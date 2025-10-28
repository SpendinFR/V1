# AGI_Evolutive/social/interaction_miner.py
# Induction de règles sociales ⟨Contexte→Tactique→Effets_attendus⟩
# depuis des dialogues inbox. Zéro LLM, heuristiques + ontologie si dispo.
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import re
import time
import hashlib
import math
import unicodedata
from collections import defaultdict


from AGI_Evolutive.social.interaction_rule import (
    InteractionRule, Predicate, TacticSpec, ContextBuilder
)
from AGI_Evolutive.utils.llm_service import (
    LLMIntegrationError,
    LLMUnavailableError,
    get_llm_manager,
    is_llm_enabled,
)

# ---------------------- utilitaires ----------------------
# ---------------------- utilitaires ----------------------
LOGGER = logging.getLogger(__name__)


def _now() -> float: return time.time()
def _hash(s: str) -> str: return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]
def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))

def _strip_accents(text: str) -> str:
    if not text:
        return ""
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def _normalize(text: str) -> str:
    return _strip_accents(text.lower())


class OnlineTextActClassifier:
    """Simple online multinomial logistic classifier on hashed n-grams."""

    def __init__(self, bucket_count: int = 2048, lr: float = 0.08, reg: float = 1e-5):
        self.bucket_count = max(32, bucket_count)
        self.lr = lr
        self.reg = reg
        self._weights: Dict[str, List[float]] = {}
        self._bias: Dict[str, float] = defaultdict(float)
        self._label_counts: Dict[str, int] = defaultdict(int)

    # --------- feature extraction ---------
    def _hash_feature(self, key: str) -> int:
        return int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16) % self.bucket_count

    def _vectorize(self, text: str) -> Dict[int, float]:
        norm = _normalize(text)
        tokens = re.findall(r"[\w']+", norm)
        feats: Dict[int, float] = defaultdict(float)
        # unigrams
        for tok in tokens:
            feats[self._hash_feature(f"uni:{tok}")] += 1.0
        # bigrams
        for i in range(len(tokens) - 1):
            feats[self._hash_feature(f"bi:{tokens[i]}_{tokens[i+1]}")] += 1.0
        # punctuation / emoji cues
        feats[self._hash_feature("feat:question_mark")] += 1.0 if "?" in text else 0.0
        feats[self._hash_feature("feat:exclam")] += text.count("!")
        feats[self._hash_feature("feat:ellipsis")] += 1.0 if "..." in text else 0.0
        feats[self._hash_feature("feat:emoji")] += len(re.findall(r"[\U0001F300-\U0001F6FF\u2600-\u26FF]", text))
        feats[self._hash_feature("feat:length_bucket")] += min(5.0, len(text) / 80.0)
        return feats

    # --------- inference / training ---------
    def _ensure_weights(self, label: str) -> None:
        if label not in self._weights:
            self._weights[label] = [0.0] * self.bucket_count
            self._bias[label] = 0.0

    def predict_with_conf(self, texts: List[str]) -> List[Tuple[str, float]]:
        results: List[Tuple[str, float]] = []
        for text in texts:
            feats = self._vectorize(text)
            best_label = "statement"
            best_score = -float("inf")
            for label, weights in self._weights.items():
                z = self._bias[label]
                for idx, val in feats.items():
                    z += weights[idx] * val
                if z > best_score:
                    best_label = label
                    best_score = z
            # sigmoid for confidence; fallback to neutral 0.5 if no weights
            conf = 1.0 / (1.0 + math.exp(-best_score)) if best_score != -float("inf") else 0.5
            results.append((best_label, conf))
        return results

    def partial_fit(self, texts: List[str], labels: List[str]) -> None:
        for text, label in zip(texts, labels):
            if not label:
                continue
            feats = self._vectorize(text)
            self._ensure_weights(label)
            weights = self._weights[label]
            bias = self._bias[label]
            # one-vs-rest logistic update
            z = bias + sum(weights[idx] * val for idx, val in feats.items())
            pred = 1.0 / (1.0 + math.exp(-z))
            target = 1.0
            error = pred - target
            for idx, val in feats.items():
                weights[idx] -= self.lr * (error * val + self.reg * weights[idx])
            self._bias[label] -= self.lr * error
            # small negative update for other labels to keep separation
            for other_label, other_weights in self._weights.items():
                if other_label == label:
                    continue
                oz = self._bias[other_label] + sum(other_weights[idx] * val for idx, val in feats.items())
                opred = 1.0 / (1.0 + math.exp(-oz))
                oerror = opred - 0.0
                for idx, val in feats.items():
                    other_weights[idx] -= self.lr * (oerror * val + self.reg * other_weights[idx])
                self._bias[other_label] -= self.lr * oerror
            self._label_counts[label] += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bucket_count": self.bucket_count,
            "lr": self.lr,
            "reg": self.reg,
            "weights": {lbl: list(wts) for lbl, wts in self._weights.items()},
            "bias": dict(self._bias),
            "label_counts": dict(self._label_counts),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "OnlineTextActClassifier":
        inst = cls(
            bucket_count=int(payload.get("bucket_count", 2048)),
            lr=float(payload.get("lr", 0.08)),
            reg=float(payload.get("reg", 1e-5)),
        )
        weights = payload.get("weights") or {}
        inst._weights = {lbl: list(vals) for lbl, vals in weights.items()}
        inst._bias = defaultdict(float, payload.get("bias") or {})
        inst._label_counts = defaultdict(int, payload.get("label_counts") or {})
        return inst


class OnlineScoreCalibrator:
    """Online GLM used to adjust rule scores with mild non-linearity."""

    def __init__(self, bucket_count: int = 256, lr: float = 0.05, reg: float = 1e-5):
        self.bucket_count = max(32, bucket_count)
        self.lr = lr
        self.reg = reg
        self._weights = [0.0] * self.bucket_count
        self._bias = 0.0

    def _hash_feature(self, key: str) -> int:
        return int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16) % self.bucket_count

    def _vectorize(self, metrics: Dict[str, float]) -> Dict[int, float]:
        feats: Dict[int, float] = {}
        for key, value in metrics.items():
            feats[self._hash_feature(f"linear:{key}")] = value
            feats[self._hash_feature(f"quad:{key}")] = value * value
        # pairwise interactions
        keys = list(metrics.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                prod = metrics[keys[i]] * metrics[keys[j]]
                feats[self._hash_feature(f"cross:{keys[i]}:{keys[j]}")] = prod
        return feats

    def predict(self, metrics: Dict[str, float], default_score: float) -> float:
        feats = self._vectorize(metrics)
        z = self._bias
        for idx, val in feats.items():
            z += self._weights[idx] * val
        calibrated = 1.0 / (1.0 + math.exp(-z))
        # Blend learned calibration with heuristic default for stability
        return clamp(0.5 * calibrated + 0.5 * default_score)

    def update(self, metrics: Dict[str, float], target: float) -> None:
        feats = self._vectorize(metrics)
        z = self._bias + sum(self._weights[idx] * val for idx, val in feats.items())
        pred = 1.0 / (1.0 + math.exp(-z))
        error = pred - clamp(target)
        for idx, val in feats.items():
            self._weights[idx] -= self.lr * (error * val + self.reg * self._weights[idx])
        self._bias -= self.lr * error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bucket_count": self.bucket_count,
            "lr": self.lr,
            "reg": self.reg,
            "weights": list(self._weights),
            "bias": self._bias,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "OnlineScoreCalibrator":
        inst = cls(
            bucket_count=int(payload.get("bucket_count", 256)),
            lr=float(payload.get("lr", 0.05)),
            reg=float(payload.get("reg", 1e-5)),
        )
        weights = payload.get("weights")
        if isinstance(weights, list):
            inst._weights = list(weights)
        inst._bias = float(payload.get("bias", 0.0))
        return inst


# Détections rapides FR (tu peux enrichir sans rien casser)
RE_QMARK = re.compile(r"(\?\s*$|(?:est\s+(?:ce|ce\s+que|un|une|le|la|l'|ceci|cela))|(?:qui|quoi|ou|où|quand|comment|pourquoi)\b)", re.I)
RE_COMPLIMENT = re.compile(r"\b(bravo|bien\s+jou[ée]|trop\s+fort|j[' ]adore|excellent|g[ée]nial|super|incroyable|magnifique)\b", re.I)
RE_THANKS     = re.compile(r"\b(merci|thanks?|thx|gratitude|remercie)\b", re.I)
RE_DISAGREE   = re.compile(r"\b(je\s+ne\s+suis\s+pas\s+d[' ]accord|pas\s+d[' ]accord|non|je\s+pense\s+pas|pas\s+s[uû]r|bof|je\s+refuse)\b", re.I)
RE_EXPLAIN    = re.compile(r"\b(parce\s+que|car|en\s+fait|la\s+raison|explication|voil[aà]\s+pou?rquoi|j[' ]explique)\b", re.I)
RE_CONFUSED   = re.compile(r"\b(je\s+ne\s+comprends?\s+pas|c[' ]est\s+quoi|hein+\??|pardon+\??|qu[' ]est[- ]ce\s+que|explique|signifie)\b", re.I)
RE_CLARIFY    = re.compile(r"\b(alors|donc|autrement\s+dit|en\s+clair|pour\s+[êe]tre\s+clair|c[aç]a\s+veut\s+dire|clarifie|r[eé]sumons)\b", re.I)
RE_INSINUATE  = re.compile(r"\b(si\s+tu\s+le\s+dis|bien\s+s[uû]r+|okay+y+|h?m+m+|lol+|mdr+|ouais\s+c[' ]est\s+c[aç]a|genre)\b", re.I)
RE_ACK        = re.compile(r"\b(ok(?:ay)?|d[' ]accord|not[eé]|je\s+vois|c[aç]a\s+marche|compris|vu|entendu)\b", re.I)

# ---------------------- structures ----------------------
@dataclass
class DialogueTurn:
    speaker: str
    text: str
    act: Optional[str] = None
    valence: float = 0.0

# ---------------------- InteractionMiner ----------------------
class InteractionMiner:
    """
    Lit du texte de conversation H↔H et fabrique des InteractionRule
    basées sur paires adjacentes + indices d'implicature.
    S'appuie sur Ontology/Beliefs/analyzers si dispo, sinon heuristiques.
    """
    def __init__(self, arch):
        self.arch = arch
        self._fallback_classifier = self._init_fallback_classifier()
        self._score_calibrator = self._init_score_calibrator()
        
    def _llm_enabled(self) -> bool:
        return is_llm_enabled()

    def _llm_manager(self):
        return get_llm_manager()

    def _init_fallback_classifier(self) -> OnlineTextActClassifier:
        existing = getattr(self.arch, "_interaction_miner_act_classifier", None)
        if isinstance(existing, OnlineTextActClassifier):
            return existing
        state = getattr(self.arch, "_interaction_miner_act_classifier_state", None)
        clf = OnlineTextActClassifier.from_dict(state) if isinstance(state, dict) else OnlineTextActClassifier()
        setattr(self.arch, "_interaction_miner_act_classifier", clf)
        return clf

    def _init_score_calibrator(self) -> OnlineScoreCalibrator:
        existing = getattr(self.arch, "_interaction_miner_score_calibrator", None)
        if isinstance(existing, OnlineScoreCalibrator):
            return existing
        state = getattr(self.arch, "_interaction_miner_score_calibrator_state", None)
        calibrator = OnlineScoreCalibrator.from_dict(state) if isinstance(state, dict) else OnlineScoreCalibrator()
        setattr(self.arch, "_interaction_miner_score_calibrator", calibrator)
        return calibrator

    def _persist_learners(self) -> None:
        try:
            setattr(self.arch, "_interaction_miner_act_classifier_state", self._fallback_classifier.to_dict())
            setattr(self.arch, "_interaction_miner_score_calibrator_state", self._score_calibrator.to_dict())
            mem = getattr(self.arch, "memory", None)
            if mem and hasattr(mem, "update_memory"):
                payload = {
                    "kind": "interaction_miner_models",
                    "act_classifier": self._fallback_classifier.to_dict(),
                    "score_calibrator": self._score_calibrator.to_dict(),
                    "ts": _now(),
                }
                try:
                    mem.update_memory(payload)
                except Exception:
                    if hasattr(mem, "add_memory"):
                        mem.add_memory(payload)
        except Exception:
            pass

    # ----------- API publique -----------
    def mine_file(self, path: str) -> List[InteractionRule]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception:
            return []
        return self.mine_text(text, source=f"inbox:{path}")

    def mine_text(self, text: str, source: str = "inbox:unknown") -> List[InteractionRule]:
        turns = self._parse_turns(text)
        self._annotate_acts(turns)            # remplit .act (heuristique + analyzers si dispo)
        llm_annotations = self._llm_annotate(turns, source)
        if llm_annotations and llm_annotations.get("speech_act") and turns:
            primary_act = str(llm_annotations["speech_act"]).strip()
            if primary_act:
                turns[0].act = primary_act
        rules = self._extract_rules(turns, source, llm_annotations)
        rules = self._merge_duplicates(rules) # fusionne mêmes règles avec +evidence
        return rules

    # ----------- Auto-évaluation & simulation -----------
    def schedule_self_evaluation(self, ctx, payload: Dict[str, Any]):
        """Background job entry point used by DocumentIngest."""
        rule_dict = payload.get("rule") if isinstance(payload, dict) else None
        arch = payload.get("arch") if isinstance(payload, dict) else None
        if rule_dict is None:
            return {"status": "skipped", "reason": "no_rule"}
        try:
            rule = InteractionRule.from_dict(rule_dict)
        except Exception:
            return {"status": "skipped", "reason": "invalid_rule"}
        arch = arch or self.arch
        outcome = self._simulate_rule(rule)
        self._persist_evaluation(rule, outcome, arch)
        return outcome

    def _simulate_rule(self, rule: InteractionRule) -> Dict[str, Any]:
        """Cheap multi-factor simulation -> returns a pseudo-outcome payload."""
        now = time.time()
        conf = float(getattr(rule, "confidence", 0.5) or 0.5)
        support = len(getattr(rule, "evidence", []) or [])
        predicate_weight = sum(float(getattr(pred, "confidence", 0.6) or 0.6) for pred in rule.predicates)
        predicate_weight = predicate_weight / max(1, len(rule.predicates)) if rule.predicates else 0.5
        provenance = getattr(rule, "provenance", {}) or {}
        last_outcome = (provenance.get("evidence") or {}).get("auto_score")

        # heuristique: plus de support & predicates cohérents => meilleur score
        base = 0.45 + (conf * 0.35) + (predicate_weight * 0.2)
        support_boost = math.tanh(support / 4.0) * 0.1
        heuristic_score = clamp(base + support_boost)

        calibrator_metrics = {
            "conf": conf,
            "support": float(support),
            "predicate_weight": predicate_weight,
            "base": base,
            "support_boost": support_boost,
            "evidence_len": float(len((provenance.get("evidence_multi") or []))),
            "has_prev_outcome": 1.0 if last_outcome is not None else 0.0,
        }
        score = self._score_calibrator.predict(calibrator_metrics, heuristic_score)
        self._score_calibrator.update(calibrator_metrics, heuristic_score if last_outcome is None else float(last_outcome))
        risk = max(0.0, 1.0 - score)
        action = "deploy" if score >= 0.62 else ("review" if score >= 0.5 else "hold")
        counterexamples: List[Dict[str, Any]] = []
        if action != "deploy":
            counterexamples.append({
                "rule_id": rule.id,
                "reason": "score_below_threshold",
                "score": score,
                "ts": now,
            })
        outcome = {
            "rule_id": rule.id,
            "score": round(score, 3),
            "risk": round(risk, 3),
            "action": action,
            "support": support,
            "timestamp": now,
            "counterexamples": counterexamples,
        }
        self._persist_learners()
        return outcome

    def _persist_evaluation(self, rule: InteractionRule, outcome: Dict[str, Any], arch) -> None:
        mem = getattr(arch, "memory", None)
        if not mem or not hasattr(mem, "add_memory"):
            return
        try:
            payload = {
                "kind": "interaction_rule_evaluation",
                "rule_id": rule.id,
                "outcome": outcome,
                "ts": outcome.get("timestamp", time.time()),
            }
            mem.add_memory(payload)
        except Exception:
            pass
        try:
            if outcome.get("counterexamples"):
                for ce in outcome["counterexamples"]:
                    mem.add_memory({
                        "kind": "interaction_counterexample",
                        "rule_id": rule.id,
                        "details": ce,
                    })
        except Exception:
            pass
        try:
            rule_dict = rule.to_dict()
            rule_dict.setdefault("evidence", {})["last_review_ts"] = outcome.get("timestamp", time.time())
            rule_dict["evidence"]["auto_score"] = outcome.get("score")
            rule_dict["evidence"]["recommended_action"] = outcome.get("action")
            if hasattr(mem, "update_memory"):
                mem.update_memory(rule_dict)
            else:
                mem.add_memory(rule_dict)
        except Exception:
            pass

    # ----------- Parsing basique des tours ----------
    def _parse_turns(self, text: str) -> List[DialogueTurn]:
        """
        Heuristique tolérante: détecte 'A:', 'B:', 'User:', '—', ou lignes sèches.
        On essaye de conserver l'alternance; sinon on attribue speakers génériques.
        """
        lines = [l.strip() for l in re.split(r"[\r\n]+", text or "") if l.strip()]
        turns: List[DialogueTurn] = []
        cur_speaker = "A"
        for ln in lines:
            m = re.match(r"^([A-Za-z]{1,12})\s*[:\-]\s*(.+)$", ln)
            if m:
                spk, utt = m.group(1), m.group(2)
                turns.append(DialogueTurn(spk.strip(), utt.strip()))
                cur_speaker = spk.strip()
            else:
                # bullets/quote style
                if ln.startswith(("-", "•", "—", ">")):
                    ln = ln.lstrip("-•—> ").strip()
                turns.append(DialogueTurn(cur_speaker, ln))
                cur_speaker = "B" if cur_speaker == "A" else "A"
        return turns

    # ----------- Annotation d'actes de dialogue ----------
    def _annotate_acts(self, turns: List[DialogueTurn]) -> None:
        """
        Si un analyzer existe (arch.conversation_classifier, arch.analyzers), on l'utilise.
        Sinon heuristiques simples.
        """
        # Essayez analyzers si dispo
        used_external = False
        try:
            clf = getattr(self.arch, "conversation_classifier", None)
            if clf and hasattr(clf, "predict_acts"):
                acts = clf.predict_acts([t.text for t in turns])
                for t, a in zip(turns, acts):
                    t.act = a
                used_external = True
        except Exception:
            pass

        if used_external:
            return

        # Fallback classifier pre-pass
        fallback = self._fallback_classifier
        if fallback and turns:
            try:
                texts = [tr.text for tr in turns]
                for t, (label, conf) in zip(turns, fallback.predict_with_conf(texts)):
                    if conf >= 0.55 and not t.act:
                        t.act = label
            except Exception:
                pass

        # Heuristique locale enrichie
        fallback_fit_texts: List[str] = []
        fallback_fit_labels: List[str] = []
        for t in turns:
            low = t.text.lower()
            norm = _normalize(t.text)
            assigned_before = t.act
            heuristic_hit = False
            if RE_QMARK.search(t.text) or RE_QMARK.search(norm):
                t.act = "question"
                heuristic_hit = True
            elif RE_COMPLIMENT.search(low) or RE_COMPLIMENT.search(norm):
                t.act = "compliment"
                heuristic_hit = True
            elif RE_DISAGREE.search(low) or RE_DISAGREE.search(norm):
                t.act = "disagreement"
                heuristic_hit = True
            elif RE_CONFUSED.search(low) or RE_CONFUSED.search(norm):
                t.act = "confusion"
                heuristic_hit = True
            elif RE_INSINUATE.search(low) or RE_INSINUATE.search(norm):
                t.act = "insinuation"
                heuristic_hit = True
            elif RE_THANKS.search(low) or RE_THANKS.search(norm):
                t.act = "thanks"
                heuristic_hit = True
            elif RE_ACK.search(low) or RE_ACK.search(norm):
                t.act = "ack"
                heuristic_hit = True
            elif RE_EXPLAIN.search(low) or RE_EXPLAIN.search(norm):
                t.act = "explain"
                heuristic_hit = True
            elif RE_CLARIFY.search(low) or RE_CLARIFY.search(norm):
                t.act = "clarify"
                heuristic_hit = True
            else:
                t.act = t.act or "statement"

            if heuristic_hit or (assigned_before and assigned_before != "statement"):
                fallback_fit_texts.append(t.text)
                fallback_fit_labels.append(t.act)

        if fallback and fallback_fit_texts:
            try:
                fallback.partial_fit(fallback_fit_texts, fallback_fit_labels)
            except Exception:
                pass
        self._persist_learners()

    # ----------- Extraction de règles (paires + implicatures) ----------
    def _extract_rules(
        self,
        turns: List[DialogueTurn],
        source: str,
        llm_annotations: Optional[Dict[str, Any]] = None,
    ) -> List[InteractionRule]:
        rules: List[InteractionRule] = []
        if llm_annotations:
            llm_rules = self._rules_from_llm(turns, source, llm_annotations)
            if llm_rules:
                rules.extend(llm_rules)
        n = len(turns)
        for i in range(n - 1):
            a, b = turns[i], turns[i + 1]
            pair = (a.act, b.act)

            # 1) Question -> Answer concise
            if a.act == "question" and b.act not in ("question", None):
                rules.append(self._rule_from_pair(
                    kind="question_answer",
                    a=a, b=b, source=source,
                    preds=[
                        Predicate("dialogue_act", "eq", "question", 1.0),
                        Predicate("risk_level",   "in", ["low","medium"], 0.6),
                        Predicate("persona_alignment", "ge", 0.2, 0.4),
                    ],
                    tactic=TacticSpec("answer_concise", {"max_len": 200, "ensure_ack": True}),
                    base_conf=0.68
                ))

            # 2) Compliment -> Acknowledge grateful
            if RE_COMPLIMENT.search(a.text) and (b.act in ("thanks","ack","statement")):
                rules.append(self._rule_from_pair(
                    kind="compliment_thanks",
                    a=a, b=b, source=source,
                    preds=[
                        Predicate("dialogue_act", "eq", "compliment", 1.1),
                        Predicate("risk_level",   "in", ["low","medium"], 0.7),
                    ],
                    tactic=TacticSpec("ack_grateful", {"warmth": "auto"}),
                    base_conf=0.70
                ))

            # 3) Insinuation -> Banter léger ou Reformulation empathique (selon risk)
            if a.act == "insinuation" and (b.act in ("ack","statement","insinuation")):
                # On crée DEUX règles candidates (policy/selector trancheront selon context)
                rules.append(self._rule_from_pair(
                    kind="insinuation_banter",
                    a=a, b=b, source=source,
                    preds=[
                        Predicate("dialogue_act", "eq", "insinuation", 1.2),
                        Predicate("risk_level",   "in", ["low","medium"], 0.8),
                        Predicate("persona_alignment", "ge", 0.3, 0.6),
                    ],
                    tactic=TacticSpec("banter_leger", {"soft": True, "max_len_delta": 40}),
                    base_conf=0.64
                ))
                rules.append(self._rule_from_pair(
                    kind="insinuation_reformulate",
                    a=a, b=b, source=source,
                    preds=[
                        Predicate("dialogue_act", "eq", "insinuation", 1.2),
                        Predicate("risk_level",   "in", ["medium","high"], 0.9),
                    ],
                    tactic=TacticSpec("reformulation_empathique", {"mirror_ratio": 0.4}),
                    base_conf=0.62
                ))

            # 4) Disagreement -> Explain/Clarify
            if a.act == "disagreement" and (b.act in ("explain","clarify","statement")):
                rules.append(self._rule_from_pair(
                    kind="disagree_explain",
                    a=a, b=b, source=source,
                    preds=[
                        Predicate("dialogue_act", "eq", "disagreement", 1.0),
                        Predicate("risk_level",   "in", ["low","medium","high"], 0.7),
                    ],
                    tactic=TacticSpec("clarification_rationale", {"connective": "parce que"}),
                    base_conf=0.66
                ))

            # 5) Confusion -> Clarify / Reformulation
            if a.act == "confusion" and (b.act in ("clarify","explain","statement")):
                rules.append(self._rule_from_pair(
                    kind="confusion_clarify",
                    a=a, b=b, source=source,
                    preds=[
                        Predicate("dialogue_act", "eq", "confusion", 1.0),
                        Predicate("risk_level",   "in", ["low","medium"], 0.8),
                    ],
                    tactic=TacticSpec("clarify_definition", {"ensure_example": True}),
                    base_conf=0.69
                ))
                rules.append(self._rule_from_pair(
                    kind="confusion_reformulation",
                    a=a, b=b, source=source,
                    preds=[
                        Predicate("dialogue_act", "eq", "confusion", 1.0),
                        Predicate("risk_level",   "in", ["low","medium","high"], 0.7),
                        Predicate("persona_alignment", "ge", 0.2, 0.4),
                    ],
                    tactic=TacticSpec("reformulation_empathique", {"mirror_ratio": 0.6}),
                    base_conf=0.65
                ))

        # enrichissement par ontologie (synonymes -> implicature_hint)
        for r in rules:
            try:
                onto = getattr(self.arch, "ontology", None)
                if not onto: 
                    continue
                # Exemple: si act=insinuation, ajoute un prédicat implicature_hint ∈ {ironie, sous-entendu}
                if any(p.op == "eq" and p.key == "dialogue_act" and p.value == "insinuation"
                       for p in r.context_predicates):
                    r.context_predicates.append(Predicate("implicature_hint", "in", ["sous-entendu","ironie"], 0.4))
            except Exception:
                pass

        return rules

    def _llm_payload(self, turns: List[DialogueTurn], source: str) -> Dict[str, Any]:
        return {
            "source": source,
            "turns": [
                {
                    "speaker": t.speaker,
                    "text": t.text,
                    "act": t.act,
                    "valence": t.valence,
                }
                for t in turns[:12]
            ],
        }

    def _llm_annotate(
        self, turns: List[DialogueTurn], source: str
    ) -> Optional[Dict[str, Any]]:
        if not turns or not self._llm_enabled():
            return None

        payload = self._llm_payload(turns, source)
        try:
            response = self._llm_manager().call_dict(
                "social_interaction_miner",
                input_payload=payload,
            )
        except (LLMUnavailableError, LLMIntegrationError):
            LOGGER.debug("LLM interaction miner unavailable", exc_info=True)
            return None

        if not isinstance(response, dict):
            return None
        return response

    def _rules_from_llm(
        self,
        turns: List[DialogueTurn],
        source: str,
        annotations: Dict[str, Any],
    ) -> List[InteractionRule]:
        suggestions = annotations.get("suggested_rules")
        if not isinstance(suggestions, list):
            return []

        speech_act = annotations.get("speech_act")
        base_conf = float(annotations.get("confidence", 0.0) or 0.0)
        base_conf = max(0.0, min(1.0, base_conf)) or 0.72
        rules: List[InteractionRule] = []
        for entry in suggestions:
            if not isinstance(entry, dict):
                continue
            rule_name = str(entry.get("rule") or "").strip()
            if not rule_name:
                continue
            params = entry.get("parameters") if isinstance(entry.get("parameters"), dict) else {}
            rationale = entry.get("rationale") or entry.get("explanation")
            expected_effect = entry.get("expected_effect")
            intensity = float(entry.get("confidence", base_conf) or base_conf)
            intensity = max(0.05, min(1.0, intensity))

            predicates: List[Predicate] = []
            if speech_act:
                predicates.append(Predicate("dialogue_act", "eq", speech_act, 1.0))
            if entry.get("context") and isinstance(entry["context"], dict):
                for key, val in list(entry["context"].items())[:4]:
                    predicates.append(Predicate(str(key), "eq", val, 0.6))
            if not predicates:
                predicates.append(Predicate("dialogue_act", "exists", None, 0.4))

            tactic = TacticSpec(rule_name, dict(params))
            provenance = {
                "source": source,
                "kind": "llm_suggestion",
                "llm": {
                    "speech_act": speech_act,
                    "suggestion": entry,
                },
            }
            if expected_effect:
                provenance.setdefault("llm", {})["expected_effect"] = expected_effect
            rule = InteractionRule.build(predicates, tactic, provenance=provenance)
            rule.confidence = clamp(float(intensity), 0.05, 1.0)
            rule.tags = list(set((rule.tags or []) + ["llm", "social"]))
            if rationale:
                rule.provenance.setdefault("llm", {})["rationale"] = rationale
            if turns:
                rule.provenance.setdefault("evidence", {})["a"] = turns[0].text
                if len(turns) > 1:
                    rule.provenance.setdefault("evidence", {})["b"] = turns[1].text
            rules.append(rule)

        return rules

    def _rule_from_pair(self, kind: str, a: DialogueTurn, b: DialogueTurn, source: str,
                        preds: List[Predicate], tactic: TacticSpec, base_conf: float) -> InteractionRule:
        r = InteractionRule.build(preds, tactic, provenance={
            "source": source, "kind": kind,
            "evidence": {"a": a.text, "b": b.text}
        })
        # amorce de confiance
        r.confidence = clamp(base_conf, 0.0, 1.0)
        # tags utiles
        r.tags = list(set((r.tags or []) + ["mined","inbox"]))
        return r

    # ----------- fusion de doublons / agrégation d'évidence ----------
    def _merge_duplicates(self, rules: List[InteractionRule]) -> List[InteractionRule]:
        merged: Dict[str, InteractionRule] = {}
        for r in rules:
            if r.id in merged:
                # on augmente légèrement la confiance si plusieurs occurrences
                merged[r.id].confidence = clamp(0.6*merged[r.id].confidence + 0.4*r.confidence, 0.0, 1.0)
                # concat (sans explosion)
                ev = merged[r.id].provenance.get("evidence", {})
                if isinstance(ev, dict) and isinstance(r.provenance.get("evidence"), dict):
                    # garde 3 exemples max
                    ex = []
                    if isinstance(ev.get("a"), str) and isinstance(ev.get("b"), str):
                        ex.append(ev)
                    ex.append(r.provenance["evidence"])
                    merged[r.id].provenance["evidence_multi"] = (merged[r.id].provenance.get("evidence_multi", []) + ex)[-3:]
            else:
                merged[r.id] = r
        return list(merged.values())
