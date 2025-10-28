from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Iterable
import logging
import re, time, json, os, unicodedata, math

# ---------- utils ----------
def _now(): return time.time()
def clamp(x,a=0,b=1): return max(a, min(b, x))

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = re.sub(r"\s+", " ", s.strip())
    return s

def _clean_term(t: str) -> str:
    t = t.strip().lower()
    t = re.sub(r"[^\w\- ]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t

from AGI_Evolutive.utils.llm_service import (
    LLMIntegrationError,
    LLMUnavailableError,
    get_llm_manager,
    is_llm_enabled,
)


LOGGER = logging.getLogger(__name__)


STOP = set("""
le la les un une des de du au aux et ou mais donc car que qui quoi dont où
je tu il elle on nous vous ils elles ne pas plus moins très trop ce cette ces
mon ton son ma ta sa mes tes ses est es suis êtes sont c'est ça d' l'
""".split())

SUF_N = ("ie","té","tion","sion","isme","ence","ance","eur","ure","ment","esse","isme")
SUF_A = ("ique","if","ive","el","elle","al","ale","aire","eux","euse")
SUF_V = ("iser","ifier","iser","iser","ifier","er","ir","re")  # repères faibles

# ---------- modèle léger en ligne ----------

class OnlineConceptModel:
    """Classifieur logistique léger entraîné en ligne sur les retours."""

    def __init__(self, path: str = "data/concept_model.json", lr: float = 0.25):
        self.path = path
        self.lr = lr
        self.state = self._load()

    # --- persistance ---
    def _load(self) -> Dict[str, Any]:
        try:
            if os.path.exists(self.path):
                data = json.load(open(self.path, "r", encoding="utf-8"))
                if isinstance(data, dict):
                    data.setdefault("bias", 0.0)
                    data.setdefault("weights", {})
                    return data
        except Exception:
            pass
        return {"bias": 0.0, "weights": {}}

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            json.dump(self.state, open(self.path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        except Exception:
            pass

    # --- features ---
    def build_features(
        self,
        kind: str,
        label: str,
        base_score: float,
        families: Iterable[str],
        evidence: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        families = list(families or [])
        ev = evidence or {}
        feats: Dict[str, float] = {
            "bias": 1.0,
            f"kind={kind}": 1.0,
            "base_score": float(base_score),
            "len_char": float(len(label)),
            "len_tokens": float(len(label.split())),
            "has_space": 1.0 if " " in label else 0.0,
            "families_count": float(len(families)),
        }
        if extra:
            for k, v in extra.items():
                if isinstance(v, (int, float)):
                    feats[f"extra:{k}"] = float(v)
        for fam in families:
            feats[f"fam={fam}"] = 1.0
            count = 0.0
            try:
                fam_ev = ev.get(fam, [])
                if isinstance(fam_ev, list):
                    count = float(len(fam_ev))
            except Exception:
                count = 0.0
            feats[f"famhits={fam}"] = count
        feats["evidence_families"] = float(len(ev))
        return feats

    # --- scoring ---
    def _dot(self, feats: Dict[str, float]) -> float:
        weights: Dict[str, float] = self.state.get("weights", {})
        return sum(weights.get(k, 0.0) * v for k, v in feats.items() if k != "bias") + self.state.get("bias", 0.0)

    def predict(self, feats: Dict[str, float]) -> float:
        z = self._dot(feats)
        try:
            return 1.0 / (1.0 + math.exp(-z))
        except OverflowError:
            return 1.0 if z > 0 else 0.0

    def update(self, feats: Dict[str, float], target: float, weight: float = 1.0):
        if not feats:
            return
        pred = self.predict(feats)
        err = (target - pred) * self.lr * weight
        weights: Dict[str, float] = self.state.setdefault("weights", {})
        # bias update
        self.state["bias"] = float(self.state.get("bias", 0.0) + err)
        for k, v in feats.items():
            if k == "bias":
                continue
            weights[k] = float(clamp(weights.get(k, 0.0) + err * v, -2.5, 2.5))
        self.save()

# ---------- patrons génériques (déf, reformulation, étiquette, citations…) ----------
# Permet également « terme – est » ou « terme — est » et variations d'espaces
RE_DEF_1 = re.compile(r"\b([A-Za-zÀ-ÿ][\w\- ]{2,})\b\s*(?:[–—-]\s*)?(?:est|sont|c['e]st|signifie|désigne|correspond(?:\s+à)?|implique|se manifeste(?:\s+par)?)\s", re.I)
RE_DEF_2 = re.compile(r"(?:qu['e]st-ce que|c['e]st quoi)\s+([A-Za-zÀ-ÿ][\w\- ]{2,})\b\??", re.I)
RE_LABEL = re.compile(r"^([A-Za-zÀ-ÿ][\w\- ]{2,})\s*[:\-]\s+.+$", re.M)
RE_QUOTE = re.compile(r"[«\"]([A-Za-zÀ-ÿ][\w\- ]{2,})[»\"]")
RE_REFORM = re.compile(r"\b(autrement dit|en d'autres termes|en clair|pour dire simple|c'est-à-dire)\b", re.I)
RE_RHET_Q = re.compile(r"\?\s*$")

# ---------- indices de style ----------
RE_IRONY = re.compile(r"\b(si tu le dis|bien sûr+|ouais c'est ça|lol+|mdr+|ptdr+)\b", re.I)
RE_FORMAL = re.compile(r"\b(cependant|toutefois|néanmoins|en revanche|par conséquent|de surcroît)\b", re.I)
RE_SLANG = re.compile(r"\b(chelou|wesh|grave|nan|ouais|bg|relou|bcp|tmtc)\b", re.I)
RE_EMOJIS = re.compile(r"[\U0001F300-\U0001F6FF\U0001F900-\U0001F9FF\U00002600-\U000026FF\U00002700-\U000027BF]+")

def _is_stopish(t: str) -> bool:
    toks = [w for w in t.split() if w]
    if not toks: return True
    if len(toks) == 1 and toks[0] in STOP: return True
    return False

def _is_concepty(t: str) -> bool:
    # "conceptuel" par morpho/longueur/structure
    if _is_stopish(t): return False
    if len(t) < 3: return False
    if any(t.endswith(s) for s in SUF_N): return True
    if " " in t and len(t.split()) <= 4: return True
    if any(t.endswith(s) for s in SUF_A): return True
    return True  # garde ouvert

@dataclass
class ItemCandidate:
    # kind: "concept" | "term" | "style" | "construction"
    kind: str
    label: str
    score: float
    evidence: Dict[str, Any]
    features: Dict[str, Any]
    ts: float

# ---------- ConceptRecognizer ----------
class ConceptRecognizer:
    """
    Détecte des candidats "à apprendre" dans un texte. Hybride:
    - patrons génériques (définitions/étiquettes/citations/reformulations/questions)
    - morphologie FR (suffixes nom/adjectif/verbe)
    - OOV/termes fréquents inconnus (skills/ontology)
    - indices dialogiques (via miner: acts -> hint concept)
    - styles et constructions
    Apprend (renforce) ses propres patrons quand un item est confirmé.
    """
    def __init__(self, arch, patterns_path: str = "data/concept_patterns.json"):
        self.arch = arch
        self.path = getattr(arch, "concept_patterns_path", patterns_path)
        self.patterns: Dict[str, Dict[str, Any]] = self._load_patterns()
        self.feedback_log_path = getattr(arch, "concept_feedback_path", "data/concept_feedback.jsonl")
        model_path = getattr(arch, "concept_model_path", "data/concept_model.json")
        self.model_mix = float(getattr(arch, "concept_model_mix", 0.45))
        self.model = OnlineConceptModel(model_path)

    # --- patterns dynamiques : poids par famille ---
    def _load_patterns(self) -> Dict[str, Any]:
        try:
            if os.path.exists(self.path):
                data = json.load(open(self.path, "r", encoding="utf-8")) or {}
                if isinstance(data, dict):
                    for fam, default in {
                        "def_1": 0.45,
                        "def_2": 0.40,
                        "label": 0.30,
                        "quote": 0.25,
                        "reform": 0.18,
                        "morph": 0.22,
                        "verb": 0.15,
                        "oov": 0.25,
                        "dialog": 0.20,
                        "style_irony": 0.40,
                        "style_formal": 0.35,
                        "style_slang": 0.35,
                        "style_emoji": 0.25,
                        "construction_rhetq": 0.22,
                    }.items():
                        data.setdefault(fam, {"w": default})
                return data
        except Exception:
            pass
        return {
            # familles -> poids init (évoluent)
            "def_1": {"w": 0.45},       # "X est / c'est / signifie ..."
            "def_2": {"w": 0.40},       # "qu'est-ce que X ? / c'est quoi X ?"
            "label": {"w": 0.30},       # "X : ..."
            "quote": {"w": 0.25},       # «X»
            "reform": {"w": 0.18},      # "autrement dit", etc. -> construction
            "morph": {"w": 0.22},       # suffixes conceptuels
            "verb":  {"w": 0.15},       # suffixes verbaux (faible)
            "oov":   {"w": 0.25},       # inconnu des skills/ontology (freq≥2)
            "dialog": {"w":0.20},       # indice via actes (ex: empathy_act)
            # styles
            "style_irony":  {"w": 0.40},
            "style_formal": {"w": 0.35},
            "style_slang":  {"w": 0.35},
            "style_emoji":  {"w": 0.25},
            # constructions (questions rhétoriques, etc.)
            "construction_rhetq": {"w": 0.22},
        }

    def save_patterns(self):
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            json.dump(self.patterns, open(self.path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        except Exception:
            pass

    # --- helpers internes ---
    def _families_from_evidence(self, evidence: Optional[Dict[str, Any]],
                                fallback: Optional[Iterable[str]] = None) -> List[str]:
        if isinstance(evidence, dict) and evidence:
            return sorted(list(evidence.keys()))
        if fallback:
            return sorted(list(fallback))
        return []

    def _log_feedback(self, payload: Dict[str, Any]):
        try:
            os.makedirs(os.path.dirname(self.feedback_log_path), exist_ok=True)
            with open(self.feedback_log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _model_features(self, kind: str, label: str, base_score: float,
                        families: Iterable[str], evidence: Optional[Dict[str, Any]] = None,
                        features: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        extra = {}
        if features:
            for key in ("raw_freq", "dialog_hint", "score_boost"):
                if key in features and isinstance(features[key], (int, float)):
                    extra[key] = features[key]
            hits = features.get("family_hits")
            if isinstance(hits, dict):
                for fam, val in hits.items():
                    if isinstance(val, (int, float)):
                        extra[f"hit:{fam}"] = val
        return self.model.build_features(kind, label, base_score, families, evidence, extra=extra)

    def _apply_model(self, kind: str, label: str, base_score: float,
                     families: Iterable[str], evidence: Dict[str, Any],
                     features: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        if not self.model_mix:
            return clamp(base_score, 0.0, 1.0), {"raw": base_score, "ml": None}
        feat_vector = self._model_features(kind, label, base_score, families, evidence, features)
        ml_score = self.model.predict(feat_vector)
        # mix: base + correction centrée
        delta = (ml_score - 0.5) * self.model_mix
        final = clamp(base_score + delta, 0.0, 1.0)
        model_info = {
            "raw": base_score,
            "ml_score": ml_score,
            "delta": delta,
            "mix": self.model_mix,
            "vector": feat_vector,
        }
        return final, model_info

    def _log_tracker(self, arch, goal_id: str, item: ItemCandidate):
        mem = getattr(arch, "memory", None)
        if not mem or not hasattr(mem, "add_memory"):
            return
        try:
            payload = {
                "kind": "concept_learning_tracker",
                "goal_id": goal_id,
                "label": item.label,
                "score": item.score,
                "families": item.features.get("families", []) if isinstance(item.features, dict) else [],
                "ts": item.ts,
                "evidence": item.evidence,
            }
            model_info = item.features.get("model") if isinstance(item.features, dict) else None
            if isinstance(model_info, dict):
                payload["model"] = {
                    "raw": model_info.get("raw"),
                    "ml_score": model_info.get("ml_score"),
                    "delta": model_info.get("delta"),
                }
            mem.add_memory(payload)
        except Exception:
            pass

    # --- extraction principale ---
    def extract_candidates(self, text: str, dialog_hints: Optional[Dict[str, Any]] = None) -> List[ItemCandidate]:
        textN = _norm(text)
        evC: Dict[str, Dict[str, Any]] = {}   # evidence par label
        scores: Dict[Tuple[str,str], float] = {}  # (kind,label) -> score brut
        feats: Dict[Tuple[str,str], Dict[str, Any]] = {}

        def bump(kind: str, label: str, family: str, amount: float, evidence: Any, *, meta: Optional[Dict[str, Any]] = None):
            k = (kind, label)
            scores[k] = scores.get(k, 0.0) + amount
            evC.setdefault(label, {}).setdefault(family, [])
            evC[label][family].append(evidence)
            entry = feats.setdefault(k, {"families": set(), "family_hits": {}, "meta": {}})
            entry["families"].add(family)
            hits = entry.setdefault("family_hits", {})
            hits[family] = hits.get(family, 0) + 1
            if meta:
                entry_meta = entry.setdefault("meta", {})
                for mk, mv in meta.items():
                    if isinstance(mv, (int, float)):
                        entry_meta[mk] = float(mv)

        def _dialog_boost(concept: str, weight: float, reason: Any):
            label = _clean_term(concept)
            if not label or _is_stopish(label):
                return
            boost = clamp(weight, 0.0, 1.0) * self.patterns["dialog"]["w"]
            bump("concept", label, "dialog", boost, reason, meta={"dialog_hint": boost})

        # A) Concepts par patrons définitionnels / labels / citations
        for m in RE_DEF_1.finditer(textN):
            t = _clean_term(m.group(1))
            if _is_concepty(t):
                bump("concept", t, "def_1", self.patterns["def_1"]["w"], m.group(0))
        for m in RE_DEF_2.finditer(textN):
            t = _clean_term(m.group(1))
            if _is_concepty(t):
                bump("concept", t, "def_2", self.patterns["def_2"]["w"], m.group(0))
        for m in RE_LABEL.finditer(textN):
            t = _clean_term(m.group(1))
            if _is_concepty(t):
                bump("concept", t, "label", self.patterns["label"]["w"], m.group(0))
        for m in RE_QUOTE.finditer(textN):
            t = _clean_term(m.group(1))
            if _is_concepty(t):
                bump("concept", t, "quote", self.patterns["quote"]["w"], m.group(0))

        # B) Constructions (reformulation, c-à-d)
        for m in RE_REFORM.finditer(textN):
            bump("construction", "reformulation", "reform", self.patterns["reform"]["w"], m.group(0))

        # C) Styles
        if RE_IRONY.search(textN):
            bump("style", "ironie", "style_irony", self.patterns["style_irony"]["w"], "irony_marker")
        if RE_FORMAL.search(textN):
            bump("style", "formel", "style_formal", self.patterns["style_formal"]["w"], "formal_markers")
        if RE_SLANG.search(textN):
            bump("style", "argot", "style_slang", self.patterns["style_slang"]["w"], "slang_markers")
        if RE_EMOJIS.search(textN):
            bump("style", "emoji_usage", "style_emoji", self.patterns["style_emoji"]["w"], "emoji_present")

        # D) Morphologie (concepts & verbes candidats)
        for w in re.findall(r"[A-Za-zÀ-ÿ][\w\-]{2,}", textN):
            t = _clean_term(w)
            if not t or t in STOP: 
                continue
            if any(t.endswith(s) for s in SUF_N) or any(t.endswith(s) for s in SUF_A):
                bump("concept", t, "morph", self.patterns["morph"]["w"], t)
            if any(t.endswith(s) for s in SUF_V):
                bump("term", t, "verb", self.patterns["verb"]["w"], t)

        # E) OOV / fréquences -> termes à apprendre (même 1 mot)
        known = set()
        try:
            # skills connus
            skills_path = getattr(self.arch, "skills_path", "data/skills.json")
            if os.path.exists(skills_path):
                data = json.load(open(skills_path, "r", encoding="utf-8")) or {}
                known |= set(data.keys())
        except Exception: pass
        try:
            onto = getattr(self.arch, "ontology", None)
            if onto and hasattr(onto, "all_concepts"):
                known |= set(onto.all_concepts())
        except Exception: pass

        freqs: Dict[str,int] = {}
        for w in re.findall(r"[A-Za-zÀ-ÿ][\w\-]{2,}", textN):
            t = _clean_term(w)
            if not _is_stopish(t):
                freqs[t] = freqs.get(t,0)+1

        for t, f in freqs.items():
            if t not in known and f >= 2:
                # "term" (mot/locution) candidat à apprendre
                amount = self.patterns["oov"]["w"] * min(1.0, (f-1)/3.0)
                bump("term", t, "oov", amount, {"freq": f}, meta={"raw_freq": f})

        # F) Indices dialogiques (hints passés par le miner)
        # Exemple: {"empathy_act": True} → augmente le poids du concept "empathie"
        if dialog_hints:
            if dialog_hints.get("empathy_act"):
                bump("concept", "empathie", "dialog", self.patterns["dialog"]["w"], "empathy_act",
                     meta={"dialog_hint": self.patterns["dialog"]["w"]})
            concepts_hint = dialog_hints.get("concepts")
            if isinstance(concepts_hint, dict):
                for concept, weight in concepts_hint.items():
                    _dialog_boost(concept, float(weight), {"hint": concept, "weight": weight})
            elif isinstance(concepts_hint, (list, tuple)):
                for concept in concepts_hint:
                    _dialog_boost(str(concept), 0.6, concept)

        # G) Questions rhétoriques (construction)
        for line in re.split(r"[\r\n]+", textN):
            if RE_RHET_Q.search(line.strip()):
                bump("construction", "question_rhetorique", "construction_rhetq", self.patterns["construction_rhetq"]["w"], line.strip())

        # Assemblage final
        items: List[ItemCandidate] = []
        for (kind, label), s in scores.items():
            label_clean = _clean_term(label)
            if not label_clean:
                continue
            # concepty pour "concept", souple pour "term"
            if kind == "concept" and not _is_concepty(label_clean):
                continue
            base_score = clamp(s, 0.0, 1.0)
            evidence = evC.get(label, {})
            feat_entry = feats.get((kind, label), {"families": set(), "family_hits": {}, "meta": {}})
            fams = feat_entry.get("families", set())
            final_score, model_info = self._apply_model(kind, label_clean, base_score, fams, evidence, {
                "family_hits": feat_entry.get("family_hits", {}),
                **(feat_entry.get("meta", {}) or {}),
            })
            features_payload = {
                "families": sorted(list(fams)),
                "family_hits": feat_entry.get("family_hits", {}),
                "raw_score": base_score,
                "model": model_info,
                "meta": feat_entry.get("meta", {}),
            }
            items.append(ItemCandidate(kind=kind, label=label_clean, score=final_score,
                                       evidence=evidence, features=features_payload, ts=_now()))
        items = self._apply_llm_guidance(textN, items)
        # tri: score puis richesse de l'evidence
        items.sort(key=lambda c: (c.score, sum(len(v) if isinstance(v,list) else 1 for v in c.evidence.values())), reverse=True)
        return items

    def _apply_llm_guidance(self, text: str, items: List[ItemCandidate]) -> List[ItemCandidate]:
        if not items or not is_llm_enabled():
            return items

        payload = {
            "text": text[:2000],
            "candidates": [
                {
                    "kind": item.kind,
                    "label": item.label,
                    "score": item.score,
                    "evidence": item.evidence,
                    "features": item.features,
                }
                for item in items[:10]
            ],
        }

        if not payload["candidates"]:
            return items

        try:
            response = get_llm_manager().call_dict(
                "concept_recognizer",
                input_payload=payload,
            )
        except (LLMUnavailableError, LLMIntegrationError):
            LOGGER.debug("LLM concept recognizer unavailable", exc_info=True)
            return items

        if not isinstance(response, dict):
            return items

        mapping: Dict[str, Dict[str, Any]] = {}
        for entry in response.get("candidates", []):
            if not isinstance(entry, dict):
                continue
            name = _clean_term(str(entry.get("candidate") or entry.get("label") or ""))
            if not name:
                continue
            mapping[name] = entry

        for item in items:
            data = mapping.get(item.label)
            if not data:
                continue
            status = str(data.get("status") or "").strip().lower()
            justification = data.get("justification")
            rec_learning = data.get("recommended_learning")
            llm_record = {
                "status": status,
                "justification": justification,
                "recommended_learning": rec_learning,
                "notes": data.get("notes"),
            }
            item.features.setdefault("llm", llm_record)
            item.evidence.setdefault("llm", []).append({k: v for k, v in llm_record.items() if v})

            if status in {"à_apprendre", "apprendre", "retain", "promote", "a_apprendre"}:
                item.score = clamp(0.6 * item.score + 0.4 * float(data.get("confidence", 0.85) or 0.85), 0.0, 1.0)
            elif status in {"surveiller", "review", "a_surveiller"}:
                item.score = clamp(0.5 * item.score + 0.5 * 0.55, 0.0, 1.0)
            elif status in {"rejeter", "rejet", "reject", "ignore"}:
                item.score = clamp(min(item.score, 0.25), 0.0, 1.0)

            suggested_kind = str(data.get("kind") or "").strip().lower()
            if suggested_kind and suggested_kind in {"concept", "term", "style", "construction"}:
                item.kind = suggested_kind

        notes = response.get("notes")
        if notes:
            LOGGER.debug("LLM concept recognizer notes: %s", notes)

        return items

    # --- apprentissage des patrons à partir d'une confirmation ---
    def learn_from_confirmation(self, kind: str, label: str, evidence: Dict[str, Any],
                                 reward: float = 0.8, features: Optional[Dict[str, Any]] = None):
        families = self._families_from_evidence(evidence, (features or {}).get("families"))
        stable = {"def_1", "def_2", "label", "quote"}
        exploratory = {"morph", "verb", "oov", "dialog", "style_irony", "style_slang", "style_emoji"}
        base_score = float((features or {}).get("raw_score", 0.5))
        for fam in families:
            if fam in self.patterns:
                w = float(self.patterns[fam].get("w", 0.2))
                if fam in stable:
                    delta = 0.035 * reward
                elif fam in exploratory:
                    delta = 0.065 * reward
                else:
                    delta = 0.05 * reward
                self.patterns[fam]["w"] = clamp(w + delta, 0.05, 0.95)
        self.save_patterns()

        feat_vector = self._model_features(kind, _clean_term(label), base_score, families, evidence, features)
        self.model.update(feat_vector, 1.0, weight=reward)
        self._log_feedback({
            "ts": _now(),
            "outcome": "confirm",
            "kind": kind,
            "label": _clean_term(label),
            "reward": reward,
            "families": families,
            "raw_score": base_score,
            "features": feat_vector,
        })

    def learn_from_rejection(self, kind: str, label: str, evidence: Dict[str, Any],
                              penalty: float = 0.5, features: Optional[Dict[str, Any]] = None):
        families = self._families_from_evidence(evidence, (features or {}).get("families"))
        exploratory = {"morph", "verb", "oov", "dialog", "style_slang", "style_emoji"}
        base_score = float((features or {}).get("raw_score", 0.5))
        for fam in families:
            if fam in self.patterns:
                w = float(self.patterns[fam].get("w", 0.2))
                if fam in exploratory:
                    delta = 0.055 * penalty
                else:
                    delta = 0.04 * penalty
                self.patterns[fam]["w"] = clamp(w - delta, 0.03, 0.9)
        self.save_patterns()

        feat_vector = self._model_features(kind, _clean_term(label), base_score, families, evidence, features)
        self.model.update(feat_vector, 0.0, weight=penalty)
        self._log_feedback({
            "ts": _now(),
            "outcome": "reject",
            "kind": kind,
            "label": _clean_term(label),
            "penalty": penalty,
            "families": families,
            "raw_score": base_score,
            "features": feat_vector,
        })

    # --- helpers d'intégration (faciles à appeler) ---
    def commit_candidates_to_memory(self, source: str, items: List[ItemCandidate], arch=None):
        arch = arch or self.arch
        for it in items:
            arch.memory.add_memory({
                "kind": f"{it.kind}_candidate",
                "label": it.label,
                "score": it.score,
                "evidence": it.evidence,
                "features": it.features,
                "ts": it.ts,
                "source": source
            })

    def autogoals_for_high_confidence(self, items: List[ItemCandidate], arch=None,
                                      th_concept: float = 0.72, th_term: float = 0.75,
                                      th_style: float = 0.80, th_constr: float = 0.70):
        """
        Crée des goals non préemptifs selon le type :
        - concept -> learn_concept::<label>
        - term    -> learn_term::<label> (ou learn_concept si tu n'as pas d'action 'learn_term')
        - style   -> learn_style::<label> (ajustement de voix)
        - construction -> learn_construction::<label>
        """
        arch = arch or self.arch
        for it in items:
            if it.kind == "concept" and it.score >= th_concept and not self._known(arch, it.label):
                gid = f"learn_concept::{it.label}"
                arch.planner.ensure_goal(gid, f"Apprendre le concept « {it.label} »", priority=0.72, tags=["background"])
                arch.planner.add_action_step(gid, "learn_concept", {"concept": it.label}, priority=0.70)
                self._log_tracker(arch, gid, it)
            elif it.kind == "term" and it.score >= th_term and not self._known(arch, it.label):
                # si tu n'as pas d'action 'learn_term', retombe sur learn_concept
                action = "learn_term" if hasattr(getattr(arch, "io", None), "_h_learn_term") else "learn_concept"
                gid = f"{action}::{it.label}"
                arch.planner.ensure_goal(gid, f"Apprendre le terme « {it.label} »", priority=0.68, tags=["background"])
                arch.planner.add_action_step(gid, action, {"term": it.label}, priority=0.66)
                self._log_tracker(arch, gid, it)
            elif it.kind == "style" and it.score >= th_style:
                gid = f"learn_style::{it.label}"
                arch.planner.ensure_goal(gid, f"Comprendre le style « {it.label} »", priority=0.60, tags=["style"])
                arch.planner.add_action_step(gid, "learn_style", {"style": it.label, "evidence": it.evidence}, priority=0.58)
                self._log_tracker(arch, gid, it)
            elif it.kind == "construction" and it.score >= th_constr:
                gid = f"learn_construction::{it.label}"
                arch.planner.ensure_goal(gid, f"Étudier la construction « {it.label} »", priority=0.60, tags=["linguistics"])
                arch.planner.add_action_step(gid, "learn_construction", {"construction": it.label}, priority=0.58)
                self._log_tracker(arch, gid, it)

    def _known(self, arch, label: str) -> bool:
        # connu si présent en skills/ontology (tu peux étendre)
        try:
            skills_path = getattr(arch, "skills_path", "data/skills.json")
            if os.path.exists(skills_path):
                data = json.load(open(skills_path, "r", encoding="utf-8")) or {}
                if label in data: return True
        except Exception:
            pass
        try:
            onto = getattr(arch, "ontology", None)
            if onto and hasattr(onto, "has_concept"):
                return bool(onto.has_concept(label))
        except Exception:
            pass
        return False
