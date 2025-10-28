import hashlib
import logging
import math
import os
import json
import random
import time
from typing import List, Dict, Any, Optional, Iterable, Tuple, Mapping
from collections import Counter

from AGI_Evolutive.utils.jsonsafe import json_sanitize
from AGI_Evolutive.utils.llm_service import (
    LLMIntegrationError,
    LLMUnavailableError,
    get_llm_manager,
    is_llm_enabled,
)


STOPWORDS = {
    "et", "le", "la", "de", "des", "les", "un", "une", "du", "en", "à",
    "au", "aux", "ou", "dans", "pour", "par", "sur", "avec", "ce", "cet",
    "cette", "ces", "que", "qui", "quoi", "où", "ne", "pas", "plus", "moins",
    "très", "bien", "mal", "est", "sont", "été", "être", "avoir", "je", "tu",
    "il", "elle", "nous", "vous", "ils", "elles", "moi", "toi", "suis"
}

TRIVIAL_GOAL_TOKENS = {
    "bonjour",
    "salut",
    "bonsoir",
    "hello",
    "hi",
    "hey",
    "coucou",
    "yo",
    "allo",
}

POSITIVE_WORDS = {"succès", "réussi", "brillant", "clair", "fiable", "robuste", "juste"}
NEGATIVE_WORDS = {"échec", "raté", "confus", "incertain", "risque", "problème", "erreur"}

DOMAIN_KEYWORDS = {
    "language": {
        "langue", "language", "texte", "lexique", "mot", "phrase", "syntaxe",
        "grammaire", "communication", "discours", "écriture", "traduction"
    },
    "humans": {
        "humain", "humains", "personne", "personnes", "émotion", "relation",
        "social", "interaction", "comportement", "psychologie", "culture"
    },
    "planning": {
        "plan", "planification", "objectif", "stratégie", "tâche", "priorité",
        "étape", "roadmap", "deadline", "agenda", "exécution", "process"
    }
}

TTL_OPTIONS = (3, 7, 14, 30)


LOGGER = logging.getLogger(__name__)


def _llm_enabled() -> bool:
    return is_llm_enabled()


def _llm_manager():
    return get_llm_manager()


class OnlineLinear:
    """Simple bounded online logistic model with optional L2 regularisation."""

    def __init__(
        self,
        feature_names: Iterable[str],
        bounds: Tuple[float, float] = (0.0, 1.0),
        lr: float = 0.05,
        l2: float = 0.001,
        max_grad: float = 0.3,
        warmup: int = 12,
        init_weight: float = 0.1,
    ):
        self.feature_names: List[str] = list(feature_names)
        self.index = {name: idx for idx, name in enumerate(self.feature_names)}
        self.bounds = (min(bounds[0], bounds[1]), max(bounds[0], bounds[1]))
        self.lr = max(1e-4, float(lr))
        self.l2 = max(0.0, float(l2))
        self.max_grad = max(0.01, float(max_grad))
        self.bias = 0.0
        self.init_weight = float(init_weight)
        self.weights: List[float] = [float(init_weight) for _ in self.feature_names]
        self.update_count = 0
        self.warmup = max(1, int(warmup))

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    def _ensure_features(self, features: Dict[str, float]):
        for name in features.keys():
            if name not in self.index:
                self.index[name] = len(self.feature_names)
                self.feature_names.append(name)
                self.weights.append(float(self.init_weight))

    def predict(self, features: Dict[str, float]) -> float:
        if not features:
            return 0.5
        total = self.bias
        for name, value in features.items():
            if name in self.index:
                total += self.weights[self.index[name]] * value
        return self._sigmoid(total)

    def update(self, features: Dict[str, float], target: float):
        self._ensure_features(features)
        target = max(0.0, min(1.0, float(target)))
        pred = self.predict(features)
        error = pred - target
        for name, value in features.items():
            idx = self.index[name]
            grad = error * value + self.l2 * self.weights[idx]
            grad = max(-self.max_grad, min(self.max_grad, grad))
            new_weight = self.weights[idx] - self.lr * grad
            self.weights[idx] = max(self.bounds[0], min(self.bounds[1], new_weight))
        bias_grad = max(-self.max_grad, min(self.max_grad, error))
        self.bias -= self.lr * bias_grad
        max_abs = max(abs(self.bounds[0]), abs(self.bounds[1]))
        bias_limit = 2.0 * max_abs if max_abs > 0.0 else 1.0
        self.bias = max(-bias_limit, min(bias_limit, self.bias))
        self.update_count += 1

    def confidence(self) -> float:
        return max(0.0, min(1.0, self.update_count / float(self.warmup)))

    def to_state(self) -> Dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "weights": list(self.weights),
            "bias": float(self.bias),
            "update_count": int(self.update_count),
        }

    @classmethod
    def from_state(
        cls,
        payload: Optional[Dict[str, Any]],
        bounds: Tuple[float, float] = (0.0, 1.0),
        lr: float = 0.05,
        l2: float = 0.001,
        max_grad: float = 0.3,
        warmup: int = 12,
        init_weight: float = 0.1,
    ) -> "OnlineLinear":
        feature_names: Iterable[str] = payload.get("feature_names") if isinstance(payload, dict) else []
        model = cls(
            feature_names or [],
            bounds=bounds,
            lr=lr,
            l2=l2,
            max_grad=max_grad,
            warmup=warmup,
            init_weight=init_weight,
        )
        if isinstance(payload, dict):
            weights = payload.get("weights")
            if isinstance(weights, list) and len(weights) == len(model.feature_names):
                model.weights = [max(model.bounds[0], min(model.bounds[1], float(w))) for w in weights]
            bias = payload.get("bias")
            if isinstance(bias, (int, float)):
                model.bias = float(bias)
            update_count = payload.get("update_count")
            if isinstance(update_count, int):
                model.update_count = max(0, update_count)
        return model

class MetaCognition:
    """
    Évalue l'incertitude, repère des lacunes, génère des learning-goals,
    et enregistre des réflexions (inner-monologue).
    Persiste état dans data/metacog.json
    """
    def __init__(self, memory_store, planner, self_model, data_dir: str = "data"):
        self.memory = memory_store
        self.planner = planner
        self.self_model = self_model
        self.path = os.path.join(data_dir, "metacog.json")
        self.state = {
            "last_assessment_ts": 0.0,
            "open_questions": [],     # [{"q": "<question>", "topic": "<theme>", "priority": 0.0-1.0}]
            "uncertainty": 0.5,       # 0..1
            "domains": {},            # {"language": {"confidence": 0.6, "gaps": ["lexique", "raisonnement"]}}
            "history": [],            # log compact
            "models": {},             # états des modèles online
            "bandits": {},            # paramètres TS sur TTL
            "feature_stats": {}
        }
        self._load()
        self.state.setdefault("recent_goals", {})
        self.state.setdefault("last_logged_goal_digest", "")
        self._init_models()
        self._last_llm_assessment: Optional[Mapping[str, Any]] = None

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.state = json.load(f)
            except Exception:
                pass

    def _init_models(self):
        models_state = self.state.setdefault("models", {})
        self.uncertainty_model = OnlineLinear.from_state(
            models_state.get("uncertainty"),
            bounds=(0.0, 1.0),
            lr=0.04,
            l2=0.002,
            max_grad=0.25,
            warmup=16,
            init_weight=0.08,
        )
        if not self.uncertainty_model.feature_names:
            base_features = (
                "question_ratio",
                "answer_ratio",
                "error_ratio",
                "lesson_ratio",
                "novelty_ratio",
                "success_rate",
                "reflection_sentiment",
                "token_diversity",
            )
            self.uncertainty_model = OnlineLinear(base_features, bounds=(0.0, 1.0), lr=0.04, l2=0.002, max_grad=0.25)

        goal_state = models_state.get("goal_priority")
        self.goal_model = OnlineLinear.from_state(
            goal_state,
            bounds=(0.0, 1.0),
            lr=0.05,
            l2=0.002,
            max_grad=0.3,
            warmup=10,
            init_weight=0.12,
        )
        if not self.goal_model.feature_names:
            goal_features = ("uncertainty", "domain_confidence", "hazard", "novelty")
            self.goal_model = OnlineLinear(goal_features, bounds=(0.0, 1.0), lr=0.05, l2=0.002, max_grad=0.3)

        self.domain_bandits = self.state.setdefault("bandits", {})

    def _llm_assessment(self, payload: Dict[str, Any]) -> Optional[Mapping[str, Any]]:
        if not _llm_enabled():
            return None

        try:
            response = _llm_manager().call_dict(
                "meta_cognition",
                input_payload=payload,
            )
        except (LLMUnavailableError, LLMIntegrationError):
            LOGGER.debug("LLM meta-cognition unavailable", exc_info=True)
            return None

        if not isinstance(response, Mapping):
            return None

        self._last_llm_assessment = dict(response)
        return self._last_llm_assessment

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.state.setdefault("models", {})
        self.state["models"]["uncertainty"] = self.uncertainty_model.to_state()
        self.state["models"]["goal_priority"] = self.goal_model.to_state()
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(json_sanitize(self.state), f, ensure_ascii=False, indent=2)

    # --------- Feature extraction helpers ---------
    def _initial_domain_state(self, domain: str) -> Dict[str, Any]:
        if domain not in self.state["domains"]:
            self.state["domains"][domain] = {
                "confidence": 0.5,
                "gaps": [],
                "revision_ttl": TTL_OPTIONS[1],
                "hazard": 0.3,
            }
        return self.state["domains"][domain]

    def _compute_sentiment(self, text: str) -> float:
        tokens = [t.strip(".,;:!?()[]\"").lower() for t in text.split()]
        pos = sum(1 for t in tokens if t in POSITIVE_WORDS)
        neg = sum(1 for t in tokens if t in NEGATIVE_WORDS)
        if pos == 0 and neg == 0:
            return 0.0
        return (pos - neg) / float(pos + neg)

    def _extract_stats(self, recents: List[Dict[str, Any]]) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "total": len(recents),
            "questions": 0,
            "answers": 0,
            "errors": 0,
            "lessons": 0,
            "reflections": 0,
            "successes": 0,
            "success_count": 0,
            "token_counter": Counter(),
            "question_tokens": Counter(),
            "lesson_tokens": Counter(),
            "error_tokens": Counter(),
            "length_sum": 0,
            "novel_tokens": set(),
            "sentiment_sum": 0.0,
            "last_error_ts": 0.0,
        }

        now = time.time()
        for mem in recents:
            kind = (mem.get("kind") or "").lower()
            text = (mem.get("text") or "")
            tokens = [t.strip(".,;:!?()[]\"").lower() for t in text.split() if t]
            alpha_tokens = [t for t in tokens if t.isalpha() and t not in STOPWORDS]
            stats["token_counter"].update(alpha_tokens)
            stats["length_sum"] += len(tokens)
            if kind in ("question", "interaction") and text.strip().endswith("?"):
                stats["questions"] += 1
                stats["question_tokens"].update(alpha_tokens)
            if kind in ("lesson", "reflection"):
                stats["lessons"] += 1
                stats["lesson_tokens"].update(alpha_tokens)
            if "error" in kind or mem.get("ok") is False:
                stats["errors"] += 1
                stats["error_tokens"].update(alpha_tokens)
                ts = mem.get("ts")
                if isinstance(ts, (int, float)):
                    stats["last_error_ts"] = max(stats["last_error_ts"], float(ts))
            if any(k in kind for k in ("lesson", "insight")) or ("answer" in text.lower()):
                stats["answers"] += 1
            if kind == "reflection":
                stats["reflections"] += 1
                stats["sentiment_sum"] += self._compute_sentiment(text)
            success = mem.get("success")
            score = mem.get("score")
            if isinstance(success, bool):
                stats["success_count"] += 1
                stats["successes"] += 1 if success else 0
            if isinstance(score, (int, float)):
                stats.setdefault("score_sum", 0.0)
                stats.setdefault("score_count", 0)
                stats["score_sum"] += float(score)
                stats["score_count"] += 1
            if mem.get("novel"):
                stats["novel_tokens"].update(alpha_tokens)

        stats["token_total"] = sum(stats["token_counter"].values())
        stats["unique_tokens"] = len(stats["token_counter"])
        stats["novel_tokens"] = set(stats["novel_tokens"])
        stats["now"] = now
        return stats

    def _build_features(self, stats: Dict[str, Any]) -> Dict[str, float]:
        interactions = max(1, stats["questions"] + stats["answers"] + stats["lessons"])
        token_total = max(1, stats["token_total"])
        unique_tokens = max(1, stats["unique_tokens"])
        reflection_count = max(1, stats["reflections"]) if stats["reflections"] else 1
        success_rate = stats["successes"] / float(stats["success_count"]) if stats["success_count"] else 0.0
        score_avg = stats.get("score_sum", 0.0) / stats.get("score_count", 1) if stats.get("score_count") else 0.0
        avg_sentiment = stats["sentiment_sum"] / reflection_count if stats["reflections"] else 0.0
        novelty_ratio = len(stats["novel_tokens"]) / float(unique_tokens)
        token_diversity = unique_tokens / float(token_total)

        features = {
            "question_ratio": stats["questions"] / float(interactions),
            "answer_ratio": stats["answers"] / float(interactions),
            "error_ratio": stats["errors"] / float(interactions),
            "lesson_ratio": stats["lessons"] / float(interactions),
            "novelty_ratio": max(0.0, min(1.0, novelty_ratio)),
            "success_rate": max(0.0, min(1.0, success_rate)),
            "score_avg": max(0.0, min(1.0, score_avg / 10.0)),
            "reflection_sentiment": max(-1.0, min(1.0, avg_sentiment)),
            "token_diversity": max(0.0, min(1.0, token_diversity)),
            "length_density": stats["length_sum"] / (stats["total"] or 1) / 40.0,
        }
        features["length_density"] = max(0.0, min(1.0, features["length_density"]))
        net_questions = max(0.0, stats["questions"] - stats["answers"])
        features["question_pressure"] = max(0.0, min(1.0, net_questions / float(interactions)))
        features["error_energy"] = max(0.0, min(1.0, stats["errors"] / float(stats["total"] or 1)))
        features["error_ratio_sq"] = features["error_ratio"] ** 2
        features["lesson_ratio_sq"] = features["lesson_ratio"] ** 2

        # Domain concentration features
        for domain, vocab in DOMAIN_KEYWORDS.items():
            domain_hits = sum(stats["token_counter"].get(tok, 0) for tok in vocab)
            features[f"focus_{domain}"] = domain_hits / float(token_total)

        return features

    def _heuristic_uncertainty(self, stats: Dict[str, Any]) -> float:
        interactions = max(1, stats["questions"] + stats["answers"])
        baseline = 0.5 + 0.35 * ((stats["questions"] - stats["answers"]) / interactions)
        baseline += 0.15 * min(1.0, stats["errors"] / float(interactions))
        baseline -= 0.1 * min(0.5, stats["lessons"] / float(interactions * 1.5))
        return max(0.0, min(1.0, baseline))

    def _target_uncertainty(self, stats: Dict[str, Any]) -> float:
        interactions = max(1, stats["questions"] + stats["answers"] + stats["lessons"])
        err_component = stats["errors"] / float(interactions)
        pressure = max(0.0, stats["questions"] - stats["answers"]) / float(interactions)
        sentiment = 0.0
        if stats["reflections"]:
            sentiment = max(-1.0, min(1.0, stats["sentiment_sum"] / stats["reflections"]))
        sentiment_component = 0.2 * (0.5 - sentiment * 0.5)
        return max(0.0, min(1.0, 0.6 * err_component + 0.4 * pressure + sentiment_component))

    def _domain_hazard(self, domain: str, stats: Dict[str, Any]) -> float:
        domain_state = self._initial_domain_state(domain)
        last_error_ts = stats["last_error_ts"]
        now = stats["now"]
        decay = now - domain_state.get("last_error_ts", 0.0) if domain_state.get("last_error_ts") else None
        if last_error_ts:
            domain_state["last_error_ts"] = last_error_ts
        if decay is None:
            decay = now - self.state.get("last_assessment_ts", now)
        horizon_days = max(0.1, decay / 86400.0)
        recent_error_intensity = stats["errors"] / float(max(1, stats["questions"] + stats["answers"]))
        hazard = max(0.05, min(0.95, 0.4 * recent_error_intensity + 0.2 / horizon_days))
        domain_state["hazard"] = hazard
        return hazard

    def _select_ttl(self, domain: str, hazard: float) -> int:
        bandit = self.domain_bandits.setdefault(domain, {"success": {}, "fail": {}})
        best_score = None
        best_opt = TTL_OPTIONS[1]
        for option in TTL_OPTIONS:
            alpha = bandit["success"].get(option, 1.0)
            beta = bandit["fail"].get(option, 1.0)
            sample = random.betavariate(alpha, beta)
            if best_score is None or sample > best_score:
                best_score = sample
                best_opt = option

        reward = max(0.0, min(1.0, 1.0 - hazard))
        bandit["success"][best_opt] = bandit["success"].get(best_opt, 1.0) + reward
        bandit["fail"][best_opt] = bandit["fail"].get(best_opt, 1.0) + (1.0 - reward)
        self._initial_domain_state(domain)["revision_ttl"] = best_opt
        return best_opt

    def _compute_domain_confidence(
        self, domain: str, features: Dict[str, float], uncertainty: float, stats: Dict[str, Any]
    ) -> float:
        focus = features.get(f"focus_{domain}", 0.0)
        novelty = features.get("novelty_ratio", 0.0)
        success_rate = features.get("success_rate", 0.0)
        hazard = self._domain_hazard(domain, stats)
        base = max(0.1, min(0.9, 0.7 * success_rate + 0.3 * focus - 0.4 * uncertainty))
        base = base * (1.0 - 0.5 * hazard) + 0.1 * (1.0 - novelty)
        return max(0.05, min(0.95, base))

    def _domain_gaps(self, domain: str, stats: Dict[str, Any]) -> List[str]:
        gaps: List[Tuple[str, float]] = []
        vocab = DOMAIN_KEYWORDS.get(domain, set())
        question_tokens = stats["question_tokens"]
        lesson_tokens = stats["lesson_tokens"]
        error_tokens = stats["error_tokens"]
        for token, freq in stats["token_counter"].most_common(60):
            if token in STOPWORDS or len(token) <= 2:
                continue
            if vocab and token not in vocab and freq < 2:
                continue
            q = question_tokens.get(token, 0)
            l = lesson_tokens.get(token, 0)
            e = error_tokens.get(token, 0)
            target = 1.0 if (q > l or e > 0) else 0.0
            features = {
                "question_ratio": q / float(stats["questions"] or 1),
                "lesson_ratio": l / float(stats["lessons"] or 1),
                "error_ratio": e / float(stats["errors"] or 1),
                "novelty": 1.0 if token in stats["novel_tokens"] else 0.0,
            }
            score = 0.6 * features["question_ratio"] + 0.3 * features["error_ratio"] - 0.2 * features["lesson_ratio"]
            if target > 0:
                score += 0.15
            if vocab and token not in vocab:
                score *= 0.7
            gaps.append((token, max(0.0, score)))
        gaps.sort(key=lambda x: x[1], reverse=True)
        return [token for token, _ in gaps[:5]]

    # --------- Analyse & incertitude ---------
    def assess_understanding(self, horizon: int = 150) -> Dict[str, Any]:
        recents = self.memory.get_recent_memories(n=horizon) if self.memory else []
        stats = self._extract_stats(recents)
        features = self._build_features(stats)
        heur_unc = self._heuristic_uncertainty(stats)
        model_pred = self.uncertainty_model.predict(features)
        weight = 0.3 + 0.6 * self.uncertainty_model.confidence()
        uncertainty = (1 - weight) * heur_unc + weight * model_pred

        target = self._target_uncertainty(stats)
        self.uncertainty_model.update(features, target)

        domains: Dict[str, Dict[str, Any]] = {}
        for domain in DOMAIN_KEYWORDS.keys():
            conf = self._compute_domain_confidence(domain, features, uncertainty, stats)
            hazard = self._initial_domain_state(domain)["hazard"]
            ttl = self._select_ttl(domain, hazard)
            domains[domain] = {
                "confidence": conf,
                "gaps": self._domain_gaps(domain, stats),
                "revision_ttl": ttl,
                "hazard": hazard,
            }

        history_entry = {
            "ts": stats["now"],
            "uncertainty": float(uncertainty),
            "heuristic": heur_unc,
            "model_pred": model_pred,
            "features": features,
        }
        self.state.setdefault("history", []).append(history_entry)
        if len(self.state["history"]) > 300:
            self.state["history"] = self.state["history"][-300:]

        previous_uncertainty = self.state.get("uncertainty", 0.5)
        drift = abs(previous_uncertainty - uncertainty)
        if drift > 0.25 and self.memory:
            self.memory.add_memory(
                {
                    "kind": "metacog_drift",
                    "text": f"Variation forte de l'incertitude ({previous_uncertainty:.2f} -> {uncertainty:.2f})",
                    "delta": drift,
                    "ts": stats["now"],
                }
            )

        self.state["uncertainty"] = float(uncertainty)
        self.state["domains"].update(domains)
        self.state["last_assessment_ts"] = stats["now"]
        self.state["feature_stats"] = features
        self._save()
        assessment = {
            "uncertainty": float(uncertainty),
            "domains": domains,
            "stats": {
                "questions": stats["questions"],
                "answers": stats["answers"],
                "errors": stats["errors"],
                "lessons": stats["lessons"],
            },
        }

        llm_payload = {
            "uncertainty": float(uncertainty),
            "domains": domains,
            "stats": assessment["stats"],
        }
        llm_bundle = self._llm_assessment(llm_payload)
        if llm_bundle:
            assessment["llm"] = llm_bundle
            knowledge_gaps = llm_bundle.get("knowledge_gaps")
            if isinstance(knowledge_gaps, list):
                meta_domain = domains.setdefault(
                    "llm_insights",
                    {"confidence": 0.5, "gaps": [], "revision_ttl": TTL_OPTIONS[1], "hazard": 0.3},
                )
                for gap in knowledge_gaps:
                    topic = gap.get("topic") if isinstance(gap, Mapping) else None
                    if isinstance(topic, str) and topic and topic not in meta_domain["gaps"]:
                        meta_domain["gaps"].append(topic)
            self.state["domains"].update(domains)
            self.state["last_llm_assessment"] = llm_bundle
        return assessment

    # --------- Génération de learning goals ---------
    def propose_learning_goals(self, max_goals: int = 3) -> List[Dict[str, Any]]:
        assessment = self.assess_understanding()
        uncertainty = assessment["uncertainty"]
        candidate_goals = []
        for domain, d in assessment["domains"].items():
            hazard = d.get("hazard", 0.3)
            novelty = max(0.0, 1.0 - d.get("confidence", 0.5))
            goal_features = {
                "uncertainty": uncertainty,
                "domain_confidence": d.get("confidence", 0.5),
                "hazard": hazard,
                "novelty": novelty,
            }
            priority_base = self.goal_model.predict(goal_features)
            target_priority = 0.8 if novelty > 0.4 or hazard > 0.4 else 0.3
            self.goal_model.update(goal_features, target_priority)
            ttl = d.get("revision_ttl", TTL_OPTIONS[1])
            for g in d.get("gaps", []):
                normalized = (g or "").strip().lower()
                if not normalized or len(normalized) < 3:
                    continue
                if normalized in STOPWORDS or normalized in TRIVIAL_GOAL_TOKENS:
                    continue
                prio = max(priority_base, 0.5 * uncertainty + 0.5 * (1.0 - d["confidence"]))
                candidate_goals.append(
                    {
                        "id": f"learn_{domain}_{g}",
                        "desc": f"Comprendre le concept '{g}' dans le domaine {domain}",
                        "priority": prio,
                        "domain": domain,
                        "revision_ttl": ttl,
                    }
                )
        llm_bundle = self._last_llm_assessment or assessment.get("llm")
        if isinstance(llm_bundle, Mapping):
            for entry in llm_bundle.get("learning_goals", []):
                if not isinstance(entry, Mapping):
                    continue
                goal_text = entry.get("goal")
                if not isinstance(goal_text, str) or not goal_text.strip():
                    continue
                impact = str(entry.get("impact", "")).lower()
                impact_priority = {
                    "haut": 0.9,
                    "élevé": 0.9,
                    "eleve": 0.9,
                    "moyen": 0.65,
                    "faible": 0.45,
                    "bas": 0.45,
                }.get(impact, 0.6)
                goal_id = entry.get("id")
                if not isinstance(goal_id, str) or not goal_id:
                    goal_id = "llm_" + hashlib.sha1(goal_text.encode("utf-8")).hexdigest()[:10]
                candidate_goals.append(
                    {
                        "id": goal_id,
                        "desc": goal_text.strip(),
                        "priority": max(0.05, min(1.0, impact_priority)),
                        "domain": entry.get("domain") or entry.get("topic") or "llm",
                        "revision_ttl": entry.get("revision_ttl", TTL_OPTIONS[1]),
                        "source": "llm",
                    }
                )
        candidate_goals.sort(key=lambda x: x["priority"], reverse=True)
        now = time.time()
        recent = self.state.setdefault("recent_goals", {})
        goals: List[Dict[str, Any]] = []
        for candidate in candidate_goals:
            if len(goals) >= max_goals:
                break
            gid = candidate.get("id")
            if not gid:
                continue
            last_ts = float(recent.get(gid, 0.0) or 0.0)
            revision_ttl = candidate.get("revision_ttl")
            cooldown = 1800.0
            if isinstance(revision_ttl, (int, float)) and revision_ttl > 0:
                cooldown = max(cooldown, float(revision_ttl) * 3600.0)
            if now - last_ts < cooldown:
                continue
            goals.append(candidate)

        digest = ""
        if goals:
            for goal in goals:
                gid = goal.get("id")
                if gid:
                    recent[gid] = now
            goal_ids = sorted(goal.get("id", "") for goal in goals)
            digest = hashlib.sha1("|".join(goal_ids).encode("utf-8")).hexdigest()
            self.state["recent_goals"] = recent

        # Planifier des micro-actions pour chaque goal
        for goal in goals:
            if self.planner:
                plan = self.planner.plan_for_goal(goal["id"], goal["desc"])
                steps = plan.get("steps", []) if isinstance(plan, dict) else []
                if not steps:
                    self.planner.add_step(goal["id"], f"Collecter exemples concrets pour '{goal['desc']}'")
                    self.planner.add_step(goal["id"], f"Poser 2 questions ciblées sur '{goal['desc']}'")
                if goal.get("revision_ttl"):
                    self.planner.add_step(goal["id"], f"Réviser après {goal['revision_ttl']} jours")
        # journaliser en mémoire
        should_log_reflection = bool(goals)
        if digest and digest == self.state.get("last_logged_goal_digest"):
            should_log_reflection = False
        if should_log_reflection and digest:
            self.state["last_logged_goal_digest"] = digest
            if self.memory:
                self.memory.add_memory({
                    "kind": "reflection",
                    "text": f"Génération de {len(goals)} learning-goals basés sur incertitude",
                    "goals": goals,
                    "ts": time.time()
                })
        if goals:
            self._save()
        return goals

    # --------- Journal réflexif ---------
    def log_inner_monologue(self, text: str, tags: Optional[List[str]] = None):
        self.memory.add_memory({
            "kind": "reflection",
            "text": text,
            "tags": tags or [],
            "ts": time.time()
        })
