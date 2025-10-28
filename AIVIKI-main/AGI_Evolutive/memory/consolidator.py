import json
import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from AGI_Evolutive.utils.jsonsafe import json_sanitize
from AGI_Evolutive.utils.llm_service import try_call_llm_dict


LOGGER = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class OnlineLogisticRegression:
    """Minimal logistic regression with online updates."""

    def __init__(self, weights: Optional[Dict[str, float]] = None, lr: float = 0.15):
        self.weights: Dict[str, float] = weights or {}
        self.lr = lr

    def predict_proba(self, features: Dict[str, float]) -> float:
        score = 0.0
        for name, value in features.items():
            if not value:
                continue
            score += self.weights.get(name, 0.0) * value
        return _sigmoid(score)

    def update(self, features: Dict[str, float], label: float, sample_weight: float = 1.0) -> None:
        prediction = self.predict_proba(features)
        gradient = (label - prediction) * sample_weight
        if not gradient:
            return
        for name, value in features.items():
            if not value:
                continue
            current = self.weights.get(name, 0.0)
            self.weights[name] = current + self.lr * gradient * value

    def to_state(self) -> Dict[str, Any]:
        return {"weights": self.weights, "lr": self.lr}

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "OnlineLogisticRegression":
        return cls(weights=state.get("weights", {}), lr=state.get("lr", 0.15))


@dataclass
class ExponentialTrend:
    value: float = 0.0
    alpha: float = 0.3

    def update(self, observation: float) -> float:
        self.value = self.alpha * observation + (1.0 - self.alpha) * self.value
        return self.value

    def to_state(self) -> Dict[str, Any]:
        return {"value": self.value, "alpha": self.alpha}

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "ExponentialTrend":
        return cls(value=state.get("value", 0.0), alpha=state.get("alpha", 0.3))


class StreamingTopics:
    def __init__(self, counts: Optional[Dict[str, float]] = None, decay: float = 0.92):
        self.counts: Dict[str, float] = counts or {}
        self.decay = decay

    def decay_once(self) -> None:
        if not self.counts:
            return
        for word in list(self.counts.keys()):
            self.counts[word] *= self.decay
            if self.counts[word] < 0.1:
                del self.counts[word]

    def update(self, tokens: Iterable[str]) -> None:
        for token in tokens:
            self.counts[token] = self.counts.get(token, 0.0) + 1.0

    def top(self, k: int = 5) -> List[str]:
        if not self.counts:
            return []
        ranked = sorted(self.counts.items(), key=lambda item: item[1], reverse=True)
        return [word for word, _ in ranked[:k]]

    def to_state(self) -> Dict[str, Any]:
        return {"counts": self.counts, "decay": self.decay}

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "StreamingTopics":
        return cls(counts=state.get("counts", {}), decay=state.get("decay", 0.92))


class Consolidator:
    """Adaptive consolidator that learns online from recent memories."""

    POSITIVE_KEYWORDS = {"lesson", "lessons", "learned", "retenir", "leçon", "rappel"}
    PRAISE_KEYWORDS = {"bravo", "bien", "good", "merci", "awesome", "great", "génial"}
    ERROR_KEYWORDS = {"error", "erreur", "oops", "raté", "failure"}

    def __init__(self, memory_store, state_path: str = "data/consolidator.json"):
        self.memory = memory_store
        self.path = state_path
        self.state: Dict[str, Any] = {
            "last_ts": 0.0,
            "lessons": [],
            "lesson_model": {"weights": {}, "lr": 0.15},
            "lesson_threshold": 0.55,
            "feedback": {
                "error": {"trend": {"value": 0.0, "alpha": 0.35}, "threshold": 0.6},
                "praise": {"trend": {"value": 0.0, "alpha": 0.35}, "threshold": 0.55},
            },
            "topics": {"counts": {}, "decay": 0.92},
        }
        self._load()
        self._ensure_state_defaults()

    def _ensure_state_defaults(self) -> None:
        self.state.setdefault("last_ts", 0.0)
        self.state.setdefault("lessons", [])
        self.state.setdefault("lesson_model", {"weights": {}, "lr": 0.15})
        self.state.setdefault("lesson_threshold", 0.55)
        feedback = self.state.setdefault("feedback", {})
        feedback.setdefault("error", {"trend": {"value": 0.0, "alpha": 0.35}, "threshold": 0.6})
        feedback.setdefault("praise", {"trend": {"value": 0.0, "alpha": 0.35}, "threshold": 0.55})
        self.state.setdefault("topics", {"counts": {}, "decay": 0.92})

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    self.state = json.load(fh)
            except Exception:
                self.state = {
                    "last_ts": 0.0,
                    "lessons": [],
                    "lesson_model": {"weights": {}, "lr": 0.15},
                    "lesson_threshold": 0.55,
                    "feedback": {
                        "error": {"trend": {"value": 0.0, "alpha": 0.35}, "threshold": 0.6},
                        "praise": {"trend": {"value": 0.0, "alpha": 0.35}, "threshold": 0.55},
                    },
                    "topics": {"counts": {}, "decay": 0.92},
                }

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(json_sanitize(self.state), fh, ensure_ascii=False, indent=2)

    def run_once_now(self) -> Dict[str, Any]:
        last_ts = self.state.get("last_ts", 0.0)
        recents = [
            m
            for m in self.memory.get_recent_memories(200)
            if m.get("ts", 0) > last_ts
        ]
        if not recents:
            return {"lessons": [], "processed": 0, "proposals": []}

        lesson_model = OnlineLogisticRegression.from_state(self.state.get("lesson_model", {}))
        lesson_threshold = float(self.state.get("lesson_threshold", 0.55))

        error_state = self.state.setdefault("feedback", {}).setdefault(
            "error", {"trend": {"value": 0.0, "alpha": 0.35}, "threshold": 0.6}
        )
        praise_state = self.state.setdefault("feedback", {}).setdefault(
            "praise", {"trend": {"value": 0.0, "alpha": 0.35}, "threshold": 0.55}
        )
        error_trend = ExponentialTrend.from_state(error_state.get("trend", {}))
        praise_trend = ExponentialTrend.from_state(praise_state.get("trend", {}))
        error_threshold = float(error_state.get("threshold", 0.6))
        praise_threshold = float(praise_state.get("threshold", 0.55))

        topics_tracker = StreamingTopics.from_state(self.state.get("topics", {}))
        topics_tracker.decay_once()

        lessons: List[str] = []
        lesson_candidates: List[Tuple[float, Dict[str, Any]]] = []
        interactions: List[Dict[str, Any]] = []

        for mem in recents:
            kind = (mem.get("kind") or "").lower()
            text = (mem.get("text") or mem.get("content") or "").strip()
            features = self._lesson_features(kind, text)
            probability = lesson_model.predict_proba(features)
            lesson_candidates.append((probability, mem))
            label = self._infer_lesson_label(kind, text)
            if label is not None:
                weight = 1.0 if label > 0.5 else 0.7
                lesson_model.update(features, label, weight)

            if kind == "interaction":
                interactions.append(mem)

            tokens = self._tokenize_for_topics(text)
            if tokens:
                topics_tracker.update(tokens)

            lowered = text.lower()
            error_signal = 1.0 if ("error" in kind or any(word in lowered for word in self.ERROR_KEYWORDS)) else 0.0
            praise_signal = 1.0 if any(word in lowered for word in self.PRAISE_KEYWORDS) else 0.0
            error_trend.update(error_signal)
            praise_trend.update(praise_signal)

        lesson_candidates.sort(key=lambda item: item[0], reverse=True)

        for probability, mem in lesson_candidates:
            if len(lessons) >= 5:
                break
            text = (mem.get("text") or mem.get("content") or "").strip()
            if not text:
                continue
            lesson_text = text if len(text) <= 180 else text[:177].rstrip() + "…"
            if lesson_text not in lessons and probability >= lesson_threshold:
                lessons.append(lesson_text)

        if len(lessons) < 3 and interactions:
            last_msg = (interactions[-1].get("text") or "").strip()
            if last_msg:
                summary = last_msg if len(last_msg) <= 120 else last_msg[:117].rstrip() + "…"
                lessons.append(f"Revenir sur l'interaction récente: {summary}")

        top_topics = topics_tracker.top(5)
        if top_topics and len(lessons) < 5:
            topic_sentence = ", ".join(top_topics)
            if topic_sentence and topic_sentence not in lessons:
                lessons.append(f"Sujets dominants: {topic_sentence}")

        proposals: List[Dict[str, Any]] = []
        if error_trend.value > error_threshold:
            proposals.append(
                {
                    "type": "update",
                    "path": ["persona", "tone"],
                    "value": "careful",
                    "reason": f"Tendance d'erreurs élevée (score={error_trend.value:.2f})",
                }
            )
            error_threshold = min(0.9, 0.5 * error_threshold + 0.5 * error_trend.value)
        else:
            error_threshold = max(0.3, 0.8 * error_threshold + 0.2 * error_trend.value)

        if praise_trend.value > praise_threshold:
            proposals.append(
                {
                    "type": "reinforce",
                    "path": ["strategies", "positive_feedback"],
                    "value": True,
                    "reason": f"Tendance de feedback positif élevée (score={praise_trend.value:.2f})",
                }
            )
            praise_threshold = min(0.9, 0.5 * praise_threshold + 0.5 * praise_trend.value)
        else:
            praise_threshold = max(0.25, 0.8 * praise_threshold + 0.2 * praise_trend.value)

        if lesson_candidates:
            accepted = [prob for prob, _ in lesson_candidates if prob >= lesson_threshold]
            if accepted:
                lesson_threshold = 0.7 * lesson_threshold + 0.3 * min(accepted)
            else:
                lesson_threshold = max(0.25, lesson_threshold * 0.92)

        llm_response = self._llm_recommendations(
            lesson_candidates,
            topics_tracker.top(5),
            error_trend.value,
            praise_trend.value,
            fallback_lessons=lessons,
            fallback_proposals=proposals,
        )

        if isinstance(llm_response, Mapping):
            llm_lessons = self._coerce_llm_lessons(llm_response.get("lessons"))
            if llm_lessons:
                lessons = llm_lessons

        max_ts = max((m.get("ts", 0) for m in recents), default=last_ts)
        self.state["last_ts"] = max(last_ts, max_ts)
        if lessons:
            self.state.setdefault("lessons", []).extend(lessons)
            self.state["lessons"] = self.state["lessons"][-200:]

        self.state["lesson_model"] = lesson_model.to_state()
        self.state["lesson_threshold"] = lesson_threshold
        error_state["trend"] = error_trend.to_state()
        error_state["threshold"] = error_threshold
        praise_state["trend"] = praise_trend.to_state()
        praise_state["threshold"] = praise_threshold
        self.state["topics"] = topics_tracker.to_state()

        self._save()
        result: Dict[str, Any] = {
            "lessons": lessons,
            "processed": len(recents),
            "proposals": proposals,
        }
        if isinstance(llm_response, Mapping):
            result["llm"] = dict(llm_response)
        return result

    # ------------------------------------------------------------------
    # LLM integration helpers

    def _candidate_snapshot(
        self,
        probability: float,
        memory: Mapping[str, Any],
    ) -> Dict[str, Any]:
        text = (memory.get("text") or memory.get("content") or "").strip()
        snippet = text[:280]
        metadata = memory.get("metadata") if isinstance(memory.get("metadata"), Mapping) else {}
        safe_meta = {
            key: metadata[key]
            for key in ("kind", "source", "channel", "location")
            if metadata.get(key) is not None
        }
        return {
            "id": memory.get("id"),
            "kind": memory.get("kind"),
            "probability": round(float(probability), 4),
            "ts": memory.get("ts"),
            "text": snippet,
            "metadata": safe_meta,
        }

    def _llm_recommendations(
        self,
        lesson_candidates: Sequence[Tuple[float, Mapping[str, Any]]],
        topics: Sequence[str],
        error_trend: float,
        praise_trend: float,
        *,
        fallback_lessons: Sequence[str],
        fallback_proposals: Sequence[Mapping[str, Any]],
    ) -> Optional[Mapping[str, Any]]:
        if not lesson_candidates:
            return None

        payload = {
            "candidates": [
                self._candidate_snapshot(prob, mem)
                for prob, mem in list(lesson_candidates)[:6]
            ],
            "topics": list(topics),
            "signals": {
                "error_trend": error_trend,
                "praise_trend": praise_trend,
            },
            "fallback": {
                "lessons": list(fallback_lessons)[:5],
                "proposals": [dict(p) for p in fallback_proposals],
            },
        }

        response = try_call_llm_dict(
            "memory_consolidator",
            input_payload=payload,
            logger=LOGGER,
        )
        if isinstance(response, Mapping):
            return dict(response)
        return None

    def _coerce_llm_lessons(
        self,
        lessons: Any,
        *,
        limit: int = 3,
    ) -> List[str]:
        if lessons is None:
            return []
        results: List[str] = []
        if isinstance(lessons, Mapping):
            lessons = lessons.get("items")
        if isinstance(lessons, Iterable) and not isinstance(lessons, (str, bytes)):
            for entry in lessons:
                if isinstance(entry, Mapping):
                    title = str(entry.get("title") or entry.get("lesson") or "").strip()
                    action = str(entry.get("action") or entry.get("follow_up") or "").strip()
                    if title and action:
                        results.append(f"{title} — {action}")
                    elif title:
                        results.append(title)
                    elif action:
                        results.append(action)
                elif isinstance(entry, str):
                    stripped = entry.strip()
                    if stripped:
                        results.append(stripped)
                if len(results) >= limit:
                    break
        return results[:limit]

    @staticmethod
    def _lesson_features(kind: str, text: str) -> Dict[str, float]:
        lowered = text.lower()
        words = lowered.split()
        length = len(words)
        char_len = len(text)
        features: Dict[str, float] = {
            "bias": 1.0,
            "kind_reflection": 1.0 if kind in {"reflection", "lesson"} else 0.0,
            "kind_interaction": 1.0 if kind == "interaction" else 0.0,
            "length_norm": min(length / 80.0, 1.5),
            "char_length_norm": min(char_len / 400.0, 1.5),
            "has_question": 1.0 if "?" in text else 0.0,
            "has_exclaim": 1.0 if "!" in text else 0.0,
            "has_bullet": 1.0 if any(token.startswith(("-", "*")) for token in text.split()) else 0.0,
            "keyword_positive": 1.0 if any(token in Consolidator.POSITIVE_KEYWORDS for token in words) else 0.0,
            "keyword_error": 1.0 if any(token in Consolidator.ERROR_KEYWORDS for token in words) else 0.0,
            "sentence_count": min(text.count(".") + text.count("!"), 4),
        }
        uppercase_chars = sum(1 for ch in text if ch.isupper())
        features["upper_ratio"] = (uppercase_chars / char_len) if char_len else 0.0
        return features

    @staticmethod
    def _infer_lesson_label(kind: str, text: str) -> Optional[float]:
        if not text:
            return 0.0
        lowered = text.lower()
        if kind in {"reflection", "lesson"}:
            return 1.0
        if "lesson" in lowered or "leçon" in lowered or "retenir" in lowered:
            return 1.0
        if kind == "interaction" and len(text) < 200:
            return 0.0
        if len(text) < 40:
            return 0.0
        if kind in {"error", "warning"}:
            return 0.0
        return None

    @staticmethod
    def _tokenize_for_topics(text: str) -> List[str]:
        if not text:
            return []
        tokens: List[str] = []
        for raw in text.lower().split():
            clean = "".join(ch for ch in raw if ch.isalpha())
            if len(clean) >= 4:
                tokens.append(clean)
        return tokens
