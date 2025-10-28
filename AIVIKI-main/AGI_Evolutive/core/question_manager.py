
# core/question_manager.py
"""Gestionnaire de questions proactives / active learning."""

from __future__ import annotations

import math
import time
import uuid
from collections import deque
import re
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import logging

from AGI_Evolutive.utils.jsonsafe import json_sanitize
from AGI_Evolutive.utils.llm_service import try_call_llm_dict


LOGGER = logging.getLogger(__name__)

try:  # numpy may be unavailable in lightweight environments
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover - runtime fallback
    np = None  # type: ignore


class _FeatureHasher:
    """Encode context / metadata into a small dense feature vector."""

    def __init__(self, dim: int = 24) -> None:
        if dim < 8:
            raise ValueError("Feature dimension trop faible pour l'encodage")
        self.dim = dim

    def encode(
        self,
        payload: Dict[str, Any],
        *,
        now: float,
        novelty: float,
        asked_recently: Dict[str, float],
    ) -> Any:
        meta = payload.get("meta") or {}
        if np is None:
            x = [0.0] * self.dim
        else:
            x = np.zeros(self.dim, dtype=float)

        # Bias
        x[0] = 1.0

        severity = float(meta.get("severity", payload.get("score", 0.5)))
        severity = max(0.0, min(1.0, severity))
        x[1] = severity
        x[2] = severity * severity

        ts = float(meta.get("ts") or now)
        elapsed = max(0.0, now - ts)
        # Recence normalisée (<=1)
        x[3] = math.exp(-elapsed / 3600.0)

        # Cooldown déjà respecté ?
        text = payload.get("text", "")
        last_asked = asked_recently.get(text, 0.0)
        delta = max(0.0, now - last_asked)
        x[4] = math.exp(-delta / 1800.0)

        # Mesure de nouveauté contrôlée
        x[5] = max(0.0, min(1.5, novelty))

        # Hachage léger sur type / source / topic
        cursor = 6
        cursor = self._hash_feature(cursor, x, "type", payload.get("type"))
        cursor = self._hash_feature(cursor, x, "topic", meta.get("topic"))
        cursor = self._hash_feature(cursor, x, "source", meta.get("source"))

        # Flag sur présence d'une suggestion explicite
        x[min(self.dim - 1, cursor)] = 1.0 if meta.get("explicit") else 0.0
        return x

    def _hash_feature(self, cursor: int, x: Any, name: str, value: Optional[str]) -> int:
        if value:
            bucket = 1 + (hash((name, value)) % (self.dim - cursor - 1))
            idx = min(self.dim - 2, cursor + bucket)
            x[idx] += 1.0
            return idx
        return cursor


class LinearBanditScorer:
    """Petit LinUCB avec dérive et forgetting factor."""

    def __init__(
        self,
        dim: int,
        *,
        alpha: float = 0.4,
        ridge: float = 1.0,
        forget: float = 0.98,
    ) -> None:
        self.dim = dim
        self.alpha = alpha
        self.forget = forget
        if np is not None:
            self.A = np.eye(dim, dtype=float) * ridge
            self.b = np.zeros(dim, dtype=float)
            self._last_theta = np.zeros(dim, dtype=float)
        else:
            self.A = None
            self.b = None
            self._last_theta = None
        self._last_decay = time.time()

    def _apply_decay(self) -> None:
        if np is None or self.forget >= 1.0:
            return
        now = time.time()
        if now - self._last_decay < 30.0:
            return
        # Décroissance exponentielle
        factor = self.forget ** ((now - self._last_decay) / 30.0)
        self.A *= factor
        self.b *= factor
        self._last_decay = now

    def predict(self, features: Any, *, base: float) -> float:
        if np is None or self.A is None or self.b is None:
            return base
        self._apply_decay()
        try:
            theta = np.linalg.solve(self.A, self.b)
            self._last_theta = theta
        except np.linalg.LinAlgError:
            theta = self._last_theta
        mean = float(features @ theta)
        try:
            inv_A_x = np.linalg.solve(self.A, features)
            ucb = math.sqrt(max(0.0, float(features @ inv_A_x)))
        except np.linalg.LinAlgError:
            ucb = 0.0
        score = mean + self.alpha * ucb
        # Mélange léger avec le score historique
        return 0.6 * score + 0.4 * base

    def update(self, features: Any, reward: float) -> None:
        if np is None or self.A is None or self.b is None:
            return
        if features.shape != (self.dim,):
            raise ValueError("Dimension de features inattendue")
        self._apply_decay()
        x = features.reshape(-1, 1)
        self.A += x @ x.T
        self.b += reward * features



def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


class QuestionManager:
    """Centralise la génération de questions à forte valeur informationnelle.

    Le gestionnaire maintient une *question bank* priorisée à partir
    d'incertitudes observées, garde en mémoire les questions déjà posées
    récemment et propose un petit lot de questions courtes et univoques.
    """

    QUESTION_LIBRARY: Dict[str, Iterable[str]] = {
        "goal_focus": (
            "Quel est l'objectif prioritaire que tu veux atteindre ?",
            "Sur quelle mission dois-je concentrer mon attention en premier ?",
        ),
        "constraint": (
            "Quelle est la contrainte non négociable à respecter ?",
            "Y a-t-il une échéance précise à ne pas manquer ?",
        ),
        "evidence": (
            "As-tu un exemple ou un document de référence à me partager ?",
            "Peux-tu citer un cas concret pour illustrer ton attente ?",
        ),
        "success_metric": (
            "Quel indicateur me dira que le résultat est satisfaisant ?",
        ),
        "intent_confirmation": (
            "Est-ce que cet objectif reste bien d'actualité pour toi ?",
        ),
    }

    GENERIC_TOPICS = {
        "goal_focus",
        "constraint",
        "evidence",
        "success_metric",
        "intent_confirmation",
    }

    def __init__(self, arch):
        self.arch = arch
        self.pending_questions: List[Dict[str, Any]] = []
        self.question_bank: List[Dict[str, Any]] = []
        self.uncertainty_log: Deque[Dict[str, Any]] = deque(maxlen=200)
        self.asked_recently: Dict[str, float] = {}
        self.question_history: Deque[Dict[str, Any]] = deque(maxlen=200)
        self.feedback_history: Deque[Dict[str, Any]] = deque(maxlen=200)
        self.max_primary = 50
        self.max_immediate = 10
        self._channel_limits = {"primary": self.max_primary, "immediate": self.max_immediate}
        self._channel_block: Dict[str, bool] = {"primary": False, "immediate": False}
        self._overflow: Dict[str, List[Dict[str, Any]]] = {"primary": [], "immediate": []}
        self.last_generated = 0.0
        self.cooldown = 8.0  # secondes minimum entre générations
        self.reask_cooldown = 3600.0
        self._feature_hasher = _FeatureHasher()
        self._bandit = LinearBanditScorer(self._feature_hasher.dim)
        self._novelty_tracker: Dict[str, int] = {}
        self._asked_features: Dict[str, Tuple[float, Tuple[float, ...]]] = {}
        self._llm_stale_threshold = 2 * 3600.0  # seconds (2h)
        self._llm_retry_cooldown = 3600.0
        self._auto_retry_cooldown = 900.0

    # ------------------------------------------------------------------
    # Public API
    def add_question(
        self,
        text: str,
        qtype: str = "custom",
        metadata: Optional[Dict[str, Any]] = None,
        priority: float = 0.6,
    ) -> None:
        """Push une question directement dans la banque."""

        if not text:
            return

        now = time.time()
        base_meta = {**(metadata or {})}
        base_meta.setdefault("ts", now)
        entry = {
            "id": base_meta.get("id") or str(uuid.uuid4()),
            "type": qtype,
            "text": text.strip(),
            "score": _clip(priority),
            "meta": base_meta,
        }
        self._push_to_bank(entry)
        if entry["text"] and not any(entry["text"] == q.get("text") for q in self.pending_questions):
            self._enqueue_pending(entry)

    def record_information_need(
        self,
        topic: str,
        severity: float,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        explicit_question: Optional[str] = None,
    ) -> None:
        """Enregistre une incertitude observée pour enrichir la question bank."""

        ts = time.time()
        entry = {
            "ts": ts,
            "topic": topic,
            "severity": _clip(severity),
            "metadata": metadata or {},
        }
        if explicit_question:
            entry["explicit_question"] = explicit_question
        self.uncertainty_log.append(entry)

        base_meta = {**(metadata or {})}
        base_meta.setdefault("slot", topic)
        base_meta.setdefault("topic", topic)
        focus = self._extract_focus_topic(topic, base_meta)
        if focus:
            base_meta["focus"] = focus

        llm_questions = self._llm_suggest_questions(
            topic,
            severity=entry["severity"],
            base_meta=base_meta,
            explicit_question=explicit_question,
        )

        if llm_questions:
            for suggestion in llm_questions:
                self._add_llm_question(entry, base_meta, suggestion)
            return

        question_text = (explicit_question or "").strip()
        if not question_text:
            question_text = self._design_question(topic, base_meta, focus_override=focus)
        if not question_text:
            return
        self._add_llm_question(
            entry,
            base_meta,
            {"text": question_text, "reason": base_meta.get("focus")},
        )

    def _add_llm_question(
        self,
        base_entry: Dict[str, Any],
        base_meta: Dict[str, Any],
        suggestion: Dict[str, Any],
    ) -> None:
        text = str((suggestion or {}).get("text", "")).strip()
        if not text:
            return
        severity = float(base_entry.get("severity", 0.0))
        ts = float(base_entry.get("ts", time.time()))
        payload_meta = {
            **base_meta,
            "severity": severity,
            "ts": ts,
            "explicit": bool(base_entry.get("explicit_question")),
        }
        reason = suggestion.get("reason")
        if reason:
            payload_meta["llm_reason"] = reason
        payload = {
            "id": payload_meta.get("id") or str(uuid.uuid4()),
            "type": payload_meta.get("type", base_meta.get("topic")),
            "text": text,
            "score": _clip(0.4 + 0.6 * severity),
            "meta": payload_meta,
        }
        insights = suggestion.get("insights")
        if isinstance(insights, list) and insights:
            payload_meta["insights"] = [str(item) for item in insights if str(item).strip()]
        self._push_to_bank(payload)
        if payload["text"] and not any(payload["text"] == q.get("text") for q in self.pending_questions):
            self._enqueue_pending(payload)

    def _llm_suggest_questions(
        self,
        topic: str,
        *,
        severity: float,
        base_meta: Dict[str, Any],
        explicit_question: Optional[str],
    ) -> Optional[List[Dict[str, Any]]]:
        payload = {
            "topic": topic,
            "severity": severity,
            "metadata": base_meta,
            "explicit_question": explicit_question,
            "recent_questions": [
                {
                    "text": item.get("question"),
                    "ts": item.get("ts"),
                }
                for item in list(self.question_history)[-10:]
            ],
            "pending": [q.get("text") for q in self.pending_questions],
        }
        response = try_call_llm_dict(
            "question_manager",
            input_payload=payload,
            logger=LOGGER,
            max_retries=2,
        )
        if not response:
            return None
        questions = response.get("questions")
        if not isinstance(questions, list):
            return None
        cleaned: List[Dict[str, Any]] = []
        for entry in questions:
            if not isinstance(entry, dict):
                continue
            text = str(entry.get("text", "")).strip()
            if not text:
                continue
            cleaned.append(entry)
        return cleaned or None

    def maybe_generate_questions(self) -> None:
        """Sélectionne un petit lot de questions à proposer."""

        now = time.time()
        if now - self.last_generated < self.cooldown:
            return

        self._refresh_from_models()
        self._purge_stale_bank(now)
        ranked = sorted(self.question_bank, key=lambda x: x.get("score", 0.0), reverse=True)

        for candidate in ranked:
            text = candidate.get("text", "")
            if not text:
                continue
            last_asked = self.asked_recently.get(text, 0.0)
            if now - last_asked < self.reask_cooldown:
                continue
            if any(text == q.get("text") for q in self.pending_questions):
                continue
            if not self._enqueue_pending(candidate):
                continue
            if any(self.is_channel_blocked(ch) for ch in self._channel_limits):
                break

        if self.pending_questions:
            self.last_generated = now

    def pop_questions(self) -> List[Dict[str, Any]]:
        """Retourne les questions planifiées et trace l'historique."""

        now = time.time()
        out = []
        for q in self.pending_questions:
            meta = q.get("meta") or {}
            if meta.get("state") == "answered":
                continue
            if meta.get("served_at"):
                continue
            text = q.get("text")
            if not text:
                continue
            qid = q.get("id") or meta.get("id") or str(uuid.uuid4())
            meta["served_at"] = now
            meta.setdefault("state", "pending")
            self.asked_recently[text] = now
            features = q.get("_features")
            if features:
                self._asked_features[qid] = (now, tuple(features))
            self.question_history.append({"ts": now, "id": qid, "question": text, "meta": dict(meta)})
            self._novelty_tracker[text] = self._novelty_tracker.get(text, 0) + 1
            out.append(q)
        return out

    def resolve_question(self, question_id: str, answer: Optional[str] = None) -> bool:
        """Marque une question comme résolue et met à jour l'état de blocage."""

        if not question_id:
            return False
        resolved = False
        for idx in range(len(self.pending_questions) - 1, -1, -1):
            q = self.pending_questions[idx]
            qid = q.get("id") or q.get("meta", {}).get("id")
            if qid != question_id:
                continue
            channel = self._channel_for_entry(q)
            meta = q.get("meta") or {}
            meta["state"] = "answered"
            meta["answered_at"] = time.time()
            if answer:
                meta["answer"] = answer
            self.pending_questions.pop(idx)
            resolved = True
            self._update_channel_block(channel)
            break
        if not resolved:
            return False
        try:
            self._replenish_from_overflow()
        except Exception:
            pass
        return True

    def register_outcome(
        self,
        question_id: str,
        *,
        success: bool,
        latency: float = 0.0,
        reward: Optional[float] = None,
        notes: Optional[str] = None,
    ) -> None:
        """Met à jour le bandit avec un feedback utilisateur."""

        if question_id not in self._asked_features:
            return

        asked_ts, features = self._asked_features.pop(question_id)
        if np is not None:
            features_vec = np.asarray(features, dtype=float)
        else:
            features_vec = None
        if reward is None:
            base = 1.0 if success else -0.3
            delay_penalty = min(0.6, max(0.0, latency) / 120.0)
            freshness_penalty = min(0.3, max(0.0, time.time() - asked_ts - 30.0) / 600.0)
            reward_value = base - delay_penalty - freshness_penalty
        else:
            reward_value = reward
        if features_vec is not None:
            self._bandit.update(features_vec, reward_value)
        self.feedback_history.append(
            {
                "ts": time.time(),
                "question_id": question_id,
                "success": success,
                "latency": latency,
                "reward": reward_value,
                "notes": notes or "",
            }
        )

    # ------------------------------------------------------------------
    # Internals
    def _refresh_from_models(self) -> None:
        """Alimente la question bank depuis les modules connexes."""

        # Intent model uncertainties → confirmation questions
        intent_model = getattr(self.arch, "intent_model", None)
        if intent_model and hasattr(intent_model, "pending_clarifications"):
            for item in intent_model.pending_clarifications():
                question, priority, meta = item
                base_meta = meta or {}
                enriched_meta = {
                    **base_meta,
                    "ts": time.time(),
                    "source": base_meta.get("source", "intent"),
                }
                self._push_to_bank(
                    {
                        "id": enriched_meta.get("id", str(uuid.uuid4())),
                        "type": enriched_meta.get("type", "intent_confirmation"),
                        "text": question,
                        "score": _clip(priority),
                        "meta": enriched_meta,
                    }
                )

        # Metacognitive uncertainty → clarifier objectif / contraintes
        meta = getattr(self.arch, "metacognition", None)
        try:
            awareness = 1.0 - float(meta.metacognitive_states.get("awareness_level", 0.5))
        except Exception:
            awareness = 0.5
        if awareness > 0.45:
            self.record_information_need(
                "goal_focus",
                awareness,
                metadata={"source": "metacog", "type": "goal_focus", "topic": "goal_focus"},
            )

    def _push_to_bank(self, payload: Dict[str, Any]) -> None:
        text = payload.get("text", "").strip()
        if not text:
            return

        score, features = self._score_with_bandit(payload)
        payload["score"] = score
        payload["_features"] = tuple(features)

        existing = next((q for q in self.question_bank if q.get("text") == text), None)
        if existing:
            existing["score"] = max(existing.get("score", 0.0), payload.get("score", 0.0))
            existing.setdefault("meta", {}).update(payload.get("meta", {}))
            existing["_features"] = payload["_features"]
            return

        self.question_bank.append(payload)
        if len(self.question_bank) > 50:
            # prune the lowest scored entries
            self.question_bank = sorted(self.question_bank, key=lambda x: x.get("score", 0.0), reverse=True)[:50]

    def _enqueue_pending(self, payload: Dict[str, Any]) -> bool:
        channel = self._channel_for_entry(payload)
        meta = payload.setdefault("meta", {})
        now = time.time()
        meta.setdefault("channel", channel)
        meta.setdefault("queued_at", now)
        meta.pop("served_at", None)
        meta.setdefault("state", "pending")

        if self._channel_count(channel) >= self._channel_limits[channel]:
            meta.setdefault("channel", channel)
            self._overflow[channel].append(payload)
            self._update_channel_block(channel)
            return False

        self.pending_questions.append(payload)
        self.pending_questions.sort(key=lambda item: (item.get("meta", {}).get("queued_at", 0.0)))
        self._update_channel_block(channel)
        return True

    def _channel_for_entry(self, payload: Dict[str, Any]) -> str:
        meta = payload.get("meta") or {}
        channel = meta.get("channel")
        if isinstance(channel, str) and channel in self._channel_limits:
            return channel
        source = str(meta.get("source") or payload.get("source") or "").lower()
        qtype = str(payload.get("type") or "").lower()
        immediacy = meta.get("immediacy")
        try:
            immediacy_val = float(immediacy)
        except (TypeError, ValueError):
            immediacy_val = 0.0
        if immediacy_val >= 0.75:
            return "immediate"
        if any(token in source for token in ("trigger", "reflex", "alert", "threat")):
            return "immediate"
        if qtype in {"trigger", "urgent", "reflex"}:
            return "immediate"
        return "primary"

    def _channel_count(self, channel: str) -> int:
        return sum(1 for q in self.pending_questions if self._channel_for_entry(q) == channel)

    def _update_channel_block(self, channel: str, blocked: Optional[bool] = None) -> None:
        if blocked is None:
            blocked = self._channel_count(channel) >= self._channel_limits[channel] or bool(
                self._overflow[channel]
            )
        prev = self._channel_block.get(channel)
        new_val = bool(blocked)
        self._channel_block[channel] = new_val
        if prev != new_val:
            logger = getattr(self.arch, "logger", None)
            if logger and hasattr(logger, "write"):
                try:
                    logger.write(
                        "questions.block_state",
                        channel=channel,
                        blocked=new_val,
                        pending=self._channel_count(channel),
                        overflow=len(self._overflow[channel]),
                    )
                except Exception:
                    pass

    def _replenish_from_overflow(self) -> None:
        for channel in self._channel_limits:
            queue = self._overflow[channel]
            if not queue:
                continue
            while queue and self._channel_count(channel) < self._channel_limits[channel]:
                entry = queue.pop(0)
                entry_meta = entry.setdefault("meta", {})
                entry_meta["channel"] = channel
                entry_meta.pop("served_at", None)
                entry_meta["state"] = "pending"
                entry_meta["queued_at"] = time.time()
                self.pending_questions.append(entry)
            self.pending_questions.sort(key=lambda item: (item.get("meta", {}).get("queued_at", 0.0)))
            self._update_channel_block(channel)

    def blocked_channels(self) -> List[str]:
        return [name for name, blocked in self._channel_block.items() if blocked]

    def is_channel_blocked(self, channel: str) -> bool:
        return bool(self._channel_block.get(channel, False))

    def is_blocked(self) -> bool:
        return any(self._channel_block.values())

    def attempt_auto_answers(self, channels: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
        """Essaie de proposer une réponse en autonomie pour les questions bloquantes."""

        if channels is not None:
            target_channels = {ch for ch in channels if ch in self._channel_limits}
        else:
            target_channels = set(self._channel_limits)
        attempts: List[Dict[str, Any]] = []
        now = time.time()
        for question in list(self.pending_questions):
            meta = question.setdefault("meta", {})
            channel = self._channel_for_entry(question)
            if channel not in target_channels:
                continue
            if meta.get("state") == "answered":
                continue
            last_auto_attempt = self._as_timestamp(meta.get("auto_attempted"))
            if last_auto_attempt and now - last_auto_attempt < self._auto_retry_cooldown:
                continue
            meta["auto_attempted"] = now
            suggestion = self._attempt_auto_answer_for(question)
            if not suggestion and self._should_attempt_llm(meta):
                suggestion = self._attempt_llm_answer_for(question)
            if suggestion:
                meta.setdefault("auto_suggestions", []).append(suggestion)
                attempts.append({"question": question, "suggestion": suggestion})
                if suggestion.get("source") == "llm":
                    meta["auto_resolution"] = "llm"
                    meta["resolved_by"] = "llm"
                    qid = question.get("id") or meta.get("id")
                    answer_text = str(suggestion.get("text") or "").strip()
                    if answer_text:
                        meta["answer"] = answer_text
                    meta["answered_at"] = time.time()
                    self._record_llm_answer(question, suggestion)
                    if qid:
                        try:
                            self.resolve_question(qid, answer=answer_text)
                        except Exception:
                            pass
        return attempts

    def _should_attempt_llm(self, meta: Dict[str, Any]) -> bool:
        if not isinstance(meta, dict):
            return False
        if meta.get("state") == "answered":
            return False
        suggestions = meta.get("auto_suggestions")
        if suggestions and any((s or {}).get("source") == "llm" for s in suggestions if isinstance(s, dict)):
            return False
        now = time.time()
        last_attempt_val = self._as_timestamp(meta.get("llm_auto_attempted"))
        if last_attempt_val and now - last_attempt_val < self._llm_retry_cooldown:
            return False
        base_ts_val = self._as_timestamp(meta.get("served_at") or meta.get("queued_at") or meta.get("ts"))
        if not base_ts_val:
            return False
        if now - base_ts_val < self._llm_stale_threshold:
            return False
        return True

    def _attempt_llm_answer_for(self, question: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        text = (question.get("text") or "").strip()
        if not text:
            return None
        meta = question.setdefault("meta", {})
        meta["llm_auto_attempted"] = time.time()
        base_ts_val = self._as_timestamp(meta.get("queued_at") or meta.get("ts"))
        payload: Dict[str, Any] = {
            "question": text,
            "type": question.get("type"),
            "metadata": json_sanitize(meta),
            "elapsed_seconds": float(max(0.0, time.time() - (base_ts_val or time.time()))),
        }
        focus = meta.get("focus") or meta.get("topic")
        if focus:
            payload["focus"] = focus
        active_goal = self._active_goal_description()
        if active_goal:
            payload["active_goal"] = active_goal
        recent_history = list(self.question_history)[-5:]
        if recent_history:
            payload["recent_questions"] = [
                {
                    "question": item.get("question"),
                    "answer": (item.get("meta") or {}).get("answer"),
                    "ts": item.get("ts"),
                }
                for item in recent_history
            ]
        response = try_call_llm_dict(
            "question_auto_answer",
            input_payload=json_sanitize(payload),
            logger=LOGGER,
            max_retries=1,
        )
        if not response:
            return None
        answer_text = str(response.get("answer") or response.get("text") or "").strip()
        if not answer_text:
            return None
        suggestion: Dict[str, Any] = {
            "text": answer_text,
            "source": "llm",
            "created_at": time.time(),
            "payload": json_sanitize(response),
        }
        confidence = response.get("confidence")
        try:
            confidence_val = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence_val = None
        if confidence_val is not None:
            suggestion["confidence"] = _clip(confidence_val)
        for key in ("concepts", "keywords", "insights"):
            value = response.get(key)
            if isinstance(value, list):
                cleaned: List[Any] = []
                for item in value:
                    sanitized = json_sanitize(item)
                    if sanitized:
                        cleaned.append(sanitized)
                if cleaned:
                    suggestion[key] = cleaned
        notes = response.get("notes")
        if isinstance(notes, str) and notes.strip():
            suggestion["notes"] = notes.strip()
        return suggestion

    def _active_goal_description(self) -> Optional[str]:
        goals = getattr(self.arch, "goals", None)
        if not goals or not hasattr(goals, "get_active_goal"):
            return None
        try:
            active = goals.get_active_goal()
        except Exception:
            return None
        if not active:
            return None
        if isinstance(active, dict):
            for key in ("description", "title", "name"):
                value = active.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        if isinstance(active, str):
            return active.strip()
        return None

    def _record_llm_answer(self, question: Dict[str, Any], suggestion: Dict[str, Any]) -> None:
        memory = getattr(self.arch, "memory", None)
        if not memory or not hasattr(memory, "add_memory"):
            return
        entry = {
            "kind": "question_auto_answer",
            "question": question.get("text"),
            "answer": suggestion.get("text"),
            "confidence": suggestion.get("confidence"),
            "source": suggestion.get("source"),
            "concepts": suggestion.get("concepts"),
            "keywords": suggestion.get("keywords"),
            "insights": suggestion.get("insights"),
            "metadata": {
                "question_id": question.get("id") or (question.get("meta") or {}).get("id"),
                "topic": (question.get("meta") or {}).get("topic"),
                "type": question.get("type"),
            },
            "notes": suggestion.get("notes", ""),
            "ts": time.time(),
        }
        try:
            memory.add_memory(json_sanitize(entry))
        except Exception:
            pass

    def _as_timestamp(self, value: Any) -> float:
        try:
            return float(value) if value is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    def _attempt_auto_answer_for(self, question: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        text = (question.get("text") or "").strip()
        if not text:
            return None
        tokens = self._tokenize(text)
        if not tokens:
            return None
        store = getattr(getattr(self.arch, "memory", None), "store", None)
        if not store or not hasattr(store, "get_recent_memories"):
            return None
        candidates = store.get_recent_memories(200)
        best_score = 0.0
        best_item: Optional[Dict[str, Any]] = None
        for item in reversed(candidates):
            snippet = str(item.get("text") or item.get("content") or "")
            if not snippet.strip():
                continue
            score = self._overlap_score(tokens, snippet)
            if score > best_score:
                best_score = score
                best_item = item
        if not best_item or best_score < 0.35:
            return None
        return {
            "text": best_item.get("text") or best_item.get("content") or "",
            "source": best_item.get("kind", "memory"),
            "memory_id": best_item.get("id"),
            "score": best_score,
        }

    def _tokenize(self, text: str) -> List[str]:
        return [tok for tok in re.findall(r"[a-z0-9éèêàùûôîïç]+", text.lower()) if len(tok) > 2]

    def _overlap_score(self, tokens: List[str], snippet: str) -> float:
        if not tokens:
            return 0.0
        words = self._tokenize(snippet)
        if not words:
            return 0.0
        overlap = len(set(tokens) & set(words))
        return overlap / max(len(set(tokens)), 1)

    def _design_question(
        self,
        topic: str,
        metadata: Optional[Dict[str, Any]],
        *,
        focus_override: Optional[str] = None,
    ) -> Optional[str]:
        focus = focus_override or self._extract_focus_topic(topic, metadata)
        if focus:
            contextual = self._contextual_question_for(topic, focus)
            if contextual:
                return contextual

        library = self.QUESTION_LIBRARY.get(topic)
        if not library:
            if focus:
                return f"Quelles précisions peux-tu apporter à propos de « {focus} » ?"
            return None

        # Choisir la formulation la moins utilisée récemment
        history = {entry["question"] for entry in self.question_history}
        for candidate in library:
            if candidate not in history:
                return candidate if not focus else candidate.replace(" ?", f" concernant « {focus} » ?")
        fallback = next(iter(library), None)
        if fallback and focus:
            return fallback.replace(" ?", f" pour « {focus} » ?")
        return fallback

    def _sanitize_focus(self, text: Optional[str]) -> Optional[str]:
        if not text or not isinstance(text, str):
            return None
        candidate = text.strip().strip("«»\"'“”‘’ ")
        if len(candidate) < 3:
            return None
        normalized = candidate.lower()
        if normalized in self.GENERIC_TOPICS:
            return None
        return candidate

    def _extract_focus_topic(
        self, topic: Optional[str], metadata: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        meta = metadata or {}
        for key in ("focus", "concept", "topic", "subject", "goal", "goal_topic"):
            focus = self._sanitize_focus(meta.get(key))
            if focus:
                return focus

        # Active goal description
        arch_goals = getattr(self.arch, "goals", None)
        if arch_goals and hasattr(arch_goals, "get_active_goal"):
            try:
                active = arch_goals.get_active_goal()
            except Exception:
                active = None
            if active:
                for key in ("description", "desc", "title"):
                    focus = self._focus_from_text(active.get(key))
                    if focus:
                        return focus

        # Metadata textual hints
        for key in ("user_msg", "text", "surface", "hint"):
            focus = self._focus_from_text(meta.get(key))
            if focus:
                return focus

        sanitized_topic = self._sanitize_focus(topic)
        if sanitized_topic:
            return sanitized_topic
        return None

    def _focus_from_text(self, text: Optional[str]) -> Optional[str]:
        if not text or not isinstance(text, str):
            return None
        # Priorité aux segments entre guillemets
        quote_match = re.search(r"[«\"“”'‘’]\s*([^«\"“”'‘’]+?)\s*[»\"“”'‘’]", text)
        if quote_match:
            focus = self._sanitize_focus(quote_match.group(1))
            if focus:
                return focus

        lowered = text.lower()
        trigger_phrases = ["comprendre", "apprendre", "clarifier", "maîtriser"]
        for phrase in trigger_phrases:
            if phrase in lowered:
                start = lowered.find(phrase) + len(phrase)
                snippet = text[start:]
                snippet = snippet.split(".")[0]
                snippet = snippet.split("?")[0]
                snippet = snippet.split("!")[0]
                snippet = snippet.split(",")[0]
                snippet = snippet.strip(" :;-—")
                if snippet:
                    words = snippet.split()
                    if words:
                        candidate = " ".join(words[:8])
                        focus = self._sanitize_focus(candidate)
                        if focus:
                            return focus
        return None

    def _contextual_question_for(self, topic: str, focus: str) -> Optional[str]:
        clean_focus = self._sanitize_focus(focus)
        if not clean_focus:
            return None
        topic_key = (topic or "").lower()
        if topic_key == "goal_focus":
            return f"Peux-tu préciser ce que tu attends de moi concernant « {clean_focus} » ?"
        if topic_key == "constraint":
            return f"Y a-t-il des contraintes spécifiques liées à « {clean_focus} » que je dois respecter ?"
        if topic_key == "evidence":
            return f"As-tu un exemple concret ou une ressource à propos de « {clean_focus} » ?"
        if topic_key == "success_metric":
            return f"Quel indicateur me permettra de savoir que « {clean_focus} » est accompli ?"
        if topic_key == "intent_confirmation":
            return f"Confirme-tu que je dois poursuivre « {clean_focus} » en priorité ?"
        return f"Quelles informations te semblent importantes concernant « {clean_focus} » ?"

    def _score_with_bandit(self, payload: Dict[str, Any]) -> Tuple[float, Any]:
        base_score = _clip(payload.get("score", 0.5))
        now = time.time()
        text = payload.get("text", "")
        novelty = 1.0 / (1 + self._novelty_tracker.get(text, 0))
        features = self._feature_hasher.encode(
            payload,
            now=now,
            novelty=novelty,
            asked_recently=self.asked_recently,
        )
        if np is not None:
            learned_score = self._bandit.predict(features, base=base_score)
            final_score = _clip(0.5 * base_score + 0.5 * learned_score)
            return final_score, features
        # Fallback sans numpy → score basé sur priorité de base
        return base_score, tuple(float(x) for x in features)

    def _purge_stale_bank(self, now: float) -> None:
        freshness_horizon = 8 * 3600  # 8h
        filtered: List[Dict[str, Any]] = []
        for item in self.question_bank:
            meta = item.get("meta") or {}
            ts = meta.get("ts") or 0.0
            if ts and now - float(ts) > freshness_horizon:
                continue
            filtered.append(item)
        self.question_bank = filtered
