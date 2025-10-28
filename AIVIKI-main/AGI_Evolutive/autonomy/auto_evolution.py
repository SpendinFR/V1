"""Auto-evolution coordination across cognitive modules.

This module wires together long-term memory, metacognition, the skill sandbox,
mechanism store and the self-improver so that any new concept/insight/emotion
can autonomously bloom into persistent intentions, actions and evaluation
signals.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from AGI_Evolutive.autonomy.auto_signals import (
    derive_signals_for_description,
    extract_keywords,
)
from AGI_Evolutive.core.structures.mai import ImpactHypothesis, MAI
from AGI_Evolutive.utils.jsonsafe import json_sanitize
from AGI_Evolutive.utils.llm_service import try_call_llm_dict

try:  # GoalType is optional when goal system is not available
    from AGI_Evolutive.goals import GoalType
except Exception:  # pragma: no cover - optional dependency
    GoalType = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _slugify(text: str, *, prefix: str = "") -> str:
    base = "".join(ch.lower() if ch.isalnum() else "_" for ch in text)
    base = re.sub(r"_+", "_", base).strip("_")
    if not base:
        base = "auto"
    if prefix:
        base = f"{prefix}_{base}" if not base.startswith(prefix) else base
    return base[:96]


@dataclass
class AutoIntention:
    """Intermediate representation used before promotion to skills/MAIs."""

    action_type: str
    description: str
    source: str
    signals: List[Dict[str, Any]] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_emotion: Optional[Dict[str, float]] = None


class AutoEvolutionCoordinator:
    """Promotes new knowledge into long-term autonomous skills and signals."""

    _CANDIDATE_KINDS = (
        "lesson",
        "insight",
        "knowledge",
        "concept",
        "thought",
        "belief",
        "reflection",
        "interaction",
        "emotion",
        "intention",
        "desire",
        "goal",
        "subgoal",
        "plan",
        "skill",
        "ability",
        "prediction",
        "pattern",
    )

    def __init__(
        self,
        *,
        memory: Optional[Any],
        metacog: Optional[Any],
        skill_sandbox: Optional[Any],
        evolution_manager: Optional[Any],
        mechanism_store: Optional[Any],
        self_improver: Optional[Any],
        goals: Optional[Any] = None,
        emotions: Optional[Any] = None,
        interval: float = 45.0,
        modules: Optional[Sequence[Any]] = None,
        signal_registry: Optional[AutoSignalRegistry] = None,
    ) -> None:
        self.memory = memory
        self.metacog = metacog
        self.skill_sandbox = skill_sandbox
        self.evolution_manager = evolution_manager
        self.mechanism_store = mechanism_store
        self.self_improver = self_improver
        self.goals = goals
        self.emotions = emotions
        self.interval = max(5.0, float(interval))
        self.modules: List[Any] = [module for module in modules or [] if module is not None]
        self.signal_registry = signal_registry

        self._lock = threading.RLock()
        self._processed: OrderedDict[str, float] = OrderedDict()
        self._last_ts: float = 0.0
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._goal_links: Dict[str, str] = {}

    # ------------------------------------------------------------------
    def install(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="auto-evolution", daemon=True)
        self._thread.start()
        logger.info("AutoEvolutionCoordinator installé et démarré")

    def shutdown(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=1.0)

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:
                logger.exception("Erreur dans la boucle auto-évolution")
            self._stop.wait(self.interval)

    # ------------------------------------------------------------------
    def tick(self) -> None:
        intentions = self._collect_intentions()
        if not intentions:
            return
        for intention in intentions:
            evaluation = self._evaluate_intention(intention)
            if not evaluation.get("accepted"):
                continue
            self._promote_intention(intention, evaluation)

    # ------------------------------------------------------------------
    def _collect_intentions(self) -> List[AutoIntention]:
        memory = self.memory
        if memory is None:
            return []
        try:
            snapshot = memory.get_all_time_history(
                include_raw=True,
                include_digests=False,
                include_expanded=False,
                include_self_model=False,
                limit_recent=256,
                limit_digests=0,
                since_ts=self._last_ts or None,
            )
        except Exception:
            return []

        timeline = snapshot.get("timeline", {}) if isinstance(snapshot, Mapping) else {}
        combined = timeline.get("combined", []) if isinstance(timeline, Mapping) else []
        knowledge = snapshot.get("knowledge", {}) if isinstance(snapshot, Mapping) else {}

        candidates: List[AutoIntention] = []
        for entry in combined:
            intention = self._intention_from_entry(entry)
            if intention is not None:
                candidates.append(intention)

        for section in ("lessons", "beliefs", "completed_goals"):
            records = knowledge.get(section, []) if isinstance(knowledge, Mapping) else []
            for record in records:
                intention = self._intention_from_entry(record, source=section)
                if intention is not None:
                    candidates.append(intention)

        return candidates

    # ------------------------------------------------------------------
    def _intention_from_entry(
        self,
        entry: Mapping[str, Any],
        *,
        source: Optional[str] = None,
    ) -> Optional[AutoIntention]:
        if not isinstance(entry, Mapping):
            return None
        kind = str(entry.get("kind") or source or "generic")
        lowered_kind = kind.lower()
        if not any(token in lowered_kind for token in self._CANDIDATE_KINDS):
            return None

        text_fields = [
            entry.get("text"),
            entry.get("content"),
            entry.get("description"),
            entry.get("summary"),
            entry.get("title"),
            entry.get("idea"),
            entry.get("lesson"),
            entry.get("observation"),
        ]
        description = next((str(value) for value in text_fields if value), None)
        if not description:
            return None

        ts_value = entry.get("ts") or entry.get("timestamp") or entry.get("created_at")
        try:
            ts = float(ts_value)
        except (TypeError, ValueError):
            ts = time.time()

        key = str(entry.get("id") or f"{kind}:{hash(description)}:{int(ts)}")
        if self._already_processed(key, ts):
            return None

        action_type = _slugify(description, prefix=lowered_kind.split(".")[0])
        tags = entry.get("tags", [])
        requirements = extract_keywords(description, " ".join(tags))
        signals = derive_signals_for_description(
            action_type,
            description,
            requirements=requirements,
            hints=tags,
        )
        metadata = {
            "source_kind": kind,
            "raw": json_sanitize(dict(entry)),
            "keywords": requirements,
        }
        emotion_snapshot = self._current_emotion()
        self._remember_processed(key, ts)
        self._last_ts = max(self._last_ts, ts)
        return AutoIntention(
            action_type=action_type,
            description=description,
            source=kind,
            signals=signals,
            requirements=requirements,
            metadata=metadata,
            source_emotion=emotion_snapshot,
        )

    # ------------------------------------------------------------------
    def _already_processed(self, key: str, ts: float) -> bool:
        with self._lock:
            if key in self._processed:
                return True
            # prune old entries lazily
            while len(self._processed) > 512:
                self._processed.popitem(last=False)
            return False

    def _remember_processed(self, key: str, ts: float) -> None:
        with self._lock:
            self._processed[key] = ts

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _current_emotion(self) -> Optional[Dict[str, float]]:
        emotions = self.emotions
        if emotions is None:
            return None
        payload: Dict[str, float] = {}
        intensity = getattr(emotions, "emotional_intensity", None)
        if isinstance(intensity, (int, float)):
            payload["intensity"] = float(intensity)
        dims = getattr(emotions, "affective_dimensions", None)
        if isinstance(dims, Mapping):
            valence = dims.get("valence_energy", {}).get("valence") if isinstance(dims.get("valence_energy"), Mapping) else None
            tension = dims.get("valence_energy", {}).get("tension") if isinstance(dims.get("valence_energy"), Mapping) else None
            if isinstance(valence, (int, float)):
                payload["valence"] = float(valence)
            if isinstance(tension, (int, float)):
                payload["tension"] = float(tension)
        return payload or None

    # ------------------------------------------------------------------
    def _evaluate_intention(self, intention: AutoIntention) -> Dict[str, Any]:
        metacog = self.metacog
        if metacog and hasattr(metacog, "evaluate_auto_intention"):
            try:
                result = metacog.evaluate_auto_intention(
                    {
                        "action_type": intention.action_type,
                        "description": intention.description,
                        "source": intention.source,
                        "signals": intention.signals,
                        "requirements": intention.requirements,
                        "metadata": intention.metadata,
                        "source_emotion": intention.source_emotion,
                    }
                )
                if isinstance(result, Mapping):
                    return dict(result)
            except Exception:
                logger.exception("Évaluation métacognitive échouée")

        llm_payload = json_sanitize(
            {
                "intention": {
                    "action_type": intention.action_type,
                    "description": intention.description,
                    "source": intention.source,
                    "signals": intention.signals,
                    "requirements": intention.requirements,
                    "metadata": intention.metadata,
                    "source_emotion": intention.source_emotion,
                },
                "metacog_available": bool(metacog),
            }
        )
        response = try_call_llm_dict(
            "autonomy_intention_evaluation",
            input_payload=llm_payload,
            logger=logger,
        )
        if isinstance(response, Mapping):
            def _coerce_float(value: Any, default: float) -> float:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return default

            evaluation = {
                "accepted": bool(response.get("accepted", False)),
                "significance": _coerce_float(
                    response.get("significance", response.get("score", 0.6)),
                    0.6,
                ),
                "alignment": _coerce_float(response.get("alignment", 0.5), 0.5),
                "emotional_drive": _coerce_float(
                    response.get("emotional_drive", (intention.source_emotion or {}).get("intensity", 0.5)),
                    (intention.source_emotion or {}).get("intensity", 0.5) or 0.5,
                ),
            }
            for key, value in response.items():
                if key in evaluation:
                    continue
                evaluation[key] = value
            return evaluation

        # fallback heuristic
        score = 0.5 + 0.05 * len(intention.signals) + 0.02 * len(intention.requirements)
        return {
            "accepted": score >= 0.6,
            "significance": score,
            "alignment": 0.5,
            "emotional_drive": (intention.source_emotion or {}).get("intensity", 0.5),
        }

    # ------------------------------------------------------------------
    def _promote_intention(self, intention: AutoIntention, evaluation: Mapping[str, Any]) -> None:
        self_assessment = self._build_self_assessment(intention, evaluation)

        request_payload = {
            "origin": intention.source,
            "signals": intention.signals,
            "requirements": intention.requirements,
            "metadata": intention.metadata,
            "evaluation": dict(evaluation),
        }
        if self_assessment:
            request_payload["self_assessment"] = self_assessment
        request_view: Optional[Dict[str, Any]] = None
        if self.skill_sandbox and hasattr(self.skill_sandbox, "register_intention"):
            try:
                request_view = self.skill_sandbox.register_intention(
                    action_type=intention.action_type,
                    description=intention.description,
                    payload=request_payload,
                    requirements=intention.requirements,
                )
            except Exception:
                logger.exception("Impossible d'enregistrer l'intention dans le sandbox")

        if self.signal_registry is not None and intention.signals:
            try:
                self.signal_registry.register(
                    intention.action_type,
                    intention.signals,
                    evaluation=evaluation,
                    blueprint=self_assessment,
                    description=intention.description,
                    requirements=intention.requirements,
                    hints=intention.metadata.get("keywords")
                    if isinstance(intention.metadata, Mapping)
                    else None,
                )
            except Exception:
                logger.exception("Impossible d'enregistrer les signaux auto-évolutifs")

        if self.mechanism_store is not None:
            try:
                self._upsert_mechanism(intention, evaluation, request_view, self_assessment)
            except Exception:
                logger.exception("Mise à jour de MechanismStore échouée")

        try:
            self._schedule_metacog_followup(intention, evaluation, self_assessment)
        except Exception:
            logger.exception("Planification métacognitive échouée")

        try:
            self._ensure_goal_alignment(intention, evaluation)
        except Exception:
            logger.exception("Alignement des buts auto-évolutifs impossible")

        try:
            self._log_memory_plan(intention, evaluation, self_assessment)
        except Exception:
            logger.exception("Impossibilité d'enregistrer la trace mémoire auto-évolutive")

        try:
            self._notify_modules(intention, evaluation, self_assessment, request_view)
        except Exception:
            logger.exception("Diffusion inter-modulaire de l'intention auto-évolutive échouée")

        if self.evolution_manager and hasattr(self.evolution_manager, "record_feedback_event"):
            try:
                self.evolution_manager.record_feedback_event(
                    "auto_evolution",
                    label=intention.action_type,
                    success=None,
                    confidence=float(evaluation.get("significance", 0.6)),
                    heuristic="intention_promoted",
                    payload={
                        "signals": intention.signals,
                        "requirements": intention.requirements,
                        "metadata": intention.metadata,
                        "self_assessment": self_assessment,
                    },
                )
            except Exception:
                logger.exception("Trace evolution_manager échouée")

        if self.self_improver and hasattr(self.self_improver, "run_cycle"):
            try:
                self.self_improver.run_cycle(n_candidates=2)
            except Exception:
                logger.exception("SelfImprover n'a pas pu lancer un cycle")

    # ------------------------------------------------------------------
    def _upsert_mechanism(
        self,
        intention: AutoIntention,
        evaluation: Mapping[str, Any],
        request_view: Optional[Mapping[str, Any]],
        self_assessment: Optional[Mapping[str, Any]],
    ) -> None:
        store = self.mechanism_store
        if store is None:
            return
        mai_id = f"auto::{intention.action_type}"
        metadata = {
            "auto_generated": True,
            "signals": intention.signals,
            "requirements": intention.requirements,
            "source": intention.source,
            "evaluation": dict(evaluation),
            "request": json_sanitize(request_view) if request_view else None,
            "self_assessment": json_sanitize(self_assessment) if self_assessment else None,
        }
        existing = None
        try:
            existing = store.get(mai_id)
        except Exception:
            existing = None
        if existing is None:
            mai = MAI(
                id=mai_id,
                title=intention.description[:80],
                summary=intention.description,
                status="draft",
                tags=["auto_evolution", intention.source],
                expected_impact=ImpactHypothesis(
                    trust_delta=0.05,
                    competence_delta=0.1,
                    confidence=float(evaluation.get("significance", 0.6)),
                ),
                metadata=metadata,
            )
            store.add(mai)
        else:
            existing.summary = intention.description
            existing.metadata = {**(existing.metadata or {}), **metadata}
            existing.updated_at = time.time()
            store.update(existing)

    # ------------------------------------------------------------------
    def _schedule_metacog_followup(
        self,
        intention: AutoIntention,
        evaluation: Mapping[str, Any],
        self_assessment: Optional[Mapping[str, Any]],
    ) -> None:
        metacog = self.metacog
        if metacog is None:
            return
        payload = {
            "action_type": intention.action_type,
            "description": intention.description,
            "signals": intention.signals,
            "evaluation": dict(evaluation),
        }
        if self_assessment:
            payload["self_assessment"] = self_assessment
        if hasattr(metacog, "plan_auto_intention_followup"):
            metacog.plan_auto_intention_followup(
                payload,
                dict(evaluation),
                intention.signals,
                self_assessment,
            )
            return

        engine = getattr(metacog, "reflection_engine", None)
        if isinstance(engine, Mapping):
            scheduled = engine.setdefault("scheduled_reflections", [])
            if isinstance(scheduled, list):
                plan = {
                    "ts": time.time(),
                    "kind": "auto_intention_followup",
                    "action_type": intention.action_type,
                    "description": intention.description,
                    "targets": intention.signals,
                    "score": float(evaluation.get("score", evaluation.get("significance", 0.6))),
                    "self_assessment": self_assessment,
                }
                scheduled.append(plan)

    def _ensure_goal_alignment(self, intention: AutoIntention, evaluation: Mapping[str, Any]) -> None:
        goals = self.goals
        if goals is None or not hasattr(goals, "add_goal"):
            return

        keywords = set(intention.metadata.get("keywords", [])) if isinstance(intention.metadata, Mapping) else set()
        criteria = [
            f"{signal['name']} >= {signal.get('target', 0.6)}"
            for signal in intention.signals
            if isinstance(signal, Mapping) and signal.get("name")
        ]
        description = f"Institutionnaliser: {intention.description}"[:180]

        existing_id = self._goal_links.get(intention.action_type)
        if existing_id and hasattr(goals, "metadata"):
            metadata = goals.metadata.get(existing_id)
            if metadata is not None:
                metadata.success_criteria = criteria
                metadata.updated_at = time.time()
                return

        goal_kwargs: Dict[str, Any] = {
            "description": description,
            "criteria": criteria,
            "created_by": "auto_evolution",
            "value": float(evaluation.get("significance", 0.6)),
            "competence": float(evaluation.get("alignment", 0.5)),
            "curiosity": 0.3 + 0.2 * float(evaluation.get("significance", 0.5)),
            "urgency": float(evaluation.get("emotional_drive", 0.4)),
            "structural_seeded": False,
        }

        goal_type = self._resolve_goal_type(keywords)
        if goal_type is not None:
            goal_kwargs["goal_type"] = goal_type

        node = goals.add_goal(**goal_kwargs)
        if node is not None:
            self._goal_links[intention.action_type] = getattr(node, "id", intention.action_type)

    def _resolve_goal_type(self, keywords: Iterable[str]) -> Optional[Any]:
        if GoalType is None:
            return None
        keyword_set = {kw.lower() for kw in keywords}
        if keyword_set.intersection({"relation", "relationship", "social", "empathy", "compassion"}):
            return GoalType.SOCIAL
        if keyword_set.intersection({"learn", "knowledge", "mastery", "improve", "growth"}):
            return GoalType.GROWTH
        if keyword_set.intersection({"plan", "strategy", "reason", "cognitive"}):
            return GoalType.COGNITIVE
        if keyword_set.intersection({"creative", "creation", "imagine", "novel"}):
            return GoalType.CREATIVE
        if keyword_set.intersection({"self", "awareness", "identity", "introspect"}):
            return GoalType.SELF_ACTUALISATION
        if keyword_set.intersection({"survival", "safety", "protection"}):
            return GoalType.SURVIVAL
        return GoalType.GROWTH

    def _log_memory_plan(
        self,
        intention: AutoIntention,
        evaluation: Mapping[str, Any],
        self_assessment: Optional[Mapping[str, Any]],
    ) -> None:
        memory = self.memory
        if memory is None or not hasattr(memory, "add_memory"):
            return
        payload = {
            "kind": "auto_intention_promoted",
            "action_type": intention.action_type,
            "description": intention.description,
            "signals": intention.signals,
            "evaluation": dict(evaluation),
            "source": intention.source,
            "metadata": intention.metadata,
        }
        if self_assessment:
            payload["self_assessment"] = self_assessment
        memory.add_memory("auto_evolution", payload)

    # ------------------------------------------------------------------
    def register_module(self, module: Any) -> None:
        if module is None:
            return
        if module not in self.modules:
            self.modules.append(module)

    def _notify_modules(
        self,
        intention: AutoIntention,
        evaluation: Mapping[str, Any],
        self_assessment: Optional[Mapping[str, Any]],
        request_view: Optional[Mapping[str, Any]],
    ) -> None:
        if not self.modules:
            return
        base_event = {
            "action_type": intention.action_type,
            "description": intention.description,
            "source": intention.source,
            "signals": json_sanitize(intention.signals),
            "requirements": list(intention.requirements),
            "metadata": json_sanitize(intention.metadata),
            "source_emotion": json_sanitize(intention.source_emotion),
            "timestamp": time.time(),
        }
        if isinstance(intention.metadata, Mapping):
            keywords = list(intention.metadata.get("keywords", []))
            if keywords:
                base_event["keywords"] = keywords
        if request_view is not None:
            base_event["skill_request"] = json_sanitize(request_view)
        if self_assessment is not None:
            base_event["self_assessment"] = json_sanitize(self_assessment)

        evaluation_map = dict(evaluation)
        for module in list(self.modules):
            if module is None:
                continue
            handler = getattr(module, "on_auto_intention_promoted", None)
            if not callable(handler):
                continue
            try:
                handler(base_event, evaluation_map, self_assessment)
            except TypeError:
                handler(base_event)

    # ------------------------------------------------------------------
    def _build_self_assessment(
        self, intention: AutoIntention, evaluation: Mapping[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Synthesize a self-judgement blueprint for autonomous monitoring."""

        if not intention.signals:
            return None

        checkpoints: List[Dict[str, Any]] = []
        weights: List[float] = []
        for signal in intention.signals:
            if not isinstance(signal, Mapping):
                continue
            name = str(signal.get("name") or "").strip()
            if not name:
                continue
            target = float(signal.get("target", 0.6) or 0.6)
            direction = str(signal.get("direction", "above") or "above")
            metric = str(signal.get("metric") or name)
            weight = max(0.1, float(signal.get("weight", 1.0) or 1.0))
            weights.append(weight)
            checkpoints.append(
                {
                    "signal": name,
                    "metric": metric,
                    "target": target,
                    "direction": direction,
                    "weight": weight,
                    "reward_if_met": round(0.5 + target * 0.5, 3),
                    "penalty_if_missed": round(0.2 + (1.0 - target) * 0.4, 3),
                }
            )

        if not checkpoints:
            return None

        total_weight = sum(weights) or 1.0
        normalised = [weight / total_weight for weight in weights]
        composite_target = sum(
            checkpoint["target"] * ratio
            for checkpoint, ratio in zip(checkpoints, normalised)
        )

        now = time.time()
        plan = {
            "action_type": intention.action_type,
            "baseline_score": float(evaluation.get("score", 0.5)),
            "significance": float(evaluation.get("significance", 0.5)),
            "alignment": float(evaluation.get("alignment", 0.5)),
            "emotional_drive": float(evaluation.get("emotional_drive", 0.4)),
            "checkpoints": checkpoints,
            "composite_target": round(composite_target, 3),
            "reward_curve": {
                "max_reward": round(sum(cp["reward_if_met"] for cp in checkpoints), 3),
                "decay": 0.85,
                "bonus_on_growth": 0.25,
            },
            "next_review_ts": now + max(300.0, self.interval * 2.0),
            "created_at": now,
        }
        return plan


__all__ = ["AutoEvolutionCoordinator"]
