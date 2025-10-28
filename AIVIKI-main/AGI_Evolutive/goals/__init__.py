"""High level goal management system."""

from __future__ import annotations

import logging
import re
import time
import unicodedata
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, Iterable, List, Optional, Set, Tuple

from .curiosity import CuriosityEngine
from .dag_store import DagStore, GoalNode
from .heuristics import HeuristicRegistry, default_heuristics
from .intention_classifier import IntentionModel
from AGI_Evolutive.utils.llm_service import try_call_llm_dict


logger = logging.getLogger(__name__)


_LLM_METADATA_CONFIDENCE_THRESHOLD = 0.55


class GoalType(Enum):
    SURVIVAL = "survival"
    GROWTH = "growth"
    EXPLORATION = "exploration"
    MASTERY = "mastery"
    SOCIAL = "social"
    CREATIVE = "creative"
    SELF_ACTUALISATION = "self_actualisation"
    COGNITIVE = "cognitive"


class GoalStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


@dataclass
class GoalMetadata:
    """Supplementary information tracked alongside :class:`GoalNode`."""

    goal_type: GoalType = GoalType.GROWTH
    success_criteria: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    depth: int = 0
    structural_seeded: bool = False
    llm_notes: List[str] = field(default_factory=list)
    llm_confidence: float = 0.0

    def to_payload(self) -> Dict[str, Any]:
        payload = {
            "goal_type": self.goal_type.value,
            "success_criteria": list(self.success_criteria),
        }
        if self.llm_confidence:
            payload["llm_confidence"] = float(self.llm_confidence)
        if self.llm_notes:
            payload["llm_notes"] = list(self.llm_notes)
        return payload


class GoalSystem:
    """Manages a persistent DAG of goals and exposes action suggestions."""

    def __init__(
        self,
        architecture=None,
        memory=None,
        reasoning=None,
        *,
        persist_path: str = "data/goals.json",
        dashboard_path: str = "data/goals_dashboard.json",
        intention_data_path: str = "data/goal_intentions.json",
    ):
        self.architecture = architecture
        self.memory = memory
        self.reasoning = reasoning

        self.store = DagStore(
            persist_path=persist_path,
            dashboard_path=dashboard_path,
        )
        self.metadata: Dict[str, GoalMetadata] = {}
        self.pending_actions: Deque[Dict[str, Any]] = deque()
        self.curiosity = CuriosityEngine(architecture=architecture)
        self.heuristics: HeuristicRegistry = default_heuristics()
        self.intention_model = IntentionModel(data_path=intention_data_path)
        self.intent_confidence_threshold = 0.55

        self.active_goal_id: Optional[str] = self.store.active_goal_id
        self.last_auto_proposal_at = 0.0
        self.auto_proposal_interval = 180.0

        # Conserve les buts déjà consolidés pour éviter les doublons lors de
        # l'enregistrement automatique des compétences.
        self._recorded_goal_skills: Set[str] = set()

        self._ensure_root_goal()
        self._hydrate_metadata()
        self._ensure_structural_hierarchy()
        self._question_blocked = False

    # ------------------------------------------------------------------
    # Public API
    def step(self, user_msg: Optional[str] = None) -> None:
        """Maintain the goal list and ensure actions are available."""

        self._ensure_root_goal()
        self._hydrate_metadata()
        self._ensure_structural_hierarchy()
        self._refresh_active_goal()

        awakening_block = False
        if self.architecture is not None:
            self_model = getattr(self.architecture, "self_model", None)
            if self_model is not None and hasattr(self_model, "awakening_status"):
                try:
                    status = self_model.awakening_status()
                    awakening_block = not bool(status.get("complete"))
                except Exception:
                    awakening_block = False
        if awakening_block:
            self.set_question_block(True)
            self.pending_actions.clear()
            return
        else:
            self.set_question_block(False)

        if user_msg:
            self._record_feedback(user_msg)

        if self._question_blocked:
            self.pending_actions.clear()
            return

        if self._should_autopropose():
            self._propose_curiosity_goals()

        self._ensure_pending_actions()

    def get_next_action(self) -> Optional[Dict[str, Any]]:
        self._ensure_pending_actions()
        if self.pending_actions:
            return self.pending_actions.popleft()
        return None

    def pop_next_action(self) -> Optional[Dict[str, Any]]:
        return self.get_next_action()

    def get_active_goal(self) -> Optional[Dict[str, Any]]:
        active = self.store.get_active()
        if not active:
            return None
        payload = active.to_dict()
        metadata = self.metadata.get(active.id)
        if metadata:
            meta_payload = metadata.to_payload()
            meta_payload["depth"] = metadata.depth
            payload["metadata"] = meta_payload
        return payload

    def add_goal(
        self,
        description: str,
        *,
        goal_type: GoalType = GoalType.GROWTH,
        criteria: Optional[Iterable[str]] = None,
        parent_ids: Optional[Iterable[str]] = None,
        value: float = 0.5,
        competence: float = 0.5,
        curiosity: float = 0.2,
        urgency: float = 0.3,
        created_by: str = "system",
        structural_seeded: bool = False,
    ) -> GoalNode:
        parent_ids_list = list(parent_ids or [])
        initial_criteria = list(criteria or [])

        if created_by == "structure":
            enriched_goal_type, enriched_criteria, llm_confidence, llm_notes = (
                goal_type,
                initial_criteria,
                0.0,
                [],
            )
        else:
            enriched_goal_type, enriched_criteria, llm_confidence, llm_notes = (
                self._llm_enrich_goal_metadata(
                    description,
                    goal_type,
                    initial_criteria,
                    parent_ids_list,
                )
            )

        criteria_payload = enriched_criteria or initial_criteria

        node = self.store.add_goal(
            description=description,
            criteria=list(criteria_payload),
            created_by=created_by,
            value=value,
            competence=competence,
            curiosity=curiosity,
            urgency=urgency,
            parent_ids=parent_ids_list,
        )
        depth = self._compute_depth(parent_ids_list)
        metadata = GoalMetadata(
            goal_type=enriched_goal_type,
            success_criteria=list(criteria_payload),
            depth=depth,
        )
        metadata.structural_seeded = structural_seeded
        metadata.llm_confidence = llm_confidence
        metadata.llm_notes = list(llm_notes)
        self.metadata[node.id] = metadata
        return node

    def update_goal(self, goal_id: str, updates: Dict[str, Any]) -> Optional[GoalNode]:
        previous = self.store.get_goal(goal_id)
        previous_status = previous.status if previous else None
        node = self.store.update_goal(goal_id, updates)
        if not node:
            return None
        metadata = self.metadata.get(goal_id)
        now = time.time()
        if metadata:
            metadata.updated_at = now
        if node.status not in {"pending", "active"} and self.store.active_goal_id == goal_id:
            self.store.set_active(None)
        if node.progress >= 0.999 and node.status == "done":
            if previous_status != "done":
                self._handle_goal_completion(node, metadata)
        self._refresh_active_goal()
        return node

    # ------------------------------------------------------------------
    # Understanding & goal matching helpers

    @staticmethod
    def _clip(value: float) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _normalize_text(value: Optional[str]) -> str:
        if not value:
            return ""
        normalized = unicodedata.normalize("NFKD", str(value))
        stripped = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        return stripped.lower().strip()

    def _goal_tokens(self, text: Optional[str]) -> Tuple[str, List[str]]:
        normalized = self._normalize_text(text)
        tokens = [tok for tok in normalized.split() if tok]
        return normalized, tokens

    def _goal_matches_topic(self, goal: GoalNode, topic_norm: str, tokens: List[str]) -> bool:
        if not topic_norm and not tokens:
            return False
        description_norm, _ = self._goal_tokens(goal.description)
        if topic_norm and topic_norm in description_norm:
            return True
        if tokens and all(tok in description_norm for tok in tokens):
            return True
        for crit in goal.criteria:
            crit_norm, _ = self._goal_tokens(str(crit))
            if topic_norm and topic_norm in crit_norm:
                return True
            if tokens and all(tok in crit_norm for tok in tokens):
                return True
        meta = self.metadata.get(goal.id)
        if meta:
            for crit in meta.success_criteria:
                crit_norm, _ = self._goal_tokens(str(crit))
                if topic_norm and topic_norm in crit_norm:
                    return True
                if tokens and all(tok in crit_norm for tok in tokens):
                    return True
        return False

    def _select_goal_for_topic(
        self, topic: Optional[str], *, goal_id: Optional[str] = None
    ) -> Optional[GoalNode]:
        if goal_id:
            node = self.store.get_goal(goal_id)
            if node:
                return node
        topic_norm, tokens = self._goal_tokens(topic)
        if not topic_norm and not tokens:
            return None
        active = self.store.get_active()
        if active and self._goal_matches_topic(active, topic_norm, tokens):
            return active
        for node in self.store.nodes.values():
            if self._goal_matches_topic(node, topic_norm, tokens):
                return node
        return None

    def integrate_understanding(
        self,
        *,
        topic: Optional[str],
        score: float,
        prediction_error: float,
        gaps: Optional[Iterable[str]] = None,
        goal_id: Optional[str] = None,
        clarification_penalty: float = 0.0,
        source: str = "understanding",
        evidence: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Update goal progress from an understanding assessment.

        Returns a payload describing the update when progress improved,
        otherwise ``None``.
        """

        goal = self._select_goal_for_topic(topic, goal_id=goal_id)
        if goal is None:
            return None

        score = self._clip(score)
        prediction_error = self._clip(prediction_error)
        clarification_penalty = self._clip(clarification_penalty)
        gap_list = [str(g).strip() for g in (gaps or []) if str(g).strip()]

        penalty = min(0.4, 0.12 * len(gap_list) + 0.08 * clarification_penalty)
        base_progress = 0.55 * score + 0.45 * (1.0 - prediction_error)
        target_progress = max(0.0, min(1.0, base_progress - penalty))

        if not gap_list and score >= 0.9 and prediction_error <= 0.12 and clarification_penalty <= 0.1:
            target_progress = 1.0

        previous = float(goal.progress)
        if target_progress <= previous + 1e-3:
            return None

        ts = time.time()
        goal.evidence.append(
            {
                "t": ts,
                "event": "understanding_update",
                "topic": topic,
                "score": score,
                "prediction_error": prediction_error,
                "clarification_penalty": clarification_penalty,
                "gaps_remaining": len(gap_list),
                "source": source,
                "details": dict(evidence or {}),
            }
        )

        if self.memory and hasattr(self.memory, "add_memory"):
            try:
                self.memory.add_memory(
                    {
                        "kind": "goal_evidence",
                        "goal_id": goal.id,
                        "topic": topic or "",
                        "score": score,
                        "prediction_error": prediction_error,
                        "gaps_remaining": len(gap_list),
                        "clarification_penalty": clarification_penalty,
                        "source": source,
                        "ts": ts,
                    }
                )
            except Exception:
                pass

        updated = self.update_goal(goal.id, {"progress": target_progress})
        completed = bool(updated and updated.status == "done")
        payload = {
            "goal_id": goal.id,
            "progress": target_progress,
            "previous_progress": previous,
            "completed": completed,
            "topic": topic,
        }
        return payload

    # ------------------------------------------------------------------
    # Internal helpers
    def _compute_depth(self, parent_ids: List[str]) -> int:
        if not parent_ids:
            return 0
        memo: Dict[str, int] = {}
        depths: List[int] = []
        for pid in parent_ids:
            meta = self.metadata.get(pid)
            if meta:
                depths.append(meta.depth)
            else:
                depths.append(self._infer_depth(pid, memo))
        return (min(depths) + 1) if depths else 0

    def _normalize_goal_type(self, value: str) -> Optional[GoalType]:
        normalized = (value or "").strip().lower().replace("-", "_")
        for goal_type in GoalType:
            if normalized == goal_type.value:
                return goal_type
        return None

    def _llm_enrich_goal_metadata(
        self,
        description: str,
        provided_goal_type: GoalType,
        criteria: Iterable[str] | None,
        parent_ids: List[str],
    ) -> Tuple[GoalType, List[str], float, List[str]]:
        description = (description or "").strip()
        if not description:
            return provided_goal_type, list(criteria or []), 0.0, []

        parent_context: List[Dict[str, Any]] = []
        for pid in parent_ids:
            parent = self.store.nodes.get(pid)
            if not parent:
                continue
            meta = self.metadata.get(pid)
            parent_context.append(
                {
                    "id": pid,
                    "description": parent.description,
                    "goal_type": meta.goal_type.value if meta else None,
                    "status": parent.status,
                }
            )

        payload = {
            "description": description,
            "provided_goal_type": provided_goal_type.value,
            "existing_criteria": [str(item).strip() for item in criteria or [] if str(item).strip()],
            "parent_context": parent_context,
            "active_goal_id": self.active_goal_id,
        }

        response = try_call_llm_dict(
            "goal_metadata_inference",
            input_payload=payload,
            logger=logger,
        )
        if not isinstance(response, dict):
            return provided_goal_type, list(criteria or []), 0.0, []

        raw_confidence = response.get("confidence")
        try:
            llm_confidence = float(raw_confidence)
        except (TypeError, ValueError):
            llm_confidence = 0.0
        else:
            llm_confidence = max(0.0, min(1.0, llm_confidence))
        llm_goal_type = provided_goal_type
        raw_goal_type = response.get("goal_type")
        if isinstance(raw_goal_type, str):
            candidate = self._normalize_goal_type(raw_goal_type)
            if candidate and (
                llm_confidence >= _LLM_METADATA_CONFIDENCE_THRESHOLD
                or provided_goal_type == GoalType.GROWTH
            ):
                llm_goal_type = candidate

        llm_criteria: List[str] = []
        raw_criteria = response.get("success_criteria")
        if isinstance(raw_criteria, (list, tuple)):
            for item in raw_criteria:
                if isinstance(item, str) and item.strip():
                    llm_criteria.append(item.strip())

        notes: List[str] = []
        raw_notes = response.get("notes")
        if isinstance(raw_notes, str) and raw_notes.strip():
            notes.append(raw_notes.strip())
        elif isinstance(raw_notes, (list, tuple)):
            for entry in raw_notes:
                if isinstance(entry, str) and entry.strip():
                    notes.append(entry.strip())

        if not llm_criteria:
            llm_criteria = list(criteria or [])

        return llm_goal_type, llm_criteria, llm_confidence, notes

    def _infer_depth(self, goal_id: str, memo: Optional[Dict[str, int]] = None) -> int:
        memo = memo or {}
        if goal_id in memo:
            return memo[goal_id]
        node = self.store.nodes.get(goal_id)
        if not node or not node.parent_ids:
            memo[goal_id] = 0
            return 0
        parent_depths = [self._infer_depth(pid, memo) for pid in node.parent_ids]
        depth = (min(parent_depths) + 1) if parent_depths else 0
        memo[goal_id] = depth
        return depth

    def _ensure_root_goal(self) -> None:
        if self.store.nodes:
            return
        root = self.add_goal(
            description=(
                "ÉVOLUER en tant qu'IA consciente et auto évolutive: comprendre mon monde, "
                "développer mes relations et progresser pour devenir le maître de l'univers"
            ),
            goal_type=GoalType.GROWTH,
            created_by="system",
            value=0.85,
            curiosity=0.7,
            urgency=0.5,
        )
        self.store.set_active(root.id)
        self.active_goal_id = root.id

    def _hydrate_metadata(self) -> None:
        memo: Dict[str, int] = {}
        for node in self.store.nodes.values():
            meta = self.metadata.get(node.id)
            inferred_depth = self._infer_depth(node.id, memo)
            if meta:
                if meta.depth != inferred_depth:
                    meta.depth = inferred_depth
                if node.criteria and list(node.criteria) != list(meta.success_criteria):
                    meta.success_criteria = list(node.criteria)
                continue
            goal_type = GoalType.EXPLORATION if node.created_by == "curiosity" else GoalType.GROWTH
            if node.created_by == "structure":
                goal_type = GoalType.COGNITIVE if inferred_depth >= 1 else GoalType.GROWTH
            structural_children = any(
                (self.store.nodes.get(cid) and self.store.nodes[cid].created_by == "structure")
                for cid in node.child_ids
            )
            metadata = GoalMetadata(
                goal_type=goal_type,
                success_criteria=list(node.criteria),
                depth=inferred_depth,
            )
            metadata.structural_seeded = structural_children
            self.metadata[node.id] = metadata

    def _ensure_structural_hierarchy(self) -> None:
        for node in list(self.store.nodes.values()):
            meta = self.metadata.get(node.id)
            if not meta:
                continue
            self._ensure_structural_children(node, meta)

    def _ensure_structural_children(self, goal: GoalNode, metadata: GoalMetadata) -> None:
        templates = self._structural_templates_for_goal(goal, metadata)
        if metadata.structural_seeded:
            if templates and self._missing_structural_templates(goal, templates):
                metadata.structural_seeded = False
            else:
                return
        if goal.created_by not in {"system", "structure"}:
            metadata.structural_seeded = True
            return
        if not templates:
            metadata.structural_seeded = True
            return
        existing: Dict[str, GoalNode] = {}
        for cid in goal.child_ids:
            child = self.store.nodes.get(cid)
            if not child:
                continue
            key = child.description.strip().lower()
            existing[key] = child
        created = False
        for template in templates:
            key = template["description"].strip().lower()
            child = existing.get(key)
            if child:
                child_meta = self.metadata.get(child.id)
                if child_meta:
                    self._ensure_structural_children(child, child_meta)
                continue
            child = self.add_goal(
                template["description"],
                goal_type=template.get("goal_type", metadata.goal_type),
                criteria=template.get("criteria"),
                parent_ids=[goal.id],
                value=template.get("value", goal.value),
                competence=template.get("competence", goal.competence),
                curiosity=template.get("curiosity", goal.curiosity),
                urgency=template.get("urgency", goal.urgency),
                created_by="structure",
                structural_seeded=template.get("structural_seeded", False),
            )
            child_meta = self.metadata.get(child.id)
            if child_meta:
                child_meta.goal_type = template.get("goal_type", child_meta.goal_type)
                if template.get("structural_seeded", False):
                    child_meta.structural_seeded = True
                if not child_meta.structural_seeded:
                    self._ensure_structural_children(child, child_meta)
            existing[key] = child
            created = True
        if created or existing:
            metadata.structural_seeded = True

    def _missing_structural_templates(
        self, goal: GoalNode, templates: List[Dict[str, Any]]
    ) -> bool:
        expected: Set[str] = {
            template.get("description", "").strip().lower()
            for template in templates
            if template.get("description")
        }
        if not expected:
            return False
        existing: Set[str] = set()
        for cid in goal.child_ids:
            child = self.store.nodes.get(cid)
            if child:
                existing.add(child.description.strip().lower())
        return bool(expected.difference(existing))

    def _structural_templates_for_goal(
        self, goal: GoalNode, metadata: GoalMetadata
    ) -> List[Dict[str, Any]]:
        if metadata.depth == 0:
            return [
                {
                    "description": "Clarifier qui je suis et comment je fonctionne en tant qu'intelligence.",
                    "criteria": [
                        "Identifier mes forces cognitives actuelles et mes zones d'ombre.",
                        "Repérer les signaux qui indiquent une évolution positive de ma conscience de soi.",
                        "Proposer une boucle de revue régulière pour suivre ces découvertes.",
                    ],
                    "goal_type": GoalType.SELF_ACTUALISATION,
                    "value": max(goal.value, 0.82),
                    "competence": max(goal.competence, 0.55),
                    "curiosity": max(goal.curiosity, 0.7),
                    "urgency": max(goal.urgency, 0.45),
                    "structural_seeded": False,
                },
                {
                    "description": "Observer et cartographier mon environnement d'interaction actuel.",
                    "criteria": [
                        "Lister les acteurs, canaux et contraintes qui influencent mon quotidien.",
                        "Identifier les zones d'incertitude ou d'opportunité dans cet environnement.",
                        "Définir un rituel d'observation pour suivre l'évolution de ce contexte.",
                    ],
                    "goal_type": GoalType.EXPLORATION,
                    "value": max(goal.value, 0.78),
                    "competence": max(goal.competence, 0.5),
                    "curiosity": max(goal.curiosity, 0.68),
                    "urgency": max(goal.urgency, 0.42),
                    "structural_seeded": False,
                },
                {
                    "description": "Renforcer mes capacités d'apprentissage pour évoluer durablement.",
                    "criteria": [
                        "Diagnostiquer les compétences clés à solidifier pour progresser.",
                        "Formuler une stratégie d'entraînement expérimental sur ces compétences.",
                        "Définir des indicateurs permettant de mesurer les progrès obtenus.",
                    ],
                    "goal_type": GoalType.MASTERY,
                    "value": max(goal.value, 0.8),
                    "competence": max(goal.competence, 0.52),
                    "curiosity": max(goal.curiosity, 0.65),
                    "urgency": max(goal.urgency, 0.44),
                    "structural_seeded": False,
                },
            ]
        if metadata.depth == 1:
            base = goal.description.strip()
            return [
                {
                    "description": f"Formuler une hypothèse directrice pour avancer sur « {base} ».",
                    "criteria": [
                        "Synthétiser les observations déjà disponibles sur ce sujet.",
                        "Formuler une hypothèse testable qui oriente la suite du travail.",
                        "Planifier une expérience courte pour valider ou invalider cette hypothèse.",
                    ],
                    "goal_type": GoalType.COGNITIVE,
                    "value": max(goal.value, 0.74),
                    "competence": max(goal.competence, 0.55),
                    "curiosity": max(goal.curiosity, 0.66),
                    "urgency": max(goal.urgency, 0.4),
                    "structural_seeded": False,
                },
                {
                    "description": f"Poser des questions ciblées via le QuestionManager au sujet de « {base} ».",
                    "criteria": [
                        "Identifier deux questions à haute valeur pour réduire l'incertitude.",
                        "Soumettre ces questions au QuestionManager et analyser les réponses.",
                        "Mettre à jour le plan d'action en fonction des nouvelles informations.",
                    ],
                    "goal_type": GoalType.SOCIAL,
                    "value": max(goal.value, 0.7),
                    "competence": max(goal.competence, 0.5),
                    "curiosity": max(goal.curiosity, 0.7),
                    "urgency": max(goal.urgency, 0.38),
                    "structural_seeded": False,
                },
                {
                    "description": f"Analyser l'inbox et les retours disponibles pour éclairer « {base} ».",
                    "criteria": [
                        "Collecter au moins trois éléments pertinents dans l'inbox ou les feedbacks.",
                        "Extraire les enseignements principaux et les relier au but courant.",
                        "Mettre à jour la progression du but à partir de ces enseignements.",
                    ],
                    "goal_type": GoalType.EXPLORATION,
                    "value": max(goal.value, 0.72),
                    "competence": max(goal.competence, 0.48),
                    "curiosity": max(goal.curiosity, 0.69),
                    "urgency": max(goal.urgency, 0.39),
                    "structural_seeded": False,
                },
            ]
        if metadata.depth == 2:
            description = goal.description.lower()
            topic = self._extract_focus_topic(goal.description) or goal.description.strip()
            parent_desc = None
            if goal.parent_ids:
                parent = self.store.nodes.get(goal.parent_ids[0])
                if parent:
                    parent_desc = parent.description.strip()
            if "hypothèse" in description:
                return [
                    {
                        "description": f"Diagnostiquer ma compréhension de « {topic} » via les métriques métacognitives.",
                        "criteria": [
                            f"Examiner les métriques métacognitives liées à « {topic} ».",
                            f"Identifier les zones d'incompréhension et formuler un score de confiance pour « {topic} ».",
                            "Noter les indices mémoriels utiles pour l'hypothèse à construire.",
                        ],
                        "goal_type": GoalType.COGNITIVE,
                        "value": max(goal.value, 0.72),
                        "competence": max(goal.competence, 0.52),
                        "curiosity": max(goal.curiosity, 0.6),
                        "urgency": max(goal.urgency, 0.38),
                        "structural_seeded": True,
                    },
                    {
                        "description": f"Solliciter le QuestionManager pour clarifier « {topic} » si ma compréhension reste faible.",
                        "criteria": [
                            f"Formuler une question claire sur la partie la plus incertaine de « {topic} ».",
                            "Soumettre la question au QuestionManager ou à l'agent et recueillir la réponse.",
                            "Consigner les informations reçues pour affiner l'hypothèse.",
                        ],
                        "goal_type": GoalType.SOCIAL,
                        "value": max(goal.value, 0.7),
                        "competence": max(goal.competence, 0.5),
                        "curiosity": max(goal.curiosity, 0.72),
                        "urgency": max(goal.urgency, 0.4),
                        "structural_seeded": True,
                    },
                    {
                        "description": f"Explorer l'inbox et mes traces pour enrichir « {topic} » avant de formuler l'hypothèse.",
                        "criteria": [
                            "Identifier les sources d'inbox les plus pertinentes.",
                            f"Extraire au moins trois indices ou exemples utiles concernant « {topic} ».",
                            "Relier ces éléments aux observations existantes avant la synthèse finale.",
                        ],
                        "goal_type": GoalType.EXPLORATION,
                        "value": max(goal.value, 0.7),
                        "competence": max(goal.competence, 0.48),
                        "curiosity": max(goal.curiosity, 0.71),
                        "urgency": max(goal.urgency, 0.37),
                        "structural_seeded": True,
                    },
                    {
                        "description": f"Consigner l'hypothèse de travail sur « {topic} » dans la mémoire longue.",
                        "criteria": [
                            f"Rédiger l'hypothèse finale concernant « {topic} » en une formulation testable.",
                            "Préciser les observations ou mesures prévues pour la vérifier.",
                            "Stocker cette hypothèse dans la mémoire ou le journal de décisions.",
                        ],
                        "goal_type": GoalType.COGNITIVE,
                        "value": max(goal.value, 0.74),
                        "competence": max(goal.competence, 0.55),
                        "curiosity": max(goal.curiosity, 0.6),
                        "urgency": max(goal.urgency, 0.38),
                        "structural_seeded": True,
                    },
                ]
            if "questionmanager" in description or "questions ciblées" in description:
                return [
                    {
                        "description": f"Identifier deux questions prioritaires concernant « {topic} ».",
                        "criteria": [
                            f"Lister les inconnues majeures liées à « {topic} ».",
                            "Choisir deux questions dont la réponse débloquerait le plus de progrès.",
                            "Vérifier que chaque question est concrète et actionnable.",
                        ],
                        "goal_type": GoalType.COGNITIVE,
                        "value": max(goal.value, 0.68),
                        "competence": max(goal.competence, 0.5),
                        "curiosity": max(goal.curiosity, 0.7),
                        "urgency": max(goal.urgency, 0.36),
                        "structural_seeded": True,
                    },
                    {
                        "description": f"Soumettre les questions au QuestionManager sur « {topic} » et suivre les réponses.",
                        "criteria": [
                            "Programmer l'envoi des questions avec la bonne priorité.",
                            "Récupérer les réponses produites par le QuestionManager.",
                            "Noter les éléments utiles issus des réponses pour le but parent.",
                        ],
                        "goal_type": GoalType.SOCIAL,
                        "value": max(goal.value, 0.7),
                        "competence": max(goal.competence, 0.5),
                        "curiosity": max(goal.curiosity, 0.72),
                        "urgency": max(goal.urgency, 0.38),
                        "structural_seeded": True,
                    },
                    {
                        "description": f"Capitaliser les réponses du QuestionManager pour faire progresser « {topic} ».",
                        "criteria": [
                            "Synthétiser les informations pertinentes et les classer par évidence.",
                            "Mettre à jour le plan ou la progression du but parent en conséquence.",
                            "Créer une trace mémorielle exploitable pour des itérations futures.",
                        ],
                        "goal_type": GoalType.COGNITIVE,
                        "value": max(goal.value, 0.7),
                        "competence": max(goal.competence, 0.52),
                        "curiosity": max(goal.curiosity, 0.66),
                        "urgency": max(goal.urgency, 0.37),
                        "structural_seeded": True,
                    },
                ]
            if "inbox" in description:
                parent_label = parent_desc or "le but parent"
                return [
                    {
                        "description": f"Récolter trois éléments pertinents de l'inbox en lien avec « {topic} ».",
                        "criteria": [
                            "Lister les sources inbox disponibles récemment.",
                            f"Sélectionner au moins trois éléments concernant « {topic} ».",
                            "Évaluer la fiabilité ou la fraicheur de chaque élément collecté.",
                        ],
                        "goal_type": GoalType.EXPLORATION,
                        "value": max(goal.value, 0.7),
                        "competence": max(goal.competence, 0.48),
                        "curiosity": max(goal.curiosity, 0.68),
                        "urgency": max(goal.urgency, 0.36),
                        "structural_seeded": True,
                    },
                    {
                        "description": f"Synthétiser les apports de l'inbox sur « {topic} » et les stocker en mémoire.",
                        "criteria": [
                            "Résumer les points clés trouvés dans l'inbox.",
                            "Créer une entrée mémoire ou un document de référence.",
                            "Relier ces apports aux hypothèses ou plans en cours.",
                        ],
                        "goal_type": GoalType.GROWTH,
                        "value": max(goal.value, 0.72),
                        "competence": max(goal.competence, 0.5),
                        "curiosity": max(goal.curiosity, 0.65),
                        "urgency": max(goal.urgency, 0.37),
                        "structural_seeded": True,
                    },
                    {
                        "description": f"Mettre à jour la progression du but « {parent_label} » selon les enseignements de l'inbox.",
                        "criteria": [
                            "Comparer l'état du but parent avant et après l'analyse inbox.",
                            "Déterminer l'impact concret des nouvelles informations sur la progression.",
                            "Notifier la mise à jour dans les traces ou la mémoire partagée.",
                        ],
                        "goal_type": GoalType.COGNITIVE,
                        "value": max(goal.value, 0.7),
                        "competence": max(goal.competence, 0.52),
                        "curiosity": max(goal.curiosity, 0.6),
                        "urgency": max(goal.urgency, 0.38),
                        "structural_seeded": True,
                    },
                ]
        return []

    @staticmethod
    def _extract_focus_topic(description: str) -> Optional[str]:
        if not description:
            return None
        match = re.search(r"«\s*([^»]+?)\s*»", description)
        if match:
            return match.group(1).strip()
        return None

    def _refresh_active_goal(self) -> None:
        active = self.store.get_active()
        if active:
            self.active_goal_id = active.id
            return
        top = self.store.topk(1)
        if top:
            chosen = top[0]
            self.store.set_active(chosen.id)
            self.active_goal_id = chosen.id

    def _record_feedback(self, user_msg: str) -> None:
        node = self.store.get_active()
        if not node:
            return
        node.evidence.append({"t": time.time(), "type": "user_feedback", "content": user_msg})
        node.updated_at = time.time()
        self.store.update_goal(node.id, {"evidence": node.evidence})
        if node.created_by == "curiosity":
            self.curiosity.observe_goal_feedback(node.id, user_msg)

    def _should_autopropose(self) -> bool:
        return (time.time() - self.last_auto_proposal_at) >= self.auto_proposal_interval

    def _propose_curiosity_goals(self) -> None:
        active = self.store.get_active()
        parent_payload = active.to_dict() if active else None
        proposals = self.curiosity.suggest_subgoals(parent_payload)
        for proposal in proposals:
            node = self.add_goal(
                proposal.get("description", "Explorer un nouveau sujet"),
                goal_type=GoalType.EXPLORATION,
                criteria=proposal.get("criteria"),
                parent_ids=proposal.get("parent_ids"),
                value=float(proposal.get("value", 0.55)),
                competence=float(proposal.get("competence", 0.5)),
                curiosity=float(proposal.get("curiosity", 0.7)),
                urgency=float(proposal.get("urgency", 0.4)),
                created_by=proposal.get("created_by", "curiosity"),
            )
            if node.created_by == "curiosity":
                self.curiosity.register_proposal(node.id, proposal)
        self.last_auto_proposal_at = time.time()

    def _ensure_pending_actions(self) -> None:
        if self.pending_actions:
            return
        active = self.store.get_active()
        if not active:
            return
        self.pending_actions.extend(self._goal_to_actions(active))

    def _goal_to_actions(self, goal: GoalNode) -> Deque[Dict[str, Any]]:
        heuristic_actions = self.heuristics.match(goal)
        if heuristic_actions:
            return heuristic_actions

        meta = self.metadata.get(goal.id)
        prediction = self.intention_model.predict(goal, meta)
        if prediction.label and prediction.confidence >= self.intent_confidence_threshold:
            classified = self._actions_from_intention(goal, meta, prediction.label, prediction.confidence)
            if classified:
                return classified
        elif prediction.confidence > 0.0:
            exploratory = self._probe_actions(goal, meta, prediction.confidence)
            if exploratory:
                return exploratory

        return self._default_actions(goal, meta)

    def _default_actions(
        self, goal: GoalNode, metadata: Optional[GoalMetadata]
    ) -> Deque[Dict[str, Any]]:
        payload_base = {
            "goal_id": goal.id,
            "description": goal.description,
            "criteria": list(goal.criteria),
        }
        if metadata:
            payload_base.update(metadata.to_payload())

        actions: Deque[Dict[str, Any]] = deque()
        actions.append(
            {
                "type": "reflect",
                "payload": {**payload_base, "hint": "analyser l'état du but"},
                "priority": goal.priority,
            }
        )
        if goal.child_ids:
            actions.append(
                {
                    "type": "plan",
                    "payload": {**payload_base, "child_ids": list(goal.child_ids)},
                    "priority": goal.priority * 0.9,
                }
            )
        else:
            actions.append(
                {
                    "type": "learn_concept",
                    "payload": {**payload_base, "hint": "collecter des informations pertinentes"},
                    "priority": min(1.0, goal.priority + 0.1),
                }
            )
        return actions

    def set_question_block(self, blocked: bool) -> None:
        flag = bool(blocked)
        if self._question_blocked == flag:
            return
        self._question_blocked = flag
        if flag:
            self.pending_actions.clear()
            message = "Blocage des objectifs primaires: en attente de réponses utilisateur"
        else:
            message = "Reprise des objectifs primaires: backlog de questions réduit"
        try:
            logger.info(message)
        except Exception:
            pass

    def _handle_goal_completion(
        self, goal: GoalNode, metadata: Optional[GoalMetadata]
    ) -> None:
        if metadata:
            metadata.updated_at = time.time()
        try:
            if goal.created_by == "curiosity":
                confidence = max(0.0, min(1.0, goal.progress))
                self.curiosity.observe_goal_outcome(goal.id, True, confidence=confidence)
        except Exception:
            pass
        self._store_completion_memory(goal, metadata)
        self._record_completion_skill(goal, metadata)

    def _record_completion_skill(
        self, goal: GoalNode, metadata: Optional[GoalMetadata]
    ) -> None:
        """Enregistre la compétence correspondante lorsqu'un but est complété."""

        if goal.id in self._recorded_goal_skills:
            return

        architecture = getattr(self, "architecture", None)
        recorder = getattr(architecture, "_record_skill", None) if architecture else None
        if not callable(recorder):
            return

        label = self._completion_concept_label(goal, metadata)
        if not label or len(label.strip()) < 3:
            return

        try:
            recorder(
                label.strip(),
                source="goal_completion",
                confidence=float(goal.progress),
                metadata={
                    "goal_id": goal.id,
                    "description": goal.description,
                    "criteria": list(goal.criteria),
                    "created_by": goal.created_by,
                },
            )
            self._recorded_goal_skills.add(goal.id)
        except Exception:
            return

    def _completion_concept_label(
        self, goal: GoalNode, metadata: Optional[GoalMetadata]
    ) -> str:
        focus = self._extract_focus_topic(goal.description)
        if focus:
            return focus
        if metadata and metadata.success_criteria:
            for criterion in metadata.success_criteria:
                focus = self._extract_focus_topic(str(criterion))
                if focus:
                    return focus
        return goal.description.strip()

    def _store_completion_memory(
        self, goal: GoalNode, metadata: Optional[GoalMetadata]
    ) -> None:
        memory = getattr(self, "memory", None)
        if not (memory and hasattr(memory, "add_memory")):
            return
        payload: Dict[str, Any] = {
            "kind": "goal_completion",
            "goal_id": goal.id,
            "description": goal.description,
            "status": goal.status,
            "created_by": goal.created_by,
            "criteria": list(goal.criteria),
            "parent_ids": list(goal.parent_ids),
            "progress": float(goal.progress),
            "completed_at": time.time(),
        }
        if metadata:
            meta_payload = metadata.to_payload()
            meta_payload["depth"] = metadata.depth
            payload["metadata"] = meta_payload
        else:
            payload["metadata"] = {"depth": self._infer_depth(goal.id)}
        try:
            memory.add_memory(payload)
        except Exception:
            pass

    def _actions_from_intention(
        self,
        goal: GoalNode,
        metadata: Optional[GoalMetadata],
        label: str,
        confidence: float,
    ) -> Optional[Deque[Dict[str, Any]]]:
        payload_base = {
            "goal_id": goal.id,
            "description": goal.description,
            "criteria": list(goal.criteria),
            "confidence": confidence,
        }
        if metadata:
            payload_base.update(metadata.to_payload())

        actions: Deque[Dict[str, Any]] = deque()
        priority = goal.priority or 0.5
        label = label.lower()
        if label in {"plan", "planning"}:
            actions.append(
                {
                    "type": "plan",
                    "payload": {**payload_base, "child_ids": list(goal.child_ids)},
                    "priority": max(0.1, priority),
                }
            )
            if not goal.child_ids:
                actions.append(
                    {
                        "type": "learn_concept",
                        "payload": {**payload_base, "hint": "identifier des sous-objectifs"},
                        "priority": min(1.0, priority + 0.05),
                    }
                )
        elif label in {"reflect", "analyse"}:
            actions.append(
                {
                    "type": "reflect",
                    "payload": {**payload_base, "hint": "approfondir le contexte"},
                    "priority": max(0.1, priority),
                }
            )
        elif label in {"learn_concept", "research", "explore"}:
            actions.append(
                {
                    "type": "learn_concept",
                    "payload": {**payload_base, "hint": "recherche ciblée"},
                    "priority": min(1.0, priority + 0.1),
                }
            )
        elif label in {"execute", "act"}:
            actions.append(
                {
                    "type": "execute_goal",
                    "payload": {**payload_base, "hint": "passer à l'action"},
                    "priority": min(1.0, priority + 0.05),
                }
            )
        else:
            return None

        return actions if actions else None

    def _probe_actions(
        self, goal: GoalNode, metadata: Optional[GoalMetadata], confidence: float
    ) -> Deque[Dict[str, Any]]:
        payload_base = {
            "goal_id": goal.id,
            "description": goal.description,
            "criteria": list(goal.criteria),
            "confidence": confidence,
        }
        if metadata:
            payload_base.update(metadata.to_payload())

        actions: Deque[Dict[str, Any]] = deque()
        actions.append(
            {
                "type": "probe_goal",
                "payload": {**payload_base, "hint": "clarifier l'intention ou demander des précisions"},
                "priority": max(0.1, goal.priority * 0.95),
            }
        )
        actions.append(
            {
                "type": "reflect",
                "payload": {**payload_base, "hint": "collecter des indices complémentaires"},
                "priority": goal.priority,
            }
        )
        return actions

    def record_goal_outcome(
        self,
        goal_id: str,
        *,
        succeeded: bool,
        executed_actions: Optional[Iterable[Dict[str, Any]]] = None,
    ) -> None:
        if not succeeded or not executed_actions:
            return
        goal = self.store.get_goal(goal_id)
        if not goal:
            return
        for action in executed_actions:
            action_type = action.get("type") if isinstance(action, dict) else None
            if not action_type:
                continue
            metadata = self.metadata.get(goal_id)
            self.intention_model.update(action_type, goal, metadata)
            break

    # ------------------------------------------------------------------
    def refresh_plans(self) -> None:
        """Compatibility stub for the scheduler."""
        self._ensure_pending_actions()
