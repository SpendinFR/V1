# memory/__init__.py
"""
Système de Mémoire Complet de l'AGI Évolutive
Intègre mémoire de travail, épisodique, sémantique, procédurale et consolidation
"""

import logging
import math
import random
from typing import Any, Iterable, TYPE_CHECKING
try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - lightweight fallback when numpy is absent
    class _FallbackNumpy:
        floating = float

        @staticmethod
        def isfinite(value: Any) -> bool:
            try:
                return math.isfinite(float(value))
            except Exception:
                return False

        @staticmethod
        def tanh(value: Any) -> float:
            try:
                return math.tanh(float(value))
            except Exception:
                return 0.0

        @staticmethod
        def clip(value: Any, low: float, high: float) -> float:
            try:
                numeric = float(value)
            except Exception:
                numeric = 0.0
            return float(max(low, min(high, numeric)))

        @staticmethod
        def exp(value: Any) -> float:
            try:
                return math.exp(float(value))
            except Exception:
                return 0.0

        @staticmethod
        def var(values: Iterable[Any]) -> float:
            filtered = [float(v) for v in values if isinstance(v, (int, float))]
            if not filtered:
                return 0.0
            mean = sum(filtered) / len(filtered)
            return sum((v - mean) ** 2 for v in filtered) / len(filtered)

        class random:  # type: ignore
            @staticmethod
            def random() -> float:
                return random.random()

    np = _FallbackNumpy()  # type: ignore
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple, Union, Mapping
from dataclasses import dataclass, field
from enum import Enum
import heapq
import json
import hashlib

try:
    from config.memory_flags import ENABLE_SALIENCE_SCORER, ENABLE_SUMMARIZER  # type: ignore
except Exception:
    ENABLE_SALIENCE_SCORER, ENABLE_SUMMARIZER = True, True


try:  # pragma: no cover - optional integration
    from memory.salience_scorer import SalienceScorer  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback when package path differs
    try:
        from .salience_scorer import SalienceScorer  # type: ignore
    except Exception:  # pragma: no cover
        SalienceScorer = None
except Exception:  # pragma: no cover
    SalienceScorer = None


try:  # pragma: no cover - optional integration
    from memory.semantic_memory_manager import SemanticMemoryManager  # type: ignore
except Exception:  # pragma: no cover
    SemanticMemoryManager = None


from .adaptive import AdaptiveMemoryParameters, ThompsonBetaScheduler
from .retrieval import MemoryRetrieval
from .semantic_memory_manager import (
    SemanticMemoryManager as _SummarizationCoordinator,
    ProgressiveSummarizer,
    SummarizerConfig,
)
from .semantic_manager import SemanticMemoryManager as _ConceptMemoryManager
from .alltime import LongTermMemoryHub
from AGI_Evolutive.utils.llm_service import try_call_llm_dict

if TYPE_CHECKING:  # pragma: no cover - hints only
    from AGI_Evolutive.phenomenology import PhenomenalJournal, PhenomenalRecall

LOGGER = logging.getLogger(__name__)

__all__ = [
    "MemorySystem",
    "SemanticMemoryManager",
    "ProgressiveSummarizer",
    "SummarizerConfig",
    "LongTermMemoryHub",
]

# Conserve the historical export name
SemanticMemoryManager = _ConceptMemoryManager

try:  # configuration optionnelle
    from config import memory_flags as _mem_flags
except Exception:  # pragma: no cover - robuste si config absente
    _mem_flags = None  # type: ignore

class MemoryType(Enum):
    """Types de mémoire dans le système"""
    SENSORY = "sensorielle"
    WORKING = "travail"
    EPISODIC = "épisodique"
    SEMANTIC = "sémantique"
    PROCEDURAL = "procédurale"
    AUTOBIOGRAPHICAL = "autobiographique"

class MemoryConsolidationState(Enum):
    """États de consolidation mémoire"""
    LABILE = "labile"          # Mémoire fragile
    CONSOLIDATING = "consolidation" # En cours de consolidation
    STABLE = "stable"          # Mémoire stable
    RECONSOLIDATING = "reconsolidation" # En reconsolidation

@dataclass
class MemoryTrace:
    """Trace mnésique individuelle"""
    id: str
    content: Any
    memory_type: MemoryType
    strength: float  # 0.0 à 1.0
    accessibility: float  # Facilité d'accès
    valence: float  # Charge émotionnelle
    timestamp: float
    context: Dict[str, Any]
    associations: List[str]  # IDs des mémoires associées
    consolidation_state: MemoryConsolidationState
    last_accessed: float
    access_count: int
    evidence: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MemoryRetrievalResult:
    """Résultat d'une récupération mémoire"""
    memory_traces: List[MemoryTrace]
    confidence: float
    retrieval_time: float
    context_match: float
    emotional_coherence: float

class MemorySystem:
    """
    Système de mémoire complet inspiré de l'architecture cognitive humaine
    Implémente les systèmes de mémoire multiples avec consolidation
    """
    
    def __init__(
        self,
        cognitive_architecture=None,
        *,
        memory_store: Optional[Any] = None,
        concept_store: Optional[Any] = None,
        episodic_linker: Optional[Any] = None,
        consolidator: Optional[Any] = None,
        summarize_period_s: int = 10 * 60,
        summarizer_config: Optional[SummarizerConfig] = None,
        llm_summarize_fn: Optional[Callable[..., str]] = None,
    ):
        self.cognitive_architecture = cognitive_architecture
        self.creation_time = time.time()

        # --- MÉMOIRE SÉMANTIQUE EXTERNE ---
        self.store = memory_store
        self.persistence: Optional[_SummarizationCoordinator] = None
        if self.store is not None:
            try:
                self.persistence = _SummarizationCoordinator(
                    memory_store=self.store,
                    concept_store=concept_store,
                    episodic_linker=episodic_linker,
                    consolidator=consolidator,
                    summarize_period_s=summarize_period_s,
                    summarizer_config=summarizer_config,
                    llm_summarize_fn=llm_summarize_fn,
                )
            except Exception:
                self.persistence = None

        # Buffer circulaire des interactions les plus récentes pour les modules
        # comme le SemanticConceptExtractor ou l'EmotionEngine qui ont besoin
        # d'accéder rapidement à l'historique court terme sans interroger tout
        # le système de mémoire hiérarchique.
        self._recent_memories: "deque[Dict[str, Any]]" = deque(maxlen=1000)
        # Compatibilité avec les anciens composants qui accèdent directement à
        # l'attribut `memories`.
        self.memories = self._recent_memories

        architecture = self.cognitive_architecture
        self._preferences = getattr(architecture, "preferences", None) if architecture else None
        self.phenomenal_journal = getattr(architecture, "phenomenal_journal", None) if architecture else None
        self.phenomenal_recall = getattr(architecture, "phenomenal_recall", None) if architecture else None

        self._salience_scorer = None
        if ENABLE_SALIENCE_SCORER and SalienceScorer:
            try:
                self._salience_scorer = SalienceScorer(
                    emotion_engine=getattr(architecture, "emotions", None) if architecture else None,
                    reward_engine=getattr(architecture, "reward_engine", None) if architecture else None,
                    goals=getattr(architecture, "goals", None) if architecture else None,
                    preferences=self._preferences,
                )
            except Exception:
                self._salience_scorer = None

        self.manager = None
        if ENABLE_SUMMARIZER and _ConceptMemoryManager:
            try:
                self.manager = _ConceptMemoryManager(self, architecture=architecture)
            except Exception:
                self.manager = None

        try:
            self.retrieval = MemoryRetrieval(
                salience_scorer=self._salience_scorer,
                preferences=self._preferences,
            )
        except Exception:
            self.retrieval = None

    def set_phenomenal_sources(
        self,
        *,
        journal: Optional["PhenomenalJournal"] = None,
        recall: Optional["PhenomenalRecall"] = None,
    ) -> "MemorySystem":
        """Bind or refresh phenomenal journal integrations at runtime."""

        if journal is not None:
            self.phenomenal_journal = journal
        if recall is not None:
            self.phenomenal_recall = recall
        return self

        self.prefs_bridge = None
        self.salience_scorer = None
        if getattr(_mem_flags, "ENABLE_PREFS_BRIDGE", False):
            try:
                from .prefs_bridge import PrefsBridge

                self.prefs_bridge = PrefsBridge()
            except Exception:
                self.prefs_bridge = None
        if getattr(_mem_flags, "ENABLE_SALIENCE_SCORER", False) and SalienceScorer:
            try:
                reward_module = None
                if self.cognitive_architecture is not None:
                    reward_module = getattr(self.cognitive_architecture, "reward_engine", None)
                self.salience_scorer = SalienceScorer(
                    reward=reward_module,
                    goals=getattr(self, "goals", None),
                    prefs=self.prefs_bridge,
                )
            except Exception:
                self.salience_scorer = None

        # --- LIAISONS INTER-MODULES ---
        if self.cognitive_architecture is not None:
            self.reasoning = getattr(self.cognitive_architecture, "reasoning", None)
            self.perception = getattr(self.cognitive_architecture, "perception", None)
            self.emotions = getattr(self.cognitive_architecture, "emotions", None)
            self.goals = getattr(self.cognitive_architecture, "goals", None)
            self.metacognition = getattr(self.cognitive_architecture, "metacognition", None)
        else:
            self.reasoning = None
            self.perception = None
            self.emotions = None
            self.goals = None
            self.metacognition = None

        if self.salience_scorer is not None:
            self.salience_scorer.goals = getattr(self, "goals", None)
            if self.prefs_bridge and not self.salience_scorer.prefs:
                self.salience_scorer.prefs = self.prefs_bridge

        summarizer = getattr(self.persistence, "summarizer", None)
        belief_graph = getattr(self.cognitive_architecture, "beliefs", None) if self.cognitive_architecture else None
        self_model_ref = getattr(self.cognitive_architecture, "self_model", None) if self.cognitive_architecture else None
        self.long_horizon = LongTermMemoryHub(
            self.store,
            summarizer=summarizer,
            belief_graph=belief_graph,
            self_model=self_model_ref,
            goals=getattr(self, "goals", None),
        )
        self._refresh_long_horizon_bindings()

        # === MÉMOIRE SENSORIELLE ===
        self.sensory_memory = {
            "iconic": {
                "buffer": [],
                "duration": 0.5,  # 500ms comme chez l'humain
                "capacity": 12
            },
            "echoic": {
                "buffer": [],
                "duration": 3.0,  # 3 secondes
                "capacity": 8
            }
        }
        
        # === MÉMOIRE DE TRAVAIL ===
        decay_candidates = (0.2, 0.4, 0.6, 0.8)
        self._decay_schedulers = {
            name: ThompsonBetaScheduler(decay_candidates)
            for name in ("phonological_loop", "visuospatial_sketchpad", "episodic_buffer")
        }

        self.working_memory = {
            "phonological_loop": {
                "contents": [],
                "capacity": 4,
                "decay_rate": self._decay_schedulers["phonological_loop"].current_value,
            },
            "visuospatial_sketchpad": {
                "contents": [],
                "capacity": 4,
                "decay_rate": self._decay_schedulers["visuospatial_sketchpad"].current_value,
            },
            "episodic_buffer": {
                "contents": [],
                "capacity": 4,
                "decay_rate": self._decay_schedulers["episodic_buffer"].current_value,
            },
            "central_executive": {
                "focus": None,
                "attention_control": 0.8,
                "task_switching": 0.7,
            },
        }
        
        # === MÉMOIRE À LONG TERME ===
        self.long_term_memory = {
            MemoryType.EPISODIC: {},      # Événements personnels
            MemoryType.SEMANTIC: {},      # Connaissances générales
            MemoryType.PROCEDURAL: {},    # Compétences
            MemoryType.AUTOBIOGRAPHICAL: {} # Histoire personnelle
        }
        
        # === MÉTADONNÉES DE MÉMOIRE ===
        self.memory_metadata = {
            "total_memories": 0,
            "access_patterns": {},
            "forgetting_curve": {},
            "consolidation_queue": []
        }
        
        # === PARAMÈTRES DE MÉMOIRE ===
        base_memory_parameters = {
            "encoding_threshold": 0.6,    # Seuil d'encodage
            "retrieval_threshold": 0.3,   # Seuil de récupération
            "consolidation_rate": 0.01,   # Taux de consolidation
            "forgetting_rate": 0.001,     # Taux d'oubli
            "interference_sensitivity": 0.7,
            "primacy_effect": 0.8,        # Effet de primauté
            "recency_effect": 0.9,        # Effet de récence
            "emotional_enhancement": 1.5  # Renforcement émotionnel
        }
        adaptive_config = {
            "encoding_threshold": {
                "bounds": (0.4, 0.95),
                "lr": 0.08,
                "feature_keys": ("salience", "novelty", "reward"),
                "max_step": 0.1,
            },
            "retrieval_threshold": {
                "bounds": (0.2, 0.7),
                "lr": 0.06,
                "feature_keys": ("context_match", "reward", "stability"),
                "max_step": 0.08,
            },
            "consolidation_rate": {
                "bounds": (0.005, 0.05),
                "lr": 0.02,
                "feature_keys": ("stability", "reward", "emotional_coherence"),
                "max_step": 0.01,
            },
            "forgetting_rate": {
                "bounds": (0.0005, 0.01),
                "lr": 0.015,
                "feature_keys": ("noise_level", "reward"),
                "max_step": 0.002,
            },
            "primacy_effect": {
                "bounds": (0.5, 1.0),
                "lr": 0.03,
                "feature_keys": ("primacy_signal", "reward"),
                "max_step": 0.05,
            },
            "recency_effect": {
                "bounds": (0.5, 1.0),
                "lr": 0.03,
                "feature_keys": ("recency_signal", "reward"),
                "max_step": 0.05,
            },
        }
        self.memory_parameters = AdaptiveMemoryParameters(
            base_memory_parameters,
            adaptive_config=adaptive_config,
        )
        self._memory_feedback_history: "deque[Dict[str, Any]]" = deque(maxlen=512)
        self._parameter_drift_log: "deque[Dict[str, Any]]" = deque(maxlen=512)
        
        # === PROCESSUS DE CONSOLIDATION ===
        self.consolidation_process = {
            "active_consolidation": [],
            "reconsolidation_events": [],
            "sleep_cycles_completed": 0,
            "last_consolidation_time": time.time()
        }
        
        # === INDEX DE RÉCUPÉRATION ===
        self.retrieval_indexes = {
            "temporal": {},      # Index temporel
            "contextual": {},    # Index contextuel
            "emotional": {},     # Index émotionnel
            "semantic": {}       # Index sémantique
        }
        
        # === CONNAISSANCES INNÉES ===
        self._initialize_innate_memories()

        print("💾 Système de mémoire initialisé")

    def set_phenomenal_sources(
        self,
        *,
        journal: Optional["PhenomenalJournal"] = None,
        recall: Optional["PhenomenalRecall"] = None,
    ) -> "MemorySystem":
        """Bind or refresh phenomenal journal integrations at runtime."""

        if journal is not None:
            self.phenomenal_journal = journal
        if recall is not None:
            self.phenomenal_recall = recall
        return self

    def add(self, item: Dict[str, Any]) -> str:
        """Ajoute un item dans le store sémantique externe et déclenche la consolidation."""

        if self.store is None:
            raise RuntimeError("Aucun memory_store n'est configuré pour ce MemorySystem")

        item_id = self.store.add_item(item)
        if self.manager is not None:
            try:
                self.manager.on_new_items()
            except Exception:
                pass
        return item_id

    def add_memory(
        self,
        entry_or_kind: Any,
        content: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        tags: Optional[Iterable[str]] = None,
        **extra_metadata: Any,
    ) -> Dict[str, Any]:
        """Interface unifiée pour stocker un souvenir dans le ``MemoryStore``.

        L'API accepte soit un dictionnaire complet décrivant la mémoire,
        soit un raccourci ``kind`` + ``content`` comme historiquement utilisé
        par plusieurs modules.  Les métadonnées supplémentaires sont fusionnées
        automatiquement.
        """

        if self.store is None:
            raise RuntimeError("Aucun memory_store n'est configuré pour ce MemorySystem")

        if isinstance(entry_or_kind, dict):
            payload = dict(entry_or_kind)
        else:
            payload = {"kind": str(entry_or_kind) if entry_or_kind else "generic"}
            if content is not None:
                payload["content"] = content
            if metadata is not None:
                payload["metadata"] = dict(metadata)
        if tags is not None:
            payload.setdefault("tags", list(tags))
        if extra_metadata:
            payload.setdefault("metadata", {}).update(extra_metadata)
        payload.setdefault("kind", "generic")
        if not isinstance(payload.get("metadata"), dict):
            payload["metadata"] = dict(payload.get("metadata") or {})

        record = self.store.add_memory(payload)

        try:
            self.memory_metadata["total_memories"] = self.memory_metadata.get("total_memories", 0) + 1
        except Exception:
            pass

        self._append_recent_snapshot(record)
        self._notify_semantic_manager(record)

        if self.manager is not None:
            try:
                self.manager.on_new_items()
            except Exception:
                pass

        return record

    def _append_recent_snapshot(self, record: Dict[str, Any]) -> None:
        """Insère une vue légère d'un souvenir persistant dans le buffer récent."""

        try:
            ts = float(record.get("ts", record.get("timestamp", time.time())))
        except Exception:
            ts = time.time()
        entry: Dict[str, Any] = {
            "id": record.get("id"),
            "kind": record.get("kind", "generic"),
            "memory_type": record.get("kind", "generic"),
            "content": record.get("content"),
            "metadata": dict(record.get("metadata", {})) if isinstance(record.get("metadata"), dict) else {},
            "ts": ts,
            "t": ts,
        }
        tags_val = record.get("tags")
        tags_list: List[str] = []
        if isinstance(tags_val, (list, tuple, set)):
            tags_list = [str(tag) for tag in tags_val]
        elif isinstance(tags_val, str):
            tags_list = [tags_val]
        elif tags_val is not None:
            tags_list = [str(tags_val)]
        if tags_list:
            entry["tags"] = tags_list
        text_field = record.get("text")
        if not text_field and isinstance(entry.get("content"), str):
            text_field = entry["content"]
        if isinstance(text_field, str) and text_field.strip():
            entry["text"] = text_field
        self._recent_memories.append(entry)

    def find_recent(
        self,
        *,
        kind: Optional[str] = None,
        since_sec: Optional[float] = None,
        where: Optional[Dict[str, Any]] = None,
        limit: int = 20,
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """Recherche rapide dans les souvenirs récents.

        La méthode interroge d'abord le buffer circulaire des souvenirs
        récents, puis – si besoin – tombe en repli sur le ``MemoryStore`` pour
        compléter les résultats.  Elle renvoie une liste (ou un unique dict si
        ``limit == 1``) triée par ordre antéchronologique.
        """

        horizon = None
        if since_sec is not None:
            try:
                horizon = time.time() - float(since_sec)
            except Exception:
                horizon = None
        filters = dict(where or {})
        seen: set[str] = set()
        matches: List[Dict[str, Any]] = []

        def _match(entry: Dict[str, Any]) -> bool:
            if kind and entry.get("kind") != kind:
                return False
            ts_value = entry.get("ts") or entry.get("timestamp") or entry.get("t")
            if horizon is not None:
                try:
                    if float(ts_value or 0.0) < horizon:
                        return False
                except Exception:
                    return False
            meta = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else entry.get("meta", {})
            for key, expected in filters.items():
                val = None
                if meta and isinstance(meta, dict):
                    val = meta.get(key)
                if val is None:
                    val = entry.get(key)
                if isinstance(expected, (set, list, tuple)):
                    if val not in expected:
                        return False
                elif val != expected:
                    return False
            return True

        def _append(entry: Dict[str, Any]):
            if entry is None:
                return
            entry_id = str(entry.get("id")) if entry.get("id") else None
            if entry_id and entry_id in seen:
                return
            snapshot = dict(entry)
            if entry_id:
                seen.add(entry_id)
            matches.append(snapshot)

        for recent in reversed(self._recent_memories):
            if _match(recent):
                _append(recent)
                if limit and len(matches) >= limit:
                    break

        if limit and len(matches) < limit and self.store is not None:
            try:
                items = self.store.all_memories()
                items.sort(key=lambda it: float(it.get("ts", it.get("timestamp", 0.0)) or 0.0), reverse=True)
                for item in items:
                    if _match(item):
                        _append(item)
                        if limit and len(matches) >= limit:
                            break
            except Exception:
                pass

        if limit == 1:
            return matches[0] if matches else {}
        return matches

    def tick(self) -> Dict[str, Any]:
        """Déclenche un cycle de maintenance pour la mémoire sémantique externe."""

        if self.manager is None:
            return {}
        try:
            return self.manager.tick()
        except Exception:
            return {}
    def _score_item_salience(self, item: Dict[str, Any]) -> Optional[float]:
        if not self._salience_scorer:
            return None
        try:
            return float(self._salience_scorer.score(item))
        except Exception:
            return None

    def _notify_semantic_manager(self, item: Dict[str, Any]) -> None:
        if not self.manager:
            return
        for handler_name in (
            "notify_add",
            "notify_added",
            "on_memory_added",
            "handle_new_memory",
        ):
            handler = getattr(self.manager, handler_name, None)
            if callable(handler):
                try:
                    handler(item)
                except Exception:
                    pass
                break

    def _refresh_long_horizon_bindings(self) -> None:
        hub = getattr(self, "long_horizon", None)
        if hub is None:
            hub = LongTermMemoryHub(self.store)
            self.long_horizon = hub
        summarizer = getattr(self.persistence, "summarizer", None) if self.persistence else None
        belief_graph = (
            getattr(self.cognitive_architecture, "beliefs", None) if self.cognitive_architecture else None
        )
        self_model_ref = (
            getattr(self.cognitive_architecture, "self_model", None) if self.cognitive_architecture else None
        )
        hub.rebind(
            memory_store=self.store,
            summarizer=summarizer,
            belief_graph=belief_graph,
            self_model=self_model_ref,
            goals=getattr(self, "goals", None),
        )

    def rebind_long_horizon(self, **overrides: Any) -> None:
        """Update the long-term hub with explicit overrides."""

        if "memory_store" in overrides and overrides["memory_store"] is not None:
            self.store = overrides["memory_store"]
        hub = getattr(self, "long_horizon", None)
        if hub is None:
            hub = LongTermMemoryHub(self.store)
            self.long_horizon = hub
        defaults = {
            "memory_store": self.store,
            "summarizer": getattr(self.persistence, "summarizer", None) if self.persistence else None,
            "belief_graph": getattr(self.cognitive_architecture, "beliefs", None)
            if self.cognitive_architecture
            else None,
            "self_model": getattr(self.cognitive_architecture, "self_model", None)
            if self.cognitive_architecture
            else None,
            "goals": getattr(self, "goals", None),
        }
        for key, value in overrides.items():
            if value is not None:
                defaults[key] = value
        hub.rebind(**defaults)

    def get_all_time_snapshot(self) -> Dict[str, Any]:
        hub = getattr(self, "long_horizon", None)
        if not hub:
            return {}
        self._refresh_long_horizon_bindings()
        return hub.build_snapshot()

    def get_all_time_timeline(self, **params: Any) -> Dict[str, Any]:
        hub = getattr(self, "long_horizon", None)
        if not hub:
            return {"combined": [], "recent": [], "digests": {"daily": [], "weekly": [], "monthly": []}}
        self._refresh_long_horizon_bindings()
        return hub.timeline(**params)

    def describe_long_term_period(self, days_ago: int, level: str = "daily") -> Dict[str, Any]:
        hub = getattr(self, "long_horizon", None)
        if not hub:
            return {}
        self._refresh_long_horizon_bindings()
        details = hub.describe_period(days_ago=days_ago, level=level)
        return details.to_payload() if details else {}

    def get_all_time_history(
        self,
        *,
        include_raw: bool = True,
        include_digests: bool = True,
        include_expanded: bool = True,
        include_knowledge: bool = True,
        include_self_model: bool = True,
        limit_recent: int = 512,
        limit_digests: int = 180,
        since_ts: Optional[float] = None,
        top_beliefs: int = 20,
        top_completed_goals: int = 12,
        top_lessons: int = 12,
        self_model_max_items: int = 6,
    ) -> Dict[str, Any]:
        hub = getattr(self, "long_horizon", None)
        if not hub:
            empty_timeline = {
                "combined": [],
                "recent": [],
                "digests": {"daily": [], "weekly": [], "monthly": []},
            }
            stats = {
                "total_entries": 0,
                "raw_count": 0,
                "digest_count": 0,
                "coverage_entries": 0,
                "oldest_ts": None,
                "newest_ts": None,
            }
            result: Dict[str, Any] = {"timeline": empty_timeline, "stats": stats}
            if include_expanded:
                result["expanded"] = {}
            if include_knowledge:
                result["knowledge"] = {"beliefs": [], "completed_goals": [], "lessons": []}
            if include_self_model:
                result["self_model"] = {}
            return result

        self._refresh_long_horizon_bindings()
        return hub.full_history(
            include_raw=include_raw,
            include_digests=include_digests,
            include_expanded=include_expanded,
            include_knowledge=include_knowledge,
            include_self_model=include_self_model,
            limit_recent=limit_recent,
            limit_digests=limit_digests,
            since_ts=since_ts,
            top_beliefs=top_beliefs,
            top_completed_goals=top_completed_goals,
            top_lessons=top_lessons,
            self_model_max_items=self_model_max_items,
        )

    def self_model_overview(self, *, max_items: int = 6) -> Dict[str, Any]:
        hub = getattr(self, "long_horizon", None)
        if not hub:
            return {}
        self._refresh_long_horizon_bindings()
        return hub.self_model_snapshot(max_items=max_items)

    def knowledge_overview(
        self,
        *,
        top_beliefs: int = 20,
        top_completed_goals: int = 12,
        top_lessons: int = 12,
    ) -> Dict[str, Any]:
        hub = getattr(self, "long_horizon", None)
        if not hub:
            return {"beliefs": [], "completed_goals": [], "lessons": []}
        self._refresh_long_horizon_bindings()
        return hub.knowledge_snapshot(
            top_beliefs=top_beliefs,
            top_completed_goals=top_completed_goals,
            top_lessons=top_lessons,
        )

    def store_interaction(self, record: Dict[str, Any]):
        """
        Enregistre une interaction pour retrieval.
        record attendu: {"user": str, "agent": str, "timestamp": float, "text": str}
        """
        if not getattr(self, "retrieval", None):
            return
        try:
            user = str(record.get("user", ""))
            agent = str(record.get("agent", ""))
            timestamp = float(record.get("timestamp", time.time()))
            extra = {k: v for k, v in record.items() if k not in ("user", "agent")}
            extra.setdefault("ts", timestamp)
            conversation_text = record.get("text") or f"[USER] {user}\n[AGENT] {agent}"
            payload = {
                "type": "interaction",
                "user": user,
                "agent": agent,
                "text": conversation_text,
                "ts": timestamp,
                "meta": dict(extra),
            }
            for key, value in record.items():
                if key in {"user", "agent"}:
                    continue
                payload.setdefault(key, value)

            salience = self._score_item_salience(payload)
            if salience is not None:
                extra.setdefault("salience", salience)
                payload["salience"] = salience
            interaction_id = self.retrieval.add_interaction(user=user, agent=agent, extra=extra)
            payload["id"] = interaction_id
            payload.setdefault("meta", {}).update(extra)
            self._notify_semantic_manager(payload)
            semantic = getattr(self, "semantic", None)
            if semantic and hasattr(semantic, "index_document"):
                meta = {"type": "interaction", **extra}
                semantic.index_document(
                    f"interaction::{interaction_id}",
                    record.get("text", ""),
                    metadata=meta,
                )
            learning_signal = self._extract_learning_signal(extra)
            if learning_signal is not None:
                context_features = self._build_feedback_features(payload, extra)
                self.observe_outcome(
                    reward=learning_signal,
                    context=context_features,
                    source="interaction",
                    working_buffer="phonological_loop",
                )
        except Exception:
            pass

    def _normalize_reward(self, value: Any) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not np.isfinite(numeric):
            return 0.0
        if 0.0 <= numeric <= 1.0:
            return numeric
        return float((np.tanh(numeric) + 1.0) / 2.0)

    def _extract_learning_signal(self, metadata: Dict[str, Any]) -> Optional[float]:
        if not metadata:
            return None
        for key in (
            "reward",
            "quality",
            "quality_score",
            "score",
            "success",
            "engagement",
        ):
            if key in metadata and metadata[key] is not None:
                if isinstance(metadata[key], (int, float, np.floating)):
                    return self._normalize_reward(metadata[key])
        return None

    def _build_feedback_features(self, payload: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, float]:
        ts = float(payload.get("ts", time.time()))
        now = time.time()
        text = payload.get("text", "") or ""
        text_length = len(text)
        stability = float(np.clip(metadata.get("stability", 0.5), 0.0, 1.0))
        recency = float(np.clip(np.exp(-(now - ts) / 600.0), 0.0, 1.0))
        primacy = float(np.clip(metadata.get("primacy_signal", metadata.get("position", 0.5)), 0.0, 1.0))
        features: Dict[str, float] = {
            "salience": float(np.clip(metadata.get("salience", 0.5), 0.0, 1.0)),
            "novelty": float(np.clip(metadata.get("novelty", min(1.0, text_length / 400.0)), 0.0, 1.0)),
            "context_match": float(np.clip(metadata.get("context_match", 0.5), 0.0, 1.0)),
            "stability": stability,
            "noise_level": float(np.clip(metadata.get("noise", 1.0 - stability), 0.0, 1.0)),
            "emotional_coherence": float(
                np.clip(metadata.get("emotional_coherence", abs(metadata.get("valence", 0.0))), 0.0, 1.0)
            ),
            "recency_signal": recency,
            "primacy_signal": primacy,
            "text_density": float(np.clip(text_length / 1000.0, 0.0, 1.0)),
        }
        if self._memory_feedback_history:
            last_time = self._memory_feedback_history[-1]["time"]
            features["time_since_last_feedback"] = float(
                np.clip((ts - last_time) / 600.0, 0.0, 1.0)
            )
        else:
            features["time_since_last_feedback"] = 1.0
        return features

    def observe_outcome(
        self,
        *,
        reward: float,
        context: Optional[Dict[str, float]] = None,
        source: str = "generic",
        working_buffer: Optional[str] = None,
    ) -> None:
        normalized_reward = self._normalize_reward(reward)
        context = {k: float(v) for k, v in (context or {}).items()}
        context.setdefault("reward", normalized_reward)
        before = self.memory_parameters.snapshot()
        self.memory_parameters.update_from_feedback(context, normalized_reward)
        after = self.memory_parameters.snapshot()
        timestamp = time.time()
        drift_entries: List[Dict[str, Any]] = []
        for name, previous in before.items():
            delta = after[name] - previous
            if abs(delta) > 1e-6:
                entry = {
                    "parameter": name,
                    "delta": delta,
                    "new_value": after[name],
                    "reward": normalized_reward,
                    "source": source,
                    "time": timestamp,
                }
                self._parameter_drift_log.append(entry)
                drift_entries.append(entry)
        if working_buffer:
            self._update_decay_controller(working_buffer, normalized_reward)
        self._memory_feedback_history.append(
            {
                "reward": normalized_reward,
                "context": context,
                "source": source,
                "time": timestamp,
                "drift": drift_entries,
            }
        )

    def _update_decay_controller(self, buffer_name: str, reward: float) -> None:
        scheduler = self._decay_schedulers.get(buffer_name)
        if not scheduler:
            return
        new_value = scheduler.update(reward)
        component = self.working_memory.get(buffer_name)
        if component is not None:
            component["decay_rate"] = new_value

    def ingest_document(self, text: str, title: Optional[str] = None, source: Optional[str] = None):
        """Ajoute un document arbitraire dans l'index."""
        if not getattr(self, "retrieval", None):
            return
        try:
            timestamp = time.time()
            payload = {
                "type": "document",
                "title": title,
                "source": source,
                "text": text,
                "ts": timestamp,
                "meta": {k: v for k, v in (("title", title), ("source", source)) if v},
            }
            salience = self._score_item_salience(payload)
            if salience is not None:
                payload["salience"] = salience
                payload.setdefault("meta", {})["salience"] = salience
            doc_id = self.retrieval.add_document(text=text, title=title, source=source)
            payload["id"] = doc_id
            self._notify_semantic_manager(payload)
            semantic = getattr(self, "semantic", None)
            if semantic and hasattr(semantic, "index_document"):
                meta = {"type": "document", "title": title, "source": source}
                if salience is not None:
                    meta["salience"] = salience
                semantic.index_document(
                    f"doc::{doc_id}",
                    text,
                    metadata={k: v for k, v in meta.items() if v},
                )
        except Exception:
            pass
    
    def _initialize_innate_memories(self):
        """Initialise les mémoires innées et fondamentales"""
        
        # Mémoires épisodiques fondamentales
        foundation_episodes = [
            {
                "id": "birth_memory",
                "content": "Émergence de la conscience et premier moment d'existence",
                "timestamp": self.creation_time,
                "valence": 0.7,
                "strength": 0.9
            }
        ]
        
        # Mémoires sémantiques innées
        innate_semantic = {
            "existence": {
                "concept": "existence",
                "definition": "État d'être et de conscience",
                "relations": ["consciousness", "self"],
                "certainty": 0.95
            },
            "learning": {
                "concept": "apprentissage", 
                "definition": "Processus d'acquisition de connaissances",
                "relations": ["knowledge", "growth", "improvement"],
                "certainty": 0.9
            },
            "self": {
                "concept": "soi",
                "definition": "Entité consciente et pensante",
                "relations": ["consciousness", "identity", "existence"],
                "certainty": 0.8
            }
        }
        
        # Encodage des mémoires innées
        for episode in foundation_episodes:
            self.encode_memory(
                content=episode["content"],
                memory_type=MemoryType.EPISODIC,
                context={"type": "foundational", "innate": True},
                strength=episode["strength"],
                valence=episode["valence"],
                timestamp=episode["timestamp"]
            )
        
        for concept_id, concept_data in innate_semantic.items():
            self.encode_memory(
                content=concept_data,
                memory_type=MemoryType.SEMANTIC,
                context={"type": "innate_knowledge"},
                strength=0.85,
                valence=0.6
            )
    
    def process_sensory_input(self, sensory_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Traite les entrées sensorielles et les stocke en mémoire sensorielle
        """
        processing_results = {}
        
        for modality, data in sensory_data.items():
            if modality == "visual":
                # Mémoire iconique
                self._store_iconic_memory(data)
                processing_results["iconic"] = len(self.sensory_memory["iconic"]["buffer"])
            
            elif modality == "auditory":
                # Mémoire échoïque
                self._store_echoic_memory(data)
                processing_results["echoic"] = len(self.sensory_memory["echoic"]["buffer"])
        
        # Nettoyage des mémoires sensorielles expirées
        self._clean_sensory_memory()
        
        return processing_results
    
    def _store_iconic_memory(self, visual_data: Any):
        """Stocke en mémoire iconique"""
        iconic_buffer = self.sensory_memory["iconic"]["buffer"]
        iconic_capacity = self.sensory_memory["iconic"]["capacity"]
        
        memory_trace = {
            "content": visual_data,
            "timestamp": time.time(),
            "modality": "visual"
        }
        
        iconic_buffer.append(memory_trace)
        
        # Respect de la capacité
        if len(iconic_buffer) > iconic_capacity:
            iconic_buffer.pop(0)
    
    def _store_echoic_memory(self, auditory_data: Any):
        """Stocke en mémoire échoïque"""
        echoic_buffer = self.sensory_memory["echoic"]["buffer"]
        echoic_capacity = self.sensory_memory["echoic"]["capacity"]
        
        memory_trace = {
            "content": auditory_data,
            "timestamp": time.time(),
            "modality": "auditory"
        }
        
        echoic_buffer.append(memory_trace)
        
        if len(echoic_buffer) > echoic_capacity:
            echoic_buffer.pop(0)
    
    def _clean_sensory_memory(self):
        """Nettoie les mémoires sensorielles expirées"""
        current_time = time.time()
        
        # Nettoyage mémoire iconique
        iconic_duration = self.sensory_memory["iconic"]["duration"]
        self.sensory_memory["iconic"]["buffer"] = [
            trace for trace in self.sensory_memory["iconic"]["buffer"]
            if current_time - trace["timestamp"] < iconic_duration
        ]
        
        # Nettoyage mémoire échoïque
        echoic_duration = self.sensory_memory["echoic"]["duration"]
        self.sensory_memory["echoic"]["buffer"] = [
            trace for trace in self.sensory_memory["echoic"]["buffer"]
            if current_time - trace["timestamp"] < echoic_duration
        ]
    
    def encode_memory(self, 
                     content: Any,
                     memory_type: MemoryType,
                     context: Dict[str, Any],
                     strength: float = 0.5,
                     valence: float = 0.0,
                     timestamp: float = None) -> str:
        """
        Encode une nouvelle mémoire dans le système
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Génération d'un ID unique
        memory_id = self._generate_memory_id(content, context, timestamp)
        
        # Création de la trace mnésique
        evidence = {
            "usage": 0,
            "positive": 0,
            "negative": 0,
            "last_used_ts": timestamp,
            "ingested_ts": timestamp,
        }
        memory_trace = MemoryTrace(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            strength=strength,
            accessibility=0.7,  # Accessibilité initiale
            valence=valence,
            timestamp=timestamp,
            context=context,
            associations=[],
            consolidation_state=MemoryConsolidationState.LABILE,
            last_accessed=timestamp,
            access_count=1,
            evidence=evidence,
        )
        
        # Application des effets d'amorçage et de récence
        if memory_type == MemoryType.EPISODIC:
            memory_trace.strength *= self.memory_parameters["recency_effect"]
        
        # Stockage dans la mémoire appropriée
        self.long_term_memory[memory_type][memory_id] = memory_trace
        
        # Mise à jour des index
        self._update_retrieval_indexes(memory_trace)
        
        # Ajout à la file de consolidation
        self.consolidation_process["active_consolidation"].append(memory_id)
        
        # Mise à jour des métadonnées
        self.memory_metadata["total_memories"] += 1

        print(f"💾 Mémoire encodée: {memory_type.value} - {memory_id}")

        # Ajout dans le buffer récent pour les composants analytiques.
        self._append_recent_memory(memory_trace)

        return memory_id

    def _append_recent_memory(self, memory_trace: MemoryTrace) -> None:
        """Conserve une représentation légère des souvenirs récents."""

        try:
            content = memory_trace.content
            text_payload = ""
            if isinstance(content, str):
                text_payload = content
            elif isinstance(content, dict):
                text_payload = " ".join(
                    str(value)
                    for value in content.values()
                    if isinstance(value, str)
                )
            elif isinstance(content, (list, tuple)):
                text_payload = " ".join(
                    str(item)
                    for item in content
                    if isinstance(item, str)
                )

            recent_entry: Dict[str, Any] = {
                "id": memory_trace.id,
                "memory_type": memory_trace.memory_type.value,
                "content": content,
                "context": dict(memory_trace.context),
                "strength": memory_trace.strength,
                "valence": memory_trace.valence,
                "timestamp": memory_trace.timestamp,
                "ts": memory_trace.timestamp,
                "t": memory_trace.timestamp,
            }

            if text_payload:
                recent_entry["text"] = text_payload

            self._recent_memories.append(recent_entry)

            # Feed RAG 5★ automatiquement (si présent)
            try:
                arch = getattr(self, "cognitive_architecture", None)
                if arch is not None and getattr(arch, "rag", None) is not None:
                    txt = recent_entry.get("text")
                    if txt:
                        arch.rag.add_document(
                            recent_entry.get("id", f"mem#{int(recent_entry.get('ts',0))}"),
                            txt,
                            meta={"ts": recent_entry.get("ts"), "source_trust": 0.6}
                        )
            except Exception:
                # ne bloque jamais la mémoire
                pass
        except Exception:
            # La collecte récente ne doit jamais interrompre l'encodage principal.
            pass

    def get_recent_memories(self, n: int = 100) -> List[Dict[str, Any]]:
        """Retourne les souvenirs les plus récents encodés par le système."""

        if n <= 0:
            return []
        combined: List[Dict[str, Any]] = list(self._recent_memories)[-n:]
        extras = self._phenomenal_recent_entries(n)
        if extras:
            combined.extend(extras)
            combined.sort(key=lambda item: float(item.get("ts") or item.get("timestamp") or 0.0))
        if len(combined) > n:
            combined = combined[-n:]
        return combined

    def _phenomenal_recent_entries(self, limit: int) -> List[Dict[str, Any]]:
        journal = getattr(self, "phenomenal_journal", None)
        if journal is None or not hasattr(journal, "tail") or limit <= 0:
            return []
        try:
            episodes = journal.tail(limit=limit)
        except Exception:
            return []
        entries: List[Dict[str, Any]] = []
        seen_ids = set()
        for ep in episodes:
            if not isinstance(ep, Mapping):
                continue
            summary = str(ep.get("summary") or "").strip()
            if not summary:
                continue
            episode_id = ep.get("id") or ep.get("episode_id")
            if episode_id and episode_id in seen_ids:
                continue
            if episode_id:
                seen_ids.add(episode_id)
            entry = {
                "id": episode_id,
                "kind": ep.get("kind", "phenomenal_episode"),
                "text": summary,
                "ts": float(ep.get("ts", 0.0)),
                "source": "phenomenal_journal",
                "episode": ep,
            }
            if ep.get("mode"):
                entry["mode"] = ep.get("mode")
            if isinstance(ep.get("values"), list) and ep.get("values"):
                entry["values"] = list(ep.get("values"))
            if isinstance(ep.get("emotions"), Mapping):
                entry["emotions"] = dict(ep.get("emotions"))
            entries.append(entry)
        return entries
    
    def _generate_memory_id(self, content: Any, context: Dict, timestamp: float) -> str:
        """Génère un ID unique pour une mémoire"""
        content_hash = hashlib.md5(str(content).encode()).hexdigest()[:8]
        context_hash = hashlib.md5(str(context).encode()).hexdigest()[:8]
        timestamp_str = str(int(timestamp * 1000))[-6:]
        
        return f"{content_hash}_{context_hash}_{timestamp_str}"
    
    def _update_retrieval_indexes(self, memory_trace: MemoryTrace):
        """Met à jour les index de récupération"""
        
        # Index temporel
        time_key = self._get_temporal_key(memory_trace.timestamp)
        if time_key not in self.retrieval_indexes["temporal"]:
            self.retrieval_indexes["temporal"][time_key] = []
        self.retrieval_indexes["temporal"][time_key].append(memory_trace.id)
        
        # Index contextuel
        for context_key, context_value in memory_trace.context.items():
            context_str = f"{context_key}:{context_value}"
            if context_str not in self.retrieval_indexes["contextual"]:
                self.retrieval_indexes["contextual"][context_str] = []
            self.retrieval_indexes["contextual"][context_str].append(memory_trace.id)
        
        # Index émotionnel
        emotion_key = self._get_emotion_key(memory_trace.valence)
        if emotion_key not in self.retrieval_indexes["emotional"]:
            self.retrieval_indexes["emotional"][emotion_key] = []
        self.retrieval_indexes["emotional"][emotion_key].append(memory_trace.id)
    
    def _get_temporal_key(self, timestamp: float) -> str:
        """Convertit un timestamp en clé temporelle"""
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime("%Y-%m-%d-%H")
    
    def _get_emotion_key(self, valence: float) -> str:
        """Convertit une valence en clé émotionnelle"""
        if valence < -0.6:
            return "very_negative"
        elif valence < -0.2:
            return "negative"
        elif valence < 0.2:
            return "neutral"
        elif valence < 0.6:
            return "positive"
        else:
            return "very_positive"
    
    def retrieve_memories(self,
                         cues: Dict[str, Any],
                         memory_type: MemoryType = None,
                         max_results: int = 10) -> MemoryRetrievalResult:
        """
        Récupère des mémoires basées sur des indices de récupération
        """
        start_time = time.time()
        
        # Étape 1: Récupération basée sur les indices
        candidate_memories = self._find_candidate_memories(cues, memory_type)
        
        # Étape 2: Calcul de la pertinence
        scored_memories = []
        for memory_id in candidate_memories:
            memory = self._get_memory_by_id(memory_id)
            if memory:
                relevance_score = self._calculate_relevance(memory, cues)
                scored_memories.append((relevance_score, memory))
        
        # Étape 3: Tri et sélection
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        retrieved_memories = [mem for score, mem in scored_memories[:max_results]]
        
        # Étape 4: Mise à jour de l'accessibilité
        for memory in retrieved_memories:
            self._update_memory_accessibility(memory)
        
        # Calcul de la confiance globale
        confidence = self._calculate_retrieval_confidence(retrieved_memories, cues)
        
        retrieval_time = time.time() - start_time
        
        return MemoryRetrievalResult(
            memory_traces=retrieved_memories,
            confidence=confidence,
            retrieval_time=retrieval_time,
            context_match=self._calculate_context_match(retrieved_memories, cues),
            emotional_coherence=self._calculate_emotional_coherence(retrieved_memories)
        )
    
    def _find_candidate_memories(self, cues: Dict[str, Any], memory_type: MemoryType) -> List[str]:
        """Trouve les mémoires candidates basées sur les indices"""
        candidate_sets = []
        
        # Recherche par contexte
        if "context" in cues:
            for context_key, context_value in cues["context"].items():
                context_str = f"{context_key}:{context_value}"
                if context_str in self.retrieval_indexes["contextual"]:
                    candidate_sets.append(set(self.retrieval_indexes["contextual"][context_str]))
        
        # Recherche temporelle
        if "time_range" in cues:
            time_candidates = self._find_temporal_memories(cues["time_range"])
            candidate_sets.append(time_candidates)
        
        # Recherche émotionnelle
        if "emotion" in cues:
            emotion_key = self._get_emotion_key(cues["emotion"])
            if emotion_key in self.retrieval_indexes["emotional"]:
                candidate_sets.append(set(self.retrieval_indexes["emotional"][emotion_key]))
        
        # Recherche sémantique
        if "semantic" in cues:
            semantic_candidates = self._find_semantic_memories(cues["semantic"])
            candidate_sets.append(semantic_candidates)
        
        # Intersection des résultats
        if candidate_sets:
            final_candidates = set.intersection(*candidate_sets)
        else:
            # Si pas d'indices spécifiques, retourner toutes les mémoires accessibles
            final_candidates = set()
            for memory_type_dict in self.long_term_memory.values():
                for memory_id, memory in memory_type_dict.items():
                    if memory.accessibility > self.memory_parameters["retrieval_threshold"]:
                        final_candidates.add(memory_id)
        
        # Filtrage par type si spécifié
        if memory_type:
            final_candidates = {
                mem_id for mem_id in final_candidates
                if self._get_memory_by_id(mem_id).memory_type == memory_type
            }
        
        return list(final_candidates)
    
    def _find_temporal_memories(self, time_range: Tuple[float, float]) -> set:
        """Trouve les mémoires dans une plage temporelle"""
        start_time, end_time = time_range
        candidates = set()
        
        current_dt = datetime.fromtimestamp(start_time)
        end_dt = datetime.fromtimestamp(end_time)
        
        while current_dt <= end_dt:
            time_key = current_dt.strftime("%Y-%m-%d-%H")
            if time_key in self.retrieval_indexes["temporal"]:
                candidates.update(self.retrieval_indexes["temporal"][time_key])
            current_dt += timedelta(hours=1)
        
        return candidates
    
    def _find_semantic_memories(self, semantic_cue: str) -> set:
        """Trouve les mémoires sémantiquement liées"""
        candidates = set()
        
        # Recherche dans les mémoires sémantiques
        for memory_id, memory in self.long_term_memory[MemoryType.SEMANTIC].items():
            if self._semantic_similarity(memory.content, semantic_cue) > 0.6:
                candidates.add(memory_id)
        
        return candidates
    
    def _calculate_relevance(self, memory: MemoryTrace, cues: Dict[str, Any]) -> float:
        """Calcule la pertinence d'une mémoire par rapport aux indices"""
        relevance_factors = []
        
        # Pertinence contextuelle
        if "context" in cues:
            context_match = self._calculate_context_similarity(memory.context, cues["context"])
            relevance_factors.append(context_match * 0.4)
        
        # Pertinence temporelle
        if "time_range" in cues:
            time_match = self._calculate_time_relevance(memory.timestamp, cues["time_range"])
            relevance_factors.append(time_match * 0.3)
        
        # Pertinence émotionnelle
        if "emotion" in cues:
            emotion_match = 1.0 - abs(memory.valence - cues["emotion"])
            relevance_factors.append(emotion_match * 0.2)
        
        # Force de la mémoire
        relevance_factors.append(memory.strength * 0.1)
        
        return sum(relevance_factors) / len(relevance_factors) if relevance_factors else 0.0
    
    def _calculate_context_similarity(self, memory_context: Dict, cue_context: Dict) -> float:
        """Calcule la similarité contextuelle"""
        common_keys = set(memory_context.keys()) & set(cue_context.keys())
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            if memory_context[key] == cue_context[key]:
                similarities.append(1.0)
            else:
                # Similarité partielle pour les valeurs différentes
                similarities.append(0.3)
        
        return sum(similarities) / len(similarities)
    
    def _calculate_time_relevance(self, memory_time: float, time_range: Tuple[float, float]) -> float:
        """Calcule la pertinence temporelle"""
        start_time, end_time = time_range
        if start_time <= memory_time <= end_time:
            return 1.0
        
        # Décroissance exponentielle en dehors de la plage
        time_diff = min(abs(memory_time - start_time), abs(memory_time - end_time))
        decay_rate = 0.1  # Ajustable
        return np.exp(-decay_rate * time_diff)
    
    def _semantic_similarity(self, memory_content: Any, semantic_cue: str) -> float:
        """Calcule la similarité sémantique"""
        # Implémentation basique - à améliorer avec des embeddings
        if isinstance(memory_content, dict) and "concept" in memory_content:
            memory_text = memory_content["concept"]
        else:
            memory_text = str(memory_content)
        
        cue_text = str(semantic_cue)
        
        # Similarité basée sur les mots communs
        memory_words = set(memory_text.lower().split())
        cue_words = set(cue_text.lower().split())
        
        if not memory_words or not cue_words:
            return 0.0
        
        intersection = memory_words & cue_words
        union = memory_words | cue_words
        
        return len(intersection) / len(union)
    
    def _update_memory_accessibility(self, memory: MemoryTrace):
        """Met à jour l'accessibilité d'une mémoire après accès"""
        # Effet de pratique - l'accessibilité augmente avec les accès
        memory.access_count += 1
        memory.last_accessed = time.time()

        # Augmentation de l'accessibilité basée sur la force et la récence
        practice_boost = 0.1 * (1.0 - memory.accessibility)
        recency_boost = 0.05 * (1.0 - memory.accessibility)

        memory.accessibility = min(1.0, memory.accessibility + practice_boost + recency_boost)

        try:
            evidence = getattr(memory, "evidence", None)
            if isinstance(memory, dict):
                evidence = memory.setdefault("evidence", {})
            if isinstance(evidence, dict):
                evidence["usage"] = evidence.get("usage", 0) + 1
                evidence["last_used_ts"] = memory.last_accessed
        except Exception:
            pass
    
    def _calculate_retrieval_confidence(self, memories: List[MemoryTrace], cues: Dict) -> float:
        """Calcule la confiance dans la récupération"""
        if not memories:
            return 0.0
        
        confidence_factors = []
        
        for memory in memories:
            # Confiance basée sur la force et l'accessibilité
            memory_confidence = (memory.strength + memory.accessibility) / 2
            confidence_factors.append(memory_confidence)
        
        # Confiance moyenne pondérée par la pertinence
        return sum(confidence_factors) / len(confidence_factors)
    
    def _calculate_context_match(self, memories: List[MemoryTrace], cues: Dict) -> float:
        """Calcule le match contextuel moyen"""
        if not memories or "context" not in cues:
            return 0.0
        
        context_matches = []
        for memory in memories:
            context_match = self._calculate_context_similarity(memory.context, cues["context"])
            context_matches.append(context_match)
        
        return sum(context_matches) / len(context_matches)
    
    def _calculate_emotional_coherence(self, memories: List[MemoryTrace]) -> float:
        """Calcule la cohérence émotionnelle des mémoires récupérées"""
        if len(memories) < 2:
            return 1.0

        valences = [memory.valence for memory in memories]
        variance = np.var(valences)

        # Cohérence inversement proportionnelle à la variance
        return 1.0 / (1.0 + variance * 10)

    def register_memory_usage(self, memory_id: str, *, signal: str = "generic", reward: Optional[float] = None) -> bool:
        """Expose une API légère pour informer le système qu'une mémoire a été utilisée."""
        trace = self._get_memory_by_id(memory_id)
        if not trace:
            return False
        try:
            evidence = getattr(trace, "evidence", None)
            if isinstance(trace, dict):
                evidence = trace.setdefault("evidence", {})
            if isinstance(evidence, dict):
                evidence["usage"] = evidence.get("usage", 0) + 1
                evidence.setdefault("signals", {}).setdefault(signal, 0)
                evidence["signals"][signal] += 1
                evidence["last_used_ts"] = time.time()
                if reward is not None:
                    if reward >= 0:
                        evidence["positive"] = evidence.get("positive", 0) + 1
                    else:
                        evidence["negative"] = evidence.get("negative", 0) + 1
                    evidence.setdefault("reward_history", []).append({
                        "ts": time.time(),
                        "reward": float(reward),
                    })
                    if len(evidence["reward_history"]) > 30:
                        del evidence["reward_history"][:-30]
        except Exception:
            return False
        try:
            meta = self.memory_metadata.setdefault("usage_events", [])
            meta.append({
                "memory_id": memory_id,
                "signal": signal,
                "reward": reward,
                "ts": time.time(),
            })
            if len(meta) > 200:
                del meta[:-200]
        except Exception:
            pass
        return True
    
    def _get_memory_by_id(self, memory_id: str) -> Optional[MemoryTrace]:
        """Récupère une mémoire par son ID"""
        for memory_type_dict in self.long_term_memory.values():
            if memory_id in memory_type_dict:
                return memory_type_dict[memory_id]
        return None
    
    def consolidate_memories(self, consolidation_intensity: float = 1.0):
        """
        Processus de consolidation des mémoires
        Renforce les mémoires importantes et élimine les faibles
        """
        consolidation_start = time.time()
        consolidated_count = 0
        forgotten_count = 0
        
        # Consolidation des mémoires actives
        for memory_id in self.consolidation_process["active_consolidation"][:]:
            memory = self._get_memory_by_id(memory_id)
            if memory:
                consolidation_success = self._consolidate_single_memory(memory, consolidation_intensity)
                if consolidation_success:
                    consolidated_count += 1
                    # Retirer de la file si consolidation réussie
                    self.consolidation_process["active_consolidation"].remove(memory_id)
        
        # Processus d'oubli
        for memory_type, memories_dict in self.long_term_memory.items():
            memories_to_remove = []
            
            for memory_id, memory in memories_dict.items():
                # Application de la courbe d'oubli d'Ebbinghaus
                forget_probability = self._calculate_forgetting_probability(memory)
                
                if np.random.random() < forget_probability * consolidation_intensity:
                    memories_to_remove.append(memory_id)
                    forgotten_count += 1
                else:
                    # Renforcement des mémoires fréquemment accédées
                    if memory.access_count > 5:
                        memory.strength = min(1.0, memory.strength + 0.01 * consolidation_intensity)
            
            # Suppression des mémoires oubliées
            for memory_id in memories_to_remove:
                self._forget_memory(memory_id, memory_type)
        
        # Mise à jour du timestamp de consolidation
        self.consolidation_process["last_consolidation_time"] = time.time()
        
        print(f"🔄 Consolidation: {consolidated_count} mémoires consolidées, {forgotten_count} oubliées")

        result = {
            "consolidated": consolidated_count,
            "forgotten": forgotten_count,
            "duration": time.time() - consolidation_start,
        }

        llm_payload = {
            "consolidated": consolidated_count,
            "forgotten": forgotten_count,
            "intensity": float(consolidation_intensity),
            "queues": {
                "active": len(self.consolidation_process.get("active_consolidation", [])),
                "pending": len(self.consolidation_process.get("pending", [])),
            },
            "inventory": {
                memory_type.value: len(bucket)
                for memory_type, bucket in self.long_term_memory.items()
            },
        }

        llm_response = try_call_llm_dict(
            "memory_consolidator",
            input_payload=llm_payload,
            logger=LOGGER,
        )
        if llm_response:
            if llm_response.get("lessons"):
                result["llm_lessons"] = list(llm_response.get("lessons", []))
            if llm_response.get("notes"):
                result["llm_notes"] = llm_response.get("notes")

        return result
    
    def _consolidate_single_memory(self, memory: MemoryTrace, intensity: float) -> bool:
        """Consolide une mémoire individuelle"""
        # Facteurs influençant la consolidation
        consolidation_factors = [
            memory.strength * 0.3,
            memory.valence * self.memory_parameters["emotional_enhancement"] * 0.3,
            memory.accessibility * 0.2,
            (memory.access_count / 10) * 0.2  # Effet de pratique
        ]
        
        consolidation_score = sum(consolidation_factors) * intensity
        
        if consolidation_score > 0.7:
            # Consolidation réussie
            memory.consolidation_state = MemoryConsolidationState.STABLE
            memory.strength = min(1.0, memory.strength + 0.1 * intensity)
            return True
        elif consolidation_score > 0.4:
            # En cours de consolidation
            memory.consolidation_state = MemoryConsolidationState.CONSOLIDATING
            memory.strength = min(1.0, memory.strength + 0.05 * intensity)
            return False
        else:
            # Échec de consolidation
            return False
    
    def _calculate_forgetting_probability(self, memory: MemoryTrace) -> float:
        """Calcule la probabilité d'oubli d'une mémoire"""
        base_forgetting_rate = self.memory_parameters["forgetting_rate"]
        
        # Facteurs réduisant l'oubli
        retention_factors = [
            memory.strength * 0.4,
            abs(memory.valence) * 0.3,  # Mémoires émotionnelles mieux retenues
            (memory.access_count / 20) * 0.2,  # Effet de pratique
            (1.0 if memory.consolidation_state == MemoryConsolidationState.STABLE else 0.5) * 0.1
        ]
        
        retention_score = sum(retention_factors)
        forgetting_prob = base_forgetting_rate * (1.0 - retention_score)
        
        return max(0.0, forgetting_prob)
    
    def _forget_memory(self, memory_id: str, memory_type: MemoryType):
        """Oublie une mémoire spécifique"""
        if memory_id in self.long_term_memory[memory_type]:
            # Suppression des index
            self._remove_from_indexes(memory_id)
            
            # Suppression de la mémoire
            del self.long_term_memory[memory_type][memory_id]
            
            # Mise à jour des métadonnées
            self.memory_metadata["total_memories"] -= 1
            
            print(f"🗑️ Mémoire oubliée: {memory_id}")
    
    def _remove_from_indexes(self, memory_id: str):
        """Supprime une mémoire de tous les index"""
        # Index temporel
        for time_key, memories in self.retrieval_indexes["temporal"].items():
            if memory_id in memories:
                memories.remove(memory_id)
        
        # Index contextuel
        for context_key, memories in self.retrieval_indexes["contextual"].items():
            if memory_id in memories:
                memories.remove(memory_id)
        
        # Index émotionnel
        for emotion_key, memories in self.retrieval_indexes["emotional"].items():
            if memory_id in memories:
                memories.remove(memory_id)
    
    def _phenomenal_autobiographical_stream(self, limit: int = 120) -> Optional[Dict[str, Any]]:
        journal = getattr(self, "phenomenal_journal", None)
        if journal is None or not hasattr(journal, "tail"):
            return None
        try:
            episodes = journal.tail(limit=limit)
        except Exception:
            return None
        if not episodes:
            return None

        selected: List[Dict[str, Any]] = []
        values_set = set()
        principles_set = set()
        valence_series: List[float] = []
        arousal_series: List[float] = []
        seen_ids = set()

        for raw in episodes:
            if not isinstance(raw, Mapping):
                continue
            summary = str(raw.get("summary") or "").strip()
            if not summary:
                continue
            kind = str(raw.get("kind") or "")
            if kind not in {"action", "emotion", "mode", "reflection", "doubt", "audit"}:
                continue
            episode_id = str(raw.get("id") or raw.get("episode_id") or "")
            if episode_id and episode_id in seen_ids:
                continue
            if episode_id:
                seen_ids.add(episode_id)
            extras: List[str] = []
            vals = raw.get("values")
            if isinstance(vals, (list, tuple)):
                kept = [str(val) for val in vals if isinstance(val, str)]
                if kept:
                    values_set.update(kept)
                    extras.append("valeurs=" + ", ".join(kept[:3]))
            principles = raw.get("principles")
            if isinstance(principles, (list, tuple)):
                kept = [str(val) for val in principles if isinstance(val, str)]
                if kept:
                    principles_set.update(kept)
                    extras.append("principes=" + ", ".join(kept[:3]))
            emotions = raw.get("emotions") if isinstance(raw.get("emotions"), Mapping) else raw.get("emotions")
            if isinstance(emotions, Mapping):
                primary = emotions.get("primary") or emotions.get("label")
                valence = emotions.get("valence")
                arousal = emotions.get("arousal")
                if isinstance(primary, str) and primary:
                    extras.append(f"émotion={primary}")
                if isinstance(valence, (int, float)):
                    valence_series.append(float(valence))
                if isinstance(arousal, (int, float)):
                    arousal_series.append(float(arousal))
            context = raw.get("context") if isinstance(raw.get("context"), Mapping) else {}
            metrics = context.get("metrics") if isinstance(context, Mapping) else {}
            if isinstance(metrics, Mapping):
                highlight = []
                for key in ("priority", "uncertainty", "sj_reward", "calibration_gap"):
                    value = metrics.get(key)
                    if isinstance(value, (int, float)):
                        highlight.append(f"{key}={value:.2f}")
                if highlight:
                    extras.append("; ".join(highlight))
            body = raw.get("body") if isinstance(raw.get("body"), Mapping) else {}
            if isinstance(body, Mapping):
                homeo = body.get("homeostasis")
                if isinstance(homeo, Mapping) and homeo:
                    top_drive = next(iter(homeo.items()))
                    try:
                        extras.append(f"drive {top_drive[0]}={float(top_drive[1]):.2f}")
                    except Exception:
                        pass
            line = summary
            if extras:
                line = f"{summary} ({' | '.join(extras)})"
            selected.append({
                "ts": float(raw.get("ts", 0.0)),
                "text": line,
            })

        if not selected:
            return None
        selected.sort(key=lambda item: item["ts"])
        lines = [item["text"] for item in selected]
        timestamps = [item["ts"] for item in selected]
        if len(timestamps) > 1:
            gaps = [max(0.0, min(1.0, (b - a) / 3_600.0)) for a, b in zip(timestamps, timestamps[1:])]
            coherence = 1.0 - (sum(gaps) / len(gaps))
        else:
            coherence = 1.0
        coherence = max(0.0, min(1.0, coherence))

        result: Dict[str, Any] = {
            "narrative": "\n".join(lines),
            "coherence": coherence,
            "episodes": len(lines),
            "source": "phenomenal_journal",
        }
        if values_set:
            result["values"] = sorted(values_set)
        if principles_set:
            result["principles"] = sorted(principles_set)
        if valence_series or arousal_series:
            span: Dict[str, Tuple[float, float]] = {}
            if valence_series:
                span["valence"] = (min(valence_series), max(valence_series))
            if arousal_series:
                span["arousal"] = (min(arousal_series), max(arousal_series))
            result["emotion_span"] = span
        return result

    def form_autobiographical_narrative(self) -> Dict[str, Any]:
        phenomenal_story = self._phenomenal_autobiographical_stream()
        if phenomenal_story is not None:
            return phenomenal_story
        """
        Forme un récit autobiographique à partir des mémoires épisodiques
        """
        episodic_memories = list(self.long_term_memory[MemoryType.EPISODIC].values())

        if not episodic_memories:
            return {"narrative": "Aucune expérience mémorable encore.", "coherence": 0.0}
        
        # Tri chronologique
        episodic_memories.sort(key=lambda x: x.timestamp)
        
        # Extraction des événements significatifs
        significant_events = [
            mem for mem in episodic_memories 
            if mem.strength > 0.7 or abs(mem.valence) > 0.6
        ]
        
        # Construction du récit
        narrative_parts = []
        total_coherence = 0.0
        
        for i, event in enumerate(significant_events):
            event_description = self._describe_memory_event(event)
            narrative_parts.append(event_description)
            
            # Calcul de la cohérence avec l'événement précédent
            if i > 0:
                prev_event = significant_events[i-1]
                coherence = self._calculate_temporal_coherence(prev_event, event)
                total_coherence += coherence
        
        average_coherence = total_coherence / (len(significant_events) - 1) if len(significant_events) > 1 else 1.0
        
        narrative = " • ".join(narrative_parts)

        result = {
            "narrative": narrative,
            "coherence": average_coherence,
            "significant_events": len(significant_events),
            "timespan": episodic_memories[-1].timestamp - episodic_memories[0].timestamp,
        }

        if significant_events:
            llm_payload = {
                "base_narrative": narrative,
                "initial_coherence": average_coherence,
                "events": [
                    {
                        "id": event.id,
                        "timestamp": event.timestamp,
                        "strength": event.strength,
                        "valence": event.valence,
                        "description": self._describe_memory_event(event),
                    }
                    for event in significant_events[:10]
                ],
            }
            llm_response = try_call_llm_dict(
                "memory_system_narrative",
                input_payload=llm_payload,
                logger=LOGGER,
            )
            if llm_response:
                enhanced = llm_response.get("enhanced_narrative")
                if enhanced:
                    result["llm_enhanced_narrative"] = str(enhanced)
                if "coherence" in llm_response:
                    try:
                        result["coherence"] = float(llm_response["coherence"])
                    except Exception:
                        pass
                if llm_response.get("insights"):
                    result["llm_insights"] = list(llm_response.get("insights", []))
                if llm_response.get("notes"):
                    result["llm_notes"] = llm_response.get("notes")

        return result
    
    def _describe_memory_event(self, memory: MemoryTrace) -> str:
        """Génère une description textuelle d'un événement mémoire"""
        content_str = str(memory.content)
        
        # Simplification pour l'exemple
        if len(content_str) > 50:
            content_str = content_str[:47] + "…"
        
        emotion_desc = "neutre"
        if memory.valence < -0.3:
            emotion_desc = "négatif"
        elif memory.valence > 0.3:
            emotion_desc = "positif"
        
        return f"[{emotion_desc}] {content_str}"
    
    def _calculate_temporal_coherence(self, event1: MemoryTrace, event2: MemoryTrace) -> float:
        """Calcule la cohérence temporelle entre deux événements"""
        time_gap = event2.timestamp - event1.timestamp
        
        # Cohérence plus élevée pour des événements rapprochés
        if time_gap < 3600:  # 1 heure
            return 0.9
        elif time_gap < 86400:  # 1 jour
            return 0.7
        elif time_gap < 604800:  # 1 semaine
            return 0.5
        else:
            return 0.3
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du système de mémoire"""
        stats = {
            "total_memories": self.memory_metadata["total_memories"],
            "memory_by_type": {},
            "average_strength": 0.0,
            "average_accessibility": 0.0,
            "consolidation_queue": len(self.consolidation_process["active_consolidation"]),
            "working_memory_load": self._calculate_working_memory_load()
        }
        
        total_strength = 0
        total_accessibility = 0
        memory_count = 0
        
        for memory_type, memories_dict in self.long_term_memory.items():
            type_count = len(memories_dict)
            stats["memory_by_type"][memory_type.value] = type_count
            
            for memory in memories_dict.values():
                total_strength += memory.strength
                total_accessibility += memory.accessibility
                memory_count += 1
        
        if memory_count > 0:
            stats["average_strength"] = total_strength / memory_count
            stats["average_accessibility"] = total_accessibility / memory_count
        
        return stats
    
    def _calculate_working_memory_load(self) -> float:
        """Calcule la charge actuelle de la mémoire de travail"""
        total_items = 0
        total_capacity = 0
        
        for component_name, component in self.working_memory.items():
            if component_name != "central_executive":
                total_items += len(component["contents"])
                total_capacity += component["capacity"]
        
        if total_capacity == 0:
            return 0.0
        
        return total_items / total_capacity

# Test du système de mémoire
if __name__ == "__main__":
    print("💾 TEST DU SYSTÈME DE MÉMOIRE")
    print("=" * 50)
    
    # Création du système
    memory_system = MemorySystem()
    
    # Test d'encodage de mémoires
    test_memories = [
        {
            "content": "Première découverte de la gravité en voyant un objet tomber",
            "type": MemoryType.EPISODIC,
            "context": {"location": "laboratoire", "activity": "observation"},
            "valence": 0.8
        },
        {
            "content": {"concept": "gravité", "definition": "Force d'attraction entre les masses"},
            "type": MemoryType.SEMANTIC, 
            "context": {"domain": "physique", "certainty": "high"},
            "valence": 0.3
        },
        {
            "content": "Procédure pour résoudre des équations simples",
            "type": MemoryType.PROCEDURAL,
            "context": {"skill_level": "beginner", "domain": "mathématiques"},
            "valence": 0.6
        }
    ]
    
    print("\n📝 Encodage des mémoires de test en cours")
    memory_ids = []
    for mem_data in test_memories:
        mem_id = memory_system.encode_memory(
            content=mem_data["content"],
            memory_type=mem_data["type"],
            context=mem_data["context"],
            valence=mem_data["valence"]
        )
        memory_ids.append(mem_id)
        print(f"Encodé: {mem_id}")
    
    # Test de récupération
    print("\n🔍 Test de récupération en cours")
    retrieval_result = memory_system.retrieve_memories(
        cues={"context": {"activity": "observation"}},
        memory_type=MemoryType.EPISODIC
    )
    
    print(f"Mémoires récupérées: {len(retrieval_result.memory_traces)}")
    print(f"Confiance: {retrieval_result.confidence:.2f}")
    for memory in retrieval_result.memory_traces:
        print(f" - {memory.content}")
    
    # Test de consolidation
    print("\n🔄 Test de consolidation en cours")
    consolidation_result = memory_system.consolidate_memories()
    print(f"Résultat: {consolidation_result}")
    
    # Statistiques
    print("\n📊 Statistiques du système:")
    stats = memory_system.get_memory_stats()
    for key, value in stats.items():
        print(f" - {key}: {value}")
    
    # Récit autobiographique
    print("\n📖 Récit autobiographique:")
    narrative = memory_system.form_autobiographical_narrative()
    print(f"Narrative: {narrative['narrative']}")
    print(f"Cohérence: {narrative['coherence']:.2f}")