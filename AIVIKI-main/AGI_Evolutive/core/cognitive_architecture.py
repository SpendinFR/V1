import json
import datetime as dt
import logging
import os
import re
import time
import unicodedata
from collections import deque
from copy import deepcopy
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple

from AGI_Evolutive.autonomy import AutonomyManager
from AGI_Evolutive.autonomy.auto_evolution import AutoEvolutionCoordinator
from AGI_Evolutive.autonomy.auto_signals import AutoSignalRegistry
from AGI_Evolutive.beliefs.graph import BeliefGraph, Evidence
from AGI_Evolutive.knowledge.ontology_facade import EntityLinker, Ontology
from AGI_Evolutive.cognition.evolution_manager import EvolutionManager
from AGI_Evolutive.cognition.reward_engine import RewardEngine, RewardEvent
from AGI_Evolutive.core.telemetry import Telemetry
from AGI_Evolutive.core.question_manager import QuestionManager
from AGI_Evolutive.creativity import CreativitySystem
from AGI_Evolutive.emotions import EmotionalSystem
from AGI_Evolutive.goals import GoalSystem, GoalType
from AGI_Evolutive.goals.dag_store import GoalDAG
from AGI_Evolutive.io.action_interface import ActionInterface
from AGI_Evolutive.io.perception_interface import PerceptionInterface
from AGI_Evolutive.language.understanding import SemanticUnderstanding
from AGI_Evolutive.language.style_policy import StylePolicy
from AGI_Evolutive.language.social_reward import extract_social_reward
from AGI_Evolutive.language.style_profiler import StyleProfiler
from AGI_Evolutive.language.nlg import NLGContext, apply_mai_bids_to_nlg
from AGI_Evolutive.learning import ExperientialLearning
from AGI_Evolutive.memory import MemorySystem
from AGI_Evolutive.memory.concept_extractor import ConceptExtractor
from AGI_Evolutive.memory.episodic_linker import EpisodicLinker
from AGI_Evolutive.memory.vector_store import VectorStore
from AGI_Evolutive.retrieval.adaptive_controller import RAGAdaptiveController
from AGI_Evolutive.metacog.calibration import CalibrationMeter, NoveltyDetector
from AGI_Evolutive.metacognition import MetacognitiveSystem
from AGI_Evolutive.models import IntentModel, UserModel
from AGI_Evolutive.perception import PerceptionSystem
from AGI_Evolutive.reasoning import ReasoningSystem
from AGI_Evolutive.reasoning.abduction import AbductiveReasoner, Hypothesis
from AGI_Evolutive.reasoning.causal import CounterfactualSimulator, SCMStore
from AGI_Evolutive.reasoning.question_engine import QuestionEngine
from AGI_Evolutive.runtime.logger import JSONLLogger
from AGI_Evolutive.runtime.response import (
    ensure_contract,
    format_agent_reply,
    humanize_reasoning_block,
)
from AGI_Evolutive.runtime.scheduler import Scheduler
from AGI_Evolutive.runtime.job_manager import JobManager
from AGI_Evolutive.world_model import PhysicsEngine
from AGI_Evolutive.self_improver import SelfImprover
from AGI_Evolutive.self_improver.code_evolver import CodeEvolver
from AGI_Evolutive.self_improver.promote import PromotionManager
from AGI_Evolutive.self_improver.skill_acquisition import SkillSandboxManager
from AGI_Evolutive.planning.htn import HTNPlanner
from AGI_Evolutive.core.persistence import PersistenceManager
from AGI_Evolutive.core.self_model import SelfModel
from AGI_Evolutive.core.config import cfg
from AGI_Evolutive.utils.jsonsafe import json_sanitize
from AGI_Evolutive.utils.llm_service import try_call_llm_dict


logger = logging.getLogger(__name__)


class CognitiveArchitecture:
    """Central coordinator for the agent's cognitive subsystems."""

    def __init__(self, boot_minimal: bool = False):
        self.boot_minimal = boot_minimal
        logger.info(
            "Initialisation de la CognitiveArchitecture",
            extra={"boot_minimal": bool(boot_minimal)},
        )
        # Observability
        self.logger = JSONLLogger("runtime/agent_events.jsonl")
        self.telemetry = Telemetry()
        self.style_policy = StylePolicy()
        self.intent_model = IntentModel()
        self.question_manager = QuestionManager(self)
        self._last_intent_decay = time.time()
        self.goal_dag = GoalDAG("runtime/goal_dag.json")

        # Global state
        self.global_activation = 0.5
        self.start_time = time.time()
        self.reflective_mode = True
        self.last_output_text = "OK"
        self.last_user_id = "default"
        self._memory_request_goal_id: Optional[str] = None
        self.rag_adaptive: Optional[RAGAdaptiveController] = None
        self._rag_last_context: Optional[Dict[str, Any]] = None
        self._rag_feedback_queue = deque(maxlen=24)

        # Core subsystems
        self.telemetry.log("init", "core", {"stage": "memory"})
        self.vector_store = VectorStore()
        self.memory = MemorySystem(self)
        from AGI_Evolutive.memory.semantic_manager import (  # type: ignore  # local import avoids circular init
            SemanticMemoryManager,
        )

        self.memory.semantic = SemanticMemoryManager(
            self.memory,
            architecture=self,
            index_backend=self.vector_store,
        )
        if getattr(self.memory, "retrieval", None) is not None:
            setattr(self.memory.retrieval, "vector_store", self.vector_store)

        # === RAG 5★ : init lazy & optionnel ===
        self.rag = None
        self.rag_cfg = None
        try:
            try:
                with open("configs/rag.json", "r", encoding="utf-8") as fh:
                    self.rag_cfg = json.load(fh)
            except Exception:
                self.rag_cfg = {
                    "retrieval": {
                        "topk_dense": 200,
                        "topk_sparse": 100,
                        "topk_fused": 80,
                        "alpha_dense": 0.6,
                        "beta_sparse": 0.4,
                        "recency_boost": 0.2,
                        "recency_half_life_days": 14.0,
                    },
                    "ann": {
                        "backend": "faiss",
                        "hnsw": True,
                        "efSearch": 128,
                        "M": 32,
                        "metric": "ip",
                    },
                    "rerank": {"topk": 30, "mmr_lambda": 0.7},
                    "compose": {
                        "budget_tokens": 1500,
                        "snippet_chars": 420,
                        "tokenizer": "bert-base-multilingual-cased",
                    },
                    "guards": {
                        "min_support_docs": 2,
                        "min_support_score": 0.15,
                        "min_top1_score": 0.25,
                        "refuse_message": "Je ne peux pas répondre de façon fiable : support insuffisant. Peux-tu préciser la question ou partager une source ?",
                    },
                }

            try:
                self.rag_adaptive = RAGAdaptiveController(self.rag_cfg)
                self.rag_cfg = self.rag_adaptive.current_config()
            except Exception as adaptive_exc:
                self.rag_adaptive = None
                if hasattr(self, "logger"):
                    try:
                        self.logger.warning(
                            "RAG adaptatif désactivé: %s", adaptive_exc
                        )
                    except Exception:
                        pass

            # Import du pipeline uniquement ici, et seulement si on va l’utiliser
            try:
                from AGI_Evolutive.retrieval.rag5.pipeline import RAGPipeline

                self.rag = RAGPipeline(self.rag_cfg)
                if self.rag_adaptive:
                    self._apply_rag_config(self.rag_cfg)
                # Seed initial depuis la mémoire (non bloquant)
                try:
                    for m in self.memory.get_recent_memories(n=2000):
                        txt = m.get("text")
                        if txt:
                            self.rag.add_document(
                                m.get("id", f"mem#{int(m.get('ts', 0))}"),
                                txt,
                                meta={"ts": m.get("ts"), "source_trust": 0.6},
                            )
                except Exception:
                    pass
            except Exception as e:
                # RAG restera désactivé si dépendances absentes
                self.rag = None
                if hasattr(self, "logger"):
                    self.logger.warning(
                        "RAG désactivé (dépendances manquantes ou erreur d'init): %s", e
                    )
        except Exception:
            # Ne jamais empêcher l'architecture de booter
            self.rag = None

        logger.info(
            "Statut RAG",
            extra={
                "actif": bool(self.rag),
                "adaptatif": bool(self.rag_adaptive),
            },
        )

        self.telemetry.log("init", "core", {"stage": "perception"})
        self.perception = PerceptionSystem(self, self.memory)

        self.telemetry.log("init", "core", {"stage": "reasoning"})
        self.reasoning = ReasoningSystem(self, self.memory, self.perception)

        self.telemetry.log("init", "core", {"stage": "goals"})
        self.goals = GoalSystem(self, self.memory, self.reasoning)

        self.telemetry.log("init", "core", {"stage": "metacognition"})
        self.metacognition = MetacognitiveSystem(self, self.memory, self.reasoning)

        self.telemetry.log("init", "core", {"stage": "emotions"})
        self.emotions = EmotionalSystem(self, self.memory, self.metacognition)

        self.telemetry.log("init", "core", {"stage": "learning"})
        self.learning = ExperientialLearning(self)

        self.telemetry.log("init", "core", {"stage": "creativity"})
        self.creativity = CreativitySystem(
            self, self.memory, self.reasoning, self.emotions, self.metacognition
        )

        self.telemetry.log("init", "core", {"stage": "world_model"})
        self.world_model = PhysicsEngine(self, self.memory)

        self.telemetry.log("init", "core", {"stage": "language"})
        self.language = SemanticUnderstanding(self, self.memory, intent_model=self.intent_model)

        self.concept_extractor = ConceptExtractor(self.memory)
        self.episodic_linker = EpisodicLinker(self.memory)

        self.action_interface = ActionInterface()
        self.auto_signals = AutoSignalRegistry()
        self.perception_interface = PerceptionInterface()

        # Advanced subsystems
        self.style_profiler = StyleProfiler(persist_path="data/style_profiles.json")
        self.ontology = Ontology()
        self.beliefs = BeliefGraph(ontology=self.ontology)
        self.entity_linker = EntityLinker(self.ontology, self.beliefs)
        self.beliefs.set_entity_linker(self.entity_linker)
        self.scm = SCMStore(self.beliefs, self.ontology)
        self.simulator = CounterfactualSimulator(self.scm)
        self.planner = HTNPlanner(self.beliefs, self.ontology)
        self.user_model = UserModel()
        self.self_model = SelfModel()
        try:
            setattr(self.self_model, "arch", self)
        except Exception:
            pass
        try:
            persona_tone = (self.user_model.describe().get("persona", {}) or {}).get("tone")
            if persona_tone:
                self.style_policy.update_persona_tone(persona_tone)
        except Exception:
            pass
        self.calibration = CalibrationMeter()
        self.calibration_abduction = CalibrationMeter(path="data/calibration_abduction.jsonl")
        self.calibration_concepts = CalibrationMeter(path="data/calibration_concepts.jsonl")
        self.calibration_causal = CalibrationMeter(path="data/calibration_causal.jsonl")
        self.calibration_plan = CalibrationMeter(path="data/calibration_plan.jsonl")
        self.novelty_detector = NoveltyDetector()
        self.abduction = AbductiveReasoner(self.beliefs, self.user_model)
        self.abduction.qengine = QuestionEngine(self.beliefs, self.user_model)
        self.reward_engine = RewardEngine(
            architecture=self,
            memory=self.memory,
            emotions=self.emotions,
            goals=self.goals,
            metacognition=self.metacognition,
            persist_dir="data",
        )
        try:
            self.reward_engine.register_listener(self._on_reward_event)
        except Exception:
            pass

        self.autonomy = AutonomyManager(
            architecture=self,
            goal_system=self.goals,
            metacognition=self.metacognition,
            memory=self.memory,
            perception=self.perception,
            language=self.language,
        )

        def _apply_overrides(arch: "CognitiveArchitecture", ov: Dict[str, Any]) -> None:
            if not ov:
                return
            style_policy = getattr(arch, "style_policy", None)
            if style_policy and hasattr(style_policy, "params"):
                if "style.hedging" in ov:
                    style_policy.params["hedging"] = max(0.0, min(1.0, float(ov["style.hedging"])))

            if "learning.self_assess.threshold" in ov and hasattr(arch, "learning"):
                try:
                    threshold = float(ov["learning.self_assess.threshold"])
                    setattr(arch.learning, "self_assess_threshold", threshold)
                except Exception:
                    pass

            abduction = getattr(arch, "abduction", None)
            if not abduction:
                return
            if "abduction.tie_gap" in ov:
                setattr(abduction, "tie_gap", float(ov["abduction.tie_gap"]))
            if "abduction.weights.prior" in ov:
                setattr(abduction, "w_prior", float(ov["abduction.weights.prior"]))
            if "abduction.weights.boost" in ov:
                setattr(abduction, "w_boost", float(ov["abduction.weights.boost"]))
            if "abduction.weights.match" in ov:
                setattr(abduction, "w_match", float(ov["abduction.weights.match"]))

        def _arch_factory(overrides: Dict[str, Any]) -> "CognitiveArchitecture":
            fresh: "CognitiveArchitecture"
            try:
                fresh = self.__class__(boot_minimal=True)
            except Exception:
                fresh = self
            try:
                _apply_overrides(fresh, overrides or {})
            except Exception:
                pass
            return fresh

        self._arch_factory = _arch_factory
        self.promotions: Optional[PromotionManager]
        if not boot_minimal:
            self.self_improver = SelfImprover(
                arch_factory=self._arch_factory,
                memory=self.memory,
                question_manager=getattr(self, "question_manager", None),
                apply_overrides=lambda overrides: _apply_overrides(self, overrides),
            )
            self.promotions = self.self_improver.prom
            self.code_evolver: Optional[CodeEvolver] = getattr(self.self_improver, "code_evolver", None)
        else:
            self.self_improver = None
            self.promotions = None
            self.code_evolver = None

        # Persistence layer shared with Autopilot/logger
        existing_persistence = getattr(self, "persistence", None)
        if isinstance(existing_persistence, PersistenceManager):
            self.persistence = existing_persistence
        else:
            self.persistence = PersistenceManager(self)
        self.logger.persistence = self.persistence

        # Job manager for background/offloaded actions
        try:
            data_dir = cfg().get("DATA_DIR", "data")
        except Exception:
            data_dir = "data"
        self.jobs = JobManager(self, data_dir=data_dir)

        skill_storage = os.path.join(data_dir, "skills")
        self.skill_sandbox = SkillSandboxManager(
            storage_dir=skill_storage,
            min_trials=3,
            success_threshold=0.75,
            approval_required=True,
        )
        self.skill_sandbox.bind(
            memory=self.memory,
            language=self.language,
            simulator=self.simulator,
            jobs=self.jobs,
            arch=self,
            interface=self.action_interface,
            question_manager=getattr(self, "question_manager", None),
            perception=self.perception_interface,
        )

        # Bind helper components
        self._bind_interfaces()
        self._bind_extractors()

        # Long-term modules
        self.evolution = EvolutionManager(data_dir="data")
        self.evolution.bind(
            architecture=self,
            memory=self.memory,
            metacog=self.metacognition,
            goals=self.goals,
            learning=self.learning,
            emotions=self.emotions,
            language=self.language,
        )

        # Scheduler runs background maintenance work (daemon thread)
        self.scheduler = None
        if not boot_minimal:
            self.scheduler = Scheduler(self, data_dir="data")
            self.scheduler.start()

        # Auto evolution orchestrator (runs even if scheduler disabled)
        mechanism_store = getattr(self, "mechanism_store", None)
        auto_modules = self._auto_evolution_targets()
        self.auto_evolution = AutoEvolutionCoordinator(
            memory=self.memory,
            metacog=self.metacognition,
            skill_sandbox=self.skill_sandbox,
            evolution_manager=self.evolution,
            mechanism_store=mechanism_store,
            self_improver=self.self_improver,
            goals=self.goals,
            emotions=self.emotions,
            modules=auto_modules,
            signal_registry=self.auto_signals,
        )
        try:
            self.auto_evolution.install()
        except Exception:
            logger.exception("Échec du démarrage de l'AutoEvolutionCoordinator")

        self.telemetry.log("ready", "core", {"status": "initialized"})
        self._cycle_counter = 0
        self._decay_period = 8

        logger.info(
            "CognitiveArchitecture prête",
            extra={
                "boot_minimal": bool(boot_minimal),
                "scheduler_actif": bool(self.scheduler),
            },
        )

    # ------------------------------------------------------------------
    def _auto_evolution_targets(self) -> List[Any]:
        candidates: List[Any] = [
            self,
            getattr(self, "autonomy", None),
            getattr(self, "action_interface", None),
            getattr(self, "perception", None),
            getattr(self, "perception_interface", None),
            getattr(self, "reasoning", None),
            getattr(self, "language", None),
            getattr(self, "learning", None),
            getattr(self, "emotions", None),
            getattr(self, "creativity", None),
            getattr(self, "goals", None),
            getattr(self, "memory", None),
            getattr(self, "metacognition", None),
            getattr(self, "evolution", None),
            getattr(self, "reward_engine", None),
            getattr(self, "planner", None),
            getattr(self, "simulator", None),
            getattr(self, "self_improver", None),
            getattr(self, "world_model", None),
        ]
        extras = getattr(self, "auto_evolution_receivers", None)
        if isinstance(extras, (list, tuple, set)):
            candidates.extend(extras)
        seen: Set[int] = set()
        filtered: List[Any] = []
        for module in candidates:
            if module is None:
                continue
            identity = id(module)
            if identity in seen:
                continue
            seen.add(identity)
            filtered.append(module)
        return filtered

    # ------------------------------------------------------------------
    # Helpers
    def _bind_interfaces(self) -> None:
        policy = getattr(self, "policy", None)
        self.action_interface.bind(
            arch=self,
            goals=self.goals,
            policy=policy,
            memory=self.memory,
            metacog=self.metacognition,
            emotions=self.emotions,
            language=self.language,
            simulator=self.simulator,
            jobs=self.jobs,
            skills=self.skill_sandbox,
            perception=self.perception_interface,
            auto_signals=self.auto_signals,
        )

    # ------------------------------------------------------------------
    def on_auto_intention_promoted(
        self,
        event: Mapping[str, Any],
        evaluation: Optional[Mapping[str, Any]] = None,
        self_assessment: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not isinstance(event, Mapping):
            return
        summary = {
            "action_type": event.get("action_type"),
            "score": (evaluation or {}).get("score"),
            "significance": (evaluation or {}).get("significance"),
            "alignment": (evaluation or {}).get("alignment"),
        }
        try:
            self.telemetry.log("auto_evolution", "intent_promoted", summary)
        except Exception:
            pass
        try:
            if self.logger:
                self.logger.write(
                    "auto_evolution.intent",
                    event=event,
                    evaluation=evaluation,
                    self_assessment=self_assessment,
                )
        except Exception:
            pass

    def _bind_extractors(self) -> None:
        self.concept_extractor.bind(
            memory=self.memory,
            emotions=self.emotions,
            metacog=self.metacognition,
            language=self.language,
        )
        self.episodic_linker.bind(
            memory=self.memory,
            language=self.language,
            metacog=self.metacognition,
            emotions=self.emotions,
        )

    def _apply_rag_config(self, config: Dict[str, Any]) -> None:
        if not isinstance(config, dict):
            return
        try:
            cfg = deepcopy(config)
        except Exception:
            cfg = config
        self.rag_cfg = cfg
        if self.rag is not None:
            try:
                self.rag.cfg = cfg
            except Exception:
                pass

    def prepare_rag_query(self, question: str) -> Optional[Dict[str, Any]]:
        if not self.rag_adaptive:
            return None
        try:
            context = self.rag_adaptive.prepare_query(question)
            cfg = context.get("config")
            if isinstance(cfg, dict):
                self._apply_rag_config(cfg)
            self._rag_last_context = context
            return context
        except Exception as exc:
            self._rag_last_context = None
            if hasattr(self, "logger"):
                try:
                    self.logger.warning("RAG adaptatif – préparation échouée: %s", exc)
                except Exception:
                    pass
            return None

    def observe_rag_outcome(
        self,
        question: str,
        rag_out: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.rag_adaptive:
            return
        ctx = context or self._rag_last_context
        if not isinstance(ctx, dict):
            return
        payload = dict(ctx)
        payload["question"] = question
        try:
            self.rag_adaptive.observe_outcome(payload, rag_out)
            self._rag_feedback_queue.append({"ts": time.time()})
        except Exception as exc:
            if hasattr(self, "logger"):
                try:
                    self.logger.warning("RAG adaptatif – observation échouée: %s", exc)
                except Exception:
                    pass
        finally:
            self._rag_last_context = None

    def _on_reward_event(self, event: RewardEvent) -> None:
        if not self.rag_adaptive:
            return
        if not isinstance(event, RewardEvent):
            return
        if getattr(event, "channel", "") != "chat":
            return
        now = float(getattr(event, "timestamp", time.time()))
        while self._rag_feedback_queue and now - self._rag_feedback_queue[0]["ts"] > 60.0:
            self._rag_feedback_queue.popleft()
        if not self._rag_feedback_queue:
            return
        intensity = getattr(event, "intensity", 0.0)
        try:
            intensity = float(intensity)
        except Exception:
            intensity = 0.0
        multiplier = max(0.2, min(1.0, intensity)) if intensity > 0 else 0.2
        reward = getattr(event, "extrinsic_reward", 0.0)
        try:
            reward = float(reward)
        except Exception:
            reward = 0.0
        reward = max(0.0, min(1.0, (reward + 1.0) / 2.0))
        reward *= multiplier
        try:
            self.rag_adaptive.apply_feedback(reward)
        except Exception as exc:
            if hasattr(self, "logger"):
                try:
                    self.logger.warning("RAG adaptatif – feedback rejeté: %s", exc)
                except Exception:
                    pass
        finally:
            if self._rag_feedback_queue:
                self._rag_feedback_queue.popleft()

    def bump_global_activation(self, delta: float, reinforce: Optional[float] = None) -> None:
        base = max(0.0, min(1.0, float(self.global_activation) + float(delta)))
        if self.rag_adaptive:
            try:
                self.global_activation = self.rag_adaptive.update_global_activation(base, reinforce=reinforce)
                return
            except Exception:
                pass
        self.global_activation = base

    def on_affect_modulators(self, mods: Dict[str, Any]) -> None:
        if not isinstance(mods, dict):
            return
        delta = mods.get("activation_delta", 0.0)
        try:
            delta = float(delta)
        except Exception:
            delta = 0.0
        feedback = mods.get("activation_feedback")
        reinforce = None
        if isinstance(feedback, (int, float)):
            reinforce = max(0.0, min(1.0, 0.5 + 0.5 * float(feedback)))
        self.bump_global_activation(delta, reinforce=reinforce)
        mods["_activation_handled"] = True

    def _language_state_snapshot(self) -> Dict[str, Any]:
        language = getattr(self, "language", None)
        dialogue = None
        if language is not None:
            dialogue = getattr(language, "state", None)
            dialogue = getattr(language, "dialogue_state", dialogue)
        return {
            "beliefs": getattr(self, "beliefs", None),
            "self_model": getattr(self, "self_model", None),
            "dialogue": dialogue,
            "world": getattr(self, "world_model", None),
            "memory": getattr(self, "memory", None),
        }

    def _predicate_registry_for_state(self, state: Dict[str, Any]) -> Dict[str, Callable[..., bool]]:
        policy = getattr(self, "policy", None)
        if policy is not None and hasattr(policy, "build_predicate_registry"):
            try:
                registry = policy.build_predicate_registry(state)
                if isinstance(registry, dict):
                    return registry
            except Exception:
                pass

        dialogue = state.get("dialogue")
        world = state.get("world")
        self_model = state.get("self_model")
        beliefs = state.get("beliefs")

        def _belief_contains(topic: Any) -> bool:
            if beliefs is None:
                return False
            for accessor in ("contains", "has_fact", "has_edge"):
                fn = getattr(beliefs, accessor, None)
                if callable(fn):
                    try:
                        if fn(topic):
                            return True
                    except Exception:
                        continue
            return False

        def _belief_confidence(topic: Any, threshold: float) -> bool:
            if beliefs is None:
                return False
            confidence_for = getattr(beliefs, "confidence_for", None)
            if not callable(confidence_for):
                return False
            try:
                return float(confidence_for(topic)) >= float(threshold)
            except Exception:
                return False

        registry: Dict[str, Callable[..., bool]] = {
            "request_is_sensitive": lambda st: getattr(dialogue, "is_sensitive", False) if dialogue else False,
            "audience_is_not_owner": lambda st: (
                getattr(dialogue, "audience_id", None) != getattr(dialogue, "owner_id", None)
                if dialogue
                else False
            ),
            "has_consent": lambda st: getattr(dialogue, "has_consent", False) if dialogue else False,
            "imminent_harm_detected": lambda st: getattr(world, "imminent_harm", False) if world else False,
            "has_commitment": lambda st, key: (
                self_model.has_commitment(key)
                if hasattr(self_model, "has_commitment")
                else False
            ),
            "belief_mentions": lambda st, topic: _belief_contains(topic),
            "belief_confidence_above": lambda st, topic, threshold: _belief_confidence(topic, threshold),
        }
        return registry

    def _resolve_nlg_hint_applier(self) -> Callable[[str, str], str]:
        renderer = getattr(self, "renderer", None)
        if renderer is not None and hasattr(renderer, "apply_action_hint"):
            fn = getattr(renderer, "apply_action_hint")
            if callable(fn):
                return lambda text, hint: fn(text, hint)

        try:
            from AGI_Evolutive.language.renderer import _apply_action_hint

            return lambda text, hint: _apply_action_hint(text, hint)
        except Exception:
            return lambda text, hint: text

    def reindex_rag_from_memory(self, limit: int = 10000) -> int:
        """(Ré)indexe des souvenirs texte récents dans le RAG. Retourne le nb de docs ajoutés."""
        if not getattr(self, "rag", None):
            return 0
        n = 0
        try:
            for m in self.memory.get_recent_memories(n=limit):
                txt = m.get("text")
                if txt:
                    self.rag.add_document(
                        m.get("id", f"mem#{int(m.get('ts', 0))}"),
                        txt,
                        meta={"ts": m.get("ts"), "source_trust": 0.6},
                    )
                    n += 1
        except Exception:
            pass
        return n

    # ------------------------------------------------------------------
    # Status & reporting
    def _present_subsystems(self) -> Dict[str, bool]:
        names = [
            "memory",
            "perception",
            "reasoning",
            "goals",
            "metacognition",
            "emotions",
            "learning",
            "creativity",
            "world_model",
            "language",
        ]
        return {name: getattr(self, name, None) is not None for name in names}

    def get_cognitive_status(self) -> Dict[str, Any]:
        wm_load = 0.0
        try:
            wm = getattr(self.memory, "working_memory", None)
            if wm and hasattr(wm, "__len__"):
                wm_load = min(len(wm) / 10.0, 1.0)
        except Exception:
            wm_load = 0.0

        return {
            "uptime_s": int(time.time() - self.start_time),
            "global_activation": float(self.global_activation),
            "working_memory_load": float(wm_load),
            "subsystems": self._present_subsystems(),
            "style_policy": self.style_policy.as_dict(),
            "goal_focus": self.goal_dag.choose_next_goal(),
        }

    def diagnostic_snapshot(self, tail: int = 30) -> Dict[str, Any]:
        snapshot = {
            "status": self.get_cognitive_status(),
            "tail": self.telemetry.tail(tail),
        }
        analysis = self._llm_diagnostic(snapshot)
        if analysis:
            snapshot["analysis"] = analysis
        return snapshot

    def _llm_diagnostic(self, snapshot: Mapping[str, Any]) -> Dict[str, Any]:
        fallback = self._fallback_diagnostic(snapshot)
        payload = {
            "status": snapshot.get("status", {}),
            "tail": list(snapshot.get("tail", []))[:15],
            "recent_reply": self.last_output_text,
        }
        response = try_call_llm_dict(
            "cognitive_state_summary",
            input_payload=json_sanitize(payload),
            logger=getattr(self, "logger", logger),
            max_retries=2,
        )
        if not isinstance(response, Mapping):
            return fallback

        summary = self._clean_text(response.get("summary")) or fallback.get("summary", "")
        notes = self._clean_text(response.get("notes")) or fallback.get("notes", "")
        alerts = self._clean_list(response.get("alerts"))
        recommendations = self._clean_list(response.get("recommended_actions"))
        confidence = fallback.get("confidence", 0.6)
        try:
            confidence = float(response.get("confidence", confidence))
        except Exception:
            confidence = float(confidence)

        enriched = dict(fallback)
        enriched.update(
            {
                "source": "llm",
                "summary": summary,
                "alerts": alerts,
                "recommended_actions": recommendations or fallback.get("recommended_actions", []),
                "confidence": max(0.0, min(1.0, confidence)),
                "notes": notes,
            }
        )
        return enriched

    def _fallback_diagnostic(self, snapshot: Mapping[str, Any]) -> Dict[str, Any]:
        status = snapshot.get("status", {})
        activation = float(status.get("global_activation", 0.0) or 0.0)
        wm_load = float(status.get("working_memory_load", 0.0) or 0.0)
        subsystems = status.get("subsystems", {}) if isinstance(status, Mapping) else {}
        missing = [name for name, present in subsystems.items() if not present]
        alerts: List[str] = []
        if activation < 0.25:
            alerts.append("Activation globale faible : risque de léthargie cognitive.")
        if wm_load > 0.8:
            alerts.append("Charge mémoire de travail élevée : envisager un vidage.")
        for name in missing:
            alerts.append(f"Sous-système absent : {name}")
        recommendations: List[str] = []
        if activation < 0.25:
            recommendations.append("Initier un cycle d'exploration légère pour relancer l'activation.")
        if wm_load > 0.8:
            recommendations.append("Archiver ou résumer le contenu mémoire non critique.")
        if missing:
            recommendations.append("Vérifier l'initialisation des sous-systèmes manquants.")

        summary = "État cognitif stable." if not alerts else "État cognitif à surveiller."
        notes = "Analyse heuristique basée sur les métriques disponibles."
        return {
            "source": "heuristic",
            "summary": summary,
            "alerts": alerts,
            "recommended_actions": recommendations or ["Continuer la surveillance standard."],
            "confidence": 0.58 if alerts else 0.7,
            "notes": notes,
        }

    @staticmethod
    def _clean_text(value: Any) -> str:
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned:
                return cleaned
        return ""

    @staticmethod
    def _clean_list(value: Any) -> List[str]:
        if not isinstance(value, (list, tuple)):
            return []
        results: List[str] = []
        for item in value:
            text = CognitiveArchitecture._clean_text(item)
            if text:
                results.append(text)
        return results

    def summarize_beliefs(self, timeframe: str = "daily") -> Dict[str, Any]:
        beliefs = getattr(self, "beliefs", None)
        if not beliefs:
            return {}
        try:
            summary = beliefs.latest_summary()
            if timeframe:
                return summary.get(timeframe, {})
            return summary
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Cycle
    def cycle(
        self,
        user_msg: Optional[str] = None,
        inbox_docs=None,
        user_id: str = "default",
    ) -> str:
        self.last_user_id = user_id or "default"

        if not user_msg:
            self._tick_background_systems()
            return self.last_output_text

        self._cycle_counter += 1
        if self._cycle_counter % self._decay_period == 0:
            self._apply_belief_decay()

        try:
            if self.self_improver and self.self_improver.try_promote_from_reply(user_msg):
                return "✅ Challenger promu. Les nouveaux paramètres sont actifs."
        except Exception:
            pass

        if isinstance(user_msg, str):
            normalized = user_msg.strip().lower()
            if normalized in {"améliore-toi", "self-improve", "optimize"}:
                cid = None
                try:
                    if self.self_improver:
                        cid = self.self_improver.run_cycle(n_candidates=4)
                except Exception:
                    cid = None
                if cid:
                    return (
                        "J’ai un challenger candidat ({cid}). Je te demande validation avant promotion.".format(
                            cid=cid
                        )
                    )
                return "Aucun challenger n’a surclassé le champion sur les métriques définies."

        trimmed = (user_msg or "").strip()
        trimmed_lower = trimmed.lower()
        explicit_mode_cmd = trimmed_lower.startswith("/mode")
        detected_mode = None
        try:
            detected_mode = self.style_policy.detect_mode_command(trimmed_lower)
        except Exception:
            detected_mode = None
        if detected_mode:
            phrase_command = trimmed_lower in {
                f"mode {detected_mode}",
                f"mode {detected_mode}.",
                f"mode {detected_mode}!",
                f"mode {detected_mode}?",
            }
            if phrase_command:
                explicit_mode_cmd = True
            self.style_policy.set_mode(detected_mode, persona_tone=self.style_policy.persona_tone)
            if explicit_mode_cmd and len(trimmed.split()) <= 2:
                ack = f"✅ Mode de communication réglé sur « {detected_mode} »."
                try:
                    if hasattr(self.memory, "add_memory"):
                        self.memory.add_memory(
                            kind="mode_switch",
                            content=detected_mode,
                            metadata={"source": "user"},
                        )
                except Exception:
                    pass
                self.last_output_text = ack
                return ack

        try:
            persona_tone = (self.user_model.describe().get("persona", {}) or {}).get("tone")
            if persona_tone:
                self.style_policy.update_persona_tone(persona_tone)
        except Exception:
            pass

        try:
            self.intent_model.observe_user_message(user_msg)
        except Exception:
            pass

        self._maybe_update_calibration(user_msg)

        self.telemetry.log("input", "language", {"text": user_msg})

        try:
            self.autonomy.notify_user_activity()
        except Exception:
            pass

        try:
            self.style_profiler.observe(self.last_user_id, user_msg)
        except Exception:
            pass

        try:
            context = {
                "last_assistant_output": self.last_output_text,
                "active_goal_id": getattr(self.goals, "active_goal_id", None),
            }
            try:
                if self.goals and hasattr(self.goals, "get_active_goal"):
                    context["active_goal"] = self.goals.get_active_goal()
            except Exception:
                context.setdefault("active_goal", None)
            try:
                if self.emotions and hasattr(self.emotions, "get_affect"):
                    context["emotional_state"] = self.emotions.get_affect()
            except Exception:
                pass
            kernel_state = None
            try:
                kernel_state = getattr(self, "phenomenal_kernel_state", None)
                if not kernel_state:
                    kernel_state = getattr(self, "_phenomenal_kernel_state", None)
            except Exception:
                kernel_state = None
            if isinstance(kernel_state, dict) and kernel_state:
                context["phenomenal_kernel"] = dict(kernel_state)
            self.reward_engine.ingest_user_message(
                self.last_user_id, user_msg, context=context, channel="chat"
            )
        except Exception:
            pass

        special_reply = self._handle_recent_memory_request(trimmed)
        if special_reply:
            self.last_output_text = special_reply
            return special_reply

        def _looks_like_abduction(s: str) -> bool:
            s = (s or "").lower()
            return any(
                k in s
                for k in ["devine", "pourquoi", "hypothèse", "à ton avis", "raison", "résous l'énigme"]
            )

        abduction_result: Optional[Dict[str, Any]] = None
        if _looks_like_abduction(user_msg):
            abduction_result = self._handle_abduction_request(user_msg)
        else:
            if self._looks_like_causal(trimmed_lower):
                causal_reply = self._handle_causal(user_msg)
                if causal_reply:
                    self.last_output_text = causal_reply
                    return causal_reply
            if self._looks_like_plan(trimmed_lower):
                plan_reply = self._handle_plan(user_msg)
                if plan_reply:
                    self.last_output_text = plan_reply
                    return plan_reply

        surface = user_msg
        hints = {}
        try:
            hints = self.style_policy.adapt_from_instruction(user_msg)
        except Exception:
            hints = {}
        try:
            parsed = self.language.parse_utterance(
                user_msg,
                context={
                    "style_hints": hints,
                    "style_mode": self.style_policy.current_mode,
                    "user_intents": self.intent_model.as_constraints(),
                },
            )
            surface = getattr(parsed, "surface_form", user_msg)
        except Exception:
            pass

        novelty_score = 0.0
        novelty_flag = False
        try:
            novelty_score, novelty_flag = self.novelty_detector.assess(surface, update=True)
            if novelty_flag and hasattr(self.memory, "add_memory"):
                self.memory.add_memory(
                    kind="novel_case",
                    content=surface[:160],
                    metadata={"score": float(novelty_score), "user_id": self.last_user_id},
                )
            try:
                self.logger.write(
                    "novelty.assessment",
                    score=float(novelty_score),
                    flagged=bool(novelty_flag),
                    text=surface[:120],
                )
            except Exception:
                pass
        except Exception:
            novelty_score = 0.0
            novelty_flag = False

        self._emit_structured_memories(surface)

        # Capture d'une confirmation utilisateur pour valider un apprentissage récent
        try:
            low = (user_msg or "").lower()
            if any(k in low for k in ["oui", "c'est correct", "exact"]) and hasattr(self, "memory"):
                recents = self.memory.get_recent_memories(50)
                concept = None
                for item in reversed(recents):
                    if item.get("kind") == "validation_request":
                        concept = (item.get("metadata") or {}).get("concept")
                        if concept:
                            break
                if concept:
                    self._record_skill(concept)
            else:
                refusal_markers = (
                    "pas correct",
                    "incorrect",
                    "c'est faux",
                    "ce n'est pas correct",
                    "je ne suis pas d'accord",
                    "pas vrai",
                    "mauvais",
                )
                refuse_short = low.strip() in {"non", "nope", "pas vraiment", "absolument pas"}
                refusal_hit = refuse_short or any(m in low for m in refusal_markers)
                if refusal_hit and hasattr(self, "memory"):
                    recents = self.memory.get_recent_memories(50)
                    concept = None
                    last_request = None
                    for item in reversed(recents):
                        if item.get("kind") == "validation_request":
                            last_request = item
                            concept = (item.get("metadata") or {}).get("concept")
                            if concept:
                                break
                    if concept:
                        try:
                            self.memory.add_memory(
                                {
                                    "kind": "validation_request:refusé",
                                    "content": f"Validation refusée pour {concept}",
                                    "metadata": {
                                        "concept": concept,
                                        "source": "user_feedback",
                                        "raw": user_msg[:160],
                                        "request_id": (last_request or {}).get("id"),
                                    },
                                    "ts": time.time(),
                                }
                            )
                        except Exception:
                            pass
                        try:
                            if not getattr(self, "concept_recognizer", None):
                                from AGI_Evolutive.knowledge.concept_recognizer import ConceptRecognizer

                                self.concept_recognizer = ConceptRecognizer(self)
                            evidence_payload = {}
                            features_payload = {}
                            mem = getattr(self, "memory", None)
                            if mem and hasattr(mem, "find_recent"):
                                ev = mem.find_recent(
                                    kind="concept_candidate",
                                    since_sec=3600 * 24,
                                    where={"label": concept},
                                ) or {}
                                if isinstance(ev, dict):
                                    evidence_payload = ev.get("evidence", {}) or {}
                                    features_payload = ev.get("features", {}) or {}
                                elif isinstance(ev, list) and ev:
                                    first = ev[0]
                                    if isinstance(first, dict):
                                        evidence_payload = first.get("evidence", {}) or {}
                                        features_payload = first.get("features", {}) or {}
                            if getattr(self, "concept_recognizer", None):
                                self.concept_recognizer.learn_from_rejection(
                                    kind="concept",
                                    label=concept,
                                    evidence=evidence_payload,
                                    penalty=0.6,
                                    features=features_payload,
                                )
                        except Exception:
                            pass
        except Exception:
            pass

        reason_out: Dict[str, Any] = {}
        if abduction_result:
            reason_out = dict(abduction_result.get("reason_out") or {})
        else:
            try:
                reasoning_context = {
                    "inbox_docs": inbox_docs,
                    "user_intents": self.intent_model.as_constraints(),
                    "style_mode": self.style_policy.current_mode,
                }
                reason_out = self.reasoning.reason_about(surface, context=reasoning_context)
            except Exception as exc:
                self.logger.write("reasoning.error", error=str(exc), user_msg=surface)
                reason_out = {
                    "summary": "Raisonnement basique uniquement (fallback).",
                    "chosen_hypothesis": "clarifier intention + proposer 1 test",
                    "tests": ["proposer 2 options et valider"],
                    "final_confidence": 0.5,
                    "appris": ["garder une trace même en cas d'erreur"],
                    "prochain_test": "valider l'option la plus utile",
                }

        ask_prompts: List[str] = []
        abstain = False
        calibration_domain = None
        raw_confidence = float(reason_out.get("final_confidence", 0.5))
        adjusted_confidence = max(0.0, min(1.0, raw_confidence))

        if not abduction_result:
            calibration_domain = "planning" if reason_out.get("tests") or reason_out.get("prochain_test") else "decision"
            if novelty_flag:
                adjusted_confidence *= max(0.4, 1.0 - 0.35 * novelty_score)
            adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))
            reason_out["final_confidence"] = adjusted_confidence
            try:
                abstain = self.calibration.should_abstain(calibration_domain, adjusted_confidence)
            except Exception:
                abstain = False
            if novelty_flag and adjusted_confidence < 0.75:
                abstain = True
            event_id = None
            try:
                meta = {
                    "domain": calibration_domain,
                    "novelty": float(novelty_score),
                    "abstain": bool(abstain),
                    "raw_confidence": float(raw_confidence),
                }
                event_id = self.calibration.log_prediction(
                    domain=calibration_domain,
                    p=adjusted_confidence,
                    meta=meta,
                )
                if hasattr(self.memory, "add_memory"):
                    self.memory.add_memory(
                        kind="calibration_observation",
                        content=f"{calibration_domain or 'decision'}_pred",
                        metadata={
                            "event_id": event_id,
                            "domain": calibration_domain,
                            "p": float(adjusted_confidence),
                        },
                    )
            except Exception:
                event_id = None
            if abstain:
                reason_out.setdefault("appris", []).append(
                    "Appliquer un refus calibré quand la confiance est insuffisante."
                )
                ask_prompts.append(
                    "Confiance trop faible → peux-tu préciser ton objectif ou les contraintes clés ?"
                )
                reason_out["summary"] = (
                    reason_out.get("summary", "")
                    + " | ⚠️ abstention calibrée (demander précisions)."
                )
                try:
                    if hasattr(self.memory, "add_memory"):
                        self.memory.add_memory(
                            kind="abstain",
                            content=surface[:160],
                            metadata={"domain": calibration_domain, "confidence": adjusted_confidence},
                        )
                except Exception:
                    pass
        try:
            if not abduction_result and hasattr(self, "question_manager") and self.question_manager:
                severity = max(0.0, 1.0 - adjusted_confidence)
                if severity > 0.45:
                    explicit_q = ask_prompts[0] if ask_prompts else None
                    self.question_manager.record_information_need(
                        "goal_focus",
                        severity,
                        metadata={"source": "confidence", "user_msg": surface[:160]},
                        explicit_question=explicit_q,
                    )
        except Exception:
            pass
        if novelty_flag:
            if abduction_result:
                adjusted_confidence = max(
                    0.0,
                    min(1.0, adjusted_confidence * max(0.4, 1.0 - 0.35 * novelty_score)),
                )
                reason_out["final_confidence"] = adjusted_confidence
            reason_out.setdefault("appris", []).append(
                "Détecter les cas atypiques et demander un éclairage supplémentaire."
            )
            if "Cas inhabituel détecté → partage un exemple ou le contexte exact." not in ask_prompts:
                ask_prompts.append(
                    "Cas inhabituel détecté → partage un exemple ou le contexte exact."
                )
            try:
                if hasattr(self, "question_manager") and self.question_manager:
                    self.question_manager.record_information_need(
                        "evidence",
                        max(0.4, min(1.0, float(novelty_score) or 0.5)),
                        metadata={"source": "novelty", "user_msg": surface[:160]},
                    )
            except Exception:
                pass

        friendly_fallback: Optional[str] = None
        if not abduction_result and (abstain or adjusted_confidence < 0.45):
            friendly_fallback = self._build_clarification_reply(ask_prompts)

        apprentissages = [
            "associer récompense sociale ↔ style",
            "tenir un journal d'épisodes de raisonnement",
        ] + list(reason_out.get("appris", []))

        next_test = reason_out.get("prochain_test") or "-"
        if abstain:
            next_test = "clarifier avec toi les contraintes et objectifs avant d'avancer"

        base_besoins = ["confirmer si tu veux patch immédiat ou plan en étapes"]
        besoins: List[str] = []
        for item in base_besoins + ask_prompts:
            if not item:
                continue
            if item not in besoins:
                besoins.append(item)

        contract = ensure_contract(
            {
                "hypothese_choisie": reason_out.get("chosen_hypothesis", "clarifier intention"),
                "incertitude": float(max(0.0, min(1.0, 1.0 - adjusted_confidence))),
                "prochain_test": next_test,
                "appris": apprentissages,
                "besoin": besoins,
            }
        )

        base_text = self._generate_base_text(surface, reason_out)
        nlg_state = self._language_state_snapshot()
        predicate_registry = self._predicate_registry_for_state(nlg_state)
        nlg_context = NLGContext(base_text, self._resolve_nlg_hint_applier())
        try:
            apply_mai_bids_to_nlg(nlg_context, nlg_state, predicate_registry)
        except Exception:
            pass
        applied_hints = nlg_context.applied_hints()
        if applied_hints:
            reason_out["applied_hints"] = applied_hints
        base_reply = nlg_context.text.strip() or ""
        if not base_reply:
            base_reply = reason_out.get("summary") or surface.strip()
        if not base_reply:
            base_reply = "Je n'ai pas encore de réponse claire, peux-tu préciser ta demande ?"

        normalized_text, reasoning_diag = humanize_reasoning_block(base_reply)
        user_reply = normalized_text or base_reply

        if friendly_fallback:
            user_reply = friendly_fallback

        diagnostics_text = None
        try:
            diagnostics_text = format_agent_reply(user_reply, **contract)
        except Exception:
            diagnostics_text = None

        if abduction_result:
            user_reply = abduction_result.get("response", user_reply)

        try:
            user_reply = self.style_profiler.rewrite_to_match(user_reply, self.last_user_id)
        except Exception:
            pass

        self.last_output_text = user_reply

        try:
            if hasattr(self.memory, "store_interaction"):
                payload = {
                    "ts": time.time(),
                    "user": user_msg,
                    "agent": user_reply,
                    "agent_raw": diagnostics_text or user_reply,
                    "lang_state": getattr(
                        getattr(self.language, "state", None), "to_dict", lambda: {}
                    )(),
                }
                if reasoning_diag:
                    payload["reasoning_trace"] = reasoning_diag
                self.memory.store_interaction(payload)
        except Exception:
            pass

        self.logger.write(
            "dialogue.turn",
            user_msg=user_msg,
            surface=surface,
            hypothesis=contract["hypothese_choisie"],
            incertitude=contract["incertitude"],
            test=contract["prochain_test"],
            reward=extract_social_reward(user_msg).get("reward", 0.0),
            style=self.style_policy.as_dict(),
            reason_summary=reason_out.get("summary", ""),
        )

        try:
            mem = self.memory.get_recent_memories(60)
            self.user_model.ingest_memories(mem)
        except Exception:
            pass

        output_payload: Dict[str, Any] = {
            "text": user_reply,
            "raw": base_reply,
            "contract": contract,
            "reasoning": reason_out,
        }
        if friendly_fallback:
            output_payload.setdefault("metadata", {})["clarification_reply"] = True
        if reasoning_diag:
            output_payload["reasoning_trace"] = reasoning_diag
        if applied_hints:
            output_payload.setdefault("metadata", {})["applied_hints"] = list(applied_hints)
        needs_list = [str(item).strip() for item in contract.get("besoin", []) if str(item).strip()]
        if needs_list:
            output_payload["bullets"] = needs_list
        if diagnostics_text and diagnostics_text.strip() and diagnostics_text.strip() != user_reply.strip():
            output_payload["diagnostics_text"] = diagnostics_text

        self._tick_background_systems()
        return output_payload

    def _handle_abduction_request(self, user_msg: str) -> Dict[str, Any]:
        reason_out: Dict[str, Any] = {
            "summary": "Aucune hypothèse abductive trouvée.",
            "chosen_hypothesis": "aucune",
            "final_confidence": 0.0,
            "prochain_test": None,
            "appris": ["reconnaître une demande abductive"],
        }
        try:
            hyps = self.abduction.generate(user_msg)
        except Exception:
            hyps = []
        if not hyps:
            return {
                "response": "Je manque d'indices pour formuler une hypothèse utile.",
                "reason_out": reason_out,
            }

        top = hyps[0]
        score = float(getattr(top, "score", 0.0))
        reason_out.update(
            {
                "summary": top.explanation or "Hypothèse générée via abduction.",
                "chosen_hypothesis": top.label,
                "final_confidence": score,
                "prochain_test": getattr(top, "ask_next", None),
            }
        )
        appris = list(reason_out.get("appris") or [])
        if getattr(top, "priors", None):
            appris.append("mobiliser les priors abductifs")
        reason_out["appris"] = appris
        try:
            if hasattr(self.memory, "add_memory"):
                self.memory.add_memory(
                    kind="hypothesis",
                    content=top.label,
                    metadata={
                        "score": top.score,
                        "explanation": top.explanation,
                        "priors": top.priors,
                    },
                )
        except Exception:
            pass

        ev_id = None
        try:
            ev_id = self.calibration_abduction.log_prediction(
                domain="abduction", p=float(top.score), meta={"label": top.label}
            )
            if hasattr(self.memory, "add_memory"):
                self.memory.add_memory(
                    kind="calibration_observation",
                    content="abduction_pred",
                    metadata={
                        "event_id": ev_id,
                        "label": top.label,
                        "p": float(top.score),
                        "domain": "abduction",
                    },
                )
        except Exception:
            ev_id = None

        if top.ask_next:
            handled = False
            try:
                qm = getattr(self, "question_manager", None)
                if qm and hasattr(qm, "add_question"):
                    qm.add_question(top.ask_next)
                    handled = True
            except Exception:
                handled = False
            if not handled:
                try:
                    if hasattr(self.memory, "add_memory"):
                        self.memory.add_memory(
                            kind="question_active",
                            content=top.ask_next,
                            metadata={"source": "abduction"},
                        )
                except Exception:
                    pass
            reason_out["summary"] = top.explanation or "Question de clarification abductive."
            return {"response": f"{top.ask_next}", "reason_out": reason_out}

        conf = int(round(score * 100))
        response = (
            f"Mon hypothèse la plus probable : **{top.label}** ({conf}% confiance). "
            "Je peux réviser si tu me donnes un indice contraire."
        )
        return {"response": response, "reason_out": reason_out}

    def _maybe_update_calibration(self, user_msg: Optional[str]) -> None:
        t = (user_msg or "").strip().lower()
        if t not in {"oui", "non", "exact", "c'est correct"}:
            return
        try:
            recents = []
            if hasattr(self.memory, "get_recent_memories"):
                recents = self.memory.get_recent_memories(50)
            ev_id = None
            event_meta: Dict[str, Any] = {}
            for item in reversed(recents):
                if item.get("kind") == "calibration_observation":
                    event_meta = item.get("metadata") or {}
                    ev_id = event_meta.get("event_id")
                    if ev_id:
                        break
            if ev_id:
                success = t in {"oui", "exact", "c'est correct"}
                domain = event_meta.get("domain", "abduction")
                self.calibration.log_outcome(ev_id, success=success)
                delta = self.calibration.suggested_hedging_delta(domain=domain)
                if hasattr(self.style_policy, "params"):
                    hedging = self.style_policy.params.get("hedging", 0.3)
                    self.style_policy.params["hedging"] = max(0.0, min(1.0, hedging + delta))
                if hasattr(self.memory, "add_memory"):
                    self.memory.add_memory(
                        kind="calibration_feedback",
                        content=t,
                        metadata={"event_id": ev_id, "delta_hedging": delta, "domain": domain},
                    )
        except Exception:
            pass

    def _build_clarification_reply(self, prompts: List[str]) -> str:
        questions: List[str] = []
        for prompt in prompts[:2]:
            cleaned = self._normalize_clarification_prompt(prompt)
            if cleaned:
                questions.append(cleaned)
        if not questions:
            fallback = self._normalize_clarification_prompt(None)
            if fallback:
                questions.append(fallback)
        question_text = " ".join(questions) if questions else "Peux-tu préciser ta demande ?"
        base = "Je ne suis pas certain d'avoir bien compris."
        return f"{base} {question_text}".strip()

    def _normalize_clarification_prompt(self, prompt: Optional[str]) -> Optional[str]:
        text = (prompt or "").strip()
        if "→" in text:
            text = text.split("→", 1)[1].strip()
        elif ":" in text:
            head, tail = text.split(":", 1)
            tail = tail.strip()
            text = tail or head.strip()
        text = re.sub(r"^[•\-\s]+", "", text)
        text = text.strip()
        if not text:
            text = "Peux-tu préciser ta demande ?"
        if not text:
            return None
        if text.endswith("?"):
            return text
        text = text.rstrip(".")
        candidate = text.rstrip()
        if not candidate:
            return None
        return f"{candidate} ?"

    def _emit_structured_memories(self, text: str) -> None:
        if not text or not hasattr(self.memory, "add_memory"):
            return
        try:
            entities = re.findall(r"\b[A-ZÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ][\w'’\-]{2,}\b", text)
            seen = set()
            for ent in entities[:6]:
                key = ent.lower()
                if key in seen:
                    continue
                seen.add(key)
                linked = self.entity_linker.link(ent)
                self.memory.add_memory(
                    kind="entity_detected",
                    content=ent,
                    metadata={"type": linked["type"], "canonical": linked["canonical"]},
                )
        except Exception:
            pass

        try:
            for match in re.finditer(
                r"([A-Za-zÀ-ÖØ-öø-ÿ'’\-]{2,})\s+(est|sont|sera|éta(?:is|it|ient))\s+([^\.;]+)",
                text,
            ):
                subject = match.group(1).strip()
                relation = match.group(2).strip().lower()
                value = match.group(3).strip()
                if not subject or not value:
                    continue
                subj_link = self.entity_linker.link(subject)
                val_link = self.entity_linker.link(value)
                self.memory.add_memory(
                    kind="fact_extracted",
                    content=f"{subject} {relation} {value}",
                    metadata={
                        "subject": subj_link["canonical"],
                        "relation": relation,
                        "value": val_link["canonical"],
                        "polarity": +1,
                    },
                )
        except Exception:
            pass

    def _normalize_memory_request_text(self, text: str) -> str:
        base = unicodedata.normalize("NFKD", text or "")
        stripped = "".join(ch for ch in base if not unicodedata.combining(ch))
        lowered = stripped.lower()
        lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
        return re.sub(r"\s+", " ", lowered).strip()

    def _is_recent_memory_request(self, normalized: str) -> bool:
        if not normalized:
            return False
        memory_cues = ("memoire", "memoires", "souvenir", "souvenirs")
        if not any(cue in normalized for cue in memory_cues):
            return False
        recency_cues = ("recent", "recents", "dern", "aujourd", "derni", "dernier", "derniers")
        if not any(cue in normalized for cue in recency_cues):
            return False
        action_cues = (
            "dis",
            "donne",
            "montre",
            "partage",
            "affiche",
            "liste",
            "raconte",
            "envoie",
        )
        return any(cue in normalized for cue in action_cues)

    def _resolve_memory_requester_identity(self, normalized_text: str) -> Dict[str, Any]:
        configuration = cfg()
        configured_names = configuration.get("MEMORY_SHARING_TRUSTED_NAMES") or []
        if not isinstance(configured_names, list):
            configured_names = [str(configured_names)]
        normalized_map: Dict[str, str] = {}
        for raw_name in configured_names:
            if not raw_name:
                continue
            normalized_map[self._normalize_memory_request_text(str(raw_name))] = str(raw_name)

        default_name = str(configuration.get("PRIMARY_USER_NAME", configured_names[0] if configured_names else "utilisateur"))
        expected_role = str(configuration.get("PRIMARY_USER_ROLE", "creator"))
        roles_map = configuration.get("MEMORY_SHARING_ROLES_BY_NAME") or {}
        if not isinstance(roles_map, dict):
            roles_map = {}

        matched_norm = None
        for alias_norm in normalized_map:
            if alias_norm and alias_norm in normalized_text:
                matched_norm = alias_norm
                break

        match_source = "unknown"
        resolved_name: Optional[str] = None
        if matched_norm:
            resolved_name = normalized_map.get(matched_norm)
            match_source = "alias"
        elif self.last_user_id == "default" and default_name:
            resolved_name = default_name
            match_source = "default_user_id"
        elif self.last_user_id:
            resolved_name = str(self.last_user_id)
            match_source = "user_id"

        if resolved_name in roles_map:
            resolved_role = str(roles_map[resolved_name])
        elif resolved_name == default_name:
            resolved_role = expected_role
        elif resolved_name and self._normalize_memory_request_text(resolved_name) in normalized_map:
            resolved_role = expected_role
        else:
            resolved_role = "unknown"

        confidence = 0.55
        if match_source == "alias":
            confidence = 0.9
        elif match_source == "default_user_id":
            confidence = 0.8
        elif match_source == "user_id":
            confidence = 0.65

        trusted_roles = configuration.get("MEMORY_SHARING_TRUSTED_ROLES") or []
        if not isinstance(trusted_roles, list):
            trusted_roles = [trusted_roles]
        trusted_roles_norm = [str(role).lower() for role in trusted_roles]
        name_norm = self._normalize_memory_request_text(resolved_name) if resolved_name else ""
        trusted_name_norms = set(normalized_map.keys())
        role_norm = resolved_role.lower() if isinstance(resolved_role, str) else ""
        trusted = bool(resolved_name and name_norm in trusted_name_norms and (not trusted_roles_norm or role_norm in trusted_roles_norm))

        return {
            "user_id": self.last_user_id or "default",
            "name": resolved_name,
            "role": resolved_role,
            "confidence": float(confidence),
            "matched_via": match_source,
            "trusted": trusted,
        }

    def _summarize_memory_entry(self, entry: Dict[str, Any]) -> Optional[str]:
        if not isinstance(entry, dict):
            return None
        kind = entry.get("kind") or entry.get("type") or entry.get("label") or "mémoire"
        kind_label = str(kind).replace("_", " ").replace(":", " ").strip().capitalize() or "Mémoire"
        text = entry.get("text") or entry.get("content") or ""
        if not isinstance(text, str):
            text = str(text)
        text = text.strip()
        metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
        if not text and metadata:
            for key in ("summary", "description", "note"):
                value = metadata.get(key)
                if isinstance(value, str) and value.strip():
                    text = value.strip()
                    break
        if not text and metadata:
            for value in metadata.values():
                if isinstance(value, str) and value.strip():
                    text = value.strip()
                    break
        text = text.replace("\n", " ").strip()
        if len(text) > 140:
            text = text[:137].rstrip() + "…"
        timestamp = entry.get("ts")
        if not isinstance(timestamp, (int, float)) and isinstance(metadata, dict):
            timestamp = metadata.get("timestamp")
        date_hint = ""
        if isinstance(timestamp, (int, float)):
            try:
                dt_obj = dt.datetime.fromtimestamp(timestamp)
                date_hint = dt_obj.strftime("%d/%m %H:%M")
            except Exception:
                date_hint = ""
        summary = f"{kind_label}: {text}" if text else kind_label
        if date_hint:
            summary = f"{date_hint} · {summary}"
        source = metadata.get("source") if isinstance(metadata, dict) else None
        if isinstance(source, str) and source.strip():
            summary = f"{summary} (source: {source.strip()})"
        return summary

    def _collect_recent_memory_summaries(self, limit: int) -> List[str]:
        memory = getattr(self, "memory", None)
        if not memory or not hasattr(memory, "get_recent_memories"):
            return []
        try:
            raw = memory.get_recent_memories(n=max(limit * 6, limit))
        except Exception:
            return []
        summaries: List[str] = []
        seen_ids: set[str] = set()
        seen_texts: set[str] = set()
        for entry in reversed(raw):
            if not isinstance(entry, dict):
                continue
            entry_id = entry.get("id")
            if entry_id and entry_id in seen_ids:
                continue
            if entry_id:
                seen_ids.add(entry_id)
            if str(entry.get("kind", "")) == "memory_share_event":
                continue
            summary = self._summarize_memory_entry(entry)
            if not summary or summary in seen_texts:
                continue
            seen_texts.add(summary)
            summaries.append(summary)
            if len(summaries) >= limit:
                break
        return summaries

    def _infer_memory_request_hypotheses(self, normalized: str, identity: Dict[str, Any]) -> List[str]:
        hypotheses: List[str] = []
        if "comprend" in normalized or "analyse" in normalized:
            hypotheses.append("Tu vérifies ma compréhension actuelle du monde via mes traces récentes.")
        if "objectif" in normalized or "but" in normalized:
            hypotheses.append("Tu veux relier ces souvenirs à un objectif explicite.")
        if identity.get("role") == "creator":
            hypotheses.append("En tant que créateur, tu testes ma transparence et ma fiabilité.")
        if "contexte" in normalized or "monde" in normalized:
            hypotheses.append("Tu souhaites synchroniser nos contextes cognitifs.")
        if not hypotheses:
            hypotheses.append("Tu veux simplement savoir ce que j'ai vécu dernièrement pour mieux me guider.")
        return hypotheses

    def _ensure_memory_request_goal(self) -> Optional[str]:
        goals = getattr(self, "goals", None)
        if goals is None:
            return None
        store = getattr(goals, "store", None)
        if store is None:
            return None
        try:
            if self._memory_request_goal_id:
                existing = store.get_goal(self._memory_request_goal_id)
                if existing is not None:
                    return f"Objectif déjà suivi ({self._memory_request_goal_id})."
                self._memory_request_goal_id = None
            node = goals.add_goal(
                "Comprendre la motivation derrière les demandes de mémoires récentes",
                goal_type=GoalType.SOCIAL,
                criteria=[
                    "Formuler une hypothèse sur la motivation de William",
                    "Valider l'hypothèse via un échange ou une observation.",
                ],
                value=0.62,
                curiosity=0.58,
                urgency=0.32,
                created_by="memory_share_handler",
            )
            self._memory_request_goal_id = node.id
            return f"Nouvel objectif créé ({node.id}) pour approfondir ta motivation."
        except Exception:
            return None

    def _handle_recent_memory_request(self, user_msg: str) -> Optional[str]:
        normalized = self._normalize_memory_request_text(user_msg)
        if not self._is_recent_memory_request(normalized):
            return None

        identity = self._resolve_memory_requester_identity(normalized)
        share_allowed = bool(identity.get("trusted"))
        config = cfg()
        max_items = int(config.get("MEMORY_SHARING_MAX_ITEMS", 5))
        shared_memories = self._collect_recent_memory_summaries(max_items) if share_allowed else []

        request_analysis = "Je comprends que tu me demandes explicitement de partager mes mémoires récentes."
        if identity.get("name"):
            user_analysis = (
                f"Je t'identifie comme {identity['name']} (rôle : {identity.get('role', 'inconnu')})"
                f" avec une confiance d'environ {int(identity.get('confidence', 0.0) * 100)} %."
            )
        else:
            user_analysis = (
                "Je n'ai pas de nom explicite pour toi, mais je relie la demande à l'utilisateur courant"
                f" (id : {identity.get('user_id')}) avec une confiance d'environ"
                f" {int(identity.get('confidence', 0.0) * 100)} %."
            )

        if share_allowed and shared_memories:
            decision = (
                f"Je décide de partager {len(shared_memories)} souvenir(s) car tu es reconnu comme {identity.get('role', 'partenaire')}"
                " et je peux te faire confiance."
            )
        elif share_allowed:
            decision = (
                "Je suis prêt à partager, mais aucune mémoire exploitable n'est disponible dans ma fenêtre récente."
            )
        else:
            decision = (
                "Je préfère ne rien divulguer tant que je n'ai pas confirmé ton identité comme autorisée."
            )

        hypotheses = self._infer_memory_request_hypotheses(normalized, identity)
        goal_note = self._ensure_memory_request_goal()

        try:
            metadata = {
                "requested_by": identity.get("name") or identity.get("user_id"),
                "trusted": share_allowed,
                "matched_via": identity.get("matched_via"),
                "shared_count": len(shared_memories),
            }
            if hasattr(self.memory, "add_memory"):
                self.memory.add_memory(
                    {
                        "kind": "memory_share_event",
                        "content": "Partage de mémoires récentes",
                        "metadata": metadata,
                    }
                )
        except Exception:
            pass

        try:
            self.telemetry.log(
                "memory",
                "share_recent_memories",
                {
                    "trusted": share_allowed,
                    "shared_count": len(shared_memories),
                    "requester": identity.get("name") or identity.get("user_id"),
                    "matched_via": identity.get("matched_via"),
                },
            )
        except Exception:
            pass

        response_lines = [
            "Voici comment j'ai traité ta demande :",
            "",
            "**1. Compréhension de ta requête**",
            f"- {request_analysis}",
            "",
            "**2. Analyse de ton identité**",
            f"- {user_analysis}",
            "",
            "**3. Décision**",
            f"- {decision}",
        ]

        response_lines.append("")
        if share_allowed and shared_memories:
            response_lines.append("**4. Mémoires récentes partagées**")
            for item in shared_memories:
                response_lines.append(f"- {item}")
        elif share_allowed:
            response_lines.append("**4. Mémoires récentes partagées**")
            response_lines.append("- Aucune mémoire pertinente n'a été retrouvée dans l'instant.")
        else:
            response_lines.append("**4. Mémoires récentes**")
            response_lines.append("- Accès suspendu en attendant une identification confirmée.")

        response_lines.append("")
        response_lines.append("**5. Hypothèses sur ta motivation**")
        for hyp in hypotheses:
            response_lines.append(f"- {hyp}")

        if goal_note:
            response_lines.append("")
            response_lines.append("**6. Prochaine étape interne**")
            response_lines.append(f"- {goal_note}")

        return "\n".join(response_lines)

    def _looks_like_causal(self, text: str) -> bool:
        if not text:
            return False
        cues = [
            "pourquoi",
            "cause",
            "causal",
            "que se passerait-il",
            "que se passerait il",
        ]
        if any(cue in text for cue in cues):
            return True
        return " si " in text and "alors" in text

    def _looks_like_plan(self, text: str) -> bool:
        if not text:
            return False
        cues = ["planifie", "planifier", "plan", "comment atteindre", "objectif"]
        return any(cue in text for cue in cues)

    def _parse_cause_effect(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        match = re.search(r"si\s+(.+?)\s+alors\s+(.+)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        return None, None

    def _handle_causal(self, user_msg: str) -> Optional[str]:
        if not getattr(self, "simulator", None):
            return None
        cause, effect = self._parse_cause_effect(user_msg)
        query = {
            "cause": cause,
            "effect": effect,
            "scenario": {"utterance": user_msg},
        }
        try:
            report = self.simulator.run(query)
        except Exception as exc:
            return f"Je n'ai pas pu exécuter la simulation causale ({exc})."

        try:
            if hasattr(self.memory, "add_memory"):
                self.memory.add_memory(
                    kind="causal_query",
                    content=user_msg[:160],
                    metadata={"cause": cause, "effect": effect},
                )
        except Exception:
            pass

        probability = 0.7 if report.supported else 0.35
        event_id = None
        try:
            event_id = self.calibration_causal.log_prediction(
                "causal", probability, meta={"cause": cause, "effect": effect}
            )
            if hasattr(self.memory, "add_memory"):
                self.memory.add_memory(
                    kind="calibration_observation",
                    content="causal_pred",
                    metadata={"event_id": event_id, "p": probability, "domain": "causal"},
                )
        except Exception:
            event_id = None

        try:
            if hasattr(self.memory, "add_memory"):
                self.memory.add_memory(
                    kind="counterfactual_result",
                    content=f"{cause or 'cause?'} → {effect or 'effet?'}",
                    metadata={
                        "supported": report.supported,
                        "evidence": report.evidence,
                        "intervention": report.intervention,
                    },
                )
        except Exception:
            pass

        evidence_text = (
            "J'observe un lien causal existant dans ma base." if report.supported else "Je ne possède pas de lien causal établi pour cette relation."
        )
        sim_texts = [sim.get("outcome", "") for sim in report.simulations if sim]
        if sim_texts:
            evidence_text += " " + sim_texts[0][:160]
        return evidence_text

    def _handle_plan(self, user_msg: str) -> Optional[str]:
        if not getattr(self, "planner", None):
            return None
        goal = user_msg
        match = re.search(r"(?:planifie|planifier|plan pour|comment atteindre)\s+(.+)", user_msg, re.IGNORECASE)
        if match:
            goal = match.group(1).strip()
        steps = self.planner.plan("diagnostic_general", context={"goal": goal}) or [
            f"Clarifier le résultat attendu pour « {goal} ».",
            "Identifier ressources et contraintes majeures.",
            "Découper en trois actions concrètes et dater la première.",
        ]

        probability = min(0.95, 0.55 + 0.05 * len(steps))
        event_id = None
        try:
            event_id = self.calibration_plan.log_prediction(
                "plan", probability, meta={"goal": goal}
            )
            if hasattr(self.memory, "add_memory"):
                self.memory.add_memory(
                    kind="calibration_observation",
                    content="plan_pred",
                    metadata={"event_id": event_id, "p": probability, "domain": "plan"},
                )
        except Exception:
            event_id = None

        try:
            if hasattr(self.memory, "add_memory"):
                self.memory.add_memory(
                    kind="plan_created",
                    content=goal[:160],
                    metadata={"steps": steps},
                )
        except Exception:
            pass

        numbered = "\n".join(f"{idx+1}. {step}" for idx, step in enumerate(steps))
        return f"Plan proposé pour « {goal} » :\n{numbered}"

    def _apply_belief_decay(self) -> None:
        try:
            now = time.time()
            decayed = 0
            for belief in self.beliefs.iter_beliefs():
                age = now - belief.updated_at
                if age < 180 or belief.confidence <= 0.2:
                    continue
                belief.confidence = max(0.0, belief.confidence - 0.02)
                belief.updated_at = now
                decayed += 1
            if decayed:
                self.beliefs.flush()
                if hasattr(self.memory, "add_memory"):
                    self.memory.add_memory(
                        kind="belief_decay",
                        content=f"{decayed} croyances ajustées",
                        metadata={"timestamp": now},
                    )
        except Exception:
            pass

    def _tick_background_systems(self) -> None:
        try:
            if getattr(self, "jobs", None):
                self.jobs.drain_to_memory(self.memory)
        except Exception:
            pass

        try:
            if self.perception_interface:
                self.perception_interface.step()
        except Exception:
            pass

        try:
            if self.emotions and hasattr(self.emotions, "step"):
                self.emotions.step()
        except Exception:
            pass

        try:
            if self.action_interface:
                self.action_interface.step()
        except Exception:
            pass

        try:
            self.concept_extractor.step(self.memory)
        except Exception:
            pass

        try:
            self.episodic_linker.step(self.memory)
        except Exception:
            pass

        try:
            self.autonomy.tick()
        except Exception:
            pass

        try:
            if time.time() - self._last_intent_decay > 600:
                self.intent_model.decay()
                self._last_intent_decay = time.time()
        except Exception:
            pass

        try:
            if self.memory and hasattr(self.memory, "add_memory"):
                self.memory.add_memory(
                    "dialog_turn",
                    {
                        "t": time.time(),
                        "user_id": self.last_user_id,
                        "assistant_msg": self.last_output_text,
                    },
                )
        except Exception:
            pass

    def _record_skill(
        self,
        concept: str,
        *,
        source: str = "learn_concept",
        confidence: float | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """
        Enregistre un 'skill/concept' appris et consolide la connaissance :
          - Persistance JSON (data/skills.json)
          - Mémoire milestone (tags: lesson, concept:<name>)
          - Ontology/Beliefs: crée le nœud concept:*, evidence 'defined_by'
          - Consolidator: ajoute une 'lesson' (résumé court)
          - Émet un événement 'virtue_learned' (si concept fait sens pour la persona)
        Robuste: aucun crash si un composant (ontology/beliefs/consolidator/voice) est absent.
        """
        import os, json, time, traceback

        t0 = time.time()
        out = {"ok": True, "concept": concept, "path": None, "errors": []}
        concept_norm = (concept or "").strip()
        if not concept_norm:
            return {"ok": False, "concept": concept, "errors": ["empty_concept"]}

        # ---------------------------------------
        # 1) Persistance JSON (skills.json)
        # ---------------------------------------
        skills_path = getattr(self, "skills_path", os.path.join("data", "skills.json"))
        os.makedirs(os.path.dirname(skills_path) or ".", exist_ok=True)

        try:
            skills = {}
            if os.path.exists(skills_path):
                with open(skills_path, "r", encoding="utf-8") as f:
                    try:
                        skills = json.load(f) or {}
                    except Exception:
                        skills = {}

            entry = skills.get(concept_norm) or {}
            entry.update({
                "name": concept_norm,
                "acquired": True,
                "last_update": t0,
                "source": source,
                "confidence": float(confidence) if confidence is not None else entry.get("confidence", 1.0),
                "meta": {**(entry.get("meta", {}) or {}), **(metadata or {})},
            })
            skills[concept_norm] = entry

            with open(skills_path, "w", encoding="utf-8") as f:
                json.dump(skills, f, ensure_ascii=False, indent=2)
            out["path"] = skills_path
        except Exception as e:
            out["ok"] = False
            out["errors"].append(f"skills_persist:{e}")
            # on continue malgré tout

        # ---------------------------------------
        # 2) Mémoire milestone (trace datée)
        # ---------------------------------------
        try:
            if hasattr(self, "memory") and getattr(self, "memory"):
                self.memory.add_memory({
                    "kind": "milestone",
                    "text": f"Compréhension de {concept_norm} validée",
                    "ts": t0,
                    "tags": ["lesson", f"concept:{concept_norm}"],
                    "metadata": {
                        "concept": concept_norm,
                        "source": source,
                        "confidence": confidence,
                        **(metadata or {})
                    }
                })
        except Exception as e:
            out["errors"].append(f"milestone:{e}")

        # ---------------------------------------
        # 3) Ontology + Beliefs (evidence)
        # ---------------------------------------
        try:
            # Ontology
            if hasattr(self, "ontology") and getattr(self, "ontology"):
                # API tolérante: add_entity(id, attrs=...) ou add_entity(id, **attrs)
                try:
                    self.ontology.add_entity(f"concept:{concept_norm}", attrs={"kind": "concept", "label": concept_norm, "source": source})
                except TypeError:
                    self.ontology.add_entity(f"concept:{concept_norm}", kind="concept", label=concept_norm, source=source)
            # Beliefs
            if hasattr(self, "beliefs") and getattr(self, "beliefs"):
                ev = None
                try:
                    from AGI_Evolutive.beliefs.graph import Evidence
                    # Evidence.new(type, via, info, weight)
                    ev = Evidence.new("action", source, f"appris:{concept_norm}", weight=0.8)
                except Exception:
                    ev = None
                # tolérance d’API: add_fact(...) ou add_evidence(...)
                if hasattr(self.beliefs, "add_fact"):
                    self.beliefs.add_fact(subject=f"concept:{concept_norm}",
                                          predicate="defined_by",
                                          obj=source,
                                          evidence=ev)
                elif hasattr(self.beliefs, "add_evidence"):
                    self.beliefs.add_evidence(subject=f"concept:{concept_norm}",
                                              predicate="defined_by",
                                              obj=source,
                                              weight=0.8)
                if hasattr(self.beliefs, "flush"):
                    self.beliefs.flush()
        except Exception as e:
            out["errors"].append(f"ontology_beliefs:{e}")

        # ---------------------------------------
        # 4) Consolidator: lesson/synthèse
        # ---------------------------------------
        try:
            if hasattr(self, "consolidator") and getattr(self, "consolidator"):
                summary = f"Concept : {concept_norm} — défini et validé (source={source}). " \
                          f"Confiance={confidence if confidence is not None else 'n/a'}."
                st = getattr(self.consolidator, "state", None)
                if st is not None:
                    lessons = st.setdefault("lessons", [])
                    lessons.append({
                        "topic": concept_norm,
                        "summary": summary,
                        "sources": [f"goal:{source}"] if source else [],
                        "ts": t0,
                        "tags": ["lesson", f"concept:{concept_norm}"]
                    })
                    # _save() si dispo
                    if hasattr(self.consolidator, "_save"):
                        self.consolidator._save()
        except Exception as e:
            out["errors"].append(f"consolidator:{e}")

        # ---------------------------------------
        # 5) Événement pour la persona/voix (optionnel, non bloquant)
        #    -> permettra au Proposer de faire évoluer persona.values/tone
        # ---------------------------------------
        try:
            if hasattr(self, "memory") and getattr(self, "memory"):
                # Si c’est une “vertu”/valeur probable, on émet un hook
                virtues = {"empathy", "compassion", "kindness", "honesty", "precision"}
                if concept_norm.lower() in virtues:
                    self.memory.add_memory({
                        "kind": "virtue_learned",
                        "value": concept_norm.lower(),
                        "ts": t0,
                        "tags": ["persona_hook", f"concept:{concept_norm}"]
                    })
            # petit coup de pouce voix (si présent) – non bloquant
            if hasattr(self, "voice_profile") and getattr(self, "voice_profile"):
                try:
                    if concept_norm.lower() in {"empathy", "compassion", "kindness"}:
                        self.voice_profile.bump("warmth", +0.03)
                        self.voice_profile.bump("emoji", +0.02)
                    elif concept_norm.lower() in {"precision", "rigor", "rigueur"}:
                        self.voice_profile.bump("conciseness", +0.02)
                        self.voice_profile.bump("analytical", +0.02)
                except Exception:
                    pass
        except Exception as e:
            out["errors"].append(f"voice_event:{e}")

        try:
            if not getattr(self, "concept_recognizer", None):
                from AGI_Evolutive.knowledge.concept_recognizer import ConceptRecognizer
                self.concept_recognizer = ConceptRecognizer(self)
            mem = getattr(self, "memory", None)
            ev = {}
            if mem and hasattr(mem, "find_recent"):
                ev = mem.find_recent(kind="concept_candidate", since_sec=3600 * 24, where={"label": concept_norm}) or {}
            evidence_payload = {}
            features_payload = {}
            if isinstance(ev, dict):
                evidence_payload = ev.get("evidence", {}) or {}
                features_payload = ev.get("features", {}) or {}
            elif isinstance(ev, list) and ev:
                first = ev[0]
                if isinstance(first, dict):
                    evidence_payload = first.get("evidence", {}) or {}
                    features_payload = first.get("features", {}) or {}
            if getattr(self, "concept_recognizer", None):
                self.concept_recognizer.learn_from_confirmation(
                    kind="concept",
                    label=concept_norm,
                    evidence=evidence_payload,
                    reward=0.85,
                    features=features_payload,
                )
        except Exception:
            pass

        out["duration_s"] = round(time.time() - t0, 3)
        # Si on a rencontré des erreurs non critiques, on reste ok=True mais on les remonte
        return out

    # ------------------------------------------------------------------
    def _generate_base_text(self, surface: str, reason_out: Dict[str, Any]) -> str:
        status = self.get_cognitive_status()
        status_line = (
            f"⏱️{status['uptime_s']}s | 🔋act={status['global_activation']:.2f} | "
            f"🧠wm={status['working_memory_load']:.2f}"
        )
        focus = status["goal_focus"]
        focus_line = (
            f"🎯focus:{focus['id']} (EVI={focus['evi']:.2f}, prog={focus['progress']:.2f})"
            if isinstance(focus, dict)
            else "🎯focus: n/a"
        )
        summary = reason_out.get("summary", "")
        return f"Reçu: {surface}\n{status_line}\n{focus_line}\n🧠 {summary}"
