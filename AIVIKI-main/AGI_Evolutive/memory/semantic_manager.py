import random
import time
import logging
from typing import Any, Dict, List, Optional, Tuple

from .concept_store import ConceptStore, Concept, Relation
from .concept_extractor import ConceptExtractor
from .episodic_linker import EpisodicLinker
from .vector_store import VectorStore

from AGI_Evolutive.utils.llm_service import try_call_llm_dict

LOGGER = logging.getLogger(__name__)


class _AdaptiveSignal:
    """Lightweight Thompson-sampling controller over EMA smoothing."""

    def __init__(
        self,
        label: str,
        betas: Tuple[float, ...] = (0.2, 0.4, 0.6, 0.8),
        *,
        decay: float = 0.97,
    ) -> None:
        self.label = label
        self.betas = betas
        self.decay = decay
        self.posterior: Dict[float, Tuple[float, float]] = {
            beta: (1.2, 1.2) for beta in betas
        }
        self.value = 0.0
        self.beta = betas[0] if betas else 0.5

    def update(self, signal: float, target: float) -> Dict[str, float]:
        signal = max(0.0, min(1.0, float(signal)))
        target = max(0.0, min(1.0, float(target)))

        best_beta = self.beta
        best_sample = float("-inf")
        for beta in self.betas:
            alpha, beta_param = self.posterior.get(beta, (1.2, 1.2))
            try:
                sample = random.betavariate(alpha, beta_param)
            except ValueError:
                sample = 0.0
            if sample > best_sample:
                best_sample = sample
                best_beta = beta

        previous = self.value
        self.beta = best_beta
        self.value = (1.0 - best_beta) * self.value + best_beta * signal
        drift = abs(self.value - previous)

        reward = 1.0 - min(1.0, abs(signal - target))
        alpha, beta_param = self.posterior.get(best_beta, (1.2, 1.2))
        alpha = alpha * self.decay + reward
        beta_param = beta_param * self.decay + (1.0 - reward)
        self.posterior[best_beta] = (max(1e-3, alpha), max(1e-3, beta_param))

        return {
            "value": self.value,
            "beta": best_beta,
            "drift": drift,
            "reward": reward,
            "signal": signal,
            "target": target,
        }


class SemanticMemoryManager:
    """Coordonne l'extraction de concepts et le chaînage épisodique."""

    def __init__(
        self,
        memory_system,
        architecture=None,
        *,
        index_backend: Optional[VectorStore] = None,
    ):
        self.memory = memory_system
        self.architecture = architecture
        self.store = ConceptStore()
        self.extractor = ConceptExtractor(self.memory)
        self.extractor.store = self.store
        self.linker = EpisodicLinker(self.memory)
        self.index_backend = index_backend or VectorStore()
        self._doc_metadata: Dict[str, Dict[str, Any]] = {}
        self.last_concept_step = 0.0
        self.last_link_step = 0.0
        self.concept_period = 5.0
        self.episodic_period = 5.0
        self._concept_signal = _AdaptiveSignal("concept")
        self._episodic_signal = _AdaptiveSignal("episodic")
        self._concept_backlog_window = 400
        self._episodic_backlog_window = 300
        self._concept_log_ts = 0.0
        self._episodic_log_ts = 0.0
        self.metrics: Dict[str, Any] = {
            "concept_backlog": 0,
            "concept_load": 0.0,
            "concept_quality": 0.0,
            "concept_beta": 0.2,
            "concept_last_score": 0.0,
            "episodic_backlog": 0,
            "episodic_load": 0.0,
            "episodic_quality": 0.0,
            "episodic_beta": 0.2,
            "episodic_reward": 0.0,
        }

    def step(self) -> None:
        now = time.time()
        concept_backlog = self._safe_pending(self.extractor.pending_backlog, self.memory, self._concept_backlog_window)
        concept_quality = self.extractor.quality_signal()
        concept_load = self._normalize(concept_backlog, self._concept_backlog_window)
        concept_info = self._concept_signal.update(concept_load, concept_quality)
        self.metrics.update(
            {
                "concept_backlog": concept_backlog,
                "concept_load": concept_info["value"],
                "concept_quality": concept_quality,
                "concept_beta": concept_info["beta"],
            }
        )

        target_concept_period = self._schedule_period(
            concept_info["value"],
            concept_quality,
            min_period=1.0,
            max_period=14.0,
        )
        self.concept_period = self._limit_period_change(self.concept_period, target_concept_period, max_delta=2.0)

        concept_due = concept_backlog > 0 and (
            (now - self.last_concept_step) >= self.concept_period
            or concept_info["value"] > 0.65
        )

        if concept_due:
            batch = self._batch_size(concept_backlog, base=300, ceiling=self._concept_backlog_window)
            try:
                self.extractor.run_once(max_batch=batch)
                self.last_concept_step = now
                self.metrics["concept_last_score"] = self.extractor.last_batch_score
                self.metrics["concept_quality"] = self.extractor.quality_signal()
            except Exception as exc:
                print(f"[semantic] concept step error: {exc}")
        elif concept_info["drift"] > 0.25 and now - self._concept_log_ts > 5.0:
            print(
                "[semantic] concept load drift "
                f"{concept_info['drift']:.3f} (beta={concept_info['beta']:.2f})"
            )
            self._concept_log_ts = now

        episodic_backlog = self._safe_pending(
            self.linker.pending_backlog, self._episodic_backlog_window
        )
        episodic_quality = self.linker.quality_signal()
        episodic_load = self._normalize(episodic_backlog, self._episodic_backlog_window)
        episodic_info = self._episodic_signal.update(episodic_load, episodic_quality)
        self.metrics.update(
            {
                "episodic_backlog": episodic_backlog,
                "episodic_load": episodic_info["value"],
                "episodic_quality": episodic_quality,
                "episodic_beta": episodic_info["beta"],
            }
        )

        target_episodic_period = self._schedule_period(
            episodic_info["value"],
            episodic_quality,
            min_period=5.0,
            max_period=60.0,
        )
        self.episodic_period = self._limit_period_change(
            self.episodic_period, target_episodic_period, max_delta=6.0
        )

        episodic_due = episodic_backlog > 0 and (
            (now - self.last_link_step) >= self.episodic_period
            or episodic_info["value"] > 0.55
        )

        if episodic_due:
            batch = self._batch_size(
                episodic_backlog, base=200, ceiling=self._episodic_backlog_window
            )
            try:
                self.linker.run_once(max_batch=batch)
                self.last_link_step = now
                self.metrics["episodic_reward"] = self.linker.last_reward
                self.metrics["episodic_quality"] = self.linker.quality_signal()
            except Exception as exc:
                print(f"[semantic] episodic step error: {exc}")
        elif episodic_info["drift"] > 0.2 and now - self._episodic_log_ts > 10.0:
            print(
                "[semantic] episodic load drift "
                f"{episodic_info['drift']:.3f} (beta={episodic_info['beta']:.2f})"
            )
            self._episodic_log_ts = now

        llm_payload = {
            "concept": {
                "backlog": concept_backlog,
                "quality": concept_quality,
                "load": concept_info["value"],
                "beta": concept_info["beta"],
            },
            "episodic": {
                "backlog": episodic_backlog,
                "quality": episodic_quality,
                "load": episodic_info["value"],
                "beta": episodic_info["beta"],
            },
        }
        response = try_call_llm_dict(
            "semantic_memory_manager",
            input_payload=llm_payload,
            logger=LOGGER,
        )
        if response:
            tasks = response.get("tasks")
            if tasks is not None:
                self.metrics["llm_tasks"] = tasks
            notes = response.get("notes")
            if notes:
                self.metrics["llm_notes"] = notes

    def get_top_concepts(self, k: int = 20) -> List[Concept]:
        return self.store.get_top_concepts(k)

    def neighbors(self, concept_label_prefix: str, k: int = 10) -> List[Tuple[Relation, Concept]]:
        candidates = self.store.find_by_label_prefix(concept_label_prefix, k=1)
        if not candidates:
            return []
        return self.store.neighbors(candidates[0].id, rtype=None, k=k)

    # ------------------------------------------------------------------
    # Vector index helpers
    def index_document(
        self,
        doc_id: str,
        text: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist ``text`` in the vector store and keep lightweight metadata."""

        if not doc_id:
            return
        try:
            self.index_backend.upsert(doc_id, text or "")
            if metadata:
                self._doc_metadata[doc_id] = dict(metadata)
        except Exception as exc:
            print(f"[semantic] vector index failure for {doc_id}: {exc}")

    def search_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search indexed documents using the vector backend."""

        try:
            hits = self.index_backend.search(query or "", k=k)
        except Exception:
            return []

        results: List[Dict[str, Any]] = []
        for doc_id, score in hits:
            payload: Dict[str, Any] = {
                "id": doc_id,
                "score": float(score),
            }
            if doc_id in self._doc_metadata:
                payload["metadata"] = dict(self._doc_metadata[doc_id])
            results.append(payload)
        return results

    # ------------------------------------------------------------------
    # Diagnostics helpers
    def diagnostics(self) -> Dict[str, Any]:
        """Return the latest adaptive metrics."""

        return dict(self.metrics)

    # ------------------------------------------------------------------
    # Scheduling helpers
    def _normalize(self, backlog: int, window: int) -> float:
        if window <= 0:
            return 0.0
        return max(0.0, min(1.0, float(backlog) / float(window)))

    def _schedule_period(
        self,
        load: float,
        quality: float,
        *,
        min_period: float,
        max_period: float,
    ) -> float:
        pressure = 0.7 * max(0.0, min(1.0, load)) + 0.3 * max(0.0, min(1.0, 1.0 - quality))
        span = max(0.0, max_period - min_period)
        return max_period - span * pressure

    def _limit_period_change(self, current: float, desired: float, *, max_delta: float) -> float:
        desired = max(0.1, desired)
        delta = desired - current
        if delta > max_delta:
            delta = max_delta
        elif delta < -max_delta:
            delta = -max_delta
        updated = current + delta
        return max(0.1, updated)

    def _batch_size(self, backlog: int, *, base: int, ceiling: int) -> int:
        if backlog <= 0:
            return max(1, base)
        normalized = self._normalize(backlog, max(1, ceiling))
        factor = 0.6 + 0.8 * normalized
        return max(50, min(int(base * factor), max(base, ceiling)))

    def _safe_pending(self, func, *args, **kwargs) -> int:
        try:
            value = func(*args, **kwargs)
            return int(value or 0)
        except Exception:
            return 0
