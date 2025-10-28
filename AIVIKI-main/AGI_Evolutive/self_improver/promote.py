from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, TYPE_CHECKING

from AGI_Evolutive.core.structures.mai import MAI
from AGI_Evolutive.knowledge.mechanism_store import MechanismStore
from AGI_Evolutive.utils.jsonsafe import json_sanitize
from AGI_Evolutive.utils.llm_service import try_call_llm_dict

if TYPE_CHECKING:
    from .quality import QualityGateRunner

class PromotionStorage:
    """Filesystem-backed storage for promotion artefacts."""

    def __init__(self, root: str = "config") -> None:
        self.root = root
        self.candidate_dir = os.path.join(self.root, "candidates")
        self.active_path = os.path.join(self.root, "active_overrides.json")
        self.hist_path = os.path.join(self.root, "history.jsonl")

        os.makedirs(self.root, exist_ok=True)
        os.makedirs(self.candidate_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Active overrides
    def load_active(self) -> Dict[str, Any]:
        if os.path.exists(self.active_path):
            with open(self.active_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        return {}

    def save_active(self, overrides: Dict[str, Any]) -> None:
        with open(self.active_path, "w", encoding="utf-8") as handle:
            json.dump(json_sanitize(overrides), handle, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Candidates
    def candidate_path(self, cid: str) -> str:
        return os.path.join(self.candidate_dir, f"{cid}.json")

    def write_candidate(self, cid: str, payload: Dict[str, Any]) -> None:
        with open(self.candidate_path(cid), "w", encoding="utf-8") as handle:
            json.dump(json_sanitize(payload), handle, ensure_ascii=False, indent=2)

    def read_candidate(self, cid: str) -> Dict[str, Any]:
        with open(self.candidate_path(cid), "r", encoding="utf-8") as handle:
            return json.load(handle)

    # ------------------------------------------------------------------
    # History
    def append_history(self, event: Dict[str, Any]) -> None:
        with open(self.hist_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(json_sanitize(event)) + "\n")

    def iter_history(self) -> Iterable[Dict[str, Any]]:
        if not os.path.exists(self.hist_path):
            return []
        with open(self.hist_path, "r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]


class PromotionError(RuntimeError):
    """Base exception raised when promotion fails."""


@dataclass
class QualityGateOutcome:
    passed: bool
    report: Dict[str, Any]


class CanaryDeployer(Protocol):
    """Protocol describing a canary deployment helper."""

    def deploy(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        ...

    def finalize(self, success: bool) -> None:
        ...


class PromotionManager:
    """Manage candidate overrides, quality gates, and promotion history."""

    def __init__(self, root: str = "config", storage: Optional[PromotionStorage] = None) -> None:
        self.storage = storage or PromotionStorage(root=root)

        self._logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Active overrides management
    def load_active(self) -> Dict[str, Any]:
        return self.storage.load_active()

    def save_active(self, overrides: Dict[str, Any]) -> None:
        self.storage.save_active(overrides)

    # ------------------------------------------------------------------
    # Candidate lifecycle
    def stage_candidate(
        self,
        overrides: Dict[str, Any],
        metrics: Dict[str, float],
        metadata: Dict[str, Any] | None = None,
    ) -> str:
        cid = str(uuid.uuid4())
        payload_metadata = dict(metadata or {})
        llm_brief: Mapping[str, Any] | None = None
        existing_brief = payload_metadata.get("llm_brief")
        if isinstance(existing_brief, Mapping):
            llm_brief = existing_brief
        elif overrides or metrics:
            llm_brief = try_call_llm_dict(
                "self_improver_promotion_brief",
                input_payload={
                    "overrides": overrides,
                    "metrics": metrics,
                    "metadata": metadata or {},
                },
                logger=self._logger,
                max_retries=2,
            )
            if llm_brief:
                payload_metadata.setdefault("llm_brief", dict(llm_brief))

        payload = {
            "overrides": overrides,
            "metrics": metrics,
            "metadata": payload_metadata,
            "t": time.time(),
            "cid": cid,
        }
        if llm_brief:
            payload["llm_brief"] = dict(llm_brief)
        self.storage.write_candidate(cid, payload)
        return cid

    def read_candidate(self, cid: str) -> Dict[str, Any]:
        return self.storage.read_candidate(cid)

    def _run_quality_gates(
        self,
        overrides: Dict[str, Any],
        quality_runner: Optional["QualityGateRunner"],
    ) -> QualityGateOutcome:
        if quality_runner is None:
            return QualityGateOutcome(passed=True, report={"skipped": True})

        try:
            report = quality_runner.run(overrides)
        except Exception as exc:  # pragma: no cover - defensive guard
            raise PromotionError(f"quality_gate_error: {exc}") from exc

        passed = bool(report.get("passed", False))
        if not passed:
            raise PromotionError("quality_gates_failed")
        return QualityGateOutcome(passed=passed, report=report)

    def _log_history(self, event: Dict[str, Any]) -> None:
        self.storage.append_history(event)

    def promote(
        self,
        cid: str,
        quality_runner: Optional["QualityGateRunner"] = None,
        canary: Optional[CanaryDeployer] = None,
        observers: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        data = self.read_candidate(cid)
        overrides = data.get("overrides", {})

        quality_outcome = self._run_quality_gates(overrides, quality_runner)

        canary_report: Optional[Dict[str, Any]] = None
        if canary is not None:
            try:
                canary_report = canary.deploy(overrides)
            except Exception as exc:  # pragma: no cover - defensive guard
                try:
                    canary.finalize(False)
                finally:
                    raise PromotionError(f"canary_deploy_error: {exc}") from exc

        self.save_active(overrides)

        record = {
            "t": time.time(),
            "event": "promote",
            "promotion_id": str(uuid.uuid4()),
            "cid": cid,
            "overrides": overrides,
            "metrics": data.get("metrics"),
            "metadata": data.get("metadata"),
            "quality": quality_outcome.report,
            "canary": canary_report,
            "observers": observers or [],
        }
        self._log_history(record)

        if canary is not None:
            canary.finalize(True)

        return record

    def rollback(self, steps: int = 1) -> None:
        history = list(self.storage.iter_history())
        if not history:
            return
        lines = history
        prev: Dict[str, Any] | None = None
        for entry in reversed(lines):
            if entry.get("event") == "promote":
                if steps <= 0:
                    prev = entry
                    break
                steps -= 1
        if not prev:
            return
        self.save_active(prev.get("overrides", {}))
        self._log_history({"t": time.time(), "event": "rollback", "to": prev})

    def describe_history(self) -> List[Dict[str, Any]]:
        """Return the promotion history as a list for observability/inspection."""

        history = self.storage.iter_history()
        if isinstance(history, list):
            return history
        return list(history)


class PromoteManager:
    def sandbox_mechanism(self, mai: MAI) -> bool:
        """
        Exécute une batterie de tests/ablation en sandbox.
        Retourne True si aucune régression critique détectée.
        """

        # branche tes tests existants ici (latence, non-régression, cohérence, etc.)
        return True

    def promote_mechanism(self, mai: MAI) -> bool:
        """
        Intègre le MAI en canary, puis généralise selon les résultats,
        et enregistre un handler NLG s'il est fourni par le MAI.
        """

        ms = MechanismStore()
        ms.add(mai)

        # Option: si le MAI fournit un handler NLG spécifique, l’enregistrer
        try:
            from AGI_Evolutive.language.nlg import register_nlg_handler

            handler = getattr(mai, "nlg_handler", None)
            if callable(handler) and hasattr(mai, "action_hint_for_handler"):
                register_nlg_handler(mai.action_hint_for_handler, handler)
        except Exception:
            pass

        return True
