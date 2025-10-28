from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Callable, Dict, Optional

from AGI_Evolutive.utils.jsonsafe import json_sanitize
from AGI_Evolutive.utils.llm_service import try_call_llm_dict

from .metrics import aggregate_metrics, bootstrap_superiority, dominates
from .mutations import generate_overrides
from .promote import PromotionManager
from .sandbox import SandboxRunner, ArchFactory
from .quality import QualityGateRunner
from .code_evolver import CodeEvolver


_LOGGER = logging.getLogger(__name__)


class SelfImprover:
    """Coordinate champion/challenger evaluation and promotion workflow."""

    def __init__(
        self,
        arch_factory: ArchFactory,
        memory: Any,
        question_manager: Optional[Any] = None,
        config_root: str = "config",
        eval_root: str = "data/eval",
        apply_overrides: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.arch_factory = arch_factory
        self.memory = memory
        self.questions = question_manager
        self.prom = PromotionManager(config_root)
        self.sandbox = SandboxRunner(arch_factory, eval_root=eval_root)
        self.quality = QualityGateRunner(arch_factory)
        self.code_evolver = CodeEvolver(
            repo_root=os.getcwd(),
            sandbox=self.sandbox,
            quality=self.quality,
            arch_factory=arch_factory,
        )
        self._apply_overrides = apply_overrides
        self.curriculum_level = "base"

        # Ensure any previously promoted overrides affect the live architecture.
        self._refresh_live_overrides()

    # ------------------------------------------------------------------
    # Internal helpers
    def _active_overrides(self) -> Dict[str, Any]:
        return self.prom.load_active()

    def _refresh_live_overrides(self) -> None:
        try:
            overrides = self.prom.load_active()
        except Exception:
            overrides = {}
        level = str(overrides.get("curriculum_level", self.curriculum_level))
        self.curriculum_level = level
        try:
            self.sandbox.set_curriculum_level(level)
        except Exception:
            pass
        if not self._apply_overrides:
            return
        try:
            self._apply_overrides(overrides)
        except Exception:
            pass

    def _log_experiment(self, payload: Dict[str, Any]) -> None:
        os.makedirs("data/self_improve", exist_ok=True)
        record = {"t": time.time(), **payload}
        with open("data/self_improve/experiments.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(json_sanitize(record)) + "\n")

    def _llm_guard_candidate(
        self,
        overrides: Dict[str, Any],
        metrics: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> tuple[bool, Dict[str, Any], Optional[Dict[str, Any]]]:
        metadata = dict(metadata or {})
        response = try_call_llm_dict(
            "self_improver_promotion_brief",
            input_payload={
                "overrides": overrides,
                "metrics": metrics,
                "metadata": metadata,
            },
            logger=_LOGGER,
            max_retries=2,
        )
        response_data = dict(response) if response else None
        if response_data:
            metadata.setdefault("llm_brief", response_data)
            go_value = response_data.get("go")
            confidence = response_data.get("confidence", 0.0)
            try:
                conf_val = float(confidence)
            except (TypeError, ValueError):
                conf_val = 0.0
            if isinstance(go_value, bool) and not go_value and conf_val >= 0.75:
                self._log_experiment(
                    {
                        "kind": "llm_gate_reject",
                        "overrides": overrides,
                        "metrics": metrics,
                        "llm_brief": response_data,
                    }
                )
                return True, metadata, response_data
        return False, metadata, response_data

    # ------------------------------------------------------------------
    # Public API
    def set_curriculum_level(self, level: str) -> None:
        self.curriculum_level = str(level)
        try:
            self.sandbox.set_curriculum_level(self.curriculum_level)
        except Exception:
            pass

    def rotate_curriculum(self, level: str) -> Optional[str]:
        self.set_curriculum_level(level)
        overrides = dict(self._active_overrides())
        overrides["curriculum_level"] = self.curriculum_level
        cid = self.prom.stage_candidate(overrides, {"acc": 0.0}, metadata={"kind": "curriculum"})
        self._log_experiment({"kind": "curriculum_rotate", "level": self.curriculum_level, "cid": cid})
        try:
            if hasattr(self.memory, "add_memory"):
                self.memory.add_memory(
                    kind="curriculum_rotated",
                    content=self.curriculum_level,
                    metadata={"candidate_id": cid},
                )
        except Exception:
            pass
        return cid

    def run_code_cycle(self, n_candidates: int = 2) -> Optional[str]:
        if not self.code_evolver:
            return None
        base = self._active_overrides()
        champion_eval = self.sandbox.run_all(base)
        champion_metrics = aggregate_metrics(champion_eval.get("samples", []))
        best_cid: Optional[str] = None
        for patch in self.code_evolver.generate_candidates(max(1, n_candidates)):
            serialised = self.code_evolver.serialise_patch(patch)
            report = self.code_evolver.evaluate_patch(patch, champion_metrics)
            payload = {
                "kind": "code_cycle",
                "summary": report.get("summary"),
                "passed": report.get("passed", False),
                "file": report.get("file"),
            }
            self._log_experiment(payload)
            if not report.get("passed", False):
                try:
                    if hasattr(self.memory, "add_memory"):
                        self.memory.add_memory(
                            kind="code_tests_failed",
                            content=str(report.get("summary", ""))[:160],
                            metadata={"lint": report.get("lint"), "static": report.get("static")},
                        )
                except Exception:
                    pass
                continue
            evaluation = report.get("evaluation", {})
            aggregated = aggregate_metrics(evaluation.get("samples", []))
            metadata = {
                "kind": "code_patch",
                "file": report.get("file"),
                "diff": report.get("diff"),
                "summary": report.get("summary"),
                "quality": report.get("quality"),
                "canary": report.get("canary"),
                "patch": serialised,
            }
            skip_candidate, metadata, llm_brief = self._llm_guard_candidate(base, aggregated, metadata)
            if skip_candidate:
                continue
            cid = self.prom.stage_candidate(base, aggregated, metadata=metadata)
            best_cid = cid
            try:
                if hasattr(self.memory, "add_memory"):
                    self.memory.add_memory(
                        kind="code_candidate",
                        content=str((metadata.get("llm_brief") or {}).get("summary") or report.get("summary", ""))[:160],
                        metadata={"candidate_id": cid, "file": report.get("file")},
                    )
                    self.memory.add_memory(
                        kind="code_tests_passed",
                        content=str(report.get("summary", ""))[:160],
                        metadata={"candidate_id": cid, "evaluation": aggregated},
                    )
            except Exception:
                pass
        return best_cid

    def run_cycle(self, n_candidates: int = 4) -> Optional[str]:
        base = self._active_overrides()
        champion_eval = self.sandbox.run_all(base)
        champ_aggr = aggregate_metrics(champion_eval.get("samples", []))
        champ_scores = champion_eval.get("scores", [])
        self._log_experiment({"kind": "champion_eval", "metrics": champ_aggr})

        candidates = generate_overrides(base, n=n_candidates)
        best_cid: Optional[str] = None
        best_metrics: Optional[Dict[str, float]] = None
        best_metadata: Optional[Dict[str, Any]] = None

        for cand in candidates:
            evaluation = self.sandbox.run_all(cand)
            aggregated = aggregate_metrics(evaluation.get("samples", []))
            scores = evaluation.get("scores", [])
            p_value = bootstrap_superiority(champ_scores, scores)
            quality = self.quality.run(cand)
            safety = evaluation.get("security", {"passed": True})
            curriculum = evaluation.get("curriculum", [])
            mutation = evaluation.get("mutation_testing")
            if not quality.get("passed", False) or not safety.get("passed", False):
                self._log_experiment(
                    {
                        "kind": "gate_reject",
                        "reason": "quality" if not quality.get("passed", False) else "security",
                        "overrides": cand,
                        "metrics": aggregated,
                    }
                )
                continue
            self._log_experiment(
                {
                    "kind": "challenger_eval",
                    "overrides": cand,
                    "metrics": aggregated,
                    "p": p_value,
                    "quality": quality,
                    "security": safety,
                    "curriculum": curriculum,
                    "mutation": mutation,
                }
            )
            if not (dominates(champ_aggr, aggregated) and p_value < 0.2):
                continue
            canary = self.sandbox.run_canary(cand, champ_aggr)
            if not canary.get("passed", False):
                self._log_experiment(
                    {
                        "kind": "gate_reject",
                        "reason": "canary",
                        "overrides": cand,
                        "metrics": aggregated,
                        "canary": canary,
                    }
                )
                continue
            metadata = {
                "kind": "override",
                "quality": quality,
                "safety": safety,
                "curriculum": curriculum,
                "mutation": mutation,
                "canary": canary,
            }
            skip_candidate, metadata, _ = self._llm_guard_candidate(cand, aggregated, metadata)
            if skip_candidate:
                continue
            cid = self.prom.stage_candidate(cand, aggregated, metadata=metadata)
            best_cid = cid
            best_metrics = aggregated
            best_metadata = metadata

        # Evaluate code-level challengers via the CodeEvolver
        code_patches = self.code_evolver.generate_candidates(max(1, n_candidates // 2))
        for patch in code_patches:
            report = self.code_evolver.evaluate_patch(patch, champ_aggr)
            evaluation = report.get("evaluation", {})
            aggregated = aggregate_metrics(evaluation.get("samples", []))
            scores = evaluation.get("scores", [])
            p_value = bootstrap_superiority(champ_scores, scores)
            self._log_experiment(
                {
                    "kind": "code_challenger_eval",
                    "patch": report.get("diff"),
                    "metrics": aggregated,
                    "p": p_value,
                    "quality": report.get("quality"),
                    "lint": report.get("lint"),
                    "static": report.get("static"),
                    "canary": report.get("canary"),
                }
            )
            if not report.get("passed", False):
                continue
            safety = evaluation.get("security", {"passed": True})
            if not safety.get("passed", False):
                continue
            if not (dominates(champ_aggr, aggregated) and p_value < 0.2):
                continue
            canary = report.get("canary", {})
            if not canary.get("passed", False):
                continue
            metadata = {
                "kind": "code_patch",
                "quality": report.get("quality"),
                "safety": safety,
                "canary": canary,
                "lint": report.get("lint"),
                "static": report.get("static"),
                "curriculum": evaluation.get("curriculum"),
                "mutation": evaluation.get("mutation_testing"),
                "patch": self.code_evolver.serialise_patch(patch),
            }
            skip_candidate, metadata, _ = self._llm_guard_candidate(base, aggregated, metadata)
            if skip_candidate:
                continue
            cid = self.prom.stage_candidate(base, aggregated, metadata=metadata)
            best_cid = cid
            best_metrics = aggregated
            best_metadata = metadata

        if not best_cid or not best_metrics:
            try:
                if hasattr(self.memory, "add_memory"):
                    self.memory.add_memory(
                        kind="self_improve",
                        content="Aucun challenger ne surpasse le champion",
                        metadata={"champion": champ_aggr},
                    )
            except Exception:
                pass
            return None

        llm_brief = (best_metadata or {}).get("llm_brief") if best_metadata else None
        summary = ""
        if isinstance(llm_brief, dict):
            summary = str(llm_brief.get("summary") or "").strip()
        metric_clause = (
            f"acc={best_metrics.get('acc', 0.0):.3f}, "
            f"ece={best_metrics.get('cal_ece', 0.0):.3f}"
        )
        prompt_intro = summary or f"Promotion du challenger {best_cid}".strip()
        question = (
            f"{prompt_intro} ? ({metric_clause}) "
            "Réponds 'oui' pour promouvoir."
        )
        try:
            if hasattr(self.memory, "add_memory"):
                self.memory.add_memory(
                    kind="promotion_request",
                    content=question,
                    metadata={
                        "cid": best_cid,
                        "metrics": best_metrics,
                        "details": best_metadata,
                        "llm_summary": llm_brief,
                    },
                )
            if self.questions and hasattr(self.questions, "add_question"):
                self.questions.add_question(question)
        except Exception:
            pass
        return best_cid

    def try_promote_from_reply(self, user_text: str) -> bool:
        normalized = (user_text or "").strip().lower()
        if normalized not in {"oui", "yes", "go", "ok"}:
            return False
        cid: Optional[str] = None
        if hasattr(self.memory, "get_recent_memories"):
            try:
                recents = self.memory.get_recent_memories(60)
            except Exception:
                recents = []
        else:
            recents = []
        for item in reversed(recents):
            if item.get("kind") == "promotion_request":
                meta = item.get("metadata") or {}
                cid = meta.get("cid")
                if cid:
                    break
        if not cid:
            return False
        self.promote(cid)
        return True

    def promote(self, cid: str) -> None:
        candidate = self.prom.read_candidate(cid)
        metadata = candidate.get("metadata", {})
        quality = metadata.get("quality", {"passed": True})
        safety = metadata.get("safety", {"passed": True})
        canary = metadata.get("canary", {"passed": True})
        if not quality.get("passed", False):
            raise RuntimeError("Quality gates failed, promotion aborted")
        if not safety.get("passed", False):
            raise RuntimeError("Security gate failed, promotion aborted")
        if not canary.get("passed", False):
            raise RuntimeError("Canary run failed, promotion aborted")

        if metadata.get("kind") == "code_patch":
            patch_payload = metadata.get("patch") or {}
            patch_payload.setdefault("metadata", metadata)
            self.code_evolver.promote_patch(patch_payload)

        self.prom.promote(cid, quality_runner=self.quality)
        self._refresh_live_overrides()
        try:
            if hasattr(self.memory, "add_memory"):
                self.memory.add_memory(kind="promotion", content=f"Promu {cid}")
        except Exception:
            pass

    def rollback(self, steps: int = 1) -> None:
        self.prom.rollback(steps)
        self._refresh_live_overrides()
        try:
            if hasattr(self.memory, "add_memory"):
                self.memory.add_memory(kind="rollback", content=f"Rollback {steps}")
        except Exception:
            pass
