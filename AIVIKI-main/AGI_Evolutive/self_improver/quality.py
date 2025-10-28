from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from AGI_Evolutive.utils.llm_service import try_call_llm_dict


_LOGGER = logging.getLogger(__name__)


@dataclass
class GateResult:
    name: str
    passed: bool
    details: Dict[str, Any]


class QualityGateRunner:
    """Run unit and integration quality gates before a promotion."""

    def __init__(self, arch_factory: Callable[[Dict[str, Any]], Any]) -> None:
        self.arch_factory = arch_factory
        self._unit_modules = [
            "AGI_Evolutive.reasoning.abduction",
            "AGI_Evolutive.core.cognitive_architecture",
            "AGI_Evolutive.self_improver.metrics",
        ]

    # ------------------------------------------------------------------
    def _run_unit_suite(self) -> GateResult:
        failures: List[str] = []
        for module_name in self._unit_modules:
            try:
                module = importlib.import_module(module_name)
                importlib.reload(module)
            except Exception as exc:
                failures.append(f"{module_name}: {exc}")
        return GateResult(
            name="unit",
            passed=not failures,
            details={"failures": failures},
        )

    def _run_integration_suite(self, overrides: Dict[str, Any]) -> GateResult:
        diagnostics: Dict[str, Any] = {}
        try:
            arch = self.arch_factory(overrides)
        except Exception as exc:
            return GateResult(
                name="integration",
                passed=False,
                details={"error": f"arch_init: {exc}"},
            )

        abduction_ok = True
        learning_ok = True
        try:
            if hasattr(arch, "abduction"):
                sample = arch.abduction.generate("Symptômes : fièvre, fatigue")
                diagnostics["abduction_top"] = sample[0].label if sample else ""
                abduction_ok = bool(sample)
        except Exception as exc:
            diagnostics["abduction_error"] = str(exc)
            abduction_ok = False

        try:
            learning = getattr(arch, "learning", None)
            if learning and hasattr(learning, "self_assess_concept"):
                res = learning.self_assess_concept("principe_X")
                if isinstance(res, dict):
                    confidence = float(res.get("confidence", 0.0))
                else:
                    confidence = float(getattr(res, "confidence", 0.0))
                diagnostics["learning_confidence"] = confidence
                learning_ok = confidence >= 0.0
        except Exception as exc:
            diagnostics["learning_error"] = str(exc)
            learning_ok = False

        return GateResult(
            name="integration",
            passed=abduction_ok and learning_ok,
            details=diagnostics,
        )

    # ------------------------------------------------------------------
    def run(self, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        overrides = overrides or {}
        unit = self._run_unit_suite()
        integration = self._run_integration_suite(overrides)
        passed = unit.passed and integration.passed
        payload = {
            "passed": passed,
            "unit": unit.__dict__,
            "integration": integration.__dict__,
        }

        llm_review = try_call_llm_dict(
            "self_improver_quality_review",
            input_payload={
                "overrides": overrides,
                "unit": payload["unit"],
                "integration": payload["integration"],
            },
            logger=_LOGGER,
            max_retries=2,
        )
        if llm_review:
            payload["llm_review"] = dict(llm_review)
            llm_passed = llm_review.get("llm_passed")
            if isinstance(llm_passed, bool) and not llm_passed:
                payload["passed"] = False

        return payload
