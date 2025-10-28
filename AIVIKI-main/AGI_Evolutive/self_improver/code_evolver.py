from __future__ import annotations

import ast
import contextlib
import difflib
import importlib
import json
import logging
import re
import random
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from AGI_Evolutive.utils.jsonsafe import json_sanitize
from AGI_Evolutive.utils.llm_service import try_call_llm_dict

from .quality import QualityGateRunner


logger = logging.getLogger(__name__)


@dataclass
class CodePatch:
    patch_id: str
    target_file: Path
    module_name: str
    original_source: str
    patched_source: str
    summary: str
    target_id: str = ""


class OnlineHeuristicLearner:
    """Simple online learner to adapt heuristic weights from feedback."""

    def __init__(
        self,
        key: str,
        state: Optional[Dict[str, Any]] = None,
        *,
        lower: float = 0.0,
        upper: float = 2.0,
    ) -> None:
        self.key = key
        self.lower = lower
        self.upper = upper
        state = state or {}
        span = max(upper - lower, 1e-6)
        self.mu: float = float(state.get("mu", (lower + upper) / 2))
        self.sigma: float = float(state.get("sigma", span / 4))
        self.learning_rate: float = float(state.get("lr", 0.2))
        self.successes: int = int(state.get("success", 0))
        self.failures: int = int(state.get("failure", 0))
        self._last_proposal: Optional[Dict[str, float]] = None

    @property
    def confidence(self) -> float:
        total = self.successes + self.failures + 1
        return self.successes / total

    def propose(self, current_value: float) -> float:
        center = 0.7 * self.mu + 0.3 * current_value
        candidate = random.gauss(center, self.sigma)
        candidate = max(self.lower, min(self.upper, candidate))
        self._last_proposal = {
            "original": float(current_value),
            "candidate": float(candidate),
        }
        return candidate

    def update(self, success: bool) -> None:
        if not self._last_proposal:
            return
        candidate = self._last_proposal["candidate"]
        error = candidate - self.mu
        direction = 1.0 if success else -1.0
        self.mu = max(self.lower, min(self.upper, self.mu + self.learning_rate * direction * error))
        span = max(self.upper - self.lower, 1e-6)
        adjust = 0.85 if success else 1.15
        self.sigma = min(max(self.sigma * adjust, span * 0.02), span)
        if success:
            self.successes += 1
        else:
            self.failures += 1
        self._last_proposal = None

    def to_state(self) -> Dict[str, Any]:
        return {
            "mu": self.mu,
            "sigma": self.sigma,
            "lr": self.learning_rate,
            "success": self.successes,
            "failure": self.failures,
        }


class _ScoreHeuristicTweaker(ast.NodeTransformer):
    """Mutate numeric constants in the abduction score heuristic."""

    def __init__(
        self,
        learner: Optional[OnlineHeuristicLearner] = None,
        *,
        learner_key: Optional[str] = None,
        forced_value: Optional[float] = None,
    ) -> None:
        self._context: List[str] = []
        self._changed: bool = False
        self._metadata: Dict[str, Any] = {}
        self._learner = learner
        self._learner_key = learner_key or (learner.key if learner else None)
        self._forced_value = forced_value

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self._context.append(node.name)
        new_node = self.generic_visit(node)
        self._context.pop()
        return new_node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:  # type: ignore[override]
        if self._changed:
            return node
        if not self._context or self._context[-1] != "_score":
            return node
        if not isinstance(node.value, (int, float)):
            return node
        original = float(node.value)
        if original <= 0.0 or original > 2.0:
            return node
        if self._forced_value is not None:
            candidate = float(max(0.0, min(2.0, self._forced_value)))
            self._forced_value = None
            self._metadata["llm_override"] = True
        elif self._learner:
            candidate = self._learner.propose(original)
        else:
            delta = random.uniform(-0.08, 0.08)
            candidate = max(0.0, min(2.0, original + delta))
        if candidate == original:
            return node
        self._changed = True
        rounded = round(candidate, 3)
        self._metadata = {
            "kind": "heuristic_weight_shift",
            "from": original,
            "to": rounded,
            "delta": round(rounded - original, 3),
            "learner_key": self._learner_key,
        }
        if self._learner:
            self._metadata["confidence"] = round(self._learner.confidence, 4)
            self._metadata["proposal"] = {
                "original": round(original, 4),
                "candidate": round(candidate, 4),
            }
        return ast.copy_location(ast.Constant(value=rounded), node)


def _extract_numeric_patch(diff_text: str) -> Optional[float]:
    """Extract the first numeric literal from a unified diff snippet."""

    if not diff_text:
        return None
    for line in diff_text.splitlines():
        line = line.strip()
        if not line.startswith("+"):
            continue
        tokens = [tok for tok in re.split(r"[^0-9.,-]", line[1:]) if tok]
        for token in tokens:
            cleaned = token.replace(",", ".")
            try:
                return float(cleaned)
            except ValueError:
                continue
    return None


class CodeEvolver:
    """Propose and evaluate AST-level patches for the cognitive architecture."""

    def __init__(
        self,
        repo_root: str,
        sandbox: Any,
        quality: QualityGateRunner,
        arch_factory: Any,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.sandbox = sandbox
        self.quality = quality
        self.arch_factory = arch_factory
        self._state_dir = self.repo_root / "data" / "self_improve"
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._state_path = self._state_dir / "code_evolver_state.json"
        self._history_path = self._state_dir / "code_evolver_history.jsonl"
        self._state: Dict[str, Any] = self._load_state()
        self._targets: List[Dict[str, str]] = self._initialise_targets()
        self._bandit_state: Dict[str, Dict[str, int]] = dict(self._state.get("bandit", {}))
        self._learners: Dict[str, OnlineHeuristicLearner] = {}
        for target in self._targets:
            learner_key = target.get("learner_key")
            if not learner_key:
                continue
            self._learners[learner_key] = OnlineHeuristicLearner(
                learner_key, self._state.get("learners", {}).get(learner_key, {})
            )

    # ------------------------------------------------------------------
    def _initialise_targets(self) -> List[Dict[str, str]]:
        defaults: List[Dict[str, str]] = [
            {
                "id": "reasoning_abduction_score",
                "file": "AGI_Evolutive/reasoning/abduction.py",
                "module": "AGI_Evolutive.reasoning.abduction",
                "learner_key": "AGI_Evolutive.reasoning.abduction._score",
                "summary": "Ajustement automatique d'un poids heuristique dans abduction._score",
            }
        ]
        resolved: List[Dict[str, str]] = []
        for target in defaults:
            file_path = self.repo_root / target["file"]
            if not file_path.exists() and target["file"].startswith("AGI_Evolutive/"):
                alt = target["file"].split("AGI_Evolutive/", 1)[1]
                if (self.repo_root / alt).exists():
                    target = dict(target)
                    target["file"] = alt
                    file_path = self.repo_root / target["file"]
            if file_path.exists():
                resolved.append(target)

        config_path = self.repo_root / "configs" / "evolution_targets.json"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as handle:
                    config_targets = json.load(handle)
            except Exception:
                config_targets = []
            for entry in config_targets or []:
                if not isinstance(entry, dict):
                    continue
                file_rel = entry.get("file")
                module = entry.get("module")
                if not file_rel or not module:
                    continue
                file_path = self.repo_root / str(file_rel)
                if not file_path.exists():
                    continue
                target: Dict[str, str] = {
                    "id": str(entry.get("id") or file_rel),
                    "file": str(file_rel),
                    "module": str(module),
                }
                if entry.get("learner_key"):
                    target["learner_key"] = str(entry["learner_key"])
                if entry.get("summary"):
                    target["summary"] = str(entry["summary"])
                resolved.append(target)

        unique: Dict[str, Dict[str, str]] = {}
        for target in resolved:
            unique[target.get("id", target["file"])] = target
        final_targets = list(unique.values())
        if not final_targets:
            raise RuntimeError("No valid targets configured for CodeEvolver")
        return final_targets

    def _load_state(self) -> Dict[str, Any]:
        if not self._state_path.exists():
            return {}
        try:
            with open(self._state_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            return {}

    def _save_state(self) -> None:
        payload = {
            "bandit": self._bandit_state,
            "learners": {k: learner.to_state() for k, learner in self._learners.items()},
        }
        with open(self._state_path, "w", encoding="utf-8") as handle:
            json.dump(json_sanitize(payload), handle, indent=2, sort_keys=True)

    def _select_target(self) -> Dict[str, str]:
        best: Optional[Dict[str, str]] = None
        best_score = -1.0
        for target in self._targets:
            stats = self._bandit_state.get(target.get("id", target["file"]), {})
            success = int(stats.get("success", 0))
            failure = int(stats.get("failure", 0))
            sample = random.betavariate(success + 1, failure + 1)
            if sample > best_score or best is None:
                best_score = sample
                best = target
        return best or self._targets[0]

    def _update_bandit_state(self, target_id: Optional[str], success: bool) -> None:
        if not target_id:
            return
        stats = self._bandit_state.setdefault(target_id, {"success": 0, "failure": 0})
        if success:
            stats["success"] = int(stats.get("success", 0)) + 1
        else:
            stats["failure"] = int(stats.get("failure", 0)) + 1

    def _update_learner(self, metadata: Dict[str, Any], success: bool) -> None:
        learner_key = metadata.get("learner_key") if metadata else None
        if not learner_key:
            return
        learner = self._learners.get(learner_key)
        if learner is None:
            learner = OnlineHeuristicLearner(learner_key)
            self._learners[learner_key] = learner
        learner.update(success)

    def _record_patch_outcome(self, patch: CodePatch, report: Dict[str, Any]) -> None:
        entry = {
            "timestamp": time.time(),
            "patch_id": patch.patch_id,
            "target_id": patch.target_id or report.get("metadata", {}).get("target_id"),
            "module": patch.module_name,
            "file": self._relative_to_repo(patch.target_file),
            "summary": report.get("summary"),
            "passed": report.get("passed", False),
            "quality_passed": report.get("quality", {}).get("passed"),
            "security_passed": report.get("evaluation", {}).get("security", {}).get("passed"),
            "canary_passed": report.get("canary", {}).get("passed"),
            "metadata": report.get("metadata", {}),
        }
        with open(self._history_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(json_sanitize(entry)) + "\n")
        target_id = patch.target_id or report.get("metadata", {}).get("target_id")
        self._update_bandit_state(target_id, report.get("passed", False))
        self._update_learner(report.get("metadata", {}), report.get("passed", False))
        self._save_state()

    # ------------------------------------------------------------------
    def _load_source(self, file_path: Path) -> str:
        with open(file_path, "r", encoding="utf-8") as handle:
            return handle.read()

    def _relative_to_repo(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.repo_root))
        except ValueError:
            return str(path)

    def _patch_source(self, source: str, target: Dict[str, str]) -> Optional[CodePatch]:
        tree = ast.parse(source)
        learner_key = target.get("learner_key")
        forced_value: Optional[float] = None
        llm_metadata: Dict[str, Any] = {}
        payload = {
            "target_id": target.get("id"),
            "summary": target.get("summary"),
            "module": target.get("module"),
            "current_constant_values": re.findall(r"_score\s*\([^)]*\)", source),
        }
        response = try_call_llm_dict(
            "code_evolver",
            input_payload=payload,
            logger=logger,
        )
        if response:
            llm_metadata = dict(response)
            forced_value = _extract_numeric_patch(str(response.get("suggested_patch", "")))
        mutator = _ScoreHeuristicTweaker(
            self._learners.get(learner_key),
            learner_key=learner_key,
            forced_value=forced_value,
        )
        patched = mutator.visit(tree)
        if not mutator.metadata:
            return None
        ast.fix_missing_locations(patched)
        patched_source = ast.unparse(patched)
        summary = target.get(
            "summary",
            "Ajustement automatique d'un poids heuristique dans abduction._score",
        )
        patch = CodePatch(
            patch_id=str(uuid.uuid4()),
            target_file=Path(),  # placeholder
            module_name="",
            original_source=source,
            patched_source=patched_source,
            summary=summary,
        )
        patch.target_id = target.get("id", "")
        # Attach metadata onto object for reporting
        metadata = dict(mutator.metadata)
        if llm_metadata:
            metadata.setdefault("llm_suggestion", llm_metadata)
        if patch.target_id and "target_id" not in metadata:
            metadata["target_id"] = patch.target_id
        setattr(patch, "_metadata", metadata)
        return patch

    def _prepare_patch(self, target: Dict[str, str]) -> Optional[CodePatch]:
        path = self.repo_root / target["file"]
        if not path.exists():
            return None
        source = self._load_source(path)
        candidate = self._patch_source(source, target)
        if not candidate:
            return None
        candidate.target_file = path
        candidate.module_name = target["module"]
        return candidate

    def generate_candidates(self, n: int = 2) -> List[CodePatch]:
        candidates: List[CodePatch] = []
        for _ in range(max(1, n)):
            target = self._select_target()
            patch = self._prepare_patch(target)
            if patch:
                candidates.append(patch)
        return candidates

    # ------------------------------------------------------------------
    @contextlib.contextmanager
    def _apply_patch(self, patch: CodePatch) -> Iterable[None]:
        path = patch.target_file
        backup = self._load_source(path)
        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(patch.patched_source)
            importlib.invalidate_caches()
            module = importlib.import_module(patch.module_name)
            importlib.reload(module)
            yield
        finally:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(backup)
            importlib.invalidate_caches()
            try:
                module = importlib.import_module(patch.module_name)
                importlib.reload(module)
            except Exception:
                pass

    # ------------------------------------------------------------------
    def _lint_patch(self, patch: CodePatch) -> Dict[str, Any]:
        temp_dir = tempfile.mkdtemp(prefix="code_evolver_")
        temp_file = Path(temp_dir) / patch.target_file.name
        with open(temp_file, "w", encoding="utf-8") as handle:
            handle.write(patch.patched_source)
        errors: List[str] = []
        try:
            compile(patch.patched_source, str(temp_file), "exec")
        except Exception as exc:
            errors.append(str(exc))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        return {"passed": not errors, "errors": errors}

    def _static_analysis(self, patch: CodePatch) -> Dict[str, Any]:
        tree = ast.parse(patch.patched_source)
        banned = {ast.Exec, ast.Global, ast.Nonlocal}
        violations: List[str] = []
        for node in ast.walk(tree):
            if type(node) in banned:  # noqa: E721 - intentional exact type check
                violations.append(type(node).__name__)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in {"eval", "exec"}:
                    violations.append(f"call:{node.func.id}")
        return {"passed": not violations, "violations": violations}

    def _build_diff(self, patch: CodePatch) -> str:
        original_lines = patch.original_source.splitlines(keepends=True)
        patched_lines = patch.patched_source.splitlines(keepends=True)
        diff = difflib.unified_diff(
            original_lines,
            patched_lines,
            fromfile=str(patch.target_file),
            tofile=f"{patch.target_file} (patched)",
        )
        return "".join(diff)

    # ------------------------------------------------------------------
    def evaluate_patch(
        self,
        patch: CodePatch,
        baseline_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        lint_report = self._lint_patch(patch)
        static_report = self._static_analysis(patch)
        diff = self._build_diff(patch)
        metadata = getattr(patch, "_metadata", {})
        file_rel = self._relative_to_repo(patch.target_file)
        base_report: Dict[str, Any] = {
            "lint": lint_report,
            "static": static_report,
            "diff": diff,
            "summary": patch.summary,
            "metadata": metadata,
            "module": patch.module_name,
            "file": file_rel,
            "target_id": patch.target_id,
        }
        if not lint_report["passed"] or not static_report["passed"]:
            report = {
                **base_report,
                "passed": False,
                "quality": {},
                "evaluation": {},
                "canary": {},
            }
            self._record_patch_outcome(patch, report)
            return report

        with self._apply_patch(patch):
            quality_report = self.quality.run({})
            evaluation = self.sandbox.run_all({})
            canary = self.sandbox.run_canary({}, baseline_metrics or {})

        report = {
            **base_report,
            "quality": quality_report,
            "evaluation": evaluation,
            "canary": canary,
            "passed": quality_report.get("passed", False)
            and evaluation.get("security", {}).get("passed", True)
            and canary.get("passed", False),
        }
        self._record_patch_outcome(patch, report)
        return report

    # ------------------------------------------------------------------
    def promote_patch(self, patch_payload: Dict[str, Any]) -> None:
        file_rel = patch_payload.get("file")
        if not file_rel:
            raise ValueError("Missing file information for patch promotion")
        path = self.repo_root / file_rel
        patched_source = patch_payload.get("patched_source")
        if not patched_source:
            raise ValueError("Missing patched source for promotion")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(patched_source)
        importlib.invalidate_caches()
        try:
            module = importlib.import_module(patch_payload.get("module", ""))
            importlib.reload(module)
        except Exception:
            pass

    def serialise_patch(self, patch: CodePatch) -> Dict[str, Any]:
        return {
            "patch_id": patch.patch_id,
            "file": self._relative_to_repo(patch.target_file),
            "module": patch.module_name,
            "patched_source": patch.patched_source,
            "summary": patch.summary,
            "metadata": getattr(patch, "_metadata", {}),
            "diff": self._build_diff(patch),
            "target_id": patch.target_id,
        }
