"""High level helpers for the input/output subsystem orchestration."""

from __future__ import annotations

import importlib
import inspect
import logging
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

from AGI_Evolutive.utils.llm_service import try_call_llm_dict

logger = logging.getLogger(__name__)


_INTERFACE_TEMPLATES: Sequence[Mapping[str, Any]] = (
    {
        "name": "perception",
        "module": "AGI_Evolutive.io.perception_interface",
        "summary_hint": "Interface de perception fusionnée (inbox + liaisons mémoire).",
        "status": "stable",
        "entrypoints": ["PerceptionInterface"],
        "llm_hooks": ["perception_preprocess"],
        "responsibilities": (
            "Surveiller le dossier inbox",
            "Lier mémoire, émotions et langage",
            "Pré-analyser les contenus entrants",
        ),
        "fallback_capabilities": (
            "Scan heuristique des fichiers",
            "Journalisation JSONL structurée",
        ),
    },
    {
        "name": "intent",
        "module": "AGI_Evolutive.io.intent_classifier",
        "summary_hint": "Classification d'intentions avec règles et LLM en surcouche.",
        "status": "stable",
        "entrypoints": ["classify", "log_uncertain_intent"],
        "llm_hooks": ["intent_classification"],
        "responsibilities": (
            "Détecter la classe d'intention",
            "Journaliser les cas incertains",
            "Combiner heuristiques et modèle ML léger",
        ),
        "fallback_capabilities": (
            "Regex et patronage multiclasse",
            "Modèle JSON entraîné offline",
        ),
    },
    {
        "name": "action",
        "module": "AGI_Evolutive.io.action_interface",
        "summary_hint": "Priorisation et orchestration des actions candidates.",
        "status": "stable",
        "entrypoints": ["ActionInterface"],
        "llm_hooks": ["action_interface"],
        "responsibilities": (
            "Normaliser les actions candidates",
            "Évaluer impact, effort et risque",
            "Synchroniser les micro-actions AutoSignalRegistry",
        ),
        "fallback_capabilities": (
            "Thompson sampling et GLM adaptatif",
            "Journalisation des évaluations",
        ),
    },
)


def describe_io_interfaces(
    context: Optional[Mapping[str, Any]] = None,
    *,
    use_llm: bool = True,
) -> Mapping[str, Any]:
    """Return a structured overview of the IO subsystem.

    The helper first assembles a baseline snapshot using static knowledge and
    lightweight introspection, then optionally asks the local LLM (via the
    :mod:`AGI_Evolutive.utils.llm_service` helpers) for a richer assessment.
    When the LLM is disabled or fails, the heuristic snapshot is returned as
    is.  The shape matches :data:`utils.llm_specs.LLM_INTEGRATION_SPECS` for
    the ``io_overview`` key.
    """

    baseline = _build_baseline_snapshot(context=context)
    if not use_llm:
        return baseline

    payload = {"baseline": baseline, "context": dict(context or {})}
    llm_response = try_call_llm_dict(
        "io_overview",
        input_payload=payload,
        logger=logger,
    )

    if isinstance(llm_response, Mapping):
        merged = _merge_snapshots(baseline, llm_response)
        if merged:
            return merged

    return baseline


def _build_baseline_snapshot(
    *, context: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    interfaces = [_build_interface_entry(template) for template in _INTERFACE_TEMPLATES]

    summary = (
        "Le sous-système I/O relie perception, classification d'intentions et "
        "passerelle d'action avec supervision LLM optionnelle."
    )
    risks: List[Mapping[str, Any]] = [
        {
            "label": "Intent_classifier reste majoritairement heuristique",
            "severity": "medium",
        },
        {
            "label": "Journalisation inbox volumineuse",
            "severity": "low",
        },
    ]
    recommended_actions: List[str] = [
        "Auditer la précision LLM vs heuristique sur intent_classifier",
        "Consolider la télémétrie perception_preprocess",
    ]

    if context:
        context_summary = _summarise_context(context)
        if context_summary:
            recommended_actions.append(context_summary)

    snapshot: Dict[str, Any] = {
        "summary": summary,
        "interfaces": interfaces,
        "risks": risks,
        "recommended_actions": recommended_actions[:3],
        "notes": "Baseline heuristique construite sans appel LLM.",
    }
    return snapshot


def _build_interface_entry(template: Mapping[str, Any]) -> Dict[str, Any]:
    module_name = str(template.get("module"))
    module_obj = _safe_import(module_name)

    summary_hint = str(template.get("summary_hint") or "")
    summary = summary_hint
    entrypoints: List[str] = []
    if module_obj is not None:
        summary = _first_doc_line(inspect.getdoc(module_obj)) or summary_hint
        for symbol in template.get("entrypoints", ()):  # type: ignore[arg-type]
            if hasattr(module_obj, symbol):
                entrypoints.append(str(symbol))
    status = str(template.get("status") or "unknown").strip() or "unknown"

    entry: Dict[str, Any] = {
        "name": str(template.get("name")),
        "module": module_name,
        "status": status,
        "summary": summary,
        "responsibilities": list(template.get("responsibilities", ())),
        "fallback_capabilities": list(template.get("fallback_capabilities", ())),
    }
    if entrypoints:
        entry["entrypoints"] = entrypoints

    llm_hooks = [str(hook) for hook in template.get("llm_hooks", ()) if hook]
    if llm_hooks:
        entry["llm_hooks"] = llm_hooks

    if module_obj is None:
        entry["status"] = "unknown"
        entry.setdefault(
            "notes",
            "Import du module échoué lors de la construction baseline.",
        )

    return entry


def _safe_import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception:  # pragma: no cover - import best effort
        logger.debug("Impossible d'importer %s", module_name, exc_info=True)
        return None


def _first_doc_line(doc: Optional[str]) -> str:
    if not doc:
        return ""
    for line in doc.splitlines():
        cleaned = line.strip()
        if cleaned:
            return cleaned
    return ""


def _merge_snapshots(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> Dict[str, Any]:
    merged: MutableMapping[str, Any] = dict(baseline)

    for key, value in candidate.items():
        if value is None:
            continue
        if key == "summary" and isinstance(value, str) and value.strip():
            merged[key] = value.strip()
        elif (
            key in {"interfaces", "risks", "recommended_actions"}
            and isinstance(value, Iterable)
            and not isinstance(value, (str, bytes))
        ):
            merged[key] = list(value)
        elif key == "notes" and isinstance(value, str):
            merged[key] = value.strip()
        else:
            merged[key] = value

    if not merged.get("interfaces"):
        merged["interfaces"] = list(baseline.get("interfaces", []))

    return dict(merged)


def _summarise_context(context: Mapping[str, Any]) -> str:
    if not context:
        return ""

    hints: List[str] = []
    llm_flag = context.get("llm_enabled")
    if isinstance(llm_flag, bool):
        hints.append(
            "Confirmer l'état LLM (activé)" if llm_flag else "Réactiver LLM si pertinent"
        )

    pending = context.get("pending_interfaces")
    if isinstance(pending, Iterable) and not isinstance(pending, (str, bytes)):
        pending_list = [str(item) for item in pending if item]
        if pending_list:
            hints.append(
                "Prioriser la complétion pour : " + ", ".join(sorted(set(pending_list)))
            )

    return hints[0] if hints else ""


__all__ = ["describe_io_interfaces"]
