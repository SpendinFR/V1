"""Top-level helpers for the :mod:`AGI_Evolutive` package.

In addition to exposing the package namespace, this module now offers a
lightweight *package overview* helper.  The function gathers structural
statistics about the repository and, when available, enriches them with a
structured summary produced by the local LLM integration.  When the LLM layer
is disabled or returns an invalid payload, the function falls back to a
deterministic heuristic profile so callers always receive a consistent shape.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Mapping, MutableMapping, Sequence, Tuple

from AGI_Evolutive.utils.llm_service import try_call_llm_dict


LOGGER = logging.getLogger(__name__)
_PACKAGE_NAME = "AGI_Evolutive"
_PACKAGE_ROOT = Path(__file__).resolve().parent


def package_overview(*, extra_notes: Sequence[str] | None = None) -> MutableMapping[str, object]:
    """Return a structured overview of the package, enriched by the LLM.

    Parameters
    ----------
    extra_notes:
        Optional list of additional strings to share with the LLM.  The
        heuristics ignore this argument but it is accepted to keep the public
        API stable should future versions decide to surface the hints.

    Returns
    -------
    dict
        A mapping containing the original structural statistics, an optional
        LLM synthesis, and deterministic fallback fields ensuring that the
        return type is stable regardless of the integration status.
    """

    stats = _collect_package_stats()
    llm_payload = try_call_llm_dict(
        "package_overview",
        input_payload={"stats": stats, "extra_notes": list(extra_notes or ())},
        logger=LOGGER,
    )

    if isinstance(llm_payload, Mapping):
        normalized = _normalise_llm_overview(llm_payload, stats)
        if normalized is not None:
            return normalized

    return _fallback_overview(stats)


def _collect_package_stats() -> MutableMapping[str, object]:
    modules: List[str] = []

    stack: List[Tuple[Path, str]] = [(_PACKAGE_ROOT, _PACKAGE_NAME)]
    while stack:
        current_path, current_package = stack.pop()

        if current_path != _PACKAGE_ROOT and (current_path / "__init__.py").exists():
            modules.append(current_package)

        for child in sorted(current_path.iterdir(), key=lambda path: path.name):
            if child.name == "__pycache__":
                continue
            if child.is_file() and child.suffix == ".py" and child.name != "__init__.py":
                modules.append(f"{current_package}.{child.stem}")
            elif child.is_dir() and (child / "__init__.py").exists():
                stack.append((child, f"{current_package}.{child.name}"))

    subpackages: List[MutableMapping[str, object]] = []
    for child in sorted(_PACKAGE_ROOT.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "__init__.py").exists():
            continue
        python_count = sum(1 for _ in child.rglob("*.py"))
        subpackages.append(
            {
                "name": f"{_PACKAGE_NAME}.{child.name}",
                "python_files": python_count,
            }
        )

    return {
        "package": _PACKAGE_NAME,
        "module_count": len(modules),
        "docstring": (__doc__ or "").strip(),
        "subpackages": subpackages,
    }


def _normalise_llm_overview(
    payload: Mapping[str, object],
    stats: Mapping[str, object],
) -> MutableMapping[str, object] | None:
    summary = _clean_string(payload.get("summary") or payload.get("overview"))
    capabilities = _clean_string_list(payload.get("capabilities"))
    focus = _clean_string_list(payload.get("recommended_focus") or payload.get("next_steps"))
    alerts = _clean_string_list(payload.get("alerts") or payload.get("warnings"))
    notes = _clean_string(payload.get("notes"))

    confidence = _safe_confidence(payload.get("confidence"))

    fallback = _fallback_overview(stats)

    if not summary:
        summary = fallback["summary"]  # type: ignore[index]
    if not capabilities:
        capabilities = fallback["capabilities"]  # type: ignore[index]
    if not focus:
        focus = fallback["recommended_focus"]  # type: ignore[index]
    if not alerts:
        alerts = []
    if not notes:
        notes = fallback["notes"]  # type: ignore[index]
    if confidence <= 0.0:
        confidence = max(0.55, float(fallback.get("confidence", 0.5)))

    overview: MutableMapping[str, object] = {
        "source": "llm",
        "summary": summary,
        "capabilities": capabilities,
        "recommended_focus": focus,
        "alerts": alerts,
        "confidence": confidence,
        "notes": notes,
        "stats": stats,
    }
    return overview


def _fallback_overview(stats: Mapping[str, object]) -> MutableMapping[str, object]:
    summary = (
        "AGI_Evolutive regroupe les briques de perception, cognition, mémoire et "
        "auto-amélioration au sein d'un même noyau expérimental."
    )
    capabilities = [
        "Orchestration multi-domaines (perception, cognition, mémoire, autonomie)",
        "Heuristiques déterministes prêtes à l'emploi pour fonctionner sans LLM",
        "Infrastructure d'intégrations LLM via des specs structurées",
    ]
    recommended_focus = [
        "Prioriser l'harmonisation des sorties LLM entre sous-systèmes",
        "Renforcer l'observabilité des pipelines heuristiques critiques",
    ]

    return {
        "source": "heuristic",
        "summary": summary,
        "capabilities": capabilities,
        "recommended_focus": recommended_focus,
        "alerts": [],
        "confidence": 0.35,
        "notes": "Profil généré sans assistance LLM (fallback heuristique).",
        "stats": stats,
    }


def _clean_string(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _clean_string_list(value: object) -> List[str]:
    if isinstance(value, str):
        value = [value]
    if isinstance(value, Sequence):
        cleaned = [str(item).strip() for item in value if isinstance(item, str) and item.strip()]
        return cleaned
    return []


def _safe_confidence(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))


__all__ = ["package_overview"]
