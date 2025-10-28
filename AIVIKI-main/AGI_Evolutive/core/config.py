import json
import logging
import os
from functools import lru_cache
from typing import Any, Dict, Mapping, MutableMapping, Optional

from AGI_Evolutive.utils.jsonsafe import json_sanitize
from AGI_Evolutive.utils.llm_service import try_call_llm_dict

_DEFAULTS: Dict[str, Any] = {
    "version": 1,
    "name": "agi_evolutive",
    "DATA_DIR": "data",
    "MEM_DIR": "data/memories",
    "PLANS_PATH": "data/plans.json",
    "SELF_PATH": "data/self_model.json",
    "SELF_VERSIONS_DIR": "data/self_model_versions",
    "HOMEOSTASIS_PATH": "data/homeostasis.json",
    "VECTOR_DIR": "data/vector_store",
    "LOGS_DIR": "logs",
    "GOALS_DAG_PATH": "logs/goals_dag.json",
    "PRIMARY_USER_NAME": "William",
    "PRIMARY_USER_ROLE": "creator",
    "MEMORY_SHARING_TRUSTED_NAMES": ["William"],
    "MEMORY_SHARING_ROLES_BY_NAME": {},
    "MEMORY_SHARING_TRUSTED_ROLES": ["creator", "owner"],
    "MEMORY_SHARING_MAX_ITEMS": 5,
}

_DIR_KEYS = ("DATA_DIR", "MEM_DIR", "SELF_VERSIONS_DIR", "VECTOR_DIR", "LOGS_DIR")

_cfg: Optional[Dict[str, Any]] = None
LOGGER = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_config(path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from *path*, applying defaults and creating directories."""
    global _cfg

    if os.path.isabs(path):
        candidate_paths = [path]
    else:
        here = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        candidate_paths = [os.path.join(here, path), path]

    cfg = dict(_DEFAULTS)
    for candidate in candidate_paths:
        if os.path.exists(candidate):
            try:
                with open(candidate, "r", encoding="utf-8") as fh:
                    cfg.update(json.load(fh))
                break
            except Exception:
                continue

    for key in _DIR_KEYS:
        os.makedirs(cfg[key], exist_ok=True)

    _cfg = cfg
    return _cfg


def cfg() -> Dict[str, Any]:
    """Return the cached configuration, loading it if required."""
    global _cfg
    return _cfg or load_config()


def summarize_config(config: Mapping[str, Any] | None = None) -> MutableMapping[str, Any]:
    """Return a structured summary of the configuration, enriched by the LLM."""

    current = dict(config or cfg())
    heuristic = _fallback_summary(current)

    payload = {
        "config": {key: current.get(key) for key in sorted(current.keys())[:40]},
        "custom_keys": sorted(set(current.keys()) - set(_DEFAULTS.keys())),
        "directory_keys": {key: current.get(key) for key in _DIR_KEYS},
    }
    response = try_call_llm_dict(
        "config_profile",
        input_payload=json_sanitize(payload),
        logger=LOGGER,
        max_retries=2,
    )

    if isinstance(response, Mapping):
        summary = _clean_text(response.get("summary")) or heuristic["summary"]
        alerts = _clean_list(response.get("alerts"))
        recommendations = _clean_list(response.get("recommended_actions"))
        notes = _clean_text(response.get("notes")) or heuristic["notes"]
        confidence = heuristic.get("confidence", 0.6)
        try:
            confidence = float(response.get("confidence", confidence))
        except Exception:
            confidence = float(confidence)

        enriched = dict(heuristic)
        enriched.update(
            {
                "source": "llm",
                "summary": summary,
                "alerts": alerts,
                "recommended_actions": recommendations or heuristic["recommended_actions"],
                "confidence": max(0.0, min(1.0, confidence)),
                "notes": notes,
            }
        )
        return enriched

    return heuristic


def _fallback_summary(config: Mapping[str, Any]) -> MutableMapping[str, Any]:
    overrides = sorted(set(config.keys()) - set(_DEFAULTS.keys()))
    missing_dirs = [key for key in _DIR_KEYS if not os.path.exists(str(config.get(key, "")))]
    alerts = [f"Répertoire absent: {key}" for key in missing_dirs]
    summary = "Configuration standard appliquée."
    if overrides:
        summary = "Configuration personnalisée détectée."
    if missing_dirs:
        summary = "Configuration incomplète : répertoires manquants."
    recommendations = []
    if missing_dirs:
        recommendations.append("Créer les répertoires manquants ou ajuster les chemins.")
    if not overrides:
        recommendations.append("Documenter les surcharges futures pour traçabilité.")
    else:
        recommendations.append("Vérifier la cohérence des valeurs personnalisées.")

    return {
        "source": "heuristic",
        "summary": summary,
        "alerts": alerts,
        "recommended_actions": recommendations,
        "config_snapshot": {key: config.get(key) for key in list(_DEFAULTS.keys())[:6]},
        "overrides": overrides,
        "confidence": 0.6 if missing_dirs else 0.72,
        "notes": "Synthèse calculée sans LLM à partir des chemins et surcharges.",
    }


def _clean_text(value: Any) -> str:
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned:
            return cleaned
    return ""


def _clean_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    result: list[str] = []
    for item in value:
        text = _clean_text(item)
        if text:
            result.append(text)
    return result
