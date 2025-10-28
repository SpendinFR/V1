from __future__ import annotations

import logging
from typing import Any, Dict, List

from AGI_Evolutive.utils.llm_service import try_call_llm_dict

POSITIVE_MARKERS = {
    "merci": 0.25,
    "thanks": 0.25,
    "awesome": 0.25,
    "great": 0.25,
    "parfait": 0.25,
    "bravo": 0.25,
    "bien": 0.2,
    "excellent": 0.25,
    "top": 0.2,
    "génial": 0.25,
    "super": 0.25,
}

NEGATIVE_MARKERS = {
    "nul": -0.3,
    "bad": -0.3,
    "horrible": -0.3,
    "déçu": -0.25,
    "angry": -0.3,
    "non": -0.2,
    "pas clair": -0.25,
    "mauvais": -0.25,
    "bof": -0.2,
    "trop flou": -0.25,
    "à côté": -0.25,
}

LOGGER = logging.getLogger(__name__)
LLM_SPEC_KEY = "language_social_reward"


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _heuristic_reward(text: str) -> Dict[str, Any]:
    lowered = (text or "").lower()
    score = 0.0
    positive_hits: List[str] = []
    negative_hits: List[str] = []

    for marker, weight in POSITIVE_MARKERS.items():
        if marker in lowered:
            score += weight
            positive_hits.append(marker)

    for marker, weight in NEGATIVE_MARKERS.items():
        if marker in lowered:
            score += weight
            negative_hits.append(marker)

    score = max(-1.0, min(1.0, score))
    return {
        "reward": float(score),
        "hits": {"positive": positive_hits, "negative": negative_hits},
    }


def _merge_llm_response(heuristics: Dict[str, Any], response: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(heuristics)

    reward = _coerce_float(response.get("reward"))
    if reward is not None:
        merged["reward"] = max(-1.0, min(1.0, reward))

    label = response.get("label")
    if isinstance(label, str) and label.strip():
        merged["label"] = label.strip().lower()

    evidence = response.get("evidence")
    if isinstance(evidence, list):
        cleaned = [str(item).strip() for item in evidence if isinstance(item, str) and item.strip()]
        if cleaned:
            merged["evidence"] = cleaned

    notes = response.get("notes")
    if isinstance(notes, str) and notes.strip():
        merged["notes"] = notes.strip()

    confidence = _coerce_float(response.get("confidence"))
    if confidence is not None:
        merged["confidence"] = max(0.0, min(1.0, confidence))

    rationale = response.get("rationale")
    if isinstance(rationale, str) and rationale.strip():
        merged["rationale"] = rationale.strip()

    return merged


def extract_social_reward(text: str) -> Dict[str, Any]:
    """Évalue la valence sociale d'un texte en privilégiant l'LLM quand disponible."""

    sample = (text or "").strip()
    heuristics = _heuristic_reward(sample)
    payload = {
        "text": sample,
        "heuristics": heuristics,
        "positive_markers": list(POSITIVE_MARKERS.keys()),
        "negative_markers": list(NEGATIVE_MARKERS.keys()),
    }

    response = try_call_llm_dict(LLM_SPEC_KEY, input_payload=payload, logger=LOGGER)
    if not response:
        return heuristics

    if not isinstance(response, dict):
        return heuristics

    return _merge_llm_response(heuristics, response)
