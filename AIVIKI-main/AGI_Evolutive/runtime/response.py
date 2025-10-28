from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

import logging
import re

from AGI_Evolutive.utils.llm_service import (
    LLMIntegrationError,
    LLMUnavailableError,
    get_llm_manager,
    is_llm_enabled,
)

LOGGER = logging.getLogger(__name__)


def _llm_enabled() -> bool:
    return is_llm_enabled()


def _llm_manager():
    return get_llm_manager()


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if isinstance(value, str):
            candidate = value.replace(",", ".").strip()
            return float(candidate)
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _extract_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    text = str(value).strip()
    return text or None


def _split_reasoning_bullets(text: str) -> List[str]:
    if not text:
        return []
    items: List[str] = []
    for chunk in text.split("•"):
        cleaned = chunk.strip().strip("- ")
        if cleaned:
            items.append(cleaned)
    return items


def _split_reasoning_segments(text: str) -> List[str]:
    segments: List[str] = []
    for raw_line in (text or "").replace("\r\n", "\n").split("\n"):
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        parts = re.split(r"\s*\|\s*", raw_line)
        for part in parts:
            piece = part.strip()
            if not piece:
                continue
            if " - 🧩" in piece and not piece.strip().startswith("- 🧩"):
                prefix, rest = piece.split("- 🧩", 1)
                if prefix.strip():
                    segments.append(prefix.strip())
                segments.append("- 🧩" + rest.strip())
            else:
                segments.append(piece)

    expanded: List[str] = []
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        if " - 🧩" in seg and not seg.startswith("- 🧩"):
            prefix, rest = seg.split("- 🧩", 1)
            if prefix.strip():
                expanded.append(prefix.strip())
            expanded.append("- 🧩" + rest.strip())
        else:
            expanded.append(seg)

    normalized: List[str] = []
    for seg in expanded:
        seg = seg.strip()
        if not seg:
            continue
        if "•" in seg and not seg.startswith("•"):
            if ":" in seg:
                head, tail = seg.split(":", 1)
                normalized.append(head.strip() + ":")
                for item in _split_reasoning_bullets(tail):
                    normalized.append("• " + item)
            else:
                for item in _split_reasoning_bullets(seg):
                    normalized.append("• " + item)
        else:
            normalized.append(seg)
    return normalized


def humanize_reasoning_block(text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    if not text:
        return text, None

    sentinel_tokens = ("Hypothèse prise", "Incertitude", "Ce que j'apprends", "🔧", "❓")
    if not any(token in text for token in sentinel_tokens):
        llm_result = _llm_rewrite_reasoning(text)
        if llm_result:
            return llm_result
        return text, None

    lines = _split_reasoning_segments(text)
    summary: Optional[str] = None
    hypothesis: Optional[str] = None
    incertitude: Optional[float] = None
    next_test: Optional[str] = None
    caution_notes: List[str] = []
    needs: List[str] = []
    learnings: List[str] = []
    questions: List[str] = []
    current_section: Optional[str] = None

    for line in lines:
        if not line:
            continue
        if line.startswith("Reçu:") or line.startswith("⏱️") or line.startswith("🎯"):
            continue
        if line.startswith("🧠"):
            summary = line.lstrip("🧠").strip()
            continue
        if line.startswith("⚠️"):
            caution_notes.append(line.lstrip("⚠️").strip())
            current_section = None
            continue
        if "Hypothèse prise" in line:
            try:
                payload = line.split("Hypothèse prise", 1)[1]
                if ":" in payload:
                    payload = payload.split(":", 1)[1]
                hypothesis = payload.strip()
            except Exception:
                hypothesis = line
            current_section = None
            continue
        if "Incertitude" in line:
            match = re.search(r"Incertitude\s*:\s*([0-9]+[,.]?[0-9]*)", line)
            if match:
                try:
                    incertitude = float(match.group(1).replace(",", "."))
                except ValueError:
                    incertitude = None
            current_section = None
            continue
        if line.startswith("🧪"):
            try:
                next_test = line.split(":", 1)[1].strip()
            except IndexError:
                next_test = line.lstrip("🧪").strip()
            current_section = None
            continue
        if line.startswith("📗"):
            current_section = "learn"
            rest = line.split(":", 1)[1].strip() if ":" in line else ""
            for item in _split_reasoning_bullets(rest):
                learnings.append(item)
            continue
        if line.startswith("🔧"):
            current_section = "need"
            rest = line.split(":", 1)[1].strip() if ":" in line else ""
            for item in _split_reasoning_bullets(rest):
                needs.append(item)
            continue
        if line.startswith("•"):
            item = line.lstrip("• ").strip()
            if not item:
                continue
            if current_section == "need":
                needs.append(item)
            elif current_section == "learn":
                learnings.append(item)
            continue
        if line.startswith("❓"):
            question = line.lstrip("❓").strip()
            if question:
                questions.append(question)
            current_section = None
            continue
        current_section = None

    if not (summary or hypothesis or incertitude is not None or next_test or needs or questions or caution_notes):
        return text, None

    sentences: List[str] = []
    if hypothesis:
        sentences.append(f"Je pars de l'hypothèse que {hypothesis}.")
    if summary:
        cleaned_summary = summary.rstrip(".")
        sentences.append(f"Synthèse : {cleaned_summary}.")
    if incertitude is not None:
        confidence = max(0.0, min(1.0, 1.0 - incertitude))
        confidence_pct = int(round(confidence * 100))
        sentences.append(f"Niveau de confiance approximatif : {confidence_pct}\xa0%.")
    for note in caution_notes:
        note_clean = note.rstrip(".")
        sentences.append(f"Attention : {note_clean}.")
    if next_test:
        sentences.append(f"Prochain test envisagé : {next_test}.")
    if needs:
        sentences.append("Pour avancer, j'aurais besoin de : " + "; ".join(needs) + ".")
    if questions:
        formatted = []
        for question in questions:
            stripped = question.rstrip(" ?")
            formatted.append(stripped + " ?")
        sentences.append("Peux-tu préciser : " + " ; ".join(formatted))

    normalized_text = " ".join(sentences).strip()
    if not normalized_text:
        return text, None

    diagnostics = {
        "summary": summary,
        "hypothesis": hypothesis,
        "incertitude": incertitude,
        "needs": needs,
        "questions": questions,
    }
    return normalized_text, diagnostics


def _llm_rewrite_reasoning(text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    if not text or not _llm_enabled():
        return None
    try:
        response = _llm_manager().call_dict("response_formatter", input_payload={"reasoning": text})
    except (LLMUnavailableError, LLMIntegrationError):
        LOGGER.debug("LLM response formatter unavailable", exc_info=True)
        return None
    if not isinstance(response, Mapping):
        return None

    hypothesis = _extract_str(response.get("hypothese") or response.get("hypothesis"))
    incertitude_value = response.get("incertitude")
    incertitude = _coerce_float(incertitude_value)
    incertitude_note = None if incertitude is not None else _extract_str(incertitude_value)
    needs = _coerce_list(response.get("besoins"))
    questions = _coerce_list(response.get("questions"))
    summary = _extract_str(response.get("summary") or response.get("notes"))
    notes = _extract_str(response.get("notes"))

    sentences: List[str] = []
    if hypothesis:
        hypothesis_clean = hypothesis.rstrip(".")
        sentences.append(f"Hypothèse : {hypothesis_clean}.")
    if summary:
        summary_clean = summary.rstrip(".")
        sentences.append(f"Synthèse : {summary_clean}.")
    if incertitude is not None:
        confidence = max(0.0, min(1.0, 1.0 - incertitude))
        sentences.append(f"Confiance estimée : {int(round(confidence * 100))}%.")
    elif incertitude_note:
        sentences.append(f"Incertitude : {incertitude_note}.")
    if needs:
        sentences.append("Besoins : " + "; ".join(needs) + ".")
    if questions:
        formatted = [item.rstrip(" ?") + " ?" for item in questions]
        sentences.append("Questions : " + " ; ".join(formatted))
    if notes and notes not in (summary or ""):
        sentences.append(f"Notes : {notes}.")

    normalized_text = " ".join(sentences).strip()
    if not normalized_text:
        normalized_text = summary or text

    diagnostics: Dict[str, Any] = {
        "hypothesis": hypothesis,
        "incertitude": incertitude,
        "needs": needs,
        "questions": questions,
        "notes": notes,
    }
    if incertitude is None and incertitude_note:
        diagnostics["incertitude_text"] = incertitude_note

    return normalized_text, diagnostics

CONTRACT_KEYS = [
    "hypothese_choisie",
    "incertitude",
    "prochain_test",
    "appris",
    "besoin",
]

CONTRACT_DEFAULTS: Dict[str, Any] = {
    "hypothese_choisie": "clarifier l'intention et la granularité attendue",
    "incertitude": 0.5,
    "prochain_test": "proposer 2 chemins d'action et demander ton choix",
    "appris": ["prioriser le concret et la traçabilité"],
    "besoin": ["confirmer si tu préfères plan en étapes ou patch direct"],
}


def _stringify_list(items: Optional[List[str]]) -> str:
    if not items:
        return "-"
    return "\n".join([f"• {x}" for x in items])


def _ensure_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, (set, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def ensure_contract(contract: Dict[str, Any]) -> Dict[str, Any]:
    """Merge defaults with provided values and normalise the payload."""

    normalised: Dict[str, Any] = dict(CONTRACT_DEFAULTS)
    normalised.update(contract or {})

    normalised["appris"] = _ensure_list(normalised.get("appris"))
    normalised["besoin"] = _ensure_list(normalised.get("besoin"))

    try:
        incertitude = float(normalised.get("incertitude", 0.5))
    except (TypeError, ValueError):
        incertitude = 0.5
    normalised["incertitude"] = max(0.0, min(1.0, incertitude))

    if not normalised.get("prochain_test"):
        normalised["prochain_test"] = "-"

    return normalised


def format_agent_reply(base_text: str, **contract: Any) -> str:
    """Formats an agent reply mixing the base text with the social contract."""

    enriched = ensure_contract(contract)

    learned = _stringify_list(enriched.get("appris"))
    needs = _stringify_list(enriched.get("besoin"))
    test_line = enriched.get("prochain_test") or "-"

    return (
        f"{base_text}\n\n"
        f"-\n"
        f"🧩 Hypothèse prise: {enriched['hypothese_choisie']}\n"
        f"🤔 Incertitude: {enriched['incertitude']:.2f}\n"
        f"🧪 Prochain test: {test_line}\n"
        f"📗 Ce que j'apprends: \n{learned}\n"
        f"🔧 Besoins: \n{needs}"
    )
