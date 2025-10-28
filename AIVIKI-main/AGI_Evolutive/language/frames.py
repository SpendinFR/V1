from dataclasses import dataclass, field
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum, auto
import logging

from AGI_Evolutive.utils.llm_service import try_call_llm_dict


LOGGER = logging.getLogger(__name__)
LLM_SPEC_KEY = "language_frames"


class DialogueAct(Enum):
    ASK = auto()
    INFORM = auto()
    REQUEST = auto()
    FEEDBACK_POS = auto()
    FEEDBACK_NEG = auto()
    GREET = auto()
    BYE = auto()
    THANKS = auto()
    META_HELP = auto()
    CLARIFY = auto()
    REFLECT = auto()


@dataclass
class UtteranceFrame:
    text: str
    normalized_text: str
    intent: str
    confidence: float
    uncertainty: float
    acts: List[DialogueAct] = field(default_factory=list)
    slots: Dict[str, Any] = field(default_factory=dict)
    unknown_terms: List[str] = field(default_factory=list)
    needs: List[str] = field(default_factory=list)      # ce dont l'IA a "besoin" pour bien répondre
    meta: Dict[str, Any] = field(default_factory=dict)  # ex: language, tone, user_profile hints

    @property
    def surface_form(self) -> str:
        """Compat: utilisé par ton ancien cycle."""
        return self.text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "normalized_text": self.normalized_text,
            "intent": self.intent,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "acts": [a.name for a in self.acts],
            "slots": self.slots,
            "unknown_terms": self.unknown_terms,
            "needs": self.needs,
            "meta": self.meta,
        }


def analyze_utterance(
    text: str,
    *,
    normalized_text: Optional[str] = None,
    intent_hint: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> UtteranceFrame:
    normalized = normalized_text or text.strip().lower()
    acts: List[DialogueAct] = []
    stripped = text.strip()
    if stripped.endswith("?"):
        acts.append(DialogueAct.ASK)
    if stripped.endswith("!"):
        acts.append(DialogueAct.FEEDBACK_POS)
    if any(token in normalized for token in ("merci", "thanks")):
        acts.append(DialogueAct.THANKS)
    if any(token in normalized for token in ("bonjour", "salut", "hello")):
        acts.append(DialogueAct.GREET)

    intent = intent_hint or "inform"
    if "aide" in normalized or "help" in normalized:
        intent = "help_request"
    elif "comment" in normalized or "how" in normalized:
        intent = "ask_info"

    confidence = 0.55
    uncertainty = 0.35 if acts and acts[0] == DialogueAct.ASK else 0.2

    frame = UtteranceFrame(
        text=text,
        normalized_text=normalized,
        intent=intent,
        confidence=confidence,
        uncertainty=uncertainty,
        acts=acts,
    )

    llm_update = _llm_enrich_frame(frame, context=context)
    if llm_update:
        _apply_llm_update(frame, llm_update)

    return frame


def _llm_enrich_frame(frame: UtteranceFrame, *, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    payload = {
        "text": frame.text,
        "normalized": frame.normalized_text,
        "intent_hint": frame.intent,
        "acts": [act.name for act in frame.acts],
        "context": context or {},
    }
    response = try_call_llm_dict(LLM_SPEC_KEY, input_payload=payload, logger=LOGGER)
    if not isinstance(response, dict):
        return {}
    return response


def _apply_llm_update(frame: UtteranceFrame, llm_payload: Dict[str, Any]) -> None:
    intent = llm_payload.get("intent")
    if isinstance(intent, str) and intent:
        frame.intent = intent

    confidence = llm_payload.get("confidence")
    if isinstance(confidence, (int, float)):
        frame.confidence = max(0.0, min(1.0, float(confidence)))

    uncertainty = llm_payload.get("uncertainty")
    if isinstance(uncertainty, (int, float)):
        frame.uncertainty = max(0.0, min(1.0, float(uncertainty)))

    acts = llm_payload.get("acts")
    if isinstance(acts, list):
        enriched: List[DialogueAct] = []
        for act in acts:
            if isinstance(act, DialogueAct):
                enriched.append(act)
            elif isinstance(act, str):
                try:
                    enriched.append(DialogueAct[act])
                except KeyError:
                    continue
        if enriched:
            frame.acts = enriched

    slots = llm_payload.get("slots")
    if isinstance(slots, dict):
        frame.slots.update(slots)

    unknown_terms = llm_payload.get("unknown_terms")
    if isinstance(unknown_terms, list):
        frame.unknown_terms = [str(term) for term in unknown_terms if term]

    needs = llm_payload.get("needs")
    if isinstance(needs, list):
        frame.needs = [str(need) for need in needs if need]

    meta = llm_payload.get("meta")
    if isinstance(meta, dict):
        frame.meta.update(meta)

    notes = llm_payload.get("notes")
    if notes:
        frame.meta.setdefault("llm_notes", str(notes))


__all__ = ["DialogueAct", "UtteranceFrame", "analyze_utterance"]
