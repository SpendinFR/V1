from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Mapping
import logging
import time

from AGI_Evolutive.utils.llm_contracts import enforce_llm_contract
from AGI_Evolutive.utils.llm_service import try_call_llm_dict


logger = logging.getLogger(__name__)


@dataclass
class DialogueState:
    conversation_id: str = "default"
    turn_index: int = 0
    last_speaker: str = "user"
    user_profile: Dict[str, Any] = field(default_factory=dict)
    known_entities: Dict[str, Any] = field(default_factory=dict)
    pending_questions: List[str] = field(default_factory=list)
    recent_frames: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    analysis: Dict[str, Any] = field(default_factory=dict)

    def update_with_frame(self, frame: Dict[str, Any]):
        self.turn_index += 1
        self.last_speaker = "user"
        self.recent_frames.append(frame)
        if len(self.recent_frames) > 20:
            self.recent_frames = self.recent_frames[-20:]

        # Retenir éventuels nouveaux référents
        for k, v in frame.get("slots", {}).items():
            if isinstance(v, str) and len(v) <= 128:
                self.known_entities[k] = v

        self.refresh_summary(extra_context={"reason": "new_frame"})

    def add_pending_question(self, q: str):
        if q and q not in self.pending_questions:
            self.pending_questions.append(q)
            if len(self.pending_questions) > 5:
                self.pending_questions = self.pending_questions[-5:]
            self.refresh_summary(extra_context={"reason": "pending_question"})

    def consume_pending_questions(self, max_q: int = 2) -> List[str]:
        qs = self.pending_questions[:max_q]
        self.pending_questions = self.pending_questions[max_q:]
        return qs

    def remember_unknown_term(self, term: str):
        if "unknown_terms" not in self.user_profile:
            self.user_profile["unknown_terms"] = []
        if term not in self.user_profile["unknown_terms"]:
            self.user_profile["unknown_terms"].append(term)
            if len(self.user_profile["unknown_terms"]) > 50:
                self.user_profile["unknown_terms"] = self.user_profile["unknown_terms"][-50:]
            self.refresh_summary(extra_context={"reason": "unknown_term"})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "turn_index": self.turn_index,
            "last_speaker": self.last_speaker,
            "user_profile": self.user_profile,
            "known_entities": self.known_entities,
            "pending_questions": list(self.pending_questions),
            "recent_frames": list(self.recent_frames),
            "created_at": self.created_at,
            "analysis": dict(self.analysis),
        }

    def _fallback_summary(self) -> Dict[str, Any]:
        frames = [frame for frame in self.recent_frames[-3:] if isinstance(frame, Mapping)]
        focus_bits: List[str] = []
        for frame in frames:
            for key in ("summary", "text", "utterance", "intent"):
                value = frame.get(key)
                if isinstance(value, str) and value:
                    focus_bits.append(value)
                    break
        summary_text = " / ".join(focus_bits) if focus_bits else "Conversation en cours"

        commitments_raw = self.user_profile.get("commitments") if isinstance(self.user_profile, Mapping) else None
        commitments: List[Dict[str, Any]] = []
        if isinstance(commitments_raw, list):
            for item in commitments_raw:
                if not isinstance(item, Mapping):
                    continue
                commitment = str(item.get("commitment") or item.get("label") or "").strip()
                if not commitment:
                    continue
                entry: Dict[str, Any] = {"commitment": commitment}
                deadline = item.get("deadline") or item.get("due")
                if isinstance(deadline, str) and deadline:
                    entry["deadline"] = deadline
                commitments.append(entry)

        return {
            "state_summary": summary_text,
            "open_commitments": commitments,
            "pending_questions": list(self.pending_questions),
            "notes": f"{len(self.known_entities)} entités suivies",
        }

    def refresh_summary(self, extra_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {
            "conversation_id": self.conversation_id,
            "turn_index": self.turn_index,
            "last_speaker": self.last_speaker,
            "user_profile": self.user_profile,
            "known_entities": self.known_entities,
            "pending_questions": self.pending_questions,
            "recent_frames": self.recent_frames[-5:],
            "context": extra_context or {},
        }

        llm_result = try_call_llm_dict(
            "dialogue_state",
            input_payload=payload,
            logger=logger,
        )

        cleaned = enforce_llm_contract("dialogue_state", llm_result)
        if cleaned is not None:
            llm_result = cleaned

        summary: Dict[str, Any]
        if isinstance(llm_result, Mapping):
            summary = {
                "state_summary": llm_result.get("state_summary") or llm_result.get("summary") or "",
                "open_commitments": llm_result.get("open_commitments") or [],
                "pending_questions": llm_result.get("pending_questions") or self.pending_questions,
            }
            notes = llm_result.get("notes")
            if isinstance(notes, str) and notes.strip():
                summary["notes"] = notes.strip()
        else:
            summary = self._fallback_summary()

        summary["timestamp"] = time.time()
        self.analysis = summary
        return summary
