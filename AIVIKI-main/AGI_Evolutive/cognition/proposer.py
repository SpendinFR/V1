from typing import Any, Dict, List, Mapping
import logging

from AGI_Evolutive.utils.llm_service import try_call_llm_dict


logger = logging.getLogger(__name__)


class Proposer:
    """Generate proposals for self-model updates based on drives and memory."""

    def __init__(self, memory_store, planner, homeostasis) -> None:
        self.memory = memory_store
        self.planner = planner
        self.homeo = homeostasis

    def run_once_now(self) -> List[Dict[str, Any]]:
        proposals: List[Dict[str, Any]] = []
        drives = self.homeo.state["drives"]
        recent = self.memory.get_recent_memories(n=50)
        error_count = sum(1 for memo in recent if memo.get("kind") == "error")
        if drives["curiosity"] > 0.55 and error_count >= 3:
            proposals.append({"type": "update", "path": ["persona", "tone"], "value": "inquisitive-analytical"})

        if drives["social_bonding"] > 0.55:
            proposals.append(
                {
                    "type": "add",
                    "path": ["persona", "values"],
                    "value": ["growth", "truth", "help", "empathy"],
                }
            )

        payload = {
            "drives": drives,
            "recent_errors": error_count,
            "recent_memories": [memo for memo in recent[:10] if isinstance(memo, Mapping)],
            "fallback_proposals": proposals,
        }

        llm_result = try_call_llm_dict(
            "cognition_proposer",
            input_payload=payload,
            logger=logger,
        )

        if isinstance(llm_result, Mapping):
            suggestions = llm_result.get("suggestions")
            if isinstance(suggestions, list):
                enriched: List[Dict[str, Any]] = []
                for item in suggestions:
                    if not isinstance(item, Mapping):
                        continue
                    suggestion = {
                        "type": item.get("type", "update"),
                        "path": item.get("path") or item.get("target") or [],
                        "value": item.get("value") or item.get("adjustment"),
                        "cause": item.get("cause"),
                    }
                    enriched.append(suggestion)
                if enriched:
                    proposals = enriched
            notes = llm_result.get("notes")
            if isinstance(notes, str) and notes.strip():
                proposals.append({"type": "note", "value": notes.strip()})

        return proposals
