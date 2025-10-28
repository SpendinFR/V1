from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from AGI_Evolutive.utils.llm_service import try_call_llm_dict

from .global_workspace import GlobalWorkspace


LOGGER = logging.getLogger(__name__)


class ConsciousnessEngine:
    """Maintains the conscious focus by consulting the global workspace."""

    def __init__(self, workspace: Optional[GlobalWorkspace] = None) -> None:
        self.gw = workspace or GlobalWorkspace()
        self._last_summary: Optional[Dict[str, Any]] = None

    def push(self, content: Any) -> None:
        """Broadcast raw content to the global workspace."""

        self.gw.broadcast(content)

    def describe_focus(self, *, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Return a description of the current conscious focus.

        The LLM provides the primary narrative; heuristics fall back to
        summarising the recent winners when the LLM is unavailable.
        """

        winners = self.gw.last_trace()
        payload = {
            "recent_bids": [
                {
                    "source": getattr(bid, "source", None),
                    "action_hint": getattr(bid, "action_hint", None),
                    "rationale": getattr(bid, "rationale", None),
                    "expected_info_gain": getattr(bid, "expected_info_gain", None),
                    "urgency": getattr(bid, "urgency", None),
                }
                for bid in winners
            ],
            "context": context or {},
        }
        response = try_call_llm_dict(
            "consciousness_engine",
            input_payload=payload,
            logger=LOGGER,
            max_retries=2,
        )
        if response:
            self._last_summary = dict(response)
            return self._last_summary

        fallback_focus = self._fallback_summary(winners)
        self._last_summary = fallback_focus
        return fallback_focus

    def last_summary(self) -> Optional[Dict[str, Any]]:
        """Return the most recent conscious focus summary."""

        return self._last_summary

    @staticmethod
    def _fallback_summary(winners: List[Any]) -> Dict[str, Any]:
        if not winners:
            return {"active_focus": None, "conflicts": [], "notes": "aucun bid actif"}
        primary = winners[0]
        focus = getattr(primary, "action_hint", None)
        conflicts: List[Dict[str, Any]] = []
        for bid in winners[1:3]:
            conflicts.append(
                {
                    "with": getattr(bid, "action_hint", None),
                    "impact": "modéré" if getattr(bid, "urgency", 0.0) > 0.4 else "faible",
                }
            )
        return {
            "active_focus": focus,
            "conflicts": conflicts,
            "notes": "Synthèse heuristique faute de réponse LLM.",
        }


__all__ = ["ConsciousnessEngine"]
