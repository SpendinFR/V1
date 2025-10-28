"""Gestion légère des préférences likes/dislikes."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Iterable

from AGI_Evolutive.utils.llm_service import try_call_llm_dict

LOGGER = logging.getLogger(__name__)


def _norm_concept(value: str) -> str:
    return value.strip().lower()


class PrefsBridge:
    """Agrège feedback préférentiel (JSON + graphe facultatif)."""

    def __init__(self, path: str = "data/preferences.json", graph: Any = None) -> None:
        self.path = path
        self.graph = graph
        self._mem: Dict[str, Dict[str, float]] = {"likes": {}, "dislikes": {}}
        self._last_guidance: Dict[str, Any] = {}
        self._load()

    # ------------------------------------------------------------------
    def get_affinity(self, concepts: Iterable[str], tags: Iterable[str]) -> float:
        likes = 0.0
        dislikes = 0.0
        for raw in list(concepts) + list(tags):
            if not raw:
                continue
            it = _norm_concept(str(raw))
            likes += self._score("likes", it)
            dislikes += self._score("dislikes", it)
        # map -> [0,1]
        total = likes + dislikes
        base_affinity = 0.0 if total <= 0 else max(0.0, min(1.0, likes / total))
        payload = {
            "concepts": [str(c) for c in concepts],
            "tags": [str(tag) for tag in tags],
            "base_affinity": base_affinity,
        }
        response = try_call_llm_dict(
            "memory_preferences_guidance",
            input_payload=payload,
            logger=LOGGER,
        )
        if response:
            try:
                adjusted = float(response.get("adjusted_affinity", base_affinity))
                base_affinity = max(0.0, min(1.0, adjusted))
            except Exception:
                pass
            self._last_guidance = dict(response)
        else:
            self._last_guidance = {}
        return base_affinity

    def last_guidance(self) -> Dict[str, Any]:
        return dict(self._last_guidance)

    def observe_feedback(
        self,
        concept: str,
        *,
        sign: int,
        evidence_id: str,
        strength: float = 1.0,
    ) -> None:
        concept = _norm_concept(concept)
        strength = max(0.1, min(2.0, float(strength)))
        if self.graph and hasattr(self.graph, "add_evidence"):
            # duck-typed: subject="Agent", predicate="likes|dislikes", object=concept
            try:
                pred = "likes" if sign >= 0 else "dislikes"
                self.graph.add_evidence(
                    "Agent",
                    pred,
                    concept,
                    weight=strength,
                    source_id=evidence_id,
                    ts=time.time(),
                )
                return
            except Exception:
                pass
        # fallback JSON
        bucket = "likes" if sign >= 0 else "dislikes"
        self._mem.setdefault(bucket, {})
        self._mem[bucket][concept] = self._mem[bucket].get(concept, 0.0) + float(strength)
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._mem, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # --- internes ---
    def _score(self, bucket: str, key: str) -> float:
        key = key.strip().lower()
        # graph path
        if self.graph and hasattr(self.graph, "edge_weight"):
            try:
                w = float(self.graph.edge_weight("Agent", bucket.rstrip("s"), key))  # likes/dislikes
                if w > 0:
                    return w
            except Exception:
                pass
        # fallback JSON
        return float(self._mem.get(bucket, {}).get(key, 0.0))

    def _load(self) -> None:
        if not self.path:
            return
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                likes = payload.get("likes", {})
                dislikes = payload.get("dislikes", {})
                if isinstance(likes, dict):
                    self._mem["likes"] = {str(k).strip().lower(): float(v) for k, v in likes.items()}
                if isinstance(dislikes, dict):
                    self._mem["dislikes"] = {str(k).strip().lower(): float(v) for k, v in dislikes.items()}
        except Exception:
            pass
