"""Suppression douce des items expirés (sauf `pinned`). Safe si `delete_item` absent."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Iterable, List, Mapping

from AGI_Evolutive.utils.llm_service import try_call_llm_dict

LOGGER = logging.getLogger(__name__)


def _prepare_candidate(item: Mapping[str, Any]) -> Dict[str, Any]:
    text = str(item.get("text") or item.get("content") or "")
    if len(text) > 160:
        text = text[:157].rstrip() + "…"
    return {
        "id": str(item.get("id")),
        "kind": item.get("kind"),
        "tags": list(item.get("tags", [])) if isinstance(item.get("tags"), Iterable) else [],
        "salience": item.get("salience"),
        "ts": item.get("ts"),
        "text": text,
        "pinned": bool(item.get("pinned")),
        "record": item,
    }


def _apply_batch(memory_store, batch: List[Dict[str, Any]], deleted: int, marked: int) -> tuple[int, int]:
    if not batch:
        return deleted, marked
    payload = {
        "candidates": [
            {k: v for k, v in candidate.items() if k != "record"}
            for candidate in batch
            if candidate.get("id")
        ]
    }
    response = try_call_llm_dict(
        "memory_janitor_triage",
        input_payload=payload,
        logger=LOGGER,
    )
    decisions: Dict[str, str] = {}
    if response:
        for entry in response.get("decisions", []):
            if isinstance(entry, Mapping) and entry.get("id"):
                decisions[str(entry["id"])] = str(entry.get("action", ""))
    for candidate in batch:
        record = candidate["record"]
        item_id = candidate.get("id")
        if not item_id:
            continue
        decision = decisions.get(item_id, "")
        if decision == "soft_keep":
            continue
        hard_delete = decision == "delete"
        if hasattr(memory_store, "delete_item") and hard_delete:
            try:
                memory_store.delete_item(item_id)
                deleted += 1
                continue
            except Exception:
                LOGGER.debug("Hard delete failed for %s, falling back to soft delete", item_id)
        try:
            memory_store.update_item(item_id, {"deleted": True})
            marked += 1
        except Exception:
            LOGGER.debug("Soft delete failed for %s", item_id, exc_info=True)
    return deleted, marked


def run_once(memory_store) -> dict:
    now = time.time() if not hasattr(memory_store, "now") else memory_store.now()
    deleted = 0
    marked = 0
    batch: List[Dict[str, Any]] = []
    for item in memory_store.list_items({"newer_than_ts": 0, "limit": 5000}):
        exp = item.get("expiry_ts")
        if exp and exp < now and not item.get("pinned"):
            candidate = _prepare_candidate(item)
            batch.append(candidate)
            if len(batch) >= 25:
                deleted, marked = _apply_batch(memory_store, batch, deleted, marked)
                batch = []
    if batch:
        deleted, marked = _apply_batch(memory_store, batch, deleted, marked)
    return {"deleted": deleted, "marked": marked}
