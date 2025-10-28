import hashlib
import logging
import os
from typing import Any, Dict, Iterable, List, Optional

from . import DATA_DIR, _json_load, _json_save
from AGI_Evolutive.utils.llm_service import try_call_llm_dict

CACHE = os.path.join(DATA_DIR, "inbox_cache.json")
LLM_SPEC_KEY = "language_inbox_ingest"
LOGGER = logging.getLogger(__name__)


def _hash_path(p: str) -> str:
    return hashlib.sha1(p.encode("utf-8", errors="ignore")).hexdigest()


def ingest_inbox_paths(paths: Iterable[str], *, arch) -> int:
    """
    Ingère des fichiers texte depuis l'inbox :
      - alimente le Lexicon
      - observe le style
      - ajoute des citations à la QuoteMemory

    Un cache persistant empêche la ré-ingestion inutile.
    """

    cache = _json_load(CACHE, {})
    added = 0
    qm = getattr(arch, "quote_memory", None) or getattr(
        getattr(arch, "voice_profile", None), "quote_memory", None
    )
    lex = getattr(arch, "lexicon", None)
    style_obs = getattr(arch, "style_observer", None)

    for p in paths:
        if not p or not os.path.isfile(p):
            continue
        h = _hash_path(p)
        if cache.get(h):
            continue
        try:
            entries: List[tuple[int, str]] = []
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                for idx, line in enumerate(f, start=1):
                    text = (line or "").strip()
                    if not text:
                        continue
                    entries.append((idx, text))

            decisions = _llm_filter_entries(p, entries, arch=arch)

            if decisions is None:
                for _, text in entries:
                    if lex:
                        try:
                            lex.add_from_text(text, liked=False)
                        except Exception:
                            pass
                    if style_obs:
                        try:
                            if hasattr(style_obs, "observe_text"):
                                style_obs.observe_text(
                                    text,
                                    source=f"inbox:{p}",
                                    channel="inbox",
                                )
                            else:
                                style_obs.observe(text)
                        except Exception:
                            pass
                if qm:
                    try:
                        qm.ingest_file_units(p, liked=True)
                    except Exception:
                        pass
            else:
                for entry_id, (lineno, text) in enumerate(entries):
                    decision = decisions.get(entry_id)
                    if not decision or not decision.get("accept", False):
                        continue
                    targets = decision.get("targets") or ["lexicon", "style", "quote"]
                    liked = bool(decision.get("liked", False))
                    tags = decision.get("tags") or []
                    channel = decision.get("channel", "inbox")
                    if lex and "lexicon" in targets:
                        try:
                            lex.add_from_text(text, liked=liked)
                        except Exception:
                            pass
                    if style_obs and "style" in targets:
                        try:
                            if hasattr(style_obs, "observe_text"):
                                style_obs.observe_text(
                                    text,
                                    source=f"inbox:{p}#L{lineno}",
                                    channel=channel,
                                )
                            else:
                                style_obs.observe(text)
                        except Exception:
                            pass
                    if qm and "quote" in targets:
                        try:
                            qm.ingest(
                                text,
                                source=f"inbox:{p}#L{lineno}",
                                liked=liked,
                                tags=tags,
                            )
                        except Exception:
                            pass
            cache[h] = {"path": p}
            added += 1
        except Exception:
            continue

    _json_save(CACHE, cache)
    if qm:
        try:
            qm.save()
        except Exception:
            pass
    return added


def _llm_filter_entries(
    path: str,
    entries: List[tuple[int, str]],
    *,
    arch,
) -> Optional[Dict[int, Dict[str, Any]]]:
    if not entries:
        return {}

    limited = entries[:200]
    payload = {
        "path": path,
        "entries": [
            {"id": idx, "lineno": lineno, "text": text[:240]} for idx, (lineno, text) in enumerate(limited)
        ],
        "has_style_observer": bool(getattr(arch, "style_observer", None)),
        "has_quote_memory": bool(getattr(arch, "quote_memory", None)),
        "has_lexicon": bool(getattr(arch, "lexicon", None)),
    }
    response = try_call_llm_dict(LLM_SPEC_KEY, input_payload=payload, logger=LOGGER)
    if not isinstance(response, dict):
        return None

    decisions = response.get("decisions")
    if not isinstance(decisions, list):
        return None

    decision_map: Dict[int, Dict[str, Any]] = {}
    for item in decisions:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        decision_map[idx] = {
            "accept": bool(item.get("accept", False)),
            "targets": item.get("targets", []),
            "liked": bool(item.get("liked", False)),
            "tags": item.get("tags", []),
            "channel": item.get("channel", "inbox"),
        }

    for idx in range(len(limited)):
        decision_map.setdefault(idx, {"accept": True, "targets": ["lexicon", "style", "quote"]})

    if len(entries) > len(limited):
        for idx in range(len(limited), len(entries)):
            decision_map[idx] = {"accept": True, "targets": ["lexicon", "style", "quote"]}

    return decision_map
