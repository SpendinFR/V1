from collections import Counter
from math import exp
from typing import Any, Dict, Iterable, Tuple

from AGI_Evolutive.utils.llm_service import try_call_llm_dict


def _memory_recent(arch, limit: int):
    store = None
    memory = getattr(arch, "memory", None)
    if memory is not None:
        store = getattr(memory, "store", None)
    if store is None:
        store = getattr(arch, "_memory_store", None)
    if store is None or not hasattr(store, "get_recent"):
        return []
    try:
        data = store.get_recent(limit)
    except Exception:
        return []
    return list(data) if isinstance(data, list) else []


def _flatten_text_fragments(fragments: Iterable[str]) -> str:
    return " ".join(fragments).strip().lower()


def _decay_weight(index: int, decay: float = 0.85) -> float:
    if index <= 0:
        return 1.0
    return decay ** index


def _extract_language_from_text(text: str) -> Tuple[str, float]:
    if "en français" in text or "parle français" in text:
        return "fr", 1.0
    if "in english" in text or "english please" in text:
        return "en", 1.0
    if "habla español" in text or "en español" in text:
        return "es", 0.8
    return "", 0.0


def _extract_sentiment_tokens(text: str) -> Tuple[Iterable[str], Iterable[str]]:
    positive = []
    negative = []
    if any(token in text for token in ("merci", "thank you", "parfait", "great")):
        positive.append("gratitude")
    if any(token in text for token in ("super", "excellent", "nice", "j'adore")):
        positive.append("praise")
    if any(token in text for token in ("pas", "not", "n'aime pas", "hate", "horrible")):
        negative.append("critique")
    if "too slow" in text or "trop lent" in text:
        negative.append("speed")
    return positive, negative


def infer_preferences(arch, window: int = 100) -> Dict[str, Any]:
    """
    Observe des corrections/choix récents pour déduire values/likes/dislikes/style.
    Retourne un patch + score de confiance (0..1).
    """
    likes, dislikes, values = set(), set(), set()
    style: Dict[str, Any] = {}
    feature_counts: Counter[str] = Counter()

    events = _memory_recent(arch, window)
    for idx, ev in enumerate(reversed(events)):
        weight = _decay_weight(idx)
        fragments = [
            str(ev.get("text") or ""),
            str(ev.get("summary") or ""),
        ]
        metadata = ev.get("metadata")
        if isinstance(metadata, dict):
            fragments.append(str(metadata.get("note") or ""))
        text = _flatten_text_fragments(fragment for fragment in fragments if fragment)

        if "plus court" in text or "résume" in text or "shorter" in text:
            style["conciseness"] = "high"
            feature_counts["style_conciseness"] += weight
        if "plus détaillé" in text or "explique davantage" in text or "more details" in text:
            style["conciseness"] = "low"
            feature_counts["style_conciseness"] += weight

        lang_hint, lang_weight = _extract_language_from_text(text)
        if lang_hint:
            style["lang"] = lang_hint
            feature_counts["style_lang"] += weight * lang_weight

        if "sources" in text or "citer" in text or "references" in text:
            values.add("traceability")
            feature_counts["values"] += weight

        if "tone" in text and "professional" in text:
            style["tone"] = "formal"
            feature_counts["style_tone"] += weight
        if "friendly" in text or "amical" in text:
            style["tone"] = "friendly"
            feature_counts["style_tone"] += weight

        positive_tokens, negative_tokens = _extract_sentiment_tokens(text)
        if positive_tokens:
            likes.update(positive_tokens)
            feature_counts["likes"] += weight * len(positive_tokens)
        if negative_tokens:
            dislikes.update(negative_tokens)
            feature_counts["dislikes"] += weight * len(negative_tokens)

        request_like_tokens = [
            ("j'aime", likes),
            ("i like", likes),
            ("je préfère", likes),
            ("préférer", likes),
        ]
        for token, target_set in request_like_tokens:
            if token in text:
                target_set.add(token)
                feature_counts["likes"] += weight

        request_dislike_tokens = [
            ("n'aime pas", dislikes),
            ("je déteste", dislikes),
            ("i dislike", dislikes),
            ("évite", dislikes),
        ]
        for token, target_set in request_dislike_tokens:
            if token in text:
                target_set.add(token)
                feature_counts["dislikes"] += weight

        kind = str(ev.get("kind"))
        if kind == "decision":
            action = str(ev.get("action") or "").lower()
            if "safeguard" in action or "redact" in action:
                values.add("care")
                feature_counts["values"] += weight
            if "explore" in action or "hypotheses" in action:
                values.add("curiosity")
                feature_counts["values"] += weight

        if isinstance(metadata, dict):
            lang_meta = metadata.get("lang") or metadata.get("language")
            if isinstance(lang_meta, str) and lang_meta:
                style["lang"] = lang_meta.lower()
                feature_counts["style_lang"] += weight

            tone_meta = metadata.get("tone")
            if isinstance(tone_meta, str) and tone_meta:
                style["tone"] = tone_meta.lower()
                feature_counts["style_tone"] += weight

            values_meta = metadata.get("values")
            if isinstance(values_meta, (list, set, tuple)):
                for val in values_meta:
                    if not isinstance(val, str):
                        continue
                    values.add(val)
                    feature_counts["values"] += weight

            feedback = metadata.get("feedback")
            if isinstance(feedback, dict):
                sentiment = str(feedback.get("sentiment") or "").lower()
                if sentiment in {"positive", "satisfied"}:
                    likes.add("feedback_positive")
                    feature_counts["likes"] += weight
                if sentiment in {"negative", "unsatisfied", "frustrated"}:
                    dislikes.add("feedback_negative")
                    feature_counts["dislikes"] += weight

    weight_map = {
        "values": 1.4,
        "style_conciseness": 1.2,
        "style_lang": 1.5,
        "style_tone": 1.1,
        "likes": 1.0,
        "dislikes": 1.0,
    }

    bias = -0.25
    raw_score = bias
    for key, count in feature_counts.items():
        raw_score += weight_map.get(key, 0.8) * count

    try:
        score = 1.0 / (1.0 + exp(-raw_score))
    except OverflowError:
        score = 1.0 if raw_score > 0 else 0.0
    score = max(0.0, min(1.0, score))
    patch: Dict[str, Any] = {
        "preferences": {},
    }
    if values:
        patch["preferences"]["values"] = sorted(values)
    if likes:
        patch["preferences"]["likes"] = sorted(likes)
    if dislikes:
        patch["preferences"]["dislikes"] = sorted(dislikes)
    if style:
        patch["preferences"]["style"] = style

    heuristics = {
        "feature_counts": {key: float(value) for key, value in feature_counts.items()},
        "likes": sorted(likes),
        "dislikes": sorted(dislikes),
        "values": sorted(values),
        "style": style,
    }
    result = {"patch": patch if patch["preferences"] else {"preferences": {}}, "score": score, "heuristics": heuristics}

    llm_payload = {
        "heuristics": heuristics,
        "score": score,
        "patch": patch,
    }
    llm_response = try_call_llm_dict(
        "cognition_preferences_inference",
        input_payload=llm_payload,
        logger=getattr(arch, "logger", None),
    )
    if isinstance(llm_response, dict):
        if "patch" in llm_response and isinstance(llm_response["patch"], dict):
            llm_patch = llm_response["patch"]
            preferences = llm_patch.get("preferences") if isinstance(llm_patch, dict) else None
            if isinstance(preferences, dict):
                result["patch"] = {"preferences": dict(preferences)}
            else:
                result["patch"] = llm_patch
        if "score" in llm_response:
            try:
                result["score"] = max(0.0, min(1.0, float(llm_response["score"])))
            except (TypeError, ValueError):
                pass
        if isinstance(llm_response.get("notes"), str) and llm_response["notes"].strip():
            result["notes"] = llm_response["notes"].strip()
        if "confidence" in llm_response:
            try:
                result["confidence"] = max(0.0, min(1.0, float(llm_response["confidence"])))
            except (TypeError, ValueError):
                pass
        result["llm"] = llm_response

    return result


def apply_preferences_if_confident(arch, threshold: float = 0.75) -> Dict[str, Any]:
    res = infer_preferences(arch)
    patch = res.get("patch", {"preferences": {}})
    score = float(res.get("score", 0.0))
    preferences = patch.get("preferences", {}) if isinstance(patch, dict) else {}
    if not preferences:
        return {"status": "no_change", "score": score}
    if score >= threshold:
        try:
            arch.self_model.set_identity_patch(patch)
            return {"status": "applied", "score": score, "patch": patch}
        except Exception:
            return {"status": "error", "score": score, "patch": patch}
    return {"status": "needs_confirmation", "score": score, "patch": patch}
