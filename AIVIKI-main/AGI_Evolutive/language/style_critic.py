from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple

from AGI_Evolutive.utils.llm_service import try_call_llm_dict


EMOJI_RE = re.compile(
    "[\U0001F300-\U0001F6FF\U0001F900-\U0001FAFF\U0001F1E6-\U0001F1FF]"
)
HEDGING_RE = re.compile(
    r"\b(p(?:e|é|è|ê)ut[-\s]?(?:etre|être))\b",
    flags=re.IGNORECASE,
)
DOUBLE_ADVERB_RE = re.compile(r"\b(tr[eéè]s)\s+(tr[eéè]s)\b", flags=re.IGNORECASE)
COPULA_RE = re.compile(
    r"\b(?:c['’]?est|est)\s+(?:un|une|le|la|l['’])",
    flags=re.IGNORECASE,
)
PUNCT_BEFORE_RE = re.compile(r"\s+([!?;:])")
PUNCT_AFTER_RE = re.compile(r"([!?;:])(?!\s)")


LOGGER = logging.getLogger(__name__)
LLM_SPEC_KEY = "language_style_critique"


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _strip_accents(text: str) -> str:
    """Return a lowercase, accent-less representation of *text*."""

    normalized = unicodedata.normalize("NFD", text)
    return "".join(
        ch.lower()
        for ch in normalized
        if not unicodedata.combining(ch)
    )


@dataclass
class SignalSnapshot:
    """Expose the internal state of an adaptive signal."""

    observation: float
    pressure: float
    momentum: float
    baseline: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "observation": self.observation,
            "pressure": self.pressure,
            "momentum": self.momentum,
            "baseline": self.baseline,
        }


@dataclass
class AdaptiveSignal:
    """Streaming signal with a touch of evolution via momentum."""

    baseline: float
    sensitivity: float = 1.0
    decay: float = 0.35
    reward_strength: float = 0.3
    floor: float = 0.0
    momentum: float = 0.0

    def observe(self, observation: float) -> SignalSnapshot:
        pressure = max(0.0, observation - self.baseline)
        self.momentum = (1.0 - self.decay) * self.momentum + self.decay * pressure
        amplified = pressure * (1.0 + self.sensitivity * self.momentum)
        return SignalSnapshot(
            observation=observation,
            pressure=amplified,
            momentum=self.momentum,
            baseline=self.baseline,
        )

    def reinforce(self, reward: float) -> None:
        delta = reward * self.reward_strength
        self.baseline = max(self.floor, self.baseline + delta)
        self.momentum *= 0.5


class EvolvingSignalModel:
    """Minimal orchestration for adaptive signals."""

    def __init__(self, signals: Dict[str, AdaptiveSignal]):
        self._signals = signals
        self._snapshots: Dict[str, SignalSnapshot] = {}

    def observe(self, name: str, observation: float) -> SignalSnapshot:
        signal = self._signals[name]
        snapshot = signal.observe(observation)
        self._snapshots[name] = snapshot
        return snapshot

    def reinforce(self, name: str, reward: float) -> None:
        if name in self._signals:
            self._signals[name].reinforce(reward)

    def last_snapshots(self) -> Dict[str, Dict[str, float]]:
        return {name: snapshot.as_dict() for name, snapshot in self._snapshots.items()}


def _expressive_density(text: str) -> float:
    bang_runs = sum(1 for _ in re.finditer(r"!{2,}", text))
    question_runs = sum(1 for _ in re.finditer(r"\?{2,}", text))
    ellipsis = text.count("...")
    emoji = len(EMOJI_RE.findall(text))
    words = re.findall(r"\b\w+\b", text, flags=re.UNICODE)
    caps_ratio = 0.0
    if words:
        caps_words = sum(1 for word in words if len(word) > 2 and word.isupper())
        caps_ratio = caps_words / len(words)

    return (
        0.6 * bang_runs
        + 0.45 * question_runs
        + 0.3 * ellipsis
        + 0.4 * emoji
        + 1.5 * caps_ratio
    )


class StyleCritic:
    """Critique légère du style pour post-traiter les réponses."""

    def __init__(self, max_chars: int = 1200, signal_overrides: Dict[str, AdaptiveSignal] | None = None):
        self.max_chars = int(max_chars)
        defaults: Dict[str, AdaptiveSignal] = {
            "too_long": AdaptiveSignal(baseline=float(self.max_chars), sensitivity=0.02, decay=0.4),
            "excess_bang": AdaptiveSignal(baseline=0.4, sensitivity=1.2, decay=0.3),
            "hedging_maybe": AdaptiveSignal(baseline=0.1, sensitivity=1.5, decay=0.35),
            "adverb_dup": AdaptiveSignal(baseline=0.15, sensitivity=1.4, decay=0.3),
            "copula_definition": AdaptiveSignal(baseline=0.4, sensitivity=1.1, decay=0.3),
            "expressive_noise": AdaptiveSignal(baseline=0.8, sensitivity=1.0, decay=0.25),
        }
        if signal_overrides:
            defaults.update(signal_overrides)
        self._signals = EvolvingSignalModel(defaults)

    def analyze(self, text: str) -> Dict[str, Any]:
        sample = (text or "").strip()
        heuristic_report = self._run_heuristics(sample)
        payload = self._build_llm_payload(sample, heuristic_report)
        response = try_call_llm_dict(LLM_SPEC_KEY, input_payload=payload, logger=LOGGER)
        if not response:
            return heuristic_report
        return self._merge_llm_response(sample, heuristic_report, response)

    def _run_heuristics(self, sample: str) -> Dict[str, Any]:
        normalized = _strip_accents(sample)
        issues: List[Tuple[str, Any]] = []

        if len(sample) > self.max_chars:
            snapshot = self._signals.observe("too_long", float(len(sample)))
            issues.append(("too_long", round(snapshot.pressure, 3)))

        double_bang_matches = list(re.finditer(r"!{2,}", sample))
        if double_bang_matches:
            snapshot = self._signals.observe("excess_bang", float(len(double_bang_matches)))
            if snapshot.pressure > 0.0:
                issues.append(("excess_bang", round(snapshot.pressure, 3)))

        double_adverbs = DOUBLE_ADVERB_RE.findall(normalized)
        if double_adverbs:
            snapshot = self._signals.observe("adverb_dup", float(len(double_adverbs)))
            if snapshot.pressure > 0.0:
                issues.append(("adverb_dup", {"severity": round(snapshot.pressure, 3), "examples": double_adverbs}))

        if HEDGING_RE.search(normalized):
            snapshot = self._signals.observe("hedging_maybe", 1.0)
            if snapshot.pressure > 0.0:
                issues.append(("hedging_maybe", round(snapshot.pressure, 3)))

        copula_hits = list(COPULA_RE.finditer(sample))
        if copula_hits:
            snapshot = self._signals.observe("copula_definition", float(len(copula_hits)))
            if snapshot.pressure > 0.0:
                issues.append(("copula_definition", round(snapshot.pressure, 3)))

        expressive = _expressive_density(sample)
        if expressive:
            snapshot = self._signals.observe("expressive_noise", expressive)
            if snapshot.pressure > 0.0:
                issues.append(("expressive_noise", round(snapshot.pressure, 3)))

        return {
            "length": len(sample),
            "issues": issues,
            "signals": self._signals.last_snapshots(),
        }

    def _build_llm_payload(self, sample: str, heuristics: Mapping[str, Any]) -> Dict[str, Any]:
        issues = []
        for item in heuristics.get("issues", []):
            if isinstance(item, (tuple, list)) and len(item) == 2:
                code, payload = item
                issues.append({"code": code, "payload": payload})
        return {
            "text": sample,
            "max_chars": self.max_chars,
            "heuristics": {
                "length": heuristics.get("length"),
                "issues": issues,
                "signals": heuristics.get("signals"),
            },
        }

    def _merge_llm_response(
        self,
        sample: str,
        heuristics: Mapping[str, Any],
        response: Mapping[str, Any],
    ) -> Dict[str, Any]:
        llm_issues_raw = response.get("issues")
        llm_issue_map: Dict[str, Mapping[str, Any]] = {}
        if isinstance(llm_issues_raw, list):
            for entry in llm_issues_raw:
                if isinstance(entry, Mapping):
                    code = str(
                        entry.get("code")
                        or entry.get("name")
                        or entry.get("issue")
                        or ""
                    ).strip()
                    if code:
                        llm_issue_map[code] = entry

        combined: List[Tuple[str, Any]] = []
        llm_feedback: Dict[str, Dict[str, Any]] = {}
        seen: set[str] = set()

        for item in heuristics.get("issues", []):
            if not (isinstance(item, (tuple, list)) and len(item) == 2):
                continue
            code, payload = item
            info = llm_issue_map.get(code)
            new_payload = payload
            if info:
                severity = _coerce_float(info.get("severity"))
                explanation = info.get("explanation") or info.get("rationale") or info.get("reason")
                suggestion = info.get("suggested_fix") or info.get("suggestion")

                if isinstance(payload, dict):
                    new_payload = dict(payload)
                    if severity is not None:
                        base = _coerce_float(new_payload.get("severity"))
                        if base is not None:
                            new_payload["severity"] = round((base + severity) / 2.0, 3)
                        else:
                            new_payload["severity"] = severity
                else:
                    if severity is not None:
                        base = _coerce_float(payload)
                        if base is not None:
                            new_payload = round((base + severity) / 2.0, 3)
                        else:
                            new_payload = severity

                feedback_entry: Dict[str, Any] = {}
                if severity is not None:
                    feedback_entry["severity"] = severity
                if explanation:
                    explanation_str = str(explanation).strip()
                    if explanation_str:
                        feedback_entry["explanation"] = explanation_str
                        if isinstance(new_payload, dict):
                            new_payload.setdefault("explanation", explanation_str)
                if suggestion:
                    suggestion_str = str(suggestion).strip()
                    if suggestion_str:
                        feedback_entry["suggested_fix"] = suggestion_str
                        if isinstance(new_payload, dict):
                            new_payload.setdefault("suggested_fix", suggestion_str)
                if feedback_entry:
                    llm_feedback[code] = feedback_entry

            combined.append((code, new_payload))
            seen.add(str(code))

        for code, info in llm_issue_map.items():
            if str(code) in seen:
                continue
            severity = _coerce_float(info.get("severity"))
            if severity is None:
                continue
            explanation = info.get("explanation") or info.get("rationale") or info.get("reason")
            suggestion = info.get("suggested_fix") or info.get("suggestion")
            payload: Any
            feedback_entry: Dict[str, Any] = {"severity": severity}
            explanation_str = str(explanation).strip() if isinstance(explanation, str) else None
            suggestion_str = str(suggestion).strip() if isinstance(suggestion, str) else None
            if explanation_str:
                feedback_entry["explanation"] = explanation_str
            if suggestion_str:
                feedback_entry["suggested_fix"] = suggestion_str
            if explanation_str or suggestion_str:
                payload = {"severity": severity}
                if explanation_str:
                    payload["explanation"] = explanation_str
                if suggestion_str:
                    payload["suggested_fix"] = suggestion_str
            else:
                payload = severity
            combined.append((code, payload))
            llm_feedback[str(code)] = feedback_entry

        length_value = _coerce_float(response.get("length"))
        if length_value is not None:
            length_report = int(length_value)
        else:
            length_report = int(heuristics.get("length", len(sample)) or 0)

        merged: Dict[str, Any] = {
            "length": length_report,
            "issues": combined,
            "signals": heuristics.get("signals", {}),
        }

        confidence = _coerce_float(response.get("confidence"))
        if confidence is not None:
            merged["llm_confidence"] = max(0.0, min(1.0, confidence))

        notes = response.get("notes")
        if isinstance(notes, str) and notes.strip():
            merged["llm_notes"] = notes.strip()

        if llm_feedback:
            merged["llm_feedback"] = llm_feedback

        return merged

    def rewrite(self, text: str) -> str:
        if not text:
            return ""

        cleaned = re.sub(r"[ \t]+", " ", text)
        cleaned = re.sub(r" ?\n ?", "\n", cleaned)

        def _replace_probablement(match: re.Match[str]) -> str:
            original = match.group(0)
            replacement = "probablement"
            return replacement.capitalize() if original[:1].isupper() else replacement

        cleaned = HEDGING_RE.sub(_replace_probablement, cleaned)
        cleaned = DOUBLE_ADVERB_RE.sub(lambda m: m.group(1), cleaned)
        cleaned = re.sub(r"([!?]){2,}", r"\1", cleaned)
        cleaned = PUNCT_BEFORE_RE.sub(r"\1", cleaned)
        cleaned = PUNCT_AFTER_RE.sub(r"\1 ", cleaned)
        cleaned = cleaned.strip()

        if len(cleaned) > self.max_chars:
            cleaned = cleaned[: self.max_chars].rstrip()
            if not cleaned.endswith((".", "!", "?", "…")):
                cleaned = cleaned.rstrip("…") + "…"

        return cleaned

    def nudge(self, issue_name: str, reward: float) -> None:
        """Integrate external feedback to keep the critic evolving."""

        self._signals.reinforce(issue_name, reward)
