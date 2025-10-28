from __future__ import annotations

import logging
import math
import random
import re
import unicodedata
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from AGI_Evolutive.core.structures.mai import Bid, MAI
from AGI_Evolutive.utils.llm_service import try_call_llm_dict

# --- MAI dispatcher (NLG) ---
# Registre de handlers extensible à chaud (aucune liste figée)
_MAI_NLG_HANDLERS: Dict[str, Callable[[Bid, "NLGContext"], None]] = {}


logger = logging.getLogger(__name__)


def register_nlg_handler(action_hint: str, fn: Callable[[Bid, "NLGContext"], None]) -> None:
    """Appelé lors de la promotion d’un MAI si celui-ci fournit un handler dédié."""

    _MAI_NLG_HANDLERS[action_hint] = fn


class NLGContext:
    """Context runtime minimaliste pour appliquer des bids MAI au rendu NLG."""

    def __init__(self, base_text: str, apply_hint: Callable[[str, str], str]):
        self.text = base_text
        self._apply_hint = apply_hint
        self._applied: List[Dict[str, str]] = []

    # --- API utilisée par les handlers ---
    def mark_applied(self, bid: Bid) -> None:
        entry = {"origin": bid.source, "hint": bid.action_hint}
        if entry not in self._applied:
            self._applied.append(entry)

    def register_custom_action(self, origin: str, hint: str) -> None:
        entry = {"origin": origin, "hint": hint}
        if entry not in self._applied:
            self._applied.append(entry)

    def apply_bid_hint(self, bid: Bid) -> None:
        hint = (bid.action_hint or "").strip()
        self.text = self._apply_hint(self.text, hint)
        self.mark_applied(bid)

    def redact(self, fields: Any, *, bid: Optional[Bid] = None) -> None:
        if not fields:
            return
        if isinstance(fields, (set, list, tuple)):
            payload = ", ".join(sorted(str(x) for x in fields if x is not None))
        else:
            payload = str(fields)
        note = f"(Je masque {payload} pour protéger la confidentialité.)"
        current = (self.text or "").strip()
        if note.lower() not in current.lower():
            self.text = f"{current}\n\n{note}" if current else note
        if bid is not None:
            self.mark_applied(bid)

    def rephrase_politely(self, *, bid: Optional[Bid] = None) -> None:
        self.text = self._apply_hint(self.text, "RephraseRespectfully")
        if bid is not None:
            self.mark_applied(bid)

    def applied_hints(self) -> List[Dict[str, str]]:
        return list(self._applied)


def apply_generic(bid: Bid, nlg_context: "NLGContext") -> None:
    """Fallback générique basé sur le contenu d’un Bid."""

    if isinstance(bid.target, dict) and "redact" in bid.target:
        nlg_context.redact(bid.target["redact"], bid=bid)
    elif (bid.action_hint or "").lower().startswith("rephrase"):
        nlg_context.rephrase_politely(bid=bid)
    else:
        nlg_context.apply_bid_hint(bid)


def apply_mai_bids_to_nlg(
    nlg_context: "NLGContext",
    state: Optional[Dict[str, Any]],
    predicate_registry: Optional[Dict[str, Any]],
) -> List[MAI]:
    from AGI_Evolutive.knowledge.mechanism_store import MechanismStore

    ms = MechanismStore()
    try:
        mechanisms: Iterable[MAI] = ms.scan_applicable(state or {}, predicate_registry or {})
    except Exception:
        mechanisms = []

    applied: List[MAI] = []
    for mechanism in mechanisms:
        applied.append(mechanism)
        try:
            bids = mechanism.propose(state or {})
        except Exception:
            continue
        for bid in bids or []:
            handler = _MAI_NLG_HANDLERS.get(bid.action_hint)
            if handler:
                try:
                    handler(bid, nlg_context)
                except Exception:
                    continue
                nlg_context.mark_applied(bid)
            else:
                apply_generic(bid, nlg_context)
    sections = try_call_llm_dict(
        "language_nlg",
        input_payload={
            "base_text": nlg_context.text,
            "applied_hints": applied,
        },
        logger=logger,
    )
    if sections:
        parts = [sections.get("introduction"), sections.get("body"), sections.get("conclusion")]
        assembled = "\n\n".join(part.strip() for part in parts if isinstance(part, str) and part.strip())
        if assembled:
            nlg_context.text = assembled
            nlg_context.register_custom_action("llm", "structured_sections")
            setattr(nlg_context, "llm_sections", dict(sections))
    return applied


ELIDE_MAP = {"le": "l’", "la": "l’", "de": "d’", "que": "qu’"}


def _starts_vowel(word: str) -> bool:
    return bool(re.match(r"^[aàâæeéèêëiîïoôœuùûüyh]", (word or "").lower()))


def elide_token(prev: str, next_word: str) -> str:
    base = prev.lower()
    if base in ELIDE_MAP and _starts_vowel(next_word):
        out = ELIDE_MAP[base]
        return out if prev.islower() else out.capitalize()
    return prev


def join_tokens(tokens: Iterable[str]) -> str:
    tokens = list(tokens)
    if not tokens:
        return ""
    out: List[str] = []
    for idx, token in enumerate(tokens):
        if idx > 0 and out[-1].lower() in ELIDE_MAP:
            out[-1] = elide_token(out[-1], token)
        out.append(token)
    text = " ".join(out)
    text = text.replace(" ’", "’").replace(" l’ ", " l’").replace(" d’ ", " d’").replace(" qu’ ", " qu’")
    text = re.sub(r"\s+([!?;:])", r"\1", text)
    text = re.sub(r"([!?;:])(?=\S)", r"\1 ", text)
    return text


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text or "")
    without_accents = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return without_accents.lower()


def _restore_casing(original: str, candidate: str) -> str:
    if not original:
        return candidate
    if original.isupper():
        return candidate.upper()
    if original[0].isupper():
        return candidate[:1].upper() + candidate[1:]
    return candidate


def _strip_punct(token: str) -> Tuple[str, str, str]:
    prefix = re.match(r"^\W+", token)
    suffix = re.search(r"\W+$", token)
    start = prefix.group(0) if prefix else ""
    end = suffix.group(0) if suffix else ""
    core = token[len(start) : len(token) - len(end)] if len(token) > len(start) + len(end) else ""
    return start, core, end


def _rebuild_token(prefix: str, replacement: str, suffix: str) -> str:
    return f"{prefix}{replacement}{suffix}"


DEFAULT_PARAPHRASES = {
    "par defaut": ["d’ordinaire", "en général"],
    "vraiment": ["franchement", "réellement"],
    "rapide": ["vite", "prompt"],
    "simple": ["basique", "élémentaire"],
    "important": ["clé", "central", "majeur"],
    "idee": ["intuition", "piste"],
    "guide": ["fil conducteur", "repère"],
    "plan": ["feuille de route", "stratégie"],
    "astuce": ["conseil", "tour de main"],
    "note": ["remarque", "annotation"],
    "bien": ["au top", "remarquable"],
}


FALLBACK_PARAPHRASES = {
    "cool": ["sympa", "amusant"],
    "ok": ["d’accord", "reçu"],
}


EST_VARIANT_PATTERN = re.compile(
    r"(?i)est[\s\u00A0]+(?:(?:un|une|le|la|l['’])|des|du|de\s+la)"
)


@dataclass
class ThompsonArm:
    alpha: float = 1.0
    beta: float = 1.0

    def sample(self, rng: random.Random) -> float:
        return rng.betavariate(self.alpha, self.beta)

    def update(self, reward: float) -> None:
        reward = min(max(reward, 0.0), 1.0)
        self.alpha += reward
        self.beta += 1.0 - reward


class BanditExplorer:
    """Hybrid Thompson Sampling + epsilon-greedy explorer for lexical choices."""

    def __init__(self, epsilon: float = 0.15, seed: Optional[int] = None):
        self._epsilon = max(0.0, min(1.0, epsilon))
        self._rng = random.Random(seed)
        self._arms: Dict[str, Dict[str, ThompsonArm]] = defaultdict(dict)

    def select(self, key: str, candidates: Sequence[str]) -> Optional[str]:
        candidates = list(dict.fromkeys(candidates))
        if not candidates:
            return None
        arms = self._arms[key]
        for candidate in candidates:
            arms.setdefault(candidate, ThompsonArm())
        if self._rng.random() < self._epsilon:
            choice = self._rng.choice(candidates)
            logger.debug("Exploring randomly for key '%s': %s", key, choice)
            return choice
        sampled = {candidate: arms[candidate].sample(self._rng) for candidate in candidates}
        choice = max(sampled.items(), key=lambda item: item[1])[0]
        logger.debug("Thompson sampling choice for key '%s': %s", key, choice)
        return choice

    def update(self, key: str, candidate: str, reward: float) -> None:
        arms = self._arms.get(key)
        if not arms or candidate not in arms:
            return
        logger.debug(
            "Updating explorer for key '%s' with reward %.3f (candidate=%s)",
            key,
            reward,
            candidate,
        )
        arms[candidate].update(reward)

    def arm_parameters(self, key: str) -> Dict[str, Tuple[float, float]]:
        arms = self._arms.get(key, {})
        return {candidate: (arm.alpha, arm.beta) for candidate, arm in arms.items()}


class AdaptiveEMA:
    """EMA whose smoothing factor is itself selected via Thompson Sampling."""

    def __init__(
        self,
        betas: Sequence[float] = (0.2, 0.4, 0.6, 0.8),
        explorer: Optional[BanditExplorer] = None,
        drift_threshold: float = 0.15,
    ) -> None:
        self._betas = tuple(sorted(set(betas)))
        self._explorer = explorer or BanditExplorer(epsilon=0.0)
        self._drift_threshold = max(drift_threshold, 0.0)
        self._values: Dict[float, float] = {}
        self._history: deque[float] = deque(maxlen=64)
        self._last_choice: Optional[float] = None

    def apply(self, value: float) -> float:
        if not self._betas:
            return value
        beta_choice = self._choose_beta()
        previous = self._values.get(beta_choice)
        smoothed = value if previous is None else beta_choice * value + (1.0 - beta_choice) * previous
        self._values[beta_choice] = smoothed
        self._detect_drift(smoothed)
        self._history.append(smoothed)
        self._last_choice = beta_choice
        return smoothed

    def feedback(self, reward: float) -> None:
        if self._last_choice is None:
            return
        self._explorer.update("__ema__", f"beta={self._last_choice:.1f}", reward)

    def history(self) -> List[float]:
        return list(self._history)

    def _choose_beta(self) -> float:
        labels = [f"beta={beta:.1f}" for beta in self._betas]
        label = self._explorer.select("__ema__", labels)
        if label is None:
            return self._betas[0]
        try:
            return float(label.split("=")[1])
        except (IndexError, ValueError):
            return self._betas[0]

    def _detect_drift(self, value: float) -> None:
        if not self._history:
            return
        delta = abs(value - self._history[-1])
        if delta > self._drift_threshold:
            logger.info("Qualité : dérive détectée (Δ=%.3f)", delta)


class DefaultQualityEvaluator:
    def score(self, original: str, candidate: str) -> float:
        if not original and not candidate:
            return 1.0
        if not original or not candidate:
            return 0.0
        orig_tokens = set(_normalize(original).split()) or {_normalize(original)}
        cand_tokens = set(_normalize(candidate).split()) or {_normalize(candidate)}
        intersection = len(orig_tokens & cand_tokens)
        union = len(orig_tokens | cand_tokens)
        jaccard = intersection / union if union else 1.0
        length_ratio = min(len(candidate), len(original)) / max(len(candidate), len(original))
        punctuation_penalty = 0.0
        if bool(re.search(r"[!?]{2,}", candidate)) and not re.search(r"[!?]{2,}", original):
            punctuation_penalty = 0.2
        score = max(0.0, min(1.0, 0.6 * jaccard + 0.4 * length_ratio - punctuation_penalty))
        return score


class QualityTracker:
    def __init__(
        self,
        evaluator: Optional[DefaultQualityEvaluator] = None,
        ema: Optional[AdaptiveEMA] = None,
    ) -> None:
        self._evaluator = evaluator or DefaultQualityEvaluator()
        self._ema = ema or AdaptiveEMA()

    def observe(self, original: str, candidate: str) -> float:
        raw_score = self._evaluator.score(original, candidate)
        smoothed = self._ema.apply(raw_score)
        self._ema.feedback(raw_score)
        return smoothed

    def quality_history(self) -> List[float]:
        return self._ema.history()


EMOJI_PATTERN = re.compile(r"[\U0001F300-\U0001FAFF]")


class OnlineFallbackClassifier:
    """Light-weight online classifier using simple lexical features."""

    def __init__(self, learning_rate: float = 0.25, max_step: float = 0.5) -> None:
        self._weights: Dict[str, float] = defaultdict(float)
        self._bias = -1.5
        self._lr = learning_rate
        self._max_step = max_step

    def _features(self, token: str) -> Dict[str, float]:
        base = token or ""
        length = len(base)
        vowel_ratio = sum(ch.lower() in "aeiouyàâäéèêëîïôöùûü" for ch in base) / length if length else 0.0
        punct_ratio = sum(not ch.isalnum() for ch in base) / length if length else 0.0
        upper_ratio = sum(ch.isupper() for ch in base) / length if length else 0.0
        emoji_presence = 1.0 if EMOJI_PATTERN.search(base) else 0.0
        features = {
            "bias": 1.0,
            "length": min(length, 12) / 12.0,
            "vowel_ratio": vowel_ratio,
            "punct_ratio": punct_ratio,
            "upper_ratio": upper_ratio,
            "emoji": emoji_presence,
        }
        for n in (2, 3):
            for idx in range(max(0, length - n + 1)):
                gram = base[idx : idx + n].lower()
                features[f"ng:{n}:{gram}"] = 1.0
        return features

    def predict(self, token: str) -> float:
        feats = self._features(token)
        z = self._bias
        for name, value in feats.items():
            z += self._weights.get(name, 0.0) * value
        return 1.0 / (1.0 + math.exp(-z))

    def update(self, token: str, reward: float) -> None:
        reward = min(max(reward, 0.0), 1.0)
        prediction = self.predict(token)
        error = reward - prediction
        step = max(-self._max_step, min(self._max_step, self._lr * error))
        feats = self._features(token)
        for name, value in feats.items():
            self._weights[name] += step * value
        self._bias += step

    def should_paraphrase(self, token: str, threshold: float = 0.65) -> bool:
        return self.predict(token) >= threshold


GLOBAL_EXPLORER = BanditExplorer()
GLOBAL_QUALITY_TRACKER = QualityTracker()
GLOBAL_CLASSIFIER = OnlineFallbackClassifier()


def paraphrase_light(
    text: str,
    prob: float = 0.35,
    *,
    explorer: Optional[BanditExplorer] = None,
    quality_tracker: Optional[QualityTracker] = None,
    classifier: Optional[OnlineFallbackClassifier] = None,
    rng: Optional[random.Random] = None,
) -> str:
    if not text:
        return ""

    explorer = explorer or GLOBAL_EXPLORER
    quality_tracker = quality_tracker or GLOBAL_QUALITY_TRACKER
    classifier = classifier or GLOBAL_CLASSIFIER
    rng = rng or random.Random()

    tokens = re.split(r"(\s+)", text)
    rebuilt: List[str] = []

    for token in tokens:
        if not token or token.isspace():
            rebuilt.append(token)
            continue

        prefix, core, suffix = _strip_punct(token)
        if not core:
            rebuilt.append(token)
            continue

        inner_prefix = ""
        core_body = core
        apostrophe_match = re.match(r"^([A-Za-zÀ-ÖØ-öø-ÿ]{1,3}['’])(.*)$", core_body)
        if apostrophe_match:
            inner_prefix = apostrophe_match.group(1)
            core_body = apostrophe_match.group(2)

        if not core_body:
            rebuilt.append(token)
            continue

        normalized = _normalize(core_body)
        synonyms = DEFAULT_PARAPHRASES.get(normalized)
        fallback_synonyms = FALLBACK_PARAPHRASES.get(normalized)
        selected: Optional[str] = None

        if synonyms:
            if rng.random() < prob:
                choice = explorer.select(normalized, synonyms)
                if choice:
                    selected = choice
        elif fallback_synonyms and classifier.should_paraphrase(core_body):
            choice = explorer.select(f"fallback:{normalized}", fallback_synonyms)
            if choice:
                selected = choice

        if selected:
            replaced = _restore_casing(core_body, selected)
            rebuilt_token = _rebuild_token(prefix + inner_prefix, replaced, suffix)
            reward = quality_tracker.observe(core_body, replaced)
            key = normalized if synonyms else f"fallback:{normalized}"
            explorer.update(key, selected, reward)
            classifier.update(core_body, reward)
            rebuilt.append(rebuilt_token)
            logger.debug(
                "Paraphrase applied: '%s' -> '%s' (reward=%.3f)",
                core_body,
                replaced,
                reward,
            )
        else:
            if EST_VARIANT_PATTERN.search(token):
                logger.debug("Pattern 'est …' detected, no safe paraphrase applied for '%s'", token)
            rebuilt.append(token)

    output = "".join(rebuilt)
    output = re.sub(r"\s+\n", "\n", output)
    return output.strip()
