"""
EmotionEngineV2 — moteur émotionnel évolutif, compatible avec l'existant
-------------------------------------------------------------------------
- PAD sur 2 échelles : épisode (rapide) + humeur (lente, set‑point dynamique)
- Appraisal *pluginisé* (erreur, succès, récompense, charge cog., fatigue, social)
  avec poids apprenants en ligne (SGD borné) et confiance par signal
- Incertitude estimée et propagée → exploration, focus, ton, safety gate
- Journalisation JSONL des épisodes + dashboard JSON d'état (poids, mood, etc.)
- Meta-contrôle des plugins (gating doux, entropie cible, apprentissage contextuel)
- Auto-synthèse de patterns contextuels → plugin émergent latent
- Plasticité multi-échelles : demi-vies adaptatives corrélées aux modulateurs
- Rituels auto-scénarisés (auto-régulation) avec mémoire synthétique
- **Compat 100%** avec l'existant :
  - API: bind(), register_event(), step(), get_modulators(), get_state(),
          update_from_recent_memories(), modulate_homeostasis()
  - Modulateurs: mêmes clés + alias
    *tone* **et** *language_tone*, *goal_priority_bias* **dict** + *goal_priority_bias_scalar*
  - Bump explicite de arch.global_activation via activation_delta
  - Cible de décroissance configurable: vers "mood" (par défaut) ou "neutral"

Auteur: Toi (refonte assistée) — 2025-10-19
Licence: MIT
"""
from __future__ import annotations

import logging
import os
import json
import time
import math
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Tuple, TYPE_CHECKING

from AGI_Evolutive.utils.llm_service import (
    LLMIntegrationError,
    LLMUnavailableError,
    get_llm_manager,
    is_llm_enabled,
)

# ========================= Utilitaires ========================= #

def clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    from AGI_Evolutive.phenomenology import PhenomenalJournal


LOGGER = logging.getLogger(__name__)

_POSITIVE_EMOTIONS = {
    "joie",
    "fierté",
    "soulagement",
    "gratitude",
    "calme",
    "confiance",
    "enthousiasme",
    "espoir",
    "satisfaction",
}
_NEGATIVE_EMOTIONS = {
    "stress",
    "colère",
    "tristesse",
    "anxiété",
    "peur",
    "culpabilité",
    "frustration",
    "fatigue",
    "inquiétude",
}


def _softmax(logits: Dict[str, float], temperature: float = 1.0) -> Dict[str, float]:
    if not logits:
        return {}
    temp = max(1e-3, float(temperature))
    scaled = {k: float(v) / temp for k, v in logits.items()}
    max_logit = max(scaled.values())
    exps = {k: math.exp(v - max_logit) for k, v in scaled.items()}
    total = sum(exps.values()) or 1.0
    return {k: exps[k] / total for k in logits}


def json_sanitize(obj: Any) -> Any:
    """Sanitise en JSON sans dépendances externes."""
    try:
        json.dumps(obj)
        return obj
    except Exception:
        if isinstance(obj, dict):
            return {str(k): json_sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [json_sanitize(v) for v in obj]
        try:
            if obj != obj:  # NaN
                return None
        except Exception:
            pass
        return str(obj)


NEUTRAL = {"valence": 0.0, "arousal": 0.2, "dominance": 0.5}

# ========================= États ========================= #

@dataclass
class AffectState:
    t: float = field(default_factory=time.time)
    valence: float = NEUTRAL["valence"]      # -1..1
    arousal: float = NEUTRAL["arousal"]      #  0..1
    dominance: float = NEUTRAL["dominance"]  #  0..1
    label: str = "neutral"


@dataclass
class MoodState:
    t: float = field(default_factory=time.time)
    valence: float = NEUTRAL["valence"]
    arousal: float = 0.15
    dominance: float = NEUTRAL["dominance"]


@dataclass
class EmotionEpisode:
    id: str
    onset: float
    dt: float
    dv: float
    da: float
    dd: float
    label: str
    causes: List[Tuple[str, float]]  # (plugin, contribution)
    confidence: float
    meta: Dict[str, Any]


# ========================= Appraisal (plugins) ========================= #

@dataclass
class AppraisalOutput:
    dv: float
    da: float
    dd: float
    confidence: float = 1.0
    meta: Dict[str, Any] = field(default_factory=dict)


class AppraisalPlugin:
    name: str = "base"
    def __call__(self, ctx: Dict[str, Any]) -> AppraisalOutput:
        raise NotImplementedError


class CognitiveLoadPlugin(AppraisalPlugin):
    name = "cog_load"
    def __call__(self, ctx: Dict[str, Any]) -> AppraisalOutput:
        load = float(ctx.get("cognitive_load", 0.5) or 0.0)
        conf = 0.7 if "cognitive_load" in ctx else 0.3
        return AppraisalOutput(dv=0.0, da=0.30 * load, dd=0.0, confidence=conf, meta={"load": load})


class ErrorPlugin(AppraisalPlugin):
    name = "error"
    def __call__(self, ctx: Dict[str, Any]) -> AppraisalOutput:
        errs = ctx.get("recent_errors", []) or []
        n = float(len(errs))
        magnitude = clip(n / 5.0, 0.0, 1.0)
        conf = 0.6 if n > 0 else 0.25
        return AppraisalOutput(dv=-0.40 * magnitude, da=0.10 * magnitude, dd=-0.20 * magnitude,
                               confidence=conf, meta={"n_errors": n})


class SuccessPlugin(AppraisalPlugin):
    name = "success"
    def __call__(self, ctx: Dict[str, Any]) -> AppraisalOutput:
        succ = float(ctx.get("recent_success", 0.0) or 0.0)  # ∈[0,1]
        conf = 0.6 if "recent_success" in ctx else 0.3
        return AppraisalOutput(dv=+0.30 * succ, da=+0.05 * succ, dd=+0.20 * succ,
                               confidence=conf, meta={"success": succ})


class RewardPlugin(AppraisalPlugin):
    name = "reward"
    def __call__(self, ctx: Dict[str, Any]) -> AppraisalOutput:
        rew = float(ctx.get("reward_signal", 0.0) or 0.0)
        rew = clip(rew, -1.0, 1.0)
        conf = 0.7 if "reward_signal" in ctx else 0.3
        return AppraisalOutput(dv=0.35 * rew, da=0.10 * abs(rew), dd=0.05 * rew,
                               confidence=conf, meta={"reward": rew})


class IntrinsicPleasurePlugin(AppraisalPlugin):
    name = "intrinsic_pleasure"

    def __call__(self, ctx: Dict[str, Any]) -> AppraisalOutput:
        kernel = ctx.get("phenomenal_kernel") or {}
        pleasure = ctx.get("hedonic_signal")
        if pleasure is None and isinstance(kernel, dict):
            pleasure = kernel.get("hedonic_reward")
        try:
            pleasure_val = float(pleasure)
        except (TypeError, ValueError):
            pleasure_val = 0.0
        pleasure_val = clip(pleasure_val, -1.0, 1.0)
        intensity = abs(pleasure_val)
        meta = {"pleasure": pleasure_val}
        if isinstance(kernel, dict):
            meta.update({
                "mode": kernel.get("mode"),
                "suggestion": kernel.get("mode_suggestion"),
            })
        return AppraisalOutput(
            dv=0.40 * pleasure_val,
            da=0.12 * intensity,
            dd=0.06 * pleasure_val,
            confidence=0.4 + 0.4 * intensity,
            meta=meta,
        )


class FatiguePlugin(AppraisalPlugin):
    name = "fatigue"
    def __call__(self, ctx: Dict[str, Any]) -> AppraisalOutput:
        fatigue = float(ctx.get("fatigue", 0.0) or 0.0)  # 0–1
        conf = 0.6 if "fatigue" in ctx else 0.25
        return AppraisalOutput(dv=-0.20 * fatigue, da=-0.15 * fatigue, dd=-0.05 * fatigue,
                               confidence=conf, meta={"fatigue": fatigue})


class SocialFeedbackPlugin(AppraisalPlugin):
    name = "social"
    POS = ("bravo", "merci", "thanks", "bien", "good", "+1")
    NEG = ("mauvais", "nul", "pas bien", "bad", "-1", "wrong")
    def __call__(self, ctx: Dict[str, Any]) -> AppraisalOutput:
        cues = ctx.get("social_cues", []) or []
        pos = sum(any(p in str(c).lower() for p in self.POS) for c in cues)
        neg = sum(any(n in str(c).lower() for n in self.NEG) for c in cues)
        score = clip((pos - neg) / 3.0, -1.0, 1.0)
        conf = 0.5 if cues else 0.2
        return AppraisalOutput(dv=0.25 * score, da=0.05 * abs(score), dd=0.05 * score,
                               confidence=conf, meta={"pos": pos, "neg": neg})


class PluginMetaController:
    def __init__(self, plugin_names: List[str], lr: float = 0.03, target_entropy: float = 1.2):
        self.plugin_names = list(plugin_names)
        self.lr = float(lr)
        self.params: Dict[str, defaultdict] = {name: defaultdict(float) for name in self.plugin_names}
        self.temperature: float = 1.3
        self.target_entropy = float(target_entropy)
        self._avg_quality: float = 0.0
        self._last_feats: Dict[str, float] = {}
        self._last_gates: Dict[str, float] = {name: 1.0 / max(1, len(self.plugin_names)) for name in self.plugin_names}

    def _feature_vector(self, ctx: Dict[str, Any]) -> Dict[str, float]:
        feats: Dict[str, float] = {"bias": 1.0}
        feats["cognitive_load"] = float(ctx.get("cognitive_load", 0.5) or 0.0)
        feats["error_density"] = float(len(ctx.get("recent_errors", []) or [])) / 5.0
        feats["success"] = float(ctx.get("recent_success", 0.0) or 0.0)
        feats["reward"] = float(ctx.get("reward_signal", 0.0) or 0.0)
        feats["fatigue"] = float(ctx.get("fatigue", 0.0) or 0.0)
        tod = int(ctx.get("time_of_day", -1))
        if tod >= 0:
            feats["is_night"] = 1.0 if tod < 6 or tod > 21 else 0.0
        latent = ctx.get("_latent_features")
        if isinstance(latent, dict):
            for key, value in latent.items():
                if len(feats) > 24:
                    break
                if isinstance(value, (int, float)):
                    feats[f"latent_{key}"] = float(value)
        return feats

    def compute(self, ctx: Dict[str, Any], confidences: Dict[str, float]) -> Dict[str, float]:
        feats = self._feature_vector(ctx)
        logits: Dict[str, float] = {}
        for name in self.plugin_names:
            params = self.params.setdefault(name, defaultdict(float))
            logit = 0.0
            for feat, value in feats.items():
                logit += params[feat] * float(value)
            logit += math.log(max(confidences.get(name, 1e-3), 1e-3))
            logits[name] = logit
        gates = _softmax(logits, temperature=self.temperature or 1.0)
        if not gates:
            gates = {name: 1.0 / max(1, len(self.plugin_names)) for name in self.plugin_names}
        self._last_feats = feats
        self._last_gates = gates
        return gates

    def update(self, quality: Optional[float]) -> None:
        if quality is None or not self._last_feats:
            return
        q = clip(float(quality), -1.0, 1.0)
        self._avg_quality = 0.92 * self._avg_quality + 0.08 * q
        advantage = q - self._avg_quality
        for name, gate_value in self._last_gates.items():
            params = self.params.setdefault(name, defaultdict(float))
            for feat, value in self._last_feats.items():
                delta = self.lr * advantage * float(value) * (1.0 - gate_value)
                params[feat] = clip(params[feat] + delta, -3.0, 3.0)
        entropy = 0.0
        for g in self._last_gates.values():
            if g > 0:
                entropy -= g * math.log(g)
        self.temperature = clip(self.temperature + 0.05 * (entropy - self.target_entropy), 0.4, 2.5)

    @property
    def last_gates(self) -> Dict[str, float]:
        return dict(self._last_gates)


class AppraisalAggregator:
    def __init__(self, plugins: List[AppraisalPlugin], lr: float = 0.02, w_max: float = 3.0,
                 meta_controller: Optional[PluginMetaController] = None):
        self.plugins: Dict[str, AppraisalPlugin] = {p.name: p for p in plugins}
        self.w: Dict[str, float] = {p.name: 1.0 for p in plugins}
        self.lr = float(lr)
        self.w_max = float(w_max)
        self.meta_controller = meta_controller or PluginMetaController(list(self.plugins))
        self.last_gates: Dict[str, float] = {name: 1.0 / max(1, len(self.plugins)) for name in self.plugins}

    def step(self, ctx: Dict[str, Any], quality: Optional[float]) -> Tuple[float, float, float, Dict[str, float], Dict[str, float]]:
        dv = da = dd = 0.0
        parts: Dict[str, float] = {}
        cache: Dict[str, AppraisalOutput] = {}
        confidences: Dict[str, float] = {}
        for name, p in self.plugins.items():
            out = p(ctx)
            cache[name] = out
            confidences[name] = float(out.confidence)
        gates = self.meta_controller.compute(ctx, confidences)
        for name, out in cache.items():
            gate = gates.get(name, 1.0)
            contrib = gate * float(self.w[name]) * float(out.confidence)
            dv += contrib * out.dv
            da += contrib * out.da
            dd += contrib * out.dd
            parts[name] = contrib * (abs(out.dv) + 0.5 * abs(out.da) + 0.3 * abs(out.dd))
        if quality is not None:
            target = clip(quality, -1.0, 1.0)
            for name, out in cache.items():
                signal = out.dv + 0.5 * out.da + 0.3 * out.dd
                grad = target * signal * out.confidence * gates.get(name, 1.0)
                self.w[name] = clip(self.w[name] + self.lr * grad, 0.0, self.w_max)
        self.meta_controller.update(quality)
        self.last_gates = dict(gates)
        return dv, da, dd, parts, gates


class ContextAutoSynthesizer:
    def __init__(self, max_patterns: int = 12):
        self.max_patterns = max_patterns
        self.pattern_stats: Dict[str, Dict[str, float]] = {}
        self.history: deque = deque(maxlen=200)

    @staticmethod
    def _clean_token(token: str) -> Optional[str]:
        token = "".join(ch for ch in token.lower() if ch.isalpha())
        if len(token) < 4:
            return None
        return token

    def _extract_tokens(self, ctx: Dict[str, Any]) -> List[str]:
        tokens: List[str] = []
        memories = ctx.get("recent_memories") or []
        for mem in memories[:40]:
            if not isinstance(mem, dict):
                continue
            text = str(mem.get("text", ""))
            for raw in text.split():
                tok = self._clean_token(raw)
                if tok:
                    tokens.append(tok)
        return tokens

    def augment(self, ctx: Dict[str, Any]) -> Dict[str, float]:
        features: Dict[str, float] = {}
        memories = ctx.get("recent_memories") or []
        if memories:
            questions = sum("?" in str((m or {}).get("text", "")) for m in memories)
            features["question_ratio"] = questions / max(1.0, len(memories))
            positive = sum("bravo" in str((m or {}).get("text", "")).lower() or "merci" in str((m or {}).get("text", "")).lower()
                           for m in memories)
            negative = sum("erreur" in str((m or {}).get("text", "")).lower() or "fail" in str((m or {}).get("text", "")).lower()
                           for m in memories)
            features["memory_valence"] = clip((positive - negative) / max(1.0, len(memories)), -1.0, 1.0)
        tokens = self._extract_tokens(ctx)
        unique_tokens = set(tokens)
        features["novelty"] = clip(len(unique_tokens) / 30.0, 0.0, 1.0)
        features["token_count"] = clip(len(tokens) / 60.0, 0.0, 1.0)
        self.history.append({"tokens": list(unique_tokens)})
        return features

    def observe(self, ctx: Dict[str, Any], quality: Optional[float]) -> None:
        if quality is None:
            return
        q = clip(float(quality), -1.0, 1.0)
        for token in self._extract_tokens(ctx):
            stats = self.pattern_stats.setdefault(token, {"sum": 0.0, "count": 0.0})
            stats["sum"] += q
            stats["count"] += 1.0
        if len(self.pattern_stats) > self.max_patterns * 2:
            # prune lowest confidence patterns
            sorted_items = sorted(self.pattern_stats.items(), key=lambda kv: abs(kv[1]["sum"]) / max(1.0, kv[1]["count"]))
            for key, _ in sorted_items[: max(0, len(sorted_items) - self.max_patterns)]:
                self.pattern_stats.pop(key, None)

    def build_output(self, ctx: Dict[str, Any]) -> AppraisalOutput:
        tokens = set(self._extract_tokens(ctx))
        dv = da = dd = 0.0
        details: List[Tuple[str, float]] = []
        for token in list(tokens)[:8]:
            stats = self.pattern_stats.get(token)
            if not stats or stats["count"] < 3:
                continue
            mean_reward = clip(stats["sum"] / max(1.0, stats["count"]), -1.0, 1.0)
            weight = mean_reward
            dv += 0.15 * weight
            da += 0.08 * abs(weight)
            dd += 0.05 * weight
            details.append((token, round(mean_reward, 3)))
        confidence = clip(0.25 + 0.05 * len(details), 0.1, 0.8)
        return AppraisalOutput(dv=dv, da=da, dd=dd, confidence=confidence, meta={"patterns": details[:5]})


class SynthesizedPlugin(AppraisalPlugin):
    name = "latent_synth"

    def __init__(self, synthesizer: ContextAutoSynthesizer):
        self._synth = synthesizer

    def __call__(self, ctx: Dict[str, Any]) -> AppraisalOutput:
        return self._synth.build_output(ctx)


class HalfLifePlasticity:
    def __init__(self, episode_half_life: float, mood_half_life: float,
                 min_episode: float = 60.0, max_episode: float = 60.0 * 60.0,
                 min_mood: float = 60.0 * 10.0, max_mood: float = 60.0 * 60.0 * 24.0):
        self.episode_half_life = float(episode_half_life)
        self.mood_half_life = float(mood_half_life)
        self.min_episode = float(min_episode)
        self.max_episode = float(max_episode)
        self.min_mood = float(min_mood)
        self.max_mood = float(max_mood)
        self.history: deque = deque(maxlen=120)
        self._last_adjust = 0.0

    @staticmethod
    def _corr(xs: List[float], ys: List[float]) -> float:
        n = len(xs)
        if n < 3:
            return 0.0
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        den_x = sum((x - mean_x) ** 2 for x in xs)
        den_y = sum((y - mean_y) ** 2 for y in ys)
        denom = math.sqrt(max(den_x, 1e-9) * max(den_y, 1e-9))
        if denom <= 1e-6:
            return 0.0
        return clip(num / denom, -1.0, 1.0)

    def observe(self, now: float, quality: Optional[float], modulators: Optional[Dict[str, Any]]) -> Optional[Tuple[float, float]]:
        if quality is None or modulators is None:
            return None
        entry = {
            "quality": clip(float(quality), -1.0, 1.0),
            "curiosity": float(modulators.get("curiosity_gain", 0.0) or 0.0),
            "focus": float(modulators.get("focus_narrowing", 0.0) or 0.0),
            "activation": float(modulators.get("activation_delta", 0.0) or 0.0),
        }
        self.history.append(entry)
        if now - self._last_adjust < 10.0 or len(self.history) < 12:
            return None
        qualities = [h["quality"] for h in self.history]
        focus = [h["focus"] for h in self.history]
        curiosity = [h["curiosity"] for h in self.history]
        activation = [h["activation"] for h in self.history]
        corr_focus = self._corr(qualities, focus)
        corr_curiosity = self._corr(qualities, curiosity)
        corr_activation = self._corr(qualities, activation)
        changed = False
        if corr_curiosity > 0.15:
            new_ep = clip(self.episode_half_life * (1.0 + 0.08 * corr_curiosity), self.min_episode, self.max_episode)
            if abs(new_ep - self.episode_half_life) > 1e-3:
                self.episode_half_life = new_ep
                changed = True
        elif corr_curiosity < -0.15:
            new_ep = clip(self.episode_half_life * (1.0 + 0.08 * corr_curiosity), self.min_episode, self.max_episode)
            if abs(new_ep - self.episode_half_life) > 1e-3:
                self.episode_half_life = new_ep
                changed = True
        if corr_focus > 0.20:
            new_mood = clip(self.mood_half_life * (1.0 - 0.05 * corr_focus), self.min_mood, self.max_mood)
            if abs(new_mood - self.mood_half_life) > 1e-3:
                self.mood_half_life = new_mood
                changed = True
        elif corr_activation < -0.20:
            new_mood = clip(self.mood_half_life * (1.0 + 0.04 * corr_activation), self.min_mood, self.max_mood)
            if abs(new_mood - self.mood_half_life) > 1e-3:
                self.mood_half_life = new_mood
                changed = True
        if changed:
            self._last_adjust = now
            return self.episode_half_life, self.mood_half_life
        return None


class RitualPlanner:
    def __init__(self, cooldown: float = 180.0):
        self.cooldown = float(cooldown)
        self._last_trigger = 0.0
        self.trace: deque = deque(maxlen=20)

    def observe(self, now: float, quality: Optional[float], state: AffectState, mood: MoodState) -> None:
        self.trace.append({
            "t": now,
            "valence": float(state.valence),
            "arousal": float(state.arousal),
            "quality": None if quality is None else float(quality),
            "mood_valence": float(mood.valence),
        })

    def maybe_plan(self, now: float) -> Optional[Dict[str, Any]]:
        if now - self._last_trigger < self.cooldown or len(self.trace) < 4:
            return None
        recent = list(self.trace)[-4:]
        if all(item["valence"] < -0.35 for item in recent):
            self._last_trigger = now
            return {
                "kind": "self_soothing",
                "intensity": 0.55,
                "valence_hint": 0.6,
                "arousal_hint": -0.25,
                "dominance_hint": 0.15,
                "meta": {"ritual": "breathing_reset"},
            }
        if all(item["arousal"] > 0.75 for item in recent):
            self._last_trigger = now
            return {
                "kind": "grounding_focus",
                "intensity": 0.50,
                "valence_hint": 0.2,
                "arousal_hint": -0.4,
                "dominance_hint": 0.2,
                "meta": {"ritual": "micro_grounding"},
            }
        return None

# ========================= Labelling ========================= #

EMO_LABELS = [
    ("elated",       (0.6, 0.6, 0.5)),
    ("excited",      (0.5, 0.7, 0.5)),
    ("content",      (0.4, 0.3, 0.5)),
    ("calm",         (0.2, 0.2, 0.5)),
    ("neutral",      (0.0, 0.2, 0.5)),
    ("tense",        (-0.2, 0.6, 0.4)),
    ("frustrated",   (-0.5, 0.6, 0.4)),
    ("sad",          (-0.5, 0.3, 0.4)),
    ("bored",        (-0.2, 0.1, 0.5)),
]


def label_from_pad(v: float, a: float, d: float) -> str:
    best, bd = "neutral", 9e9
    for name, (lv, la, ld) in EMO_LABELS:
        dist = (v - lv) ** 2 + (a - la) ** 2 + 0.2 * (d - ld) ** 2
        if dist < bd: best, bd = name, dist
    return best


# ========================= Moteur principal ========================= #

class EmotionEngine:
    """Moteur émotionnel évolutif — **compat 100%** avec l'existant.

    API stable: bind(), register_event(), step(), get_modulators(), get_state(),
                update_from_recent_memories(), modulate_homeostasis().
    """
    def __init__(self,
                 path_state: str = "data/mood.json",
                 path_dashboard: str = "data/mood_dashboard.json",
                 path_log: str = "data/mood_episodes.jsonl",
                 half_life_sec: float = 60.0 * 15.0,
                 step_period: float = 2.0,
                 mood_half_life_sec: float = 60.0 * 60.0 * 6.0,  # 6h
                 decay_target: str = "mood",  # "mood" (défaut) ou "neutral"
                 seed: Optional[int] = None):
        os.makedirs(os.path.dirname(path_state) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(path_dashboard) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(path_log) or ".", exist_ok=True)
        self.path_state = path_state
        self.path_dashboard = path_dashboard
        self.path_log = path_log

        self.state = AffectState()
        self.mood = MoodState()
        self._last_step = 0.0
        self.half_life_sec = float(half_life_sec)
        self.step_period = float(step_period)
        self.mood_half_life_sec = float(mood_half_life_sec)
        self.decay_target = decay_target
        self.rng = (seed and int(seed)) or None

        # Plugins & agrégateur
        self._synthesizer = ContextAutoSynthesizer()
        plugins: List[AppraisalPlugin] = [
            CognitiveLoadPlugin(),
            ErrorPlugin(),
            SuccessPlugin(),
            RewardPlugin(),
            IntrinsicPleasurePlugin(),
            FatiguePlugin(),
            SocialFeedbackPlugin(),
            SynthesizedPlugin(self._synthesizer),
        ]
        self.aggregator = AppraisalAggregator(plugins)

        # Plasticité multi-échelles + rituels
        self._plasticity = HalfLifePlasticity(self.half_life_sec, self.mood_half_life_sec)
        self._rituals = RitualPlanner()

        # Liaison vers d'autres modules
        self.bound: Dict[str, Any] = {
            "arch": None,
            "memory": None,
            "metacog": None,
            "goals": None,
            "language": None,
            "evolution": None,
            "phenomenal_journal": None,
        }

        # Épisodes récents
        self._recent_episodes: List[EmotionEpisode] = []
        self.max_recent_episodes = 200

        # Cache modulators
        self.last_modulators: Dict[str, Any] = {}
        self._last_llm_annotation: Optional[Dict[str, Any]] = None

        # Charger si présent
        self.load()

    # ---------- Binding ----------
    def bind(self, arch=None, memory=None, metacog=None, goals=None, language=None, evolution=None):
        self.bound.update({
            "arch": arch,
            "memory": memory,
            "metacog": metacog,
            "goals": goals,
            "language": language,
            "evolution": evolution,
        })
        try:
            if arch is not None and getattr(arch, "phenomenal_journal", None) is not None:
                self.bound["phenomenal_journal"] = getattr(arch, "phenomenal_journal")
        except Exception:
            pass
        return self

    def _phenomenal_journal(self) -> Optional["PhenomenalJournal"]:
        journal = self.bound.get("phenomenal_journal")
        if journal is None:
            arch = self.bound.get("arch")
            if arch is not None:
                journal = getattr(arch, "phenomenal_journal", None)
                if journal is not None:
                    self.bound["phenomenal_journal"] = journal
        return journal

    def _emit_phenomenal_experience(self, experience: Any, context: Optional[Dict[str, Any]]) -> None:
        journal = self._phenomenal_journal()
        if journal is None:
            return
        try:
            arch = self.bound.get("arch")
            values: List[str] = []
            principles: List[str] = []
            if arch is not None and hasattr(arch, "self_model"):
                self_model = getattr(arch, "self_model", None)
                persona = getattr(self_model, "persona", {}) if self_model else {}
                if isinstance(persona, dict):
                    raw_values = persona.get("values")
                    if isinstance(raw_values, list):
                        values.extend(str(val) for val in raw_values if isinstance(val, str))
                identity = getattr(self_model, "identity", {}) if self_model else {}
                if isinstance(identity, dict):
                    declared = identity.get("principles")
                    if isinstance(declared, list):
                        principles.extend(str(val) for val in declared if isinstance(val, str))
            journal.record_emotion(
                experience,
                context=context or {},
                values=values,
                principles=principles,
            )
        except Exception:
            pass

    # ---------- API externe ----------
    def register_event(self, kind: str, intensity: float = 0.4,
                       valence_hint: Optional[float] = None,
                       arousal_hint: Optional[float] = None,
                       dominance_hint: Optional[float] = None,
                       confidence: float = 1.0,
                       meta: Optional[Dict[str, Any]] = None):
        m = float(clip(intensity, 0.0, 1.0))
        dv = (valence_hint if valence_hint is not None else 0.0) * m
        da = (arousal_hint if arousal_hint is not None else 0.0) * m
        dd = (dominance_hint if dominance_hint is not None else 0.0) * 0.5 * m
        self._nudge(dv, da, dd, source=f"event:{kind}", confidence=confidence, meta=meta or {})

    def register_intrinsic_pleasure(self, intensity: float, meta: Optional[Dict[str, Any]] = None) -> None:
        try:
            value = float(intensity)
        except (TypeError, ValueError):
            value = 0.0
        value = clip(value, -1.0, 1.0)
        self.register_event(
            "intrinsic_pleasure",
            intensity=abs(value),
            valence_hint=value,
            arousal_hint=0.2 * abs(value),
            dominance_hint=0.1 * value,
            confidence=0.6,
            meta=meta or {},
        )

    def get_modulators(self) -> Dict[str, Any]:
        self.step(force=True)
        if not self.last_modulators:
            self.last_modulators = self._compute_modulators()
        return dict(self.last_modulators)

    def get_state(self) -> Dict[str, Any]:
        return {
            "episode": asdict(self.state),
            "mood": asdict(self.mood),
            "uncertainty": self._estimate_uncertainty(),
            "label": self.state.label,
            "weights": dict(self.aggregator.w),
            "gates": dict(self.aggregator.last_gates),
            "half_life": self.half_life_sec,
            "mood_half_life": self.mood_half_life_sec,
            "latent_patterns": {
                token: {
                    "score": clip(stats.get("sum", 0.0) / max(1.0, stats.get("count", 0.0)), -1.0, 1.0),
                    "count": stats.get("count", 0.0),
                }
                for token, stats in list(self._synthesizer.pattern_stats.items())[:10]
            },
            "recent_causes": [
                {"id": e.id, "label": e.label, "causes": e.causes[-3:], "dv": e.dv, "da": e.da, "dd": e.dd, "confidence": e.confidence, "dt": e.dt}
                for e in self._recent_episodes[-5:]
            ],
        }

    def get_affect(self) -> Dict[str, float]:
        self.step(force=True)
        return {
            "valence": float(self.state.valence),
            "arousal": float(self.state.arousal),
            "dominance": float(self.state.dominance),
            "label": self.state.label,
        }

    # ---------- Compat: helpers hérités ----------
    def update_from_recent_memories(self, recent: List[Dict[str, Any]]):
        if not isinstance(recent, list):
            recent = []
        positive = ("bravo", "merci", "good", "bien")
        negative = ("erreur", "fail", "mauvais", "wrong", "error")
        for m in recent:
            text = str((m or {}).get("text", "")).lower()
            kind = str((m or {}).get("kind", "")).lower()
            if any(t in text for t in positive):
                self.register_event("positive_feedback", 0.5, valence_hint=+0.7, arousal_hint=+0.2, dominance_hint=+0.1)
            if "error" in kind or any(t in text for t in negative):
                self.register_event("error_feedback", 0.6, valence_hint=-0.7, arousal_hint=+0.3, dominance_hint=-0.2)
            if "?" in text:
                self.register_event("curiosity_signal", 0.3, valence_hint=+0.1, arousal_hint=+0.2)
        self.step(force=True)

    def modulate_homeostasis(self, homeostasis) -> None:
        if homeostasis is None:
            return
        self.step(force=True)
        mods = self.last_modulators or self._compute_modulators()
        curiosity_boost = float(mods.get("curiosity_gain", 0.0))
        activation_delta = float(mods.get("activation_delta", 0.0))
        def _current(name: str) -> float:
            try:
                return float(homeostasis.state.get("drives", {}).get(name, 0.5))
            except Exception:
                return 0.5
        def _apply(name: str, delta: float) -> None:
            try:
                if hasattr(homeostasis, "adjust_drive"):
                    homeostasis.adjust_drive(name, delta)
                    return
            except Exception:
                pass
            # fallback direct
            drives_dict = homeostasis.state.setdefault("drives", {})
            drives_dict[name] = clip(_current(name) + delta, 0.0, 1.0)
            if hasattr(homeostasis, "_save"):
                try: homeostasis._save()
                except Exception: pass
        kernel_state = mods.get("phenomenal_kernel")
        if not kernel_state:
            arch = self.bound.get("arch")
            if arch is not None:
                try:
                    kernel_state = getattr(arch, "phenomenal_kernel_state", None)
                except Exception:
                    kernel_state = None
        slowdown_meta: Dict[str, Any] = {}
        slowdown_val = None
        if isinstance(kernel_state, dict):
            slowdown_val = kernel_state.get("global_slowdown")
            slowdown_meta = {
                "mode": kernel_state.get("mode") or kernel_state.get("mode_suggestion") or "travail",
                "energy": kernel_state.get("energy"),
                "feel_like": kernel_state.get("feel_like"),
                "budget": kernel_state.get("flanerie_budget_remaining"),
            }
        if isinstance(kernel_state, dict) and hasattr(homeostasis, "register_hedonic_state"):
            hedonic_val = kernel_state.get("hedonic_reward")
            if isinstance(hedonic_val, (int, float)) and abs(hedonic_val) > 1e-3:
                mode = slowdown_meta.get("mode", "travail")
                meta = {
                    "energy": kernel_state.get("energy"),
                    "feel_like": kernel_state.get("feel_like"),
                    "budget": kernel_state.get("flanerie_budget_remaining"),
                }
                slowdown_meta.update({k: v for k, v in meta.items() if v is not None})
                try:
                    homeostasis.register_hedonic_state(float(hedonic_val), mode=mode, meta=meta)
                except Exception:
                    pass
        if isinstance(slowdown_val, (int, float)) and hasattr(homeostasis, "register_global_slowdown"):
            try:
                homeostasis.register_global_slowdown(float(slowdown_val), meta=slowdown_meta)
            except Exception:
                pass
        _apply("curiosity", +0.05 + 0.10 * curiosity_boost)
        _apply("task_activation", activation_delta)

    # ---------- Tick principal ----------
    def step(self, force: bool = False, quality: Optional[float] = None, now: Optional[float] = None):
        now = float(now or time.time())
        if not force and (now - self._last_step) < self.step_period:
            return

        # 1) Décroissance épisode → baseline (mood ou neutral selon flag)
        self._decay_episode(now)

        # 2) Contexte + appraisal via agrégateur (avec proxy qualité)
        ctx = self._collect_context()
        quality_signal = quality if quality is not None else self._proxy_quality(ctx)
        dv, da, dd, parts, gates = self.aggregator.step(ctx, quality=quality_signal)
        self._synthesizer.observe(ctx, quality_signal)

        dv, da, dd, llm_annotation = self._llm_appraise(ctx, dv, da, dd)
        self._last_llm_annotation = llm_annotation

        meta_payload = {"parts": parts, "ctx_keys": list(ctx.keys()), "gates": gates}
        if llm_annotation:
            meta_payload["llm"] = llm_annotation

        # 3) Appliquer delta + journaliser
        self._nudge(dv, da, dd, source="aggregator", confidence=self._confidence_from_ctx(ctx),
                    meta=meta_payload)

        # 3b) Rituels potentiels (après nudge pour état à jour)
        self._rituals.observe(now, quality_signal, self.state, self.mood)

        # 4) Mise à jour humeur (filtre lent)
        self._update_mood(now)

        # 5) Recalcul modulateurs & dispatch
        self.last_modulators = self._compute_modulators()
        if llm_annotation:
            self.last_modulators.setdefault("llm_annotation", llm_annotation)
            suggestion = llm_annotation.get("regulation_suggestion") if isinstance(llm_annotation, dict) else None
            if suggestion and isinstance(suggestion, str):
                self.last_modulators.setdefault("regulation_suggestion", suggestion)
        adjust = self._plasticity.observe(now, quality_signal, self.last_modulators)
        if adjust:
            self.half_life_sec = self._plasticity.episode_half_life
            self.mood_half_life_sec = self._plasticity.mood_half_life
            self._emit_plasticity_annotation(now)
        self._maybe_trigger_ritual(now)
        self._dispatch_modulators(self.last_modulators)

        # 6) Persistance légère
        self.save()
        self._last_step = now

    # ========================= Interne ========================= #
    def _llm_appraise(
        self,
        ctx: Dict[str, Any],
        dv: float,
        da: float,
        dd: float,
    ) -> Tuple[float, float, float, Optional[Dict[str, Any]]]:
        if not is_llm_enabled():
            return dv, da, dd, None

        payload = {
            "state": {
                "valence": self.state.valence,
                "arousal": self.state.arousal,
                "dominance": self.state.dominance,
            },
            "aggregated_delta": {"dv": dv, "da": da, "dd": dd},
            "context_keys": list(ctx.keys())[:24],
        }

        try:
            response = get_llm_manager().call_dict(
                "emotion_engine",
                input_payload=payload,
            )
        except (LLMUnavailableError, LLMIntegrationError):
            LOGGER.debug("LLM emotion engine unavailable", exc_info=True)
            return dv, da, dd, None

        if not isinstance(response, dict):
            return dv, da, dd, None

        emotions = response.get("emotions")
        annotation: Dict[str, Any] = {
            "emotions": emotions,
            "regulation_suggestion": response.get("regulation_suggestion"),
            "notes": response.get("notes"),
        }

        if not isinstance(emotions, list) or not emotions:
            return dv, da, dd, annotation

        valence_score = 0.0
        total_intensity = 0.0
        for entry in emotions:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "").lower()
            intensity = float(entry.get("intensity", 0.0) or 0.0)
            total_intensity += intensity
            if name in _POSITIVE_EMOTIONS:
                valence_score += intensity
            elif name in _NEGATIVE_EMOTIONS:
                valence_score -= intensity
            else:
                valence_score += intensity * (1.0 if dv >= 0 else -1.0) * 0.25

        valence_score = clip(valence_score, -1.0, 1.0)
        arousal_level = clip(total_intensity / max(1.0, len(emotions)), 0.0, 1.0)
        dominance_delta = clip(0.35 * valence_score, -0.6, 0.6)

        agg_valence = clip(self.state.valence + dv, -1.0, 1.0)
        agg_arousal = clip(self.state.arousal + da, 0.0, 1.0)
        agg_dominance = clip(self.state.dominance + dd, 0.0, 1.0)

        llm_valence = clip(self.state.valence + valence_score, -1.0, 1.0)
        llm_arousal = clip(0.5 * self.state.arousal + 0.5 * arousal_level, 0.0, 1.0)
        llm_dominance = clip(self.state.dominance + dominance_delta, 0.0, 1.0)

        combined_valence = clip(0.6 * agg_valence + 0.4 * llm_valence, -1.0, 1.0)
        combined_arousal = clip(0.5 * agg_arousal + 0.5 * llm_arousal, 0.0, 1.0)
        combined_dominance = clip(0.6 * agg_dominance + 0.4 * llm_dominance, 0.0, 1.0)

        new_dv = clip(combined_valence - self.state.valence, -1.0, 1.0)
        new_da = clip(combined_arousal - self.state.arousal, -1.0, 1.0)
        new_dd = clip(combined_dominance - self.state.dominance, -1.0, 1.0)

        return new_dv, new_da, new_dd, annotation

    def _collect_context(self) -> Dict[str, Any]:
        ctx: Dict[str, Any] = {}
        metacog = self.bound.get("metacog")
        memory = self.bound.get("memory")
        language = self.bound.get("language")
        arch = self.bound.get("arch")
        if metacog is not None:
            ctx.setdefault("cognitive_load", getattr(metacog, "load", None))
            # fatigue (si disponible)
            try:
                mon = getattr(getattr(metacog, "cognitive_monitoring", {}), "resource_monitor", None)
                if mon and hasattr(mon, "assess_fatigue"):
                    ctx.setdefault("fatigue", float(clip(mon.assess_fatigue(getattr(metacog, "metacognitive_history", None), arch), 0.0, 1.0)))
            except Exception:
                pass
            ctx.setdefault("recent_errors", getattr(metacog, "recent_errors", None))
            ctx.setdefault("reward_signal", getattr(metacog, "reward_signal", None))
            ctx.setdefault("recent_success", getattr(metacog, "recent_success", None))
        if memory is not None and hasattr(memory, "get_recent_memories"):
            try: ctx.setdefault("recent_memories", memory.get_recent_memories(n=40))
            except Exception: pass
        if language is not None:
            ctx.setdefault("social_cues", getattr(language, "recent_user_cues", None))
        kernel_state = None
        if arch is not None:
            try:
                kernel_state = getattr(arch, "phenomenal_kernel_state", None)
                if not kernel_state:
                    kernel_state = getattr(arch, "_phenomenal_kernel_state", None)
            except Exception:
                kernel_state = None
        if isinstance(kernel_state, dict) and kernel_state:
            ctx.setdefault("phenomenal_kernel", dict(kernel_state))
            hedonic_val = kernel_state.get("hedonic_reward")
            if "hedonic_signal" not in ctx and isinstance(hedonic_val, (int, float)):
                ctx["hedonic_signal"] = float(hedonic_val)
            if "fatigue" not in ctx and isinstance(kernel_state.get("fatigue"), (int, float)):
                ctx["fatigue"] = float(kernel_state.get("fatigue"))
        ctx.setdefault("time_of_day", time.localtime().tm_hour)
        ctx = {k: v for k, v in ctx.items() if v is not None}
        latent = self._synthesizer.augment(ctx)
        if latent:
            ctx["_latent_features"] = latent
        return ctx

    def _confidence_from_ctx(self, ctx: Dict[str, Any]) -> float:
        filled = sum(1 for _ in ctx.items())
        return clip(0.2 + 0.05 * filled, 0.2, 1.0)

    def _proxy_quality(self, ctx: Dict[str, Any]) -> Optional[float]:
        succ = float(ctx.get("recent_success", 0.0) or 0.0)
        errs = float(len(ctx.get("recent_errors", []) or []))
        fatigue = float(ctx.get("fatigue", 0.0) or 0.0)
        return clip(succ - 0.2 * errs - 0.3 * fatigue, -1.0, 1.0)

    def _decay_episode(self, now: float):
        dt = max(0.001, now - self.state.t)
        hl = max(1e-3, self.half_life_sec)
        k = 0.6931 / hl
        def towards(curr, base):
            return base + (curr - base) * math.exp(-k * dt)
        if self.decay_target == "neutral":
            base_v, base_a, base_d = 0.0, 0.2, 0.5
        else:  # mood
            base_v, base_a, base_d = self.mood.valence, max(0.05, self.mood.arousal), self.mood.dominance
        self.state.valence = clip(towards(self.state.valence, base_v), -1.0, 1.0)
        self.state.arousal = clip(towards(self.state.arousal, base_a), 0.0, 1.0)
        self.state.dominance = clip(towards(self.state.dominance, base_d), 0.0, 1.0)
        self.state.t = now
        self.state.label = label_from_pad(self.state.valence, self.state.arousal, self.state.dominance)

    def _update_mood(self, now: float):
        dt = max(0.001, now - self.mood.t)
        hl = max(1e-3, self.mood_half_life_sec)
        k = 0.6931 / hl
        def lowpass(m, s):
            return m + (s - m) * (1.0 - math.exp(-k * dt))
        self.mood.valence = clip(lowpass(self.mood.valence, self.state.valence), -1.0, 1.0)
        self.mood.arousal = clip(lowpass(self.mood.arousal, self.state.arousal * 0.8), 0.05, 1.0)
        self.mood.dominance = clip(lowpass(self.mood.dominance, self.state.dominance), 0.0, 1.0)
        self.mood.t = now

    def _estimate_uncertainty(self) -> float:
        base = clip(0.6 * self.state.arousal + 0.4 * (1.0 - self.state.dominance), 0.0, 1.0)
        return clip(0.5 * base + 0.5 * (0.5 - 0.5 * self.mood.dominance), 0.0, 1.0)

    def _nudge(self, dv: float, da: float, dd: float, source: str, confidence: float, meta: Dict[str, Any]):
        self.state.valence = clip(self.state.valence + dv, -1.0, 1.0)
        self.state.arousal = clip(self.state.arousal + da, 0.0, 1.0)
        self.state.dominance = clip(self.state.dominance + dd, 0.0, 1.0)
        self.state.label = label_from_pad(self.state.valence, self.state.arousal, self.state.dominance)
        ep = EmotionEpisode(
            id=str(uuid.uuid4()), onset=time.time(), dt=self.step_period,
            dv=dv, da=da, dd=dd, label=self.state.label, confidence=confidence,
            causes=sorted([(k, float(v)) for k, v in (meta.get("parts") or {}).items()], key=lambda x: -x[1])[:5],
            meta={k: v for k, v in meta.items() if k != "parts"}
        )
        self._recent_episodes.append(ep)
        if len(self._recent_episodes) > self.max_recent_episodes:
            self._recent_episodes = self._recent_episodes[-self.max_recent_episodes:]
        self._emit_phenomenal_experience(ep, {"source": source, **meta})
        try:
            with open(self.path_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(json_sanitize(asdict(ep)), ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _goal_priority_bias_dict(self, v: float, a: float, d: float) -> Dict[str, float]:
        """Reconstruit un dict de biais par domaines (compat ancien code)."""
        bias: Dict[str, float] = {}
        # heuristiques simples mais stables
        if v < -0.2:
            bias["résolution_problème"] = bias.get("résolution_problème", 0.0) + 0.15
            bias["apprentissage"] = bias.get("apprentissage", 0.0) + 0.05
        if a > 0.6:
            bias["attention"] = bias.get("attention", 0.0) + 0.15
            bias["prise_décision"] = bias.get("prise_décision", 0.0) + 0.10
        if v > 0.3:
            bias["social_cognition"] = bias.get("social_cognition", 0.0) + 0.10
            bias["langage"] = bias.get("langage", 0.0) + 0.05
        # clé globale pour consommateurs génériques
        bias["global"] = clip(0.15 * v + 0.10 * (d - 0.5), -0.3, 0.3)
        return bias

    def _compute_modulators(self) -> Dict[str, Any]:
        v, a, d = self.state.valence, self.state.arousal, self.state.dominance
        unc = self._estimate_uncertainty()

        curiosity_gain = clip(0.2 + 0.4 * max(0.0, v) + 0.2 * (1.0 - unc), 0.0, 1.0)
        exploration_rate = clip(0.15 + 0.5 * unc + 0.2 * a, 0.0, 1.0)
        focus_narrowing = clip(0.3 + 0.5 * a - 0.3 * unc, 0.0, 1.0)

        tone = {
            "warmth": clip(0.55 + 0.35 * v - 0.15 * unc, 0.0, 1.0),
            "energy": clip(0.30 + 0.60 * a, 0.0, 1.0),
            "assertiveness": clip(0.30 + 0.50 * d - 0.25 * unc, 0.0, 1.0),
            "hedging": clip(0.25 + 0.50 * unc - 0.15 * d, 0.0, 1.0),
        }
        # Safety tone gate
        if d < 0.4 and unc > 0.5:
            tone["assertiveness"] = min(tone["assertiveness"], 0.45)

        activation_delta = clip(0.25 * a + 0.15 * (v - 0.2), -0.3, 0.7)
        goal_priority_bias_scalar = clip(0.15 * v + 0.10 * (d - 0.5), -0.3, 0.3)
        goal_priority_bias = self._goal_priority_bias_dict(v, a, d)

        mods = {
            "curiosity_gain": curiosity_gain,
            "exploration_rate": exploration_rate,
            "focus_narrowing": focus_narrowing,
            "tone": tone,
            "language_tone": dict(tone),  # alias compat
            "activation_delta": activation_delta,
            "goal_priority_bias": goal_priority_bias,  # dict (compat)
            "goal_priority_bias_scalar": goal_priority_bias_scalar,  # scalaire (nouveau)
            "uncertainty": unc,
            "label": self.state.label,  # alias compat si l'ancien code lisait label ici
        }
        arch = self.bound.get("arch")
        kernel_state = None
        if arch is not None:
            try:
                kernel_state = getattr(arch, "phenomenal_kernel_state", None)
            except Exception:
                kernel_state = None
        if isinstance(kernel_state, dict) and kernel_state:
            mods["phenomenal_kernel"] = dict(kernel_state)
            if "hedonic_reward" in kernel_state:
                try:
                    mods["hedonic_reward"] = float(kernel_state["hedonic_reward"])
                except Exception:
                    pass
        return mods

    def _dispatch_modulators(self, mods: Dict[str, Any]):
        arch = self.bound.get("arch")
        lang = self.bound.get("language")
        goals = self.bound.get("goals")

        def call(obj, name: str, *args, **kwargs):
            try:
                fn = getattr(obj, name, None)
                if callable(fn): fn(*args, **kwargs)
            except Exception:
                pass

        # Hook générique
        if arch is not None:
            call(arch, "on_affect_modulators", mods)

        # **Compat 1**: Ton — set_style_hints (ancien) OU set_tone_param (nouveau)
        tone = mods.get("language_tone") or mods.get("tone") or {}
        if lang is not None:
            if hasattr(lang, "set_style_hints"):
                try: lang.set_style_hints(tone)
                except Exception: pass
            elif hasattr(lang, "set_tone_param"):
                for k, v in tone.items():
                    try: lang.set_tone_param(k, v)
                    except Exception: pass

        # **Compat 2**: Goal priority — dict (apply_emotional_bias) OU scalaire (set_priority_bias)
        if goals is not None:
            if hasattr(goals, "apply_emotional_bias"):
                try: goals.apply_emotional_bias(mods.get("goal_priority_bias", {}), mods.get("curiosity_gain", 0.0))
                except Exception: pass
            elif hasattr(goals, "set_priority_bias"):
                try: goals.set_priority_bias(float(mods.get("goal_priority_bias_scalar", 0.0)))
                except Exception: pass

        # **Compat 3**: Global activation bump explicite
        if arch is not None and hasattr(arch, "bump_global_activation"):
            if not mods.get("_activation_handled"):
                try:
                    arch.bump_global_activation(float(mods.get("activation_delta", 0.0)))
                except Exception:
                    pass
        elif arch is not None and hasattr(arch, "global_activation"):
            try:
                ga = float(getattr(arch, "global_activation", 0.5))
                ga = clip(ga + float(mods.get("activation_delta", 0.0)), 0.0, 1.0)
                setattr(arch, "global_activation", ga)
            except Exception:
                pass

    def _emit_plasticity_annotation(self, now: float) -> None:
        note = {
            "ts": now,
            "episode_half_life": round(self._plasticity.episode_half_life, 3),
            "mood_half_life": round(self._plasticity.mood_half_life, 3),
        }
        evolution = self.bound.get("evolution")
        if evolution is not None and hasattr(evolution, "state"):
            try:
                extra = evolution.state.setdefault("history_extra", {})
                bucket = extra.setdefault("emotion_plasticity", [])
                bucket.append(note)
                extra["emotion_plasticity"] = bucket[-120:]
            except Exception:
                pass
        memory = self.bound.get("memory")
        if memory is not None and hasattr(memory, "add_memory"):
            try:
                memory.add_memory({
                    "kind": "emotion_plasticity",
                    "content": "Ajustement demi-vies émotionnelles",
                    "metadata": note,
                })
            except Exception:
                pass

    def _emit_ritual_memory(self, plan: Dict[str, Any], now: float) -> None:
        memory = self.bound.get("memory")
        if memory is None or not hasattr(memory, "add_memory"):
            return
        try:
            memory.add_memory({
                "kind": "emotion_ritual",
                "content": plan.get("meta", {}).get("ritual", plan.get("kind", "ritual")),
                "metadata": {
                    "plan": {k: v for k, v in plan.items() if k != "meta"},
                    "ts": now,
                },
            })
        except Exception:
            pass

    def _maybe_trigger_ritual(self, now: float) -> None:
        plan = self._rituals.maybe_plan(now)
        if not plan:
            return
        meta = dict(plan.get("meta") or {})
        meta.setdefault("source", "EmotionEngine.ritual")
        self.register_event(
            plan.get("kind", "ritual"),
            intensity=float(plan.get("intensity", 0.4)),
            valence_hint=plan.get("valence_hint"),
            arousal_hint=plan.get("arousal_hint"),
            dominance_hint=plan.get("dominance_hint"),
            confidence=0.8,
            meta=meta,
        )
        self._emit_ritual_memory(plan, now)

    # ========================= Persistance ========================= #
    def save(self):
        payload = {
            "state": asdict(self.state),
            "mood": asdict(self.mood),
            "weights": dict(self.aggregator.w),
            "last_modulators": self.last_modulators,
            "t": time.time(),
            "half_life_sec": self.half_life_sec,
            "mood_half_life_sec": self.mood_half_life_sec,
            "meta": {
                "temperature": getattr(self.aggregator.meta_controller, "temperature", None),
                "params": {
                    name: dict(params)
                    for name, params in getattr(self.aggregator.meta_controller, "params", {}).items()
                },
                "gates": dict(getattr(self.aggregator, "last_gates", {})),
            },
            "latent_patterns": {
                key: {"sum": val.get("sum", 0.0), "count": val.get("count", 0.0)}
                for key, val in self._synthesizer.pattern_stats.items()
            },
            "plasticity": {
                "episode": self._plasticity.episode_half_life,
                "mood": self._plasticity.mood_half_life,
            },
        }
        try:
            with open(self.path_state, "w", encoding="utf-8") as f:
                json.dump(json_sanitize(payload), f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        dash = {
            "t": time.time(),
            "label": self.state.label,
            "v": round(self.state.valence, 3),
            "a": round(self.state.arousal, 3),
            "d": round(self.state.dominance, 3),
            "m_v": round(self.mood.valence, 3),
            "m_a": round(self.mood.arousal, 3),
            "m_d": round(self.mood.dominance, 3),
            "unc": round(self._estimate_uncertainty(), 3),
            "weights": dict(self.aggregator.w),
            "half_life": round(self.half_life_sec, 2),
            "mood_half_life": round(self.mood_half_life_sec, 2),
            "gates": {
                name: round(float(val), 4)
                for name, val in self.aggregator.last_gates.items()
            },
        }
        try:
            with open(self.path_dashboard, "w", encoding="utf-8") as f:
                json.dump(json_sanitize(dash), f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def load(self):
        try:
            if os.path.exists(self.path_state):
                with open(self.path_state, "r", encoding="utf-8") as f:
                    data = json.load(f)
                st = data.get("state", {}) or data.get("episode", {}) or {}
                md = data.get("mood", {})
                self.state = AffectState(
                    t=float(st.get("t", time.time())),
                    valence=float(st.get("valence", NEUTRAL["valence"])),
                    arousal=float(st.get("arousal", NEUTRAL["arousal"])),
                    dominance=float(st.get("dominance", NEUTRAL["dominance"])),
                    label=str(st.get("label", "neutral")),
                )
                self.mood = MoodState(
                    t=float(md.get("t", time.time())),
                    valence=float(md.get("valence", NEUTRAL["valence"])),
                    arousal=float(md.get("arousal", 0.15)),
                    dominance=float(md.get("dominance", NEUTRAL["dominance"])),
                )
                w = data.get("weights", {}) or data.get("state", {}).get("weights", {})
                for k, v in w.items():
                    if k in self.aggregator.w:
                        self.aggregator.w[k] = float(v)
                if "half_life_sec" in data:
                    self.half_life_sec = float(data.get("half_life_sec", self.half_life_sec))
                if "mood_half_life_sec" in data:
                    self.mood_half_life_sec = float(data.get("mood_half_life_sec", self.mood_half_life_sec))
                self._plasticity.episode_half_life = self.half_life_sec
                self._plasticity.mood_half_life = self.mood_half_life_sec
                meta = data.get("meta", {})
                params = meta.get("params", {})
                if isinstance(params, dict):
                    for name, feats in params.items():
                        bucket = self.aggregator.meta_controller.params.setdefault(name, defaultdict(float))
                        if isinstance(feats, dict):
                            for feat, value in feats.items():
                                bucket[feat] = float(value)
                temp = meta.get("temperature")
                if temp is not None:
                    self.aggregator.meta_controller.temperature = float(temp)
                gates = meta.get("gates")
                if isinstance(gates, dict):
                    self.aggregator.last_gates = {k: float(v) for k, v in gates.items()}
                latent = data.get("latent_patterns")
                if isinstance(latent, dict):
                    self._synthesizer.pattern_stats = {
                        str(k): {
                            "sum": float(v.get("sum", 0.0)),
                            "count": float(v.get("count", 0.0)),
                        }
                        for k, v in latent.items()
                        if isinstance(v, dict)
                    }
                plasticity = data.get("plasticity")
                if isinstance(plasticity, dict):
                    ep = plasticity.get("episode")
                    md = plasticity.get("mood")
                    if ep is not None:
                        self._plasticity.episode_half_life = float(ep)
                        self.half_life_sec = float(ep)
                    if md is not None:
                        self._plasticity.mood_half_life = float(md)
                        self.mood_half_life_sec = float(md)
        except Exception:
            # fallback silencieux
            self.state = AffectState()
            self.mood = MoodState()


# ========================= Cli léger ========================= #
if __name__ == "__main__":
    eng = EmotionEngine()
    # Exemple de bind minimal tolérant
    class _Meta: load=0.6; recent_errors=[1,2]; recent_success=0.3; reward_signal=0.1
    eng.bind(metacog=_Meta())
    for _ in range(5):
        eng.step(force=True)
        print(eng.get_modulators())
