from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Tuple, Callable
import logging
import math
import re
import time
import json
import os
from collections import deque

from AGI_Evolutive.utils.jsonsafe import json_sanitize
from AGI_Evolutive.goals import GoalType
from AGI_Evolutive.goals.curiosity import CuriosityEngine
from AGI_Evolutive.utils.llm_service import try_call_llm_dict


logger = logging.getLogger(__name__)


@dataclass
class RewardEvent:
    timestamp: float
    user_id: str
    text: str
    channel: str               # "chat", "system", "tool"
    extrinsic_reward: float    # [-1.0, 1.0]
    polarity: str              # "positive" | "negative" | "neutral"
    intensity: float           # [0, 1] (force du signal)
    features: Dict[str, Any]   # détails (emojis, !, lexiques, etc.)
    context: Dict[str, Any]    # ex: last_assistant_output, active_goal_id, emotional_state


class OnlineCalibratedClassifier:
    """Perceptron-like online classifier with Platt-style calibration."""

    def __init__(
        self,
        state_path: str,
        learning_rate: float = 0.1,
        temperature: float = 1.0,
    ) -> None:
        self.state_path = state_path
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.weights: Dict[str, float] = {}
        self.bias: float = 0.0
        self.example_count: int = 0
        self._load_state()

    def predict(self, features: Dict[str, float]) -> Tuple[float, float, float]:
        z = self.bias
        for name, value in features.items():
            z += self.weights.get(name, 0.0) * value
        calibrated_prob = self._sigmoid(self._apply_temperature(z))
        reward = float(max(-1.0, min(1.0, (calibrated_prob - 0.5) * 2.0)))
        confidence = float(min(1.0, max(0.0, abs(calibrated_prob - 0.5) * 2.0)))
        return reward, calibrated_prob, confidence

    def partial_fit(self, features: Dict[str, float], target_reward: float) -> None:
        target_prob = max(0.0, min(1.0, 0.5 + 0.5 * target_reward))
        z = self.bias
        for name, value in features.items():
            z += self.weights.get(name, 0.0) * value
        prob = self._sigmoid(z)
        error = target_prob - prob

        lr = self.learning_rate * (0.5 + 0.5 / (1 + self.example_count / 50.0))
        for name, value in features.items():
            self.weights[name] = self.weights.get(name, 0.0) + lr * error * value
        self.bias += lr * error
        self.example_count += 1

        # Platt-like calibration update (temperature scaling)
        confidence_target = max(0.05, min(0.95, target_prob))
        confidence_pred = max(0.05, min(0.95, self._sigmoid(self._apply_temperature(z))))
        temp_error = confidence_target - confidence_pred
        self.temperature = float(max(0.5, min(3.0, self.temperature + 0.05 * temp_error)))

        self._save_state()

    def _apply_temperature(self, logit: float) -> float:
        if self.temperature == 0:
            return logit
        return logit / max(1e-6, self.temperature)

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0:
            z = math.exp(-value)
            return 1.0 / (1.0 + z)
        z = math.exp(value)
        return z / (1.0 + z)

    def _load_state(self) -> None:
        if not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return
        self.weights = data.get("weights", {})
        self.bias = data.get("bias", 0.0)
        self.example_count = int(data.get("example_count", 0))
        self.temperature = float(data.get("temperature", self.temperature))

    def _save_state(self) -> None:
        try:
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "weights": self.weights,
                        "bias": self.bias,
                        "example_count": self.example_count,
                        "temperature": self.temperature,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception:
            pass


class RewardEngine:
    """
    Analyse le feedback social/utilisateur et génère des récompenses extrinsèques
    + ajuste drives/émotions/goals + log en mémoire + notifie métacognition.
    """

    POS_FR = {
        "bravo",
        "bien",
        "super",
        "parfait",
        "excellent",
        "incroyable",
        "merci",
        "génial",
        "top",
        "nickel",
        "parfait",
        "cool",
        "magnifique",
        "impeccable",
    }
    NEG_FR = {
        "nul",
        "mauvais",
        "horrible",
        "pas bien",
        "déçu",
        "décevant",
        "non",
        "faux",
        "pourri",
        "n'importe quoi",
        "t'es nul",
        "nulle",
        "c'est nul",
    }
    POS_EN = {
        "great",
        "good",
        "nice",
        "awesome",
        "amazing",
        "perfect",
        "thanks",
        "thank you",
        "well done",
        "brilliant",
        "wonderful",
        "excellent",
    }
    NEG_EN = {
        "bad",
        "terrible",
        "awful",
        "wrong",
        "no",
        "useless",
        "garbage",
        "disappointing",
        "nonsense",
        "makes no sense",
        "stupid",
        "dumb",
    }
    EMOJI_POS = set(list("👍👌👏💯✨🔥😄😁😊🤝❤️💪🤩🙌🌟✅🆗🙂🤗"))
    EMOJI_NEG = set(list("👎😡😠😞😔💢❌🛑🤬😤🙄😒"))

    def __init__(
        self,
        architecture=None,
        memory=None,
        emotions=None,
        goals=None,
        metacognition=None,
        persist_dir="data",
    ):
        self.arch = architecture
        self.memory = memory
        self.emotions = emotions
        self.goals = goals
        self.metacognition = metacognition
        self.persist_dir = persist_dir
        os.makedirs(os.path.join(self.persist_dir, "logs"), exist_ok=True)
        self.curiosity_engine = CuriosityEngine(architecture=architecture)
        self._last_intrinsic_goal = 0.0
        self._last_reward_event = time.time()
        self._last_events: deque = deque(maxlen=12)
        self._profile_state_path = os.path.join(self.persist_dir, "logs", "reward_profiles.json")
        self._user_profiles: Dict[str, Dict[str, float]] = self._load_profiles()

        model_state_path = os.path.join(self.persist_dir, "logs", "reward_model.json")
        self.classifier = OnlineCalibratedClassifier(state_path=model_state_path)
        self._listeners: List[Callable[[RewardEvent], None]] = []

    def ingest_user_message(
        self,
        user_id: str,
        text: str,
        context: Optional[Dict[str, Any]] = None,
        channel: str = "chat",
    ) -> RewardEvent:
        """
        À appeler depuis la boucle de dialogue - analyse textuelle, calcule une récompense,
        met à jour systèmes, et retourne l'événement.
        """
        ev = self._analyze_feedback(user_id, text, context or {}, channel)
        self._apply_reward(ev, intrinsic=False)
        self._log_reward_event(ev, intrinsic=False)
        self._notify_metacognition(ev, intrinsic=False)
        self._notify_listeners(ev)
        return ev

    def register_listener(self, listener: Callable[[RewardEvent], None]) -> None:
        if not callable(listener):
            return
        self._listeners.append(listener)

    def _notify_listeners(self, event: RewardEvent) -> None:
        if not self._listeners:
            return
        for listener in list(self._listeners):
            try:
                listener(event)
            except Exception:
                continue

    def register_intrinsic_reward(
        self,
        source: str,
        pleasure: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> RewardEvent:
        ctx = dict(context or {})
        kernel = ctx.get("phenomenal_kernel")
        intensity = max(0.0, min(1.0, abs(float(pleasure))))
        polarity = "positive" if pleasure > 0 else "negative" if pleasure < 0 else "neutral"
        features: Dict[str, Any] = {
            "hedonic_source": source,
            "hedonic_value": float(pleasure),
            "hedonic_mode": ctx.get("mode"),
            "intensity": intensity,
        }
        if isinstance(kernel, dict):
            features["kernel"] = dict(kernel)
        ev = RewardEvent(
            timestamp=time.time(),
            user_id=source or "internal",
            text=str(ctx.get("text", "[intrinsic_reward]")),
            channel="internal",
            extrinsic_reward=0.0,
            polarity=polarity,
            intensity=intensity,
            features=features,
            context=ctx,
        )
        self._apply_reward(ev, intrinsic=True)
        self._log_reward_event(ev, intrinsic=True)
        self._notify_metacognition(ev, intrinsic=True)
        self._notify_listeners(ev)
        return ev

    def _analyze_feedback(
        self, user_id: str, text: str, context: Dict[str, Any], channel: str
    ) -> RewardEvent:
        t = text.strip()
        t_lower = t.lower()

        lex_features = self._extract_lexical_features(t, t_lower)
        surface_features = self._extract_surface_features(t)
        context_features = self._extract_context_features(context)
        sequential_features = self._sequential_features()
        personalization_features = self._personalization_features(user_id)

        base_reward, base_intensity = self._heuristic_reward(lex_features, surface_features)

        model_features = self._prepare_model_features(
            text=t,
            lowercase=t_lower,
            lex=lex_features,
            surface=surface_features,
            context=context_features,
            sequential=sequential_features,
            personalization=personalization_features,
        )

        model_reward, calibrated_prob, confidence = self.classifier.predict(model_features)
        extrinsic = self._blend_rewards(base_reward, model_reward, confidence)
        intensity = float(max(base_intensity, min(1.0, confidence * 0.9 + 0.1)))

        llm_data: Optional[Dict[str, Any]] = None
        payload = {
            "text": text,
            "channel": channel,
            "base_reward": base_reward,
            "model_reward": model_reward,
            "features": {
                "lex": lex_features,
                "surface": surface_features,
                "context": context_features,
            },
        }
        response = try_call_llm_dict(
            "reward_engine",
            input_payload=payload,
            logger=logger,
        )
        if response:
            llm_data = dict(response)
            llm_valence = response.get("valence")
            try:
                if llm_valence is not None:
                    extrinsic = self._blend_rewards(extrinsic, float(llm_valence), 0.6)
            except (TypeError, ValueError):
                logger.debug("Invalid LLM valence: %r", llm_valence)
            if response.get("irony_detected"):
                intensity = max(intensity, 0.6)

        polarity = "neutral"
        if extrinsic > 0.05:
            polarity = "positive"
        elif extrinsic < -0.05:
            polarity = "negative"

        features = {
            "pos_hits": lex_features["pos_hits"],
            "neg_hits": lex_features["neg_hits"],
            "pos_emoji": surface_features["pos_emoji"],
            "neg_emoji": surface_features["neg_emoji"],
            "exclam": surface_features["exclam"],
            "question": surface_features["question"],
            "caps_ratio": surface_features["caps_ratio"],
            "intensity": intensity,
            "emoji_list": surface_features["emoji_list"],
            "model_reward": model_reward,
            "model_confidence": confidence,
            "calibrated_prob": calibrated_prob,
            "base_reward": base_reward,
            "sequential": sequential_features,
            "contextual": context_features,
            "personalization": personalization_features,
        }
        if llm_data:
            features["llm"] = llm_data

        self._update_user_profile(user_id, extrinsic, intensity)
        self._record_event_history(user_id, extrinsic, polarity, features)
        self._update_model_from_event(model_features, base_reward, extrinsic, context)

        return RewardEvent(
            timestamp=time.time(),
            user_id=user_id,
            text=text,
            channel=channel,
            extrinsic_reward=extrinsic,
            polarity=polarity,
            intensity=intensity,
            features=features,
            context=context,
        )

    def _apply_reward(self, ev: RewardEvent, intrinsic: bool = False):
        try:
            emo = self.emotions
            hedonic_val = 0.0
            hedonic_mode = None
            if isinstance(ev.features, dict):
                raw_hedonic = ev.features.get("hedonic_value")
                if isinstance(raw_hedonic, (int, float)):
                    hedonic_val = float(raw_hedonic)
                hedonic_mode = ev.features.get("hedonic_mode") or (ev.features.get("kernel") or {}).get("mode")
            if isinstance(ev.context, dict) and not hedonic_mode:
                hedonic_mode = ev.context.get("mode")
            if emo and hedonic_val and hasattr(emo, "register_intrinsic_pleasure"):
                try:
                    emo.register_intrinsic_pleasure(hedonic_val, meta=asdict(ev))
                except Exception:
                    pass
            if not intrinsic and emo:
                if hasattr(emo, "register_social_feedback"):
                    emo.register_social_feedback(
                        ev.extrinsic_reward,
                        ev.intensity,
                        ev.polarity,
                        meta=asdict(ev),
                    )
                elif hasattr(emo, "state"):
                    st = emo.state
                    st["valence"] = max(
                        -1.0,
                        min(1.0, st.get("valence", 0.0) + 0.3 * ev.extrinsic_reward),
                    )
                    st["arousal"] = max(
                        0.0,
                        min(1.0, st.get("arousal", 0.5) + 0.2 * ev.intensity),
                    )
        except Exception:
            pass

        if hedonic_val:
            homeo = getattr(self.arch, "homeostasis", None)
            if homeo and hasattr(homeo, "register_hedonic_state"):
                try:
                    homeo.register_hedonic_state(
                        hedonic_val,
                        mode=str(hedonic_mode or "travail"),
                        meta=ev.features if isinstance(ev.features, dict) else {},
                    )
                except Exception:
                    pass

        if not intrinsic:
            try:
                if self.goals:
                    active = None
                    if hasattr(self.goals, "get_active_goal"):
                        active = self.goals.get_active_goal()
                    elif hasattr(self.goals, "current_goal"):
                        active = getattr(self.goals, "current_goal")

                    if active:
                        lr = 0.25 * (0.5 + 0.5 * ev.intensity)
                        new_val = max(
                            0.0,
                            min(1.0, active.get("value", 0.5) + lr * ev.extrinsic_reward),
                        )
                        active["value"] = new_val
                        if "evidence" not in active:
                            active["evidence"] = []
                        active["evidence"].append(
                            {
                                "t": ev.timestamp,
                                "type": "social_feedback",
                                "delta_value": lr * ev.extrinsic_reward,
                                "user_id": ev.user_id,
                                "text": ev.text,
                            }
                        )
                        if hasattr(self.goals, "update_goal"):
                            self.goals.update_goal(active.get("id"), active)
            except Exception:
                pass

            if ev.extrinsic_reward > 0.05:
                self._last_reward_event = ev.timestamp

            self._maybe_spawn_curiosity_goals(ev)

        try:
            if self.memory:
                payload = {
                    "type": "intrinsic_feedback" if intrinsic else "social_feedback",
                    "reward": ev.extrinsic_reward,
                    "polarity": ev.polarity,
                    "intensity": ev.intensity,
                    "hedonic_value": hedonic_val,
                    "features": ev.features,
                    "user_id": ev.user_id,
                    "text": ev.text,
                    "context": ev.context,
                }
                if hasattr(self.memory, "add_memory"):
                    self.memory.add_memory("feedback", payload)
                elif hasattr(self.memory, "store_event"):
                    self.memory.store_event("feedback", payload)
        except Exception:
            pass

        self._metacognitive_loop_adjustment(ev)

    def _notify_metacognition(self, ev: RewardEvent, intrinsic: bool = False):
        try:
            m = self.metacognition
            if not m:
                return
            if hasattr(m, "_record_metacognitive_event"):
                event_type = "intrinsic_feedback" if intrinsic else "social_feedback"
                description = (
                    f"Intrinsic pleasure {ev.polarity} (I={ev.intensity:.2f})"
                    if intrinsic
                    else f"Feedback {ev.polarity} (r={ev.extrinsic_reward:.2f}, I={ev.intensity:.2f})"
                )
                significance = ev.intensity if intrinsic else abs(ev.extrinsic_reward) * ev.intensity
                hedonic_val = 0.0
                if intrinsic and isinstance(ev.features, dict):
                    try:
                        hedonic_val = float(ev.features.get("hedonic_value", 0.0))
                    except (TypeError, ValueError):
                        hedonic_val = 0.0
                m._record_metacognitive_event(
                    event_type=event_type,
                    domain=getattr(m, "CognitiveDomain", None).DECISION_MAKING
                    if hasattr(m, "CognitiveDomain")
                    else None,
                    description=description,
                    significance=significance,
                    confidence=0.7,
                    emotional_valence=ev.extrinsic_reward if not intrinsic else hedonic_val,
                    cognitive_load=0.0,
                    related_memories=[],
                    action_taken=None,
                )
        except Exception:
            pass

    def _log_reward_event(self, ev: RewardEvent, intrinsic: bool = False):
        try:
            filename = "intrinsic_feedback.jsonl" if intrinsic else "social_feedback.jsonl"
            path = os.path.join(self.persist_dir, "logs", filename)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(json_sanitize(asdict(ev)), ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _count_hits(self, text_lower: str, vocab: set) -> int:
        hits = 0
        for w in vocab:
            pattern = re.escape(w).replace(r"\ ", r"\s+")
            if re.search(rf"(?<!\w){pattern}(?!\w)", text_lower):
                hits += 1
        return hits

    def _extract_lexical_features(self, text: str, text_lower: str) -> Dict[str, Any]:
        pos_hits = self._count_hits(text_lower, self.POS_FR) + self._count_hits(
            text_lower, self.POS_EN
        )
        neg_hits = self._count_hits(text_lower, self.NEG_FR) + self._count_hits(
            text_lower, self.NEG_EN
        )
        direct_pos = 1 if re.search(r"\b(merci|thanks)\b", text_lower) else 0
        direct_neg = (
            1
            if re.search(
                r"\b(ta\sfaux|you(\sare)?\swrong|c'?est\s(faux|nul))\b", text_lower
            )
            else 0
        )
        return {
            "pos_hits": pos_hits + direct_pos,
            "neg_hits": neg_hits + direct_neg,
            "length": len(text),
            "word_count": len(text_lower.split()),
        }

    def _extract_surface_features(self, text: str) -> Dict[str, Any]:
        emojis = [ch for ch in text if ch in self.EMOJI_POS or ch in self.EMOJI_NEG]
        pos_emoji = sum(1 for e in emojis if e in self.EMOJI_POS)
        neg_emoji = sum(1 for e in emojis if e in self.EMOJI_NEG)
        exclam = text.count("!")
        quest = text.count("?")
        caps_ratio = self._caps_ratio(text)
        return {
            "emoji_list": emojis,
            "pos_emoji": pos_emoji,
            "neg_emoji": neg_emoji,
            "exclam": exclam,
            "question": quest,
            "caps_ratio": caps_ratio,
        }

    def _extract_context_features(self, context: Dict[str, Any]) -> Dict[str, float]:
        if not context:
            return {}
        features: Dict[str, float] = {}
        last_assistant = context.get("last_assistant_output")
        if isinstance(last_assistant, str):
            features["last_assistant_length"] = float(min(500, len(last_assistant)))
        emotional_state = context.get("emotional_state") or {}
        if isinstance(emotional_state, dict):
            for key in ("valence", "arousal"):
                if isinstance(emotional_state.get(key), (int, float)):
                    features[f"emo_{key}"] = float(emotional_state[key])
        active_goal = context.get("active_goal") or {}
        if isinstance(active_goal, dict):
            for key in ("value", "confidence", "progress"):
                if isinstance(active_goal.get(key), (int, float)):
                    features[f"goal_{key}"] = float(active_goal[key])
        if isinstance(context.get("expected_reward"), (int, float)):
            features["expected_reward"] = float(context["expected_reward"])
        if isinstance(context.get("conversation_turn"), (int, float)):
            features["conversation_turn"] = float(context["conversation_turn"])
        kernel = context.get("phenomenal_kernel")
        if isinstance(kernel, dict):
            energy = kernel.get("energy")
            if isinstance(energy, (int, float)):
                features["kernel_energy"] = float(energy)
            fatigue = kernel.get("fatigue")
            if isinstance(fatigue, (int, float)):
                features["kernel_fatigue"] = float(fatigue)
            hedonic = kernel.get("hedonic_reward")
            if isinstance(hedonic, (int, float)):
                features["kernel_hedonic"] = float(hedonic)
            slowdown = kernel.get("global_slowdown")
            if isinstance(slowdown, (int, float)):
                features["kernel_slowdown"] = float(slowdown)
            alert_pressure = kernel.get("alert_pressure")
            if isinstance(alert_pressure, (int, float)):
                features["kernel_alert_pressure"] = float(alert_pressure)
            feel_like = kernel.get("feel_like")
            if isinstance(feel_like, str):
                features["kernel_feel_pause"] = 1.0 if feel_like == "pause" else 0.5 if feel_like == "slow" else 0.0
        return features

    def _heuristic_reward(
        self, lex: Dict[str, Any], surface: Dict[str, Any]
    ) -> Tuple[float, float]:
        raw_pos = lex["pos_hits"] + surface["pos_emoji"]
        raw_neg = lex["neg_hits"] + surface["neg_emoji"]
        raw = raw_pos - raw_neg
        base = 0.0
        if raw > 0:
            base = min(1.0, 0.2 + 0.15 * raw)
        elif raw < 0:
            base = max(-1.0, -0.2 + 0.15 * raw)
        intensity = min(
            1.0,
            0.3
            + 0.1 * surface["exclam"]
            + 0.05 * surface["question"]
            + 0.6 * surface["caps_ratio"],
        )
        extrinsic = float(max(-1.0, min(1.0, base * (0.6 + 0.4 * intensity))))
        return extrinsic, intensity

    def _prepare_model_features(
        self,
        text: str,
        lowercase: str,
        lex: Dict[str, Any],
        surface: Dict[str, Any],
        context: Dict[str, float],
        sequential: Dict[str, float],
        personalization: Dict[str, float],
    ) -> Dict[str, float]:
        features: Dict[str, float] = {
            "bias": 1.0,
            "lex_pos": float(lex["pos_hits"]),
            "lex_neg": float(lex["neg_hits"]),
            "emoji_pos": float(surface["pos_emoji"]),
            "emoji_neg": float(surface["neg_emoji"]),
            "exclam": float(surface["exclam"]),
            "question": float(surface["question"]),
            "caps_ratio": float(surface["caps_ratio"]),
            "length": min(500.0, float(len(text))),
            "word_count": min(120.0, float(len(lowercase.split()))),
        }

        for key, value in context.items():
            features[f"ctx_{key}"] = float(value)
        for key, value in sequential.items():
            features[f"seq_{key}"] = float(value)
        for key, value in personalization.items():
            features[f"pers_{key}"] = float(value)

        if surface["emoji_list"]:
            top_emoji = surface["emoji_list"][0]
            features[f"emoji_{top_emoji}"] = 1.0

        if "!" in text:
            features["contains_exclamation"] = 1.0
        if lowercase.endswith("?"):
            features["ends_question"] = 1.0
        if re.search(r"\b(bizarre|sarcasm|ironie)\b", lowercase):
            features["lex_sarcasm"] = 1.0

        return features

    def _blend_rewards(self, base: float, model: float, confidence: float) -> float:
        alpha = 0.35 + 0.45 * confidence
        return float(max(-1.0, min(1.0, (1 - alpha) * base + alpha * model)))

    def _sequential_features(self) -> Dict[str, float]:
        if not self._last_events:
            return {}
        positives = sum(1 for ev in self._last_events if ev.get("reward", 0.0) > 0.05)
        negatives = sum(1 for ev in self._last_events if ev.get("reward", 0.0) < -0.05)
        neutral = len(self._last_events) - positives - negatives
        last = self._last_events[-1]
        time_delta = time.time() - last.get("timestamp", time.time())
        return {
            "recent_positive_ratio": positives / max(1, len(self._last_events)),
            "recent_negative_ratio": negatives / max(1, len(self._last_events)),
            "recent_neutral_ratio": neutral / max(1, len(self._last_events)),
            "time_since_last": min(600.0, float(time_delta)),
            "last_reward": float(last.get("reward", 0.0)),
        }

    def _record_event_history(
        self, user_id: str, reward: float, polarity: str, features: Dict[str, Any]
    ) -> None:
        self._last_events.append(
            {
                "timestamp": time.time(),
                "user_id": user_id,
                "reward": reward,
                "polarity": polarity,
                "intensity": features.get("intensity", 0.0),
            }
        )

    def _personalization_features(self, user_id: str) -> Dict[str, float]:
        profile = self._user_profiles.get(user_id)
        if not profile:
            return {"seen": 0.0}
        return {
            "seen": 1.0,
            "bias": profile.get("bias", 0.0),
            "moving_avg": profile.get("moving_avg", 0.0),
            "volatility": profile.get("volatility", 0.0),
        }

    def _update_user_profile(self, user_id: str, reward: float, intensity: float) -> None:
        profile = self._user_profiles.setdefault(
            user_id,
            {"count": 0.0, "bias": 0.0, "moving_avg": 0.0, "volatility": 0.0},
        )
        count = profile.get("count", 0.0) + 1.0
        moving_avg = profile.get("moving_avg", 0.0)
        moving_avg = moving_avg * 0.8 + reward * 0.2
        bias = profile.get("bias", 0.0) * 0.9 + reward * 0.1
        volatility = profile.get("volatility", 0.0)
        volatility = volatility * 0.85 + abs(reward - moving_avg) * 0.15

        profile.update(
            {
                "count": count,
                "bias": bias,
                "moving_avg": moving_avg,
                "volatility": volatility,
                "last_intensity": intensity,
            }
        )
        self._save_profiles()

    def _update_model_from_event(
        self,
        features: Dict[str, float],
        base_reward: float,
        extrinsic: float,
        context: Dict[str, Any],
    ) -> None:
        supervision_candidates: List[float] = [base_reward]
        if isinstance(context.get("expected_reward"), (int, float)):
            supervision_candidates.append(float(context["expected_reward"]))
        if isinstance(context.get("goal_correction"), (int, float)):
            supervision_candidates.append(float(context["goal_correction"]))
        if supervision_candidates:
            target = sum(supervision_candidates) / len(supervision_candidates)
        else:
            target = extrinsic
        self.classifier.partial_fit(features, target)

    def _metacognitive_loop_adjustment(self, ev: RewardEvent) -> None:
        context = ev.context or {}
        correction = None
        if isinstance(context.get("metacognitive_signal"), (int, float)):
            correction = float(context["metacognitive_signal"])
        elif isinstance(context.get("goal_progress_delta"), (int, float)) and abs(
            context["goal_progress_delta"]
        ) > 0:
            correction = float(max(-1.0, min(1.0, context["goal_progress_delta"])) )
        if correction is None:
            return
        adjusted = (ev.extrinsic_reward + correction) / 2.0
        features = self._prepare_model_features(
            text=ev.text,
            lowercase=ev.text.lower(),
            lex=self._extract_lexical_features(ev.text, ev.text.lower()),
            surface=self._extract_surface_features(ev.text),
            context=self._extract_context_features(ev.context or {}),
            sequential=self._sequential_features(),
            personalization=self._personalization_features(ev.user_id),
        )
        self.classifier.partial_fit(features, adjusted)

    def _load_profiles(self) -> Dict[str, Dict[str, float]]:
        if not os.path.exists(self._profile_state_path):
            return {}
        try:
            with open(self._profile_state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            return {}
        return {}

    def _save_profiles(self) -> None:
        try:
            with open(self._profile_state_path, "w", encoding="utf-8") as f:
                json.dump(self._user_profiles, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _caps_ratio(self, text: str) -> float:
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return 0.0
        upper = sum(1 for c in letters if c.isupper())
        return upper / max(1, len(letters))

    def _maybe_spawn_curiosity_goals(self, ev: RewardEvent) -> None:
        goals = self.goals or getattr(self.arch, "goals", None)
        if not goals or not hasattr(goals, "curiosity"):
            return
        now = time.time()
        if ev.extrinsic_reward > 0.05:
            self._last_reward_event = now
            return
        if (now - self._last_intrinsic_goal) < 180.0:
            return
        if (now - self._last_reward_event) < 90.0 and ev.extrinsic_reward >= 0.0:
            return
        parent = None
        try:
            if hasattr(goals, "store"):
                parent = goals.store.get_active()
        except Exception:
            parent = None
        parent_payload = parent.to_dict() if parent and hasattr(parent, "to_dict") else None
        engine = getattr(goals, "curiosity", None) or self.curiosity_engine
        try:
            proposals = engine.suggest_subgoals(parent_payload, k=2)
        except Exception:
            proposals = []
        added = False
        for proposal in proposals:
            try:
                goals.add_goal(
                    proposal.get("description", "Explorer un sujet intrigant."),
                    goal_type=GoalType.EXPLORATION,
                    criteria=proposal.get("criteria"),
                    parent_ids=proposal.get("parent_ids"),
                    value=float(proposal.get("value", 0.55)),
                    competence=float(proposal.get("competence", 0.5)),
                    curiosity=float(proposal.get("curiosity", 0.7)),
                    urgency=float(proposal.get("urgency", 0.35)),
                    created_by=proposal.get("created_by", "curiosity"),
                )
                added = True
            except Exception:
                continue
        if added:
            self._last_intrinsic_goal = now
