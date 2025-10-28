import json
import logging
import math
import os
import random
import time
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Tuple, Mapping

from AGI_Evolutive.utils.jsonsafe import json_sanitize
from AGI_Evolutive.utils.llm_service import try_call_llm_dict

from AGI_Evolutive.core.config import cfg

_PATH = cfg()["HOMEOSTASIS_PATH"]


logger = logging.getLogger(__name__)


def _now() -> float:
    """Allow easier testing."""

    return time.time()


_DEFAULT_DRIVES: Dict[str, float] = {
    "curiosity": 0.6,
    "self_preservation": 0.7,
    "social_bonding": 0.4,
    "competence": 0.5,
    "play": 0.3,
    "restoration": 0.45,
    "task_activation": 0.55,
    "energy": 0.65,
    "respiration": 0.7,
    "thermal_regulation": 0.75,
    "memory_balance": 0.6,
}


class OnlineLinearModel:
    """Very small online ridge-regression helper."""

    def __init__(
        self,
        feature_names: Iterable[str],
        weights: Optional[Dict[str, float]] = None,
        lr: float = 0.05,
        l2: float = 0.01,
        forget: float = 0.001,
        max_step: float = 0.1,
    ) -> None:
        self.feature_names = list(feature_names)
        self.weights: Dict[str, float] = {name: 0.0 for name in self.feature_names}
        if weights:
            self.weights.update({k: float(v) for k, v in weights.items() if k in self.weights})
        self.lr = lr
        self.l2 = l2
        self.forget = forget
        self.max_step = max_step

    def predict(self, features: MutableMapping[str, float]) -> float:
        value = 0.0
        for name in self.feature_names:
            value += self.weights.get(name, 0.0) * float(features.get(name, 0.0))
        return value

    def update(self, features: MutableMapping[str, float], target: float, prediction: float) -> None:
        error = prediction - target
        for name in self.feature_names:
            grad = error * float(features.get(name, 0.0)) + self.l2 * self.weights.get(name, 0.0)
            step = self.lr * grad
            step = max(-self.max_step, min(self.max_step, step))
            self.weights[name] -= step
            self.weights[name] *= 1.0 - self.forget


class Homeostasis:
    """Internal drives and reward shaping."""

    _DECAY_OPTIONS_HOURS: Tuple[int, ...] = (3, 7, 14, 30)
    _POSITIVE_TERMS: Dict[str, float] = {
        "bravo": 0.35,
        "bien": 0.3,
        "good": 0.35,
        "thanks": 0.25,
        "merci": 0.25,
        "excellent": 0.45,
        "great": 0.4,
        "super": 0.3,
    }
    _NEGATIVE_TERMS: Dict[str, float] = {
        "mauvais": -0.35,
        "non": -0.25,
        "wrong": -0.35,
        "bad": -0.35,
        "triste": -0.2,
        "fail": -0.4,
        "erreur": -0.3,
        "bug": -0.25,
    }
    _NEUTRAL_TERMS: Dict[str, float] = {
        "peut": 0.05,
        "essaye": 0.05,
    }

    def __init__(self) -> None:
        self.state: Dict[str, Any] = {
            "drives": dict(_DEFAULT_DRIVES),
            "last_update": _now(),
            "intrinsic_reward": 0.0,
            "extrinsic_reward": 0.0,
            "hedonic_reward": 0.0,
            "meta": {},
        }
        self._load()
        self._ensure_schema()

    def _load(self) -> None:
        if os.path.exists(_PATH):
            try:
                with open(_PATH, "r", encoding="utf-8") as f:
                    self.state = json.load(f)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "Impossible de charger l'état homeostasis, utilisation des valeurs par défaut: %s",
                    exc,
                )

    def _ensure_schema(self) -> None:
        drives = self.state.setdefault("drives", dict(_DEFAULT_DRIVES))
        for key, default in _DEFAULT_DRIVES.items():
            drives.setdefault(key, default)

        meta = self.state.setdefault("meta", {})
        decay_meta = meta.setdefault("decay", {})
        timestamp = float(self.state.get("last_update", _now()))
        for drive in drives:
            if drive not in decay_meta:
                decay_meta[drive] = {
                    "options": list(self._DECAY_OPTIONS_HOURS),
                    "alpha": [1.0 for _ in self._DECAY_OPTIONS_HOURS],
                    "beta": [1.0 for _ in self._DECAY_OPTIONS_HOURS],
                    "last_choice": 0,
                    "last_decay": timestamp,
                    "last_reward_time": timestamp,
                }
            else:
                entry = decay_meta[drive]
                entry.setdefault("options", list(self._DECAY_OPTIONS_HOURS))
                if len(entry.get("options", [])) != len(self._DECAY_OPTIONS_HOURS):
                    entry["options"] = list(self._DECAY_OPTIONS_HOURS)
                entry.setdefault("alpha", [1.0 for _ in self._DECAY_OPTIONS_HOURS])
                entry.setdefault("beta", [1.0 for _ in self._DECAY_OPTIONS_HOURS])
                if len(entry["alpha"]) != len(self._DECAY_OPTIONS_HOURS):
                    entry["alpha"] = [1.0 for _ in self._DECAY_OPTIONS_HOURS]
                if len(entry["beta"]) != len(self._DECAY_OPTIONS_HOURS):
                    entry["beta"] = [1.0 for _ in self._DECAY_OPTIONS_HOURS]
                entry.setdefault("last_choice", 0)
                entry.setdefault("last_decay", timestamp)
                entry.setdefault("last_reward_time", timestamp)

        intrinsic_meta = meta.setdefault("intrinsic_model", {})
        intrinsic_meta.setdefault(
            "weights",
            {
                "bias": 0.1,
                "curiosity": 0.4,
                "competence": 0.3,
                "play": 0.2,
                "restoration": 0.25,
            },
        )
        intrinsic_meta.setdefault("lr", 0.05)
        intrinsic_meta.setdefault("l2", 0.01)
        intrinsic_meta.setdefault("forget", 0.002)
        intrinsic_meta.setdefault("max_step", 0.1)
        intrinsic_meta.setdefault("last_features", None)
        intrinsic_meta.setdefault("last_prediction", 0.0)

        extrinsic_meta = meta.setdefault("extrinsic_model", {})
        extrinsic_meta.setdefault("adaptive_terms", {})
        extrinsic_meta.setdefault("decay", 0.01)
        extrinsic_meta.setdefault("max_weight", 0.6)

        drift_meta = meta.setdefault("drift", {})
        drift_meta.setdefault("history", [])
        drift_meta.setdefault("window", 25)
        drift_meta.setdefault("threshold", 0.18)
        drift_meta.setdefault("last_logged", None)

        self.state.setdefault("last_update", timestamp)
        self.state.setdefault("intrinsic_reward", 0.0)
        self.state.setdefault("extrinsic_reward", 0.0)
        self.state.setdefault("hedonic_reward", 0.0)
        meta.setdefault("hedonic", {})

    def _save(self) -> None:
        os.makedirs(os.path.dirname(_PATH), exist_ok=True)
        with open(_PATH, "w", encoding="utf-8") as f:
            json.dump(json_sanitize(self.state), f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Decay and survival-style TTL selection
    # ------------------------------------------------------------------

    def _select_ttl_index(self, drive: str, timestamp: float) -> int:
        decay_meta = self.state["meta"]["decay"][drive]
        alpha = decay_meta["alpha"]
        beta = decay_meta["beta"]
        best_idx = 0
        best_score = float("-inf")
        for idx, (a, b) in enumerate(zip(alpha, beta)):
            sample = random.betavariate(a, b)
            hazard = self._hazard_rate(drive, idx, timestamp)
            score = sample + hazard * 0.1
            if score > best_score:
                best_idx = idx
                best_score = score
        return best_idx

    def _hazard_rate(self, drive: str, option_index: int, timestamp: float) -> float:
        decay_meta = self.state["meta"]["decay"][drive]
        ttl_hours = decay_meta["options"][option_index]
        ttl_seconds = max(1.0, ttl_hours * 3600.0)
        age = max(0.0, timestamp - float(decay_meta.get("last_reward_time", timestamp)))
        hazard = 1.0 - math.exp(-age / ttl_seconds)
        return hazard

    def _update_decay_bandit(self, drive: str, success: bool) -> None:
        decay_meta = self.state["meta"]["decay"][drive]
        idx = decay_meta.get("last_choice", 0)
        alpha = decay_meta["alpha"][idx]
        beta = decay_meta["beta"][idx]
        if success:
            alpha += 1.0
        else:
            beta += 1.0
        decay_meta["alpha"][idx] = alpha
        decay_meta["beta"][idx] = beta

    def _decay_factor(self, ttl_hours: int, delta_seconds: float) -> float:
        ttl_seconds = max(1.0, ttl_hours * 3600.0)
        # Convert to exponential decay factor equivalent to half-life.
        factor = 1.0 - 0.5 ** (delta_seconds / ttl_seconds)
        return max(0.005, min(0.12, factor))

    def decay(self) -> None:
        timestamp = _now()
        delta_seconds = max(0.0, timestamp - float(self.state.get("last_update", timestamp)))
        self.state["last_update"] = timestamp
        for drive, value in list(self.state["drives"].items()):
            idx = self._select_ttl_index(drive, timestamp)
            ttl_hours = self.state["meta"]["decay"][drive]["options"][idx]
            factor = self._decay_factor(ttl_hours, delta_seconds)
            new_value = value + factor * (0.5 - value)
            new_value = max(0.0, min(1.0, new_value))
            self.state["drives"][drive] = new_value
            decay_meta = self.state["meta"]["decay"][drive]
            decay_meta["last_choice"] = idx
            decay_meta["last_decay"] = timestamp
            self._record_drive_history(drive, new_value)
        self._save()

    # ------------------------------------------------------------------
    # Intrinsic rewards
    # ------------------------------------------------------------------

    def _intrinsic_model(self) -> OnlineLinearModel:
        meta = self.state["meta"]["intrinsic_model"]
        model = OnlineLinearModel(
            feature_names=["bias", "curiosity", "competence", "play", "restoration"],
            weights=meta.get("weights", {}),
            lr=float(meta.get("lr", 0.05)),
            l2=float(meta.get("l2", 0.01)),
            forget=float(meta.get("forget", 0.002)),
            max_step=float(meta.get("max_step", 0.1)),
        )
        return model

    def _store_intrinsic_model(self, model: OnlineLinearModel) -> None:
        self.state["meta"]["intrinsic_model"]["weights"] = dict(model.weights)

    def _prepare_intrinsic_features(self, info_gain: float, progress: float) -> Dict[str, float]:
        curiosity_drive = float(self.state["drives"].get("curiosity", 0.5))
        competence_drive = float(self.state["drives"].get("competence", 0.5))
        play_drive = float(self.state["drives"].get("play", 0.3))
        restoration_drive = float(self.state["drives"].get("restoration", 0.45))
        features = {
            "bias": 1.0,
            "curiosity": curiosity_drive * float(info_gain),
            "competence": competence_drive * float(progress),
            "play": play_drive * float(info_gain),
            "restoration": restoration_drive * (1.0 - float(progress)),
        }
        return features

    def compute_intrinsic_reward(self, info_gain: float, progress: float) -> float:
        features = self._prepare_intrinsic_features(info_gain, progress)
        model = self._intrinsic_model()
        prediction = model.predict(features)
        reward = max(0.0, min(1.0, prediction))
        self.state["intrinsic_reward"] = reward
        meta = self.state["meta"]["intrinsic_model"]
        meta["last_features"] = features
        meta["last_prediction"] = reward
        self._store_intrinsic_model(model)
        self._save()
        return reward

    # ------------------------------------------------------------------
    # Extrinsic rewards
    # ------------------------------------------------------------------

    def _normalise_text(self, text: str) -> List[str]:
        return [token for token in text.lower().split() if token]

    def _sentiment_score(self, tokens: Iterable[str]) -> float:
        extrinsic_meta = self.state["meta"]["extrinsic_model"]
        adaptive_terms: Dict[str, float] = extrinsic_meta.setdefault("adaptive_terms", {})
        decay = float(extrinsic_meta.get("decay", 0.01))
        max_weight = float(extrinsic_meta.get("max_weight", 0.6))
        counts: Dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1

        score = 0.0
        total = sum(counts.values()) or 1

        for token, count in counts.items():
            weight = 0.0
            if token in self._POSITIVE_TERMS:
                weight += self._POSITIVE_TERMS[token]
            if token in self._NEGATIVE_TERMS:
                weight += self._NEGATIVE_TERMS[token]
            if token in self._NEUTRAL_TERMS:
                weight += self._NEUTRAL_TERMS[token]
            adaptive_weight = adaptive_terms.get(token, 0.0)
            weight += adaptive_weight
            score += weight * (count / total)

            # Lightweight online update so that recurring tokens adapt.
            adaptive_terms[token] = max(
                -max_weight,
                min(
                    max_weight,
                    adaptive_weight * (1.0 - decay)
                    + 0.1 * weight * (count / total),
                ),
            )

        score = max(-1.0, min(1.0, math.tanh(score)))
        return score

    def _llm_adjust_from_feedback(
        self,
        text: str,
        tokens: Iterable[str],
        baseline_reward: float,
    ) -> Dict[str, Any]:
        payload = {
            "feedback": text,
            "tokens": list(tokens)[:50],
            "current_drives": self.state.get("drives", {}),
            "recent_rewards": {
                "intrinsic": self.state.get("intrinsic_reward"),
                "extrinsic": self.state.get("extrinsic_reward"),
                "hedonic": self.state.get("hedonic_reward"),
            },
            "baseline_reward": baseline_reward,
        }

        llm_result = try_call_llm_dict(
            "homeostasis",
            input_payload=payload,
            logger=logger,
        )

        report: Dict[str, Any] = {}
        if isinstance(llm_result, Dict):
            drive_updates = llm_result.get("drive_updates")
            if isinstance(drive_updates, Mapping):
                applied: Dict[str, float] = {}
                for drive, delta in drive_updates.items():
                    try:
                        adjustment = float(delta)
                    except (TypeError, ValueError):
                        continue
                    if drive not in self.state.get("drives", {}):
                        continue
                    self._apply_drive_update(drive, adjustment, max_step=0.12)
                    applied[drive] = float(self.state["drives"].get(drive, 0.5))
                if applied:
                    report["drive_updates"] = applied
            reward_signal = llm_result.get("reward_signal")
            try:
                if reward_signal is not None:
                    reward_value = max(-1.0, min(1.0, float(reward_signal)))
                    report["reward_signal"] = reward_value
            except (TypeError, ValueError):
                pass
            notes = llm_result.get("notes")
            if isinstance(notes, str) and notes.strip():
                report["notes"] = notes.strip()
        return report

    def compute_extrinsic_reward_from_memories(self, recent_feedback: str) -> float:
        text = (recent_feedback or "").strip()
        tokens = self._normalise_text(text)
        bonus = self._sentiment_score(tokens)
        llm_report = self._llm_adjust_from_feedback(text, tokens, bonus)
        reward_value = bonus
        if isinstance(llm_report, Mapping) and "reward_signal" in llm_report:
            reward_value = float(llm_report["reward_signal"])
        self.state["extrinsic_reward"] = max(-1.0, min(1.0, reward_value))
        if isinstance(llm_report, Mapping):
            feedback_meta = self.state.setdefault("meta", {}).setdefault("llm_feedback", {})
            feedback_meta.update(
                {
                    "last_report": llm_report,
                    "text": text[:200],
                    "ts": _now(),
                }
            )
        self._save()
        return self.state["extrinsic_reward"]

    # ------------------------------------------------------------------
    # Drive updates and drift detection
    # ------------------------------------------------------------------

    def _apply_drive_update(self, drive: str, delta: float, max_step: float = 0.08) -> None:
        capped_delta = max(-max_step, min(max_step, delta))
        updated = max(0.0, min(1.0, self.state["drives"].get(drive, 0.5) + capped_delta))
        self.state["drives"][drive] = updated
        self._record_drive_history(drive, updated)

    def adjust_drive(self, drive: str, delta: float, max_step: float = 0.08) -> None:
        """Public helper used by other modules (e.g. EmotionEngine)."""
        self._apply_drive_update(drive, delta, max_step=max_step)
        self._save()

    def integrate_system_metrics(self, metrics: MutableMapping[str, Any]) -> Dict[str, float]:
        """Translate raw machine metrics into drive levels.

        Returns the subset of drives that were updated with their new values.
        """

        if not isinstance(metrics, MutableMapping):
            return {}

        timestamp = float(metrics.get("timestamp", _now())) if isinstance(metrics.get("timestamp"), (int, float)) else _now()

        def _inverse_scale(value: Optional[float], comfort: float, critical: float) -> Optional[float]:
            if value is None:
                return None
            try:
                val = float(value)
            except Exception:
                return None
            if val <= comfort:
                return 1.0
            if val >= critical:
                return 0.0
            return max(0.0, min(1.0, (critical - val) / (critical - comfort)))

        def _avg(values: Iterable[float]) -> Optional[float]:
            vals = [float(v) for v in values if v is not None]
            if not vals:
                return None
            return sum(vals) / len(vals)

        cpu_info = metrics.get("cpu") if isinstance(metrics.get("cpu"), MutableMapping) else {}
        mem_info = metrics.get("memory") if isinstance(metrics.get("memory"), MutableMapping) else {}
        gpu_info = metrics.get("gpu") if isinstance(metrics.get("gpu"), MutableMapping) else {}
        power_info = metrics.get("power") if isinstance(metrics.get("power"), MutableMapping) else {}

        cpu_load = cpu_info.get("load") if isinstance(cpu_info, MutableMapping) else None
        cpu_temp = cpu_info.get("temp_c") if isinstance(cpu_info, MutableMapping) else None
        mem_percent = mem_info.get("percent") if isinstance(mem_info, MutableMapping) else None
        gpu_temp = gpu_info.get("temp_c") if isinstance(gpu_info, MutableMapping) else None
        gpu_util = gpu_info.get("util_pct") if isinstance(gpu_info, MutableMapping) else None
        power_draw = power_info.get("draw_w") if isinstance(power_info, MutableMapping) else None

        respiration_target = _inverse_scale(cpu_load, 45.0, 95.0)
        memory_target = _inverse_scale(mem_percent, 65.0, 95.0)
        thermal_candidates = [
            _inverse_scale(cpu_temp, 65.0, 90.0),
            _inverse_scale(gpu_temp, 65.0, 85.0),
        ]
        thermal_target = _avg(v for v in thermal_candidates if v is not None)

        energy_components: List[Optional[float]] = [
            _inverse_scale(cpu_load, 55.0, 95.0),
            _inverse_scale(gpu_util, 60.0, 100.0),
            _inverse_scale(power_draw, 60.0, 150.0),
        ]
        energy_target = _avg(v for v in energy_components if v is not None)

        updates: Dict[str, float] = {}
        changed = False

        def _apply_target(name: str, target: Optional[float], step: float = 0.4) -> None:
            nonlocal changed
            current = float(self.state["drives"].get(name, _DEFAULT_DRIVES.get(name, 0.5)))
            if target is None:
                updates[name] = current
                return
            target = max(0.0, min(1.0, float(target)))
            delta = target - current
            if abs(delta) < 0.01:
                updates[name] = current
                return
            prev = current
            self.adjust_drive(name, delta, max_step=step)
            new_value = float(self.state["drives"].get(name, current))
            updates[name] = new_value
            if abs(new_value - prev) >= 1e-3:
                changed = True

        _apply_target("respiration", respiration_target)
        _apply_target("memory_balance", memory_target)
        _apply_target("thermal_regulation", thermal_target)
        _apply_target("energy", energy_target)

        if changed:
            physiology_meta = self.state.setdefault("meta", {}).setdefault("physiology", {})
            physiology_meta["last_snapshot"] = {
                "timestamp": timestamp,
                "cpu_load": float(cpu_load) if isinstance(cpu_load, (int, float)) else None,
                "cpu_temp": float(cpu_temp) if isinstance(cpu_temp, (int, float)) else None,
                "mem_percent": float(mem_percent) if isinstance(mem_percent, (int, float)) else None,
                "gpu_temp": float(gpu_temp) if isinstance(gpu_temp, (int, float)) else None,
                "gpu_util": float(gpu_util) if isinstance(gpu_util, (int, float)) else None,
                "power_draw": float(power_draw) if isinstance(power_draw, (int, float)) else None,
                "drives": {k: updates.get(k, self.state["drives"].get(k)) for k in ("energy", "respiration", "thermal_regulation", "memory_balance")},
            }
            self._save()
        return updates

    def register_hedonic_state(self, pleasure: float, mode: str = "travail", meta: Optional[Dict[str, Any]] = None) -> None:
        """Track hedonic events (flânerie, repos) and adjust drives accordingly."""
        pleasure = max(-1.0, min(1.0, float(pleasure)))
        self.state["hedonic_reward"] = pleasure
        play_delta = 0.08 * pleasure
        restoration_delta = 0.12 * pleasure if mode == "flanerie" else 0.04 * pleasure
        self._apply_drive_update("play", play_delta)
        self._apply_drive_update("restoration", restoration_delta, max_step=0.12)
        hedonic_meta = self.state.setdefault("meta", {}).setdefault("hedonic", {})
        hedonic_meta.update({
            "last_mode": mode,
            "last_ts": _now(),
            "last_meta": meta or {},
        })
        self._save()

    def register_global_slowdown(self, slowdown: float, meta: Optional[Dict[str, Any]] = None) -> None:
        """Record a physiological slowdown and bias drives toward recovery."""

        slowdown = max(0.0, min(1.0, float(slowdown)))
        phenom_meta = self.state.setdefault("meta", {}).setdefault("phenomenal", {})
        phenom_meta["global_slowdown"] = {
            "value": slowdown,
            "ts": _now(),
            "meta": meta or {},
        }
        if slowdown <= 0.0:
            self._save()
            return
        self.adjust_drive("restoration", 0.15 * slowdown, max_step=0.12)
        self.adjust_drive("task_activation", -0.18 * slowdown)

    def _record_drive_history(self, drive: str, value: float) -> None:
        drift_meta = self.state["meta"]["drift"]
        history: List[Tuple[str, float, float]] = drift_meta.setdefault("history", [])
        history.append((drive, _now(), float(value)))
        window = int(drift_meta.get("window", 25))
        if len(history) > 5 * window:
            del history[: len(history) - 5 * window]
        self._detect_drift(drive)

    def _detect_drift(self, drive: str) -> None:
        drift_meta = self.state["meta"]["drift"]
        window = int(drift_meta.get("window", 25))
        history: List[Tuple[str, float, float]] = drift_meta.get("history", [])
        recent = [value for drv, _, value in reversed(history) if drv == drive][:window]
        if len(recent) < window:
            return
        mean_value = sum(recent) / len(recent)
        variance = sum((val - mean_value) ** 2 for val in recent) / len(recent)
        std_dev = math.sqrt(variance)
        threshold = float(drift_meta.get("threshold", 0.18))
        if std_dev > threshold:
            last_logged = drift_meta.get("last_logged")
            timestamp = _now()
            if not last_logged or timestamp - float(last_logged) > 300:
                logger.warning(
                    "Drive %s exhibiting high volatility (std=%.3f > %.3f)",
                    drive,
                    std_dev,
                    threshold,
                )
                drift_meta["last_logged"] = timestamp

    def _update_intrinsic_model(self, intrinsic: float, extrinsic: float) -> None:
        meta = self.state["meta"]["intrinsic_model"]
        features = meta.get("last_features")
        prediction = float(meta.get("last_prediction", intrinsic))
        if not isinstance(features, dict):
            return
        target = max(0.0, min(1.0, 0.5 + 0.5 * extrinsic))
        model = self._intrinsic_model()
        model.update(features, target, prediction)
        self._store_intrinsic_model(model)

    def _register_reward_time(self) -> None:
        timestamp = _now()
        for drive in self.state["drives"]:
            self.state["meta"]["decay"][drive]["last_reward_time"] = timestamp

    def update_from_rewards(self, intrinsic: float, extrinsic: float) -> None:
        intrinsic_delta = 0.05 * (intrinsic - 0.5)
        extrinsic_delta = 0.05 * extrinsic
        self._apply_drive_update("curiosity", intrinsic_delta)
        self._apply_drive_update("competence", intrinsic_delta)
        self._apply_drive_update("social_bonding", extrinsic_delta)
        self._register_reward_time()
        self._update_intrinsic_model(intrinsic, extrinsic)
        for drive, value in self.state["drives"].items():
            success = 0.35 <= value <= 0.65
            self._update_decay_bandit(drive, success)
        self._save()
