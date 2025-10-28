"""Phenomenal kernel orchestrating hedonic vs. instrumental modes.

This simplified kernel drops direct machine physiology sampling.  External
modules may inject alerts (e.g. overheating) via :meth:`PhenomenalKernel.register_alert`
which immediately affect the shared phenomenological state and propagate to the
rest of the architecture.  Energy, arousal and veto are derived from emotional
hints, progress signals and these alerts.  A :class:`ModeManager` keeps
historical ratios between Travail and Flânerie while enforcing configurable
budgets.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, Optional

from AGI_Evolutive.utils.llm_service import try_call_llm_dict


__all__ = ["PhenomenalKernel", "ModeManager", "SystemAlert"]


LOGGER = logging.getLogger(__name__)


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


@dataclass
class SystemAlert:
    """Represents an immediate physiological-like alert injected by triggers."""

    kind: str
    intensity: float = 1.0
    slowdown: float = 1.0
    duration: float = 30.0
    timestamp: float = field(default_factory=lambda: float(time.time()))

    def remaining(self, now: Optional[float] = None) -> float:
        now = float(now or time.time())
        return max(0.0, self.duration - (now - self.timestamp))

    def as_dict(self, now: Optional[float] = None) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "intensity": _clip(self.intensity),
            "slowdown": _clip(self.slowdown),
            "remaining": self.remaining(now),
        }


class PhenomenalKernel:
    """Blends affective hints with injected alerts to drive phenomenology."""

    def __init__(self, smoothing: float = 0.35) -> None:
        self.smoothing = _clip(float(smoothing), 0.05, 0.9)
        self._state: Dict[str, Any] = {}
        self._ema: Dict[str, float] = {}
        self._alerts: Deque[SystemAlert] = deque()

    # ------------------------------------------------------------------
    # Alert handling
    # ------------------------------------------------------------------
    def register_alert(
        self,
        kind: str,
        *,
        intensity: float = 1.0,
        slowdown: Optional[float] = None,
        duration: float = 30.0,
        timestamp: Optional[float] = None,
    ) -> None:
        """Register an alert that should immediately bias the kernel state."""

        if not kind:
            return
        intensity = _clip(float(intensity))
        slowdown_val = float(slowdown) if slowdown is not None else intensity
        slowdown_val = _clip(slowdown_val)
        duration = max(0.5, float(duration))
        ts = float(timestamp) if timestamp is not None else float(time.time())
        self._alerts.append(
            SystemAlert(kind=kind, intensity=intensity, slowdown=slowdown_val, duration=duration, timestamp=ts)
        )

    def _merge_external_alerts(self, alerts: Optional[Iterable[Dict[str, Any]]]) -> None:
        if not alerts:
            return
        for alert in alerts:
            if not isinstance(alert, dict):
                continue
            kind = str(alert.get("kind") or "external")
            intensity = float(alert.get("intensity", alert.get("priority", 1.0)))
            slowdown = alert.get("slowdown")
            duration = alert.get("duration", 30.0)
            timestamp = alert.get("timestamp")
            self.register_alert(
                kind,
                intensity=intensity,
                slowdown=slowdown,
                duration=duration,
                timestamp=timestamp,
            )

    def _active_alerts(self, now: Optional[float] = None) -> Deque[SystemAlert]:
        now = float(now or time.time())
        active: Deque[SystemAlert] = deque()
        while self._alerts:
            alert = self._alerts.popleft()
            if alert.remaining(now) > 0.0:
                active.append(alert)
        self._alerts = active
        return deque(active)

    # ------------------------------------------------------------------
    # EMA helper
    # ------------------------------------------------------------------
    def _ema_update(self, key: str, value: float) -> float:
        if key not in self._ema:
            self._ema[key] = float(value)
        else:
            self._ema[key] = self.smoothing * float(value) + (1.0 - self.smoothing) * self._ema[key]
        return self._ema[key]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(
        self,
        *,
        emotional_state: Optional[Dict[str, float]] = None,
        novelty: Optional[float] = None,
        belief: Optional[float] = None,
        progress: Optional[float] = None,
        extrinsic_reward: Optional[float] = None,
        hedonic_signal: Optional[float] = None,
        fatigue: Optional[float] = None,
        alerts: Optional[Iterable[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Update the phenomenal state using affective hints and alerts."""

        self._merge_external_alerts(alerts)
        now = time.time()
        active_alerts = list(self._active_alerts(now))
        alert_pressure = max((alert.intensity for alert in active_alerts), default=0.0)
        slowdown_pressure = max((alert.slowdown for alert in active_alerts), default=0.0)

        emotional_state = emotional_state or {}
        valence = float(emotional_state.get("valence", 0.0) or 0.0)
        arousal = float(emotional_state.get("arousal", 0.0) or 0.0)
        novelty = 0.5 if novelty is None else float(novelty)
        belief = 0.5 if belief is None else float(belief)
        progress = 0.3 if progress is None else float(progress)
        extrinsic_reward = 0.0 if extrinsic_reward is None else float(extrinsic_reward)
        hedonic_signal = 0.0 if hedonic_signal is None else float(hedonic_signal)
        fatigue = 0.5 if fatigue is None else float(fatigue)

        base_energy = 0.55 + 0.25 * valence - 0.35 * fatigue + 0.2 * progress
        base_energy -= 0.5 * alert_pressure
        energy = _clip(self._ema_update("energy", base_energy))

        arousal_mix = 0.45 + 0.3 * arousal + 0.15 * novelty + 0.15 * extrinsic_reward - 0.2 * alert_pressure
        arousal_level = _clip(self._ema_update("arousal", arousal_mix))

        resonance_val = 0.45 + 0.35 * belief - 0.25 * alert_pressure
        resonance = _clip(self._ema_update("resonance", resonance_val))

        surprise_val = 0.25 + 0.5 * novelty + 0.2 * abs(extrinsic_reward)
        surprise = _clip(self._ema_update("surprise", surprise_val))

        fatigue_hint = _clip(self._ema_update("fatigue", fatigue))

        veto_val = 0.1 + 0.55 * alert_pressure + 0.35 * fatigue_hint
        veto_probability = _clip(self._ema_update("veto_probability", veto_val))

        hedonic_base = 0.35 + 0.45 * max(0.0, hedonic_signal) - 0.25 * alert_pressure
        hedonic_reward = _clip(self._ema_update("hedonic_reward", hedonic_base))

        feel_like = "continue"
        if alert_pressure > 0.6 or energy < 0.3:
            feel_like = "pause"
        elif alert_pressure > 0.35 or energy < 0.4:
            feel_like = "slow"

        mode_bias = energy - 0.3 * fatigue_hint + 0.2 * (hedonic_reward - 0.5) - 0.4 * alert_pressure
        mode_suggestion = "travail" if mode_bias >= 0.0 else "flanerie"
        if feel_like == "pause":
            mode_suggestion = "flanerie"

        global_slowdown = _clip(self._ema_update("global_slowdown", slowdown_pressure))

        self._state = {
            "timestamp": now,
            "energy": energy,
            "arousal": arousal_level,
            "resonance": resonance,
            "surprise": surprise,
            "veto_probability": veto_probability,
            "feel_like": feel_like,
            "mode_suggestion": mode_suggestion,
            "hedonic_reward": hedonic_reward,
            "fatigue": fatigue_hint,
            "global_slowdown": global_slowdown,
            "alert_pressure": alert_pressure,
            "active_alerts": [alert.as_dict(now) for alert in active_alerts],
        }
        enrichment = self._llm_interpret(
            emotional_state=emotional_state,
            kernel_state=self._state,
        )
        if enrichment:
            self._state["llm_interpretation"] = enrichment
        return dict(self._state)

    def _llm_interpret(
        self,
        *,
        emotional_state: Dict[str, float],
        kernel_state: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        payload = {
            "emotional_state": emotional_state,
            "kernel_state": kernel_state,
        }
        response = try_call_llm_dict(
            "phenomenal_kernel",
            input_payload=payload,
            logger=LOGGER,
        )
        if not response:
            return None
        current_state = response.get("current_state")
        recommended = response.get("recommended_mode")
        justification = response.get("justification")
        if not isinstance(current_state, str) or not isinstance(recommended, str):
            return None
        if justification is not None and not isinstance(justification, str):
            justification = None
        return {
            "current_state": current_state,
            "recommended_mode": recommended,
            "justification": justification or "",
            "notes": response.get("notes", ""),
        }

    @property
    def state(self) -> Dict[str, Any]:
        return dict(self._state)


class ModeManager:
    """Soft arbitration between instrumental work and flânerie."""

    def __init__(
        self,
        target_work_ratio: float = 0.8,
        window: float = 900.0,
        enter_energy_threshold: float = 0.35,
        exit_energy_threshold: float = 0.5,
        exit_veto_threshold: float = 0.4,
        min_switch_interval: float = 60.0,
    ) -> None:
        self.target_work_ratio = _clip(target_work_ratio, 0.5, 0.95)
        self.window = max(60.0, float(window))
        self.enter_energy_threshold = max(0.0, min(1.0, float(enter_energy_threshold)))
        self.exit_energy_threshold = max(0.0, min(1.0, float(exit_energy_threshold)))
        self.exit_veto_threshold = max(0.0, min(1.0, float(exit_veto_threshold)))
        self.min_switch_interval = max(10.0, float(min_switch_interval))
        self._history: Deque[tuple[float, float, str]] = deque()
        self._mode = "travail"
        self._last_ts = time.time()
        self._last_switch_ts = self._last_ts

    def _prune(self, now: float) -> None:
        limit = now - self.window
        while self._history and self._history[0][0] < limit:
            self._history.popleft()

    def _ratio(self) -> float:
        total = sum(dt for _, dt, _ in self._history) or 1.0
        flanerie = sum(dt for _, dt, mode in self._history if mode == "flanerie")
        return flanerie / total

    def update(self, kernel_state: Dict[str, Any], urgent: bool = False) -> Dict[str, Any]:
        now = time.time()
        dt = max(1.0, now - self._last_ts)
        self._last_ts = now
        previous_mode = self._mode
        self._history.append((now, dt, previous_mode))
        self._prune(now)

        ratio = self._ratio()
        remaining_budget = max(0.0, (1.0 - self.target_work_ratio) - ratio)
        suggestion = kernel_state.get("mode_suggestion", "travail")
        energy = float(kernel_state.get("energy", 0.5))
        veto = float(kernel_state.get("veto_probability", 0.0))
        feel_like = kernel_state.get("feel_like", "continue")
        slowdown = float(kernel_state.get("global_slowdown", 0.0))
        since_switch = now - self._last_switch_ts

        if urgent or slowdown >= 0.9:
            if self._mode != "travail":
                self._mode = "travail"
                self._last_switch_ts = now
        else:
            if previous_mode == "travail":
                if (
                    remaining_budget > 0.02
                    and since_switch >= self.min_switch_interval
                    and (
                        suggestion == "flanerie"
                        or feel_like in {"pause", "slow"}
                        or energy < self.enter_energy_threshold
                    )
                ):
                    self._mode = "flanerie"
                    self._last_switch_ts = now
            else:  # currently in flânerie
                should_exit = (
                    remaining_budget <= 0.0
                    or suggestion == "travail"
                    or energy > self.exit_energy_threshold
                    or slowdown > 0.5
                )
                energy_ok = energy > max(0.0, self.exit_energy_threshold * 0.8)
                if remaining_budget <= 0.0:
                    energy_ok = True
                if (
                    since_switch >= self.min_switch_interval
                    and should_exit
                    and energy_ok
                    and veto < self.exit_veto_threshold
                ):
                    self._mode = "travail"
                    self._last_switch_ts = now

        return {
            "mode": self._mode,
            "flanerie_ratio": ratio,
            "flanerie_budget_remaining": max(0.0, (1.0 - self.target_work_ratio) - ratio),
            "urgent": urgent,
        }
