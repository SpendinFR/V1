from __future__ import annotations
from typing import Dict, Any, Optional, List
import os, json, datetime as dt, time, math, logging

from AGI_Evolutive.utils.jsonsafe import json_sanitize
from AGI_Evolutive.utils.llm_service import try_call_llm_dict


LOGGER = logging.getLogger(__name__)

class UserModel:
    """
    - valeurs & persona (déclaratif)
    - préférences (Beta-like: (pos, neg) -> prob)
    - routines temporelles (jour/heure -> activités probables)
    - extraction/learning implicite depuis la mémoire
    """
    def __init__(self, path: str = "data/user_model.json") -> None:
        self.path = path
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self.pref_half_life = 60.0 * 60.0 * 24.0 * 14.0  # 2 semaines
        self.pref_learning_rate = 0.15
        self.pref_clip = 5.0
        self.pref_blend = 0.5
        self.routine_half_life = 60.0 * 60.0 * 24.0 * 21.0  # 3 semaines
        self.routine_baseline = 0.2

        self.state: Dict[str, Any] = {
            "persona": {"tone": "neutral", "values": {"honesty": 1.0, "helpfulness": 1.0}},
            "preferences": {},  # label -> {"pos": int, "neg": int, "prob": float}
            "routines": {},     # "Tue:12" -> {"fast_food": 0.6, "gym": 0.1}
            "last_update": time.time(),
        }
        self._load()

    def _touch(self) -> None:
        self.state["last_update"] = time.time()

    def _load(self) -> None:
        if os.path.exists(self.path):
            try: self.state = json.load(open(self.path, "r", encoding="utf-8"))
            except Exception: pass

    def save(self) -> None:
        self._touch()
        json.dump(json_sanitize(self.state), open(self.path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # ---------- Persona / valeurs ----------
    def set_value(self, key: str, level: float) -> None:
        self.state["persona"]["values"][key] = float(max(0.0, min(1.0, level))); self.save()

    def set_tone(self, tone: str) -> None:
        self.state["persona"]["tone"] = tone; self.save()

    def describe(self) -> Dict[str, Any]:
        return self.state

    # ---------- Préférences (apprentissage Beta-like) ----------
    def _ensure_pref(self, label: str) -> Dict[str, Any]:
        p = self.state["preferences"].get(label)
        if not p:
            p = {"pos": 1.0, "neg": 1.0, "prob": 0.5, "last_seen": time.time(), "glm": {"weights": {}}}
            self.state["preferences"][label] = p
        else:
            if "glm" not in p:
                p["glm"] = {"weights": {}}
            if "last_seen" not in p:
                p["last_seen"] = self.state.get("last_update", time.time())
            if isinstance(p.get("pos"), int):
                p["pos"] = float(p["pos"])
            if isinstance(p.get("neg"), int):
                p["neg"] = float(p["neg"])
        return p

    def _decay_factor(self, elapsed: float, half_life: float) -> float:
        if elapsed <= 0 or half_life <= 0:
            return 1.0
        return math.exp(-elapsed / half_life)

    @staticmethod
    def _sigmoid(x: float) -> float:
        # valeurs bornées pour rester stable
        if x < -60.0:
            return 0.0
        if x > 60.0:
            return 1.0
        return 1.0 / (1.0 + math.exp(-x))

    def _normalize_timestamp(self, value: float) -> float:
        if not isinstance(value, (int, float)):
            return time.time()
        ts = float(value)
        if not math.isfinite(ts):
            return time.time()
        # Heuristically downscale timestamps provided in ms/us/ns
        # to keep them within datetime.fromtimestamp's supported range.
        for _ in range(3):
            if abs(ts) <= 1e11:
                break
            ts /= 1000.0
        return ts

    def _feature_vector(self, timestamp: float, context: Optional[Dict[str, Any]]) -> Dict[str, float]:
        features: Dict[str, float] = {"bias": 1.0}
        ts = self._normalize_timestamp(timestamp)
        try:
            dt_obj = dt.datetime.fromtimestamp(ts)
        except (OverflowError, ValueError, OSError):
            dt_obj = dt.datetime.fromtimestamp(time.time())
        features[f"hour::{dt_obj.hour:02d}"] = 1.0
        features[f"weekday::{dt_obj.weekday()}"] = 1.0
        if context:
            for key, value in context.items():
                if value is None:
                    continue
                if isinstance(value, (int, float)):
                    features[f"num::{key}"] = float(value)
                else:
                    features[f"cat::{key}::{value}"] = 1.0
        return features

    def _predict_glm(self, weights: Dict[str, float], features: Dict[str, float]) -> float:
        score = 0.0
        for name, val in features.items():
            if not isinstance(val, (int, float)):
                continue
            score += weights.get(name, 0.0) * float(val)
        return self._sigmoid(score)

    def _update_glm(self, p: Dict[str, Any], liked: bool, features: Dict[str, float]) -> float:
        weights: Dict[str, float] = p.setdefault("glm", {}).setdefault("weights", {})
        prediction = self._predict_glm(weights, features)
        target = 1.0 if liked else 0.0
        error = target - prediction
        for name, val in features.items():
            if not isinstance(val, (int, float)):
                continue
            w = weights.get(name, 0.0)
            w += self.pref_learning_rate * error * float(val)
            w = max(-self.pref_clip, min(self.pref_clip, w))
            weights[name] = w
        return self._predict_glm(weights, features)

    def _blend_probabilities(self, beta_prob: float, glm_prob: float) -> float:
        return float(max(0.0, min(1.0, (1.0 - self.pref_blend) * beta_prob + self.pref_blend * glm_prob)))

    def _top_preferences(self, limit: int = 5) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        preferences = self.state.get("preferences", {})
        for label, payload in preferences.items():
            if not isinstance(payload, dict):
                continue
            prob = payload.get("prob")
            if isinstance(prob, (int, float)):
                items.append({"label": label, "probability": float(max(0.0, min(1.0, prob)))})
        items.sort(key=lambda entry: entry.get("probability", 0.0), reverse=True)
        return items[:limit]

    def _sample_routines(self, limit: int = 4) -> List[Dict[str, Any]]:
        routines = []
        buckets = self.state.get("routines", {})
        for bucket, entries in buckets.items():
            if not isinstance(entries, dict):
                continue
            for label, payload in entries.items():
                prob = payload.get("prob") if isinstance(payload, dict) else None
                if isinstance(prob, (int, float)):
                    routines.append(
                        {
                            "time_bucket": bucket,
                            "activity": label,
                            "probability": float(max(0.0, min(1.0, prob))),
                        }
                    )
        routines.sort(key=lambda entry: entry.get("probability", 0.0), reverse=True)
        return routines[:limit]

    def _build_llm_payload(self, memories: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        recent_memories: List[Dict[str, Any]] = []
        for memory in reversed(memories[-20:]):
            if not isinstance(memory, dict):
                continue
            text = str(memory.get("content") or memory.get("text") or "").strip()
            if not text:
                continue
            metadata = memory.get("metadata") if isinstance(memory.get("metadata"), dict) else {}
            recent_memories.append(
                {
                    "text": text[:240],
                    "metadata": {
                        key: metadata[key]
                        for key in ("channel", "partner", "location", "sentiment")
                        if metadata.get(key) is not None
                    },
                }
            )
        persona = self.state.get("persona", {})
        payload = {
            "current_persona": {
                "tone": persona.get("tone", "neutral"),
                "values": persona.get("values", {}),
            },
            "top_preferences": self._top_preferences(),
            "routines": self._sample_routines(),
            "recent_memories": recent_memories,
        }
        if not any(payload.values()):
            return None
        return payload

    def _llm_update_profile(self, memories: List[Dict[str, Any]]) -> None:
        payload = self._build_llm_payload(memories)
        if not payload:
            return
        response = try_call_llm_dict(
            "user_model",
            input_payload=payload,
            logger=LOGGER,
        )
        if not response:
            return

        persona = self.state.setdefault("persona", {})
        dirty = False

        tone = response.get("tone")
        if isinstance(tone, str) and tone:
            if persona.get("tone") != tone:
                persona["tone"] = tone
                dirty = True

        satisfaction = response.get("satisfaction")
        if isinstance(satisfaction, (int, float)):
            clipped = float(max(0.0, min(1.0, satisfaction)))
            if persona.get("satisfaction") != clipped:
                persona["satisfaction"] = clipped
                dirty = True

        traits = response.get("persona_traits")
        if isinstance(traits, list):
            persona["traits"] = traits
            dirty = True

        notes = response.get("notes")
        if isinstance(notes, str):
            self.state.setdefault("insights", {})["user_model_notes"] = notes
            dirty = True

        self.state.setdefault("llm_profiles", {})["user_model"] = response
        dirty = True
        if dirty:
            self.save()

    def observe_preference(
        self,
        label: str,
        liked: bool,
        weight: float = 1.0,
        context: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        p = self._ensure_pref(label)
        now = timestamp or time.time()
        elapsed = now - float(p.get("last_seen", now))
        decay = self._decay_factor(elapsed, self.pref_half_life)
        p["pos"] = max(1e-6, p["pos"] * decay)
        p["neg"] = max(1e-6, p["neg"] * decay)
        weight = max(0.0, float(weight))
        if liked:
            p["pos"] += max(1e-6, weight)
        else:
            p["neg"] += max(1e-6, weight)

        features = self._feature_vector(now, context)
        glm_prob = self._update_glm(p, liked, features)
        beta_prob = p["pos"] / float(p["pos"] + p["neg"])
        p["prob"] = self._blend_probabilities(beta_prob, glm_prob)
        p["last_seen"] = now
        self.save()

    def prior(
        self,
        label: str,
        context: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> float:
        p = self._ensure_pref(label)
        now = timestamp or time.time()
        elapsed = now - float(p.get("last_seen", now))
        decay = self._decay_factor(elapsed, self.pref_half_life)
        pos = max(1e-6, p["pos"] * decay)
        neg = max(1e-6, p["neg"] * decay)
        beta_prob = pos / float(pos + neg)
        features = self._feature_vector(now, context)
        glm_prob = self._predict_glm(p.get("glm", {}).get("weights", {}), features)
        return self._blend_probabilities(beta_prob, glm_prob)

    # ---------- Routines temporelles ----------
    @staticmethod
    def _key_for_time(t: Optional[dt.datetime] = None) -> str:
        t = t or dt.datetime.now()
        return f"{['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][t.weekday()]}:{t.hour:02d}"

    def _ensure_routine_entry(self, bucket: Dict[str, Any], label: str) -> Dict[str, Any]:
        entry = bucket.get(label)
        if isinstance(entry, dict):
            entry.setdefault("pos", 1.0)
            entry.setdefault("neg", 1.0)
            entry.setdefault("prob", entry["pos"] / (entry["pos"] + entry["neg"]))
            entry.setdefault("last_seen", self.state.get("last_update", time.time()))
            return entry
        # compatibilité avec anciennes versions stockant un float
        score = float(entry) if entry is not None else 0.0
        converted = {
            "pos": 1.0 + max(0.0, score) * 4.0,
            "neg": 1.0 + max(0.0, 1.0 - score) * 4.0,
            "prob": float(max(0.0, min(1.0, score))),
            "last_seen": self.state.get("last_update", time.time()),
        }
        bucket[label] = converted
        return converted

    def observe_routine(
        self,
        label: str,
        t: Optional[dt.datetime] = None,
        strength: float = 0.2,
        timestamp: Optional[float] = None,
    ) -> None:
        k = self._key_for_time(t)
        cur = self.state["routines"].get(k, {})
        now = timestamp or (t.timestamp() if t else time.time())
        entry = self._ensure_routine_entry(cur, label)
        elapsed = now - float(entry.get("last_seen", now))
        decay = self._decay_factor(elapsed, self.routine_half_life)
        entry["pos"] = max(1e-6, entry["pos"] * decay)
        entry["neg"] = max(1e-6, entry["neg"] * decay)
        weight = max(0.0, float(strength))
        entry["pos"] += max(weight, 1e-6)
        entry["neg"] += self.routine_baseline * (1.0 - decay)
        entry["prob"] = entry["pos"] / float(entry["pos"] + entry["neg"])
        entry["last_seen"] = now
        cur[label] = entry
        self.state["routines"][k] = cur
        self.save()

    def routine_bias(
        self,
        label: str,
        t: Optional[dt.datetime] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        k = self._key_for_time(t)
        bucket = self.state["routines"].get(k, {})
        entry = self._ensure_routine_entry(bucket, label)
        now = time.time()
        elapsed = now - float(entry.get("last_seen", now))
        decay = self._decay_factor(elapsed, self.routine_half_life)
        pos = max(1e-6, entry["pos"] * decay)
        neg = max(1e-6, entry["neg"] * decay)
        prob = pos / float(pos + neg)
        bucket[label].update({"pos": pos, "neg": neg, "prob": prob, "last_seen": now})
        if context:
            # légère modulation contextuelle basée sur préférences existantes
            for ctx_key, ctx_val in context.items():
                if ctx_key == "activity_preference":
                    pref_prob = self.prior(ctx_val)
                    prob = self._blend_probabilities(prob, pref_prob)
        self.state["routines"][k] = bucket
        return float(max(0.0, min(1.0, prob)))

    # ---------- Learning implicite depuis la mémoire ----------
    def ingest_memories(self, memories: List[Dict[str, Any]]) -> int:
        """
        Parse des entrées mémoire pour détecter des signaux:
        - 'like:<label>' / 'dislike:<label>'
        - 'did:<activity>'
        - 'tone:<style>'
        """
        n = 0
        for m in memories[-200:]:
            text = (m.get("content") or m.get("text") or "").lower()
            meta = m.get("metadata") or {}
            if not text:
                continue

            context = {k: meta.get(k) for k in ("channel", "partner", "location") if meta.get(k) is not None}
            timestamp = meta.get("timestamp")
            if isinstance(timestamp, (int, float)):
                ts = self._normalize_timestamp(timestamp)
            else:
                ts = time.time()

            tokens = text.replace(",", " ").replace(";", " ").split()
            for tok in tokens:
                if tok.startswith("like:"):
                    self.observe_preference(tok.split(":", 1)[1], True, context=context, timestamp=ts)
                    n += 1
                elif tok.startswith("love:") or tok.startswith("adore:"):
                    self.observe_preference(tok.split(":", 1)[1], True, weight=1.5, context=context, timestamp=ts)
                    n += 1
                elif tok.startswith("dislike:") or tok.startswith("hate:"):
                    self.observe_preference(tok.split(":", 1)[1], False, context=context, timestamp=ts)
                    n += 1
                elif tok.startswith("did:"):
                    self.observe_routine(tok.split(":", 1)[1], strength=0.4, timestamp=ts)
                    n += 1
                elif tok.startswith("tone:"):
                    self.set_tone(tok.split(":", 1)[1])
                    n += 1

            if "sentiment" in meta:
                sent = str(meta.get("sentiment"))
                if sent.startswith("pos:"):
                    label = sent.split(":", 1)[1]
                    self.observe_preference(label, True, weight=1.2, context=context, timestamp=ts)
                    n += 1
                elif sent.startswith("neg:"):
                    label = sent.split(":", 1)[1]
                    self.observe_preference(label, False, weight=1.2, context=context, timestamp=ts)
                    n += 1

            if meta.get("activities"):
                for activity in meta["activities"]:
                    self.observe_routine(str(activity), strength=0.3, timestamp=ts)
                    n += 1
        if n:
            try:
                self._llm_update_profile(memories)
            except Exception:
                LOGGER.debug("LLM user model update failed", exc_info=True)
        return n
