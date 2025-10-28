from __future__ import annotations

from typing import Dict, Any, List
import logging
import math
import re
import time
import unicodedata
from collections import deque

from AGI_Evolutive.utils.llm_service import try_call_llm_dict


LOGGER = logging.getLogger(__name__)
LLM_SPEC_KEY = "language_style_observer"


class OnlineLinearModel:
    """Mini modèle linéaire online avec régularisation et facteur d'oubli."""

    def __init__(
        self,
        base_weights: Dict[str, float],
        learning_rate: float = 0.05,
        l2: float = 0.01,
        decay: float = 0.995,
    ) -> None:
        self.weights: Dict[str, float] = dict(base_weights)
        self.lr = learning_rate
        self.l2 = l2
        self.decay = decay

    def predict(self, features: Dict[str, float]) -> float:
        score = sum(self.weights.get(k, 0.0) * v for k, v in features.items())
        # utilisation d'une sigmoïde douce pour rester entre 0 et 1
        try:
            return 1.0 / (1.0 + math.exp(-score))
        except OverflowError:
            return 0.0 if score < 0 else 1.0

    def update(self, features: Dict[str, float], target: float) -> None:
        prediction = self.predict(features)
        error = prediction - target
        # facteur d'oubli : décroît légèrement les poids avant correction
        for key in list(self.weights.keys()):
            self.weights[key] *= self.decay
        for name, value in features.items():
            grad = error * value + self.l2 * self.weights.get(name, 0.0)
            self.weights[name] = self.weights.get(name, 0.0) - self.lr * grad


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


class StyleObserver:
    """Observe les textes lus et ajuste la voix / le lexique en douceur."""

    def __init__(self, self_model, homeostasis, voice_profile, lexicon, user_model=None) -> None:
        self.self_model = self_model
        self.homeo = homeostasis
        self.voice = voice_profile
        self.lex = lexicon
        self.user_model = user_model
        self.last_updates: deque[float] = deque(maxlen=32)  # timestamps pour rate limit
        self._state = self._ensure_state()
        self._channel_models: Dict[str, OnlineLinearModel] = {}
        self._fallback_model = OnlineLinearModel(
            self._state.setdefault(
                "fallback_weights",
                {
                    "bias": -0.8,
                    "sent_len": 0.04,
                    "exclam": 0.15,
                    "emoji": 0.12,
                    "question": 0.05,
                },
            ),
            learning_rate=0.03,
            l2=0.005,
            decay=0.998,
        )
        self._channel_base = {
            "bias": -0.2,
            "align": 0.9,
            "novelty": 0.8,
            "humor": 0.6,
            "analytic": 0.45,
            "cur": 0.5,
            "bond": 0.4,
            "chan_user": 0.25,
        }
        self._fallback_threshold = 0.62
        self._metrics = self._state.setdefault("metrics", {"events": []})
        self._regex_cache: Dict[str, re.Pattern] = {}

    # --------- API publique ----------
    def observe_text(self, text: str, source: str, *, channel: str) -> None:
        """Analyse un texte et déclenche des micro-ajustements si pertinent."""
        if not text or len(text) < 20:
            return

        # 1) extraire candidates (collocations, refs, blagues courtes)
        items = self._extract_candidates(text)

        # 2) scorer l'appétence (0..1)
        persona = self._persona_profile()
        drives = (getattr(self.homeo, "state", {}) or {}).get("drives", {})
        scored: List[tuple[float, Dict[str, Any]]] = []
        for it in items:
            score = self._like_score(it, persona, drives, channel=channel)
            if score >= 0.65 and self._aligned(it, persona):
                scored.append((score, it))
            else:
                self._record_feedback(it, channel, accepted=False)

        if not scored:
            return

        # 3) rate limit (max 5 ajouts / 5min)
        now = time.time()
        while self.last_updates and now - self.last_updates[0] >= 300:
            self.last_updates.popleft()
        budget = max(0, 5 - len(self.last_updates))
        if budget <= 0:
            return

        scored.sort(reverse=True)
        picked = self._llm_rank_candidates(
            scored,
            persona=persona,
            drives=drives,
            channel=channel,
            budget=budget,
        )
        if not picked:
            picked = [it for _, it in scored[:budget]]

        # 4) appliquer (petits incréments)
        for it in picked:
            self._apply_like(it)
            self.last_updates.append(now)
            self._record_feedback(it, channel, accepted=True)

    def _llm_rank_candidates(
        self,
        scored: List[tuple[float, Dict[str, Any]]],
        *,
        persona: Dict[str, Any],
        drives: Dict[str, float],
        channel: str,
        budget: int,
    ) -> List[Dict[str, Any]]:
        if not scored or budget <= 0:
            return []

        candidate_map: Dict[int, tuple[float, Dict[str, Any]]] = {
            idx: (score, item) for idx, (score, item) in enumerate(scored)
        }
        payload = {
            "channel": channel,
            "budget": int(budget),
            "persona": {
                "tone": str(persona.get("tone", "")) if isinstance(persona, dict) else "",
                "values": [
                    str(v) for v in (persona.get("values") or [])
                ]
                if isinstance(persona, dict)
                else [],
            },
            "drives": {k: float(v) for k, v in drives.items()},
            "candidates": [
                {
                    "id": idx,
                    "text": (item.get("text", "") or "")[:200],
                    "type": item.get("type", "unknown"),
                    "heuristic_score": round(float(score), 3),
                }
                for idx, (score, item) in candidate_map.items()
            ],
        }
        response = try_call_llm_dict(LLM_SPEC_KEY, input_payload=payload, logger=LOGGER)
        if not response:
            return []

        decisions = response.get("decisions")
        if not isinstance(decisions, list):
            return []

        accepted: list[tuple[float, int]] = []
        rejected_ids: set[int] = set()
        decision_map: Dict[int, Dict[str, Any]] = {}
        for entry in decisions:
            if not isinstance(entry, dict):
                continue
            try:
                cand_id = int(entry.get("id"))
            except (TypeError, ValueError):
                continue
            if cand_id not in candidate_map:
                continue
            decision_map[cand_id] = entry
            priority = float(entry.get("priority", candidate_map[cand_id][0]))
            if entry.get("accept"):
                accepted.append((priority, cand_id))
            else:
                rejected_ids.add(cand_id)

        for cand_id in rejected_ids:
            _, item = candidate_map[cand_id]
            item.setdefault("llm_feedback", {})["reason"] = decision_map.get(cand_id, {}).get(
                "justification", ""
            )
            self._record_feedback(item, channel, accepted=False)

        selected_items: List[Dict[str, Any]] = []
        used_ids: set[int] = set()

        if accepted:
            accepted.sort(key=lambda x: x[0], reverse=True)
            for priority, cand_id in accepted:
                if len(selected_items) >= budget:
                    break
                score, item = candidate_map[cand_id]
                item.setdefault("llm_feedback", {})["priority"] = priority
                item.setdefault("llm_feedback", {})["justification"] = decision_map.get(
                    cand_id, {}
                ).get("justification", "")
                selected_items.append(item)
                used_ids.add(cand_id)

        idx_order = list(range(len(candidate_map)))
        for idx in idx_order:
            if len(selected_items) >= budget:
                break
            if idx in used_ids or idx in rejected_ids:
                continue
            _, item = candidate_map[idx]
            selected_items.append(item)
            used_ids.add(idx)

        return selected_items

    # --------- Extracteurs ----------
    def _extract_candidates(self, text: str) -> List[Dict[str, Any]]:
        # Collocations 2–5 mots + détection simple d’idiomes/blagues/refs
        words = re.findall(r"[A-Za-zÀ-ÿ]+(?:'[A-Za-zÀ-ÿ]+)?", text)
        items: List[Dict[str, Any]] = []
        for n in (2, 3, 4, 5):
            for i in range(len(words) - n + 1):
                phr = " ".join(words[i : i + n])
                if len(phr) < 8:
                    continue
                items.append({"type": "collocation", "text": phr})

        # blagues ou punchlines ultra basiques (heuristique)
        normalized_text = _strip_accents(text)
        if self._pattern("c'est (pas )?faux|est (un|une|le|la|l') (twist|fait)\b").search(
            normalized_text
        ):
            items.append({"type": "punch", "text": "fun fact"})
        # références pop (à étendre)
        if self._pattern("matrix|inception|star wars|one piece|the expanse|gundam").search(
            normalized_text
        ):
            items.append({"type": "reference", "text": "réf pop-culture"})

        # fallback classifier pour détecter des phrases stylées
        items.extend(self._fallback_candidates(text))

        return items

    # --------- Scoring ----------
    def _like_score(
        self,
        item: Dict[str, Any],
        persona: Dict[str, Any],
        drives: Dict[str, float],
        *,
        channel: str,
    ) -> float:
        tone = (persona.get("tone") or "").lower()
        raw_vals = persona.get("values", [])
        if isinstance(raw_vals, dict):
            vals = [key.lower() for key, level in raw_vals.items() if level]
        else:
            vals = [v.lower() for v in raw_vals]

        # features simples
        txt = item["text"].lower()
        novelty = 1.0 if not self._has_collocation(txt) else 0.4
        humor = 1.0 if item["type"] in ("punch",) else 0.2
        analytic = 1.0 if len(txt.split()) >= 3 and item["type"] == "collocation" else 0.4

        # match drives (si homeo impulsion “social_bonding”/“curiosity”)
        cur = float(drives.get("curiosity", 0.5))
        bond = float(drives.get("social_bonding", 0.5))

        # alignement persona simple
        align = 0.5
        if "analytical" in tone:
            align += 0.2 * analytic
        if "friendly" in tone or "warm" in tone:
            align += 0.2 * bond
        if "precision" in vals:
            align += 0.1

        # canal : booste l’utilisateur (on apprend plus de toi)
        chan = 0.15 if channel == "user" else 0.0

        features = {
            "bias": 1.0,
            "align": align,
            "novelty": novelty,
            "humor": humor,
            "analytic": analytic,
            "cur": cur,
            "bond": bond,
        }
        if channel == "user":
            features["chan_user"] = 1.0
        model = self._channel_model(channel)
        score = model.predict(features)
        if item.get("type") == "fallback":
            fallback_feats = item.setdefault(
                "_fallback_features", self._fallback_features(item["text"])
            )
            if fallback_feats:
                fallback_score = self._fallback_model.predict(fallback_feats)
                score = (score + fallback_score) / 2.0
        item["_features"] = features
        return max(0.0, min(1.0, score))

    def _aligned(self, item: Dict[str, Any], persona: Dict[str, Any]) -> bool:
        # garde-fou : blacklist mots, pas d’insulte, pas contraire aux valeurs, etc.
        txt = item["text"].lower()
        if len(txt) > 60:
            return False
        # exemples de filtres:
        bad = ["insulte", "raciste"]  # à remplacer par ta vraie liste
        if any(b in txt for b in bad):
            return False
        return True

    # --------- Application ----------
    def _apply_like(self, item: Dict[str, Any]) -> None:
        txt = item["text"].strip()
        # Favoriser la collocation dans le lexique
        try:
            self.lex.prefer(txt)
        except Exception:
            pass
        # Petite adaptation de la voix si pertinent
        try:
            if item["type"] == "punch":
                self.voice.bump("humor", +0.03)
                self.voice.bump("storytelling", +0.02)
            else:
                self.voice.bump("analytical", +0.02)
        except Exception:
            pass
        # Enregistre une mémoire de style appris (pour traçabilité)
        try:
            state = getattr(self.self_model, "state", None)
            if state is not None:
                state.setdefault("style_learned", []).append({"text": txt, "ts": time.time(), "source": "style_observer"})
        except Exception:
            pass

    def _persona_profile(self) -> Dict[str, Any]:
        persona: Dict[str, Any] = {}
        try:
            if self.self_model is not None:
                base_state = getattr(self.self_model, "state", {}) or {}
                persona = dict(base_state.get("persona", {}) or {})
        except Exception:
            persona = {}
        if self.user_model is not None and hasattr(self.user_model, "describe"):
            try:
                user_state = self.user_model.describe() or {}
                user_persona = user_state.get("persona") or {}
                if user_persona:
                    merged = dict(persona)
                    merged.update(user_persona)
                    persona = merged
            except Exception:
                pass
        return persona

    # --------- Helpers internes ----------
    def _ensure_state(self) -> Dict[str, Any]:
        if self.self_model is None:
            return {"channel_weights": {}}
        state = getattr(self.self_model, "state", None)
        if state is None:
            state = {}
            setattr(self.self_model, "state", state)
        bucket = state.setdefault("style_observer", {})
        bucket.setdefault("channel_weights", {})
        return bucket

    def _channel_model(self, channel: str) -> OnlineLinearModel:
        if channel not in self._channel_models:
            stored = self._state["channel_weights"].get(channel, {})
            weights = dict(self._channel_base)
            weights.update(stored)
            self._channel_models[channel] = OnlineLinearModel(
                weights,
                learning_rate=0.04,
                l2=0.01,
                decay=0.997,
            )
        return self._channel_models[channel]

    def _persist_weights(self, channel: str) -> None:
        if self.self_model is None:
            return
        model = self._channel_models.get(channel)
        if model is None:
            return
        self._state["channel_weights"][channel] = dict(model.weights)

    def _persist_fallback(self) -> None:
        if self.self_model is None:
            return
        self._state["fallback_weights"] = dict(self._fallback_model.weights)

    def _pattern(self, expr: str) -> re.Pattern:
        key = f"pattern::{expr}"
        pat = self._regex_cache.get(key)
        if pat is None:
            flags = re.IGNORECASE
            normalized = _strip_accents(expr)
            pat = re.compile(normalized, flags)
            self._regex_cache[key] = pat
        return pat

    def _has_collocation(self, text: str) -> bool:
        try:
            lex_data = getattr(self.lex, "lex", {}) or {}
            collocs = lex_data.get("collocations", {}) if isinstance(lex_data, dict) else {}
            if hasattr(collocs, "get"):
                return bool(collocs.get(text))
        except Exception:
            return False
        return False

    def _fallback_candidates(self, text: str) -> List[Dict[str, Any]]:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        items: List[Dict[str, Any]] = []
        for sent in sentences:
            feats = self._fallback_features(sent)
            if not feats:
                continue
            score = self._fallback_model.predict(feats)
            if score >= self._fallback_threshold:
                items.append({"type": "fallback", "text": sent, "_fallback_features": feats})
        return items

    def _fallback_features(self, sent: str) -> Dict[str, float]:
        if len(sent) < 6:
            return {}
        normalized = _strip_accents(sent)
        emoji = len(re.findall(r"[\u2600-\u27BF\U0001F300-\U0001FAD6]", sent))
        return {
            "bias": 1.0,
            "sent_len": min(len(normalized) / 80.0, 2.0),
            "exclam": sent.count("!"),
            "emoji": float(emoji > 0),
            "question": float("?" in sent),
        }

    def _record_feedback(self, item: Dict[str, Any], channel: str, *, accepted: bool) -> None:
        feats = item.get("_features")
        if feats and channel in self._channel_models:
            self._channel_models[channel].update(feats, 1.0 if accepted else 0.0)
            self._persist_weights(channel)
        fallback_feats = item.get("_fallback_features")
        if fallback_feats:
            self._fallback_model.update(fallback_feats, 1.0 if accepted else 0.0)
            self._persist_fallback()
        self._log_event(
            {
                "text": item.get("text", "")[:160],
                "channel": channel,
                "accepted": accepted,
                "type": item.get("type"),
                "ts": time.time(),
            }
        )

    def _log_event(self, payload: Dict[str, Any]) -> None:
        events: List[Dict[str, Any]] = self._metrics.setdefault("events", [])
        events.append(payload)
        # garde une trace raisonnable
        if len(events) > 200:
            del events[:-200]
