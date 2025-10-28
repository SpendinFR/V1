from __future__ import annotations

import json
import logging
import math
import os
import random
import re
import time
import unicodedata
from collections import defaultdict, deque
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from AGI_Evolutive.utils.llm_service import try_call_llm_dict

Number = float


def _now() -> float:
    return time.time()


def _safe(d: Any, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def _normalize_text(s: str) -> str:
    if not s:
        return ""
    norm = unicodedata.normalize("NFD", s)
    norm = "".join(ch for ch in norm if unicodedata.category(ch) != "Mn")
    return norm.lower()


def _tok(s: str) -> List[str]:
    return re.findall(r"[a-z]{3,}", _normalize_text(s))


class OnlineWeightLearner:
    """Online ridge-like learner with drift control."""

    def __init__(self, base_weights: Dict[str, float], state: Dict[str, Any] | None = None):
        self.base_weights = dict(base_weights)
        self.weights: Dict[str, float] = dict(base_weights)
        self.count: Dict[str, float] = defaultdict(lambda: 1.0)
        self.ewma_short: Dict[str, float] = defaultdict(float)
        self.ewma_long: Dict[str, float] = defaultdict(float)
        self.lr = 0.2
        self.l2 = 0.001
        if state:
            for k, v in state.get("weights", {}).items():
                try:
                    self.weights[k] = float(v)
                except (TypeError, ValueError):
                    continue
            for k, v in state.get("count", {}).items():
                try:
                    self.count[k] = max(1.0, float(v))
                except (TypeError, ValueError):
                    continue
            for k, v in state.get("ewma_short", {}).items():
                try:
                    self.ewma_short[k] = float(v)
                except (TypeError, ValueError):
                    continue
            for k, v in state.get("ewma_long", {}).items():
                try:
                    self.ewma_long[k] = float(v)
                except (TypeError, ValueError):
                    continue

    def get_weights(self, names: Iterable[str], fallback: Dict[str, float]) -> Dict[str, float]:
        return {n: self.weights.get(n, fallback.get(n, 0.0)) for n in names}

    def predict(self, features: Dict[str, float]) -> float:
        return sum(self.weights.get(name, self.base_weights.get(name, 0.0)) * float(val)
                   for name, val in features.items())

    def update(self, features: Dict[str, float], reward: float | None) -> None:
        if reward is None:
            return
        try:
            reward = float(reward)
        except (TypeError, ValueError):
            return
        pred = self.predict(features)
        err = max(-1.0, min(1.0, reward - pred))
        if not math.isfinite(err):
            return
        for name, raw_val in features.items():
            if not isinstance(raw_val, (int, float)):
                continue
            val = float(raw_val)
            if not math.isfinite(val):
                continue
            lr = self.lr / math.sqrt(self.count[name])
            grad = err * val
            w = self.weights.get(name, self.base_weights.get(name, 0.0))
            w = (1.0 - self.l2 * lr) * w + lr * grad
            self.weights[name] = w
            self.count[name] += abs(val) + 1e-3
            self._update_drift(name, val)

    def _update_drift(self, name: str, val: float) -> None:
        short_alpha = 0.3
        long_alpha = 0.05
        self.ewma_short[name] = (1 - short_alpha) * self.ewma_short[name] + short_alpha * val
        self.ewma_long[name] = (1 - long_alpha) * self.ewma_long[name] + long_alpha * val
        if abs(self.ewma_short[name] - self.ewma_long[name]) > 0.35:
            self.weights[name] = self.weights.get(name, self.base_weights.get(name, 0.0)) * 0.8
            self.count[name] = max(1.0, self.count[name] * 0.7)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weights": self.weights,
            "count": dict(self.count),
            "ewma_short": dict(self.ewma_short),
            "ewma_long": dict(self.ewma_long),
        }


class OnlineBanditTTL:
    """Discrete Thompson Sampling over TTL candidates."""

    def __init__(self, candidates: Iterable[float] | None = None, state: Dict[str, Any] | None = None):
        default_candidates = [30 * 60, 3 * 3600, 7 * 3600, 14 * 3600, 30 * 3600]
        self.candidates = [float(c) for c in (candidates or default_candidates)]
        self.state: Dict[str, Dict[str, Dict[str, float]]] = {}
        if state:
            for key, options in state.items():
                safe_opts: Dict[str, Dict[str, float]] = {}
                for ttl, stats in options.items():
                    try:
                        ttl_f = str(float(ttl))
                    except (TypeError, ValueError):
                        continue
                    safe_opts[ttl_f] = {
                        "s": max(1.0, float(stats.get("s", 1.0))),
                        "f": max(1.0, float(stats.get("f", 1.0))),
                    }
                if safe_opts:
                    self.state[key] = safe_opts

    def _ensure_key(self, key: str) -> Dict[str, Dict[str, float]]:
        if key not in self.state:
            self.state[key] = {
                str(c): {"s": 1.0, "f": 1.0} for c in self.candidates
            }
        else:
            # ensure new candidates exist
            for c in self.candidates:
                self.state[key].setdefault(str(c), {"s": 1.0, "f": 1.0})
        return self.state[key]

    def sample(self, key: str) -> float:
        options = self._ensure_key(key)
        best_ttl = None
        best_draw = -1.0
        for ttl, stats in options.items():
            s = max(1.0, float(stats.get("s", 1.0)))
            f = max(1.0, float(stats.get("f", 1.0)))
            draw = random.betavariate(s, f)
            if draw > best_draw:
                best_draw = draw
                best_ttl = float(ttl)
        return best_ttl or self.candidates[0]

    def update(self, key: str, ttl: float, success: bool) -> None:
        options = self._ensure_key(key)
        ttl_key = str(float(ttl))
        stats = options.setdefault(ttl_key, {"s": 1.0, "f": 1.0})
        if success:
            stats["s"] = stats.get("s", 1.0) + 1.0
        else:
            stats["f"] = stats.get("f", 1.0) + 1.0

    def to_dict(self) -> Dict[str, Any]:
        return self.state


class OnlineTextClassifier:
    """Lightweight perceptron-style classifier for urgency detection."""

    def __init__(self, state: Dict[str, Any] | None = None):
        self.weights: Dict[str, float] = defaultdict(float)
        self.bias = 0.0
        self.lr = 0.08
        self.decay = 0.995
        if state:
            for k, v in state.get("weights", {}).items():
                try:
                    self.weights[k] = float(v)
                except (TypeError, ValueError):
                    continue
            try:
                self.bias = float(state.get("bias", 0.0))
            except (TypeError, ValueError):
                self.bias = 0.0

    def _features(self, text: str) -> List[str]:
        norm = _normalize_text(text)
        tokens = re.findall(r"[a-z]{2,}", norm)
        feats = set(tokens)
        for i in range(len(tokens) - 1):
            feats.add(tokens[i] + "_" + tokens[i + 1])
        return list(feats)

    def predict(self, text: str) -> float:
        feats = self._features(text)
        score = self.bias + sum(self.weights.get(f, 0.0) for f in feats)
        score = max(-20.0, min(20.0, score))
        return 1.0 / (1.0 + math.exp(-score))

    def update(self, text: str, label: float | int | bool) -> None:
        if text is None:
            return
        feats = self._features(text)
        if not feats:
            return
        y = 1.0 if label else 0.0
        pred = self.predict(text)
        err = y - pred
        for f in feats:
            self.weights[f] = self.weights.get(f, 0.0) * self.decay + self.lr * err
        self.bias = self.bias * self.decay + self.lr * err

    def to_dict(self) -> Dict[str, Any]:
        items = sorted(self.weights.items(), key=lambda kv: -abs(kv[1]))
        if len(items) > 1500:
            items = items[:1500]
        return {"weights": {k: v for k, v in items}, "bias": self.bias}


class GoalPrioritizer:
    """
    Prioritiseur multi-signaux.
    - Lit persona/values (+ alias & poids si dispo), ontology/beliefs, skills appris,
      homeostasis.drives, questions/uncertainty, deadlines, dépendances parent↔enfant,
      progrès/ancienneté, directives utilisateur récentes, coût/risque estimé des actions,
      et avis de Policy.
    - Produit: plan["priority"] ∈ [0..1] + plan["tags"] (ex: ["urgent"] ou ["background"]).
    - N'enlève rien: si un composant manque, il s'efface (fallback = 0).
    - Configurable par JSON: data/prioritizer_config.json (facultatif).
    """

    def __init__(self, arch):
        self.arch = arch
        self._logger = getattr(arch, "logger", logging.getLogger(__name__))
        self.cfg = self._load_cfg()
        # petite mémoire interne pour anti-famine et “dernier tick”
        self._last_seen: Dict[str, float] = {}
        self._lane_thresholds = self.cfg.get(
            "lane_thresholds",
            {
                "urgent_pr": 0.92,
                "background_pr": 0.60,
            },
        )
        self.state_path = getattr(
            self.arch, "prioritizer_state_path", "data/prioritizer_state.json"
        )
        persisted = self._load_state()
        base_weights = self.cfg.setdefault("weights", {})
        self.weight_model = OnlineWeightLearner(base_weights, persisted.get("weight_model"))
        self.ttl_bandit = OnlineBanditTTL(state=persisted.get("ttl_bandit"))
        self.text_clf = OnlineTextClassifier(persisted.get("text_classifier"))
        self._last_feedback_ts = float(persisted.get("last_feedback_ts", 0.0))
        seen_ids = [str(s) for s in persisted.get("seen_feedback_ids", [])][:512]
        self._seen_feedback_ids: deque[str] = deque(seen_ids, maxlen=512)
        self._feature_history: Dict[str, Dict[str, float]] = {}
        self._feature_fifo: deque[str] = deque()
        self._ttl_assignments: Dict[str, Tuple[str, float, float]] = {}
        self._dirty_state = False
        self._last_state_flush = 0.0
        # patterns renforcées pour capturer différentes tournures FR
        self._urgency_patterns = [
            (re.compile(r"\b(urgent|urgence|prioritaire|critique)\b"), "user_urgent"),
            (
                re.compile(
                    r"\b(tout\s*(?:de\s*)?suite|sans\s+delai|imm?ediatement|maintenant)\b"
                ),
                "user_urgent",
            ),
            (
                re.compile(r"\b(dans|avant)\s+(?:les?|la)\s+\d+\s*(minutes?|heures?|jours?)\b"),
                "user_time_ref",
            ),
            (re.compile(r"\baujourd['’]?hui\b"), "user_time_ref"),
            (re.compile(r"\bce\s+(soir|matin|week\s*end)\b"), "user_time_ref"),
        ]
        self._directive_patterns = [
            re.compile(r"\bfais\s+preuve\s+d['e]\w+"),
            re.compile(r"\bs(?:ois|oyons)\s+\w+"),
            re.compile(r"\badopte\s+(?:un|une|le|la|l')\s+\w+"),
        ]
        self._urgency_classifier_threshold = 0.6

    # ---------- CONFIG ----------
    def _load_cfg(self) -> Dict[str, Any]:
        path = getattr(self.arch, "prioritizer_cfg_path", "data/prioritizer_config.json")
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f) or {}
        except Exception:
            pass
        # défaut raisonnable
        return {
            "weights": {
                "user_urgency": 0.30,
                "deadline": 0.28,
                "uncertainty": 0.18,
                "drive_alignment": 0.14,
                "identity_alignment": 0.18,
                "parent_waiting": 0.10,
                "staleness": 0.10,
                "base_priority": 0.10,
                "policy_feasibility": 0.10,
                "competence_fit": 0.10,
                "novelty_drive": 0.08,
                "action_cost": -0.08,  # coût pénalise
                "risk_penalty": -0.10,  # risque pénalise
            },
            "lane_thresholds": {
                "urgent_pr": 0.92,
                "background_pr": 0.60,
            },
            "background_period_sec": 5 * 60,  # utile si tu filtres ailleurs la fréquence BG
        }

    def _load_state(self) -> Dict[str, Any]:
        path = self.state_path
        if not path:
            return {}
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f) or {}
        except Exception:
            return {}
        return {}

    def _maybe_flush_state(self, force: bool = False) -> None:
        if not self._dirty_state:
            return
        if not force and (_now() - self._last_state_flush) < 30.0:
            return
        payload = {
            "weight_model": self.weight_model.to_dict(),
            "ttl_bandit": self.ttl_bandit.to_dict(),
            "text_classifier": self.text_clf.to_dict(),
            "last_feedback_ts": self._last_feedback_ts,
            "seen_feedback_ids": list(self._seen_feedback_ids),
        }
        path = self.state_path
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = f"{path}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
            self._dirty_state = False
            self._last_state_flush = _now()
        except Exception:
            pass

    def _cache_features(self, goal_id: str, features: Dict[str, float]) -> None:
        self._feature_history[goal_id] = dict(features)
        try:
            self._feature_fifo.remove(goal_id)
        except ValueError:
            pass
        self._feature_fifo.append(goal_id)
        while len(self._feature_fifo) > 256:
            old = self._feature_fifo.popleft()
            if old not in self._ttl_assignments:
                self._feature_history.pop(old, None)

    def _ttl_key(self, plan: Dict[str, Any]) -> str:
        key = plan.get("kind") or plan.get("lane") or plan.get("category")
        if isinstance(key, str) and key.strip():
            return key.lower()
        tags = plan.get("tags") or []
        if tags:
            return str(tags[0]).lower()
        return "default"

    def _detect_urgency(self, text: str, meta: Dict[str, Any] | None = None) -> Tuple[float, str]:
        if not text:
            return (0.0, "")
        norm = _normalize_text(text)
        for pattern, label in self._urgency_patterns:
            if pattern.search(norm):
                return (1.0, label)
        for pattern in self._directive_patterns:
            if pattern.search(norm):
                return (0.65, "user_directive")
        prob = 0.0
        if self.text_clf:
            prob = self.text_clf.predict(text)
        if meta:
            labels = set(map(str, meta.get("labels", []) or []))
            labels.update(str(t) for t in meta.get("tags", []) or [])
            urgent_flags = {"urgent", "rush", "haut_niveau"}
            not_urgent = {"not_urgent", "background"}
            if labels & urgent_flags:
                return (1.0, "label_urgent")
            if labels & not_urgent:
                return (0.0, "")
            explicit = meta.get("urgency_label")
            if isinstance(explicit, str):
                explicit = explicit.lower()
                if explicit in ("urgent", "rush", "eleve"):
                    return (1.0, "explicit_label")
                if explicit in ("faible", "basse", "background"):
                    return (0.0, "")
        if prob >= self._urgency_classifier_threshold:
            return (min(1.0, prob), f"clf:{prob:.2f}")
        return (0.0, "")

    def _update_from_reward(self, goal_id: str, reward: Any) -> None:
        features = self._feature_history.get(goal_id)
        if not features:
            return
        try:
            reward_f = float(reward)
        except (TypeError, ValueError):
            return
        self.weight_model.update(features, reward_f)
        self._dirty_state = True

    def _register_completion(self, goal_id: str, plan: Dict[str, Any]) -> None:
        assign = self._ttl_assignments.pop(goal_id, None)
        if not assign:
            return
        key, ttl, start_ts = assign
        elapsed = plan.get("elapsed_sec") or plan.get("duration_sec")
        if elapsed is None and start_ts:
            done_ts = plan.get("completed_ts") or plan.get("last_tick_done")
            if done_ts:
                elapsed = max(0.0, float(done_ts) - float(start_ts))
        if elapsed is None:
            return
        try:
            elapsed = float(elapsed)
        except (TypeError, ValueError):
            return
        success = elapsed <= float(ttl)
        self.ttl_bandit.update(key, float(ttl), success)
        self._dirty_state = True

    def _ingest_feedback(self) -> None:
        memory = getattr(self.arch, "memory", None)
        if not memory or not hasattr(memory, "get_recent_memories"):
            return
        try:
            recent = memory.get_recent_memories(limit=200)
        except Exception:
            return
        max_ts = self._last_feedback_ts
        dirty = False
        for m in recent:
            if not isinstance(m, dict):
                continue
            mid = m.get("id")
            if mid is not None:
                mid = str(mid)
                if mid in self._seen_feedback_ids:
                    continue
            ts = float(m.get("ts") or 0.0)
            if ts and ts <= self._last_feedback_ts:
                continue
            kind = m.get("kind")
            if kind == "priority_feedback":
                goal = m.get("goal") or m.get("goal_id")
                reward = m.get("reward")
                if goal is not None and reward is not None:
                    self._update_from_reward(str(goal), reward)
            elif kind in {"urgency_feedback", "interaction_feedback"}:
                text = m.get("text") or m.get("sample")
                label = (
                    m.get("label")
                    or m.get("urgency_label")
                    or (m.get("tags") and ("urgent" in m.get("tags")))
                )
                if isinstance(label, str):
                    label = label.lower() in ("urgent", "rush", "eleve")
                elif isinstance(label, (list, tuple, set)):
                    label = any(str(x).lower() in ("urgent", "rush", "eleve") for x in label)
                if text:
                    self.text_clf.update(text, bool(label))
                    self._dirty_state = True
            elif kind == "goal_completed":
                goal = m.get("goal") or m.get("goal_id")
                if goal:
                    proxy_plan = {
                        "elapsed_sec": m.get("elapsed_sec") or m.get("duration_sec"),
                        "completed_ts": m.get("ts"),
                    }
                    self._register_completion(str(goal), proxy_plan)
            if mid is not None:
                self._seen_feedback_ids.append(mid)
                dirty = True
            if ts:
                max_ts = max(max_ts, ts)
        if max_ts > self._last_feedback_ts:
            self._last_feedback_ts = max_ts
            dirty = True
        if dirty:
            self._dirty_state = True

    # ---------- FEATURES (chacune retourne (score, reason)) ----------
    def feat_user_urgency(self) -> Tuple[Number, str]:
        # détecte "maintenant/urgent", directives NL récentes, etc.
        try:
            rec = self.arch.memory.get_recent_memories(limit=30)
        except Exception:
            rec = []
        for m in reversed(rec):
            if (m.get("kind") or "") != "interaction":
                continue
            if (m.get("role") or "") != "user":
                continue
            text = m.get("text") or ""
            score, reason = self._detect_urgency(text, m)
            if score > 0:
                return (min(1.0, score), reason)
            break
        return (0.0, "")

    def feat_deadline(self, plan: Dict[str, Any]) -> Tuple[Number, str]:
        dl = plan.get("deadline_ts")
        if not dl:
            return (0.0, "")
        rem = dl - _now()
        if rem <= 0:
            return (1.0, "deadline_passed")
        # mapping: <=1h -> ~1.0 ; 48h -> ~0
        v = max(0.0, min(1.0, 1.0 - rem / (48 * 3600)))
        return (v, "deadline")

    def feat_uncertainty(self, goal_id: str) -> Tuple[Number, str]:
        # incertitude haute -> priorité à lever les blocages / clarifier
        u = 0.0
        reason = ""
        try:
            pol = getattr(self.arch, "policy", None)
            if pol and hasattr(pol, "confidence"):
                c = float(pol.confidence())
                u = max(u, 1.0 - c)
                if u > 0.5:
                    reason = "low_policy_conf"
        except Exception:
            pass
        try:
            qm = getattr(self.arch, "question_manager", None)
            if qm:
                for q in getattr(qm, "pending_questions", []):
                    # si question liée au goal_id, boost supplémentaire
                    if goal_id.lower() in (q.get("topic", "") or q.get("text", "")).lower():
                        u = max(u, 0.7)
                        reason = reason or "pending_question"
                        break
        except Exception:
            pass
        return (u, reason)

    def feat_drive_alignment(self, goal_id: str) -> Tuple[Number, str]:
        # aligne sur drives (homeostasis)
        drives = _safe(self.arch, "homeostasis", "state", "drives", default={}) or {}
        g = goal_id.lower()
        w = 0.0
        why: List[str] = []
        # mapping heuristique + extensible via ontology (voir feat_identity_alignment)
        if any(k in g for k in ("understand", "learn", "research", "investigate")):
            v = float(drives.get("curiosity", 0.5))
            w += 0.5 * v
            why.append(f"curiosity={v:.2f}")
        if any(k in g for k in ("human", "emotion", "empathy", "social")):
            v = float(drives.get("social_bonding", 0.5))
            w += 0.5 * v
            why.append(f"social_bonding={v:.2f}")
        if "self" in g or "evolve" in g:
            v = float(drives.get("self_actualization", drives.get("autonomy", 0.5)))
            w += 0.4 * v
            why.append(f"self_actualization={v:.2f}")
        return (min(1.0, w), ",".join(why))

    def _goal_concepts(self, goal_id: str, plan: Dict[str, Any]) -> List[str]:
        # 1) meta stockée
        cs = (plan.get("concepts") or []) + (plan.get("tags") or [])
        # 2) parsage id/title
        cs += [
            w
            for w in _tok(goal_id)
            if w not in ("understand", "learn", "goal", "task", "subgoal")
        ]
        title = plan.get("title") or ""
        cs += [w for w in _tok(title)]
        for text in (goal_id, title):
            norm = _normalize_text(text)
            for match in re.findall(r"est\s+(?:un|une|le|la|l')\s+([a-z]{3,})", norm):
                cs.append(match)
        return list(dict.fromkeys(cs))[:10]

    def _value_aliases(self) -> Dict[str, List[str]]:
        # persona.values + alias éventuels
        persona = _safe(self.arch, "self_model", "state", "persona", default={}) or {}
        alias = persona.get("value_aliases") or {}
        vals = persona.get("values") or []
        for v in vals:
            alias.setdefault(v.lower(), [v.lower()])
        return alias

    def _ontology_neighbors(self, term: str, max_depth: int = 2) -> List[str]:
        # essaie via ontology (si présente), sinon retour vide
        out: List[str] = []
        onto = getattr(self.arch, "ontology", None)
        if not onto:
            return out
        try:
            nid = f"concept:{term}"
            if not onto.has_entity(nid):
                return out
            # neighbors 1..max_depth (API hypothétique -> adapte à la tienne)
            frontier: List[Tuple[str, int]] = [(nid, 0)]
            seen = {nid}
            while frontier:
                cur, d = frontier.pop(0)
                if d >= max_depth:
                    continue
                neigh: List[str] = []
                try:
                    neigh = list(onto.neighbors(cur))
                except Exception:
                    pass
                for nb in neigh:
                    if nb in seen:
                        continue
                    seen.add(nb)
                    out.append(nb)
                    frontier.append((nb, d + 1))
        except Exception:
            pass
        # dé-normalise "concept:xxx" -> "xxx"
        clean: List[str] = []
        for n in out:
            if isinstance(n, str) and n.startswith("concept:"):
                clean.append(n.split(":", 1)[1])
        return clean

    def feat_identity_alignment(self, goal_id: str, plan: Dict[str, Any]) -> Tuple[Number, str]:
        """
        Score d'identité riche:
        - correspondance concepts du goal ↔ persona.values (avec alias)
        - expansion via ontologie jusqu'à 2 sauts (synonymes, proches)
        - support par beliefs (évidences récentes)
        - renforcement si la valeur est récente (feedback utilisateur, virtue_learned)
        """

        values_alias = self._value_aliases()
        if not values_alias:
            return (0.0, "")

        concepts = self._goal_concepts(goal_id, plan)
        # expand via ontology
        expanded = set(concepts)
        for c in concepts:
            for nb in self._ontology_neighbors(c, max_depth=2):
                expanded.add(nb)

        # matching alias
        score = 0.0
        matches: List[str] = []
        for val, aliases in values_alias.items():
            # Intersection entre alias de la valeur et concepts/voisins
            if set(aliases) & set(expanded):
                score += 0.35
                matches.append(val)

        # beliefs support (si un predicate 'promotes' lie la valeur au concept)
        beliefs = getattr(self.arch, "beliefs", None)
        if beliefs and hasattr(beliefs, "support"):
            try:
                for c in concepts[:5]:
                    s = beliefs.support(
                        subject=f"concept:{c}", predicate="promotes_identity", default=0.0
                    )
                    if s > 0:
                        score += min(0.25, s)
                        matches.append(f"belief:{c}")
            except Exception:
                pass

        # événements récents (virtue_learned, feedback) -> boost léger
        try:
            rec = self.arch.memory.get_recent_memories(limit=50)
            for m in rec[-50:]:
                if (m.get("kind") or "") == "virtue_learned":
                    v = (m.get("value") or "").lower()
                    if v in values_alias.keys():
                        score += 0.15
                        matches.append(f"virtue:{v}")
                if (m.get("kind") or "") == "feedback" and "style" in (m.get("tags") or []):
                    score += 0.05
        except Exception:
            pass

        return (min(1.0, score), ",".join(matches[:4]))

    def feat_parent_waiting(self, goal_id: str) -> Tuple[Number, str]:
        parent = _safe(self.arch, "planner", "state", "parents", default={}).get(goal_id)
        if not parent:
            return (0.0, "")
        parent_plan = _safe(self.arch, "planner", "state", "plans", default={}).get(parent, {})
        if parent_plan and parent_plan.get("status") != "done":
            return (0.4, f"parent:{parent}")
        return (0.0, "")

    def feat_staleness(self, goal_id: str, plan: Dict[str, Any]) -> Tuple[Number, str]:
        last = plan.get("last_tick_done") or plan.get("created_ts") or (_now() - 1800)
        age = _now() - last
        key = self._ttl_key(plan)
        assign = self._ttl_assignments.get(goal_id)
        if assign and assign[0] == key:
            ttl = float(assign[1])
        else:
            ttl = max(60.0, self.ttl_bandit.sample(key))
            self._ttl_assignments[goal_id] = (key, ttl, plan.get("created_ts") or _now())
        v = max(0.0, min(0.5, age / ttl))
        return (v, f"stale:{int(age)}s<{int(ttl)}s")

    def feat_base_priority(self, plan: Dict[str, Any]) -> Tuple[Number, str]:
        bp = float(plan.get("priority", 0.5))
        return (bp, f"base:{bp:.2f}")

    def feat_policy_feasibility(self, goal_id: str, plan: Dict[str, Any]) -> Tuple[Number, str]:
        # si la policy va probablement refuser la majorité des steps => baisse
        pol = getattr(self.arch, "policy", None)
        if not pol:
            return (0.0, "")
        # heuristique: si plan a beaucoup d'ops "restricted" (selon policy)
        ops: List[str] = []
        for st in plan.get("steps", []):
            if isinstance(st, dict) and st.get("kind") == "act":
                ops.append(st.get("op"))
        if not ops:
            return (0.0, "")
        bad = 0
        for op in ops[:6]:
            try:
                r = pol.simulate(op, plan) if hasattr(pol, "simulate") else None
                if isinstance(r, dict) and r.get("decision") == "deny":
                    bad += 1
            except Exception:
                pass
        if bad >= max(2, len(ops) // 2):
            return (-0.5, "policy_block_risk")
        return (0.0, "")

    def feat_competence_fit(self, goal_id: str, plan: Dict[str, Any]) -> Tuple[Number, str]:
        # si on possède déjà les skills utiles, on livre plus vite → boost
        # si totalement inconnu mais curiosité haute → boost via novelty_drive plutôt
        skills: Dict[str, Any] = {}
        try:
            path = getattr(self.arch, "skills_path", "data/skills.json")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    skills = json.load(f) or {}
        except Exception:
            pass
        concepts = self._goal_concepts(goal_id, plan)
        have = sum(1 for c in concepts if c in skills and skills[c].get("acquired"))
        if have == 0:
            return (0.0, "")
        v = min(1.0, 0.15 * have)
        return (v, f"skills:{have}")

    def feat_novelty_drive(self, goal_id: str, plan: Dict[str, Any]) -> Tuple[Number, str]:
        # curiosité + absence de skill => exploration
        drives = _safe(self.arch, "homeostasis", "state", "drives", default={}) or {}
        curiosity = float(drives.get("curiosity", 0.5))
        skills: Dict[str, Any] = {}
        try:
            path = getattr(self.arch, "skills_path", "data/skills.json")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    skills = json.load(f) or {}
        except Exception:
            pass
        concepts = self._goal_concepts(goal_id, plan)
        unknown = sum(1 for c in concepts if c not in skills)
        if unknown == 0:
            return (0.0, "")
        v = min(1.0, 0.1 * unknown * (0.6 + 0.7 * curiosity))
        return (v, f"novel:{unknown}")

    def feat_action_cost(self, plan: Dict[str, Any]) -> Tuple[Number, str]:
        # si on a des stats sur les ops (avg_ms), on pénalise légèrement les plans “chers”
        stats = getattr(self.arch, "actions", None)
        if not stats or not hasattr(stats, "stats"):
            return (0.0, "")
        total_ms = 0.0
        n = 0
        for st in plan.get("steps", []):
            if isinstance(st, dict) and st.get("kind") == "act":
                op = st.get("op")
                meta = stats.stats.get(op) if isinstance(stats.stats, dict) else None
                if meta and "avg_ms" in meta:
                    total_ms += float(meta["avg_ms"])
                    n += 1
        if n == 0:
            return (0.0, "")
        avg = total_ms / n
        # 0 si <30ms ; -0.2 vers -0.4 si >300ms
        pen = -min(0.4, max(0.0, (avg - 30.0) / 300.0))
        return (pen, f"cost_ms~{avg:.0f}")

    def feat_risk_penalty(self, plan: Dict[str, Any]) -> Tuple[Number, str]:
        # si le plan a un tag 'risky' ou des steps marqués 'needs_human' → pénalité (laisse Policy trancher)
        tags = set(plan.get("tags", []))
        if "risky" in tags:
            return (-0.3, "risky_tag")
        if any(
            (isinstance(st, dict) and st.get("policy") == "needs_human")
            for st in plan.get("steps", [])
        ):
            return (-0.15, "needs_human_steps")
        return (0.0, "")

    # ---------- SCORING GLOBAL ----------
    def _compute_heuristic_score(self, goal_id: str, plan: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        base_weights = self.cfg.setdefault("weights", {})
        feats: Dict[str, Tuple[float, str]] = {}

        def add(name: str, val_reason: Tuple[Number, str]):
            v, r = val_reason
            feats[name] = (float(v), r)

        # calcule toutes les features (avec fallbacks)
        add("user_urgency", self.feat_user_urgency())
        add("deadline", self.feat_deadline(plan))
        add("uncertainty", self.feat_uncertainty(goal_id))
        add("drive_alignment", self.feat_drive_alignment(goal_id))
        add("identity_alignment", self.feat_identity_alignment(goal_id, plan))
        add("parent_waiting", self.feat_parent_waiting(goal_id))
        add("staleness", self.feat_staleness(goal_id, plan))
        add("base_priority", self.feat_base_priority(plan))
        add("policy_feasibility", self.feat_policy_feasibility(goal_id, plan))
        add("competence_fit", self.feat_competence_fit(goal_id, plan))
        add("novelty_drive", self.feat_novelty_drive(goal_id, plan))
        add("action_cost", self.feat_action_cost(plan))
        add("risk_penalty", self.feat_risk_penalty(plan))

        feature_values = {name: val for name, (val, _) in feats.items()}
        weights = self.weight_model.get_weights(feature_values.keys(), base_weights)

        pr = 0.0
        reasons: List[str] = []
        for name, (v, r) in feats.items():
            w = float(weights.get(name, 0.0))
            pr += w * v
            if r:
                reasons.append(f"{name}:{r}({v:.2f}×{w:.2f})")

        pr = max(0.0, min(1.0, pr))

        # tags de voie (ne supprime pas les autres tags)
        tags = set(plan.get("tags", []))
        tags.discard("urgent")
        tags.discard("background")
        if pr >= self._lane_thresholds["urgent_pr"]:
            tags.add("urgent")
        elif pr < self._lane_thresholds["background_pr"]:
            tags.add("background")

        # anti-famine: si jamais vu depuis > N sec, petit boost (post)
        last = self._last_seen.get(goal_id, plan.get("created_ts", _now()))
        idle = _now() - last
        if idle > 20 * 60 and pr < 0.85:
            pr = min(1.0, pr + 0.05)
            reasons.append("anti_famine:+0.05")

        self._cache_features(goal_id, feature_values)

        breakdown = {
            name: {
                "value": float(feature_values.get(name, 0.0)),
                "weight": float(weights.get(name, 0.0)),
                "reason": str(feats[name][1] or ""),
            }
            for name in feature_values
        }

        heuristic_result = {"priority": pr, "tags": list(tags), "explain": reasons[:6]}
        context = {
            "features": breakdown,
            "thresholds": {
                "urgent_pr": float(self._lane_thresholds.get("urgent_pr", 0.92)),
                "background_pr": float(self._lane_thresholds.get("background_pr", 0.60)),
            },
            "base_weights": {k: float(v) for k, v in base_weights.items()},
        }
        return heuristic_result, context

    def _summarize_plan(self, plan: Mapping[str, Any]) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        text_fields = [
            "title",
            "name",
            "summary",
            "description",
            "goal",
            "user_text",
        ]
        for field in text_fields:
            value = plan.get(field)
            if isinstance(value, str) and value.strip():
                summary[field] = value.strip()[:600]

        for field in ("kind", "lane", "category", "status", "parent", "owner"):
            value = plan.get(field)
            if isinstance(value, str) and value.strip():
                summary[field] = value.strip()

        for field in ("deadline", "deadline_ts", "due", "due_ts"):
            if field in plan and plan.get(field) is not None:
                summary[field] = plan.get(field)

        priority_hint = plan.get("priority")
        if isinstance(priority_hint, (int, float)):
            summary["reported_priority"] = float(priority_hint)

        tags = plan.get("tags")
        if isinstance(tags, (list, tuple, set)):
            summary["tags"] = [str(t) for t in list(tags)[:8]]

        metrics = plan.get("metrics")
        if isinstance(metrics, Mapping):
            cleaned_metrics: Dict[str, Any] = {}
            for key, value in metrics.items():
                if isinstance(value, (int, float, str)):
                    cleaned_metrics[str(key)] = value
            if cleaned_metrics:
                summary["metrics"] = cleaned_metrics

        dependencies = plan.get("dependencies") or plan.get("depends_on")
        if isinstance(dependencies, (list, tuple, set)):
            summary["dependencies"] = [str(dep) for dep in list(dependencies)[:6]]

        steps = plan.get("steps")
        if isinstance(steps, list) and steps:
            summary["steps"] = self._summarize_steps(steps)

        return summary

    def _summarize_steps(self, steps: List[Any]) -> List[Any]:
        summarized: List[Any] = []
        for step in steps[:5]:
            if isinstance(step, Mapping):
                entry = {}
                for field in ("id", "kind", "op", "summary", "description"):
                    value = step.get(field)
                    if isinstance(value, str) and value.strip():
                        entry[field] = value.strip()[:240]
                status = step.get("status")
                if isinstance(status, str) and status.strip():
                    entry["status"] = status.strip()
                if entry:
                    summarized.append(entry)
            elif isinstance(step, str):
                summarized.append(step.strip()[:240])
        return summarized

    def _build_llm_payload(
        self,
        goal_id: str,
        plan: Mapping[str, Any],
        heuristic_result: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "goal_id": goal_id,
            "plan_summary": self._summarize_plan(plan),
            "heuristics": {
                "priority": float(heuristic_result.get("priority", 0.0)),
                "tags": list(heuristic_result.get("tags", [])),
                "explain": list(heuristic_result.get("explain", [])),
                "features": context.get("features", {}),
                "thresholds": context.get("thresholds", {}),
            },
        }
        base_weights = context.get("base_weights")
        if isinstance(base_weights, Mapping):
            payload["heuristics"]["base_weights"] = dict(base_weights)
        return payload

    def _merge_llm_result(
        self, heuristic_result: Dict[str, Any], llm_response: Mapping[str, Any]
    ) -> Dict[str, Any]:
        result = dict(heuristic_result)

        priority = llm_response.get("priority")
        try:
            if priority is not None:
                priority_f = float(priority)
                if math.isfinite(priority_f):
                    result["priority"] = max(0.0, min(1.0, priority_f))
        except (TypeError, ValueError):
            pass

        tags = llm_response.get("tags")
        if isinstance(tags, (list, tuple, set)):
            cleaned_tags = [str(tag).strip() for tag in tags if str(tag).strip()]
            if cleaned_tags:
                result["tags"] = list(dict.fromkeys(cleaned_tags))

        explain = llm_response.get("explain")
        if isinstance(explain, (list, tuple)):
            cleaned_explain = [str(item) for item in explain if str(item).strip()]
            if cleaned_explain:
                result["explain"] = cleaned_explain[:6]

        confidence = llm_response.get("confidence")
        try:
            if confidence is not None:
                conf_f = float(confidence)
                if math.isfinite(conf_f):
                    result["confidence"] = max(0.0, min(1.0, conf_f))
        except (TypeError, ValueError):
            pass

        notes = llm_response.get("notes")
        if isinstance(notes, str) and notes.strip():
            result["notes"] = notes.strip()

        return result

    def score_goal(self, goal_id: str, plan: Dict[str, Any]) -> Dict[str, Any]:
        heuristic_result, context = self._compute_heuristic_score(goal_id, plan)

        llm_payload = self._build_llm_payload(goal_id, plan, heuristic_result, context)
        llm_response = try_call_llm_dict(
            "cognition_goal_prioritizer",
            input_payload=llm_payload,
            logger=self._logger,
        )

        if llm_response:
            try:
                return self._merge_llm_result(heuristic_result, llm_response)
            except Exception:  # pragma: no cover - defensive safety net
                if self._logger:
                    try:
                        self._logger.debug(
                            "Failed to merge LLM result for goal %s", goal_id, exc_info=True
                        )
                    except Exception:
                        pass
        return heuristic_result

    def reprioritize_all(self):
        planner = getattr(self.arch, "planner", None)
        if not planner:
            return

        self._ingest_feedback()

        plans: Dict[str, Dict[str, Any]] = {}
        state = getattr(planner, "state", None)
        if isinstance(state, dict):
            plans = state.get("plans") or {}
        elif isinstance(getattr(planner, "plans", None), dict):
            plans = getattr(planner, "plans") or {}

        if not isinstance(plans, dict):
            plans = {}
        if not plans:
            return
        for gid, plan in plans.items():
            if plan.get("status") == "done":
                self._register_completion(gid, plan)
                self._feature_history.pop(gid, None)
                try:
                    self._feature_fifo.remove(gid)
                except ValueError:
                    pass
                continue
            reward = plan.get("priority_reward")
            if reward is None:
                metrics_candidate = plan.get("metrics")
                metrics = metrics_candidate if isinstance(metrics_candidate, dict) else None
                if metrics:
                    reward = metrics.get("priority_reward") or metrics.get("reward")
            if reward is not None:
                self._update_from_reward(gid, reward)
            s = self.score_goal(gid, plan)
            plan["priority"] = s["priority"]
            plan["tags"] = s["tags"]
            self._last_seen[gid] = _now()
            # trace d'explication légère (optionnel)
            try:
                self.arch.memory.add_memory(
                    {
                        "kind": "priority_trace",
                        "goal": gid,
                        "priority": round(s["priority"], 3),
                        "tags": s["tags"],
                        "why": s["explain"],
                        "ts": _now(),
                    }
                )
            except Exception:
                pass
        self._maybe_flush_state()
