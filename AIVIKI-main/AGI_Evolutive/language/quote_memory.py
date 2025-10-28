from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any, Tuple
import logging
import time, re, math, os, unicodedata, hashlib

import numpy as np

from . import DATA_DIR, _json_load, _json_save
from AGI_Evolutive.utils.llm_service import try_call_llm_dict


LOGGER = logging.getLogger(__name__)
LLM_SPEC_KEY = "language_quote_memory"


@dataclass
class Quote:
    text: str
    source: str           # "inbox:path#L42" | "chat:<msg_id>" | "assistant:<ts>"
    ts: float
    liked: bool = False
    uses: int = 0
    last_used: float = 0.0
    lang: str = "fr"
    tags: List[str] = field(default_factory=list)


class OnlineLinearBandit:
    def __init__(
        self,
        dim: int,
        prior_mean: Optional[List[float]] = None,
        prior_var: float = 1.0,
        noise: float = 0.25,
        decay: float = 0.995,
    ):
        self.dim = dim
        self.noise = max(1e-3, noise)
        self.decay = min(max(decay, 0.9), 0.999)
        self.prior_mean = np.array(prior_mean or [0.0] * dim, dtype=float)
        self.prior_precision = (1.0 / max(prior_var, 1e-3)) * np.eye(dim)
        self.A = self.prior_precision.copy()
        self.b = self.prior_precision @ self.prior_mean

    def _regularise(self):
        self.A = self.decay * self.A + (1.0 - self.decay) * self.prior_precision
        self.b = self.decay * self.b + (1.0 - self.decay) * (self.prior_precision @ self.prior_mean)

    def sample_weights(self) -> np.ndarray:
        precision = self.A + 1e-6 * np.eye(self.dim)
        cov = np.linalg.inv(precision)
        mean = cov @ self.b
        try:
            sample = np.random.multivariate_normal(mean, (self.noise ** 2) * cov)
        except np.linalg.LinAlgError:
            sample = mean
        return sample

    def mean_weights(self) -> np.ndarray:
        precision = self.A + 1e-6 * np.eye(self.dim)
        return np.linalg.solve(precision, self.b)

    def update(self, features: np.ndarray, reward: float):
        if features is None:
            return
        x = np.asarray(features, dtype=float)
        self._regularise()
        self.A += np.outer(x, x)
        self.b += x * reward

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dim": self.dim,
            "prior_mean": self.prior_mean.tolist(),
            "prior_var": float(1.0 / self.prior_precision[0, 0]) if self.dim else 1.0,
            "noise": self.noise,
            "decay": self.decay,
            "A": self.A.tolist(),
            "b": self.b.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        dim = data.get("dim", 0)
        inst = cls(
            dim=dim,
            prior_mean=data.get("prior_mean"),
            prior_var=data.get("prior_var", 1.0),
            noise=data.get("noise", 0.25),
            decay=data.get("decay", 0.995),
        )
        if data.get("A") and data.get("b"):
            inst.A = np.array(data["A"], dtype=float)
            inst.b = np.array(data["b"], dtype=float)
        return inst


class OnlineTextModel:
    def __init__(self, dim: int = 2048, lr: float = 0.15, decay: float = 0.997):
        self.dim = dim
        self.lr = lr
        self.decay = decay
        self.weights: List[float] = [0.0] * dim
        self.bias: float = 0.0

    def _features(self, context: str, quote: str) -> Dict[int, int]:
        text = f"{context}\n>>>\n{quote}".lower()
        counts: Dict[int, int] = {}
        for n in (3, 4):
            for i in range(len(text) - n + 1):
                chunk = text[i : i + n].encode("utf-8")
                digest = hashlib.blake2b(chunk, digest_size=8).digest()
                idx = int.from_bytes(digest, "little") % self.dim
                counts[idx] = counts.get(idx, 0) + 1
        return counts

    def score(self, context: str, quote: str) -> Tuple[float, Dict[int, int]]:
        feats = self._features(context, quote)
        if not feats:
            return 0.0, feats
        norm = sum(feats.values())
        weighted = sum(self.weights[idx] * count for idx, count in feats.items()) / max(1, norm)
        score = self.bias + weighted
        return score, feats

    def update(self, feats: Dict[int, int], reward: float):
        if not feats or reward == 0:
            return
        target = 1.0 if reward > 0 else 0.0
        norm = sum(feats.values())
        weighted = sum(self.weights[idx] * count for idx, count in feats.items()) / max(1, norm)
        score = self.bias + weighted
        prob = 1.0 / (1.0 + math.exp(-score))
        grad = prob - target
        self.bias = self.bias * self.decay - self.lr * grad
        step = self.lr * grad / max(1, norm)
        for idx, count in feats.items():
            self.weights[idx] = self.weights[idx] * self.decay - step * count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dim": self.dim,
            "lr": self.lr,
            "decay": self.decay,
            "bias": self.bias,
            "weights": self.weights,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        inst = cls(dim=data.get("dim", 2048), lr=data.get("lr", 0.15), decay=data.get("decay", 0.997))
        weights = data.get("weights")
        if isinstance(weights, list) and len(weights) == inst.dim:
            inst.weights = [float(w) for w in weights]
        inst.bias = float(data.get("bias", 0.0))
        return inst


class QuoteMemory:
    DEFAULT_WEIGHTS = [0.0, 0.45, 0.35, 0.20, -0.60, -0.06, 0.10]

    def __init__(self, storage: str = None, max_items: int = 1200):
        self.storage = storage or os.path.join(DATA_DIR, "quotes.json")
        self.state_storage = self.storage + ".state.json"
        self.max = max_items
        self.items: List[Quote] = []
        self._last_served: Optional[int] = None
        self._last_features: Optional[np.ndarray] = None
        self._last_text_feats: Optional[Dict[int, int]] = None
        self._bandit = OnlineLinearBandit(dim=len(self.DEFAULT_WEIGHTS), prior_mean=self.DEFAULT_WEIGHTS)
        self._text_model = OnlineTextModel()
        self.load()

    # ---------- Persistence ----------
    def load(self):
        data = _json_load(self.storage, [])
        self.items = [Quote(**q) for q in data if isinstance(q, dict) and "text" in q]
        self.items = self.items[-self.max:]
        self._load_state()

    def save(self):
        data = [asdict(q) for q in self.items[-self.max:]]
        _json_save(self.storage, data)
        self._save_state()

    def _load_state(self):
        state = _json_load(self.state_storage, None)
        if isinstance(state, dict):
            bandit_state = state.get("bandit")
            text_state = state.get("text")
            if isinstance(bandit_state, dict):
                try:
                    self._bandit = OnlineLinearBandit.from_dict(bandit_state)
                except Exception:
                    self._bandit = OnlineLinearBandit(dim=len(self.DEFAULT_WEIGHTS), prior_mean=self.DEFAULT_WEIGHTS)
            if isinstance(text_state, dict):
                try:
                    self._text_model = OnlineTextModel.from_dict(text_state)
                except Exception:
                    self._text_model = OnlineTextModel()

    def _save_state(self):
        state = {
            "bandit": self._bandit.to_dict(),
            "text": self._text_model.to_dict(),
        }
        _json_save(self.state_storage, state)

    # ---------- Core ----------
    def ingest(self, text: str, source: str, liked: bool=False, tags: Optional[List[str]]=None):
        text = (text or "").strip()
        if not text or len(text.split()) < 3 or len(text) > 180:
            return
        if re.search(r"https?://", text):  # évite bouts de liens
            return
        q = Quote(text=text, source=source, ts=time.time(), liked=liked, tags=(tags or []))
        self.items.append(q)
        if len(self.items) > self.max:
            self.items = self.items[-self.max:]

    def _tok(self, s: str):
        norm = self._normalize(s)
        return set(re.findall(r"[a-z0-9']{2,}", norm))

    def _normalize(self, text: str) -> str:
        text = unicodedata.normalize("NFD", text or "")
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        text = re.sub(r"est\s+(un|une|le|la|l')", "est ", text, flags=re.IGNORECASE)
        return text.lower()

    def _sim(self, a: str, b: str) -> float:
        A, B = self._tok(a), self._tok(b)
        if not A or not B:
            return 0.0
        return len(A & B) / max(1, len(A | B))

    def _prepare_features(
        self, context: str, q: Quote, now: float
    ) -> Tuple[np.ndarray, float, Dict[int, int]]:
        recency = math.exp(-(now - q.ts) / (3 * 24 * 3600))
        overlap = self._sim(context, q.text)
        liked_flag = 1.0 if q.liked else 0.0
        cooldown_penalty = 1.0 if now - q.last_used <= 3 * 3600 else 0.0
        fatigue = min(q.uses, 20) / 20.0
        text_score_raw, text_feats = self._text_model.score(context, q.text)
        text_score = math.tanh(text_score_raw)
        features = np.array(
            [
                1.0,
                recency,
                overlap,
                liked_flag,
                cooldown_penalty,
                fatigue,
                text_score,
            ],
            dtype=float,
        )
        return features, text_score, text_feats

    def sample(self, context: str) -> Optional[str]:
        if not self.items:
            self._last_served = None
            self._last_features = None
            self._last_text_feats = None
            return None

        context = context or ""
        now = time.time()
        theta_sample = self._bandit.sample_weights()
        theta_mean = self._bandit.mean_weights()

        candidates = []
        for idx, q in enumerate(self.items):
            feats, text_score, text_feats = self._prepare_features(context, q, now)
            sample_score = float(np.dot(theta_sample, feats))
            mean_score = float(np.dot(theta_mean, feats))
            candidates.append((sample_score, mean_score, idx, feats, text_score, text_feats))

        candidates.sort(key=lambda item: item[0], reverse=True)
        selected = self._llm_select_candidate(context, candidates[:12])
        fallback = None
        for entry in candidates[:12]:
            if fallback is None or entry[4] > fallback[4]:
                fallback = entry
            if selected is None:
                sample_score, mean_score, _, _, text_score, _ = entry
                if mean_score > 0.05 or text_score > 0.45:
                    selected = entry
                    break

        if selected is None and fallback and fallback[4] > 0.3:
            selected = fallback

        if selected is None:
            self._last_served = None
            self._last_features = None
            self._last_text_feats = None
            return None

        _, _, pick, feats, _, text_feats = selected
        self.items[pick].uses += 1
        self.items[pick].last_used = now
        self._last_served = pick
        self._last_features = feats
        self._last_text_feats = text_feats
        return self.items[pick].text

    def reward_last(self, r: float):
        if self._last_served is None:
            return
        q = self.items[self._last_served]
        if r > 0:
            q.liked = True
            q.uses = max(0, q.uses - 1)
        elif r < 0:
            q.uses += 1
        reward = max(-1.0, min(1.0, r))
        if self._last_features is not None:
            self._bandit.update(self._last_features, reward)
        if self._last_text_feats is not None:
            self._text_model.update(self._last_text_feats, reward)
        self._save_state()

    def _llm_select_candidate(
        self, context: str, candidates: List[Tuple[float, float, int, np.ndarray, float, Dict[int, int]]]
    ) -> Optional[Tuple[float, float, int, np.ndarray, float, Dict[int, int]]]:
        if not candidates:
            return None

        feature_labels = [
            "bias",
            "recency",
            "overlap",
            "liked_flag",
            "cooldown_penalty",
            "fatigue",
            "text_score",
        ]
        payload_candidates = []
        index_map: Dict[int, Tuple[float, float, int, np.ndarray, float, Dict[int, int]]] = {}
        for local_id, entry in enumerate(candidates):
            sample_score, mean_score, idx, feats, text_score, _ = entry
            quote = self.items[idx]
            feature_values = {name: float(feats[i]) for i, name in enumerate(feature_labels)}
            payload_candidates.append(
                {
                    "id": local_id,
                    "quote_index": idx,
                    "text": quote.text[:240],
                    "liked": bool(quote.liked),
                    "uses": int(quote.uses),
                    "last_used_seconds": float(max(0.0, time.time() - quote.last_used)),
                    "sample_score": float(sample_score),
                    "mean_score": float(mean_score),
                    "text_score": float(text_score),
                    "features": feature_values,
                    "tags": list(quote.tags or [])[:8],
                }
            )
            index_map[local_id] = entry

        payload = {
            "context": (context or "")[-600:],
            "candidates": payload_candidates,
        }
        response = try_call_llm_dict(LLM_SPEC_KEY, input_payload=payload, logger=LOGGER)
        if not isinstance(response, dict):
            return None

        if response.get("reject_all"):
            return None

        selected_id = response.get("selected_id")
        try:
            selected_idx = int(selected_id)
        except (TypeError, ValueError):
            selected_idx = None

        if selected_idx is not None and selected_idx in index_map:
            entry = index_map[selected_idx]
            quote = self.items[entry[2]]
            quote.tags = list(set((quote.tags or []) + response.get("selected_tags", [])))
            return entry

        alternatives = response.get("alternatives")
        if isinstance(alternatives, list):
            for alt in alternatives:
                try:
                    alt_idx = int(alt)
                except (TypeError, ValueError):
                    continue
                if alt_idx in index_map:
                    return index_map[alt_idx]

        return None

    # util: ingestion de fichier texte en extraits "réutilisables"
    def ingest_file_units(self, path: str, liked: bool=True):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f):
                    t = (line or "").strip()
                    if 8 <= len(t.split()) <= 24:
                        self.ingest(t, source=f"inbox:{path}#L{i+1}", liked=liked)
        except Exception:
            pass
