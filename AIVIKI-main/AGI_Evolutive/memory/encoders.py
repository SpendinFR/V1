import hashlib
import logging
import math
import random
import re
from collections import Counter
from collections.abc import Iterable as IterableABC
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from AGI_Evolutive.utils.llm_service import try_call_llm_dict

logger = logging.getLogger(__name__)


_BASE_TOKEN_PATTERN = re.compile(
    r"(?:[#@][\wàâäéèêëîïôöùûüçœæ]+|[a-zàâäéèêëîïôöùûüçœæ0-9]+(?:['’`-][a-zàâäéèêëîïôöùûüçœæ0-9]+)*)",
    re.IGNORECASE | re.UNICODE,
)


def tokenize(text: str) -> List[str]:
    """Tokenisation minimale (compatibilité historique)."""
    text = (text or "").lower()
    return _BASE_TOKEN_PATTERN.findall(text)


def _hash_to_dim(token: str, dim: int, seed: int = 13) -> int:
    h = hashlib.sha1((str(seed) + token).encode("utf-8")).hexdigest()
    return int(h, 16) % dim


def l2_normalize(vec: List[float]) -> List[float]:
    s = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / s for v in vec]


def cosine(a: List[float], b: List[float]) -> float:
    # assume both L2-normalized
    if len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


class AdaptiveTokenizer:
    """Tokenisation hybride auto-adaptative."""

    def __init__(
        self,
        min_token_freq: int = 5,
        min_phrase_freq: int = 3,
        max_vocab: int = 8000,
    ) -> None:
        self._min_token_freq = min_token_freq
        self._min_phrase_freq = min_phrase_freq
        self._max_vocab = max_vocab
        self._token_counts: Counter[str] = Counter()
        self._phrase_counts: Counter[str] = Counter()
        self._stable_tokens: set[str] = set()
        self._learned_phrases: set[str] = set()

    def tokenize(self, text: str, update: bool = False) -> Tuple[List[str], List[str]]:
        raw_tokens = tokenize(text)
        tokens = self._merge_phrases(raw_tokens)
        if update:
            self.learn(raw_tokens)
        return tokens, raw_tokens

    def learn(self, raw_tokens: Iterable[str]) -> None:
        tokens = list(raw_tokens)
        unique_tokens = set(tokens)
        for tok in unique_tokens:
            self._token_counts[tok] += 1
            if self._token_counts[tok] >= self._min_token_freq:
                self._stable_tokens.add(tok)
        for a, b in zip(tokens, tokens[1:]):
            phrase = f"{a}_{b}"
            self._phrase_counts[phrase] += 1
            if self._phrase_counts[phrase] >= self._min_phrase_freq:
                self._learned_phrases.add(phrase)
        self._trim_vocab()

    def _merge_phrases(self, raw_tokens: List[str]) -> List[str]:
        merged: List[str] = []
        i = 0
        while i < len(raw_tokens):
            if i < len(raw_tokens) - 1:
                candidate = f"{raw_tokens[i]}_{raw_tokens[i + 1]}"
                if candidate in self._learned_phrases:
                    merged.append(candidate)
                    i += 2
                    continue
            merged.append(raw_tokens[i])
            i += 1
        return merged

    def _trim_vocab(self) -> None:
        if len(self._learned_phrases) <= self._max_vocab:
            return
        # Retire les phrases les moins fréquentes en premier
        sorted_phrases = sorted(
            self._learned_phrases,
            key=lambda p: self._phrase_counts[p],
        )
        overflow = len(self._learned_phrases) - self._max_vocab
        for phrase in sorted_phrases[:overflow]:
            self._learned_phrases.remove(phrase)


class OnlineContextEmbeddings:
    """Projection continue via moyennes de contexte."""

    def __init__(self, dim: int = 256, seed: int = 37, lr: float = 0.05, window: int = 2):
        self.dim = dim
        self.seed = seed
        self.lr = lr
        self.window = window
        self._embeddings: Dict[str, List[float]] = {}

    def encode(self, tokens: Iterable[str], normalize: bool = False) -> List[float]:
        tokens_list = list(tokens)
        if not tokens_list:
            return [0.0] * self.dim
        vec = [0.0] * self.dim
        for tok in tokens_list:
            emb = self._embeddings.get(tok)
            if emb is None:
                emb = self._seed_vector(tok)
            for i, value in enumerate(emb):
                vec[i] += value
        if normalize:
            return l2_normalize(vec)
        return vec

    def update(self, tokens: List[str]) -> None:
        if not tokens:
            return
        for idx, token in enumerate(tokens):
            context_tokens = tokens[max(0, idx - self.window) : idx] + tokens[idx + 1 : idx + 1 + self.window]
            if not context_tokens:
                continue
            context_vec = [0.0] * self.dim
            for ctx in context_tokens:
                source = self._embeddings.get(ctx)
                if source is None:
                    source = self._seed_vector(ctx)
                for i, value in enumerate(source):
                    context_vec[i] += value
            scale = 1.0 / len(context_tokens)
            context_vec = [v * scale for v in context_vec]
            target = self._embeddings.get(token)
            if target is None:
                target = self._seed_vector(token)
                self._embeddings[token] = target
            for i in range(self.dim):
                target[i] = (1.0 - self.lr) * target[i] + self.lr * context_vec[i]

    def _seed_vector(self, token: str) -> List[float]:
        rnd = random.Random(hash((token, self.seed)))
        return [(rnd.random() * 2.0 - 1.0) / math.sqrt(self.dim) for _ in range(self.dim)]


@dataclass
class EncoderMonitor:
    smoothing: float = 0.05
    recalibration_interval: int = 128
    avg_token_len: float = 0.0
    avg_hash_norm: float = 0.0
    avg_semantic_norm: float = 0.0
    avg_similarity: float = 0.0
    calls: int = 0

    def observe(self, token_len: int, hash_norm: float, semantic_norm: float) -> None:
        self.calls += 1
        self.avg_token_len += (token_len - self.avg_token_len) * self.smoothing
        self.avg_hash_norm += (hash_norm - self.avg_hash_norm) * self.smoothing
        self.avg_semantic_norm += (semantic_norm - self.avg_semantic_norm) * self.smoothing

    def observe_similarity(self, value: float) -> None:
        self.avg_similarity += (value - self.avg_similarity) * self.smoothing

    def should_recalibrate(self) -> bool:
        return self.calls > 0 and self.calls % self.recalibration_interval == 0


class TinyEncoder:
    """Encodeur hybride évolutif combinant hashing et projection contextuelle."""

    def __init__(
        self,
        dim: int = 256,
        seed: int = 13,
        semantic_weight: float = 0.35,
        tokenizer: Optional[AdaptiveTokenizer] = None,
    ) -> None:
        self.dim = dim
        self.seed = seed
        self.semantic_weight = max(0.05, min(0.95, semantic_weight))
        self.tokenizer = tokenizer or AdaptiveTokenizer()
        self.semantic = OnlineContextEmbeddings(dim=dim, seed=seed + 17)
        self.monitor = EncoderMonitor()
        self.doc_freq: Counter[str] = Counter()
        self.total_docs = 0
        self._last_keywords: List[str] = []

    def encode(self, text: str, *, train: bool = True) -> List[float]:
        tokens, raw_tokens = self.tokenizer.tokenize(text, update=False)
        hash_vec = self._hash_features(tokens) if tokens else [0.0] * self.dim
        semantic_vec = (
            self.semantic.encode(tokens, normalize=False) if tokens else [0.0] * self.dim
        )
        combined = [
            (1.0 - self.semantic_weight) * h + self.semantic_weight * s
            for h, s in zip(hash_vec, semantic_vec)
        ]
        fallback_embedding = l2_normalize(combined) if tokens else [0.0] * self.dim
        fallback_keywords = self._extract_keywords(tokens, raw_tokens)

        llm_embedding: Optional[List[float]] = None
        llm_keywords: Optional[List[str]] = None
        if text.strip():
            llm_response = try_call_llm_dict(
                "memory_encoders",
                input_payload={
                    "text": text,
                    "token_sample": tokens[:32],
                    "dim": self.dim,
                    "stats": {
                        "token_count": len(tokens),
                        "unique_tokens": len(set(tokens)),
                    },
                },
                logger=logger,
            )
            if llm_response:
                llm_embedding = self._coerce_embedding(llm_response.get("embedding"))
                llm_keywords = self._normalize_keywords(llm_response.get("keywords"))

        if train:
            self._after_encode(tokens, raw_tokens, hash_vec, semantic_vec)

        if llm_embedding is not None:
            self._last_keywords = llm_keywords or fallback_keywords
            return llm_embedding

        self._last_keywords = fallback_keywords
        return fallback_embedding

    @property
    def last_keywords(self) -> List[str]:
        return list(self._last_keywords)

    def encode_with_keywords(self, text: str, *, train: bool = True) -> Dict[str, object]:
        embedding = self.encode(text, train=train)
        return {
            "embedding": embedding,
            "keywords": list(self._last_keywords),
        }

    def _extract_keywords(self, tokens: List[str], raw_tokens: List[str]) -> List[str]:
        counts = Counter(token for token in tokens if len(token) >= 3)
        keywords = [token.replace("_", " ") for token, _ in counts.most_common(6)]
        if keywords:
            return keywords
        if raw_tokens:
            fallback_counts = Counter(tok for tok in raw_tokens if len(tok) >= 3)
            return [token for token, _ in fallback_counts.most_common(6)]
        return []

    def _normalize_keywords(self, value: object) -> List[str]:
        if isinstance(value, str):
            normalized = value.strip()
            return [normalized] if normalized else []
        if isinstance(value, IterableABC):
            out: List[str] = []
            for item in value:
                if isinstance(item, str):
                    cleaned = item.strip()
                    if cleaned:
                        out.append(cleaned)
            return out
        return []

    def _coerce_embedding(self, value: object) -> Optional[List[float]]:
        if not isinstance(value, IterableABC):
            return None
        numbers: List[float] = []
        for item in value:  # type: ignore[assignment]
            try:
                numbers.append(float(item))
            except (TypeError, ValueError):
                return None
        if not numbers:
            return None
        if len(numbers) < self.dim:
            numbers.extend([0.0] * (self.dim - len(numbers)))
        elif len(numbers) > self.dim:
            numbers = numbers[: self.dim]
        norm = math.sqrt(sum(v * v for v in numbers)) or 1.0
        return [v / norm for v in numbers]

    def log_similarity(self, cosine_similarity: float) -> None:
        """Expose une API pour alimenter la boucle d'auto-évaluation."""
        self.monitor.observe_similarity(cosine_similarity)

    def snapshot(self) -> Dict[str, object]:
        return {
            "semantic_weight": self.semantic_weight,
            "tokenizer": {
                "token_counts": dict(self.tokenizer._token_counts),
                "phrase_counts": dict(self.tokenizer._phrase_counts),
                "learned_phrases": set(self.tokenizer._learned_phrases),
                "stable_tokens": set(self.tokenizer._stable_tokens),
            },
            "semantic_embeddings": {k: v[:] for k, v in self.semantic._embeddings.items()},
            "doc_freq": dict(self.doc_freq),
            "total_docs": self.total_docs,
        }

    def restore(self, snapshot: Dict[str, object]) -> None:
        self.semantic_weight = float(snapshot.get("semantic_weight", self.semantic_weight))
        tokenizer_state = snapshot.get("tokenizer", {})
        self.tokenizer._token_counts = Counter(tokenizer_state.get("token_counts", {}))
        self.tokenizer._phrase_counts = Counter(tokenizer_state.get("phrase_counts", {}))
        self.tokenizer._learned_phrases = set(tokenizer_state.get("learned_phrases", set()))
        self.tokenizer._stable_tokens = set(tokenizer_state.get("stable_tokens", set()))
        self.semantic._embeddings = {
            k: list(v)
            for k, v in ((snapshot.get("semantic_embeddings") or {}).items())
        }
        self.doc_freq = Counter(snapshot.get("doc_freq", {}))
        self.total_docs = int(snapshot.get("total_docs", 0))

    def _hash_features(self, tokens: List[str]) -> List[float]:
        counts = Counter(tokens)
        vec = [0.0] * self.dim
        for token, tf in counts.items():
            idx = _hash_to_dim(token, self.dim, self.seed)
            idf = self._idf(token)
            vec[idx] += tf * idf
        return vec

    def _idf(self, token: str) -> float:
        df = self.doc_freq.get(token, 0)
        return 1.0 + math.log(1.0 + (self.total_docs + 1) / (1.0 + df))

    def _after_encode(
        self,
        tokens: List[str],
        raw_tokens: List[str],
        hash_vec: List[float],
        semantic_vec: List[float],
    ) -> None:
        self.tokenizer.learn(raw_tokens)
        self.semantic.update(tokens)
        self.total_docs += 1
        for token in set(tokens):
            self.doc_freq[token] += 1
        hash_norm = math.sqrt(sum(v * v for v in hash_vec)) or 1.0
        semantic_norm = math.sqrt(sum(v * v for v in semantic_vec)) or 1.0
        self.monitor.observe(len(tokens), hash_norm, semantic_norm)
        if self.monitor.should_recalibrate():
            self._recalibrate()

    def _recalibrate(self) -> None:
        # Ajuste la pondération sémantique pour éviter la dérive
        hash_norm = max(1e-6, self.monitor.avg_hash_norm)
        ratio = self.monitor.avg_semantic_norm / hash_norm
        if ratio > 1.2:
            self.semantic_weight *= 0.9
        elif ratio < 0.8:
            self.semantic_weight *= 1.1
        self.semantic_weight = max(0.05, min(0.95, self.semantic_weight))
