from __future__ import annotations

import json
import logging
import math
import os
import random
import re
import time
import unicodedata
from typing import Any, Dict, Iterable, List, Tuple

from AGI_Evolutive.utils.llm_service import try_call_llm_dict

class LiveLexicon:
    def __init__(self, path: str = "data/lexicon.json"):
        self.path = path
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self.lex: Dict[str, Any] = {
            "collocations": {},  # phrase -> {freq:int, liked:int, last:ts}
            "synonyms": {},      # key -> [variants]
        }
        # Metadata store to track usage statistics without breaking legacy data format
        self._load()
        self.lex.setdefault("_synonyms_meta", {})
        self.logger = logging.getLogger(f"{__name__}.LiveLexicon")

    def _load(self):
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    self.lex.update(json.load(f))
        except Exception:
            pass

    def save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.lex, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def add_from_text(self, text: str, liked: bool = False):
        # extrait des n-grammes simples (2-5 mots) – heuristique légère
        normalized = self._normalize_text(text)
        words = self._tokenize(normalized)
        triggered_llm = False
        for n in (2,3,4,5):
            for i in range(len(words)-n+1):
                phrase = " ".join(words[i:i+n])
                if len(phrase) < 8:
                    continue
                c = self.lex["collocations"].setdefault(
                    phrase,
                    {"freq": 0, "liked": 0, "last": 0, "last_selected": 0},
                )
                c["freq"] += 1
                c["last"] = time.time()
                if liked:
                    c["liked"] += 1
                if self._should_query_llm(c, liked):
                    if self._maybe_expand_with_llm(phrase, c, liked=liked):
                        triggered_llm = True
        # Nettoyage léger pour éviter l'accumulation de bruit
        self._cleanup_collocations()
        if triggered_llm:
            self.save()

    def prefer(self, phrase: str):
        key = self._normalize_text(phrase)
        c = self.lex["collocations"].setdefault(
            key,
            {"freq": 0, "liked": 0, "last": 0, "last_selected": 0},
        )
        c["liked"] += 1
        c["last"] = time.time()
        if self._should_query_llm(c, liked=True):
            if self._maybe_expand_with_llm(key, c, liked=True):
                self.save()
        self.save()

    def sample_variant(self, key: str, default: str) -> str:
        # renvoie un synonyme/variation si dispo
        variants = self.lex["synonyms"].get(key, [])
        if not variants:
            entry = self.lex["collocations"].get(key)
            if entry and self._should_query_llm(entry, liked=False):
                if self._maybe_expand_with_llm(key, entry, liked=False):
                    variants = self.lex["synonyms"].get(key, [])
                    self.save()
            if not variants:
                return default
        population = [default] + variants
        meta_store: Dict[str, Dict[str, float]] = self.lex.setdefault("_synonyms_meta", {}).setdefault(key, {})
        scores: List[Tuple[float, str]] = []
        now = time.time()
        for candidate in population:
            stats = meta_store.setdefault(
                candidate,
                {"freq": 0, "liked": 0, "last_selected": 0.0},
            )
            score = self._score_synonym(stats, now)
            scores.append((score, candidate))

        chosen = self._weighted_choice(scores, temperature=0.8)
        stats = meta_store.setdefault(
            chosen,
            {"freq": 0, "liked": 0, "last_selected": 0.0},
        )
        stats["freq"] += 1
        stats["last_selected"] = time.time()
        self.logger.debug("sample_variant key=%s choice=%s", key, chosen)
        return chosen

    def sample_collocation(self, novelty: float = 0.3) -> str | None:
        # favorise ce qui est liked ; dose nouveauté
        items = list(self.lex["collocations"].items())
        if not items:
            return None
        now = time.time()
        scored: List[Tuple[float, str]] = []
        for phrase, meta in items:
            # normalisation des champs au cas où des anciennes entrées manquent des clés
            if "last_selected" not in meta:
                meta["last_selected"] = meta.get("last", 0)
            score = self._score_collocation(meta, now)
            scored.append((score, phrase))

        if not scored:
            return None

        temperature = max(0.3, min(2.5, 0.8 + novelty))
        chosen = self._weighted_choice(scored, temperature=temperature)
        if chosen is None:
            return None

        meta = self.lex["collocations"].setdefault(
            chosen,
            {"freq": 0, "liked": 0, "last": 0, "last_selected": 0},
        )
        meta["last_selected"] = now
        meta.setdefault("freq", 0)
        meta.setdefault("liked", 0)
        self.logger.debug(
            "sample_collocation choice=%s score=%.3f novelty=%.2f",
            chosen,
            self._score_collocation(meta, now),
            novelty,
        )
        return chosen

    # --- helpers ---------------------------------------------------------

    def _normalize_text(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text or "")
        text = text.casefold()
        return text

    def _tokenize(self, text: str) -> List[str]:
        pattern = r"[a-zà-ÿ]+(?:'[a-zà-ÿ]+)?"
        return re.findall(pattern, text, flags=re.IGNORECASE)

    def _score_collocation(self, meta: Dict[str, Any], now: float) -> float:
        liked = max(0, meta.get("liked", 0))
        freq = max(0, meta.get("freq", 0))
        last_seen = max(meta.get("last", 0), meta.get("last_selected", 0))
        if last_seen:
            age_hours = max(0.0, (now - last_seen) / 3600.0)
            recency_factor = math.exp(-age_hours / 72.0)  # ~3 jours de demi-vie
        else:
            recency_factor = 0.5
        base = (liked * 2.5) + math.sqrt(freq + 1)
        exploration_bonus = 1.0 + 0.3 * random.random()
        return base * (0.6 + 0.4 * recency_factor) * exploration_bonus

    def _score_synonym(self, stats: Dict[str, Any], now: float) -> float:
        freq = max(0, stats.get("freq", 0))
        liked = max(0, stats.get("liked", 0))
        last_selected = stats.get("last_selected", 0)
        if last_selected:
            age_hours = max(0.0, (now - last_selected) / 3600.0)
            recency = math.exp(-age_hours / 48.0)
        else:
            recency = 0.7
        return (1 + liked) + 0.5 * math.sqrt(freq + 1) + recency

    def _weighted_choice(
        self,
        scored: Iterable[Tuple[float, str]],
        temperature: float = 1.0,
    ) -> str | None:
        scored_list = list(scored)
        if not scored_list:
            return None
        weights = []
        phrases = []
        max_score = max(score for score, _ in scored_list)
        for score, phrase in scored_list:
            adjusted = (score - max_score) / max(0.1, temperature)
            weight = math.exp(adjusted)
            weights.append(weight)
            phrases.append(phrase)
        total = sum(weights)
        if total <= 0:
            return random.choice(phrases)
        r = random.random() * total
        cumulative = 0.0
        for phrase, weight in zip(phrases, weights):
            cumulative += weight
            if r <= cumulative:
                return phrase
        return phrases[-1]

    def _cleanup_collocations(
        self,
        max_age_days: float = 180.0,
        min_freq: int = 1,
    ) -> None:
        now = time.time()
        threshold = now - max_age_days * 86400.0
        to_delete = []
        for phrase, meta in list(self.lex["collocations"].items()):
            freq = meta.get("freq", 0)
            last_seen = max(meta.get("last", 0), meta.get("last_selected", 0))
            if freq <= min_freq and last_seen and last_seen < threshold:
                to_delete.append(phrase)
        for phrase in to_delete:
            self.logger.debug("cleanup removing stale collocation=%s", phrase)
            self.lex["collocations"].pop(phrase, None)

    # --- LLM integration -------------------------------------------------

    def _should_query_llm(self, meta: Dict[str, Any], liked: bool) -> bool:
        now = time.time()
        last = float(meta.get("last_llm_ts", 0.0))
        cooldown = 6 * 3600.0 if liked else 12 * 3600.0
        if now - last < cooldown:
            return False
        freq = int(meta.get("freq", 0))
        if liked:
            return True
        return freq in {3, 5, 8} or (freq % 10 == 0 and freq > 0)

    def _maybe_expand_with_llm(self, phrase: str, meta: Dict[str, Any], *, liked: bool) -> bool:
        payload = {
            "phrase": phrase,
            "statistics": {
                "freq": int(meta.get("freq", 0)),
                "liked": int(meta.get("liked", 0)),
                "last_seen": meta.get("last", 0),
            },
            "existing_synonyms": self.lex["synonyms"].get(phrase, []),
            "liked": bool(liked),
        }
        response = try_call_llm_dict(
            "language_lexicon",
            input_payload=payload,
            logger=self.logger,
        )
        meta["last_llm_ts"] = time.time()
        if not response:
            return False

        changed = False
        synonyms = response.get("synonyms") or response.get("variants")
        if isinstance(synonyms, list):
            bucket = self.lex["synonyms"].setdefault(phrase, [])
            for candidate in synonyms:
                if not isinstance(candidate, str):
                    continue
                normalized = candidate.strip()
                if not normalized or normalized.lower() == phrase.lower():
                    continue
                if normalized not in bucket:
                    bucket.append(normalized)
                    changed = True

        related = response.get("collocations") or response.get("related_collocations")
        if isinstance(related, list):
            for colloc in related:
                if not isinstance(colloc, str):
                    continue
                cleaned = colloc.strip()
                if len(cleaned) < 8:
                    continue
                slot = self.lex["collocations"].setdefault(
                    cleaned,
                    {"freq": 0, "liked": 0, "last": 0, "last_selected": 0},
                )
                slot.setdefault("source", "llm")
                slot["last"] = time.time()
                changed = True

        register = response.get("register")
        if isinstance(register, str):
            meta["register"] = register

        if changed:
            self.logger.debug(
                "LLM expanded lexicon", extra={"phrase": phrase, "response": response}
            )
        return changed

