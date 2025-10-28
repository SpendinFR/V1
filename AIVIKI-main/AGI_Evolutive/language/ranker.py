# language/ranker.py
import hashlib
import logging
import math
import os
from typing import Any, Dict, List, Mapping, Optional

from . import DATA_DIR, _json_load, _json_save
from AGI_Evolutive.utils.llm_service import try_call_llm_dict


logger = logging.getLogger(__name__)


def sigmoid(x):
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


class RankerModel:
    """
    Ranker linéaire léger (logit) + apprentissage pairwise online.
    On featurise (contexte, candidate) et on ajuste w via descente sur perte BT-LR.
    """

    def __init__(self, storage=None):
        self.storage = storage or os.path.join(DATA_DIR, "ranker.json")
        self.w: Dict[str, float] = {}
        self.load()
        self._llm_cache: Dict[str, float] = {}
        self._llm_explanations: Dict[str, str] = {}

    # --------- Persistence ----------
    def load(self):
        data = _json_load(self.storage, {})
        self.w = data.get("w", {})

    def save(self):
        _json_save(self.storage, {"w": self.w})

    # --------- Features ----------
    def featurize(self, context: Dict[str, Any], text: str) -> Dict[str, float]:
        # Features simples mais robustes
        f: Dict[str, float] = {}
        L = len(text)
        words = text.split()
        sents = max(1, text.count(".") + text.count("!") + text.count("?"))
        avg_len = L / max(1, len(words))
        # basiques
        f["len_chars/1k"] = min(2.0, L / 1000.0)
        f["len_words/100"] = min(2.0, len(words) / 100.0)
        f["avg_token_len"] = min(1.5, avg_len / 7.0)
        f["sent_density"] = min(1.5, sents / max(1, len(words) / 18))

        # “risques” simples
        f["has_allcaps"] = 1.0 if any(w.isupper() and len(w) >= 3 for w in words) else 0.0
        f["too_many_..."] = min(1.5, text.count("...") / 3.0)

        # diversité lexicale approx
        uniq = len(set(w.lower().strip(",.;:!?") for w in words))
        f["type_token"] = min(1.5, uniq / max(1, len(words)))

        # adéquation préférences (si présentes dans context)
        style = context.get("style", {})
        f["pref_warmth"] = float(style.get("warmth", 0.5))
        f["pref_directness"] = float(style.get("directness", 0.5))
        f["pref_hedging"] = float(style.get("hedging", 0.5))

        # pénalités quote longue (tag dans context)
        f["has_quote"] = 1.0 if context.get("has_quote") else 0.0
        f["quote_len/100"] = min(1.5, context.get("quote_len", 0) / 100.0)

        return f

    def _dot(self, f: Dict[str, float]) -> float:
        # auto-init poids à 0
        s = 0.0
        for k, v in f.items():
            s += v * self.w.get(k, 0.0)
        return s

    def score(self, context: Dict[str, Any], text: str) -> float:
        cache_key = self._cache_key(context, text)
        cached = self._llm_cache.get(cache_key)
        if cached is not None:
            return cached

        llm_score = self._score_with_llm(context, text, cache_key)
        if llm_score is not None:
            return llm_score

        f = self.featurize(context, text)
        return sigmoid(self._dot(f))

    def update_pair(self, context: Dict[str, Any], winner: str, loser: str, lr: float = 0.2):
        fw = self.featurize(context, winner)
        fl = self.featurize(context, loser)
        # prob préférée(w) > préférée(l)
        s_w = self._dot(fw)
        s_l = self._dot(fl)
        # gradient approché (logit pairwise)
        p = sigmoid(s_w - s_l)
        err = 1.0 - p  # on veut p→1
        # mise à jour
        for k in set(list(fw.keys()) + list(fl.keys())):
            gw = fw.get(k, 0.0)
            gl = fl.get(k, 0.0)
            self.w[k] = self.w.get(k, 0.0) + lr * err * (gw - gl)
        # régule doucement
        for k in list(self.w.keys()):
            self.w[k] *= 0.999

    def rank_candidates(self, context: Dict[str, Any], candidates: Mapping[str, str]) -> Dict[str, Dict[str, Any]]:
        if not candidates:
            return {}
        payload = {
            "context": self._llm_context_snapshot(context),
            "candidates": [
                {"id": str(candidate_id), "text": text}
                for candidate_id, text in candidates.items()
                if isinstance(text, str)
            ],
        }
        response = try_call_llm_dict("ranker_model", input_payload=payload, logger=logger)
        results: Dict[str, Dict[str, Any]] = {}
        if not response or not isinstance(response.get("ranking"), list):
            return results
        for entry in response["ranking"]:
            if not isinstance(entry, Mapping):
                continue
            candidate_id = str(entry.get("id") or "")
            if candidate_id not in candidates:
                continue
            score = entry.get("score")
            explanation = entry.get("explanation")
            try:
                float_score = max(0.0, min(1.0, float(score)))
            except (TypeError, ValueError):
                continue
            results[candidate_id] = {
                "score": float_score,
                "explanation": str(explanation) if explanation is not None else "",
            }
            cache_key = self._cache_key(context, candidates[candidate_id])
            self._llm_cache[cache_key] = float_score
            if explanation is not None:
                self._llm_explanations[cache_key] = str(explanation)
        return results

    def llm_explanation(self, context: Dict[str, Any], text: str) -> Optional[str]:
        cache_key = self._cache_key(context, text)
        if cache_key not in self._llm_cache:
            self._score_with_llm(context, text, cache_key)
        return self._llm_explanations.get(cache_key)

    def _score_with_llm(
        self, context: Dict[str, Any], text: str, cache_key: str
    ) -> Optional[float]:
        if not text.strip():
            return None
        candidate_id = cache_key[:12]
        payload = {
            "context": self._llm_context_snapshot(context),
            "candidates": [{"id": candidate_id, "text": text}],
        }
        response = try_call_llm_dict("ranker_model", input_payload=payload, logger=logger)
        if not response or not isinstance(response.get("ranking"), list):
            return None
        entry = next((item for item in response["ranking"] if isinstance(item, Mapping)), None)
        if not entry:
            return None
        score = entry.get("score")
        explanation = entry.get("explanation")
        try:
            float_score = max(0.0, min(1.0, float(score)))
        except (TypeError, ValueError):
            return None
        self._llm_cache[cache_key] = float_score
        if explanation is not None:
            self._llm_explanations[cache_key] = str(explanation)
        return float_score

    def _cache_key(self, context: Dict[str, Any], text: str) -> str:
        style = context.get("style") if isinstance(context, Mapping) else {}
        fragments = []
        if isinstance(style, Mapping):
            for key, value in style.items():
                try:
                    fragments.append(f"{key}:{float(value):.3f}")
                except (TypeError, ValueError):
                    fragments.append(f"{key}:?")
        raw = f"{';'.join(sorted(fragments))}|{text}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _llm_context_snapshot(self, context: Dict[str, Any]) -> Dict[str, Any]:
        snapshot: Dict[str, Any] = {}
        if not isinstance(context, Mapping):
            return snapshot
        style = context.get("style")
        if isinstance(style, Mapping):
            snapshot["style"] = {
                str(k): float(v)
                for k, v in style.items()
                if isinstance(v, (int, float))
            }
        for key, value in context.items():
            if key == "style":
                continue
            if isinstance(value, (str, int, float, bool)):
                snapshot.setdefault("hints", {})[str(key)] = value
        return snapshot
