from typing import List, Tuple, Dict
import math, time
def _sigmoid(x): return 1/(1+math.exp(-x))
def recency_boost_fn(ts: float, half_life_days: float = 14.0) -> float:
    if not ts: return 0.0
    age_days = max(0.0, (time.time() - ts)/86400.0)
    return 2**(-age_days/half_life_days)
def fuse(dense: List[Tuple[str,float]], sparse: List[Tuple[str,float]],
         alpha=0.6, beta=0.4, meta: Dict[str, Dict] = None,
         recency_weight: float = 0.2, half_life_days: float = 14.0) -> List[Tuple[str,float]]:
    meta = meta or {}; scores: Dict[str,float] = {}
    for doc, s in dense: scores[doc]=scores.get(doc,0.0)+alpha*s
    for doc, s in sparse: scores[doc]=scores.get(doc,0.0)+beta*s
    if recency_weight > 0.0:
        for doc in list(scores.keys()):
            m = meta.get(doc) or {}; trust = float(m.get("source_trust", 0.5)); ts = float(m.get("ts", 0.0))
            scores[doc] += recency_boost_fn(ts, half_life_days) * trust * recency_weight
    fused = list(scores.items()); fused.sort(key=lambda x: x[1], reverse=True); return fused
