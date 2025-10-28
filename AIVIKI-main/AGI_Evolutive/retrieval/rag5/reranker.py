from typing import List, Tuple, Dict
import os
from sentence_transformers import CrossEncoder
_MODEL = os.getenv("RAG_RERANKER", "cross-encoder/ms-marco-MiniLM-L-6-v2")
_ce = CrossEncoder(_MODEL)
def rerank(query: str, docs: List[Tuple[str, str, float]], topk=20) -> List[Tuple[str, float]]:
    if not docs: return []
    pairs = [(query, text) for _, text, _ in docs]
    ce_scores = _ce.predict(pairs).tolist()
    rescored = []
    for (doc_id, _, fused), s in zip(docs, ce_scores):
        score = 0.7*float(s) + 0.3*float(fused)
        rescored.append((doc_id, score))
    rescored.sort(key=lambda x: x[1], reverse=True); return rescored[:topk]
def mmr(query_vec, doc_vecs: Dict[str, List[float]], ranked: List[Tuple[str, float]], lam=0.7, k=10):
    import math
    def cos(a,b):
        if not a or not b: return 0.0
        na=math.sqrt(sum(x*x for x in a)); nb=math.sqrt(sum(x*x for x in b))
        if na==0 or nb==0: return 0.0
        return sum(x*y for x,y in zip(a,b))/(na*nb)
    selected=[]; cand=[d for d,_ in ranked]
    while cand and len(selected)<k:
        best=None; best_score=-1e9
        for cid in cand:
            rel=dict(ranked).get(cid,0.0)
            div=max([cos(doc_vecs.get(cid,[]), doc_vecs.get(s,[])) for s in selected], default=0.0)
            score = lam*rel - (1-lam)*div
            if score>best_score: best=(cid,rel); best_score=score
        selected.append(best[0]); cand.remove(best[0])
    return [(cid, dict(ranked).get(cid,0.0)) for cid in selected]
