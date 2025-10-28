from typing import Dict, List, Tuple, Optional
import time
from .encoders import encode_query, encode_passage
from .store_ann import ANNIndex
from .store_sparse import BM25Index
from .hybrid import fuse
from .reranker import rerank, mmr
from .planner import expand_queries, decompose
from .compose import assemble_citations
from .guards import should_refuse, refusal_message
from .telemetry import log_event
from . import request_llm_guidance

class RAGPipeline:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.ann = ANNIndex(config=cfg.get('ann', {}))
        self.bm25 = BM25Index()
        self.docs: Dict[str, str] = {}
        self.meta: Dict[str, Dict] = {}
        self.doc_vecs: Dict[str, List[float]] = {}
    def add_document(self, doc_id: str, text: str, meta: Optional[Dict] = None):
        meta = meta or {}
        self.docs[doc_id] = text; self.meta[doc_id] = meta
        self.bm25.add(doc_id, text, meta)
        vec = encode_passage(text); self.doc_vecs[doc_id] = vec; self.ann.add(vec, doc_id)
    def ask(self, question: str) -> Dict:
        t0 = time.perf_counter()
        q_exp = expand_queries(question)
        llm_guidance = request_llm_guidance(
            question,
            config=self.cfg,
            expansions=q_exp,
        )
        if llm_guidance:
            suggested = llm_guidance.get("reformulated_queries") or llm_guidance.get("expansions")
            if isinstance(suggested, list):
                supplemental = [q for q in suggested if isinstance(q, str) and q.strip()]
                if supplemental:
                    combined = q_exp + [q for q in supplemental if q not in q_exp]
                    q_exp = combined
        dense_all = []; sparse_all = []
        for q in q_exp:
            qvec = encode_query(q)
            dense_all += self.ann.topk(qvec, k=self.cfg['retrieval']['topk_dense'])
            sparse_all += self.bm25.topk(q, k=self.cfg['retrieval']['topk_sparse'])
        fused = fuse(
            dense_all,
            sparse_all,
            alpha=self.cfg['retrieval']['alpha_dense'],
            beta=self.cfg['retrieval']['beta_sparse'],
            meta=self.meta,
            recency_weight=self.cfg['retrieval'].get('recency_boost', 0.0),
            half_life_days=self.cfg['retrieval'].get('recency_half_life_days', 14.0),
        )[: self.cfg['retrieval']['topk_fused']]
        t_retr = time.perf_counter()
        payload = [(doc_id, self.docs.get(doc_id, ''), score) for doc_id, score in fused]
        ranked = rerank(question, payload, topk=self.cfg['rerank']['topk'])
        ranked = mmr(encode_query(question), self.doc_vecs, ranked,
                     lam=self.cfg['rerank']['mmr_lambda'], k=self.cfg['rerank']['topk'])
        t_rer = time.perf_counter()
        citations = assemble_citations(question, self.docs, ranked,
                                       budget_tokens=self.cfg['compose']['budget_tokens'],
                                       snippet_chars=self.cfg['compose']['snippet_chars'],
                                       tokenizer=self.cfg['compose'].get('tokenizer'))
        t_comp = time.perf_counter()
        diagnostics = {
            "dense_hits": len(dense_all),
            "sparse_hits": len(sparse_all),
            "fused_hits": len(fused),
            "top_scores": [float(score) for _, score in ranked[:3]],
            "latency_ms": {
                "retrieval": (t_retr - t0) * 1000,
                "rerank": (t_rer - t_retr) * 1000,
                "compose": (t_comp - t_rer) * 1000,
            },
        }
        if llm_guidance:
            diagnostics["llm_guidance"] = llm_guidance
        if should_refuse(citations,
                         min_docs=self.cfg['guards']['min_support_docs'],
                         min_score=self.cfg['guards']['min_support_score'],
                         min_top1_score=self.cfg['guards'].get('min_top1_score', 0.25)):
            log_event({"type":"rag_result","status":"refused","q":question,"expanded":len(q_exp),
                       "retrieval":{"dense":len(dense_all),"sparse":len(sparse_all),"fused":len(fused)},
                       "latency_ms":{"retrieval":(t_retr-t0)*1000,"rerank":(t_rer-t_retr)*1000,"compose":(t_comp-t_rer)*1000}})
            diagnostics["decision"] = "refused"
            return refusal_message(self.cfg['guards']['refuse_message'], diagnostics=diagnostics)
        answer = " ".join(c['snippet'] for c in citations[:3])
        log_event({"type":"rag_result","status":"ok","q":question,"expanded":len(q_exp),
                   "retrieval":{"dense":len(dense_all),"sparse":len(sparse_all),"fused":len(fused)},
                   "top_docs":[c['doc_id'] for c in citations[:5]],
                   "latency_ms":{"retrieval":(t_retr-t0)*1000,"rerank":(t_rer-t_retr)*1000,"compose":(t_comp-t_rer)*1000}})
        diagnostics["decision"] = "ok"
        return {"answer": answer, "citations": citations, "status": "ok", "diagnostics": diagnostics}
