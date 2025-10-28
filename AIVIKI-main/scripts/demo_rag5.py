#!/usr/bin/env python3
import json, time, os
from AGI_Evolutive.retrieval.rag5.pipeline import RAGPipeline
if __name__=='__main__':
    os.makedirs("logs", exist_ok=True)
    os.environ.setdefault("RAG_LOG_PATH", "logs/rag.jsonl")
    cfg = json.loads(open('configs/rag.json','r',encoding='utf-8').read())
    rag = RAGPipeline(cfg)
    rag.add_document('d1', "Le RAG hybride combine recherche dense et sparse; un reranker croisé améliore la précision.", meta={"ts": time.time()-3600, "source_trust": 0.9})
    rag.add_document('d2', "Le Global Workspace diffuse les informations saillantes.", meta={"ts": time.time()-86400*10, "source_trust": 0.6})
    out = rag.ask("Explique le RAG hybride et le Global Workspace")
    print(out["status"])
    if out["status"]=="ok":
        print([(c['doc_id'], c['start'], c['end']) for c in out["citations"]])
        print(out["answer"][:200], "...")
    else:
        print(out["reason"])
