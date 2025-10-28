from typing import List, Tuple, Dict
import os
_USE_ES = os.getenv("RAG_ES_HOST") is not None
if _USE_ES:
    from opensearchpy import OpenSearch
    _ES_HOST = os.getenv("RAG_ES_HOST", "http://localhost:9200")
    _ES_USER = os.getenv("RAG_ES_USER", "")
    _ES_PASS = os.getenv("RAG_ES_PASS", "")
    _ES_INDEX = os.getenv("RAG_ES_INDEX", "rag_docs")
    class BM25Index:
        def __init__(self):
            self.client = OpenSearch(hosts=[_ES_HOST],
                                     http_auth=(_ES_USER, _ES_PASS) if _ES_USER else None,
                                     timeout=10)
            if not self.client.indices.exists(_ES_INDEX):
                self.client.indices.create(_ES_INDEX, body={
                    "settings": {"index": {"number_of_shards": 1}},
                    "mappings": {"properties": {"doc_id": {"type": "keyword"},
                                                "text": {"type": "text"},
                                                "ts": {"type": "date"},
                                                "source_trust": {"type": "float"}}}
                })
        def add(self, doc_id: str, text: str, meta: Dict = None):
            meta = meta or {}
            self.client.index(index=_ES_INDEX, id=doc_id,
                              body={"doc_id": doc_id, "text": text,
                                    "ts": meta.get("ts"), "source_trust": meta.get("source_trust", 0.5)},
                              refresh="false")
        def topk(self, query: str, k=10) -> List[Tuple[str, float]]:
            res = self.client.search(index=_ES_INDEX, body={"size": k, "query": {"match": {"text": {"query": query}}}, "_source": False})
            hits = res.get("hits", {}).get("hits", [])
            return [(h["_id"], float(h["_score"])) for h in hits]
else:
    from collections import defaultdict
    import math
    class BM25Index:
        def __init__(self, k1=1.5, b=0.75):
            self.k1=k1; self.b=b; self.docs: Dict[str, Dict[str,int]]={}; self.df=defaultdict(int)
            self.avg_len=0.0; self.N=0
        def add(self, doc_id: str, text: str, meta: Dict = None):
            terms=text.lower().split(); freqs=defaultdict(int)
            for t in terms: freqs[t]+=1
            self.docs[doc_id]=dict(freqs)
            for t in freqs: self.df[t]+=1
            self.N+=1
            self.avg_len = sum(sum(d.values()) for d in self.docs.values())/self.N
        def score(self, query: str, doc_id: str) -> float:
            q_terms=query.lower().split(); dl=sum(self.docs[doc_id].values()); score=0.0
            for t in q_terms:
                if t not in self.docs[doc_id]: continue
                f=self.docs[doc_id][t]
                idf=math.log(1 + (self.N - self.df[t] + 0.5)/(self.df[t] + 0.5))
                denom = f + self.k1*(1-self.b + self.b*dl/self.avg_len)
                score += idf * (f*(self.k1+1))/denom
            return score
        def topk(self, query: str, k=10) -> List[Tuple[str,float]]:
            scored=[(doc_id, self.score(query, doc_id)) for doc_id in self.docs.keys()]
            scored.sort(key=lambda x:x[1], reverse=True); return scored[:k]
