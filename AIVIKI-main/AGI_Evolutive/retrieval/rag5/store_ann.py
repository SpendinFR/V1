from typing import List, Tuple
import os, numpy as np
_BACKEND = os.getenv("RAG_ANN", "").lower().strip() or None
class ANNIndex:
    def __init__(self, dim: int = None, config: dict = None):
        self.dim = dim; self.config = config or {}; self.ids: List[str] = []; self.index=None
        self.backend = (_BACKEND or self.config.get("backend") or "faiss").lower()
        if self.backend not in ("faiss","hnsw"): self.backend = "faiss"
    def _ensure_faiss(self, vec: np.ndarray):
        import faiss
        d = vec.shape[-1] if self.dim is None else self.dim; self.dim=d
        hnsw = bool(self.config.get("hnsw", False))
        metric = (self.config.get("metric") or "ip").lower()
        metric_id = faiss.METRIC_INNER_PRODUCT if metric=="ip" else faiss.METRIC_L2
        if hnsw:
            M = int(self.config.get("M", 32)); self.index = faiss.IndexHNSWFlat(d, M, metric_id)
            self.index.hnsw.efSearch = int(self.config.get("efSearch", 128))
        else:
            self.index = faiss.IndexFlatIP(d) if metric_id==faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(d)
    def _ensure_hnsw(self, vec: np.ndarray):
        import hnswlib
        d = vec.shape[-1] if self.dim is None else self.dim; self.dim=d
        space = (self.config.get("space") or "cosine").lower()
        self.index = hnswlib.Index(space=space, dim=d)
        self.index.init_index(max_elements=int(self.config.get("max_elements", 2000000)),
                              ef_construction=int(self.config.get("ef_construction", 200)),
                              M=int(self.config.get("M", 32)))
        self.index.set_ef(int(self.config.get("ef_search", 128)))
    def _ensure(self, vec: np.ndarray):
        if self.index is not None: return
        if (self.backend or "faiss")=="faiss": self._ensure_faiss(vec)
        else: self._ensure_hnsw(vec)
    def add(self, vec: List[float], doc_id: str):
        v = np.asarray([vec], dtype=np.float32); self._ensure(v)
        if self.backend=="faiss": self.index.add(v)
        else:
            import numpy as np
            self.index.add_items(v, np.array([len(self.ids)]))
        self.ids.append(doc_id)
    def topk(self, qvec: List[float], k=10) -> List[Tuple[str, float]]:
        if self.index is None: return []
        v = np.asarray([qvec], dtype=np.float32)
        if self.backend=="faiss":
            scores, I = self.index.search(v, k); out=[]
            for idx, s in zip(I[0], scores[0]):
                if idx == -1: continue
                out.append((self.ids[idx], float(s)))
            return out
        else:
            I, D = self.index.knn_query(v, k=k)
            return [(self.ids[idx], float(1.0 - dist)) for idx, dist in zip(I[0], D[0])]
