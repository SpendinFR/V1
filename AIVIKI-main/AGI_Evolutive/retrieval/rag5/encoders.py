# encoders.py — lazy load (aucune dépendance lourde au import)
from typing import List, Iterable
import os

_q = None
_p = None
_DEVICE = None

def _ensure_models():
    global _q, _p, _DEVICE
    if _q is not None and _p is not None:
        return
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError(f"Encoders indisponibles (installez torch & sentence-transformers) : {e}")

    _DEVICE = "cuda" if hasattr(torch, "cuda") and torch.cuda.is_available() else "cpu"
    model_q = os.getenv("RAG_QUERY_ENCODER", "intfloat/multilingual-e5-base")
    model_p = os.getenv("RAG_PASSAGE_ENCODER", model_q)

    _q = SentenceTransformer(model_q, device=_DEVICE)
    _p = SentenceTransformer(model_p, device=_DEVICE)

def _encode(texts: Iterable[str], is_query: bool) -> list:
    _ensure_models()
    model = _q if is_query else _p
    embs = model.encode(
        list(texts),
        batch_size=int(os.getenv("RAG_BATCH", "32")),
        normalize_embeddings=True,   # L2 norm -> cos/IP ok
        convert_to_numpy=True,
        show_progress_bar=False
    )
    return embs[0].astype("float32").tolist()

def encode_query(text: str) -> List[float]:
    return _encode([text], True)

def encode_passage(text: str) -> List[float]:
    return _encode([text], False)
