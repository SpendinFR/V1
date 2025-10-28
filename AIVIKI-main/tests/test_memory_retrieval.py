import importlib.machinery
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "AGI_Evolutive" / "memory" / "retrieval.py"

agi_pkg = sys.modules.setdefault(
    "AGI_Evolutive",
    importlib.util.module_from_spec(importlib.machinery.ModuleSpec("AGI_Evolutive", None)),
)
if not hasattr(agi_pkg, "__path__"):
    agi_pkg.__path__ = [str(REPO_ROOT / "AGI_Evolutive")]

memory_pkg = sys.modules.setdefault(
    "AGI_Evolutive.memory",
    importlib.util.module_from_spec(importlib.machinery.ModuleSpec("AGI_Evolutive.memory", None)),
)
if not hasattr(memory_pkg, "__path__"):
    memory_pkg.__path__ = [str(REPO_ROOT / "AGI_Evolutive" / "memory")]

loader = importlib.machinery.SourceFileLoader("AGI_Evolutive.memory.retrieval", str(MODULE_PATH))
spec = importlib.util.spec_from_loader("AGI_Evolutive.memory.retrieval", loader)
if spec is None:
    raise RuntimeError("Unable to load memory.retrieval module for tests")
memory_retrieval = importlib.util.module_from_spec(spec)
memory_retrieval.__package__ = "AGI_Evolutive.memory"
sys.modules["AGI_Evolutive.memory.retrieval"] = memory_retrieval
spec.loader.exec_module(memory_retrieval)  # type: ignore[union-attr]

MemoryRetrieval = memory_retrieval.MemoryRetrieval
_FRENCH_PATTERN = memory_retrieval._FRENCH_PATTERN
_normalize_text = memory_retrieval._normalize_text


class DummySalienceScorer:
    def score(self, record: Dict[str, Any]) -> float:  # pragma: no cover - simple heuristic
        return 0.6 if "important" in record.get("text", "").lower() else 0.2


class DummyVectorStore:
    def __init__(self) -> None:
        self._docs: Dict[str, str] = {}
        self.feedback: List[Tuple[str, float]] = []

    def upsert(self, key: str, text: str) -> None:
        self._docs[key] = text

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        query_tokens = _normalize_text(query).split()
        results: List[Tuple[str, float]] = []
        for key, text in self._docs.items():
            norm_text = _normalize_text(text)
            score = 0.1
            for token in query_tokens:
                if token and token in norm_text:
                    score += 0.4
            if "important" in norm_text:
                score += 0.2
            if "emoji" in norm_text:
                score += 0.2
            results.append((key, min(score, 1.0)))
        results.sort(key=lambda item: (item[1], item[0]), reverse=True)
        return results[: max(1, k)]

    def register_feedback(self, key: str, reward: float) -> None:
        self.feedback.append((key, reward))


@pytest.fixture()
def retrieval_system() -> MemoryRetrieval:
    return MemoryRetrieval(
        vector_store=DummyVectorStore(),
        salience_scorer=DummySalienceScorer(),
    )


def _score_for(results: List[Dict[str, Any]], doc_id: int) -> float:
    for result in results:
        if result["id"] == doc_id:
            return float(result["score"])
    raise AssertionError(f"Document {doc_id} not found in results")


def test_adaptive_reranker_improves_after_feedback(retrieval_system: MemoryRetrieval) -> None:
    doc_good = retrieval_system.add_document(
        "Ceci est un texte important sur la mémoire évolutive !"
    )
    _doc_bad = retrieval_system.add_document("Contenu banal sans grand intérêt.")

    before = retrieval_system.search_text("texte important", top_k=2)
    assert len(before) == 2
    base_score = _score_for(before, doc_good)
    assert retrieval_system._reranker is not None  # type: ignore[truthy-bool]
    weight_snapshot = dict(  # type: ignore[attr-defined]
        retrieval_system._reranker._weight_model.weights
    )

    retrieval_system.register_feedback(doc_good, 1.0)

    after = retrieval_system.search_text("texte important", top_k=2)
    updated_score = _score_for(after, doc_good)

    assert updated_score + 1e-6 >= base_score
    weights_after = retrieval_system._reranker._weight_model.weights  # type: ignore[attr-defined]
    assert any(
        abs(weights_after[name] - weight_snapshot.get(name, 0.0)) >= 1e-6
        for name in weights_after
    )


def test_fallback_classifier_handles_accented_pattern(retrieval_system: MemoryRetrieval) -> None:
    doc_id = retrieval_system.add_document(
        "ÉST Une Réponse claire : l'intelligence est clé 😊"
    )

    retrieval_system.index.search_text = lambda query, top_k=5: []  # type: ignore[assignment]

    results = retrieval_system.search_text("Est-ce une bonne réponse ?", top_k=1)
    assert results
    result = results[0]
    assert result["id"] == doc_id
    assert result["_adaptive_features"]["fallback"] >= 0.5
    features = retrieval_system._reranker._fallback._extract_features(  # type: ignore[attr-defined]
        "Est-ce une bonne réponse ?",
        "ÉST Une Réponse claire : l'intelligence est clé 😊",
    )
    assert math.isclose(features["french_pattern"], 1.0)
    assert _FRENCH_PATTERN.search(_normalize_text("ÉST Une Réponse claire"))


def test_fallback_classifier_learns_from_feedback(retrieval_system: MemoryRetrieval) -> None:
    doc_id = retrieval_system.add_document("Réponse emoji 😊 ultra concise")

    retrieval_system.index.search_text = lambda query, top_k=5: []  # type: ignore[assignment]

    initial = retrieval_system.search_text("emoji", top_k=1)
    base_fallback = initial[0]["_adaptive_features"]["fallback"]

    retrieval_system.register_feedback(doc_id, 1.0)

    updated = retrieval_system.search_text("emoji", top_k=1)
    new_fallback = updated[0]["_adaptive_features"]["fallback"]

    assert new_fallback >= base_fallback
    vec_store: DummyVectorStore = retrieval_system.vector_store  # type: ignore[assignment]
    assert any(key.startswith("doc::") for key, _ in vec_store.feedback)
