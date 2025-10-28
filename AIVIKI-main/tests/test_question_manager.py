"""Tests for contextual question generation in the QuestionManager."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AGI_Evolutive.core.question_manager import QuestionManager


class _MemoryStoreStub:
    def __init__(self, items: Optional[List[Dict[str, Any]]] = None) -> None:
        self._items = list(items or [])

    def add(self, entry: Dict[str, Any]) -> None:
        self._items.append(entry)

    def get_recent_memories(self, n: int = 50) -> List[Dict[str, Any]]:
        return list(self._items[-n:])


class _MemoryStub:
    def __init__(self, items: Optional[List[Dict[str, Any]]] = None) -> None:
        self.store = _MemoryStoreStub(items)

    def add_memory(self, entry: Dict[str, Any]) -> None:
        self.store.add(entry)


class _DummyGoals:
    def __init__(self, description: Optional[str]) -> None:
        self._description = description

    def get_active_goal(self) -> Optional[Dict[str, Any]]:
        if not self._description:
            return None
        return {"description": self._description}


class _DummyArch:
    def __init__(
        self,
        goal_description: Optional[str],
        *,
        memory: Optional[_MemoryStub] = None,
    ) -> None:
        self.goals = _DummyGoals(goal_description)
        self.memory = memory or _MemoryStub()


def _pop_question_texts(qm: QuestionManager) -> list[str]:
    qm.maybe_generate_questions()
    pending = qm.pop_questions()
    return [item["text"] for item in pending]


def test_question_manager_anchors_on_active_goal_topic():
    arch = _DummyArch("Comprendre le concept « empathie » pour mieux aider l'utilisateur")
    qm = QuestionManager(arch)
    qm.record_information_need("goal_focus", 0.72)

    questions = _pop_question_texts(qm)

    assert questions, "A contextual question should be generated"
    assert any("empathie" in q.lower() for q in questions)


def test_question_manager_uses_metadata_topic_override():
    arch = _DummyArch("Explorer les émotions complexes")
    qm = QuestionManager(arch)
    qm.record_information_need(
        "evidence",
        0.65,
        metadata={"topic": "auto-régulation émotionnelle"},
    )

    questions = _pop_question_texts(qm)

    assert questions, "A contextual evidence question should be generated"
    assert any("auto-régulation" in q.lower() for q in questions)


def test_question_manager_falls_back_to_library_when_no_focus():
    arch = _DummyArch(None)
    qm = QuestionManager(arch)
    qm.record_information_need("success_metric", 0.5)

    questions = _pop_question_texts(qm)

    assert questions, "Library question should be available"
    assert any("indicateur" in q.lower() for q in questions)


def test_primary_queue_blocks_at_capacity():
    arch = _DummyArch(None)
    qm = QuestionManager(arch)
    for idx in range(qm.max_primary + 5):
        qm.add_question(f"Question {idx}")

    assert len(qm.pending_questions) == qm.max_primary
    assert "primary" in qm.blocked_channels()

    # Résolution de plusieurs questions pour libérer la file
    removed = 0
    while removed < 6 and qm.pending_questions:
        q = qm.pending_questions[0]
        qm.resolve_question(q.get("id"))
        removed += 1

    assert "primary" not in qm.blocked_channels()


def test_immediate_queue_limit_and_overflow():
    arch = _DummyArch(None)
    qm = QuestionManager(arch)
    for idx in range(qm.max_immediate + 3):
        qm.add_question(
            f"Alerte {idx}",
            qtype="trigger",
            metadata={"source": "trigger", "immediacy": 0.9},
        )

    immediates = [q for q in qm.pending_questions if q.get("meta", {}).get("channel") == "immediate"]
    assert len(immediates) == qm.max_immediate
    assert "immediate" in qm.blocked_channels()

    # Libère toutes les questions immédiates
    for q in list(immediates):
        qm.resolve_question(q.get("id"))

    assert not qm.is_channel_blocked("immediate")


def test_attempt_auto_answers_uses_recent_memory():
    memory = _MemoryStub(
        [
            {
                "id": "mem1",
                "text": "Le projet Phoenix est en cours de finalisation",
                "kind": "note",
            }
        ]
    )
    arch = _DummyArch(None, memory=memory)
    qm = QuestionManager(arch)
    qm.add_question("Quel est l'état du projet Phoenix ?")
    qm.attempt_auto_answers()

    question = qm.pending_questions[0]
    meta = question.get("meta", {})
    suggestions = meta.get("auto_suggestions")
    assert suggestions, "Une suggestion automatique doit être enregistrée"
    assert "phoenix" in suggestions[0]["text"].lower()


def test_llm_auto_answer_resolves_stale_question(monkeypatch):
    from AGI_Evolutive.core import question_manager as qm_module

    arch = _DummyArch("Comprendre l'empathie humaine")
    qm = QuestionManager(arch)
    qm.add_question("Qu'est-ce que l'empathie ?")

    assert qm.pending_questions, "La question doit être enregistrée"
    question = qm.pending_questions[0]
    meta = question.setdefault("meta", {})
    meta["queued_at"] = time.time() - (3 * 3600)
    meta.pop("auto_attempted", None)

    captured: Dict[str, Any] = {}

    def fake_try_call_llm_dict(spec_key: str, **kwargs):
        captured["spec_key"] = spec_key
        captured["payload"] = kwargs.get("input_payload")
        return {
            "answer": "L'empathie est la capacité à comprendre et partager les émotions d'autrui.",
            "confidence": 0.82,
            "concepts": [
                {
                    "label": "empathie émotionnelle",
                    "definition": "Résonance avec l'état affectif d'une autre personne",
                    "example": "Ressentir la tristesse d'un ami et l'accompagner avec bienveillance",
                }
            ],
            "keywords": ["empathie", "émotions"],
            "insights": ["L'empathie combine perception émotionnelle et cognition."],
            "notes": "Réponse générée automatiquement",
        }

    monkeypatch.setattr(qm_module, "try_call_llm_dict", fake_try_call_llm_dict)

    attempts = qm.attempt_auto_answers()

    assert captured.get("spec_key") == "question_auto_answer"
    assert not qm.pending_questions, "La question doit être résolue après la réponse LLM"
    assert attempts and attempts[0]["suggestion"]["source"] == "llm"

    stored = arch.memory.store.get_recent_memories()
    assert any(item.get("kind") == "question_auto_answer" for item in stored)
