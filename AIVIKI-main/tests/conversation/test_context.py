import pathlib
import sys
import time

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AGI_Evolutive.conversation.context import ContextBuilder, OnlineTopicClassifier


class DummyMemory:
    def __init__(self, memories):
        self._memories = list(memories)

    def get_recent_memories(self, n=100):
        return list(self._memories)[-n:]


class DummyUserModel:
    def __init__(self, persona=None, user_id="user-1"):
        self._persona = persona or {}
        self._user_id = user_id

    def describe(self):
        return {"id": self._user_id, "persona": self._persona}


class DummyArch:
    def __init__(self, memories, persona=None):
        self.memory = DummyMemory(memories)
        self.user_model = DummyUserModel(persona=persona)
        self.consolidator = type("Consolidator", (), {"state": {"lessons": ["note A", "note B"]}})()
        self.related_index = None


def _make_msg(text, ts=None, kind="interaction", tags=None):
    if ts is None:
        ts = time.time()
    return {
        "text": text,
        "kind": kind,
        "memory_type": kind,
        "tags": tags or [],
        "ts": ts,
    }


def test_ttl_selection_and_monitoring_updates():
    base = time.time() - 6 * 86400
    memories = [
        _make_msg("premier échange", ts=base + i * 86400) for i in range(6)
    ]
    arch = DummyArch(memories)
    builder = ContextBuilder(arch)

    ctx = builder.build("ping")

    ttl_info = ctx["monitoring"]["ttl"]["interaction"]
    assert ttl_info["days"] in ContextBuilder.TTL_OPTIONS_DAYS
    assert ttl_info["days"] == 3  # intervals ≈ 1 jour ⇒ demi-vie la plus proche : 3 jours
    assert ctx["active_thread"]


def test_topics_handle_french_structures_and_classifier():
    now = time.time()
    memories = [
        _make_msg("C'est une Innovation majeure pour l'équipe", ts=now - 60),
        _make_msg("Analyse des métriques financières", ts=now - 30),
    ]
    arch = DummyArch(memories)
    builder = ContextBuilder(arch)

    topics = builder._topics(memories)
    assert "innovation" in topics
    assert "analyse" in topics


def test_user_style_persona_overrides():
    now = time.time()
    memories = [
        _make_msg("Message très long " + ("x" * 140), ts=now - 120),
        _make_msg("Une question?", ts=now - 60),
    ]
    persona = {"tone": "verbose-analytical", "values": {"curiosity": 0.8}}
    arch = DummyArch(memories, persona=persona)
    builder = ContextBuilder(arch)

    ctx = builder.build("Salut")
    style = ctx["user_style"]
    assert style["prefers_long"] is True
    assert style["asks_questions"] is True


def test_sequence_motifs_detection():
    base = time.time()
    texts = [
        "Plan d'action pour le projet",
        "Plan d'action détaillé",
        "Plan d'action validé",
        "Analyse finale",
    ]
    memories = [_make_msg(txt, ts=base + idx * 120) for idx, txt in enumerate(texts)]
    arch = DummyArch(memories)
    builder = ContextBuilder(arch)

    ctx = builder.build("ok")
    motifs = ctx["sequence_motifs"]
    assert any("plan → d" in motif or "plan → action" in motif for motif in motifs)


def test_online_topic_classifier_updates():
    clf = OnlineTopicClassifier()
    txt = "Analyse des données financières !"
    clf.observe(txt, ["analyse", "données"])
    score = clf.score("analyse")
    assert score > 0.5
    fallback = clf.fallback_topics("Analyse complémentaire des données", exclude=[], top_k=2)
    assert "analyse" in fallback
