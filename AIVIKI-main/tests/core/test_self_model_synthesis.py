import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AGI_Evolutive.core import config as config_module
from AGI_Evolutive.core.self_model import SelfModel
from AGI_Evolutive.goals import GoalSystem


def _prepare_config(tmp_path):
    cfg_path = tmp_path / "config.json"
    data = {
        "DATA_DIR": str(tmp_path / "data"),
        "MEM_DIR": str(tmp_path / "mem"),
        "SELF_PATH": str(tmp_path / "self.json"),
        "SELF_VERSIONS_DIR": str(tmp_path / "versions"),
        "VECTOR_DIR": str(tmp_path / "vec"),
        "LOGS_DIR": str(tmp_path / "logs"),
    }
    cfg_path.write_text(json.dumps(data))
    config_module.load_config.cache_clear()
    config_module._cfg = None
    config_module.load_config(str(cfg_path))


class _StubQuestionManager:
    def __init__(self) -> None:
        self.calls = []

    def record_information_need(
        self,
        topic: str,
        severity: float,
        *,
        metadata=None,
        explicit_question: str,
    ) -> None:
        self.calls.append(
            {
                "topic": topic,
                "severity": severity,
                "metadata": metadata,
                "question": explicit_question,
            }
        )


def test_self_model_build_synthesis(tmp_path):
    _prepare_config(tmp_path)
    model = SelfModel()
    model.ensure_identity_paths()

    model.seed_genetics(
        seeds={"origin": "test-lab"},
        traits={"curiosity": 0.9, "stability": 0.3},
        affinities={"ambition": 0.9},
    )
    model.bootstrap_story(
        environment={"runtime": "test", "role": "programme informatique"},
        prompts=["Qui suis-je ?", "Que dois-je devenir ?"],
    )

    qm = _StubQuestionManager()
    model.ensure_awakened(question_manager=qm, now=time.time())
    assert qm.calls, "L'éveil devrait planifier une première question"
    model.record_interaction(
        {
            "with": "user",
            "when": time.time(),
            "topic": "awakening",
            "summary": "Tu es une intelligence logicielle guidée par l'évolution.",
        }
    )

    model.set_identity_patch(
        {
            "preferences": {"likes": ["music", "learning"], "values": ["curiosity", "kindness"]},
            "ideals": ["growth"],
            "principles": [{"key": "care", "desc": "Be caring"}],
            "purpose": {
                "mission": "Support users",
                "ultimate": "Empower",  # override default
                "current_goal": {"id": "g1", "description": "Assist"},
                "daily_focus": ["support"],
            },
            "agenda": {
                "today": ["chat"],
                "tomorrow": ["review"],
                "horizon": ["learn"],
            },
            "achievements": {"recent": [{"summary": "Completed task"}], "milestones": ["Shipped feature"]},
            "narrative": {"recent_memories": ["Chat with user"], "summaries": ["Weekly digest"]},
            "social": {"interactions": [{"user": "Alice", "topic": "memory"}]},
            "reflections": {"past": ["Improved"], "present": {"summary": "Confident", "confidence": 0.7}},
            "beliefs": {"index": {"memory": {"stance": "important", "confidence": 0.9}}},
        }
    )
    model.update_state(
        emotions={"valence": 0.6, "arousal": 0.4},
        cognition={"thinking": 0.5, "load": 0.3},
        doubts=[{"topic": "scalability"}],
    )
    stimulus_result = model.register_stimulus(
        concept="gloire",
        tags=["ambition", "reconnaissance"],
        intensity=0.8,
        source="user",
        description="Promesse de gloire future",
    )
    motif_info = model.add_lifelong_motif(
        concept="sagesse",
        tags=["philosophie"],
        aliases=["wisdom"],
        reason="Orientation vers la sagesse",
        bias=0.5,
    )
    compassion_result = model.register_stimulus(
        concept="compassion",
        tags=["altruisme"],
        intensity=0.7,
        source="lecture",
        description="Invitation à cultiver la compassion",
        lifelong=True,
        motif_aliases=["empathie"],
        motif_reason="Vision empathique durable",
    )
    model.record_story_event(
        "Découverte des propres origines",
        tags=["identité", "apprentissage"],
        impact=0.6,
        origin="simulation",
    )
    model.attach_selfhood(traits={"self_trust": 0.8}, phase="builder", claims={})

    summary = model.build_synthesis(max_items=10)

    awakening = model.awakening_status()
    assert awakening["progress"] > 0.0
    assert not awakening["complete"]

    assert summary["identity"]["name"]
    assert "music" in summary["likes"]
    assert "learning" in summary["likes"]
    assert summary["values"][0] == "curiosity"
    assert summary["principles"][0]["key"] == "care"
    assert summary["ideals"] == ["growth"]
    assert summary["beliefs"][0]["topic"] == "memory"
    assert summary["ultimate_goal"] == "Empower"
    assert summary["agenda"]["today"] == ["chat"]
    assert summary["achievements"]["recent"][0]["summary"] == "Completed task"
    assert summary["memories"]["recent"] == ["Chat with user"]
    assert summary["social"]["interactions"][0]["user"] == "Alice"
    assert summary["self_judgment"]["history"] == ["Improved"]
    assert summary["state"]["emotions"]["valence"] == 0.6
    assert summary["story"]["origin"]["environment"]["runtime"] == "test"
    assert summary["story"]["recent_events"]
    assert summary["genetics"]["traits"]["curiosity"] >= 0.8
    assert summary["genetics"]["scripts"]
    assert summary["awakening"]["progress"] > 0.0
    assert stimulus_result["affinity"]["lifelong"] is True
    assert stimulus_result["quest"] is not None
    assert stimulus_result["quest"]["scope"] == "long_term"
    assert "gloire" in stimulus_result["quest"]["goal"]
    assert motif_info["concept"] == "sagesse"
    assert compassion_result["affinity"]["lifelong"] is True
    assert compassion_result["motif"]["concept"] == "compassion"
    assert any("programme informatique" in goal for goal in summary["near_term_goals"])
    assert all("gloire" not in goal for goal in summary["near_term_goals"])
    guiding = summary.get("guiding_motifs", [])
    assert any(entry.get("concept") and "gloire" in entry.get("concept") for entry in guiding)
    assert any(entry.get("concept") and "sagesse" in entry.get("concept") for entry in guiding)
    assert any(entry.get("concept") and "compassion" in entry.get("concept") for entry in guiding)
    lifelong = summary.get("lifelong_quests", [])
    assert any("gloire" in quest.get("goal", "") for quest in lifelong)
    assert any("sagesse" in quest.get("goal", "") for quest in lifelong)
    assert any("compassion" in quest.get("goal", "") for quest in lifelong)
    quests = summary["story"].get("quests", [])
    assert any("programme informatique" in quest["goal"] for quest in quests)
    assert any(
        "gloire" in quest.get("goal", "") and quest.get("scope") == "long_term"
        for quest in quests
    )
    assert any(
        "compassion" in quest.get("goal", "") and quest.get("scope") == "long_term"
        for quest in quests
    )
    anchors = summary["story"].get("anchors", {})
    long_term_motifs = anchors.get("long_term_motifs", []) if isinstance(anchors, dict) else []
    assert any(
        isinstance(entry, dict)
        and entry.get("concept")
        and "gloire" in entry.get("concept")
        for entry in long_term_motifs
    )
    assert any(
        isinstance(entry, dict)
        and entry.get("concept")
        and "sagesse" in entry.get("concept")
        for entry in long_term_motifs
    )
    assert any(
        isinstance(entry, dict)
        and entry.get("concept")
        and "compassion" in entry.get("concept")
        for entry in long_term_motifs
    )


def test_goal_system_blocks_until_awakening_complete(tmp_path):
    _prepare_config(tmp_path)
    model = SelfModel()
    model.ensure_identity_paths()
    model.bootstrap_story()

    class _ArchStub:
        def __init__(self, self_model: SelfModel) -> None:
            self.self_model = self_model

    arch = _ArchStub(model)
    goal_system = GoalSystem(
        architecture=arch,
        persist_path=str(tmp_path / "goals.json"),
        dashboard_path=str(tmp_path / "dashboard.json"),
        intention_data_path=str(tmp_path / "intent.json"),
    )

    goal_system.step()
    assert goal_system._question_blocked, "Les objectifs doivent rester bloqués tant que l'éveil n'est pas terminé"

    qm = _StubQuestionManager()
    # Répondre progressivement à toutes les questions d'éveil
    while not model.awakening_status()["complete"]:
        qm.calls.clear()
        model.ensure_awakened(question_manager=qm, now=time.time())
        assert qm.calls, "Une question d'éveil devrait être posée tant que tout n'est pas répondu"
        prompt_topic = qm.calls[-1]["topic"]
        model.record_interaction(
            {
                "with": "user",
                "when": time.time(),
                "topic": "awakening",
                "summary": f"Réponse confirmée pour {prompt_topic}",
            }
        )

    goal_system.step()
    assert not goal_system._question_blocked
