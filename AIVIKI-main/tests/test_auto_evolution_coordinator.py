import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pytest

from AGI_Evolutive.autonomy.auto_evolution import AutoEvolutionCoordinator
from AGI_Evolutive.autonomy.auto_signals import AutoSignalRegistry, extract_keywords


class MemoryStub:
    def __init__(self, entry: Mapping[str, Any]) -> None:
        self.entry = dict(entry)
        self.added: List[tuple[str, Mapping[str, Any]]] = []

    def get_all_time_history(
        self,
        *,
        include_raw: bool = True,
        include_digests: bool = True,
        include_expanded: bool = True,
        include_knowledge: bool = True,
        include_self_model: bool = True,
        limit_recent: int = 512,
        limit_digests: int = 180,
        since_ts: Optional[float] = None,
        **_: Any,
    ) -> Mapping[str, Any]:
        ts = float(self.entry.get("ts", time.time()))
        if since_ts is not None and since_ts >= ts:
            combined: Sequence[Mapping[str, Any]] = []
        else:
            combined = [dict(self.entry)]
        return {
            "timeline": {"combined": combined},
            "knowledge": {},
        }

    def add_memory(self, kind: str, payload: Mapping[str, Any]) -> None:
        self.added.append((kind, payload))


class MetacogStub:
    def __init__(self) -> None:
        self.evaluations: List[Mapping[str, Any]] = []
        self.followups: List[Mapping[str, Any]] = []

    def evaluate_auto_intention(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        self.evaluations.append(payload)
        return {
            "accepted": True,
            "score": 0.9,
            "significance": 0.88,
            "alignment": 0.82,
            "emotional_drive": 0.76,
        }

    def plan_auto_intention_followup(
        self,
        payload: Mapping[str, Any],
        evaluation: Mapping[str, Any],
        signals: Sequence[Mapping[str, Any]],
        self_assessment: Optional[Mapping[str, Any]],
    ) -> None:
        self.followups.append(
            {
                "payload": dict(payload),
                "evaluation": dict(evaluation),
                "signals": list(signals),
                "self_assessment": dict(self_assessment or {}),
            }
        )


class SkillSandboxStub:
    def __init__(self) -> None:
        self.requests: List[tuple[str, str, Dict[str, Any], Sequence[str]]] = []

    def register_intention(
        self,
        *,
        action_type: str,
        description: str,
        payload: Optional[Dict[str, Any]] = None,
        requirements: Optional[Sequence[str]] = None,
    ) -> Mapping[str, Any]:
        record = (
            action_type,
            description,
            dict(payload or {}),
            list(requirements or []),
        )
        self.requests.append(record)
        return {
            "action_type": action_type,
            "description": description,
            "status": "pending",
        }


class MechanismStoreStub:
    def __init__(self) -> None:
        self.items: Dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self.items[key]

    def add(self, mai: Any) -> None:
        self.items[mai.id] = mai

    def update(self, mai: Any) -> None:
        self.items[mai.id] = mai


class GoalMeta:
    def __init__(self) -> None:
        self.success_criteria: List[str] = []
        self.updated_at: Optional[float] = None


class GoalSystemStub:
    def __init__(self) -> None:
        self.added: List[Mapping[str, Any]] = []
        self.metadata: Dict[str, GoalMeta] = {}

    def add_goal(self, **kwargs: Any) -> Any:
        goal_id = f"goal-{len(self.added) + 1}"
        self.added.append(dict(kwargs))
        meta = GoalMeta()
        self.metadata[goal_id] = meta
        return type("GoalNode", (), {"id": goal_id})()


class EvolutionStub:
    def __init__(self) -> None:
        self.events: List[Mapping[str, Any]] = []

    def record_feedback_event(
        self,
        channel: str,
        *,
        label: str,
        success: Optional[bool],
        confidence: float,
        heuristic: str,
        payload: Mapping[str, Any],
    ) -> None:
        self.events.append(
            {
                "channel": channel,
                "label": label,
                "success": success,
                "confidence": confidence,
                "heuristic": heuristic,
                "payload": dict(payload),
            }
        )


class SelfImproverStub:
    def __init__(self) -> None:
        self.cycles: List[int] = []

    def run_cycle(self, n_candidates: int = 1) -> None:
        self.cycles.append(n_candidates)


class ModuleListener:
    def __init__(self) -> None:
        self.events: List[tuple[Mapping[str, Any], Mapping[str, Any], Optional[Mapping[str, Any]]]] = []

    def on_auto_intention_promoted(
        self,
        event: Mapping[str, Any],
        evaluation: Optional[Mapping[str, Any]] = None,
        self_assessment: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.events.append((dict(event), dict(evaluation or {}), dict(self_assessment or {})))


class EmotionStub:
    emotional_intensity = 0.42
    affective_dimensions = {
        "valence_energy": {"valence": 0.15, "tension": 0.33}
    }


@pytest.fixture()
def coordinator_setup() -> Dict[str, Any]:
    ts = time.time()
    entry = {
        "id": "entry-1",
        "kind": "concept",
        "ts": ts,
        "text": "Construire une relation durable et empathique",
        "tags": ["relation", "empathie", "profondeur"],
    }
    memory = MemoryStub(entry)
    setup = {
        "memory": memory,
        "metacog": MetacogStub(),
        "sandbox": SkillSandboxStub(),
        "store": MechanismStoreStub(),
        "goals": GoalSystemStub(),
        "evolution": EvolutionStub(),
        "improver": SelfImproverStub(),
        "listener": ModuleListener(),
        "emotions": EmotionStub(),
        "signals": AutoSignalRegistry(),
    }
    return setup


def test_auto_evolution_promotes_concepts(coordinator_setup):
    setup = coordinator_setup
    coordinator = AutoEvolutionCoordinator(
        memory=setup["memory"],
        metacog=setup["metacog"],
        skill_sandbox=setup["sandbox"],
        evolution_manager=setup["evolution"],
        mechanism_store=setup["store"],
        self_improver=setup["improver"],
        goals=setup["goals"],
        emotions=setup["emotions"],
        modules=[setup["listener"]],
        interval=999.0,
        signal_registry=setup["signals"],
    )

    coordinator.tick()

    assert setup["sandbox"].requests, "Skill sandbox should receive the autonomous intention"
    action_type, description, payload, requirements = setup["sandbox"].requests[-1]
    assert action_type.startswith("concept_")
    assert "relation" in description.lower()
    assert payload.get("self_assessment")
    payload_signals = payload["signals"]
    tag_fragments = set(
        extract_keywords(
            setup["memory"].entry.get("text", ""),
            " ".join(setup["memory"].entry.get("tags", [])),
        )
    )
    assert tag_fragments, "memory tags should yield fragments for dynamic signals"
    for fragment in tag_fragments:
        assert any(
            fragment in sig.get("metric", "") or fragment in sig.get("name", "")
            for sig in payload_signals
        ), "payload signals should reflect extracted knowledge keywords"
    assert "relation" in requirements

    mai_id = f"auto::{action_type}"
    assert mai_id in setup["store"].items
    stored = setup["store"].items[mai_id]
    assert stored.metadata["auto_generated"]
    assert stored.metadata["self_assessment"]["checkpoints"]

    assert setup["metacog"].followups
    followup = setup["metacog"].followups[-1]
    assert followup["payload"]["action_type"] == action_type

    assert setup["goals"].added
    goal_kwargs = setup["goals"].added[-1]
    assert goal_kwargs["description"].startswith("Institutionnaliser")

    assert setup["memory"].added
    kind, payload_log = setup["memory"].added[-1]
    assert kind == "auto_evolution"
    assert payload_log["action_type"] == action_type

    assert setup["listener"].events
    event_payload, evaluation_payload, self_assessment = setup["listener"].events[-1]
    assert event_payload["action_type"] == action_type
    assert self_assessment["checkpoints"]

    signals = setup["signals"].get_signals(action_type)
    for fragment in tag_fragments:
        assert any(fragment in sig.metric or fragment in sig.name for sig in signals)

    assert setup["evolution"].events
    evo_event = setup["evolution"].events[-1]
    assert evo_event["label"] == action_type

    assert setup["improver"].cycles == [2]

    # Second tick should not duplicate once processed
    coordinator.tick()
    assert len(setup["sandbox"].requests) == 1
    assert len(setup["memory"].added) == 1
