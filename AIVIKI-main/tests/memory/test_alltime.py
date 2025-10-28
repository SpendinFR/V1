import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AGI_Evolutive.memory.alltime import LongTermMemoryHub
from AGI_Evolutive.memory.memory_store import MemoryStore


class _DummyBelief:
    def __init__(self, **fields):
        self.__dict__.update(fields)


class _DummyBeliefGraph:
    def iter_beliefs(self):
        now = time.time()
        return [
            _DummyBelief(
                id="belief-1",
                subject="self",
                subject_label="Self",
                relation="likes",
                value="learning",
                value_label="Learning",
                confidence=0.92,
                polarity=1,
                updated_at=now,
            )
        ]


class _StubSelfModel:
    def build_synthesis(self, max_items: int = 6):
        return {"identity": {"name": "Test Agent"}, "values": ["care"]}


def _make_store(tmp_path):
    path = tmp_path / "mem.json"
    return MemoryStore(path=str(path), max_items=200)


def test_timeline_and_digest_expansion(tmp_path):
    store = _make_store(tmp_path)
    now = time.time()
    raw = store.add_memory({"kind": "interaction", "text": "Hello", "ts": now - 120})
    digest = store.add_memory(
        {
            "kind": "digest.daily",
            "text": "Conversation summary",
            "ts": now - 60,
            "metadata": {"start_ts": now - 3600, "end_ts": now},
            "lineage": [raw["id"]],
        }
    )

    hub = LongTermMemoryHub(store)
    timeline = hub.timeline()
    ids = {item.get("id") for item in timeline["combined"]}
    assert raw["id"] in ids
    assert digest["id"] in ids

    expanded = hub.expand_digest(digest["id"])
    assert expanded and expanded[0]["id"] == raw["id"]

    details = hub.describe_period(days_ago=0, level="daily")
    assert details is not None
    assert details.digest_id == digest["id"]
    assert any(entry["id"] == raw["id"] for entry in details.coverage)


def test_full_history_includes_expansion_and_stats(tmp_path):
    store = _make_store(tmp_path)
    now = time.time()
    raw_one = store.add_memory({"kind": "interaction", "text": "Hi", "ts": now - 300})
    raw_two = store.add_memory({"kind": "reflection", "text": "Thinking", "ts": now - 200})
    digest = store.add_memory(
        {
            "kind": "digest.daily",
            "text": "Summary",
            "ts": now - 100,
            "metadata": {"start_ts": now - 400, "end_ts": now},
            "lineage": [raw_one["id"], raw_two["id"]],
        }
    )

    hub = LongTermMemoryHub(store)
    history = hub.full_history(
        limit_recent=50,
        limit_digests=50,
        include_expanded=True,
        include_knowledge=False,
        include_self_model=False,
    )

    assert "knowledge" not in history
    assert "self_model" not in history
    stats = history["stats"]
    assert stats["digest_count"] >= 1
    assert stats["raw_count"] >= 2
    assert stats["coverage_entries"] >= 2

    expanded_daily = history["expanded"]["daily"][0]
    assert expanded_daily["digest"]["id"] == digest["id"]
    expanded_ids = {item["id"] for item in expanded_daily["entries"]}
    assert {raw_one["id"], raw_two["id"]}.issubset(expanded_ids)


def test_snapshot_with_knowledge_and_self_model(tmp_path):
    store = _make_store(tmp_path)
    now = time.time()
    store.add_memory(
        {
            "kind": "goal_completion",
            "description": "Finish module",
            "goal_id": "goal-1",
            "completed_at": now - 10,
        }
    )
    store.add_memory({"kind": "lesson", "text": "Learned about memory hubs", "ts": now - 5})

    hub = LongTermMemoryHub(
        store,
        belief_graph=_DummyBeliefGraph(),
        self_model=_StubSelfModel(),
    )

    knowledge = hub.knowledge_snapshot(top_beliefs=5, top_completed_goals=5, top_lessons=5)
    assert knowledge["beliefs"]
    assert knowledge["completed_goals"]
    assert knowledge["lessons"]

    self_summary = hub.self_model_snapshot(max_items=3)
    assert self_summary.get("identity", {}).get("name") == "Test Agent"

    snapshot = hub.build_snapshot()
    assert "timeline" in snapshot and "knowledge" in snapshot and "self_model" in snapshot
    assert "stats" in snapshot and snapshot["stats"]["digest_count"] >= 0
