import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AGI_Evolutive.goals.dag_store import DagStore


class _Counter:
    def __init__(self):
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        return {"priority": 0.5, "confidence": 0.0}


def _make_store(tmp_path, monkeypatch):
    counter = _Counter()
    monkeypatch.setattr("AGI_Evolutive.goals.dag_store.try_call_llm_dict", counter)
    persist = tmp_path / "goals.json"
    dashboard = tmp_path / "dashboard.json"
    store = DagStore(str(persist), str(dashboard))
    return store, counter


def test_priority_review_called_once_for_stable_node(tmp_path, monkeypatch):
    store, counter = _make_store(tmp_path, monkeypatch)
    node = store.add_goal("test goal")
    assert counter.count == 1

    store._recompute_priority(node)
    assert counter.count == 1

    node.priority_last_review_at = time.time() - (store.priority_review_max_age + 1.0)
    store._recompute_priority(node)
    assert counter.count == 2


def test_priority_review_skips_within_cooldown_on_change(tmp_path, monkeypatch):
    store, counter = _make_store(tmp_path, monkeypatch)
    node = store.add_goal("test goal")
    assert counter.count == 1

    node.value = 0.6
    node.priority_last_review_at = time.time()
    store._recompute_priority(node)
    assert counter.count == 1

    node.priority_last_review_at = time.time() - (store.priority_review_cooldown + 1.0)
    store._recompute_priority(node)
    assert counter.count == 2
