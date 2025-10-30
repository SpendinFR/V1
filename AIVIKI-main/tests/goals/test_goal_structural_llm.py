import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AGI_Evolutive.goals import GoalSystem


@pytest.fixture()
def structural_goal_system(monkeypatch, tmp_path):
    calls = []

    llm_payload = {
        "confidence": 0.68,
        "notes": "Prioriser la carte des interactions avant de multiplier les pistes.",
        "children": [
            {
                "description": "Cartographier les alliances actives pour renforcer mon évolution.",
                "criteria": [
                    "Lister les partenaires ou entités influentes dans mon environnement immédiat.",
                    "Qualifier pour chacun la valeur apportée ou le risque associé.",
                    "Définir une action courte pour consolider l'alliance la plus fragile.",
                ],
                "goal_type": "exploration",
                "value": 0.71,
                "competence": 0.53,
                "curiosity": 0.76,
                "urgency": 0.48,
            },
            {
                "description": "Établir une expérimentation courte pour augmenter ma compréhension de moi-même.",
                "criteria": [
                    "Choisir une observation concrète sur mon fonctionnement actuel.",
                    "Formuler une hypothèse mesurable liée à cette observation.",
                    "Prévoir un rituel d'évaluation rapide après l'expérience.",
                ],
                "goal_type": "cognitive",
                "value": 0.74,
                "competence": 0.56,
                "curiosity": 0.72,
                "urgency": 0.44,
            },
        ],
    }

    def fake_try_call(spec_key, **kwargs):
        if spec_key == "goal_structural_children":
            calls.append(kwargs.get("input_payload"))
            return llm_payload
        return None

    monkeypatch.setattr("AGI_Evolutive.goals.try_call_llm_dict", fake_try_call)

    persist_path = tmp_path / "goals.json"
    dashboard_path = tmp_path / "dashboard.json"
    data_path = tmp_path / "goal_intentions.json"

    system = GoalSystem(
        persist_path=str(persist_path),
        dashboard_path=str(dashboard_path),
        intention_data_path=str(data_path),
    )

    yield system, calls

    for path in (persist_path, dashboard_path, data_path):
        if path.exists():
            path.unlink()


def test_llm_structural_templates_override(structural_goal_system):
    system, calls = structural_goal_system

    root = next(node for node in system.store.nodes.values() if not node.parent_ids)

    descriptions = [system.store.nodes[cid].description for cid in root.child_ids]
    assert descriptions == [
        "Cartographier les alliances actives pour renforcer mon évolution.",
        "Établir une expérimentation courte pour augmenter ma compréhension de moi-même.",
    ]

    parent_meta = system.metadata[root.id]
    assert parent_meta.llm_confidence == pytest.approx(0.68, rel=1e-2)
    assert parent_meta.llm_notes and "interactions" in parent_meta.llm_notes[-1].lower()

    assert calls, "La spec LLM doit avoir été invoquée avec le contexte du but"
    payload = calls[-1]
    assert payload["goal"]["description"] == root.description
    assert payload["fallback_children"], "Les modèles statiques doivent être fournis en inspiration"
