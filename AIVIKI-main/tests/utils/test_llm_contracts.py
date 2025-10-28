import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AGI_Evolutive.utils.llm_contracts import enforce_llm_contract


def test_dialogue_state_contract_filters_and_clips():
    raw = {
        "state_summary": "  incident en cours  ",
        "open_commitments": [
            {
                "label": "  informer équipe  ",
                "deadline": "2024-05-10T09:00:00",
                "status": "  pending  ",
                "confidence": 1.7,
                "origin": 5,
            },
            "pas une entrée",
            {
                "commitment": "  archiver le log  ",
                "confidence": 0.2,
            },
        ],
        "pending_questions": {"q1": "  besoin de confirmation  ", "q2": "   "},
        "notes": "  beaucoup de bruit  ",
    }

    cleaned = enforce_llm_contract("dialogue_state", raw)

    assert cleaned == {
        "state_summary": "incident en cours",
        "open_commitments": [
            {
                "commitment": "informer équipe",
                "deadline": "2024-05-10T09:00:00",
                "status": "pending",
                "confidence": 1.0,
                "origin": "5",
            },
            {
                "commitment": "archiver le log",
                "confidence": 0.2,
            },
        ],
        "pending_questions": ["besoin de confirmation"],
        "notes": "beaucoup de bruit",
    }


def test_planner_support_contract_generates_stable_steps():
    raw = {
        "plan": [
            {
                "id": "",
                "description": "  observer l'environnement  ",
                "depends_on": ["phase-0", 42, ""],
                "priority": "1.4",
                "context": {"clue": "terrain", 3: "ignored"},
            },
            {
                "id": "phase-0",
                "description": "stabiliser position",
                "priority": "not-a-number",
            },
            {
                "id": "phase-0",
                "description": "planter balise",
            },
            "entrée invalide",
        ],
        "notes": {"raw": "kept"},
    }

    cleaned = enforce_llm_contract("planner_support", raw)

    assert cleaned["plan"] == [
        {
            "id": "llm_step_1",
            "description": "observer l'environnement",
            "depends_on": ["phase-0"],
            "priority": 1.0,
            "context": {"clue": "terrain"},
        },
        {
            "id": "phase-0",
            "description": "stabiliser position",
        },
        {
            "id": "phase-0_3",
            "description": "planter balise",
        },
    ]
    assert cleaned["notes"] == {"raw": "kept"}


def test_planner_support_contract_preserves_valid_payload():
    raw = {
        "plan": [
            {
                "id": "phase-1",
                "description": "consolider le camp",
                "depends_on": ["phase-0"],
                "priority": 0.75,
                "action_type": "log",
                "context": {"zone": "clairière"},
            },
            {
                "id": "phase-2",
                "description": "collecter témoignages",
                "depends_on": ["phase-1"],
                "priority": 0.55,
            },
        ],
        "notes": "plan cohérent",
    }

    cleaned = enforce_llm_contract("planner_support", raw)

    assert cleaned == raw


def test_episodic_linker_contract_clips_confidence():
    raw = {
        "links": [
            {
                "from": "  souvenir_a  ",
                "to": "souvenir_b",
                "type_lien": "  support  ",
                "confidence": -0.5,
                "notes": "  important  ",
            },
            {
                "from": "",
                "to": "souvenir_c",
                "type_lien": "cause",
            },
        ],
        "notes": "  conserver  ",
    }

    cleaned = enforce_llm_contract("episodic_linker", raw)

    assert cleaned == {
        "links": [
            {
                "from": "souvenir_a",
                "to": "souvenir_b",
                "type_lien": "support",
                "confidence": 0.0,
                "notes": "important",
            }
        ],
        "notes": "conserver",
    }


def test_enforce_llm_contract_handles_unknown_specs():
    payload = {"any": "value"}
    cleaned = enforce_llm_contract("non_critique", payload)
    assert cleaned == payload


def test_enforce_llm_contract_rejects_non_mappings():
    assert enforce_llm_contract("dialogue_state", [1, 2, 3]) is None
