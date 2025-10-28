import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from AGI_Evolutive.self_improver.skill_acquisition import (
    SkillRequest,
    SkillSandboxManager,
    SkillTrial,
)


def test_list_skills_filters_and_formats(tmp_path):
    manager = SkillSandboxManager(storage_dir=str(tmp_path / "skills"), run_async=False)

    req_pending = SkillRequest(
        identifier="req-1",
        action_type="alpha_test",
        description="Alpha",
        payload={},
        created_at=1.0,
        status="awaiting_approval",
    )
    req_pending.trials = [
        SkillTrial(index=0, coverage=0.8, success=True),
        SkillTrial(index=1, coverage=0.6, success=False),
    ]
    req_pending.attempts = 2
    req_pending.successes = 1
    req_pending.requirements = ["alpha", "test"]

    req_failed = SkillRequest(
        identifier="req-2",
        action_type="beta_test",
        description="Beta",
        payload={},
        created_at=2.0,
        status="failed",
    )
    req_failed.trials = [
        SkillTrial(index=0, coverage=0.2, success=False),
    ]
    req_failed.attempts = 1
    req_failed.successes = 0
    req_failed.requirements = ["beta"]

    with manager._lock:  # type: ignore[attr-defined]
        manager._requests = {
            req_pending.action_type: req_pending,
            req_failed.action_type: req_failed,
        }

    awaiting = manager.list_skills(status="awaiting_approval", include_trials=True)
    assert len(awaiting) == 1
    entry = awaiting[0]
    assert entry["action_type"] == "alpha_test"
    assert entry["success_rate"] == 0.5
    assert len(entry["trials"]) == 2

    failed = manager.list_skills(status="failed")
    assert len(failed) == 1
    assert failed[0]["trial_count"] == 1


class _SimulatorStub:
    def __init__(self) -> None:
        self.calls = []

    def run(self, query):
        self.calls.append(query)
        requirements = query.get("requirements") or []
        assert any(
            "implémentation exécutable" in str(req) for req in requirements
        ), "Sandbox should request executable implementation details"
        assert query.get("implementation_required") is True
        details = query.get("implementation_details") or {}
        assert details.get("expect_operations") is True
        assert details.get("expect_steps") is True
        return {
            "success": True,
            "summary": "Essai validé dans le simulateur",
            "feedback": "Validation complète",
            "evidence": ["magie", "présentation"],
            "implementation": {
                "operations": {
                    "consigner": {
                        "type": "python",
                        "code": (
                            "result = {\n"
                            "    'ok': True,\n"
                            "    'value': 'validation',\n"
                            "    'timestamp': inputs.get('timestamp', 0)\n"
                            "}\n"
                        ),
                    }
                },
                "steps": [
                    {"op": "consigner", "inputs": {"timestamp": 1.0}, "store_as": "trace"},
                ],
            },
        }


class _MemoryStub:
    def __init__(self, items):
        self.items = list(items)
        self.calls = []

    def search(self, query, top_k=8):  # pragma: no cover - simple container
        self.calls.append((query, top_k))
        return list(self.items)


class _QuestionManagerStub:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def add_question(
        self,
        text: str,
        qtype: str = "custom",
        metadata: Optional[Dict[str, Any]] = None,
        priority: float = 0.6,
    ) -> None:
        self.calls.append(
            {
                "text": text,
                "qtype": qtype,
                "metadata": metadata or {},
                "priority": priority,
            }
        )


class _PerceptionStub:
    def __init__(self, inbox_dir: str) -> None:
        self.inbox_dir = inbox_dir


def test_training_records_practice_feedback(tmp_path):
    manager = SkillSandboxManager(
        storage_dir=str(tmp_path / "skills"),
        run_async=False,
        min_trials=1,
        success_threshold=0.4,
        training_interval=0.0,
    )
    simulator = _SimulatorStub()
    manager.bind(simulator=simulator)

    act = SimpleNamespace(
        type="perform_magic",
        payload={"description": "Faire un tour de magie", "requirements": ["magie", "présentation"]},
        context={},
    )

    result = manager.handle_simulation(act, interface=None)

    assert result["ok"] is False
    assert result["reason"] in {"skill_waiting_user_approval", "skill_training_in_progress"}
    assert simulator.calls, "Simulator should be invoked during training"

    status = manager.status("perform_magic")
    assert status["status"] in {"awaiting_approval", "active"}
    assert status["success_rate"] >= 1.0

    trials = status.get("trials", [])
    assert trials, "Trials should be recorded with simulator feedback"
    trial = trials[0]
    assert trial["mode"] == "simulator"
    assert trial["success"] is True
    assert "Essai validé" in (trial.get("summary") or "")
    assert any("magie" in evidence.lower() for evidence in trial.get("evidence", []))


def test_execute_runs_real_implementation(tmp_path):
    storage = tmp_path / "skills"
    manager = SkillSandboxManager(storage_dir=str(storage), run_async=False)

    data_file = tmp_path / "data.txt"
    data_file.write_text("hello world", encoding="utf-8")

    request = SkillRequest(
        identifier="req-impl",
        action_type="inspect_document",
        description="Lire un document",
        payload={},
        created_at=1.0,
        status="active",
    )
    request.implementation = {
        "kind": "sequence",
        "operations": {
            "read_file": {
                "type": "python",
                "code": (
                    "from pathlib import Path\n"
                    "path = inputs.get('path')\n"
                    "text = Path(path).read_text(encoding='utf-8')\n"
                    "result = {'ok': True, 'value': text, 'path': str(Path(path))}\n"
                ),
            }
        },
        "steps": [
            {"use": "read_file", "inputs": {"path": str(data_file)}, "store_as": "content"},
        ],
    }
    request.trials = [SkillTrial(index=0, coverage=1.0, success=True)]

    with manager._lock:  # type: ignore[attr-defined]
        manager._requests = {request.action_type: request}

    result = manager.execute("inspect_document", {})

    assert result["ok"] is True
    assert "result" in result
    outputs = result["result"]["outputs"]
    assert outputs["content"]["value"] == "hello world"

    status = manager.status("inspect_document")
    assert status["last_execution"] is not None
    assert Path(outputs["content"]["path"]).exists()


def test_gather_knowledge_uses_inbox_when_available(tmp_path):
    storage = tmp_path / "skills"
    inbox_dir = tmp_path / "inbox"
    inbox_dir.mkdir()
    (inbox_dir / "procedure.txt").write_text(
        "Procédure Alpha : étapes détaillées pour l'action.", encoding="utf-8"
    )

    manager = SkillSandboxManager(storage_dir=str(storage), run_async=False)
    perception = _PerceptionStub(str(inbox_dir))
    manager.bind(perception=perception)

    request = SkillRequest(
        identifier="req-inbox",
        action_type="alpha_action",
        description="Alpha",
        payload={},
        created_at=1.0,
    )

    knowledge = manager._gather_knowledge(request)

    assert any(item.get("kind") == "inbox" for item in knowledge)
    assert any(str(inbox_dir) in item.get("path", "") for item in knowledge)


def test_missing_information_triggers_question_manager(tmp_path):
    storage = tmp_path / "skills"
    manager = SkillSandboxManager(storage_dir=str(storage), run_async=False)
    qm = _QuestionManagerStub()
    manager.bind(question_manager=qm)

    request = SkillRequest(
        identifier="req-question",
        action_type="beta_action",
        description="Beta",
        payload={},
        created_at=1.0,
    )

    knowledge = manager._gather_knowledge(request)

    assert knowledge == []
    assert qm.calls, "QuestionManager should be solicited when knowledge is missing"
    queued = qm.calls[-1]
    assert queued["qtype"] == "skill_requirement"
    assert queued["metadata"].get("action_type") == "beta_action"

def test_execute_supports_custom_operations(tmp_path):
    storage = tmp_path / "skills"
    manager = SkillSandboxManager(storage_dir=str(storage), run_async=False)

    captured: List[Any] = []

    def remember_value(context, step):  # type: ignore[no-untyped-def]
        value = context.resolve_value(step.get("value"))
        captured.append(value)
        return {"ok": True, "value": value}

    manager.register_operation("remember_value", remember_value)

    request = SkillRequest(
        identifier="req-custom",
        action_type="memorise_fact",
        description="Retenir une valeur",
        payload={},
        created_at=1.0,
        status="active",
    )
    request.implementation = {
        "operations": {
            "prepare_value": {
                "type": "python",
                "code": (
                    "value = inputs.get('value')\n"
                    "result = {'ok': True, 'value': value}\n"
                ),
            }
        },
        "steps": [
            {
                "op": "prepare_value",
                "inputs": {"value": 42},
                "store_as": "prepared",
            },
            {"op": "remember_value", "value": "result.prepared.value", "store_as": "answer"},
            {
                "op": "prepare_value",
                "inputs": {"value": "skip"},
                "store_as": "skipped",
                "when": False,
            },
            {"op": "remember_value", "value": "result.answer.value", "store_as": "echo"},
        ]
    }
    request.trials = [SkillTrial(index=0, coverage=1.0, success=True)]

    with manager._lock:  # type: ignore[attr-defined]
        manager._requests = {request.action_type: request}

    response = manager.execute("memorise_fact", {})

    assert response["ok"] is True
    outputs = response["result"]["outputs"]
    assert "skipped" not in outputs
    assert outputs["answer"]["value"] == 42
    assert outputs["echo"]["value"] == 42
    assert captured == [42, 42]


def test_execute_python_operation_can_create_module(tmp_path):
    storage = tmp_path / "skills"
    manager = SkillSandboxManager(storage_dir=str(storage), run_async=False)

    module_path = tmp_path / "generated_module.py"

    request = SkillRequest(
        identifier="req-module",
        action_type="generate_module",
        description="Créer un module Python",
        payload={},
        created_at=1.0,
        status="active",
    )
    request.implementation = {
        "operations": {
            "build_module": {
                "type": "python",
                "code": (
                    "import textwrap\n"
                    "path = Path(inputs['path'])\n"
                    "body = textwrap.dedent(str(inputs.get('content', '')))\n"
                    "path.write_text(body, encoding='utf-8')\n"
                    "result = {'ok': True, 'value': str(path)}\n"
                ),
            },
            "load_answer": {
                "type": "python",
                "code": (
                    "import importlib.util\n"
                    "path = Path(inputs['path'])\n"
                    "name = inputs.get('name', 'generated_module')\n"
                    "spec = importlib.util.spec_from_file_location(name, path)\n"
                    "module = importlib.util.module_from_spec(spec)\n"
                    "spec.loader.exec_module(module)\n"
                    "answer = getattr(module, 'ANSWER', None)\n"
                    "result = {'ok': True, 'value': answer}\n"
                ),
            },
        },
        "steps": [
            {
                "op": "build_module",
                "inputs": {
                    "path": str(module_path),
                    "content": "ANSWER = 7\n",
                },
                "store_as": "module_file",
            },
            {
                "op": "load_answer",
                "inputs": {
                    "path": str(module_path),
                    "name": "generated_module",
                },
                "store_as": "module_answer",
            },
        ],
    }
    request.trials = [SkillTrial(index=0, coverage=1.0, success=True)]

    with manager._lock:  # type: ignore[attr-defined]
        manager._requests = {request.action_type: request}

    response = manager.execute("generate_module", {})

    assert response["ok"] is True
    outputs = response["result"]["outputs"]
    assert module_path.exists()
    assert outputs["module_file"]["value"] == str(module_path)
    assert outputs["module_answer"]["value"] == 7


def test_training_uses_knowledge_implementation(tmp_path):
    storage = tmp_path / "skills"
    manager = SkillSandboxManager(
        storage_dir=str(storage),
        run_async=False,
        min_trials=1,
        success_threshold=0.5,
        approval_required=False,
    )

    knowledge_entry = {
        "source": "memoire",
        "title": "Procédure analyse logs",
        "description": "Procédure pour analyser logs système",
        "content": {
            "implementation": {
                "kind": "sequence",
                "operations": {
                    "memo_plan": {
                        "type": "python",
                        "code": (
                            "info = inputs.get('info', '')\n"
                            "note = inputs.get('note', '')\n"
                            "result = {'ok': True, 'value': str(info) + ' :: ' + str(note)}\n"
                        ),
                    }
                },
                "steps": [
                    {
                        "op": "memo_plan",
                        "inputs": {
                            "info": "payload.description",
                            "note": {"from_knowledge": {"index": 0, "path": "description"}},
                        },
                        "store_as": "memo",
                    }
                ],
            }
        },
    }

    memory = _MemoryStub([knowledge_entry])
    manager.bind(memory=memory)

    manager.register_intention(
        action_type="analyse_logs",
        description="Analyser les journaux système",
        payload={"requirements": ["logs", "analyse"]},
        approval_required=False,
    )

    status = manager.status("analyse_logs")
    assert status["status"] == "active"
    assert status["implementation"] is not None
    assert status["trials"], "Knowledge practice should register a trial"
    assert status["trials"][0]["mode"] == "knowledge"

    result = manager.execute("analyse_logs", {"description": "Analyse journaux"})

    assert result["ok"] is True
    outputs = result["result"]["outputs"]
    assert outputs["memo"]["value"].startswith("Analyse journaux")
    assert "logs" in outputs["memo"]["value"]


def test_inject_requirements_skips_when_implementation_ready(tmp_path):
    manager = SkillSandboxManager(storage_dir=str(tmp_path / "skills"), run_async=False)

    request = SkillRequest(
        identifier="req-ready",
        action_type="act_ready",
        description="Action prête",
        payload={},
        created_at=1.0,
        status="training",
    )
    request.implementation = {
        "operations": {"noop": {"type": "python", "code": ("result = {'ok': True}\n",)}},
        "steps": [{"op": "noop"}],
    }

    snapshot = {"requirements": ["alpha"]}

    manager._inject_implementation_requirements(request, snapshot)

    assert snapshot["requirements"] == ["alpha"]
    assert "implementation_required" not in snapshot
    assert "implementation_details" not in snapshot
