from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pytest

from AGI_Evolutive.language.voice import VoiceProfile


class DummyModel:
    def __init__(self, persona: dict | None = None):
        self.state = {"persona": persona or {}}


def build_profile(tmp_path: Path, persona: dict | None = None) -> VoiceProfile:
    return VoiceProfile(DummyModel(persona or {}), path=str(tmp_path / "voice.json"))


def test_feedback_regex_adjusts_conciseness(tmp_path: Path) -> None:
    vp = build_profile(tmp_path, {"tone": "friendly", "values": ["precision"]})
    before = vp.style()
    vp.update_from_feedback("C'est beaucoup trop long, merci de résumer.", positive=False)
    after = vp.style()
    assert after["conciseness"] > before["conciseness"]
    assert vp.classifier.is_trained


def test_register_blacklist_and_coupled_emoji(tmp_path: Path) -> None:
    vp = build_profile(tmp_path, {"tone": "formal"})
    before = vp.style()
    vp.update_from_feedback("Peux-tu être plus pro et évite les emojis stp", positive=False)
    after = vp.style()
    assert after["formality"] > before["formality"]
    assert after["emoji"] < before["emoji"]
    assert any(token.startswith("emoji") for token in vp.state["register_blacklist"])


def test_liked_source_boosts_warmth_and_emoji(tmp_path: Path) -> None:
    vp = build_profile(tmp_path, {"tone": "warm"})
    before = vp.style()
    vp.update_from_liked_source("inbox:42", ["J'adore 😄 ce ton !"])
    after = vp.style()
    assert "inbox:42" in vp.state["liked_sources"]
    assert after["emoji"] >= before["emoji"]
    assert after["warmth"] >= before["warmth"]


def test_persona_bootstrap_runs_once(tmp_path: Path) -> None:
    path = tmp_path / "persist.json"
    persona = {"tone": "friendly analytical"}
    vp = VoiceProfile(DummyModel(persona), path=str(path))
    first = vp.style()
    vp_reloaded = VoiceProfile(DummyModel(persona), path=str(path))
    reloaded = vp_reloaded.style()
    for knob in ("warmth", "analytical", "directness"):
        assert pytest.approx(first[knob], rel=0.01, abs=1e-3) == reloaded[knob]

