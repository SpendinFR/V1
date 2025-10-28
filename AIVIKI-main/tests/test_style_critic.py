import pathlib
import sys

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AGI_Evolutive.language.style_critic import StyleCritic


@pytest.fixture()
def critic() -> StyleCritic:
    return StyleCritic(max_chars=80)


def _issue_map(report: dict) -> dict:
    return {name: payload for name, payload in report["issues"]}


def test_analyze_handles_accent_and_spacing(critic: StyleCritic) -> None:
    text = "Ceci peut etre un essai. C'est une phrase très très longue!!"
    report = critic.analyze(text)

    issues = _issue_map(report)
    issue_types = set(issues)
    assert "hedging_maybe" in issue_types
    assert "adverb_dup" in issue_types
    assert "excess_bang" in issue_types
    assert "copula_definition" in issue_types
    assert issues["adverb_dup"]["examples"], "Should expose matched adverbs"


def test_rewrite_normalises_variants(critic: StyleCritic) -> None:
    text = "Peut être que c'est très très bien !!"
    rewritten = critic.rewrite(text)

    assert "probablement" in rewritten.lower()
    assert "très très" not in rewritten.lower()
    assert "!!" not in rewritten


def test_expressive_signal_flags_text(critic: StyleCritic) -> None:
    text = "OMG!!! 😂😂"
    report = critic.analyze(text)

    issues = _issue_map(report)
    assert "expressive_noise" in issues
    assert issues["expressive_noise"] > 0


def test_signal_momentum_accumulates_pressure(critic: StyleCritic) -> None:
    text = "C'est VRAIMENT!!!! incroyable!!!!"

    first = critic.analyze(text)
    second = critic.analyze(text)

    first_issue = _issue_map(first)["excess_bang"]
    second_issue = _issue_map(second)["excess_bang"]

    assert second_issue > first_issue


def test_feedback_nudge_recalibrates_signal(critic: StyleCritic) -> None:
    text = "Peut être que c'est acceptable."

    before = critic.analyze(text)
    severity_before = _issue_map(before)["hedging_maybe"]

    critic.nudge("hedging_maybe", reward=1.5)

    after = critic.analyze(text)
    severity_after = _issue_map(after)["hedging_maybe"]

    assert severity_after < severity_before
