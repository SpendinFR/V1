from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from AGI_Evolutive.language.style_profiler import StyleProfiler


def test_accented_tokenization_preserves_words(tmp_path):
    persist_path = tmp_path / "style_profiles.json"
    profiler = StyleProfiler(persist_path=str(persist_path))

    tokens = profiler._simple_tokens("créateur très inspiré")

    assert "créateur" in tokens
    assert "cre" not in tokens
    assert "ateur" not in tokens

    profiler.observe("user", "Salut, je suis ton créateur préféré !")
    profile = profiler.style_of("user")

    assert "créateur" in profile.fav_lexicon
    assert "cre" not in profile.fav_lexicon
    assert "ateur" not in profile.fav_lexicon


def test_personal_name_and_memory_tracking(tmp_path):
    persist_path = tmp_path / "style_profiles.json"
    profiler = StyleProfiler(persist_path=str(persist_path))

    profiler.observe(
        "default",
        "Salut ! Je m'appelle william et j'adore les films de SF depuis toujours. "
        "Tu m'avais dit il y a 25 jours que tu voulais savoir ça.",
    )

    profile = profiler.style_of("default")

    assert profile.personal_names.get("William") == 1
    assert any(
        "j'adore les films de sf" in memory["text"].lower()
        for memory in profile.associative_memory
    )

    rewritten = profiler.rewrite_to_match("Merci pour ton retour.", "default")

    assert "William" in rewritten
    assert "Je me souviens" in rewritten
    assert "j'adore les films de sf" in rewritten.lower()

