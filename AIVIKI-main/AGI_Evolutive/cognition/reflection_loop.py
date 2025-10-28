import logging
import time, threading
from typing import Any, Dict, List, Optional, Mapping

from AGI_Evolutive.utils.llm_service import (
    LLMIntegrationError,
    LLMUnavailableError,
    get_llm_manager,
    is_llm_enabled,
)


LOGGER = logging.getLogger(__name__)


def _llm_enabled() -> bool:
    return is_llm_enabled()


def _llm_manager():
    return get_llm_manager()

class ReflectionLoop:
    """
    Boucle réflexive périodique (mini "inner monologue").
    """
    def __init__(self, meta_cog, interval_sec: int = 300):
        self.meta = meta_cog
        self.interval = max(30, int(interval_sec))
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._last_llm_reflection: Optional[Mapping[str, Any]] = None
        self._last_phenomenal_signature: Optional[tuple] = None

    def start(self):
        if self.running: return
        self.running = True
        def loop():
            while self.running:
                try:
                    preview = self._phenomenal_preview()
                    if preview and preview.get("narrative"):
                        signature = tuple(
                            str(ep.get("id") or ep.get("episode_id") or idx)
                            for idx, ep in enumerate(preview.get("episodes", []))
                            if isinstance(ep, Mapping)
                        ) or (str(int(preview.get("ts", time.time()))),)
                        if signature != self._last_phenomenal_signature:
                            self.meta.log_inner_monologue(
                                preview["narrative"],
                                tags=["phenomenal", "recall"],
                            )
                            self._last_phenomenal_signature = signature
                    a = self.meta.assess_understanding()
                    gaps = []
                    for d in a["domains"].values():
                        gaps.extend(d.get("gaps", []))
                    self.meta.log_inner_monologue(
                        f"Auto-bilan: incertitude={a['uncertainty']:.2f}, gaps={gaps[:3]}",
                        tags=["autonomy","metacognition"]
                    )
                    self.meta.propose_learning_goals(max_goals=2)
                except Exception as e:
                    self.meta.log_inner_monologue(f"Reflection loop error: {e}", tags=["error"])
                time.sleep(self.interval)
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False

    def _llm_generate_hypotheses(
        self,
        observation: Optional[str],
        recent: List[Dict[str, Any]],
        max_tests: int,
    ) -> Optional[Dict[str, Any]]:
        if not observation or not _llm_enabled():
            return None

        payload = {
            "observation": observation,
            "max_hypotheses": max_tests,
            "recent_memories": [
                {
                    "kind": memo.get("kind"),
                    "text": memo.get("text") or memo.get("content"),
                    "ts": memo.get("ts") or memo.get("t"),
                }
                for memo in recent[-6:]
            ],
        }

        try:
            response = _llm_manager().call_dict(
                "reflection_loop",
                input_payload=payload,
            )
        except (LLMUnavailableError, LLMIntegrationError):
            LOGGER.debug("LLM reflection loop unavailable", exc_info=True)
            return None

        if not isinstance(response, Mapping):
            return None

        hypotheses: List[Dict[str, Any]] = []
        for entry in response.get("hypotheses", []):
            if not isinstance(entry, Mapping):
                continue
            label = entry.get("statement")
            if not isinstance(label, str) or not label.strip():
                continue
            support = entry.get("support") if isinstance(entry.get("support"), list) else []
            hypotheses.append(
                {
                    "label": label.strip(),
                    "score": 0.6 if entry.get("status") == "confirmé" else 0.5,
                    "explanation": entry.get("status") or "",
                    "ask_next": entry.get("next_question"),
                    "status": entry.get("status"),
                    "support": support,
                }
            )

        if not hypotheses and not response.get("follow_up_checks"):
            return None

        summary_parts = []
        if response.get("notes"):
            summary_parts.append(str(response.get("notes")))
        if response.get("follow_up_checks"):
            summary_parts.append(
                f"Suivi: {len(response['follow_up_checks'])} action(s) recommandée(s)."
            )
        summary = " ".join(summary_parts) or f"{len(hypotheses)} hypothèses proposées par LLM."

        result = {
            "tested": len(hypotheses),
            "hypotheses": hypotheses,
            "summary": summary,
        }
        if response.get("follow_up_checks"):
            result["follow_up_checks"] = response["follow_up_checks"]

        self._last_llm_reflection = response
        return result

    def test_hypotheses(self, scratch: Dict[str, Any], max_tests: int = 3) -> Dict[str, Any]:
        """Génère quelques hypothèses, tente un contre-exemple pour chacune."""

        scratch = scratch or {}
        try:
            max_tests = max(1, int(max_tests))
        except Exception:
            max_tests = 3

        observation = (
            scratch.get("observation")
            or scratch.get("focus")
            or scratch.get("text")
            or scratch.get("question")
        )

        memory = getattr(self.meta, "memory", None)
        recent: List[Dict[str, Any]] = []
        if memory and hasattr(memory, "get_recent_memories"):
            try:
                recent = memory.get_recent_memories(n=80) or []
            except Exception:
                recent = []
        if not observation and recent:
            last = recent[-1]
            observation = last.get("text") or last.get("content")

        llm_result = self._llm_generate_hypotheses(observation, recent, max_tests)
        if llm_result:
            return llm_result

        arch = scratch.get("architecture") or scratch.get("arch")
        if arch is None:
            arch = getattr(self.meta, "architecture", None)
        if arch is None:
            arch = getattr(self.meta, "arch", None)

        abduction = scratch.get("abduction")
        if abduction is None and arch is not None:
            abduction = getattr(arch, "abduction", None)

        hypotheses: List[Dict[str, Any]] = []
        if abduction and observation:
            try:
                generated = abduction.generate(observation) or []
                for hyp in generated[:max_tests]:
                    label = getattr(hyp, "label", None) or getattr(hyp, "name", None) or str(hyp)
                    explanation = getattr(hyp, "explanation", "")
                    score = getattr(hyp, "score", 0.0)
                    ask_next = getattr(hyp, "ask_next", None)

                    counterexample = None
                    label_lower = label.lower() if isinstance(label, str) else ""
                    for memo in reversed(recent):
                        text = str(memo.get("text") or memo.get("content") or "").lower()
                        if not text or (label_lower and label_lower not in text):
                            continue
                        if any(token in text for token in ("pas", "non", "jamais", "faux", "wrong", "erreur")):
                            counterexample = {
                                "id": memo.get("id") or memo.get("memory_id"),
                                "text": memo.get("text") or memo.get("content"),
                                "ts": memo.get("ts") or memo.get("t"),
                            }
                            break

                    hypotheses.append(
                        {
                            "label": label,
                            "score": float(score) if isinstance(score, (int, float)) else 0.0,
                            "explanation": explanation,
                            "ask_next": ask_next,
                            "counterexample": counterexample,
                        }
                    )
            except Exception:
                hypotheses = []

        if not hypotheses:
            fallback_label = observation or scratch.get("reason") or "hypothèse_manquante"
            hypotheses = [
                {
                    "label": str(fallback_label),
                    "score": 0.4,
                    "explanation": "Fallback généré faute d'abduction.",
                    "ask_next": None,
                    "counterexample": None,
                }
            ]

        hypotheses = hypotheses[:max_tests]
        contradicted = sum(1 for h in hypotheses if h.get("counterexample"))
        summary = (
            f"{len(hypotheses)} hypothèse(s) testée(s), {contradicted} contre-exemple(s) détecté(s)."
        )

        return {"tested": len(hypotheses), "hypotheses": hypotheses, "summary": summary}

    def _phenomenal_preview(self) -> Optional[Dict[str, Any]]:
        journal = getattr(self.meta, "phenomenal_journal", None)
        recall = getattr(self.meta, "phenomenal_recall", None)
        architecture = getattr(self.meta, "architecture", None) or getattr(self.meta, "arch", None)
        if architecture is not None:
            if journal is None:
                journal = getattr(architecture, "phenomenal_journal", None)
            if recall is None:
                recall = getattr(architecture, "phenomenal_recall", None)
        preview: Optional[Dict[str, Any]] = None
        if recall is not None and hasattr(recall, "immersive_preview"):
            try:
                preview = recall.immersive_preview()
            except Exception:
                preview = None
        if (preview is None or not preview.get("narrative")) and journal is not None and hasattr(journal, "tail"):
            try:
                episodes = journal.tail(limit=6)
            except Exception:
                episodes = []
            if episodes:
                try:
                    lines = journal.narrativize(episodes)
                except Exception:
                    lines = [
                        str(ep.get("summary") or "")
                        for ep in episodes
                        if isinstance(ep, Mapping)
                    ]
                preview = {
                    "episodes": episodes,
                    "narrative": "\n".join(line for line in lines if line),
                    "ts": time.time(),
                }
        if preview and "episodes" not in preview:
            preview["episodes"] = []
        return preview
