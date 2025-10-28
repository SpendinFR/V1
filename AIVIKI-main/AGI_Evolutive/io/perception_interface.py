"""Interfaces d'entrée sensorielle pour l'architecture."""

from __future__ import annotations

import glob
import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Mapping

from AGI_Evolutive.utils.jsonsafe import json_sanitize
def _now() -> float:
    return time.time()


LOGGER = logging.getLogger(__name__)


def _llm_enabled() -> bool:
    return is_llm_enabled()


def _llm_manager():
    return get_llm_manager()


class PerceptionInterface:
    """Gestion centralisée des perceptions utilisateurs et fichiers inbox.

    L'implémentation fusionne deux versions historiques du module :
    - la version avancée qui journalise les événements, supporte le binding
      vers la mémoire, les émotions et la métacognition ;
    - la version minimaliste utilisée par certains modules externes qui
      attendent un constructeur recevant directement le *memory store* et des
      helpers simples comme :meth:`scan_inbox` ou
      :meth:`ingest_user_utterance`.

    Pour rester compatible avec les deux approches, le constructeur accepte
    un *memory store* optionnel en premier argument et expose à la fois les
    méthodes détaillées et leurs alias simplifiés.
    """

    def __init__(
        self,
        memory_store: Optional[Any] = None,
        inbox_dir: str = "inbox",
        path_log: str = "data/perception_log.jsonl",
        index_path: str = "data/perception_index.json",
    ) -> None:
        os.makedirs(os.path.dirname(path_log), exist_ok=True)
        os.makedirs(inbox_dir, exist_ok=True)

        self.inbox_dir = inbox_dir
        self.path_log = path_log
        self.index_path = index_path
        self.memory = memory_store
        self.arch = None

        self.bound: Dict[str, Any] = {
            "arch": None,
            "memory": memory_store,
            "metacog": None,
            "emotions": None,
            "language": None,
        }

        self._index = self._load_index()
        self.scan_interval = 3.0
        self._last_scan = 0.0
        self._last_llm_preanalysis: Optional[Mapping[str, Any]] = None

    # ------------------------------------------------------------------
    # Binding helpers
    def bind(
        self,
        arch: Any = None,
        memory: Any = None,
        metacog: Any = None,
        emotions: Any = None,
        language: Any = None,
    ) -> None:
        """Associe les différents sous-systèmes nécessaires."""

        if memory is not None:
            self.memory = memory
        if arch is not None:
            self.arch = arch
        self.bound.update(
            {
                "arch": arch,
                "memory": self.memory,
                "metacog": metacog,
                "emotions": emotions,
                "language": language,
            }
        )

    # ------------------------------------------------------------------
    # Inbox handling
    def step(self, force: bool = False) -> List[str]:
        """Appelé périodiquement par la boucle principale."""

        return self.scan_inbox(force=force)

    def scan_inbox(self, force: bool = False) -> List[str]:
        """Parcourt le dossier ``inbox`` et ingère les nouveaux fichiers."""

        now = time.time()
        if not force and (now - self._last_scan < self.scan_interval):
            return []
        self._last_scan = now
        return self._scan_inbox()

    def _scan_inbox(self) -> List[str]:
        files = glob.glob(os.path.join(self.inbox_dir, "*"))
        seen = set(self._index.get("seen_files", []))
        added: List[str] = []

        for path in files:
            if path in seen:
                continue
            if self._ingest_file(path):
                added.append(os.path.basename(path))
            seen.add(path)

        if added:
            self._index["seen_files"] = sorted(seen)
            self._save_index()

        return added

    def _llm_preprocess(self, text: str, meta: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
        if not _llm_enabled():
            return None

        payload = {"text": text, "metadata": dict(meta)}

        try:
            response = _llm_manager().call_dict(
                "perception_preprocess",
                input_payload=payload,
            )
        except (LLMUnavailableError, LLMIntegrationError):
            LOGGER.debug("LLM perception preprocessing unavailable", exc_info=True)
            return None

        if not isinstance(response, Mapping):
            return None

        self._last_llm_preanalysis = dict(response)
        return self._last_llm_preanalysis

    def _ingest_file(self, path: str) -> bool:
        memory = self.bound.get("memory")
        emotions = self.bound.get("emotions")

        meta = {
            "source": "inbox",
            "filename": os.path.basename(path),
            "size": os.path.getsize(path) if os.path.exists(path) else 0,
            "mtype": self._guess_mtype(path),
            "ingested_at": _now(),
            "id": str(uuid.uuid4()),
        }

        content_text = ""
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content_text = f.read()[:100000]
        except Exception:
            content_text = f"(binaire) {meta['filename']}"

        llm_analysis = self._llm_preprocess(content_text, meta)
        record = {"kind": "inbox_file", "meta": meta}
        if llm_analysis:
            record["llm"] = llm_analysis
            meta.setdefault("llm_preanalysis", llm_analysis)
        self._log(record)

        if memory and hasattr(memory, "add_memory"):
            try:
                memory.add_memory(
                    {
                        "kind": "perception_inbox",
                        "content": content_text,
                        "metadata": meta,
                        "llm": llm_analysis,
                    }
                )
            except Exception:
                pass

        if emotions and hasattr(emotions, "register_emotion_event"):
            try:
                emotions.register_emotion_event(
                    kind="perceived_input",
                    intensity=0.2,
                    arousal_hint=0.1,
                    meta=meta,
                )
            except Exception:
                pass

        return True

    def _guess_mtype(self, path: str) -> str:
        name = path.lower()
        if name.endswith((".txt", ".md", ".json", ".log")):
            return "text"
        if name.endswith((".png", ".jpg", ".jpeg", ".gif")):
            return "image"
        if name.endswith((".pdf",)):
            return "pdf"
        return "blob"

    # ------------------------------------------------------------------
    # Dialogue ingestion
    def ingest_user_message(
        self,
        text: str,
        speaker: str = "user",
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        record = {
            "kind": "dialogue_turn",
            "speaker": speaker,
            "text": text,
            "t": _now(),
            "meta": meta or {},
        }
        llm_analysis = self._llm_preprocess(text, record["meta"])
        if llm_analysis:
            record["llm"] = llm_analysis
            record["meta"].setdefault("llm_preanalysis", llm_analysis)
        self._log(record)

        memory = self.bound.get("memory")
        if memory and hasattr(memory, "add_memory"):
            try:
                memory.add_memory(
                    {
                        "kind": "dialogue_turn",
                        "content": text,
                        "metadata": {"speaker": speaker, **(meta or {})},
                        "llm": llm_analysis,
                    }
                )
            except Exception:
                pass

        try:
            from AGI_Evolutive.social.social_critic import SocialCritic
            from AGI_Evolutive.social.interaction_rule import ContextBuilder

            arch = self.arch or self.bound.get("arch")
            if arch and getattr(arch, "memory", None):
                critic = getattr(arch, "social_critic", None)
                if critic is None:
                    critic = SocialCritic(arch)
                    setattr(arch, "social_critic", critic)

                trace = None
                memory_obj = arch.memory
                finder = getattr(memory_obj, "find_recent", None)
                if callable(finder):
                    trace = finder(kind="decision_trace", since_sec=240)
                else:
                    recent: List[Dict[str, Any]] = []
                    getter = getattr(memory_obj, "get_recent_memories", None)
                    if callable(getter):
                        try:
                            recent = getter(200)
                        except TypeError:
                            recent = getter(n=200)
                    now_ts = time.time()
                    for item in reversed(recent):
                        if not isinstance(item, dict):
                            continue
                        if item.get("kind") != "decision_trace":
                            continue
                        ts = item.get("ts") or item.get("t") or item.get("timestamp")
                        if ts is None or now_ts - float(ts) <= 240:
                            trace = item
                            break

                if trace and trace.get("rule_id"):
                    post_ctx = ContextBuilder.build(arch, extra={
                        "pending_questions_count": len(
                            getattr(getattr(arch, "question_manager", None), "pending_questions", [])
                            or []
                        )
                    })
                    pre_ctx = trace.get("ctx_snapshot") or {}
                    outcome = critic.compute_outcome(
                        user_msg=text,
                        decision_trace=trace,
                        pre_ctx=pre_ctx,
                        post_ctx=post_ctx,
                    )
                    critic.update_rule_with_outcome(trace["rule_id"], outcome)
                    try:
                        from AGI_Evolutive.social.tactic_selector import TacticSelector

                        arch.tactic_selector = getattr(arch, "tactic_selector", TacticSelector(arch))
                        reward01 = float(outcome.get("reward", 0.5))
                        arch.tactic_selector.bandit_update(trace["rule_id"], pre_ctx, reward01)
                    except Exception:
                        pass
        except Exception:
            pass

        emotions = self.bound.get("emotions")
        if emotions and hasattr(emotions, "register_emotion_event"):
            try:
                emotions.register_emotion_event(
                    kind="dialogue_input",
                    intensity=0.25,
                    arousal_hint=0.15,
                    meta={"speaker": speaker},
                )
            except Exception:
                pass

    # Backwards compatible alias (anciens modules)
    def ingest_user_utterance(
        self, text: str, author: str = "user", meta: Optional[Dict[str, Any]] = None
    ) -> None:
        self.ingest_user_message(text, speaker=author, meta=meta)
        try:
            arch = self.arch or self.bound.get("arch")
            if arch and hasattr(arch, "style_observer"):
                arch.style_observer.observe_text(text, source="chat:user", channel="user")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Persistence helpers
    def _log(self, rec: Dict[str, Any]) -> None:
        rec["logged_at"] = _now()
        with open(self.path_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(json_sanitize(rec), ensure_ascii=False) + "\n")

    def _load_index(self) -> Dict[str, Any]:
        if not os.path.exists(self.index_path):
            return {"seen_files": []}
        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"seen_files": []}

    def _save_index(self) -> None:
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(json_sanitize(self._index), f, ensure_ascii=False, indent=2)
