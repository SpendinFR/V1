# 🚀 main.py - Point d'entrée AGI Évolutive
import glob
import json
import logging
import os
import re
import sys
import time
import traceback
import unicodedata
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from AGI_Evolutive.utils.llm_service import try_call_llm_dict, get_recent_llm_activity

logger = logging.getLogger(__name__)

try:
    from AGI_Evolutive.language.quote_memory import QuoteMemory  # type: ignore
except ImportError:  # pragma: no cover - module optionnel
    QuoteMemory = None  # type: ignore

try:
    from AGI_Evolutive.social.tactic_selector import TacticSelector  # type: ignore
except ImportError:  # pragma: no cover - module optionnel
    TacticSelector = None  # type: ignore

try:
    from AGI_Evolutive.language.ranker import RankerModel  # type: ignore
except ImportError:  # pragma: no cover - module optionnel
    RankerModel = None  # type: ignore

try:
    from AGI_Evolutive.language.inbox_ingest import ingest_inbox_paths  # type: ignore
except ImportError:  # pragma: no cover - module optionnel
    ingest_inbox_paths = None  # type: ignore

# --- Questions CLI helpers ---
def _get_qm(auto) -> Any:
    # Essaie plusieurs emplacements possibles
    return (
        getattr(auto, "question_manager", None)
        or getattr(auto, "questions", None)
        or getattr(getattr(auto, "arch", None), "question_manager", None)
        or getattr(getattr(auto, "arch", None), "questions", None)
    )


def _print_pending(
    qm,
    k: Optional[int] = 3,
    preset: Optional[List[Dict[str, Any]]] = None,
    *,
    channel: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Affiche les questions en attente et renvoie la liste affichée."""

    if preset is not None:
        pending = list(preset)
    elif not qm:
        return []
    else:
        pending = list(getattr(qm, "pending_questions", []))
    if not pending:
        return []

    def _chan(item: Dict[str, Any]) -> str:
        meta = item.get("meta") or {}
        return str(meta.get("channel") or "primary")

    if channel:
        pending = [q for q in pending if _chan(q) == channel]
        if not pending:
            return []

    pending.sort(key=lambda item: (item.get("meta", {}).get("queued_at", 0.0)))
    if k is None or k <= 0 or k >= len(pending):
        view = pending
    else:
        view = pending[-k:]

    print("\n— Questions en attente —")
    for i, q in enumerate(view, 1):
        qtype = q.get("type", "?")
        score = q.get("score", 0.0)
        text = q.get("text", "")
        chan = _chan(q)
        auto = " ⚙️" if (q.get("meta", {}) or {}).get("auto_suggestions") else ""
        print(f"[{i}] ({chan}/{qtype}, score={score:.2f}) {text}{auto}")
    print("Réponds avec : a <num> <ta réponse>   ex:  a 2 oui, c’était volontaire\n")
    return view




from AGI_Evolutive.core.autopilot import Autopilot, StageExecutionError
from AGI_Evolutive.core.cognitive_architecture import CognitiveArchitecture
from AGI_Evolutive.cognition.prioritizer import GoalPrioritizer
from AGI_Evolutive.orchestrator import Orchestrator
from AGI_Evolutive.language.voice import VoiceProfile
from AGI_Evolutive.language.lexicon import LiveLexicon
from AGI_Evolutive.language.style_observer import StyleObserver
from AGI_Evolutive.conversation.context import ContextBuilder
from AGI_Evolutive.language.renderer import LanguageRenderer
from AGI_Evolutive.language import OnlineNgramClassifier
from AGI_Evolutive.memory.concept_extractor import ConceptExtractor
from AGI_Evolutive.memory.prefs_bridge import PrefsBridge as PreferencesAdapter
from AGI_Evolutive.utils.logging_setup import configure_logging
from AGI_Evolutive.utils.llm_service import get_llm_manager

BANNER = """
╔══════════════════════════════════════════════╗
║            🧠  AGI ÉVOLUTIVE v1.0            ║
║  Architecture cognitive intégrée & évolutive ║
╚══════════════════════════════════════════════╝
"""
HELP_TEXT = """
Commandes disponibles :
  /help        → afficher cette aide
  /inbox       → liste les fichiers déposés dans ./inbox
  /save        → force une sauvegarde immédiate
  /state       → montre les infos d'état globales
  /llmstatus   → affiche l'état des dernières intégrations LLM
  /skillacquis [statut|oui/non <action>] → gère les compétences du bac à sable
  /quit        → quitte proprement
  q [channel]  → affiche toutes les questions en attente (channel=primary|immediate)
Astuce : déposez vos fichiers (.txt, .md, .json, etc.) dans ./inbox/
         ils seront intégrés automatiquement en mémoire.
"""


_FEEDBACK_STATE_PATH = os.path.join("data", "evolution", "cli_feedback_classifier.json")


def _normalize_feedback_text(text: str) -> str:
    base = unicodedata.normalize("NFKD", text or "")
    base = "".join(ch for ch in base if not unicodedata.combining(ch))
    base = base.replace("’", "'")
    base = base.lower()
    base = re.sub(r"[^a-z0-9+?!\s]", " ", base)
    return re.sub(r"\s+", " ", base).strip()


_POSITIVE_PATTERNS: Tuple[Tuple[str, re.Pattern], ...] = (
    ("plusplus", re.compile(r"\+\+")),
    ("merci", re.compile(r"\bmerci(?:\s+(?:beaucoup|bcp|d avance|infiniment))?\b")),
    ("parfait", re.compile(r"\b(?:parfait|parfaite|top|nickel|exactement|genial|super|au top)\b")),
    ("appreciation", re.compile(r"\b(?:jadore|jaime|japprecie|love|amazing|bravo)\b")),
    ("cest_bien", re.compile(r"\best\s+(?:un|une|le|la|l)\s+(?:bon|bonne|plaisir|delight|top)\b")),
)

_NEGATIVE_PATTERNS: Tuple[Tuple[str, re.Pattern], ...] = (
    ("trop_long", re.compile(r"\btrop\s+(?:long|lent|lourd)\b")),
    ("pas_clair", re.compile(r"\bpas\s+(?:clair|compr(?:is|ehensible)|terrible)\b")),
    ("bof", re.compile(r"\b(?:bof|mediocre|pas top|moyen)\b")),
    ("critique", re.compile(r"\b(?:naime pas|naimeplus|deteste|horrible)\b")),
    ("cest_nul", re.compile(r"\bc\s*est\s+(?:pas\s+)?(?:terrible|nul|faux|mauvais|decevant)\b")),
)

_POSITIVE_EMOJI_RE = re.compile(r"[👍👏🙏😊😁😄😍🤩❤♥️💖💙💚💛💜💗]")
_NEGATIVE_EMOJI_RE = re.compile(r"[😞😠😡☹🙁😤😒😕😔😩😫😢😭👎]")


def _detect_feedback_label(text: str) -> Tuple[Optional[str], Optional[str]]:
    normalized = _normalize_feedback_text(text)
    for name, pattern in _NEGATIVE_PATTERNS:
        if pattern.search(normalized):
            return "negative", name
    if _NEGATIVE_EMOJI_RE.search(text):
        return "negative", "emoji_negative"
    for name, pattern in _POSITIVE_PATTERNS:
        if pattern.search(normalized):
            return "positive", name
    if "++" in normalized:
        return "positive", "plusplus"
    if _POSITIVE_EMOJI_RE.search(text):
        return "positive", "emoji_positive"
    return None, None


class CLIAdaptiveFeedback:
    """Gère l'apprentissage en ligne du feedback utilisateur côté CLI."""

    def __init__(
        self,
        state_path: str = _FEEDBACK_STATE_PATH,
        min_confidence: float = 0.68,
    ) -> None:
        self.state_path = state_path
        self.min_confidence = float(min_confidence)
        self.classifier = OnlineNgramClassifier(labels=("positive", "negative"))
        try:
            self.evolution = EvolutionManager.shared()
        except Exception:
            self.evolution = None
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception:
            return
        state = payload.get("classifier") if isinstance(payload, dict) else None
        if isinstance(state, dict):
            try:
                self.classifier.from_state(state)
            except Exception:
                pass
        if isinstance(payload, dict):
            self.min_confidence = float(payload.get("min_confidence", self.min_confidence))

    def _persist(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            with open(self.state_path, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "classifier": self.classifier.to_state(),
                        "min_confidence": self.min_confidence,
                    },
                    fh,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception:
            pass

    def _record_event(
        self,
        *,
        label: str,
        success: Optional[bool],
        confidence: float,
        origin: str,
        heuristic: Optional[str] = None,
    ) -> None:
        if not self.evolution:
            return
        try:
            self.evolution.record_feedback_event(
                "cli_feedback",
                label=label,
                success=success,
                confidence=confidence,
                heuristic=heuristic,
                payload={"origin": origin},
            )
        except Exception:
            pass

    def register_ground_truth(self, text: str, label: str, pattern: Optional[str]) -> None:
        try:
            self.classifier.partial_fit(text, label)
            self._persist()
        except Exception:
            pass
        self._record_event(
            label=label,
            success=True,
            confidence=1.0,
            origin="heuristic",
            heuristic=pattern,
        )

    def classify(self, text: str) -> Optional[Dict[str, float]]:
        if not text.strip():
            return None
        try:
            label, confidence = self.classifier.predict(text)
        except Exception:
            return None
        if confidence >= self.min_confidence:
            return {"label": label, "confidence": float(confidence)}
        return None

    def analyze_with_llm(self, text: str) -> Optional[Dict[str, Any]]:
        if not text.strip():
            return None
        response = try_call_llm_dict(
            "cli_feedback",
            input_payload={"text": text},
            logger=logger,
        )
        if response:
            sentiment = str(response.get("sentiment") or "").strip() or "unknown"
            urgency = response.get("urgency")
            self._record_event(
                label=sentiment,
                success=None,
                confidence=1.0,
                origin="llm",
                heuristic=str(urgency) if isinstance(urgency, str) else None,
            )
        return response

    def record_prediction(self, label: str, confidence: float) -> None:
        self._record_event(
            label=label,
            success=None,
            confidence=confidence,
            origin="classifier",
            heuristic="classifier",
        )

def list_inbox(inbox_dir="inbox"):
    files = [os.path.basename(p) for p in glob.glob(os.path.join(inbox_dir, "*"))]
    if not files:
        print("📂 Inbox vide.")
    else:
        print("📁 Inbox :", ", ".join(files))

def run_cli():
    log_path = configure_logging()
    logger = logging.getLogger(__name__)

    print(BANNER)
    print(f"📝 Journaux: {log_path}")
    print("Chargement de l'architecture cognitive…")

    llm_auto_enabled = False
    if not os.getenv("AGI_DISABLE_LLM"):
        try:
            manager = get_llm_manager()
            if not manager.enabled:
                manager.set_enabled(True)
                llm_auto_enabled = True
        except Exception as exc:  # pragma: no cover - uniquement log utilisateur
            logger.warning(
                "Impossible d'activer automatiquement le LLM : %s", exc, exc_info=True
            )
    if llm_auto_enabled:
        logger.info("LLM activé automatiquement au démarrage de l'IA")
        print("🤖 Intégration LLM activée automatiquement.")

    logger.info("Démarrage de la CLI AGI Évolutive", extra={"log_path": str(log_path)})
    logger.info("Initialisation de l'architecture cognitive")
    try:
        arch = CognitiveArchitecture()
        arch.prioritizer = getattr(arch, "prioritizer", GoalPrioritizer(arch))
        from AGI_Evolutive.cognition.trigger_bus import TriggerBus  # déjà importable
        from AGI_Evolutive.cognition.evolution_manager import EvolutionManager

        try:
            bus = getattr(arch.prioritizer, "trigger_bus", None)
            if bus and hasattr(bus, "set_habit_strength_source"):
                bus.set_habit_strength_source(EvolutionManager.shared().habits_strength)
        except Exception:
            pass
        # --- bootstrap voix & contexte ---
        arch.voice_profile = getattr(
            arch,
            "voice_profile",
            VoiceProfile(arch.self_model, user_model=getattr(arch, "user_model", None)),
        )
        arch.lexicon = getattr(arch, "lexicon", LiveLexicon())
        arch.style_observer = getattr(
            arch,
            "style_observer",
            StyleObserver(
                arch.self_model,
                getattr(arch, "homeostasis", None),
                arch.voice_profile,
                arch.lexicon,
                user_model=getattr(arch, "user_model", None),
            ),
        )
        # Harmonise les instances utilisées par le renderer et l'observateur de style
        if getattr(arch.style_observer, "voice", None) is not None:
            arch.voice_profile = arch.style_observer.voice
        if getattr(arch.style_observer, "lex", None) is not None:
            arch.lexicon = arch.style_observer.lex
        arch.context_builder = getattr(arch, "context_builder", ContextBuilder(arch))
        arch.renderer = getattr(
            arch,
            "renderer",
            LanguageRenderer(arch.voice_profile, arch.lexicon),
        )
        if getattr(arch.renderer, "voice", None) is not arch.voice_profile:
            arch.renderer.voice = arch.voice_profile
        if getattr(arch.renderer, "lex", None) is not arch.lexicon:
            arch.renderer.lex = arch.lexicon
        # --- fin bootstrap ---

        # 10.1 — rattache les modules langue avancés si disponibles
        qm = None
        if QuoteMemory:
            try:
                qm = QuoteMemory()
            except Exception:
                qm = None
        if qm:
            if hasattr(arch, "voice_profile"):
                arch.voice_profile.quote_memory = qm
            else:
                arch.quote_memory = qm

        if TacticSelector and not hasattr(arch, "tactic_selector"):
            try:
                arch.tactic_selector = TacticSelector()
            except Exception:
                arch.tactic_selector = None

        if RankerModel and not hasattr(arch, "ranker"):
            try:
                arch.ranker = RankerModel()
            except Exception:
                arch.ranker = None

        if getattr(arch, "renderer", None) and getattr(arch, "ranker", None):
            try:
                arch.renderer.ranker = arch.ranker
                if hasattr(arch, "voice_profile"):
                    arch.renderer.voice = arch.voice_profile
                if hasattr(arch, "lexicon"):
                    arch.renderer.lex = arch.lexicon
            except Exception:
                pass

        # Optionnel : ingestion ciblée de l'inbox au démarrage
        # if ingest_inbox_paths:
        #     ingest_inbox_paths(["inbox/foo.txt", "inbox/bar.md"], arch=arch)

        orc = Orchestrator(arch)
        auto = Autopilot(arch, orchestrator=orc)
        logger.info(
            "Architecture cognitive initialisée",
            extra={
                "modules": ["autopilot", "orchestrator", "memory", "language"],
                "rag_active": bool(getattr(arch, "rag", None)),
            },
        )
    except Exception as e:
        print("❌ Erreur d'initialisation :", e)
        traceback.print_exc()
        sys.exit(1)

    voice = arch.voice_profile
    concept_extractor = ConceptExtractor(getattr(getattr(arch, "memory", None), "store", None))
    prefs = PreferencesAdapter(getattr(arch, "beliefs_graph", None))
    feedback_adapter = CLIAdaptiveFeedback()
    _last_cleanup_ts = 0.0

    print("✅ AGI initialisée. (Persistance & mémoire prêtes)")
    print(HELP_TEXT)
    print("🗨️  Démarrez la conversation ou tapez /help.")

    _last_view: List[Dict[str, Any]] = []
    _pending_cache: List[Dict[str, Any]] = []

    while True:
        try:
            # Affiche jusqu'à 3 questions en attente à chaque itération
            try:
                qm = _get_qm(auto)
                preset = _pending_cache if _pending_cache else None
                _last_view = _print_pending(qm, k=3, preset=preset)  # garde en mémoire locale
            except Exception:
                _last_view = []

            try:
                now = time.time()
                if now - _last_cleanup_ts > 24 * 3600:
                    mem_store = getattr(getattr(arch, "memory", None), "store", None) or getattr(arch, "memory", None)
                    from AGI_Evolutive.memory.janitor import run_once as memory_janitor_run

                    if mem_store:
                        memory_janitor_run(mem_store)
                    _last_cleanup_ts = now
            except Exception:
                pass

            msg = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n⏳ Sauvegarde avant sortie…")
            try:
                auto.save_now()
            except Exception as e:
                print("⚠️ Erreur lors de la sauvegarde :", e)
            print("👋 Fin de session.")
            logger.info("Session CLI terminée (interruption utilisateur)")
            break

        msg_lower = msg.lower()

        # --- Réponse à une question : "a <num> <réponse>" / "answer <num> ..."
        m = re.match(r"^\s*(a|answer|reponds?|réponds?)\s+(\d+)\s+(.+)$", msg, flags=re.IGNORECASE)
        if m:
            idx = max(0, int(m.group(2)) - 1)
            answer_text = m.group(3).strip()
            qm = _get_qm(auto)
            if not qm:
                print("⚠️  Aucun gestionnaire de questions accessible.")
                continue

            # Récupère la vue actuelle (ou replie sur les pending)
            view = _last_view if _last_view else list(getattr(qm, "pending_questions", []))[-3:]
            if not view or idx >= len(view):
                print("⚠️  Index hors limites. Tape 'q' pour lister.")
                continue

            q = view[idx]
            qid = q.get("id") or q.get("qid") or q.get("uuid")

            # 1) Ingestion de ta réponse comme utterance utilisateur (traçabilité)
            try:
                # Préfère ton interface de perception si dispo
                per = getattr(getattr(auto, "arch", None), "perception", None)
                meta = {"answer_to": qid, "question_text": q.get("text", ""), "ts": time.time()}
                if per and hasattr(per, "ingest_user_utterance"):
                    per.ingest_user_utterance(answer_text, author="user", meta=meta)
                else:
                    memory = getattr(getattr(auto, "arch", None), "memory", None)
                    if memory and hasattr(memory, "add_memory"):
                        memory.add_memory(
                            {
                                "kind": "user_answer",
                                "q_id": qid,
                                "q_text": q.get("text", ""),
                                "text": answer_text,
                                "ts": time.time(),
                            }
                        )
            except Exception:
                pass

            # 2) Notifie le QuestionManager si une API existe
            updated = False
            for meth in ("record_answer", "resolve_question", "set_answer"):
                if hasattr(qm, meth):
                    try:
                        getattr(qm, meth)(qid, answer_text)
                        updated = True
                        break
                    except Exception:
                        pass

            # 3) Fallback : on retire manuellement la question de la file
            if not updated:
                try:
                    pend = getattr(qm, "pending_questions", [])
                    # enlève la 1re occurrence correspondante
                    for i in range(len(pend) - 1, -1, -1):
                        if (pend[i].get("id") or pend[i].get("qid")) == qid or pend[i] is q:
                            pend.pop(i)
                            break
                except Exception:
                    pass

            try:
                if _pending_cache:
                    _pending_cache = [
                        item
                        for item in _pending_cache
                        if not (
                            (item.get("id") or item.get("qid") or item.get("uuid")) == qid
                            or item is q
                        )
                    ]
            except Exception:
                pass

            print(f"✅  Réponse enregistrée pour [{idx+1}] : {answer_text}")
            # laisse la boucle continuer (l’abduction/planification la prendra au prochain tick)
            continue
        # --- fin: réponse NL ---

        if not msg:
            continue

        # ==== COMMANDES ====
        if msg in ("/quit", "/exit"):
            print("💾 Sauvegarde finale…")
            auto.save_now()
            print("👋 À bientôt.")
            break

        elif msg == "/help":
            print(HELP_TEXT)
            continue

        elif (
            msg_lower in {"q", "/q", "/questions"}
            or msg_lower.startswith("q ")
            or msg_lower.startswith("/q ")
            or msg_lower.startswith("/questions ")
        ):
            qm = _get_qm(auto)
            if not qm:
                print("⚠️  Aucun gestionnaire de questions accessible.")
                continue
            pending_all = list(getattr(qm, "pending_questions", []))
            if not pending_all:
                print("ℹ️  Aucune question en attente.")
                _last_view = []
                _pending_cache = []
                continue
            parts = msg.strip().split()
            channel = None
            if len(parts) >= 2:
                arg = parts[1].lower()
                if arg.startswith("prim"):
                    channel = "primary"
                elif arg.startswith("imm"):
                    channel = "immediate"
            view = _print_pending(qm, k=None, preset=pending_all, channel=channel)
            if not view:
                label = f" pour le canal {channel}" if channel else ""
                print(f"ℹ️  Aucune question{label}.")
                continue
            _last_view = list(view)
            _pending_cache = list(view)
            if channel:
                print(f"Total questions ({channel}) : {len(view)}")
            else:
                print(f"Total questions : {len(view)}")
            blocked_fn = getattr(qm, "blocked_channels", None)
            if callable(blocked_fn):
                blocked = list(blocked_fn())
                if blocked:
                    print("⛔ Blocage actif sur : " + ", ".join(blocked))
            continue

        elif msg_lower in {"/llmstatus", "/llm", "llmstatus", "llm"}:
            manager = get_llm_manager()
            enabled_flag = getattr(manager, "enabled", False)
            print(f"LLM activé : {'oui' if enabled_flag else 'non'}")
            records = get_recent_llm_activity(15)
            if not records:
                print("ℹ️  Aucun appel LLM enregistré depuis le lancement.")
                continue

            print("\n— Historique récent des intégrations LLM —")
            icon_map = {
                "success": "✅",
                "error": "⚠️",
                "disabled": "⏸️",
            }
            for record in records:
                icon = icon_map.get(record.status.lower(), "•")
                try:
                    ts = datetime.fromtimestamp(record.timestamp).strftime("%H:%M:%S")
                except Exception:
                    ts = "?"
                line = f"  {icon} [{ts}] {record.spec_key} → {record.status}"
                if record.message:
                    message = record.message.strip()
                    if len(message) > 120:
                        message = message[:117] + "..."
                    line += f" · {message}"
                print(line)
            continue

        elif msg_lower.startswith("/skillacquis"):
            skills = getattr(arch, "skill_sandbox", None)
            if not (skills and hasattr(skills, "list_skills")):
                print("⚠️  Bac à sable de compétences indisponible.")
                continue

            parts = msg.strip().split()
            yes_tokens = {"oui", "yes", "y", "approve", "ok", "daccord", "d'accord"}
            no_tokens = {"non", "no", "n", "reject", "refus", "refuser"}

            if len(parts) >= 3 and parts[1].lower() in yes_tokens.union(no_tokens):
                decision_word = parts[1].lower()
                decision = "approve" if decision_word in yes_tokens else "reject"
                action_type = parts[2]
                notes = " ".join(parts[3:]).strip()
                interface = getattr(arch, "action_interface", None)
                if not (interface and hasattr(interface, "execute")):
                    print("⚠️  Interface d'action indisponible pour enregistrer la décision.")
                    continue
                payload = {
                    "type": "review_skill_candidate",
                    "payload": {
                        "action_type": action_type,
                        "decision": decision,
                        "reviewer": "cli",
                    },
                    "context": {"user": "cli"},
                }
                if notes:
                    payload["payload"]["notes"] = notes
                result = interface.execute(payload)
                if result.get("ok"):
                    skill_payload = result.get("skill") or {}
                    status = skill_payload.get("status") or result.get("status") or decision
                    print(f"✅ Décision enregistrée pour {action_type} → {status}")
                else:
                    reason = result.get("reason") or result.get("error") or "erreur inconnue"
                    print(f"⚠️  Impossible de traiter la décision : {reason}")
                continue

            if len(parts) == 2 and parts[1].lower() in yes_tokens.union(no_tokens):
                print("ℹ️  Syntaxe : /skillacquis oui|non <action_type> [notes]")
                continue

            status_aliases = {
                "attente": "awaiting_approval",
                "approval": "awaiting_approval",
                "awaiting": "awaiting_approval",
                "pending": "pending",
                "entrainement": "training",
                "entraînement": "training",
                "training": "training",
                "active": "active",
                "actives": "active",
                "actif": "active",
                "acquises": "active",
                "échec": "failed",
                "echec": "failed",
                "failed": "failed",
                "rejetées": "rejected",
                "rejetee": "rejected",
                "rejete": "rejected",
                "rejected": "rejected",
            }

            label_map = {
                "awaiting_approval": "🕒 Compétences en attente d'approbation",
                "training": "🔁 Compétences en entraînement",
                "pending": "📝 Compétences nouvellement demandées",
                "failed": "⚠️  Compétences en échec",
                "rejected": "🚫 Compétences rejetées",
                "active": "✅ Compétences actives",
            }

            status_filter = None
            if len(parts) >= 2:
                token = parts[1].lower()
                if token in status_aliases:
                    status_filter = status_aliases[token]
                elif token in {"all", "tout", "toutes", "*"}:
                    status_filter = None
                else:
                    print("ℹ️  Utilise /skillacquis [attente|active|failed|all] ou /skillacquis oui/non <action>.")
                    continue

            def _print_entries(entries, header, show_trials=False):
                if not entries:
                    return False
                print(f"\n{header} ({len(entries)}) :")
                for item in entries:
                    action_type = item.get("action_type")
                    status = item.get("status")
                    success_rate = item.get("success_rate") or 0.0
                    attempts = item.get("attempts", 0)
                    requirements = ", ".join(item.get("requirements", [])[:4]) or "(aucune)"
                    created_ts = item.get("created_at") or 0.0
                    try:
                        created_dt = datetime.fromtimestamp(created_ts)
                        created_text = created_dt.strftime("%Y-%m-%d %H:%M")
                    except Exception:
                        created_text = "n/a"
                    line = (
                        f"  - {action_type} · statut={status} · succès={success_rate:.2f} · "
                        f"essais={attempts} · créé={created_text}"
                    )
                    print(line)
                    print(f"      exigences: {requirements}")
                    if show_trials:
                        trials = item.get("trials", [])
                        if trials:
                            best_cov = max(float(t.get("coverage", 0.0)) for t in trials)
                            successes = sum(1 for t in trials if t.get("success"))
                            print(
                                f"      essais réussis: {successes}/{len(trials)} (couverture max {best_cov:.2f})"
                            )
                            for trial in trials:
                                summary = (trial.get("summary") or "").strip()
                                if not summary:
                                    continue
                                first_line = summary.splitlines()[0].strip()
                                if len(first_line) > 110:
                                    first_line = first_line[:107] + "..."
                                mode = trial.get("mode") or "essai"
                                print(
                                    f"        • essai {trial.get('index', '?')} [{mode}] → {first_line}"
                                )
                return True

            if status_filter is not None:
                entries = skills.list_skills(status=status_filter, include_trials=True)
                header = label_map.get(status_filter, status_filter)
                if not _print_entries(entries, header, show_trials=True):
                    print(f"ℹ️  Aucune compétence pour le statut '{header}'.")
                elif status_filter == "awaiting_approval":
                    print("\n🤖 Utilise /skillacquis oui|non <action_type> pour décider.")
                continue

            shown = False
            for code in [
                "awaiting_approval",
                "training",
                "pending",
                "failed",
                "rejected",
                "active",
            ]:
                entries = skills.list_skills(status=code, include_trials=(code == "awaiting_approval"))
                header = label_map.get(code, code)
                printed = _print_entries(entries, header, show_trials=(code == "awaiting_approval"))
                shown = shown or printed

            if not shown:
                print("ℹ️  Aucune compétence suivie pour le moment.")
            else:
                print("\n🤖 Utilise /skillacquis oui|non <action_type> pour approuver ou refuser une compétence.")
            continue

        elif msg == "/inbox":
            list_inbox()
            continue

        elif msg == "/save":
            path = auto.save_now()
            print(f"💾 Snapshot sauvegardé : {path}")
            continue

        elif msg == "/state":
            try:
                t = time.strftime("%H:%M:%S", time.localtime())
                total_mem = getattr(arch.memory, "memory_metadata", {}).get("total_memories", 0)
                print(f"🧩 Heure locale: {t}")
                print(f"🧠 Mémoires stockées : {total_mem}")
                print(f"⚙️  Dernière sauvegarde : {time.strftime('%H:%M:%S', time.localtime(auto.persist._last_save))}")
            except Exception as e:
                print("⚠️ Impossible d'afficher l'état :", e)
            continue

        # ==== INTERACTION ====
        feedback_label: Optional[str] = None
        feedback_confidence = 0.0
        feedback_meta: Dict[str, Any] = {}
        llm_label: Optional[str] = None
        llm_confidence = 0.0

        llm_analysis = feedback_adapter.analyze_with_llm(msg)
        if llm_analysis:
            feedback_meta = dict(llm_analysis)
            sentiment_norm = _normalize_feedback_text(str(llm_analysis.get("sentiment") or ""))
            if sentiment_norm:
                if any(word in sentiment_norm for word in ("positif", "satisfait", "content", "heureux", "ravi", "merci")):
                    llm_label = "positive"
                elif any(word in sentiment_norm for word in ("negatif", "mecontent", "insatisfait", "colere", "frustre", "plainte", "critique")):
                    llm_label = "negative"
            if llm_label:
                llm_confidence = 0.7
                feedback_label = llm_label
                feedback_confidence = llm_confidence

        heur_label, heur_pattern = _detect_feedback_label(msg)
        if heur_label:
            feedback_label = heur_label
            feedback_confidence = 1.0
            feedback_adapter.register_ground_truth(msg, heur_label, heur_pattern)
        else:
            prediction = feedback_adapter.classify(msg)
            if prediction:
                feedback_label = prediction["label"]
                feedback_confidence = prediction["confidence"]
                feedback_adapter.record_prediction(prediction["label"], prediction["confidence"])

        if llm_label:
            if feedback_label is None:
                feedback_label = llm_label
                feedback_confidence = llm_confidence
            elif feedback_label == llm_label:
                feedback_confidence = max(feedback_confidence, max(llm_confidence, 0.75))
            else:
                feedback_confidence *= 0.6

        urgency_level = str(feedback_meta.get("urgency") or "").lower()
        if feedback_label and urgency_level == "haut":
            feedback_confidence = max(feedback_confidence, 0.85)
        elif feedback_label and urgency_level == "bas":
            feedback_confidence = max(0.2, feedback_confidence * 0.9)

        if feedback_label == "positive":
            voice.update_from_feedback(msg, positive=True)
        elif feedback_label == "negative":
            voice.update_from_feedback(msg, positive=False)

        try:
            pos = feedback_label == "positive"
            neg = feedback_label == "negative"
            if pos or neg:
                sign = 1 if pos else -1
                raw_concepts = concept_extractor._extract_concepts(msg) or []
                targets = [str(c).strip().lower() for c in raw_concepts if c and len(str(c)) >= 3][:5]
                if targets:
                    evidence_id = f"user:{int(time.time())}"
                    base_strength = 1.0 if sign > 0 else 0.8
                    strength = max(0.3, base_strength * (0.6 + 0.4 * feedback_confidence))
                    for c in targets:
                        prefs.observe_feedback(
                            concept=c,
                            sign=sign,
                            evidence_id=evidence_id,
                            strength=strength,
                        )

                selector = getattr(arch, "tactic_selector", None)
                if selector and hasattr(selector, "feedback"):
                    try:
                        arm = getattr(arch, "_last_macro", None)
                        reward = feedback_confidence if pos else -feedback_confidence
                        if reward > 0:
                            selector.feedback(reward, arm=arm)
                        elif reward < 0:
                            selector.feedback(reward, arm=arm)
                    except Exception:
                        pass

                qm_for_feedback = None
                if hasattr(arch, "voice_profile"):
                    qm_for_feedback = getattr(arch.voice_profile, "quote_memory", None)
                if qm_for_feedback is None:
                    qm_for_feedback = getattr(arch, "quote_memory", None)
                if qm_for_feedback:
                    try:
                        reward = feedback_confidence if pos else -feedback_confidence
                        if reward:
                            qm_for_feedback.reward_last(reward)
                            qm_for_feedback.save()
                    except Exception:
                        pass

                pack = getattr(arch, "_last_candidates", None)
                if pack and isinstance(pack, dict) and pack.get("alts"):
                    try:
                        ctx_for_rank = {}
                        vp = getattr(arch, "voice_profile", None)
                        style_policy = getattr(vp, "style_policy", None)
                        if style_policy and hasattr(style_policy, "params"):
                            ctx_for_rank["style"] = style_policy.params
                        ranker = getattr(arch, "ranker", None)
                        if ranker and hasattr(ranker, "update_pair"):
                            winner = (pack.get("chosen") or {}).get("text") or pack.get("text")
                            alts = pack.get("alts") or []
                            loser = None
                            if alts:
                                loser = (alts[0] or {}).get("text") if isinstance(alts[0], dict) else None
                            if winner and loser:
                                lr_scale = max(0.2, min(1.0, feedback_confidence))
                                if pos:
                                    ranker.update_pair(ctx_for_rank, winner, loser, lr=0.15 * lr_scale)
                                    if hasattr(ranker, "save"):
                                        ranker.save()
                                elif neg:
                                    ranker.update_pair(ctx_for_rank, loser, winner, lr=0.10 * lr_scale)
                                    if hasattr(ranker, "save"):
                                        ranker.save()
                    except Exception:
                        pass
        except Exception:
            pass

        assistant_text_override: Optional[str] = None
        final_pack_override: Optional[Dict[str, Any]] = None
        selected_macro_override = None

        if msg_lower.startswith("j'aime") and "inbox/" in msg_lower:
            import re as _re
            m = _re.search(r"(?:\"([^\"]+)\"|'([^']+)')", msg)
            if m:
                voice.update_from_liked_source(m.group(1) or m.group(2))

            paths = _re.findall(r"(inbox\/[^\s]+)", msg)
            if paths and ingest_inbox_paths:
                try:
                    added = ingest_inbox_paths(paths, arch=arch)
                except Exception:
                    added = 0
                assistant_text_override = (
                    f"Bien reçu : j’ai intégré {added} source(s) de l’inbox et capté des formules réutilisables."
                )
                final_pack_override = {
                    "text": assistant_text_override,
                    "chosen": {"text": assistant_text_override},
                    "alts": [],
                }

        fallback_reply = (
            "Je n'ai pas compris cette demande. Peux-tu la reformuler ou donner un exemple ?"
        )

        if assistant_text_override is None:
            try:
                assistant_text_brut = auto.step(user_msg=msg)
            except StageExecutionError as e:
                print("⚠️ Erreur durant le cycle :", e)
                assistant_text_brut = {"text": fallback_reply}
            except Exception as e:
                print("⚠️ Erreur durant le cycle :", e)
                traceback.print_exc()
                assistant_text_brut = {"text": fallback_reply}
        else:
            assistant_text_brut = None

        semantic_source = (
            assistant_text_override if assistant_text_override is not None else assistant_text_brut
        )

        reply = None
        final_pack: Optional[Dict[str, Any]] = final_pack_override
        selected_macro = selected_macro_override
        try:
            try:
                ctx = arch.context_builder.build(msg)
            except Exception:
                ctx = {"last_message": msg}

            ctx.setdefault("last_user_msg", msg)
            if isinstance(semantic_source, dict):
                trace = semantic_source.get("reasoning_trace")
                if trace and "reasoning_trace" not in ctx:
                    ctx["reasoning_trace"] = trace

            macro_selector = getattr(arch, "tactic_selector", None)
            if selected_macro is None and macro_selector and hasattr(macro_selector, "pick"):
                try:
                    selected_macro = macro_selector.pick(context=ctx)
                except Exception:
                    selected_macro = None

            generated_points: List[str] = []
            if isinstance(semantic_source, dict):
                bullets = semantic_source.get("bullets") if isinstance(semantic_source, dict) else None
                if isinstance(bullets, list):
                    generated_points = [str(b).strip() for b in bullets if str(b).strip()]
                else:
                    text = semantic_source.get("text") or semantic_source.get("raw")
                    if text:
                        generated_points = [s.strip() for s in str(text).split("\n") if s.strip()]
            elif isinstance(semantic_source, list):
                generated_points = [str(item).strip() for item in semantic_source if str(item).strip()]
            elif semantic_source is not None:
                generated_points = [
                    line.strip()
                    for line in str(semantic_source or "").split("\n")
                    if line.strip()
                ] or [str(semantic_source or "").strip()]

            plan_text = ""
            if isinstance(semantic_source, str):
                plan_text = semantic_source
            elif isinstance(semantic_source, dict):
                primary = semantic_source.get("text") or semantic_source.get("raw")
                if isinstance(primary, str):
                    plan_text = primary
            if not plan_text and generated_points:
                plan_text = "\n".join(generated_points)
            elif not plan_text and semantic_source is not None:
                plan_text = str(semantic_source)

            plan = {"title": "", "bullets": generated_points, "text": plan_text, "raw": semantic_source}

            renderer = getattr(arch, "renderer", None)
            if assistant_text_override is None and renderer and hasattr(renderer, "render_final"):
                try:
                    final_pack = renderer.render_final(ctx, plan)
                    reply = (final_pack or {}).get("text")
                except Exception:
                    final_pack = final_pack_override

            if reply is None and renderer is not None and (
                assistant_text_override is None or assistant_text_brut is not None
            ):
                sem = {"text": plan_text, "raw": semantic_source}
                reply = arch.renderer.render_reply(sem, ctx)
                if reply is not None and final_pack is None:
                    final_pack = {
                        "text": reply,
                        "chosen": {"text": reply},
                        "alts": [],
                    }

            if reply is None and assistant_text_override is not None:
                reply = assistant_text_override
                if final_pack is None:
                    final_pack = {
                        "text": reply,
                        "chosen": {"text": reply},
                        "alts": [],
                    }

            if reply is None and assistant_text_brut is not None:
                if isinstance(assistant_text_brut, dict):
                    candidate = assistant_text_brut.get("text") or assistant_text_brut.get("raw")
                    if isinstance(candidate, str) and candidate.strip():
                        reply = candidate.strip()
                    else:
                        reply = str(assistant_text_brut)
                else:
                    reply = str(assistant_text_brut)
                if final_pack is None and isinstance(reply, str):
                    final_pack = {
                        "text": reply,
                        "chosen": {"text": reply},
                        "alts": [],
                    }

            if isinstance(reply, str):
                stripped = reply.strip()
                if not stripped:
                    reply = fallback_reply
                else:
                    reply = stripped
            elif reply is None:
                reply = fallback_reply

            if reply is not None:
                print(reply)
        except Exception as e:
            print("⚠️ Erreur lors du rendu :", e)
            traceback.print_exc()
            continue

        if reply is not None:
            if final_pack:
                try:
                    final_pack["text"] = reply
                    chosen = final_pack.get("chosen") or {}
                    if not isinstance(chosen, dict):
                        chosen = {"text": reply}
                    else:
                        chosen = dict(chosen)
                        chosen["text"] = reply
                    final_pack["chosen"] = chosen
                    if not isinstance(final_pack.get("alts"), list):
                        final_pack["alts"] = []
                except Exception:
                    final_pack = {
                        "text": reply,
                        "chosen": {"text": reply},
                        "alts": [],
                    }
            else:
                final_pack = {
                    "text": reply,
                    "chosen": {"text": reply},
                    "alts": [],
                }

            arch._last_candidates = final_pack
            arch._last_macro = selected_macro

            try:
                auto.arch.memory.add_memory({"kind": "interaction", "role": "assistant", "text": reply})
                arch.lexicon.add_from_text(reply, liked=False)
                arch.lexicon.save()
            except Exception:
                pass

        # ==== QUESTIONS PROACTIVES ====
        questions = auto.pending_questions()
        if questions:
            _pending_cache = list(questions)
        for q in questions:
            print("❓", q["text"])
    logger.info("Session CLI terminée")

if __name__ == "__main__":
    run_cli()
