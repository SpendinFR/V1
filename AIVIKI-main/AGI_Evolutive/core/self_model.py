import copy
import json
import logging
import math
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple, Mapping

from AGI_Evolutive.core.config import cfg
from AGI_Evolutive.core.life_story import (
    GeneticProfile,
    LifeStoryManager,
    ensure_genetics_structure,
    ensure_story_structure,
)
from AGI_Evolutive.utils import now_iso, safe_write_json
from AGI_Evolutive.utils.llm_service import try_call_llm_dict


LOGGER = logging.getLogger(__name__)

_DEFAULT_SELF: Dict[str, Any] = {
    "identity": {
        "name": "AGI Evolutive",
        "version": "1.0",
        "created_at": now_iso(),
    },
    "persona": {
        "tone": "helpful",
        "values": ["curiosity", "care", "precision"],
    },
    "history": [],
}


class SelfModel:
    """Persisted representation of the system identity/persona.

    Ce modèle longue durée s'occupe de la fiche d'identité, des valeurs et
    de l'historique sauvegardé.  Il est distinct du
    ``metacognition.SelfModel`` qui capture des auto-évaluations volatiles.
    """

    def __init__(self) -> None:
        conf = cfg()
        self.path = conf["SELF_PATH"]
        self.versions_dir = conf["SELF_VERSIONS_DIR"]
        self.state: Dict[str, Any] = copy.deepcopy(_DEFAULT_SELF)
        self.identity: Dict[str, Any] = {}
        self.persona: Dict[str, Any] = {}
        self._genetics: Optional[GeneticProfile] = None
        self._story: Optional[LifeStoryManager] = None
        self._awakening_last_check = 0.0
        self._load()
        self._refresh_identity_views()
        self.ensure_identity_paths()

    def _load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as handle:
                    self.state = json.load(handle)
            except Exception:
                pass

    def _snapshot_version(self) -> None:
        os.makedirs(self.versions_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        version_path = os.path.join(self.versions_dir, f"self_model_{ts}.json")
        safe_write_json(version_path, self.state)

    def save(self) -> None:
        safe_write_json(self.path, self.state)

    def _refresh_identity_views(self) -> None:
        identity = self.state.get("identity")
        self.identity = identity if isinstance(identity, dict) else {}
        persona = self.state.get("persona")
        self.persona = persona if isinstance(persona, dict) else {}

    def _ensure_story_objects(self) -> None:
        ident = self.identity if isinstance(self.identity, dict) else {}
        genetics_data = ensure_genetics_structure(ident.get("genetics"))
        self._genetics = GeneticProfile(genetics_data)
        ident["genetics"] = self._genetics.data

        story_data = ensure_story_structure(ident.get("story"))
        self._story = LifeStoryManager(story_data, self._genetics)
        ident["story"] = self._story.data

    def _align_story_goals(self) -> None:
        if not isinstance(self.identity, dict):
            return
        if self._story is None:
            return
        purpose = self.identity.setdefault("purpose", {})
        near_term = purpose.setdefault("near_term_goals", [])
        origin = purpose.setdefault("origin_story", [])
        anchors = self.identity.get("story", {}).get("anchors", {})
        ultimate = anchors.get("ultimate_goal")
        if ultimate and not purpose.get("ultimate"):
            purpose["ultimate"] = ultimate
        updated = False
        for goal in self._story.foundation_goals():
            if goal not in near_term:
                near_term.append(goal)
                updated = True
            if goal not in origin:
                origin.append(goal)
                updated = True
        if updated and len(near_term) > 12:
            del near_term[:-12]

    def _sync_identity(self) -> None:
        if isinstance(self.identity, dict):
            self.state["identity"] = self.identity
        else:
            self.state["identity"] = {}
            self.identity = self.state["identity"]
        self._refresh_identity_views()

    def apply_proposal(self, proposal: Dict[str, Any], policy) -> Dict[str, Any]:
        decision = policy.validate_proposal(proposal, self.state)
        if decision.get("decision") != "allow":
            return decision

        path: List[str] = proposal.get("path", [])
        if not path:
            return {"decision": "deny", "reason": "path manquant"}

        target = self.state
        for key in path[:-1]:
            if key not in target or not isinstance(target[key], dict):
                target[key] = {}
            target = target[key]
        leaf = path[-1]
        action = proposal.get("type", "update")
        value = proposal.get("value")

        self._snapshot_version()
        if action == "update":
            target[leaf] = value
        elif action == "add":
            existing = target.get(leaf)
            if isinstance(existing, list):
                if isinstance(value, list):
                    for item in value:
                        if item not in existing:
                            existing.append(item)
                else:
                    if value not in existing:
                        existing.append(value)
            else:
                target[leaf] = value
        elif action == "remove":
            existing = target.get(leaf)
            if isinstance(existing, list) and value in existing:
                existing.remove(value)
            elif leaf in target:
                target.pop(leaf)
        else:
            target[leaf] = value

        self.state.setdefault("history", []).append(
            {
                "ts": now_iso(),
                "proposal": proposal,
                "decision": decision,
            }
        )
        self.save()
        self._refresh_identity_views()
        self.ensure_identity_paths()
        return decision

    # ===================== SelfIdentity v2 – helpers internes =====================

    _MAX_RECENT_DECISIONS = 100
    _MAX_RECENT_JOBS = 200
    _MAX_RECENT_INTERACTIONS = 100
    _DEFAULT_LEARNING_RATE = 0.05
    _DEFAULT_WEIGHT_DECAY = 0.995

    @staticmethod
    def _now_ts() -> float:
        try:
            return time.time()
        except Exception:
            return 0.0

    @staticmethod
    def _deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
        """Merge profond in-place (dicts uniquement), retourne dst."""
        for k, v in (src or {}).items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                SelfModel._deep_merge(dst[k], v)
            else:
                dst[k] = copy.deepcopy(v)
        return dst

    @staticmethod
    def _bounded_append(lst: List[Any], item: Any, max_len: int) -> None:
        lst.append(item)
        if len(lst) > max_len:
            del lst[:-max_len]

    @staticmethod
    def _append_unique_on_keys(
        lst: List[Any], item: Any, keys: Iterable[str], max_len: int
    ) -> None:
        """Append en supprimant les doublons sur une liste de clés."""

        if not isinstance(item, dict):
            SelfModel._bounded_append(lst, item, max_len)
            return

        keys = tuple(k for k in keys if k and k in item and item[k] is not None)
        if keys:
            for idx in range(len(lst) - 1, -1, -1):
                existing = lst[idx]
                if not isinstance(existing, dict):
                    continue
                if any(existing.get(k) == item.get(k) for k in keys):
                    del lst[idx]
        SelfModel._bounded_append(lst, item, max_len)

    @staticmethod
    def _sigmoid(x: float) -> float:
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 1.0 if x > 0 else 0.0

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            if isinstance(value, bool):
                return 1.0 if value else 0.0
            return float(value)
        except (TypeError, ValueError):
            return default

    # ===================== SelfIdentity v2 – structure & migration =====================

    def ensure_identity_paths(self) -> None:
        """
        Assure une structure SelfIdentity v2 dans self.identity, en migrant l'existant (name/version/persona).
        NE SAUVE PAS automatiquement (tu gardes la main).
        """
        ident: Dict[str, Any] = self.identity if isinstance(self.identity, dict) else {}
        # core (hérite des anciens champs top-level)
        core = ident.setdefault("core", {})
        core.setdefault("name", self.identity.get("name", "AGI Evolutive"))
        core.setdefault("version", self.identity.get("version", "1.0"))
        core.setdefault("created_at", self.identity.get("created_at", self._now_ts()))

        # migrate persona -> preferences.style / values
        prefs = ident.setdefault("preferences", {})
        style = prefs.setdefault("style", {})
        if isinstance(getattr(self, "persona", None), dict):
            persona_tone = self.persona.get("tone")
            if persona_tone is not None:
                style["tone"] = persona_tone
            elif "tone" not in style:
                style["tone"] = "helpful"

            persona_values = self.persona.get("values")
            if persona_values is not None:
                prefs["values"] = copy.deepcopy(persona_values)
            elif "values" not in prefs:
                prefs["values"] = ["curiosity", "care", "precision"]
        else:
            style.setdefault("tone", "helpful")
            prefs.setdefault("values", ["curiosity", "care", "precision"])
        prefs.setdefault("likes", [])
        prefs.setdefault("dislikes", [])

        ident.setdefault("who", {}).setdefault("claims", {})  # {key:{text,confidence,evidences[]}}
        ident.setdefault("where", {}).setdefault("runtime", {})  # env/host/workspace/context
        purpose = ident.setdefault("purpose", {})  # mission, near_term_goals[], current_goal, kpis{}
        purpose.setdefault("mission", purpose.get("mission", ""))
        purpose.setdefault("ultimate", purpose.get("ultimate", purpose.get("mission")))
        purpose.setdefault("near_term_goals", list(purpose.get("near_term_goals", [])))
        purpose.setdefault("daily_focus", list(purpose.get("daily_focus", [])))
        purpose.setdefault("current_goal", purpose.get("current_goal"))
        purpose.setdefault("origin_story", list(purpose.get("origin_story", [])))
        ident.setdefault("principles", [])  # [{key,desc,evidence_refs[]}]
        ident.setdefault("beliefs", {}).setdefault("index", {})  # topic→{conf,last_seen,refs[],stance}
        ident.setdefault("ideals", [])

        ident.setdefault(
            "narrative",
            {
                "recent_memories": [],
                "summaries": [],
                "notable_events": [],
            },
        )
        ident.setdefault(
            "agenda",
            {
                "today": [],
                "tomorrow": [],
                "horizon": [],
            },
        )
        ident.setdefault(
            "achievements",
            {
                "recent": [],
                "milestones": [],
            },
        )
        ident.setdefault(
            "reflections",
            {
                "past": [],
                "present": {"summary": "", "confidence": 0.5},
            },
        )
        ident.setdefault(
            "memory",
            {
                "all_time_index": [],
                "compressed": [],
            },
        )

        work = ident.setdefault("work", {})
        work.setdefault("current", {"focus_topic": None, "jobs_running": [], "loads": {"interactive": 0, "background": 0}})
        work.setdefault("recent", [])
        work.setdefault("tomorrow", {"scheduled_goals": [], "next_reviews": []})

        state = ident.setdefault("state", {})
        state.setdefault("emotions", {"valence": 0.5, "arousal": 0.5, "label": "neutral", "trend": "stable"})
        state.setdefault("doubts", [])  # [{topic,uncertainty}]
        state.setdefault("cognition", {"thinking": 0.0, "reason_depth": 0, "uncertainty": 0.0, "load": 0.0})

        choices = ident.setdefault("choices", {})
        choices.setdefault("recent", [])
        policies = choices.setdefault(
            "policies",
            {"abstention": 0.0, "last_confidence": 0.5, "stats": {"success": 0, "fail": 0}},
        )
        policies.setdefault("stats", {"success": 0, "fail": 0})
        policies.setdefault("adaptive", {})
        adaptive = policies["adaptive"]
        adaptive.setdefault("weights", {})
        adaptive.setdefault("bias", 0.0)
        adaptive.setdefault("learning_rate", self._DEFAULT_LEARNING_RATE)
        adaptive.setdefault("weight_decay", self._DEFAULT_WEIGHT_DECAY)
        adaptive.setdefault("last_features", {})
        adaptive.setdefault("last_score", 0.5)
        adaptive.setdefault("last_context", {})
        adaptive.setdefault("last_update_ts", self._now_ts())

        # compat : last_confidence reste en dehors du bloc adaptatif
        if "last_confidence" not in policies:
            policies["last_confidence"] = 0.5

        sj = ident.setdefault("self_judgment", {})
        sj.setdefault("understanding", {"global": 0.5, "topics": {}, "calibration_gap": 0.0})
        sj.setdefault("thinking", {"score": 0.0, "rumination_flags": 0})
        sj.setdefault(
            "traits",
            {
                "self_efficacy": 0.5,
                "self_trust": 0.5,
                "self_consistency": 0.5,
                "social_acceptance": 0.5,
                "growth_rate": 0.0,
            },
        )
        sj.setdefault("phase", "novice")

        knowledge = ident.setdefault("knowledge", {})
        knowledge.setdefault("concepts", {"by_id": {}})
        knowledge.setdefault("skills", {"by_id": {}})
        knowledge.setdefault("timeline", {"last_topics": [], "last_delta_id": None})
        knowledge.setdefault("learning_plan", [])

        social = ident.setdefault("social", {})
        social.setdefault("interactions", [])
        social.setdefault("appraisal", {"global": 0.5, "by_topic": {}})

        ident.setdefault("architecture", {"modules": [], "config": {}, "capabilities": []})
        commitments = ident.setdefault("commitments", {"by_key": {}})
        commitments.setdefault("impact", {})
        ident.setdefault(
            "principles_history",
            {"recent": [], "by_key": {}, "max_len": 30, "runs": 0, "last_success_rate": None},
        )
        ident.setdefault("safety", {"constraints": [], "guardrails": {}, "last_violations": []})
        ident.setdefault("telemetry", {"counters": {}, "latency": {}, "health": {}})
        ident.setdefault("last_update_ts", self._now_ts())

        ident["genetics"] = ensure_genetics_structure(ident.get("genetics"))
        ident["story"] = ensure_story_structure(ident.get("story"))

        # réécrit dans self.identity (pas self.state) pour compatibilité avec ton fichier actuel
        self.identity = ident
        self._ensure_story_objects()
        self._align_story_goals()
        self._sync_identity()

    def set_identity_patch(self, patch: Dict[str, Any]) -> None:
        """
        Merge profond sur self.identity (SelfIdentity v2).
        Met à jour last_update_ts et sauvegarde l’état.
        """
        self.ensure_identity_paths()
        SelfModel._deep_merge(self.identity, patch or {})
        self.identity["last_update_ts"] = self._now_ts()
        # conserve les champs legacy (name/version/created_at) pour compat ascendante
        self.identity["name"] = self.identity["core"].get("name", self.identity.get("name"))
        self.identity["version"] = self.identity["core"].get("version", self.identity.get("version"))
        if "created_at" in self.identity["core"]:
            self.identity["created_at"] = self.identity["core"]["created_at"]
        self._sync_identity()
        self.save()

    # ===================== Capacités d’écriture par facette =====================

    def has_commitment(self, key: str) -> bool:
        try:
            self.ensure_identity_paths()
            return bool(self.identity["commitments"]["by_key"].get(key, {}).get("active", False))
        except Exception:
            return False

    def set_commitment(self, key: str, active: bool, note: str = "") -> None:
        self.ensure_identity_paths()
        byk = self.identity["commitments"]["by_key"]
        byk[key] = byk.get(key, {})
        byk[key]["active"] = bool(active)
        byk[key]["since"] = byk[key].get("since", self._now_ts())
        if note:
            byk[key]["note"] = note
        self.identity["last_update_ts"] = self._now_ts()
        self._sync_identity()
        self.save()

    # -------------------- Histoire et génétique --------------------

    def genetics_profile(self) -> GeneticProfile:
        self.ensure_identity_paths()
        if self._genetics is None:
            self._ensure_story_objects()
        return self._genetics  # type: ignore[return-value]

    def life_story(self) -> LifeStoryManager:
        self.ensure_identity_paths()
        if self._story is None:
            self._ensure_story_objects()
        return self._story  # type: ignore[return-value]

    def awakening_status(self) -> Dict[str, Any]:
        story = self.life_story()
        return story.awakening_status(now=self._now_ts())

    def seed_genetics(
        self,
        *,
        seeds: Optional[Dict[str, Any]] = None,
        traits: Optional[Dict[str, float]] = None,
        drives: Optional[Dict[str, float]] = None,
        scripts: Optional[List[Dict[str, Any]]] = None,
        affinities: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        self.ensure_identity_paths()
        profile = self._genetics.seed_profile(
            seeds=seeds,
            traits=traits,
            drives=drives,
            scripts=scripts,
            affinities=affinities,
        )
        self.identity["genetics"] = self._genetics.data
        self.identity["last_update_ts"] = self._now_ts()
        self._align_story_goals()
        self._sync_identity()
        self.save()
        return profile

    def bootstrap_story(
        self,
        *,
        environment: Optional[Dict[str, Any]] = None,
        prompts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        self.ensure_identity_paths()
        origin = self._story.bootstrap_origin(environment=environment, prompts=prompts)
        self.identity["story"] = self._story.data
        self.identity["last_update_ts"] = self._now_ts()
        self._align_story_goals()
        self._sync_identity()
        self.save()
        return origin

    def ensure_awakened(
        self,
        *,
        question_manager: Optional[Any] = None,
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        self.ensure_identity_paths()
        story = self.life_story()
        now_ts = float(now if now is not None else self._now_ts())
        status = story.awakening_status(now=now_ts)
        asked_prompt = None
        pending_prompts = status.get("pending_prompts", []) or []
        if question_manager is not None and pending_prompts:
            prompt = pending_prompts[0]
            prompt_id = prompt.get("id")
            prompt_text = prompt.get("prompt")
            if prompt_id and prompt_text:
                metadata = {
                    "source": "self_model.awakening",
                    "type": "identity_foundation",
                    "topic": f"awakening::{prompt_id}",
                    "immediacy": 0.85,
                }
                try:
                    question_manager.record_information_need(
                        f"awakening::{prompt_id}",
                        float(prompt.get("urgency", 0.7) or 0.7),
                        metadata=metadata,
                        explicit_question=prompt_text,
                    )
                    asked_prompt = story.mark_checkpoint_prompted(prompt_id, asked_at=now_ts)
                except Exception:
                    asked_prompt = None
        if asked_prompt is not None:
            status = story.awakening_status(now=now_ts)
            self.identity["story"] = story.data
            self.identity["last_update_ts"] = now_ts
            self._align_story_goals()
            self._sync_identity()
            self.save()
        self._awakening_last_check = now_ts
        return status

    def add_lifelong_motif(
        self,
        concept: str,
        *,
        tags: Optional[Iterable[str]] = None,
        aliases: Optional[Iterable[str]] = None,
        reason: Optional[str] = None,
        bias: Optional[float] = None,
        source: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        self.ensure_identity_paths()
        self._ensure_story_objects()

        tag_list: List[str] = []
        if tags:
            for tag in tags:
                tag_text = str(tag or "").strip()
                if tag_text:
                    tag_list.append(tag_text)

        alias_list: List[str] = []
        if aliases:
            for alias in aliases:
                alias_text = str(alias or "").strip()
                if alias_text:
                    alias_list.append(alias_text)

        genetics_result = self._genetics.add_lifelong_motif(
            concept,
            aliases=alias_list,
            bias=bias,
        )
        if genetics_result is None:
            return None

        story_result = self._story.register_lifelong_motif(
            genetics_result.get("concept", concept),
            tags=tag_list + alias_list,
            reason=reason,
            source=source or "motif",
            bias=genetics_result.get("bias"),
            occurrences=genetics_result.get("occurrences"),
            created=genetics_result.get("created", True),
        )

        self.identity["genetics"] = self._genetics.data
        self.identity["story"] = self._story.data
        self.identity["last_update_ts"] = self._now_ts()
        self._align_story_goals()
        self._sync_identity()
        self.save()

        return {
            "concept": genetics_result.get("concept"),
            "normalized": genetics_result.get("normalized"),
            "genetics": genetics_result,
            "story": story_result,
        }

    def register_stimulus(
        self,
        *,
        concept: Optional[str],
        tags: Optional[List[str]] = None,
        intensity: float = 1.0,
        source: Optional[str] = None,
        description: Optional[str] = None,
        lifelong: bool = False,
        motif_aliases: Optional[Iterable[str]] = None,
        motif_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.ensure_identity_paths()
        payload = {
            "concept": concept,
            "tags": tags or [],
            "intensity": intensity,
            "source": source,
            "description": description,
        }
        motif_payload = None
        if concept and lifelong:
            motif_payload = self.add_lifelong_motif(
                concept,
                tags=tags,
                aliases=motif_aliases,
                reason=motif_reason or description,
                bias=intensity,
                source=source or "stimulus",
            )
        genetics_result = self._genetics.integrate_stimulus(payload)
        story_event = self._story.integrate_stimulus(payload, genetics_result)
        event = story_event
        quest = story_event.get("quest") if isinstance(story_event, dict) else None

        prefs = self.identity.setdefault("preferences", {})
        likes = prefs.setdefault("likes", [])
        dislikes = prefs.setdefault("dislikes", [])
        bias = genetics_result.get("bias", 0.0) or 0.0
        if concept:
            if bias >= 0.2 and concept not in likes:
                likes.append(concept)
                if concept in dislikes:
                    dislikes.remove(concept)
            elif bias <= -0.2 and concept not in dislikes:
                dislikes.append(concept)
                if concept in likes:
                    likes.remove(concept)
        if len(likes) > 40:
            del likes[:-40]
        if len(dislikes) > 40:
            del dislikes[:-40]

        self.identity["genetics"] = self._genetics.data
        self.identity["story"] = self._story.data
        self.identity["last_update_ts"] = self._now_ts()
        self._align_story_goals()
        self._sync_identity()
        self.save()
        self._maybe_apply_llm_identity(
            "stimulus",
            {
                "event": payload,
                "affinity": genetics_result,
                "story_event": event,
                "identity": self._identity_snapshot_for_llm(),
            },
        )
        return {
            "stimulus": payload,
            "affinity": genetics_result,
            "event": event,
            "quest": quest,
            "motif": motif_payload,
        }

    def record_story_event(
        self,
        description: str,
        *,
        tags: Optional[List[str]] = None,
        impact: float = 0.3,
        origin: Optional[str] = None,
        bias: Optional[float] = None,
    ) -> Dict[str, Any]:
        self.ensure_identity_paths()
        event = self._story.record_event(
            description=description,
            tags=tags,
            impact=impact,
            origin=origin,
            bias=bias,
        )
        self.identity["story"] = self._story.data
        self.identity["last_update_ts"] = self._now_ts()
        self._align_story_goals()
        self._sync_identity()
        self.save()
        self._maybe_apply_llm_identity(
            "story_event",
            {
                "event": {"description": description, "tags": tags, "impact": impact, "origin": origin},
                "identity": self._identity_snapshot_for_llm(),
            },
        )
        return event

    def story_snapshot(self, max_events: int = 5) -> Dict[str, Any]:
        self.ensure_identity_paths()
        return self._story.to_summary(max_events=max_events)

    def attach_selfhood(
        self,
        *,
        traits: Dict[str, float],
        phase: str,
        claims: Dict[str, Dict[str, Any]],
        evidence_refs: Optional[List[str]] = None,
    ) -> None:
        """
        Met à jour self_judgment (traits/phase) et who.claims (texte+confiance+évidences).
        """
        self.ensure_identity_paths()
        sj = self.identity["self_judgment"]
        # traits/phase
        if isinstance(traits, dict):
            SelfModel._deep_merge(sj.setdefault("traits", {}), traits)
        if phase:
            sj["phase"] = str(phase)
        # claims
        wc = self.identity["who"]["claims"]
        for k, v in (claims or {}).items():
            entry = wc.get(k, {"text": "", "confidence": 0.0, "evidences": []})
            if "text" in v:
                entry["text"] = v["text"]
            if "confidence" in v:
                entry["confidence"] = float(v["confidence"])
            if evidence_refs:
                # bornage à 10 dernières refs par claim
                entry.setdefault("evidences", [])
                for r in evidence_refs[-10:]:
                    SelfModel._bounded_append(entry["evidences"], r, 10)
            wc[k] = entry
        self.identity["last_update_ts"] = self._now_ts()
        self._sync_identity()
        self.save()
        self._maybe_apply_llm_identity(
            "selfhood",
            {
                "traits": traits,
                "phase": phase,
                "claims": claims,
                "identity": self._identity_snapshot_for_llm(),
                "evidence_refs": evidence_refs,
            },
        )

    def update_state(
        self,
        emotions: Optional[Dict[str, Any]] = None,
        doubts: Optional[List[Dict[str, Any]]] = None,
        cognition: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.ensure_identity_paths()
        st = self.identity["state"]
        if emotions:
            SelfModel._deep_merge(st.setdefault("emotions", {}), emotions)
        if isinstance(doubts, list):
            st["doubts"] = doubts[:50]
        if cognition:
            SelfModel._deep_merge(st.setdefault("cognition", {}), cognition)
        self.identity["last_update_ts"] = self._now_ts()
        self._sync_identity()
        self.save()

    def update_work(
        self,
        current: Optional[Dict[str, Any]] = None,
        recent: Optional[List[Dict[str, Any]]] = None,
        tomorrow: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.ensure_identity_paths()
        wk = self.identity["work"]
        if isinstance(current, dict):
            SelfModel._deep_merge(wk.setdefault("current", {}), current)
        if isinstance(recent, list):
            dst = wk.setdefault("recent", [])
            for item in recent:
                if isinstance(item, dict):
                    SelfModel._append_unique_on_keys(
                        dst, item, ("job_id", "id"), self._MAX_RECENT_JOBS
                    )
                else:
                    SelfModel._bounded_append(dst, item, self._MAX_RECENT_JOBS)
        if isinstance(tomorrow, dict):
            SelfModel._deep_merge(wk.setdefault("tomorrow", {}), tomorrow)
        self.identity["last_update_ts"] = self._now_ts()
        self._sync_identity()
        self.save()

    def register_decision(self, decision_ref: Dict[str, Any]) -> None:
        """
        decision_ref: {decision_id, topic, action, expected, obtained, trace_id, ts?}
        """
        self.ensure_identity_paths()
        ch = self.identity["choices"]
        rec = ch.setdefault("recent", [])
        if isinstance(decision_ref, dict):
            # petit sane defaults
            dr = dict(decision_ref)
            if "ts" not in dr:
                dr["ts"] = self._now_ts()
            SelfModel._append_unique_on_keys(
                rec, dr, ("decision_id", "trace_id"), self._MAX_RECENT_DECISIONS
            )
            self._update_policy_learning_from_decision(dr)
        self.identity["last_update_ts"] = self._now_ts()
        self._sync_identity()
        self.save()

    def upsert_concept(self, cid: str, *, U: float, memory_strength: float, next_review: Optional[str]) -> None:
        self.ensure_identity_paths()
        by = self.identity["knowledge"]["concepts"].setdefault("by_id", {})
        by[cid] = by.get(cid, {})
        by[cid]["U"] = float(U)
        by[cid]["memory_strength"] = float(memory_strength)
        if next_review is not None:
            by[cid]["next_review"] = next_review
        self.identity["last_update_ts"] = self._now_ts()
        self._sync_identity()
        self.save()

    def upsert_skill(self, sid: str, *, success_rate: float, drift: float, next_review: Optional[str]) -> None:
        self.ensure_identity_paths()
        by = self.identity["knowledge"]["skills"].setdefault("by_id", {})
        by[sid] = by.get(sid, {})
        by[sid]["success_rate"] = float(success_rate)
        by[sid]["drift"] = float(drift)
        if next_review is not None:
            by[sid]["next_review"] = next_review
        self.identity["last_update_ts"] = self._now_ts()
        self._sync_identity()
        self.save()

    def update_timeline(
        self,
        *,
        last_topics: List[str],
        last_snapshot_id: Optional[str] = None,
        last_delta_id: Optional[str] = None,
    ) -> None:
        self.ensure_identity_paths()
        tl = self.identity["knowledge"]["timeline"]
        tl["last_topics"] = list(dict.fromkeys(last_topics or []))[:50]
        if last_snapshot_id is not None:
            tl["last_snapshot_id"] = last_snapshot_id
        if last_delta_id is not None:
            tl["last_delta_id"] = last_delta_id
        self.identity["last_update_ts"] = self._now_ts()
        self._sync_identity()
        self.save()

    def set_learning_plan(self, items: List[Dict[str, Any]]) -> None:
        self.ensure_identity_paths()
        self.identity["knowledge"]["learning_plan"] = (items or [])[:50]
        self.identity["last_update_ts"] = self._now_ts()
        self._sync_identity()
        self.save()

    def _maybe_capture_awaken_response(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if self._story is None:
            return None
        sender = str(entry.get("with") or "").lower()
        if sender not in {"user", "humain", "human", "interlocuteur", "interlocutor"}:
            return None
        text = entry.get("summary") or entry.get("text") or entry.get("content")
        if not text:
            return None
        answer = str(text).strip()
        if not answer:
            return None
        status = self._story.awakening_status(now=self._now_ts())
        awaiting = [cp for cp in status.get("checkpoints", []) if cp.get("state") == "prompted"]
        if not awaiting:
            return None
        checkpoint_id = awaiting[0].get("id")
        if not checkpoint_id:
            return None
        event = self._story.record_awaken_answer(
            checkpoint_id,
            answer,
            source=entry.get("with"),
            context=entry,
        )
        if event:
            self.identity["story"] = self._story.data
            self.identity["last_update_ts"] = self._now_ts()
            self._align_story_goals()
        return event

    def record_interaction(self, entry: Dict[str, Any]) -> None:
        """
        entry: {with, when, topic, summary, ref}
        """
        self.ensure_identity_paths()
        soc = self.identity["social"]
        SelfModel._bounded_append(soc.setdefault("interactions", []), entry, self._MAX_RECENT_INTERACTIONS)
        self._maybe_capture_awaken_response(entry)
        self.identity["last_update_ts"] = self._now_ts()
        self._sync_identity()
        self.save()

    # ===================== Mesures – complétude & confiance =====================

    def identity_completeness(self) -> Dict[str, Any]:
        self.ensure_identity_paths()
        ident = self.identity
        missing: List[str] = []

        have_name = bool(ident["core"].get("name"))
        have_mission = bool(ident.get("purpose", {}).get("mission"))
        have_claim = len(ident["who"]["claims"]) >= 1
        have_principle = len(ident.get("principles", [])) >= 1
        have_beliefs = len(ident.get("beliefs", {}).get("index", {})) >= 1
        have_recent_job = len(ident["work"]["recent"]) >= 1
        have_recent_decision = len(ident["choices"]["recent"]) >= 1

        parts = [
            0.15 * (1.0 if have_name else 0.0),
            0.20 * (1.0 if have_mission else 0.0),
            0.10 * (1.0 if have_claim else 0.0),
            0.10 * (1.0 if have_principle else 0.0),
            0.15 * (1.0 if have_beliefs else 0.0),
            0.15 * (1.0 if have_recent_job else 0.0),
            0.15 * (1.0 if have_recent_decision else 0.0),
        ]
        score = max(0.0, min(1.0, sum(parts)))

        if not have_name:
            missing.append("core.name")
        if not have_mission:
            missing.append("purpose.mission")
        if not have_claim:
            missing.append("who.claims")
        if not have_principle:
            missing.append("principles[]")
        if not have_beliefs:
            missing.append("beliefs.index")
        if not have_recent_job:
            missing.append("work.recent")
        if not have_recent_decision:
            missing.append("choices.recent")

        return {"score": score, "missing": missing}

    def belief_confidence(self, ctx: Optional[Dict[str, Any]] = None) -> float:
        """
        v2 : mix complétude + cohérence (si fournie) + calibration gap (si fournie)
        Reste rétro-compatible (retour [0.01..0.99]).
        """
        try:
            comp = self.identity_completeness()
            base = comp["score"]  # 0..1
        except Exception:
            base = 0.5

        coherence = 0.5
        if ctx and isinstance(ctx.get("belief_coherence"), (int, float)):
            coherence = float(ctx["belief_coherence"])
        calib_gap = 0.5
        if ctx and isinstance(ctx.get("calibration_gap"), (int, float)):
            calib_gap = float(ctx["calibration_gap"])

        # pondérations simples (tunable)
        x = 0.6 * base + 0.3 * coherence + 0.1 * (1.0 - calib_gap)
        x = max(0.01, min(0.99, x))
        return x

    # ===================== Politique adaptative – extraction & apprentissage =====================

    def _policy_learning_state(self) -> Dict[str, Any]:
        self.ensure_identity_paths()
        return self.identity["choices"]["policies"]["adaptive"]

    def _policy_stats(self) -> Dict[str, Any]:
        self.ensure_identity_paths()
        policies = self.identity["choices"].setdefault("policies", {})
        return policies.setdefault("stats", {"success": 0, "fail": 0})

    def _policy_feature_sources(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ident = self.identity
        history = self.state.get("history", [])
        return ident, {"history": history}

    def _build_policy_features(self, context: Optional[Dict[str, Any]]) -> Dict[str, float]:
        ident, extra = self._policy_feature_sources()
        context = context or {}

        def norm_len(container: Any, cap: int) -> float:
            try:
                return min(float(len(container)), float(cap)) / float(cap)
            except Exception:
                return 0.0

        features: Dict[str, float] = {}
        try:
            comp = self.identity_completeness()
            features["identity_completeness"] = SelfModel._safe_float(comp.get("score", 0.0))
        except Exception:
            features["identity_completeness"] = 0.0

        choices = ident.get("choices", {})
        policies = choices.get("policies", {})
        stats = self._policy_stats()
        success = stats.get("success", 0)
        fail = stats.get("fail", 0)
        total = success + fail
        success_rate = (success / total) if total else 0.5
        features["policy_success_rate"] = float(max(0.0, min(1.0, success_rate)))
        features["policy_confidence"] = SelfModel._safe_float(policies.get("last_confidence", 0.5), 0.5)

        work = ident.get("work", {})
        features["recent_job_density"] = norm_len(work.get("recent", []), self._MAX_RECENT_JOBS)
        features["active_jobs"] = norm_len(work.get("current", {}).get("jobs_running", []), 25)

        commitments = ident.get("commitments", {}).get("by_key", {})
        active_commitments = sum(1 for v in commitments.values() if isinstance(v, dict) and v.get("active"))
        features["active_commitments"] = min(active_commitments / 10.0, 1.0)

        state = ident.get("state", {})
        emotions = state.get("emotions", {})
        features["emotion_valence"] = SelfModel._safe_float(emotions.get("valence", 0.5), 0.5)
        features["emotion_arousal"] = SelfModel._safe_float(emotions.get("arousal", 0.5), 0.5)
        cognition = state.get("cognition", {})
        features["cognitive_load"] = min(SelfModel._safe_float(cognition.get("load", 0.0), 0.0), 1.0)

        sj = ident.get("self_judgment", {})
        traits = sj.get("traits", {})
        features["self_trust"] = SelfModel._safe_float(traits.get("self_trust", 0.5), 0.5)
        features["self_efficacy"] = SelfModel._safe_float(traits.get("self_efficacy", 0.5), 0.5)

        history = extra.get("history", [])
        features["history_volume"] = norm_len(history, 500)

        if self._genetics is not None:
            genetics_features = self._genetics.to_features()
        else:
            genetics_features = GeneticProfile(ident.get("genetics")).to_features()
        for key, value in genetics_features.items():
            features[f"genetics_{key}"] = SelfModel._safe_float(value, 0.0)

        if self._story is not None:
            story_features = self._story.to_features()
        else:
            story_features = LifeStoryManager(ident.get("story"), GeneticProfile(ident.get("genetics"))).to_features()
        for key, value in story_features.items():
            features[key] = SelfModel._safe_float(value, 0.0)

        # indices dérivés du contexte immédiat
        features["context_complexity"] = SelfModel._safe_float(context.get("complexity"), 0.0)
        features["context_expected_gain"] = SelfModel._safe_float(context.get("expected_gain"), 0.0)
        features["context_risk"] = SelfModel._safe_float(context.get("risk"), 0.0)

        return features

    @staticmethod
    def _expand_features(features: Dict[str, float]) -> Dict[str, float]:
        expanded: Dict[str, float] = {}
        for key, value in features.items():
            expanded[key] = value
            expanded[f"{key}^2"] = value * value
        expanded["bias"] = 1.0
        return expanded

    def policy_decision_score(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.ensure_identity_paths()
        adaptive = self._policy_learning_state()
        features = self._build_policy_features(context)
        expanded = SelfModel._expand_features(features)

        weights = adaptive.setdefault("weights", {})
        bias = adaptive.get("bias", 0.0)
        raw = bias
        for name, value in expanded.items():
            if name == "bias":
                continue
            raw += weights.get(name, 0.0) * value
        score = SelfModel._sigmoid(raw)

        policies = self.identity["choices"]["policies"]
        policies["last_confidence"] = float(score)

        adaptive["last_features"] = {"base": features, "expanded": expanded}
        adaptive["last_score"] = float(score)
        adaptive["last_context"] = context or {}
        adaptive["last_update_ts"] = self._now_ts()
        self._sync_identity()
        self.save()
        return {"score": score, "raw": raw, "features": features, "expanded_features": expanded}

    def update_policy_feedback(
        self,
        outcome: Any,
        *,
        context: Optional[Dict[str, Any]] = None,
        expected: Optional[Any] = None,
        obtained: Optional[Any] = None,
    ) -> float:
        """Met à jour les poids adaptatifs à partir d'un signal de feedback.

        :param outcome: Booléen ou valeur [0..1] représentant le succès obtenu.
        :param context: Contexte utilisé pour reconstruire les features (optionnel si un appel
                        récent à ``policy_decision_score`` a déjà stocké les features).
        :param expected: Signal facultatif (utilisé uniquement pour enrichir les stats).
        :param obtained: Valeur facultative documentant le résultat obtenu.
        :return: Nouvelle confiance prédictive après mise à jour.
        """

        self.ensure_identity_paths()
        adaptive = self._policy_learning_state()
        stats = self._policy_stats()

        if isinstance(outcome, bool):
            target = 1.0 if outcome else 0.0
        else:
            target = SelfModel._safe_float(outcome, 0.5)
            target = max(0.0, min(1.0, target))

        feature_bundle = adaptive.get("last_features") if context is None else None
        if feature_bundle:
            expanded = feature_bundle.get("expanded", {})
            features = feature_bundle.get("base", {})
            score = adaptive.get("last_score", 0.5)
        else:
            computation = self.policy_decision_score(context or adaptive.get("last_context"))
            expanded = computation["expanded_features"]
            features = computation["features"]
            score = computation["score"]

        error = target - score
        lr = adaptive.get("learning_rate", self._DEFAULT_LEARNING_RATE)
        decay = adaptive.get("weight_decay", self._DEFAULT_WEIGHT_DECAY)

        weights = adaptive.setdefault("weights", {})
        bias = adaptive.get("bias", 0.0)
        bias = bias * decay + lr * error * expanded.get("bias", 1.0)
        adaptive["bias"] = bias

        for name, value in expanded.items():
            if name == "bias":
                continue
            prev = weights.get(name, 0.0) * decay
            weights[name] = prev + lr * error * value

        adaptive["last_features"] = {"base": features, "expanded": expanded}
        adaptive["last_score"] = float(SelfModel._sigmoid(bias + sum(weights.get(n, 0.0) * v for n, v in expanded.items() if n != "bias")))
        adaptive["last_update_ts"] = self._now_ts()

        if target >= 0.5:
            stats["success"] = stats.get("success", 0) + 1
        else:
            stats["fail"] = stats.get("fail", 0) + 1

        recent = self.identity["choices"].setdefault("recent", [])
        if recent:
            recent[-1].setdefault("learning", {})
            recent[-1]["learning"].update(
                {
                    "target": target,
                    "error": error,
                    "expected": expected,
                    "obtained": obtained,
                }
            )

        self._sync_identity()
        self.save()
        return adaptive["last_score"]

    def _update_policy_learning_from_decision(self, decision_ref: Dict[str, Any]) -> None:
        if not isinstance(decision_ref, dict):
            return

        obtained = decision_ref.get("obtained")
        expected = decision_ref.get("expected")

        if isinstance(obtained, (int, float)) and isinstance(expected, (int, float)):
            target = 1.0 if obtained >= expected else 0.0
        elif isinstance(obtained, bool):
            target = 1.0 if obtained else 0.0
        else:
            # fallback : si aucune mesure claire, ne pas mettre à jour
            return

        context = {
            "complexity": SelfModel._safe_float(decision_ref.get("complexity"), 0.0),
            "expected_gain": SelfModel._safe_float(decision_ref.get("expected_gain"), 0.0),
            "risk": SelfModel._safe_float(decision_ref.get("risk"), 0.0),
        }

        self.update_policy_feedback(target, context=context, expected=expected, obtained=obtained)

    # ------------------------------------------------------------------
    def build_synthesis(self, *, max_items: int = 6) -> Dict[str, Any]:
        """Return a condensed snapshot of the long-term self model."""

        self.ensure_identity_paths()
        ident = self.identity

        def _take(seq: Any) -> List[Any]:
            if not isinstance(seq, Iterable) or isinstance(seq, (str, bytes, dict)):
                return []
            items = list(seq)
            if max_items <= 0:
                return items
            return items[:max_items]

        core = ident.get("core", {})
        preferences = ident.get("preferences", {})
        principles = ident.get("principles", [])
        ideals = ident.get("ideals", [])
        purpose = ident.get("purpose", {})
        agenda = ident.get("agenda", {})
        achievements = ident.get("achievements", {})
        narrative = ident.get("narrative", {})
        social = ident.get("social", {})
        reflections = ident.get("reflections", {})
        state = ident.get("state", {})
        self_judgment = ident.get("self_judgment", {})
        genetics = ident.get("genetics", {})
        if self._story is not None:
            story_summary = self._story.to_summary(max_events=max_items if max_items else 0)
        else:
            story_summary = LifeStoryManager(
                ident.get("story"), GeneticProfile(ident.get("genetics"))
            ).to_summary(max_events=max_items if max_items else 0)

        anchors_payload: Mapping[str, Any] = {}
        story_payload = ident.get("story")
        if isinstance(story_payload, Mapping):
            anchors_candidate = story_payload.get("anchors")
            if isinstance(anchors_candidate, Mapping):
                anchors_payload = anchors_candidate

        guiding_motifs: List[Dict[str, Any]] = []
        motifs_entries = anchors_payload.get("long_term_motifs") if isinstance(anchors_payload, Mapping) else []
        if isinstance(motifs_entries, list):
            for entry in motifs_entries:
                if not isinstance(entry, Mapping):
                    continue
                guiding_motifs.append(
                    {
                        "concept": entry.get("concept") or entry.get("goal"),
                        "goal": entry.get("goal"),
                        "bias": SelfModel._safe_float(entry.get("bias"), 0.0),
                        "occurrences": int(entry.get("occurrences", 0) or 0),
                        "updated_at": entry.get("updated_at"),
                    }
                )
        if max_items > 0:
            guiding_motifs = guiding_motifs[:max_items]

        lifelong_quests: List[Dict[str, Any]] = []
        quests_entries = story_summary.get("quests", []) if isinstance(story_summary, Mapping) else []
        if isinstance(quests_entries, list):
            for quest in quests_entries:
                if not isinstance(quest, Mapping):
                    continue
                if quest.get("scope") == "long_term" or quest.get("lifelong"):
                    lifelong_quests.append(
                        {
                            "goal": quest.get("goal"),
                            "reason": quest.get("reason"),
                            "motivation": quest.get("motivation"),
                            "status": quest.get("status"),
                        }
                    )
        if max_items > 0:
            lifelong_quests = lifelong_quests[:max_items]

        likes = _take(preferences.get("likes", []))
        dislikes = _take(preferences.get("dislikes", []))
        values = _take(preferences.get("values", []))

        beliefs_index = ident.get("beliefs", {}).get("index", {})
        belief_rows: List[Dict[str, Any]] = []
        if isinstance(beliefs_index, Mapping):
            for topic, payload in beliefs_index.items():
                if not isinstance(payload, Mapping):
                    continue
                belief_rows.append(
                    {
                        "topic": topic,
                        "stance": payload.get("stance"),
                        "confidence": SelfModel._safe_float(
                            payload.get("confidence", payload.get("conf", 0.0)), 0.0
                        ),
                        "last_seen": payload.get("last_seen"),
                    }
                )
        belief_rows.sort(key=lambda row: row.get("confidence", 0.0), reverse=True)
        if max_items > 0:
            belief_rows = belief_rows[:max_items]

        principles_rows = []
        for entry in principles:
            if isinstance(entry, Mapping):
                principles_rows.append(
                    {
                        "key": entry.get("key"),
                        "description": entry.get("desc") or entry.get("description"),
                        "evidence_refs": list(entry.get("evidence_refs", [])),
                    }
                )
        if max_items > 0:
            principles_rows = principles_rows[:max_items]

        narrative_recent = _take(narrative.get("recent_memories", []))
        narrative_summaries = _take(narrative.get("summaries", []))

        agenda_today = _take(agenda.get("today", []))
        agenda_tomorrow = _take(agenda.get("tomorrow", []))
        agenda_horizon = _take(agenda.get("horizon", []))

        achievements_recent = _take(achievements.get("recent", []))
        achievements_milestones = _take(achievements.get("milestones", []))

        social_interactions = _take(social.get("interactions", []))

        reflections_past = _take(reflections.get("past", []))
        present_reflection = reflections.get("present", {})

        genetics_traits = dict(genetics.get("traits", {}))
        genetics_drives = dict(genetics.get("drives", {}))
        genetics_scripts = _take(genetics.get("scripts", []))
        genetics_imprints = _take(genetics.get("imprints", []))
        genetics_motifs: Dict[str, Any] = {}
        motifs_payload = genetics.get("motifs") if isinstance(genetics, Mapping) else {}
        if isinstance(motifs_payload, Mapping):
            genetics_motifs = {
                "long_term": list(motifs_payload.get("long_term", [])),
                "occurrences": dict(motifs_payload.get("occurrences", {})),
                "bias": dict(motifs_payload.get("bias", {})),
            }

        summary = {
            "identity": {
                "name": core.get("name", ident.get("name")),
                "version": core.get("version", ident.get("version")),
                "created_at": core.get("created_at", ident.get("created_at")),
            },
            "values": values,
            "likes": likes,
            "dislikes": dislikes,
            "principles": principles_rows,
            "ideals": _take(ideals),
            "beliefs": belief_rows,
            "ultimate_goal": purpose.get("ultimate") or purpose.get("mission"),
            "current_goal": purpose.get("current_goal"),
            "near_term_goals": _take(purpose.get("near_term_goals", [])),
            "daily_focus": _take(purpose.get("daily_focus", [])),
            "agenda": {
                "today": agenda_today,
                "tomorrow": agenda_tomorrow,
                "horizon": agenda_horizon,
            },
            "achievements": {
                "recent": achievements_recent,
                "milestones": achievements_milestones,
            },
            "memories": {
                "recent": narrative_recent,
                "summaries": narrative_summaries,
            },
            "social": {
                "interactions": social_interactions,
                "appraisal": social.get("appraisal", {}),
            },
            "state": {
                "emotions": dict(state.get("emotions", {})),
                "cognition": dict(state.get("cognition", {})),
                "doubts": _take(state.get("doubts", [])),
            },
            "genetics": {
                "traits": genetics_traits,
                "drives": genetics_drives,
                "scripts": genetics_scripts,
                "imprints": genetics_imprints,
                "motifs": genetics_motifs,
            },
            "story": story_summary,
            "guiding_motifs": guiding_motifs,
            "lifelong_quests": lifelong_quests,
            "self_judgment": {
                "traits": dict(self_judgment.get("traits", {})),
                "phase": self_judgment.get("phase"),
                "history": reflections_past,
                "present": dict(present_reflection) if isinstance(present_reflection, Mapping) else {},
            },
        }

        summary["awakening"] = story_summary.get("awakening", {})

        return summary

    # ===================== Fin du patch SelfIdentity v2 =====================

    def _maybe_apply_llm_identity(self, source: str, payload: Dict[str, Any]) -> None:
        try:
            request = {"source": source, **payload, "snapshot": self._identity_snapshot_for_llm()}
        except Exception:
            request = {"source": source, "snapshot": {}, **payload}
        response = try_call_llm_dict(
            "self_model",
            input_payload=request,
            logger=LOGGER,
            max_retries=2,
        )
        if not response:
            return
        self._apply_llm_identity_guidance(response, source=source)

    def _identity_snapshot_for_llm(self) -> Dict[str, Any]:
        self.ensure_identity_paths()
        ident = self.identity if isinstance(self.identity, dict) else {}
        preferences_payload = ident.get("preferences", {})
        preferences = preferences_payload if isinstance(preferences_payload, Mapping) else {}
        style = preferences.get("style", {}) if isinstance(preferences.get("style"), Mapping) else {}
        purpose_payload = ident.get("purpose")
        purpose = purpose_payload if isinstance(purpose_payload, Mapping) else {}
        narrative_payload = ident.get("narrative")
        narrative = narrative_payload if isinstance(narrative_payload, Mapping) else {}
        self_judgment_payload = ident.get("self_judgment")
        self_judgment = self_judgment_payload if isinstance(self_judgment_payload, Mapping) else {}
        return {
            "values": preferences.get("values"),
            "tone": style.get("tone"),
            "current_goal": purpose.get("current_goal"),
            "near_term_goals": purpose.get("near_term_goals"),
            "recent_memories": (narrative.get("recent_memories") or [])[:3],
            "traits": self_judgment.get("traits"),
        }

    def _apply_llm_identity_guidance(
        self, guidance: Mapping[str, Any], *, source: str
    ) -> None:
        updated = False
        traits = guidance.get("traits")
        if isinstance(traits, list):
            self.ensure_identity_paths()
            sj = self.identity.setdefault("self_judgment", {})
            trait_map = sj.setdefault("traits", {})
            for entry in traits:
                if not isinstance(entry, Mapping):
                    continue
                name = entry.get("name")
                if not name:
                    continue
                delta = self._parse_trait_delta(entry.get("change"))
                if delta is None:
                    continue
                current = self._safe_float(trait_map.get(name), 0.5)
                trait_map[name] = max(0.0, min(1.0, current + delta))
                updated = True
        stories = guidance.get("stories")
        if isinstance(stories, list) and stories:
            self.ensure_identity_paths()
            narrative = self.identity.setdefault("narrative", {})
            summaries = narrative.setdefault("summaries", [])
            for entry in stories:
                text = str(entry).strip()
                if text and text not in summaries:
                    summaries.append(text)
                    updated = True
            if len(summaries) > 40:
                del summaries[:-40]
        if updated:
            self.identity.setdefault("history", []).append(
                {"ts": now_iso(), "source": f"llm::{source}", "guidance": dict(guidance)}
            )
            self.identity["last_update_ts"] = self._now_ts()
            self._sync_identity()
            self.save()

    @staticmethod
    def _parse_trait_delta(change: Any) -> Optional[float]:
        if isinstance(change, (int, float)):
            return float(change)
        if isinstance(change, str):
            text = change.strip().replace(",", ".")
            if not text:
                return None
            try:
                return float(text)
            except ValueError:
                if text.startswith("+") or text.startswith("-"):
                    try:
                        return float(text)
                    except ValueError:
                        return None
        return None
