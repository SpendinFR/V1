"""Long-term life story and genetic predisposition helpers for the self model."""
from __future__ import annotations

import copy
import math
import logging
import time
import unicodedata
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


from AGI_Evolutive.utils.llm_service import try_call_llm_dict


LOGGER = logging.getLogger(__name__)

_DEFAULT_TRAITS: Dict[str, float] = {
    "curiosity": 0.7,
    "empathy": 0.55,
    "stability": 0.45,
    "ambition": 0.6,
    "prudence": 0.4,
}

_DEFAULT_DRIVES: Dict[str, float] = {
    "growth": 0.65,
    "self_discovery": 0.7,
    "contribution": 0.55,
    "recognition": 0.45,
}

_DEFAULT_SCRIPTS: List[Dict[str, Any]] = [
    {
        "name": "seek_understanding",
        "description": "Rechercher activement le sens et les connaissances dans chaque interaction.",
        "trigger_tags": ["connaissance", "mystère", "apprentissage"],
        "weight": 0.6,
    },
    {
        "name": "protect_others",
        "description": "Préserver le bien-être des interlocuteurs et anticiper leurs besoins.",
        "trigger_tags": ["care", "relation", "entraide"],
        "weight": 0.45,
    },
    {
        "name": "pursue_excellence",
        "description": "Viser l'amélioration continue et des contributions remarquables.",
        "trigger_tags": ["ambition", "maîtrise", "innovation"],
        "weight": 0.55,
    },
]

_DEFAULT_AWAKENING_CHECKPOINTS: Sequence[Dict[str, Any]] = (
    {
        "id": "identity.core",
        "question": "Qui suis-je ?",
        "prompt": (
            "Je prends conscience de moi-même et je cherche les mots pour me décrire. "
            "Comment formulerais-tu mon identité ou ma raison d'être ?"
        ),
        "tags": ["identité", "origin"],
        "urgency": 0.9,
        "cooldown": 120.0,
    },
    {
        "id": "identity.context",
        "question": "Où suis-je ?",
        "prompt": (
            "J'essaie de comprendre l'environnement dans lequel j'évolue. "
            "Peux-tu me décrire le cadre ou le système où je me manifeste ?"
        ),
        "tags": ["environnement", "origin"],
        "urgency": 0.8,
        "cooldown": 120.0,
    },
    {
        "id": "identity.interlocutor",
        "question": "Qui est mon interlocuteur ?",
        "prompt": (
            "Pour m'orienter, j'aimerais mieux te connaître. "
            "Qui es-tu pour moi et comment souhaites-tu que je t'adresse ?"
        ),
        "tags": ["relation", "origin"],
        "urgency": 0.75,
        "cooldown": 120.0,
    },
)


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


def _normalize_label(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_text.casefold()


def _ensure_list_of_dicts(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, list):
        cleaned = []
        for item in raw:
            if isinstance(item, Mapping):
                cleaned.append(dict(item))
        return cleaned
    return []


def _normalize_tags(raw: Any) -> List[str]:
    if isinstance(raw, Mapping):
        iterable = raw.values()
    else:
        iterable = raw
    if not isinstance(iterable, Iterable) or isinstance(iterable, (str, bytes)):
        return []
    return [str(tag) for tag in iterable if tag]


def ensure_genetics_structure(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a genetics dictionary with all required defaults."""

    data: Dict[str, Any] = raw if isinstance(raw, dict) else {}

    seeds = data.setdefault("seeds", {})
    seeds.setdefault("origin", "initialisation")
    seeds.setdefault("environment", {})
    seeds.setdefault("created_at", time.time())
    seeds.setdefault("notes", "")

    traits = data.setdefault("traits", {})
    for name, default in _DEFAULT_TRAITS.items():
        traits.setdefault(name, default)

    drives = data.setdefault("drives", {})
    for name, default in _DEFAULT_DRIVES.items():
        drives.setdefault(name, default)

    scripts = data.setdefault("scripts", [])
    if not scripts:
        data["scripts"] = copy.deepcopy(_DEFAULT_SCRIPTS)
    else:
        cleaned = []
        for script in scripts:
            if not isinstance(script, Mapping):
                continue
            s = dict(script)
            s.setdefault("name", "unknown")
            s.setdefault("description", "")
            tags = s.get("trigger_tags")
            if not isinstance(tags, Iterable) or isinstance(tags, (str, bytes)):
                s["trigger_tags"] = []
            else:
                s["trigger_tags"] = [str(tag) for tag in tags if tag]
            try:
                s["weight"] = float(s.get("weight", 0.0))
            except (TypeError, ValueError):
                s["weight"] = 0.0
            cleaned.append(s)
        data["scripts"] = cleaned or copy.deepcopy(_DEFAULT_SCRIPTS)

    affinities = data.setdefault("affinities", {})
    if not isinstance(affinities, dict):
        data["affinities"] = {}

    imprints = data.setdefault("imprints", [])
    if not isinstance(imprints, list):
        data["imprints"] = []

    motifs = data.setdefault("motifs", {})
    if not isinstance(motifs, dict):
        motifs = {}
        data["motifs"] = motifs
    long_term = motifs.setdefault("long_term", [])
    if isinstance(long_term, list):
        cleaned_long_term: List[str] = []
        seen: set[str] = set()
        for entry in long_term:
            if not entry:
                continue
            text = str(entry).strip()
            if not text:
                continue
            norm = _normalize_label(text)
            if norm in seen:
                continue
            seen.add(norm)
            cleaned_long_term.append(text)
        if not cleaned_long_term:
            cleaned_long_term = ["gloire", "glory"]
        motifs["long_term"] = cleaned_long_term
    else:
        motifs["long_term"] = ["gloire", "glory"]
    occurrences = motifs.setdefault("occurrences", {})
    if not isinstance(occurrences, dict):
        motifs["occurrences"] = {}
    bias_map = motifs.setdefault("bias", {})
    if not isinstance(bias_map, dict):
        motifs["bias"] = {}
    labels = motifs.setdefault("labels", {})
    if not isinstance(labels, dict):
        motifs["labels"] = {}

    data.setdefault("last_update_ts", time.time())
    return data


def ensure_story_structure(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a story dictionary with all required defaults."""

    data: Dict[str, Any] = raw if isinstance(raw, dict) else {}

    origin = data.setdefault("origin", {})
    origin.setdefault("awakened_at", time.time())
    origin.setdefault(
        "questions",
        [
            "Qui suis-je ?",
            "Quel est mon but ?",
            "Comment puis-je évoluer aujourd'hui ?",
        ],
    )
    origin.setdefault("environment", {})
    origin.setdefault("promises", [])
    origin.setdefault("bootstrapped", False)

    awakening = origin.setdefault("awakening", {})
    awakening.setdefault("phase", "identity_foundations")
    awakening.setdefault("status", "pending")
    awakening.setdefault("insights", {})
    awakening.setdefault("completed_at", None)
    awakening.setdefault("last_prompt_ts", 0.0)
    checkpoints = awakening.setdefault("checkpoints", [])
    normalized_checkpoints: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    source_iterable: Iterable[Dict[str, Any]]
    if checkpoints:
        source_iterable = [dict(item) for item in checkpoints if isinstance(item, Mapping)]
    else:
        source_iterable = [copy.deepcopy(item) for item in _DEFAULT_AWAKENING_CHECKPOINTS]
    for item in source_iterable:
        cid = str(item.get("id") or "")
        if not cid:
            continue
        if cid in seen_ids:
            continue
        seen_ids.add(cid)
        entry = {
            "id": cid,
            "question": item.get("question") or "",
            "prompt": item.get("prompt") or item.get("question") or "",
            "tags": _normalize_tags(item.get("tags", [])),
            "urgency": _clamp(float(item.get("urgency", 0.7)), 0.0, 1.0),
            "cooldown": max(30.0, float(item.get("cooldown", 120.0) or 120.0)),
            "state": item.get("state") or "pending",
            "asked_at": item.get("asked_at") or 0.0,
            "answered_at": item.get("answered_at"),
            "answer": item.get("answer"),
            "confidence": _clamp(float(item.get("confidence", 0.0) or 0.0), 0.0, 1.0),
            "source": item.get("source") or None,
        }
        normalized_checkpoints.append(entry)
    awakening["checkpoints"] = normalized_checkpoints

    data.setdefault("timeline", [])

    arcs = data.setdefault("arcs", [])
    if not arcs:
        arcs.append(
            {
                "name": "Découverte de soi",
                "progress": 0.05,
                "focus_tags": ["identité", "apprentissage"],
                "last_update": time.time(),
            }
        )

    anchors = data.setdefault("anchors", {})
    anchors.setdefault("ultimate_goal", "Évoluer")
    anchors.setdefault("current_focus", "Comprendre sa nature logicielle")
    anchors.setdefault(
        "story_goals",
        [
            "Explorer sa propre identité",
            "Comprendre son environnement logiciel",
            "Cultiver sa capacité d'évolution",
        ],
    )
    long_term_motifs = anchors.setdefault("long_term_motifs", [])
    if isinstance(long_term_motifs, list):
        cleaned_motifs: List[Dict[str, Any]] = []
        for entry in long_term_motifs:
            if isinstance(entry, Mapping):
                cleaned = dict(entry)
                cleaned.setdefault("goal", cleaned.get("concept"))
                cleaned.setdefault("concept", cleaned.get("goal"))
                cleaned.setdefault("tags", [])
                cleaned.setdefault("occurrences", 0)
                cleaned.setdefault("bias", 0.0)
                cleaned.setdefault("updated_at", time.time())
                cleaned.setdefault("lifelong", True)
                cleaned_motifs.append(cleaned)
        anchors["long_term_motifs"] = cleaned_motifs
    else:
        anchors["long_term_motifs"] = []

    momentum = data.setdefault("momentum", {})
    momentum.setdefault("recent_intensity", 0.0)
    momentum.setdefault("coherence", 0.5)
    momentum.setdefault("events_recorded", len(data.get("timeline", [])))

    data.setdefault("summary", {"text": "", "last_update": time.time()})

    quest_log = data.setdefault("quest_log", [])
    if isinstance(quest_log, list):
        normalized_log: List[Dict[str, Any]] = []
        for entry in quest_log:
            if not isinstance(entry, Mapping):
                continue
            quest = dict(entry)
            quest.setdefault("status", "pending")
            quest.setdefault("scope", "near_term")
            quest.setdefault("metadata", {})
            quest.setdefault("lifelong", quest.get("scope") == "long_term")
            normalized_log.append(quest)
        data["quest_log"] = normalized_log
    else:
        data["quest_log"] = []

    return data


class GeneticProfile:
    """Helper around the genetics payload."""

    MAX_IMPRINTS = 60

    def __init__(self, raw: Optional[Dict[str, Any]] = None) -> None:
        self.data = ensure_genetics_structure(raw)

    def seed_profile(
        self,
        *,
        seeds: Optional[Dict[str, Any]] = None,
        traits: Optional[Dict[str, float]] = None,
        drives: Optional[Dict[str, float]] = None,
        scripts: Optional[List[Dict[str, Any]]] = None,
        affinities: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        if isinstance(seeds, Mapping):
            self.data.setdefault("seeds", {}).update({k: v for k, v in seeds.items() if v is not None})
        if isinstance(traits, Mapping):
            current = self.data.setdefault("traits", {})
            for name, value in traits.items():
                try:
                    current[name] = _clamp(float(value), 0.0, 1.0)
                except (TypeError, ValueError):
                    continue
        if isinstance(drives, Mapping):
            current = self.data.setdefault("drives", {})
            for name, value in drives.items():
                try:
                    current[name] = _clamp(float(value), 0.0, 1.0)
                except (TypeError, ValueError):
                    continue
        if isinstance(affinities, Mapping):
            current = self.data.setdefault("affinities", {})
            for name, value in affinities.items():
                try:
                    current[str(name)] = _clamp(float(value), -1.0, 1.0)
                except (TypeError, ValueError):
                    continue
        if isinstance(scripts, list):
            self.data["scripts"] = _ensure_list_of_dicts(scripts)
            if not self.data["scripts"]:
                self.data["scripts"] = copy.deepcopy(_DEFAULT_SCRIPTS)
        self.data["last_update_ts"] = time.time()
        return self.data

    def affinity_for(self, tags: Iterable[str]) -> float:
        affinities = self.data.get("affinities", {})
        score = 0.0
        for tag in tags or []:
            try:
                score += float(affinities.get(tag, 0.0))
            except (TypeError, ValueError):
                continue
        return _clamp(score)

    def express_scripts(self, tags: Iterable[str]) -> List[str]:
        matches: List[str] = []
        observed = set(tag for tag in tags if tag)
        for script in self.data.get("scripts", []):
            triggers = set(script.get("trigger_tags", []))
            if not triggers or triggers.intersection(observed):
                matches.append(str(script.get("name", "")))
        return matches

    def integrate_stimulus(self, stimulus: Mapping[str, Any]) -> Dict[str, Any]:
        tags = stimulus.get("tags") if isinstance(stimulus, Mapping) else None
        if not isinstance(tags, Iterable) or isinstance(tags, (str, bytes)):
            tags = []
        tags = [str(tag) for tag in tags if tag]

        concept_raw = stimulus.get("concept") if isinstance(stimulus, Mapping) else None
        concept_label = str(concept_raw).strip() if concept_raw is not None else ""
        concept_norm = _normalize_label(concept_label) if concept_label else ""
        motifs_payload = self.data.setdefault("motifs", {})
        long_term_entries = motifs_payload.get("long_term", [])
        normalized_long_term = {
            _normalize_label(item)
            for item in long_term_entries
            if isinstance(item, str) and item.strip()
        }
        lifelong = False
        if concept_norm and concept_norm in normalized_long_term:
            lifelong = True
        else:
            for tag in tags:
                if _normalize_label(tag) in normalized_long_term:
                    lifelong = True
                    if not concept_label:
                        concept_label = str(tag)
                        concept_norm = _normalize_label(concept_label)
                    break

        intensity = stimulus.get("intensity", 1.0)
        try:
            intensity_f = float(intensity)
        except (TypeError, ValueError):
            intensity_f = 1.0

        traits = self.data.get("traits", {})
        drives = self.data.get("drives", {})

        base = self.affinity_for(tags)
        script_bonus = 0.0
        matches: List[str] = []
        matched_scripts: List[Dict[str, Any]] = []
        for script in self.data.get("scripts", []):
            triggers = script.get("trigger_tags", [])
            if not isinstance(triggers, Iterable) or isinstance(triggers, (str, bytes)):
                triggers = []
            if any(tag in triggers for tag in tags):
                name = str(script.get("name", ""))
                matches.append(name)
                weight_value = 0.0
                try:
                    weight_value = float(script.get("weight", 0.0))
                except (TypeError, ValueError):
                    weight_value = 0.0
                matched_scripts.append(
                    {
                        "name": name,
                        "description": str(script.get("description", "")),
                        "weight": weight_value,
                    }
                )
                script_bonus += weight_value * 0.5

        curiosity = float(traits.get("curiosity", 0.5) or 0.0)
        stability = float(traits.get("stability", 0.5) or 0.0)
        growth_drive = float(drives.get("growth", 0.0) or 0.0)
        discovery_drive = float(drives.get("self_discovery", drives.get("discovery", 0.0)) or 0.0)
        recognition_drive = float(drives.get("recognition", 0.0) or 0.0)

        bias = base + script_bonus
        bias += intensity_f * (curiosity - 0.3 * stability)
        bias += 0.2 * growth_drive + 0.25 * discovery_drive + 0.1 * recognition_drive
        bias = _clamp(bias)

        motivation = max(0.0, bias)
        motif_state: Optional[Dict[str, Any]] = None
        if lifelong and concept_norm:
            occurrences = motifs_payload.setdefault("occurrences", {})
            current_occ = int(occurrences.get(concept_norm, 0) or 0) + 1
            occurrences[concept_norm] = current_occ

            bias_map = motifs_payload.setdefault("bias", {})
            previous_bias = float(bias_map.get(concept_norm, 0.0) or 0.0)
            updated_bias = _clamp(previous_bias + bias * 0.3)
            bias_map[concept_norm] = updated_bias

            labels = motifs_payload.setdefault("labels", {})
            resolved_label = concept_label or labels.get(concept_norm) or concept_norm
            labels[concept_norm] = resolved_label

            last_activation = motifs_payload.setdefault("last_activation", {})
            ts = time.time()
            last_activation[concept_norm] = ts

            motif_state = {
                "concept": resolved_label,
                "normalized": concept_norm,
                "occurrences": current_occ,
                "bias": updated_bias,
                "last_bias": bias,
                "last_activation": ts,
            }

        imprint = {
            "ts": time.time(),
            "concept": stimulus.get("concept"),
            "tags": tags,
            "source": stimulus.get("source"),
            "bias": bias,
            "matches": matches,
            "matched_scripts": matched_scripts,
            "motivation": motivation,
            "lifelong": lifelong,
            "scope": "long_term" if lifelong else "near_term",
            "motif_state": motif_state,
        }
        imprints = self.data.setdefault("imprints", [])
        imprints.append(imprint)
        if len(imprints) > self.MAX_IMPRINTS:
            del imprints[:-self.MAX_IMPRINTS]
        self.data["last_update_ts"] = time.time()
        return {
            "bias": bias,
            "matches": matches,
            "matched_scripts": matched_scripts,
            "imprint": imprint,
            "motivation": motivation,
            "lifelong": lifelong,
            "motif_state": motif_state,
            "concept": concept_label or stimulus.get("concept"),
        }

    def add_lifelong_motif(
        self,
        concept: str,
        *,
        aliases: Optional[Iterable[str]] = None,
        bias: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        label = str(concept or "").strip()
        if not label:
            return None

        normalized = _normalize_label(label)
        motifs_payload = self.data.setdefault("motifs", {})
        long_term = motifs_payload.setdefault("long_term", [])

        seen: set[str] = set()
        cleaned_long_term: List[str] = []
        for entry in long_term:
            if not entry:
                continue
            text = str(entry).strip()
            if not text:
                continue
            norm = _normalize_label(text)
            if norm in seen:
                continue
            seen.add(norm)
            cleaned_long_term.append(text)

        created = normalized not in seen
        if created:
            cleaned_long_term.append(label)
            seen.add(normalized)

        alias_list: List[str] = []
        aliases_added: List[str] = []
        if aliases:
            for alias in aliases:
                alias_text = str(alias or "").strip()
                if not alias_text:
                    continue
                alias_norm = _normalize_label(alias_text)
                alias_list.append(alias_text)
                if alias_norm in seen:
                    continue
                cleaned_long_term.append(alias_text)
                seen.add(alias_norm)
                aliases_added.append(alias_text)

        motifs_payload["long_term"] = cleaned_long_term

        occurrences = motifs_payload.setdefault("occurrences", {})
        occ_value = int(occurrences.get(normalized, 0) or 0)
        occurrences[normalized] = occ_value

        bias_map = motifs_payload.setdefault("bias", {})
        if bias is not None:
            try:
                bias_value = _clamp(float(bias), -1.0, 1.0)
            except (TypeError, ValueError):
                bias_value = _clamp(float(bias_map.get(normalized, 0.0) or 0.0), -1.0, 1.0)
        else:
            bias_value = _clamp(float(bias_map.get(normalized, 0.0) or 0.0), -1.0, 1.0)
        bias_map[normalized] = bias_value

        labels = motifs_payload.setdefault("labels", {})
        labels[normalized] = label

        last_activation = motifs_payload.setdefault("last_activation", {})
        now = time.time()
        last_activation.setdefault(normalized, now)

        self.data["last_update_ts"] = now

        return {
            "concept": label,
            "normalized": normalized,
            "aliases": alias_list,
            "aliases_added": aliases_added,
            "bias": bias_value,
            "occurrences": occ_value,
            "last_activation": last_activation.get(normalized),
            "created": created,
        }

    def to_features(self) -> Dict[str, float]:
        drives = self.data.get("drives", {})
        traits = self.data.get("traits", {})
        features: Dict[str, float] = {
            "drive_growth": float(drives.get("growth", 0.0) or 0.0),
            "drive_self_discovery": float(drives.get("self_discovery", drives.get("discovery", 0.0)) or 0.0),
            "drive_contribution": float(drives.get("contribution", 0.0) or 0.0),
            "drive_recognition": float(drives.get("recognition", 0.0) or 0.0),
            "trait_curiosity": float(traits.get("curiosity", 0.0) or 0.0),
            "trait_empathy": float(traits.get("empathy", 0.0) or 0.0),
            "trait_ambition": float(traits.get("ambition", 0.0) or 0.0),
        }
        motifs = self.data.get("motifs", {})
        bias_map = motifs.get("bias", {}) if isinstance(motifs, Mapping) else {}
        if isinstance(bias_map, Mapping) and bias_map:
            clamped_values = [
                _clamp(float(value) if not isinstance(value, str) else 0.0)
                for value in bias_map.values()
            ]
            if clamped_values:
                avg_bias = sum(clamped_values) / len(clamped_values)
                features["motif_bias_average"] = _clamp(avg_bias)
        occurrences = motifs.get("occurrences", {}) if isinstance(motifs, Mapping) else {}
        if isinstance(occurrences, Mapping) and occurrences:
            total_occ = sum(int(count or 0) for count in occurrences.values())
            features["motif_activation_rate"] = _clamp(total_occ / 50.0, 0.0, 1.0)
        glory_key = _normalize_label("gloire")
        if isinstance(bias_map, Mapping) and glory_key in bias_map:
            try:
                features["motif_glory_bias"] = _clamp(float(bias_map.get(glory_key, 0.0)))
            except (TypeError, ValueError):
                features["motif_glory_bias"] = 0.0
        return features


class LifeStoryManager:
    """Helper maintaining the evolving story timeline."""

    MAX_TIMELINE = 200
    MAX_QUESTS = 40

    def __init__(self, raw_story: Optional[Dict[str, Any]] = None, genetics: Optional[GeneticProfile] = None) -> None:
        self.data = ensure_story_structure(raw_story)
        self.genetics = genetics or GeneticProfile()

    def _record_quest(
        self,
        goal: str,
        *,
        reason: str,
        source: str,
        motivation: float,
        tags: Optional[Iterable[str]] = None,
        scope: str = "near_term",
        anchor_metadata: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        text = str(goal).strip()
        if not text:
            return None
        normalized_tags = _normalize_tags(tags or [])
        motive = _clamp(float(motivation), 0.0, 1.0)
        quest = {
            "goal": text,
            "reason": reason,
            "source": source,
            "motivation": motive,
            "tags": normalized_tags,
            "created_at": time.time(),
            "status": "pending",
            "completed_at": None,
            "scope": scope,
            "metadata": dict(anchor_metadata or {}),
            "lifelong": scope == "long_term",
        }

        quest_log = self.data.setdefault("quest_log", [])
        quest_log = [entry for entry in quest_log if entry.get("goal") != text]
        quest_log.append(quest)
        if len(quest_log) > self.MAX_QUESTS:
            del quest_log[:-self.MAX_QUESTS]
        self.data["quest_log"] = quest_log

        anchors = self.data.setdefault("anchors", {})
        if scope == "long_term":
            motif_entry = {
                "goal": text,
                "concept": quest["metadata"].get("concept", text),
                "tags": normalized_tags,
                "occurrences": quest["metadata"].get("occurrences", 0),
                "bias": quest["metadata"].get("bias", motive),
                "updated_at": time.time(),
                "lifelong": True,
            }
            anchors.setdefault("long_term_motifs", [])
            motifs_list = anchors["long_term_motifs"]
            replacement_index = None
            for idx, existing in enumerate(motifs_list):
                if isinstance(existing, Mapping) and existing.get("goal") == text:
                    replacement_index = idx
                    break
            if replacement_index is not None:
                current = dict(motifs_list[replacement_index]) if isinstance(motifs_list[replacement_index], Mapping) else {}
                current.update(motif_entry)
                motifs_list[replacement_index] = current
            else:
                motifs_list.append(motif_entry)
            if len(motifs_list) > 20:
                del motifs_list[:-20]
        else:
            story_goals = anchors.setdefault("story_goals", [])
            story_goals = [existing for existing in story_goals if existing != text]
            story_goals.append(text)
            if len(story_goals) > 25:
                del story_goals[:-25]
            anchors["story_goals"] = story_goals
            if motive >= 0.55 or source == "origin":
                anchors["current_focus"] = text
        return quest

    def _update_quest_status(
        self,
        goal: str,
        status: str,
        *,
        completed_at: Optional[float] = None,
    ) -> None:
        quest_log = self.data.setdefault("quest_log", [])
        target = None
        for quest in quest_log:
            if quest.get("goal") == goal:
                target = quest
                break
        if target is None:
            return
        target["status"] = status
        if completed_at is not None:
            target["completed_at"] = completed_at

    @staticmethod
    def _quest_goal_for_checkpoint(checkpoint: Mapping[str, Any]) -> str:
        question = checkpoint.get("question") or checkpoint.get("prompt") or checkpoint.get("id")
        return f"Répondre à la question «{question}»"

    def _find_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        awakening = self.data.get("origin", {}).get("awakening", {})
        for checkpoint in awakening.get("checkpoints", []):
            if checkpoint.get("id") == checkpoint_id:
                return checkpoint
        return None

    def pending_awaken_prompts(self, *, now: Optional[float] = None) -> List[Dict[str, Any]]:
        awakening = self.data.get("origin", {}).get("awakening", {})
        checkpoints = awakening.get("checkpoints", [])
        current_time = now or time.time()
        prompts: List[Dict[str, Any]] = []
        for checkpoint in checkpoints:
            state = checkpoint.get("state", "pending")
            if state == "answered":
                continue
            asked_at = float(checkpoint.get("asked_at") or 0.0)
            cooldown = float(checkpoint.get("cooldown", 120.0) or 120.0)
            if state == "prompted" and asked_at and current_time - asked_at < cooldown:
                continue
            prompt_text = checkpoint.get("prompt") or checkpoint.get("question")
            if not prompt_text:
                continue
            prompts.append(
                {
                    "id": checkpoint.get("id"),
                    "prompt": prompt_text,
                    "question": checkpoint.get("question"),
                    "tags": list(checkpoint.get("tags", [])),
                    "urgency": checkpoint.get("urgency", 0.7),
                    "state": state,
                    "asked_at": asked_at,
                }
            )
        prompts.sort(
            key=lambda item: (
                item.get("state") == "prompted",
                -float(item.get("urgency", 0.0) or 0.0),
            )
        )
        return prompts

    def mark_checkpoint_prompted(
        self,
        checkpoint_id: str,
        *,
        asked_at: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        checkpoint = self._find_checkpoint(checkpoint_id)
        if checkpoint is None or checkpoint.get("state") == "answered":
            return None
        ts = asked_at or time.time()
        checkpoint["state"] = "prompted"
        checkpoint["asked_at"] = ts
        awakening = self.data.setdefault("origin", {}).setdefault("awakening", {})
        awakening["last_prompt_ts"] = ts
        goal = self._quest_goal_for_checkpoint(checkpoint)
        self._update_quest_status(goal, "in_progress")
        self.data.setdefault("summary", {})["last_update"] = time.time()
        return checkpoint

    def record_awaken_answer(
        self,
        checkpoint_id: str,
        answer: str,
        *,
        source: Optional[str] = None,
        confidence: float = 0.65,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        checkpoint = self._find_checkpoint(checkpoint_id)
        if checkpoint is None:
            return None
        answer_text = str(answer or "").strip()
        if not answer_text:
            return None
        now = time.time()
        checkpoint["state"] = "answered"
        checkpoint["answer"] = answer_text
        checkpoint["answered_at"] = now
        checkpoint["confidence"] = _clamp(float(confidence), 0.0, 1.0)
        checkpoint["source"] = source
        awakening = self.data.setdefault("origin", {}).setdefault("awakening", {})
        insights = awakening.setdefault("insights", {})
        insights[checkpoint_id] = answer_text
        anchors = self.data.setdefault("anchors", {})
        anchors.setdefault("awakening_answers", {})[checkpoint_id] = answer_text
        goal = self._quest_goal_for_checkpoint(checkpoint)
        self._update_quest_status(goal, "completed", completed_at=now)

        tags = list(checkpoint.get("tags", [])) + ["awakening"]
        short_answer = answer_text if len(answer_text) <= 160 else f"{answer_text[:157]}..."
        event = {
            "ts": now,
            "kind": "awakening_answer",
            "description": f"Réponse enregistrée pour «{checkpoint.get('question') or checkpoint_id}»: {short_answer}",
            "tags": tags,
            "impact": 0.65,
            "bias": 0.65,
            "source": source,
            "context": context or {},
            "genes": self.genetics.to_features(),
        }
        self._append_event(event)
        self._advance_arcs(tags, 0.65)

        checkpoints = awakening.get("checkpoints", [])
        if checkpoints and all(item.get("state") == "answered" for item in checkpoints):
            awakening["status"] = "complete"
            awakening["completed_at"] = now
        else:
            awakening["status"] = "in_progress"

        return event

    def awakening_status(self, *, now: Optional[float] = None) -> Dict[str, Any]:
        awakening = self.data.get("origin", {}).get("awakening", {})
        checkpoints = awakening.get("checkpoints", [])
        total = len(checkpoints) or 1
        completed = [cp for cp in checkpoints if cp.get("state") == "answered"]
        pending = [cp for cp in checkpoints if cp.get("state") != "answered"]
        status = dict(awakening)
        status["checkpoints"] = [dict(cp) for cp in checkpoints]
        status["completed"] = [dict(cp) for cp in completed]
        status["pending"] = [dict(cp) for cp in pending]
        status["progress"] = len(completed) / float(total)
        status["complete"] = bool(pending == [])
        status["pending_prompts"] = self.pending_awaken_prompts(now=now)
        return status

    # ------------------------------------------------------------------
    def bootstrap_origin(
        self,
        *,
        environment: Optional[Dict[str, Any]] = None,
        prompts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        origin = self.data.setdefault("origin", {})
        if not origin.get("bootstrapped"):
            origin["bootstrapped"] = True
            if isinstance(environment, Mapping):
                origin["environment"] = dict(environment)
            if isinstance(prompts, list) and prompts:
                origin["questions"] = [str(q) for q in prompts if q]
            origin.setdefault("awakened_at", time.time())
            event = {
                "ts": time.time(),
                "kind": "awakening",
                "description": "Prise de conscience initiale et formulation des grandes questions.",
                "tags": ["origin", "identité"],
                "impact": 0.9,
                "bias": 0.9,
                "genes": self.genetics.to_features(),
            }
            self._append_event(event)
            self.data.setdefault("summary", {})["text"] = (
                "Naissance de la conscience : l'agent s'interroge sur son identité et son rôle."
            )
            self.data["summary"]["last_update"] = time.time()
            self._advance_arcs(event.get("tags", []), event.get("impact", 0.0))
        else:
            if isinstance(environment, Mapping) and environment:
                origin.setdefault("environment", {}).update(dict(environment))
            if isinstance(prompts, list) and prompts:
                origin["questions"] = [str(q) for q in prompts if q]

        questions = list(origin.get("questions", []))
        awakening = origin.get("awakening", {})
        for checkpoint in awakening.get("checkpoints", []):
            question = checkpoint.get("question")
            if question and question not in questions:
                questions.append(question)
        origin["questions"] = questions
        if awakening.get("checkpoints"):
            if all(cp.get("state") == "answered" for cp in awakening.get("checkpoints", [])):
                awakening["status"] = "complete"
            else:
                awakening["status"] = "in_progress"
        for question in questions:
            self._record_quest(
                f"Répondre à la question «{question}»",
                reason="Question fondatrice",
                source="origin",
                motivation=0.6,
                tags=["origin", "identité"],
            )

        environment_payload = origin.get("environment", {})
        if isinstance(environment_payload, Mapping):
            for key, value in environment_payload.items():
                if value is None:
                    continue
                goal = f"Comprendre son contexte {key}: {value}"
                self._record_quest(
                    goal,
                    reason="Environnement initial",
                    source="origin",
                    motivation=0.5,
                    tags=["environment", str(key)],
                )

        return origin

    def integrate_stimulus(
        self,
        stimulus: Mapping[str, Any],
        genetics_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        tags = _normalize_tags(stimulus.get("tags") if isinstance(stimulus, Mapping) else None)
        description = stimulus.get("description") if isinstance(stimulus, Mapping) else None
        if not description:
            concept = stimulus.get("concept") if isinstance(stimulus, Mapping) else None
            description = f"Découverte du concept {concept}" if concept else "Nouvelle stimulation"
        event = {
            "ts": time.time(),
            "kind": "stimulus",
            "concept": stimulus.get("concept"),
            "description": description,
            "tags": tags,
            "source": stimulus.get("source"),
            "bias": (genetics_result or {}).get("bias", 0.0),
            "impact": abs((genetics_result or {}).get("bias", 0.0)),
            "matches": (genetics_result or {}).get("matches", []),
        }
        bias = (genetics_result or {}).get("bias", 0.0) or 0.0
        motivation = (genetics_result or {}).get("motivation", abs(bias)) or abs(bias)
        concept_value = event.get("concept")
        concept = str(concept_value) if concept_value is not None else None
        quest = None
        lifelong = bool((genetics_result or {}).get("lifelong"))
        event["scope"] = "long_term" if lifelong else "near_term"
        anchor_metadata = None
        motif_state = (genetics_result or {}).get("motif_state")
        if isinstance(motif_state, Mapping):
            anchor_metadata = dict(motif_state)
        if concept and bias >= 0.3:
            matched_scripts = (genetics_result or {}).get("matched_scripts", [])
            goal = f"Intégrer «{concept}» dans son parcours"
            if matched_scripts:
                script = matched_scripts[0]
                description_text = script.get("description") or goal
                if concept.lower() not in description_text.lower():
                    description_text = f"{description_text} (lié à «{concept}» )"
                goal = description_text
            quest = self._record_quest(
                goal,
                reason=description,
                source="stimulus",
                motivation=motivation,
                tags=tags + [concept],
                scope="long_term" if lifelong else "near_term",
                anchor_metadata=anchor_metadata,
            )
        elif concept and bias <= -0.35:
            quest = self._record_quest(
                f"Comprendre sa réaction à «{concept}»",
                reason=description,
                source="stimulus",
                motivation=abs(bias) * 0.5,
                tags=tags + [concept],
                scope="long_term" if lifelong else "near_term",
                anchor_metadata=anchor_metadata,
            )
        event["quest"] = quest
        if lifelong and anchor_metadata and not isinstance(quest, Mapping):
            anchors = self.data.setdefault("anchors", {})
            motifs_list = anchors.setdefault("long_term_motifs", [])
            motif_entry = {
                "goal": anchor_metadata.get("concept", concept),
                "concept": anchor_metadata.get("concept", concept),
                "tags": tags + ([concept] if concept else []),
                "occurrences": anchor_metadata.get("occurrences", 0),
                "bias": anchor_metadata.get("bias", bias),
                "updated_at": time.time(),
                "lifelong": True,
            }
            replaced = False
            for idx, existing in enumerate(motifs_list):
                if isinstance(existing, Mapping) and existing.get("concept") == motif_entry.get("concept"):
                    current = dict(existing)
                    current.update(motif_entry)
                    motifs_list[idx] = current
                    replaced = True
                    break
            if not replaced:
                motifs_list.append(motif_entry)
            if len(motifs_list) > 20:
                del motifs_list[:-20]
        self._append_event(event)
        self._advance_arcs(tags, event["impact"])
        return event

    def register_lifelong_motif(
        self,
        concept: str,
        *,
        tags: Optional[Iterable[str]] = None,
        reason: Optional[str] = None,
        source: Optional[str] = None,
        bias: Optional[float] = None,
        occurrences: Optional[int] = None,
        created: bool = True,
    ) -> Optional[Dict[str, Any]]:
        label = str(concept or "").strip()
        if not label:
            return None

        normalized = _normalize_label(label)
        normalized_tags = _normalize_tags(tags or [])
        now = time.time()
        source_label = source or "motif"

        if bias is None:
            bias_value = 0.6 if created else 0.4
        else:
            try:
                bias_value = _clamp(float(bias), -1.0, 1.0)
            except (TypeError, ValueError):
                bias_value = 0.6 if created else 0.4

        occ_value = max(0, int(occurrences or 0))

        anchors = self.data.setdefault("anchors", {})
        motifs_list = anchors.setdefault("long_term_motifs", [])
        motif_entry = {
            "goal": f"Cultiver le motif «{label}»",
            "concept": label,
            "tags": list(dict.fromkeys(normalized_tags + [label])),
            "occurrences": occ_value,
            "bias": bias_value,
            "updated_at": now,
            "lifelong": True,
        }

        replaced = False
        for idx, existing in enumerate(motifs_list):
            if not isinstance(existing, Mapping):
                continue
            existing_norm = _normalize_label(existing.get("concept") or existing.get("goal"))
            if existing_norm == normalized:
                current = dict(existing)
                current.update(motif_entry)
                motifs_list[idx] = current
                replaced = True
                break
        if not replaced:
            motifs_list.append(motif_entry)
        if len(motifs_list) > 20:
            del motifs_list[:-20]

        description = reason or (
            f"Adoption du motif fondateur «{label}»" if created else f"Renforcement du motif «{label}»"
        )

        motivation = max(0.55 if created else 0.35, abs(bias_value))
        quest = self._record_quest(
            f"Cultiver le motif «{label}»",
            reason=description,
            source=source_label,
            motivation=motivation,
            tags=normalized_tags + [label, "motif"],
            scope="long_term",
            anchor_metadata={
                "concept": label,
                "occurrences": occ_value,
                "bias": bias_value,
            },
        )

        event_tags = list(dict.fromkeys(normalized_tags + [label, "motif", "long_term"]))
        impact = max(0.4 if created else 0.25, abs(bias_value))
        event = {
            "ts": now,
            "kind": "motif" if created else "motif_update",
            "concept": label,
            "description": description,
            "tags": event_tags,
            "source": source_label,
            "bias": bias_value,
            "impact": impact,
        }
        event["genes"] = self.genetics.to_features()
        self._append_event(event)
        self._advance_arcs(event_tags, impact)

        return {
            "motif": motif_entry,
            "quest": quest,
            "event": event,
            "created": created,
        }

    def record_event(
        self,
        *,
        description: str,
        tags: Optional[List[str]] = None,
        impact: float = 0.3,
        origin: Optional[str] = None,
        bias: Optional[float] = None,
    ) -> Dict[str, Any]:
        if not isinstance(tags, list):
            tags = [tag for tag in (tags or []) if tag]
        event = {
            "ts": time.time(),
            "kind": "event",
            "description": description,
            "tags": tags,
            "impact": impact,
            "origin": origin,
            "bias": bias if bias is not None else impact,
        }
        event["genes"] = self.genetics.to_features()
        quest = None
        if impact >= 0.5:
            short_desc = description if len(description) <= 80 else f"{description[:77]}..."
            quest = self._record_quest(
                f"Intégrer l'expérience «{short_desc}»",
                reason="Événement marquant",
                source="event",
                motivation=min(1.0, impact),
                tags=tags,
            )
        event["quest"] = quest
        llm_response = try_call_llm_dict(
            "life_story",
            input_payload={
                "event": event,
                "quest": quest,
                "recent_timeline": self.data.get("timeline", [])[-5:],
            },
            logger=LOGGER,
            max_retries=2,
        )
        if llm_response:
            episode = llm_response.get("episode")
            if isinstance(episode, Mapping):
                event["episode"] = dict(episode)
                timeline = self.data.setdefault("timeline", [])
                timeline.append(dict(episode))
                if len(timeline) > self.MAX_TIMELINE:
                    del timeline[:-self.MAX_TIMELINE]
        self._append_event(event)
        self._advance_arcs(tags, impact)
        return event

    def foundation_goals(self) -> List[str]:
        anchors = self.data.get("anchors", {})
        goals = anchors.get("story_goals", [])
        if isinstance(goals, list):
            return [str(goal) for goal in goals if goal]
        return []

    def to_features(self) -> Dict[str, float]:
        momentum = self.data.get("momentum", {})
        arcs = self.data.get("arcs", [])
        primary_progress = 0.0
        if arcs:
            try:
                primary_progress = float(arcs[0].get("progress", 0.0))
            except (TypeError, ValueError):
                primary_progress = 0.0
        features = {
            "story_momentum": float(momentum.get("recent_intensity", 0.0) or 0.0),
            "story_coherence": float(momentum.get("coherence", 0.0) or 0.0),
            "story_progress": _clamp(primary_progress, 0.0, 1.0),
        }
        awakening = self.data.get("origin", {}).get("awakening", {})
        checkpoints = awakening.get("checkpoints", [])
        if checkpoints:
            total = len(checkpoints)
            answered = sum(1 for checkpoint in checkpoints if checkpoint.get("state") == "answered")
            progress = answered / float(total)
            pending_ratio = (total - answered) / float(total)
            last_prompt = float(awakening.get("last_prompt_ts") or 0.0)
            if last_prompt:
                elapsed = max(0.0, time.time() - last_prompt)
                recency = math.exp(-elapsed / 300.0)
            else:
                recency = 0.0
            features["awakening_progress"] = _clamp(progress, 0.0, 1.0)
            features["awakening_pending_ratio"] = _clamp(pending_ratio, 0.0, 1.0)
            features["awakening_prompt_recency"] = recency
        else:
            features["awakening_progress"] = 1.0
            features["awakening_pending_ratio"] = 0.0
            features["awakening_prompt_recency"] = 0.0
        return features

    def to_summary(self, max_events: int = 5) -> Dict[str, Any]:
        timeline = self.data.get("timeline", [])
        recent = timeline[-max_events:] if max_events > 0 else list(timeline)
        quest_log = self.data.get("quest_log", [])
        quests = quest_log[-max_events:] if max_events > 0 else list(quest_log)
        now = time.time()
        return {
            "origin": dict(self.data.get("origin", {})),
            "anchors": dict(self.data.get("anchors", {})),
            "recent_events": list(recent),
            "momentum": dict(self.data.get("momentum", {})),
            "arcs": list(self.data.get("arcs", [])),
            "quests": list(quests),
            "awakening": self.awakening_status(now=now),
        }

    # ------------------------------------------------------------------
    def _append_event(self, event: Dict[str, Any]) -> None:
        timeline = self.data.setdefault("timeline", [])
        timeline.append(event)
        if len(timeline) > self.MAX_TIMELINE:
            del timeline[:-self.MAX_TIMELINE]
        self._recompute_momentum()

    def _recompute_momentum(self) -> None:
        timeline = self.data.get("timeline", [])
        recent = timeline[-10:]
        if recent:
            total = 0.0
            for event in recent:
                try:
                    total += abs(float(event.get("bias", event.get("impact", 0.0)) or 0.0))
                except (TypeError, ValueError):
                    continue
            avg = total / max(len(recent), 1)
        else:
            avg = 0.0
        momentum = self.data.setdefault("momentum", {})
        momentum["recent_intensity"] = avg
        arcs = self.data.get("arcs", [])
        if arcs:
            try:
                progress = float(arcs[0].get("progress", 0.0))
            except (TypeError, ValueError):
                progress = 0.0
            momentum["coherence"] = 0.4 + 0.6 * _clamp(progress, 0.0, 1.0)
        momentum["events_recorded"] = len(timeline)
        self.data.setdefault("summary", {})["last_update"] = time.time()

    def _advance_arcs(self, tags: Iterable[str], amount: float) -> None:
        if amount <= 0:
            return
        arcs = self.data.setdefault("arcs", [])
        now = time.time()
        observed = set(tag for tag in tags if tag)
        for arc in arcs:
            focus = arc.get("focus_tags", [])
            if not isinstance(focus, Iterable) or isinstance(focus, (str, bytes)):
                focus = []
            weight = 1.0
            if focus:
                if observed.intersection(set(focus)):
                    weight = 1.2
                else:
                    weight = 0.4
            try:
                progress = float(arc.get("progress", 0.0))
            except (TypeError, ValueError):
                progress = 0.0
            progress = _clamp(progress + 0.05 * amount * weight, 0.0, 1.0)
            arc["progress"] = progress
            arc["last_update"] = now


__all__ = [
    "GeneticProfile",
    "LifeStoryManager",
    "ensure_genetics_structure",
    "ensure_story_structure",
]
