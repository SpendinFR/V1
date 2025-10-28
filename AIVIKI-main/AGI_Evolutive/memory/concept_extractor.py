from __future__ import annotations

import glob
import json
import logging
import math
import os
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from AGI_Evolutive.utils.jsonsafe import json_sanitize
from AGI_Evolutive.utils.llm_service import (
    LLMIntegrationError,
    LLMUnavailableError,
    get_llm_manager,
    is_llm_enabled,
)

from .concept_store import ConceptStore

BASIC_STOP = set(
    """
    a au aux avec c ce ces ça dans de des du elle elles en et eux il ils je la le les leur lui ma mais me même mes
    moi mon ne nos notre nous on ou où par pas pour qu que qui sa se ses son sur ta te tes toi ton tu un une vos votre
    vous y d l n t s qu' j' c' m' n' t' s' auj aujourd'hui puis donc alors comme si car or ni soit été être suis es est
    sont étais étions étiez étaient sera seront serai serons serez serais seraient étais étiez étaient
    the of and to a in is it that for on as with this these from by at or an be are was were i you he she they we me him her them my your our their
    not do does did have has had can could will would should into about over more no so than then very just
    """.split()
)

LOGGER = logging.getLogger(__name__)


def _llm_enabled() -> bool:
    return is_llm_enabled()


def _llm_manager():
    return get_llm_manager()

CAP_WORD = re.compile(r"\b[A-ZÉÈÀÂÎÔÙÛÇ][\w\-']+\b")
TOKEN = re.compile(r"[a-z0-9àâäéèêëîïôöùûüç']{3,}")

_RELATION_SYNONYMS = {
    "support": "supports",
    "supports": "supports",
    "soutien": "supports",
    "soutient": "supports",
    "renforce": "supports",
    "encourage": "supports",
    "favorise": "supports",
    "cause": "causes",
    "entraîne": "causes",
    "provoque": "causes",
    "leads": "causes",
    "implies": "causes",
    "contradiction": "contradicts",
    "contredit": "contradicts",
    "oppose": "contradicts",
    "refers": "refers_to",
    "reference": "refers_to",
    "référence": "refers_to",
    "associe": "related_to",
    "relie": "related_to",
    "related": "related_to",
    "link": "related_to",
    "appartient": "part_of",
    "part": "part_of",
}


def _normalize_relation_label(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    lowered = str(value).strip().lower()
    if not lowered:
        return None
    mapped = _RELATION_SYNONYMS.get(lowered)
    if mapped:
        return mapped
    normalized = lowered.replace(" ", "_")
    return normalized or None
COPULA_PATTERN = re.compile(
    r"\b(est|étais|sera|sont|suis|serai|seront)\s+(?:un|une|le|la|l'|des|de la)\s+([\w\-']{3,})",
    re.IGNORECASE,
)
NOUN_PATTERN = re.compile(r"\b([\w\-']{3,})\b", re.UNICODE)


@dataclass(frozen=True)
class Profile:
    """Configuration used by the adaptive bandit."""

    identifier: str
    k: int = 6
    df_decay: float = 0.0
    idf_floor: float = 0.0
    support_gain: float = 1.0
    salience_gain: float = 0.1
    relation_gain: float = 0.1
    confidence: float = 0.6
    bigram_weight: float = 1.0
    min_concepts: int = 1
    fallback_weight: float = 0.8
    max_sources: int = 8


DEFAULT_PROFILES: Tuple[Profile, ...] = (
    Profile("baseline"),
    Profile(
        "explore_bigram",
        k=8,
        bigram_weight=1.4,
        idf_floor=2.0,
        fallback_weight=1.2,
    ),
    Profile(
        "fast_decay",
        df_decay=0.05,
        support_gain=1.2,
        salience_gain=0.18,
        relation_gain=0.2,
    ),
    Profile(
        "precision",
        k=5,
        idf_floor=1.0,
        min_concepts=2,
        confidence=0.7,
    ),
)


def _now() -> float:
    return time.time()


def _safe_json_load(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return default


def _safe_jsonl_append(path: str, obj: Dict[str, Any]) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_sanitize(obj), ensure_ascii=False) + "\n")


def _normalize_text(value: Any) -> Optional[str]:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, dict):
        for key in ("content", "text", "summary", "description", "message", "label", "concept", "value"):
            candidate = value.get(key)
            normalized = _normalize_text(candidate)
            if normalized:
                return normalized
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            normalized = _normalize_text(item)
            if normalized:
                return normalized
        return None
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _memory_to_text(memory: Dict[str, Any]) -> Optional[str]:
    for key in ("content", "text", "summary", "message", "body"):
        if key in memory:
            normalized = _normalize_text(memory.get(key))
            if normalized:
                return normalized
    metadata = memory.get("metadata")
    if isinstance(metadata, dict):
        normalized = _normalize_text(metadata)
        if normalized:
            return normalized
    return None


class ConceptExtractor:
    """Concept detection pipeline shared across orchestrator and semantic manager."""

    def __init__(
        self,
        memory_store: Optional[Any],
        data_path: str = "data/concepts.json",
    ) -> None:
        concept_events_path: Optional[str]
        if not data_path:
            self.data_dir = "."
            concept_events_path = os.path.join(self.data_dir, "concept_events.jsonl")
        elif data_path.endswith(".jsonl"):
            base_dir = os.path.dirname(data_path)
            self.data_dir = base_dir if base_dir else "."
            concept_events_path = data_path
        elif data_path.endswith(".json"):
            base_dir = os.path.dirname(data_path)
            self.data_dir = base_dir if base_dir else "."
            concept_events_path = os.path.join(self.data_dir, "concept_events.jsonl")
        else:
            self.data_dir = data_path
            concept_events_path = os.path.join(self.data_dir, "concept_events.jsonl")

        os.makedirs(self.data_dir, exist_ok=True)
        self.paths = {
            "concept_index": os.path.join(self.data_dir, "concept_index.json"),
            "concept_graph": os.path.join(self.data_dir, "concept_graph.json"),
            "concept_events": concept_events_path,
            "state": os.path.join(self.data_dir, "concept_state.json"),
        }
        self.store: Optional[ConceptStore] = None
        self.bound: Dict[str, Optional[Any]] = {
            "memory": memory_store,
            "emotions": None,
            "metacog": None,
            "language": None,
        }

        self.state = _safe_json_load(
            self.paths["state"],
            {
                "last_run": 0.0,
                "processed_ids": [],
                "df": {},
                "total_docs": 0.0,
                "profile_stats": {},
                "score_history": [],
                "classifier": {"weights": {}, "bias": 0.0},
            },
        )
        self.state.setdefault("profile_stats", {})
        self.state.setdefault("score_history", [])
        self.state.setdefault("classifier", {"weights": {}, "bias": 0.0})
        self.state.setdefault("df", {})
        self.state.setdefault("total_docs", 0.0)
        self.concept_index: Dict[str, Dict[str, Any]] = _safe_json_load(
            self.paths["concept_index"],
            {},
        )
        self.graph = _safe_json_load(
            self.paths["concept_graph"],
            {"nodes": {}, "edges": {}},
        )

        self.period_s = 15.0
        self._last_step = 0.0
        self._last_batch_score = 0.0
        self._last_llm_feedback: Optional[Mapping[str, Any]] = None
        self._last_llm_suggestions: Optional[List[Tuple[str, float]]] = None
        self._last_llm_relations: Optional[List[Tuple[str, str, str, float]]] = None

    # ---------- binding ----------
    def bind(self, memory: Any = None, emotions=None, metacog=None, language=None) -> None:
        if memory is not None:
            self.bound["memory"] = memory
        self.bound.update({"emotions": emotions, "metacog": metacog, "language": language})

    # ---------- scheduling helpers ----------
    def step(self, memory: Any = None, max_batch: int = 300) -> None:
        if memory is not None:
            self.bound["memory"] = memory
        memory_system = memory if memory is not None else self.bound.get("memory")
        if memory_system is None:
            return
        now = time.time()
        if now - self._last_step < self.period_s:
            return
        self._last_step = now
        self._run_batch(self._collect_memories(memory_system, limit=max_batch))

    def run_once(self, max_batch: int = 400) -> None:
        mems = self._collect_memories(self.bound.get("memory"), limit=max_batch)
        self._run_batch(mems)

    def extract_from_recent(self, n: int = 200) -> List[Dict[str, Any]]:
        """Compatibility helper used by the orchestrator scheduler."""
        mems = self._collect_memories(self.bound.get("memory"), limit=n)
        aggregated = self._run_batch(mems)
        result = [
            {"concept": concept, "score": score}
            for concept, score in aggregated
        ]
        return result

    def process_memories(self, memories: Iterable[Mapping[str, Any]]) -> List[Tuple[str, float]]:
        """Process a batch immediately (used by memory store hooks)."""

        batch: List[Dict[str, Any]] = []
        for memory in memories:
            if isinstance(memory, Mapping):
                batch.append(dict(memory))
        if not batch:
            return []
        return self._run_batch(batch)

    # ---------- core processing ----------
    def _run_batch(self, memories: Iterable[Dict[str, Any]]) -> List[Tuple[str, float]]:
        docs = list(self._prepare_documents(memories))
        if not docs:
            return []

        profile = self._select_profile()
        self._apply_decay(profile.df_decay)

        aggregated: Counter[str] = Counter()
        ids_in_batch: List[str] = []

        for mem_id, text, meta in docs:
            concepts = self._extract_concepts(text, meta, profile)
            llm_feedback = self._last_llm_feedback
            if not concepts:
                continue
            tf = Counter(concepts)
            uniq_concepts = set(tf.keys())
            self.state["total_docs"] += 1.0
            for concept in uniq_concepts:
                current = float(self.state["df"].get(concept, 0.0))
                self.state["df"][concept] = current + 1.0

            top = self._score_select(
                tf,
                self.state["df"],
                float(self.state["total_docs"]),
                profile=profile,
            )
            for concept, score in top:
                aggregated[concept] += score

            self._update_index(top, mem_id, text, profile)
            relations_for_mem = list(self._last_llm_relations or [])
            self._update_graph(top, profile, relations=relations_for_mem)
            self._log_event(mem_id, meta, top, profile, llm_feedback)
            self._update_store(mem_id, top, profile, relations=relations_for_mem)
            self._log_memory(mem_id, top)
            self._update_classifier(top)

            if mem_id:
                ids_in_batch.append(mem_id)

            self._last_llm_feedback = None
            self._last_llm_relations = None

        self.state["processed_ids"] += ids_in_batch
        if len(self.state["processed_ids"]) > 5000:
            self.state["processed_ids"] = self.state["processed_ids"][-2500:]

        self._save_state()
        top_global = aggregated.most_common(20)
        batch_score = sum(score for _, score in top_global) / max(1, len(top_global))
        self._update_profile_stats(profile.identifier, batch_score)
        self._last_batch_score = float(batch_score)
        self._update_score_history(batch_score)
        return top_global

    # ---------- concept helpers ----------
    def _extract_concepts(self, text: str, meta: Dict[str, Any], profile: Profile) -> List[str]:
        self._last_llm_feedback = None
        self._last_llm_suggestions = None
        self._last_llm_relations = None
        lang = self._detect_language(text, meta)
        lowered = text.lower()
        dynamic_stop = self._language_stopwords(lang)
        tokens = [token for token in TOKEN.findall(lowered) if token not in dynamic_stop]

        caps = [match.group(0) for match in CAP_WORD.finditer(text)]
        caps_norm = [cap.strip("'").lower() for cap in caps if len(cap) >= 3]

        bigrams: List[str] = []
        for idx in range(len(tokens) - 1):
            w1, w2 = tokens[idx], tokens[idx + 1]
            if w1 in dynamic_stop or w2 in dynamic_stop:
                continue
            bigram = f"{w1} {w2}"
            bigrams.append(bigram)

        concepts = list(tokens)
        if profile.bigram_weight:
            concepts.extend(bigrams * max(1, int(profile.bigram_weight)))
        concepts.extend(caps_norm)

        if len(concepts) < profile.min_concepts:
            fallback = self._fallback_concepts(text, lang)
            if fallback:
                weight = max(1, int(profile.fallback_weight))
                concepts.extend(fallback * weight)

        self._augment_with_llm(concepts, text, meta, profile)

        return concepts

    def _augment_with_llm(
        self,
        concepts: List[str],
        text: str,
        meta: Dict[str, Any],
        profile: Profile,
    ) -> None:
        if not _llm_enabled():
            return

        should_trigger = len(concepts) < max(profile.min_concepts * 2, 6)
        if not should_trigger:
            return

        payload = {
            "text": text,
            "metadata": meta.get("metadata", {}),
            "kind": meta.get("kind"),
            "tags": meta.get("tags"),
        }
        try:
            response = _llm_manager().call_dict("concept_extraction", input_payload=payload)
        except (LLMUnavailableError, LLMIntegrationError):
            LOGGER.debug("LLM concept extraction unavailable", exc_info=True)
            return

        if not isinstance(response, Mapping):
            return

        suggestions: List[Tuple[str, float]] = []
        for entry in response.get("concepts", []):
            if not isinstance(entry, Mapping):
                continue
            term = str(entry.get("term", "")).strip()
            if not term:
                continue
            try:
                raw_score = entry.get("score", 0.7)
                score = float(raw_score)
            except (TypeError, ValueError):
                score = 0.7
            score = max(0.2, min(1.0, score))
            suggestions.append((term, score))

        relations_payload: List[Tuple[str, str, str, float]] = []

        relations_raw = response.get("relations")
        if isinstance(relations_raw, Sequence):
            for entry in relations_raw:
                if not isinstance(entry, Mapping):
                    continue
                subject = str(entry.get("subject") or entry.get("from") or "").strip()
                obj = str(entry.get("object") or entry.get("to") or "").strip()
                rel = _normalize_relation_label(entry.get("verb") or entry.get("relation"))
                if not subject or not obj:
                    continue
                try:
                    confidence = float(entry.get("confidence", entry.get("weight", 0.6)))
                except (TypeError, ValueError):
                    confidence = 0.6
                confidence = max(0.1, min(1.0, confidence))
                relations_payload.append((subject.lower(), rel or "related_to", obj.lower(), confidence))

        if not suggestions and not relations_payload:
            self._last_llm_feedback = dict(response)
            return

        self._last_llm_suggestions = suggestions or None
        self._last_llm_relations = relations_payload or None

        feedback = dict(response)
        if suggestions:
            feedback.setdefault("suggestions", suggestions)
        if relations_payload:
            feedback.setdefault("relations", relations_payload)
        self._last_llm_feedback = feedback

        base_weight = max(1, int(round(profile.support_gain or 1.0)))
        for term, score in suggestions:
            normalized = term.lower().strip()
            if not normalized:
                continue
            multiplier = max(1, int(round(1 + score * (1 + base_weight))))
            concepts.extend([normalized] * multiplier)

    def _score_select(
        self,
        tf: Counter,
        df: Dict[str, int],
        total_docs: float,
        profile: Profile,
    ) -> List[Tuple[str, float]]:
        scored: List[Tuple[str, float]] = []
        for term, freq in tf.items():
            denom = max(1.0, float(df.get(term, 1.0)))
            idf = math.log(1.0 + ((total_docs + profile.idf_floor) / denom))
            scored.append((term, freq * idf))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: profile.k]

    def _update_index(
        self,
        top: List[Tuple[str, float]],
        mem_id: Optional[str],
        text: str,
        profile: Profile,
    ) -> None:
        now = _now()
        for concept, _ in top:
            entry = self.concept_index.get(
                concept,
                {
                    "count": 0,
                    "first_seen": now,
                    "last_seen": now,
                    "sources": [],
                    "sample": "",
                },
            )
            entry["count"] += 1
            entry["last_seen"] = now
            if mem_id and len(entry["sources"]) < profile.max_sources:
                entry["sources"].append(mem_id)
            if not entry["sample"]:
                entry["sample"] = text[:240]
            self.concept_index[concept] = entry

        with open(self.paths["concept_index"], "w", encoding="utf-8") as handle:
            json.dump(json_sanitize(self.concept_index), handle, ensure_ascii=False, indent=2)

    def _update_graph(
        self,
        top: List[Tuple[str, float]],
        profile: Profile,
        *,
        relations: Optional[Sequence[Tuple[str, str, str, float]]] = None,
    ) -> None:
        if not top:
            return
        for concept, _ in top:
            node = self.graph["nodes"].get(concept, {"count": 0})
            node["count"] += 1
            self.graph["nodes"][concept] = node

        concepts = [concept for concept, _ in top]
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                key = f"{concepts[i]}|||{concepts[j]}"
                edge = self.graph["edges"].get(key, {"w": 0.0})
                edge["w"] += profile.relation_gain
                self.graph["edges"][key] = edge

        if relations:
            for subject, rel, obj, confidence in relations:
                if not subject or not obj:
                    continue
                label = rel or "related_to"
                key = f"{subject}|||{label}|||{obj}"
                edge = self.graph["edges"].get(key, {"w": 0.0})
                gain = profile.relation_gain * (1.0 + 0.5 * confidence)
                edge.update(
                    {
                        "w": edge.get("w", 0.0) + gain,
                        "type": label,
                        "directed": True,
                        "confidence": confidence,
                    }
                )
                self.graph["edges"][key] = edge

        with open(self.paths["concept_graph"], "w", encoding="utf-8") as handle:
            json.dump(json_sanitize(self.graph), handle, ensure_ascii=False)

    def _log_event(
        self,
        mem_id: Optional[str],
        meta: Dict[str, Any],
        top: List[Tuple[str, float]],
        profile: Profile,
        llm_feedback: Optional[Mapping[str, Any]] = None,
    ) -> None:
        avg_score = sum(score for _, score in top) / max(1, len(top))
        _safe_jsonl_append(
            self.paths["concept_events"],
            {
                "t": _now(),
                "memory_id": mem_id,
                "top_concepts": top,
                "kind": meta.get("kind"),
                "meta": meta.get("metadata", {}),
                "profile": profile.identifier,
                "avg_score": avg_score,
                "llm": json_sanitize(llm_feedback) if llm_feedback else None,
            },
        )

    def _log_memory(self, mem_id: Optional[str], top: List[Tuple[str, float]]) -> None:
        memory = self.bound.get("memory")
        if mem_id and memory and hasattr(memory, "add_memory") and top:
            try:
                memory.add_memory(
                    kind="concept_extracted",
                    content="; ".join(concept for concept, _ in top),
                    metadata={"source_memory": mem_id, "concepts": top},
                )
            except Exception:
                pass

    def _update_store(
        self,
        mem_id: Optional[str],
        top: List[Tuple[str, float]],
        profile: Profile,
        *,
        relations: Optional[Sequence[Tuple[str, str, str, float]]] = None,
    ) -> None:
        if not self.store or not top:
            return
        max_score = top[0][1] if top else 1.0
        emotion_boost = self._emotion_factor()
        concept_ids: Dict[str, str] = {}
        for concept, score in top:
            support_delta = profile.support_gain * (float(score / max_score) if max_score else 0.0)
            salience_delta = (
                profile.salience_gain * (float(score / max_score) if max_score else 0.0) * emotion_boost
            )
            obj = self.store.upsert_concept(
                label=concept,
                support_delta=support_delta,
                salience_delta=salience_delta,
                example_mem_id=mem_id,
                confidence=profile.confidence,
            )
            concept_ids[concept] = obj.id
        concepts = [concept for concept, _ in top]
        relation_gain = profile.relation_gain * emotion_boost
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                self.store.upsert_relation(
                    concept_ids.get(concepts[i]) or concepts[i],
                    concept_ids.get(concepts[j]) or concepts[j],
                    "related_to",
                    weight_delta=relation_gain,
                    mem_id=mem_id,
                    confidence=profile.confidence,
                )
        if relations:
            def ensure(label: str, base_conf: float) -> str:
                cached = concept_ids.get(label)
                if cached:
                    return cached
                obj = self.store.upsert_concept(
                    label=label,
                    support_delta=profile.support_gain * 0.3,
                    salience_delta=profile.salience_gain * 0.3 * emotion_boost,
                    example_mem_id=mem_id,
                    confidence=max(base_conf, 0.3),
                )
                concept_ids[label] = obj.id
                return obj.id

            for subject, rel, obj_label, confidence in relations:
                if not subject or not obj_label:
                    continue
                src_id = ensure(subject, profile.confidence)
                dst_id = ensure(obj_label, profile.confidence * 0.9)
                weight = relation_gain * (0.5 + 0.5 * confidence)
                self.store.upsert_relation(
                    src_id,
                    dst_id,
                    rel or "related_to",
                    weight_delta=weight,
                    mem_id=mem_id,
                    confidence=max(profile.confidence, confidence),
                )
        self.store.save()

    def _emotion_factor(self) -> float:
        emotions = self.bound.get("emotions")
        if emotions and hasattr(emotions, "get_state"):
            try:
                state = emotions.get_state() or {}
                valence = float(state.get("valence", 0.0))
                arousal = float(state.get("arousal", 0.0))
                return max(0.6, min(1.6, 1.0 + 0.4 * valence + 0.2 * arousal))
            except Exception:
                return 1.0
        return 1.0

    # ---------- persistence helpers ----------
    def _save_state(self) -> None:
        self.state["last_run"] = _now()
        with open(self.paths["state"], "w", encoding="utf-8") as handle:
            json.dump(json_sanitize(self.state), handle, ensure_ascii=False, indent=2)

    # ---------- adaptive helpers ----------
    def _select_profile(self) -> Profile:
        stats: Dict[str, Dict[str, float]] = self.state.setdefault("profile_stats", {})
        chosen: Optional[Profile] = None
        best_sample = float("-inf")
        for profile in DEFAULT_PROFILES:
            record = stats.setdefault(
                profile.identifier,
                {"success": 1.0, "failure": 1.0, "avg": 0.0},
            )
            success = max(1.0, float(record.get("success", 1.0)))
            failure = max(1.0, float(record.get("failure", 1.0)))
            sample = random.betavariate(success, failure)
            if sample > best_sample:
                best_sample = sample
                chosen = profile
        return chosen or DEFAULT_PROFILES[0]

    def _update_profile_stats(self, profile_id: str, batch_score: float) -> None:
        stats: Dict[str, Dict[str, float]] = self.state.setdefault("profile_stats", {})
        record = stats.setdefault(profile_id, {"success": 1.0, "failure": 1.0, "avg": 0.0})
        moving = record.get("avg", 0.0)
        updated = moving * 0.8 + batch_score * 0.2
        record["avg"] = updated
        baseline = self._score_baseline()
        if batch_score >= baseline:
            record["success"] = float(record.get("success", 1.0)) + 1.0
        else:
            record["failure"] = float(record.get("failure", 1.0)) + 1.0
        stats[profile_id] = record

    def _score_baseline(self) -> float:
        history: List[float] = self.state.setdefault("score_history", [])
        if not history:
            return 0.0
        return sum(history[-10:]) / min(len(history), 10)

    def _update_score_history(self, batch_score: float) -> None:
        history: List[float] = self.state.setdefault("score_history", [])
        history.append(float(batch_score))
        if len(history) > 200:
            del history[: len(history) - 200]

    def _apply_decay(self, df_decay: float) -> None:
        if not df_decay:
            return
        df = self.state.setdefault("df", {})
        for concept in list(df.keys()):
            df[concept] = float(df[concept]) * (1.0 - df_decay)
            if df[concept] < 0.01:
                df.pop(concept, None)
        total_docs = float(self.state.get("total_docs", 0.0))
        self.state["total_docs"] = max(0.0, total_docs * (1.0 - df_decay))

    def _update_classifier(self, top: List[Tuple[str, float]]) -> None:
        if not top:
            return
        classifier = self.state.setdefault("classifier", {"weights": {}, "bias": 0.0})
        weights: Dict[str, float] = classifier.setdefault("weights", {})
        bias = float(classifier.get("bias", 0.0))
        lr = 0.05
        for concept, score in top:
            weights[concept] = float(weights.get(concept, 0.0)) + lr * float(score)
        classifier["bias"] = bias + lr * len(top)
        # shrink weights to avoid runaway growth
        for concept in list(weights.keys()):
            weights[concept] *= 0.995
            if abs(weights[concept]) < 1e-3:
                weights.pop(concept, None)
        self.state["classifier"] = classifier

    def _classifier_score(self, token: str) -> float:
        classifier = self.state.setdefault("classifier", {"weights": {}, "bias": 0.0})
        weights: Dict[str, float] = classifier.setdefault("weights", {})
        bias = float(classifier.get("bias", 0.0))
        return float(weights.get(token, 0.0)) + bias

    # ---------- language helpers ----------
    def _detect_language(self, text: str, meta: Optional[Dict[str, Any]] = None) -> str:
        meta_lang = None
        if meta:
            meta_lang = meta.get("language") or meta.get("lang")
            if isinstance(meta_lang, str):
                meta_lang = meta_lang.lower()
        if meta_lang in {"fr", "fr_fr", "french"}:
            return "fr"
        if meta_lang in {"en", "en_us", "english"}:
            return "en"
        lowered = text.lower()
        accented = sum(1 for char in lowered if char in "àâäéèêëîïôöùûüç")
        ascii_letters = sum(1 for char in lowered if "a" <= char <= "z")
        if accented > 2 and accented > ascii_letters * 0.1:
            return "fr"
        if ascii_letters >= len(lowered) * 0.2:
            return "en"
        return "mixed"

    def _language_stopwords(self, lang: str) -> Set[str]:
        if lang == "fr":
            return BASIC_STOP
        if lang == "en":
            return BASIC_STOP
        return BASIC_STOP

    # ---------- fallback extraction ----------
    def _fallback_concepts(self, text: str, lang: str) -> List[str]:
        concepts: List[str] = []
        for match in COPULA_PATTERN.finditer(text):
            candidate = match.group(2)
            if candidate:
                concepts.append(candidate.lower())
        if concepts:
            return concepts
        # fallback on noun extraction with simple heuristics
        tokens = [token.lower() for token in NOUN_PATTERN.findall(text) if len(token) >= 4]
        seen: Dict[str, int] = {}
        for token in tokens:
            seen[token] = seen.get(token, 0) + 1
        scored_tokens = [
            (token, freq + self._classifier_score(token)) for token, freq in seen.items()
        ]
        scored_tokens.sort(key=lambda item: item[1], reverse=True)
        return [token for token, _ in scored_tokens[:3]]

    # ---------- memory access ----------
    def _collect_memories(self, memory_system: Any, limit: int) -> List[Dict[str, Any]]:
        if memory_system is None:
            return self._fallback_read_files(limit)
        try:
            if hasattr(memory_system, "get_recent_memories"):
                res = memory_system.get_recent_memories(n=limit)
                if res:
                    return list(res)
        except Exception:
            pass
        try:
            if hasattr(memory_system, "iter_memories"):
                result: List[Dict[str, Any]] = []
                for item in memory_system.iter_memories():
                    result.append(item)
                    if len(result) >= limit:
                        break
                if result:
                    return result
        except Exception:
            pass
        return self._fallback_read_files(limit)

    def _fallback_read_files(self, limit: int) -> List[Dict[str, Any]]:
        root = os.path.join(self.data_dir, "memories")
        if not os.path.isdir(root):
            return []
        files = sorted(glob.glob(os.path.join(root, "*.json")), reverse=True)
        out: List[Dict[str, Any]] = []
        for path in files[:limit]:
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    out.append(json.load(handle))
            except Exception:
                continue
        return out

    def _prepare_documents(self, memories: Iterable[Dict[str, Any]]) -> Iterable[Tuple[Optional[str], str, Dict[str, Any]]]:
        for memory in memories:
            mem_id = memory.get("id") or memory.get("_id") or memory.get("memory_id")
            if mem_id and mem_id in self.state["processed_ids"]:
                continue
            text = _memory_to_text(memory)
            if not text:
                continue
            yield mem_id, text, memory

    # ---------- diagnostics ----------
    def pending_backlog(self, memory: Any = None, max_batch: int = 400) -> int:
        """Estimate the number of unprocessed memories awaiting extraction."""

        source = memory if memory is not None else self.bound.get("memory")
        candidates = list(self._prepare_documents(self._collect_memories(source, limit=max_batch)))
        return len(candidates)

    @property
    def last_batch_score(self) -> float:
        return float(getattr(self, "_last_batch_score", 0.0))

    @property
    def last_llm_feedback(self) -> Optional[Mapping[str, Any]]:
        return self._last_llm_feedback

    def quality_signal(self, window: int = 5) -> float:
        """Return a normalized quality indicator derived from recent batches."""

        history: List[float] = self.state.get("score_history", [])
        if not history:
            return 0.0
        window = max(1, min(window, len(history)))
        recent = history[-window:]
        baseline = max(max(history[-20:]), 1e-6)
        avg_recent = sum(recent) / float(window)
        normalized = max(0.0, min(1.0, avg_recent / baseline))
        return normalized
