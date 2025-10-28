
# core/document_ingest.py
"""
DocumentIngest: intègre les documents de ./inbox dans la mémoire.
- Parse des fichiers texte/markdown/json (binaire -> ignoré)
- Crée des traces mnésiques épisodiques avec associations légères
- Hook simple à appeler dans la boucle
"""
import logging
import os
import time
import glob
from typing import Dict, Any, Iterable

from AGI_Evolutive.core.config import cfg
from AGI_Evolutive.knowledge.concept_recognizer import ConceptRecognizer
from AGI_Evolutive.memory import MemoryType
from AGI_Evolutive.utils.llm_service import try_call_llm_dict


LOGGER = logging.getLogger(__name__)

def _hash(s: str) -> str:
    import hashlib
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def _summarize_text(text: str, *, max_lines: int = 12, max_chars: int = 2000) -> str:
    """Heuristic summary used when a document adds little novelty."""
    if not text:
        return ""
    lines = []
    for raw in text.splitlines():
        cleaned = raw.strip()
        if cleaned:
            lines.append(cleaned)
        if len(lines) >= max_lines:
            break
    summary = " ".join(lines)
    if not summary:
        summary = text[:max_chars]
    return summary[:max_chars]


def _document_evidence_job(ctx, args):
    """Background maintenance of document traces based on usage evidence."""
    arch = args.get("arch")
    memory_id = args.get("memory_id")
    memory_type_name = args.get("memory_type")
    now = time.time()
    if not arch or not memory_id or not memory_type_name:
        return {"status": "skipped", "reason": "missing_args"}
    memory = getattr(arch, "memory", None)
    if memory is None:
        return {"status": "skipped", "reason": "no_memory"}
    try:
        mem_type = MemoryType[memory_type_name]
    except Exception:
        mem_type = getattr(memory, "MemoryType", MemoryType).EPISODIC

    trace = None
    try:
        trace = memory.long_term_memory.get(mem_type, {}).get(memory_id)
    except Exception:
        trace = None
    if trace is None:
        return {"status": "skipped", "reason": "missing_trace"}

    evidence = getattr(trace, "evidence", None)
    if isinstance(trace, dict):
        evidence = trace.setdefault("evidence", {})
    elif evidence is None:
        trace.evidence = {}
        evidence = trace.evidence

    usage = float(evidence.get("usage", 0))
    last_used = float(evidence.get("last_used_ts", 0) or 0)
    age = now - float(getattr(trace, "timestamp", evidence.get("ingested_ts", now)))
    idle = now - last_used if last_used else age

    if usage < 1 and idle > 48 * 3600:
        # Candidate for purge → gently decay strength and flag for review
        evidence["status"] = "stale"
        evidence["decayed_strength"] = float(getattr(trace, "strength", 0.6)) * 0.85
        if isinstance(trace, dict):
            trace["strength"] = evidence["decayed_strength"]
        else:
            trace.strength = evidence["decayed_strength"]
    else:
        evidence["status"] = "active"
        boost = min(0.1, 0.02 * usage)
        new_strength = min(1.0, float(getattr(trace, "strength", 0.6)) + boost)
        evidence["reinforced_strength"] = new_strength
        if isinstance(trace, dict):
            trace["strength"] = new_strength
        else:
            trace.strength = new_strength

    evidence["last_audit_ts"] = now
    try:
        meta = memory.memory_metadata.setdefault("document_ingest", {})
        audits = meta.setdefault("audits", [])
        audits.append({
            "memory_id": memory_id,
            "status": evidence.get("status"),
            "usage": usage,
            "idle": idle,
            "ts": now,
        })
        if len(audits) > 100:
            del audits[:-100]
    except Exception:
        pass
    return {"status": evidence.get("status"), "usage": usage, "idle": idle}


class DocumentIngest:
    def __init__(self, arch, inbox_dir: str | None = None):
        self.arch = arch
        data_dir = cfg().get("DATA_DIR", "data")
        fallback = os.path.join(data_dir, "inbox")
        self.inbox_dir = os.path.abspath(inbox_dir or fallback)
        os.makedirs(self.inbox_dir, exist_ok=True)
        self._index = {}  # filename -> last_hash
        self._concept_seen: set[str] = set()
    
    def scan(self) -> Dict[str, Any]:
        docs = {}
        for path in glob.glob(os.path.join(self.inbox_dir, "*")):
            name = os.path.basename(path)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                docs[name] = content
            except Exception:
                continue
        return docs
    
    def integrate(self):
        mem = getattr(self.arch, "memory", None)
        if mem is None:
            return 0
        docs = self.scan()
        added = 0
        for name, content in docs.items():
            try:
                if hasattr(self.arch, "style_observer"):
                    self.arch.style_observer.observe_text(
                        content, source=f"inbox:{name}", channel="inbox"
                    )
            except Exception:
                pass
            h = _hash(content[:10000])
            if self._index.get(name) == h:
                continue  # déjà intégré
            novelty_payload: Dict[str, Any] = {"candidates": 0, "new": 0, "ratio": 0.0, "labels": []}
            items: Iterable[Any] = []
            # --- DÉTECTION DE CONCEPTS, TERMES, STYLES ---
            try:
                arch = self.arch
                if not getattr(arch, "concept_recognizer", None):
                    arch.concept_recognizer = ConceptRecognizer(arch)
                dialog_hints = {}
                try:
                    hints = arch.memory.find_recent(kind="dialog_hints", since_sec=3600)
                    dialog_hints = hints or {}
                except Exception:
                    dialog_hints = {}
                items = arch.concept_recognizer.extract_candidates(content, dialog_hints=dialog_hints)
                arch.concept_recognizer.commit_candidates_to_memory(source=f"inbox:{name}", items=items, arch=arch)
                arch.concept_recognizer.autogoals_for_high_confidence(items, arch=arch)
                labels = sorted({getattr(item, "label", "").lower() for item in items if getattr(item, "label", None)})
                novelty_payload["candidates"] = len(labels)
                new_labels = [lab for lab in labels if lab not in self._concept_seen]
                novelty_payload["new"] = len(new_labels)
                if labels:
                    novelty_payload["ratio"] = round(len(new_labels) / max(1, len(labels)), 3)
                novelty_payload["labels"] = new_labels
                self._concept_seen.update(labels)
            except Exception:
                items = []

            llm_response = try_call_llm_dict(
                "document_ingest",
                input_payload={
                    "name": name,
                    "content": content[:6000],
                    "novelty": novelty_payload,
                },
                logger=LOGGER,
                max_retries=2,
            )
            llm_summary = None
            llm_tags = None
            llm_sections = None
            if llm_response:
                llm_summary = llm_response.get("summary")
                if isinstance(llm_summary, str) and llm_summary.strip():
                    llm_summary = llm_summary.strip()
                else:
                    llm_summary = None
                tags = llm_response.get("tags")
                if isinstance(tags, list):
                    llm_tags = [str(tag) for tag in tags if str(tag).strip()]
                sections = llm_response.get("critical_sections")
                if isinstance(sections, list):
                    llm_sections = [str(section) for section in sections if str(section).strip()]

            ingest_mode = "full"
            text_to_store = content[:200000]
            if novelty_payload["candidates"] and novelty_payload["ratio"] < 0.25 and len(content) > 4000:
                ingest_mode = "summary"
                text_to_store = _summarize_text(content)
            if llm_summary:
                ingest_mode = "llm_summary"
                text_to_store = llm_summary[:200000]

            trace_context = {
                "source": "inbox",
                "filename": name,
                "hash": h,
                "ingest_mode": ingest_mode,
            }
            if llm_sections:
                trace_context["critical_sections"] = llm_sections
            trace_content = {
                "text": text_to_store,
                "title": name,
                "source": f"inbox:{name}",
                "novelty": novelty_payload,
                "original_length": len(content),
                "ingest_mode": ingest_mode,
            }
            if llm_tags:
                trace_content.setdefault("metadata", {})["tags"] = llm_tags
            if llm_response:
                trace_content.setdefault("metadata", {})["llm"] = llm_response
            trace_content.setdefault("metadata", {})["hash"] = h
            trace_content["metadata"]["ingested_ts"] = time.time()

            memory_id = None
            trace = None
            try:
                memory_id = mem.encode_memory(
                    trace_content,
                    MemoryType.EPISODIC,
                    context=trace_context,
                    strength=0.6,
                    valence=0.0,
                )
                trace = mem.long_term_memory[MemoryType.EPISODIC][memory_id]
            except Exception:
                memory_id = None

            if memory_id is None:
                trace_id = f"doc::{name}::{h[:8]}"
                trace = {
                    "id": trace_id,
                    "content": trace_content,
                    "memory_type": getattr(mem, "MemoryType", MemoryType).EPISODIC if hasattr(mem, "MemoryType") else "ep",
                    "strength": 0.6,
                    "accessibility": 0.7,
                    "valence": 0.0,
                    "timestamp": time.time(),
                    "context": trace_context,
                    "associations": [],
                    "consolidation_state": "LABILE",
                    "last_accessed": time.time(),
                    "access_count": 0,
                    "evidence": {
                        "usage": 0,
                        "positive": 0,
                        "negative": 0,
                        "last_used_ts": None,
                        "ingested_ts": trace_content["metadata"]["ingested_ts"],
                    },
                }
                try:
                    LTM = mem.long_term_memory
                    mem_type = getattr(mem, "MemoryType", MemoryType).EPISODIC if hasattr(mem, "MemoryType") else MemoryType.EPISODIC
                    LTM.setdefault(mem_type, {})[trace_id] = trace
                    mem.memory_metadata["total_memories"] = mem.memory_metadata.get("total_memories", 0) + 1
                    memory_id = trace_id
                except Exception:
                    memory_id = trace_id
            else:
                evidence = {
                    "usage": 0,
                    "positive": 0,
                    "negative": 0,
                    "last_used_ts": None,
                    "ingested_ts": trace_content["metadata"]["ingested_ts"],
                }
                if isinstance(trace, dict):
                    trace["evidence"] = evidence
                else:
                    trace.evidence = evidence

            try:
                metrics = mem.memory_metadata.setdefault("document_ingest", {})
                history = metrics.setdefault("history", [])
                history.append({
                    "doc": name,
                    "hash": h,
                    "mode": ingest_mode,
                    "novelty": novelty_payload,
                    "ts": time.time(),
                    "memory_id": memory_id,
                })
                if len(history) > 50:
                    del history[:-50]
                backlog = metrics.setdefault("backlog", [])
                backlog.append({
                    "doc": name,
                    "ratio": novelty_payload.get("ratio", 0.0),
                    "memory_id": memory_id,
                    "ts": time.time(),
                })
                backlog.sort(key=lambda item: item.get("ratio", 0.0), reverse=True)
                if len(backlog) > 40:
                    backlog[:] = backlog[:40]
            except Exception:
                pass

            # pipeline indexing -> semantic vector store
            try:
                if hasattr(mem, "ingest_document"):
                    mem.ingest_document(text_to_store, title=name, source=f"inbox:{name}")
            except Exception:
                pass

            # --- INDUCTION DE RÈGLES SOCIALES ---
            try:
                from AGI_Evolutive.social.interaction_miner import InteractionMiner
                from AGI_Evolutive.social.interaction_rule import InteractionRule

                self.arch.interaction_miner = getattr(self.arch, "interaction_miner", InteractionMiner(self.arch))
                rules = self.arch.interaction_miner.mine_text(content, source=f"inbox:{name}")

                # seuil bas pour enregistrer, moyen/haut pour proposer validation
                for rule in rules:
                    rule_dict = rule.to_dict()
                    rule_dict.setdefault("evidence", {})["source"] = f"inbox:{name}"
                    rule_dict["evidence"]["usage"] = 0
                    rule_dict["evidence"]["last_review_ts"] = None
                    self.arch.memory.add_memory(rule_dict)

                    # (optionnel) goal de validation quand confiance moyenne
                    if 0.55 <= float(rule.confidence) < 0.75:
                        gid = f"validate_rule::{rule.id}"
                        self.arch.planner.ensure_goal(gid, f"Valider règle sociale {rule.id}", priority=0.62)
                        # pipeline simple : simuler ou chercher contre-exemples
                        self.arch.planner.add_action_step(gid, "simulate_dialogue", {"rule_id": rule.id}, priority=0.60)
                        self.arch.planner.add_action_step(gid, "search_counterexample", {"rule_id": rule.id}, priority=0.58)
                    try:
                        jm = getattr(self.arch, "job_manager", None)
                        if jm and hasattr(jm, "submit"):
                            jm.submit(
                                kind="social_self_eval",
                                fn=self.arch.interaction_miner.schedule_self_evaluation,
                                args={"rule": rule.to_dict(), "arch": self.arch},
                                queue="background",
                                priority=0.55,
                                key=f"rule_self_eval::{rule.id}::{h[:6]}",
                            )
                    except Exception:
                        pass
            except Exception as e:
                # ne casse jamais l'ingestion si la mine échoue
                try:
                    self.arch.memory.add_memory({"kind": "warn", "text": f"interaction_miner_failed:{e}"})
                except Exception:
                    pass
            # --- fin induction ---
            self._index[name] = h
            added += 1

            if memory_id:
                try:
                    jm = getattr(self.arch, "job_manager", None)
                    if jm and hasattr(jm, "submit"):
                        jm.submit(
                            kind="memory_audit",
                            fn=_document_evidence_job,
                            args={
                                "arch": self.arch,
                                "memory_id": memory_id,
                                "memory_type": MemoryType.EPISODIC.name,
                            },
                            queue="background",
                            priority=0.42,
                            key=f"doc_audit::{memory_id}",
                            timeout_s=10.0,
                        )
                except Exception:
                    pass
        return added
