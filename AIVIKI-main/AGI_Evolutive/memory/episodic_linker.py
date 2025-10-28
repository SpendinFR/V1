import logging
import os
import json
import time
import glob
from typing import Any, Dict, Iterable, List, Optional, Tuple, Mapping

try:
    from AGI_Evolutive.learning import OnlineLinearModel, ThompsonBandit
except Exception:  # pragma: no cover - learning module optional in some builds
    OnlineLinearModel = None  # type: ignore[assignment]
    ThompsonBandit = None  # type: ignore[assignment]

from AGI_Evolutive.utils.jsonsafe import json_sanitize
from AGI_Evolutive.utils.llm_contracts import enforce_llm_contract
from AGI_Evolutive.utils.llm_service import try_call_llm_dict


logger = logging.getLogger(__name__)


def _now():
    return time.time()


CAUSE_HINTS = [
    "parce que",
    "car",
    "donc",
    "alors",
    "ainsi",
    "du coup",
    "because",
    "therefore",
    "so that",
    "so ",
    "hence",
]
REL_NEXT = "NEXT"
REL_CAUSES = "CAUSES"
REL_REFERS = "REFERS_TO"
REL_SUPPORTS = "SUPPORTS"
REL_CONTRADICTS = "CONTRADICTS"

LLM_RELATION_MAP = {
    "cause": REL_CAUSES,
    "causal": REL_CAUSES,
    "causation": REL_CAUSES,
    "root_cause": REL_CAUSES,
    "support": REL_SUPPORTS,
    "soutien": REL_SUPPORTS,
    "backing": REL_SUPPORTS,
    "contradiction": REL_CONTRADICTS,
    "oppose": REL_CONTRADICTS,
    "ref": REL_REFERS,
    "reference": REL_REFERS,
    "cite": REL_REFERS,
    "sequence": REL_NEXT,
    "follow": REL_NEXT,
    "suivi": REL_NEXT,
}


class _AdaptiveScheduler:
    """Bandit-based tuner for the run cadence and episode window."""

    def __init__(
        self,
        default_period: float,
        default_window: float,
        state: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.default_period = default_period
        self.default_window = default_window
        self.actions = [
            "period=10.0_window=600",
            "period=15.0_window=900",
            "period=20.0_window=900",
            "period=25.0_window=1200",
            "period=30.0_window=1500",
        ]
        self.bandit: Optional[ThompsonBandit] = None
        if ThompsonBandit is not None:
            self.bandit = ThompsonBandit(prior_alpha=1.5, prior_beta=1.5)
        self.last_action: Optional[str] = None
        self.decay = 0.97
        if isinstance(state, dict):
            self.last_action = state.get("last_action")
            if self.bandit and isinstance(state.get("bandit"), dict):
                params: Dict[str, Tuple[float, float]] = {}
                for key, value in state["bandit"].items():
                    if not isinstance(value, (list, tuple)) or len(value) != 2:
                        continue
                    alpha = max(1e-3, float(value[0]))
                    beta = max(1e-3, float(value[1]))
                    params[str(key)] = (alpha, beta)
                if params:
                    self.bandit.parameters = params

    @staticmethod
    def _parse(action: Optional[str]) -> Tuple[float, float]:
        if not action:
            return 20.0, 15 * 60.0
        parts = action.split("_")
        period = 20.0
        window = 15 * 60.0
        for part in parts:
            if part.startswith("period="):
                try:
                    period = float(part.split("=", 1)[1])
                except (TypeError, ValueError):
                    period = 20.0
            elif part.startswith("window="):
                try:
                    window = float(part.split("=", 1)[1])
                except (TypeError, ValueError):
                    window = 15 * 60.0
        return period, window

    def select(self, load_signal: float) -> Tuple[float, float, Optional[str]]:
        if not self.bandit:
            return self.default_period, self.default_window, None
        priors = {}
        load_signal = max(0.0, min(1.0, float(load_signal)))
        for action in self.actions:
            period, window = self._parse(action)
            # Encourage shorter periods when load is high
            expected = 1.0 / (1.0 + period / max(1.0, load_signal * 20.0 + 1.0))
            expected *= max(0.1, min(1.0, window / (30.0 * 60.0)))
            priors[action] = max(0.0, min(1.0, expected))
        action = self.bandit.select(self.actions, priors=priors, fallback=lambda: self.actions[0])
        self.last_action = action
        period, window = self._parse(action)
        return period, window, action

    def update(self, reward: float) -> None:
        if not self.bandit or not self.last_action:
            return
        reward = max(0.0, min(1.0, float(reward)))
        alpha, beta = self.bandit._params(self.last_action)
        alpha = alpha * self.decay + reward
        beta = beta * self.decay + (1.0 - reward)
        self.bandit.parameters[self.last_action] = (max(1e-3, alpha), max(1e-3, beta))

    def to_state(self) -> Dict[str, Any]:
        if not self.bandit:
            return {}
        return {
            "last_action": self.last_action,
            "bandit": {key: list(value) for key, value in self.bandit.parameters.items()},
        }


class _SalienceLearner:
    """Online learner for relation salience scoring with forgetting."""

    def __init__(self, state: Optional[Dict[str, Any]] = None) -> None:
        self.model: Optional[OnlineLinearModel] = None
        if OnlineLinearModel is not None:
            feature_dim = 6 + len({REL_CAUSES, REL_SUPPORTS, REL_CONTRADICTS, REL_REFERS, REL_NEXT})
            self.model = OnlineLinearModel(
                feature_dim=feature_dim,
                learning_rate=0.12,
                l2=5e-3,
                bounds=(0.0, 1.0),
            )
        self.forget = 0.985
        if self.model and isinstance(state, dict):
            weights = state.get("weights")
            bias = state.get("bias")
            try:
                if isinstance(weights, list):
                    self.model._ensure_dim(len(weights))
                    for i, value in enumerate(weights):
                        self.model.weights[i] = float(value)
                if bias is not None:
                    self.model.bias = float(bias)
            except Exception:
                self.model.reset()

    def _decay(self) -> None:
        if not self.model:
            return
        for i in range(len(self.model.weights)):
            self.model.weights[i] *= self.forget
        self.model.bias *= self.forget

    def features(
        self,
        baseline: float,
        recency: float,
        emotion_factor: float,
        relation: str,
    ) -> List[float]:
        link_types = [REL_CAUSES, REL_SUPPORTS, REL_CONTRADICTS, REL_REFERS, REL_NEXT]
        features: List[float] = [
            max(0.0, min(1.0, baseline)),
            max(0.0, min(1.0, recency)),
            max(0.0, min(1.0, emotion_factor)),
            max(0.0, min(1.0, recency * emotion_factor)),
            max(0.0, min(1.0, recency * baseline)),
            max(0.0, min(1.0, emotion_factor * baseline)),
        ]
        for rel_type in link_types:
            features.append(1.0 if relation == rel_type else 0.0)
        return features

    def score(self, features: Iterable[float], fallback: float) -> float:
        if not self.model:
            return max(0.0, min(1.0, fallback))
        prediction = self.model.predict(list(features))
        return max(0.0, min(1.0, prediction))

    def learn(self, features: Iterable[float], target: float) -> None:
        if not self.model:
            return
        self._decay()
        self.model.update(list(features), max(0.0, min(1.0, target)))

    def to_state(self) -> Dict[str, Any]:
        if not self.model:
            return {}
        return {
            "weights": list(self.model.weights),
            "bias": float(self.model.bias),
        }


class EpisodicLinker:
    """
    - Regroupe des mémoires proches dans le temps en 'épisodes' (fenêtre)
    - Ajoute des liens temporels/causaux simples entre mémoires
    - Écrit episodes.jsonl + backlinks (memory_backlinks.json)
    - Pousse des mémoires 'episode_summary' si possible
    """

    def __init__(self, memory_store: Optional[Any], graph_path: str = "data/episodic_graph.json"):
        if graph_path and not graph_path.endswith(".json"):
            self.data_dir = graph_path
            self.graph_path = os.path.join(self.data_dir, "episodic_graph.json")
        else:
            self.graph_path = graph_path
            base_dir = os.path.dirname(graph_path)
            self.data_dir = base_dir if base_dir else "."

        os.makedirs(self.data_dir, exist_ok=True)
        self.paths = {
            "episodes": os.path.join(self.data_dir, "episodes.jsonl"),
            "backlinks": os.path.join(self.data_dir, "memory_backlinks.json"),
            "state": os.path.join(self.data_dir, "episodic_state.json"),
        }
        self.bound = {
            "memory": memory_store,
            "language": None,
            "metacog": None,
            "emotions": None,
        }

        self.state = self._load(
            self.paths["state"],
            {"last_run": 0.0, "processed_ids": [], "last_episode_id": 0},
        )
        self.backlinks = self._load(self.paths["backlinks"], {})  # mem_id -> [{to, rel}]
        self.graph = self._load(self.graph_path, {"nodes": [], "edges": []})

        scheduler_state = self.state.get("scheduler") if isinstance(self.state, dict) else None
        self._scheduler = _AdaptiveScheduler(20.0, 15 * 60.0, scheduler_state)
        initial_period, initial_window = _AdaptiveScheduler._parse(
            scheduler_state.get("last_action") if isinstance(scheduler_state, dict) else None
        )
        self.period_s = initial_period
        self._last_step = 0.0
        self.window_s = initial_window  # 15 minutes pour grouper en épisode (adaptatif)
        self._salient_queue: List[Dict[str, Any]] = []
        self._salient_seen: Dict[str, float] = {}
        self._duplicate_ttl = 3 * 3600.0
        self._salience_features: Dict[str, List[float]] = {}
        salience_state = self.state.get("salience") if isinstance(self.state, dict) else None
        self._salience_learner = _SalienceLearner(salience_state)
        self._relation_priors = {
            REL_CAUSES: 0.9,
            REL_SUPPORTS: 0.75,
            REL_CONTRADICTS: 0.85,
            REL_REFERS: 0.6,
            REL_NEXT: 0.5,
        }
        self._last_reward: float = 0.0

    def bind(self, memory=None, language=None, metacog=None, emotions=None):
        self.bound.update(
            {
                "memory": memory,
                "language": language,
                "metacog": metacog,
                "emotions": emotions,
            }
        )

    # ---------- stepping ----------
    def step(self, memory: Any = None, max_batch: int = 200) -> None:
        if memory is not None:
            self.bound["memory"] = memory
        if self.bound.get("memory") is None:
            return
        now = time.time()
        if now - self._last_step < self.period_s:
            return
        self._last_step = now
        self.run_once(max_batch=max_batch)

    # ---------- core ----------
    def run_once(self, max_batch: int = 400):
        mems = self._fetch_recent_memories(limit=max_batch)
        self._process_batch(mems, limit=max_batch)

    def process_memories(self, memories: Iterable[Mapping[str, Any]]) -> None:
        batch: List[Dict[str, Any]] = []
        for memory in memories:
            if isinstance(memory, Mapping):
                batch.append(dict(memory))
        if not batch:
            return
        self._process_batch(batch, limit=len(batch))

    def _process_batch(self, mems: List[Dict[str, Any]], *, limit: Optional[int] = None) -> None:
        if not mems:
            return

        reward_signal = 0.0
        if hasattr(self, "_scheduler") and isinstance(self._scheduler, _AdaptiveScheduler):
            load_signal = min(1.0, len(mems) / float(max(1, limit or len(mems))))
            period, window, action = self._scheduler.select(load_signal)
            self.period_s = 0.7 * self.period_s + 0.3 * period
            self.window_s = 0.7 * self.window_s + 0.3 * window
            if action:
                self.state.setdefault("scheduler", {})["last_action"] = action

        # trier par timestamp si dispo
        def _ts(memory):
            metadata = memory.get("metadata", {})
            return float(metadata.get("timestamp", memory.get("t", memory.get("ts", _now()))))

        mems = sorted(mems, key=_ts)

        # filtrer déjà traités
        batch: List[Dict[str, Any]] = []
        for memory in mems:
            mid = memory.get("id") or memory.get("_id") or memory.get("memory_id")
            if not mid:
                continue
            if mid in self.state["processed_ids"]:
                continue
            batch.append(memory)

        if not batch:
            return

        # grouper en épisodes
        episodes = self._group_into_episodes(batch, key_ts=_ts)
        total_relations = 0
        total_memories = sum(len(ep.get("memories", [])) for ep in episodes)
        for episode in episodes:
            ep_id = self._next_episode_id()
            rels = self._link_relations(episode["memories"])
            total_relations += len(rels)
            self._queue_salient_associations(episode["memories"], rels)
            summary = self._summarize_episode(episode["memories"])
            record = {
                "episode_id": ep_id,
                "start": episode["start"],
                "end": episode["end"],
                "size": len(episode["memories"]),
                "memory_ids": [
                    memory.get("id") or memory.get("_id") or memory.get("memory_id")
                    for memory in episode["memories"]
                ],
                "relations": rels,
                "summary": summary,
            }
            self._append_jsonl(self.paths["episodes"], record)
            self._apply_backlinks(record["relations"])
            self._emit_episode_memory(summary, record)

        # marquer traités
        new_ids = [
            memory.get("id") or memory.get("_id") or memory.get("memory_id")
            for memory in batch
        ]
        self.state["processed_ids"] += new_ids
        if len(self.state["processed_ids"]) > 5000:
            self.state["processed_ids"] = self.state["processed_ids"][-2500:]
        self.state["last_run"] = _now()
        denom = max(1, total_memories)
        relation_density = total_relations / denom
        episode_count = len(episodes)
        reward = 0.6 * max(0.0, min(1.0, relation_density / 3.0))
        reward += 0.4 * max(0.0, min(1.0, episode_count / float(len(batch))))
        reward_signal = reward
        if hasattr(self, "_scheduler") and isinstance(self._scheduler, _AdaptiveScheduler):
            self._scheduler.update(reward)
            self.state["scheduler"] = self._scheduler.to_state()
        self._last_reward = float(reward_signal)
        metrics = self.state.setdefault("metrics", {})
        metrics.update(
            {
                "reward": float(reward_signal),
                "relation_density": float(relation_density),
                "episodes": int(episode_count),
                "batch": int(len(batch)),
            }
        )
        if hasattr(self, "_salience_learner") and isinstance(self._salience_learner, _SalienceLearner):
            self.state["salience"] = self._salience_learner.to_state()
        self._save(self.paths["state"], self.state)
        self._save(self.paths["backlinks"], self.backlinks)

    def link_recent(self, n: int = 60) -> Dict[str, int]:
        mems = self._fetch_recent_memories(limit=n)
        if not mems:
            return {"nodes": 0, "edges": 0}

        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        for memory in mems:
            ts = memory.get("ts") or memory.get("t") or _now()
            node_id = f"{int(ts)}_{memory.get('kind', 'mem')}"
            nodes.append(
                {
                    "id": node_id,
                    "kind": memory.get("kind"),
                    "ts": ts,
                    "text": (memory.get("text") or memory.get("content") or "")[:120],
                }
            )

        for i in range(len(nodes) - 1):
            a = nodes[i]
            b = nodes[i + 1]
            if a["kind"] in {"action_exec", "action_sim"} and b["kind"] in {"interaction", "lesson", "reflection"}:
                edges.append({"from": a["id"], "to": b["id"], "rel": "causes_like"})
            if (a.get("kind") or "").startswith("error") and b["kind"] in {"reflection", "lesson"}:
                edges.append({"from": a["id"], "to": b["id"], "rel": "triggered_reflection"})

        existing_nodes = {node.get("id"): node for node in self.graph.get("nodes", []) if node.get("id")}
        for node in nodes:
            existing_nodes[node["id"]] = node
        self.graph["nodes"] = list(existing_nodes.values())[-120:]
        self.graph.setdefault("edges", []).extend(edges)
        self.graph["edges"] = self.graph["edges"][-240:]
        self._save(self.graph_path, self.graph)
        return {"nodes": len(nodes), "edges": len(edges)}

    # ---------- grouping ----------
    def _group_into_episodes(self, mems: List[Dict[str, Any]], key_ts) -> List[Dict[str, Any]]:
        episodes: List[Dict[str, Any]] = []
        cur: List[Dict[str, Any]] = []
        cur_start: Optional[float] = None
        last_t: Optional[float] = None

        for memory in mems:
            timestamp = key_ts(memory)
            if last_t is None:
                cur = [memory]
                cur_start = timestamp
                last_t = timestamp
                continue
            if timestamp - last_t <= self.window_s:
                cur.append(memory)
                last_t = timestamp
            else:
                episodes.append({"start": cur_start, "end": last_t, "memories": cur[:]})
                cur = [memory]
                cur_start = timestamp
                last_t = timestamp
        if cur:
            episodes.append({"start": cur_start, "end": last_t, "memories": cur[:]})
        return episodes

    def _memory_timestamp(self, memory: Optional[Dict[str, Any]]) -> float:
        if not isinstance(memory, dict):
            return _now()
        metadata = memory.get("metadata") if isinstance(memory.get("metadata"), dict) else {}
        for key in ("timestamp", "ts", "t"):
            if key in metadata:
                try:
                    return float(metadata[key])
                except (TypeError, ValueError):
                    continue
        for key in ("ts", "t", "timestamp"):
            if key in memory:
                try:
                    return float(memory[key])
                except (TypeError, ValueError):
                    continue
        return _now()

    # ---------- linking ----------
    def _link_relations(self, mems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rels: List[Dict[str, Any]] = []
        ids = [memory.get("id") or memory.get("_id") or memory.get("memory_id") for memory in mems]
        texts = [str(memory.get("content", "")) for memory in mems]

        alias_map: Dict[str, str] = {}
        id_to_alias: Dict[str, str] = {}
        llm_memories: List[Dict[str, Any]] = []
        sample_count = min(len(mems), 12)
        for idx in range(sample_count):
            mem_id = ids[idx]
            if not mem_id:
                continue
            alias = f"m{idx}"
            alias_map[alias] = mem_id
            id_to_alias[mem_id] = alias
            memory = mems[idx]
            llm_memories.append(
                {
                    "alias": alias,
                    "id": mem_id,
                    "kind": memory.get("kind"),
                    "timestamp": self._memory_timestamp(memory),
                    "summary": (memory.get("content") or memory.get("text") or "")[:280],
                }
            )

        # NEXT relations (séquentiel)
        for i in range(len(ids) - 1):
            rels.append({"src": ids[i], "dst": ids[i + 1], "rel": REL_NEXT})

        # heuristique CAUSES/REFERS via texte
        for i in range(len(ids)):
            t_i = texts[i].lower()
            for j in range(i + 1, min(i + 5, len(ids))):
                t_j = texts[j].lower()
                if any(hint in t_j for hint in CAUSE_HINTS) or "car " in t_j or "parce " in t_j:
                    rels.append({"src": ids[i], "dst": ids[j], "rel": REL_CAUSES})
                if "voir" in t_j or "cf." in t_j or "réf" in t_j or "ref " in t_j:
                    rels.append({"src": ids[i], "dst": ids[j], "rel": REL_REFERS})

        # petites contradictions/soutiens naïfs
        for i in range(len(ids) - 1):
            a = texts[i].lower()
            b = texts[i + 1].lower()
            if any(word in a for word in ["oui", "vrai", "possible"]) and any(
                word in b for word in ["non", "faux", "impossible"]
            ):
                rels.append({"src": ids[i], "dst": ids[i + 1], "rel": REL_CONTRADICTS})
            if ("appris" in b or "confirm" in b or "conclu" in b) and len(a) > 10:
                rels.append({"src": ids[i], "dst": ids[i + 1], "rel": REL_SUPPORTS})

        fallback_relations = list(rels)

        llm_notes: Optional[str] = None
        llm_added: List[Dict[str, Any]] = []
        if llm_memories:
            existing_for_llm: List[Dict[str, Any]] = []
            for rel in fallback_relations:
                src_alias = id_to_alias.get(rel.get("src"))
                dst_alias = id_to_alias.get(rel.get("dst"))
                if not src_alias or not dst_alias:
                    continue
                existing_for_llm.append(
                    {
                        "from": src_alias,
                        "to": dst_alias,
                        "relation": rel.get("rel"),
                    }
                )

            llm_payload = {
                "memories": llm_memories,
                "existing_relations": existing_for_llm,
            }

            llm_result = try_call_llm_dict(
                "episodic_linker",
                input_payload=json_sanitize(llm_payload),
                logger=logger,
            )

            cleaned = enforce_llm_contract("episodic_linker", llm_result)
            if cleaned is not None:
                llm_result = cleaned

            if isinstance(llm_result, Mapping):
                links = llm_result.get("links")
                if isinstance(links, (list, tuple)):
                    existing_keys = {
                        (rel.get("src"), rel.get("dst"), rel.get("rel"))
                        for rel in fallback_relations
                    }
                    for item in links:
                        if not isinstance(item, Mapping):
                            continue
                        src_val = item.get("from") or item.get("src")
                        dst_val = item.get("to") or item.get("dst")
                        rel_val = item.get("type_lien") or item.get("relation") or item.get("rel")
                        if not rel_val:
                            continue
                        rel_code = LLM_RELATION_MAP.get(str(rel_val).lower(), None)
                        if rel_code is None:
                            rel_code = str(rel_val).upper()
                        src_candidate = str(src_val).strip() if src_val is not None else ""
                        dst_candidate = str(dst_val).strip() if dst_val is not None else ""
                        src_id = alias_map.get(src_candidate) if src_candidate else None
                        if src_id is None and src_candidate in ids:
                            src_id = src_candidate
                        if src_id is None and src_val in ids:
                            src_id = src_val  # type: ignore[arg-type]
                        dst_id = alias_map.get(dst_candidate) if dst_candidate else None
                        if dst_id is None and dst_candidate in ids:
                            dst_id = dst_candidate
                        if dst_id is None and dst_val in ids:
                            dst_id = dst_val  # type: ignore[arg-type]
                        if not src_id or not dst_id:
                            continue
                        key = (src_id, dst_id, rel_code)
                        if key in existing_keys:
                            continue
                        confidence_val = item.get("confidence")
                        confidence = None
                        try:
                            if confidence_val is not None:
                                confidence = max(0.0, min(1.0, float(confidence_val)))
                        except (TypeError, ValueError):
                            confidence = None
                        relation_entry: Dict[str, Any] = {
                            "src": src_id,
                            "dst": dst_id,
                            "rel": rel_code,
                        }
                        if confidence is not None:
                            relation_entry["confidence"] = confidence
                        rels.append(relation_entry)
                        existing_keys.add(key)
                        llm_added.append(relation_entry)
                notes_val = llm_result.get("notes")
                if isinstance(notes_val, str) and notes_val.strip():
                    llm_notes = notes_val.strip()

        if (llm_added or llm_notes) and isinstance(self.state, dict):
            llm_state = self.state.setdefault("llm", {})
            llm_state["episodic_linker"] = {
                "links": json_sanitize(llm_added),
                "notes": llm_notes,
                "timestamp": _now(),
            }

        return rels

    def _queue_salient_associations(
        self, memories: List[Dict[str, Any]], relations: List[Dict[str, Any]]
    ) -> None:
        if not relations:
            return

        emotion_factor = 0.5
        emotions = self.bound.get("emotions")
        if emotions and hasattr(emotions, "get_state"):
            try:
                state = emotions.get_state() or {}
                valence = float(state.get("valence", 0.0))
                emotion_factor = max(0.1, min(1.0, 0.5 + 0.5 * valence))
            except Exception:
                emotion_factor = 0.5

        mem_index = {
            memory.get("id") or memory.get("_id") or memory.get("memory_id"): memory
            for memory in memories
            if isinstance(memory, dict)
        }
        now = _now()
        expired_keys = [key for key, ts in self._salient_seen.items() if now - ts > self._duplicate_ttl]
        for key in expired_keys:
            self._salient_seen.pop(key, None)
            self._salience_features.pop(key, None)

        for rel in relations:
            src = rel.get("src")
            dst = rel.get("dst")
            rel_type = rel.get("rel")
            if not src or not dst or not rel_type:
                continue
            key = f"{src}->{dst}:{rel_type}"
            last_seen = self._salient_seen.get(key)
            if last_seen and now - last_seen < self._duplicate_ttl:
                continue

            src_ts = self._memory_timestamp(mem_index.get(src))
            dst_ts = self._memory_timestamp(mem_index.get(dst))
            recency = max(0.1, min(1.0, 1.0 - (now - max(src_ts, dst_ts)) / (6 * 3600)))
            link_strength = self._relation_priors.get(rel_type, 0.4)
            baseline = max(0.0, min(1.0, recency * emotion_factor * link_strength))
            features = self._salience_learner.features(baseline, recency, emotion_factor, rel_type)
            score = self._salience_learner.score(features, baseline)

            self._salient_queue.append(
                {
                    "key": key,
                    "source": src,
                    "target": dst,
                    "relation": rel_type,
                    "score": round(score, 3),
                    "timestamp": max(src_ts, dst_ts),
                }
            )
            self._salience_features[key] = features
            self._salient_seen[key] = now

        if len(self._salient_queue) > 40:
            self._salient_queue.sort(key=lambda item: item.get("score", 0.0), reverse=True)
            kept = self._salient_queue[:40]
            discarded = self._salient_queue[40:]
            self._salient_queue = kept
            for item in discarded:
                key = item.get("key")
                if key and key in self._salience_features:
                    self._salience_learner.learn(self._salience_features[key], 0.05)
                    self._salience_features.pop(key, None)
                if key:
                    self._salient_seen.pop(key, None)

    def pop_salient_associations(self, max_n=2) -> Dict[str, Any]:
        try:
            limit = max(1, int(max_n))
        except Exception:
            limit = 2

        items: List[Dict[str, Any]] = []
        while self._salient_queue and len(items) < limit:
            entry = self._salient_queue.pop(0)
            key = entry.get("key")
            if key and key in self._salience_features:
                self._salience_learner.learn(self._salience_features.pop(key), 0.9)
            if key:
                self._salient_seen.pop(key, None)
                entry.pop("key", None)
            items.append(entry)

        if items:
            return {"items": items, "count": len(items), "source": "queue"}

        # fallback: derive from backlinks recency
        candidates: List[Dict[str, Any]] = []
        for dst, links in self.backlinks.items():
            if not isinstance(links, list):
                continue
            for link in links[-5:]:
                if not isinstance(link, dict):
                    continue
                candidates.append(
                    {
                        "source": link.get("from"),
                        "target": dst,
                        "relation": link.get("rel"),
                        "timestamp": link.get("t", 0.0),
                        "score": 0.35,
                    }
                )

        candidates.sort(key=lambda item: item.get("timestamp", 0.0), reverse=True)
        fallback = candidates[:limit]
        return {"items": fallback, "count": len(fallback), "source": "backlinks"}

    # ---------- summary ----------
    def _summarize_episode(self, mems: List[Dict[str, Any]]) -> str:
        lang = self.bound.get("language")
        # tenter un résumé via module language
        if lang and hasattr(lang, "summarize"):
            try:
                return lang.summarize([memory.get("content", "") for memory in mems])
            except Exception:
                pass
        # fallback minimal : 1ère et dernière phrases tronquées
        def _preview(memory: Dict[str, Any]) -> str:
            value = memory.get("content")
            if isinstance(value, str):
                return value[:180]
            if value is None:
                return ""
            try:
                return json.dumps(value, ensure_ascii=False)[:180]
            except Exception:
                return str(value)[:180]

        first = _preview(mems[0])
        last = _preview(mems[-1])
        return f"Épisode ({len(mems)} mémoires) - début: {first} · fin: {last}"

    # ---------- apply backlinks + emit episode mem ----------
    def _apply_backlinks(self, relations: List[Dict[str, Any]]):
        mem = self.bound.get("memory")
        for relation in relations:
            src, dst, rel = relation["src"], relation["dst"], relation["rel"]
            if not src or not dst:
                continue
            self.backlinks.setdefault(dst, [])
            self.backlinks[dst].append({"from": src, "rel": rel, "t": _now()})
            # enregistre une mémoire "lien" optionnelle
            if mem and hasattr(mem, "add_memory"):
                try:
                    mem.add_memory(
                        kind="memory_link",
                        content=f"{src} -> {dst} [{rel}]",
                        metadata={"from": src, "to": dst, "relation": rel},
                    )
                except Exception:
                    pass

    def _emit_episode_memory(self, summary: str, record: Dict[str, Any]):
        mem = self.bound.get("memory")
        if mem and hasattr(mem, "add_memory"):
            try:
                mem.add_memory(
                    kind="episode_summary",
                    content=summary,
                    metadata={
                        "episode_id": record["episode_id"],
                        "start": record["start"],
                        "end": record["end"],
                        "size": record["size"],
                        "memory_ids": record["memory_ids"],
                    },
                )
            except Exception:
                pass

    # ---------- io ----------
    def _append_jsonl(self, path: str, obj: Dict[str, Any]):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(json_sanitize(obj), ensure_ascii=False) + "\n")

    def _load(self, path: str, default):
        if not os.path.exists(path):
            return default
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default

    def _save(self, path: str, obj):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(json_sanitize(obj), f, ensure_ascii=False, indent=2)

    # ---------- memory fetch ----------
    def _fetch_recent_memories(self, limit: int = 400) -> List[Dict[str, Any]]:
        mem = self.bound.get("memory")
        if mem is None:
            return self._fallback_read_files(limit)
        try:
            if hasattr(mem, "get_recent_memories"):
                res = mem.get_recent_memories(n=limit) or []
                return res
        except Exception:
            pass
        try:
            if hasattr(mem, "iter_memories"):
                res = []
                for memory in mem.iter_memories():
                    res.append(memory)
                    if len(res) >= limit:
                        break
                return res
        except Exception:
            pass
        return self._fallback_read_files(limit)

    def _fallback_read_files(self, limit: int = 400) -> List[Dict[str, Any]]:
        root = os.path.join(self.data_dir, "memories")
        if not os.path.isdir(root):
            return []
        files = sorted(glob.glob(os.path.join(root, "*.json")), reverse=True)
        out: List[Dict[str, Any]] = []
        for path in files[:limit]:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    out.append(json.load(f))
            except Exception:
                continue
        return out

    # ---------- diagnostics ----------
    def pending_backlog(self, limit: int = 300) -> int:
        """Estimate the number of memories that still need episodic linking."""

        mems = self._fetch_recent_memories(limit=limit)
        if not mems:
            return 0
        processed = set(self.state.get("processed_ids", []))
        backlog = 0
        for memory in mems:
            mid = memory.get("id") or memory.get("_id") or memory.get("memory_id")
            if not mid or mid in processed:
                continue
            backlog += 1
        return backlog

    def quality_signal(self) -> float:
        """Return the last normalized reward produced by the episodic linker."""

        metrics = self.state.get("metrics") if isinstance(self.state, dict) else None
        if isinstance(metrics, dict) and "reward" in metrics:
            value = float(metrics.get("reward", 0.0))
        else:
            value = self._last_reward
        return max(0.0, min(1.0, float(value)))

    @property
    def last_reward(self) -> float:
        return max(0.0, min(1.0, float(self._last_reward)))

    # ---------- helpers ----------
    def _next_episode_id(self) -> int:
        self.state["last_episode_id"] += 1
        return self.state["last_episode_id"]
