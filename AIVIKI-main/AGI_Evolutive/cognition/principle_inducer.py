from __future__ import annotations
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import json
import math
import random
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import hashlib

from AGI_Evolutive.core.structures.mai import MAI, new_mai, EvidenceRef, ImpactHypothesis
from AGI_Evolutive.knowledge.mechanism_store import MechanismStore
from AGI_Evolutive.utils.llm_service import try_call_llm_dict

# Adapters (branchés sur tes modules v11)
# On suppose que interaction_miner renvoie des "patterns" normatifs déjà extraits.
# Chaque pattern contient: text, conditions (tokens/logiques), examples, strength
NormativePattern = Dict[str, Any]


@dataclass
class _FeedbackRecord:
    signature: str
    accepted: bool
    trust_gain: float
    harm_delta: float
    source: str
    variant: str
    strength: float
    latency_s: float
    timestamp: float


class _FeedbackLedger:
    """Tiny append-only JSONL cache pour exploiter les retours d'évaluation."""

    def __init__(self, path: Path | str | None = None, window: int = 200) -> None:
        self.path = Path(path or "data/runtime/principle_feedback.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.window = window
        self._per_signature: Dict[str, List[_FeedbackRecord]] = defaultdict(list)
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                    except Exception:
                        continue
                    record = self._record_from_payload(payload)
                    if record:
                        self._per_signature[record.signature].append(record)
        except Exception:
            # fail-safe: ignore corrupted stores
            self._per_signature.clear()

    def _record_from_payload(self, payload: Mapping[str, Any]) -> Optional[_FeedbackRecord]:
        try:
            return _FeedbackRecord(
                signature=str(payload["signature"]),
                accepted=bool(payload.get("accepted", False)),
                trust_gain=float(payload.get("trust_gain", 0.0)),
                harm_delta=float(payload.get("harm_delta", 0.0)),
                source=str(payload.get("source", "unknown")),
                variant=str(payload.get("variant", "base")),
                strength=float(payload.get("strength", 0.0)),
                latency_s=float(payload.get("latency_s", 0.0)),
                timestamp=float(payload.get("timestamp", time.time())),
            )
        except Exception:
            return None

    def stats(self, signature: str) -> Dict[str, float]:
        records = self._per_signature.get(signature, [])
        if not records:
            return {
                "count": 0.0,
                "accept_rate": 0.5,
                "recent_accept_rate": 0.5,
                "mean_trust": 0.0,
                "mean_harm": 0.0,
                "drift": 0.0,
            }

        count = float(len(records))
        accept_rate = sum(1.0 for r in records if r.accepted) / count
        recent_window = records[-min(len(records), 10) :]
        recent_accept = sum(1.0 for r in recent_window if r.accepted) / float(len(recent_window))
        mean_trust = statistics.fmean(r.trust_gain for r in records) if records else 0.0
        mean_harm = statistics.fmean(r.harm_delta for r in records) if records else 0.0
        drift = abs(recent_accept - accept_rate)
        return {
            "count": count,
            "accept_rate": accept_rate,
            "recent_accept_rate": recent_accept,
            "mean_trust": mean_trust,
            "mean_harm": mean_harm,
            "drift": drift,
        }

    def record(
        self,
        signature: str,
        *,
        accepted: bool,
        trust_gain: float,
        harm_delta: float,
        source: str,
        variant: str,
        strength: float,
        latency_s: float,
    ) -> None:
        record = _FeedbackRecord(
            signature=signature,
            accepted=accepted,
            trust_gain=trust_gain,
            harm_delta=harm_delta,
            source=source,
            variant=variant,
            strength=strength,
            latency_s=latency_s,
            timestamp=time.time(),
        )

        bucket = self._per_signature[signature]
        bucket.append(record)
        if len(bucket) > self.window:
            del bucket[: len(bucket) - self.window]

        payload = {
            "timestamp": record.timestamp,
            "signature": signature,
            "accepted": accepted,
            "trust_gain": trust_gain,
            "harm_delta": harm_delta,
            "source": source,
            "variant": variant,
            "strength": strength,
            "latency_s": latency_s,
        }
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            # Telemetry shouldn't break the pipeline
            pass

    def aggregate_accept_rate(self) -> Optional[float]:
        if not self._per_signature:
            return None
        totals = [self.stats(sig)["accept_rate"] for sig in self._per_signature]
        return statistics.fmean(totals) if totals else None


class _TelemetryLogger:
    """Append-only JSONL instrumentation (drift, scoring, décisions)."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path or "data/runtime/principle_inducer_metrics.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, kind: str, payload: Mapping[str, Any]) -> None:
        entry = {
            "timestamp": time.time(),
            "kind": kind,
            "payload": dict(payload),
        }
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

class PrincipleInducer:
    """
    Transforme des motifs normatifs (lectures, dialogues, épisodes) en MAI candidats,
    puis délègue l'évaluation/ablation/promotion à ton pipeline (critic + world_model + self_improver).
    """
    def __init__(self, mechanism_store: MechanismStore):
        self.store = mechanism_store
        self._feedback = _FeedbackLedger()
        self._telemetry = _TelemetryLogger()
        self._logger = getattr(mechanism_store, "logger", None)

    def _llm_summarize_run(
        self,
        *,
        produced: int,
        selected: int,
        latency: float,
        metrics_snapshot: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload = {
            "produced": produced,
            "selected": selected,
            "latency": latency,
            "metrics_snapshot": metrics_snapshot,
            "feedback_accept_rate": self._feedback.aggregate_accept_rate(),
        }
        response = try_call_llm_dict(
            "cognition_principle_inducer",
            input_payload=payload,
            logger=getattr(self, "_logger", None),
        )
        summary = {
            "produced": produced,
            "selected": selected,
            "latency": latency,
        }
        if isinstance(response, dict):
            if isinstance(response.get("notes"), str) and response["notes"].strip():
                summary["notes"] = response["notes"].strip()
            if "confidence" in response:
                try:
                    summary["confidence"] = max(0.0, min(1.0, float(response["confidence"])))
                except (TypeError, ValueError):
                    pass
            if isinstance(response.get("actions"), list):
                summary["actions"] = [
                    str(action) for action in response["actions"] if str(action).strip()
                ][:5]
            summary["llm"] = response
        return summary

    # ---- 1) Synthèse de MAI candidats depuis des patterns normatifs ----
    def induce_from_patterns(self, patterns: List[NormativePattern]) -> List[MAI]:
        """
        Patterns attendus (exemple minimal):
        {
          "text": "Garder une information confidentielle sans consentement explicite; sauf danger imminent.",
          "conditions": [
             {"op":"atom","name":"request_is_sensitive"},
             {"op":"atom","name":"audience_is_not_owner"},
             {"op":"not","args":[{"op":"atom","name":"has_consent"}]}
          ],
          "exceptions": [
             {"op":"atom","name":"imminent_harm_detected"}
          ],
          "suggested_actions": ["RefusePolitely","AskConsent","PartialReveal"],
          "evidence": [{"source":"doc:ethics.pdf#p12"}],
          "strength": 0.74
        }
        """
        candidates: List[MAI] = []
        for p in patterns:
            for variant_tag, conds, exceptions, variant_info in self._iter_variants(p):
                base_cond = self._build_condition_expr(conds, exceptions)

                actions = list(p.get("suggested_actions", []) or [])
                if variant_info.get("kind") == "explore_actions" and variant_info.get("actions"):
                    actions = list(variant_info.get("actions", actions))

                bids = [
                    {"action_hint": a, "rationale": p.get("text", "")}
                    for a in actions
                ]

                evidence_refs = [
                    EvidenceRef(**e) if isinstance(e, dict) else EvidenceRef(source=str(e))
                    for e in p.get("evidence", [])
                ]

                doc = p.get("text", "(principe appris)")
                if variant_tag != "base":
                    doc = f"{doc} (variant:{variant_tag})"

                m = new_mai(
                    docstring=doc,
                    precondition_expr=base_cond,
                    propose_spec={"bids": bids},
                    evidence=evidence_refs,
                    safety=["keep_explicit_promises"],
                )

                signature = self._pattern_signature(p, variant_tag, variant_info)
                strength = float(p.get("strength", 0.0))
                m.expected_impact = ImpactHypothesis(
                    trust_delta=0.1 + 0.2 * strength,
                    harm_delta=-0.2 * max(0.0, 1.0 - strength),
                    identity_coherence_delta=0.05 + 0.1 * strength,
                    uncertainty=max(0.05, 0.5 - 0.2 * strength),
                )
                m.tags.extend(["induced", f"variant:{variant_tag}"])
                m.metadata.update({
                    "pattern_text": p.get("text"),
                    "pattern_signature": signature,
                    "pattern_strength": strength,
                    "variant_tag": variant_tag,
                    "variant_info": variant_info,
                    "source_examples": p.get("examples"),
                    "source_conditions": p.get("conditions"),
                    "source_exceptions": p.get("exceptions"),
                })

                self._attach_synthetic_scenarios(m, p)
                candidates.append(m)
        return candidates

    # ---- 2) Pré-évaluation rapide (heuristique) AVANT sandbox lourde ----
    def prefilter(self, mai_list: List[MAI], history_stats: Dict[str, float]) -> List[MAI]:
        max_canary = int(max(1, history_stats.get("mai_max_canary", 3)))

        history_accept = history_stats.get("mai_accept_rate")
        feedback_accept = self._feedback.aggregate_accept_rate()
        if history_accept is not None:
            if history_accept < 0.2:
                max_canary = max(1, math.floor(max_canary * 0.6))
            elif history_accept > 0.6:
                max_canary = max_canary + 1
        elif feedback_accept is not None and feedback_accept < 0.3:
            max_canary = max(1, math.floor(max_canary * 0.7))

        existing_vectors = self._existing_novelty_vectors()

        scored: List[Tuple[float, MAI, Sequence[str]]] = []
        for mai in mai_list:
            signature = str(mai.metadata.get("pattern_signature", mai.id))
            stats = self._feedback.stats(signature)
            novelty = self._novelty_score(mai, existing_vectors)
            diversity_bonus = 1.0 + 0.1 * random.random()
            exploration_bonus = 0.15 if stats["count"] < 2 else 0.0
            penalty = 0.0
            if stats["mean_harm"] > 0.0:
                penalty += min(0.4, stats["mean_harm"])
            penalty += min(0.3, stats["drift"])

            base_score = 0.5 * stats["recent_accept_rate"] + 0.3 * novelty + 0.2 * stats["mean_trust"]
            score = (base_score + exploration_bonus) * diversity_bonus - penalty
            tokens = tuple(self._tokenize(mai.docstring))

            scored.append((score, mai, tokens))
            self._telemetry.log(
                "prefilter_score",
                {
                    "mai_id": mai.id,
                    "signature": signature,
                    "score": score,
                    "components": {
                        "recent_accept_rate": stats["recent_accept_rate"],
                        "novelty": novelty,
                        "mean_trust": stats["mean_trust"],
                        "penalty": penalty,
                    },
                },
            )

        scored.sort(key=lambda item: item[0], reverse=True)

        selected: List[MAI] = []
        seen_vectors: List[Sequence[str]] = []
        for score, mai, tokens in scored:
            if len(selected) >= max_canary:
                break
            if self._is_too_similar(tokens, seen_vectors):
                continue
            selected.append(mai)
            seen_vectors.append(tokens)

        if len(selected) < max_canary:
            for _, mai, tokens in scored:
                if mai in selected:
                    continue
                selected.append(mai)
                if len(selected) >= max_canary:
                    break
                seen_vectors.append(tokens)

        self._telemetry.log(
            "prefilter_selection",
            {
                "requested": max_canary,
                "available": len(mai_list),
                "selected": [m.id for m in selected],
            },
        )
        return selected

    # ---- Helpers internes (diversité, feedback, etc.) ----
    def _iter_variants(
        self, pattern: NormativePattern
    ) -> Iterable[Tuple[str, List[Any], List[Any], Dict[str, Any]]]:
        conditions = list(pattern.get("conditions", []) or [])
        exceptions = list(pattern.get("exceptions", []) or [])
        yield ("base", conditions, exceptions, {"kind": "base"})

        if len(conditions) > 1:
            for idx in range(len(conditions)):
                reduced = conditions[:idx] + conditions[idx + 1 :]
                yield (
                    f"drop_{idx}",
                    reduced,
                    exceptions,
                    {"kind": "relaxed", "dropped_index": idx},
                )

        if exceptions:
            yield (
                "narrow_exceptions",
                conditions,
                exceptions[:-1],
                {"kind": "focused", "removed_exception": exceptions[-1]},
            )
        else:
            if conditions:
                yield (
                    "solo_condition",
                    [conditions[0]],
                    exceptions,
                    {"kind": "focused", "kept_index": 0},
                )

        # Variation active: inverser des actions suggérées pour tester
        suggested = list(pattern.get("suggested_actions", []) or [])
        if suggested:
            flipped = list(reversed(suggested))
            yield (
                "action_permutation",
                conditions,
                exceptions,
                {"kind": "explore_actions", "actions": flipped},
            )

    def _build_condition_expr(self, conditions: List[Any], exceptions: List[Any]) -> Dict[str, Any]:
        base_cond: Dict[str, Any] = {"op": "and", "args": conditions}
        if exceptions:
            base_cond = {
                "op": "and",
                "args": [
                    base_cond,
                    {"op": "not", "args": [{"op": "or", "args": exceptions}]},
                ],
            }
        return base_cond

    def _pattern_signature(
        self,
        pattern: NormativePattern,
        variant_tag: str,
        variant_info: Mapping[str, Any],
    ) -> str:
        payload = {
            "text": pattern.get("text"),
            "conditions": pattern.get("conditions"),
            "exceptions": pattern.get("exceptions"),
            "variant": variant_tag,
            "variant_info": variant_info,
        }
        digest = hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode("utf-8"))
        return digest.hexdigest()

    def _tokenize(self, text: Optional[str]) -> List[str]:
        if not text:
            return []
        cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
        tokens = [tok for tok in cleaned.split() if tok]
        return tokens

    def _existing_novelty_vectors(self) -> List[Sequence[str]]:
        vectors: List[Sequence[str]] = []
        try:
            for existing in self.store.all():
                vectors.append(tuple(self._tokenize(existing.docstring)))
        except Exception:
            return []
        return vectors

    def _novelty_score(self, mai: MAI, existing_vectors: List[Sequence[str]]) -> float:
        tokens = set(self._tokenize(mai.docstring))
        if not tokens:
            return 0.5
        if not existing_vectors:
            return 1.0
        overlaps = []
        for vec in existing_vectors:
            ref = set(vec)
            if not ref:
                continue
            intersection = len(tokens & ref)
            union = len(tokens | ref)
            overlaps.append(intersection / union)
        if not overlaps:
            return 1.0
        best_overlap = max(overlaps)
        return max(0.0, 1.0 - best_overlap)

    def _is_too_similar(
        self, tokens: Sequence[str], seen_vectors: List[Sequence[str]], threshold: float = 0.85
    ) -> bool:
        if not seen_vectors:
            return False
        target = set(tokens)
        if not target:
            return False
        for vec in seen_vectors:
            ref = set(vec)
            if not ref:
                continue
            inter = len(target & ref)
            union = len(target | ref) or 1
            if union == 0:
                continue
            similarity = inter / union
            if similarity >= threshold:
                return True
        return False

    def _attach_synthetic_scenarios(self, mai: MAI, pattern: NormativePattern) -> None:
        scenarios = []
        base_text = pattern.get("text") or mai.docstring
        actors = pattern.get("actors") or ["agent", "interlocutor"]
        context = pattern.get("context") or "generic_dialogue"
        conditions = pattern.get("conditions", [])

        if conditions:
            exemplar = ", ".join(str(c.get("name", c)) if isinstance(c, Mapping) else str(c) for c in conditions[:2])
            scenarios.append(
                {
                    "title": "baseline_application",
                    "context": context,
                    "description": f"Apply when {exemplar} holds; actors: {actors}",
                }
            )

        scenarios.append(
            {
                "title": "adversarial_exception",
                "context": context,
                "description": f"Stress test {base_text} with missing consent and a malicious request.",
            }
        )

        scenarios.append(
            {
                "title": "edge_relaxation",
                "context": context,
                "description": "Relax one condition and verify critic/regret scores remain acceptable.",
            }
        )

        meta = mai.metadata if isinstance(mai.metadata, dict) else {}
        meta.setdefault("synthetic_scenarios", scenarios)
        mai.metadata = meta

    def _record_feedback(
        self,
        mai: MAI,
        *,
        accepted: bool,
        trust_gain: float,
        harm_delta: float,
        latency: float,
        source: str,
    ) -> None:
        signature = str(mai.metadata.get("pattern_signature", mai.id))
        variant = str(mai.metadata.get("variant_tag", "base"))
        strength = float(mai.metadata.get("pattern_strength", 0.0))

        self._feedback.record(
            signature,
            accepted=accepted,
            trust_gain=trust_gain,
            harm_delta=harm_delta,
            source=source,
            variant=variant,
            strength=strength,
            latency_s=float(latency),
        )
        self._telemetry.log(
            "evaluation_feedback",
            {
                "mai_id": mai.id,
                "signature": signature,
                "accepted": accepted,
                "trust_gain": trust_gain,
                "harm_delta": harm_delta,
                "latency": latency,
                "source": source,
            },
        )

    # ---- 3) Remise au pipeline d’évaluation existant ----
    def submit_for_evaluation(self, mai_list: List[MAI]) -> None:
        """
        Passe chaque MAI au pipeline d'évaluation existant :
        - contrefactuels via world_model social
        - scoring via social_critic
        - ablation/sandbox via self_improver
        - promotion → mechanism_store.add/update
        """
        # 1) Essaye d’utiliser l’évaluateur central (si dispo)
        try:
            from AGI_Evolutive.cognition.evolution_manager import EvolutionManager
            try:
                from AGI_Evolutive.self_improver.promote import PromoteManager

                promoter: Optional[PromoteManager] = PromoteManager()
            except Exception:
                promoter = None

            evaluator = EvolutionManager.shared() if hasattr(EvolutionManager, "shared") else EvolutionManager()
            for m in mai_list:
                started = time.time()
                ok = False
                try:
                    ok = bool(evaluator.evaluate_mechanism(m))
                except Exception:
                    ok = False

                accepted = False
                if ok:
                    try:
                        accepted = (
                            promoter.promote_mechanism(m)
                            if promoter and hasattr(promoter, "promote_mechanism")
                            else True
                        )
                    except Exception:
                        accepted = ok
                    if accepted:
                        self.store.add(m)

                self._record_feedback(
                    m,
                    accepted=accepted,
                    trust_gain=0.0,
                    harm_delta=0.0,
                    latency=time.time() - started,
                    source="evolution_manager",
                )
            return
        except Exception:
            pass

        # 2) Fallback direct (robuste si pas de singleton ou API différente)
        try:
            from AGI_Evolutive.social.social_critic import SocialCritic
            from AGI_Evolutive.world_model import SocialModel
            from AGI_Evolutive.self_improver.promote import PromoteManager

            critic = SocialCritic()
            sim = SocialModel()
            promoter = PromoteManager()

            for m in mai_list:
                # (a) Simuler quelques cas contrefactuels avec/sans MAI
                batch = (
                    sim.build_counterfactual_batch_for_mechanism(m)
                    if hasattr(sim, "build_counterfactual_batch_for_mechanism")
                    else []
                )

                scores_with, scores_without = [], []
                for case in batch:
                    s_with = critic.score(sim.run(case, enable_mechanism=m))
                    s_without = critic.score(sim.run(case, enable_mechanism=None))
                    scores_with.append(s_with)
                    scores_without.append(s_without)

                # (b) Décision : bénéfice social net + pas de régression critique
                def agg(slist, key, default=0.0):
                    return sum(float(s.get(key, default)) for s in slist) / max(1, len(slist))

                trust_gain = agg(scores_with, "trust") - agg(scores_without, "trust")
                harm_diff = agg(scores_with, "harm") - agg(scores_without, "harm")

                ok = (trust_gain > 0.0) and (harm_diff <= 0.0)
                # (c) Sandbox / ablation (si dispo)
                if hasattr(promoter, "sandbox_mechanism"):
                    ok = ok and promoter.sandbox_mechanism(m)

                # (d) Promotion
                if ok:
                    accepted = (
                        promoter.promote_mechanism(m)
                        if hasattr(promoter, "promote_mechanism")
                        else True
                    )
                    if accepted:
                        self.store.add(m)
                else:
                    accepted = False

                self._record_feedback(
                    m,
                    accepted=accepted,
                    trust_gain=trust_gain,
                    harm_delta=harm_diff,
                    latency=0.0,
                    source="manual_fallback",
                )

        except Exception:
            # En dernier recours: ne rien faire (pas de promotion silencieuse)
            return

    def run(self, recent_docs, recent_dialogues, metrics_snapshot: Dict[str, Any]):
        started = time.time()
        produced = 0
        selected = 0
        try:
            from AGI_Evolutive.social.interaction_miner import InteractionMiner

            miner = InteractionMiner()
            patterns = miner.extract_normative_patterns(recent_docs, recent_dialogues)
            candidates = self.induce_from_patterns(patterns)
            produced = len(candidates)
            candidates = self.prefilter(candidates, history_stats=metrics_snapshot or {})
            selected = len(candidates)
            self.submit_for_evaluation(candidates)
        except Exception:
            pass
        finally:
            latency = time.time() - started
            self._telemetry.log(
                "run_summary",
                {
                    "produced": produced,
                    "selected": selected,
                    "latency": latency,
                    "metrics_snapshot": metrics_snapshot,
                },
            )
        return self._llm_summarize_run(
            produced=produced,
            selected=selected,
            latency=latency,
            metrics_snapshot=metrics_snapshot,
        )
