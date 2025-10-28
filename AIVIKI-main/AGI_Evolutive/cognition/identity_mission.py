from typing import Any, Dict, List, Optional, Tuple, Mapping
import logging
import math
import random
import time

from AGI_Evolutive.cognition.homeostasis import OnlineLinearModel
from AGI_Evolutive.utils.jsonsafe import json_sanitize
from AGI_Evolutive.utils.llm_contracts import enforce_llm_contract
from AGI_Evolutive.utils.llm_service import try_call_llm_dict


logger = logging.getLogger(__name__)


def _norm(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _resolve_policy(arch) -> Any:
    if arch is None:
        return None
    if hasattr(arch, "policy"):
        return arch.policy
    core = getattr(arch, "core", None)
    if core is not None and hasattr(core, "policy"):
        return core.policy
    return getattr(arch, "_policy_engine", None)


def _memory_recent(arch, limit: int) -> List[Dict[str, Any]]:
    store = None
    memory = getattr(arch, "memory", None)
    if memory is not None:
        store = getattr(memory, "store", None)
    if store is None:
        store = getattr(arch, "_memory_store", None)
    if store is None or not hasattr(store, "get_recent"):
        return []
    try:
        data = store.get_recent(limit)
        return list(data) if isinstance(data, list) else []
    except Exception:
        return []


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def _mission_state(arch: Any) -> Dict[str, Any]:
    state = getattr(arch, "_identity_mission_state", None)
    if state is None:
        state = {
            "model": {
                "weights": {
                    "bias": 0.15,
                    "coverage": 0.45,
                    "constraint_overlap": 0.35,
                    "feedback": 0.25,
                    "topic_focus": 0.2,
                    "constraint_focus": 0.2,
                    "growth_focus": 0.2,
                    "coverage_feedback": 0.15,
                    "constraint_feedback": 0.15,
                    "coverage_squared": 0.1,
                    "constraint_squared": 0.1,
                },
                "lr": 0.05,
                "l2": 0.01,
                "forget": 0.002,
                "max_step": 0.05,
            },
            "calibration": {
                "a": 1.0,
                "b": 0.0,
                "lr": 0.05,
                "forget": 0.001,
                "max_step": 0.2,
            },
            "bandits": {
                "threshold": {
                    "options": [0.65, 0.7, 0.75, 0.8],
                    "alpha": [1.0, 1.0, 1.0, 1.0],
                    "beta": [1.0, 1.0, 1.0, 1.0],
                    "last_choice": 2,
                },
                "delta": {
                    "options": [0.05, 0.1, 0.15],
                    "alpha": [1.0, 1.0, 1.0],
                    "beta": [1.0, 1.0, 1.0],
                    "last_choice": 1,
                },
                "jobs_ttl": {
                    "options": [3.0, 7.0, 14.0, 30.0],
                    "alpha": [1.0, 1.0, 1.0, 1.0],
                    "beta": [1.0, 1.0, 1.0, 1.0],
                    "last_choice": 1,
                },
                "decisions_ttl": {
                    "options": [3.0, 7.0, 14.0, 30.0],
                    "alpha": [1.0, 1.0, 1.0, 1.0],
                    "beta": [1.0, 1.0, 1.0, 1.0],
                    "last_choice": 1,
                },
            },
            "stats": {"success": 0.0, "fail": 0.0},
            "drift": {"threshold": 0.35},
            "drift_log": [],
            "meta": {
                "jobs_per_day": 20,
                "decisions_per_day": 35,
            },
            "last_candidates": [],
            "last_choice": 0,
        }
        setattr(arch, "_identity_mission_state", state)
    return state


def _ensure_bandit_entry(
    container: Dict[str, Any], name: str, options: List[float], default_idx: int
) -> Dict[str, Any]:
    entry = container.setdefault(name, {})
    if entry.get("options") != options:
        entry["options"] = list(options)
    entry.setdefault("alpha", [1.0 for _ in options])
    entry.setdefault("beta", [1.0 for _ in options])
    if len(entry["alpha"]) != len(options):
        entry["alpha"] = [1.0 for _ in options]
    if len(entry["beta"]) != len(options):
        entry["beta"] = [1.0 for _ in options]
    entry.setdefault("last_choice", default_idx)
    return entry


def _bandit_sample(entry: Dict[str, Any]) -> Tuple[float, int]:
    options = entry.get("options", [])
    if not options:
        return 0.0, -1
    alpha = entry.setdefault("alpha", [1.0 for _ in options])
    beta = entry.setdefault("beta", [1.0 for _ in options])
    if len(alpha) != len(options):
        alpha[:] = [1.0 for _ in options]
    if len(beta) != len(options):
        beta[:] = [1.0 for _ in options]
    samples = [random.betavariate(max(1e-3, a), max(1e-3, b)) for a, b in zip(alpha, beta)]
    idx = max(range(len(options)), key=samples.__getitem__)
    entry["last_choice"] = idx
    entry["last_sample"] = options[idx]
    entry["last_time"] = time.time()
    return float(options[idx]), idx


def _bandit_update(entry: Dict[str, Any], reward: Optional[float]) -> None:
    if reward is None:
        return
    reward = _norm(reward)
    options = entry.get("options", [])
    if not options:
        return
    idx = int(entry.get("last_choice", -1))
    if idx < 0 or idx >= len(options):
        return
    alpha = entry.setdefault("alpha", [1.0 for _ in options])
    beta = entry.setdefault("beta", [1.0 for _ in options])
    alpha[idx] += reward
    beta[idx] += 1.0 - reward
    # prevent unbounded growth
    cap = entry.get("cap", 200.0)
    alpha[idx] = min(alpha[idx], cap)
    beta[idx] = min(beta[idx], cap)


def _mission_model_from_state(state: Dict[str, Any]) -> OnlineLinearModel:
    meta = state.setdefault("model", {})
    feature_names = [
        "bias",
        "coverage",
        "constraint_overlap",
        "feedback",
        "topic_focus",
        "constraint_focus",
        "growth_focus",
        "coverage_feedback",
        "constraint_feedback",
        "coverage_squared",
        "constraint_squared",
    ]
    weights = meta.get("weights") or {}
    model = OnlineLinearModel(
        feature_names,
        weights=weights,
        lr=float(meta.get("lr", 0.05)),
        l2=float(meta.get("l2", 0.01)),
        forget=float(meta.get("forget", 0.002)),
        max_step=float(meta.get("max_step", 0.05)),
    )
    if "weights" not in meta:
        meta["weights"] = dict(model.weights)
    return model


def _store_model(state: Dict[str, Any], model: OnlineLinearModel, before: Optional[Dict[str, float]] = None) -> None:
    meta = state.setdefault("model", {})
    for name in model.feature_names:
        if name != "bias":
            model.weights[name] = _norm(model.weights.get(name, 0.0))
    meta["weights"] = dict(model.weights)
    if before is not None:
        diff = 0.0
        for name in model.feature_names:
            prev = before.get(name, 0.0)
            diff += abs(model.weights.get(name, 0.0) - prev)
        drift_meta = state.setdefault("drift", {})
        threshold = float(drift_meta.get("threshold", 0.35))
        if diff >= threshold:
            log = state.setdefault("drift_log", [])
            log.append({
                "ts": time.time(),
                "delta": diff,
                "weights": dict(model.weights),
            })
            if len(log) > 50:
                del log[:-50]


def _apply_calibration(state: Dict[str, Any], raw_score: float) -> float:
    cal = state.setdefault("calibration", {})
    cal.setdefault("a", 1.0)
    cal.setdefault("b", 0.0)
    return _sigmoid(cal.get("a", 1.0) * raw_score + cal.get("b", 0.0))


def _update_calibration(state: Dict[str, Any], raw_score: float, target: float) -> None:
    cal = state.setdefault("calibration", {})
    lr = float(cal.get("lr", 0.05))
    forget = float(cal.get("forget", 0.001))
    max_step = float(cal.get("max_step", 0.2))
    a = float(cal.get("a", 1.0))
    b = float(cal.get("b", 0.0))
    pred = _sigmoid(a * raw_score + b)
    error = pred - target
    step_a = max(-max_step, min(max_step, lr * error * raw_score))
    step_b = max(-max_step, min(max_step, lr * error))
    a -= step_a
    b -= step_b
    a *= 1.0 - forget
    b *= 1.0 - forget
    cal["a"] = max(-5.0, min(5.0, a))
    cal["b"] = max(-5.0, min(5.0, b))


def _update_model(state: Dict[str, Any], features: Dict[str, float], target: float) -> None:
    model = _mission_model_from_state(state)
    before = dict(model.weights)
    raw = model.predict(features)
    prediction = _sigmoid(raw)
    model.update(features, target, prediction)
    _store_model(state, model, before=before)
    _update_calibration(state, raw, target)


def _extract_reward(policy: Any, state: Dict[str, Any]) -> Optional[float]:
    stats = getattr(policy, "stats", None)
    if not isinstance(stats, dict):
        return None
    success = float(stats.get("success", 0.0))
    fail = float(stats.get("fail", 0.0))
    stored = state.setdefault("stats", {"success": 0.0, "fail": 0.0})
    delta_success = max(0.0, success - float(stored.get("success", 0.0)))
    delta_fail = max(0.0, fail - float(stored.get("fail", 0.0)))
    total = delta_success + delta_fail
    stored["success"] = success
    stored["fail"] = fail
    if total <= 0.0:
        return None
    return delta_success / total


def mine_frequent_goals(
    arch,
    horizon_jobs: int = 500,
    horizon_decisions: Optional[int] = None,
    ttl_jobs: float = 7.0,
    ttl_decisions: float = 7.0,
) -> Dict[str, Any]:
    """
    Analyse les jobs/decisions récents pour extraire les thèmes dominants.
    Retourne {"topics":[(topic, coverage_score), ...], "evidence_refs":[...]}
    """
    topics: Dict[str, float] = {}
    refs: List[str] = []

    jm = getattr(arch, "job_manager", None)
    if jm and hasattr(jm, "snapshot_identity_view"):
        try:
            view = jm.snapshot_identity_view()
        except Exception:
            view = {}
        if horizon_decisions is None:
            horizon_decisions = min(200, horizon_jobs)
        recent_jobs = (view.get("recent") or [])[:horizon_jobs]
        for idx, r in enumerate(recent_jobs):
            topic = str(r.get("kind") or r.get("topic") or "__unknown__")
            weight = math.exp(-idx / max(1.0, ttl_jobs))
            topics[topic] = topics.get(topic, 0.0) + weight
            if r.get("job_id"):
                refs.append(f"job:{r['job_id']}")

    decision_limit = horizon_decisions if horizon_decisions is not None else min(200, horizon_jobs)
    recent_memories = list(reversed(_memory_recent(arch, min(400, max(50, decision_limit)))))
    count = 0
    for idx, entry in enumerate(recent_memories):
        if count >= decision_limit:
            break
        if str(entry.get("kind")) == "decision":
            topic = entry.get("topic") or "__generic__"
            weight = math.exp(-idx / max(1.0, ttl_decisions))
            topics[topic] = topics.get(topic, 0.0) + weight
            if entry.get("decision_id"):
                refs.append(f"decision:{entry['decision_id']}")
            count += 1

    total = sum(topics.values()) or 1
    ranked = sorted(((k, v / total) for k, v in topics.items()), key=lambda kv: kv[1], reverse=True)
    return {"topics": ranked[:10], "evidence_refs": refs[:200]}


def cluster_intent_constraints(arch, top_k: int = 5) -> Dict[str, Any]:
    """
    Interroge le modèle d'intent/contraintes pour obtenir les contraintes récurrentes.
    Retourne {"constraints":[("desc", weight), ...]}
    """
    lst: List[Tuple[str, float]] = []
    im = getattr(arch, "intent_model", None)
    if im and hasattr(im, "constraints_view"):
        try:
            view = im.constraints_view(top_k=top_k) or []
        except Exception:
            view = []
        for c in view:
            if not isinstance(c, dict):
                continue
            desc = c.get("description") or c.get("name") or ""
            if not desc:
                continue
            w = float(c.get("weight", 0.5))
            lst.append((desc, _norm(w)))
    return {"constraints": lst}


def _candidate_features(
    coverage: float,
    constraint_overlap: float,
    feedback_score: float,
    topic_focus: float,
    constraint_focus: float,
    growth_focus: float,
) -> Dict[str, float]:
    return {
        "bias": 1.0,
        "coverage": _norm(coverage),
        "constraint_overlap": _norm(constraint_overlap),
        "feedback": _norm(feedback_score),
        "topic_focus": _norm(topic_focus),
        "constraint_focus": _norm(constraint_focus),
        "growth_focus": _norm(growth_focus),
        "coverage_feedback": _norm(coverage * feedback_score),
        "constraint_feedback": _norm(constraint_overlap * feedback_score),
        "coverage_squared": _norm(coverage * coverage),
        "constraint_squared": _norm(constraint_overlap * constraint_overlap),
    }


def draft_mission_hypotheses(
    freq: Dict[str, Any],
    cons: Dict[str, Any],
    feedback_score: float,
    state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Produit 2–3 formulations candidates avec un score appris (GLM online + calibration).
    """
    topics = [t for t, _ in (freq.get("topics") or [])[:3]]
    cons_desc = [d for d, _ in (cons.get("constraints") or [])[:3]]

    cov = _norm(sum(w for _, w in (freq.get("topics") or [])[:3]))
    if cons.get("constraints"):
        con = sum(w for _, w in cons["constraints"][:3]) / max(1, len(cons["constraints"][:3]))
    else:
        con = 0.0
    con = _norm(con)
    fb = _norm(feedback_score)

    candidates: List[Tuple[str, Dict[str, float]]] = []
    if topics:
        joined_topics = ", ".join(topics)
        guard = ", ".join(cons_desc) if cons_desc else "les contraintes usuelles"
        features = _candidate_features(cov, con, fb, topic_focus=1.0, constraint_focus=0.7 if cons_desc else 0.3, growth_focus=0.3)
        candidates.append((f"Aider efficacement sur {joined_topics} en respectant {guard}.", features))
    hyp2 = "Maximiser la valeur des échanges en apprenant en continu et en garantissant la sécurité des informations."
    if all(hyp2 != text for text, _ in candidates):
        features = _candidate_features(cov * 0.6, con * 0.5, max(fb, 0.5), topic_focus=0.4, constraint_focus=0.4, growth_focus=1.0)
        candidates.append((hyp2, features))
    if cons_desc:
        hyp3 = f"Prioriser {cons_desc[0]} tout en améliorant la précision et la clarté des réponses."
        if all(hyp3 != text for text, _ in candidates):
            features = _candidate_features(cov * 0.4, min(1.0, con * 1.2), fb * 0.7, topic_focus=0.3, constraint_focus=1.0, growth_focus=0.4)
            candidates.append((hyp3, features))

    model_state = state or {}
    model = _mission_model_from_state(model_state)
    scored: List[Dict[str, Any]] = []
    for text, features in candidates[:3]:
        raw = model.predict(features)
        calibrated = _apply_calibration(model_state, raw)
        scored.append({
            "text": text,
            "score": _norm(calibrated),
            "raw": raw,
            "features": features,
        })

    scored.sort(key=lambda item: item["score"], reverse=True)
    if state is not None:
        state["last_candidates"] = scored
        state["last_choice"] = 0
    return {"candidates": [(item["text"], item["score"]) for item in scored]}


def recommend_and_apply_mission(
    arch,
    threshold: Optional[float] = None,
    delta_gate: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Chaîne complète: mine -> cluster -> draft -> décider d'écrire.
    Ecrit via Policy.apply_proposal si score >= threshold ET (best - second) >= delta_gate,
    sinon retourne une recommandation (l'orchestrateur pourra déclencher QM si blocage).
    """
    state = _mission_state(arch)
    bandits = state.setdefault("bandits", {})

    bandit_threshold = _ensure_bandit_entry(
        bandits, "threshold", [0.65, 0.7, 0.75, 0.8], 2
    )
    bandit_delta = _ensure_bandit_entry(
        bandits, "delta", [0.05, 0.1, 0.15], 1
    )
    bandit_jobs = _ensure_bandit_entry(
        bandits, "jobs_ttl", [3.0, 7.0, 14.0, 30.0], 1
    )
    bandit_decisions = _ensure_bandit_entry(
        bandits, "decisions_ttl", [3.0, 7.0, 14.0, 30.0], 1
    )

    auto_threshold = threshold is None
    auto_delta = delta_gate is None

    if threshold is None:
        threshold, _ = _bandit_sample(bandit_threshold)
    if delta_gate is None:
        delta_gate, _ = _bandit_sample(bandit_delta)

    jobs_ttl, _ = _bandit_sample(bandit_jobs)
    decisions_ttl, _ = _bandit_sample(bandit_decisions)

    jobs_ttl = jobs_ttl or 7.0
    decisions_ttl = decisions_ttl or 7.0

    jobs_per_day = max(10, int(state.get("meta", {}).get("jobs_per_day", 20)))
    decisions_per_day = max(10, int(state.get("meta", {}).get("decisions_per_day", 35)))

    horizon_jobs = max(50, int(jobs_per_day * jobs_ttl))
    horizon_decisions = max(50, int(decisions_per_day * decisions_ttl))

    freq = mine_frequent_goals(
        arch,
        horizon_jobs=horizon_jobs,
        horizon_decisions=horizon_decisions,
        ttl_jobs=jobs_ttl,
        ttl_decisions=decisions_ttl,
    )
    cons = cluster_intent_constraints(arch)

    fb = 0.6
    policy = _resolve_policy(arch)
    try:
        stats = getattr(policy, "stats", None)
        if isinstance(stats, dict):
            success = float(stats.get("success", 0))
            fail = float(stats.get("fail", 0))
            total = success + fail
            if total > 0:
                fb = _norm(success / total)
    except Exception:
        pass

    draft = draft_mission_hypotheses(freq, cons, fb, state=state)
    candidates = draft.get("candidates", [])
    if not candidates:
        return {"status": "no_candidates", "candidates": []}

    best = candidates[0]
    second = candidates[1] if len(candidates) > 1 else (None, 0.0)
    delta = best[1] - (second[1] if second else 0.0)

    original_best_text = best[0]
    selected_text = best[0]
    llm_axes: Optional[Dict[str, str]] = None
    llm_notes: Optional[str] = None

    llm_payload = {
        "frequent_goals": freq,
        "constraints": cons,
        "feedback_score": fb,
        "candidates": [{"text": text, "score": score} for text, score in candidates[:3]],
        "selected": {"text": best[0], "score": best[1]},
        "delta": delta,
        "threshold": threshold,
        "auto_threshold": auto_threshold,
        "auto_delta": auto_delta,
    }

    llm_result = try_call_llm_dict(
        "identity_mission",
        input_payload=json_sanitize(llm_payload),
        logger=logger,
    )

    cleaned = enforce_llm_contract("identity_mission", llm_result)
    if cleaned is not None:
        llm_result = cleaned

    if isinstance(llm_result, Mapping):
        mission_block = llm_result.get("mission")
        if not isinstance(mission_block, dict):
            keys = {"prioritaire", "support", "vision"}.intersection(llm_result.keys())
            if keys:
                mission_block = {key: llm_result.get(key) for key in ("prioritaire", "support", "vision")}
        if isinstance(mission_block, Mapping):
            axes: Dict[str, str] = {}
            axis_aliases = {
                "prioritaire": ("prioritaire", "priority", "focus"),
                "support": ("support", "backbone", "soutien"),
                "vision": ("vision", "aspiration", "north_star"),
            }
            for axis, aliases in axis_aliases.items():
                value = None
                for alias in aliases:
                    if alias in mission_block and mission_block.get(alias):
                        value = mission_block.get(alias)
                        break
                if isinstance(value, str) and value.strip():
                    axes[axis] = value.strip()
            if axes:
                llm_axes = axes
        mission_text = llm_result.get("mission_text") or llm_result.get("mission_statement")
        if not mission_text and llm_axes:
            mission_text = " | ".join(f"{k}: {v}" for k, v in llm_axes.items())
        if isinstance(mission_text, str) and mission_text.strip():
            selected_text = mission_text.strip()
        notes_val = llm_result.get("notes")
        if isinstance(notes_val, str) and notes_val.strip():
            llm_notes = notes_val.strip()

    best = (selected_text, best[1])

    decided = False
    if best[1] >= float(threshold) and delta >= float(delta_gate):
        proposal = {
            "type": "update",
            "path": ["identity", "purpose", "mission"],
            "value": selected_text,
            "rationale": "Mission inferred from frequent goals, constraints and feedback.",
            "evidence_refs": freq.get("evidence_refs", [])[:50],
        }
        try:
            if hasattr(arch, "self_model") and policy is not None:
                arch.self_model.apply_proposal(proposal, policy)
                decided = True
        except Exception:
            decided = False

    reward = _extract_reward(policy, state)
    if reward is not None and state.get("last_candidates"):
        idx = int(state.get("last_choice", 0))
        idx = max(0, min(idx, len(state["last_candidates"]) - 1))
        features = dict(state["last_candidates"][idx]["features"])
        _update_model(state, features, reward)
        if auto_threshold:
            _bandit_update(bandit_threshold, reward)
        if auto_delta:
            _bandit_update(bandit_delta, reward)
        _bandit_update(bandit_jobs, reward)
        _bandit_update(bandit_decisions, reward)

    llm_info: Optional[Dict[str, Any]] = None
    if llm_axes or llm_notes or selected_text != original_best_text:
        llm_info = {
            "mission": llm_axes,
            "text": selected_text,
            "notes": llm_notes,
        }
        llm_info = {k: v for k, v in llm_info.items() if v}
        if llm_info:
            llm_info["timestamp"] = time.time()
            state.setdefault("llm", {})["identity_mission"] = json_sanitize(llm_info)

    result = {
        "status": "applied" if decided else "needs_confirmation",
        "best": best,
        "second": second,
        "delta": delta,
        "freq": freq,
        "constraints": cons,
    }
    if llm_info:
        result["llm"] = llm_info
    return result
