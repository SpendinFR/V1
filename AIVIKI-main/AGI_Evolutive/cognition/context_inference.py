from __future__ import annotations

from typing import Any, Dict, Tuple
import math
import os
import platform
import statistics
import time

from AGI_Evolutive.utils.llm_service import try_call_llm_dict

DEFAULT_THRESHOLD = 0.70
DEFAULT_STABLE_CYCLES = 2


def _ensure_where_state(arch: Any) -> Dict[str, Any]:
    """Initialise ou récupère l'état d'inférence contextuelle persistant."""

    state = getattr(arch, "_where_state", None)
    if isinstance(state, dict):
        return state

    state = {
        "logreg": {"weights": {}, "bias": 0.0, "lr": 0.05, "seen": 0},
        "platt": {"A": 0.0, "B": 0.0, "lr": 0.01, "seen": 0},
        "beta": {
            "runtime": [1.0, 1.0],
            "workspace": [1.0, 1.0],
            "context": [1.0, 1.0],
        },
        "drift": {
            "ema": DEFAULT_THRESHOLD,
            "cusum": 0.0,
            "min_cusum": 0.0,
            "delta": 0.01,
            "lambda": 0.1,
            "adaptive_threshold": DEFAULT_THRESHOLD,
        },
        "history": {
            "where": [],
            "max_len": 25,
        },
    }
    arch._where_state = state
    return state


def _bounded_log(history: Dict[str, Any], entry: Dict[str, Any]) -> None:
    """Ajoute une entrée dans le log circulaire de l'historique."""

    log = history.setdefault("where", [])
    max_len = int(history.get("max_len", 25))
    log.append(entry)
    if len(log) > max_len:
        del log[0 : len(log) - max_len]


def _update_beta(beta_state: Dict[str, list[float]], signals: Dict[str, bool]) -> Dict[str, float]:
    """Met à jour les distributions Beta-Bernoulli pour chaque signal."""

    confidences: Dict[str, float] = {}
    for name, success in signals.items():
        params = beta_state.setdefault(name, [1.0, 1.0])
        if success:
            params[0] += 1.0
        else:
            params[1] += 1.0
        total = params[0] + params[1]
        confidences[name] = params[0] / total if total else 0.5
    return confidences


def _page_hinkley_update(drift_state: Dict[str, Any], value: float) -> None:
    """Détecteur de dérive (Page-Hinkley) pour ajuster le seuil adaptatif."""

    ema = drift_state.get("ema", value)
    ema = 0.9 * ema + 0.1 * value
    drift_state["ema"] = ema

    delta = float(drift_state.get("delta", 0.01))
    lam = float(drift_state.get("lambda", 0.1))
    cusum = drift_state.get("cusum", 0.0) + value - ema - delta
    min_cusum = min(drift_state.get("min_cusum", 0.0), cusum)
    drift_state["cusum"] = cusum
    drift_state["min_cusum"] = min_cusum

    adaptive = drift_state.get("adaptive_threshold", DEFAULT_THRESHOLD)
    if cusum - min_cusum > lam:
        # dérive détectée → réduire légèrement le seuil pour se réadapter
        adaptive = max(0.5, adaptive - 0.05)
        drift_state["cusum"] = 0.0
        drift_state["min_cusum"] = 0.0
    else:
        # relâcher progressivement vers la moyenne observée
        adaptive = min(0.95, 0.95 * adaptive + 0.05 * max(ema, DEFAULT_THRESHOLD))
    drift_state["adaptive_threshold"] = adaptive


def _logistic_predict(logreg_state: Dict[str, Any], features: Dict[str, float]) -> float:
    bias = float(logreg_state.get("bias", 0.0))
    weights: Dict[str, float] = logreg_state.setdefault("weights", {})
    z = bias
    for key, value in features.items():
        z += weights.get(key, 0.0) * value
    try:
        return 1.0 / (1.0 + math.exp(-z))
    except OverflowError:
        return 1.0 if z > 0 else 0.0


def _logistic_update(logreg_state: Dict[str, Any], features: Dict[str, float], label: float, pred: float) -> None:
    lr = float(logreg_state.get("lr", 0.05))
    error = label - pred
    weights = logreg_state.setdefault("weights", {})
    for key, value in features.items():
        weights[key] = weights.get(key, 0.0) + lr * error * value
    logreg_state["bias"] = float(logreg_state.get("bias", 0.0) + lr * error)
    logreg_state["seen"] = int(logreg_state.get("seen", 0)) + 1


def _platt_calibrate(platt_state: Dict[str, Any], prob: float) -> float:
    prob = min(max(prob, 1e-6), 1.0 - 1e-6)
    logit = math.log(prob / (1.0 - prob))
    A = float(platt_state.get("A", 0.0))
    B = float(platt_state.get("B", 0.0))
    try:
        calibrated = 1.0 / (1.0 + math.exp(A * logit + B))
    except OverflowError:
        calibrated = 0.0 if (A * logit + B) > 0 else 1.0
    return calibrated


def _platt_update(platt_state: Dict[str, Any], prob: float, label: float) -> None:
    prob = min(max(prob, 1e-6), 1.0 - 1e-6)
    logit = math.log(prob / (1.0 - prob))
    A = float(platt_state.get("A", 0.0))
    B = float(platt_state.get("B", 0.0))
    lr = float(platt_state.get("lr", 0.01))
    pred = 1.0 / (1.0 + math.exp(A * logit + B))
    error = label - pred
    platt_state["A"] = A + lr * error * logit
    platt_state["B"] = B + lr * error
    platt_state["seen"] = int(platt_state.get("seen", 0)) + 1


def _extract_features(
    where: Dict[str, Any],
    hits: int,
    beta_conf: Dict[str, float],
    last: Dict[str, Any] | None,
) -> Dict[str, float]:
    features: Dict[str, float] = {}

    runtime_score = float(where.get("runtime_score") or 0.0)
    workspace_score = float(where.get("workspace_score") or 0.0)
    context_score = float(where.get("context_score") or 0.0)
    global_score = float(where.get("global_score") or 0.0)

    features["runtime_score"] = runtime_score
    features["workspace_score"] = workspace_score
    features["context_score"] = context_score
    features["global_score"] = global_score

    mean_conf = statistics.fmean(beta_conf.values()) if beta_conf else 0.5
    features["mean_conf"] = mean_conf
    features["runtime_conf"] = beta_conf.get("runtime", mean_conf)
    features["workspace_conf"] = beta_conf.get("workspace", mean_conf)
    features["context_conf"] = beta_conf.get("context", mean_conf)

    features["stable_hits"] = min(hits / 10.0, 1.0)

    ws = where.get("workspace") or {}
    files = ws.get("files") or []
    features["workspace_files_norm"] = min(len(files) / 200.0, 1.0)
    features["has_git"] = 1.0 if ws.get("has_git") else 0.0

    ctx = where.get("context") or {}
    lang = (ctx.get("user_lang") or "").lower()
    features["lang_fr"] = 1.0 if lang == "fr" else 0.0
    features["lang_en"] = 1.0 if lang == "en" else 0.0

    if last and last.get("stamp"):
        elapsed = time.time() - float(last.get("stamp", 0.0))
        features["elapsed_norm"] = min(elapsed / 30.0, 1.0)
    else:
        features["elapsed_norm"] = 0.0

    return features


def detect_runtime(last_runtime: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Détecte l'environnement d'exécution et mesure la stabilité."""

    now = time.time()
    rt = {
        "os": platform.system(),
        "os_release": platform.release(),
        "python": platform.python_version(),
        "python_build": " ".join(platform.python_build()),
        "python_compiler": platform.python_compiler(),
        "hostname": platform.node(),
        "cwd": os.getcwd(),
        "time": now,
        "pid": os.getpid(),
    }

    score = 0.9
    if last_runtime:
        same_os = (
            last_runtime.get("os") == rt["os"]
            and last_runtime.get("os_release") == rt["os_release"]
        )
        same_python = last_runtime.get("python") == rt["python"]
        same_host = last_runtime.get("hostname") == rt["hostname"]
        if not (same_os and same_python and same_host):
            score -= 0.15
        if last_runtime.get("cwd") != rt["cwd"]:
            score -= 0.05
    return {"runtime": rt, "score": max(0.6, score), "contradictions": 0}


def detect_workspace(
    paths_hint: Dict[str, str] | None = None,
    last_workspace: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    ws: Dict[str, Any] = {}
    try:
        cwd = os.getcwd()
        ws["root"] = cwd
        ws["has_git"] = os.path.exists(os.path.join(cwd, ".git"))
        ws["files"] = [
            fn
            for fn in os.listdir(cwd)[:200]
            if fn.endswith((".py", ".md", ".json", ".yml", ".yaml", ".toml", ".ts", ".js", ".ipynb"))
        ]
        if paths_hint:
            ws["paths_hint"] = {k: v for k, v in paths_hint.items() if v}
        ws["file_extensions"] = sorted({os.path.splitext(fn)[1] for fn in ws["files"]})
        ws["file_count"] = len(ws["files"])
        if last_workspace:
            ws["root_changed"] = last_workspace.get("root") != ws["root"]
            last_files = set(last_workspace.get("files", []))
            ws["new_files"] = [fn for fn in ws["files"] if fn not in last_files]
            ws["disappeared_files"] = [fn for fn in last_files if fn not in set(ws["files"])]
    except Exception:
        pass
    score = 0.7 + 0.2 * (1.0 if ws.get("has_git") else 0.0)
    if ws.get("root_changed"):
        score -= 0.05
    return {"workspace": ws, "score": min(0.95, score), "contradictions": 0}


def _memory_recent_interactions(arch, limit: int) -> list[Dict[str, Any]]:
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
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return [m for m in data if m.get("kind") in {"interaction", "user", "message"}]


def _runtime_consistent(previous: Dict[str, Any] | None, current: Dict[str, Any]) -> bool:
    if not previous:
        return True
    keys = ["os", "os_release", "python", "hostname"]
    return all(previous.get(k) == current.get(k) for k in keys)


def _workspace_consistent(previous: Dict[str, Any] | None, current: Dict[str, Any]) -> bool:
    if not previous:
        return True
    return previous.get("root") == current.get("root")


def _context_consistent(previous: Dict[str, Any] | None, current: Dict[str, Any]) -> bool:
    if not previous:
        return True
    return previous.get("user_lang") == current.get("user_lang")


def infer_user_context(
    arch,
    lookback: int = 50,
    last_context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Déduit langue/style/préfs observées à partir des interactions récentes.
    """
    lang = "fr"
    style: Dict[str, Any] = {}
    intents = {"pref_detail": 0, "pref_concise": 0}
    try:
        msgs = _memory_recent_interactions(arch, lookback)
        fr_hits = 0
        for msg in msgs:
            text = (msg.get("summary") or msg.get("text") or "").lower()
            if any(ch in text for ch in "éàèùçôâîû"):
                fr_hits += 1
            if "résume" in text or "plus court" in text:
                intents["pref_concise"] += 1
            if "plus détaillé" in text or "explique davantage" in text:
                intents["pref_detail"] += 1
            if "en français" in text:
                style["lang"] = "fr"
            if "english" in text and "please" in text:
                style["lang"] = "en"
        if msgs and fr_hits < (len(msgs) / 3):
            lang = "en"
    except Exception:
        pass
    if intents["pref_concise"] and not intents["pref_detail"]:
        style["conciseness"] = "high"
    elif intents["pref_detail"] and not intents["pref_concise"]:
        style["conciseness"] = "low"
    elif intents["pref_concise"] and intents["pref_detail"]:
        style["conciseness"] = "balanced"

    if last_context and last_context.get("user_lang") != style.get("lang", lang):
        # signal d'incohérence → réduire confiance
        score = 0.6
    else:
        score = 0.7
    ctx = {"user_lang": style.get("lang", lang), "style_hint": style}
    return {"context": ctx, "score": score, "contradictions": 0}


def situational_summary(where: Dict[str, Any]) -> Tuple[str, float]:
    """
    Concatène un petit résumé et renvoie (texte, score global).
    """
    rt = where.get("runtime", {})
    ws = where.get("workspace", {})
    ctx = where.get("context", {})
    parts = []
    if rt:
        parts.append(
            f"Environnement: {rt.get('os')} {rt.get('os_release')} / Python {rt.get('python')} sur {rt.get('hostname')}."
        )
    if ws:
        root = ws.get("root")
        if root:
            parts.append(f"Espace de travail: {os.path.basename(root)} (git={ws.get('has_git')}).")
        if ws.get("file_count"):
            parts.append(
                f"{ws.get('file_count')} fichiers repérés ({', '.join(ws.get('file_extensions', [])[:5])})."
            )
    if ctx:
        parts.append(
            f"Contexte utilisateur: langue={ctx.get('user_lang')}, style_hint={ctx.get('style_hint')}."
        )
    summary = " ".join(parts) or "Contexte en cours d'inférence."
    sub_scores = [
        where.get("runtime_score"),
        where.get("workspace_score"),
        where.get("context_score"),
    ]
    sub = [float(s) for s in sub_scores if isinstance(s, (int, float))]
    score = sum(sub) / len(sub) if sub else 0.75
    return summary, score


def _build_llm_payload(
    *, where: Dict[str, Any], decision_payload: Dict[str, Any], state: Dict[str, Any]
) -> Dict[str, Any]:
    history = state.get("history") if isinstance(state, dict) else None
    recent = []
    if isinstance(history, dict):
        log = history.get("where")
        if isinstance(log, list):
            recent = log[-5:]
    return {
        "where": where,
        "decision": decision_payload,
        "recent_history": recent,
    }


def _merge_llm_context(
    heuristic_result: Dict[str, Any], llm_response: Dict[str, Any]
) -> Dict[str, Any]:
    result = dict(heuristic_result)

    if isinstance(llm_response.get("status"), str):
        result["status"] = llm_response["status"].strip()
    if isinstance(llm_response.get("summary"), str) and llm_response["summary"].strip():
        result["summary"] = llm_response["summary"].strip()
    if "score" in llm_response:
        try:
            result["score"] = max(0.0, min(1.0, float(llm_response["score"])))
        except (TypeError, ValueError):
            pass
    if "threshold" in llm_response:
        try:
            result["threshold"] = max(0.3, min(0.99, float(llm_response["threshold"])))
        except (TypeError, ValueError):
            pass
    if "confidence" in llm_response:
        try:
            result["confidence"] = max(0.0, min(1.0, float(llm_response["confidence"])))
        except (TypeError, ValueError):
            pass
    if isinstance(llm_response.get("notes"), str) and llm_response["notes"].strip():
        result["notes"] = llm_response["notes"].strip()
    if isinstance(llm_response.get("actions"), list):
        result["actions"] = [str(a) for a in llm_response["actions"] if str(a).strip()][:6]
    result["llm"] = llm_response
    return result


def infer_where_and_apply(
    arch,
    threshold: float = DEFAULT_THRESHOLD,
    stable_cycles: int = DEFAULT_STABLE_CYCLES,
) -> Dict[str, Any]:
    """
    Infère runtime/workspace/context. Si score>=threshold et stable N cycles, écrit dans identity.where.*.
    La stabilité est estimée via un cache léger sur l'orchestrateur.
    """
    state = _ensure_where_state(arch)

    last = getattr(arch, "_where_last", None)
    last_where = last.get("where") if isinstance(last, dict) else None

    rt = detect_runtime((last_where or {}).get("runtime"))
    paths_hint = getattr(getattr(arch, "job_manager", None), "paths", None)
    ws = detect_workspace(paths_hint, (last_where or {}).get("workspace"))
    uc = infer_user_context(arch, last_context=(last_where or {}).get("context"))

    where = {
        "runtime": rt.get("runtime"),
        "runtime_score": rt.get("score"),
        "workspace": ws.get("workspace"),
        "workspace_score": ws.get("score"),
        "context": uc.get("context"),
        "context_score": uc.get("score"),
    }
    summary, global_score = situational_summary(where)
    where["summary"] = summary
    where["global_score"] = global_score

    # Signal consistency & Beta fusion
    signals_consistency = {
        "runtime": _runtime_consistent((last_where or {}).get("runtime"), where["runtime"]),
        "workspace": _workspace_consistent((last_where or {}).get("workspace"), where["workspace"]),
        "context": _context_consistent((last_where or {}).get("context"), where["context"]),
    }
    beta_conf = _update_beta(state.setdefault("beta", {}), signals_consistency)

    # Stabilité temporelle
    stable = False
    try:
        if isinstance(last, dict):
            previous = last.get("where")
            stamp = float(last.get("stamp", 0.0))
            if previous == where and (time.time() - stamp) >= 2.0:
                hits = int(last.get("hits", 0)) + 1
            else:
                hits = 1
        else:
            hits = 1
    except Exception:
        hits = 1

    mean_conf = statistics.fmean(beta_conf.values()) if beta_conf else 0.5
    required_hits = max(1, int(round(stable_cycles * (1.0 + max(0.0, 0.8 - mean_conf)))))
    stable = hits >= required_hits

    # Features & probabilistic fusion
    features = _extract_features(where, hits, beta_conf, last)
    raw_prob = _logistic_predict(state.setdefault("logreg", {}), features)
    calibrated_prob = _platt_calibrate(state.setdefault("platt", {}), raw_prob)
    fused_prob = (calibrated_prob + features.get("mean_conf", 0.5) + global_score) / 3.0

    # Drift-aware thresholding
    _page_hinkley_update(state.setdefault("drift", {}), fused_prob)
    adaptive_threshold = state["drift"].get("adaptive_threshold", threshold)
    effective_threshold = max(0.5, min(0.95, 0.5 * threshold + 0.5 * adaptive_threshold))

    decision_payload = {
        "where": where,
        "hits": hits,
        "required_hits": required_hits,
        "fused_prob": fused_prob,
        "global_score": global_score,
        "effective_threshold": effective_threshold,
        "beta_conf": beta_conf,
        "stable": stable,
    }

    arch._where_last = {
        "stamp": time.time(),
        "where": where,
        "hits": hits,
        "decision": decision_payload,
    }

    status = "pending"
    if fused_prob >= effective_threshold and stable:
        try:
            arch.self_model.set_identity_patch(
                {
                    "where": {
                        "runtime": where["runtime"],
                        "workspace": where["workspace"],
                        "context": where["context"],
                        "summary": where["summary"],
                    }
                }
            )
            status = "applied"
        except Exception:
            status = "error"
    else:
        status = "pending"

    # Feedback loop & apprentissage en ligne
    feedback_entry = {
        "ts": time.time(),
        "status": status,
        "fused_prob": fused_prob,
        "features": features,
        "threshold": effective_threshold,
        "stable": stable,
    }
    _bounded_log(state.setdefault("history", {}), feedback_entry)

    label: float | None
    if status == "applied":
        label = 1.0
    elif status == "error":
        label = 0.0
    else:
        label = None

    if label is not None:
        _logistic_update(state.setdefault("logreg", {}), features, label, raw_prob)
        _platt_update(state.setdefault("platt", {}), raw_prob, label)

    heuristic_result = {
        "status": status,
        "score": fused_prob,
        "summary": summary,
        "threshold": effective_threshold,
        "stable": stable,
        "hits": hits,
        "required_hits": required_hits,
    }
    llm_payload = _build_llm_payload(where=where, decision_payload=decision_payload, state=state)
    llm_response = try_call_llm_dict(
        "cognition_context_inference",
        input_payload=llm_payload,
        logger=getattr(arch, "logger", None),
    )
    if isinstance(llm_response, dict):
        try:
            return _merge_llm_context(heuristic_result, dict(llm_response))
        except Exception:
            pass
    return heuristic_result
