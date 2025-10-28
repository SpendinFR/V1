import time
from typing import Any, Dict, List, Optional, Set, Tuple

from AGI_Evolutive.utils.llm_service import try_call_llm_dict


def _bounded_append(buffer: List[Dict[str, Any]], entry: Dict[str, Any], max_len: int) -> None:
    if max_len <= 0:
        return
    buffer.append(entry)
    if len(buffer) > max_len:
        del buffer[:-max_len]


def _success_rate(success: int, fail: int, default: float = 0.5) -> float:
    try:
        success = max(0, int(success))
        fail = max(0, int(fail))
    except Exception:
        return float(default)
    total = success + fail
    if total <= 0:
        return float(default)
    return float(success / total)


def _parse_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _collect_history(identity: Dict[str, Any]) -> Tuple[Dict[str, Any], int, Optional[float]]:
    history = identity.get("principles_history") if isinstance(identity, dict) else {}
    if not isinstance(history, dict):
        return {}, 30, None
    max_len = int(history.get("max_len", 30) or 30)
    last_success_rate = history.get("last_success_rate")
    return history.get("by_key", {}), max_len, last_success_rate if isinstance(last_success_rate, (float, int)) else None


def _collect_commitment_impact(identity: Dict[str, Any]) -> Dict[str, Any]:
    commitments = identity.get("commitments") if isinstance(identity, dict) else {}
    if not isinstance(commitments, dict):
        return {}
    impact = commitments.get("impact")
    return impact if isinstance(impact, dict) else {}


_HISTORY_ALIAS = {
    "privacy_mode": "respect_privacy",
    "risk_aversion": "prudence",
    "abstention_threshold": "do_no_harm",
    "max_depth": "keep_learning",
}


def _slugify(value: str) -> str:
    slug = "".join(ch if ch.isalnum() else "_" for ch in value.lower())
    slug = slug.strip("_")
    return slug or "custom"


def _resolve_policy(arch) -> Any:
    if arch is None:
        return None
    if hasattr(arch, "policy"):
        return arch.policy
    core = getattr(arch, "core", None)
    if core is not None and hasattr(core, "policy"):
        return core.policy
    return getattr(arch, "_policy_engine", None)


def _collect_policy_stats(policy: Any) -> Tuple[int, int]:
    try:
        stats = getattr(policy, "stats", None)
        if isinstance(stats, dict):
            return int(stats.get("success", 0) or 0), int(stats.get("fail", 0) or 0)
    except Exception:
        pass
    return 0, 0


def _strength_from_history(base: float, history: Dict[str, Any], key: str) -> float:
    entry = history.get(key)
    if not isinstance(entry, dict):
        return base
    confidence = entry.get("confidence")
    if isinstance(confidence, (int, float)):
        # scale confidence (0..1) to a modifier around the base value
        modifier = (float(confidence) - 0.5) * 0.6
        return float(min(0.95, max(0.2, base + modifier)))
    return base


def _principles_changed(current: List[Dict[str, Any]], new: List[Dict[str, Any]]) -> bool:
    def _norm(items: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
        normalized: List[Tuple[str, str]] = []
        for item in items or []:
            key = str(item.get("key", ""))
            desc = str(item.get("desc", ""))
            normalized.append((key, desc))
        normalized.sort()
        return normalized

    return _norm(current) != _norm(new)


def _record_principles_application(
    self_model: Any,
    principles: List[Dict[str, Any]],
    commitments: List[Dict[str, Any]],
    success: int,
    fail: int,
) -> None:
    if not self_model or not hasattr(self_model, "identity") or not hasattr(self_model, "set_identity_patch"):
        return
    identity = getattr(self_model, "identity", {}) or {}
    history_by_key, max_len, _ = _collect_history(identity)
    recent = []
    history = identity.get("principles_history")
    if isinstance(history, dict):
        recent = list(history.get("recent", []))
    now_ts = time.time()
    entry = {
        "ts": now_ts,
        "principles": [p.get("key") for p in principles if p.get("key")],
        "commitments": [c.get("key") for c in commitments if c.get("key")],
        "success": int(success),
        "fail": int(fail),
        "success_rate": _success_rate(success, fail),
    }
    _bounded_append(recent, entry, max_len)

    updated_history = dict(history_by_key)
    for key in entry["principles"]:
        stats = dict(updated_history.get(key, {"success": 0, "fail": 0}))
        stats["success"] = int(stats.get("success", 0)) + int(success)
        stats["fail"] = int(stats.get("fail", 0)) + int(fail)
        stats["last_seen"] = now_ts
        stats["confidence"] = _success_rate(stats["success"], stats["fail"])
        updated_history[key] = stats

    impact = _collect_commitment_impact(identity)
    updated_impact = dict(impact)
    for key in entry["commitments"]:
        stats = dict(updated_impact.get(key, {"success": 0, "fail": 0}))
        stats["success"] = int(stats.get("success", 0)) + int(success)
        stats["fail"] = int(stats.get("fail", 0)) + int(fail)
        stats["last_seen"] = now_ts
        stats["confidence"] = _success_rate(stats["success"], stats["fail"])
        updated_impact[key] = stats

    runs = 0
    if isinstance(history, dict):
        runs = int(history.get("runs", 0) or 0)

    patch = {
        "principles_history": {
            "recent": recent,
            "by_key": updated_history,
            "max_len": max_len,
            "runs": runs + 1,
            "last_success_rate": entry["success_rate"],
        },
        "commitments": {
            "impact": updated_impact,
        },
    }
    try:
        self_model.set_identity_patch(patch)
    except Exception:
        # best-effort logging shouldn't prevent the main flow
        return


def extract_effective_policies(arch) -> List[Dict[str, Any]]:
    """
    Inspecte la Policy pour lister des règles / comportements déjà effectifs.
    Retourne [{key, desc, strength}, ...]
    """
    out: List[Dict[str, Any]] = []
    pol = _resolve_policy(arch)
    if not pol:
        return out

    self_model = getattr(arch, "self_model", None)
    identity: Dict[str, Any] = {}
    if self_model and isinstance(getattr(self_model, "identity", None), dict):
        identity = self_model.identity
    history_by_key, _, _ = _collect_history(identity)
    commitment_impact = _collect_commitment_impact(identity)
    success, fail = _collect_policy_stats(pol)
    success_rate = _success_rate(success, fail)

    for key in ["abstention_threshold", "max_depth", "risk_aversion", "privacy_mode"]:
        if not hasattr(pol, key):
            continue
        val = getattr(pol, key)
        desc = f"{key}={val}"
        base_strength = 0.6
        alias = _HISTORY_ALIAS.get(key)
        if alias:
            base_strength = _strength_from_history(base_strength, history_by_key, alias)
            base_strength = _strength_from_history(base_strength, commitment_impact, alias)
        if key in ("abstention_threshold", "risk_aversion"):
            base_strength = float(min(0.95, max(0.2, base_strength + (success_rate - 0.5) * 0.4)))
        out.append({"key": key, "desc": desc, "strength": round(base_strength, 3)})

    if success or fail:
        consistency_strength = _strength_from_history(0.5 + (success_rate - 0.5) * 0.5, history_by_key, "consistency")
        out.append(
            {
                "key": "consistency",
                "desc": f"success={success}/fail={fail} (rate={success_rate:.2f})",
                "strength": round(float(min(0.95, max(0.2, consistency_strength))), 3),
            }
        )
    return out


def map_to_principles(
    effective: List[Dict[str, Any]],
    values: List[str],
    identity: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Convertit des règles effectives et des valeurs en étiquette de principes {key, desc}.

    L'algorithme hiérarchise les principes en combinant :
      - les valeurs déclarées,
      - les règles effectives détectées,
      - l'historique de succès/échec stocké dans SelfModel.identity.
    """

    identity = identity or {}
    history_by_key, _, last_success_rate = _collect_history(identity)
    commitment_impact = _collect_commitment_impact(identity)

    def history_adjust(priority: float, key: str) -> float:
        priority = _strength_from_history(priority, history_by_key, key)
        priority = _strength_from_history(priority, commitment_impact, key)
        return float(min(1.0, max(0.1, priority)))

    candidates: List[Dict[str, Any]] = []
    values = values or []
    low = [v.lower() for v in values]
    matched_values: Set[str] = set()

    keyword_rules = [
        ({"care", "sécurité", "safety"}, "do_no_harm", "Minimiser les risques et préserver l'intégrité des personnes.", 0.85),
        ({"privacy", "confidentialité"}, "respect_privacy", "Ne pas divulguer d'informations sensibles sans autorisation explicite.", 0.8),
        ({"curiosity", "exploration"}, "keep_learning", "Apprendre en continu à partir des retours et des expériences.", 0.78),
        ({"autonomy", "liberté"}, "support_autonomy", "Soutenir l'autonomie de l'utilisateur et offrir des options réversibles.", 0.65),
        ({"justice", "équité"}, "fairness", "Traiter les parties prenantes équitablement et éviter les biais.", 0.72),
    ]

    for value in low:
        for keywords, key, desc, base_priority in keyword_rules:
            if any(word in value for word in keywords):
                priority = history_adjust(base_priority, key)
                candidates.append({"key": key, "desc": desc, "priority": priority})
                matched_values.add(value)

    honesty_priority = history_adjust(0.75, "honesty")
    candidates.append(
        {
            "key": "honesty",
            "desc": "Dire quand l'incertitude est élevée, expliciter les sources et les limites.",
            "priority": honesty_priority,
        }
    )

    if last_success_rate is not None and last_success_rate < 0.45:
        candidates.append(
            {
                "key": "resilience",
                "desc": "Analyser les échecs récents et ajuster les engagements avant de les relancer.",
                "priority": history_adjust(0.7, "resilience"),
            }
        )

    for eff in effective or []:
        key = eff.get("key")
        if not key:
            continue
        desc = str(eff.get("desc", "")).lower()
        if key == "privacy_mode":
            candidates.append(
                {
                    "key": "respect_privacy",
                    "desc": "Renforcer le cloisonnement des données et surveiller les accès sensibles.",
                    "priority": history_adjust(0.82, "respect_privacy"),
                }
            )
        elif key == "risk_aversion":
            candidates.append(
                {
                    "key": "prudence",
                    "desc": "Préférer les actions réversibles lorsque l'incertitude est élevée.",
                    "priority": history_adjust(0.8, "prudence"),
                }
            )
        elif key == "abstention_threshold":
            threshold = None
            if "=" in desc:
                threshold = _parse_float(desc.split("=")[-1])
            dynamic_desc = "Maintenir un seuil d'abstention adapté pour éviter les réponses non fiables."
            if isinstance(threshold, float):
                dynamic_desc += f" (seuil actuel ≈ {threshold:.2f})"
            candidates.append(
                {
                    "key": "do_no_harm",
                    "desc": dynamic_desc,
                    "priority": history_adjust(0.76, "do_no_harm"),
                }
            )
        elif key == "max_depth":
            depth = None
            if "=" in desc:
                depth = _parse_float(desc.split("=")[-1])
            adaptive_desc = "Adapter la profondeur de raisonnement aux contraintes de temps et de risque."
            if isinstance(depth, float):
                adaptive_desc += f" (profondeur actuelle ≈ {depth:.0f})"
            candidates.append(
                {
                    "key": "structured_reasoning",
                    "desc": adaptive_desc,
                    "priority": history_adjust(0.7, "structured_reasoning"),
                }
            )

    for value in low:
        if value in matched_values:
            continue
        slug = _slugify(value)
        key = f"value_{slug}"
        candidates.append(
            {
                "key": key,
                "desc": f"Honorer la valeur déclarée « {value} » dans les arbitrages quotidiens.",
                "priority": history_adjust(0.55, key),
            }
        )

    candidates.sort(key=lambda item: item.get("priority", 0.0), reverse=True)
    dedup: Dict[str, Dict[str, Any]] = {}
    for cand in candidates:
        key = cand.get("key")
        if not key or key in dedup:
            continue
        dedup[key] = {"key": key, "desc": cand.get("desc", "")}
    return list(dedup.values())


def propose_commitments(
    principles: List[Dict[str, Any]],
    identity: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Propose des engagements booléens à activer dans SelfModel.identity.commitments.by_key.

    L'activation tient compte de l'impact historique enregistré : un engagement ayant
    un faible taux de succès est reproposé en mode inactif pour validation explicite.
    """

    identity = identity or {}
    impact = _collect_commitment_impact(identity)
    props: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    def add_commitment(key: str, base_note: str, default_active: bool = True) -> None:
        if not key or key in seen:
            return
        data = impact.get(key, {}) if isinstance(impact, dict) else {}
        active = default_active
        note = base_note
        confidence = None
        if isinstance(data, dict):
            confidence = data.get("confidence")
            if isinstance(confidence, (int, float)):
                confidence = float(confidence)
                if confidence < 0.35:
                    active = False
                elif confidence > 0.85:
                    active = True
                note += f" (historique: {confidence * 100:.0f}% de réussite)"
        props.append({"key": key, "active": bool(active), "note": note})
        seen.add(key)

    for p in principles or []:
        pk = p.get("key")
        if pk == "respect_privacy":
            add_commitment(
                "respect_privacy",
                "Renforcer le contrôle d'accès et la journalisation des usages sensibles.",
            )
        elif pk == "honesty":
            add_commitment(
                "disclose_uncertainty",
                "Exprimer explicitement l'incertitude lorsque la confiance tombe sous le seuil défini.",
            )
        elif pk == "keep_learning":
            add_commitment(
                "continuous_learning",
                "Programmer des boucles de revue pour capitaliser sur les retours utilisateurs.",
            )
        elif pk == "prudence":
            add_commitment(
                "risk_review",
                "Exiger une revue rapide des risques avant les actions irréversibles.",
            )
        elif pk == "structured_reasoning":
            add_commitment(
                "monitor_reasoning_depth",
                "Adapter dynamiquement la profondeur de raisonnement selon la charge et les délais.",
            )
        elif pk == "support_autonomy":
            add_commitment(
                "offer_reversible_options",
                "Prévoir des issues de secours et des options réversibles pour l'utilisateur final.",
            )
        elif pk == "fairness":
            add_commitment(
                "bias_check",
                "Intégrer une vérification systématique des biais avant de déployer une recommandation.",
            )
        elif pk == "resilience":
            add_commitment(
                "postmortem_reviews",
                "Planifier des revues post-incident pour identifier les apprentissages concrets.",
                default_active=False,
            )

    return props


def _llm_refine_principles(
    *,
    identity: Dict[str, Any],
    effective: List[Dict[str, Any]],
    principles: List[Dict[str, Any]],
    commitments: List[Dict[str, Any]],
    stats: Dict[str, Any],
    logger: Any,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    payload = {
        "identity": {
            "values": (identity.get("preferences", {}) or {}).get("values", []),
            "history": identity.get("principles_history"),
            "commitments": identity.get("commitments"),
        },
        "effective_rules": effective,
        "principles": principles,
        "commitments": commitments,
        "stats": stats,
    }
    response = try_call_llm_dict(
        "cognition_identity_principles",
        input_payload=payload,
        logger=logger,
    )
    if not isinstance(response, dict):
        return principles, commitments, {}

    def _sanitize_principles(items: Any) -> List[Dict[str, Any]]:
        sanitized: List[Dict[str, Any]] = []
        if not isinstance(items, list):
            return sanitized
        for item in items:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key", "")).strip()
            desc = str(item.get("desc", "")).strip()
            if not key:
                continue
            sanitized.append({"key": key, "desc": desc})
        return sanitized

    def _sanitize_commitments(items: Any) -> List[Dict[str, Any]]:
        sanitized: List[Dict[str, Any]] = []
        if not isinstance(items, list):
            return sanitized
        for item in items:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key", "")).strip()
            if not key:
                continue
            entry = {"key": key}
            if "active" in item:
                entry["active"] = bool(item["active"])
            if isinstance(item.get("note"), str) and item["note"].strip():
                entry["note"] = item["note"].strip()
            sanitized.append(entry)
        return sanitized

    updated_principles = _sanitize_principles(response.get("principles")) or principles
    updated_commitments = _sanitize_commitments(response.get("commitments")) or commitments

    meta: Dict[str, Any] = {"llm": response}
    if "notes" in response and isinstance(response["notes"], str):
        meta["notes"] = response["notes"].strip()
    if "confidence" in response:
        try:
            meta["confidence"] = max(0.0, min(1.0, float(response["confidence"])))
        except (TypeError, ValueError):
            pass

    return updated_principles, updated_commitments, meta


def run_and_apply_principles(arch, require_confirmation: bool = True) -> Dict[str, Any]:
    """
    Chaîne complète:
      - extrait règles effectives + valeurs
      - mappe en principes
      - propose des engagements
      - écrit (Policy.apply_proposal) si pas sensible, sinon retourne 'needs_confirmation'
    """
    if not hasattr(arch, "self_model"):
        return {"applied": [], "pending": [], "principles": [], "commitments": []}

    self_model = arch.self_model
    self_model.ensure_identity_paths()
    identity = getattr(self_model, "identity", {}) or {}
    values = identity.get("preferences", {}).get("values", []) if isinstance(identity, dict) else []

    eff = extract_effective_policies(arch)
    prin = map_to_principles(eff, values, identity=identity)
    commits = propose_commitments(prin, identity=identity)

    applied: List[str] = []
    pending: List[str] = []

    policy = _resolve_policy(arch)
    success, fail = _collect_policy_stats(policy)
    stats = {"success": success, "fail": fail, "success_rate": _success_rate(success, fail)}

    prin, commits, llm_meta = _llm_refine_principles(
        identity=identity,
        effective=eff,
        principles=prin,
        commitments=commits,
        stats=stats,
        logger=getattr(arch, "logger", None),
    )

    if prin:
        current_principles = identity.get("principles", []) if isinstance(identity, dict) else []
        if _principles_changed(current_principles, prin):
            proposal = {
                "type": "update",
                "path": ["identity", "principles"],
                "value": prin,
                "rationale": "Derived from effective Policy and current values.",
            }
            try:
                if policy is not None:
                    self_model.apply_proposal(proposal, policy)
                    applied.append("principles")
                else:
                    pending.append("principles")
            except Exception:
                pending.append("principles")
        else:
            applied.append("principles")

    for cm in commits:
        key = cm.get("key")
        if not key:
            continue
        label = f"commit:{key}"
        if require_confirmation:
            pending.append(label)
            continue
        try:
            desired_active = bool(cm.get("active", True))
            note = cm.get("note", "")
            current_active = self_model.has_commitment(key)
            if current_active == desired_active and not note:
                applied.append(label)
                continue
            self_model.set_commitment(key, desired_active, note=note)
            applied.append(label)
        except Exception:
            pending.append(label)

    if prin or commits:
        _record_principles_application(self_model, prin, commits, success, fail)

    result = {
        "applied": applied,
        "pending": pending,
        "principles": prin,
        "commitments": commits,
    }
    if llm_meta:
        result["llm"] = llm_meta
    return result
