"""Reasoning system responsible for producing structured hypothesis/test plans."""

import logging
import math
import random
import time
import unicodedata
from collections import deque
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..utils.llm_service import try_call_llm_dict
from .structures import Evidence, Hypothesis, Test, episode_record


logger = logging.getLogger(__name__)


class OnlineLinear:
    """Simple online generalized linear model with bounded weights."""

    def __init__(
        self,
        feature_names: Sequence[str],
        learning_rate: float = 0.15,
        l2: float = 0.01,
        weight_bounds: Sequence[float] = (-1.5, 1.5),
    ) -> None:
        self.feature_names = list(feature_names)
        self.learning_rate = learning_rate
        self.l2 = l2
        low, high = weight_bounds
        self.low, self.high = (low, high) if low <= high else (high, low)
        self.weights: Dict[str, float] = {name: 0.0 for name in self.feature_names}

    def predict(self, features: Dict[str, float]) -> float:
        z = sum(self.weights.get(name, 0.0) * features.get(name, 0.0) for name in self.feature_names)
        # Logistic squashing to keep score in [0, 1]
        return 1.0 / (1.0 + math.exp(-z))

    def update(self, features: Dict[str, float], target: float) -> float:
        prediction = self.predict(features)
        error = prediction - target
        for name in self.feature_names:
            value = features.get(name, 0.0)
            grad = error * value + self.l2 * self.weights.get(name, 0.0)
            self.weights[name] = float(min(self.high, max(self.low, self.weights.get(name, 0.0) - self.learning_rate * grad)))
        return prediction


class ThompsonSelector:
    """Discrete Thompson Sampling helper for adaptively choosing options."""

    def __init__(self, option_keys: Sequence[str]) -> None:
        self.params: Dict[str, Dict[str, float]] = {
            key: {"alpha": 1.0, "beta": 1.0} for key in option_keys
        }

    def draw(self, option: str) -> float:
        params = self.params.setdefault(option, {"alpha": 1.0, "beta": 1.0})
        return random.betavariate(params["alpha"], params["beta"])

    def sample(self) -> str:
        return max(
            self.params,
            key=lambda key: self.draw(key),
        )

    def update(self, option: str, reward: float) -> None:
        if option not in self.params:
            self.params[option] = {"alpha": 1.0, "beta": 1.0}
        # Reward expected in [0, 1]
        self.params[option]["alpha"] += max(0.0, min(1.0, reward))
        self.params[option]["beta"] += max(0.0, min(1.0, 1.0 - reward))


class ReasoningSystem:
    """Generates structured reasoning episodes with traceable hypotheses and tests."""

    def __init__(self, architecture, memory_system=None, perception_system=None):
        self.arch = architecture
        self.memory = memory_system
        self.perception = perception_system

        self._feature_names = [
            "bias",
            "prompt_len",
            "contains_question",
            "contains_test",
            "strategy_match",
            "hypothesis_prior",
        ]
        self.strategy_models: Dict[str, OnlineLinear] = {
            strategy: OnlineLinear(self._feature_names)
            for strategy in ("abduction", "deduction", "analogy")
        }

        self._test_templates = self._build_test_templates()
        self.test_bandits: Dict[str, ThompsonSelector] = {}
        for strategy in ("abduction", "deduction", "analogy"):
            templates = list(self._test_templates.get("common", [])) + list(
                self._test_templates.get(strategy, [])
            )
            option_keys = [self._test_option_key(strategy, t["key"]) for t in templates]
            self.test_bandits[strategy] = ThompsonSelector(option_keys)

        smoothing_keys = ["beta_0.2", "beta_0.4", "beta_0.6", "beta_0.8"]
        self.preference_smoother = ThompsonSelector(smoothing_keys)
        self._current_smoothing_key: Optional[str] = None

        self.reasoning_history: Dict[str, Any] = {
            "recent_inferences": deque(maxlen=200),
            "learning_trajectory": [],
            "errors": deque(maxlen=200),
            "stats": {
                "n_episodes": 0,
                "avg_confidence": 0.50,
                "strategy_preferences": {
                    "abduction": 0.33,
                    "deduction": 0.34,
                    "analogy": 0.33,
                },
            },
            "auto_intentions": deque(maxlen=120),
        }

    # ------------------- API publique -------------------
    def on_auto_intention_promoted(
        self,
        event: Mapping[str, Any],
        evaluation: Optional[Mapping[str, Any]] = None,
        self_assessment: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not isinstance(event, Mapping):
            return
        record = {
            "ts": time.time(),
            "action_type": event.get("action_type"),
            "score": (evaluation or {}).get("score"),
            "keywords": list(event.get("keywords", [])),
        }
        history: deque = self.reasoning_history.setdefault("auto_intentions", deque(maxlen=120))
        history.append(record)
        prefs = self.reasoning_history.get("stats", {}).get("strategy_preferences", {})
        if isinstance(prefs, dict):
            keywords = {str(k).lower() for k in record.get("keywords", []) if k}
            if keywords.intersection({"relation", "relationship", "empathy", "social"}):
                prefs["analogy"] = min(1.0, prefs.get("analogy", 0.33) + 0.05)
            if keywords.intersection({"plan", "strategy", "predict"}):
                prefs["deduction"] = min(1.0, prefs.get("deduction", 0.34) + 0.05)
            if keywords.intersection({"hypothesis", "uncertain", "explore"}):
                prefs["abduction"] = min(1.0, prefs.get("abduction", 0.33) + 0.05)
            total = sum(max(0.0, v) for v in prefs.values()) or 1.0
            for key in list(prefs.keys()):
                prefs[key] = max(0.0, min(1.0, prefs.get(key, 0.0) / total))
        if self_assessment and isinstance(self_assessment, Mapping):
            checkpoints = self_assessment.get("checkpoints")
            if isinstance(checkpoints, list) and checkpoints:
                self.reasoning_history.setdefault("learning_trajectory", []).append(
                    {
                        "ts": time.time(),
                        "action_type": event.get("action_type"),
                        "targets": checkpoints,
                    }
                )

    def reason_about(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Réalise un raisonnement structuré sur *prompt* et retourne un épisode détaillé."""
        t0 = time.time()

        smoothing_value = self._select_smoothing_factor()
        strategy = self._pick_strategy(prompt)
        hypos = self._make_hypotheses(prompt, strategy=strategy)
        scores, features_by_idx = self._score_hypotheses(prompt, hypos, strategy)
        chosen_idx = max(range(len(scores)), key=lambda i: scores[i]) if scores else 0

        tests, chosen_test_keys = self._propose_tests(prompt, strategy=strategy)
        note = self._simulate_micro_inference(prompt, strategy, hypos[chosen_idx])
        evidence = Evidence(notes=note, confidence=min(0.55 + 0.15 * scores[chosen_idx], 0.95))

        reasoning_time = max(0.05, time.time() - t0)
        complexity = self._estimate_complexity(prompt, strategy, hypos)
        final_conf = float(min(0.9, 0.45 + 0.4 * scores[chosen_idx] + 0.10 * (1.0 - complexity)))

        deliberation = self._deliberate_actions(
            prompt,
            strategy,
            hypos,
            scores,
            context or {},
            complexity=complexity,
        )

        base_summary = self._make_readable_summary(
            strategy, hypos, chosen_idx, tests, evidence, final_conf
        )
        metadata: Dict[str, Any] = {}
        if deliberation:
            metadata["deliberation"] = deliberation
        metadata["heuristic_summary"] = base_summary

        ep = episode_record(
            user_msg=prompt,
            hypotheses=hypos,
            chosen_index=chosen_idx,
            tests=tests,
            evidence=evidence,
            result_text=evidence.notes,
            final_confidence=final_conf,
            metadata=metadata or None,
        )
        ep["reasoning_time"] = reasoning_time
        ep["complexity"] = complexity
        ep["strategy"] = strategy
        ep["hypothesis_features"] = features_by_idx
        ep["smoothing_value"] = smoothing_value
        ep["test_keys"] = chosen_test_keys
        ep["heuristic_summary"] = base_summary
        if deliberation:
            ep["deliberation"] = deliberation

        chosen_hypothesis_text = hypos[chosen_idx].content if hypos else ""
        tests_display = [t.description for t in tests]
        prochain_test = tests_display[0] if tests_display else "-"
        appris_entries = [
            f"Stratégie={strategy}, complexité≈{complexity:.2f}",
            "Toujours relier hypothèse→test→évidence (traçabilité).",
        ]

        llm_summary: Optional[str] = None
        llm_tests_display: Optional[List[str]] = None
        llm_learning: List[str] = []
        llm_notes: Optional[str] = None
        llm_actions: Optional[List[Dict[str, Any]]] = None

        def _coerce_float(value: Any) -> Optional[float]:
            try:
                if value is None:
                    return None
                return float(value)
            except (TypeError, ValueError):
                return None

        def _coerce_int(value: Any) -> Optional[int]:
            try:
                if value is None:
                    return None
                return int(value)
            except (TypeError, ValueError):
                return None

        heuristics_payload = {
            "strategy": strategy,
            "scores": [float(s) for s in scores],
            "chosen_index": int(chosen_idx),
            "hypotheses": [h.to_dict() for h in hypos],
            "tests": [t.to_dict() for t in tests],
            "evidence": evidence.to_dict(),
            "summary": base_summary,
            "complexity": complexity,
            "smoothing_value": smoothing_value,
            "features": [
                {
                    "index": int(idx),
                    "values": {name: float(val) for name, val in feats.items()},
                }
                for idx, feats in features_by_idx.items()
            ],
            "deliberation": deliberation,
            "reasoning_time": reasoning_time,
        }

        llm_response = try_call_llm_dict(
            "reasoning_episode",
            input_payload={
                "prompt": prompt,
                "context": self._prepare_llm_context(context or {}),
                "heuristics": heuristics_payload,
            },
            logger=logger,
        )

        if isinstance(llm_response, Mapping):
            llm_guidance = ep.setdefault("llm_guidance", {})
            llm_guidance["raw"] = llm_response
            summary_text = str(llm_response.get("summary") or "").strip()
            if summary_text:
                llm_summary = summary_text

            llm_notes_value = str(llm_response.get("notes") or "").strip()
            if llm_notes_value:
                llm_notes = llm_notes_value
                llm_guidance["notes"] = llm_notes

            confidence_value = _coerce_float(llm_response.get("confidence"))
            hypothesis_payload = llm_response.get("hypothesis")
            if isinstance(hypothesis_payload, Mapping):
                label_value = str(
                    hypothesis_payload.get("label")
                    or hypothesis_payload.get("name")
                    or ""
                ).strip()
                if label_value:
                    chosen_hypothesis_text = label_value
                    llm_guidance.setdefault("hypothesis", {})["label"] = label_value
                rationale = str(hypothesis_payload.get("rationale") or "").strip()
                if rationale:
                    llm_guidance.setdefault("hypothesis", {})["rationale"] = rationale
                if confidence_value is None:
                    confidence_value = _coerce_float(hypothesis_payload.get("confidence"))

            if confidence_value is not None:
                final_conf = float(max(0.0, min(1.0, confidence_value)))
                evidence.confidence = final_conf
                ep["evidence"]["confidence"] = final_conf
                ep["final_confidence"] = final_conf

            tests_payload = llm_response.get("tests")
            if isinstance(tests_payload, Sequence):
                llm_tests_display = []
                llm_tests_details: List[Dict[str, Any]] = []
                for item in tests_payload:
                    if not isinstance(item, Mapping):
                        continue
                    description = str(item.get("description") or item.get("label") or "").strip()
                    if not description:
                        continue
                    goal = str(item.get("goal") or item.get("objectif") or "").strip()
                    priority_val = _coerce_int(item.get("priority"))
                    detail: Dict[str, Any] = {"description": description}
                    if goal:
                        detail["goal"] = goal
                    if priority_val is not None:
                        detail["priority"] = priority_val
                    expected_gain = _coerce_float(item.get("expected_gain"))
                    if expected_gain is not None:
                        detail["expected_gain"] = float(max(0.0, min(1.0, expected_gain)))
                    llm_tests_details.append(detail)

                    parts: List[str] = []
                    if priority_val is not None:
                        parts.append(f"[p{max(1, min(9, priority_val))}]")
                    parts.append(description)
                    if goal:
                        parts.append(f"objectif: {goal}")
                    llm_tests_display.append(" ".join(parts))

                if llm_tests_details:
                    llm_guidance["tests"] = llm_tests_details

            learning_payload = llm_response.get("learning")
            if isinstance(learning_payload, Sequence):
                llm_learning = [
                    str(item).strip()
                    for item in learning_payload
                    if isinstance(item, (str, bytes)) and str(item).strip()
                ]
                if llm_learning:
                    llm_guidance["learning"] = llm_learning

            actions_payload = llm_response.get("actions")
            if isinstance(actions_payload, Sequence):
                parsed_actions: List[Dict[str, Any]] = []
                for item in actions_payload:
                    if not isinstance(item, Mapping):
                        continue
                    label = str(item.get("label") or item.get("action") or "").strip()
                    if not label:
                        continue
                    action_entry: Dict[str, Any] = {"label": label}
                    utility_val = _coerce_float(item.get("utility"))
                    if utility_val is not None:
                        action_entry["utility"] = float(max(0.0, min(1.0, utility_val)))
                    notes_val = str(item.get("notes") or item.get("why") or "").strip()
                    if notes_val:
                        action_entry["notes"] = notes_val
                    parsed_actions.append(action_entry)
                if parsed_actions:
                    llm_actions = parsed_actions
                    llm_guidance["actions"] = parsed_actions

        if llm_tests_display:
            tests_display = llm_tests_display
            prochain_test = llm_tests_display[0]

        if llm_learning:
            appris_entries.extend(llm_learning)

        if llm_notes:
            appris_entries.append(f"Note LLM: {llm_notes}")

        if llm_actions:
            if deliberation and isinstance(deliberation, Mapping):
                deliberation = dict(deliberation)
            else:
                deliberation = {}
            deliberation["llm_actions"] = llm_actions
            ep["deliberation"] = deliberation

        summary = llm_summary or base_summary
        if deliberation and deliberation.get("chosen"):
            best = deliberation["chosen"]
            summary += (
                f" Choix opératoire privilégié: {best.get('label')} (utilité≈{best.get('utility', 0.0):.2f})."
            )

        self._push_episode(ep)

        self._learn(
            final_conf,
            strategy,
            features_by_idx.get(chosen_idx, {}),
            smoothing_value,
            chosen_test_keys,
        )
        if deliberation:
            self._record_deliberation(prompt, deliberation, final_conf, complexity)

        try:
            if hasattr(self.arch, "logger") and self.arch.logger:
                self.arch.logger.write("reasoning.episode", episode=ep)
        except Exception:
            pass

        result: Dict[str, Any] = {
            "summary": summary,
            "chosen_hypothesis": chosen_hypothesis_text,
            "tests": tests_display,
            "final_confidence": final_conf,
            "appris": appris_entries,
            "prochain_test": prochain_test,
            "episode": ep,
        }
        if llm_notes:
            result["notes"] = llm_notes
        if deliberation:
            result["deliberation"] = deliberation
            cooldown = deliberation.get("rules", {}).get("ask_user_cooldown", 0)
            if cooldown:
                result.setdefault("appris", []).append(
                    f"Respecter un délai d'au moins {cooldown} interactions avant de redemander à l'utilisateur."
                )
        return result

    def get_reasoning_stats(self) -> Dict[str, Any]:
        """Résumé statistique utilisé par la métacognition."""
        stats = self.reasoning_history["stats"]
        recents = list(self.reasoning_history["recent_inferences"])
        if recents:
            stats["avg_confidence"] = sum(x.get("final_confidence", 0.5) for x in recents) / len(recents)
        return {
            "episodes": stats["n_episodes"],
            "average_confidence": float(stats.get("avg_confidence", 0.5)),
            "strategy_preferences": dict(stats.get("strategy_preferences", {})),
        }

    # ------------------- internes -------------------
    def _pick_strategy(self, prompt: str) -> str:
        lowered = prompt.lower()
        if any(keyword in lowered for keyword in ["pourquoi", "why", "cause"]):
            return "abduction"
        if any(keyword in lowered for keyword in ["comment", "plan", "steps", "étapes"]):
            return "deduction"
        return "analogy"

    def _make_hypotheses(self, prompt: str, strategy: str) -> List[Hypothesis]:
        base = [
            Hypothesis(content="tu veux une explication avec étapes et tests", prior=0.55),
            Hypothesis(content="tu veux que j'auto-documente ce que j'apprends", prior=0.50),
            Hypothesis(content="tu veux un patch/exécution immédiate", prior=0.45),
        ]
        if strategy == "analogy":
            base.append(Hypothesis(content="tu veux comparer avec un cas passé", prior=0.48))
        return base

    def _score_hypotheses(
        self, prompt: str, hypos: List[Hypothesis], strategy: str
    ) -> Tuple[List[float], Dict[int, Dict[str, float]]]:
        lowered = prompt.lower()
        scores: List[float] = []
        features_by_idx: Dict[int, Dict[str, float]] = {}
        for idx, hypo in enumerate(hypos):
            features = self._extract_features(lowered, hypo, strategy)
            model = self.strategy_models[strategy]
            learned_score = model.predict(features)
            # Blend learned score with prior for stability
            score = 0.55 * learned_score + 0.45 * min(1.0, max(0.0, hypo.prior))
            # Encourage hypotheses aligned with explicit cues
            if "test" in lowered and "test" in hypo.content:
                score += 0.05
            features_by_idx[idx] = features
            scores.append(min(1.0, score))
        return scores, features_by_idx

    def _propose_tests(self, prompt: str, strategy: str) -> Tuple[List[Test], List[str]]:
        lowered = prompt.lower()
        templates = list(self._test_templates.get("common", [])) + list(self._test_templates.get(strategy, []))
        bandit_key = strategy
        if bandit_key not in self.test_bandits:
            option_keys = [self._test_option_key(strategy, t["key"]) for t in templates]
            self.test_bandits[bandit_key] = ThompsonSelector(option_keys)
        bandit = self.test_bandits[bandit_key]

        scored_templates = []
        for template in templates:
            option_key = self._test_option_key(strategy, template["key"])
            posterior_sample = bandit.draw(option_key)
            keyword_bonus = 0.05 if any(word in lowered for word in template.get("keywords", [])) else 0.0
            score = posterior_sample + keyword_bonus + template.get("base_gain", 0.0)
            scored_templates.append((score, template, option_key))

        scored_templates.sort(key=lambda x: x[0], reverse=True)
        selected = scored_templates[:3]
        tests: List[Test] = []
        chosen_keys: List[str] = []
        for _, template, option_key in selected:
            tests.append(
                Test(
                    description=template["description"],
                    cost_est=template["cost_est"],
                    expected_information_gain=min(1.0, max(0.0, template.get("expected_information_gain", 0.5))),
                )
            )
            chosen_keys.append(option_key)
        return tests, chosen_keys

    def _simulate_micro_inference(self, prompt: str, strategy: str, hypothesis: Hypothesis) -> str:
        if strategy == "abduction":
            return f"Hypothèse choisie: {hypothesis.content}. Cause probable: manque de contexte explicite."
        if strategy == "deduction":
            return f"Plan issu de l'hypothèse '{hypothesis.content}': découper la tâche, tester chaque étape et consigner."
        return f"En comparant avec des cas passés, '{hypothesis.content}' semble le plus prometteur."

    def _estimate_complexity(self, prompt: str, strategy: str, hypos: List[Hypothesis]) -> float:
        length_factor = min(1.0, len(prompt) / 500.0)
        strategy_factor = 0.6 if strategy == "deduction" else 0.5
        hypo_factor = min(1.0, 0.2 * len(hypos))
        return float(min(1.0, 0.3 + length_factor * 0.4 + strategy_factor * 0.2 + hypo_factor * 0.2))

    def _prepare_llm_context(self, payload: Mapping[str, Any] | None) -> Dict[str, Any]:
        if not isinstance(payload, Mapping):
            return {}

        def _simplify(value: Any, depth: int = 0) -> Any:
            if depth >= 2:
                if isinstance(value, (str, int, float, bool)) or value is None:
                    return value
                return str(value)
            if isinstance(value, Mapping):
                simplified: Dict[str, Any] = {}
                for idx, (key, item) in enumerate(value.items()):
                    if idx >= 6:
                        remaining = 0
                        try:
                            remaining = max(0, len(value) - 6)
                        except Exception:
                            remaining = 0
                        simplified["..."] = f"+{remaining} entrées" if remaining else "…"
                        break
                    simplified[str(key)] = _simplify(item, depth + 1)
                return simplified
            if isinstance(value, (list, tuple, set)):
                items = list(value)
                limited = items[:6]
                simplified_list = [_simplify(item, depth + 1) for item in limited]
                if len(items) > 6:
                    simplified_list.append(f"... +{len(items) - 6} éléments")
                return simplified_list
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            return str(value)

        simplified_context: Dict[str, Any] = {}
        for idx, (key, value) in enumerate(payload.items()):
            if idx >= 8:
                remaining = 0
                try:
                    remaining = max(0, len(payload) - 8)
                except Exception:
                    remaining = 0
                simplified_context["..."] = f"+{remaining} clés" if remaining else "…"
                break
            simplified_context[str(key)] = _simplify(value, 0)
        return simplified_context

    def _make_readable_summary(
        self,
        strategy: str,
        hypos: List[Hypothesis],
        chosen_idx: int,
        tests: List[Test],
        evidence: Evidence,
        final_conf: float,
    ) -> str:
        hypo_text = hypos[chosen_idx].content if hypos else "hypothèse principale"
        test_text = tests[0].description if tests else "observer le prochain signal"
        return (
            f"Stratégie {strategy}: retenir '{hypo_text}'. "
            f"Test prioritaire: {test_text}. Évidence: {evidence.notes}. "
            f"Confiance finale≈{final_conf:.2f}."
        )

    def _push_episode(self, episode: Dict[str, Any]) -> None:
        history = self.reasoning_history
        history["recent_inferences"].append(
            {
                "final_confidence": episode.get("final_confidence", 0.5),
                "strategy": episode.get("strategy"),
                "ts": time.time(),
            }
        )
        history["stats"]["n_episodes"] += 1

    def _learn(
        self,
        confidence: float,
        strategy: str,
        features: Dict[str, float],
        smoothing_value: float,
        test_keys: List[str],
    ) -> None:
        self.reasoning_history["learning_trajectory"].append({"ts": time.time(), "confidence": confidence})
        stats = self.reasoning_history["stats"]
        prefs = stats.setdefault("strategy_preferences", {})
        for strat in self.strategy_models.keys():
            prefs.setdefault(strat, 1.0 / len(self.strategy_models))

        baseline = stats.get("avg_confidence", 0.5)

        # Update strategy preference with adaptive smoothing
        old_pref = prefs.get(strategy, baseline)
        prefs[strategy] = (1.0 - smoothing_value) * old_pref + smoothing_value * confidence
        for strat in prefs:
            if strat != strategy:
                prefs[strat] *= 1.0 - 0.05 * smoothing_value
        self._normalize_preferences(prefs)

        stats["avg_confidence"] = (1.0 - smoothing_value) * baseline + smoothing_value * confidence

        if features:
            model = self.strategy_models[strategy]
            model.update(features, confidence)

        smoothing_key = self._current_smoothing_key
        if smoothing_key:
            reward = 1.0 if confidence >= baseline else 0.0
            self.preference_smoother.update(smoothing_key, reward)
        self._current_smoothing_key = None

        reward_conf = max(0.0, min(1.0, confidence))
        bandit = self.test_bandits.get(strategy)
        if bandit:
            for key in test_keys:
                bandit.update(key, reward_conf)

    def _self_preferences(self) -> Dict[str, Any]:
        self_model = getattr(self.arch, "self_model", None)
        preferences = {
            "values": [],
            "autonomy_bias": 0.5,
            "caution_bias": 0.5,
            "verification_bias": 0.5,
            "cognitive_load": 0.0,
            "uncertainty": 0.0,
            "recent_decisions": {},
            "rules": {},
            "feedback_lexicon": {},
            "storage_path": None,
            "last_refreshed_ts": time.time(),
        }
        if not self_model:
            return preferences
        ensure_paths = getattr(self_model, "ensure_identity_paths", None)
        if callable(ensure_paths):
            try:
                ensure_paths()
            except Exception:
                pass
        identity = getattr(self_model, "identity", {}) or {}
        values = identity.get("preferences", {}).get("values", [])
        normalized_values = [str(v).lower() for v in values if isinstance(v, str)]
        preferences["values"] = normalized_values

        rules = identity.get("choices", {}).get("rules", {})
        if isinstance(rules, dict):
            preferences["rules"] = rules

        feedback_info = identity.get("feedback", {})
        lexicon = feedback_info.get("lexicon") if isinstance(feedback_info, dict) else None
        if isinstance(lexicon, dict):
            parsed_lexicon: Dict[str, float] = {}
            for token, value in lexicon.items():
                if not token:
                    continue
                try:
                    weight = float(value)
                except (TypeError, ValueError):
                    continue
                normalized_token = str(token).lower().strip()
                if not normalized_token:
                    continue
                parsed_lexicon[normalized_token] = float(max(-1.0, min(1.0, weight)))
            preferences["feedback_lexicon"] = parsed_lexicon

        storage_path = getattr(self_model, "path", None)
        if isinstance(storage_path, str):
            preferences["storage_path"] = storage_path

        stats = (
            identity.get("choices", {})
            .get("policies", {})
            .get("stats", {})
        )
        success = float(stats.get("success", 0))
        fail = float(stats.get("fail", 0))
        total = success + fail
        success_rate = success / total if total else 0.5
        preferences["autonomy_bias"] = 0.4 + 0.4 * success_rate
        preferences["caution_bias"] = 0.4 + 0.4 * (1.0 - success_rate)
        preferences["verification_bias"] = 0.5

        if "curiosity" in normalized_values:
            preferences["autonomy_bias"] += 0.1
        if "care" in normalized_values:
            preferences["caution_bias"] += 0.1
        if "precision" in normalized_values:
            preferences["verification_bias"] += 0.1

        state = identity.get("state", {})
        cognition = state.get("cognition", {})
        load = cognition.get("load", 0.0)
        uncertainty = cognition.get("uncertainty", 0.0)
        try:
            preferences["cognitive_load"] = float(max(0.0, min(1.0, load)))
        except Exception:
            preferences["cognitive_load"] = 0.0
        try:
            preferences["uncertainty"] = float(max(0.0, min(1.0, uncertainty)))
        except Exception:
            preferences["uncertainty"] = 0.0

        preferences["recent_decisions"] = self._recent_decision_counts(identity)

        for key in ("autonomy_bias", "caution_bias", "verification_bias"):
            try:
                preferences[key] = float(min(0.95, max(0.05, preferences[key])))
            except Exception:
                preferences[key] = 0.5

        return preferences

    def _recent_decision_counts(self, identity: Dict[str, Any], window: float = 3600.0) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        try:
            recent = identity.get("choices", {}).get("recent", [])
        except Exception:
            recent = []
        now = time.time()
        for entry in recent or []:
            if not isinstance(entry, dict):
                continue
            action = entry.get("action")
            if not action:
                continue
            ts = entry.get("ts", now)
            try:
                ts_f = float(ts)
            except (TypeError, ValueError):
                ts_f = now
            if now - ts_f > window:
                continue
            counts[action] = counts.get(action, 0) + 1
        return counts

    def _user_preference(self, label: str) -> float:
        user_model = getattr(self.arch, "user_model", None)
        if not user_model:
            return 0.5
        prior_fn = getattr(user_model, "prior", None)
        if not callable(prior_fn):
            return 0.5
        try:
            return float(max(0.0, min(1.0, prior_fn(label))))
        except Exception:
            return 0.5

    def _analyze_feedback_text(self, text: str, preferences: Dict[str, Any]) -> Optional[Dict[str, float]]:
        if not text:
            return None
        normalized = unicodedata.normalize("NFKD", str(text))
        lowered = normalized.lower()
        tokens = self._tokenize_feedback_text(lowered)
        if not tokens:
            return None

        seed_lexicon = self._seed_feedback_lexicon()
        learned_lexicon = preferences.get("feedback_lexicon", {}) or {}
        combined_lexicon: Dict[str, float] = {**seed_lexicon, **learned_lexicon}

        weighted_sum = 0.0
        magnitude = 0.0
        unknown_tokens = 0
        for token in tokens:
            weight = combined_lexicon.get(token)
            if weight is None:
                unknown_tokens += 1
                continue
            weighted_sum += weight
            magnitude += abs(weight)

        if magnitude == 0.0 and weighted_sum == 0.0:
            heuristic_score = self._heuristic_feedback_score(tokens)
            if heuristic_score == 0.0:
                return None
            weighted_sum = heuristic_score
            magnitude = abs(heuristic_score)

        magnitude = max(1.0, magnitude + 0.15 * unknown_tokens)
        polarity = max(-1.0, min(1.0, weighted_sum / magnitude))

        exclamations = lowered.count("!")
        questions = lowered.count("?")
        emphasis = min(1.0, 0.1 * min(4, exclamations) + 0.05 * min(4, questions))
        intensity_base = min(1.0, abs(weighted_sum) / max(1.0, len(tokens)))
        intensity = max(0.05, min(1.0, intensity_base + emphasis))

        if "care" in preferences.get("values", []):
            intensity = min(1.0, intensity * 1.1)
        if "curiosity" in preferences.get("values", []):
            polarity *= 0.95

        return {
            "polarity": float(polarity),
            "intensity": float(intensity),
        }

    def _collect_feedback_signals(
        self,
        prompt: str,
        context: Dict[str, Any],
        preferences: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        signals: List[Dict[str, Any]] = []

        def _add_signal(
            target: Optional[str],
            polarity: float,
            intensity: float,
            source: str,
            payload: Optional[Dict[str, Any]] = None,
            confidence: float = 1.0,
        ) -> None:
            if intensity <= 0.0 or polarity == 0.0:
                return
            signals.append(
                {
                    "target": target or "general",
                    "polarity": float(max(-1.0, min(1.0, polarity))),
                    "intensity": float(max(0.0, min(1.0, intensity))),
                    "confidence": float(max(0.0, min(1.0, confidence))),
                    "source": source,
                    "payload": payload or {},
                }
            )

        text_feedback = self._analyze_feedback_text(prompt, preferences)
        if text_feedback:
            _add_signal(
                target="ask_user",
                polarity=text_feedback["polarity"],
                intensity=text_feedback["intensity"],
                source="prompt",
                payload={"text": prompt[:120]},
            )

        ctx_feedback_candidates: List[Any] = []
        for key in ("feedback_events", "user_feedback", "feedback"):
            value = context.get(key) if context else None
            if not value:
                continue
            if isinstance(value, (list, tuple, set)):
                ctx_feedback_candidates.extend(list(value))
            else:
                ctx_feedback_candidates.append(value)

        for entry in ctx_feedback_candidates:
            if isinstance(entry, dict):
                target = entry.get("target") or entry.get("action")
                polarity = entry.get("polarity") or entry.get("valence")
                message = entry.get("message") or entry.get("text")
                confidence_raw = entry.get("confidence", 1.0)
                try:
                    confidence = float(confidence_raw)
                except (TypeError, ValueError):
                    confidence = 1.0
                intensity = entry.get("intensity") or entry.get("weight")
                if polarity is None and message:
                    text_signal = self._analyze_feedback_text(str(message), preferences)
                    if text_signal:
                        polarity = text_signal["polarity"]
                        intensity = intensity or text_signal["intensity"]
                try:
                    polarity_f = float(polarity)
                except (TypeError, ValueError):
                    polarity_f = 0.0
                try:
                    intensity_f = float(intensity) if intensity is not None else 0.0
                except (TypeError, ValueError):
                    intensity_f = 0.0
                if polarity_f != 0.0 and intensity_f <= 0.0:
                    intensity_f = 0.4
                if polarity_f:
                    _add_signal(
                        target=str(target or "general"),
                        polarity=polarity_f,
                        intensity=intensity_f or 0.4,
                        source=entry.get("source") or "context",
                        payload={
                            k: v
                            for k, v in entry.items()
                            if k not in {"target", "action", "polarity", "valence"}
                        },
                        confidence=confidence,
                    )
            elif isinstance(entry, str):
                feedback = self._analyze_feedback_text(entry, preferences)
                if feedback:
                    _add_signal(
                        target="general",
                        polarity=feedback["polarity"],
                        intensity=feedback["intensity"],
                        source="context:text",
                        payload={"text": entry[:120]},
                    )

        rules = preferences.get("rules", {}) if isinstance(preferences, dict) else {}
        if isinstance(rules, dict):
            for action, rule in rules.items():
                if not isinstance(rule, dict):
                    continue
                cooldown = rule.get("cooldown") or rule.get("ask_user_cooldown")
                try:
                    cooldown_val = float(cooldown)
                except (TypeError, ValueError):
                    cooldown_val = 0.0
                if cooldown_val > 0:
                    _add_signal(
                        target=str(action),
                        polarity=-1.0,
                        intensity=min(1.0, cooldown_val / 5.0),
                        source="memory:cooldown",
                        payload={"cooldown": cooldown_val},
                        confidence=0.8,
                    )
                feedback_info = rule.get("feedback")
                if isinstance(feedback_info, dict):
                    adjustment = feedback_info.get("adjustment")
                    try:
                        adj_val = float(adjustment)
                    except (TypeError, ValueError):
                        adj_val = 0.0
                    if adj_val:
                        polarity = 1.0 if adj_val > 0 else -1.0
                        _add_signal(
                            target=str(action),
                            polarity=polarity,
                            intensity=min(1.0, abs(adj_val)),
                            source="memory:feedback",
                            payload={"adjustment": adj_val},
                            confidence=float(feedback_info.get("confidence", 0.7)),
                        )
                    stored_signals = feedback_info.get("signals")
                    if isinstance(stored_signals, (list, tuple)):
                        for sig in stored_signals:
                            if not isinstance(sig, dict):
                                continue
                            try:
                                pol = float(sig.get("polarity", 0.0))
                            except (TypeError, ValueError):
                                pol = 0.0
                            try:
                                inten = float(sig.get("intensity", 0.0))
                            except (TypeError, ValueError):
                                inten = 0.0
                            conf = float(sig.get("confidence", feedback_info.get("confidence", 0.7)))
                            if pol == 0.0 or inten <= 0.0:
                                continue
                            _add_signal(
                                target=str(sig.get("target") or action),
                                polarity=pol,
                                intensity=inten,
                                source=str(sig.get("source") or "memory:signal"),
                                payload={
                                    k: v
                                    for k, v in sig.items()
                                    if k not in {"target", "polarity", "intensity", "confidence", "source"}
                                },
                                confidence=conf,
                            )

        return signals

    def _apply_feedback_signals(
        self,
        base_prior: float,
        signals: Sequence[Dict[str, Any]],
        *,
        target: str,
        preferences: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        multiplier = 1.0
        if "care" in preferences.get("values", []):
            multiplier += 0.1
        if "autonomy" in preferences.get("values", []):
            multiplier += 0.05
        delta = 0.0
        applied: List[Dict[str, Any]] = []
        for signal in signals:
            tgt = signal.get("target") or "general"
            if tgt not in {target, "general", "all"}:
                continue
            polarity = float(max(-1.0, min(1.0, signal.get("polarity", 0.0))))
            intensity = float(max(0.0, min(1.0, signal.get("intensity", 0.0))))
            confidence = float(max(0.0, min(1.0, signal.get("confidence", 1.0))))
            weight = intensity * confidence * multiplier
            if weight <= 0.0:
                continue
            adjustment = polarity * weight * 0.4
            delta += adjustment
            applied.append(
                {
                    "target": target,
                    "source": signal.get("source"),
                    "polarity": polarity,
                    "intensity": intensity,
                    "confidence": confidence,
                    "delta": adjustment,
                    "payload": signal.get("payload", {}),
                }
            )
        new_prior = float(max(0.05, min(0.95, base_prior + delta)))
        return new_prior, {"delta": delta, "signals": applied}

    def _tokenize_feedback_text(self, text: str) -> List[str]:
        normalized = unicodedata.normalize("NFKD", str(text).lower())
        cleaned = "".join(
            ch if ch.isalnum() or ch in {" ", "-", "'"} else " " for ch in normalized
        )
        tokens = []
        for raw_token in cleaned.split():
            token = raw_token.strip("-' ")
            if len(token) < 3:
                continue
            tokens.append(token)
        return tokens

    def _seed_feedback_lexicon(self) -> Dict[str, float]:
        return {
            "merci": 0.4,
            "merciii": 0.4,
            "cool": 0.35,
            "parfait": 0.45,
            "bravo": 0.4,
            "top": 0.35,
            "nickel": 0.35,
            "super": 0.35,
            "utile": 0.3,
            "ok": 0.2,
            "compris": 0.25,
            "chiant": -0.7,
            "relou": -0.6,
            "agace": -0.6,
            "agacer": -0.6,
            "ennuyeux": -0.55,
            "fatigue": -0.5,
            "fatigant": -0.5,
            "lourd": -0.45,
            "lassant": -0.45,
            "trop": -0.3,
            "stop": -0.5,
            "arrete": -0.5,
            "arretez": -0.5,
            "jamais": -0.3,
            "plus": -0.25,
            "question": -0.1,
            "questions": -0.1,
            "autonome": 0.2,
            "autonomie": 0.2,
            "apprendre": -0.2,
            "seul": -0.2,
        }

    def _heuristic_feedback_score(self, tokens: Sequence[str]) -> float:
        negative_markers = {
            "pas",
            "jamais",
            "trop",
            "stop",
            "fatigue",
            "fatigant",
            "lourd",
            "lassant",
            "marre",
            "ralentit",
            "ralentir",
            "deborde",
            "debordee",
            "derange",
        }
        positive_markers = {
            "merci",
            "bravo",
            "cool",
            "parfait",
            "genial",
            "top",
            "nickel",
        }
        score = 0.0
        for token in tokens:
            if token in positive_markers:
                score += 0.25
            if token in negative_markers:
                score -= 0.25
        return score

    def _feedback_token_updates(
        self,
        prompt: str,
        signals: Sequence[Dict[str, Any]],
        adjustment: float,
    ) -> Dict[str, float]:
        updates: Dict[str, float] = {}

        def _accumulate(text: Optional[str], weight: float) -> None:
            if not text or not weight:
                return
            for token in self._tokenize_feedback_text(text):
                updates[token] = updates.get(token, 0.0) + weight

        for signal in signals or []:
            if not isinstance(signal, dict):
                continue
            try:
                polarity = float(signal.get("polarity", 0.0))
                intensity = float(signal.get("intensity", 0.0))
                confidence = float(signal.get("confidence", 1.0))
            except (TypeError, ValueError):
                continue
            weight = polarity * max(0.1, intensity) * max(0.2, confidence)
            if weight == 0.0:
                continue
            payload = signal.get("payload") if isinstance(signal.get("payload"), dict) else {}
            extracted_texts: List[str] = []
            for key in ("text", "message", "excerpt", "content"):
                value = payload.get(key)
                if isinstance(value, str):
                    extracted_texts.append(value)
            if not extracted_texts and signal.get("source") == "prompt":
                extracted_texts.append(prompt)
            for text_value in extracted_texts:
                _accumulate(text_value, weight)

        if adjustment and prompt:
            _accumulate(prompt, adjustment * 0.4)

        return updates

    def _build_feedback_lexicon_patch(
        self,
        self_model,
        updates: Dict[str, float],
    ) -> Optional[Dict[str, float]]:
        if not updates:
            return None
        identity = getattr(self_model, "identity", {}) or {}
        feedback_info = identity.get("feedback", {}) if isinstance(identity, dict) else {}
        current = feedback_info.get("lexicon") if isinstance(feedback_info, dict) else {}
        lexicon: Dict[str, float] = {}
        if isinstance(current, dict):
            for token, value in current.items():
                try:
                    lexicon[str(token).lower()] = float(value)
                except (TypeError, ValueError):
                    continue
        changed = False
        for token, delta in updates.items():
            if not token:
                continue
            prev = float(lexicon.get(token, 0.0))
            new_val = 0.85 * prev + delta
            if abs(new_val) < 0.05:
                if token in lexicon:
                    del lexicon[token]
                    changed = True
                continue
            new_val = max(-1.0, min(1.0, new_val))
            if lexicon.get(token) != new_val:
                lexicon[token] = new_val
                changed = True
        if not changed:
            return None
        if len(lexicon) > 150:
            top_items = sorted(lexicon.items(), key=lambda item: abs(item[1]), reverse=True)[:150]
            lexicon = {token: value for token, value in top_items}
        return lexicon

    def _deliberate_actions(
        self,
        prompt: str,
        strategy: str,
        hypos: List[Hypothesis],
        scores: List[float],
        context: Dict[str, Any],
        *,
        complexity: float,
    ) -> Dict[str, Any]:
        preferences = self._self_preferences()
        user_help_prior = self._user_preference("ask_user_help")
        feedback_signals = self._collect_feedback_signals(prompt, context, preferences)
        user_help_prior, feedback_effect = self._apply_feedback_signals(
            user_help_prior,
            feedback_signals,
            target="ask_user",
            preferences=preferences,
        )
        if not feedback_effect.get("signals"):
            feedback_effect["signals"] = []
        if "delta" not in feedback_effect:
            feedback_effect["delta"] = 0.0
        inbox_docs = context.get("inbox_docs", [])
        inbox_count = 0
        if isinstance(inbox_docs, (list, tuple, set)):
            inbox_count = len(inbox_docs)
        elif isinstance(inbox_docs, dict):
            inbox_count = len(inbox_docs)
        else:
            try:
                inbox_count = int(inbox_docs) if inbox_docs else 0
            except Exception:
                inbox_count = 0

        top_score = max(scores) if scores else 0.5
        top_score = float(max(0.05, min(0.95, top_score)))
        memory_gain = min(0.9, top_score + 0.1 + 0.05 * min(math.log1p(max(inbox_count, 0)), 3.0))
        memory_cost = min(0.9, 0.25 + 0.05 * min(inbox_count, 8) + 0.15 * preferences["cognitive_load"])

        autonomous_cost = min(0.9, 0.3 + 0.2 * preferences["cognitive_load"])
        autonomous_gain = min(0.9, top_score + 0.05 * (1.0 - preferences["uncertainty"]))

        ask_user_recent = preferences["recent_decisions"].get("ask_user", 0)
        self_solved_recent = preferences["recent_decisions"].get("reason_alone", 0)
        ask_user_cost = 0.15 + 0.08 * ask_user_recent
        if user_help_prior < 0.5:
            ask_user_cost += 0.2 * (0.5 - user_help_prior)
        if self_solved_recent >= 3:
            user_help_prior = min(0.95, user_help_prior + 0.1)
            ask_user_cost = max(0.1, ask_user_cost - 0.05)

        question_manager_available = bool(getattr(self.arch, "question_manager", None))
        manager_cost = 0.28 + 0.1 * preferences["cognitive_load"]
        manager_gain = 0.65 if question_manager_available else 0.5

        cooldown_hint = 0
        rules = preferences.get("rules") if isinstance(preferences, dict) else {}
        ask_user_rule = rules.get("ask_user") if isinstance(rules, dict) else None
        rule_cooldown = 0.0
        if isinstance(ask_user_rule, dict):
            try:
                rule_cooldown = float(ask_user_rule.get("cooldown", 0.0))
            except (TypeError, ValueError):
                rule_cooldown = 0.0
        if rule_cooldown:
            cooldown_hint = max(cooldown_hint, int(math.ceil(rule_cooldown)))
        if user_help_prior < 0.4:
            cooldown_hint = max(cooldown_hint, 3)
        elif user_help_prior < 0.5:
            cooldown_hint = max(cooldown_hint, 2)
        cooldown_hint = max(cooldown_hint, ask_user_recent)
        negative_pressure = sum(
            -item.get("delta", 0.0)
            for item in feedback_effect.get("signals", [])
            if item.get("delta", 0.0) < 0
        )
        if negative_pressure > 0:
            cooldown_hint = max(cooldown_hint, min(5, int(math.ceil(negative_pressure * 8))))

        ask_user_cost_boost = sum(
            max(0.0, -item.get("delta", 0.0))
            for item in feedback_effect.get("signals", [])
            if item.get("delta", 0.0) < 0
        )
        if ask_user_cost_boost:
            ask_user_cost += min(0.25, ask_user_cost_boost)

        def _pref_weight(kind: str) -> float:
            base = 1.0
            if kind == "autonomy":
                base += preferences["autonomy_bias"] - 0.5
            elif kind == "analysis":
                base += preferences["verification_bias"] - 0.5
            elif kind == "interaction":
                base += preferences["caution_bias"] - 0.5
            return float(min(1.5, max(0.5, base)))

        reason_time = max(0.6, 2.0 + 4.0 * complexity + 2.0 * preferences["uncertainty"])
        if self_solved_recent >= 3:
            reason_time = max(0.5, reason_time - 0.6)
        scan_time = max(
            0.7,
            1.4 + 0.35 * min(inbox_count, 12) + 1.5 * preferences["cognitive_load"],
        )
        ask_user_time = max(0.3, 0.6 + 0.4 * cooldown_hint + 0.2 * ask_user_recent)
        ask_manager_time = max(0.5, 0.8 + 0.3 * preferences["cognitive_load"])

        action_times = {
            "reason_alone": reason_time,
            "scan_memory": scan_time,
            "ask_user": ask_user_time,
            "ask_manager": ask_manager_time,
        }

        def _cause_effect(option_key: str, weight: float, success: float, cost: float) -> str:
            if option_key == "reason_alone":
                return (
                    "Parce que l'autonomie est pondérée à "
                    f"{weight:.2f} et que {self_solved_recent} succès récents soutiennent cette voie,"
                    f" l'agent anticipe une utilité nette≈{max(0.0, weight * success - cost):.2f}."
                )
            if option_key == "scan_memory":
                return (
                    f"Comme {inbox_count} éléments restent à analyser et que la vérification pèse {weight:.2f},"
                    f" extraire des preuves devrait augmenter le gain attendu≈{(weight * success):.2f}."
                )
            if option_key == "ask_user":
                delta = feedback_effect.get("delta", 0.0)
                if delta < 0:
                    signal_phrase = "Les retours récents suggèrent de limiter les sollicitations"
                elif delta > 0:
                    signal_phrase = "Les retours positifs autorisent une sollicitation ciblée"
                else:
                    signal_phrase = "Les préférences actuelles autorisent une sollicitation mesurée"
                return (
                    f"{signal_phrase} (prior={user_help_prior:.2f}),"
                    f" d'où un coût≈{cost:.2f} modulé par le feedback et un éventuel délai."
                )
            if option_key == "ask_manager":
                availability = "disponible" if question_manager_available else "incertain"
                return (
                    f"Le gestionnaire de questions est {availability}, donc poids={weight:.2f} et gain≈{(weight * success):.2f}"
                    f" compensent un coût≈{cost:.2f}."
                )
            return ""

        options = [
            {
                "key": "reason_alone",
                "label": "Continuer le raisonnement interne",
                "success": autonomous_gain,
                "cost": autonomous_cost,
                "kind": "autonomy",
                "notes": "Préserve l'autonomie et capitalise sur les hypothèses actuelles.",
            },
            {
                "key": "scan_memory",
                "label": "Scanner les notes et l'inbox",
                "success": memory_gain,
                "cost": memory_cost,
                "kind": "analysis",
                "notes": "Explore les documents disponibles pour trouver des confirmations.",
            },
            {
                "key": "ask_user",
                "label": "Demander explicitement à l'utilisateur",
                "success": max(0.05, min(0.95, user_help_prior)),
                "cost": min(0.95, ask_user_cost),
                "kind": "interaction",
                "notes": "Sollicite l'utilisateur tout en respectant les signaux de feedback cumulés.",
            },
            {
                "key": "ask_manager",
                "label": "Consulter le question manager",
                "success": manager_gain,
                "cost": min(0.95, manager_cost),
                "kind": "interaction",
                "notes": "Escalader pour obtenir une réponse rapide si disponible.",
            },
        ]

        evaluated: List[Dict[str, Any]] = []
        timeline: List[Dict[str, Any]] = []
        for option in options:
            weight = _pref_weight(option["kind"])
            success = float(max(0.05, min(0.95, option["success"])))
            cost = float(max(0.05, min(0.95, option["cost"])))
            penalty = cost
            if option["key"] == "ask_user" and user_help_prior < 0.4:
                penalty += 0.15 * (0.4 - user_help_prior)
            expected_gain = weight * success
            utility = expected_gain - penalty
            note = option.get("notes", "")
            if option["key"] == "ask_user" and user_help_prior < 0.4:
                note += " Préférence négative détectée → risque de rejet."
            if option["key"] == "reason_alone" and self_solved_recent >= 3:
                note += " Succès récents en autonomie : confiance renforcée."
            if option["key"] == "scan_memory" and inbox_count:
                note += f" {inbox_count} éléments à analyser."
            time_est = round(action_times.get(option["key"], 1.0), 2)
            cause_effect = _cause_effect(option["key"], weight, success, cost)
            timeline.append(
                {
                    "action": option["key"],
                    "label": option["label"],
                    "estimated_minutes": time_est,
                    "cause_effect": cause_effect,
                }
            )
            evaluated.append(
                {
                    "key": option["key"],
                    "label": option["label"],
                    "success_probability": round(success, 3),
                    "effort_cost": round(cost, 3),
                    "preference_weight": round(weight, 3),
                    "expected_gain": round(expected_gain, 3),
                    "utility": round(utility, 3),
                    "notes": note.strip(),
                    "time_estimate_minutes": time_est,
                    "cause_effect": cause_effect,
                }
            )

        evaluated.sort(key=lambda item: item["utility"], reverse=True)
        chosen = evaluated[0] if evaluated else None

        cooldown = cooldown_hint

        explanation_parts: List[str] = []
        if chosen:
            explanation_parts.append(
                (
                    f"Comparaison coût/bénéfice: {chosen['label']} maximise l'utilité"
                    f" (gain≈{chosen['expected_gain']:.2f} - coût≈{chosen['effort_cost']:.2f})."
                )
            )
            if chosen["key"] == "ask_user":
                if feedback_effect.get("signals"):
                    negative_sources = sorted(
                        {
                            str(sig.get("source") or "feedback")
                            for sig in feedback_effect.get("signals", [])
                            if sig.get("delta", 0.0) < 0
                        }
                    )
                    positive_sources = sorted(
                        {
                            str(sig.get("source") or "feedback")
                            for sig in feedback_effect.get("signals", [])
                            if sig.get("delta", 0.0) > 0
                        }
                    )
                    if negative_sources:
                        explanation_parts.append(
                            "Signaux négatifs considérés (" + ", ".join(negative_sources) + ") → sollicitation limitée."
                        )
                    if positive_sources:
                        explanation_parts.append(
                            "Signaux positifs (" + ", ".join(positive_sources) + ") soutiennent la demande directe."
                        )
                elif user_help_prior < 0.4:
                    explanation_parts.append(
                        "Prior utilisateur défavorable : la sollicitation reste exceptionnelle."
                    )
            if chosen["key"] == "reason_alone" and preferences["uncertainty"] > 0.6:
                explanation_parts.append(
                    "La prudence reste élevée car l'incertitude cognitive dépasse 0.6."
                )
            for item in timeline:
                if item["action"] == chosen["key"] and item["cause_effect"]:
                    explanation_parts.append(item["cause_effect"])
                    explanation_parts.append(
                        f"Temps estimé pour cette option: {item['estimated_minutes']:.2f} minutes."
                    )
                    break
        if cooldown:
            explanation_parts.append(
                f"Imposer un délai de {cooldown} interactions avant une nouvelle sollicitation directe."
            )

        return {
            "options": evaluated,
            "chosen": chosen,
            "self_preferences": preferences,
            "user_feedback": {
                "ask_user_help_prior": round(user_help_prior, 3),
                "recent_user_requests": ask_user_recent,
                "adjustment": round(feedback_effect.get("delta", 0.0), 3),
                "signals": feedback_effect.get("signals", []),
            },
            "rules": {
                "ask_user_cooldown": cooldown,
            },
            "complexity": float(complexity),
            "explanation": " ".join(explanation_parts).strip(),
            "timeline": timeline,
        }

    def _record_deliberation(
        self,
        prompt: str,
        deliberation: Dict[str, Any],
        final_conf: float,
        complexity: float,
    ) -> None:
        self_model = getattr(self.arch, "self_model", None)
        if not self_model or not deliberation or not deliberation.get("chosen"):
            return
        chosen = deliberation["chosen"]
        expected_gain = float(chosen.get("expected_gain", 0.0))
        expected_gain = max(0.0, min(1.0, expected_gain))
        success_prob = float(chosen.get("success_probability", 0.5))
        success_prob = max(0.0, min(1.0, success_prob))
        expected_utility = float(chosen.get("utility", 0.0))
        expected_utility = max(0.0, min(1.0, expected_utility))
        context = {
            "complexity": float(complexity),
            "expected_gain": expected_gain,
            "risk": float(max(0.0, 1.0 - success_prob)),
            "ask_user_cooldown": float(deliberation.get("rules", {}).get("ask_user_cooldown", 0)),
        }
        try:
            if hasattr(self_model, "policy_decision_score"):
                self_model.policy_decision_score(context)
        except Exception:
            pass
        decision_ref = {
            "decision_id": f"reasoning::{int(time.time() * 1000)}",
            "topic": prompt[:120],
            "action": chosen.get("key"),
            "expected": expected_utility,
            "obtained": float(final_conf),
            "complexity": context["complexity"],
            "expected_gain": context["expected_gain"],
            "risk": context["risk"],
            "trace": deliberation,
        }
        if hasattr(self_model, "register_decision"):
            try:
                self_model.register_decision(decision_ref)
            except Exception:
                pass
        user_feedback_meta = deliberation.get("user_feedback") or {}
        feedback_payload: Optional[Dict[str, Any]] = None
        lexicon_updates: Dict[str, float] = {}
        if user_feedback_meta:
            signals_meta = user_feedback_meta.get("signals") or []
            try:
                adjustment_val = float(user_feedback_meta.get("adjustment", 0.0))
            except (TypeError, ValueError):
                adjustment_val = 0.0
            lexicon_updates = self._feedback_token_updates(
                prompt,
                signals_meta if isinstance(signals_meta, (list, tuple)) else [],
                adjustment_val,
            )
            filtered_signals: List[Dict[str, Any]] = []
            if isinstance(signals_meta, (list, tuple)):
                for sig in signals_meta[:6]:
                    if not isinstance(sig, dict):
                        continue
                    try:
                        pol = float(sig.get("polarity", 0.0))
                        inten = float(sig.get("intensity", 0.0))
                        conf = float(sig.get("confidence", 1.0))
                    except (TypeError, ValueError):
                        continue
                    payload = sig.get("payload") if isinstance(sig.get("payload"), dict) else None
                    filtered_signals.append(
                        {
                            "target": sig.get("target"),
                            "source": sig.get("source"),
                            "polarity": pol,
                            "intensity": inten,
                            "confidence": conf,
                            **({"payload": payload} if payload else {}),
                        }
                    )
            if filtered_signals or adjustment_val != 0.0:
                feedback_payload = {
                    "adjustment": adjustment_val,
                    "signals": filtered_signals,
                    "prior": float(user_feedback_meta.get("ask_user_help_prior", 0.5)),
                    "recent_requests": int(user_feedback_meta.get("recent_user_requests", 0)),
                    "confidence": 0.7,
                }
        rules = deliberation.get("rules", {}) if deliberation else {}
        lexicon_patch = self._build_feedback_lexicon_patch(self_model, lexicon_updates)
        patch_payload: Dict[str, Any] = {}
        if rules:
            rule_patch = {
                "cooldown": float(rules.get("ask_user_cooldown", 0)),
                "updated_at_ts": time.time(),
                "last_reasoning_topic": prompt[:80],
                "preferred_action": chosen.get("key"),
            }
            if feedback_payload:
                rule_patch["feedback"] = feedback_payload
            patch_payload.setdefault("choices", {}).setdefault("rules", {})[
                "ask_user"
            ] = rule_patch
        if lexicon_patch is not None:
            patch_payload.setdefault("feedback", {})["lexicon"] = lexicon_patch
        if patch_payload:
            set_patch = getattr(self_model, "set_identity_patch", None)
            if callable(set_patch):
                try:
                    set_patch(patch_payload)
                except Exception:
                    pass

    # ------------------- helpers adaptatifs -------------------
    def _select_smoothing_factor(self) -> float:
        key = self.preference_smoother.sample()
        self._current_smoothing_key = key
        try:
            return float(key.split("_")[1])
        except (IndexError, ValueError):
            return 0.4

    def _test_option_key(self, strategy: str, key: str) -> str:
        return f"{strategy}:{key}"

    def _build_test_templates(self) -> Dict[str, List[Dict[str, Any]]]:
        return {
            "common": [
                {
                    "key": "alt_hypotheses",
                    "description": "Formuler 2 hypothèses alternatives et demander validation",
                    "cost_est": 0.3,
                    "expected_information_gain": 0.55,
                    "base_gain": 0.25,
                    "keywords": ["hypothèse", "pourquoi"],
                },
                {
                    "key": "retrieve_examples",
                    "description": "Rechercher 2 exemples récents similaires",
                    "cost_est": 0.2,
                    "expected_information_gain": 0.45,
                    "base_gain": 0.2,
                    "keywords": ["exemple", "analogie"],
                },
            ],
            "abduction": [
                {
                    "key": "root_cause",
                    "description": "Identifier la cause racine supposée et vérifier un symptôme clé",
                    "cost_est": 0.28,
                    "expected_information_gain": 0.6,
                    "base_gain": 0.3,
                    "keywords": ["cause", "pourquoi"],
                }
            ],
            "deduction": [
                {
                    "key": "step_plan",
                    "description": "Détailler un plan en 3 étapes avec vérification de chaque sous-résultat",
                    "cost_est": 0.35,
                    "expected_information_gain": 0.62,
                    "base_gain": 0.35,
                    "keywords": ["plan", "étapes", "comment"],
                },
                {
                    "key": "assertion_check",
                    "description": "Lister les hypothèses critiques et prévoir un test de non-régression",
                    "cost_est": 0.32,
                    "expected_information_gain": 0.58,
                    "base_gain": 0.28,
                    "keywords": ["test", "vérifier"],
                },
            ],
            "analogy": [
                {
                    "key": "case_contrast",
                    "description": "Comparer avec un cas passé et noter les écarts structurants",
                    "cost_est": 0.25,
                    "expected_information_gain": 0.52,
                    "base_gain": 0.27,
                    "keywords": ["analogie", "similaire"],
                }
            ],
        }

    def _extract_features(self, lowered_prompt: str, hypothesis: Hypothesis, strategy: str) -> Dict[str, float]:
        return {
            "bias": 1.0,
            "prompt_len": min(1.0, len(lowered_prompt) / 400.0),
            "contains_question": 1.0
            if any(token in lowered_prompt for token in ["?", "pourquoi", "why", "comment", "how"])
            else 0.0,
            "contains_test": 1.0
            if any(token in lowered_prompt for token in ["test", "verifier", "vérifier", "validation"])
            else 0.0,
            "strategy_match": 1.0 if self._strategy_keyword_match(lowered_prompt, strategy) else 0.0,
            "hypothesis_prior": min(1.0, max(0.0, hypothesis.prior)),
        }

    def _strategy_keyword_match(self, lowered_prompt: str, strategy: str) -> bool:
        keywords = {
            "abduction": ["cause", "pourquoi", "root"],
            "deduction": ["plan", "étapes", "steps", "procedure"],
            "analogy": ["comme", "similaire", "analogie", "exemple"],
        }
        return any(word in lowered_prompt for word in keywords.get(strategy, []))

    def _normalize_preferences(self, prefs: Dict[str, float]) -> None:
        total = sum(prefs.values())
        if total > 0:
            for key in prefs:
                prefs[key] = float(prefs[key] / total)
