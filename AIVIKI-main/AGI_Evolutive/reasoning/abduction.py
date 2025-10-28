from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Deque, Dict, List, Optional, Tuple
import datetime as dt
import logging
import re
from collections import defaultdict, deque

from ..learning import OnlineLinearModel
from ..utils.llm_service import try_call_llm_dict


logger = logging.getLogger(__name__)


FRENCH_CAUSE_REGEX = re.compile(
    r"\best\s+(?:un|une|le|la|l'|les|des)\s+([\wàâäéèêëîïôöùûüç'-]+)",
    re.IGNORECASE,
)

from .structures import CausalStore, DomainSimulator, HTNPlanner, SimulationResult, TaskNode

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from .question_engine import QuestionEngine

@dataclass
class Hypothesis:
    """Abductive hypothesis enriched with scoring metadata.

    Utilisé uniquement par le moteur d'abduction : il prolonge la
    structure simple définie dans ``reasoning.structures`` avec des champs
    supplémentaires (priors, causal_support, plan, etc.).
    """
    label: str
    explanation: str
    score: float
    priors: Dict[str, float]
    evidence: List[str]
    ask_next: Optional[str] = None
    causal_support: List[str] = field(default_factory=list)
    simulations: List[Dict[str, Any]] = field(default_factory=list)
    plan: Optional[List[str]] = None
    generator: Optional[str] = None
    source_features: Dict[str, float] = field(default_factory=dict)


class AdaptiveTextClassifier:
    """Online text classifier used as a robust fallback."""

    def __init__(self, bounds: Tuple[float, float] = (0.0, 1.0)) -> None:
        self.bounds = bounds
        self._models: Dict[str, OnlineLinearModel] = {}

    def _get_model(self, label: str) -> OnlineLinearModel:
        if label not in self._models:
            self._models[label] = OnlineLinearModel(
                feature_dim=8,
                learning_rate=0.08,
                l2=1e-3,
                bounds=self.bounds,
            )
        return self._models[label]

    def _features(self, label: str, text: str) -> List[float]:
        lower = text.lower()
        length = max(len(text), 1)
        tokens = re.findall(r"[\wàâäéèêëîïôöùûüç']+", lower)
        token_count = max(len(tokens), 1)
        punct_count = sum(1 for ch in text if ch in "!?…")
        emoji_count = sum(1 for ch in text if ord(ch) > 1000)
        uppercase_ratio = sum(1 for ch in text if ch.isupper()) / float(length)
        accent_ratio = sum(1 for ch in text if ch in "àâäéèêëîïôöùûüç") / float(length)
        label_mentions = lower.count(label.lower())
        features = [
            1.0,
            label_mentions / float(token_count),
            uppercase_ratio,
            punct_count / 10.0,
            emoji_count / 5.0,
            accent_ratio,
            min(1.0, len(tokens) / 25.0),
            1.0 if any(token.startswith(label[:4].lower()) for token in tokens) else 0.0,
        ]
        return features

    def predict(self, label: str, text: str) -> float:
        model = self._get_model(label)
        return float(model.predict(self._features(label, text)))

    def update(self, label: str, text: str, reward: float) -> None:
        model = self._get_model(label)
        model.update(self._features(label, text), reward)

    def suggest(self, text: str, threshold: float = 0.55) -> List[Tuple[str, float]]:
        suggestions: List[Tuple[str, float]] = []
        for label in self._models:
            score = self.predict(label, text)
            if score >= threshold:
                suggestions.append((label, score))
        suggestions.sort(key=lambda item: item[1], reverse=True)
        return suggestions[:3]


class AbductiveAdaptationManager:
    """Centralises online learning signals for the abductive pipeline."""

    def __init__(self) -> None:
        self._scorer = OnlineLinearModel(
            feature_dim=9,
            learning_rate=0.06,
            l2=5e-4,
            bounds=(0.0, 1.0),
        )
        self.generator_stats: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=50))
        self.text_classifier = AdaptiveTextClassifier()

    def _feature_vector(
        self,
        base_score: float,
        priors: Dict[str, float],
        context: Dict[str, float],
    ) -> List[float]:
        return [
            1.0,
            base_score,
            priors.get("pri", 0.0),
            priors.get("boost", 0.0),
            priors.get("matches", 0.0),
            context.get("causal_strength", 0.0),
            context.get("simulation_success", 0.0),
            context.get("plan_depth", 0.0),
            context.get("text_confidence", base_score),
        ]

    def score(
        self,
        base_score: float,
        priors: Dict[str, float],
        context: Dict[str, float],
    ) -> float:
        features = self._feature_vector(base_score, priors, context)
        return float(self._scorer.predict(features))

    def update_feedback(
        self,
        label: str,
        base_score: float,
        priors: Dict[str, float],
        context: Dict[str, float],
        reward: float,
    ) -> None:
        features = self._feature_vector(base_score, priors, context)
        self._scorer.update(features, reward)
        observation = context.get("observation", "")
        if observation:
            self.text_classifier.update(label, observation, reward)
        generator = context.get("generator")
        if generator:
            self.generator_stats[generator].append(reward)

    def generator_weight(self, name: str) -> float:
        stats = self.generator_stats.get(name)
        if not stats:
            return 1.0
        avg = sum(stats) / max(1, len(stats))
        return 0.5 + avg

    def register_generator(self, name: str, priors: Optional[float] = None) -> None:
        if name not in self.generator_stats:
            self.generator_stats[name] = deque(maxlen=50)


class AbductiveReasoner:
    """
    Pipeline:
      - generators -> candidats
      - scoring -> score(H) par priors + matches + beliefs
      - causalité explicite (CausalStore)
      - vérification par simulation (DomainSimulator)
      - planification HTN (HTNPlanner)
      - active question (politique d'entropie)
      - calibration (log prédiction) gérée côté arch
    """

    def __init__(
        self,
        beliefs,
        user_model,
        generators: Optional[List[Callable[[str], List[Tuple[str, str]]]]] = None,
        causal_store: Optional[CausalStore] = None,
        simulator: Optional[DomainSimulator] = None,
        planner: Optional[HTNPlanner] = None,
        question_policy: Optional["EntropyQuestionPolicy"] = None,
        question_engine: Optional["QuestionEngine"] = None,
    ):
        self.beliefs = beliefs
        self.user = user_model
        self.generators = generators or [self._g_default]
        self.causal_store = causal_store or CausalStore()
        self.simulator = simulator or DomainSimulator()
        self.planner = planner or HTNPlanner()
        self.question_policy = question_policy or EntropyQuestionPolicy()
        if question_engine is None:
            from .question_engine import QuestionEngine as _QuestionEngine

            self.qengine = _QuestionEngine(beliefs, user_model)
        else:
            self.qengine = question_engine

        if not self.planner.has_template("diagnostic_general"):
            self._register_default_plan()

        self.adaptation = AbductiveAdaptationManager()
        for gen in self.generators:
            self.adaptation.register_generator(getattr(gen, "__name__", gen.__class__.__name__))
        self.adaptation.register_generator("adaptive_classifier")
        self._last_observation: Optional[str] = None
        self._last_contexts: Dict[str, Dict[str, Any]] = {}

    def _g_default(self, text: str) -> List[Tuple[str, str]]:
        lower = text.lower()
        candidates: List[Tuple[str, str]] = []
        if "jaune" in lower and ("ongle" in lower or "doigt" in lower):
            candidates.extend(
                [
                    ("cheddar", "résidu de fromage fondu"),
                    ("curcuma", "résidu d'épice"),
                    ("nicotine", "tache nicotine"),
                    ("infection", "sécrétion ou mycose"),
                    ("autre", "autre cause possible identifiée"),
                ]
            )
        for match in FRENCH_CAUSE_REGEX.findall(text):
            focus = match.lower()
            candidates.append(
                (
                    focus,
                    f"pattern linguistique détecté (« est {match} »)",
                )
            )
        if not candidates:
            candidates.append(("inconnu", "aucun indice déterminant détecté"))
        return candidates

    def _time(self) -> dt.datetime:
        return dt.datetime.now()

    def _score(self, label: str, text: str) -> Tuple[float, Dict[str,float], List[str]]:
        t = text.lower(); now = self._time()
        pri = 0.5
        ev: List[str] = []
        # priors utilisateur
        pri = max(pri, self.user.prior(label)) if hasattr(self.user, "prior") else pri
        pri = min(1.0, pri + self.user.routine_bias(label, now)) if hasattr(self.user, "routine_bias") else pri
        # beliefs
        boost = 0.0
        for b in self.beliefs.query(subject="user", relation="likes", min_conf=0.6):
            if b.value in {label, f"{label}_related"}:
                boost += 0.15
        # matches (features toy; à étendre par domaine)
        matches = 0.0
        if "midi" in t: matches += 0.05
        if "réchauff" in t or "micro" in t: matches += 0.05
        if label in t: matches += 0.2
        score = max(0.0, min(1.0, 0.5*pri + boost + matches))
        return score, {"pri": pri, "boost": boost, "matches": matches}, ev

    def generate(self, observation_text: str) -> List[Hypothesis]:
        self._last_observation = observation_text
        self._last_contexts = {}
        candidate_rows: List[Tuple[str, str, str]] = []
        for g in self.generators:
            name = getattr(g, "__name__", g.__class__.__name__)
            try:
                for label, why in g(observation_text):
                    candidate_rows.append((label, why, name))
            except Exception:
                continue

        classifier_suggestions = self.adaptation.text_classifier.suggest(observation_text)
        for label, score in classifier_suggestions:
            candidate_rows.append(
                (
                    label,
                    f"suggestion classifieur (score {score:.2f})",
                    "adaptive_classifier",
                )
            )

        llm_guidance: Dict[str, Dict[str, Any]] = {}
        try:
            user_snapshot = {}
            descriptor = getattr(self.user, "describe", None)
            if callable(descriptor):
                try:
                    user_snapshot = descriptor() or {}
                except Exception:
                    user_snapshot = {}
            llm_response = try_call_llm_dict(
                "abductive_reasoner",
                input_payload={
                    "observation": observation_text,
                    "candidates": [
                        {"label": lab, "rationale": why, "generator": generator}
                        for lab, why, generator in candidate_rows
                    ],
                    "user": user_snapshot,
                },
                logger=logger,
            )
        except Exception:
            llm_response = None

        if llm_response and isinstance(llm_response.get("hypotheses"), list):
            for item in llm_response["hypotheses"]:
                if not isinstance(item, dict):
                    continue
                raw_label = item.get("name") or item.get("label") or ""
                label = str(raw_label).strip()
                if not label:
                    continue
                normalized = label.lower()
                probability_raw = item.get("probability", 0.0)
                probability = 0.0
                if probability_raw is not None:
                    try:
                        probability = float(probability_raw)
                    except (TypeError, ValueError):
                        logger.warning(
                            "Non-numeric probability '%s' returned by LLM for hypothesis '%s'",
                            probability_raw,
                            label,
                        )
                mechanism = str(item.get("mechanism", "") or "").strip()
                tests = [
                    str(test).strip()
                    for test in (item.get("tests") or [])
                    if isinstance(test, (str, bytes)) and str(test).strip()
                ]
                llm_guidance[normalized] = {
                    "label": label,
                    "probability": max(0.0, min(1.0, probability)),
                    "mechanism": mechanism,
                    "tests": tests,
                }
                if not any(existing_label.lower() == normalized for existing_label, _, _ in candidate_rows):
                    candidate_rows.append(
                        (
                            label,
                            mechanism or "hypothèse proposée par le LLM",
                            "llm",
                        )
                    )

        uniq: Dict[str, Dict[str, Any]] = {}
        for lab, why, generator in candidate_rows:
            weight = self.adaptation.generator_weight(generator)
            item = uniq.get(lab)
            if not item or weight > item["weight"]:
                uniq[lab] = {"why": why, "generator": generator, "weight": weight}

        hyps: List[Hypothesis] = []
        for lab, info in uniq.items():
            normalized_label = lab.lower()
            base_score, priors, evidence = self._score(lab, observation_text)
            causal_support = self._causal_support(lab, observation_text)
            simulations = self._run_simulations(lab, observation_text)
            plan = self._plan_validation(lab, observation_text)

            causal_strength = 0.0
            if causal_support:
                positives = sum(
                    1
                    for msg in causal_support
                    if "observé" in msg.lower() or "conditions ok" in msg.lower()
                )
                causal_strength = positives / float(len(causal_support))

            simulation_success = 0.0
            if simulations:
                simulation_success = sum(1.0 if sim.get("success") else 0.0 for sim in simulations) / float(
                    len(simulations)
                )

            plan_depth = min(1.0, (len(plan) if plan else 0) / 5.0)
            text_confidence = self.adaptation.text_classifier.predict(lab, observation_text)

            llm_info = llm_guidance.get(normalized_label)
            if llm_info:
                base_score = 0.5 * base_score + 0.5 * llm_info["probability"]
                if llm_info["mechanism"]:
                    info["why"] = (
                        f"{info['why']} · {llm_info['mechanism']}"
                        if info.get("why")
                        else llm_info["mechanism"]
                    )
                    causal_support = list(causal_support) + [
                        f"Mécanisme proposé par LLM: {llm_info['mechanism']}"
                    ]
                if llm_info["tests"] and not plan:
                    plan = list(llm_info["tests"])

            context = {
                "causal_strength": causal_strength,
                "simulation_success": simulation_success,
                "plan_depth": plan_depth,
                "text_confidence": text_confidence,
                "generator": info["generator"],
                "observation": observation_text,
            }
            if llm_info:
                context["llm_probability"] = llm_info["probability"]
                if llm_info["tests"]:
                    context["llm_tests"] = list(llm_info["tests"])

            adaptive_score = self.adaptation.score(base_score, priors, context)
            score = max(0.0, min(1.0, (0.6 * base_score + 0.4 * adaptive_score) * info["weight"]))

            meta_features = {
                "base_score": base_score,
                "adaptive_score": adaptive_score,
                "generator_weight": info["weight"],
                **context,
                **priors,
            }

            hyp = Hypothesis(
                label=lab,
                explanation=info["why"],
                score=score,
                priors=priors,
                evidence=evidence,
                causal_support=causal_support,
                simulations=simulations,
                plan=plan,
                generator=info["generator"],
                source_features=meta_features,
            )
            hyps.append(hyp)
            self._last_contexts[lab] = {
                "base_score": base_score,
                "priors": dict(priors),
                **context,
            }

        hyps.sort(key=lambda h: h.score, reverse=True)
        if self.qengine:
            try:
                self.qengine.set_hypotheses(hyps)
                question = self.qengine.best_question(hyps, observation_text)
            except Exception:
                question = None
        else:
            question = None
        if not question and self.question_policy:
            question = self.question_policy.suggest(hyps, observation_text)
        if question and hyps:
            hyps[0].ask_next = question
        return hyps[:5]

    def register_feedback(
        self,
        label: str,
        reward: float,
        observation_text: Optional[str] = None,
    ) -> None:
        if observation_text is None:
            observation_text = self._last_observation
        if not observation_text:
            return
        context = self._last_contexts.get(label)
        if not context:
            return
        clamped = max(0.0, min(1.0, float(reward)))
        priors = context.get("priors", {})
        base_score = context.get("base_score", 0.0)
        update_context = {
            key: value
            for key, value in context.items()
            if key not in {"priors", "base_score"}
        }
        update_context["observation"] = observation_text
        self.adaptation.update_feedback(label, base_score, priors, update_context, clamped)
        if hasattr(self.question_policy, "register_feedback"):
            try:
                self.question_policy.register_feedback(clamped)
            except Exception:
                pass
        self._last_contexts.pop(label, None)

    def _craft_question(self, a: str, b: str) -> str:
        # squelette générique
        return f"As-tu un indice concret qui pointerait plutôt vers {a} (trace, odeur, contexte horaire) ou {b} ?"

    # -- causal reasoning -------------------------------------------------
    def _register_default_plan(self) -> None:
        diagnostic = TaskNode(
            name="diagnostic_general",
            preconditions=["observation"],
            actions=[
                "Collecter un indice supplémentaire fiable.",
                "Comparer les hypothèses en fonction des conditions observées.",
                "Chercher une contre-preuve rapide.",
            ],
            postconditions=["hypothesis_validated"],
        )
        self.planner.register_template("diagnostic_general", diagnostic)

    def _causal_support(self, label: str, text: str) -> List[str]:
        if not self.causal_store:
            return []
        lower = text.lower()
        tokens = re.findall(r"[a-zàâäéèêëîïôöùûüç]+", lower)
        observed_effects = {token for token in tokens if len(token) > 3}
        supports: List[str] = []
        for link in self.causal_store.get_effects(label):
            observed = link.effect.lower() in observed_effects or link.effect.lower() in lower
            status = "observé" if observed else "à vérifier"
            msg = f"{label} ⇒ {link.effect} (force {link.strength:.2f}, {status})"
            if link.conditions:
                msg += f" | conditions: {', '.join(link.conditions)}"
            supports.append(msg)
            test = self.causal_store.test_relation(
                cause=label,
                effect=link.effect,
                context={"observation": text},
            )
            if test["supported"]:
                if test["unsatisfied_conditions"]:
                    cond_status = f"conditions manquantes: {', '.join(test['unsatisfied_conditions'])}"
                else:
                    cond_status = "conditions ok"
                supports.append(f"Test causal: {label} ⇒ {link.effect} ({cond_status})")
        for link in self.causal_store.get_causes(label):
            if link.cause.lower() in lower:
                supports.append(
                    f"Observation compatible: {link.cause} pourrait provoquer {label} (force {link.strength:.2f})"
                )
        return supports

    def _run_simulations(self, label: str, text: str) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        scenario = {"observation": text, "hypothesis": label}
        sim_result = self.simulator.simulate(label, scenario)
        results.append(
            {
                "domain": label,
                "timestamp": self._time().isoformat(),
                **sim_result.to_dict(),
            }
        )
        if not sim_result.success:
            coherence = label.lower() in text.lower()
            mental = SimulationResult(
                success=coherence,
                outcome=(
                    "Le scénario est cohérent avec les éléments textuels."
                    if coherence
                    else "Le scénario semble faible : aucun indice direct dans l'observation."
                ),
                details={"method": "mental_check"},
            )
            results.append(
                {
                    "domain": "mental",
                    "timestamp": self._time().isoformat(),
                    **mental.to_dict(),
                }
            )
        return results

    def _plan_validation(self, label: str, text: str) -> Optional[List[str]]:
        context = {"observation": text, "hypothesis": label}
        goal = f"valider_{label}"
        plan = self.planner.plan(goal, context=context)
        if not plan:
            plan = self.planner.plan("diagnostic_general", context=context)
        if not plan:
            plan = [
                f"Observer de près les indices associés à {label}.",
                "Comparer avec au moins une hypothèse alternative.",
                "Collecter une nouvelle donnée ciblée.",
            ]
        return plan


class EntropyQuestionPolicy:
    """Active questioning policy with online calibration of entropy threshold."""

    def __init__(self, base_threshold: float = 0.05) -> None:
        self.base_threshold = base_threshold
        self._meta = OnlineLinearModel(
            feature_dim=4,
            learning_rate=0.09,
            l2=1e-3,
            bounds=(0.0, 1.0),
        )
        self._last_features: Optional[List[float]] = None

    def suggest(self, hypotheses: List[Hypothesis], observation: str) -> Optional[str]:
        if len(hypotheses) < 2:
            self._last_features = None
            return None
        top = hypotheses[:4]
        scores = [max(h.score, 1e-4) for h in top]
        total = float(sum(scores))
        if total == 0.0:
            self._last_features = None
            return None
        probs = [score / total for score in scores]
        best_pair: Optional[Tuple[Hypothesis, Hypothesis]] = None
        best_gain = 0.0
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                gain = 2.0 * min(probs[i], probs[j])
                if gain > best_gain:
                    best_gain = gain
                    best_pair = (top[i], top[j])
        if not best_pair:
            self._last_features = None
            return None

        features = [
            1.0,
            best_gain,
            min(1.0, len(hypotheses) / 5.0),
            min(1.0, len(observation) / 120.0),
        ]
        confidence = self._meta.predict(features)
        dynamic_threshold = max(0.02, self.base_threshold * (0.8 + 0.4 * (1.0 - confidence)))
        if best_gain < dynamic_threshold:
            self._last_features = None
            return None

        a, b = best_pair
        discriminants: List[str] = []
        if a.causal_support:
            discriminants.append(a.causal_support[0])
        if b.causal_support:
            discriminants.append(b.causal_support[0])
        base = f"Quelle observation permettrait de trancher entre {a.label} et {b.label}?"
        if discriminants:
            base = (
                f"Quel indice vérifierait {a.label} ({discriminants[0]}) ou {b.label} ({discriminants[-1]}) ?"
            )
        info = f"(gain info ≈ {best_gain:.2f})"
        self._last_features = features
        return f"{base} {info}"

    def register_feedback(self, reward: float) -> None:
        if self._last_features is None:
            return
        clamped = max(0.0, min(1.0, float(reward)))
        self._meta.update(self._last_features, clamped)
        self._last_features = None
