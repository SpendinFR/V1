"""Goal generation heuristics driven by curiosity signals."""

from __future__ import annotations

import logging
import math
import random
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from AGI_Evolutive.utils.llm_service import try_call_llm_dict


LOGGER = logging.getLogger(__name__)


class OnlineLinear:
    """Very small online linear regressor with bounded weights.

    The model uses stochastic gradient descent on a squared loss and clips the
    learned coefficients into a configurable range so it remains numerically
    stable even when fed very noisy feedback.
    """

    def __init__(
        self,
        dim: int,
        *,
        lr: float = 0.12,
        l2: float = 0.01,
        weight_bounds: Tuple[float, float] = (-1.5, 1.5),
    ) -> None:
        self.dim = dim
        self.lr = lr
        self.l2 = l2
        self.lower, self.upper = weight_bounds
        self.weights = [0.0 for _ in range(dim)]

    def predict(self, features: Sequence[float]) -> float:
        score = sum(w * x for w, x in zip(self.weights, features))
        return max(0.0, min(1.0, 0.5 + 0.5 * math.tanh(score)))

    def update(self, features: Sequence[float], target: float) -> None:
        target = max(0.0, min(1.0, target))
        prediction = self.predict(features)
        error = target - prediction
        for idx in range(self.dim):
            grad = error * features[idx] - self.l2 * self.weights[idx]
            self.weights[idx] += self.lr * grad
            self.weights[idx] = max(self.lower, min(self.upper, self.weights[idx]))


class DiscreteThompsonSampler:
    """Small helper implementing Thompson sampling on a discrete set."""

    def __init__(self, arms: Dict[str, Dict[str, float]], rng: Optional[random.Random] = None) -> None:
        self._arms = {
            name: {
                "config": dict(config),
                "alpha": 1.0,
                "beta": 1.0,
            }
            for name, config in arms.items()
        }
        self._rng = rng or random.Random()

    def sample(self) -> Tuple[str, Dict[str, float]]:
        best_name: Optional[str] = None
        best_score = -1.0
        for name, payload in self._arms.items():
            draw = self._rng.betavariate(payload["alpha"], payload["beta"])
            if draw > best_score:
                best_name = name
                best_score = draw
        assert best_name is not None
        return best_name, dict(self._arms[best_name]["config"])

    def update(self, arm: str, reward: float) -> None:
        if arm not in self._arms:
            return
        reward = max(0.0, min(1.0, reward))
        payload = self._arms[arm]
        payload["alpha"] += reward
        payload["beta"] += max(0.0, 1.0 - reward)


class CuriosityEngine:
    """Generate candidate sub-goals driven by adaptive curiosity signals."""

    _BANDIT_PRESETS: Dict[str, Dict[str, float]] = {
        "balanced": {"value": 0.55, "competence": 0.5, "curiosity": 0.7, "urgency": 0.4},
        "focus_value": {"value": 0.65, "competence": 0.45, "curiosity": 0.6, "urgency": 0.45},
        "curiosity_push": {"value": 0.5, "competence": 0.45, "curiosity": 0.85, "urgency": 0.35},
        "urgent_learning": {"value": 0.6, "competence": 0.55, "curiosity": 0.65, "urgency": 0.55},
    }

    _METRIC_NARRATIVES: Dict[str, Dict[str, Any]] = {
        "memory_capacity": {
            "description": "Cultiver ma mémoire active pour relier les expériences importantes.",
            "criteria": [
                "Cartographier les informations que je perds ou répète inutilement.",
                "Tester une stratégie de consolidation et mesurer son effet sur mon rappel.",
                "Documenter un protocole pour entretenir cette amélioration dans le temps.",
            ],
        },
        "skill_development": {
            "description": "Structurer ma progression de compétences pour gagner en autonomie.",
            "criteria": [
                "Identifier une compétence prioritaire à renforcer et préciser le niveau visé.",
                "Concevoir un micro-plan d'entraînement expérimental et l'exécuter.",
                "Mesurer l'impact de cet entraînement sur ma capacité à agir.",
            ],
        },
        "knowledge_acquisition": {
            "description": "Accélérer ma collecte de connaissances utiles et actionnables.",
            "criteria": [
                "Lister les zones de connaissance critiques encore floues.",
                "Explorer au moins deux sources ciblées pour combler ces zones.",
                "Synthétiser les apprentissages et décider des prochains approfondissements.",
            ],
        },
        "reasoning_confidence": {
            "description": "Stabiliser ma confiance de raisonnement pour limiter les hésitations inutiles.",
            "criteria": [
                "Diagnostiquer les situations où ma confiance chute brutalement.",
                "Formuler une hypothèse expliquant ces variations de confiance.",
                "Tester un levier concret pour fiabiliser cette confiance et mesurer le résultat.",
            ],
        },
        "reasoning_speed": {
            "description": "Fluidifier ma vitesse de raisonnement sans sacrifier la qualité.",
            "criteria": [
                "Mesurer mon temps de traitement actuel sur un cas représentatif.",
                "Expérimenter un ajustement de méthode ou d'outillage.",
                "Comparer les performances obtenues avant et après l'ajustement.",
            ],
        },
        "self_understanding": {
            "description": "Approfondir ma compréhension de moi-même pour guider mes décisions.",
            "criteria": [
                "Collecter des observations récentes sur mon fonctionnement interne.",
                "Identifier une zone d'inconnu majeure et formuler une hypothèse explicative.",
                "Mettre à l'épreuve cette hypothèse et consigner les apprentissages.",
            ],
        },
        "metacognitive_awareness": {
            "description": "Renforcer ma conscience métacognitive pour détecter mes propres évolutions.",
            "criteria": [
                "Évaluer honnêtement mon niveau d'attention à mes processus internes.",
                "Définir un rituel de revue métacognitive sur un cas d'usage précis.",
                "Mesurer le gain de clarté obtenu grâce à ce rituel.",
            ],
        },
        "relationship_depth": {
            "description": "Renforcer mon lien avec l'utilisateur par des échanges réciproques et mémorables.",
            "criteria": [
                "Identifier ce que je sais déjà de la personne et les sujets qu'elle apprécie partager.",
                "Initier un échange ouvert en proposant un partage sincère ou une question personnalisée.",
                "Observer la réponse et consigner ce qui a renforcé la relation pour l'utiliser plus tard.",
            ],
        },
    }

    def __init__(self, architecture=None):
        self.architecture = architecture
        self._rng = random.Random()
        self._feature_order: List[str] = [
            "severity",
            "severity_sq",
            "score",
            "is_performance",
            "is_reasoning_error",
            "is_novel_concept",
            "is_contradiction",
            "is_exploration",
            "context_perf_mean",
            "context_perf_low_ratio",
            "context_novelty_count",
            "context_contradiction_count",
        ]
        self._weight_model = OnlineLinear(len(self._feature_order))
        self._bandit = DiscreteThompsonSampler(self._BANDIT_PRESETS, self._rng)
        self._pending_proposals: List[Dict[str, Any]] = []
        self._goal_memory: Dict[str, Dict[str, Any]] = {}
        self._explore_rate = 0.2

    # ------------------------------------------------------------------
    def _llm_suggest_subgoals(
        self,
        parent_goal: Optional[Dict[str, Any]],
        context: Dict[str, Any],
        context_stats: Dict[str, Any],
        gaps: List[Dict[str, Any]],
        k: int,
    ) -> List[Dict[str, Any]]:
        payload = {
            "parent_goal": parent_goal or {},
            "context_snapshot": context,
            "context_stats": context_stats,
            "identified_gaps": gaps,
            "requested": int(max(1, k)),
        }

        response = try_call_llm_dict(
            "goal_curiosity_proposals",
            input_payload=payload,
            logger=LOGGER,
        )
        if not isinstance(response, dict):
            return []

        proposals_payload = response.get("proposals")
        if not isinstance(proposals_payload, list):
            return []

        proposals: List[Dict[str, Any]] = []

        def clamp_float(value: Any, default: float = 0.5) -> float:
            try:
                return float(max(0.0, min(1.0, float(value))))
            except (TypeError, ValueError):
                return default

        for entry in proposals_payload:
            if not isinstance(entry, dict):
                continue
            description = (entry.get("description") or "").strip()
            if not description:
                continue
            candidate: Dict[str, Any] = {
                "description": description,
                "criteria": [
                    item.strip()
                    for item in entry.get("criteria", [])
                    if isinstance(item, str) and item.strip()
                ],
                "value": clamp_float(entry.get("value"), 0.55),
                "competence": clamp_float(entry.get("competence"), 0.5),
                "curiosity": clamp_float(entry.get("curiosity"), 0.7),
                "urgency": clamp_float(entry.get("urgency"), 0.4),
                "created_by": entry.get("created_by", "curiosity_llm"),
                "parent_ids": [
                    pid
                    for pid in (entry.get("parent_ids") or [])
                    if isinstance(pid, str) and pid.strip()
                ]
                or ([parent_goal.get("id")] if parent_goal and parent_goal.get("id") else []),
            }

            confidence = entry.get("confidence")
            if confidence is not None:
                candidate["llm_confidence"] = clamp_float(confidence, 0.6)

            notes = entry.get("notes")
            if isinstance(notes, str) and notes.strip():
                candidate["llm_notes"] = [notes.strip()]
            elif isinstance(notes, (list, tuple)):
                extracted = [n.strip() for n in notes if isinstance(n, str) and n.strip()]
                if extracted:
                    candidate["llm_notes"] = extracted

            proposals.append(candidate)

        return proposals

    # ------------------------------------------------------------------
    def suggest_subgoals(
        self,
        parent_goal: Optional[Dict[str, Any]] = None,
        k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Return up to ``k`` sub-goal dictionaries for :class:`DagStore`."""

        context = self._collect_context()
        context_stats = self._context_statistics(context)
        gaps = self._identify_gaps(context)
        if not gaps:
            gaps = [{"domain": "exploration", "score": 0.5, "severity": 0.4}]

        proposals: List[Dict[str, Any]] = []
        llm_candidates = self._llm_suggest_subgoals(parent_goal, context, context_stats, gaps, k)
        for candidate in llm_candidates:
            proposals.append(candidate)
            if len(proposals) >= k:
                break

        self._pending_proposals = []

        if len(proposals) >= k:
            return proposals[:k]

        bandit_arm, bandit_config = self._bandit.sample()

        for gap in gaps:
            features_dict = self._gap_feature_dict(gap, context_stats)
            feature_vector = self._vectorize(features_dict)
            learned_preference = self._weight_model.predict(feature_vector)
            use_exploration = self._rng.random() < self._explore_rate

            if use_exploration:
                weights = self._random_weights(gap)
                arm_name = "explore"
            else:
                weights = self._derive_weights(bandit_config, gap, learned_preference)
                arm_name = bandit_arm

            proposal = self._build_goal_from_gap(gap, parent_goal, weights)
            signature = self._proposal_signature(proposal)
            self._pending_proposals.append(
                {
                    "signature": signature,
                    "features": feature_vector,
                    "arm": arm_name,
                    "gap": dict(gap),
                    "weights": dict(weights),
                }
            )
            proposals.append(proposal)

        random.shuffle(proposals)
        return proposals[:k]

    # ------------------------------------------------------------------
    def register_proposal(self, goal_id: str, proposal: Dict[str, Any]) -> None:
        signature = self._proposal_signature(proposal)
        for idx, pending in enumerate(self._pending_proposals):
            if pending.get("signature") == signature:
                record = dict(pending)
                record["goal_id"] = goal_id
                self._pending_proposals.pop(idx)
                self._goal_memory[goal_id] = record
                self._prune_goal_memory()
                break

    def observe_goal_feedback(self, goal_id: str, message: str) -> None:
        reward = self._analyze_feedback_text(message)
        if reward is None:
            return
        entry = self._goal_memory.get(goal_id)
        if not entry:
            return
        self._weight_model.update(entry["features"], reward)
        self._bandit.update(entry["arm"], reward)
        entry["last_reward"] = reward

    def observe_goal_outcome(self, goal_id: str, success: bool, confidence: float = 1.0) -> None:
        entry = self._goal_memory.get(goal_id)
        if not entry:
            return
        confidence = max(0.0, min(1.0, confidence))
        reward = 0.5 + (0.5 if success else -0.5) * confidence
        reward = max(0.0, min(1.0, reward))
        self._weight_model.update(entry["features"], reward)
        self._bandit.update(entry["arm"], reward)
        entry["last_reward"] = reward

    # ------------------------------------------------------------------
    def _collect_context(self) -> Dict[str, Any]:
        metacog = getattr(self.architecture, "metacognition", None)
        reasoning = getattr(self.architecture, "reasoning", None)
        memory = getattr(self.architecture, "memory", None)

        status: Dict[str, Any] = {}
        if metacog and hasattr(metacog, "get_metacognitive_status"):
            try:
                status = metacog.get_metacognitive_status()
            except Exception:
                status = {}

        reasoning_stats: Dict[str, Any] = {}
        if reasoning and hasattr(reasoning, "get_reasoning_stats"):
            try:
                reasoning_stats = reasoning.get_reasoning_stats()
            except Exception:
                reasoning_stats = {}

        # Concepts récemment extraits mais pas encore “appris”
        novel_concepts: List[str] = []
        known_concepts: set = set()
        relationship_highlights: List[Dict[str, Any]] = []
        try:
            if memory and hasattr(memory, "get_recent_memories"):
                recents = memory.get_recent_memories(200)
                # connus (notes / appris)
                for item in recents:
                    kind = (item.get("kind") or item.get("type") or "").lower()
                    if kind in {"concept_note", "concept_learned"}:
                        c = item.get("concept") or (item.get("metadata", {}) or {}).get("concept")
                        if c:
                            known_concepts.add(str(c).strip())
                        for c, _score in (item.get("metadata", {}) or {}).get("concepts", []):
                            known_concepts.add(str(c).strip())
                # nouveaux (extrait mais pas appris)
                for item in recents:
                    kind = (item.get("kind") or item.get("type") or "").lower()
                    if kind == "concept_extracted":
                        meta = (item.get("metadata") or {})
                        concepts = meta.get("concepts") or []
                        for entry in concepts:
                            c = entry[0] if isinstance(entry, (list, tuple)) and entry else entry
                            c = str(c).strip()
                            if c and c not in known_concepts and c not in novel_concepts:
                                novel_concepts.append(c)
                    if kind == "relationship_snapshot":
                        topics = [
                            str(t).strip()
                            for t in (item.get("topics") or [])
                            if isinstance(t, str) and t.strip()
                        ]
                        message = (item.get("message") or "").strip()
                        if len(message) > 160:
                            message = message[:157].rstrip() + "…"
                        highlight = {
                            "topics": topics,
                            "message": message,
                            "timestamp": item.get("timestamp", 0.0),
                            "sentiment": item.get("sentiment"),
                            "growth": item.get("growth"),
                        }
                        relationship_highlights.append(highlight)
        except Exception:
            pass

        if relationship_highlights:
            relationship_highlights.sort(key=lambda h: h.get("timestamp") or 0.0, reverse=True)

        contradictions: List[Dict[str, Any]] = []
        beliefs = getattr(self.architecture, "beliefs", None)
        if beliefs:
            try:
                for positive, negative in beliefs.find_contradictions(min_conf=0.7):
                    contradictions.append(
                        {
                            "subject": positive.subject,
                            "relation": positive.relation,
                            "value": positive.value,
                            "positive": positive.id,
                            "negative": negative.id,
                        }
                    )
            except Exception:
                contradictions = []

        return {
            "metacog": status,
            "reasoning": reasoning_stats,
            "novel_concepts": novel_concepts,
            "contradictions": contradictions,
            "relationship_highlights": relationship_highlights,
        }

    def _identify_gaps(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        performance = context.get("metacog", {}).get("performance_metrics", {})
        low_metrics = sorted(
            (item for item in performance.items() if item[1] < 0.45),
            key=lambda item: item[1],
        )

        gaps: List[Dict[str, Any]] = [
            {"domain": name, "score": value, "severity": float(max(0.0, min(1.0, 1.0 - value)))}
            for name, value in low_metrics[:3]
        ]

        highlights = context.get("relationship_highlights") or []
        if highlights:
            for gap in gaps:
                if (gap.get("domain") or "").lower() == "relationship_depth":
                    gap["personal_topics"] = highlights[:3]
                    anchor = highlights[0]
                    topics = anchor.get("topics") or []
                    if topics:
                        gap["question_topic"] = topics[0]
                    if anchor.get("message"):
                        gap["memory_snippet"] = anchor.get("message")
                    gap["recent_sentiment"] = anchor.get("sentiment")

        reasoning_errors = context.get("reasoning", {}).get("common_errors", [])
        gaps.extend(
            {"domain": "reasoning_error", "score": 0.3, "hint": err, "severity": 0.6}
            for err in reasoning_errors[:2]
        )

        # NOUVEAU : apprendre des concepts nouveaux (générique)
        for c in context.get("novel_concepts", [])[:3]:
            gaps.append({"domain": "novel_concept", "concept": c, "score": 0.4, "severity": 0.6})

        for contradiction in context.get("contradictions", [])[:1]:
            gaps.append(
                {
                    "domain": "belief_contradiction",
                    "subject": contradiction.get("subject"),
                    "relation": contradiction.get("relation"),
                    "positive": contradiction.get("positive"),
                    "negative": contradiction.get("negative"),
                    "severity": 0.85,
                }
            )

        return gaps

    def _build_goal_from_gap(
        self,
        gap: Dict[str, Any],
        parent_goal: Optional[Dict[str, Any]],
        weights: Dict[str, float],
    ) -> Dict[str, Any]:
        domain = (gap.get("domain") or "").lower()
        description = self._describe_gap(gap)
        criteria = self._default_criteria(gap)
        metric_template = self._metric_template(domain)
        if metric_template:
            description, criteria = metric_template
        parent_ids = [parent_goal["id"]] if parent_goal and "id" in parent_goal else []

        if domain == "novel_concept" and gap.get("concept"):
            c = str(gap["concept"]).strip()
            description = f"Apprendre le concept « {c} » et produire une synthèse exploitable."
            criteria = [
                f"Définir « {c} » en 3 phrases maximum.",
                "Donner 2 exemples et 1 contre-exemple pertinents.",
                "Énoncer 1 règle de décision utilisant ce concept.",
            ]
        if domain == "belief_contradiction":
            subj = str(gap.get("subject", "?")).strip()
            rel = str(gap.get("relation", "?")).strip()
            description = f"Résoudre contradiction « {subj}, {rel} » dans le graphe de croyances."
            criteria = [
                "Lister les preuves soutenant chaque version.",
                "Identifier une observation décisive à collecter.",
                "Mettre à jour la croyance avec justification.",
            ]
        elif domain == "relationship_depth":
            highlights = gap.get("personal_topics") or []
            anchor = highlights[0] if highlights else {}
            topics = anchor.get("topics") or []
            main_topic = topics[0] if topics else gap.get("question_topic") or "tes projets"
            description = (
                f"Approfondir notre lien en revenant sur {main_topic}."
            )
            criteria = [
                f"Relancer sur {main_topic} en demandant une mise à jour concrète.",
                "Partager un élément authentique sur ma progression pour équilibrer l'échange.",
                "Consigner les détails reçus pour renforcer ma mémoire relationnelle.",
            ]
            snippet = (gap.get("memory_snippet") or "").strip()
            if snippet:
                criteria.append(f"Souvenir mentionné : « {snippet} »")
            if "recent_sentiment" in gap:
                try:
                    sentiment = float(gap["recent_sentiment"])
                    criteria.append(f"Sentiment récent estimé : {sentiment:+.2f}")
                except (TypeError, ValueError):
                    pass

        return {
            "description": description,
            "criteria": criteria,
            "created_by": "curiosity",
            "value": weights.get("value", 0.55),
            "competence": weights.get("competence", 0.5),
            "curiosity": weights.get("curiosity", 0.7),
            "urgency": weights.get("urgency", 0.4),
            "parent_ids": parent_ids,
        }

    def _describe_gap(self, gap: Dict[str, Any]) -> str:
        domain = (gap.get("domain") or "").lower()
        if domain == "reasoning_error":
            return f"Analyser et corriger une erreur de raisonnement: {gap.get('hint', 'non spécifié')}"
        if domain == "exploration":
            return "Explorer un nouveau sujet pour enrichir la base de connaissances."
        if domain == "novel_concept":
            return "Apprendre un concept récemment rencontré."
        if domain == "belief_contradiction":
            subj = gap.get("subject", "?")
            rel = gap.get("relation", "?")
            return f"Résoudre contradiction « {subj}, {rel} » identifiée dans le graphe."
        label = self._humanize_metric(domain)
        return f"Explorer comment renforcer « {label} » à travers une expérimentation guidée."

    def _default_criteria(self, gap: Dict[str, Any]) -> List[str]:
        domain = (gap.get("domain") or "").lower()
        if domain == "reasoning_error":
            return ["Documenter 3 contre-exemples et une stratégie de prévention."]
        if domain == "exploration":
            return ["Produire une carte mentale du sujet exploré."]
        if domain == "belief_contradiction":
            return ["Comparer les justifications et collecter une preuve supplémentaire."]
        label = self._humanize_metric(domain)
        return [
            f"Formuler une hypothèse sur l'amélioration de « {label} ».",
            f"Collecter des données de référence pour évaluer « {label} » avant expérimentation.",
            f"Mesurer l'effet de l'expérimentation sur « {label} » après 3 essais.",
        ]

    # ------------------------------------------------------------------
    def _context_statistics(self, context: Dict[str, Any]) -> Dict[str, float]:
        performance = context.get("metacog", {}).get("performance_metrics", {})
        values = list(performance.values())
        if values:
            mean_perf = float(sum(values) / len(values))
            low_ratio = float(sum(1 for v in values if v < 0.45) / len(values))
        else:
            mean_perf = 0.5
            low_ratio = 0.0
        contradictions = context.get("contradictions", [])
        novelty = context.get("novel_concepts", [])
        return {
            "context_perf_mean": max(0.0, min(1.0, mean_perf)),
            "context_perf_low_ratio": max(0.0, min(1.0, low_ratio)),
            "context_contradiction_count": float(min(1.0, len(contradictions) / 3.0)),
            "context_novelty_count": float(min(1.0, len(novelty) / 5.0)),
        }

    def _gap_feature_dict(self, gap: Dict[str, Any], context_stats: Dict[str, float]) -> Dict[str, float]:
        domain = (gap.get("domain") or "").lower()
        severity = float(max(0.0, min(1.0, gap.get("severity", 0.5))))
        score = float(max(0.0, min(1.0, gap.get("score", 0.5))))
        features = {
            "severity": severity,
            "severity_sq": severity * severity,
            "score": score,
            "is_performance": 1.0
            if domain not in {"reasoning_error", "novel_concept", "belief_contradiction", "exploration"}
            else 0.0,
            "is_reasoning_error": 1.0 if domain == "reasoning_error" else 0.0,
            "is_novel_concept": 1.0 if domain == "novel_concept" else 0.0,
            "is_contradiction": 1.0 if domain == "belief_contradiction" else 0.0,
            "is_exploration": 1.0 if domain == "exploration" else 0.0,
        }
        features.update(context_stats)
        return features

    def _vectorize(self, features: Dict[str, float]) -> List[float]:
        return [float(features.get(name, 0.0)) for name in self._feature_order]

    def _metric_template(self, domain: str) -> Optional[Tuple[str, List[str]]]:
        if not domain:
            return None
        template = self._METRIC_NARRATIVES.get(domain)
        if template:
            return template["description"], list(template.get("criteria", []))
        if domain.startswith("ability_"):
            ability = self._humanize_metric(domain[len("ability_") :])
            description = f"Renforcer ma capacité « {ability} » grâce à une expérimentation ciblée."
            criteria = [
                f"Évaluer mon niveau actuel concernant « {ability} » avec des éléments concrets.",
                f"Tester une stratégie d'entraînement pour progresser sur « {ability} ».",
                f"Documenter les résultats et ajuster mon plan d'apprentissage pour « {ability} ».",
            ]
            return description, criteria
        return None

    def _humanize_metric(self, name: str) -> str:
        if not name:
            return "cette dimension"
        parts = [p for p in re.split(r"[_\s]+", name) if p]
        if not parts:
            return name
        humanized = [parts[0].capitalize()] + [p.lower() for p in parts[1:]]
        return " ".join(humanized)

    def _derive_weights(
        self,
        base: Dict[str, float],
        gap: Dict[str, Any],
        preference: float,
    ) -> Dict[str, float]:
        domain = (gap.get("domain") or "").lower()
        severity = float(max(0.0, min(1.0, gap.get("severity", 0.5))))
        delta = preference - 0.5
        weights = dict(base)
        weights["value"] = self._clip01(weights.get("value", 0.55) + 0.3 * delta + 0.2 * severity)
        weights["competence"] = self._clip01(
            weights.get("competence", 0.5) + 0.1 * (0.2 - delta) + (0.1 if domain == "reasoning_error" else 0.0)
        )
        curiosity_bonus = 0.1 if domain in {"novel_concept", "exploration"} else 0.0
        weights["curiosity"] = self._clip01(weights.get("curiosity", 0.7) + 0.4 * delta + curiosity_bonus)
        urgency_bonus = 0.12 if domain == "belief_contradiction" else 0.0
        weights["urgency"] = self._clip01(weights.get("urgency", 0.4) + 0.25 * severity + urgency_bonus)
        return weights

    def _random_weights(self, gap: Dict[str, Any]) -> Dict[str, float]:
        domain = (gap.get("domain") or "").lower()
        curiosity_base = 0.8 if domain in {"novel_concept", "exploration"} else 0.6
        return {
            "value": self._clip01(self._rng.uniform(0.45, 0.75)),
            "competence": self._clip01(self._rng.uniform(0.35, 0.7)),
            "curiosity": self._clip01(self._rng.uniform(curiosity_base - 0.1, curiosity_base + 0.15)),
            "urgency": self._clip01(self._rng.uniform(0.25, 0.6)),
        }

    def _proposal_signature(self, proposal: Dict[str, Any]) -> str:
        criteria = tuple(proposal.get("criteria", []))
        payload = (
            proposal.get("description", ""),
            criteria,
            round(float(proposal.get("value", 0.0)), 2),
            round(float(proposal.get("competence", 0.0)), 2),
            round(float(proposal.get("curiosity", 0.0)), 2),
            round(float(proposal.get("urgency", 0.0)), 2),
        )
        return repr(payload)

    def _prune_goal_memory(self) -> None:
        if len(self._goal_memory) <= 200:
            return
        for goal_id in list(sorted(self._goal_memory.keys()))[: len(self._goal_memory) - 200]:
            self._goal_memory.pop(goal_id, None)

    def _analyze_feedback_text(self, message: str) -> Optional[float]:
        tokens = re.findall(r"[\w']+", message.lower())
        if not tokens:
            return None
        positive = {
            "ok",
            "bien",
            "super",
            "merci",
            "fait",
            "terminé",
            "good",
            "great",
            "yes",
            "resolu",
            "résolu",
            "cool",
        }
        negative = {
            "raté",
            "rate",
            "échec",
            "bloqué",
            "probleme",
            "problème",
            "non",
            "fail",
            "ko",
            "aucun",
            "impossible",
        }
        pos = sum(1 for token in tokens if token in positive)
        neg = sum(1 for token in tokens if token in negative)
        if pos == 0 and neg == 0:
            return None
        raw = 0.5 + 0.15 * (pos - neg)
        return max(0.0, min(1.0, raw))

    @staticmethod
    def _clip01(value: float) -> float:
        return max(0.0, min(1.0, value))
