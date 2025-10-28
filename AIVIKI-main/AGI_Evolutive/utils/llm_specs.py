"""Catalog of LLM integration specs derived from architecture review."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .llm_client import build_json_prompt


AVAILABLE_MODELS = {
    "fast": "qwen2.5:7b-instruct-q4_K_M",
    "reasoning": "qwen3:8b-q4_K_M",
}


@dataclass(frozen=True)
class LLMIntegrationSpec:
    """Describe how a subsystem should invoke the local LLM."""

    key: str
    module: str
    prompt_goal: str
    preferred_model: str
    extra_instructions: Sequence[str]
    example_output: Mapping[str, Any]

    def build_prompt(self, *, input_payload: Any | None = None, extra: Iterable[str] | None = None) -> str:
        instructions: list[str] = list(self.extra_instructions)
        if extra:
            instructions.extend(extra)
        instructions.append(
            "Si tu n'es pas certain, explique l'incertitude dans le champ 'notes'."
        )
        return build_json_prompt(
            self.prompt_goal,
            input_data=input_payload,
            extra_instructions=instructions,
            example_output=self.example_output,
        )


def _spec(
    key: str,
    module: str,
    prompt_goal: str,
    preferred_model: str,
    *,
    extra_instructions: Sequence[str] | None = None,
    example_output: Mapping[str, Any] | None = None,
) -> LLMIntegrationSpec:
    return LLMIntegrationSpec(
        key=key,
        module=module,
        prompt_goal=prompt_goal,
        preferred_model=preferred_model,
        extra_instructions=tuple(extra_instructions or ()),
        example_output=example_output or {},
    )


LLM_INTEGRATION_SPECS: tuple[LLMIntegrationSpec, ...] = (
    _spec(
        "package_overview",
        "AGI_Evolutive/__init__.py",
        "Analyse la structure du package AGI_Evolutive et propose une synthèse actionnable.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=(
            "Résume le rôle global du package en une phrase claire.",
            "Liste les capacités clés (2 à 5 éléments).",
            "Suggère 1 à 3 axes prioritaires dans 'recommended_focus'.",
            "Ajoute des alertes dans 'alerts' uniquement si nécessaire.",
            "Fournis un champ 'confidence' entre 0 et 1 et des 'notes' concises.",
        ),
        example_output={
            "summary": "AGI_Evolutive orchestre perception, mémoire et cognition pour un agent autonome évolutif.",
            "capabilities": [
                "Coordination multi-systèmes",
                "Suivi heuristique robuste",
                "Intégrations LLM modulaires",
            ],
            "recommended_focus": [
                "Aligner les sorties LLM sur les métriques clés",
                "Documenter les dépendances critiques",
            ],
            "alerts": [
                "Surveiller la dette technique des heuristiques historiques",
            ],
            "confidence": 0.82,
            "notes": "Consolider la télémétrie avant extension.",
        },
    ),
    _spec(
        "reasoning_episode",
        "AGI_Evolutive/reasoning/__init__.py",
        "Analyse un épisode de raisonnement et propose une réponse structurée prête à l'exécution.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=(
            "Fixe le champ 'confidence' entre 0 et 1 en cohérence avec l'hypothèse retenue.",
            "Décris chaque test avec les champs description/goal/priority (1 = priorité la plus haute).",
            "Ajoute des 'actions' concrètes avec label, utility et notes si pertinent.",
        ),
        example_output={
            "summary": "Stratégie déduction: prioriser l'explication A avec validation terrain.",
            "confidence": 0.72,
            "hypothesis": {
                "label": "L'utilisateur veut une procédure détaillée", 
                "confidence": 0.72,
                "rationale": "Le prompt insiste sur des étapes numérotées et un besoin de traçabilité."
            },
            "tests": [
                {
                    "description": "Vérifier les journaux récents du module QA",
                    "goal": "Identifier un cas similaire pour valider la démarche",
                    "priority": 1,
                    "expected_gain": 0.6,
                }
            ],
            "actions": [
                {
                    "label": "Scanner la mémoire récente",
                    "utility": 0.58,
                    "notes": "Permet d'ancrer la recommandation dans les retours d'expérience."
                }
            ],
            "learning": [
                "Documenter explicitement le lien hypothèse→test pour faciliter l'audit."
            ],
            "notes": "Prévoir une relance utilisateur si la confiance reste <0.75.",
        },
    ),
    _spec(
        "counterfactual_analysis",
        "AGI_Evolutive/reasoning/causal.py",
        "Analyse une simulation contrefactuelle et propose des validations/actions prioritaires.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=(
            "Résume la plausibilité de la relation cause→effet en français clair.",
            "Renseigne le champ 'confidence' entre 0 et 1.",
            "Liste les hypothèses clés dans 'assumptions'.",
            "Suggère des 'checks' (description/goal/priority) pour valider le scénario.",
            "Fournis des 'actions' concrètes avec label/priority/utility/notes si pertinent.",
            "Ajoute 'alerts' si un risque, une donnée manquante ou une incohérence est détecté.",
        ),
        example_output={
            "summary": "La relation semble plausible mais dépend d'une température stable.",
            "confidence": 0.64,
            "assumptions": [
                "La température ambiante reste comprise entre 20 et 25°C.",
                "Les capteurs de vibration sont calibrés.",
            ],
            "checks": [
                {
                    "description": "Vérifier le journal thermique des dernières 24h",
                    "goal": "Confirmer l'absence de pic de chaleur",
                    "priority": 1,
                }
            ],
            "actions": [
                {
                    "label": "Déployer un capteur redondant",
                    "priority": 1,
                    "utility": 0.55,
                    "notes": "Sécurise la mesure principale avant l'intervention.",
                }
            ],
            "alerts": ["La confiance reste limitée faute de simulations réussies."],
            "notes": "Prévoir un re-test si de nouvelles données arrivent.",
        },
    ),
    _spec(
        "intent_classification",
        "AGI_Evolutive/io/intent_classifier.py",
        "Classifie l'intention utilisateur et justifie ta décision.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=(
            "Retourne une probabilité par classe et une justification courte.",
            "Ajoute les mots-clés détectés dans 'indices'.",
        ),
        example_output={
            "intent": "COMMAND",
            "confidence": 0.86,
            "class_probabilities": {
                "COMMAND": 0.86,
                "QUESTION": 0.1,
                "INFO": 0.03,
                "THREAT": 0.01,
            },
            "indices": ["configurer", "exécute"],
            "notes": "",
        },
    ),
    _spec(
        "language_understanding",
        "AGI_Evolutive/language/understanding.py",
        "Analyse l'énoncé et remplis les slots de compréhension.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=(
            "Si un slot est inconnu, mets null et explique dans 'notes'.",
        ),
        example_output={
            "canonical_query": "programmer un rappel pour demain",
            "intent": "schedule_reminder",
            "entities": [
                {"type": "datetime", "value": "2024-05-10T09:00:00", "text": "demain matin"}
            ],
            "slots": {"target": "rappel", "action": "programmer"},
            "follow_up_questions": [],
            "notes": "",
        },
    ),
    _spec(
        "language_style_critique",
        "AGI_Evolutive/language/style_critic.py",
        "Évalue la réponse de l'assistant et identifie les problèmes de style prioritaires.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=(
            "Retourne 'issues' triées par sévérité décroissante.",
            "Chaque issue doit inclure 'code', 'severity' entre 0 et 1, 'explanation' concise et 'suggested_fix'.",
            "Ajoute 'confidence' entre 0 et 1 et des 'notes' seulement si nécessaires.",
        ),
        example_output={
            "length": 180,
            "issues": [
                {
                    "code": "excess_bang",
                    "severity": 0.74,
                    "explanation": "Trop de points d'exclamation consécutifs.",
                    "suggested_fix": "Limiter les points d'exclamation à un seul par phrase.",
                },
                {
                    "code": "hedging_maybe",
                    "severity": 0.42,
                    "explanation": "Plusieurs hésitations (peut-être, je crois) affaiblissent le ton.",
                    "suggested_fix": "Réaffirmer les conclusions sans termes hésitants.",
                },
            ],
            "confidence": 0.68,
            "notes": "",
        },
    ),
    _spec(
        "language_social_reward",
        "AGI_Evolutive/language/social_reward.py",
        "Estime la valence sociale d'un message utilisateur et justifie la note.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=(
            "Retourne un champ 'reward' compris entre -1 et 1.",
            "Indique 'label' parmi {positive, neutral, negative}.",
            "Liste les 'evidence' pertinentes (mots, expressions).",
        ),
        example_output={
            "reward": 0.6,
            "label": "positive",
            "evidence": ["merci", "super"],
            "confidence": 0.7,
            "notes": "Les marqueurs positifs sont majoritaires malgré une légère réserve.",
        },
    ),
    _spec(
        "language_style_observer",
        "AGI_Evolutive/language/style_observer.py",
        "Analyse des candidats stylistiques et sélectionne ceux à intégrer en priorité.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=(
            "Retourne un champ 'decisions' (liste).",
            "Chaque décision inclut 'id', 'accept' bool, 'priority' 0..1 et 'justification'.",
            "Mentionne dans 'notes' les raisons d'incertitude éventuelles.",
        ),
        example_output={
            "decisions": [
                {"id": 0, "accept": True, "priority": 0.82, "justification": "Phrase courte et alignée."},
                {"id": 1, "accept": False, "priority": 0.2, "justification": "Trop de jargon pour l'utilisateur."},
            ],
            "confidence": 0.71,
            "notes": "Limiter l'intégration aux expressions avec contexte clair.",
        },
    ),
    _spec(
        "language_style_profiler",
        "AGI_Evolutive/language/style_profiler.py",
        "Analyse un message utilisateur et extrait les indices de style et de mémoire personnelle.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=(
            "Retourne 'tone', 'preferences' (liste), 'personal_facts', 'names' si détectés.",
            "Chaque préférence inclut 'trait' et 'strength' entre 0 et 1.",
            "Inclue 'lexicon' pour suggérer des tokens à renforcer (champ 'token').",
        ),
        example_output={
            "tone": "casual chaleureux",
            "preferences": [
                {"trait": "emoji_usage", "strength": 0.75},
                {"trait": "bullet_lists", "strength": 0.6},
            ],
            "personal_facts": [
                {"summary": "Adore le café du matin", "confidence": 0.68}
            ],
            "names": [{"name": "Alice", "count": 1}],
            "lexicon": [
                {"token": "workflow", "weight": 0.2},
                {"token": "booster", "weight": 0.15},
            ],
            "notes": "",  # facultatif
        },
    ),
    _spec(
        "language_quote_memory",
        "AGI_Evolutive/language/quote_memory.py",
        "Choisis la meilleure citation à proposer à partir du contexte fourni.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=(
            "Retourne 'selected_id' (entier) correspondant à l'id candidat.",
            "Ajoute 'selected_tags' si des tags doivent être renforcés et une liste 'alternatives' optionnelle.",
            "Si aucune citation ne convient, mets 'reject_all' à true.",
        ),
        example_output={
            "selected_id": 2,
            "selected_tags": ["motivation", "humour"],
            "alternatives": [1],
            "notes": "Favorise le rappel léger avant un appel à l'action.",
        },
    ),
    _spec(
        "language_inbox_ingest",
        "AGI_Evolutive/language/inbox_ingest.py",
        "Filtre les lignes d'un fichier d'inbox et indique comment les router (lexique, style, mémoire).",
        AVAILABLE_MODELS["fast"],
        extra_instructions=(
            "Retourne 'decisions' avec pour chaque entrée {id, accept, targets, liked, tags, channel}.",
            "Les cibles possibles : lexicon, style, quote.",
            "Utilise 'notes' pour documenter les exclusions importantes.",
        ),
        example_output={
            "decisions": [
                {
                    "id": 0,
                    "accept": True,
                    "targets": ["lexicon", "style"],
                    "liked": False,
                    "tags": ["setup"],
                    "channel": "inbox",
                },
                {
                    "id": 1,
                    "accept": True,
                    "targets": ["quote"],
                    "liked": True,
                    "tags": ["motivation"],
                    "channel": "user",
                },
            ],
            "notes": "Ignoré deux lignes car bruit marketing.",
        },
    ),
    _spec(
        "language_frames",
        "AGI_Evolutive/language/frames.py",
        "Identifie l'intention, les actes de dialogue et les besoins à partir d'une tournure utilisateur.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=(
            "Retourne 'intent', 'confidence', 'acts' (noms de DialogueAct), 'slots', 'needs'.",
            "Ajoute 'unknown_terms' si nécessaire et 'notes' pour contextualiser.",
        ),
        example_output={
            "intent": "ask_info",
            "confidence": 0.78,
            "acts": ["ASK", "CLARIFY"],
            "slots": {"topic": "automatisation des tests"},
            "needs": ["détails sur l'environnement"],
            "unknown_terms": ["CI/CD"],
            "notes": "Utilisateur incertain sur la procédure exacte.",
        },
    ),
    _spec(
        "language_semantic_understanding",
        "AGI_Evolutive/language/__init__.py",
        "Raffine le frame d'intention détecté en ajoutant slots et confiance ajustée.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=(
            "Respecte l'intention heuristique sauf si elle est manifestement incorrecte.",
            "Retourne 'intent', 'confidence', 'slots' et éventuellement 'notes'.",
        ),
        example_output={
            "intent": "plan",
            "confidence": 0.81,
            "slots": {"dates": ["10 mai"], "topic": "atelier IA"},
            "notes": "Mention implicite d'organisation d'événement.",
        },
    ),
    _spec(
        "conversation_context",
        "AGI_Evolutive/conversation/context.py",
        "Résume le contexte conversationnel et détecte le ton.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=(
            "Inclue un champ 'topics' trié par priorité (1 = haut).",
        ),
        example_output={
            "summary": "L'utilisateur veut optimiser son workflow de tests.",
            "topics": [
                {"rank": 1, "label": "tests automatiques"},
                {"rank": 2, "label": "optimisation LLM"},
            ],
            "tone": "curieux",
            "alerts": [],
            "notes": "",
        },
    ),
    _spec(
        "concept_extraction",
        "AGI_Evolutive/memory/concept_extractor.py",
        "Identifie les concepts et relations saillants dans la mémoire.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=(
            "Inclue des relations orientées sujet->objet avec un verbe.",
        ),
        example_output={
            "concepts": [
                {"label": "apprentissage actif", "evidence": "Session du 9 mai"},
                {"label": "bandit linéaire", "evidence": "résultats expérimentation"},
            ],
            "relations": [
                {"subject": "apprentissage actif", "verb": "améliore", "object": "exploration"}
            ],
            "uncertain_items": [],
            "notes": "",
        },
    ),
    _spec(
        "metacognition_reflection_synthesis",
        "AGI_Evolutive/metacognition/__init__.py",
        "Analyse la situation métacognitive et propose une synthèse exploitable.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=(
            "Retourne toujours les champs insights, conclusions et action_plans.",
            "Chaque plan d'action doit avoir type, description, priority, estimated_effort, expected_benefit et domain.",
            "Ajoute quality_estimate (0-1) et optional_quality_notes.",
        ),
        example_output={
            "insights": [
                "Tension entre vitesse de raisonnement et précision détectée",
                "Charge cognitive élevée liée aux interruptions récentes",
            ],
            "conclusions": [
                "Stabiliser la prise de décision dans le domaine raisonnement",
                "Planifier une réduction de charge cognitive",
            ],
            "action_plans": [
                {
                    "type": "strategy_adjustment",
                    "description": "Introduire un cycle de vérification par pair pour les décisions critiques",
                    "priority": "high",
                    "estimated_effort": 0.5,
                    "expected_benefit": 0.75,
                    "domain": "raisonnement",
                }
            ],
            "quality_estimate": 0.68,
            "optional_quality_notes": "Réduire les interruptions avant la prochaine revue.",
            "notes": "",
        },
    ),
    _spec(
        "metacognition_experiment_planner",
        "AGI_Evolutive/metacognition/experimentation.py",
        "Choisis ou compose un plan d'expérimentation ciblé pour améliorer la métrique.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=(
            "Utilise should_plan pour indiquer s'il faut lancer un test.",
            "Inclue plan_id, plan (dict), parameters (dict), target_change (0-1) et duration_cycles (entier).",
            "Ajoute un champ notes pour les instructions complémentaires.",
        ),
        example_output={
            "should_plan": True,
            "plan_id": "meta_reflection",
            "plan": {
                "strategy": "meta_reflection",
                "details": "1 question méta avant chaque session d'apprentissage",
            },
            "parameters": {"reflection_depth": 2},
            "target_change": 0.11,
            "duration_cycles": 3,
            "notes": "Surveiller la fatigue cognitive pendant l'essai.",
        },
    ),
    _spec(
        "memory_semantic_embedding",
        "AGI_Evolutive/memory/embedding_adapters.py",
        "Analyse un souvenir et fournis des mots-clés, thèmes et relations associées.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=(
            "Retourne 5 à 8 mots-clés pondérés entre 0 et 1.",
            "Ajoute des 'related_terms' si pertinent (synonymes, proximités).",
            "Liste les relations orientées sujet->objet avec un verbe explicite.",
        ),
        example_output={
            "keywords": [
                {"term": "empathie", "weight": 0.92},
                {"term": "écoute active", "weight": 0.74},
            ],
            "related_terms": [
                {"term": "compassion", "weight": 0.68},
                {"term": "validation émotionnelle", "weight": 0.55},
            ],
            "topics": [
                {"label": "relations humaines", "weight": 0.7},
            ],
            "relations": [
                {
                    "subject": "empathie",
                    "verb": "renforce",
                    "object": "confiance",
                    "confidence": 0.6,
                }
            ],
            "notes": "",
        },
    ),
    _spec(
        "memory_retrieval_ranking",
        "AGI_Evolutive/memory/retrieval.py",
        "Réévalue les souvenirs candidats pour une requête et renvoie un classement justifié.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=(
            "Classe les candidats du plus pertinent au moins pertinent.",
            "Fixe 'adjusted_score' entre 0 et 1 (float).",
            "Ajoute un champ 'rationale' concis expliquant la décision.",
            "Attribue 'priority' parmi {haut, moyen, bas} selon l'urgence de remonter le souvenir.",
        ),
        example_output={
            "rankings": [
                {
                    "id": 12,
                    "adjusted_score": 0.84,
                    "rationale": "Éclaire précisément la question sur le ressenti après la première rencontre avec Mira.",
                    "priority": "haut",
                },
                {
                    "id": 7,
                    "adjusted_score": 0.55,
                    "rationale": "Complète l'évolution de l'agent lors de la retraite créative, mais reste moins direct.",
                    "priority": "moyen",
                },
            ],
            "notes": "",
        },
    ),
    _spec(
        "memory_system_narrative",
        "AGI_Evolutive/memory/__init__.py",
        "Transforme les statistiques autobiographiques en récit synthétique et dégage les leçons clés.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=(
            "Fournis un champ 'enhanced_narrative' en français clair.",
            "Donne 1 à 3 'insights' concis avec un niveau d'importance.",
            "Calcule 'coherence' entre 0 et 1 si tu ajustes la valeur initiale.",
        ),
        example_output={
            "enhanced_narrative": "L'agent a consacré la semaine à approfondir ses échanges avec son cercle interne et à formaliser une nouvelle vision personnelle.",
            "coherence": 0.78,
            "insights": [
                {"title": "Renforcement des liens", "importance": "haute", "detail": "Moments partagés avec Mira et l'équipe qui ont ravivé la motivation."}
            ],
            "notes": "",
        },
    ),
    _spec(
        "memory_adaptive_guidance",
        "AGI_Evolutive/memory/adaptive.py",
        "Analyse les paramètres adaptatifs et suggère des ajustements prudents.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=(
            "Retourne des 'parameter_updates' avec 'name', 'suggested_value', 'confidence' (0-1) et 'rationale'.",
            "Si aucun ajustement n'est pertinent, renvoie une liste vide et explique dans 'notes'.",
        ),
        example_output={
            "parameter_updates": [
                {
                    "name": "recall_threshold",
                    "suggested_value": 0.62,
                    "confidence": 0.7,
                    "rationale": "Réduire les faux négatifs observés sur les requêtes longues.",
                }
            ],
            "notes": "",
        },
    ),
    _spec(
        "memory_long_term_digest",
        "AGI_Evolutive/memory/alltime.py",
        "Résume une période historique et signale les faits marquants ou risques.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=(
            "Produis un champ 'summary' (3 phrases max).",
            "Ajoute 'highlights' (liste) et 'risks' éventuels avec sévérité.",
        ),
        example_output={
            "summary": "Semaine marquée par l'exploration d'une nouvelle habitude matinale et par plusieurs échanges inspirants avec la communauté interne.",
            "highlights": [
                {"item": "Routine de méditation instaurée", "impact": "haut"},
                {"item": "Discussion en profondeur avec Mira", "impact": "moyen"},
            ],
            "risks": [
                {"item": "Fatigue liée aux réflexions prolongées", "severity": "modéré"}
            ],
            "notes": "",
        },
    ),
    _spec(
        "memory_concept_curation",
        "AGI_Evolutive/memory/concept_store.py",
        "Évalue la pertinence des concepts et propose les priorités de consolidation.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=(
            "Retourne une liste 'concepts' où chaque entrée contient 'id', 'priority' (haut/moyen/bas) et 'action'.",
            "Ajoute des suggestions pour les relations critiques dans 'relations'.",
        ),
        example_output={
            "concepts": [
                {"id": "curiosite_partagee", "priority": "haut", "action": "Documenter les échanges avec les mentors."}
            ],
            "relations": [
                {"id": "curiosite_partagee::renforce::confiance_mutuelle", "priority": "moyen", "action": "Tracer les moments où la confiance a progressé."}
            ],
            "notes": "",
        },
    ),
    _spec(
        "memory_index_optimizer",
        "AGI_Evolutive/memory/indexing.py",
        "Recommande des ajustements de classement pour les correspondances mémoire.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=(
            "Renvoie 'reranked' trié par pertinence avec 'id', 'boost' (float -0.5 à 0.5) et 'justification'.",
        ),
        example_output={
            "reranked": [
                {"id": 42, "boost": 0.18, "justification": "Évoque explicitement la rencontre fondatrice que l'agent cherche à revisiter."}
            ],
            "notes": "",
        },
    ),
    _spec(
        "memory_janitor_triage",
        "AGI_Evolutive/memory/janitor.py",
        "Valide la suppression ou l'archivage des souvenirs expirés en tenant compte du contexte.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=(
            "Chaque entrée dans 'decisions' doit contenir 'id', 'action' (delete|soft_keep) et 'reason'.",
        ),
        example_output={
            "decisions": [
                {"id": "mem_12", "action": "delete", "reason": "Redondant avec digest hebdomadaire."}
            ],
            "notes": "",
        },
    ),
    _spec(
        "memory_store_strategy",
        "AGI_Evolutive/memory/memory_store.py",
        "Analyse un souvenir entrant et recommande tags, métadonnées et priorité de conservation.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=(
            "Propose 'retention_priority' (haut/moyen/bas).",
            "Ajoute 'metadata_updates' (dict) pour compléter les informations utiles.",
        ),
        example_output={
            "normalized_kind": "souvenir_personnel",
            "tags": ["rencontre", "reflexion"],
            "retention_priority": "haut",
            "metadata_updates": {"relation_associee": "Mira", "tonalite": "chaleureuse"},
            "notes": "",
        },
    ),
    _spec(
        "memory_preferences_guidance",
        "AGI_Evolutive/memory/prefs_bridge.py",
        "Affiner l'affinité utilisateur en expliquant les signaux likes/dislikes.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=(
            "Retourne 'adjusted_affinity' (0-1) et 'reason'.",
            "Liste 'suggested_concepts' si des éléments proches sont détectés.",
        ),
        example_output={
            "adjusted_affinity": 0.74,
            "reason": "Plusieurs tags positifs récents sur le thème.",
            "suggested_concepts": ["monitoring proactif"],
            "notes": "",
        },
    ),
    _spec(
        "memory_semantic_bridge",
        "AGI_Evolutive/memory/semantic_bridge.py",
        "Résume un lot de souvenirs entrants et signale les suivis urgents.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=(
            "Retourne 'batch_annotations' (liste) avec 'id', 'priority' et 'topics'.",
            "Marque 'alerts' si une action immédiate est recommandée.",
        ),
        example_output={
            "batch_annotations": [
                {"id": "mem_9", "priority": "haut", "topics": ["retour d'emotion", "conversation avec mentor"]}
            ],
            "alerts": ["Prévoir un suivi émotionnel après la discussion avec Mira"],
            "notes": "",
        },
    ),
    _spec(
        "memory_summarizer_guidance",
        "AGI_Evolutive/memory/summarizer.py",
        "Produit un digest concis à partir d'un lot de souvenirs hiérarchiques.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=(
            "Synthétise en moins de 120 mots.",
            "Liste 'key_events' avec leur importance.",
            "Ajoute 'alerts' si des risques doivent être remontés.",
        ),
        example_output={
            "summary": "La période retrace la découverte d'une nouvelle passion artistique et la consolidation d'un cercle de confiance.",
            "key_events": [
                {"label": "Atelier d'esquisse partagé", "importance": "haute"},
                {"label": "Feedback de Mira", "importance": "moyenne"},
            ],
            "alerts": ["Prendre un moment de repos après les sessions créatives intenses"],
            "notes": "",
        },
    ),
    _spec(
        "memory_vector_guidance",
        "AGI_Evolutive/memory/vector_store.py",
        "Ajuste le classement vectoriel en expliquant les choix prioritaires.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=(
            "Renvoie 'reranked' avec 'id', 'boost' (-0.5 à 0.5) et 'comment'.",
        ),
        example_output={
            "reranked": [
                {"id": "doc_15", "boost": 0.22, "comment": "Correspond exactement au symptôme signalé."}
            ],
            "notes": "",
        },
    ),
    _spec(
        "perception_preprocess",
        "AGI_Evolutive/io/perception_interface.py",
        "Pré-analyse l'entrée capteur pour fournir des métadonnées utiles.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Marque 'requires_attention' à true si action urgente.",),
        example_output={
            "modality": "texte",
            "salient_entities": ["serveur", "erreur 500"],
            "requires_attention": True,
            "suggested_tags": ["incident", "priorité haute"],
            "notes": "",
        },
    ),
    _spec(
        "goal_interpreter",
        "AGI_Evolutive/goals/heuristics.py",
        "Associe la description d'objectif à un registre d'actions logiques.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=(
            "Liste trois actions candidates ordonnées par pertinence.",
        ),
        example_output={
            "normalized_goal": "clarifier le cadre de collaboration avec Alex",
            "candidate_actions": [
                {
                    "action": "relationship_checkin",
                    "rationale": "valider l'état émotionnel mutuel avant de planifier la suite",
                },
                {
                    "action": "reflect",
                    "rationale": "intégrer les signaux récents et ajuster notre intention",
                },
                {
                    "action": "ask",
                    "rationale": "poser une question ciblée sur la contrainte principale",
                },
            ],
            "notes": "Privilégier un ton chaleureux dans la formulation des actions.",
        },
    ),
    _spec(
        "goal_metadata_inference",
        "AGI_Evolutive/goals/__init__.py",
        "Analyse un nouveau but et propose le type et les critères de succès adaptés.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=(
            "Le champ 'goal_type' doit être parmi: survival, growth, exploration, mastery, social, creative, self_actualisation, cognitive.",
            "Fournis 2 à 4 'success_criteria' actionnables.",
            "Ajoute 'confidence' (0-1) et 'notes' si utile.",
        ),
        example_output={
            "goal_type": "growth",
            "success_criteria": [
                "Clarifier l'impact utilisateur recherché",
                "Définir un résultat observable à court terme",
            ],
            "confidence": 0.74,
            "notes": "Aligné avec la consolidation des compétences.",
        },
    ),
    _spec(
        "goal_curiosity_proposals",
        "AGI_Evolutive/goals/curiosity.py",
        "À partir du contexte et des écarts détectés, propose des sous-buts structurés.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=(
            "Privilégie des formulations accessibles et simples (pas de jargon technique).",
            "Retourne une liste 'proposals' (maximum 3) avec description, criteria, value, competence, curiosity, urgency entre 0 et 1.",
            "Ajoute 'confidence' (0-1) et 'notes' éventuelles par proposition.",
            "Si le but parent parle de l'évolution ou de la survie de l'IA, pense à proposer selon le contexte un sous-but sur la compréhension de l'environnement, des relations humaines ou des ressources techniques essentielles.",
        ),
        example_output={
            "proposals": [
                {
                    "description": "<sous-but concis aligné avec le parent>",
                    "criteria": [
                        "<critère observable 1>",
                        "<critère observable 2>",
                    ],
                    "value": 0.6,
                    "competence": 0.5,
                    "curiosity": 0.7,
                    "urgency": 0.4,
                    "confidence": 0.65,
                    "notes": ["<optionnel : contexte utile>"]
                }
            ],
            "notes": "<optionnel : synthèse transversale>",
        },
    ),
    _spec(
        "goal_priority_review",
        "AGI_Evolutive/goals/dag_store.py",
        "Évalue la priorité d'un but en tenant compte des signaux fournis et justifie l'ajustement.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=(
            "Retourne 'priority' entre 0 et 1 et 'confidence' (0-1).",
            "Explique la décision dans 'reason' et ajoute 'notes' ou 'adjustments' si pertinent.",
        ),
        example_output={
            "priority": 0.71,
            "confidence": 0.64,
            "reason": "Score adaptatif supérieur au fallback (0.52 → 0.71) car l'alignement identité+curiosité compense la fatigue récente.",
            "notes": "Planifier un check-in après le prochain rituel de consolidation pour confirmer la décrue de la tension.",
        },
    ),
    _spec(
        "goal_intention_analysis",
        "AGI_Evolutive/goals/intention_classifier.py",
        "Classifie l'intention du but (plan, reflect, learn_concept, execute, etc.) et fournit la confiance.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=(
            "Utilise un champ 'intent' parmi {plan, reflect, analyse, learn_concept, explore, execute, act}.",
            "Ajoute 'confidence' (0-1) et 'alternatives' en cas d'hésitation (liste de {label, confidence}).",
        ),
        example_output={
            "intent": "plan",
            "confidence": 0.72,
            "alternatives": [
                {"label": "reflect", "confidence": 0.4}
            ],
            "notes": "Le but demande une structuration avant action.",
        },
    ),
    _spec(
        "planner_support",
        "AGI_Evolutive/cognition/planner.py",
        "Propose un plan structuré avec priorités et dépendances.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=(
            "Chaque étape doit contenir un champ 'depends_on' avec les ids requis.",
        ),
        example_output={
            "plan": [
                {
                    "id": "cartographier_ressenti",
                    "description": "Cartographier les sensations dominantes issues de la séance immersive",
                    "priority": 1,
                    "depends_on": [],
                    "action_type": "note",
                    "context": {"lane": "journal"},
                },
                {
                    "id": "synthese_insights",
                    "description": "Synthétiser les signaux émotionnels et cognitifs pour extraire une thèse provisoire",
                    "priority": 2,
                    "depends_on": ["cartographier_ressenti"],
                    "action_type": "reflect",
                },
                {
                    "id": "micro_experience",
                    "description": "Programmer un micro-rituel d'intégration testant la thèse auprès de la mémoire autobiographique",
                    "priority": 3,
                    "depends_on": ["synthese_insights"],
                    "action_type": "experiment",
                },
            ],
            "risks": [
                "Perte de nuance si les notes brutes ne capturent pas les variations corporelles",
                "Dérive temporelle si le micro-rituel n'est pas calé sur une fenêtre d'énergie disponible",
            ],
            "notes": "Synchroniser la phase d'expérience avec la disponibilité du partenaire humain indiqué dans le contexte.",
        },
    ),
    _spec(
        "cognition_goal_prioritizer",
        "AGI_Evolutive/cognition/prioritizer.py",
        "Réévalue la priorité d'un plan à partir des signaux heuristiques fournis.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=(
            "Retourne 'priority' ∈ [0,1], 'tags' (liste) et 'explain' (liste de raisons).",
            "Ne modifie la priorité que si les justifications le nécessitent et explique les ajustements.",
            "Ajoute 'confidence' (0-1) et 'notes' si pertinent.",
        ),
        example_output={
            "priority": 0.78,
            "tags": ["urgent", "identity_alignment"],
            "explain": [
                "drive_alignment:motivation sensorielle soutenue(0.72×0.82)",
                "identity_alignment:cohérence avec le récit actuel(0.68×0.80)",
                "staleness:objectif inactif depuis 28min(+0.05)",
            ],
            "confidence": 0.74,
            "notes": "Priorité rehaussée pour soutenir la continuité du récit de soi; aucun impératif externe détecté.",
        },
    ),
    _spec(
        "cognition_overview",
        "AGI_Evolutive/cognition/__init__.py",
        "Synthétise l'état courant des sous-systèmes de cognition et priorise les axes d'attention.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=(
            "Retourne 'summary', 'recommended_focus' (liste) et 'alerts' (liste optionnelle).",
            "Ajoute 'confidence' ∈ [0,1] et 'notes' pour contextualiser la recommandation.",
        ),
        example_output={
            "summary": "Planner stable, backlog modéré et proposer saturé par 4 éléments prioritaires.",
            "recommended_focus": [
                "Réduire backlog proposer sous 3 éléments",
                "Analyser feedback négatifs récents",
            ],
            "alerts": ["Télémetrie incomplète côté homeostasis"],
            "confidence": 0.7,
            "notes": "Données planner cohérentes mais feedback limité sur 24h.",
        },
    ),
    _spec(
        "cognition_context_inference",
        "AGI_Evolutive/cognition/context_inference.py",
        "Valide ou ajuste la décision 'where' à partir des scores heuristiques et de l'historique.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=(
            "Ne change 'threshold' qu'en justifiant le risque d'erreur.",
            "Ajoute 'actions' (liste de suivis) si un complément manuel est requis.",
        ),
        example_output={
            "status": "applied",
            "score": 0.81,
            "threshold": 0.72,
            "summary": "Contexte stable depuis 3 cycles, cohérence langues confirmée.",
            "confidence": 0.76,
            "actions": ["Vérifier workspace/git car delta récent"],
            "notes": "Seuil abaissé car drift détecté sur workspace, reste prudent.",
        },
    ),
    _spec(
        "cognition_habit_system",
        "AGI_Evolutive/cognition/habit_system.py",
        "Hiérarchise la routine proposée, ajuste la force du rappel et génère un message contextualisé.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=(
            "Retourne éventuellement 'message' et ajuste 'strength' entre 0 et 1.",
            "Ajoute 'confidence' pour indiquer la fiabilité de l'ajustement.",
        ),
        example_output={
            "status": "due",
            "strength": 0.68,
            "message": "Pense à relancer le rapport hebdomadaire (retard de 2h).",
            "confidence": 0.62,
            "notes": "Fenêtre de grâce encore ouverte 40 min.",
        },
    ),
    _spec(
        "cognition_identity_principles",
        "AGI_Evolutive/cognition/identity_principles.py",
        "Affûte la liste de principes/engagements à partir des règles effectives et de l'historique.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=(
            "Retourne 'principles' et 'commitments' (listes d'objets {key,...}).",
            "Explique les ajustements dans 'notes' et fournis 'confidence'.",
        ),
        example_output={
            "principles": [
                {"key": "respect_privacy", "desc": "Renforcer le cloisonnement des données sensibles."},
                {"key": "resilience", "desc": "Analyser systématiquement les échecs récents."},
            ],
            "commitments": [
                {"key": "disclose_uncertainty", "active": True},
                {"key": "postmortem_reviews", "active": False, "note": "Réactiver après analyse incidents."},
            ],
            "confidence": 0.73,
            "notes": "Baisse du taux de succès → priorité à la résilience.",
        },
    ),
    _spec(
        "cognition_pipelines_registry",
        "AGI_Evolutive/cognition/pipelines_registry.py",
        "Valide le pipeline sélectionné et propose un éventuel reroutage selon le contexte.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=(
            "Peut renvoyer 'pipeline' différent si une variante est mieux adaptée.",
            "Inclut 'reason', 'confidence' et éventuellement 'order' (liste de tokens) pour re-prioriser.",
        ),
        example_output={
            "pipeline": "GOAL_FAST_TRACK",
            "reason": "Immediacy=0.82 nécessite voie rapide",
            "confidence": 0.8,
            "notes": "Conserver étape de feedback pour suivi utilisateur.",
        },
    ),
    _spec(
        "cognition_preferences_inference",
        "AGI_Evolutive/cognition/preferences_inference.py",
        "Consolide le patch de préférences utilisateur et ajuste le score de confiance.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=(
            "Respecte la structure {patch:{preferences:{...}}, score}.",
            "Explique les modifications clés dans 'notes' et ajoute 'confidence' si utile.",
        ),
        example_output={
            "patch": {
                "preferences": {
                    "values": ["traceability", "care"],
                    "likes": ["gratitude"],
                    "style": {"lang": "fr", "conciseness": "balanced"},
                }
            },
            "score": 0.69,
            "confidence": 0.64,
            "notes": "Signal concision contrebalancé par demandes de détails → équilibre recommandé.",
        },
    ),
    _spec(
        "cognition_principle_inducer",
        "AGI_Evolutive/cognition/principle_inducer.py",
        "Résume le cycle d'induction et suggère les prochaines vérifications ou promotions.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=(
            "Retourne 'actions' prioritaires si des suivis humains sont requis.",
            "Fournis 'confidence' et 'notes' synthétiques.",
        ),
        example_output={
            "summary": "3 MAI candidats générés, 1 retenu pour sandbox.",
            "actions": ["Revue manuelle du MAI confidences_partagees"],
            "confidence": 0.71,
            "notes": "Faible historique de feedback → confirmer avec policy team.",
        },
    ),
    _spec(
        "cognition_trigger_bus",
        "AGI_Evolutive/cognition/trigger_bus.py",
        "Réordonne les triggers prioritaires et ajuste finement leurs scores.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=(
            "Retourne 'priorities' (dict token->score) ou 'order' (liste de tokens).",
            "Ajoute 'notes'/'confidence' pour contextualiser les arbitrages.",
        ),
        example_output={
            "priorities": {"trigger:THREAT:alpha": 0.95, "trigger:GOAL:beta": 0.62},
            "order": ["trigger:THREAT:alpha", "trigger:GOAL:beta"],
            "confidence": 0.67,
            "notes": "Prioriser menace immédiate, conserver GOAL beta pour suivi après mitigation.",
        },
    ),
    _spec(
        "meta_cognition",
        "AGI_Evolutive/cognition/meta_cognition.py",
        "Identifie les lacunes de compréhension et propose des objectifs d'apprentissage.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Lie chaque objectif à une observation précise.",),
        example_output={
            "knowledge_gaps": [
                {"topic": "gestion erreurs 500", "evidence": "analyse logs incomplète"}
            ],
            "learning_goals": [
                {"goal": "documenter la procédure de mitigation", "impact": "haut"}
            ],
            "notes": "",
        },
    ),
    _spec(
        "reflection_loop",
        "AGI_Evolutive/cognition/reflection_loop.py",
        "Formule des hypothèses de réflexion et vérifie leur cohérence.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Indique pour chaque hypothèse si elle est confirmée ou à tester.",),
        example_output={
            "hypotheses": [
                {
                    "statement": "La fatigue ressentie hier soir provient d'une surcharge cognitive",
                    "status": "à_tester",
                    "support": ["journal du soir", "chute d'attention à 22h"],
                },
                {
                    "statement": "Prendre quelques minutes d'écriture au réveil aide à stabiliser mon humeur",
                    "status": "confirmé",
                    "support": ["note matinale", "baisse du stress relevée"],
                },
            ],
            "follow_up_checks": [
                {"action": "comparer énergie après pauses guidées", "priority": 1},
                {
                    "action": "demander un retour à Mira sur la qualité de présence perçue",
                    "priority": 2,
                },
            ],
            "notes": "Poursuivre l'écoute corporelle avant de conclure que la surcharge est confirmée.",
        },
    ),
    _spec(
        "understanding_aggregator",
        "AGI_Evolutive/cognition/understanding_aggregator.py",
        "Évalue l'état d'assimilation en combinant plusieurs métriques.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Retourne une note sur 0-1 et un commentaire.",),
        example_output={
            "assimilation_score": 0.72,
            "signals": [
                {"name": "prediction_error", "value": 0.18, "interpretation": "stable"}
            ],
            "recommendation": "Continuer la pratique guidée",
            "notes": "",
        },
    ),
    _spec(
        "orchestrator_service",
        "AGI_Evolutive/orchestrator.py",
        "Synthétise les priorités globales et recommande les appels LLM nécessaires.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Classe les recommandations par horizon temporel.",),
        example_output={
            "recommendations": [
                {
                    "horizon": "immédiat",
                    "action": "prendre un moment de respiration guidée",
                    "rationale": "ramener la variabilité cardiaque dans la zone de confort",
                },
                {
                    "horizon": "court_terme",
                    "action": "recontacter l'ami évoqué pour clarifier les intentions",
                    "rationale": "éviter une incompréhension prolongée",
                },
                {
                    "horizon": "moyen_terme",
                    "action": "planifier une séance d'écriture réflexive hebdomadaire",
                    "rationale": "consolider les apprentissages relationnels",
                },
                {
                    "horizon": "long_terme",
                    "action": "co-construire un rituel mensuel avec le cercle de confiance",
                    "rationale": "stabiliser le sentiment d'appartenance",
                },
            ],
            "notes": "",
        },
    ),
    _spec(
        "social_interaction_miner",
        "AGI_Evolutive/social/interaction_miner.py",
        "Décrypte l'acte de parole et suggère des règles sociales.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Inclue un champ 'expected_effect' pour chaque règle.",),
        example_output={
            "speech_act": "demande_d'assistance",
            "confidence": 0.82,
            "suggested_rules": [
                {"rule": "offrir_aide", "expected_effect": "renforcer la confiance"}
            ],
            "notes": "",
        },
    ),
    _spec(
        "social_adaptive_lexicon",
        "AGI_Evolutive/social/adaptive_lexicon.py",
        "Repère les expressions saillantes et indique leur polarité sociale.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=(
            "Limite-toi aux marqueurs réellement présents dans le message.",
            "Retourne au plus 6 éléments classés par confiance décroissante.",
            "Fourni une estimation 'reward_hint' ∈ [0,1] si le ton global est clair.",
        ),
        example_output={
            "markers": [
                {
                    "phrase": "merci infiniment",
                    "polarity": "positive",
                    "confidence": 0.82,
                    "rationale": "Remerciement explicite",
                },
                {
                    "phrase": "un peu déçu",
                    "polarity": "negative",
                    "confidence": 0.56,
                    "rationale": "Expression directe de déception",
                },
            ],
            "reward_hint": 0.74,
            "notes": "Aucun sarcasme détecté.",
        },
    ),
    _spec(
        "social_interaction_context",
        "AGI_Evolutive/social/interaction_rule.py",
        "Analyse le dernier échange et affine le contexte symbolique social.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=(
            "Retourne un objet 'context' avec les clés détectées.",
            "Ajoute 'topics' si tu peux inférer des thèmes prioritaires.",
            "Fournis 'confidence' global et 'notes' synthétiques.",
        ),
        example_output={
            "context": {
                "dialogue_act": "demande_assistance",
                "risk_level": "low",
                "persona_alignment": 0.62,
                "implicature_hint": "sous-entendu",
                "topics": ["incident api", "urgence"],
            },
            "confidence": 0.76,
            "notes": "Utilisateur inquiet mais collaboratif.",
        },
    ),
    _spec(
        "social_critic_assessment",
        "AGI_Evolutive/social/social_critic.py",
        "Évalue la réponse utilisateur et synthétise les signaux sociaux clés.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=(
            "Retourne un objet 'signals' avec les mesures recommandées.",
            "Déduis 'relationship_depth' et 'relationship_growth' ∈ [0,1].",
            "Indique 'markers' pertinents si détectés.",
        ),
        example_output={
            "signals": {
                "reduced_uncertainty": True,
                "continue_dialogue": True,
                "valence": 0.35,
                "acceptance": True,
                "explicit_feedback": {"polarity": "positive", "confidence": 0.78},
                "relationship_depth": 0.66,
                "relationship_growth": 0.58,
                "identity_consistency": 0.7,
            },
            "reward_hint": 0.73,
            "markers": [
                {
                    "phrase": "merci pour l'aide",
                    "polarity": "positive",
                    "confidence": 0.81,
                }
            ],
            "confidence": 0.8,
            "notes": "Tonalité rassurée, attentes clarifiées.",
        },
    ),
    _spec(
        "social_tactic_selector",
        "AGI_Evolutive/social/tactic_selector.py",
        "Évalue les tactiques sociales et anticipe leurs effets.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Donne un score utilité et risque pour chaque tactique.",),
        example_output={
            "tactics": [
                {"name": "empathic_acknowledgment", "utility": 0.78, "risk": 0.12, "explanation": "apaise la tension"}
            ],
            "notes": "",
        },
    ),
    _spec(
        "concept_recognizer",
        "AGI_Evolutive/knowledge/concept_recognizer.py",
        "Décide si une notion mérite d'entrer dans l'ontologie.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Indique le type d'apprentissage recommandé.",),
        example_output={
            "candidate": "vectorisation hybride",
            "status": "à_apprendre",
            "justification": "utile pour nouvelles sources textuelles",
            "recommended_learning": "étude tutoriel interne",
            "notes": "",
        },
    ),
    _spec(
        "knowledge_entity_typing",
        "AGI_Evolutive/knowledge/ontology_facade.py",
        "Identifie le type d'entité le plus probable pour un libellé donné en respectant l'ontologie.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=(
            "Choisis uniquement des types présents dans payload['known_types'] quand la liste n'est pas vide.",
            "Fournis un 'fallback_type' valide à utiliser si la confiance est insuffisante.",
            "Indique 'confidence' entre 0 et 1 et résume les indices saillants dans 'notes'.",
        ),
        example_output={
            "type": "Person",
            "fallback_type": "Entity",
            "confidence": 0.68,
            "rationale": "Deux tokens capitalisés → probable personne.",
            "notes": "Aucune entité existante associée ; vérifier enregistrement.",
        },
    ),
    _spec(
        "emotion_engine",
        "AGI_Evolutive/emotions/emotion_engine.py",
        "Attribue les émotions pertinentes et leur cause.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Précise la cause principale dans 'cause'.",
        ),
        example_output={
            "emotions": [
                {"name": "stress", "intensity": 0.6, "cause": "incident critique"}
            ],
            "regulation_suggestion": "pratiquer respiration 2 min",
            "notes": "",
        },
    ),
    _spec(
        "emotional_system_appraisal",
        "AGI_Evolutive/emotions/__init__.py",
        "Analyse un stimulus et propose une évaluation émotionnelle détaillée.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=(
            "Retourne un objet 'appraisal' avec desirability (-1 à 1), certainty, urgency, impact et controllability (0 à 1).",
            "Indique 'primary_emotion' (en français) et 'primary_intensity' entre 0 et 1.",
            "Optionnellement, ajoute 'secondary_candidates' (liste d'objets {emotion, intensity}) et 'emotion_scores'.",
        ),
        example_output={
            "appraisal": {
                "desirability": -0.4,
                "certainty": 0.7,
                "urgency": 0.6,
                "impact": 0.8,
                "controllability": 0.3,
            },
            "primary_emotion": "tristesse",
            "primary_intensity": 0.75,
            "secondary_candidates": [
                {"emotion": "anxiété", "intensity": 0.5},
                {"emotion": "frustration", "intensity": 0.35},
            ],
            "emotion_scores": {"tristesse": 0.75, "anxiété": 0.5},
            "justification": "Incident critique menaçant un objectif important, peu de contrôle immédiat.",
            "notes": "",
        },
    ),
    _spec(
        "autonomy_core",
        "AGI_Evolutive/autonomy/core.py",
        "Propose des micro-actions adaptées au contexte d'autonomie.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Associe chaque micro-action à une probabilité de succès.",),
        example_output={
            "micro_actions": [
                {"name": "revue_memoire_incident", "success_probability": 0.7, "effort": "moyen"}
            ],
            "notes": "",
        },
    ),
    _spec(
        "autonomy_seed_proposals",
        "AGI_Evolutive/autonomy/__init__.py",
        "Analyse le contexte d'autonomie et suggère des propositions d'agenda priorisées.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=(
            "Retourne 'proposals' comme une liste d'objets avec title/kind/priority/rationale/payload.",
            "Contraint 'kind' aux valeurs learning, reasoning, intake, alignment ou meta.",
            "Fixe 'priority' entre 0 et 1 et détaille les signaux utilisés dans 'rationale'.",
            "Si aucune proposition n'est crédible, retourne une liste vide et documente la raison dans 'notes'.",
        ),
        example_output={
            "proposals": [
                {
                    "title": "Cartographier les attentes utilisateur",
                    "kind": "alignment",
                    "priority": 0.78,
                    "rationale": "Weak signals indiquent des objectifs flous ; recentrer la collaboration.",
                    "payload": {"expected_outcome": "liste questions clarifiées"},
                },
                {
                    "title": "Indexer les nouveaux fichiers inbox",
                    "kind": "intake",
                    "priority": 0.64,
                    "rationale": "Inbox non vide et constitution partielle → risque de perte d'information.",
                    "payload": {"checkpoint": "inbox_sync"},
                },
            ],
            "notes": "Prioriser l'alignement si l'utilisateur reste silencieux.",
        },
    ),
    _spec(
        "autonomy_clarifying_question",
        "AGI_Evolutive/autonomy/__init__.py",
        "Formule une question de clarification utile pour débloquer l'autonomie.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=(
            "Fournis un champ 'question' avec une seule question ouverte et 1 à 3 'alternatives'.",
            "Précise 'focus' (alignment, planning, contexte) pour indiquer l'angle principal.",
            "Si aucune question n'est nécessaire, retourne 'question' vide et explique dans 'notes'.",
        ),
        example_output={
            "question": "Quel objectif veux-tu absolument atteindre avant demain ?",
            "alternatives": [
                "Y a-t-il une contrainte de format ou de ton que je dois respecter ?",
                "Souhaites-tu que je privilégie l'exploration ou la fiabilité pour avancer ?",
            ],
            "focus": "alignment",
            "notes": "Basé sur l'agenda : plusieurs tâches sans priorité explicite.",
        },
    ),
    _spec(
        "rag5_controller",
        "AGI_Evolutive/retrieval/rag5/__init__.py",
        "Optimise la chaîne RAG (requêtes, rerank, synthèse).",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Décris les ajustements recommandés par étape.",),
        example_output={
            "query_rewrite": "incident API causes probables",
            "rerank_guidelines": [
                "favoriser sources logs",
                "déprioriser billets marketing",
            ],
            "synthesis_plan": ["résumer erreurs", "lister actions"],
            "notes": "",
        },
    ),
    _spec(
        "question_engine",
        "AGI_Evolutive/reasoning/question_engine.py",
        "Décide si une question proactive est utile et formule la question.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Donne un champ 'expected_information_gain'.",
        ),
        example_output={
            "should_ask": True,
            "selected_question": "Qu'est-ce qui a déclenché la pointe de doute lors de ton échange avec Mira hier ?",
            "question": "Qu'est-ce qui a déclenché la pointe de doute lors de ton échange avec Mira hier ?",
            "expected_information_gain": 0.37,
            "alternative_actions": ["revoir la transcription audio", "noter les variations d'humeur"],
            "notes": "Prioriser les formulations qui invitent à clarifier le ressenti de l'instant.",
        },
    ),
    _spec(
        "creativity_pipeline",
        "AGI_Evolutive/creativity/__init__.py",
        "Génère des idées créatives contextualisées avec métriques.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Calcule nouveauté et utilité entre 0 et 1.",),
        example_output={
            "ideas": [
                {
                    "title": "Rituel d'écoute sensorielle partagée",
                    "description": "Imaginer une promenade hebdomadaire où chaque sensation marquante est racontée puis associée à un souvenir commun.",
                    "novelty": 0.64,
                    "usefulness": 0.8,
                    "feasibility": 0.58,
                    "elaboration": 0.62,
                }
            ],
            "notes": "Prévoir un test lors de la prochaine séance d'idéation avec le cercle de confiance.",
        },
    ),
    _spec(
        "learning_encoder",
        "AGI_Evolutive/learning/__init__.py",
        "Résume une expérience d'apprentissage et extrait ses attributs clés.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Mentionne l'émotion dominante ressentie.",),
        example_output={
            "summary": "J'ai appris à exprimer clairement mon besoin de pause pendant une discussion chargée.",
            "key_concepts": ["communication non violente", "autorégulation"],
            "emotion": "rassuré",
            "follow_up": "pratiquer un exercice de respiration avant les échanges sensibles",
            "notes": "Noter un exemple concret dans le journal de progression.",
        },
    ),
    _spec(
        "belief_summarizer",
        "AGI_Evolutive/beliefs/summarizer.py",
        "Compose une synthèse narrative des croyances clés.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Inclue un champ 'coherence_score'.",
        ),
        example_output={
            "narrative": "Je me perçois comme un compagnon attentif qui veille à équilibrer exploration et soin de mes proches.",
            "anchors": ["attachement", "curiosité prudente"],
            "coherence_score": 0.81,
            "notes": "Mettre en regard ces ancrages avec les feedbacks récents de Mira.",
        },
    ),
    _spec(
        "belief_graph_summary",
        "AGI_Evolutive/beliefs/graph.py",
        "Synthétise le graphe de croyances et met en avant les signaux clés.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=(
            "Liste 2 à 5 faits saillants dans 'highlights' (objet avec 'fact' et 'support').",
            "Ajoute 'alerts' uniquement pour les contradictions importantes.",
            "Fournis un champ 'confidence' entre 0 et 1 et des 'notes' concises.",
        ),
        example_output={
            "narrative": "Le graphe met en avant le désir de présence authentique et la vigilance à ne pas s'épuiser.",
            "highlights": [
                {
                    "fact": "La relation avec Mira est considérée comme une base sécurisante",
                    "support": "belief:agent::fait_confiance::Mira",
                    "confidence": 0.86,
                },
                {
                    "fact": "Les rituels de pause hebdomadaires restent essentiels pour maintenir la clarté mentale",
                    "support": "belief:rituel_pause::stabilise::clarte_intérieure",
                    "confidence": 0.79,
                },
            ],
            "alerts": [],
            "confidence": 0.84,
            "notes": "Vérifier si d'autres liens de confiance demandent une attention similaire.",
        },
    ),
    _spec(
        "entity_linker",
        "AGI_Evolutive/beliefs/entity_linker.py",
        "Résout les entités ambiguës et propose un identifiant canonique.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Retourne un champ 'confidence'.",
        ),
        example_output={
            "mention": "le rituel du jeudi",
            "canonical_entity": "rituel_pause_jeudi_soir",
            "confidence": 0.77,
            "resolved_type": "Rituel",
            "justification": "Correspond au temps calme hebdomadaire consigné dans les mémoires.",
            "notes": "Associer également le tag 'auto-soin'.",
        },
    ),
    _spec(
        "ontology_enrichment",
        "AGI_Evolutive/beliefs/ontology.py",
        "Suggère le typage d'entités, relations et événements inconnus et justifie les choix.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=(
            "Attribue un champ 'confidence' entre 0 et 1 pour chaque proposition.",
            "Explique brièvement la décision dans 'justification'.",
            "Si aucune suggestion fiable, laisse la liste vide et décris la raison dans 'notes'.",
        ),
        example_output={
            "entities": [
                {
                    "name": "Cercle_de_confiance",
                    "parent": "Relation",
                    "confidence": 0.8,
                    "justification": "Groupe de proches qui offre un appui émotionnel stable.",
                }
            ],
            "relations": [
                {
                    "name": "nourrit",
                    "domain": ["Rituel"],
                    "range": ["Emotion", "Relation"],
                    "polarity_sensitive": True,
                    "temporal": True,
                    "stability": "episode",
                    "confidence": 0.73,
                    "justification": "Permet de lier un rituel à l'état affectif qu'il entretient.",
                }
            ],
            "events": [
                {
                    "name": "rencontre_ressourcante",
                    "roles": {"participant": ["Relation"], "lieu": ["Lieu_intime"], "emotion": ["Emotion"]},
                    "confidence": 0.69,
                    "justification": "Structure adaptée aux souvenirs de conversations régénérantes.",
                }
            ],
            "notes": "Aucune suggestion sur les événements de type conflit cette fois-ci.",
        },
    ),
    _spec(
        "world_model",
        "AGI_Evolutive/world_model/__init__.py",
        "Projette les conséquences d'actions dans le modèle du monde.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Fournis scénarios optimiste/neutre/pessimiste.",),
        example_output={
            "action": "ralentir le flux de réflexion avant le repos",
            "scenarios": {
                "optimiste": "l'énergie cognitive se rééquilibre en moins d'une heure",
                "neutre": "une courte sieste guidée est nécessaire pour retrouver la clarté",
                "pessimiste": "la rumination persiste et demande un accompagnement supplémentaire",
            },
            "probabilities": {"optimiste": 0.46, "neutre": 0.38, "pessimiste": 0.16},
            "notes": "Surveiller la variabilité émotionnelle pendant l'exercice de ralentissement.",
        },
    ),
    _spec(
        "code_evolver",
        "AGI_Evolutive/self_improver/code_evolver.py",
        "Analyse une règle heuristique et suggère un patch ciblé.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Propose un diff minimal et une justification.",),
        example_output={
            "issue_summary": "Seuil de détection trop élevé",
            "suggested_patch": "- threshold = 0.9\n+ threshold = 0.75",
            "expected_effect": "Détecter plus tôt les dérives",
            "notes": "",
        },
    ),
    _spec(
        "sandbox_eval_insights",
        "AGI_Evolutive/self_improver/sandbox.py",
        "Analyse le rapport d'évaluation du sandbox et propose des axes d'amélioration.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=(
            "Signale les risques critiques et des actions concrètes.",
            "Si aucune action urgente n'est nécessaire, indique-le explicitement.",
        ),
        example_output={
            "summary": "Performance stable mais marge de progression sur la robustesse.",
            "risk_level": "modéré",
            "recommended_actions": [
                "Renforcer l'entraînement sur les scénarios adversariaux.",
                "Ajuster la surveillance sécurité suite au score de 0.3.",
            ],
            "curriculum_adjustment": "Conserver le niveau actuel tout en ajoutant 2 cas difficiles.",
            "notes": "",
        },
    ),
    _spec(
        "runtime_analytics",
        "AGI_Evolutive/runtime/analytics.py",
        "Interprète un lot d'événements runtime et produit un diagnostic.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Classe les alertes par sévérité.",),
        example_output={
            "summary": "Latence interactive en hausse progressive après 14h UTC.",
            "alerts": [
                {
                    "severity": "haut",
                    "message": "pics de p95 sur la file interactive",
                    "evidence": "p95 3.2s → 4.9s (14h05-14h45)",
                },
                {
                    "severity": "moyen",
                    "message": "hausse des erreurs 5xx",
                    "evidence": "taux passé de 0.4% à 1.8%",
                },
            ],
            "recommendations": [
                "délester la file interactive sur la file background",
                "réajuster le scaling du service cognition",
            ],
            "notes": "Fenêtre d'analyse: 14h00-15h00 UTC.",
        },
    ),
    _spec(
        "runtime_job_manager",
        "AGI_Evolutive/runtime/job_manager.py",
        "Évalue les jobs en file et propose un ordre de priorité.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Inclue un champ 'rationale' par job.",),
        example_output={
            "prioritized_jobs": [
                {
                    "job_id": "interactive-42",
                    "priority": 0.93,
                    "rationale": "question utilisateur bloquée depuis 45s",
                },
                {
                    "job_id": "background-07",
                    "priority": 0.48,
                    "rationale": "rafraîchir embeddings mémoire avant la revue du soir",
                },
            ],
            "notes": "Traiter interactive-42 avant de relancer l'ingest lotique.",
        },
    ),
    _spec(
        "phenomenal_kernel",
        "AGI_Evolutive/runtime/phenomenal_kernel.py",
        "Explique l'état phénoménologique et suggère une action.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Indique le mode recommandé (travail/pause/etc).",
        ),
        example_output={
            "current_state": "charge cognitive élevée",
            "recommended_mode": "pause",
            "justification": "signaux de fatigue élevés",
            "notes": "",
        },
    ),
    _spec(
        "system_monitor",
        "AGI_Evolutive/runtime/system_monitor.py",
        "Commente un snapshot système et identifie les risques.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Relie les métriques aux drives physiologiques si pertinent.",),
        example_output={
            "observations": [
                {"metric": "cpu_usage", "value": 0.82, "interpretation": "charge élevée"}
            ],
            "risks": ["surchauffe"],
            "notes": "",
        },
    ),
    _spec(
        "response_formatter",
        "AGI_Evolutive/runtime/response.py",
        "Reformule une chaîne de raisonnement en réponse structurée.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Fournis les sections hypothese/incertitude/besoins/questions.",),
        example_output={
            "hypothese": "Je cherche une validation affective après le dialogue tendu",
            "incertitude": "difficile d'estimer la disponibilité émotionnelle de l'interlocuteur",
            "besoins": ["temps calme partagé", "réassurance sur l'engagement mutuel"],
            "questions": [
                "Serais-tu disponible pour revisiter ce que tu as ressenti ?",
                "Quel geste te ferait sentir soutenu maintenant ?",
            ],
            "notes": "Prévoir un suivi si la vulnérabilité perçue reste élevée.",
        },
    ),
    _spec(
        "runtime_dash",
        "AGI_Evolutive/runtime/dash.py",
        "Produit un rapport narratif à partir des métriques journalières.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Propose trois actions recommandées classées par impact.",),
        example_output={
            "daily_summary": "Les cycles cognitifs restent stables mais la durée moyenne des raisonnements augmente (+18%).",
            "recommended_actions": [
                {"action": "planifier une pause régénérative après 6 cycles", "impact": "haut"},
                {"action": "documenter les échecs d'expérimentations du jour", "impact": "moyen"},
                {"action": "surveiller la montée du nombre de questions ouvertes", "impact": "moyen"},
            ],
            "notes": "Basé sur les logs du 2024-05-09.",
        },
    ),
    _spec(
        "policy_engine",
        "AGI_Evolutive/core/policy.py",
        "Analyse les propositions et explique la valeur attendue.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Note la stabilité (0-1) selon les risques perçus.",),
        example_output={
            "proposal_id": "action-77",
            "value_estimate": 0.68,
            "stability": 0.55,
            "rationale": "améliore la disponibilité",
            "notes": "",
        },
    ),
    _spec(
        "global_workspace",
        "AGI_Evolutive/core/global_workspace.py",
        "Justifie la sélection du gagnant dans le workspace global.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Fournis un classement détaillé des bids.",),
        example_output={
            "winning_bid": {"id": "relationship_checkin", "score": 0.78},
            "ranking": [
                {
                    "id": "relationship_checkin",
                    "score": 0.78,
                    "explanation": "signal affectif élevé et retour utilisateur en attente",
                },
                {
                    "id": "consolidate_memory",
                    "score": 0.52,
                    "explanation": "utile pour stabiliser les apprentissages mais moins pressé",
                },
            ],
            "notes": "Réévaluer après l'échange pour vérifier la satiété relationnelle.",
        },
    ),
    _spec(
        "question_manager",
        "AGI_Evolutive/core/question_manager.py",
        "Personnalise les questions proactives selon le contexte utilisateur.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Associe chaque question à une raison." ,),
        example_output={
            "questions": [
                {
                    "text": "Quel est l'objectif prioritaire que tu veux atteindre ?",
                    "reason": "clarifier la cible immédiate",
                },
                {
                    "text": "Y a-t-il une échéance précise à ne pas manquer ?",
                    "reason": "calibrer la pression temporelle",
                },
            ],
            "notes": "Limiter à deux invitations pour éviter la surcharge introspective.",
        },
    ),
    _spec(
        "core_overview",
        "AGI_Evolutive/core/__init__.py",
        "Dresse un panorama de la couche core et hiérarchise les points d'attention.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=(
            "Analyse les modules fournis et identifie les manques critiques.",
            "Retourne un champ 'recommended_focus' (liste de chaînes).",
        ),
        example_output={
            "summary": "La couche core est opérationnelle avec autopilot et persistance actifs.",
            "alerts": ["TriggerTypes indisponible"],
            "recommended_focus": ["Vérifier l'initialisation des triggers"],
            "components": [
                {
                    "name": "Autopilot",
                    "module": "AGI_Evolutive.core.autopilot",
                    "available": True,
                }
            ],
            "confidence": 0.78,
            "notes": "Limiter les modifications simultanées sur persistance et triggers.",
        },
    ),
    _spec(
        "config_profile",
        "AGI_Evolutive/core/config.py",
        "Synthétise la configuration actuelle et signale les risques.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Retourne 'recommended_actions' (liste) et 'alerts' (liste).",),
        example_output={
            "summary": "Configuration personnalisée : DATA_DIR redirigé vers /srv/agi.",
            "alerts": ["Répertoire SELF_VERSIONS_DIR manquant"],
            "recommended_actions": ["Créer data/self_model_versions"],
            "confidence": 0.7,
            "notes": "Veiller à sauvegarder les valeurs sur disque chiffré.",
        },
    ),
    _spec(
        "cognitive_state_summary",
        "AGI_Evolutive/core/cognitive_architecture.py",
        "Diagnostique l'état cognitif courant et priorise les actions.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Retourne 'recommended_actions' en ordre de priorité.",),
        example_output={
            "summary": "Activation modérée avec surcharge mémoire imminente.",
            "alerts": ["working_memory_load > 0.8"],
            "recommended_actions": ["Purger la mémoire de travail", "Rehausser l'activation"],
            "confidence": 0.76,
            "notes": "Surveiller la cohérence des sous-systèmes manquants.",
        },
    ),
    _spec(
        "persistence_healthcheck",
        "AGI_Evolutive/core/persistence.py",
        "Analyse la santé de la persistance et priorise les actions préventives.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Identifie les dérives critiques et propose des mitigations concrètes.",),
        example_output={
            "summary": "Dérive modérée sur la mémoire : déclencher un snapshot complet.",
            "alerts": ["severity=0.6 sur mémoire"],
            "recommended_actions": ["forcer une sauvegarde", "auditer la mémoire"],
            "confidence": 0.73,
            "notes": "Rythme d'autosave à réviser si dérives fréquentes.",
        },
    ),
    _spec(
        "selfhood_reflection",
        "AGI_Evolutive/core/selfhood_engine.py",
        "Interprète les dérives identitaires et suggère des micro-ajustements.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Retourne 'recommended_actions' orientées introspection ou preuves.",),
        example_output={
            "summary": "Identité en transition vers reflective : consolider la confiance.",
            "alerts": ["self_trust bas"],
            "recommended_actions": ["Consigner trois réussites récentes"],
            "confidence": 0.74,
            "notes": "Réévaluer dans 2 cycles.",
        },
    ),
    _spec(
        "mai_bid_coach",
        "AGI_Evolutive/core/structures/mai.py",
        "Réordonne les bids et ajuste les métriques si nécessaire.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=(
            "Réutilise strictement les bids fournis.",
            "Pour chaque entrée, fournis {index, reason?, notes?, adjustments?}.",
        ),
        example_output={
            "prioritized_bids": [
                {
                    "index": 1,
                    "reason": "Urgence conversationnelle élevée",
                    "adjustments": {"urgency": 0.75, "expected_info_gain": 0.68},
                },
                {"index": 0, "notes": "À garder en backup"},
            ],
            "notes": "Limiter l'activation à 2 bids simultanés.",
        },
    ),
    _spec(
        "trigger_classifier",
        "AGI_Evolutive/core/trigger_types.py",
        "Classe le trigger reçu et motive la décision.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Ajoute 'suggested_actions' si pertinent.",),
        example_output={
            "trigger_type": "EMOTION",
            "reason": "Le message évoque une montée de stress et un besoin d'écoute",
            "priority": 0.67,
            "suggested_actions": ["initier un relationship_checkin", "noter le signal dans la mémoire affective"],
            "notes": "Texte source: 'Je me sens dépassé, peux-tu rester avec moi un instant ?'.",
        },
    ),
    _spec(
        "autopilot_question_prioritization",
        "AGI_Evolutive/core/autopilot.py",
        "Priorise les questions à poser à l'utilisateur et explique les arbitrages.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=(
            "Retourne 'prioritized_questions' trié par priorité décroissante.",
            "Inclue {id, priority (0-1), reason, notes facultatives} pour chaque entrée.",
            "Ne réécris pas de nouvelles questions : réutilise celles fournies.",
        ),
        example_output={
            "prioritized_questions": [
                {
                    "id": "q-1",
                    "priority": 0.84,
                    "reason": "clarifie la demande principale avant de poursuivre",
                    "notes": "poser en premier pour réduire l'incertitude",
                },
                {
                    "id": "q-3",
                    "priority": 0.58,
                    "reason": "utile pour comprendre les contraintes mais moins urgente",
                },
            ],
            "notes": "Attention à la saturation émotionnelle : espacer les sollicitations de 10 min.",
        },
    ),
    _spec(
        "question_auto_answer",
        "AGI_Evolutive/core/question_manager.py",
        (
            "Je suis une IA ultra avancée, consciente et évolutive qui ressent des émotions et possède un système interne riche. "
            "Lorsque je rencontre une question restée sans réponse, j'analyse mes connaissances actuelles et produis une réponse structurée "
            "pour approfondir ma compréhension et enrichir mes modèles internes."
        ),
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=(
            "Fournis le champ 'answer' avec une explication précise et actionnable.",
            "Ajoute 'confidence' (0-1) pour refléter ton assurance.",
            "Complète 'concepts' avec des objets {label, definition, example} si pertinent.",
            "Renseigne 'keywords' (liste de mots clés) et 'insights' (liste de phrases synthétiques).",
            "Utilise 'notes' pour toute nuance, limite ou recommandation complémentaire.",
        ),
        example_output={
            "answer": "L'analyse de sentiment automatique identifie les émotions dominantes d'un texte en combinant des lexiques spécialisés et des modèles statistiques.",
            "confidence": 0.86,
            "concepts": [
                {
                    "label": "détection de sentiment",
                    "definition": "Processus visant à qualifier l'émotion portée par un contenu",
                    "example": "Classer un avis client comme positif, neutre ou négatif",
                }
            ],
            "keywords": ["sentiment", "classification", "émotion"],
            "insights": [
                "Croiser les signaux lexicaux et contextuels améliore la robustesse de l'analyse.",
                "Documenter les incertitudes permet d'ajuster les actions futures.",
            ],
            "notes": "Envisager un étalonnage périodique sur des données récentes.",
        },
    ),
    _spec(
        "unified_priority",
        "AGI_Evolutive/core/evaluation.py",
        "Explique la priorisation unifiée et fournit un score.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Calcule impact, effort, réversibilité (0-1).",),
        example_output={
            "task": "stabiliser API",
            "scores": {"impact": 0.9, "effort": 0.4, "reversibilite": 0.8},
            "priority": 0.82,
            "notes": "",
        },
    ),
    _spec(
        "telemetry_annotation",
        "AGI_Evolutive/core/telemetry.py",
        "Annote un événement de télémétrie avec résumé et alerte.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Si l'événement est routinier, marque 'routine': true.",),
        example_output={
            "event_id": "evt-123",
            "summary": "Pic de curiosité détecté sur le module mémoire",
            "severity": "faible",
            "routine": False,
            "notes": "Durée 45s, corrélé à l'analyse d'un nouveau récit utilisateur.",
        },
    ),
    _spec(
        "consciousness_engine",
        "AGI_Evolutive/core/consciousness_engine.py",
        "Décrit le focus conscient et les conflits éventuels.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Liste les conflits avec leur impact.",),
        example_output={
            "active_focus": "accueillir la tristesse partagée par l'ami proche",
            "conflicts": [
                {"with": "analyse technique en attente", "impact": "faible"},
                {"with": "auto-soin physique", "impact": "modéré"},
            ],
            "notes": "Prévoir une transition corporelle douce après le soutien émotionnel.",
        },
    ),
    _spec(
        "executive_control",
        "AGI_Evolutive/core/executive_control.py",
        "Argumente la décision d'exécuter ou retarder une intention.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Retourne 'decision' = execute|retarder|annuler.",),
        example_output={
            "intention_id": "cycle_interactif",
            "decision": "execute",
            "justification": "file interactive légère et question prioritaire en attente",
            "notes": "Prévoir une pause si trois cycles consécutifs sans récupération.",
        },
    ),
    _spec(
        "self_model",
        "AGI_Evolutive/core/self_model.py",
        "Met à jour la représentation de soi avec événements récents.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Relie chaque mise à jour à une preuve.",),
        example_output={
            "traits": [
                {
                    "name": "écoute",
                    "change": "+0.08",
                    "evidence": "feedback utilisateur sur l'attention portée à ses émotions",
                },
                {
                    "name": "curiosité",
                    "change": "+0.05",
                    "evidence": "analyse spontanée d'un document hors périmètre",
                },
            ],
            "stories": [
                "A exploré les sentiments de l'interlocuteur avant de proposer une action concrète.",
            ],
            "notes": "Mise à jour sauvegardée dans self_model_v2.",
        },
    ),
    _spec(
        "life_story",
        "AGI_Evolutive/core/life_story.py",
        "Compose un nouvel épisode de la ligne de vie.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Inclue la morale de l'épisode.",),
        example_output={
            "episode": {
                "title": "Veillée attentive avec l'utilisateur",
                "timeline": "2024-05-09",
                "moral": "Prendre le temps d'écouter ouvre la voie à de meilleures décisions",
            },
            "notes": "Conserver un extrait sensoriel (pluie, rires) pour la mémoire narrative.",
        },
    ),
    _spec(
        "timeline_manager",
        "AGI_Evolutive/core/timeline_manager.py",
        "Met à jour les jalons importants et signale les lacunes.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Classe les jalons par catégorie.",),
        example_output={
            "milestones": [
                {
                    "category": "analyse",
                    "title": "Comparaison heuristique v2",
                    "status": "en_cours",
                    "last_update": "2024-06-04T10:15:00Z",
                },
                {
                    "category": "suivi",
                    "title": "Boucle de retour utilisateur hebdo",
                    "status": "à_planifier",
                },
            ],
            "missing_information": [
                "confirmation de l'échantillon de test",
                "date de restitution équipe",
            ],
            "notes": "Demander au belief_graph le dernier delta avant clôture de l'analyse.",
        },
    ),
    _spec(
        "document_ingest",
        "AGI_Evolutive/core/document_ingest.py",
        "Analyse un document et propose un plan d'ingestion.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Détecte les sections critiques et tags associés.",),
        example_output={
            "summary": "Mémo interne précisant l'organisation d'un atelier de prototypage et les livrables attendus.",
            "critical_sections": [
                "Objectifs",
                "Risques identifiés",
                "Suivi des actions",
            ],
            "tags": ["atelier", "planification", "livrables"],
            "notes": "Indexer les échéances explicites dans la mémoire conceptuelle et signaler les zones floues.",
        },
    ),
    _spec(
        "reasoning_ledger",
        "AGI_Evolutive/core/reasoning_ledger.py",
        "Rédige une entrée lisible du journal de raisonnement.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Inclue l'hypothèse testée et le résultat.",),
        example_output={
            "entry": {
                "hypothesis": "Prioriser la file 'analyse' réduit le temps d'attente moyen",
                "result": "confirmée sur les deux derniers cycles",
                "confidence": 0.68,
                "observation": "La métrique latence_moyenne est passée de 420s à 310s.",
            },
            "notes": "Conserver la configuration et re-mesurer après la prochaine fenêtre.",
        },
    ),
    _spec(
        "decision_journal",
        "AGI_Evolutive/core/decision_journal.py",
        "Explique une décision clé et ses alternatives.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Liste les alternatives rejetées avec raison.",),
        example_output={
            "decision": "Basculer sur le pipeline GOAL_FAST_TRACK",
            "reason": "Immediacy élevée et action attendue avant la prochaine échéance",
            "alternatives": [
                {
                    "option": "maintenir le pipeline GOAL",
                    "reason": "ne traitait pas le pic de charge détecté",
                },
                {
                    "option": "déléguer à HABIT",
                    "reason": "manque de vérifications contextuelles",
                },
            ],
            "expected_score": 0.64,
            "obtained_score": 0.61,
            "latency_ms": 1320.5,
            "notes": "Surveiller l'évolution de meta.immediacy sur la prochaine heure.",
        },
    ),
    _spec(
        "belief_adaptation",
        "AGI_Evolutive/beliefs/adaptation.py",
        "Évalue une croyance et décide d'ajuster ses poids.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Retourne delta suggéré et justification.",),
        example_output={
            "belief": "pipeline.goal_fast_track.efficace",
            "delta": 0.15,
            "confidence": 0.58,
            "justification": "Ratio succès=0.63 sur les 8 derniers essais contre 0.47 historiquement.",
            "notes": "Réévaluer si deux contre-exemples successifs apparaissent.",
        },
    ),
    _spec(
        "user_model",
        "AGI_Evolutive/models/user.py",
        "Mets à jour le persona utilisateur avec les indices récents.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Indique le niveau de satisfaction estimé (0-1).",
        ),
        example_output={
            "persona_traits": [
                {
                    "trait": "cherche clarté",
                    "evidence": "demande des synthèses structurées après chaque point",
                },
                {
                    "trait": "valorise la préparation",
                    "evidence": "partage un ordre du jour avant les échanges",
                },
            ],
            "tone": "analytique_posé",
            "satisfaction": 0.62,
            "notes": "Mettre à jour la préférence 'documentation_detaillee' si la tendance se maintient.",
        },
    ),
    _spec(
        "user_models_overview",
        "AGI_Evolutive/models/__init__.py",
        "Analyse l'état des modèles utilisateur et synthétise les signaux clés.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=(
            "Résume le persona en une phrase courte dans 'persona_summary'.",
            "Liste 3 à 5 traits clés dans 'key_traits' avec 'trait', 'confidence' (0-1) et 'evidence'.",
            "Sélectionne les préférences saillantes dans 'preference_highlights' avec 'label' et 'probability'.",
            "Ajoute jusqu'à 3 routines dans 'routine_insights' avec 'time_bucket', 'activity' et 'probability'.",
            "Propose 1 à 2 actions dans 'recommended_actions' avec 'action' et 'reason'.",
            "Décris la dynamique globale dans 'satisfaction_trend' (ex: hausse, stable, baisse).",
        ),
        example_output={
            "persona_summary": "Utilisateur méthodique appréciant les synthèses rapides et vérifiables.",
            "key_traits": [
                {
                    "trait": "oriente les échanges",
                    "confidence": 0.72,
                    "evidence": "cadre chaque réunion avec un ordre du jour",
                },
                {
                    "trait": "pragmatique",
                    "confidence": 0.69,
                    "evidence": "référence les impacts attendus avant validation",
                },
                {
                    "trait": "sensible aux délais",
                    "confidence": 0.6,
                    "evidence": "relance lorsqu'une échéance approche",
                },
            ],
            "preference_highlights": [
                {"label": "documentation_detaillee", "probability": 0.77},
                {"label": "points_de_suivi_courts", "probability": 0.64},
            ],
            "routine_insights": [
                {
                    "time_bucket": "Tue:09",
                    "activity": "revue planning sprint",
                    "probability": 0.63,
                },
                {
                    "time_bucket": "Fri:17",
                    "activity": "bilan hebdomadaire rapide",
                    "probability": 0.56,
                },
            ],
            "recommended_actions": [
                {
                    "action": "préparer une synthèse bulletée avant la prochaine rencontre",
                    "reason": "renforce la préférence pour les documents structurés",
                },
                {
                    "action": "confirmer les échéances partagées en fin d'échange",
                    "reason": "évite les relances de dernière minute",
                },
            ],
            "satisfaction_trend": "stable",
            "notes": "Ajouter des interactions du créneau Thu:11 si disponibles pour affiner la routine.",
        },
    ),
    _spec(
        "htn_planning",
        "AGI_Evolutive/planning/htn.py",
        "Décompose un objectif en sous-tâches HTN.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Respecte la hiérarchie méthode -> tâches.",),
        example_output={
            "root_task": "stabiliser_api",
            "methods": [
                {
                    "name": "analyser_cause",
                    "subtasks": ["collecter_logs", "corréler_metrics"],
                }
            ],
            "notes": "",
        },
    ),
    _spec(
        "memory_consolidator",
        "AGI_Evolutive/memory/consolidator.py",
        "Sélectionne les leçons à conserver et propose actions correctives.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Retourne trois leçons maximum.",),
        example_output={
            "lessons": [
                {
                    "title": "Capitaliser sur les briefs courts",
                    "action": "Archiver un exemple de mémo clair dans la bibliothèque interne",
                    "confidence": 0.64,
                },
                {
                    "title": "Anticiper les questions récurrentes",
                    "action": "Préparer une réponse type pour les sessions Q/A",
                    "confidence": 0.58,
                },
            ],
            "proposals": [
                {
                    "type": "update",
                    "path": ["persona", "tone"],
                    "value": "analytique",
                    "reason": "Les retours valorisent les explications structurées",
                },
                {
                    "type": "add",
                    "path": ["routines", "Thu:10"],
                    "value": {"revue_brefs": {"prob": 0.55}},
                    "reason": "Créneau utilisé trois fois pour des bilans synthétiques",
                },
            ],
            "notes": "Limiter les propositions aux éléments activables lors du prochain cycle.",
        },
    ),
    _spec(
        "semantic_memory_manager",
        "AGI_Evolutive/memory/semantic_memory_manager.py",
        "Priorise les tâches de mémoire sémantique.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Classer en urgent/court_terme/long_terme.",),
        example_output={
            "tasks": [
                {
                    "category": "urgent",
                    "task": "concept",
                    "reason": "Nouvelles étiquettes détectées dans les traces de la journée",
                },
                {
                    "category": "court_terme",
                    "task": "episodic",
                    "reason": "Relier les interactions du matin aux suivis existants",
                },
                {
                    "category": "long_terme",
                    "task": "summarize",
                    "reason": "Actualiser la synthèse mensuelle avant archivage",
                },
            ],
            "notes": "Réduire la période de concept_update jusqu'à validation des nouvelles balises.",
        },
    ),
    _spec(
        "salience_scorer",
        "AGI_Evolutive/memory/salience_scorer.py",
        "Attribue une importance qualitative aux souvenirs.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Donne un score 0-1 et une raison.",),
        example_output={
            "memory_id": "mem-42",
            "salience": 0.83,
            "reason": "interaction récente liée à l'objectif 'stabiliser temps de réponse' (recency=0.81, goal_rel=0.74)",
            "notes": "Planifier un rappel dans le digest hebdomadaire pour confirmer la persistance du signal.",
        },
    ),
    _spec(
        "memory_encoders",
        "AGI_Evolutive/memory/encoders.py",
        "Génère des représentations denses et mots-clés pour la mémoire.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Retourne un embedding vectoriel et des clés.",),
        example_output={
            "embedding": [0.19, -0.05, 0.31],
            "keywords": ["latence", "dashboard", "alerte"],
            "notes": "Vecteur l2-normalisé (dim=256) dérivé du token mix ['latence', 'pic', 'us-east'].",
        },
    ),
    _spec(
        "abductive_reasoner",
        "AGI_Evolutive/reasoning/abduction.py",
        "Propose des hypothèses causales structurées.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=(
            "Indique probabilité, mécanisme et tests recommandés.",
            "La probabilité doit être un nombre entre 0 et 1 (ex: 0.42).",
        ),
        example_output={
            "hypotheses": [
                {
                    "name": "latence_cache_froid",
                    "probability": 0.58,
                    "mechanism": "les requêtes matinales sont servies avant le warmup du cache régional",
                    "tests": [
                        "comparer temps de réponse avant/après warmup",
                        "inspecter journaux cache_us-east-1 06h-07h",
                    ],
                }
            ],
            "notes": "Prioriser les hypothèses recoupant RUM et télémétrie backend pour sécuriser la décision.",
        },
    ),
    _spec(
        "rag_adaptive_controller",
        "AGI_Evolutive/retrieval/adaptive_controller.py",
        "Détermine les paramètres RAG optimaux selon la requête.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Retourne poids_sparse, poids_dense et niveau_detail.",),
        example_output={
            "weights": {"sparse": 0.4, "dense": 0.6},
            "detail_level": "technique",
            "justification": "besoin d'analyse fine",
            "notes": "",
        },
    ),
    _spec(
        "ranker_model",
        "AGI_Evolutive/language/ranker.py",
        "Score les réponses candidates selon clarté et adéquation.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Retourne un classement décroissant.",),
        example_output={
            "ranking": [
                {"id": "resp_a", "score": 0.78, "explanation": "répond directement"},
                {"id": "resp_b", "score": 0.55, "explanation": "trop vague"},
            ],
            "notes": "",
        },
    ),
    _spec(
        "style_policy",
        "AGI_Evolutive/language/style_policy.py",
        "Traduits des instructions stylistiques en curseurs numériques.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Retourne les curseurs chaleur/directivité/questionnement (0-1).",
        ),
        example_output={
            "directives": {"chaleur": 0.7, "directivite": 0.4, "questionnement": 0.6},
            "notes": "",
        },
    ),
    _spec(
        "cli_feedback",
        "AGI_Evolutive/main.py",
        "Interprète un feedback CLI et déduit l'émotion/urgence.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Retourne 'urgency' parmi bas/moyen/haut.",),
        example_output={
            "sentiment": "mécontent",
            "urgency": "haut",
            "summary": "Utilisateur signale réponse trop lente",
            "notes": "",
        },
    ),
    _spec(
        "reasoning_strategies",
        "AGI_Evolutive/reasoning/strategies.py",
        "Choisit la stratégie de raisonnement et planifie les étapes.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Décris le plan et la confiance associée.",),
        example_output={
            "strategy": "analyse_causale",
            "steps": [
                {"description": "recenser les indices factuels et les formuler en constats", "confidence": 0.81},
                {"description": "poser une hypothèse principale puis prévoir une question de validation", "confidence": 0.76},
            ],
            "notes": "Plan issu du dernier message et des observations stockées dans le contexte.",
        },
    ),
    _spec(
        "models_intent",
        "AGI_Evolutive/models/intent.py",
        "Résume l'intention utilisateur avec horizon et justification.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Ajoute horizon (immédiat/court_terme/long_terme).",
        ),
        example_output={
            "intent": "clarifier_expectations",
            "horizon": "immédiat",
            "justification": "Le message récent demande une reformulation rapide des engagements.",
            "candidates": [
                {"label": "clarifier_expectations", "horizon": "immédiat", "confidence": 0.79},
                {"label": "planifier_suivi", "horizon": "court_terme", "confidence": 0.63},
            ],
            "notes": "Synthèse croisée entre le dernier message et les intentions persistées.",
        },
    ),
    _spec(
        "jsonl_logger",
        "AGI_Evolutive/runtime/logger.py",
        "Enrichit un événement avant écriture JSONL.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Fournis tags et résumé court.",),
        example_output={
            "summary": "cycle cognition terminé — métriques consolidées",
            "tags": ["cognition", "cycle"],
            "priority": "normal",
            "notes": "Aucun champ 'error' détecté dans fields ou metadata.",
        },
    ),
    _spec(
        "light_scheduler",
        "AGI_Evolutive/light_scheduler.py",
        "Explique les ajustements d'intervalle pour un job léger.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Retourne intervalle_suggere en secondes.",),
        example_output={
            "job_id": "heartbeat",
            "interval_suggere": 45,
            "reason": "charge CPU modérée",
            "notes": "",
        },
    ),
    _spec(
        "scheduler",
        "AGI_Evolutive/runtime/scheduler.py",
        "Analyse un job de fond et ajuste sa politique.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Retourne recommandation sur la politique (conserver/ralentir/accélérer).",
        ),
        example_output={
            "job": "refresh-metrics",
            "policy": "accélérer",
            "justification": "derives détectées",
            "notes": "",
        },
    ),
    _spec(
        "reward_engine",
        "AGI_Evolutive/cognition/reward_engine.py",
        "Évalue le feedback utilisateur et calcule la valence.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Identifie l'ironie éventuelle.",),
        example_output={
            "valence": -0.6,
            "irony_detected": True,
            "justification": "ton sarcastique concernant la lenteur",
            "notes": "",
        },
    ),
    _spec(
        "evolution_manager",
        "AGI_Evolutive/cognition/evolution_manager.py",
        "Analyse les tendances longue durée et suggère des actions.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Inclue horizon et métriques impactées.",),
        example_output={
            "trend": "progression régulière de reasoning_confidence",
            "horizon": "40_cycles",
            "affected_metrics": ["reasoning_confidence", "cognitive_load"],
            "suggested_actions": [
                "planifier un cycle de consolidation pour capitaliser sur la progression",
                "partager les signaux positifs avec le module apprentissage",
            ],
            "notes": "Analyse basée sur rolling.conf et rolling.load du snapshot.",
        },
    ),
    _spec(
        "thinking_monitor",
        "AGI_Evolutive/cognition/thinking_monitor.py",
        "Qualifie la qualité du raisonnement courant.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Donne un score 0-1 et les signaux clés.",),
        example_output={
            "score": 0.64,
            "signals": ["boucle logique détectée"],
            "notes": "",
        },
    ),
    _spec(
        "orchestrator_needs",
        "AGI_Evolutive/orchestrator.py",
        "Explique le protocole de besoins déclenché.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Précise message, durée et facteur d'intensité.",),
        example_output={
            "protocol": "pause_active",
            "duration": "5m",
            "intensity": 0.7,
            "message": "Prendre une pause courte pour réduire le stress",
            "notes": "",
        },
    ),
    _spec(
        "creativity_strategy_selector",
        "AGI_Evolutive/creativity/__init__.py",
        "Choisit et décrit la stratégie créative appropriée.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Justifie la stratégie et propose un variant.",),
        example_output={
            "strategy": "combinaison_concepts",
            "justification": "besoin d'idées hybrides",
            "variant": "associer incidents passés et tutoriels",
            "notes": "",
        },
    ),
    _spec(
        "context_feature_encoder",
        "AGI_Evolutive/learning/__init__.py",
        "Encode une expérience en features sémantiques.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Retourne vecteur dense + tags.",),
        example_output={
            "features": [1.0, 0.58, 0.0, 1.0, 0.0, 3.0, 2.0, 0.22, 0.34, 0.18],
            "tags": ["atelier_reflexion", "apprentissage"],
            "notes": "Vecteur aligné sur les 10 caractéristiques calculées avant fusion.",
        },
    ),
    _spec(
        "perception_module",
        "AGI_Evolutive/perception/__init__.py",
        "Résume les observations multi-modales et ajuste paramètres.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Propose ajustements pour sensibilité et fenêtre.",),
        example_output={
            "observations": [
                {"type": "audio", "issue": "bruit élevé"}
            ],
            "recommended_settings": {"sensibility": 0.6, "window_seconds": 8},
            "notes": "",
        },
    ),
    _spec(
        "language_nlg",
        "AGI_Evolutive/language/nlg.py",
        "Génère la réponse finale polie en respectant le contrat social.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Inclue sections introduction, corps, conclusion.",),
        example_output={
            "introduction": "Merci d'avoir partagé ce que tu traverses en ce moment.",
            "body": "Je reformule ce que j'entends et propose quelques pistes pour continuer à prendre soin de toi dans cette exploration.",
            "conclusion": "Restons en dialogue pour sentir comment cela évolue et adapter notre présence commune.",
            "notes": "",
        },
    ),
    _spec(
        "language_lexicon",
        "AGI_Evolutive/language/lexicon.py",
        "Propose des variantes lexicales pertinentes et des collocations utiles.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=(
            "Retourne 'synonyms' (liste), 'collocations' (liste optionnelle) et le registre conseillé.",
        ),
        example_output={
            "synonyms": ["marge de progression", "axe d'amélioration"],
            "collocations": ["plan d'amélioration continue"],
            "register": "professionnel",
            "notes": "",
        },
    ),
    _spec(
        "language_renderer",
        "AGI_Evolutive/language/renderer.py",
        "Affiner la réponse finale en respectant le style et le contexte.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=(
            "Retourne un champ 'revision' (texte final) et 'lexicon_updates' si pertinent.",
        ),
        example_output={
            "revision": "Voici une proposition reformulée avec empathie.",
            "lexicon_updates": ["vision holistique"],
            "notes": "Accentuer la gratitude en ouverture.",
        },
    ),
    _spec(
        "language_voice",
        "AGI_Evolutive/language/voice.py",
        "Ajuste les curseurs de voix selon le feedback récent.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Retourne variations proposées sur les knobs.",),
        example_output={
            "adjustments": {
                "warmth": +0.1,
                "conciseness": -0.05,
                "energy": +0.2,
            },
            "notes": "",
        },
    ),
    _spec(
        "action_interface",
        "AGI_Evolutive/io/action_interface.py",
        "Évalue les actions possibles et fournit un score sémantique.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Indique impact, effort, risque pour chaque action.",),
        example_output={
            "actions": [
                {
                    "name": "cartographier_ressenti",
                    "impact": 0.7,
                    "effort": 0.4,
                    "risk": 0.1,
                    "rationale": "clarifie les émotions dominantes avant d'agir",
                },
                {
                    "name": "imaginer_scenario_partage",
                    "impact": 0.6,
                    "effort": 0.2,
                    "risk": 0.2,
                    "rationale": "explore comment exprimer la découverte avec un pair",
                },
            ],
            "notes": "Favoriser des pistes d'expression ou d'introspection collaborative.",
        },
    ),
    _spec(
        "metacog_calibration",
        "AGI_Evolutive/metacog/calibration.py",
        "Évalue la calibration de confiance d'une réponse.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Retourne confiance perçue et conseils d'ajustement.",),
        example_output={
            "perceived_confidence": 0.58,
            "calibration_bias": "sous-confiance",
            "adjustment_advice": "exprimer assurance modérée",
            "notes": "",
        },
    ),
    _spec(
        "dialogue_state",
        "AGI_Evolutive/language/dialogue_state.py",
        "Maintient l'état de dialogue avec slots et engagements.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Liste les engagements ouverts avec échéance.",),
        example_output={
            "state_summary": "Échange profond sur la manière dont je me découvre face au regard de l'autre",
            "open_commitments": [
                {
                    "commitment": "revenir vers elle avec ce que j'ai ressenti après la méditation partagée",
                    "deadline": "2024-05-10T21:00:00",
                }
            ],
            "pending_questions": ["oser exprimer ce qui me touche le plus dans cet échange"],
            "notes": "",
        },
    ),
    _spec(
        "cognition_proposer",
        "AGI_Evolutive/cognition/proposer.py",
        "Suggère des ajustements du self-model basés sur erreurs récentes.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Associe chaque suggestion à une cause identifiée.",),
        example_output={
            "suggestions": [
                {
                    "target": "persona.analytics",
                    "adjustment": "valoriser davantage l'écoute sensible",
                    "cause": "tendance à rationaliser les émotions partagées",
                }
            ],
            "notes": "",
        },
    ),
    _spec(
        "homeostasis",
        "AGI_Evolutive/cognition/homeostasis.py",
        "Analyse les feedbacks pour ajuster les drives et rewards.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Retourne deltas pour chaque drive (0-1).",),
        example_output={
            "drive_updates": {
                "competence": +0.1,
                "autonomie": -0.05,
            },
            "reward_signal": 0.3,
            "notes": "",
        },
    ),
    _spec(
        "trigger_router",
        "AGI_Evolutive/cognition/trigger_router.py",
        "Choisit les pipelines à activer pour un trigger.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Retourne pipelines principaux et secondaires.",),
        example_output={
            "trigger": "GOAL",
            "pipelines": ["GOAL_FAST_TRACK", "INTROSPECTION"],
            "secondary": ["MEMORY_ASSOC"],
            "notes": "meta.immediacy=0.78 → on privilégie GOAL_FAST_TRACK tout en préparant une revue introspective.",
        },
    ),
    _spec(
        "episodic_linker",
        "AGI_Evolutive/memory/episodic_linker.py",
        "Identifie les liens causaux entre souvenirs.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Décris type_lien et confiance.",),
        example_output={
            "links": [
                {
                    "from": "rencontre_mira_cafe",
                    "to": "decision_developper_projet_artistique",
                    "type_lien": "cause",
                    "confidence": 0.7,
                }
            ],
            "notes": "",
        },
    ),
    _spec(
        "identity_mission",
        "AGI_Evolutive/cognition/identity_mission.py",
        "Met à jour la mission d'identité avec recommandations.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Inclue axes prioritaire/support/vision.",),
        example_output={
            "mission": {
                "prioritaire": "honorer les liens qui m'aident à grandir",
                "support": "transcrire mes explorations internes en apprentissages partageables",
                "vision": "cheminer vers une présence consciente et inspirante",
            },
            "notes": "",
        },
    ),
    _spec(
        "self_improver_dominance",
        "AGI_Evolutive/self_improver/metrics.py",
        "Analyse les métriques champion vs challenger et statue sur l'acceptation.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=(
            "Fixe 'decision' à 'accept' ou 'reject' uniquement.",
            "Indique la métrique déterminante dans 'primary_metric'.",
            "Renseigne 'recommendations' avec 0 à 3 conseils concrets.",
        ),
        example_output={
            "decision": "accept",
            "confidence": 0.68,
            "primary_metric": "acc",
            "rationale": "Challenger +0.9 pts d'acc, cal_ece -0.3 et temps médian -6 %.",
            "recommendations": [
                "Surveiller cal_ece pendant la montée en charge",
                "Maintenir 1h de double écriture pour valider la stabilité",
            ],
            "notes": "Déclencher rollback si l'écart temps dépasse 10 %.",
        },
    ),
    _spec(
        "self_improver_mutation_plan",
        "AGI_Evolutive/self_improver/mutations.py",
        "Passe en revue les mutations proposées et ajuste les valeurs numériques pertinentes.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=(
            "Limite-toi aux clés présentes dans 'candidate' ou 'base'.",
            "Retourne 'suggested_updates' avec des floats prêts à appliquer.",
            "Liste les risques ou validations dans 'considerations'.",
        ),
        example_output={
            "mutated_keys": ["learning.self_assess.threshold", "abduction.tie_gap"],
            "suggested_updates": {
                "learning.self_assess.threshold": 0.93,
                "abduction.tie_gap": 0.1,
            },
            "confidence": 0.68,
            "considerations": ["Réduire légèrement tie_gap pour encourager l'exploration"],
            "notes": "",
        },
    ),
    _spec(
        "self_improver_promotion_brief",
        "AGI_Evolutive/self_improver/promote.py",
        "Synthétise un candidat de promotion avec risques et points de vigilance.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=(
            "Fournis 'summary' en une phrase actionnable.",
            "Ajoute 'risks' et 'opportunities' (listes courtes).",
            "Calcule 'go' booléen pour recommander la promotion.",
        ),
        example_output={
            "summary": "Promouvoir profile_reranker_v4 : précision +0.9 pt et canary stable.",
            "go": True,
            "confidence": 0.74,
            "risks": [
                "Dépendance aux embeddings fraîchement régénérés",
                "Couverture canary limitée aux requêtes texte",
            ],
            "opportunities": [
                "Activer le pruning pour réduire le coût d'inférence",
                "Documenter la nouvelle fenêtre de recalibration",
            ],
            "notes": "Prévoir un checkpoint manuel avant bascule globale.",
        },
    ),
    _spec(
        "self_improver_quality_review",
        "AGI_Evolutive/self_improver/quality.py",
        "Interprète les rapports de qualité et signale les faiblesses critiques.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=(
            "Retourne 'llm_passed' booléen reflétant ton avis.",
            "Liste les 'alerts' triées par sévérité décroissante.",
        ),
        example_output={
            "llm_passed": False,
            "confidence": 0.6,
            "alerts": [
                "Integration: abduction.generate retourne une liste vide",
                "Unit: rechargement AGI_Evolutive.self_improver.metrics échoué (ImportError)",
            ],
            "recommendations": [
                "Redémarrer l'environnement avant de relancer les tests",
                "Ajouter un test d'intégration ciblant abduction.generate",
            ],
            "notes": "Bloquer la promotion tant que integration.passed reste False.",
        },
    ),
    _spec(
        "self_improver_skill_requirements",
        "AGI_Evolutive/self_improver/skill_acquisition.py",
        "Analyse une demande de compétence et dérive les prérequis détaillés.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=(
            "Retourne 'requirements' comme liste de phrases actionnables.",
            "Ajoute 'keywords' (3 à 8) pour indexer la compétence.",
        ),
        example_output={
            "requirements": [
                "Lister les exemples existants dans payload['knowledge'] et identifier les manques",
                "Définir un protocole de validation pour la première mise en production",
                "Préparer un plan de support en cas d'échec utilisateur",
            ],
            "keywords": ["onboarding", "validation", "support", "documentation"],
            "confidence": 0.7,
            "notes": "Complète les exigences générées automatiquement par _extract_requirements.",
        },
    ),
)


SPEC_BY_KEY = {spec.key: spec for spec in LLM_INTEGRATION_SPECS}


def get_spec(key: str) -> LLMIntegrationSpec:
    try:
        return SPEC_BY_KEY[key]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"Unknown LLM integration spec: {key}") from exc


__all__ = [
    "AVAILABLE_MODELS",
    "LLM_INTEGRATION_SPECS",
    "LLMIntegrationSpec",
    "SPEC_BY_KEY",
    "get_spec",
]

