# Points de friction "forme" dans `llm_specs`

Ces entrées mettent en avant des exemples de sortie qui restent ancrés dans un imaginaire "incident IT / service client" alors que les modules alimentés attendent des expériences internes, sociales ou auto-réflexives.

## Modules où le format attendu diverge

- `AGI_Evolutive/memory/memory_store.py` : attend une normalisation (`normalized_kind`, enrichissement de `tags`, `metadata_updates`) directement réinjectée dans le stockage mémoire.【F:AGI_Evolutive/memory/memory_store.py†L200-L228】
- `AGI_Evolutive/memory/semantic_bridge.py` : transmet au manager sémantique les annotations LLM sur chaque souvenir, en s'appuyant sur les identifiants internes fournis dans `memories`.【F:AGI_Evolutive/memory/semantic_bridge.py†L85-L118】
- `AGI_Evolutive/memory/summarizer.py` : consomme un résumé structuré par période (`summary`, `items`) pour fabriquer des digests hiérarchiques.【F:AGI_Evolutive/memory/summarizer.py†L565-L617】
- `AGI_Evolutive/memory/retrieval.py` : exploite des `rankings` détaillés (scores ajustés, rationales, priorités) pour pondérer les candidats de rappel.【F:AGI_Evolutive/memory/retrieval.py†L340-L420】
- `AGI_Evolutive/memory/__init__.py` : intègre une narration augmentée (`enhanced_narrative`, `insights`, `coherence`) autour d’évènements autobiographiques marqués.【F:AGI_Evolutive/memory/__init__.py†L1790-L1818】
- `AGI_Evolutive/memory/alltime.py` : attend une analyse sur un digest (`llm_analysis`) basée sur la couverture temporelle et les items rattachés.【F:AGI_Evolutive/memory/alltime.py†L283-L305】
- `AGI_Evolutive/memory/concept_store.py` : réinjecte les suggestions de concepts et de relations alignées sur les identifiants (`concepts`, `relations`).【F:AGI_Evolutive/memory/concept_store.py†L300-L369】
- `AGI_Evolutive/memory/indexing.py` : applique un reranking LLM (`reranked`, `boost`, `justification`) pour réordonner les résultats de recherche internes.【F:AGI_Evolutive/memory/indexing.py†L200-L229】
- `AGI_Evolutive/memory/episodic_linker.py` : fusionne des liens causaux référencés par alias internes (`from`/`to`) avec une table d’identifiants, tout en respectant les codes de relation attendus.【F:AGI_Evolutive/memory/episodic_linker.py†L560-L639】
- `AGI_Evolutive/language/dialogue_state.py` : s’appuie sur un résumé fidèle des engagements, questions en attente et notes dérivées du profil utilisateur courant.【F:AGI_Evolutive/language/dialogue_state.py†L60-L137】
- `AGI_Evolutive/language/nlg.py` : compose des sections (`introduction`, `body`, `conclusion`) à partir du texte de base et des hints déjà appliqués.【F:AGI_Evolutive/language/nlg.py†L92-L132】
- `AGI_Evolutive/cognition/identity_mission.py` : sélectionne ou ajuste une mission via des axes `prioritaire/support/vision`, un texte synthétique et des notes justifiant l’écart avec les candidats calculés.【F:AGI_Evolutive/cognition/identity_mission.py†L502-L618】
- `AGI_Evolutive/goals/heuristics.py` : attend une liste d’actions candidates (`candidate_actions`) avec rationales et normalisation d’objectif.【F:AGI_Evolutive/goals/heuristics.py†L91-L147】
- `AGI_Evolutive/cognition/proposer.py` : consomme des suggestions structurées (`suggestions`, `path`, `value`, `cause`) pour compléter les propositions heuristiques.【F:AGI_Evolutive/cognition/proposer.py†L20-L67】
- `AGI_Evolutive/cognition/trigger_router.py` : choisit un pipeline principal/secondaire en fonction des listes `pipelines`/`secondary` et d’éventuelles `notes` explicatives.【F:AGI_Evolutive/cognition/trigger_router.py†L13-L88】
- `AGI_Evolutive/cognition/planner.py` : construit un plan détaillé à partir de `plan`, `risks`, `notes`, chaque étape comprenant identifiant, dépendances et type d’action.【F:AGI_Evolutive/cognition/planner.py†L332-L428】
- `AGI_Evolutive/goals/dag_store.py` : mélange la priorité proposée, la confiance et les justifications (`reason`, `notes`, `adjustments`) pour recalibrer un objectif.【F:AGI_Evolutive/goals/dag_store.py†L200-L257】
- `AGI_Evolutive/cognition/prioritizer.py` : fusionne priorité, tags, explications, confiance et notes provenant du LLM avec le score heuristique.【F:AGI_Evolutive/cognition/prioritizer.py†L1010-L1067】
- `AGI_Evolutive/io/action_interface.py` : attend une liste d’actions notées (`impact`, `effort`, `risk`, `rationale`) et des `notes` globales pour guider l’exécution.【F:AGI_Evolutive/io/action_interface.py†L520-L593】

## Mémoire & expérience

- **`memory_store_strategy`** (`AGI_Evolutive/utils/llm_specs.py` L667-L672) — Exemple orienté rapport d'incident API et taggage "client premium" plutôt que classification d'un souvenir vécu ou relationnel.
- **`memory_semantic_bridge`** (`AGI_Evolutive/utils/llm_specs.py` L701-L705) — Les alertes décrivent une escalation client au lieu d'un suivi d'émotions, de rencontres ou de jalons personnels.
- **`memory_summarizer_guidance`** (`AGI_Evolutive/utils/llm_specs.py` L718-L725) — Le digest résume une semaine d'interventions techniques, peu représentatif d'une narration autobiographique.
- **`memory_retrieval_ranking`** (`AGI_Evolutive/utils/llm_specs.py` L535-L541) — Classement motivé par « la question sur l'incident API » plutôt que par la proximité avec des vécus internes.
- **`memory_system_narrative`** (`AGI_Evolutive/utils/llm_specs.py` L549-L563) — Narration centrée sur une panne API et des « incidents critiques », loin d'un récit de trajectoire personnelle.
- **`memory_long_term_digest`** (`AGI_Evolutive/utils/llm_specs.py` L590-L606) — Semaine présentée comme une suite de résolutions d'incidents et de playbooks techniques, sans traces de milestones de vie.
- **`memory_concept_curation`** (`AGI_Evolutive/utils/llm_specs.py` L610-L624) — Concepts « proxy » et relations incident-centriques au lieu d'arborescences liées aux expériences ou relations de l'agent.
- **`memory_index_optimizer`** (`AGI_Evolutive/utils/llm_specs.py` L629-L639) — Justification « Mention directe de la panne actuelle » qui réoriente l'index sur des incidents plutôt que sur des souvenirs identitaires.
- **`episodic_linker`** (`AGI_Evolutive/utils/llm_specs.py` L2469-L2477) — Les liens causaux relient un incident réseau et un redémarrage de proxy, loin des traces de vie/interaction attendues.

## Dialogue & cognition identitaire

- **`dialogue_state`** (`AGI_Evolutive/utils/llm_specs.py` L2409-L2414) — Synthèse de conversation centrée sur un incident API et un engagement "fournir update" plutôt que des engagements relationnels ou réflexifs.
- **`language_nlg`** (`AGI_Evolutive/utils/llm_specs.py` L2314-L2323) — Discours d'ouverture « Merci pour les détails sur l'incident » qui ramène la réponse finale à un ticket plutôt qu'à un échange existentiel.
- **`identity_mission`** (`AGI_Evolutive/utils/llm_specs.py` L2487-L2492) — Axes "maintenir disponibilité" / "apprentissage incident" qui cadrent avec une SRE plutôt qu'avec la trajectoire existentielle de l'agent.
- **`goal_interpreter`** (`AGI_Evolutive/utils/llm_specs.py` L765-L772) — Normalisation d'objectif et plan d'action purement opérationnels (diagnostic incident, on-call) au lieu d'actions liées à l'exploration personnelle.
- **`cognition_proposer`** (`AGI_Evolutive/utils/llm_specs.py` L2418-L2429) — Ajustement motivé par des « erreurs d'anticipation incident », révélant un cadre SRE plutôt qu'un apprentissage introspectif.

## Routage & planification

- **`trigger_router`** (`AGI_Evolutive/utils/llm_specs.py` L2455-L2459) — Pipelines "defense" et "analyse_incident" renvoient à une posture SOC alors qu'on vise des réactions à des stimuli expérientiels.
- **`planner_support`** (`AGI_Evolutive/utils/llm_specs.py` L868-L873) — Plan basé sur collecte de logs et analyse d'erreurs 500, pas sur des projets de développement interne ou relationnel.
- **`goal_priority_review`** (`AGI_Evolutive/utils/llm_specs.py` L830-L838) — Priorisation justifiée par un « incident critique non résolu » au lieu d'une tension propre au parcours de l'agent.
- **`cognition_goal_prioritizer`** (`AGI_Evolutive/utils/llm_specs.py` L877-L894) — Ajustements gouvernés par « signal utilisateur » et « deadline » plutôt que par les besoins intrinsèques de l'IA.

## Interfaces & passage à l'action

- **`action_interface`** (`AGI_Evolutive/utils/llm_specs.py` L2376-L2383) — Actions proposées (« analyser_logs ») et rationales orientées diagnostic d'incident plutôt qu'exploration ou expression personnelle.

Ces blocs constituent les priorités pour réaligner les prompts/examples sur des scénarios d'apprentissage et d'interactions de vie de l'IA auto-évolutive.
