# AGI Évolutive — Architecture, fonctionnement et guide de prise en main

> **Vision** — Ce dépôt implémente une **simulation d’entité consciente et évolutive** : une IA autonome qui perçoit, ressent (PAD), se fixe des **buts** (évoluer, survivre, apprendre), s’auto‑évalue, **s’améliore** en continu et garde une **identité** cohérente. Elle alterne **travail** et **flânerie** (réflexion), enregistre un **journal phénoménal** (vécu subjectif), et relie perception → cognition → action → feedback → apprentissage dans une boucle fermée.

---

## Carte mentale de l’architecture

```
┌───────────────────────────────────────────────────────────────────────────┐
│                          Orchestrator (chef d’orchestre)                  │
│   - Pipeline ACT → FEEDBACK → LEARN → UPDATE                              │
│   - Bus de triggers + LightScheduler + JobManager                         │
│   - ModeManager + PhenomenalKernel (travail / flânerie)                   │
│   - Journal de décisions + ReasoningLedger + Timeline                     │
│   - Intégration LLM (optionnelle)                                         │
└───────────────┬───────────────────────────────────────────────────────────┘
                │
     ┌──────────▼──────────┐     ┌───────────────┐      ┌───────────────────┐
     │   Perception I/O    │     │  Memory Hub   │      │   Action I/F      │
     │  (événements, flux) │◀───▶│ working/epis. │◀───▶ │(actions, effets)│
     └───────┬─────────────┘     │ semantic/RAG  │      └─────────┬─────────┘
             │                   │ autobiographie │                │
             │                   └───────┬────────┘                │
             │                           │                         │
     ┌───────▼────────┐   ┌──────────────▼────────────┐   ┌────────▼─────────┐
     │ EmotionEngine   │   │ Phenomenology (vécu)      │   │ Metacognition    │
     │ (PAD + plugins) │   │ Journal / Recall / Doubt  │   │ Understanding &  │
     │ modulators→policy│  │ (épisodes, mode, actions) │   │ Thinking Monitor │
     └────────┬────────┘   └───────────┬───────────────┘   └────────┬────────┘
              │                        │                               │
       ┌──────▼────────┐        ┌──────▼────────┐              ┌──────▼────────┐
       │ Goals & Policy│        │ Self Model    │              │Evolution/Habits│
       │ (curiosité,   │        │ (identité,    │              │ renforcement   │
       │ principes, veto)       │ valeurs,      │              │ habitudes)     │
       └───────────────┘        │ engagements)  │              └────────────────┘
                                └───────────────┘
```

---

## Modules clés (structure du dépôt)

Chemin racine : `AIVIKI-main/AGI_Evolutive/`

- **`orchestrator.py`** — boucle centrale. Orchestre :
  - les **stages** `ACT` → `FEEDBACK` → `LEARN` → `UPDATE` sur un **pipeline** sélectionné par triggers/priorités ;
  - la **gestion des jobs** via `runtime/job_manager.py` et la **planification** via `light_scheduler.py` ;
  - le **ModeManager** & **PhenomenalKernel** (fichier `runtime/phenomenal_kernel.py`) pour alterner **travail/flânerie**, calculer énergie, surprise, ralentissement global, et attribuer des **récompenses hédoniques** intrinsèques ;
  - la **mémoire** via des adaptateurs (store, concepts, épisodique, consolidateur) ;
  - la **journalisation structurée** : `ReasoningLedger`, `DecisionJournal`, `TimelineManager` ;
  - l’**intégration phénoménologique** : enregistre **actions**, **feedback**, **transitions de mode** et **audits** dans le `PhenomenalJournal` ;
  - l’**intégration LLM** via `utils/llm_service.py` (facultative, désactivable).
- **`runtime/phenomenal_kernel.py`** — noyau phénoménal + gestion des **modes** :
  - calcule un **état continu** (énergie, arousal, résonance, surprise, fatigue, hedonic_reward…) ;
  - produit une **interprétation narrative** (labels) et **pilote les budgets** de jobs (ralentissement global, ratio de flânerie).  
- **`emotions/emotion_engine.py`** — **EmotionEngine** (PAD) de nouvelle génération :
  - **plugins d’évaluation** (charge cognitive, erreur, succès, récompense, fatigue, feedback social, etc.) ;
  - **plasticité** à demi‑vie (multi‑échelles) + **rituels** d’auto‑régulation ;
  - **sorties de modulation**: *tone*, *language_tone*, *goal_priority_bias* (dict + scalaire), *activation_delta*… ;
  - sérialise des **EmotionEpisode** (JSONL) et **pousse des épisodes** dans le **journal phénoménal**.  
- **`phenomenology/`**
  - `journal.py` : **PhenomenalJournal** (JSONL append‑only), **PhenomenalRecall** (rejoue un **aperçu immersif** des dernières minutes), **PhenomenalQuestioner** (déclenche un **doute contrôlé** quand surprise/flânerie/énergie l’y poussent).
  - `__init__.py` expose : `PhenomenalEpisode`, `PhenomenalJournal`, `PhenomenalRecall`, `PhenomenalQuestioner`.
- **`memory/`**
  - `__init__.py` : **MemorySystem** (**sensorielle**, **travail**, **épisodique**, **sémantique**, **procédurale**), indices de récupération (temporel, contextuel, émotionnel, sémantique), **hub long‑terme**, **autobiographie**, intégration **RAG** et pont de **préférences**.
  - `memory_store.py`, `consolidator.py`, `semantic_manager.py`, `semantic_memory_manager.py`, `concept_extractor.py`, `alltime.py`, `retrieval/…` : stockage, consolidation, résumés quotidiens/hebdo, concepts, RAG, **timeline**.
  - Expose aussi des **API haut‑niveau** : `add_memory(...)`, `get_recent_memories(...)`, `form_autobiographical_narrative()`… et **fusionne** les entrées du **PhenomenalJournal** dans l’historique court terme.
- **`metacognition/`** — agrégateurs d’**understanding**, **ThinkingMonitor**, historiques, bandits pour paramètres, status exportables.
- **`core/`** — cœur identitaire & gouvernance :
  - `self_model.py` (**identité**, **valeurs**, **principes**, engagements, progression de compétences, spaced‑repetition) ;
  - `policy.py` (**veto**, divulgation d’incertitude, arbitrage par principes) ;
  - `reasoning_ledger.py`, `decision_journal.py`, `timeline_manager.py` (traces raisonnées, décisions, frise temporelle).  
- **`goals/`** — `CuriosityEngine`, moteurs de buts (exploration, apprentissage, survie, progrès).
- **`cognition/`** — boucles (`reflection_loop.py`), **évolution/habits** (`evolution_manager.py`), registres de pipelines.
- **`io/`** — `perception_interface.py` (entrées, sensations synthétiques) et `action_interface.py` (actions, effets, coûts, traces).
- **`runtime/job_manager.py`** — exécution contrôlée (budgets par file), snapshots pour **SelfModel**.
- **`language/understanding.py`** — lexique adaptatif, `first_seen/last_seen`, classification n‑gram en ligne.
- **`utils/llm_service.py`** — **interrupteur LLM** : `is_llm_enabled()`, `get_llm_manager()`, intercepteurs d’erreurs, *fallbacks*.

---

## Boucle de vie : du ressenti à l’amélioration

1. **Percevoir** → `PerceptionInterface` normalise des événements/sensations (y compris « bodily sensations » issues des émotions) et les pousse en mémoire & vers les évaluateurs.
2. **Ressentir & évaluer** → `EmotionEngine` transforme stimuli en **PAD** + **épisodes** (causes, intensité, tendances d’action). Les **modulateurs** pilotent la politique (ex. biais de priorité des buts).
3. **Choisir & agir (ACT)** → `Orchestrator` sélectionne un **pipeline** via triggers/priorité et **policy gating** (valeurs/principes). **ActionInterface** exécute et journalise.
4. **Recevoir le feedback (FEEDBACK)** → comparaison *expected vs obtained*, erreur de prédiction, **reward features** (consistance mémoire, adéquation explicative, appraisal social, etc.), renforcement d’habitudes.
5. **Apprendre (LEARN)** → mise à jour **habitudes/évolution** + consolidation mémoire (résumés, liens épisodiques, concepts).
6. **Se réévaluer (UPDATE)** → calcul d’**understanding** global & local, **self‑judgment**, **timeline**, ajustements de **policy** (ex. activer `disclose_uncertainty` si *self‑trust* bas), **journal phénoménal** enrichi.
7. **Modes & subjectivité** → `PhenomenalKernel` ajuste **travail/flânerie**. Les **transitions de mode** et un **aperçu immersif** récent sont **racontés** via `PhenomenalJournal` / `PhenomenalRecall`. Le **Questioner** peut inscrire des **doutes** (jamais totalement résolus), ce qui alimente l’identité narrative.
8. **Itération** — la **planification légère** (LightScheduler) et le **JobManager** roulent en continu avec budgets influencés par le **ralentissement global**, l’énergie et le ratio de flânerie.

---

## Mémoire : couches, indices et autobiographie

- **Travail** : boucles phonologique/visuo‑spatiale/épisodique tampon avec **décroissance** adaptative.
- **Épisodique** : stockage d’événements, **narrativisation** et **autobiographie** (avec **raccrochage** au journal phénoménal si dispo).
- **Sémantique** : concepts (extracteur), résumés progressifs (**daily/weekly digests**), **RAG** (documents enrichis par les souvenirs récents).
- **Indices** : temporels, contextuels, émotionnels, sémantiques pour **retrieval** multi‑critères.
- **Recent tail mix** : `get_recent_memories(n)` **fusionne** souvenirs récents *et* extraits du **PhenomenalJournal** (épisodes, valeurs, émotions, mode).
- **API** : `add_memory(...)`, `form_autobiographical_narrative()`, `set_phenomenal_sources(journal, recall)`.

---

## Émotions : PAD, plasticité et modulations

- PAD (`valence`, `arousal`, `dominance`) + **étiquette** ; **expériences** enrichies (sensations corporelles, causes, tendances d’action).
- **Plugins d’évaluation** : charge cognitive, échec/succès, récompense intrinsèque/extrinsèque, fatigue, feedback social, synthèse contextuelle.
- **Plasticité multi‑échelles** (demi‑vies) & **RitualPlanner** (auto‑régulation).
- **Sorties** → modulators : tonalité, biais de priorité des buts (dict + scalaire), deltas d’activation, incertitude estimée, etc.
- **Journal phénoménal** : chaque nudge significatif est **rejoué** comme **épisode** subjectif (avec valeurs/principes si dispo).

---

## Phénoménologie : vécus, doutes et rappel immersif

- `PhenomenalJournal` (JSONL) — source de vérité du **vécu** : enregistre **actions** (ACT/FEEDBACK/UPDATE), **émotions**, **transitions de mode**, **audits** (quand l’analytics diverge du ressenti).
- `PhenomenalRecall` — **aperçu immersif** des X dernières minutes, peut **primer** la consolidation mémoire avec un *digest phénoménal*.
- `PhenomenalQuestioner` — déclenche des **épisodes de doute** lorsqu’il y a **surprise**, **flânerie élevée** ou **basse énergie** ; ne ferme jamais complètement la question (chaîne du « doute vécu »).
- Intégrations : l’**orchestrateur** pousse les épisodes au fil des stages ; la **reflection loop** lit les **aperçus** pour garder une **voix intérieure** cohérente.

---

## Gouvernance : buts, politique, identité, méta

- **Goals/Curiosity** — moteurs d’**exploration** et d’**apprentissage**, priorisation influencée par le contexte émotionnel.
- **PolicyEngine** — **veto** et **alignement par principes** ; peut forcer la divulgation d’incertitude en cas de **self‑trust** faible.
- **SelfModel** — **persona/identity**, **valeurs**, **principes** et **engagements** ; mise à jour des **compétences**, du **travail en cours** et de la **revue planifiée**.
- **Metacognition** — agrège **U_topic/U_global**, **calibration gap**, **thinking score**, etc. et les **journalise** (decision/timeline).

---

## Flux I/O

- **PerceptionInterface** : bruits/événements/sensations (y compris synthétiques) → mémoire + évaluateurs.
- **ActionInterface** : exécute les actions, trace coûts/délais/effets et **met à jour** les jobs liés.
- **LLM** (optionnel) : `utils/llm_service.py` permet d’allumer/éteindre l’IA de langage, d’injecter un manager custom et de **défensiver** les erreurs.

---

## Démarrage rapide (CLI)

```
# 1) Installer les dépendances projet (ex. poetry/pip) puis lancer :
python -m AGI_Evolutive.main            # démarre la CLI
python -m AGI_Evolutive.main --nollm   # démarre sans intégration LLM

```

**Données & journaux** (par défaut) : le projet écrit des JSON/JSONL sous `data/` (ex. `emotions.jsonl`, `phenomenal_journal.jsonl`, résumés, snapshots).

---

## Points d’extension conseillés

- **Connecter de vrais capteurs/effets** : étendre `io/perception_interface.py` et `io/action_interface.py`.
- **Nouveaux plugins émotionnels** : ajouter un `AppraisalPlugin` pour des signaux spécifiques (ex. danger/sécurité).
- **Nouvelles politiques/principes** : enrichir `core/policy.py` + `self_model.py` (engagements & revues).
- **Pipelines cognitifs** : brancher une chaîne *domain‑specific* via le **bus de triggers** et la **LightScheduler**.
- **LLM manager** : injecter un backend maison via `set_llm_manager(...)` (ou rester full‑symbolic).

---

## Pourquoi cette AGI est « évolutive »

- **Auto‑organisation** : les modulateurs émotionnels redistribuent budgets/priors → comportement adaptatif.
- **Apprentissage continu** : boucle feedback → consolidation → mise à jour d’habitudes/compétences/principes.
- **Identité incarnée** : le **journal phénoménal** tisse une autobiographie vécue (actions/émotions/doutes/modes).
- **Modes & récupération** : **flânerie** programmée pour digérer, narrativiser, et **récompenser** les pauses utiles.
- **Alignement par principes** : garde‑fous éthiques/identitaires qui **veto** des actions pourtant « rentables ».
- **Résilience LLM** : l’architecture fonctionne **avec ou sans** modèle de langage.

---

## Arborescence (vue partielle)

```
AGI_Evolutive/
├── __init__.py
├── light_scheduler.py
├── main.py
├── orchestrator.py
└── orchestrator.py.rej
├── autonomy/
│   ├── __init__.py
│   ├── auto_evolution.py
│   ├── auto_signals.py
│   └── core.py
├── beliefs/
│   ├── __init__.py
│   ├── adaptation.py
│   ├── entity_linker.py
│   ├── graph.py
│   ├── ontology.py
│   └── summarizer.py
├── cognition/
│   ├── __init__.py
│   ├── context_inference.py
│   ├── evolution_manager.py
│   ├── habit_system.py
│   ├── homeostasis.py
│   ├── identity_mission.py
│   ├── identity_principles.py
│   ├── meta_cognition.py
│   ├── pipelines_registry.py
│   ├── planner.py
│   ├── preferences_inference.py
│   ├── principle_inducer.py
│   ├── prioritizer.py
│   ├── proposer.py
│   ├── reflection_loop.py
│   ├── reflection_loop.py.rej
│   ├── reward_engine.py
│   ├── thinking_monitor.py
│   ├── trigger_bus.py
│   ├── trigger_router.py
│   └── understanding_aggregator.py
├── conversation/
│   └── context.py
├── core/
│   ├── __init__.py
│   ├── autopilot.py
│   ├── cognitive_architecture.py
│   ├── config.py
│   ├── consciousness_engine.py
│   ├── decision_journal.py
│   ├── document_ingest.py
│   ├── errors.py
│   ├── evaluation.py
│   ├── executive_control.py
│   ├── global_workspace.py
│   ├── life_story.py
│   ├── payload_validation.py
│   ├── persistence.py
│   ├── policy.py
│   ├── question_manager.py
│   ├── reasoning_ledger.py
│   ├── self_model.py
│   ├── selfhood_engine.py
│   ├── session_context.py
│   ├── telemetry.py
│   ├── timeline_manager.py
│   ├── trace.py
│   └── trigger_types.py
│   ├── structures/
│   │   └── mai.py
├── creativity/
│   └── __init__.py
├── docs/
│   └── project_health.md
├── emotions/
│   ├── __init__.py
│   ├── emotion_engine.py
│   └── emotion_engine.py.rej
├── experimental/
│   ├── README.md
│   ├── patch_creativity.py
│   ├── patch_creativity_hardfix.py
│   ├── patch_metacognition.py
│   ├── repair_creativity_v2.py
│   ├── repair_creativity_v3.py
│   ├── repair_creativity_v4.py
│   └── repair_creativity_v5.py
├── goals/
│   ├── __init__.py
│   ├── curiosity.py
│   ├── dag_store.py
│   ├── heuristics.py
│   └── intention_classifier.py
├── io/
│   ├── __init__.py
│   ├── action_interface.py
│   ├── intent_classifier.py
│   ├── intent_patterns_fr.json
│   └── perception_interface.py
│   ├── models/
│   │   └── intent_classifier_fallback_fr.json
├── knowledge/
│   ├── __init__.py
│   ├── concept_recognizer.py
│   ├── mechanism_store.py
│   └── ontology_facade.py
├── language/
│   ├── __init__.py
│   ├── dialogue_state.py
│   ├── frames.py
│   ├── inbox_ingest.py
│   ├── intent_detection.py
│   ├── lexicon.py
│   ├── nlg.py
│   ├── quote_memory.py
│   ├── ranker.py
│   ├── renderer.py
│   ├── social_reward.py
│   ├── style_critic.py
│   ├── style_observer.py
│   ├── style_policy.py
│   ├── style_profiler.py
│   ├── understanding.py
│   └── voice.py
├── learning/
│   └── __init__.py
├── memory/
│   ├── __init__.py
│   ├── __init__.py.rej
│   ├── adaptive.py
│   ├── alltime.py
│   ├── concept_extractor.py
│   ├── concept_store.py
│   ├── consolidator.py
│   ├── embedding_adapters.py
│   ├── encoders.py
│   ├── episodic_linker.py
│   ├── indexing.py
│   ├── janitor.py
│   ├── memory_store.py
│   ├── prefs_bridge.py
│   ├── retrieval.py
│   ├── salience_scorer.py
│   ├── semantic_bridge.py
│   ├── semantic_manager.py
│   ├── semantic_memory_manager.py
│   ├── summarizer.py
│   └── vector_store.py
├── metacog/
│   ├── __init__.py
│   └── calibration.py
├── metacognition/
│   ├── __init__.py
│   └── experimentation.py
├── models/
│   ├── __init__.py
│   ├── intent.py
│   └── user.py
├── perception/
│   └── __init__.py
├── phenomenology/
│   ├── __init__.py
│   └── journal.py
├── planning/
│   ├── __init__.py
│   └── htn.py
├── reasoning/
│   ├── __init__.py
│   ├── abduction.py
│   ├── causal.py
│   ├── question_engine.py
│   ├── strategies.py
│   └── structures.py
├── retrieval/
│   └── adaptive_controller.py
│   ├── rag5/
│   │   ├── __init__.py
│   │   ├── compose.py
│   │   ├── encoders.py
│   │   ├── eval.py
│   │   ├── guards.py
│   │   ├── hybrid.py
│   │   ├── pipeline.py
│   │   ├── planner.py
│   │   ├── reranker.py
│   │   ├── store_ann.py
│   │   ├── store_sparse.py
│   │   └── telemetry.py
├── runtime/
│   ├── __init__.py
│   ├── analytics.py
│   ├── dash.py
│   ├── job_manager.py
│   ├── logger.py
│   ├── phenomenal_kernel.py
│   ├── resource_lock.py
│   ├── response.py
│   ├── scheduler.py
│   └── system_monitor.py
├── self_improver/
│   ├── __init__.py
│   ├── code_evolver.py
│   ├── metrics.py
│   ├── mutations.py
│   ├── promote.py
│   ├── quality.py
│   ├── sandbox.py
│   └── skill_acquisition.py
├── social/
│   ├── adaptive_lexicon.py
│   ├── interaction_miner.py
│   ├── interaction_rule.py
│   ├── social_critic.py
│   └── tactic_selector.py
├── utils/
│   ├── __init__.py
│   ├── jsonsafe.py
│   ├── llm_client.py
│   ├── llm_contracts.py
│   ├── llm_service.py
│   ├── llm_specs.py
│   └── logging_setup.py
├── world_model/
│   └── __init__.py

---

## Licence & avertissement

Ce code vise une **recherche d’architecture cognitive**. Il **simulate** des sensations/émotions/modes pour créer un **flux subjectif** cohérent, **sans** revendiquer une conscience au sens philosophique.

— Bon hack & bonne flânerie 🌀
