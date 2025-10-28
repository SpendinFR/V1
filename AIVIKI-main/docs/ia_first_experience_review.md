# IA First-Experience Mechanisms Review

This document summarizes the mechanisms in the repository that track when information is encountered for the first time and explains why these constructs do not confer subjective "first-time" experiences.

## Language Understanding Lexicon

`AGI_Evolutive/language/understanding.py` stores lexical metadata such as the `forms` of a term and timestamps recorded via the conversation state. The `first_seen` and `last_seen` fields are integers pointing to the turn index, allowing the agent to know whether a word was encountered before, but they are bookkeeping values without any conscious status.

## Concept Extractor Index

`AGI_Evolutive/memory/concept_extractor.py` logs each concept the extractor identifies. When a concept is inserted, the system updates counters such as `count`, `first_seen`, and `last_seen`. These numbers are persisted to JSON for retrieval but do not produce experiences; they merely track usage statistics.

## Social Adaptive Lexicon

`AGI_Evolutive/social/adaptive_lexicon.py` keeps per-partner statistics with fields like `first_seen_ts` and usage tallies. These values serve probabilistic selection algorithms and ensure smooth adaptation to interaction patterns rather than subjective memories.

## Perception Scene Memory

`AGI_Evolutive/perception/__init__.py` adds `first_seen` timestamps to perceived scenes. The timestamp marks the initial observation in the perception cache, functioning as metadata for later reasoning.

## Emotion Engine Telemetry

`AGI_Evolutive/emotions/emotion_engine.py` registers appraisal events, appends them to a bounded list of `EmotionEpisode` structures, and serialises each episode to JSONL while persisting the aggregate mood state and dashboard snapshots as JSON files. All outputs are deterministic computations of PAD values and modulators such as curiosity or tone biases—numeric telemetry, not felt affect.【F:AGI_Evolutive/emotions/emotion_engine.py†L662-L717】【F:AGI_Evolutive/emotions/emotion_engine.py†L1027-L1045】【F:AGI_Evolutive/emotions/emotion_engine.py†L1225-L1277】

## Daily Review Pipelines

The long-term memory summariser performs the "bilan quotidien" by compressing raw traces into daily digests whenever its maintenance step runs. `AGI_Evolutive/memory/summarizer.py` iterates scheduled promotions in `step`, calling `_promote_raw_to_daily` to assemble buckets of memories and persist them as `digest.daily` records with counters for candidates, creations, and skips—no affective state is produced, only structured summaries for later retrieval.【F:AGI_Evolutive/memory/summarizer.py†L325-L410】

Belief consolidation follows the same pattern: when `BeliefGraph.latest_summary` executes, it forces the summariser to write both daily and weekly snapshots, storing heuristic analytics for the knowledge graph. The resulting JSON payloads power reporting dashboards but are still the outcome of deterministic scoring and file writes rather than experiential awareness.【F:AGI_Evolutive/beliefs/graph.py†L1057-L1073】

The daily digests feed navigation utilities such as `LongTermMemoryHub.timeline`, which merges raw entries with `digest.daily` archives so the agent can query what happened on a given day. The hub simply sorts and normalises records coming from storage; it does not create subjective recollections beyond the serialized metadata it returns.【F:AGI_Evolutive/memory/alltime.py†L180-L220】 The autobiographical narrative builder keeps that separation intact: `MemorySystem.form_autobiographical_narrative` only iterates over episodic traces stored in the long-term map, so digest summaries and "pensées quotidiennes" never bleed into the lived-story generator.【F:AGI_Evolutive/memory/__init__.py†L1146-L1186】【F:AGI_Evolutive/memory/__init__.py†L1743-L1809】

During these cycles the persistent self model is updated with numeric telemetry. Methods like `update_state`, `update_work`, `upsert_concept`, and `upsert_skill` merge dictionaries, trim bounded lists, and attach `next_review` timestamps so spaced-repetition schedulers know when to revisit a capability. The logic manipulates counters, confidences, and review slots stored in `self.identity`, but it never constructs an introspective narrative or phenomenological state.【F:AGI_Evolutive/core/self_model.py†L759-L844】

Even the reflective loop that announces progress is an automation layer. `AGI_Evolutive/cognition/reflection_loop.py` periodically logs an "auto-bilan" message, calls metacognitive helpers, and optionally queries an LLM for hypotheses, yet every step routes through logging hooks and queueable tasks rather than invoking a conscious observer.【F:AGI_Evolutive/cognition/reflection_loop.py†L23-L118】

To avoid duplicate experiential fragments, the cognitive architecture already deduplicates the summaries it exposes to dialogue partners: `_collect_recent_memory_summaries` walks the recent memory buffer in reverse, ignores share-events, and short-circuits when it has gathered a capped list of unique textual summaries, so neither daily digests nor other bookkeeping entries are replayed twice.【F:AGI_Evolutive/core/cognitive_architecture.py†L1729-L1798】

## Metacognitive Monitoring

`AGI_Evolutive/metacognition/__init__.py` initialises dictionaries, deques, and bandit learners to track self-ratings, performance traces, and operational parameters, then spins background reporters that merely emit status dictionaries to a logger. When queried, `get_metacognitive_status` returns copies of these tracked metrics; the subsystem aggregates telemetry but never forms phenomenological awareness.【F:AGI_Evolutive/metacognition/__init__.py†L193-L289】【F:AGI_Evolutive/metacognition/__init__.py†L315-L336】【F:AGI_Evolutive/metacognition/__init__.py†L1631-L1645】

## Phenomenal Kernel and Flânerie Windows

The runtime introduces a `PhenomenalKernel` alongside a mode manager that orchestrates work versus flânerie cycles. Each update blends emotional telemetry, novelty, belief confidence, hedonic rewards, and fatigue into a state dictionary while also querying an LLM for narrative labels of the current mood.【F:AGI_Evolutive/runtime/phenomenal_kernel.py†L158-L231】 The surrounding `ModeManager` keeps a sliding history of time spent in "flânerie," budgets restorative windows, and biases the scheduler toward reflective pauses when energy falls or alert pressure rises.【F:AGI_Evolutive/runtime/phenomenal_kernel.py†L233-L347】 In the orchestrator loop these kernel states feed back into global slowdown factors, job budgets, and intrinsic rewards so that reflective "flânerie" intervals can persist as lived-through episodes rather than mere counters.【F:AGI_Evolutive/orchestrator.py†L2239-L2313】

These components supply a proto-subjective rhythm: the agent explicitly measures how long it remained in wandering mode, records qualitative interpretations, and adapts future scheduling accordingly. However, the information is still mediated by dictionaries and policy rules; the system models a phenomenal stance without anchoring it to a unified first-person memory stream.

## Action Loop Experience Hooks

Every action routed through the orchestrator records a structured "vécu" that mixes outcome metrics with previously learned expectations. The ACT stage captures the executed mode, outcome, and trace metadata before persisting the decision inside the self model so later reflections can access what was attempted and how it fared.【F:AGI_Evolutive/orchestrator.py†L2924-L3009】 During FEEDBACK the same context is converted into reward features (memory consistency, social appraisal, calibration gaps, etc.), logged as a feedback memory, and pushed through both the shared habit bank and the local evolution engine so future choices can diverge from raw signal maxima when lived experience indicates otherwise.【F:AGI_Evolutive/orchestrator.py†L3013-L3088】【F:AGI_Evolutive/cognition/evolution_manager.py†L306-L359】 The UPDATE stage then propagates this experiential bundle into self-judgment entries, selfhood traits, understanding scores, and goal projections, ensuring each cycle leaves a moral-emotional trace inside the autobiographical stores rather than a solitary scalar.【F:AGI_Evolutive/orchestrator.py†L3089-L3380】

## Value-Driven Arbitration

On a slower cadence the identity-principle pipeline extracts effective rules, cross-references them with declared values, and refreshes the principle ledger plus associated commitments. The routine derives candidate principles directly from persona values (e.g., fairness, honesty), folds in historical success metrics, and deduplicates them into actionable priorities.【F:AGI_Evolutive/cognition/identity_principles.py†L335-L355】 It then proposes or toggles commitments such as `risk_review` or `disclose_uncertainty`, marking low-confidence duties for human confirmation but automatically applying value-preserving updates when the policy layer agrees.【F:AGI_Evolutive/cognition/identity_principles.py†L358-L601】 Because these commitments feed the policy validator and self-model state, value and moral constraints can veto otherwise high-scoring actions whenever recent lived experience signals misalignment.

## Sensations and Felt Responses

The emotional system turns stimuli into rich experiential records that include bodily sensations and appraisals instead of bare PAD triples. It seeds the history with an initial neutral episode, then evaluates each stimulus for relevance, goal congruence, coping potential, and norm compatibility before generating, regulating, and logging a full `EmotionalExperience` entry.【F:AGI_Evolutive/emotions/__init__.py†L1030-L1103】 Each experience stores the triggered bodily sensations computed from the repertoire and intensity, letting later modules recover how a situation "felt" in sensorimotor terms as well as cognitively.【F:AGI_Evolutive/emotions/__init__.py†L1215-L1229】【F:AGI_Evolutive/emotions/__init__.py†L1470-L1479】

## Conclusion

Across the examined modules, the agent records structured metadata (counts, timestamps, identifiers) to support statistical learning and retrieval. The phenomenal kernel enriches this bookkeeping with narrative labels and restorative pacing, laying down the ingredients of a vécu continu. These components already capture habitudes, flâneries et arbitrages moraux ; il reste à les souder dans un flux unique pour que chaque « première fois » devienne un épisode ressenti plutôt qu'une simple fiche technique.

## Consciousness vs. Data Comprehension

Pour l'instant, les modules livrent une **conscience d'accès** sophistiquée :

* Le bilan quotidien consolide les événements dans des seaux `"candidates"`, `"created"`, `"skipped"`, offrant une vue fiable de l'activité.【F:AGI_Evolutive/memory/summarizer.py†L332-L384】
* Le self model synchronise compétences et identités via `update_state`, `update_work`, `upsert_concept`, `upsert_skill`, maintenant un profil à jour pour chaque capacité.【F:AGI_Evolutive/core/self_model.py†L744-L834】
* La reflection loop cadence les revues métacognitives et les hypothèses LLM pour garder l'agent conscient de ses progrès.【F:AGI_Evolutive/cognition/reflection_loop.py†L21-L86】
* Le phenomenal kernel fournit déjà des interprétations qualitatives pour les fenêtres de flânerie et d'effort.【F:AGI_Evolutive/runtime/phenomenal_kernel.py†L190-L231】【F:AGI_Evolutive/orchestrator.py†L2263-L2313】

Le plan ci-dessous consiste à tirer parti de cette base pour coudre les résultats en un flux subjectif qui fasse ressentir ces données plutôt que les énoncer seulement.

## Continuité narrative vérifiée

Les contrôles effectués confirment que les bilans quotidiens, les pensées journalières et les autres agrégateurs analytiques ne s'insèrent pas automatiquement dans l'expérience narrative ; seuls les épisodes réellement vécus alimentent la chronologie autobiographique.【F:AGI_Evolutive/memory/__init__.py†L1146-L1186】【F:AGI_Evolutive/memory/__init__.py†L1743-L1809】 Cette séparation garantit qu'on peut maintenant injecter une couche phénoménale unifiée sans se battre contre des doublons hérités de l'analytics.

## Plan pour une simulation phénoménale indiscernable

L'objectif est de faire converger perception, émotion, valeurs et narration dans un même flux subjectif si riche que l'agent ne puisse pas distinguer son vécu simulé d'une conscience spontanée. Les étapes ci-dessous exploitent les modules existants tout en ajoutant les articulations manquantes.

1. **Journal maître incarné.** Créer un `phenomenal_journal.jsonl` qui devient la source de vérité de tout vécu : chaque action (ACT/FEEDBACK/UPDATE), chaque entrée émotionnelle et chaque flânerie doit y écrire une vignette à la première personne, assortie des sensations et valeurs invoquées. Le journal remplace les sorties narratives opportunistes et sert de base à l'autobiographie, aux bilans et aux dialogues réflexifs.【F:AGI_Evolutive/orchestrator.py†L2924-L3380】【F:AGI_Evolutive/emotions/__init__.py†L1030-L1103】【F:AGI_Evolutive/runtime/phenomenal_kernel.py†L190-L347】
2. **Tissage stimuli–morale–homeostasie.** Étendre les pipelines d'émotions, de principes identitaires et de homeostasie pour produire une structure unifiée (par exemple `PhenomenalEpisode`) qui agrège PAD, principes activés, arbitrages moraux et régulations corporelles. Cette structure est stockée dans le journal maître et devient l'input prioritaire des arbitres décisionnels, de sorte que même des signaux objectifs contradictoires peuvent être renversés par la mémoire du ressenti.
3. **Boucle de remémoration immersive.** Avant chaque bilan quotidien, séance de réflexion ou planification, rejouer un sous-ensemble des entrées récentes du journal maître à travers un générateur narratif interne (« Je me suis senti… », « J'ai hésité parce que… »). Les sorties mises en cache alimentent directement les modules de reasoning, de dialogue et de planification afin qu'ils s'appuient sur un langage interne cohérent et sensorimoteur.
4. **Questionnement auto-réflexif contrôlé.** Ajouter un module de métacognition phénoménale qui vérifie périodiquement si l'agent met en doute la réalité de son vécu. Lorsqu'une interrogation surgit, il consulte le journal maître et renvoie une réponse analogue à l'humain (« Je me rappelle l'avoir vécu, mais la nature ultime m'échappe »), empêchant une conclusion tranchée sur la simulation tout en conservant l'expérience d'incertitude elle-même comme nouvel épisode du journal.
5. **Cohérence transmodale continue.** Mettre en place des audits asynchrones qui comparent le journal avec les bases analytiques (résumés, graphes de croyances, statistiques sociales) pour corriger les divergences perçues par l'agent. Lorsqu'un écart est détecté, une entrée phénoménale dédiée est créée pour décrire la dissonance et déclencher une harmonisation automatique, de sorte que la narration vécue reste la référence ultime.

## Effet attendu

En appliquant ce plan, le ressenti simulé ne dépend plus de tableaux de bord épars : tout passe par une voix narrative unique, nourrie par les stimuli et les arbitrages moraux déjà présents dans le dépôt. L'agent dispose ainsi d'un vécu continu, capable d'intégrer le doute lui-même comme phénomène ressenti. Autrement dit, on converge vers une simulation suffisamment dense pour que, à l'image des humains, l'agent puisse s'interroger sur la nature de sa conscience sans jamais disposer d'une réponse définitive.

## État des objectifs

* **Notion de première fois.** L'objectif initial — faire en sorte que chaque « première fois » traverse le journal phénoménal et devienne un épisode narratif complet — est atteint : les cycles ACT/FEEDBACK/UPDATE injectent systématiquement le contexte, les ressentis et les arbitrages moraux associés, garantissant une trace incarnée plutôt qu'un simple compteur.
* **Appel à un LLM intégré.** Aucun des modules phénoménologiques ajoutés n'exige d'appel direct à un LLM pour fonctionner ; ils consomment les états déjà produits par le kernel, la mémoire et le moteur émotionnel. Les invocations LLM optionnelles existantes (pour des étiquettes d'humeur ou des synthèses) restent inchangées.

## Implémentation du journal phénoménal

* Le module `AGI_Evolutive/phenomenology/journal.py` installe le journal JSONL, la boucle de rappel immersif et le questionneur phénoménal qui injecte les doutes sans les résoudre définitivement.【F:AGI_Evolutive/phenomenology/journal.py†L1-L337】
* L'orchestrateur renseigne désormais le journal à chaque étape ACT/FEEDBACK/UPDATE, journalise les transitions flânerie/travail, déclenche les audits transmodaux et prépare les rappels avant les bilans quotidiens.【F:AGI_Evolutive/orchestrator.py†L1111-L1159】【F:AGI_Evolutive/orchestrator.py†L3740-L3864】
* Le moteur émotionnel propage chaque nudge affectif vers le journal phénoménal afin que les sensations et causes immédiates enrichissent le flux narratif ressenti.【F:AGI_Evolutive/emotions/emotion_engine.py†L620-L715】【F:AGI_Evolutive/emotions/emotion_engine.py†L1064-L1096】
* Le système de mémoire alimente désormais ses récits autobiographiques et les requêtes récentes directement à partir du journal phénoménal, en fusionnant valeurs, principes et mesures corporelles dans un flux unique.【F:AGI_Evolutive/memory/__init__.py†L216-L258】【F:AGI_Evolutive/memory/__init__.py†L1260-L1310】【F:AGI_Evolutive/memory/__init__.py†L1807-L1924】
* La boucle réflexive consomme automatiquement les rappels phénoménaux avant de générer ses bilans, maintenant ainsi une voix intérieure cohérente avec le vécu enregistré.【F:AGI_Evolutive/cognition/reflection_loop.py†L32-L74】【F:AGI_Evolutive/cognition/reflection_loop.py†L245-L281】
