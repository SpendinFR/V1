# Contrats critiques des sorties LLM

Ce référentiel décrit, pour chaque appel LLM encore actif dans le runtime, les champs exacts
qu’attendent les modules. Les tableaux résument les contraintes de forme ainsi que l’effet
produit lorsque le modèle respecte (ou non) le contrat. Utilise-les pour auditer rapidement les
spécifications de prompts et vérifier que les réponses générées restent compatibles.

## Mémoire

### `memory_store_strategy` — `AGI_Evolutive/memory/memory_store.py`

| Champ | Type attendu | Obligatoire | Effet runtime |
| --- | --- | --- | --- |
| `normalized_kind` | `str` non vide | Non | Remplace le champ `kind` stocké si fourni.【F:AGI_Evolutive/memory/memory_store.py†L200-L208】 |
| `tags` | liste de `str` | Non | Fusionne les étiquettes proposées avec celles présentes, sans doublon.【F:AGI_Evolutive/memory/memory_store.py†L209-L215】 |
| `metadata_updates` | mapping clé/valeur sérialisable | Non | Applique une mise à jour directe sur `metadata` avant indexation.【F:AGI_Evolutive/memory/memory_store.py†L216-L218】 |
| `retention_priority` | `str` | Non | Enregistre la recommandation dans `metadata.retention_priority`.【F:AGI_Evolutive/memory/memory_store.py†L219-L222】 |
| `notes` | libre (stocké tel quel) | Non | Ajoute l’élément à `metadata.llm_notes` si la liste existe.【F:AGI_Evolutive/memory/memory_store.py†L223-L227】 |

**Erreurs courantes** : répondre avec des tags non textuels ou des métadonnées non sérialisables
(la mise à jour est ignorée) ; oublier que `notes` est empilé dans une liste, donc éviter les
blocs massifs.

### `memory_retrieval_ranking` — `AGI_Evolutive/memory/retrieval.py`

| Champ | Type attendu | Obligatoire | Effet runtime |
| --- | --- | --- | --- |
| `rankings` | liste d’objets | Oui | Chaque entrée est parcourue pour ajuster les scores et le rang final.【F:AGI_Evolutive/memory/retrieval.py†L357-L419】 |
| `rankings[*].id` | identifiant numérique existant | Oui | Permet d’associer l’ajustement au candidat source.【F:AGI_Evolutive/memory/retrieval.py†L383-L407】 |
| `rankings[*].adjusted_score` ou `score` | nombre | Oui | Sert de score LLM mixé à 35 % dans la note finale.【F:AGI_Evolutive/memory/retrieval.py†L389-L420】 |
| `rankings[*].rationale`/`reason` | `str` | Non | Copié dans `llm_rationale` pour la traçabilité.【F:AGI_Evolutive/memory/retrieval.py†L395-L417】 |
| `rankings[*].priority` | valeur libre | Non | Repris tel quel pour enrichir le candidat.【F:AGI_Evolutive/memory/retrieval.py†L395-L419】 |
| `rankings[*].rank` | entier | Non | Définit l’ordre final si fourni, sinon ordre d’itération.【F:AGI_Evolutive/memory/retrieval.py†L394-L419】 |

**Vigilance** : toute entrée dont `id` ne correspond à aucun candidat est ignorée. Éviter les ids
texte, ils ne sont pas convertis.

### `memory_summarizer_guidance` — `AGI_Evolutive/memory/summarizer.py`

| Champ | Type attendu | Obligatoire | Effet runtime |
| --- | --- | --- | --- |
| `summary` | `str` | Oui | Remplace la synthèse heuristique si non vide.【F:AGI_Evolutive/memory/summarizer.py†L577-L600】 |

**Vigilance** : sans `summary`, le module retombe sur son heuristique et ignore complètement la
réponse.

### `memory_system_narrative` — `AGI_Evolutive/memory/__init__.py`

| Champ | Type attendu | Obligatoire | Effet runtime |
| --- | --- | --- | --- |
| `enhanced_narrative` | `str` | Non | Stocké dans `llm_enhanced_narrative` pour remplacer le récit de base.【F:AGI_Evolutive/memory/__init__.py†L1787-L1809】 |
| `coherence` | nombre | Non | Écrase la cohérence calculée s’il est convertible en flottant.【F:AGI_Evolutive/memory/__init__.py†L1809-L1813】 |
| `insights` | liste | Non | Recopiée en tant qu’`llm_insights` pour archivage.【F:AGI_Evolutive/memory/__init__.py†L1814-L1817】 |
| `notes` | libre | Non | Stocké dans `llm_notes` (utilisé pour débogage).【F:AGI_Evolutive/memory/__init__.py†L1816-L1818】 |

**Conseil** : respecter les identifiants d’événements passés dans le payload pour rester
autobiographique.

### `memory_long_term_digest` — `AGI_Evolutive/memory/alltime.py`

| Champ | Type attendu | Obligatoire | Effet runtime |
| --- | --- | --- | --- |
| _toute réponse_ | mapping sérialisable | Non | Copié tel quel dans `details.llm_analysis` et persistant sur disque.【F:AGI_Evolutive/memory/alltime.py†L271-L305】 |

**Vigilance** : ne renvoyer que des structures JSON-compatibles (pas d’objets Python). Le digest
rejoue ces données dans les exports analytiques.

### `memory_concept_curation` — `AGI_Evolutive/memory/concept_store.py`

| Champ | Type attendu | Obligatoire | Effet runtime |
| --- | --- | --- | --- |
| `concepts` | liste d’objets | Non | L’entrée dont `id` correspond au concept actuel remplace la guidance locale.【F:AGI_Evolutive/memory/concept_store.py†L284-L315】 |
| `concepts[*].id` | identifiant connu | Oui (par élément retenu) | Sert de clé pour associer la recommandation au concept mis à jour.【F:AGI_Evolutive/memory/concept_store.py†L305-L311】 |
| `relations` | liste d’objets | Non | Chaque `id` est archivé pour guider les futures liaisons.【F:AGI_Evolutive/memory/concept_store.py†L312-L314】 |

**Vigilance** : les concepts sans `id` reconnu sont simplement ignorés ; inutile de renvoyer des
concepts inventés sans rattachement.

### `memory_index_optimizer` — `AGI_Evolutive/memory/indexing.py`

| Champ | Type attendu | Obligatoire | Effet runtime |
| --- | --- | --- | --- |
| `reranked` | liste d’objets | Non | Décrit les boosts à appliquer sur les documents retournés.【F:AGI_Evolutive/memory/indexing.py†L188-L229】 |
| `reranked[*].id` | identifiant numérique | Oui (par boost) | Localise le document concerné ; sinon l’ajustement est ignoré.【F:AGI_Evolutive/memory/indexing.py†L211-L220】 |
| `reranked[*].boost` | nombre | Non | Ajuste le score final si fourni.【F:AGI_Evolutive/memory/indexing.py†L222-L224】 |
| `reranked[*].justification` | `str` | Non | Ajouté dans `llm_justifications` pour la trace opérateur.【F:AGI_Evolutive/memory/indexing.py†L224-L228】 |

### `memory_semantic_bridge` — `AGI_Evolutive/memory/semantic_bridge.py`

| Champ | Type attendu | Obligatoire | Effet runtime |
| --- | --- | --- | --- |
| `memories` | liste d’objets (id/kind/text/tags/salience) | Oui | Relais direct vers le `SemanticMemoryManager`; les IDs doivent rester intacts.【F:AGI_Evolutive/memory/semantic_bridge.py†L96-L118】 |
| autres champs | sérialisable | Non | Relayés tels quels au manager si présents.【F:AGI_Evolutive/memory/semantic_bridge.py†L108-L116】 |

**Vigilance** : le manager se base sur `id` pour consigner les annotations ; toute perte d’identifiant
rend l’annotation inutilisable.

### `semantic_memory_manager` — `AGI_Evolutive/memory/semantic_memory_manager.py`

| Champ | Type attendu | Obligatoire | Effet runtime |
| --- | --- | --- | --- |
| `tasks` | liste d’objets | Oui | Chaque entrée ajuste périodicité et prochaine exécution d’une tâche connue.【F:AGI_Evolutive/memory/semantic_memory_manager.py†L361-L418】 |
| `tasks[*].task` | alias reconnu | Oui | Doit correspondre à une tâche enregistrée, sinon l’entrée est ignorée.【F:AGI_Evolutive/memory/semantic_memory_manager.py†L389-L405】 |
| `tasks[*].category` | `str` (`urgent`, `court_terme`, `long_terme`, …) | Oui | Détermine la stratégie d’ajustement des périodes.【F:AGI_Evolutive/memory/semantic_memory_manager.py†L403-L417】 |
| `tasks[*].period`/`next_run` | nombres | Non | Valeurs recalculées à partir de la catégorie ; reprises dans l’historique si présentes.【F:AGI_Evolutive/memory/semantic_memory_manager.py†L401-L418】 |

**Vigilance** : passer des tâches inconnues laisse la guidance vide ; toujours réemployer les alias
renvoyés dans le payload entrant.

## Langage

### `dialogue_state` — `AGI_Evolutive/language/dialogue_state.py`

| Champ | Type attendu | Obligatoire | Effet runtime |
| --- | --- | --- | --- |
| `state_summary`/`summary` | `str` | Oui | Alimentent la synthèse courante du dialogue.【F:AGI_Evolutive/language/dialogue_state.py†L123-L138】 |
| `open_commitments` | liste | Oui | Remplace la liste d’engagements suivis ; utiliser le même schéma que l’entrée.【F:AGI_Evolutive/language/dialogue_state.py†L125-L129】 |
| `pending_questions` | liste | Oui | S’injecte dans la file locale ; une omission vide la liste courante.【F:AGI_Evolutive/language/dialogue_state.py†L125-L129】 |
| `notes` | `str` | Non | Ajoutées si non vides pour contextualiser l’état.【F:AGI_Evolutive/language/dialogue_state.py†L129-L132】 |

### `language_nlg` — `AGI_Evolutive/language/nlg.py`

| Champ | Type attendu | Obligatoire | Effet runtime |
| --- | --- | --- | --- |
| `introduction` | `str` | Non | Première section injectée dans le texte final si non vide.【F:AGI_Evolutive/language/nlg.py†L117-L131】 |
| `body` | `str` | Non | Corps principal de la réponse structurée.【F:AGI_Evolutive/language/nlg.py†L117-L131】 |
| `conclusion` | `str` | Non | Clôture la réponse ; absence tolérée.【F:AGI_Evolutive/language/nlg.py†L117-L131】 |

**Remarque** : au moins une section doit être fournie pour modifier la sortie. Sinon, le texte de
base est conservé.

## Mémoire épisodique

### `episodic_linker` — `AGI_Evolutive/memory/episodic_linker.py`

| Champ | Type attendu | Obligatoire | Effet runtime |
| --- | --- | --- | --- |
| `links` | liste d’objets | Non | Ajoute des relations supplémentaires entre souvenirs si valides.【F:AGI_Evolutive/memory/episodic_linker.py†L553-L615】 |
| `links[*].from`/`src` | alias fourni (`m0`, `m1`, …) ou id existant | Oui (par lien) | Converti via `alias_map` vers l’identifiant réel.【F:AGI_Evolutive/memory/episodic_linker.py†L585-L607】 |
| `links[*].to`/`dst` | alias/id | Oui | Identique à `from` ; rejeté si introuvable.【F:AGI_Evolutive/memory/episodic_linker.py†L585-L607】 |
| `links[*].type_lien`/`relation` | `str` | Oui | Converti via `LLM_RELATION_MAP` vers `CAUSES`, `SUPPORTS`, etc.【F:AGI_Evolutive/memory/episodic_linker.py†L588-L609】 |
| `links[*].confidence` | nombre ∈ [0, 1] | Non | Conservé si convertible ; sinon omis.【F:AGI_Evolutive/memory/episodic_linker.py†L607-L614】 |
| `notes` | `str` | Non | Archivé avec les liens générés pour inspection.【F:AGI_Evolutive/memory/episodic_linker.py†L613-L617】 |

**Conseil** : rester fidèle aux alias fournis (`m0`, `m1`…) décrivant les expériences internes au
lieu de réintroduire des intitulés techniques obsolètes.

## Identité & cognition

### `identity_mission` — `AGI_Evolutive/cognition/identity_mission.py`

| Champ | Type attendu | Obligatoire | Effet runtime |
| --- | --- | --- | --- |
| `mission` | mapping | Oui | Doit contenir les axes `prioritaire`, `support`, `vision` (ou alias) pour mettre à jour les axes internes.【F:AGI_Evolutive/cognition/identity_mission.py†L534-L556】 |
| `mission_text`/`mission_statement` | `str` | Non | Remplace la proposition sélectionnée si non vide.【F:AGI_Evolutive/cognition/identity_mission.py†L557-L562】 |
| `notes` | `str` | Non | Journalisées dans l’état et renvoyées à l’appelant.【F:AGI_Evolutive/cognition/identity_mission.py†L562-L607】 |

**Conseil** : toujours réutiliser le candidat `selected` fourni dans le payload pour garantir la
cohérence score/delta.

### `cognition_proposer` — `AGI_Evolutive/cognition/proposer.py`

| Champ | Type attendu | Obligatoire | Effet runtime |
| --- | --- | --- | --- |
| `suggestions` | liste d’objets | Non | Remplace la liste de propositions heuristiques si fournie.【F:AGI_Evolutive/cognition/proposer.py†L35-L64】 |
| `suggestions[*].type` | `str` | Oui (par suggestion) | Définit le type d’opération (`update`, `add`, …).【F:AGI_Evolutive/cognition/proposer.py†L55-L58】 |
| `suggestions[*].path`/`target` | liste/chemin | Oui | Sert à appliquer la proposition dans le self-model.【F:AGI_Evolutive/cognition/proposer.py†L55-L58】 |
| `suggestions[*].value`/`adjustment` | valeur sérialisable | Oui | Appliquée telle quelle lors de l’intégration.【F:AGI_Evolutive/cognition/proposer.py†L55-L58】 |
| `suggestions[*].cause` | `str` | Non | Documente la justification de la proposition.【F:AGI_Evolutive/cognition/proposer.py†L55-L60】 |
| `notes` | `str` | Non | Ajoutée comme proposition de type `note` pour archivage.【F:AGI_Evolutive/cognition/proposer.py†L64-L66】 |

### `cognition_goal_prioritizer` — `AGI_Evolutive/cognition/prioritizer.py`

| Champ | Type attendu | Obligatoire | Effet runtime |
| --- | --- | --- | --- |
| `priority` | nombre ∈ [0, 1] | Non | Écrase le score heuristique après clamp.【F:AGI_Evolutive/cognition/prioritizer.py†L1008-L1032】 |
| `tags` | liste | Non | Nettoyée puis fusionnée pour enrichir la classification.【F:AGI_Evolutive/cognition/prioritizer.py†L1033-L1038】 |
| `explain` | liste de chaînes | Non | Limité à six éléments et exposé au tableau de bord.【F:AGI_Evolutive/cognition/prioritizer.py†L1039-L1044】 |
| `confidence` | nombre ∈ [0, 1] | Non | Pondère le mélange entre score LLM et heuristique.【F:AGI_Evolutive/cognition/prioritizer.py†L1045-L1054】 |
| `notes` | `str` | Non | Ajouté tel quel au résultat consolidé.【F:AGI_Evolutive/cognition/prioritizer.py†L1055-L1057】 |

### `planner_support` — `AGI_Evolutive/cognition/planner.py`

| Champ | Type attendu | Obligatoire | Effet runtime |
| --- | --- | --- | --- |
| `plan` | liste d’objets | Oui | Chaque entrée devient une étape `todo` dans le plan.【F:AGI_Evolutive/cognition/planner.py†L348-L391】 |
| `plan[*].description` | `str` non vide | Oui | Sert de texte principal pour l’étape créée.【F:AGI_Evolutive/cognition/planner.py†L360-L376】 |
| `plan[*].id` | `str` | Non | Utilisé si fourni, sinon généré automatiquement.【F:AGI_Evolutive/cognition/planner.py†L368-L372】 |
| `plan[*].depends_on` | liste de `str` | Non | Filtrée et assignée directement au plan.【F:AGI_Evolutive/cognition/planner.py†L372-L379】 |
| `plan[*].priority` | nombre ∈ [0, 1] | Non | Clampé puis stocké comme priorité d’exécution.【F:AGI_Evolutive/cognition/planner.py†L379-L384】 |
| `plan[*].action_type` | `str` | Non | Définit le type d’action ; sinon déterminé heuristiquement.【F:AGI_Evolutive/cognition/planner.py†L384-L389】 |
| `plan[*].context` | mapping | Non | Ajouté au payload d’action si présent.【F:AGI_Evolutive/cognition/planner.py†L386-L393】 |
| `risks` | valeur libre | Non | Conservé tel quel dans l’objet plan.【F:AGI_Evolutive/cognition/planner.py†L394-L401】 |
| `notes` | valeur libre | Non | Idem, archivé pour l’opérateur.【F:AGI_Evolutive/cognition/planner.py†L394-L401】 |

### `trigger_router` — `AGI_Evolutive/cognition/trigger_router.py`

| Champ | Type attendu | Obligatoire | Effet runtime |
| --- | --- | --- | --- |
| `pipelines` | liste de `str` | Non | Définit l’ordre de priorité des pipelines ; le premier est choisi.【F:AGI_Evolutive/cognition/trigger_router.py†L37-L57】 |
| `secondary` | liste de `str` | Non | Consigné pour bascule ultérieure.【F:AGI_Evolutive/cognition/trigger_router.py†L37-L64】 |
| `notes` | `str` | Non | Journalisé pour expliquer la décision du routeur.【F:AGI_Evolutive/cognition/trigger_router.py†L43-L64】 |

## Objectifs

### `goal_interpreter` — `AGI_Evolutive/goals/heuristics.py`

| Champ | Type attendu | Obligatoire | Effet runtime |
| --- | --- | --- | --- |
| `candidate_actions` | liste d’objets | Oui | Génère la file d’actions dérivée du but courant.【F:AGI_Evolutive/goals/heuristics.py†L120-L152】 |
| `candidate_actions[*].action` | `str` | Oui | Devient `type` de l’action ajoutée à la deque.【F:AGI_Evolutive/goals/heuristics.py†L132-L147】 |
| `candidate_actions[*].rationale` | `str` | Non | Copiée dans le payload d’action pour la traçabilité.【F:AGI_Evolutive/goals/heuristics.py†L138-L145】 |
| `normalized_goal` | `str` | Non | Stocké dans chaque action créée pour contextualiser le but.【F:AGI_Evolutive/goals/heuristics.py†L128-L143】 |
| `notes` | `str` | Non | Injecté dans la première action générée.【F:AGI_Evolutive/goals/heuristics.py†L145-L147】 |

### `goal_priority_review` — `AGI_Evolutive/goals/dag_store.py`

| Champ | Type attendu | Obligatoire | Effet runtime |
| --- | --- | --- | --- |
| `priority` | nombre ∈ [0, 1] | Oui (sauf usage de `priority_delta`) | Recalcule la priorité du nœud, clampée entre 0 et 1.【F:AGI_Evolutive/goals/dag_store.py†L206-L236】 |
| `priority_delta` | nombre | Oui (alternative) | Ajoute un delta au fallback si `priority` absent.【F:AGI_Evolutive/goals/dag_store.py†L224-L236】 |
| `confidence` | nombre ∈ [0, 1] | Non | Pondère le mélange entre fallback et suggestion LLM.【F:AGI_Evolutive/goals/dag_store.py†L232-L236】 |

**Vigilance** : si ni `priority` ni `priority_delta` ne sont fournis, la revue est purement ignorée.

## Interaction & questionnement

### `question_manager` — `AGI_Evolutive/core/question_manager.py`

| Champ | Type attendu | Obligatoire | Effet runtime |
| --- | --- | --- | --- |
| `questions` | liste d’objets | Oui | Chaque élément validé alimente la banque et la file d’attente.【F:AGI_Evolutive/core/question_manager.py†L344-L377】 |
| `questions[*].text` | `str` non vide | Oui | Question réellement poussée ; les doublons sont filtrés ensuite.【F:AGI_Evolutive/core/question_manager.py†L360-L377】 |
| autres champs (`insights`, `meta`, …) | sérialisable | Non | Repris tels quels si conformes au format courant.【F:AGI_Evolutive/core/question_manager.py†L344-L377】 |

### `question_auto_answer` — `AGI_Evolutive/core/question_manager.py`

| Champ | Type attendu | Obligatoire | Effet runtime |
| --- | --- | --- | --- |
| `answer`/`text` | `str` non vide | Oui | Crée une suggestion d’auto-réponse avec source `llm`.【F:AGI_Evolutive/core/question_manager.py†L728-L754】 |
| `confidence` | nombre | Non | Clampé puis enregistré pour décider de l’auto-validation.【F:AGI_Evolutive/core/question_manager.py†L749-L756】 |
| `concepts`/`keywords`/`insights` | liste | Non | Nettoyés et ajoutés au rapport de réponse.【F:AGI_Evolutive/core/question_manager.py†L756-L767】 |
| `notes` | `str` | Non | Ajoutés en champ libre dans la suggestion.【F:AGI_Evolutive/core/question_manager.py†L767-L769】 |

## Action

### `action_interface` — `AGI_Evolutive/io/action_interface.py`

| Champ | Type attendu | Obligatoire | Effet runtime |
| --- | --- | --- | --- |
| `actions` | liste d’objets | Oui | Liste évaluée ; chaque élément remplace ou complète l’estimation heuristique.【F:AGI_Evolutive/io/action_interface.py†L540-L590】 |
| `actions[*].name` | `str` | Oui | Sert d’étiquette d’action (fallback si absent).【F:AGI_Evolutive/io/action_interface.py†L552-L566】 |
| `actions[*].type` | `str` | Oui | Identifie la catégorie d’action pour le moteur d’exécution.【F:AGI_Evolutive/io/action_interface.py†L552-L566】 |
| `actions[*].impact`/`effort`/`risk` | nombres ∈ [0, 1] | Non | Clampés puis arrondis à trois décimales avant restitution.【F:AGI_Evolutive/io/action_interface.py†L566-L589】 |
| `actions[*].rationale` | `str` | Non | Remplace la justification heuristique si fournie.【F:AGI_Evolutive/io/action_interface.py†L552-L589】 |
| `notes` | `str` | Non | Ajoutées au rapport final pour guider l’opérateur.【F:AGI_Evolutive/io/action_interface.py†L571-L589】 |

**Astuces de revue**
- Vérifie systématiquement que les identifiants (`id`, alias LLM, chemins `path`) proviennent bien du
payload d’entrée.
- Les valeurs non sérialisables (objets Python, `set`, etc.) sont silencieusement écartées ; rester sur
un sous-ensemble JSON.
- Les champs indiqués « obligatoires » peuvent désactiver tout le bénéfice de l’appel si absents :
un test rapide via mocks LLM permet de s’en assurer avant déploiement.
