# Reasoning deliberation flow

This document summarizes how the `ReasoningSystem` evaluates possible follow-up actions during a reasoning episode.

## Preference snapshot

* `ReasoningSystem._self_preferences()` queries the architecture's `self_model` for stored preferences, rules, and decision history.  It normalizes values such as autonomy, caution, and verification biases, tracks recent decisions, and records storage metadata so future episodes reuse the same location.  The snapshot is refreshed with a timestamp for every call, ensuring the latest preferences are applied.
* User attitude priors (e.g., willingness to be asked for help) are collected via `ReasoningSystem._user_preference()` from the architecture's `user_model`.
* A token-level feedback lexicon is loaded from the `self_model` (with a small seeded vocabulary as fallback).  Each token carries a learned weight so the agent can generalize beyond fixed phrases like “c’est chiant” and reuse newly discovered expressions in later turns.

## Option evaluation

`ReasoningSystem._deliberate_actions()` builds four options—continue reasoning alone, scan memory, ask the user, or ask the question manager.  For each option it:

1. Calculates success probability and cost, adjusting for cognitive load, uncertainty, inbox size, and feedback signals collected at runtime via `_collect_feedback_signals` (context events, stored rules, sentiment heuristics) and `_apply_feedback_signals`.
2. Applies preference weights derived from the self model (autonomy, caution, verification) so different agents can favor different strategies.
3. Estimates the expected utility (`weight * success - cost`) and annotates every option with natural-language notes, including summaries of the applied feedback.
4. Produces a cause–effect justification and a time estimate for the option's execution, aligning with the requested reasoning style.

## Decision storage and reuse

After selecting the best option, the system:

* Adds an explanation string that combines cost/benefit comparisons, cause–effect rationale, and any cooldown guidance.
* Registers the decision with the `self_model` (via `register_decision`) and persists updated rules, such as the cooldown for asking the user again, together with the feedback trace (signals, adjustment, prior) that led to the current choice.  It also folds extracted feedback tokens (drawn from the prompt or context payloads) back into the shared lexicon so fresh expressions influence later episodes.  These persisted rules and token weights are read back during the next `ReasoningSystem._self_preferences()` call so future episodes reuse the learned cause–effect relations.

The result is a reusable deliberation trace where the agent's preferences, user sentiment, time estimates, and cooldown rules influence both the current choice and future reasoning episodes.
