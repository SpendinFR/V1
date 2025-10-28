# Intent Classifier Maintenance Guide

## Heuristic rules
- Patterns are declared in `AGI_Evolutive/io/intent_patterns_fr.json`.
- Update the lists when adding new synonyms or idioms.
- Keep entries normalized (lowercase, no accents) to align with `normalize_text`.

## Fallback model
- Data lives in `data/intent_classifier_training_fr.json`.
- Retrain with `python scripts/retrain_intent_classifier.py`.
- Output is written to `AGI_Evolutive/io/models/intent_classifier_fallback_fr.json`.
- The fallback uses a Naive Bayes model on token n-grams and returns INFO when confidence < 0.45.

## Continuous learning loop
1. Monitor `data/intent_classifier_feedback.log` for uncertain predictions.
2. Curate misclassified rows by assigning a `label` and appending them to a dataset JSON file.
3. Re-run the retraining script with `--dataset path/to/new_samples.json`.
4. Commit the updated dataset and model.
5. Run the French intent tests with `pytest tests/test_intent_classifier_fr.py`.

## Testing
- Unit tests: `pytest tests/test_intent_classifier_fr.py`.
- Add new examples whenever heuristics or fallback behaviour change.

