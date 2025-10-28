#!/usr/bin/env python3
"""Train or retrain the French intent fallback classifier."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from AGI_Evolutive.io.intent_classifier import normalize_text
DEFAULT_DATASET = REPO_ROOT / "data" / "intent_classifier_training_fr.json"
DEFAULT_OUTPUT = REPO_ROOT / "AGI_Evolutive" / "io" / "models" / "intent_classifier_fallback_fr.json"
TOKEN_PATTERN = re.compile(r"[\w']+|[\?\!]+|[\U0001f300-\U0001fadf]")
ALPHA = 1.0


def load_examples(paths: Sequence[Path]) -> Tuple[List[str], List[str]]:
    texts: List[str] = []
    labels: List[str] = []
    for path in paths:
        with path.open(encoding="utf-8") as fh:
            payload = json.load(fh)
        for item in payload:
            text = normalize_text(item["text"])
            label = item["label"].upper()
            texts.append(text)
            labels.append(label)
    return texts, labels


def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text)


def train(texts: Sequence[str], labels: Sequence[str]) -> dict:
    classes = sorted(set(labels))
    class_counts = Counter(labels)
    token_counts = {label: Counter() for label in classes}
    vocabulary = set()

    for text, label in zip(texts, labels):
        tokens = tokenize(text)
        vocabulary.update(tokens)
        token_counts[label].update(tokens)

    vocabulary_list = sorted(vocabulary)
    log_prior = {}
    token_log_prob = {label: {} for label in classes}
    unknown_log_prob = {}
    total_docs = len(labels)

    for label in classes:
        log_prior[label] = math.log((class_counts[label] + ALPHA) / (total_docs + ALPHA * len(classes)))
        total_tokens = sum(token_counts[label].values()) + ALPHA * len(vocabulary_list)
        unknown_log_prob[label] = math.log(ALPHA / total_tokens)
        for token in vocabulary_list:
            token_log_prob[label][token] = math.log((token_counts[label][token] + ALPHA) / total_tokens)

    return {
        "classes": classes,
        "vocabulary": vocabulary_list,
        "log_prior": log_prior,
        "token_log_prob": token_log_prob,
        "unknown_log_prob": unknown_log_prob,
        "alpha": ALPHA,
        "token_pattern": TOKEN_PATTERN.pattern,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        type=Path,
        help="Additional dataset JSON files with {text,label} entries.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination for the serialized fallback model.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    dataset_paths = [DEFAULT_DATASET]
    if args.dataset:
        dataset_paths.extend(args.dataset)

    texts, labels = load_examples(dataset_paths)
    model = train(texts, labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(model, fh, ensure_ascii=False, indent=2)
    print(f"Saved fallback classifier to {args.output}")


if __name__ == "__main__":
    main()
