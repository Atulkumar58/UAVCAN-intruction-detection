"""Train a Random Forest for UAVCAN attack detection.

The model predicts a binary target: 0 is a normal frame and 1 is any attack
type. It accepts the numeric ``uavcan_processed.npz`` output from
``preprocess_uavcan.py`` or the human-readable CSV output.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from train_uavcan_model import (
    DEFAULT_FEATURES,
    load_data,
    metrics,
    stratified_split,
)


def save_model(
    path: Path,
    model: RandomForestClassifier,
    feature_names: list[str],
    threshold: float,
) -> None:
    """Save the forest and the metadata required for later inference."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        pickle.dump(
            {
                "model": model,
                "feature_names": feature_names,
                "threshold": threshold,
            },
            output,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def main() -> None:
    """Train, evaluate, and save a binary Random Forest attack detector."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("processed_data/uavcan_processed.npz"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("processed_data/uavcan_random_forest_model.pkl"),
    )
    parser.add_argument("--trees", type=int, default=200, help="Number of decision trees.")
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if args.trees <= 0:
        raise ValueError("--trees must be positive.")
    if args.max_depth is not None and args.max_depth <= 0:
        raise ValueError("--max-depth must be positive when provided.")
    if args.min_samples_leaf <= 0:
        raise ValueError("--min-samples-leaf must be positive.")
    if not 0 < args.validation_fraction + args.test_fraction < 1:
        raise ValueError("Validation and test fractions must be positive and sum to less than 1.")
    if not 0 < args.threshold < 1:
        raise ValueError("--threshold must be between 0 and 1.")

    features, labels, feature_names = load_data(args.input)
    if feature_names != DEFAULT_FEATURES:
        raise ValueError("Input features do not match the expected UAVCAN feature set.")
    if len(labels) == 0:
        raise ValueError("Input contains no samples.")

    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive.")
        if args.max_samples < len(labels):
            rng = np.random.default_rng(args.seed)
            selected = rng.choice(len(labels), size=args.max_samples, replace=False)
            features, labels = features[selected], labels[selected]

    train_indices, validation_indices, test_indices = stratified_split(
        labels, args.validation_fraction, args.test_fraction, args.seed
    )
    model = RandomForestClassifier(
        n_estimators=args.trees,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=-1,
    )
    model.fit(features[train_indices], labels[train_indices].astype(np.int64))

    validation_probabilities = model.predict_proba(features[validation_indices])[:, 1]
    validation_metrics = metrics(labels[validation_indices], validation_probabilities, args.threshold)
    test_probabilities = model.predict_proba(features[test_indices])[:, 1]
    test_metrics = metrics(labels[test_indices], test_probabilities, args.threshold)

    save_model(args.output, model, feature_names, args.threshold)
    metrics_path = args.output.with_name(f"{args.output.stem}_metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "input": str(args.input),
                "feature_names": feature_names,
                "binary_labels": {"0": "Normal", "1": "Attack"},
                "train_samples": len(train_indices),
                "validation_samples": len(validation_indices),
                "test_samples": len(test_indices),
                "trees": args.trees,
                "max_depth": args.max_depth,
                "min_samples_leaf": args.min_samples_leaf,
                "threshold": args.threshold,
                "validation_metrics": validation_metrics,
                "test_metrics": test_metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"validation: accuracy={validation_metrics['accuracy']:.4f} "
        f"precision={validation_metrics['precision']:.4f} "
        f"recall={validation_metrics['recall']:.4f} f1={validation_metrics['f1']:.4f}"
    )
    print(
        f"test: accuracy={test_metrics['accuracy']:.4f} "
        f"precision={test_metrics['precision']:.4f} "
        f"recall={test_metrics['recall']:.4f} f1={test_metrics['f1']:.4f}"
    )
    print(f"saved model: {args.output}")
    print(f"saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
