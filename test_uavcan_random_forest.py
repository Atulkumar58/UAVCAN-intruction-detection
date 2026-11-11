"""Evaluate a saved Random Forest UAVCAN attack-detection model."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from train_uavcan_model import DEFAULT_FEATURES, load_data, metrics


def load_model(path: Path) -> tuple[RandomForestClassifier, list[str], float]:
    """Load and validate a saved Random Forest artifact."""

    with path.open("rb") as source:
        artifact: Any = pickle.load(source)
    if not isinstance(artifact, dict):
        raise ValueError("Random Forest model artifact must contain a metadata dictionary.")
    model = artifact.get("model")
    feature_names = artifact.get("feature_names")
    threshold = artifact.get("threshold")
    if not isinstance(model, RandomForestClassifier):
        raise ValueError("Model artifact does not contain a RandomForestClassifier.")
    if feature_names != DEFAULT_FEATURES:
        raise ValueError(
            f"Unexpected model features. Expected {DEFAULT_FEATURES}, got {feature_names}."
        )
    if not isinstance(threshold, (float, int)) or not 0 < threshold < 1:
        raise ValueError("Model threshold must be between 0 and 1.")
    if getattr(model, "n_features_in_", None) != len(DEFAULT_FEATURES):
        raise ValueError("Random Forest model must accept 11 numeric features.")
    return model, feature_names, float(threshold)


def write_predictions(
    path: Path, probabilities: np.ndarray, labels: np.ndarray, threshold: float
) -> None:
    """Write one prediction and the expected label for every input row."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["row", "attack_probability", "prediction", "actual"])
        for index, (probability, label) in enumerate(zip(probabilities, labels)):
            writer.writerow(
                [index, f"{probability:.8f}", int(probability >= threshold), int(label)]
            )


def main() -> None:
    """Evaluate the saved model and report classification metrics."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("processed_data/uavcan_random_forest_model.pkl"),
    )
    parser.add_argument("--input", type=Path, default=Path("processed_data/uavcan_processed.npz"))
    parser.add_argument("--output", type=Path, help="Optional CSV path for row-level predictions.")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()

    model, feature_names, saved_threshold = load_model(args.model)
    features, labels, input_feature_names = load_data(args.input)
    if input_feature_names != feature_names:
        raise ValueError("Input feature names do not match the saved model.")
    if len(features) == 0:
        raise ValueError("Input contains no samples.")
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive.")
        features, labels = features[: args.max_samples], labels[: args.max_samples]

    threshold = saved_threshold if args.threshold is None else args.threshold
    if not 0 < threshold < 1:
        raise ValueError("--threshold must be between 0 and 1.")
    probabilities = model.predict_proba(features)[:, 1]
    result = metrics(labels, probabilities, threshold)

    print(f"samples: {len(features)}")
    print(f"threshold: {threshold:.4f}")
    print(
        f"accuracy={result['accuracy']:.4f} precision={result['precision']:.4f} "
        f"recall={result['recall']:.4f} f1={result['f1']:.4f}"
    )
    print(
        f"true_positive={result['true_positive']} true_negative={result['true_negative']} "
        f"false_positive={result['false_positive']} false_negative={result['false_negative']}"
    )

    if args.output:
        write_predictions(args.output, probabilities, labels, threshold)
        print(f"saved predictions: {args.output}")

    metrics_path = args.model.with_name(f"{args.model.stem}_test_metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "model": str(args.model),
                "input": str(args.input),
                "samples": len(features),
                "threshold": threshold,
                "metrics": result,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved test metrics: {metrics_path}")


if __name__ == "__main__":
    main()
