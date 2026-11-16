"""Evaluate a saved Keras 1D CNN on labeled UAVCAN data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from train_uavcan_model import load_data, metrics
from uavcan_cnn import load_artifact, predict_proba


def write_predictions(path: Path, probabilities: np.ndarray, labels: np.ndarray, threshold: float) -> None:
    """Write row-level CNN predictions and expected labels."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["row", "attack_probability", "prediction", "actual"])
        for index, (probability, label) in enumerate(zip(probabilities, labels)):
            writer.writerow([index, f"{probability:.8f}", int(probability >= threshold), int(label)])


def main() -> None:
    """Evaluate the saved CNN and write its metrics."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("processed_data/uavcan_attack_cnn.keras"))
    parser.add_argument("--input", type=Path, default=Path("processed_data/uavcan_processed.npz"))
    parser.add_argument("--output", type=Path, help="Optional CSV path for row-level predictions.")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-samples", type=int, help="Evaluate only a prefix of the input.")
    parser.add_argument("--threshold", type=float, help="Override the threshold saved with the model.")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive.")

    model, mean, scale, feature_names, saved_threshold = load_artifact(args.model)
    features, labels, input_feature_names = load_data(args.input)
    if input_feature_names != feature_names:
        raise ValueError("Input feature names do not match the saved model.")
    if len(features) == 0:
        raise ValueError("Input contains no samples.")
    if args.max_samples is not None:
        features, labels = features[: args.max_samples], labels[: args.max_samples]

    threshold = saved_threshold if args.threshold is None else args.threshold
    if not 0 < threshold < 1:
        raise ValueError("--threshold must be between 0 and 1.")
    probabilities = predict_proba(model, features, mean, scale, args.batch_size)
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
    summary_path = args.model.with_name(f"{args.model.stem}_test_metrics.json")
    summary_path.write_text(
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
    print(f"saved test metrics: {summary_path}")


if __name__ == "__main__":
    main()
