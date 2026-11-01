"""Evaluate a saved UAVCAN attack-detection model on labeled data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from train_uavcan_model import DEFAULT_FEATURES, MLP, load_data, metrics


def load_model(path: Path) -> tuple[MLP, np.ndarray, np.ndarray, list[str], float]:
    """Load model parameters and preprocessing metadata from an NPZ artifact."""

    with np.load(path, allow_pickle=False) as data:
        feature_names = [str(name) for name in data["feature_names"]]
        mean = np.asarray(data["mean"], dtype=np.float32)
        scale = np.asarray(data["scale"], dtype=np.float32)
        threshold = float(np.asarray(data["threshold"]).item())
        weights = [np.asarray(data[f"weight_{index}"], dtype=np.float32) for index in range(3)]
        biases = [np.asarray(data[f"bias_{index}"], dtype=np.float32) for index in range(3)]

    if feature_names != DEFAULT_FEATURES:
        raise ValueError(
            f"Unexpected model features. Expected {DEFAULT_FEATURES}, got {feature_names}."
        )
    if mean.shape != (len(DEFAULT_FEATURES),) or scale.shape != mean.shape:
        raise ValueError("Model mean and scale must each contain 11 values.")
    if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("Model normalization statistics must be finite and positive.")
    if not 0 < threshold < 1:
        raise ValueError("Model threshold must be between 0 and 1.")
    if [weight.shape for weight in weights] != [(11, 64), (64, 32), (32, 1)]:
        raise ValueError("Model weights do not match the expected 11-64-32-1 architecture.")
    if [bias.shape for bias in biases] != [(1, 64), (1, 32), (1, 1)]:
        raise ValueError("Model biases do not match the expected 11-64-32-1 architecture.")

    model = MLP(input_size=11, hidden_sizes=(64, 32), seed=0)
    model.weights = weights
    model.biases = biases
    return model, mean, scale, feature_names, threshold


def predict_in_batches(
    model: MLP, features: np.ndarray, mean: np.ndarray, scale: np.ndarray, batch_size: int
) -> np.ndarray:
    """Standardize features and predict without loading duplicate batches."""

    probabilities = np.empty(len(features), dtype=np.float32)
    for start in range(0, len(features), batch_size):
        batch = ((features[start : start + batch_size] - mean) / scale).astype(np.float32)
        probabilities[start : start + len(batch)] = model.predict_proba(batch)
    return probabilities


def write_predictions(path: Path, probabilities: np.ndarray, labels: np.ndarray, threshold: float) -> None:
    """Write one prediction and, when available, the expected label per row."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["row", "attack_probability", "prediction", "actual"])
        for index, (probability, label) in enumerate(zip(probabilities, labels)):
            writer.writerow(
                [
                    index,
                    f"{probability:.8f}",
                    int(probability >= threshold),
                    int(label),
                ]
            )


def main() -> None:
    """Evaluate the saved model and report classification metrics."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("processed_data/uavcan_attack_model.npz"))
    parser.add_argument("--input", type=Path, default=Path("processed_data/uavcan_processed.npz"))
    parser.add_argument("--output", type=Path, help="Optional CSV path for row-level predictions.")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-samples", type=int, help="Evaluate only a reproducible prefix of the input.")
    parser.add_argument("--threshold", type=float, help="Override the threshold saved with the model.")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive.")

    model, mean, scale, feature_names, saved_threshold = load_model(args.model)
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
    probabilities = predict_in_batches(model, features, mean, scale, args.batch_size)
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
