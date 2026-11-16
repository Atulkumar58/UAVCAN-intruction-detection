"""Train a Keras 1D CNN for binary UAVCAN attack detection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import keras
import numpy as np

from train_uavcan_model import load_data, metrics, stratified_split
from uavcan_cnn import build_model, predict_proba, save_artifact


def main() -> None:
    """Train, evaluate, and save the Keras CNN and its metadata."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("processed_data/uavcan_processed.npz"))
    parser.add_argument("--output", type=Path, default=Path("processed_data/uavcan_attack_cnn.keras"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--max-samples", type=int, help="Use a random subset for quick experiments.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        raise ValueError("--epochs, --batch-size, and --learning-rate must be positive.")
    if not 0 < args.validation_fraction + args.test_fraction < 1:
        raise ValueError("Validation and test fractions must be positive and sum to less than 1.")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive.")
    if not 0 < args.threshold < 1:
        raise ValueError("--threshold must be between 0 and 1.")

    keras.utils.set_random_seed(args.seed)
    features, labels, feature_names = load_data(args.input)
    if args.max_samples is not None and args.max_samples < len(labels):
        rng = np.random.default_rng(args.seed)
        selected = rng.choice(len(labels), size=args.max_samples, replace=False)
        features, labels = features[selected], labels[selected]
    if len(labels) == 0:
        raise ValueError("Input contains no samples.")

    train_indices, validation_indices, test_indices = stratified_split(
        labels, args.validation_fraction, args.test_fraction, args.seed
    )
    mean = features[train_indices].mean(axis=0)
    scale = features[train_indices].std(axis=0)
    scale[scale < 1e-6] = 1.0
    standardized = ((features - mean) / scale).astype(np.float32)
    train_features = standardized[train_indices, ..., np.newaxis]
    validation_features = standardized[validation_indices, ..., np.newaxis]
    train_labels = labels[train_indices]
    validation_labels = labels[validation_indices]
    positive_weight = float(np.sum(train_labels == 0) / max(1, np.sum(train_labels == 1)))

    model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy"), keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")],
    )
    model.fit(
        train_features,
        train_labels,
        validation_data=(validation_features, validation_labels),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight={0: 1.0, 1: positive_weight},
        shuffle=True,
        verbose=2,
    )

    test_probabilities = predict_proba(
        model, features[test_indices], mean, scale, args.batch_size
    )
    test_metrics = metrics(labels[test_indices], test_probabilities, args.threshold)
    sidecar = save_artifact(model_path=args.output, model=model, mean=mean, scale=scale,
                            feature_names=feature_names, threshold=args.threshold)
    metrics_path = args.output.with_name(f"{args.output.stem}_metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "model": str(args.output),
                "input": str(args.input),
                "feature_names": feature_names,
                "binary_labels": {"0": "Normal", "1": "Attack"},
                "train_samples": len(train_indices),
                "validation_samples": len(validation_indices),
                "test_samples": len(test_indices),
                "positive_weight": positive_weight,
                "threshold": args.threshold,
                "test_metrics": test_metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"test: accuracy={test_metrics['accuracy']:.4f} precision={test_metrics['precision']:.4f} "
        f"recall={test_metrics['recall']:.4f} f1={test_metrics['f1']:.4f}"
    )
    print(f"saved model: {args.output}")
    print(f"saved metadata: {sidecar}")
    print(f"saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
