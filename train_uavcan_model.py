"""Train a small NumPy neural network for UAVCAN attack detection.

The model predicts a binary target: 0 is a normal frame and 1 is any attack
type. It accepts the numeric ``uavcan_processed.npz`` output from
``preprocess_uavcan.py`` or the human-readable CSV output.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DEFAULT_FEATURES = [
    "timestamp",
    "can_id_decimal",
    "dlc",
    "data_decimal_0",
    "data_decimal_1",
    "data_decimal_2",
    "data_decimal_3",
    "data_decimal_4",
    "data_decimal_5",
    "data_decimal_6",
    "data_decimal_7",
]


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load numeric features and convert multiclass labels to Normal/Attack."""

    with np.load(path, allow_pickle=False) as data:
        features = np.asarray(data["X"], dtype=np.float32)
        labels = np.asarray(data["y"], dtype=np.int64)
        feature_names = [str(name) for name in data["feature_names"]]
    if feature_names != DEFAULT_FEATURES:
        raise ValueError(
            f"Unexpected preprocessed features. Expected {DEFAULT_FEATURES}, got {feature_names}."
        )
    if features.ndim != 2 or features.shape[1] != len(DEFAULT_FEATURES):
        raise ValueError("Preprocessed feature matrix must have 11 numeric columns.")
    return features, (labels > 0).astype(np.float32), feature_names


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load the six-field preprocessor CSV and expand payloads to eight features."""

    rows: list[list[float]] = []
    labels: list[float] = []
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = ["timestamp", "interface", "can_id_decimal", "dlc", "data_decimal", "label"]
        missing = [name for name in required if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"CSV is missing columns: {', '.join(missing)}")
        for row in reader:
            data = json.loads(row["data_decimal"] or "[]")
            dlc = int(row["dlc"] or 0)
            if not isinstance(data, list) or len(data) != dlc or len(data) > 8:
                raise ValueError("data_decimal must contain exactly the number of bytes declared by dlc")
            if any(not isinstance(byte, int) or not 0 <= byte <= 255 for byte in data):
                raise ValueError("data_decimal must contain integer byte values from 0 through 255")
            rows.append(
                [
                    float(row["timestamp"] or 0.0),
                    float(row["can_id_decimal"] or 0.0),
                    float(dlc),
                    *[float(byte) for byte in data],
                    *([0.0] * (8 - len(data))),
                ]
            )
            labels.append(float(int(row["label"]) > 0))
    return np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=np.float32), DEFAULT_FEATURES


def load_data(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load either the preprocessor NPZ output or its six-field CSV output."""

    if path.suffix.lower() == ".npz":
        return load_npz(path)
    if path.suffix.lower() == ".csv":
        return load_csv(path)
    raise ValueError("Input must be a .npz or .csv file.")


def stratified_split(
    labels: np.ndarray, validation_fraction: float, test_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create reproducible train, validation, and test indices per binary class."""

    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    validation_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for label in (0.0, 1.0):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        test_count = max(1, int(len(indices) * test_fraction))
        validation_count = max(1, int(len(indices) * validation_fraction))
        test_parts.append(indices[:test_count])
        validation_parts.append(indices[test_count : test_count + validation_count])
        train_parts.append(indices[test_count + validation_count :])

    def shuffled(parts: list[np.ndarray]) -> np.ndarray:
        combined = np.concatenate(parts)
        rng.shuffle(combined)
        return combined

    return shuffled(train_parts), shuffled(validation_parts), shuffled(test_parts)


class MLP:
    def __init__(self, input_size: int, hidden_sizes: tuple[int, int], seed: int) -> None:
        rng = np.random.default_rng(seed)
        first, second = hidden_sizes
        self.weights = [
            (rng.standard_normal((input_size, first)).astype(np.float32) * np.sqrt(2 / input_size)),
            (rng.standard_normal((first, second)).astype(np.float32) * np.sqrt(2 / first)),
            (rng.standard_normal((second, 1)).astype(np.float32) * np.sqrt(2 / second)),
        ]
        self.biases = [np.zeros((1, size), dtype=np.float32) for size in (first, second, 1)]
        self.m_weights = [np.zeros_like(weight) for weight in self.weights]
        self.v_weights = [np.zeros_like(weight) for weight in self.weights]
        self.m_biases = [np.zeros_like(bias) for bias in self.biases]
        self.v_biases = [np.zeros_like(bias) for bias in self.biases]
        self.step = 0

    @staticmethod
    def sigmoid(values: np.ndarray) -> np.ndarray:
        """Convert logits to numerically stable probabilities."""

        return 1.0 / (1.0 + np.exp(-np.clip(values, -40, 40)))

    def forward(self, features: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        """Run a forward pass and return probabilities plus hidden activations."""

        first = np.maximum(0, features @ self.weights[0] + self.biases[0])
        second = np.maximum(0, first @ self.weights[1] + self.biases[1])
        probabilities = self.sigmoid(second @ self.weights[2] + self.biases[2])
        return probabilities, (features, first, second)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return one attack probability for each feature row."""

        return self.forward(features)[0].ravel()

    def train_batch(
        self, features: np.ndarray, labels: np.ndarray, learning_rate: float, positive_weight: float
    ) -> float:
        """Update model weights on one batch and return weighted log loss."""

        probabilities, activations = self.forward(features)
        errors = probabilities - labels[:, None]
        errors *= np.where(labels[:, None] == 1, positive_weight, 1.0)
        errors /= max(1, len(labels))

        d_hidden_2 = errors @ self.weights[2].T * (activations[2] > 0)
        d_hidden_1 = d_hidden_2 @ self.weights[1].T * (activations[1] > 0)
        gradients_w = [
            activations[0].T @ d_hidden_1,
            activations[1].T @ d_hidden_2,
            activations[2].T @ errors,
        ]
        gradients_b = [d_hidden_1.sum(axis=0, keepdims=True), d_hidden_2.sum(axis=0, keepdims=True), errors.sum(axis=0, keepdims=True)]

        self.step += 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        for index in range(3):
            self.m_weights[index] = beta1 * self.m_weights[index] + (1 - beta1) * gradients_w[index]
            self.v_weights[index] = beta2 * self.v_weights[index] + (1 - beta2) * gradients_w[index] ** 2
            self.m_biases[index] = beta1 * self.m_biases[index] + (1 - beta1) * gradients_b[index]
            self.v_biases[index] = beta2 * self.v_biases[index] + (1 - beta2) * gradients_b[index] ** 2
            correction_1 = 1 - beta1**self.step
            correction_2 = 1 - beta2**self.step
            self.weights[index] -= learning_rate * (self.m_weights[index] / correction_1) / (
                np.sqrt(self.v_weights[index] / correction_2) + epsilon
            )
            self.biases[index] -= learning_rate * (self.m_biases[index] / correction_1) / (
                np.sqrt(self.v_biases[index] / correction_2) + epsilon
            )

        clipped = np.clip(probabilities.ravel(), 1e-7, 1 - 1e-7)
        weights = np.where(labels == 1, positive_weight, 1.0)
        return float(-np.mean(weights * (labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))))


def metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    """Calculate binary classification metrics at the selected threshold."""

    predictions = probabilities >= threshold
    actual = labels.astype(bool)
    true_positive = int(np.sum(predictions & actual))
    true_negative = int(np.sum(~predictions & ~actual))
    false_positive = int(np.sum(predictions & ~actual))
    false_negative = int(np.sum(~predictions & actual))
    accuracy = (true_positive + true_negative) / max(1, len(labels))
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / max(1e-12, precision + recall)),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def save_model(
    path: Path, model: MLP, mean: np.ndarray, scale: np.ndarray, feature_names: list[str], threshold: float
) -> None:
    """Save model parameters, preprocessing statistics, feature names, and threshold."""

    np.savez_compressed(
        path,
        weight_0=model.weights[0],
        weight_1=model.weights[1],
        weight_2=model.weights[2],
        bias_0=model.biases[0],
        bias_1=model.biases[1],
        bias_2=model.biases[2],
        mean=mean,
        scale=scale,
        feature_names=np.asarray(feature_names),
        threshold=np.asarray(threshold),
    )


def main() -> None:
    """Train, evaluate, and save an attack-versus-normal detection model."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("processed_data/uavcan_processed.npz"))
    parser.add_argument("--output", type=Path, default=Path("processed_data/uavcan_attack_model.npz"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--max-samples", type=int, default=None, help="Use a random subset for quick experiments.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if not 0 < args.validation_fraction + args.test_fraction < 1:
        raise ValueError("Validation and test fractions must be positive and sum to less than 1.")
    features, labels, feature_names = load_data(args.input)
    if args.max_samples is not None and args.max_samples < len(labels):
        rng = np.random.default_rng(args.seed)
        selected = rng.choice(len(labels), size=args.max_samples, replace=False)
        features, labels = features[selected], labels[selected]

    train_indices, validation_indices, test_indices = stratified_split(
        labels, args.validation_fraction, args.test_fraction, args.seed
    )
    mean = features[train_indices].mean(axis=0)
    scale = features[train_indices].std(axis=0)
    scale[scale < 1e-6] = 1.0
    features = ((features - mean) / scale).astype(np.float32)

    train_labels = labels[train_indices]
    positive_weight = float(np.sum(train_labels == 0) / max(1, np.sum(train_labels == 1)))
    model = MLP(features.shape[1], (64, 32), args.seed)
    rng = np.random.default_rng(args.seed)
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(train_indices)
        losses = []
        for start in range(0, len(order), args.batch_size):
            batch = order[start : start + args.batch_size]
            losses.append(model.train_batch(features[batch], labels[batch], args.learning_rate, positive_weight))
        validation_probabilities = model.predict_proba(features[validation_indices])
        validation_metrics = metrics(labels[validation_indices], validation_probabilities, args.threshold)
        print(
            f"epoch {epoch:02d}/{args.epochs} loss={np.mean(losses):.4f} "
            f"val_f1={validation_metrics['f1']:.4f} val_recall={validation_metrics['recall']:.4f}"
        )

    test_probabilities = model.predict_proba(features[test_indices])
    test_metrics = metrics(labels[test_indices], test_probabilities, args.threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_model(args.output, model, mean, scale, feature_names, args.threshold)
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
                "positive_weight": positive_weight,
                "threshold": args.threshold,
                "test_metrics": test_metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"test: accuracy={test_metrics['accuracy']:.4f} precision={test_metrics['precision']:.4f} "
          f"recall={test_metrics['recall']:.4f} f1={test_metrics['f1']:.4f}")
    print(f"saved model: {args.output}")
    print(f"saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()