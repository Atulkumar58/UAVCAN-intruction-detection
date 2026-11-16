"""Shared Keras 1D CNN model and artifact helpers for UAVCAN detection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from keras import layers, models

from train_uavcan_model import DEFAULT_FEATURES


CNN_INPUT_SHAPE = (len(DEFAULT_FEATURES), 1)


def build_model(input_shape: tuple[int, int] = CNN_INPUT_SHAPE) -> models.Model:
    """Build a binary 1D convolutional classifier for one UAVCAN frame."""

    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv1D(32, kernel_size=3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv1D(64, kernel_size=3, padding="same", activation="relu"),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.2),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="uavcan_1d_cnn",
    )
    return model


def metadata_path(model_path: Path) -> Path:
    """Return the JSON sidecar path associated with a Keras model artifact."""

    return model_path.with_suffix(".json")


def save_artifact(
    model_path: Path,
    model: models.Model,
    mean: np.ndarray,
    scale: np.ndarray,
    feature_names: list[str],
    threshold: float,
) -> Path:
    """Save the Keras model and preprocessing metadata used by inference."""

    if feature_names != DEFAULT_FEATURES:
        raise ValueError("Model feature names do not match the UAVCAN feature schema.")
    if mean.shape != (len(DEFAULT_FEATURES),) or scale.shape != mean.shape:
        raise ValueError("Model normalization statistics must each contain 11 values.")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("Model normalization statistics must be finite and positive.")
    if not 0 < threshold < 1:
        raise ValueError("Model threshold must be between 0 and 1.")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    sidecar = metadata_path(model_path)
    sidecar.write_text(
        json.dumps(
            {
                "feature_names": feature_names,
                "mean": mean.tolist(),
                "scale": scale.tolist(),
                "threshold": threshold,
                "input_shape": list(CNN_INPUT_SHAPE),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return sidecar


def load_artifact(
    model_path: Path, sidecar_path: Path | None = None
) -> tuple[models.Model, np.ndarray, np.ndarray, list[str], float]:
    """Load a Keras model and validate its inference metadata."""

    sidecar = sidecar_path or metadata_path(model_path)
    model = models.load_model(model_path, compile=False)
    metadata: dict[str, Any] = json.loads(sidecar.read_text(encoding="utf-8"))
    feature_names = [str(name) for name in metadata.get("feature_names", [])]
    mean = np.asarray(metadata.get("mean"), dtype=np.float32)
    scale = np.asarray(metadata.get("scale"), dtype=np.float32)
    threshold = float(metadata.get("threshold"))

    if feature_names != DEFAULT_FEATURES:
        raise ValueError(
            f"Unexpected model features. Expected {DEFAULT_FEATURES}, got {feature_names}."
        )
    if mean.shape != (len(DEFAULT_FEATURES),) or scale.shape != mean.shape:
        raise ValueError("Model mean and scale must each contain 11 values.")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("Model normalization statistics must be finite and positive.")
    if not 0 < threshold < 1:
        raise ValueError("Model threshold must be between 0 and 1.")
    if model.input_shape[1:] != CNN_INPUT_SHAPE or model.output_shape[-1] != 1:
        raise ValueError("Keras model does not match the expected 11-feature CNN shape.")
    return model, mean, scale, feature_names, threshold


def predict_proba(
    model: models.Model,
    features: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Standardize feature rows, add the channel dimension, and predict."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    standardized = ((features - mean) / scale).astype(np.float32)
    probabilities = model.predict(standardized[..., np.newaxis], batch_size=batch_size, verbose=0)
    return np.asarray(probabilities, dtype=np.float32).reshape(-1)
