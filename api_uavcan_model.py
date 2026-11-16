"""HTTP API for predicting whether a UAVCAN frame is normal or an attack."""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from test_uavcan_model import load_model as load_mlp_model
from uavcan_cnn import load_artifact, predict_proba


MAX_REQUEST_BYTES = 16 * 1024
FRONTEND_PATH = Path(__file__).with_name("Frontend") / "index.html"


def parse_can_id(value: Any) -> int:
    """Parse a CAN ID supplied as a decimal integer or 0x-prefixed hex string."""

    if isinstance(value, bool):
        raise ValueError("can_id must be an integer or hexadecimal string")
    if isinstance(value, int):
        can_id = value
    elif isinstance(value, str):
        value = value.strip()
        try:
            base = 16 if value.lower().startswith("0x") else 10
            can_id = int(value, base)
        except ValueError as error:
            raise ValueError(
                "can_id must be a decimal integer or 0x-prefixed hexadecimal string"
            ) from error
    else:
        raise ValueError("can_id must be an integer or hexadecimal string")
    if not 0 <= can_id <= 0x1FFFFFFF:
        raise ValueError("can_id must be between 0 and 0x1FFFFFFF")
    return can_id


def frame_to_features(payload: dict[str, Any]) -> np.ndarray:
    """Validate one API payload and convert it to the model's 11 features."""

    timestamp = payload.get("timestamp", 0.0)
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
        raise ValueError("timestamp must be a number")
    if not np.isfinite(timestamp):
        raise ValueError("timestamp must be finite")

    can_id = parse_can_id(payload.get("can_id"))
    dlc = payload.get("dlc")
    if isinstance(dlc, bool) or not isinstance(dlc, int) or not 0 <= dlc <= 8:
        raise ValueError("dlc must be an integer from 0 through 8")

    data = payload.get("data", [])
    if not isinstance(data, list) or len(data) != dlc:
        raise ValueError("data must be a list containing exactly dlc bytes")
    if any(isinstance(byte, bool) or not isinstance(byte, int) or not 0 <= byte <= 255 for byte in data):
        raise ValueError("data bytes must be integers from 0 through 255")

    return np.asarray(
        [[timestamp, can_id, dlc, *data, *([0] * (8 - dlc))]],
        dtype=np.float32,
    )


def prediction_response(
    payload: dict[str, Any], model: Any, mean: np.ndarray, scale: np.ndarray, threshold: float
) -> dict[str, Any]:
    """Create a JSON-compatible prediction for one validated frame."""

    features = frame_to_features(payload)
    probability = float(predict_proba(model, features, mean, scale, batch_size=1)[0])
    prediction = "Attack" if probability >= threshold else "Normal"
    return {
        "prediction": prediction,
        "is_attack": prediction == "Attack",
        "attack_probability": probability,
        "threshold": threshold,
    }


def mlp_prediction_response(
    payload: dict[str, Any], model: Any, mean: np.ndarray, scale: np.ndarray, threshold: float
) -> dict[str, Any]:
    """Create a JSON-compatible prediction using the NumPy MLP."""

    features = frame_to_features(payload)
    standardized = ((features - mean) / scale).astype(np.float32)
    probability = float(model.predict_proba(standardized)[0])
    prediction = "Attack" if probability >= threshold else "Normal"
    return {
        "prediction": prediction,
        "is_attack": prediction == "Attack",
        "attack_probability": probability,
        "threshold": threshold,
    }


def combined_prediction_response(
    payload: dict[str, Any],
    cnn_artifact: tuple[Any, np.ndarray, np.ndarray, float],
    mlp_artifact: tuple[Any, np.ndarray, np.ndarray, float],
) -> dict[str, Any]:
    """Create one response containing predictions from both trained models."""

    cnn_result = prediction_response(payload, *cnn_artifact)
    mlp_result = mlp_prediction_response(payload, *mlp_artifact)
    return {
        **cnn_result,
        "models": {"cnn": cnn_result, "mlp": mlp_result},
        "agreement": cnn_result["prediction"] == mlp_result["prediction"],
    }


def make_handler(
    artifacts: dict[str, tuple[Any, np.ndarray, np.ndarray, float]],
    allow_origin: str,
) -> type[BaseHTTPRequestHandler]:
    """Build a request handler bound to both loaded models."""

    class ModelRequestHandler(BaseHTTPRequestHandler):
        def send_json(self, status: HTTPStatus, body: dict[str, Any]) -> None:
            encoded = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", allow_origin)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_OPTIONS(self) -> None:
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Access-Control-Allow-Origin", allow_origin)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self) -> None:
            if self.path in {"/", "/index.html"}:
                try:
                    content = FRONTEND_PATH.read_bytes()
                except OSError as error:
                    self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return
            if self.path == "/health":
                self.send_json(
                    HTTPStatus.OK,
                    {"status": "ok", "models_loaded": list(artifacts)},
                )
                return
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "endpoint not found"})

        def do_POST(self) -> None:
            if self.path != "/predict":
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "endpoint not found"})
                return

            content_length = self.headers.get("Content-Length")
            try:
                length = int(content_length or "0")
            except ValueError:
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid Content-Length"})
                return
            if length <= 0 or length > MAX_REQUEST_BYTES:
                self.send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": f"request body must be between 1 and {MAX_REQUEST_BYTES} bytes"},
                )
                return
            try:
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("request body must be a JSON object")
                result = combined_prediction_response(
                    body, artifacts["cnn"], artifacts["mlp"]
                )
            except (json.JSONDecodeError, UnicodeDecodeError):
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": "request body must contain valid JSON"})
                return
            except ValueError as error:
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            self.send_json(HTTPStatus.OK, result)

        def log_message(self, format: str, *args: Any) -> None:
            print(f"{self.address_string()} - {format % args}")

    return ModelRequestHandler


def main() -> None:
    """Load the model and start the prediction server."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        "--cnn-model",
        dest="cnn_model",
        type=Path,
        default=Path("processed_data/uavcan_attack_cnn.keras"),
        help="Keras CNN model artifact.",
    )
    parser.add_argument(
        "--mlp-model",
        type=Path,
        default=Path("processed_data/uavcan_attack_model.npz"),
        help="NumPy MLP model artifact.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--allow-origin",
        default="*",
        help="Browser origin allowed to call the API (default: *).",
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be between 1 and 65535")

    cnn_model, cnn_mean, cnn_scale, _, cnn_threshold = load_artifact(args.cnn_model)
    mlp_model, mlp_mean, mlp_scale, _, mlp_threshold = load_mlp_model(args.mlp_model)
    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(
            {
                "cnn": (cnn_model, cnn_mean, cnn_scale, cnn_threshold),
                "mlp": (mlp_model, mlp_mean, mlp_scale, mlp_threshold),
            },
            args.allow_origin,
        ),
    )
    print(f"UAVCAN model API listening on http://{args.host}:{args.port}")
    print("POST JSON frames to /predict; press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping API server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
