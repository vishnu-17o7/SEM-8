import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
import torch.nn as nn


class DuelingDQNAgent(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


def load_latest_checkpoint(models_dir: Path):
    candidates = sorted(models_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No .pth model found in {models_dir}")

    model_path = candidates[0]
    checkpoint = torch.load(model_path, map_location="cpu")

    state_dim = int(checkpoint.get("state_dim", 7))
    action_dim = int(checkpoint.get("action_dim", 40))

    model = DuelingDQNAgent(state_dim, action_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint, model_path


RESEARCH_DIR = Path(__file__).resolve().parent
MODELS_DIR = RESEARCH_DIR / "outputs" / "models"
MODEL, CHECKPOINT, MODEL_PATH = load_latest_checkpoint(MODELS_DIR)


class Handler(BaseHTTPRequestHandler):
    def _set_headers(self, code=200, content_type="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(200)

    def do_GET(self):
        if self.path == "/health":
            payload = {
                "status": "ok",
                "model_path": str(MODEL_PATH),
                "architecture": CHECKPOINT.get("architecture", "DuelingDQN"),
                "state_dim": int(CHECKPOINT.get("state_dim", 7)),
                "action_dim": int(CHECKPOINT.get("action_dim", 40)),
                "device_trained_on": CHECKPOINT.get("device_trained_on", "unknown"),
            }
            self._set_headers(200)
            self.wfile.write(json.dumps(payload).encode("utf-8"))
            return

        self._set_headers(404)
        self.wfile.write(json.dumps({"error": "Not found"}).encode("utf-8"))

    def do_POST(self):
        if self.path != "/predict":
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Not found"}).encode("utf-8"))
            return

        try:
            content_len = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(content_len)
            body = json.loads(raw.decode("utf-8")) if raw else {}
            state = body.get("state", None)

            if not isinstance(state, list):
                raise ValueError("'state' must be a list of floats")

            state_dim = int(CHECKPOINT.get("state_dim", 7))
            if len(state) != state_dim:
                raise ValueError(f"state length must be {state_dim}, got {len(state)}")

            state_tensor = torch.tensor([state], dtype=torch.float32)
            with torch.no_grad():
                q_values = MODEL(state_tensor)
                action = int(torch.argmax(q_values, dim=1).item())

            self._set_headers(200)
            self.wfile.write(json.dumps({"action": action}).encode("utf-8"))

        except Exception as exc:
            self._set_headers(400)
            self.wfile.write(json.dumps({"error": str(exc)}).encode("utf-8"))


def run():
    host = os.environ.get("QKD_BACKEND_HOST", "127.0.0.1")
    port = int(os.environ.get("QKD_BACKEND_PORT", "8001"))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"QKD backend running at http://{host}:{port}")
    print(f"Loaded model: {MODEL_PATH}")
    server.serve_forever()


if __name__ == "__main__":
    run()
