"""Loopback-only browser inference transport for the production sketch CLI."""

import json
import mimetypes
import secrets
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import click
import torch
from pydantic import BaseModel, Field, FiniteFloat

from synth_setter.models.flow_onnx import branch_weights, export_flow_onnx
from synth_setter.models.vst_flow_matching_module import VSTFlowMatchingModule

_WEB_ROOT = Path(__file__).resolve().parents[1] / "web"
_MAX_REQUEST_BYTES = 1_048_576


class BrowserPrediction(BaseModel, strict=True, extra="forbid"):
    """Untrusted browser output accepted only for the active local session.

    .. attribute :: token

        URL-safe ASCII capability returned by the loopback input endpoint.

    .. attribute :: params

        Finite model-space parameter coordinates; width is checked against the checkpoint.
    """

    token: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    params: list[FiniteFloat] = Field(min_length=1)


def require_browser_assets() -> Path:
    """Require the installed ONNX Runtime Web module before creating artifacts.

    :returns: Directory containing the installed JavaScript/WASM runtime.
    :raises FileNotFoundError: ONNX Runtime Web assets have not been installed.
    """
    vendor = _WEB_ROOT / "node_modules" / "onnxruntime-web" / "dist"
    if not (vendor / "ort.wasm.min.mjs").is_file():
        raise FileNotFoundError(f"Install browser assets with: npm ci --prefix {_WEB_ROOT}")
    return vendor


def sample_in_browser(
    model: VSTFlowMatchingModule,
    batch: dict[str, torch.Tensor],
    noise: torch.Tensor,
    *,
    content_cfg_strength: float,
    sketch_cfg_strength: float,
    sample_steps: int,
    output_dir: Path,
    port: int = 0,
) -> torch.Tensor:
    """Export one pair and wait up to ten minutes for real browser inference.

    :param model: CPU evaluation checkpoint with mel/sketch conditioning.
    :param batch: One normalized mel/sketch pair.
    :param noise: Explicit finite float32 initial parameter row.
    :param content_cfg_strength: Non-negative content guidance.
    :param sketch_cfg_strength: Non-negative sketch guidance.
    :param sample_steps: RK4 steps, between one and one thousand.
    :param output_dir: Empty graph bundle destination beneath the evaluation arm.
    :param port: Loopback port; zero selects an available port.
    :returns: Browser-produced model-space parameter row for native rendering.
    :raises ValueError: Sampling settings or noise violate the browser contract.
    """
    width = model.hparams["num_params"]
    if (
        noise.shape != (1, width)
        or noise.dtype != torch.float32
        or not torch.isfinite(noise).all()
    ):
        raise ValueError("browser inference requires one finite float32 noise row")
    if batch["mel"].shape[0] != 1:
        raise ValueError("browser inference requires one content/sketch pair")
    if (
        not isinstance(sample_steps, int)
        or isinstance(sample_steps, bool)
        or not 1 <= sample_steps <= 1000
    ):
        raise ValueError("browser sample_steps must be an integer between 1 and 1000")
    guidance = torch.tensor([content_cfg_strength, sketch_cfg_strength], dtype=torch.float32)
    if not torch.isfinite(guidance).all() or (guidance < 0).any():
        raise ValueError("browser guidance must be finite and non-negative")
    vendor = require_browser_assets()
    export_flow_onnx(model, batch, output_dir)
    token = secrets.token_urlsafe(32)
    payload: dict[str, object] = {
        key: {"shape": list(batch[key].shape), "data": batch[key].flatten().tolist()}
        for key in ("mel", "sketch_ctrl")
    }
    payload.update(
        {
            "noise": noise.flatten().tolist(),
            "steps": sample_steps,
            "branch_weights": list(branch_weights("both", *guidance.tolist())),
            "token": token,
        }
    )
    (output_dir / "input.json").write_text(json.dumps(payload, allow_nan=False))
    routes = {
        f"/{name}": _WEB_ROOT / name
        for name in ("index.html", "app.mjs", "flow.mjs", "guidance.mjs", "rk4.mjs")
    }
    routes["/"] = _WEB_ROOT / "index.html"
    routes.update(
        {
            f"/{name}": output_dir / name
            for name in ("conditioning.onnx", "velocity.onnx", "input.json")
        }
    )
    routes.update(
        {
            f"/ort/{path.name}": path
            for path in vendor.iterdir()
            if path.suffix in {".mjs", ".wasm"}
        }
    )
    return _receive_prediction(routes, token=token, width=width, output_dir=output_dir, port=port)


def _browser_origin(port: int) -> str:
    """Match browser serialization of the loopback HTTP origin.

    :param port: Bound server port.
    :returns: Loopback origin with HTTP's default port omitted.
    """
    return "http://127.0.0.1" + (f":{port}" if port != 80 else "")


def _receive_prediction(
    routes: dict[str, Path], *, token: str, width: int, output_dir: Path, port: int
) -> torch.Tensor:
    """Serve an allowlisted bundle until one validated prediction arrives.

    :param routes: Exact URL paths and their local resources.
    :param token: Active session capability.
    :param width: Checkpoint parameter width.
    :param output_dir: Prediction artifact destination.
    :param port: Loopback listening port, or zero for dynamic allocation.
    :returns: Validated float32 prediction row.
    :raises TimeoutError: The browser did not finish within ten minutes.
    :raises click.ClickException: The requested loopback port cannot be bound.
    """
    prediction: torch.Tensor | None = None
    origin = ""

    class Handler(BaseHTTPRequestHandler):
        """Serve only the current bundle and installed runtime assets."""

        def setup(self) -> None:
            """Bound stalled local requests before reading their headers."""
            super().setup()
            self.connection.settimeout(10)

        def do_GET(self) -> None:
            """Return one allowlisted resource without exposing the workspace."""
            if self.headers.get("Host") != urlsplit(origin).netloc:
                self.send_error(403)
                return
            path = routes.get(urlsplit(self.path).path)
            if path is None:
                self.send_error(404)
                return
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if path.suffix == ".mjs":
                content_type = "text/javascript"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(path.stat().st_size))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    self.wfile.write(chunk)

        def do_POST(self) -> None:
            """Validate the session capability, shape, and finite prediction values.

            :raises ValueError: Invalid requests are caught and translated to HTTP 400.
            """
            nonlocal prediction
            if self.path != "/prediction" or self.headers.get("Origin") != origin:
                self.send_error(403)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if (
                    not 0 < length <= _MAX_REQUEST_BYTES
                    or self.headers.get("Content-Type") != "application/json"
                ):
                    raise ValueError("invalid prediction request")
                self.connection.settimeout(10)
                response = BrowserPrediction.model_validate_json(self.rfile.read(length))
                if (
                    not secrets.compare_digest(response.token, token)
                    or len(response.params) != width
                ):
                    raise ValueError("prediction session or width mismatch")
                result = torch.tensor([response.params], dtype=torch.float32)
                if not torch.isfinite(result).all():
                    raise ValueError("prediction overflows float32")
            except (ValueError, TimeoutError):
                self.send_error(400, "Invalid browser prediction")
                return
            (output_dir / "prediction.json").write_text(response.model_dump_json())
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()
            prediction = result

    try:
        server = HTTPServer(("127.0.0.1", port), Handler)
    except OSError as exc:
        raise click.ClickException(f"Cannot bind browser port {port}: {exc.strerror}") from exc
    with server:
        server.timeout = 1
        origin = _browser_origin(server.server_port)
        click.echo(f"Browser evaluation: {origin}")
        deadline = time.monotonic() + 600
        while prediction is None and time.monotonic() < deadline:
            server.handle_request()
    if prediction is None:
        raise TimeoutError("browser inference did not complete within ten minutes")
    return prediction
