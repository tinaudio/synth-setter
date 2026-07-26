"""Induce a real W&B filestream failure for the SkyPilot recovery gate."""

from __future__ import annotations

import http.client
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import wandb

log = logging.getLogger(__name__)

_UPSTREAM_HOST = "api.wandb.ai"
_HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "t" + "e",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


class _FailureProxy(ThreadingHTTPServer):
    """Forward W&B API traffic while rejecting filestream requests.

    .. attribute :: daemon_threads

        Let process teardown stop request handlers after the probe.
    """

    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _FailureProxyHandler)
        self.rejected_filestream = threading.Event()


class _FailureProxyHandler(BaseHTTPRequestHandler):
    """Proxy one W&B request or inject the targeted 503 response.

    .. attribute :: server

        Failure-injection server that records observed filestream traffic.
    """

    server: _FailureProxy

    def do_GET(self) -> None:  # noqa: N802
        """Route GET through the filestream-failure injector."""
        self._handle_request()

    def do_POST(self) -> None:  # noqa: N802
        """Route POST through the filestream-failure injector."""
        self._handle_request()

    def do_PUT(self) -> None:  # noqa: N802
        """Route PUT through the filestream-failure injector."""
        self._handle_request()

    def _handle_request(self) -> None:
        if "file_stream" in self.path:
            self.server.rejected_filestream.set()
            body = b'{"error":"injected filestream failure"}'
            self.send_response(http.client.SERVICE_UNAVAILABLE)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length) if content_length else None
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in _HOP_BY_HOP_HEADERS and key.lower() != "host"
        }
        connection = http.client.HTTPSConnection(_UPSTREAM_HOST, timeout=30)
        try:
            connection.request(self.command, self.path, body=body, headers=headers)
            response = connection.getresponse()
            response_body = response.read()
            self.send_response(response.status)
            for key, value in response.getheaders():
                if key.lower() not in _HOP_BY_HOP_HEADERS:
                    self.send_header(key, value)
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)
        finally:
            connection.close()


def main() -> None:
    """Create an online run whose real filestream exhausts retries.

    :raises RuntimeError: The proxy observes no filestream failure or W&B omits internal filestream
        diagnostics.
    """
    output_root = Path("/home/build/synth-setter/wandb-recovery-smoke")
    output_root.mkdir(parents=True, exist_ok=True)
    proxy = _FailureProxy()
    proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    proxy_thread.start()
    base_url = f"http://127.0.0.1:{proxy.server_port}"

    settings = wandb.Settings(
        base_url=base_url,
        console="off",
        x_file_stream_retry_max=1,
        x_file_stream_retry_wait_max_seconds=1,
        x_file_stream_retry_wait_min_seconds=1,
        x_file_stream_timeout_seconds=2,
    )
    try:
        run = wandb.init(
            dir=output_root,
            entity=os.environ.get("WANDB_ENTITY") or None,
            project=os.environ.get("WANDB_PROJECT", "synth-setter"),
            settings=settings,
        )
        if run is None:
            raise RuntimeError("wandb.init returned no run")
        for step in range(20):
            run.log({"recovery_probe/step": step}, step=step)
        run.finish()
    finally:
        proxy.shutdown()
        proxy.server_close()
        proxy_thread.join(timeout=5)

    if not proxy.rejected_filestream.is_set():
        raise RuntimeError("fault proxy observed no W&B filestream request")
    debug_log = (output_root / "wandb/latest-run/logs/debug-internal.log").resolve()
    debug_text = debug_log.read_text(encoding="utf-8", errors="replace").lower()
    if "file stream" not in debug_text and "filestream" not in debug_text:
        raise RuntimeError("W&B internal diagnostics contain no filestream evidence")
    log.info("WANDB_FILESTREAM_FAILURE_OBSERVED=1")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
