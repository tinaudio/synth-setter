"""Real loopback requests exercise browser transport boundaries."""

import json
import secrets
import socket
import threading
from collections.abc import Iterator
from concurrent.futures import Future
from http.client import HTTPConnection
from pathlib import Path
from queue import Queue
from urllib.parse import urlsplit

import click
import pytest
import torch

from synth_setter.evaluation import browser_flow

_TOKEN = secrets.token_urlsafe(32)
_BODY = json.dumps({"token": _TOKEN, "params": [0.25, 0.75]})

type BrowserSession = tuple[str, HTTPConnection, Future[torch.Tensor]]


@pytest.fixture
def browser_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[BrowserSession]:
    """Run the real receiver on an OS-assigned port and close it with valid output.

    :yields BrowserSession: Origin, real client, and the receiver's eventual prediction.
    :param tmp_path: Served resource and accepted prediction destination.
    :param monkeypatch: Observes the actual readiness message without changing HTTP behavior.
    """
    resource = tmp_path / "input.json"
    resource.write_text("{}")
    ready: Queue[str] = Queue()
    monkeypatch.setattr(browser_flow.click, "echo", ready.put)
    completed: Future[torch.Tensor] = Future()

    def serve() -> None:
        """Forward the receiver's result or exception to the test thread."""
        try:
            completed.set_result(
                browser_flow._receive_prediction(
                    {"/input.json": resource}, token=_TOKEN, width=2, output_dir=tmp_path, port=0
                )
            )
        except BaseException as exc:
            completed.set_exception(exc)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    origin = ready.get(timeout=3).removeprefix("Browser evaluation: ")
    connection = HTTPConnection("127.0.0.1", urlsplit(origin).port, timeout=3)
    try:
        yield origin, connection, completed
    finally:
        connection.close()
        try:
            if not completed.done():
                connection.request(
                    "POST",
                    "/prediction",
                    _BODY,
                    {
                        "Origin": origin,
                        "Content-Type": "application/json",
                    },
                )
                connection.getresponse().read()
        finally:
            connection.close()
            thread.join(timeout=3)
        assert not thread.is_alive(), "Browser receiver did not stop after valid output"
        completed.result(timeout=0)


@pytest.mark.parametrize(
    ("method", "path", "headers", "body", "status"),
    [
        pytest.param("GET", "/input.json", {"Host": "other.invalid"}, None, 403, id="host"),
        pytest.param("GET", "/pyproject.toml", {}, None, 404, id="resource"),
        pytest.param(
            "POST", "/prediction", {"Origin": "https://other.invalid"}, _BODY, 403, id="origin"
        ),
        pytest.param("POST", "/other", {}, _BODY, 403, id="endpoint"),
        pytest.param(
            "POST", "/prediction", {"Content-Type": "text/plain"}, _BODY, 400, id="content-type"
        ),
        pytest.param("POST", "/prediction", {"Content-Length": "0"}, None, 400, id="empty-body"),
        pytest.param(
            "POST",
            "/prediction",
            {"Content-Length": str(browser_flow._MAX_REQUEST_BYTES + 1)},
            None,
            400,
            id="oversized-body",
        ),
        pytest.param("POST", "/prediction", {}, "{", 400, id="malformed-json"),
        pytest.param(
            "POST",
            "/prediction",
            {},
            json.dumps({"token": "wrong", "params": [0.25, 0.75]}),
            400,
            id="token",
        ),
        pytest.param(
            "POST",
            "/prediction",
            {},
            json.dumps({"token": _TOKEN, "params": [0.25]}),
            400,
            id="width",
        ),
        pytest.param(
            "POST",
            "/prediction",
            {},
            json.dumps({"token": _TOKEN, "params": [1e40, 0.75]}),
            400,
            id="float32-overflow",
        ),
    ],
)
def test_browser_http_invalid_request_rejected(
    browser_server: BrowserSession,
    method: str,
    path: str,
    headers: dict[str, str],
    body: str | None,
    status: int,
) -> None:
    """Malformed requests cannot consume the active prediction session.

    :param browser_server: Real receiver and client with teardown completing the session.
    :param method: Request method at the rejection boundary.
    :param path: Requested loopback resource or prediction endpoint.
    :param headers: Deliberate header deviations from a valid same-origin request.
    :param body: Untrusted serialized prediction body.
    :param status: Required HTTP rejection status.
    """
    origin, connection, _ = browser_server
    connection.request(
        method,
        path,
        body,
        {
            "Origin": origin,
            "Content-Type": "application/json",
            **headers,
        },
    )
    response = connection.getresponse()
    response.read()
    assert response.status == status


def test_browser_http_rejection_does_not_inject_exception_headers(
    browser_server: BrowserSession,
) -> None:
    """Validation details cannot become attacker-controlled response headers.

    :param browser_server: Real receiver and client for the malformed JSON object.
    """
    origin, connection, _ = browser_server
    body = json.dumps({"token": _TOKEN, "params": [0.25, 0.75], "X-Injected: yes": 0})
    connection.request(
        "POST",
        "/prediction",
        body,
        {
            "Origin": origin,
            "Content-Type": "application/json",
        },
    )
    response = connection.getresponse()
    response.read()
    assert response.getheader("X-Injected") is None


def test_browser_http_valid_prediction_returns_float32_coordinates(
    browser_server: BrowserSession,
) -> None:
    """Accepted JSON reaches the caller as the original finite float32 row.

    :param browser_server: Real receiver and its eventual prediction result.
    """
    origin, connection, completed = browser_server
    connection.request(
        "POST",
        "/prediction",
        _BODY,
        {
            "Origin": origin,
            "Content-Type": "application/json",
        },
    )
    connection.getresponse().read()
    torch.testing.assert_close(completed.result(timeout=3), torch.tensor([[0.25, 0.75]]))


@pytest.mark.parametrize("port,suffix", [(80, ""), (443, ":443"), (8765, ":8765")])
def test_browser_origin_default_http_port_matches_browser_serialization(
    port: int, suffix: str
) -> None:
    """Only HTTP's default port is omitted from the browser origin.

    :param port: Bound server port, including HTTP and HTTPS defaults.
    :param suffix: Browser's explicit-port suffix for the HTTP scheme.
    """
    assert browser_flow._browser_origin(port) == f"http://127.0.0.1{suffix}"


def test_browser_http_occupied_port_raises_actionable_cli_error(tmp_path: Path) -> None:
    """A real port collision reports a CLI error rather than an uncaught OSError.

    :param tmp_path: Unused prediction destination because server binding must fail.
    """
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        port = occupied.getsockname()[1]
        with pytest.raises(click.ClickException, match=f"Cannot bind browser port {port}"):
            browser_flow._receive_prediction(
                {}, token=_TOKEN, width=2, output_dir=tmp_path, port=port
            )
