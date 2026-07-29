"""Production-entrypoint integration for locked local Studiorack installation."""

from __future__ import annotations

import hashlib
import http.server
import json
import os
import shutil
import ssl
import subprocess
import sys
import threading
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import pytest

from synth_setter.resources import as_file, vst_headless_wrapper

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDIORACK = PROJECT_ROOT / "node_modules/.bin/studiorack"

pytestmark = pytest.mark.skipif(
    sys.platform != "linux",
    reason="the patched Node CLI is installed in the Linux test workflow",
)


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass


@contextmanager
def _https_server(root: Path, certificate: Path, key: Path) -> Iterator[str]:
    handler = partial(_QuietHandler, directory=str(root))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certificate, key)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"https://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        worker.join()
        server.server_close()


def _certificate(root: Path) -> tuple[Path, Path]:
    """Generate a one-day loopback certificate for the local registry.

    :param root: Destination for certificate material.
    :returns: Certificate and private-key paths.
    :raises FileNotFoundError: OpenSSL is unavailable.
    """
    certificate = root / "certificate.pem"
    key = root / "key.pem"
    openssl = shutil.which("openssl")
    if openssl is None:
        raise FileNotFoundError("openssl is required for the local HTTPS integration")
    subprocess.run(  # noqa: S603 — resolved local certificate tool and fixed arguments
        [
            openssl,
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(certificate),
            "-subj",
            "/CN=127.0.0.1",
            "-addext",
            "subjectAltName=IP:127.0.0.1",
            "-days",
            "1",
        ],
        check=True,
        stderr=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
    )
    return certificate, key


def _artifact(root: Path) -> Path:
    """Build a deterministic ELF VST3 archive consumed by real Studiorack.

    :param root: Artifact destination directory.
    :returns: ZIP archive path.
    """
    artifact = root / "example-synth.zip"
    with zipfile.ZipFile(artifact, "w") as archive:
        archive.writestr(
            "Example Synth.vst3/Contents/x86_64-linux/Example Synth.so",
            b"\x7fELFdeterministic-test-plugin",
        )
        archive.writestr(
            "Example Synth.vst3/Contents/moduleinfo.json",
            json.dumps({"Version": "1.2.3"}),
        )
    return artifact


def _registry_payload(artifact_url: str, artifact: Path) -> dict[str, object]:
    """Build valid registry metadata for the local artifact.

    :param artifact_url: Loopback HTTPS artifact URL.
    :param artifact: Archive whose bytes define size and digest.
    :returns: Complete four-registry response payload.
    """
    package_version = {
        "author": "synth-setter",
        "changes": "integration fixture",
        "date": "2026-01-01T00:00:00.000Z",
        "description": "deterministic local plugin",
        "files": [
            {
                "architectures": ["x64"],
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "size": artifact.stat().st_size,
                "systems": [{"type": "linux"}],
                "type": "archive",
                "url": artifact_url,
            }
        ],
        "image": artifact_url,
        "license": "mit",
        "name": "Example Synth",
        "tags": ["synth"],
        "type": "instrument",
        "url": artifact_url,
    }
    return {
        "apps": {},
        "plugins": {"example/synth": {"versions": {"1.2.3": package_version}}},
        "presets": {},
        "projects": {},
    }


def _write_project_files(root: Path, artifact_url: str, artifact: Path) -> Path:
    """Write registry, manifest, and lock files for one pinned package.

    :param root: Integration scratch root.
    :param artifact_url: Loopback HTTPS artifact URL.
    :param artifact: Archive whose digest is locked.
    :returns: Written manifest path.
    """
    (root / "registry.json").write_text(json.dumps(_registry_payload(artifact_url, artifact)))
    manifest = root / "studiorack.json"
    manifest.write_text(
        json.dumps(
            {
                "name": "integration",
                "plugins": {"example/synth": "1.2.3"},
                "type": "project",
                "vst3Bundles": {"example/synth": "Example Synth.vst3"},
            }
        )
    )
    (root / "studiorack.lock.json").write_text(
        json.dumps(
            {
                "example/synth@1.2.3": {
                    "artifacts": [
                        {
                            "architectures": ["x64"],
                            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                            "systems": ["linux"],
                            "type": "archive",
                            "url": artifact_url,
                        }
                    ]
                }
            }
        )
    )
    return manifest


def _configure_studiorack_home(root: Path, registry_url: str) -> None:
    """Configure a private Studiorack home for the loopback registry.

    :param root: Integration scratch root.
    :param registry_url: Loopback registry metadata URL.
    """
    home = root / "home"
    app_dir = home / ".local/share/open-audio-stack"
    app_dir.mkdir(parents=True)
    (app_dir / "config.json").write_text(
        json.dumps(
            {
                "appDir": str(app_dir),
                "registries": [{"name": "integration", "url": registry_url}],
            }
        )
    )


def _run_locked_install(root: Path, manifest: Path) -> subprocess.CompletedProcess[str]:
    """Invoke the production plugin command against the local registry.

    :param root: Integration scratch root containing HOME and output directories.
    :param manifest: Pinned project manifest path.
    :returns: Completed production CLI process.
    """
    command = Path(sys.executable).with_name("synth-setter-plugins").resolve(strict=True)
    managed = root / "managed"
    links = root / "plugins"
    home = root / "home"
    return subprocess.run(  # noqa: S603 — resolved project entrypoint and validated arguments
        [
            str(command),
            "--manifest",
            str(manifest),
            "--plugins-dir",
            str(managed),
            "--links-dir",
            str(links),
            "--studiorack-executable",
            str(STUDIORACK),
            "install",
            "--plugin",
            "example/synth",
        ],
        check=False,
        capture_output=True,
        env={
            **os.environ,
            "HOME": str(home),
            "NODE_TLS_REJECT_UNAUTHORIZED": "0",
        },
        text=True,
        timeout=120,
    )


@pytest.fixture()
def installed_locked_plugin(
    tmp_path: Path,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    """Install one locked package through the real patched CLI.

    :param tmp_path: Scratch root for HTTPS registry, HOME, and plugin output.
    :returns: Completed process, managed root, and stable links root.
    """
    assert STUDIORACK.is_file(), "run `npm ci` before the integration suite"
    certificate, key = _certificate(tmp_path)
    artifact = _artifact(tmp_path)
    managed = tmp_path / "managed"
    links = tmp_path / "plugins"
    with _https_server(tmp_path, certificate, key) as base_url:
        artifact_url = f"{base_url}/{artifact.name}"
        manifest = _write_project_files(tmp_path, artifact_url, artifact)
        _configure_studiorack_home(tmp_path, f"{base_url}/registry.json")
        result = _run_locked_install(tmp_path, manifest)
    return result, managed, links


def test_synth_setter_plugins_install_propagates_lock_to_real_patched_cli(
    installed_locked_plugin: tuple[subprocess.CompletedProcess[str], Path, Path],
) -> None:
    """The Python command drives patched core from lock to sealed stable alias.

    :param installed_locked_plugin: Result and outputs from the real install path.
    """
    result, managed, links = installed_locked_plugin
    assert result.returncode == 0, result.stderr
    alias = links / "Example Synth.vst3"
    assert alias.is_symlink()
    assert (alias / "Contents/moduleinfo.json").is_file()
    seal = json.loads(
        (managed / "VST3/example/synth/1.2.3/.synth-setter-complete.json").read_text()
    )
    assert seal["package_reference"] == "example/synth@1.2.3"
    assert seal["source_kind"] == "artifact-lock"


@pytest.mark.network
@pytest.mark.requires_vst
@pytest.mark.slow
def test_repository_locked_dexed_install_alias_loads_in_headless_pedalboard(
    tmp_path: Path,
) -> None:
    """A real locked install produces the Dexed alias consumed by Pedalboard.

    :param tmp_path: Isolated Studiorack home, managed root, and stable alias root.
    """
    assert STUDIORACK.is_file(), "run `npm ci` before the integration suite"
    command = Path(sys.executable).with_name("synth-setter-plugins").resolve(strict=True)
    managed = tmp_path / "managed"
    links = tmp_path / "plugins"
    install = subprocess.run(  # noqa: S603 — resolved project entrypoint and fixed package pin
        [
            str(command),
            "--manifest",
            str(PROJECT_ROOT / "studiorack.json"),
            "--plugins-dir",
            str(managed),
            "--links-dir",
            str(links),
            "--studiorack-executable",
            str(STUDIORACK),
            "install",
            "--plugin",
            "asb2m10/dexed",
        ],
        check=False,
        capture_output=True,
        env={**os.environ, "HOME": str(tmp_path / "home")},
        text=True,
        timeout=300,
    )
    assert install.returncode == 0, install.stderr

    alias = links / "Dexed.vst3"
    probe = (
        "import json,sys; "
        "from synth_setter.data.vst.core import load_plugin; "
        "plugin=load_plugin(sys.argv[1]); "
        "print(json.dumps({'name':plugin.name,'parameters':len(plugin.parameters)}))"
    )
    with as_file(vst_headless_wrapper()) as wrapper:
        loaded = subprocess.run(  # noqa: S603 — packaged wrapper and current Python runtime
            [str(wrapper), sys.executable, "-c", probe, str(alias)],
            check=False,
            capture_output=True,
            env=os.environ.copy(),
            text=True,
            timeout=120,
        )

    assert loaded.returncode == 0, loaded.stderr
    payload = json.loads(
        next(line for line in reversed(loaded.stdout.splitlines()) if line.strip())
    )
    assert payload["name"] == "Dexed"
    assert payload["parameters"] >= 100
