"""Tests for scripts/capture-skypilot-state.sh — kind/k8s state capture.

The script captures controller-side state from the kind cluster BEFORE
``sky local down`` reaps it: apiserver objects (pods, events, nodes,
pods-yaml), per-worker-pod ``kubectl describe`` output, and the
sky-jobs-controller pod's on-pod ``~/sky_logs/sky-*/provision.log``. The
set of files written under ``${RUN_METADATA_DIR}/k8s_state/`` is the
public contract — the workflow's ``Upload run metadata`` step zips that
directory. See PR #876.

Tests stub ``kubectl`` on PATH via a small bash shim so they exercise the
real script hermetically (no kind cluster needed).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO_ROOT / "scripts" / "capture-skypilot-state.sh"


def _write_kubectl_shim(shim_path: Path, body: str) -> None:
    """Write an executable kubectl shim with the given bash body.

    :param shim_path: Path where the shim is created (must be on PATH for the script run).
    :param body: Bash body. Has access to ``"$@"`` (the kubectl argv).
    """
    shim_path.write_text(f"#!/bin/bash\n{body}\n")
    shim_path.chmod(0o755)


def _run(
    shim_dir: Path,
    run_metadata_dir: Path | None,
    home: Path,
    *,
    expect_success: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run capture-skypilot-state.sh with a stubbed kubectl on PATH.

    :param shim_dir: Directory containing the ``kubectl`` shim. Prepended to PATH.
    :param run_metadata_dir: Value for ``RUN_METADATA_DIR``. ``None`` means leave unset.
    :param home: Value for ``HOME`` — keeps the test hermetic so the script never
        reads the developer's real home dir even if a future change adds ~ lookups.
    :param expect_success: If True, fail the test when the script returns non-zero.
    :raises AssertionError: If ``expect_success`` is True and the script exits non-zero.
    :returns: The completed subprocess.
    :rtype: subprocess.CompletedProcess[str]
    """
    env: dict[str, str] = {
        "PATH": f"{shim_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}",
        "HOME": str(home),
    }
    if run_metadata_dir is not None:
        env["RUN_METADATA_DIR"] = str(run_metadata_dir)
    result = subprocess.run(  # noqa: S603 — controlled args, hermetic env
        ["bash", str(SCRIPT)],  # noqa: S607 — bash on PATH
        env=env,
        capture_output=True,
        text=True,
    )
    if expect_success and result.returncode != 0:
        raise AssertionError(
            f"script failed rc={result.returncode}\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


# ---------------------------------------------------------------------------
# Required env validation
# ---------------------------------------------------------------------------


class TestRunMetadataDirRequired:
    """``RUN_METADATA_DIR`` is required — script must fail fast if missing."""

    def test_unset_run_metadata_dir_fails(self, tmp_path: Path) -> None:
        """Script exits non-zero with a clear error when ``RUN_METADATA_DIR`` is unset.

        :param tmp_path: pytest temp directory fixture; hosts the kubectl shim and HOME.
        """
        shim_dir = tmp_path / "shims"
        shim_dir.mkdir()
        _write_kubectl_shim(shim_dir / "kubectl", "exit 0")
        result = _run(shim_dir, run_metadata_dir=None, home=tmp_path, expect_success=False)
        assert result.returncode != 0
        assert "RUN_METADATA_DIR" in result.stderr

    def test_empty_run_metadata_dir_fails(self, tmp_path: Path) -> None:
        """Script exits non-zero when ``RUN_METADATA_DIR`` is set to the empty string.

        :param tmp_path: pytest temp directory fixture; hosts the kubectl shim and HOME.
        """
        shim_dir = tmp_path / "shims"
        shim_dir.mkdir()
        _write_kubectl_shim(shim_dir / "kubectl", "exit 0")
        # Pass an empty-string RUN_METADATA_DIR via env (Path("") is awkward — use a custom call).
        env: dict[str, str] = {
            "PATH": f"{shim_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}",
            "HOME": str(tmp_path),
            "RUN_METADATA_DIR": "",
        }
        result = subprocess.run(  # noqa: S603
            ["bash", str(SCRIPT)],  # noqa: S607
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "RUN_METADATA_DIR" in result.stderr


# ---------------------------------------------------------------------------
# Happy path — base capture files always created (even with empty kubectl output)
# ---------------------------------------------------------------------------


class TestBaseCaptureFiles:
    """The four base capture files are always written, even when kubectl emits nothing."""

    def test_creates_out_dir_and_base_files(self, tmp_path: Path) -> None:
        """With a no-op kubectl shim, the script still creates ``k8s_state/`` and the four cluster-
        overview files (pods, events, nodes, pods-yaml).

        :param tmp_path: pytest temp directory fixture; hosts the kubectl shim and
            RUN_METADATA_DIR.
        """
        shim_dir = tmp_path / "shims"
        shim_dir.mkdir()
        _write_kubectl_shim(shim_dir / "kubectl", "exit 0")
        run_dir = tmp_path / "run-meta"
        result = _run(shim_dir, run_dir, home=tmp_path)
        assert result.returncode == 0

        out_dir = run_dir / "k8s_state"
        assert out_dir.is_dir()
        for expected in ("pods.txt", "events.txt", "nodes.txt", "pods-yaml.txt"):
            assert (out_dir / expected).exists(), f"missing {expected}"

    def test_tolerates_kubectl_failures(self, tmp_path: Path) -> None:
        """If kubectl exits non-zero on every call, the script still succeeds (best-effort).

        :param tmp_path: pytest temp directory fixture; hosts the kubectl shim and
            RUN_METADATA_DIR.
        """
        shim_dir = tmp_path / "shims"
        shim_dir.mkdir()
        _write_kubectl_shim(shim_dir / "kubectl", "echo 'kubectl failed' >&2; exit 1")
        run_dir = tmp_path / "run-meta"
        result = _run(shim_dir, run_dir, home=tmp_path)
        assert result.returncode == 0
        assert (run_dir / "k8s_state").is_dir()


# ---------------------------------------------------------------------------
# Worker-pod discovery
# ---------------------------------------------------------------------------


_WORKER_LISTING_SHIM = r"""
if [[ "$1" == "get" && "$2" == "pods" && "$3" == "-n" && "$4" == "default" && "$5" == "--no-headers" ]]; then
  printf 'worker-aaa  1/1 Running 0 1m\nsky-jobs-controller-xyz  1/1 Running 0 1m\nworker-bbb  1/1 Running 0 1m\n'
  exit 0
fi
exit 0
"""


_NO_WORKERS_SHIM = r"""
if [[ "$1" == "get" && "$2" == "pods" && "$3" == "-n" && "$4" == "default" && "$5" == "--no-headers" ]]; then
  exit 0
fi
exit 0
"""


# Mimics older kubectl versions that print the "No resources found" banner to
# *stdout* instead of stderr. Without an explicit filter, the script's awk
# pipeline would pick up "No" as the pod name and write describe-worker-No.txt.
_NO_RESOURCES_BANNER_SHIM = r"""
if [[ "$1" == "get" && "$2" == "pods" && "$3" == "-n" && "$4" == "default" && "$5" == "--no-headers" ]]; then
  printf 'No resources found in default namespace.\n'
  exit 0
fi
exit 0
"""


class TestWorkerPodCapture:
    """Worker pods (non-controller pods in default namespace) get per-pod describe files."""

    def test_workers_present_creates_describe_files(self, tmp_path: Path) -> None:
        """Two worker pods → two describe-worker-<name>.txt files; controller is filtered out.

        :param tmp_path: pytest temp directory fixture; hosts the kubectl shim and
            RUN_METADATA_DIR.
        """
        shim_dir = tmp_path / "shims"
        shim_dir.mkdir()
        _write_kubectl_shim(shim_dir / "kubectl", _WORKER_LISTING_SHIM)
        run_dir = tmp_path / "run-meta"
        _run(shim_dir, run_dir, home=tmp_path)

        out_dir = run_dir / "k8s_state"
        assert (out_dir / "describe-worker-worker-aaa.txt").exists()
        assert (out_dir / "describe-worker-worker-bbb.txt").exists()
        # sky-jobs-controller is filtered out of the worker-pod loop (handled in its own branch).
        assert not (out_dir / "describe-worker-sky-jobs-controller-xyz.txt").exists()
        # Sentinel is NOT written when workers exist.
        assert not (out_dir / "describe-worker-none.txt").exists()

    def test_no_workers_writes_sentinel(self, tmp_path: Path) -> None:
        """Empty `kubectl get pods -n default` → describe-worker-none.txt sentinel.

        :param tmp_path: pytest temp directory fixture; hosts the kubectl shim and
            RUN_METADATA_DIR.
        """
        shim_dir = tmp_path / "shims"
        shim_dir.mkdir()
        _write_kubectl_shim(shim_dir / "kubectl", _NO_WORKERS_SHIM)
        run_dir = tmp_path / "run-meta"
        _run(shim_dir, run_dir, home=tmp_path)

        sentinel = run_dir / "k8s_state" / "describe-worker-none.txt"
        assert sentinel.exists()
        assert "no non-controller pods" in sentinel.read_text()

    def test_no_resources_banner_does_not_become_pod_name(self, tmp_path: Path) -> None:
        """Older kubectl prints "No resources found ..." on stdout when the namespace is empty.

        The script must filter that line out of the worker-pod list — otherwise the awk
        pipeline picks up "No" as a pod name and writes ``describe-worker-No.txt``.
        Equivalent to the empty-namespace case: the sentinel is written instead.

        :param tmp_path: pytest temp directory fixture; hosts the kubectl shim and
            RUN_METADATA_DIR.
        """
        shim_dir = tmp_path / "shims"
        shim_dir.mkdir()
        _write_kubectl_shim(shim_dir / "kubectl", _NO_RESOURCES_BANNER_SHIM)
        run_dir = tmp_path / "run-meta"
        _run(shim_dir, run_dir, home=tmp_path)

        out_dir = run_dir / "k8s_state"
        assert not (out_dir / "describe-worker-No.txt").exists()
        # "No" must not have leaked through under any spelling.
        assert not list(out_dir.glob("describe-worker-No*"))
        assert (out_dir / "describe-worker-none.txt").exists()


# ---------------------------------------------------------------------------
# Controller-pod capture
# ---------------------------------------------------------------------------


_CONTROLLER_PRESENT_SHIM = r"""
if [[ "$1" == "get" && "$2" == "pods" && "$3" == "-A" && "$4" == "--no-headers" ]]; then
  printf 'skypilot  sky-jobs-controller-abc  1/1 Running 0 5m\n'
  exit 0
fi
if [[ "$1" == "describe" && "$2" == "pod" ]]; then
  echo "describe-output-$@"
  exit 0
fi
if [[ "$1" == "logs" ]]; then
  echo "logs-output-$@"
  exit 0
fi
if [[ "$1" == "exec" ]]; then
  echo "exec-output-$@"
  exit 0
fi
exit 0
"""


_CONTROLLER_ABSENT_SHIM = r"""
if [[ "$1" == "get" && "$2" == "pods" && "$3" == "-A" && "$4" == "--no-headers" ]]; then
  exit 0
fi
exit 0
"""


class TestControllerPodCapture:
    """The sky-jobs-controller pod gets describe + logs + on-pod provision.log capture."""

    def test_controller_present_creates_describe_and_logs(self, tmp_path: Path) -> None:
        """Controller pod found → describe + logs + previous-logs + provision-log files exist.

        :param tmp_path: pytest temp directory fixture; hosts the kubectl shim and
            RUN_METADATA_DIR.
        """
        shim_dir = tmp_path / "shims"
        shim_dir.mkdir()
        _write_kubectl_shim(shim_dir / "kubectl", _CONTROLLER_PRESENT_SHIM)
        run_dir = tmp_path / "run-meta"
        _run(shim_dir, run_dir, home=tmp_path)

        out_dir = run_dir / "k8s_state"
        for expected in (
            "describe-controller.txt",
            "logs-controller.txt",
            "logs-controller-previous.txt",
            "controller-provision-log.txt",
        ):
            assert (out_dir / expected).exists(), f"missing {expected}"

        # describe-controller.txt was populated by the `kubectl describe` branch — not the
        # "no controller found" sentinel.
        assert (
            "no sky-jobs-controller pod found"
            not in (out_dir / "describe-controller.txt").read_text()
        )

    def test_controller_absent_writes_sentinel(self, tmp_path: Path) -> None:
        """No sky-jobs-controller pod → describe-controller.txt with sentinel message.

        :param tmp_path: pytest temp directory fixture; hosts the kubectl shim and
            RUN_METADATA_DIR.
        """
        shim_dir = tmp_path / "shims"
        shim_dir.mkdir()
        _write_kubectl_shim(shim_dir / "kubectl", _CONTROLLER_ABSENT_SHIM)
        run_dir = tmp_path / "run-meta"
        _run(shim_dir, run_dir, home=tmp_path)

        sentinel = run_dir / "k8s_state" / "describe-controller.txt"
        assert sentinel.exists()
        assert "no sky-jobs-controller pod found" in sentinel.read_text()
        # The exec/logs files SHOULD NOT exist when no controller was found.
        assert not (run_dir / "k8s_state" / "logs-controller.txt").exists()
        assert not (run_dir / "k8s_state" / "controller-provision-log.txt").exists()


# ---------------------------------------------------------------------------
# Worker-pod name sanitization
# ---------------------------------------------------------------------------


_WORKER_WEIRD_NAME_SHIM = r"""
if [[ "$1" == "get" && "$2" == "pods" && "$3" == "-n" && "$4" == "default" && "$5" == "--no-headers" ]]; then
  printf 'weird.pod/name  1/1 Running 0 1m\n'
  exit 0
fi
exit 0
"""


def test_worker_name_with_unsafe_chars_is_sanitized(tmp_path: Path) -> None:
    """Worker pod names with `.` or `/` are sanitized to `_` for the filename, so the describe file
    path is always a single safe basename — no accidental directory traversal from a pod name
    containing `/`.

    :param tmp_path: pytest temp directory fixture; hosts the kubectl shim and RUN_METADATA_DIR.
    """
    shim_dir = tmp_path / "shims"
    shim_dir.mkdir()
    _write_kubectl_shim(shim_dir / "kubectl", _WORKER_WEIRD_NAME_SHIM)
    run_dir = tmp_path / "run-meta"
    _run(shim_dir, run_dir, home=tmp_path)

    out_dir = run_dir / "k8s_state"
    # Sanitized: . and / become _.
    sanitized = out_dir / "describe-worker-weird_pod_name.txt"
    assert sanitized.exists(), f"sanitized file missing; got: {sorted(out_dir.iterdir())}"


# ---------------------------------------------------------------------------
# Smoke: full script runs cleanly with a stubbed kubectl
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shim_body",
    [
        "exit 0",
        "echo stub-stdout; echo stub-stderr >&2; exit 0",
        "exit 1",
    ],
    ids=["no_output", "with_output", "always_fails"],
)
def test_script_exits_zero_under_various_kubectl_behaviors(
    tmp_path: Path,
    shim_body: str,
) -> None:
    """Whatever kubectl does (success, success-with-output, failure), the script completes
    successfully — best-effort tolerance is a contract.

    :param tmp_path: pytest temp directory fixture; hosts the kubectl shim and RUN_METADATA_DIR.
    :param shim_body: Parametrized bash body installed as the kubectl shim.
    """
    shim_dir = tmp_path / "shims"
    shim_dir.mkdir()
    _write_kubectl_shim(shim_dir / "kubectl", shim_body)
    run_dir = tmp_path / "run-meta"
    result = _run(shim_dir, run_dir, home=tmp_path, expect_success=False)
    assert result.returncode == 0, (
        f"expected rc=0 for shim {shim_body!r}; got rc={result.returncode}\n"
        f"stderr={result.stderr!r}"
    )
