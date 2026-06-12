"""On-demand coverage for the ``BUILD_MODE=source`` Docker build path.

Not wired into CI (a real source build is a ~1h compile); skipped unless
``SYNTH_SETTER_RUN_SOURCE_BUILD`` is set, so it runs only on explicit opt-in:

    SYNTH_SETTER_RUN_SOURCE_BUILD=1 pytest tests/docker/test_build_from_source.py -v -s
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

_SOURCE_BUILD_TIMEOUT_SECONDS = 5400  # CMake + Surge XT compiled from scratch; ~1h.
_SMOKE_RUN_TIMEOUT_SECONDS = 600  # matches tests/_vst.py VST_SUBPROCESS_TIMEOUT_SECONDS.
_DEV_SNAPSHOT_TAG = (
    "synth-setter:dev-snapshot"  # must match the make docker-build-dev-snapshot tag.
)

_REPO_ROOT = Path(__file__).resolve().parents[2]  # tests/docker/ -> tests/ -> repo root.

_skip_unless_opt_in = pytest.mark.skipif(
    not os.environ.get("SYNTH_SETTER_RUN_SOURCE_BUILD"),
    reason="set SYNTH_SETTER_RUN_SOURCE_BUILD to run this ~1h source build",
)
_skip_without_docker = pytest.mark.skipif(
    shutil.which("docker") is None or shutil.which("make") is None,
    reason="source build needs docker + make on PATH",
)


def _run_or_fail(
    args: list[str], *, label: str, timeout: int, capture: bool
) -> subprocess.CompletedProcess[str]:
    """Run ``args`` from the repo root; a non-zero exit calls ``pytest.fail`` and ends the test.

    :param args: Full argv, executed without a shell.
    :param label: Step name, surfaced in the failure message.
    :param timeout: Seconds before the call is killed.
    :param capture: Capture output; off lets long output stream to the terminal live.
    :returns: The completed process, for callers that need its stdout.
    """
    try:
        proc = subprocess.run(  # noqa: S603 — fixed argv, no shell; docker/git/make on PATH
            args,
            cwd=_REPO_ROOT,
            text=True,
            capture_output=capture,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"{label} timed out after {timeout}s", pytrace=False)
    if proc.returncode != 0:
        # pytest writes its failures to stdout, so merge both streams for a complete message.
        streams = "\n".join(s for s in (proc.stdout, proc.stderr) if s)
        pytest.fail(
            f"{label} failed (exit {proc.returncode})" + (f"\n{streams}" if streams else ""),
            pytrace=False,
        )
    return proc


@pytest.mark.slow
@pytest.mark.network
@_skip_unless_opt_in
@_skip_without_docker
def test_dev_snapshot_built_from_source_loads_surge_xt() -> None:
    """The source-built image must load the Surge XT it compiled, not merely import the package.

    A clean ``make`` exit or a bare ``import`` can pass on a stubbed or mistagged image;
    loading the compiled VST3 in the container proves the compile produced a usable artifact.
    """
    git_ref = _run_or_fail(
        ["git", "rev-parse", "HEAD"], label="git rev-parse", timeout=30, capture=True
    ).stdout.strip()

    _run_or_fail(
        ["make", "docker-build-dev-snapshot", f"GIT_REF={git_ref}", "DOCKER_BUILD_MODE=source"],
        label="BUILD_MODE=source build",
        timeout=_SOURCE_BUILD_TIMEOUT_SECONDS,
        capture=False,
    )

    smoke = _run_or_fail(
        [
            "docker",
            "run",
            "--rm",
            _DEV_SNAPSHOT_TAG,
            "src/synth_setter/scripts/run-linux-vst-headless.sh",
            "pytest",
            "tests/docker/test_smoke.py",
            "-m",
            "docker_smoke and requires_vst",
            "-v",
        ],
        label=f"in-container Surge XT smoke on {_DEV_SNAPSHOT_TAG}",
        timeout=_SMOKE_RUN_TIMEOUT_SECONDS,
        capture=True,
    )
    # requires_vst auto-skips (exit 0) when Surge XT is absent, so require a real pass, not a skip.
    if " passed" not in smoke.stdout:
        pytest.fail(
            f"no Surge XT smoke test passed — did the source build produce a loadable VST3?\n"
            f"{smoke.stdout}",
            pytrace=False,
        )
