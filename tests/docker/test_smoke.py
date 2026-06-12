"""Docker image smoke tests.

These tests verify the built Docker image works correctly. They are designed
to run INSIDE the container via the CI workflow, not on the host.

CI runs the import check by node ID and the VST loads by marker
(see docker-build-validation.yml). To run all smoke tests manually inside
a container:

    docker run --rm "$IMAGE" pytest tests/docker/test_smoke.py -m docker_smoke -v
    docker run --rm "$IMAGE" src/synth_setter/scripts/run-linux-vst-headless.sh \
        pytest tests/docker/test_smoke.py -m "docker_smoke and requires_vst" -v
"""

import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from synth_setter.resources import as_file, vst_headless_wrapper
from tests._vst import PLUGIN_PATH, VST_SUBPROCESS_TIMEOUT_SECONDS

if TYPE_CHECKING:
    from _pytest.mark import ParameterSet

skip_no_pedalboard = pytest.mark.skipif(
    not __import__("importlib").util.find_spec("pedalboard"),
    reason="pedalboard not installed (run inside Docker image)",
)

# Synths baked in by the Dockerfile's vst3-synths-fetch stage (amd64 only);
# the second element pins the plugin to instantiate in multi-plugin bundles.
# Absent bundles skip rather than fail — the Dockerfile's build-time
# validation is the hard gate that the image itself has them.
EXTRA_SYNTH_VST3S = (
    ("/usr/lib/vst3/Dexed.vst3", None),
    ("/usr/lib/vst3/Vital.vst3", None),
    ("/usr/lib/vst3/Six Sines.vst3", "Six Sines"),
    ("/usr/lib/vst3/Cardinal.vst3", None),
)


def _extra_synth_params() -> Iterator["ParameterSet"]:
    """Yield one pytest param per baked-in synth, skipping absent bundles.

    :yields: One ``(bundle_path, plugin_name)`` param per synth.
    :ytype: ParameterSet
    """
    for bundle_path, plugin_name in EXTRA_SYNTH_VST3S:
        yield pytest.param(
            bundle_path,
            plugin_name,
            id=Path(bundle_path).stem,
            marks=pytest.mark.skipif(
                not Path(bundle_path).exists(),
                reason=f"{bundle_path} not installed (run inside Docker image)",
            ),
        )


@pytest.mark.docker_smoke
@skip_no_pedalboard
def test_pedalboard_importable():
    """Verify pedalboard is installed and VST3Plugin class is available."""
    from pedalboard import VST3Plugin

    assert VST3Plugin is not None


@pytest.mark.docker_smoke
@pytest.mark.requires_vst
@skip_no_pedalboard
def test_surge_xt_loads():
    """Verify Surge XT VST3 plugin loads and exposes parameters."""
    from pedalboard import VST3Plugin

    plugin = VST3Plugin(PLUGIN_PATH)
    assert len(plugin.parameters) > 0  # type: ignore[attr-defined]


@pytest.mark.docker_smoke
@pytest.mark.requires_vst
@pytest.mark.slow
@skip_no_pedalboard
@pytest.mark.parametrize(("bundle_path", "plugin_name"), _extra_synth_params())
def test_extra_synth_vst3_loads(bundle_path: str, plugin_name: str | None) -> None:
    """Verify each baked-in synth VST3 loads and exposes parameters.

    :param bundle_path: Absolute path of the ``.vst3`` bundle under test.
    :param plugin_name: Plugin to instantiate for multi-plugin bundles, or
        ``None`` for single-plugin bundles.
    """
    # One subprocess per load — sequential in-process loads crash
    # order-dependently (#1649). Same check the Dockerfile build runs.
    # as_file materializes the wrapper to a real path (resources.py contract).
    with as_file(vst_headless_wrapper()) as wrapper_path:
        load_args = [
            str(wrapper_path),
            sys.executable,
            "-X",
            "faulthandler",
            "-m",
            "synth_setter.scripts.load_vst3_check",
            bundle_path,
            plugin_name or "",
        ]
        # capture_output stays off to avoid the fork-inherited-fd pipe deadlock
        # documented in tests/conftest.py (#695); the exit code is the contract.
        try:
            result = subprocess.run(  # noqa: S603 — fixed argv, no shell
                load_args,
                text=True,
                check=False,
                timeout=VST_SUBPROCESS_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            pytest.fail(
                f"load_vst3_check timed out after {VST_SUBPROCESS_TIMEOUT_SECONDS}s\n"
                f"command: {load_args}\n"
                f"(child stdout/stderr printed above; rerun with `pytest -s` if captured)",
                pytrace=False,
            )
    if result.returncode != 0:
        pytest.fail(
            f"load_vst3_check failed for {bundle_path} (exit {result.returncode})\n"
            f"command: {load_args}\n"
            f"(child stdout/stderr printed above; rerun with `pytest -s` if captured)",
            pytrace=False,
        )
