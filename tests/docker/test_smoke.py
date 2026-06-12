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

import pytest
from _pytest.mark import ParameterSet

from tests._vst import PLUGIN_PATH

skip_no_pedalboard = pytest.mark.skipif(
    not __import__("importlib").util.find_spec("pedalboard"),
    reason="pedalboard not installed (run inside Docker image)",
)

# Extra synths baked into the image by the vst3-synths-fetch Dockerfile stage
# (amd64 only). plugin_name disambiguates multi-plugin bundles; None means the
# bundle exposes a single plugin. Skip (not fail) when a bundle is absent so
# the suite stays green on hosts without the baked plugins; the Dockerfile's
# build-time validation is the hard gate that the image itself has them.
EXTRA_SYNTH_VST3S = [
    ("/usr/lib/vst3/Dexed.vst3", None),
    ("/usr/lib/vst3/Vital.vst3", None),
    ("/usr/lib/vst3/Six Sines.vst3", "Six Sines"),
    ("/usr/lib/vst3/Cardinal.vst3", None),
]


def _extra_synth_params() -> Iterator[ParameterSet]:
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
    from synth_setter.resources import vst_headless_wrapper

    # Loading several VST3s sequentially in one process is order-dependently
    # crashy (a Six Sines load after Dexed+Vital segfaults), so each load runs
    # in its own subprocess under the headless X11 wrapper — the same
    # isolation boundary tests/conftest.py uses for dataset-generation
    # subprocesses.
    load_script = (
        "import sys; from pedalboard import VST3Plugin; "
        "p = VST3Plugin(sys.argv[1], plugin_name=(sys.argv[2] or None)); "
        "print(f'param_count={len(p.parameters)}')"
    )
    result = subprocess.run(  # noqa: S603 — fixed argv, no shell
        [
            str(vst_headless_wrapper()),
            sys.executable,
            "-c",
            load_script,
            bundle_path,
            plugin_name or "",
        ],
        capture_output=True,
        text=True,
        # Same generous ceiling as conftest's _VST_SUBPROCESS_TIMEOUT_SECONDS;
        # the slowest load (Six Sines) takes ~155s.
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, f"load failed for {bundle_path}:\n{result.stderr}"
    param_count = int(result.stdout.split("param_count=")[1].split()[0])
    assert param_count > 0
