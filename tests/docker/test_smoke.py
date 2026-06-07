"""Docker image smoke tests.

These tests verify the built Docker image works correctly. They are designed
to run INSIDE the container via the CI workflow, not on the host.

CI runs individual tests by node ID (see docker-build-validation.yml).
To run all smoke tests manually inside a container:

    docker run --rm "$IMAGE" pytest tests/docker/test_smoke.py -m docker_smoke -v
    docker run --rm "$IMAGE" src/synth_setter/scripts/run-linux-vst-headless.sh \
        pytest tests/docker/test_smoke.py -m "docker_smoke and requires_vst" -v
"""

import pytest

from tests._vst import PLUGIN_PATH

skip_no_pedalboard = pytest.mark.skipif(
    not __import__("importlib").util.find_spec("pedalboard"),
    reason="pedalboard not installed (run inside Docker image)",
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
