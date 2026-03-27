"""Docker image smoke tests.

These tests verify the built Docker image works correctly. They are designed
to run INSIDE the container via the CI workflow, not on the host.

Usage from workflow:
    docker run --rm "$IMAGE" pytest tests/docker/test_smoke.py -m docker_smoke -v
    docker run --rm "$IMAGE" scripts/run-linux-vst-headless.sh \
        pytest tests/docker/test_smoke.py -m "docker_smoke and requires_vst" -v
"""

import pytest


@pytest.mark.docker_smoke
def test_pedalboard_importable():
    """Verify pedalboard is installed and VST3Plugin class is available."""
    from pedalboard import VST3Plugin

    assert VST3Plugin is not None


@pytest.mark.docker_smoke
@pytest.mark.requires_vst
def test_surge_xt_loads():
    """Verify Surge XT VST3 plugin loads and exposes parameters."""
    from pedalboard import VST3Plugin

    plugin = VST3Plugin("/usr/lib/vst3/Surge XT.vst3")
    assert len(plugin.parameters) > 0  # type: ignore[attr-defined]
