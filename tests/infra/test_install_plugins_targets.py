"""`make install-plugins` provisions every VST3 bundle the runtime docker image ships.

The image (docker/ubuntu22_04/Dockerfile) installs Surge XT plus three SHA256-pinned prebuilt
synths (Dexed, OB-Xf, Six Sines). The Makefile mirrors those pins for local installs; these tests
fail when either side drifts.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAKEFILE = PROJECT_ROOT / "Makefile"
DOCKERFILE = PROJECT_ROOT / "docker" / "ubuntu22_04" / "Dockerfile"

pytestmark = pytest.mark.infra

# Bound subprocess calls so a hung make can't wedge the suite.
_TIMEOUT_S = 60

# Every VST3 bundle staged into the runtime image, by plugins/ basename.
_IMAGE_BUNDLES = ("Surge XT.vst3", "Dexed.vst3", "OB-Xf.vst3", "Six Sines.vst3")

# Pins that must stay identical between the Makefile and the Dockerfile ARGs.
_SHARED_PINS = (
    "DEXED_VERSION",
    "DEXED_SHA256",
    "OBXF_VERSION",
    "OBXF_SHA256",
    "SIX_SINES_VERSION",
    "SIX_SINES_ASSET",
    "SIX_SINES_SHA256",
)

if shutil.which("make") is None:
    pytest.skip("make not on PATH", allow_module_level=True)


def _makefile_var(name: str) -> str:
    """Return the value of a simple `NAME := value` Makefile assignment.

    :param name: variable name to look up.
    :returns: the assigned value, surrounding whitespace stripped.
    """
    match = re.search(rf"^{name}\s*:?=\s*(.+?)\s*$", MAKEFILE.read_text(), re.MULTILINE)
    assert match, f"Makefile does not define {name}"
    return match.group(1)


def _dockerfile_arg(name: str) -> str:
    """Return the default value of an `ARG NAME=value` Dockerfile instruction.

    :param name: build-arg name to look up.
    :returns: the default value, surrounding whitespace stripped.
    """
    match = re.search(rf"^ARG {name}=(.+?)\s*$", DOCKERFILE.read_text(), re.MULTILINE)
    assert match, f"Dockerfile does not define ARG {name}"
    return match.group(1)


def _run_make(cwd: Path, target: str, env: dict[str, str] | None = None) -> str:
    """Run a make target in ``cwd``, returning its stdout.

    :param cwd: directory holding the Makefile under test.
    :param target: make target to invoke.
    :param env: full environment for the subprocess; inherits os.environ when None.
    :returns: the target's stdout.
    """
    return subprocess.run(  # noqa: S603 — fixed argv, no shell
        ["make", target],  # noqa: S607 — make on PATH
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=_TIMEOUT_S,
    ).stdout


@pytest.fixture
def makefile_checkout(tmp_path: Path) -> Path:
    """Provide a throwaway directory holding only the project Makefile.

    :param tmp_path: pytest-provided scratch directory.
    :returns: the directory, ready for `make <target>` runs against it.
    """
    shutil.copy(MAKEFILE, tmp_path / "Makefile")
    return tmp_path


@pytest.mark.parametrize("pin", _SHARED_PINS)
def test_makefile_pin_matches_dockerfile_arg(pin: str) -> None:
    """Each fetched-synth pin in the Makefile equals the Dockerfile ARG default.

    :param pin: shared pin variable name.
    """
    assert _makefile_var(pin) == _dockerfile_arg(pin), (
        f"{pin} drifted between Makefile and {DOCKERFILE.relative_to(PROJECT_ROOT)}"
    )


def test_surge_version_matches_dockerfile_prebuilt_package() -> None:
    """The Makefile's Surge pin names the same release the image's prebuilt path installs."""
    version = _makefile_var("SURGE_XT_VERSION")
    assert f"{version}/surge-xt-linux-x64-{version}.deb" in DOCKERFILE.read_text(), (
        f"Dockerfile prebuilt Surge package does not match Makefile SURGE_XT_VERSION={version}"
    )


def test_install_plugins_all_bundles_present_skips_every_download(
    makefile_checkout: Path,
) -> None:
    """`make install-plugins` covers every image bundle and is a no-op when all exist.

    Pre-creating all four bundles proves the aggregate target visits each image plugin without
    touching the network.

    :param makefile_checkout: throwaway checkout holding the Makefile.
    """
    plugins = makefile_checkout / "plugins"
    plugins.mkdir()
    for name in _IMAGE_BUNDLES:
        (plugins / name).mkdir()

    stdout = _run_make(makefile_checkout, "install-plugins")

    for name in _IMAGE_BUNDLES:
        assert f"plugins/{name} already exists" in stdout, f"{name} not visited"


def test_fetched_synth_target_non_x86_64_skips_without_installing(
    makefile_checkout: Path,
) -> None:
    """On a non-x86_64 host the fetched-synth targets skip, mirroring the image's amd64 gate.

    :param makefile_checkout: throwaway checkout holding the Makefile.
    """
    bindir = makefile_checkout / "bin"
    bindir.mkdir()
    fake_uname = bindir / "uname"
    fake_uname.write_text(
        '#!/bin/sh\nif [ "$1" = "-m" ]; then echo aarch64; else echo Linux; fi\n'
    )
    fake_uname.chmod(0o755)
    env = {**os.environ, "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}"}

    stdout = _run_make(makefile_checkout, "install-dexed", env=env)

    assert "skipping Dexed" in stdout
    assert not (makefile_checkout / "plugins" / "Dexed.vst3").exists()
