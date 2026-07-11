"""Invariant: image packages and launch flags keep `rclone mount` (FUSE) working.

Rationale and per-platform caveats (RunPod, OCI, devcontainers) live in
docs/reference/docker.md, section "FUSE mounts (`rclone mount`)".
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

# Pinned in the single-string "--flag=value" encoding these runArgs use; the
# two-element form ("--device", "/dev/fuse") would fail this check.
_FUSE_RUN_ARGS = ("--device=/dev/fuse", "--cap-add=SYS_ADMIN")


@pytest.mark.infra
def test_worker_dockerfile_runtime_apt_layer_installs_fuse_not_fuse3(
    project_root: Path,
) -> None:
    """The runtime apt layer ships fuse (2.x) alongside rclone, and not fuse3.

    :param project_root: Repo checkout whose Dockerfile is checked (pydoclint requires this field
        for fixture params).
    """
    dockerfile = project_root / "docker" / "ubuntu22_04" / "Dockerfile"
    text = dockerfile.read_text()
    # Scope to the runtime layer: the apt-get install block that ships rclone.
    apt_blocks = re.findall(r"apt-get install[^;]*", text)
    runtime_blocks = [block for block in apt_blocks if "rclone" in block]
    assert runtime_blocks, f"{dockerfile}: no apt-get install block ships rclone"
    # One package name per continuation line, e.g. "      rclone \".
    packages = set(re.findall(r"^\s+([a-z0-9][a-z0-9+.-]*) \\$", runtime_blocks[0], re.MULTILINE))
    assert "fuse" in packages, (
        f"{dockerfile}: runtime apt layer must install fuse so rclone 1.53's "
        f"bare `fusermount` exists when the launch path grants /dev/fuse"
    )
    # fuse3 Breaks fuse on 22.04, and rclone 1.53 execs `fusermount`, not
    # `fusermount3` — installing both fails the apt solver.
    assert "fuse3" not in packages, (
        f"{dockerfile}: fuse3 conflicts with fuse on Ubuntu 22.04 (fuse3 Breaks "
        f"fuse); install only fuse for the pinned rclone 1.53"
    )


@pytest.mark.infra
def test_every_devcontainer_run_args_grant_fuse_device_and_sys_admin(
    devcontainer_json_paths: list[Path],
) -> None:
    """Each devcontainer's runArgs include --device=/dev/fuse and --cap-add=SYS_ADMIN.

    :param devcontainer_json_paths: All .devcontainer/*/devcontainer.json files (pydoclint requires
        this field for fixture params).
    """
    for path in devcontainer_json_paths:
        run_args = json.loads(path.read_text()).get("runArgs", [])
        missing = [arg for arg in _FUSE_RUN_ARGS if arg not in run_args]
        assert not missing, (
            f"{path}: runArgs missing {missing}; without them fusermount "
            f"fails with 'failed to open /dev/fuse: Operation not permitted'"
        )


@pytest.mark.infra
def test_oci_compute_template_nested_docker_run_stays_privileged(
    project_root: Path,
) -> None:
    """The OCI template keeps --privileged, the superset grant FUSE relies on.

    :param project_root: Repo checkout whose compute template is checked (pydoclint requires this
        field for fixture params).
    """
    template = (
        project_root / "src" / "synth_setter" / "configs" / "compute" / "oci-cpu-template.yaml"
    )
    # Anchor to the docker run argument line — the flag also appears in an
    # explanatory comment, which must not satisfy this check.
    privileged_arg = re.search(r"^\s+--privileged \\$", template.read_text(), re.MULTILINE)
    assert privileged_arg, (
        f"{template}: nested docker run lost --privileged; FUSE mounts on OCI "
        f"workers need it (or explicit --device=/dev/fuse --cap-add=SYS_ADMIN)"
    )
