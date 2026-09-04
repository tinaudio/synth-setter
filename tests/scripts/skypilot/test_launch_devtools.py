"""Tests for the persistent SkyPilot devtools launcher."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import sky

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SCRIPT = REPO_ROOT / "scripts" / "skypilot" / "launch-devtools.sh"
TASK_YAML = REPO_ROOT / "scripts" / "skypilot" / "synth-devtools.yaml"


def test_task_yaml_loads_as_persistent_devtools_task() -> None:
    """SkyPilot accepts both GPU alternatives and the persistent devtools task."""
    task = sky.Task.from_yaml(str(TASK_YAML))

    resources = list(task.resources)
    assert {frozenset((resource.accelerators or {}).items()) for resource in resources} == {
        frozenset({"RTX3090": 1}.items()),
        frozenset({"RTX4090": 1}.items()),
    }
    assert all(resource.disk_size == 200 for resource in resources)
    assert all(
        resource.image_id == {None: "docker:tinaudio/synth-setter:devcontainer-tools"}
        for resource in resources
    )
    assert task.run == "exec sleep infinity\n"


def test_launcher_from_other_directory_uses_checked_in_task(tmp_path: Path) -> None:
    """Launcher resolves its task file independently of the caller's directory.

    :param tmp_path: Directory for the hermetic SkyPilot shim and its output.
    """
    sky_shim = tmp_path / "sky"
    argv_file = tmp_path / "argv"
    sky_shim.write_text('#!/bin/bash\nprintf \'%s\\n\' "$@" > "${SKY_ARGV_FILE}"\n')
    sky_shim.chmod(0o755)
    env = {
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "SKY_ARGV_FILE": str(argv_file),
    }

    subprocess.run(  # noqa: S603 — controlled script with a hermetic PATH
        ["bash", str(SCRIPT)],  # noqa: S607 — bash is resolved from the test PATH
        cwd=tmp_path,
        env=env,
        check=True,
    )

    assert argv_file.read_text().splitlines() == [
        "launch",
        str(TASK_YAML),
        "-c",
        "synth-devtools-02",
        "-d",
        "-r",
        "-y",
    ]
