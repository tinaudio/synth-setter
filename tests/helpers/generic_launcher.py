"""Subprocess harness for generic launcher-to-worker entrypoint tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _write_dispatch_patch(tmp_path: Path) -> None:
    """Patch the subprocess's SkyPilot boundary to execute its wrapped command.

    :param tmp_path: Directory receiving the Python startup hook.
    """
    (tmp_path / "sitecustomize.py").write_text(
        "import os\n"
        "import shlex\n"
        "import subprocess\n"
        "import synth_setter.pipeline.skypilot_launch as launcher\n"
        "def run_worker(sky_cfg):\n"
        "    checkout = shlex.quote(os.environ['WORKER_REPO'])\n"
        "    command = sky_cfg.cmd.replace('/home/build/synth-setter', checkout, 1)\n"
        "    command = command.replace('bash scripts/sync_worker_checkout.sh', "
        "'bash scripts/sync_worker_checkout.sh --python-ready', 1)\n"
        "    env = {**os.environ, 'WORKER_GIT_REF': ''}\n"
        "    subprocess.run(['/bin/bash', '-c', command], check=True, env=env)\n"
        "launcher.dispatch_via_skypilot = run_worker\n",
        encoding="utf-8",
    )


def run_generic_launcher_command(
    tmp_path: Path, worker_command: str, repo_root: Path
) -> subprocess.CompletedProcess[str]:
    """Run a worker command through the real decorated launcher without provisioning.

    The subprocess patch replaces only the external SkyPilot dispatch boundary. Its
    replacement executes the wrapped command in the current checkout; the real sync
    script runs in ``--python-ready`` mode without changing the tested worktree ref.

    :param tmp_path: Directory for the subprocess patch and Hydra output.
    :param worker_command: Command supplied through ``skypilot_launch.cmd``.
    :param repo_root: Current worktree used in place of the container checkout.
    :return: Completed launcher subprocess.
    """
    launcher = Path(sys.executable).with_name("synth-setter-skypilot-launch")
    _write_dispatch_patch(tmp_path)
    env = {
        **os.environ,
        "PATH": os.pathsep.join((str(Path(sys.executable).parent), os.environ["PATH"])),
        "PYTHONPATH": os.pathsep.join(filter(None, (str(tmp_path), os.environ.get("PYTHONPATH")))),
        "WORKER_REPO": str(repo_root),
    }
    return subprocess.run(  # noqa: S603 - real packaged launcher and worker CLIs
        [
            launcher,
            "skypilot_launch/compute=runpod/smoke",
            f"skypilot_launch.cmd={json.dumps(worker_command)}",
            f"hydra.run.dir={tmp_path / 'launcher'}",
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )
