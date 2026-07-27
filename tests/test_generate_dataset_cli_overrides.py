"""Production-path smoke tests for Hydra overrides on the dataset CLI."""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

import pytest

from tests._vst import PLUGIN_PATH

_ENTRYPOINT = "synth-setter-generate-dataset"
_CLI_TIMEOUT_S = 600
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_cli(override_args: str) -> None:
    """Run the installed dataset entrypoint with a Hydra-style argv tail.

    :param override_args: Arguments parsed with shell quoting before subprocess execution.
    """
    argv = [_ENTRYPOINT, *shlex.split(override_args)]
    subprocess.run(  # noqa: S603 — argv built from in-test literals
        argv, check=True, capture_output=True, text=True, timeout=_CLI_TIMEOUT_S
    )


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.r2
@pytest.mark.requires_vst
def test_generate_dataset_cli_accepts_overrides(
    fake_r2_remote: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Persist the production plugin-path override through a tiny real render.

    :param fake_r2_remote: Local filesystem backing the real rclone transport.
    :param monkeypatch: Supplies canonical storage settings for the local backend.
    """
    monkeypatch.setenv("SYNTH_SETTER_STORAGE_ACCESS_KEY_ID", "local-access-key")
    monkeypatch.setenv("SYNTH_SETTER_STORAGE_ENDPOINT_URL", "http://localhost")
    monkeypatch.setenv("SYNTH_SETTER_STORAGE_RCLONE_TYPE", "local")
    monkeypatch.setenv("SYNTH_SETTER_STORAGE_SECRET_ACCESS_KEY", "local-secret-key")
    plugin_path = Path(PLUGIN_PATH)
    resolved_plugin_path = plugin_path if plugin_path.is_absolute() else _REPO_ROOT / plugin_path
    preset_path = _REPO_ROOT / "presets" / "surge-simple.vstpreset"
    override_args = (
        "experiment=generate_dataset/smoke-shard "
        f"render.synth.plugin_path={shlex.quote(str(resolved_plugin_path))} "
        f"render.synth.plugin_state_path={shlex.quote(str(preset_path))} "
        "train_val_test_sizes=[1,1,1] "
        "render.samples_per_shard=1 "
        "render.samples_per_render_batch=1 "
        "logger=[]"
    )

    _run_cli(override_args)

    task_root = fake_r2_remote / "intermediate-data" / "data" / "smoke-shard"
    spec_paths = list(task_root.glob("*/input_spec.json"))
    assert len(spec_paths) == 1
    persisted = json.loads(spec_paths[0].read_text())
    assert persisted["render"]["synth"]["plugin_path"] == str(resolved_plugin_path)


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.r2
@pytest.mark.requires_vst
def test_generate_dataset_cli_help_accepts_experiment_override() -> None:
    """Compose the required experiment override before rendering starts."""
    _run_cli("experiment=generate_dataset/smoke-shard --help")
