"""Offline-wandb e2e for eval's ``eval-results`` artifact wiring.

The sibling ``test_eval_artifact.py`` covers the artifact builder + best-effort
logger with a real ``wandb.Artifact`` and a ``MagicMock(spec=WandbLogger)``;
nothing there drives the real ``evaluate()`` entrypoint, so the load-bearing
"log the artifact while the run is still open" ordering inside ``evaluate``
(before ``@task_wrapper`` closes the run) could regress to a no-op undetected.

This module closes that gap: it drives the real ``evaluate(cfg)`` against a
``WandbLogger(offline=True)`` and a local-backed ``r2://`` upload prefix, then
decodes the offline ``run-*.wandb`` binary to confirm the ``eval-results``
artifact actually landed on the live run.
"""

from __future__ import annotations

import glob
import os
import shutil
from pathlib import Path

import pytest
import wandb
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from synth_setter.cli.eval import evaluate
from synth_setter.data.vst import param_specs
from synth_setter.workspace import operator_workspace
from tests.helpers.wandb_offline import read_run_binary

_CONFIG_ID = "test-mps-fake-oracle"
_UPLOAD_URI = "r2://eval-artifacts/eval-run-1"


def _compose_offline_wandb_eval_cfg(
    tmp_path: Path, dataset_root: Path, upload_uri: str
) -> DictConfig:
    """Compose ``eval.yaml`` with an offline ``WandbLogger`` and an ``r2://`` upload prefix.

    Mirrors ``test_eval.py``'s in-process fake-oracle composition but adds the
    ``logger=wandb`` group pinned to ``offline=True`` (so ``evaluate`` builds a
    real, hermetic run) and an ``evaluation.upload_output_dir_uri`` so the
    artifact log path's gate (``is_global_zero and upload_uri``) fires.

    :param tmp_path: Pinned as ``paths.output_dir`` / ``paths.log_dir``; the
        offline run's ``wandb/`` dir lands beneath it via the logger's save_dir.
    :param dataset_root: Holds ``{train,val,test}.h5`` + ``stats.npz``.
    :param upload_uri: ``r2://`` prefix the output dir is mirrored to and the
        artifact references as ``s3://``.
    :returns: Composed eval ``DictConfig`` ready for ``evaluate``.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="eval.yaml",
            return_hydra_config=True,
            overrides=[
                f"experiment=surge/{_CONFIG_ID}",
                "trainer=cpu",
                # The experiment defaults to mode=predict; the artifact path is mode-agnostic
                # and test-mode gives a deterministic zero param_mse without rendering.
                "mode=test",
                f"model.net.d_out={len(param_specs['surge_4'])}",
                "callbacks.log_per_param_mse.param_spec=surge_4",
            ],
        )
    with open_dict(cfg):
        cfg.paths.root_dir = str(operator_workspace())
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.datamodule.dataset_root = str(dataset_root)
        cfg.datamodule.predict_file = str(dataset_root / "test.h5")
        cfg.datamodule.batch_size = 1
        cfg.datamodule.num_workers = 0
        cfg.ckpt_path = None
        # The experiment pins logger=null; set the wandb group inline (offline) so a
        # real WandbLogger is instantiated. log_model=False keeps the offline run to
        # the eval-results reference rather than uploading checkpoints, and console=wrap
        # avoids the redirect-drops-output non-TTY worker mode (MEMORY.md).
        cfg.logger = {
            "wandb": {
                "_target_": "lightning.pytorch.loggers.wandb.WandbLogger",
                "save_dir": str(tmp_path),
                "offline": True,
                # pin_wandb_run_id updates logger.wandb.{id,job_type} in place, so both
                # keys must exist in the struct before evaluate() instantiates the logger.
                "id": None,
                "job_type": "",
                "project": "eval-artifact-e2e-test",
                "log_model": False,
                "settings": {"_target_": "wandb.Settings", "console": "wrap"},
            }
        }
        cfg.evaluation.upload_output_dir_uri = upload_uri
    return cfg


@pytest.mark.requires_vst
@pytest.mark.slow
def test_evaluate_logs_eval_results_artifact_to_offline_wandb_run(
    tmp_path: Path,
    surge_xt_smoke_datasets: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``evaluate(cfg)`` end-to-end logs an ``eval-{config_id}`` ``eval-results`` artifact.

    Drives the real entrypoint against a ``WandbLogger(offline=True)`` and a
    local-backed ``r2://`` upload prefix (real rclone), then decodes the offline
    ``run-*.wandb`` binary to confirm the artifact landed on the live run. No
    wandb internals are mocked — name, type, ``s3://`` reference, and metadata
    are read back from the bytes the client wrote. Guards the load-bearing
    in-``evaluate`` ordering: the artifact must be logged while the run is open
    (before ``@task_wrapper`` closes it), so a regression to a post-return no-op
    leaves no artifact record in the binary and trips here.

    :param tmp_path: Hydra ``output_dir`` and the offline run's save_dir.
    :param surge_xt_smoke_datasets: Source ``{train,val,test}.h5`` + ``stats.npz``.
    :param monkeypatch: Pins a hermetic offline ``WANDB_*`` env, dummy R2 secrets,
        and the local rclone backend + cwd for the ``r2://`` upload.
    """
    if shutil.which("rclone") is None:
        pytest.skip("rclone binary not available on PATH")

    for key in [k for k in os.environ if k.startswith("WANDB_")]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DATA_DIR", str(tmp_path / "wandb-data"))
    # Dummy secrets satisfy ensure_r2_env_loaded's presence check; the local rclone
    # backend resolves r2: without dialing Cloudflare. Chdir into a remote root so a
    # URI r2://<bucket>/<key> materializes at <remote_root>/<bucket>/<key>; done here
    # (not via fake_r2_remote) so the dataset fixture's relative-path generation runs
    # against the repo cwd first.
    monkeypatch.setenv("RCLONE_CONFIG_R2_TYPE", "local")
    monkeypatch.setenv("RCLONE_CONFIG_R2_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("RCLONE_CONFIG_R2_SECRET_ACCESS_KEY", "test-secret-key")
    monkeypatch.setenv("RCLONE_CONFIG_R2_ENDPOINT", "http://localhost:0")
    wandb.teardown()

    # Compose before the chdir so Hydra + operator_workspace() resolve against the
    # repo cwd; the cfg's paths are all absolute, so the later chdir doesn't disturb it.
    cfg = _compose_offline_wandb_eval_cfg(tmp_path, surge_xt_smoke_datasets, _UPLOAD_URI)

    remote_root = tmp_path / "r2"
    remote_root.mkdir()
    monkeypatch.chdir(remote_root)

    HydraConfig().set_config(cfg)
    try:
        evaluate(cfg)
    finally:
        GlobalHydra.instance().clear()
    assert wandb.run is None, "evaluate() did not close the wandb run on return"

    offline_dirs = list((tmp_path / "wandb").glob("offline-run-*"))
    assert len(offline_dirs) == 1, (
        f"expected one offline-run dir under {tmp_path / 'wandb'}, found {offline_dirs}"
    )
    binary_files = glob.glob(str(offline_dirs[0] / "run-*.wandb"))
    assert len(binary_files) == 1, (
        f"expected one .wandb binary in {offline_dirs[0]}, found {binary_files}"
    )

    artifact_name = f"eval-{_CONFIG_ID}"
    s3_ref = "s3://eval-artifacts/eval-run-1"
    payload = read_run_binary(
        Path(binary_files[0]),
        until=lambda data: artifact_name.encode() in data and s3_ref.encode() in data,
    )
    assert artifact_name.encode() in payload, (
        f"eval-results artifact {artifact_name!r} not recorded in offline run binary — "
        "the in-evaluate artifact log may have regressed to a no-op"
    )
    assert b"eval-results" in payload, "artifact type 'eval-results' not recorded"
    assert s3_ref.encode() in payload, (
        f"upload prefix reference {s3_ref!r} not recorded on the artifact"
    )
    # Metadata block round-trips the scalar summary metrics + git_sha through the
    # real log → binary path; param_mse is the fake oracle's deterministic key.
    assert b"git_sha" in payload, "artifact metadata git_sha not recorded in offline run binary"
    assert b"test/param_mse" in payload, (
        "scalar summary metric test/param_mse not recorded in offline run binary"
    )
