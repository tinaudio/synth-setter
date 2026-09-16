"""Tests for eval's R2 output-dir upload path.

Covers the in-process ``_maybe_upload_output_dir`` helper and two end-to-end
runs of the ``synth-setter-eval`` CLI that exercise R2 dataset download and
output-dir upload through a local-backed ``rclone`` remote.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from uuid import UUID

import pytest
from omegaconf import DictConfig, OmegaConf

from synth_setter.cli.eval import _maybe_upload_output_dir


def _upload_cfg(output_dir: Path, upload_output_dir_uri: str | None) -> DictConfig:
    """Build the minimal cfg slice ``_maybe_upload_output_dir`` reads.

    :param output_dir: Resolves to ``cfg.paths.output_dir`` — the tree to copy.
    :param upload_output_dir_uri: Resolves to ``cfg.evaluation.upload_output_dir_uri``.
    :returns: A :class:`DictConfig` carrying only the keys the helper reads.
    """
    return OmegaConf.create(  # type: ignore[no-any-return]
        {
            "paths": {"output_dir": str(output_dir)},
            "evaluation": {"upload_output_dir_uri": upload_output_dir_uri},
        }
    )


@pytest.fixture()
def storage_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set canonical storage credentials for helpers that ping object storage.

    The dummy values satisfy the presence check while rclone resolves the local backend instead of
    dialing Cloudflare.

    :param monkeypatch: Sets the secret env vars for the test's duration.
    """
    monkeypatch.setenv("SYNTH_SETTER_STORAGE_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("SYNTH_SETTER_STORAGE_SECRET_ACCESS_KEY", "test-secret-key")
    monkeypatch.setenv("SYNTH_SETTER_STORAGE_ENDPOINT_URL", "http://localhost:0")
    monkeypatch.setenv("SYNTH_SETTER_STORAGE_RCLONE_TYPE", "local")


def _storage_env() -> dict[str, str]:
    """Return dummy canonical storage env for subprocess CLI tests.

    :returns: Environment variables that select the local rclone backend.
    """
    return {
        "SYNTH_SETTER_STORAGE_ACCESS_KEY_ID": "stub",
        "SYNTH_SETTER_STORAGE_SECRET_ACCESS_KEY": "stub",
        "SYNTH_SETTER_STORAGE_ENDPOINT_URL": "http://localhost:0",
        "SYNTH_SETTER_STORAGE_RCLONE_TYPE": "local",
    }


def _write_output_tree(output_dir: Path) -> None:
    """Populate ``output_dir`` with a nested file and a top-level file to mirror.

    :param output_dir: Created here, then filled with the two-level tree.
    """
    (output_dir / "predictions").mkdir(parents=True)
    (output_dir / "predictions" / "pred.json").write_text('{"ok": true}')
    (output_dir / "metrics.json").write_text('{"param_mse": 0.0}')


def test_maybe_upload_output_dir_noop_when_uri_unset(fake_r2_remote: Path, tmp_path: Path) -> None:
    """A null URI lands no objects in the remote.

    :param fake_r2_remote: Local-backed ``r2:`` remote; its tree is asserted empty.
    :param tmp_path: Holds the output dir that a non-null URI would have mirrored.
    """
    output_dir = tmp_path / "run"
    _write_output_tree(output_dir)

    published_uri = _maybe_upload_output_dir(
        _upload_cfg(output_dir, upload_output_dir_uri=None), is_global_zero=True
    )

    assert published_uri is None
    assert list(fake_r2_remote.glob("bucket/**/*")) == []


def test_maybe_upload_output_dir_skips_non_global_zero_rank(
    fake_r2_remote: Path, tmp_path: Path
) -> None:
    """A non-global-zero rank lands no objects even when a URI is set.

    Under DDP every rank runs ``main`` against the one shared ``output_dir``;
    only rank zero may copy it so the other ranks don't race redundant uploads.

    :param fake_r2_remote: Local-backed ``r2:`` remote; its tree is asserted empty.
    :param tmp_path: Holds the output dir rank zero would have mirrored.
    """
    output_dir = tmp_path / "run"
    _write_output_tree(output_dir)

    published_uri = _maybe_upload_output_dir(
        _upload_cfg(output_dir, "r2://bucket/evals/run-1"), is_global_zero=False
    )

    assert published_uri is None
    assert list(fake_r2_remote.glob("bucket/**/*")) == []


def test_maybe_upload_output_dir_publishes_under_generated_attempt_id(
    fake_r2_remote: Path, storage_credentials: None, tmp_path: Path
) -> None:
    """A suite root publishes the whole run beneath a generated attempt ID.

    :param fake_r2_remote: Local-backed ``r2:`` remote where the attempt lands.
    :param storage_credentials: Dummy secrets so the real credential check passes.
    :param tmp_path: Holds the output dir copied to R2.
    """
    output_dir = tmp_path / "run"
    _write_output_tree(output_dir)
    (output_dir / "wandb" / "run-1").mkdir(parents=True)
    (output_dir / "wandb" / "run-1" / "run.wandb").write_text("redundant run state")

    published_uri = _maybe_upload_output_dir(
        _upload_cfg(output_dir, "r2://bucket/evals/suite/"), is_global_zero=True
    )

    assert published_uri is not None
    attempt_id = published_uri.rsplit("/", maxsplit=1)[-1]
    assert UUID(attempt_id).hex == attempt_id
    destination = fake_r2_remote / "bucket" / "evals" / "suite" / attempt_id
    assert (destination / "predictions" / "pred.json").read_text() == '{"ok": true}'
    assert (destination / "metrics.json").read_text() == '{"param_mse": 0.0}'
    assert (destination / "wandb" / "run-1" / "run.wandb").read_text() == ("redundant run state")


def test_maybe_upload_output_dir_keeps_attempts_isolated(
    fake_r2_remote: Path, storage_credentials: None, tmp_path: Path
) -> None:
    """Two invocations under one suite root retain independent payloads.

    :param fake_r2_remote: Local-backed ``r2:`` remote holding both attempts.
    :param storage_credentials: Dummy secrets so the real credential check passes.
    :param tmp_path: Holds the two independent local run directories.
    """
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    _write_output_tree(first_output)
    _write_output_tree(second_output)
    (first_output / "predictions" / "pred.json").write_text("failed-attempt-evidence")
    (second_output / "predictions" / "pred.json").write_text("successful-retry")

    first_uri = _maybe_upload_output_dir(
        _upload_cfg(first_output, "r2://bucket/evals/suite"), is_global_zero=True
    )
    second_uri = _maybe_upload_output_dir(
        _upload_cfg(second_output, "r2://bucket/evals/suite"), is_global_zero=True
    )

    assert first_uri is not None
    assert second_uri is not None
    assert first_uri != second_uri
    first_attempt_id = first_uri.rsplit("/", maxsplit=1)[-1]
    second_attempt_id = second_uri.rsplit("/", maxsplit=1)[-1]
    suite_root = fake_r2_remote / "bucket" / "evals" / "suite"
    assert (suite_root / first_attempt_id / "predictions" / "pred.json").read_text() == (
        "failed-attempt-evidence"
    )
    assert (suite_root / second_attempt_id / "predictions" / "pred.json").read_text() == (
        "successful-retry"
    )


def test_maybe_upload_output_dir_rejects_non_r2_uri(tmp_path: Path) -> None:
    """A non-``r2://`` URI fails on the URI shape before any credential ping.

    Validating the URI first attributes a misconfiguration to the URI itself
    rather than surfacing it as a confusing credentials/auth error from the
    ``ensure_r2_env_loaded`` ping that would otherwise run first.

    :param tmp_path: Holds the output dir the rejected upload would have copied.
    """
    output_dir = tmp_path / "run"
    _write_output_tree(output_dir)

    with pytest.raises(ValueError, match="must be an r2:// URI"):
        _maybe_upload_output_dir(
            _upload_cfg(output_dir, "s3://bucket/evals/run-1"), is_global_zero=True
        )


@pytest.mark.requires_vst
@pytest.mark.slow
def test_eval_cli_downloads_dataset_from_r2_then_scores_oracle(
    tmp_path: Path, surge_xt_smoke_datasets: Path
) -> None:
    """End-to-end through the ``synth-setter-eval`` CLI: R2 prefetch then oracle scoring.

    No in-process shortcuts and no mocks — the real entrypoint runs with real
    ``rclone`` (local-backed remote). The test split staged under an ``r2://`` prefix
    is downloaded into an initially absent ``datamodule.dataset_root``, and the fake
    oracle's exact-zero ``test/param_mse`` reaches ``metrics.json``.

    :param tmp_path: Root for the fake R2 remote, the download target, and the output dir.
    :param surge_xt_smoke_datasets: Source ``{train,val,test}.lance`` + ``stats.npz``.
    """
    if shutil.which("rclone") is None:
        pytest.skip("rclone binary not available on PATH")

    remote_root = tmp_path / "r2"
    staged = remote_root / "intermediate-data" / "dataset"
    staged.mkdir(parents=True)
    # Lance splits are directories; stats.npz is a plain file.
    for name in ("train.lance", "val.lance", "test.lance"):
        shutil.copytree(surge_xt_smoke_datasets / name, staged / name)
    shutil.copy(surge_xt_smoke_datasets / "stats.npz", staged / "stats.npz")
    (staged / "dataset.complete").touch()

    dataset_root = tmp_path / "downloaded"
    output_dir = tmp_path / "out"

    env = {
        **os.environ,
        **_storage_env(),
    }
    proc = subprocess.run(  # noqa: S603 — controlled argv
        [
            sys.executable,
            "-m",
            "synth_setter.cli.eval",
            "experiment=surge/test-mps-fake-oracle",
            "trainer=cpu",
            "mode=test",
            # render defaults to null and is read only in mode=predict's
            # postprocessing, so mode=test needs no render group.
            "hydra.job.chdir=false",
            "synth=surge_4",
            "datamodule.download_dataset_root_uri=r2://intermediate-data/dataset",
            f"datamodule.dataset_root={dataset_root}",
            f"datamodule.predict_file={dataset_root}/test.lance",
            "datamodule.batch_size=1",
            "datamodule.num_workers=0",
            "ckpt_path=null",
            f"paths.output_dir={output_dir}",
            f"hydra.run.dir={output_dir}",
        ],
        cwd=remote_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    downloaded_test_splits = list(dataset_root.rglob("test.lance"))
    assert len(downloaded_test_splits) == 1
    assert len(list(dataset_root.rglob("stats.npz"))) == 1

    metrics = json.loads((output_dir / "metrics" / "metrics.json").read_text())
    assert metrics["test/param_mse"] == 0.0


@pytest.mark.requires_vst
@pytest.mark.slow
def test_eval_cli_uploads_output_dir_to_r2(tmp_path: Path, surge_xt_smoke_datasets: Path) -> None:
    """End-to-end through the ``synth-setter-eval`` CLI: oracle scoring then R2 upload.

    No in-process shortcuts and no mocks — the real entrypoint runs with real
    ``rclone`` (local-backed remote). The configured suite root receives the whole
    run beneath the automatically generated attempt ID, and uploaded metrics carry
    the oracle's exact-zero ``test/param_mse``.

    :param tmp_path: Root for the fake R2 remote and the local output dir.
    :param surge_xt_smoke_datasets: Source ``{train,val,test}.lance`` + ``stats.npz``.
    """
    if shutil.which("rclone") is None:
        pytest.skip("rclone binary not available on PATH")

    remote_root = tmp_path / "r2"
    remote_root.mkdir()
    log_dir = tmp_path / "logs"
    upload_uri = "r2://eval-artifacts/oracle-suite"

    env = {
        **os.environ,
        **_storage_env(),
    }
    proc = subprocess.run(  # noqa: S603 — controlled argv
        [
            sys.executable,
            "-m",
            "synth_setter.cli.eval",
            "experiment=surge/test-mps-fake-oracle",
            "trainer=cpu",
            "mode=test",
            "hydra.job.chdir=false",
            "synth=surge_4",
            f"datamodule.dataset_root={surge_xt_smoke_datasets}",
            f"datamodule.predict_file={surge_xt_smoke_datasets}/test.lance",
            "datamodule.batch_size=1",
            "datamodule.num_workers=0",
            "ckpt_path=null",
            f"paths.log_dir={log_dir}",
            f"evaluation.upload_output_dir_uri={upload_uri}",
        ],
        cwd=remote_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    local_metrics = list(log_dir.glob("**/metrics/metrics.json"))
    assert len(local_metrics) == 1, f"expected one eval run, found {local_metrics}"

    uploaded_attempts = list((remote_root / "eval-artifacts" / "oracle-suite").iterdir())
    assert len(uploaded_attempts) == 1
    uploaded_root = uploaded_attempts[0]
    assert UUID(uploaded_root.name).hex == uploaded_root.name
    uploaded_metrics = json.loads((uploaded_root / "metrics" / "metrics.json").read_text())
    assert uploaded_metrics["test/param_mse"] == 0.0
    assert (uploaded_root / "eval.log").is_file()
