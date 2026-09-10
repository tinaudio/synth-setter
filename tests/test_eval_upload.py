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

import pytest
from omegaconf import DictConfig, OmegaConf

from synth_setter.cli.eval import (
    _maybe_upload_output_dir,
    _start_output_publication_attempt,
)


def _upload_cfg(output_dir: Path, upload_output_dir_uri: str | None) -> DictConfig:
    """Build the minimal cfg slice ``_maybe_upload_output_dir`` reads.

    :param output_dir: Resolves to ``cfg.paths.output_dir`` — the tree to copy.
    :param upload_output_dir_uri: Resolves to ``cfg.evaluation.upload_output_dir_uri``.
    :returns: A :class:`DictConfig` carrying only the two keys the helper reads.
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
    """Populate ``output_dir`` with representative semantic output files.

    :param output_dir: Created here, then filled with the two-level tree.
    """
    (output_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions" / "pred.json").write_text('{"ok": true}')
    (output_dir / "metrics").mkdir()
    (output_dir / "metrics" / "metrics.json").write_text('{"param_mse": 0.0}')


def test_maybe_upload_output_dir_noop_when_uri_unset(fake_r2_remote: Path, tmp_path: Path) -> None:
    """A null URI lands no objects in the remote.

    :param fake_r2_remote: Local-backed ``r2:`` remote; its tree is asserted empty.
    :param tmp_path: Holds the output dir that a non-null URI would have mirrored.
    """
    output_dir = tmp_path / "run"
    _write_output_tree(output_dir)

    attempt = _start_output_publication_attempt(output_dir)
    _maybe_upload_output_dir(
        _upload_cfg(output_dir, upload_output_dir_uri=None),
        is_global_zero=True,
        attempt=attempt,
    )

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

    attempt = _start_output_publication_attempt(output_dir)
    _maybe_upload_output_dir(
        _upload_cfg(output_dir, "r2://bucket/evals/run-1"),
        is_global_zero=False,
        attempt=attempt,
    )

    assert list(fake_r2_remote.glob("bucket/**/*")) == []


def test_maybe_upload_output_dir_retry_publishes_only_successful_attempt(
    fake_r2_remote: Path, storage_credentials: None, tmp_path: Path
) -> None:
    """A retry publishes only files written by its successful attempt.

    :param fake_r2_remote: Local-backed ``r2:`` remote where publication lands.
    :param storage_credentials: Dummy secrets so the real credential check passes.
    :param tmp_path: Holds the reused output workspace.
    """
    output_dir = tmp_path / "run"
    failed_attempt = _start_output_publication_attempt(output_dir)
    (output_dir / "predictions" / "failed.json").write_text("failed")
    (output_dir / "metrics").mkdir()
    (output_dir / "metrics" / "metrics.json").write_text('{"failed": true}')
    (output_dir / "wandb" / "run-failed").mkdir(parents=True)
    (output_dir / "wandb" / "run-failed" / "run.wandb").write_text("operational")
    del failed_attempt

    retry_attempt = _start_output_publication_attempt(output_dir)
    (output_dir / "predictions" / "current.json").write_text("current")
    (output_dir / "metrics").mkdir()
    (output_dir / "metrics" / "metrics.json").write_text('{"param_mse": 0.0}')

    payload_uri = _maybe_upload_output_dir(
        _upload_cfg(output_dir, "r2://bucket/evals/run-1"),
        is_global_zero=True,
        attempt=retry_attempt,
    )

    publication_root = fake_r2_remote / "bucket" / "evals" / "run-1"
    manifest = json.loads((publication_root / "latest.json").read_text())
    assert payload_uri == manifest["payload_uri"]
    payload_root = publication_root / "attempts" / manifest["attempt_id"]
    assert (payload_root / "predictions" / "current.json").read_text() == "current"
    assert (payload_root / "metrics" / "metrics.json").is_file()
    assert not (payload_root / "predictions" / "failed.json").exists()
    assert not (payload_root / "wandb").exists()


def test_maybe_upload_output_dir_attempt_payload_is_immutable(
    fake_r2_remote: Path, storage_credentials: None, tmp_path: Path
) -> None:
    """Republishing one attempt cannot replace its remote payload.

    :param fake_r2_remote: Local-backed remote holding the first payload.
    :param storage_credentials: Dummy secrets so the real credential check passes.
    :param tmp_path: Holds the output workspace.
    """
    output_dir = tmp_path / "run"
    attempt = _start_output_publication_attempt(output_dir)
    (output_dir / "metrics").mkdir(parents=True)
    metrics_path = output_dir / "metrics" / "metrics.json"
    metrics_path.write_text('{"value": 1}')
    cfg = _upload_cfg(output_dir, "r2://bucket/evals/run-1")
    _maybe_upload_output_dir(cfg, is_global_zero=True, attempt=attempt)

    metrics_path.write_text('{"value": 2}')
    with pytest.raises(subprocess.CalledProcessError):
        _maybe_upload_output_dir(cfg, is_global_zero=True, attempt=attempt)

    payload_root = fake_r2_remote / "bucket" / "evals" / "run-1" / "attempts" / attempt.attempt_id
    assert json.loads((payload_root / "metrics" / "metrics.json").read_text()) == {"value": 1}


def test_maybe_upload_output_dir_payload_failure_does_not_commit_pointer(
    fake_r2_remote: Path, storage_credentials: None, tmp_path: Path
) -> None:
    """A failed real-rclone payload upload leaves no publication pointer.

    :param fake_r2_remote: Local-backed remote made invalid at the payload prefix.
    :param storage_credentials: Dummy secrets so the real credential check passes.
    :param tmp_path: Holds the output workspace.
    """
    output_dir = tmp_path / "run"
    attempt = _start_output_publication_attempt(output_dir)
    _write_output_tree(output_dir)
    publication_root = fake_r2_remote / "bucket" / "evals" / "run-1"
    publication_root.mkdir(parents=True)
    (publication_root / "attempts").write_text("blocks attempt directories")

    with pytest.raises(subprocess.CalledProcessError):
        _maybe_upload_output_dir(
            _upload_cfg(output_dir, "r2://bucket/evals/run-1"),
            is_global_zero=True,
            attempt=attempt,
        )

    assert not (publication_root / "latest.json").exists()


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
            _upload_cfg(output_dir, "s3://bucket/evals/run-1"),
            is_global_zero=True,
            attempt=_start_output_publication_attempt(output_dir),
        )


@pytest.mark.requires_vst
@pytest.mark.slow
def test_eval_cli_downloads_dataset_from_r2_then_scores_oracle(
    tmp_path: Path, surge_xt_smoke_datasets: Path
) -> None:
    """End-to-end through the ``synth-setter-eval`` CLI: R2 prefetch then oracle scoring.

    No in-process shortcuts and no mocks — the real entrypoint runs with real
    ``rclone`` (local-backed remote). A dataset staged under an ``r2://`` prefix is
    downloaded into an initially-absent ``data.dataset_root``, and the fake oracle's
    exact-zero ``test/param_mse`` reaches ``metrics.json``. Proves the new
    ``data.download_dataset_root_uri`` gate composes with eval through ``main``.

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

    for split in ("train.lance", "val.lance", "test.lance"):
        assert (dataset_root / split).is_dir(), f"{split} was not downloaded from R2"
    assert (dataset_root / "stats.npz").is_file(), "stats.npz was not downloaded from R2"

    metrics = json.loads((output_dir / "metrics" / "metrics.json").read_text())
    assert metrics["test/param_mse"] == 0.0


@pytest.mark.requires_vst
@pytest.mark.slow
def test_eval_cli_uploads_output_dir_to_r2(tmp_path: Path, surge_xt_smoke_datasets: Path) -> None:
    """End-to-end through the ``synth-setter-eval`` CLI: oracle scoring then R2 upload.

    No in-process shortcuts and no mocks — the real entrypoint runs with real
    ``rclone`` (local-backed remote). With ``evaluation.upload_output_dir_uri`` set,
    ``main``'s final step mirrors the whole run dir to that prefix, so every file
    the eval wrote locally must reappear beneath the destination and the uploaded
    ``metrics.json`` must carry the oracle's exact-zero ``test/param_mse``.

    :param tmp_path: Root for the fake R2 remote and the local output dir.
    :param surge_xt_smoke_datasets: Source ``{train,val,test}.lance`` + ``stats.npz``.
    """
    if shutil.which("rclone") is None:
        pytest.skip("rclone binary not available on PATH")

    remote_root = tmp_path / "r2"
    remote_root.mkdir()
    output_dir = tmp_path / "out"
    (output_dir / "predictions").mkdir(parents=True)
    (output_dir / "predictions" / "failed.json").write_text("failed")
    (output_dir / "wandb" / "failed-run").mkdir(parents=True)
    (output_dir / "wandb" / "failed-run" / "run.wandb").write_text("operational")
    upload_uri = "r2://eval-artifacts/run-1"

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
            f"paths.output_dir={output_dir}",
            f"hydra.run.dir={output_dir}",
            f"evaluation.upload_output_dir_uri={upload_uri}",
        ],
        cwd=remote_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    publication_root = remote_root / "eval-artifacts" / "run-1"
    manifest = json.loads((publication_root / "latest.json").read_text())
    payload_root = publication_root / "attempts" / manifest["attempt_id"]
    uploaded_metrics = json.loads((payload_root / "metrics" / "metrics.json").read_text())
    assert uploaded_metrics["test/param_mse"] == 0.0
    assert not (payload_root / "predictions" / "failed.json").exists()
    assert not (payload_root / "wandb").exists()
