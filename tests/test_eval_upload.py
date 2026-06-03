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

from synth_setter.cli.eval import _maybe_upload_output_dir
from synth_setter.data.vst import param_specs


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
def r2_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the three secret keys :func:`r2_io.ensure_r2_env_loaded` requires present.

    Paired with ``fake_r2_remote`` (which sets ``RCLONE_CONFIG_R2_TYPE=local``),
    these dummy values satisfy the presence check and let the real ``rclone lsd
    r2:`` auth ping resolve the local backend instead of dialing Cloudflare.

    :param monkeypatch: Sets the secret env vars for the test's duration.
    """
    monkeypatch.setenv("RCLONE_CONFIG_R2_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("RCLONE_CONFIG_R2_SECRET_ACCESS_KEY", "test-secret-key")
    monkeypatch.setenv("RCLONE_CONFIG_R2_ENDPOINT", "http://localhost:0")


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

    _maybe_upload_output_dir(
        _upload_cfg(output_dir, upload_output_dir_uri=None), is_global_zero=True
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

    _maybe_upload_output_dir(
        _upload_cfg(output_dir, "r2://bucket/evals/run-1"), is_global_zero=False
    )

    assert list(fake_r2_remote.glob("bucket/**/*")) == []


def test_maybe_upload_output_dir_mirrors_tree_when_uri_set(
    fake_r2_remote: Path, r2_credentials: None, tmp_path: Path
) -> None:
    """A set URI mirrors the whole output dir beneath the destination prefix.

    :param fake_r2_remote: Local-backed ``r2:`` remote where the mirror lands.
    :param r2_credentials: Dummy secrets so the real credential check passes.
    :param tmp_path: Holds the output dir copied to R2.
    """
    output_dir = tmp_path / "run"
    _write_output_tree(output_dir)

    _maybe_upload_output_dir(
        _upload_cfg(output_dir, "r2://bucket/evals/run-1"), is_global_zero=True
    )

    dest = fake_r2_remote / "bucket" / "evals" / "run-1"
    assert (dest / "predictions" / "pred.json").read_text() == '{"ok": true}'
    assert (dest / "metrics.json").read_text() == '{"param_mse": 0.0}'


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
    ``rclone`` (local-backed remote). A dataset staged under an ``r2://`` prefix is
    downloaded into an initially-absent ``data.dataset_root``, and the fake oracle's
    exact-zero ``test/param_mse`` reaches ``metrics.json``. Proves the new
    ``data.download_dataset_root_uri`` gate composes with eval through ``main``.

    :param tmp_path: Root for the fake R2 remote, the download target, and the output dir.
    :param surge_xt_smoke_datasets: Source ``{train,val,test}.h5`` + ``stats.npz``.
    """
    if shutil.which("rclone") is None:
        pytest.skip("rclone binary not available on PATH")

    remote_root = tmp_path / "r2"
    staged = remote_root / "intermediate-data" / "dataset"
    staged.mkdir(parents=True)
    splits_and_stats = ("train.h5", "val.h5", "test.h5", "stats.npz")
    for name in splits_and_stats:
        shutil.copy(surge_xt_smoke_datasets / name, staged / name)

    dataset_root = tmp_path / "downloaded"
    output_dir = tmp_path / "out"

    env = {
        **os.environ,
        "RCLONE_CONFIG_R2_TYPE": "local",
        "RCLONE_CONFIG_R2_ACCESS_KEY_ID": "stub",
        "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY": "stub",
        "RCLONE_CONFIG_R2_ENDPOINT": "stub",
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
            f"model.net.d_out={len(param_specs['surge_4'])}",
            "callbacks.log_per_param_mse.param_spec=surge_4",
            "datamodule.download_dataset_root_uri=r2://intermediate-data/dataset",
            f"datamodule.dataset_root={dataset_root}",
            f"datamodule.predict_file={dataset_root}/test.h5",
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

    for name in splits_and_stats:
        assert (dataset_root / name).is_file(), f"{name} was not downloaded from R2"

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
    :param surge_xt_smoke_datasets: Source ``{train,val,test}.h5`` + ``stats.npz``.
    """
    if shutil.which("rclone") is None:
        pytest.skip("rclone binary not available on PATH")

    remote_root = tmp_path / "r2"
    remote_root.mkdir()
    output_dir = tmp_path / "out"
    upload_uri = "r2://eval-artifacts/run-1"

    env = {
        **os.environ,
        "RCLONE_CONFIG_R2_TYPE": "local",
        "RCLONE_CONFIG_R2_ACCESS_KEY_ID": "stub",
        "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY": "stub",
        "RCLONE_CONFIG_R2_ENDPOINT": "stub",
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
            f"model.net.d_out={len(param_specs['surge_4'])}",
            "callbacks.log_per_param_mse.param_spec=surge_4",
            f"datamodule.dataset_root={surge_xt_smoke_datasets}",
            f"datamodule.predict_file={surge_xt_smoke_datasets}/test.h5",
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

    local_files = {p.relative_to(output_dir) for p in output_dir.rglob("*") if p.is_file()}
    assert local_files, "eval produced no output files to upload"

    uploaded_root = remote_root / "eval-artifacts" / "run-1"
    uploaded_files = {
        p.relative_to(uploaded_root) for p in uploaded_root.rglob("*") if p.is_file()
    }
    missing = local_files - uploaded_files
    assert not missing, f"output dir not fully uploaded; missing {sorted(map(str, missing))}"

    uploaded_metrics = json.loads((uploaded_root / "metrics" / "metrics.json").read_text())
    assert uploaded_metrics["test/param_mse"] == 0.0
