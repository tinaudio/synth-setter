"""Tests for ``synth_setter.cli.finalize_dataset`` — finalize entrypoint.

End-to-end test invokes ``main()`` against the real ``smoke-shard-wds``
experiment so the Hydra compose + ``DatasetSpec`` construction stay
exercised; rclone (download / upload / auth ping) is stubbed.
"""

from __future__ import annotations

import io
import sys
import tarfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pytest

from synth_setter.cli import finalize_dataset
from synth_setter.pipeline.schemas.spec import DatasetSpec


def _write_minimal_wds_shard(dest: Path) -> None:
    """Write a tar at ``dest`` with one ``00000000.mel_spec.npy`` member.

    ``get_stats_wds`` requires every shard to contain at least one matched
    ``*.mel_spec.npy`` payload or it raises; the payload's exact shape is
    irrelevant for the upload-order assertion this test guards.

    :param dest: Filesystem path where the tar is written.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Welford finalize needs >=2 rows total across shards (zero variance otherwise),
    # so each shard ships a 4-row batch.
    payload = np.arange(8, dtype=np.float32).reshape(4, 2)
    buf = io.BytesIO()
    np.save(buf, payload)
    member_bytes = buf.getvalue()
    with tarfile.open(dest, mode="w") as tar:
        info = tarfile.TarInfo(name="00000000.mel_spec.npy")
        info.size = len(member_bytes)
        tar.addfile(info, io.BytesIO(member_bytes))


def test_wds_finalize_uploads_stats_then_marker_through_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``main()`` against the smoke-shard-wds experiment uploads stats then marker.

    :param monkeypatch: Pytest fixture used to stub ``r2_io`` calls and ``sys.argv``.
    """
    uploaded: list[str] = []

    def record_upload(source: str | Path, destination_uri: str) -> None:
        del source
        uploaded.append(destination_uri)

    def fake_download(r2_uri: str, dest_path: Path) -> None:
        del r2_uri
        _write_minimal_wds_shard(dest_path)

    monkeypatch.setattr("synth_setter.pipeline.r2_io.upload", record_upload)
    monkeypatch.setattr("synth_setter.pipeline.r2_io.download_to_path", fake_download)
    monkeypatch.setattr(
        "synth_setter.pipeline.r2_io.ensure_r2_env_loaded",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(sys, "argv", ["finalize", "experiment=generate_dataset/smoke-shard-wds"])

    finalize_dataset.main()

    uploaded_basenames = [Path(urlparse(uri).path).name for uri in uploaded]
    assert uploaded_basenames == ["stats.npz", "dataset.complete"]


def test_hdf5_branch_raises_not_implemented_in_phase_1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``main()`` raises ``NotImplementedError`` for an hdf5 spec until Phase 2 lands.

    :param monkeypatch: Pytest fixture used to stub ``ensure_r2_env_loaded`` and ``sys.argv``.
    """
    monkeypatch.setattr(
        "synth_setter.pipeline.r2_io.ensure_r2_env_loaded",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(sys, "argv", ["finalize", "experiment=generate_dataset/smoke-shard"])

    with pytest.raises(NotImplementedError, match="hdf5 finalize lands in Phase 2"):
        finalize_dataset.main()


def test_finalize_wds_uploads_stats_npz_to_spec_stats_uri(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``finalize_wds`` posts ``stats.npz`` at exactly ``spec.r2.stats_uri()``.

    :param monkeypatch: Pytest fixture used to stub ``r2_io`` download / upload.
    :param tmp_path: Pytest tmp dir used as the in-process scratch work_dir.
    """
    monkeypatch.setattr(
        "synth_setter.pipeline.r2_io.download_to_path",
        lambda r2_uri, dest_path: _write_minimal_wds_shard(dest_path),
    )
    uploads: list[tuple[Any, str]] = []
    monkeypatch.setattr(
        "synth_setter.pipeline.r2_io.upload",
        lambda source, destination_uri: uploads.append((source, destination_uri)),
    )

    spec = _build_wds_smoke_spec()
    finalize_dataset.finalize_wds(spec, tmp_path)

    assert len(uploads) == 1
    source, destination = uploads[0]
    assert Path(source).name == "stats.npz"
    assert destination == spec.r2.stats_uri()


def _build_wds_smoke_spec() -> DatasetSpec:
    """Construct a single-shard wds ``DatasetSpec`` without invoking Hydra.

    :returns: A frozen ``DatasetSpec`` whose train split is one 4-sample shard.
    """
    kwargs: dict[str, Any] = {
        "task_name": "finalize-wds-unit",
        "output_format": "wds",
        "train_val_test_sizes": [4, 0, 0],
        "base_seed": 42,
        "r2": {"bucket": "intermediate-data"},
        "render": {
            "plugin_path": "/fake/Plugin.vst3",
            "preset_path": "presets/surge-base.vstpreset",
            "param_spec_name": "surge_simple",
            "renderer_version": "1.0.0-test",
            "sample_rate": 16000,
            "channels": 2,
            "velocity": 100,
            "signal_duration_seconds": 4.0,
            "min_loudness": -55.0,
            "samples_per_render_batch": 4,
            "samples_per_shard": 4,
            "gui_toggle_cadence": "never",
        },
    }
    return DatasetSpec(**kwargs)  # type: ignore[arg-type]
