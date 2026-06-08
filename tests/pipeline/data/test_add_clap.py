"""Behavioral tests for ``add_clap`` — the CLAP-embedding shard augmenter.

The CLAP model is the one mocked boundary (it needs a GPU and a checkpoint download); every other
path — HDF5/tar I/O, downmix/resample, the Click entrypoint — runs for real against tiny fixtures.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import pytest
from click.testing import CliRunner

from synth_setter.pipeline.data import add_clap
from synth_setter.pipeline.data.add_clap import (
    CLAP_EMBED_DIM,
    CLAP_FIELD,
    CLAP_SAMPLE_RATE,
)


class FirstSampleEmbedder:
    """Fake CLAP encoder whose embedding for each row is its first sample tiled to 512.

    Deterministic and free of production logic, so a test can pin
    ``clap[i] == <row i's first mono sample>`` without mirroring the real model.
    """

    def embed(self, audio_48k_mono: np.ndarray) -> np.ndarray:
        """Return each row's first sample broadcast across the embedding dimension.

        :param audio_48k_mono: ``(B, T)`` mono audio.
        :returns: ``(B, CLAP_EMBED_DIM)`` embeddings.
        """
        first = audio_48k_mono[:, :1]
        return np.repeat(first, CLAP_EMBED_DIM, axis=1).astype(np.float32)


def _write_h5_shard(path: Path, row_values: list[float], *, sample_rate: int) -> None:
    """Write an HDF5 shard whose row ``i`` is a stereo constant-valued signal.

    :param path: Destination HDF5 path.
    :param row_values: Constant value for each row's audio; length sets the row count.
    :param sample_rate: Stored in ``audio.attrs`` for the augmenter to read.
    """
    audio = np.zeros((len(row_values), 2, 8), dtype=np.float16)
    for i, value in enumerate(row_values):
        audio[i] = value
    with h5py.File(path, "w") as f:
        ds = f.create_dataset("audio", data=audio)
        ds.attrs["sample_rate"] = sample_rate


def _write_wds_shard(
    path: Path, key_to_values: dict[str, list[float]], *, sample_rate: int
) -> None:
    """Write a webdataset tar mirroring the production layout (audio/mel/param + metadata.json).

    :param path: Destination tar path.
    :param key_to_values: Per-key row values; each row is a stereo constant-valued signal.
    :param sample_rate: Written into the ``metadata.json`` sidecar.
    """
    metadata = {
        "velocity": 100,
        "signal_duration_seconds": 1.0,
        "sample_rate": sample_rate,
        "channels": 2,
        "min_loudness": -40.0,
    }
    with tarfile.open(path, "w") as tar:
        for key, values in key_to_values.items():
            audio = np.zeros((len(values), 2, 8), dtype=np.float16)
            for i, value in enumerate(values):
                audio[i] = value
            _add_npy(tar, f"{key}.audio.npy", audio)
            _add_npy(
                tar, f"{key}.mel_spec.npy", np.zeros((len(values), 2, 4, 4), dtype=np.float32)
            )
            _add_npy(tar, f"{key}.param_array.npy", np.zeros((len(values), 3), dtype=np.float32))
        _add_bytes(tar, "metadata.json", json.dumps(metadata).encode("utf-8"))


def _add_npy(tar: tarfile.TarFile, name: str, array: np.ndarray) -> None:
    """Append ``array`` to ``tar`` as a ``.npy`` member.

    :param tar: Open tar in write mode.
    :param name: Member name.
    :param array: Array to serialize.
    """
    buf = io.BytesIO()
    np.save(buf, array)
    _add_bytes(tar, name, buf.getvalue())


def _add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    """Append raw ``data`` to ``tar`` as a member.

    :param tar: Open tar in write mode.
    :param name: Member name.
    :param data: Raw member bytes.
    """
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _read_tar_npy(path: Path, name: str) -> np.ndarray:
    """Load a ``.npy`` member from a tar.

    :param path: Tar path.
    :param name: Member name.
    :returns: The decoded array.
    """
    with tarfile.open(path, "r") as tar:
        extracted = tar.extractfile(name)
        assert extracted is not None
        return np.load(io.BytesIO(extracted.read()))


def _tar_member_names(path: Path) -> list[str]:
    """Return the member names in a tar.

    :param path: Tar path.
    :returns: Member names in archive order.
    """
    with tarfile.open(path, "r") as tar:
        return tar.getnames()


def _read_h5(path: Path, field: str) -> np.ndarray:
    """Read a full HDF5 dataset into memory.

    :param path: HDF5 shard path.
    :param field: Dataset name.
    :returns: The dataset contents.
    """
    with h5py.File(path, "r") as f:
        return cast(h5py.Dataset, f[field])[:]


def test_to_clap_input_stereo_downmixed_to_mono_mean() -> None:
    """Stereo audio collapses to the channel-wise mean."""
    audio = np.array([[[2.0, 4.0], [6.0, 8.0]]], dtype=np.float16)  # (1, 2, 2)

    out = add_clap.to_clap_input(audio, source_sr=CLAP_SAMPLE_RATE)

    assert out.shape == (1, 2)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[0], [4.0, 6.0])


def test_to_clap_input_resamples_to_clap_rate_changing_length() -> None:
    """Audio below 48 kHz is resampled, lengthening the waveform proportionally."""
    audio = np.ones((1, 1, 100), dtype=np.float16)

    out = add_clap.to_clap_input(audio, source_sr=CLAP_SAMPLE_RATE // 2)

    assert out.shape == (1, 200)


def test_embed_audio_batch_returns_embedder_output_as_float32() -> None:
    """A batch is preprocessed and encoded, returning the embedder output as float32."""
    audio = np.full((3, 1, 8), 5.0, dtype=np.float16)

    out = add_clap.embed_audio_batch(audio, CLAP_SAMPLE_RATE, FirstSampleEmbedder())

    assert out.shape == (3, CLAP_EMBED_DIM)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, np.full((3, CLAP_EMBED_DIM), 5.0))


def test_embed_audio_batch_rejects_wrong_embedding_dim() -> None:
    """An embedder returning the wrong dimension raises a ValueError naming the expected dim."""

    class WrongDimEmbedder:
        def embed(self, audio_48k_mono: np.ndarray) -> np.ndarray:
            return np.zeros((audio_48k_mono.shape[0], CLAP_EMBED_DIM + 1), dtype=np.float32)

    with pytest.raises(ValueError, match=str(CLAP_EMBED_DIM)):
        add_clap.embed_audio_batch(
            np.ones((2, 1, 8), dtype=np.float16), CLAP_SAMPLE_RATE, WrongDimEmbedder()
        )


def test_add_clap_to_h5_creates_embedding_dataset_per_row(tmp_path: Path) -> None:
    """Each HDF5 row gets a 512-d embedding derived from its audio.

    :param tmp_path: Pytest temp directory.
    """
    shard = tmp_path / "shard-000000.h5"
    _write_h5_shard(shard, [0.0, 1.0, 2.0, 3.0], sample_rate=CLAP_SAMPLE_RATE)

    rows = add_clap.add_clap_to_h5(shard, FirstSampleEmbedder(), batch_size=2)

    clap = _read_h5(shard, CLAP_FIELD)
    assert rows == 4
    assert clap.shape == (4, CLAP_EMBED_DIM)
    assert clap.dtype == np.float32
    np.testing.assert_array_equal(clap[:, 0], [0.0, 1.0, 2.0, 3.0])


def test_add_clap_to_h5_is_idempotent_when_field_present(tmp_path: Path) -> None:
    """Re-running on a shard that already has the field is a no-op.

    :param tmp_path: Pytest temp directory.
    """
    shard = tmp_path / "shard-000000.h5"
    _write_h5_shard(shard, [0.0, 1.0], sample_rate=CLAP_SAMPLE_RATE)
    add_clap.add_clap_to_h5(shard, FirstSampleEmbedder(), batch_size=2)
    before = _read_h5(shard, CLAP_FIELD)

    rows = add_clap.add_clap_to_h5(shard, FirstSampleEmbedder(), batch_size=2)

    assert rows == 0
    np.testing.assert_array_equal(before, _read_h5(shard, CLAP_FIELD))


def test_add_clap_to_h5_threads_stored_sample_rate_through_resample(tmp_path: Path) -> None:
    """A sub-48 kHz shard is resampled via its stored rate, still yielding one row per sample.

    :param tmp_path: Pytest temp directory.
    """
    shard = tmp_path / "shard-000000.h5"
    _write_h5_shard(shard, [1.0, 2.0, 3.0], sample_rate=CLAP_SAMPLE_RATE // 2)

    rows = add_clap.add_clap_to_h5(shard, FirstSampleEmbedder(), batch_size=2)

    clap = _read_h5(shard, CLAP_FIELD)
    assert rows == 3
    assert clap.shape == (3, CLAP_EMBED_DIM)
    assert np.isfinite(clap).all()


def test_add_clap_to_h5_missing_sample_rate_attr_raises(tmp_path: Path) -> None:
    """A shard whose audio lacks the sample_rate attr fails loudly, naming the shard.

    :param tmp_path: Pytest temp directory.
    """
    shard = tmp_path / "shard-000000.h5"
    with h5py.File(shard, "w") as f:
        f.create_dataset("audio", data=np.zeros((1, 2, 8), dtype=np.float16))

    with pytest.raises(ValueError, match="sample_rate"):
        add_clap.add_clap_to_h5(shard, FirstSampleEmbedder(), batch_size=2)


def test_add_clap_to_wds_adds_clap_member_per_key(tmp_path: Path) -> None:
    """Every tar key gains a ``<key>.clap.npy`` member matching its row count.

    :param tmp_path: Pytest temp directory.
    """
    shard = tmp_path / "shard-000000.tar"
    _write_wds_shard(
        shard, {"00000000": [1.0, 2.0], "00000002": [3.0]}, sample_rate=CLAP_SAMPLE_RATE
    )

    rows = add_clap.add_clap_to_wds(shard, FirstSampleEmbedder(), batch_size=8)

    first = _read_tar_npy(shard, "00000000.clap.npy")
    second = _read_tar_npy(shard, "00000002.clap.npy")
    assert rows == 3
    assert first.shape == (2, CLAP_EMBED_DIM)
    assert second.shape == (1, CLAP_EMBED_DIM)
    np.testing.assert_array_equal(first[:, 0], [1.0, 2.0])
    np.testing.assert_array_equal(second[:, 0], [3.0])


def test_add_clap_to_wds_preserves_existing_members(tmp_path: Path) -> None:
    """The rewrite keeps the original audio/mel/param/metadata members intact.

    :param tmp_path: Pytest temp directory.
    """
    shard = tmp_path / "shard-000000.tar"
    _write_wds_shard(shard, {"00000000": [1.0, 2.0]}, sample_rate=CLAP_SAMPLE_RATE)

    add_clap.add_clap_to_wds(shard, FirstSampleEmbedder(), batch_size=8)

    names = _tar_member_names(shard)
    assert "00000000.audio.npy" in names
    assert "00000000.mel_spec.npy" in names
    assert "00000000.param_array.npy" in names
    assert "metadata.json" in names
    assert _read_tar_npy(shard, "00000000.audio.npy").shape == (2, 2, 8)


def test_add_clap_to_wds_is_idempotent_when_member_present(tmp_path: Path) -> None:
    """Re-running on a shard that already has CLAP members adds no duplicates.

    :param tmp_path: Pytest temp directory.
    """
    shard = tmp_path / "shard-000000.tar"
    _write_wds_shard(shard, {"00000000": [1.0, 2.0]}, sample_rate=CLAP_SAMPLE_RATE)
    add_clap.add_clap_to_wds(shard, FirstSampleEmbedder(), batch_size=8)

    rows = add_clap.add_clap_to_wds(shard, FirstSampleEmbedder(), batch_size=8)

    assert rows == 0
    assert _tar_member_names(shard).count("00000000.clap.npy") == 1


def test_add_clap_to_wds_concatenates_across_batches(tmp_path: Path) -> None:
    """A key with more rows than the batch size is embedded across batches and concatenated.

    :param tmp_path: Pytest temp directory.
    """
    shard = tmp_path / "shard-000000.tar"
    _write_wds_shard(shard, {"00000000": [1.0, 2.0, 3.0]}, sample_rate=CLAP_SAMPLE_RATE)

    add_clap.add_clap_to_wds(shard, FirstSampleEmbedder(), batch_size=1)

    clap = _read_tar_npy(shard, "00000000.clap.npy")
    assert clap.shape == (3, CLAP_EMBED_DIM)
    np.testing.assert_array_equal(clap[:, 0], [1.0, 2.0, 3.0])


def test_add_clap_to_wds_missing_metadata_raises(tmp_path: Path) -> None:
    """A tar without the metadata.json sidecar fails loudly, naming the shard.

    :param tmp_path: Pytest temp directory.
    """
    shard = tmp_path / "shard-000000.tar"
    with tarfile.open(shard, "w") as tar:
        _add_npy(tar, "00000000.audio.npy", np.zeros((1, 2, 8), dtype=np.float16))

    with pytest.raises(ValueError, match="metadata.json"):
        add_clap.add_clap_to_wds(shard, FirstSampleEmbedder(), batch_size=8)


def test_add_clap_to_shard_unsupported_suffix_raises(tmp_path: Path) -> None:
    """A shard path that is neither .h5 nor .tar is rejected.

    :param tmp_path: Pytest temp directory.
    """
    with pytest.raises(ValueError, match="unsupported shard type"):
        add_clap.add_clap_to_shard(
            tmp_path / "shard-000000.wav", FirstSampleEmbedder(), batch_size=8
        )


def test_cli_errors_when_no_shards_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI exits non-zero when the directory has no matching shards.

    :param tmp_path: Pytest temp directory.
    :param monkeypatch: Injects the fake embedder so no real model is built.
    """
    monkeypatch.setattr(add_clap, "_build_embedder", lambda ckpt: FirstSampleEmbedder())

    result = CliRunner().invoke(add_clap.main, [str(tmp_path)])

    assert result.exit_code != 0
    assert "no shard" in result.output


def test_cli_adds_clap_to_every_shard_in_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI augments both an HDF5 and a webdataset shard under the given directory.

    :param tmp_path: Pytest temp directory.
    :param monkeypatch: Used to inject the fake embedder for the heavy CLAP model.
    """
    _write_h5_shard(tmp_path / "shard-000000.h5", [1.0, 2.0], sample_rate=CLAP_SAMPLE_RATE)
    _write_wds_shard(
        tmp_path / "shard-000001.tar", {"00000000": [3.0]}, sample_rate=CLAP_SAMPLE_RATE
    )
    monkeypatch.setattr(add_clap, "_build_embedder", lambda ckpt: FirstSampleEmbedder())

    result = CliRunner().invoke(add_clap.main, [str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert _read_h5(tmp_path / "shard-000000.h5", CLAP_FIELD).shape == (2, CLAP_EMBED_DIM)
    assert _read_tar_npy(tmp_path / "shard-000001.tar", "00000000.clap.npy").shape == (
        1,
        CLAP_EMBED_DIM,
    )
