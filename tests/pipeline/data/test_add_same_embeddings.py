"""Behavioral tests for the SAME writer path in :mod:`add_embeddings`.

The SAME encoders are exercised through *injected* encode callables so the suite never downloads a
checkpoint; the input preparation (mono duplication, resampling), the latent-frame math, and the
fixed-shape Lance column contract are what these tests pin.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import lance
import numpy as np
import pyarrow as pa
import pytest
import torch
from structlog.testing import capture_logs

from synth_setter.data.vst.shapes import (
    AUDIO_FIELD,
    PARAM_ARRAY_FIELD,
    SAME_L_FIELD,
    SAME_S_FIELD,
)
from synth_setter.pipeline.data.add_embeddings import (
    SAME_DOWNSAMPLING_RATIO,
    SAME_EMBEDDING_DIM,
    SAME_LATENT_FRAMES,
    SAME_SAMPLE_RATE,
    SameEncodeFn,
    add_same_embeddings,
    same_encoder_input,
    same_num_latent_frames,
    same_record_batch,
)
from tests.helpers.lance_fixtures import write_lance_shard

# Fixture audio: 16 samples @ 44.1 kHz pad to one two-hop block -> 2 latent frames.
_FIXTURE_SAMPLES = 16
_FIXTURE_FRAMES = 2


def _fake_same(fill: float) -> SameEncodeFn:
    """Build a SAME encoder stub emitting a constant ``(B, 256, T)`` latent.

    :param fill: Constant latent value, distinguishing variants in round-trips.
    :returns: Encoder mapping prepared ``(B, 2, T)`` audio to a constant latent.
    """

    def encode(stereo: np.ndarray) -> np.ndarray:
        frames = same_num_latent_frames(stereo.shape[2], SAME_SAMPLE_RATE)
        return np.full((stereo.shape[0], SAME_EMBEDDING_DIM, frames), fill, dtype=np.float32)

    return encode


def _audio_dataset(uri: Path, rows: int, *, channels: int = 2) -> np.ndarray:
    """Write a Lance dataset of ``rows`` random-audio rows; return the audio array.

    :param uri: Output ``.lance`` directory.
    :param rows: Row count.
    :param channels: Audio channel count.
    :returns: The ``(rows, channels, 16)`` float16 audio written.
    """
    rng = np.random.default_rng(rows)
    audio = rng.random((rows, channels, _FIXTURE_SAMPLES)).astype(np.float16)
    params = rng.random((rows, 3)).astype(np.float32)
    write_lance_shard(uri, {AUDIO_FIELD: audio, PARAM_ARRAY_FIELD: params})
    return audio


def _run_udf_in_process(
    dataset: lance.LanceDataset, udf: Any, *, read_columns: list[str], batch_size: int
) -> None:
    """Run a Lance batch UDF synchronously for deterministic log assertions.

    :param dataset: Local test dataset supplying batches.
    :param udf: Lance batch UDF under test.
    :param read_columns: Columns supplied to the UDF.
    :param batch_size: Maximum rows per UDF invocation.
    """
    for batch in dataset.to_batches(columns=read_columns, batch_size=batch_size):
        udf(batch)


def test_add_same_embeddings_log_every_batch_reports_each_batch_and_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``log_every_batch`` emits one progress entry per batch, bracketed by version logs.

    :param tmp_path: Pytest-provided scratch directory for the dataset.
    :param monkeypatch: Runs Lance's UDF in-process so ``capture_logs`` sees it.
    """
    uri = str(tmp_path / "same-debug.lance")
    _audio_dataset(Path(uri), rows=5)

    monkeypatch.setattr(lance.LanceDataset, "add_columns", _run_udf_in_process)
    with capture_logs() as logs:
        add_same_embeddings(
            lance.dataset(uri),
            {SAME_S_FIELD: _fake_same(0.5)},
            SAME_SAMPLE_RATE,
            batch_size=2,
            log_every_batch=True,
        )

    progress = [entry for entry in logs if entry["event"] == "embedding_progress"]
    assert [entry["rows_processed"] for entry in progress] == [2, 4, 5]
    events = [entry["event"] for entry in logs]
    assert events.index("same_embedding_write_started") < events.index("embedding_progress")
    assert events.index("embedding_progress") < events.index("wrote_same_embeddings")


def test_same_latent_frames_standard_render_is_44() -> None:
    """The pinned profile length matches 4 s @ 44.1 kHz through the 4096 ratio."""
    assert SAME_LATENT_FRAMES == 44
    assert same_num_latent_frames(4 * SAME_SAMPLE_RATE, SAME_SAMPLE_RATE) == SAME_LATENT_FRAMES


def test_same_num_latent_frames_resampled_rate_counts_output_samples() -> None:
    """Frame math follows the resampled 44.1 kHz length, not the source length."""
    # 8192 samples @ 22.05 kHz resample to 16384 @ 44.1 kHz -> exactly 4 frames.
    assert same_num_latent_frames(2 * SAME_DOWNSAMPLING_RATIO, SAME_SAMPLE_RATE // 2) == 4


def test_same_num_latent_frames_pads_to_even_two_hop_blocks() -> None:
    """Frame counts land on even two-hop blocks, matching the real encoder."""
    # 1 s @ 44.1 kHz: ceil(44100 / 8192) = 6 blocks -> 12 frames (not ceil=11).
    assert same_num_latent_frames(SAME_SAMPLE_RATE, SAME_SAMPLE_RATE) == 12
    # One hop of input still pads up to a full block of two frames.
    assert same_num_latent_frames(SAME_DOWNSAMPLING_RATIO, SAME_SAMPLE_RATE) == 2


def test_same_encoder_input_mono_duplicates_to_stereo() -> None:
    """A single-channel batch is duplicated into identical stereo channels."""
    mono = np.random.default_rng(0).random((3, 1, 32)).astype(np.float16)

    prepared = same_encoder_input(mono, SAME_SAMPLE_RATE)

    assert prepared.shape == (3, 2, 32)
    assert prepared.dtype == np.float32
    np.testing.assert_array_equal(prepared[:, 0], prepared[:, 1])
    np.testing.assert_allclose(prepared[:, 0], mono[:, 0].astype(np.float32))


def test_same_encoder_input_stereo_at_44100_passes_through_as_float32() -> None:
    """Already-conformant stereo audio is only upcast, never altered."""
    stereo = np.random.default_rng(1).random((2, 2, 32)).astype(np.float16)

    prepared = same_encoder_input(stereo, SAME_SAMPLE_RATE)

    assert prepared.dtype == np.float32
    np.testing.assert_allclose(prepared, stereo.astype(np.float32))


def test_same_encoder_input_non_44100_rate_resamples_to_44100() -> None:
    """A half-rate batch doubles in length on the way to 44.1 kHz."""
    stereo = np.random.default_rng(2).random((2, 2, 512)).astype(np.float16)

    prepared = same_encoder_input(stereo, SAME_SAMPLE_RATE // 2)

    assert prepared.shape == (2, 2, 1024)
    assert prepared.dtype == np.float32
    assert np.isfinite(prepared).all()


def test_same_encoder_input_rejects_more_than_two_channels() -> None:
    """Surround audio has no defined stereo mapping and is rejected."""
    surround = np.zeros((1, 3, 32), dtype=np.float32)

    with pytest.raises(ValueError, match="channel"):
        same_encoder_input(surround, SAME_SAMPLE_RATE)


def test_same_record_batch_builds_fixed_shape_tensor_columns() -> None:
    """Each encoder lands as a float32 (256, T) fixed-shape tensor column."""
    audio = np.random.default_rng(3).random((4, 2, _FIXTURE_SAMPLES)).astype(np.float16)

    batch = same_record_batch(
        audio,
        {SAME_S_FIELD: _fake_same(0.5), SAME_L_FIELD: _fake_same(-1.5)},
        SAME_SAMPLE_RATE,
        num_frames=_FIXTURE_FRAMES,
    )

    assert batch.schema.names == [SAME_S_FIELD, SAME_L_FIELD]
    for name, fill in ((SAME_S_FIELD, 0.5), (SAME_L_FIELD, -1.5)):
        field_type = batch.schema.field(name).type
        assert tuple(field_type.shape) == (SAME_EMBEDDING_DIM, _FIXTURE_FRAMES)
        assert field_type.value_type == pa.float32()
        values = batch.column(name).to_numpy_ndarray()
        np.testing.assert_array_equal(
            values, np.full((4, SAME_EMBEDDING_DIM, _FIXTURE_FRAMES), fill, dtype=np.float32)
        )


def test_same_record_batch_stub_receives_prepared_stereo_input() -> None:
    """The writer, not the encoder, owns mono duplication and dtype prep."""
    mono = np.random.default_rng(4).random((2, 1, _FIXTURE_SAMPLES)).astype(np.float16)
    seen: list[np.ndarray] = []

    def recording(stereo: np.ndarray) -> np.ndarray:
        seen.append(stereo)
        return _fake_same(1.0)(stereo)

    same_record_batch(mono, {SAME_S_FIELD: recording}, SAME_SAMPLE_RATE, num_frames=_FIXTURE_FRAMES)

    assert seen[0].shape == (2, 2, _FIXTURE_SAMPLES)
    assert seen[0].dtype == np.float32


def test_same_record_batch_rejects_wrong_latent_shape() -> None:
    """A latent that is not (B, 256, num_frames) must not be written."""
    audio = np.zeros((2, 2, _FIXTURE_SAMPLES), dtype=np.float16)

    def wrong_dim(stereo: np.ndarray) -> np.ndarray:
        return np.zeros((stereo.shape[0], SAME_EMBEDDING_DIM // 2, 1), dtype=np.float32)

    with pytest.raises(ValueError, match="shape"):
        same_record_batch(audio, {SAME_S_FIELD: wrong_dim}, SAME_SAMPLE_RATE, num_frames=_FIXTURE_FRAMES)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_same_record_batch_rejects_non_finite_latents(value: float) -> None:
    """A NaN/inf latent cell aborts the batch before it can land.

    :param value: Non-finite value injected into one latent cell.
    """
    audio = np.zeros((2, 2, _FIXTURE_SAMPLES), dtype=np.float16)

    def poisoned(stereo: np.ndarray) -> np.ndarray:
        out = _fake_same(0.0)(stereo)
        out[0, 0, 0] = value
        return out

    with pytest.raises(ValueError, match="non-finite"):
        same_record_batch(audio, {SAME_S_FIELD: poisoned}, SAME_SAMPLE_RATE, num_frames=_FIXTURE_FRAMES)


def test_add_same_embeddings_appends_columns_and_round_trips(tmp_path: Path) -> None:
    """Both SAME columns commit and read back with their written values.

    :param tmp_path: Per-test dataset root.
    """
    uri = tmp_path / "shard.lance"
    _audio_dataset(uri, rows=5)
    dataset = lance.dataset(str(uri))

    add_same_embeddings(
        dataset,
        {SAME_S_FIELD: _fake_same(0.25), SAME_L_FIELD: _fake_same(0.75)},
        SAME_SAMPLE_RATE,
    )

    reopened = lance.dataset(str(uri))
    assert {SAME_S_FIELD, SAME_L_FIELD, PARAM_ARRAY_FIELD} <= set(reopened.schema.names)
    table = reopened.to_table(columns=[SAME_S_FIELD, SAME_L_FIELD]).combine_chunks()
    for name, fill in ((SAME_S_FIELD, 0.25), (SAME_L_FIELD, 0.75)):
        values = table.column(name).chunk(0).to_numpy_ndarray()
        np.testing.assert_array_equal(
            values, np.full((5, SAME_EMBEDDING_DIM, _FIXTURE_FRAMES), fill, dtype=np.float32)
        )


def test_add_same_embeddings_single_variant_writes_only_that_column(tmp_path: Path) -> None:
    """Selecting one variant leaves the other column unwritten.

    :param tmp_path: Per-test dataset root.
    """
    uri = tmp_path / "shard.lance"
    _audio_dataset(uri, rows=3)

    add_same_embeddings(lance.dataset(str(uri)), {SAME_L_FIELD: _fake_same(1.0)}, SAME_SAMPLE_RATE)

    names = set(lance.dataset(str(uri)).schema.names)
    assert SAME_L_FIELD in names
    assert SAME_S_FIELD not in names


def test_add_same_embeddings_mono_dataset_round_trips(tmp_path: Path) -> None:
    """Mono renders are accepted end-to-end via channel duplication.

    :param tmp_path: Per-test dataset root.
    """
    uri = tmp_path / "mono.lance"
    _audio_dataset(uri, rows=2, channels=1)

    add_same_embeddings(lance.dataset(str(uri)), {SAME_S_FIELD: _fake_same(2.0)}, SAME_SAMPLE_RATE)

    values = (
        lance.dataset(str(uri))
        .to_table(columns=[SAME_S_FIELD])
        .combine_chunks()
        .column(SAME_S_FIELD)
        .chunk(0)
        .to_numpy_ndarray()
    )
    assert values.shape == (2, SAME_EMBEDDING_DIM, _FIXTURE_FRAMES)


def test_add_same_embeddings_rejects_existing_column(tmp_path: Path) -> None:
    """Re-running against an already-augmented dataset fails fast.

    :param tmp_path: Per-test dataset root.
    """
    uri = tmp_path / "shard.lance"
    _audio_dataset(uri, rows=2)
    add_same_embeddings(lance.dataset(str(uri)), {SAME_S_FIELD: _fake_same(0.0)}, SAME_SAMPLE_RATE)

    with pytest.raises(ValueError, match="same_s"):
        add_same_embeddings(
            lance.dataset(str(uri)), {SAME_S_FIELD: _fake_same(0.0)}, SAME_SAMPLE_RATE
        )


def test_add_same_embeddings_rejects_empty_encoder_mapping(tmp_path: Path) -> None:
    """An empty encoder mapping is a caller bug, not a silent no-op.

    :param tmp_path: Per-test dataset root.
    """
    uri = tmp_path / "shard.lance"
    _audio_dataset(uri, rows=2)

    with pytest.raises(ValueError, match="encoder"):
        add_same_embeddings(lance.dataset(str(uri)), {}, SAME_SAMPLE_RATE)


def test_add_same_embeddings_rejects_non_positive_batch_size(tmp_path: Path) -> None:
    """A non-positive batch size is a caller bug caught before any encode.

    :param tmp_path: Per-test dataset root.
    """
    uri = tmp_path / "shard.lance"
    _audio_dataset(uri, rows=2)

    with pytest.raises(ValueError, match="batch_size"):
        add_same_embeddings(
            lance.dataset(str(uri)), {SAME_S_FIELD: _fake_same(0.0)}, SAME_SAMPLE_RATE, batch_size=0
        )


def test_add_same_embeddings_rejects_empty_dataset(tmp_path: Path) -> None:
    """A rowless dataset fails fast instead of committing empty columns.

    :param tmp_path: Per-test dataset root.
    """
    uri = tmp_path / "empty.lance"
    _audio_dataset(uri, rows=2)
    dataset = lance.dataset(str(uri))
    dataset.delete("true")

    with pytest.raises(ValueError, match="no rows"):
        add_same_embeddings(
            lance.dataset(str(uri)), {SAME_S_FIELD: _fake_same(0.0)}, SAME_SAMPLE_RATE
        )


def test_add_same_embeddings_rejects_dataset_without_audio_column(tmp_path: Path) -> None:
    """A dataset without the audio source column fails before the UDF runs.

    :param tmp_path: Per-test dataset root.
    """
    uri = tmp_path / "no_audio.lance"
    params = np.zeros((2, 3), dtype=np.float32)
    write_lance_shard(uri, {PARAM_ARRAY_FIELD: params})

    with pytest.raises(ValueError, match="audio"):
        add_same_embeddings(lance.dataset(str(uri)), {SAME_S_FIELD: _fake_same(0.0)}, 44100)


def test_same_profile_shape_feeds_embedpool_encoder() -> None:
    """A (256, 44) SAME latent pools through EmbeddingPool at profile settings."""
    from synth_setter.models.components.embed_pool import EmbeddingPool

    encoder = EmbeddingPool(
        embed_dim=SAME_EMBEDDING_DIM, d_model=32, num_heads=4, max_seq_len=SAME_LATENT_FRAMES
    )

    pooled = encoder(torch.randn(2, SAME_EMBEDDING_DIM, SAME_LATENT_FRAMES))

    assert pooled.shape == (2, 32)


def test_resolve_same_checkpoint_dir_keys_cache_on_full_r2_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distinct R2 URIs sharing a final path component get distinct cache dirs.

    :param monkeypatch: Fixture stubbing the credentialed rclone download.
    """
    from synth_setter.pipeline.data.add_embeddings import _resolve_same_checkpoint_dir

    downloads: list[tuple[str, Path]] = []
    monkeypatch.setattr("synth_setter.pipeline.r2_io.ensure_r2_env_loaded", lambda: None)
    monkeypatch.setattr(
        "synth_setter.pipeline.r2_io.download_dir_no_overwrite",
        lambda uri, dest: downloads.append((uri, dest)),
    )

    dir_a = _resolve_same_checkpoint_dir("r2://bucket/team-a/same-s")
    dir_b = _resolve_same_checkpoint_dir("r2://bucket/team-b/same-s/")

    assert dir_a != dir_b
    assert [uri for uri, _ in downloads] == [
        "r2://bucket/team-a/same-s",
        "r2://bucket/team-b/same-s/",
    ]
    assert downloads[0][1] == dir_a
    assert downloads[1][1] == dir_b


def test_resolve_same_checkpoint_dir_returns_existing_local_directory(tmp_path: Path) -> None:
    """A local checkpoint directory is used as-is, with no download.

    :param tmp_path: Existing local directory standing in for a checkpoint.
    """
    from synth_setter.pipeline.data.add_embeddings import _resolve_same_checkpoint_dir

    assert _resolve_same_checkpoint_dir(str(tmp_path)) == tmp_path


def test_load_same_audio_encoder_without_extra_names_install_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A missing stable_audio_tools import fails with the uv install command.

    :param monkeypatch: Fixture blanking the optional dependency's module entry.
    :param tmp_path: Local checkpoint directory placeholder.
    """
    from synth_setter.pipeline.data.add_embeddings import load_same_audio_encoder

    # A None sys.modules entry makes `from stable_audio_tools...` raise
    # ImportError even when the package is installed.
    monkeypatch.setitem(sys.modules, "stable_audio_tools", None)
    monkeypatch.setitem(sys.modules, "stable_audio_tools.models", None)
    monkeypatch.setitem(sys.modules, "stable_audio_tools.models.factory", None)

    with pytest.raises(ImportError, match="uv sync --extra same"):
        load_same_audio_encoder(str(tmp_path), device="cpu")


def test_same_num_latent_frames_rejects_non_positive_inputs() -> None:
    """Zero/negative lengths or rates are caller bugs, not zero-frame clips."""
    with pytest.raises(ValueError, match="positive"):
        same_num_latent_frames(0, SAME_SAMPLE_RATE)
    with pytest.raises(ValueError, match="positive"):
        same_num_latent_frames(SAME_SAMPLE_RATE, 0)


def test_add_same_embeddings_resume_cache_resumes_interrupted_run_without_reencoding(
    tmp_path: Path,
) -> None:
    """A rerun with the same resume cache skips SAME batches encoded before a crash.

    :param tmp_path: Per-test dataset + resume-cache root.
    """
    uri = tmp_path / "resume.lance"
    _audio_dataset(uri, rows=6)
    resume_cache = tmp_path / "resume.cache"
    first_run_batches: list[int] = []
    second_run_batches: list[int] = []

    def crash_on_third_batch(stereo: np.ndarray) -> np.ndarray:
        first_run_batches.append(len(stereo))
        if len(first_run_batches) == 3:
            raise RuntimeError("simulated crash")
        return _fake_same(0.5)(stereo)

    # Lance surfaces UDF exceptions wrapped as OSError("Invalid user input: ...").
    with pytest.raises(OSError, match="simulated crash"):
        add_same_embeddings(
            lance.dataset(str(uri)),
            {SAME_S_FIELD: crash_on_third_batch},
            SAME_SAMPLE_RATE,
            batch_size=2,
            resume_cache=resume_cache,
        )
    assert resume_cache.exists()

    def count_batches(stereo: np.ndarray) -> np.ndarray:
        second_run_batches.append(len(stereo))
        return _fake_same(0.5)(stereo)

    add_same_embeddings(
        lance.dataset(str(uri)),
        {SAME_S_FIELD: count_batches},
        SAME_SAMPLE_RATE,
        batch_size=2,
        resume_cache=resume_cache,
    )

    assert len(second_run_batches) < len(first_run_batches) + 3
    reopened = lance.dataset(str(uri))
    assert SAME_S_FIELD in reopened.schema.names
    assert not resume_cache.exists()
