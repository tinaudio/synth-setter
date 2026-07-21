"""Behavioral tests for the SAME writer path in :mod:`add_embeddings`.

The SAME encoders are exercised through *injected* encode callables so the suite never downloads a
checkpoint; the input preparation (mono duplication, resampling), the latent-frame math, and the
fixed-shape Lance column contract are what these tests pin.
"""

from __future__ import annotations

from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest
import torch
from click.testing import CliRunner

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
    main,
    same_encoder_input,
    same_num_latent_frames,
    same_record_batch,
)
from tests.helpers.finalize_shards import build_lance_smoke_spec, write_minimal_lance_shard
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


def test_add_same_embeddings_rejects_dataset_without_audio_column(tmp_path: Path) -> None:
    """A dataset without the audio source column fails before the UDF runs.

    :param tmp_path: Per-test dataset root.
    """
    uri = tmp_path / "no_audio.lance"
    params = np.zeros((2, 3), dtype=np.float32)
    write_lance_shard(uri, {PARAM_ARRAY_FIELD: params})

    with pytest.raises(ValueError, match="audio"):
        add_same_embeddings(lance.dataset(str(uri)), {SAME_S_FIELD: _fake_same(0.0)}, 44100)


@pytest.mark.slow
def test_main_same_mode_appends_selected_columns_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--same`` runs the SAME-only writer with the metadata sample rate.

    :param tmp_path: Per-test dataset root.
    :param monkeypatch: Fixture injecting a fake SAME encoder loader.
    """
    spec = build_lance_smoke_spec()
    uri = tmp_path / "shard.lance"
    write_minimal_lance_shard(uri, spec)
    seen_checkpoints: list[str] = []

    def fake_loader(checkpoint: str, device: str | None = None) -> SameEncodeFn:
        seen_checkpoints.append(checkpoint)
        return _fake_same(0.5)

    monkeypatch.setattr(
        "synth_setter.pipeline.data.add_embeddings.load_same_audio_encoder", fake_loader
    )

    result = CliRunner().invoke(
        main, [str(uri), "--same", "s", "--same-s-checkpoint", "/models/same-s"]
    )

    assert result.exit_code == 0, result.output
    assert seen_checkpoints == ["/models/same-s"]
    names = set(lance.dataset(str(uri)).schema.names)
    assert SAME_S_FIELD in names
    assert SAME_L_FIELD not in names
    # SAME mode must not require or write the m2l/clap columns.
    assert "m2l" not in names
    assert "clap" not in names


def test_main_documents_same_options() -> None:
    """The CLI documents its SAME mode and per-variant checkpoint flags."""
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0, result.output
    for flag in ("--same", "--same-s-checkpoint", "--same-l-checkpoint"):
        assert flag in result.output


def test_same_profile_shape_feeds_embedpool_encoder() -> None:
    """A (256, 44) SAME latent pools through EmbeddingPool at profile settings."""
    from synth_setter.models.components.embed_pool import EmbeddingPool

    encoder = EmbeddingPool(
        embed_dim=SAME_EMBEDDING_DIM, d_model=32, num_heads=4, max_seq_len=SAME_LATENT_FRAMES
    )

    pooled = encoder(torch.randn(2, SAME_EMBEDDING_DIM, SAME_LATENT_FRAMES))

    assert pooled.shape == (2, 32)
