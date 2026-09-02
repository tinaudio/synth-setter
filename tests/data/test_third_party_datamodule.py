"""Tests for serving third-party Lance audio corpora with mel conditioning."""

from __future__ import annotations

import io
import pickle
import re
from collections.abc import Sequence
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest
import soundfile as sf
import torch
from lance.blob import blob_array, blob_field
from pedalboard.io import AudioFile

from synth_setter.conditioning import (
    NUM_SKETCH_CONTROLS,
    SKETCH_CENTROID_ROW,
    SKETCH_CTRL_FIELD,
    SKETCH_LOUDNESS_ROW,
    SKETCH_PITCH_SLICE,
    SketchControls,
)
from synth_setter.data.third_party_datamodule import ThirdPartyAudioDataModule, decode_clip
from synth_setter.data.vst.shapes import AUDIO_FIELD, MEL_N_MELS, make_spectrogram
from tests.helpers.lance_fixtures import wav_bytes, write_blob_audio_corpus

# Keep the test corpus compact while retaining a legal mel front-end.
_TARGET_SAMPLE_RATE = 4000
_TARGET_CHANNELS = 2
_DURATION_SECONDS = 0.5
_TARGET_SAMPLES = 2000
_MEL_FRAMES = 51
_SOURCE_SAMPLE_RATE = 8000


def _write_corpus(
    path: Path,
    clips: Sequence[np.ndarray],
    *,
    sample_rate: int = _SOURCE_SAMPLE_RATE,
    audio_column: str = AUDIO_FIELD,
    with_sample_rate_column: bool = True,
    mode: str = "create",
) -> None:
    """Write a blob-encoded audio corpus in the published third-party layout.

    :param path: Destination Lance dataset.
    :param clips: One mono float32 clip per row.
    :param sample_rate: Encoded sample rate for every clip.
    :param audio_column: Blob column name, mirroring per-corpus schema drift.
    :param with_sample_rate_column: Whether to store the per-row sample rate.
    :param mode: Lance write mode; ``append`` commits a further version.
    """
    write_blob_audio_corpus(
        path,
        clips,
        sample_rate=sample_rate,
        audio_column=audio_column,
        with_sample_rate_column=with_sample_rate_column,
        mode=mode,
    )


def _tone(seconds: float, sample_rate: int = _SOURCE_SAMPLE_RATE, seed: int = 0) -> np.ndarray:
    """Draw one reproducible noise clip.

    :param seconds: Clip duration.
    :param sample_rate: Clip sample rate in Hz.
    :param seed: RNG seed.
    :returns: ``(frames,)`` float32 samples in ``[-0.5, 0.5]``.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.5, 0.5, size=int(seconds * sample_rate)).astype(np.float32)


def _datamodule(
    uri: str | Path,
    *,
    sample_rate: int = _TARGET_SAMPLE_RATE,
    audio_column: str = AUDIO_FIELD,
    amplitude_scale: float = 1.0,
    dataset_version: int = 1,
    row_limit: int | None = None,
    num_workers: int = 0,
    use_saved_mean_and_variance: bool = False,
    mel_stats_uri: str | None = None,
    mel_stats_sha256: str | None = None,
    sketch: SketchControls = None,
) -> ThirdPartyAudioDataModule:
    """Build a datamodule over a corpus with the tiny render contract.

    :param uri: Corpus Lance path, or an ``r2://`` URI, read in place.
    :param sample_rate: Target render rate the corpus is mapped onto.
    :param audio_column: Blob column the corpus stores its audio in.
    :param amplitude_scale: Gain applied to decoded audio.
    :param dataset_version: Immutable Lance snapshot to serve.
    :param row_limit: Cap on served rows, or ``None`` for the whole corpus.
    :param num_workers: Dataloader worker processes.
    :param use_saved_mean_and_variance: Whether mel is standardized.
    :param mel_stats_uri: Statistics source when standardization is on.
    :param mel_stats_sha256: Optional digest pin for the statistics bytes.
    :param sketch: Optional live sketch-control specification.
    :returns: Configured, un-setup datamodule.
    """
    return ThirdPartyAudioDataModule(
        dataset_uri=str(uri),
        sample_rate=sample_rate,
        channels=_TARGET_CHANNELS,
        signal_duration_seconds=_DURATION_SECONDS,
        dataset_version=dataset_version,
        batch_size=2,
        audio_column=audio_column,
        amplitude_scale=amplitude_scale,
        row_limit=row_limit,
        num_workers=num_workers,
        use_saved_mean_and_variance=use_saved_mean_and_variance,
        mel_stats_uri=mel_stats_uri,
        mel_stats_sha256=mel_stats_sha256,
        sketch=sketch,
    )


def _first_batch(datamodule: ThirdPartyAudioDataModule) -> dict[str, torch.Tensor]:
    """Run the predict dataloader through the datamodule's batch transform.

    :param datamodule: Datamodule under test.
    :returns: The first model batch a predict loop would see.
    """
    datamodule.setup("predict")
    batch = next(iter(datamodule.predict_dataloader()))
    return datamodule.on_before_batch_transfer(batch, 0)


def test_predict_batch_audio_matches_render_contract(tmp_path: Path) -> None:
    """Decoded audio is resampled, up-mixed, and length-pinned to the render contract.

    :param tmp_path: Isolated corpus fixture directory.
    """
    _write_corpus(tmp_path / "corpus.lance", [_tone(_DURATION_SECONDS, seed=1)])

    batch = _first_batch(_datamodule(tmp_path / "corpus.lance"))

    assert batch["audio"].shape == (1, _TARGET_CHANNELS, _TARGET_SAMPLES)
    assert batch["audio"].dtype == torch.float32


def test_predict_batch_upmixed_channels_carry_identical_samples(tmp_path: Path) -> None:
    """A mono source is duplicated across channels rather than silenced.

    :param tmp_path: Isolated corpus fixture directory.
    """
    _write_corpus(tmp_path / "corpus.lance", [_tone(_DURATION_SECONDS, seed=2)])

    audio = _first_batch(_datamodule(tmp_path / "corpus.lance"))["audio"]

    assert torch.equal(audio[0, 0], audio[0, 1])
    assert torch.any(audio[0, 0] != 0)


def test_predict_batch_short_clip_pads_tail_with_silence(tmp_path: Path) -> None:
    """A clip shorter than the render duration is zero-padded at the tail.

    :param tmp_path: Isolated corpus fixture directory.
    """
    _write_corpus(tmp_path / "corpus.lance", [_tone(_DURATION_SECONDS / 2, seed=3)])

    audio = _first_batch(_datamodule(tmp_path / "corpus.lance"))["audio"]

    assert audio.shape[-1] == _TARGET_SAMPLES
    assert torch.all(audio[0, :, _TARGET_SAMPLES // 2 :] == 0)
    assert torch.any(audio[0, :, : _TARGET_SAMPLES // 2] != 0)


def test_predict_batch_long_clip_trims_to_render_duration(tmp_path: Path) -> None:
    """A clip longer than the render duration is truncated, not squeezed to fit.

    The source holds one level for its first third and another after, so resampling the whole clip
    into the target length would leak the later level into the served audio; truncation keeps only
    the first.

    :param tmp_path: Isolated corpus fixture directory.
    """
    frames = int(_DURATION_SECONDS * 3 * _SOURCE_SAMPLE_RATE)
    stepped = np.full(frames, 0.2, dtype=np.float32)
    stepped[frames // 3 :] = 0.9
    _write_corpus(tmp_path / "corpus.lance", [stepped])

    audio = _first_batch(_datamodule(tmp_path / "corpus.lance"))["audio"]

    assert audio.shape[-1] == _TARGET_SAMPLES
    assert audio.abs().max() < 0.5


def test_predict_batch_amplitude_scale_attenuates_samples(tmp_path: Path) -> None:
    """The configured amplitude scale is applied to the decoded waveform.

    :param tmp_path: Isolated corpus fixture directory.
    """
    _write_corpus(tmp_path / "corpus.lance", [_tone(_DURATION_SECONDS, seed=5)])

    unscaled = _first_batch(_datamodule(tmp_path / "corpus.lance"))["audio"]
    scaled = _first_batch(_datamodule(tmp_path / "corpus.lance", amplitude_scale=0.5))["audio"]

    assert torch.allclose(scaled, unscaled * 0.5)
    assert scaled.abs().max() <= 1.0


def test_predict_batch_mel_matches_training_front_end_shape(tmp_path: Path) -> None:
    """Mel conditioning is emitted on the stored dataset's channel-leading grid.

    :param tmp_path: Isolated corpus fixture directory.
    """
    _write_corpus(tmp_path / "corpus.lance", [_tone(_DURATION_SECONDS, seed=6)])

    batch = _first_batch(_datamodule(tmp_path / "corpus.lance"))

    expected = make_spectrogram(batch["audio"][0].numpy(), _TARGET_SAMPLE_RATE)

    assert batch["mel"].shape == (1, _TARGET_CHANNELS, MEL_N_MELS, _MEL_FRAMES)
    assert batch["mel"].dtype == torch.float32
    assert torch.isfinite(batch["mel"]).all()
    # Identity with the canonical front-end proves the datamodule routes through it;
    # the front-end's own constants are pinned in tests/data/vst/test_shape_helpers.py.
    assert torch.allclose(batch["mel"][0], torch.from_numpy(expected), atol=1e-5)


def test_predict_batch_saved_statistics_normalize_mel(tmp_path: Path) -> None:
    """Configured statistics standardize the mel the corpus would otherwise emit.

    :param tmp_path: Isolated corpus fixture directory.
    """
    _write_corpus(tmp_path / "corpus.lance", [_tone(_DURATION_SECONDS, seed=7)])
    stats_file = tmp_path / "stats.npz"
    # Per-mel-bin arrays, as a training split stores them: a scalar would hide a
    # broadcasting mistake across the mel axis.
    mean = np.full((MEL_N_MELS, 1), -40.0, dtype=np.float32)
    std = np.full((MEL_N_MELS, 1), 4.0, dtype=np.float32)
    np.savez(stats_file, mean=mean, std=std)

    plain = _first_batch(_datamodule(tmp_path / "corpus.lance"))["mel"]
    normalized = _first_batch(
        _datamodule(
            tmp_path / "corpus.lance",
            use_saved_mean_and_variance=True,
            mel_stats_uri=str(stats_file),
        )
    )["mel"]

    assert torch.allclose(normalized, (plain + 40.0) / 4.0, atol=1e-5)


def test_mel_statistics_digest_mismatch_raises(tmp_path: Path) -> None:
    """Normalization rejects statistics that differ from their provenance pin.

    :param tmp_path: Isolated corpus and statistics directory.
    """
    corpus = tmp_path / "corpus.lance"
    _write_corpus(corpus, [_tone(_DURATION_SECONDS, seed=38)])
    stats_file = tmp_path / "stats.npz"
    np.savez(stats_file, mean=np.float32(0.0), std=np.float32(1.0))

    datamodule = _datamodule(
        corpus,
        use_saved_mean_and_variance=True,
        mel_stats_uri=str(stats_file),
        mel_stats_sha256="0" * 64,
    )

    with pytest.raises(ValueError, match="mel statistics SHA-256 mismatch"):
        datamodule.setup("predict")


def test_saved_statistics_without_uri_raises(tmp_path: Path) -> None:
    """Normalization without a statistics source fails loudly instead of passing raw mel.

    :param tmp_path: Isolated corpus fixture directory.
    """
    _write_corpus(tmp_path / "corpus.lance", [_tone(_DURATION_SECONDS, seed=8)])

    with pytest.raises(ValueError, match="mel_stats_uri"):
        _datamodule(tmp_path / "corpus.lance", use_saved_mean_and_variance=True)


def test_non_mel_conditioning_raises(tmp_path: Path) -> None:
    """Reject a conditioning mode this datamodule cannot compute.

    :param tmp_path: Isolated corpus fixture directory.
    """
    with pytest.raises(ValueError, match="mel conditioning only"):
        ThirdPartyAudioDataModule(
            dataset_uri=str(tmp_path / "corpus.lance"),
            sample_rate=_TARGET_SAMPLE_RATE,
            channels=_TARGET_CHANNELS,
            signal_duration_seconds=_DURATION_SECONDS,
            dataset_version=1,
            conditioning="clap",
        )


def test_corpus_with_its_own_blob_column_name_is_servable(tmp_path: Path) -> None:
    """Per-corpus schema drift is absorbed by config, not by a converted copy.

    :param tmp_path: Isolated corpus fixture directory.
    """
    _write_corpus(
        tmp_path / "corpus.lance",
        [_tone(_DURATION_SECONDS, seed=9)],
        audio_column="audio_wav",
        with_sample_rate_column=False,
    )

    batch = _first_batch(_datamodule(tmp_path / "corpus.lance", audio_column="audio_wav"))

    assert batch["audio"].shape == (1, _TARGET_CHANNELS, _TARGET_SAMPLES)


def test_source_rate_above_target_is_resampled_not_truncated(tmp_path: Path) -> None:
    """A corpus recorded above the render rate keeps its events at their original times.

    The source is silent except for a burst in its final quarter: truncating the
    leading ``num_samples`` would drop the burst entirely, so only a real
    resample places it in the final quarter of the served clip.

    :param tmp_path: Isolated corpus fixture directory.
    """
    source_rate = _TARGET_SAMPLE_RATE * 4
    frames = int(_DURATION_SECONDS * source_rate)
    late_burst = np.zeros(frames, dtype=np.float32)
    late_burst[3 * frames // 4 :] = 0.5
    _write_corpus(tmp_path / "corpus.lance", [late_burst], sample_rate=source_rate)

    audio = _first_batch(_datamodule(tmp_path / "corpus.lance"))["audio"]

    assert audio.shape[-1] == _TARGET_SAMPLES
    assert torch.all(audio[0, :, : _TARGET_SAMPLES // 2].abs() < 0.1)
    assert audio[0, :, 3 * _TARGET_SAMPLES // 4 :].abs().max() > 0.4


def test_row_limit_caps_served_rows(tmp_path: Path) -> None:
    """A row limit bounds an eval sweep without touching the published corpus.

    :param tmp_path: Isolated corpus fixture directory.
    """
    _write_corpus(tmp_path / "corpus.lance", [_tone(_DURATION_SECONDS, seed=i) for i in range(5)])

    datamodule = _datamodule(tmp_path / "corpus.lance", row_limit=3)
    datamodule.setup("predict")

    assert sum(len(batch["audio"]) for batch in datamodule.predict_dataloader()) == 3


def test_multichannel_source_disagreeing_with_contract_raises() -> None:
    """A source that is neither mono nor contract-width is rejected, not silently sliced."""
    stereo = np.stack([_tone(_DURATION_SECONDS), _tone(_DURATION_SECONDS, seed=1)])
    buffer = io.BytesIO()
    with AudioFile(
        buffer, "w", format="wav", samplerate=_SOURCE_SAMPLE_RATE, num_channels=2
    ) as handle:
        handle.write(stereo)

    with pytest.raises(ValueError, match="render contract wants 1"):
        decode_clip(
            buffer.getvalue(),
            sample_rate=_TARGET_SAMPLE_RATE,
            channels=1,
            num_samples=_TARGET_SAMPLES,
            amplitude_scale=1.0,
        )


def test_non_predict_stage_raises(tmp_path: Path) -> None:
    """Only prediction is served, so a fit or test stage fails instead of half-building.

    :param tmp_path: Isolated corpus fixture directory.
    """
    _write_corpus(tmp_path / "corpus.lance", [_tone(_DURATION_SECONDS, seed=16)])

    with pytest.raises(ValueError, match="prediction only"):
        _datamodule(tmp_path / "corpus.lance").setup("fit")


def test_predict_dataloader_before_setup_raises(tmp_path: Path) -> None:
    """Asking for the loader before setup fails loudly rather than serving zero rows.

    :param tmp_path: Isolated corpus fixture directory.
    """
    _write_corpus(tmp_path / "corpus.lance", [_tone(_DURATION_SECONDS, seed=17)])

    with pytest.raises(RuntimeError, match="setup"):
        _datamodule(tmp_path / "corpus.lance").predict_dataloader()


def test_worker_pickling_drops_the_open_dataset_handle(tmp_path: Path) -> None:
    """Row decode survives dataloader workers, which cannot inherit a Lance handle.

    :param tmp_path: Isolated corpus fixture directory.
    """
    _write_corpus(tmp_path / "corpus.lance", [_tone(_DURATION_SECONDS, seed=18)] * 2)
    datamodule = _datamodule(tmp_path / "corpus.lance")
    datamodule.setup("predict")

    revived = pickle.loads(pickle.dumps(datamodule.predict_dataloader().dataset))
    with_workers = _datamodule(tmp_path / "corpus.lance", num_workers=2)
    with_workers.setup("predict")
    served = [batch["audio"] for batch in with_workers.predict_dataloader()]

    assert revived[0]["audio"].shape == (_TARGET_CHANNELS, _TARGET_SAMPLES)
    assert sum(len(batch) for batch in served) == 2


def test_distinct_stats_uris_sharing_a_basename_cache_separately(tmp_path: Path) -> None:
    """Every training split names its statistics ``stats.npz``, so the cache keys on the URI.

    Two checkpoints trained on different corpora would otherwise collide in one cache slot, and the
    second eval would silently normalize with the first run's statistics.

    :param tmp_path: Isolated corpus fixture directory.
    """
    _write_corpus(tmp_path / "corpus.lance", [_tone(_DURATION_SECONDS, seed=19)])
    first = _datamodule(
        tmp_path / "corpus.lance",
        use_saved_mean_and_variance=True,
        mel_stats_uri="r2://experiments/data/corpus-a/train.lance/stats.npz",
    )
    second = _datamodule(
        tmp_path / "corpus.lance",
        use_saved_mean_and_variance=True,
        mel_stats_uri="r2://experiments/data/corpus-b/train.lance/stats.npz",
    )

    assert first.cached_stats_path() != second.cached_stats_path()


def test_r2_statistics_are_downloaded_and_normalize_the_batch(fake_r2_remote: Path) -> None:
    """An ``r2://`` statistics object is fetched once and applied to the served mel.

    Drives the real rclone download into the digest-keyed cache slot and feeds
    the result through its real consumer, so a broken fetch cannot pass as an
    un-normalized batch.

    :param fake_r2_remote: Root backing the ``r2:`` remote as a local filesystem.
    """
    corpus = fake_r2_remote / "corpus.lance"
    _write_corpus(corpus, [_tone(_DURATION_SECONDS, seed=20)])
    stats_key = fake_r2_remote / "experiments" / "corpus-a"
    stats_key.mkdir(parents=True)
    np.savez(stats_key / "stats.npz", mean=np.float32(-30.0), std=np.float32(2.0))

    plain = _first_batch(_datamodule(corpus))["mel"]
    normalized = _first_batch(
        _datamodule(
            corpus,
            use_saved_mean_and_variance=True,
            mel_stats_uri="r2://experiments/corpus-a/stats.npz",
        )
    )["mel"]

    assert torch.allclose(normalized, (plain + 30.0) / 2.0, atol=1e-5)


def test_decode_clip_rejects_out_of_range_source_before_resampling() -> None:
    """Substantive source overflow cannot be hidden by output clipping."""
    source_rate = 16_000
    buffer = io.BytesIO()
    sf.write(
        buffer,
        np.full(source_rate, 1.2, dtype=np.float32),
        source_rate,
        format="WAV",
        subtype="FLOAT",
    )

    with pytest.raises(ValueError, match="source audio leaves"):
        decode_clip(
            buffer.getvalue(),
            sample_rate=44_100,
            channels=2,
            num_samples=44_100,
            amplitude_scale=1.0,
        )


def test_decode_clip_rejects_positive_pcm16_tolerance_as_source_overflow() -> None:
    """PCM16's negative endpoint tolerance does not admit positive overflow."""
    source_rate = 16_000
    buffer = io.BytesIO()
    sf.write(
        buffer,
        np.full(source_rate, 1.00002, dtype=np.float32),
        source_rate,
        format="WAV",
        subtype="FLOAT",
    )

    with pytest.raises(ValueError, match="source audio leaves"):
        decode_clip(
            buffer.getvalue(),
            sample_rate=44_100,
            channels=2,
            num_samples=44_100,
            amplitude_scale=1.0,
        )


def test_decode_clip_clamps_resampling_ringing_to_storage_range() -> None:
    """Band-limited resampling cannot leave otherwise normalized audio above full scale."""
    source_rate = 16_000
    samples = np.arange(source_rate, dtype=np.float32)
    square = np.sign(np.sin(2 * np.pi * 3000.0 * samples / source_rate)).astype(np.float32)

    clip = decode_clip(
        wav_bytes(square, source_rate),
        sample_rate=44_100,
        channels=2,
        num_samples=44_100,
        amplitude_scale=1.0,
    )

    assert np.isfinite(clip).all()
    assert np.abs(clip).max() == 1.0


def test_decode_rejects_audio_outside_the_storage_range(tmp_path: Path) -> None:
    """Gain that drives a clip past full scale is rejected, not fed to the mel front-end.

    :param tmp_path: Isolated corpus fixture directory.
    """
    _write_corpus(tmp_path / "corpus.lance", [np.full(1000, 0.8, dtype=np.float32)])

    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        _first_batch(_datamodule(tmp_path / "corpus.lance", amplitude_scale=2.0))


def test_corpus_missing_the_configured_audio_column_raises(tmp_path: Path) -> None:
    """A corpus whose schema lacks the configured blob column fails at setup.

    :param tmp_path: Isolated corpus fixture directory.
    """
    _write_corpus(
        tmp_path / "corpus.lance", [_tone(_DURATION_SECONDS, seed=22)], audio_column="audio_wav"
    )

    with pytest.raises(KeyError, match="audio"):
        _datamodule(tmp_path / "corpus.lance").setup("predict")


@pytest.mark.parametrize("limit", [-1, 0])
def test_non_positive_row_limit_raises(tmp_path: Path, limit: int) -> None:
    """A negative cap cannot become a negative length, and zero serves nothing.

    :param tmp_path: Isolated corpus fixture directory.
    :param limit: Reachable cap that would produce no usable sweep.
    """
    _write_corpus(tmp_path / "corpus.lance", [_tone(_DURATION_SECONDS, seed=23)])

    with pytest.raises(ValueError, match="row_limit"):
        _datamodule(tmp_path / "corpus.lance", row_limit=limit)


def test_boolean_row_limit_raises(tmp_path: Path) -> None:
    """A YAML ``true`` is an int in Python; serving one row silently would be worse.

    :param tmp_path: Isolated corpus fixture directory.
    """
    _write_corpus(tmp_path / "corpus.lance", [_tone(_DURATION_SECONDS, seed=25)])

    with pytest.raises(ValueError, match="row_limit"):
        _datamodule(tmp_path / "corpus.lance", row_limit=True)  # type: ignore[arg-type]


def test_corpus_audio_column_without_blob_encoding_raises(tmp_path: Path) -> None:
    """An ordinary binary column named ``audio`` cannot be served through the blob API.

    :param tmp_path: Isolated corpus fixture directory.
    """
    table = pa.table({AUDIO_FIELD: pa.array([b"not-a-blob"], pa.large_binary())})
    lance.write_dataset(table, tmp_path / "corpus.lance", mode="create")

    with pytest.raises(ValueError, match="blob"):
        _datamodule(tmp_path / "corpus.lance").setup("predict")


@pytest.mark.parametrize("scheme", ["r2", "s3"])
def test_corpus_served_from_an_r2_backed_uri(fake_r2_remote: Path, scheme: str) -> None:
    """Both accepted URI spellings resolve through the configured R2 remote.

    :param fake_r2_remote: Root backing the ``r2:`` remote as a local filesystem.
    :param scheme: Public URI scheme used by the caller.
    """
    corpus = fake_r2_remote / "experiments" / "third_party" / "Tiny" / "test.lance"
    corpus.parent.mkdir(parents=True)
    _write_corpus(corpus, [_tone(_DURATION_SECONDS, seed=26)])

    batch = _first_batch(
        _datamodule(f"{scheme}://experiments/third_party/Tiny/test.lance")
    )

    assert batch["audio"].shape == (1, _TARGET_CHANNELS, _TARGET_SAMPLES)


@pytest.mark.parametrize("limit", [1.5, "4"])
def test_non_integer_row_limit_raises(tmp_path: Path, limit: object) -> None:
    """A non-integer cap cannot reach ``__len__``, which must return an int.

    :param tmp_path: Isolated corpus fixture directory.
    :param limit: Reachable Hydra override that is not an integer.
    """
    _write_corpus(tmp_path / "corpus.lance", [_tone(_DURATION_SECONDS, seed=27)])

    with pytest.raises(ValueError, match="row_limit"):
        _datamodule(tmp_path / "corpus.lance", row_limit=limit)  # type: ignore[arg-type]


def test_statistics_underflowing_float32_are_rejected(tmp_path: Path) -> None:
    """Statistics that survive float64 validation but vanish in float32 are rejected.

    A positive ``std`` such as ``1e-50`` casts to float32 zero, and dividing by
    it would hand the checkpoint infinities rather than features.

    :param tmp_path: Isolated corpus fixture directory.
    """
    _write_corpus(tmp_path / "corpus.lance", [_tone(_DURATION_SECONDS, seed=30)])
    stats_file = tmp_path / "stats.npz"
    np.savez(stats_file, mean=np.float64(-30.0), std=np.float64(1e-50))

    with pytest.raises(ValueError, match="float32"):
        _first_batch(
            _datamodule(
                tmp_path / "corpus.lance",
                use_saved_mean_and_variance=True,
                mel_stats_uri=str(stats_file),
            )
        )


def test_statistics_producing_non_finite_mel_are_rejected(tmp_path: Path) -> None:
    """Normalization that overflows to infinity is caught on the batch, not the statistics.

    A subnormal ``std`` such as ``1e-45`` survives every positivity check yet
    sends ordinary mel values to infinity when divided by it.

    :param tmp_path: Isolated corpus fixture directory.
    """
    _write_corpus(tmp_path / "corpus.lance", [_tone(_DURATION_SECONDS, seed=31)])
    stats_file = tmp_path / "stats.npz"
    np.savez(stats_file, mean=np.float64(-30.0), std=np.float64(1e-45))

    with pytest.raises(ValueError, match="non-finite"):
        _first_batch(
            _datamodule(
                tmp_path / "corpus.lance",
                use_saved_mean_and_variance=True,
                mel_stats_uri=str(stats_file),
            )
        )


def test_native_blob_v2_column_is_servable(tmp_path: Path) -> None:
    """A corpus written with Lance's native blob field carries no legacy marker.

    The published corpora use the ``lance.blob.v2`` extension type rather than
    ``lance-encoding:blob`` metadata, so accepting only the marker would reject
    every real dataset.

    :param tmp_path: Isolated corpus fixture directory.
    """
    clip = _tone(_DURATION_SECONDS, seed=35)
    table = pa.table(
        {AUDIO_FIELD: blob_array([wav_bytes(clip, _SOURCE_SAMPLE_RATE)])},
        schema=pa.schema([blob_field(AUDIO_FIELD)]),
    )
    lance.write_dataset(
        table, tmp_path / "corpus.lance", mode="create", data_storage_version="2.2"
    )

    batch = _first_batch(_datamodule(tmp_path / "corpus.lance"))

    assert batch["audio"].shape == (1, _TARGET_CHANNELS, _TARGET_SAMPLES)


def test_nsynth_sketch_batch_native_blob_v2_preserves_order_and_model_shapes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real Blob-v2 batch retains ordered audio and emits normalized model inputs.

    :param tmp_path: Isolated corpus and statistics directory.
    :param monkeypatch: Replaces only resource-heavy PESTO extraction for this fast contract test.
    """
    clips = [
        np.full(_SOURCE_SAMPLE_RATE // 2, 0.25, dtype=np.float32),
        np.full(_SOURCE_SAMPLE_RATE // 2, -0.5, dtype=np.float32),
    ]
    table = pa.table(
        {AUDIO_FIELD: blob_array([wav_bytes(clip, _SOURCE_SAMPLE_RATE) for clip in clips])},
        schema=pa.schema([blob_field(AUDIO_FIELD)]),
    )
    corpus = tmp_path / "nsynth-test.lance"
    lance.write_dataset(table, corpus, mode="create", data_storage_version="2.2")
    stats_file = tmp_path / "training-stats.npz"
    np.savez(
        stats_file,
        mean=np.full((MEL_N_MELS, 1), -10.0, dtype=np.float32),
        std=np.full((MEL_N_MELS, 1), 2.0, dtype=np.float32),
    )

    def fixed_controls(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        del sample_rate
        return torch.zeros(audio.shape[0], NUM_SKETCH_CONTROLS, 64, device=audio.device)

    monkeypatch.setattr(
        "synth_setter.data.third_party_datamodule.extract_sketch_controls_batch",
        fixed_controls,
    )
    datamodule = _datamodule(
        corpus,
        row_limit=None,
        use_saved_mean_and_variance=True,
        mel_stats_uri=str(stats_file),
        sketch={"column": "sketch", "num_frames": 32, "num_control_tokens": 32},
    )
    datamodule.setup("predict")
    raw_batch = next(iter(datamodule.predict_dataloader()))
    normalized = datamodule.on_before_batch_transfer(raw_batch, 0)
    batch = datamodule.on_after_batch_transfer(normalized, 0)

    expected_mel = (make_spectrogram(batch[AUDIO_FIELD][0].numpy(), _TARGET_SAMPLE_RATE) + 10.0) / 2.0
    assert set(batch) == {AUDIO_FIELD, "mel", SKETCH_CTRL_FIELD}
    assert sum(len(served[AUDIO_FIELD]) for served in datamodule.predict_dataloader()) == 2
    assert batch[AUDIO_FIELD].shape == (2, _TARGET_CHANNELS, _TARGET_SAMPLES)
    assert batch[AUDIO_FIELD][0].mean() > 0
    assert batch[AUDIO_FIELD][1].mean() < 0
    assert batch["mel"].shape == (2, _TARGET_CHANNELS, MEL_N_MELS, _MEL_FRAMES)
    assert torch.allclose(batch["mel"][0], torch.from_numpy(expected_mel), atol=1e-5)
    assert batch[SKETCH_CTRL_FIELD].shape == (2, NUM_SKETCH_CONTROLS, 32)


def test_nsynth_sketch_batch_pools_controls_and_zeroes_weak_pitch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Live controls use canonical scalar/pitch pooling and the checkpoint threshold.

    :param tmp_path: Isolated datamodule path; no corpus access occurs.
    :param monkeypatch: Supplies deterministic full-frame controls before real pooling.
    """
    controls = torch.zeros(1, NUM_SKETCH_CONTROLS, 64)
    controls[:, SKETCH_LOUDNESS_ROW, 1::2] = 1.0
    controls[:, SKETCH_CENTROID_ROW, ::2] = 1.0
    controls[:, SKETCH_PITCH_SLICE.start, :] = 0.05
    controls[:, SKETCH_PITCH_SLICE.start + 1, 1::2] = 0.2
    monkeypatch.setattr(
        "synth_setter.data.third_party_datamodule.extract_sketch_controls_batch",
        lambda audio, sample_rate: controls,
    )
    datamodule = _datamodule(
        tmp_path / "unused.lance",
        sketch={
            "column": "sketch",
            "num_frames": 32,
            "num_control_tokens": 32,
            "pitch_zero_threshold": 0.1,
        },
    )

    batch = datamodule.on_after_batch_transfer(
        {AUDIO_FIELD: torch.zeros(1, _TARGET_CHANNELS, _TARGET_SAMPLES)}, 0
    )

    pooled = batch[SKETCH_CTRL_FIELD]
    assert torch.equal(pooled[0, SKETCH_LOUDNESS_ROW], torch.full((32,), 0.5))
    assert torch.equal(pooled[0, SKETCH_CENTROID_ROW], torch.full((32,), 0.5))
    assert torch.count_nonzero(pooled[0, SKETCH_PITCH_SLICE.start]) == 0
    assert torch.equal(pooled[0, SKETCH_PITCH_SLICE.start + 1], torch.full((32,), 0.2))


def test_nsynth_sketch_noncanonical_frame_count_raises(tmp_path: Path) -> None:
    """A live sketch config cannot drift from the checkpoint's 32-frame contract.

    :param tmp_path: Isolated placeholder corpus path.
    """
    with pytest.raises(ValueError, match="32"):
        _datamodule(
            tmp_path / "unused.lance",
            sketch={"column": "sketch", "num_frames": 64, "num_control_tokens": 64},
        )


@pytest.mark.slow
def test_nsynth_sketch_batch_real_pesto_emits_finite_canonical_controls(tmp_path: Path) -> None:
    """A real PESTO extraction consumes Blob-v2 audio through the datamodule hooks.

    :param tmp_path: Isolated corpus and statistics directory.
    """
    sample_rate = 16_000
    samples = np.arange(sample_rate // 2, dtype=np.float32)
    clip = (0.5 * np.sin(2 * np.pi * 440.0 * samples / sample_rate)).astype(np.float32)
    corpus = tmp_path / "nsynth-test.lance"
    table = pa.table(
        {AUDIO_FIELD: blob_array([wav_bytes(clip, sample_rate)])},
        schema=pa.schema([blob_field(AUDIO_FIELD)]),
    )
    lance.write_dataset(table, corpus, mode="create", data_storage_version="2.2")
    stats_file = tmp_path / "training-stats.npz"
    np.savez(
        stats_file,
        mean=np.zeros((_TARGET_CHANNELS, MEL_N_MELS, _MEL_FRAMES), dtype=np.float32),
        std=np.ones((_TARGET_CHANNELS, MEL_N_MELS, _MEL_FRAMES), dtype=np.float32),
    )
    datamodule = _datamodule(
        corpus,
        sample_rate=sample_rate,
        use_saved_mean_and_variance=True,
        mel_stats_uri=str(stats_file),
        sketch={"column": "sketch", "num_frames": 32, "num_control_tokens": 32},
    )
    datamodule.setup("predict")
    raw_batch = next(iter(datamodule.predict_dataloader()))
    normalized = datamodule.on_before_batch_transfer(raw_batch, 0)

    batch = datamodule.on_after_batch_transfer(normalized, 0)

    controls = batch[SKETCH_CTRL_FIELD]
    assert controls.shape == (1, NUM_SKETCH_CONTROLS, 32)
    assert controls.dtype == torch.float32
    assert torch.isfinite(controls).all()
    assert torch.all((-1.0 <= controls[:, : SKETCH_PITCH_SLICE.start]))
    assert torch.all((controls[:, : SKETCH_PITCH_SLICE.start] <= 1.0))
    pitch = controls[0, SKETCH_PITCH_SLICE]
    assert torch.all((0.0 <= pitch) & (pitch <= 1.0))
    expected_a4_bin = 69 * 3
    assert torch.all((pitch.argmax(dim=0) - expected_a4_bin).abs() <= 1)
    assert torch.all(pitch.amax(dim=0) >= 0.1)
    assert batch[AUDIO_FIELD].shape == (1, _TARGET_CHANNELS, sample_rate // 2)
    assert "params" not in batch


def test_statistics_overflowing_float32_are_rejected(tmp_path: Path) -> None:
    """Statistics that become infinite in float32 would silently zero the features.

    Dividing by an infinite ``std`` yields 0.0, which is finite — so the batch
    check cannot catch it and the model would receive a blank mel.

    :param tmp_path: Isolated corpus fixture directory.
    """
    _write_corpus(tmp_path / "corpus.lance", [_tone(_DURATION_SECONDS, seed=36)])
    stats_file = tmp_path / "stats.npz"
    np.savez(stats_file, mean=np.float64(0.0), std=np.float64(1e39))

    with pytest.raises(ValueError, match="float32"):
        _first_batch(
            _datamodule(
                tmp_path / "corpus.lance",
                use_saved_mean_and_variance=True,
                mel_stats_uri=str(stats_file),
            )
        )


def test_non_boolean_use_saved_mean_and_variance_raises(tmp_path: Path) -> None:
    """A quoted ``"false"`` must not silently enable normalization.

    :param tmp_path: Isolated corpus fixture directory.
    """
    _write_corpus(tmp_path / "corpus.lance", [_tone(_DURATION_SECONDS, seed=37)])

    with pytest.raises(ValueError, match="must be a boolean"):
        _datamodule(
            tmp_path / "corpus.lance",
            use_saved_mean_and_variance="false",  # type: ignore[arg-type]
            mel_stats_uri="stats.npz",
        )


def _wav(clip: np.ndarray, sample_rate: int = _SOURCE_SAMPLE_RATE) -> bytes:
    """Encode one clip for the direct decode tests.

    :param clip: ``(frames,)`` float32 samples.
    :param sample_rate: Encoded sample rate in Hz.
    :returns: WAV container bytes.
    """
    return wav_bytes(clip, sample_rate)


def test_decode_clip_pads_short_audio_and_upmixes() -> None:
    """A short mono clip is duplicated across channels and zero-padded to length."""
    decoded = decode_clip(
        _wav(np.full(_SOURCE_SAMPLE_RATE // 4, 0.5, dtype=np.float32)),
        sample_rate=_TARGET_SAMPLE_RATE,
        channels=_TARGET_CHANNELS,
        num_samples=_TARGET_SAMPLES,
        amplitude_scale=1.0,
    )

    assert decoded.shape == (_TARGET_CHANNELS, _TARGET_SAMPLES)
    assert decoded.dtype == np.float32
    assert np.array_equal(decoded[0], decoded[1])
    assert np.all(decoded[:, -100:] == 0.0)


def test_decode_clip_truncates_long_audio_and_applies_gain() -> None:
    """A long clip is cut to length and the configured gain is applied to it."""
    decoded = decode_clip(
        _wav(np.full(_SOURCE_SAMPLE_RATE * 2, 0.4, dtype=np.float32)),
        sample_rate=_TARGET_SAMPLE_RATE,
        channels=_TARGET_CHANNELS,
        num_samples=_TARGET_SAMPLES,
        amplitude_scale=0.5,
    )

    assert decoded.shape == (_TARGET_CHANNELS, _TARGET_SAMPLES)
    assert decoded.mean() == pytest.approx(0.2, abs=1e-3)


def test_empty_corpus_raises(tmp_path: Path) -> None:
    """A corpus with no rows fails at setup rather than producing an empty sweep.

    :param tmp_path: Isolated corpus fixture directory.
    """
    _write_corpus(tmp_path / "corpus.lance", [])

    with pytest.raises(ValueError, match="no rows"):
        _datamodule(tmp_path / "corpus.lance").setup("predict")


@pytest.mark.parametrize(
    ("sample_rate", "duration", "channels", "message"),
    [
        (_TARGET_SAMPLE_RATE, -1.0, _TARGET_CHANNELS, "signal_duration_seconds=-1.0"),
        (_TARGET_SAMPLE_RATE, 0.0, 1, "signal_duration_seconds=0.0"),
        (100, 0.001, 1, "positive sample count"),
    ],
)
def test_render_contract_without_a_positive_sample_grid_raises(
    tmp_path: Path, sample_rate: int, duration: float, channels: int, message: str
) -> None:
    """A non-positive grid would slice clips to empty rather than fail the contract.

    :param tmp_path: Isolated corpus fixture directory.
    :param sample_rate: Render-contract sample rate in Hz.
    :param duration: Render-contract clip duration in seconds.
    :param channels: Render-contract channel count.
    :param message: Substring the raised error must name.
    """
    _write_corpus(tmp_path / "corpus.lance", [_tone(_DURATION_SECONDS, seed=39)])

    with pytest.raises(ValueError, match=re.escape(message)):
        ThirdPartyAudioDataModule(
            dataset_uri=str(tmp_path / "corpus.lance"),
            sample_rate=sample_rate,
            channels=channels,
            signal_duration_seconds=duration,
            dataset_version=1,
        )


def test_non_positive_channel_count_raises(tmp_path: Path) -> None:
    """A zero-channel contract would build a degenerate audio tensor.

    :param tmp_path: Isolated corpus fixture directory.
    """
    _write_corpus(tmp_path / "corpus.lance", [_tone(_DURATION_SECONDS, seed=40)])

    with pytest.raises(ValueError, match="channels=0"):
        ThirdPartyAudioDataModule(
            dataset_uri=str(tmp_path / "corpus.lance"),
            sample_rate=_TARGET_SAMPLE_RATE,
            channels=0,
            signal_duration_seconds=_DURATION_SECONDS,
            dataset_version=1,
        )


@pytest.mark.parametrize("field", ["sample_rate", "channels", "batch_size", "num_workers"])
def test_boolean_count_argument_raises(tmp_path: Path, field: str) -> None:
    """``bool`` subclasses ``int``, so a YAML ``true`` would silently mean one.

    :param tmp_path: Isolated corpus fixture directory.
    :param field: Count argument supplied as a boolean.
    """
    _write_corpus(tmp_path / "corpus.lance", [_tone(_DURATION_SECONDS, seed=41)])
    kwargs: dict[str, object] = {
        "dataset_uri": str(tmp_path / "corpus.lance"),
        "sample_rate": _TARGET_SAMPLE_RATE,
        "channels": _TARGET_CHANNELS,
        "signal_duration_seconds": _DURATION_SECONDS,
        "dataset_version": 1,
        field: True,
    }

    with pytest.raises(ValueError, match=f"{field}=True"):
        ThirdPartyAudioDataModule(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("amplitude_scale", [False, True])
def test_boolean_amplitude_scale_raises(tmp_path: Path, amplitude_scale: bool) -> None:
    """A YAML boolean must not silently select a numeric corpus gain.

    :param tmp_path: Isolated corpus fixture directory.
    :param amplitude_scale: Boolean value supplied where a numeric gain is required.
    """
    with pytest.raises(ValueError, match=f"amplitude_scale={amplitude_scale}"):
        ThirdPartyAudioDataModule(
            dataset_uri=str(tmp_path / "corpus.lance"),
            sample_rate=_TARGET_SAMPLE_RATE,
            channels=_TARGET_CHANNELS,
            signal_duration_seconds=_DURATION_SECONDS,
            dataset_version=1,
            amplitude_scale=amplitude_scale,
        )


@pytest.mark.parametrize(
    "amplitude_scale", [0.0, -0.5, float("inf"), float("-inf"), float("nan")]
)
def test_amplitude_scale_outside_positive_finite_domain_raises(
    tmp_path: Path, amplitude_scale: float
) -> None:
    """A gain that erases or invalidates corpus audio fails at configuration time.

    :param tmp_path: Isolated corpus fixture directory.
    :param amplitude_scale: Invalid numeric gain.
    """
    with pytest.raises(ValueError, match="must be positive and finite"):
        ThirdPartyAudioDataModule(
            dataset_uri=str(tmp_path / "corpus.lance"),
            sample_rate=_TARGET_SAMPLE_RATE,
            channels=_TARGET_CHANNELS,
            signal_duration_seconds=_DURATION_SECONDS,
            dataset_version=1,
            amplitude_scale=amplitude_scale,
        )


def test_non_positive_batch_size_raises(tmp_path: Path) -> None:
    """A zero batch size would build a loader that yields nothing.

    :param tmp_path: Isolated corpus fixture directory.
    """
    _write_corpus(tmp_path / "corpus.lance", [_tone(_DURATION_SECONDS, seed=42)])

    with pytest.raises(ValueError, match="batch_size=0"):
        ThirdPartyAudioDataModule(
            dataset_uri=str(tmp_path / "corpus.lance"),
            sample_rate=_TARGET_SAMPLE_RATE,
            channels=_TARGET_CHANNELS,
            signal_duration_seconds=_DURATION_SECONDS,
            dataset_version=1,
            batch_size=0,
        )


def test_negative_rate_and_duration_cancelling_into_a_positive_grid_raises(
    tmp_path: Path,
) -> None:
    """Two negatives multiply to a plausible sample count but no valid contract.

    :param tmp_path: Isolated corpus fixture directory.
    """
    _write_corpus(tmp_path / "corpus.lance", [_tone(_DURATION_SECONDS, seed=43)])

    with pytest.raises(ValueError, match="sample_rate=-4000"):
        ThirdPartyAudioDataModule(
            dataset_uri=str(tmp_path / "corpus.lance"),
            sample_rate=-4000,
            channels=_TARGET_CHANNELS,
            signal_duration_seconds=-0.5,
            dataset_version=1,
        )


def test_setup_transient_lance_open_failure_retries_and_serves_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transient object-store failure while opening the corpus does not abort evaluation.

    :param tmp_path: Isolated corpus fixture directory.
    :param monkeypatch: Fixture injecting one transient Lance failure.
    """
    corpus = tmp_path / "corpus.lance"
    _write_corpus(corpus, [_tone(_DURATION_SECONDS, seed=44)])
    real_open = lance.dataset
    attempts = 0

    def open_after_timeout(
        uri: str,
        *,
        version: int | None = None,
        storage_options: dict[str, str] | None = None,
    ) -> lance.LanceDataset:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("transient object-store timeout")
        return real_open(uri, version=version, storage_options=storage_options)

    monkeypatch.setattr(
        "synth_setter.data.third_party_datamodule.lance.dataset", open_after_timeout
    )

    batch = _first_batch(_datamodule(corpus))

    assert batch["audio"].shape == (1, _TARGET_CHANNELS, _TARGET_SAMPLES)


def test_dataloader_batch_reads_blobs_together_and_preserves_requested_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One native blob read serves an ordered batch with duplicate indices.

    :param tmp_path: Isolated corpus fixture directory.
    :param monkeypatch: Fixture requiring one complete Lance selection.
    """
    corpus = tmp_path / "corpus.lance"
    _write_corpus(corpus, [_tone(_DURATION_SECONDS, seed=index) for index in range(4)])
    datamodule = _datamodule(corpus)
    datamodule.setup("predict")
    pinned = lance.dataset(corpus, version=datamodule.dataset_version)

    class CompleteBatchOnly:
        def read_blobs(
            self,
            blob_column: str,
            *,
            indices: list[int],
            preserve_order: bool = True,
        ) -> list[tuple[int, bytes]]:
            if indices != [2, 0, 2, 1]:
                raise AssertionError(f"expected one complete batch, got {indices}")
            return pinned.read_blobs(
                blob_column, indices=indices, preserve_order=preserve_order
            )

    monkeypatch.setattr(
        "synth_setter.data.third_party_datamodule.lance.dataset",
        lambda *args, **kwargs: CompleteBatchOnly(),
    )
    dataset = datamodule.predict_dataloader().dataset
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=[[2, 0, 2, 1]])

    audio = next(iter(loader))["audio"]

    assert audio.shape == (4, _TARGET_CHANNELS, _TARGET_SAMPLES)
    assert torch.equal(audio[0], audio[2])
    assert not torch.equal(audio[0], audio[1])
    assert not torch.equal(audio[2], audio[3])


def test_row_read_transient_blob_failure_retries_and_returns_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transient blob read failure retries against the pinned corpus snapshot.

    :param tmp_path: Isolated corpus fixture directory.
    :param monkeypatch: Fixture injecting one transient Lance failure.
    """
    corpus = tmp_path / "corpus.lance"
    _write_corpus(
        corpus,
        [_tone(_DURATION_SECONDS, seed=45), _tone(_DURATION_SECONDS, seed=46)],
    )
    datamodule = _datamodule(corpus)
    datamodule.setup("predict")
    pinned = lance.dataset(corpus, version=datamodule.dataset_version)

    class BlobReadAfterTimeout:
        def __init__(self) -> None:
            self.attempts = 0

        def read_blobs(
            self,
            blob_column: str,
            *,
            indices: list[int],
            preserve_order: bool = True,
        ) -> list[tuple[int, bytes]]:
            if indices != [0, 1]:
                raise AssertionError(f"expected one complete batch, got {indices}")
            self.attempts += 1
            if self.attempts == 1:
                raise TimeoutError("transient object-store timeout")
            return pinned.read_blobs(
                blob_column, indices=indices, preserve_order=preserve_order
            )

    flaky_dataset = BlobReadAfterTimeout()
    monkeypatch.setattr(
        "synth_setter.data.third_party_datamodule.lance.dataset",
        lambda *args, **kwargs: flaky_dataset,
    )

    batch = next(iter(datamodule.predict_dataloader()))

    assert batch["audio"].shape == (2, _TARGET_CHANNELS, _TARGET_SAMPLES)


def test_configured_version_replays_snapshot_after_later_commit(tmp_path: Path) -> None:
    """A stored dataset version reproduces the same corpus after later commits.

    :param tmp_path: Isolated corpus fixture directory.
    """
    corpus = tmp_path / "corpus.lance"
    original = _tone(_DURATION_SECONDS, seed=47)
    _write_corpus(corpus, [original])
    _write_corpus(corpus, [_tone(_DURATION_SECONDS, seed=48)], mode="overwrite")

    served = _first_batch(_datamodule(corpus, dataset_version=1))["audio"]
    expected = decode_clip(
        wav_bytes(original, _SOURCE_SAMPLE_RATE),
        sample_rate=_TARGET_SAMPLE_RATE,
        channels=_TARGET_CHANNELS,
        num_samples=_TARGET_SAMPLES,
        amplitude_scale=1.0,
    )

    assert torch.equal(served, torch.from_numpy(expected).unsqueeze(0))


@pytest.mark.parametrize("dataset_version", [True, 0, -1])
def test_invalid_dataset_version_raises(tmp_path: Path, dataset_version: int) -> None:
    """A mutable or nonexistent snapshot identifier fails before corpus access.

    :param tmp_path: Isolated corpus fixture directory.
    :param dataset_version: Invalid Lance version.
    """
    with pytest.raises(ValueError, match="dataset_version"):
        _datamodule(tmp_path / "corpus.lance", dataset_version=dataset_version)


def test_setup_pins_the_snapshot_against_later_corpus_commits(tmp_path: Path) -> None:
    """A commit landing mid-evaluation must not change what the run scores.

    Overwriting rather than appending is what distinguishes a pinned snapshot:
    an unpinned read would serve the replacement clip under the same row index.

    :param tmp_path: Isolated corpus fixture directory.
    """
    corpus = tmp_path / "corpus.lance"
    original = _tone(_DURATION_SECONDS, seed=45)
    _write_corpus(corpus, [original])
    datamodule = _datamodule(corpus)
    datamodule.setup("predict")

    _write_corpus(corpus, [_tone(_DURATION_SECONDS, seed=46)], mode="overwrite")

    served = torch.cat([batch[AUDIO_FIELD] for batch in datamodule.predict_dataloader()])
    expected = decode_clip(
        wav_bytes(original, _SOURCE_SAMPLE_RATE),
        sample_rate=_TARGET_SAMPLE_RATE,
        channels=_TARGET_CHANNELS,
        num_samples=_TARGET_SAMPLES,
        amplitude_scale=1.0,
    )
    assert torch.equal(served, torch.from_numpy(expected).unsqueeze(0))


def test_non_numeric_duration_raises(tmp_path: Path) -> None:
    """The sample-count product must not run before its operands are checked.

    :param tmp_path: Isolated corpus fixture directory.
    """
    _write_corpus(tmp_path / "corpus.lance", [_tone(_DURATION_SECONDS, seed=48)])

    with pytest.raises(ValueError, match="must be a number"):
        ThirdPartyAudioDataModule(
            dataset_uri=str(tmp_path / "corpus.lance"),
            sample_rate=_TARGET_SAMPLE_RATE,
            channels=_TARGET_CHANNELS,
            signal_duration_seconds="0.5",  # type: ignore[arg-type]
            dataset_version=1,
        )


def test_statistics_configured_with_normalization_off_raises(tmp_path: Path) -> None:
    """Dropping configured statistics would feed the checkpoint raw mel.

    :param tmp_path: Isolated corpus fixture directory.
    """
    _write_corpus(tmp_path / "corpus.lance", [_tone(_DURATION_SECONDS, seed=49)])

    with pytest.raises(ValueError, match="would be dropped"):
        ThirdPartyAudioDataModule(
            dataset_uri=str(tmp_path / "corpus.lance"),
            sample_rate=_TARGET_SAMPLE_RATE,
            channels=_TARGET_CHANNELS,
            signal_duration_seconds=_DURATION_SECONDS,
            dataset_version=1,
            use_saved_mean_and_variance=False,
            mel_stats_uri="r2://intermediate-data/stats.json",
        )
