"""Tests for serving third-party audio corpora with live batch transforms."""

import io
import pickle
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest
import torch
from pedalboard.io import AudioFile

from synth_setter.conditioning import (
    NUM_SKETCH_CONTROLS,
    SKETCH_CTRL_FIELD,
    Conditioning,
    EmbeddingConditioningSpec,
    SketchControls,
    SketchControlSpec,
)
from synth_setter.data.third_party_datamodule import (
    LiveEmbedding,
    ThirdPartyAudioDataModule,
    decode_clip,
)
from synth_setter.data.vst.shapes import AUDIO_FIELD, MEL_N_MELS
from synth_setter.pipeline.data.add_embeddings import EMBEDDING_REGISTRY, Encoder
from synth_setter.pipeline.data.ssondo import SSONDO_EMBEDDING_DIM
from tests.helpers.lance_fixtures import write_blob_audio_corpus

# Keep the test corpus compact while retaining a legal mel front-end.
_TARGET_SAMPLE_RATE = 4000
_TARGET_CHANNELS = 2
_DURATION_SECONDS = 0.5
_TARGET_SAMPLES = 2000
_MEL_FRAMES = 51
_SOURCE_SAMPLE_RATE = 8000


def _wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Encode one mono clip as WAV bytes.

    :param samples: ``(frames,)`` float32 samples.
    :param sample_rate: Encoded sample rate in Hz.
    :returns: WAV container bytes.
    """
    buffer = io.BytesIO()
    with AudioFile(buffer, "w", format="wav", samplerate=sample_rate, num_channels=1) as handle:
        handle.write(samples.reshape(1, -1).astype(np.float32))
    return buffer.getvalue()


def _write_corpus(
    path: Path,
    clips: list[np.ndarray],
    *,
    sample_rate: int = _SOURCE_SAMPLE_RATE,
    audio_column: str = AUDIO_FIELD,
    with_sample_rate_column: bool = True,
) -> None:
    """Write a blob-encoded audio corpus in the published third-party layout.

    :param path: Destination Lance dataset.
    :param clips: One mono float32 clip per row.
    :param sample_rate: Encoded sample rate for every clip.
    :param audio_column: Blob column name, mirroring per-corpus schema drift.
    :param with_sample_rate_column: Whether to store the per-row sample rate.
    """
    write_blob_audio_corpus(
        path,
        clips,
        sample_rate=sample_rate,
        audio_column=audio_column,
        with_sample_rate_column=with_sample_rate_column,
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
    uri: Path,
    *,
    sample_rate: int = _TARGET_SAMPLE_RATE,
    audio_column: str = AUDIO_FIELD,
    amplitude_scale: float = 1.0,
    row_limit: int | None = None,
    conditioning: Conditioning = "mel",
    sketch: SketchControls = None,
    embedding_encoder: Encoder | None = None,
    use_saved_mean_and_variance: bool = False,
    mel_stats_uri: str | None = None,
) -> ThirdPartyAudioDataModule:
    """Build a datamodule over a corpus with the tiny render contract.

    :param uri: Corpus Lance dataset the datamodule reads in place.
    :param sample_rate: Target render rate the corpus is mapped onto.
    :param audio_column: Blob column the corpus stores its audio in.
    :param amplitude_scale: Gain applied to decoded audio.
    :param row_limit: Cap on served rows, or ``None`` for the whole corpus.
    :param conditioning: Conditioning mode or embedding spec under test.
    :param sketch: Sketch-control spec, or ``None`` for no control tokens.
    :param embedding_encoder: Encoder standing in for downloaded frozen weights.
    :param use_saved_mean_and_variance: Whether mel is standardized.
    :param mel_stats_uri: Statistics source when standardization is on.
    :returns: Configured, un-setup datamodule.
    """
    return ThirdPartyAudioDataModule(
        dataset_uri=str(uri),
        sample_rate=sample_rate,
        channels=_TARGET_CHANNELS,
        signal_duration_seconds=_DURATION_SECONDS,
        batch_size=2,
        audio_column=audio_column,
        amplitude_scale=amplitude_scale,
        row_limit=row_limit,
        conditioning=conditioning,
        sketch=sketch,
        embedding_encoder=embedding_encoder,
        use_saved_mean_and_variance=use_saved_mean_and_variance,
        mel_stats_uri=mel_stats_uri,
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
    """A clip longer than the render duration is truncated, not resampled to fit.

    :param tmp_path: Isolated corpus fixture directory.
    """
    _write_corpus(tmp_path / "corpus.lance", [_tone(_DURATION_SECONDS * 3, seed=4)])

    audio = _first_batch(_datamodule(tmp_path / "corpus.lance"))["audio"]

    assert audio.shape[-1] == _TARGET_SAMPLES


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

    assert batch["mel"].shape == (1, _TARGET_CHANNELS, MEL_N_MELS, _MEL_FRAMES)


def test_predict_batch_saved_statistics_normalize_mel(tmp_path: Path) -> None:
    """Configured statistics standardize the mel the corpus would otherwise emit.

    :param tmp_path: Isolated corpus fixture directory.
    """
    _write_corpus(tmp_path / "corpus.lance", [_tone(_DURATION_SECONDS, seed=7)])
    stats_file = tmp_path / "stats.npz"
    np.savez(stats_file, mean=np.float32(-40.0), std=np.float32(4.0))

    plain = _first_batch(_datamodule(tmp_path / "corpus.lance"))["mel"]
    normalized = _first_batch(
        _datamodule(
            tmp_path / "corpus.lance",
            use_saved_mean_and_variance=True,
            mel_stats_uri=str(stats_file),
        )
    )["mel"]

    assert torch.allclose(normalized, (plain + 40.0) / 4.0, atol=1e-5)


def test_saved_statistics_without_uri_raises(tmp_path: Path) -> None:
    """Normalization without a statistics source fails loudly instead of passing raw mel.

    :param tmp_path: Isolated corpus fixture directory.
    """
    _write_corpus(tmp_path / "corpus.lance", [_tone(_DURATION_SECONDS, seed=8)])

    with pytest.raises(ValueError, match="mel_stats_uri"):
        _datamodule(tmp_path / "corpus.lance", use_saved_mean_and_variance=True)


def test_embedding_conditioning_does_not_require_mel_statistics(tmp_path: Path) -> None:
    """An arm that never reads mel is servable without a statistics source.

    :param tmp_path: Isolated corpus fixture directory.
    """
    _write_corpus(tmp_path / "corpus.lance", [_tone(_DURATION_SECONDS, seed=15)])

    def encode(source: np.ndarray, sample_rate: int) -> np.ndarray:
        del sample_rate
        return np.ones((len(source), SSONDO_EMBEDDING_DIM), dtype=np.float32)

    batch = _first_batch(
        _datamodule(
            tmp_path / "corpus.lance",
            conditioning=EmbeddingConditioningSpec(
                column="ssondo", input_shape=(SSONDO_EMBEDDING_DIM,)
            ),
            embedding_encoder=encode,
            use_saved_mean_and_variance=True,
        )
    )

    assert "mel" not in batch


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


def test_live_embedding_emits_the_stored_column_shape() -> None:
    """The registry's own encode contract shapes the live conditioning tensor."""
    audio = np.zeros((2, 1, 32_000), dtype=np.float32)

    def encode(source: np.ndarray, sample_rate: int) -> np.ndarray:
        del source, sample_rate
        return np.ones((2, SSONDO_EMBEDDING_DIM), dtype=np.float32)

    live = LiveEmbedding(EMBEDDING_REGISTRY["ssondo"], encode, sample_rate=_TARGET_SAMPLE_RATE)

    assert live(audio)["ssondo"].shape == (2, SSONDO_EMBEDDING_DIM)


def test_live_embedding_rejects_output_disagreeing_with_registry() -> None:
    """A malformed live embedding fails where the stored writer would have failed."""
    audio = np.zeros((2, 1, 32_000), dtype=np.float32)

    def encode(source: np.ndarray, sample_rate: int) -> np.ndarray:
        del source, sample_rate
        return np.ones((2, 3), dtype=np.float32)

    live = LiveEmbedding(EMBEDDING_REGISTRY["ssondo"], encode, sample_rate=_TARGET_SAMPLE_RATE)

    with pytest.raises(ValueError, match="produced shape"):
        live(audio)


def test_embedding_conditioning_populates_the_model_batch_key(tmp_path: Path) -> None:
    """A configured embedding column reaches the model under the shared batch key.

    :param tmp_path: Isolated corpus fixture directory.
    """
    _write_corpus(tmp_path / "corpus.lance", [_tone(_DURATION_SECONDS, seed=11)])

    def encode(source: np.ndarray, sample_rate: int) -> np.ndarray:
        del sample_rate
        return np.ones((len(source), SSONDO_EMBEDDING_DIM), dtype=np.float32)

    datamodule = _datamodule(
        tmp_path / "corpus.lance",
        conditioning=EmbeddingConditioningSpec(
            column="ssondo", input_shape=(SSONDO_EMBEDDING_DIM,)
        ),
        embedding_encoder=encode,
    )

    assert _first_batch(datamodule)["conditioning"].shape == (1, SSONDO_EMBEDDING_DIM)


def test_embedding_conditioning_shape_mismatch_raises(tmp_path: Path) -> None:
    """A column whose live shape contradicts the checkpoint's spec is rejected.

    :param tmp_path: Isolated corpus fixture directory.
    """
    _write_corpus(tmp_path / "corpus.lance", [_tone(_DURATION_SECONDS, seed=12)])

    def encode(source: np.ndarray, sample_rate: int) -> np.ndarray:
        del sample_rate
        return np.ones((len(source), SSONDO_EMBEDDING_DIM), dtype=np.float32)

    datamodule = _datamodule(
        tmp_path / "corpus.lance",
        conditioning=EmbeddingConditioningSpec(column="ssondo", input_shape=(7,)),
        embedding_encoder=encode,
    )

    with pytest.raises(ValueError, match="input_shape"):
        _first_batch(datamodule)


def test_unknown_conditioning_column_raises(tmp_path: Path) -> None:
    """Only columns the embedding registry can produce are servable live.

    :param tmp_path: Isolated corpus fixture directory.
    """
    _write_corpus(tmp_path / "corpus.lance", [_tone(_DURATION_SECONDS, seed=13)])

    with pytest.raises(KeyError, match="nonesuch"):
        _datamodule(
            tmp_path / "corpus.lance",
            conditioning=EmbeddingConditioningSpec(column="nonesuch", input_shape=(4,)),
        )


def test_sketch_conditioning_emits_the_control_stack(tmp_path: Path) -> None:
    """Sketch controls are extracted live and reassembled into the model tensor.

    :param tmp_path: Isolated corpus fixture directory.
    """
    sketch_rate = 16_000
    _write_corpus(
        tmp_path / "corpus.lance",
        [_tone(_DURATION_SECONDS, sample_rate=sketch_rate, seed=14)],
        sample_rate=sketch_rate,
    )

    batch = _first_batch(
        _datamodule(
            tmp_path / "corpus.lance",
            sample_rate=sketch_rate,
            sketch=SketchControlSpec(num_frames=51, num_control_tokens=4),
        )
    )

    controls = batch[SKETCH_CTRL_FIELD]

    assert controls.shape == (1, NUM_SKETCH_CONTROLS, 51)
    assert controls.dtype == torch.float32


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

    assert revived[0]["audio"].shape == (_TARGET_CHANNELS, _TARGET_SAMPLES)


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


def test_sketch_frames_disagreeing_with_the_checkpoint_spec_raise(tmp_path: Path) -> None:
    """Live controls whose frame axis contradicts the checkpoint's spec are rejected.

    A corpus duration or rate that yields a different control grid would otherwise hand the model
    silently misaligned tokens.

    :param tmp_path: Isolated corpus fixture directory.
    """
    sketch_rate = 16_000
    _write_corpus(
        tmp_path / "corpus.lance",
        [_tone(_DURATION_SECONDS, sample_rate=sketch_rate, seed=21)],
        sample_rate=sketch_rate,
    )

    with pytest.raises(ValueError, match="num_frames"):
        _first_batch(
            _datamodule(
                tmp_path / "corpus.lance",
                sample_rate=sketch_rate,
                sketch=SketchControlSpec(num_frames=50, num_control_tokens=4),
            )
        )


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
