"""Native Dexed config-to-dataset-row integration."""

from pathlib import Path

import lance
import numpy as np
import pytest
from dexed import DexedSynth
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf, open_dict

from synth_setter.data.vst.dexed_renderer import DexedRenderer
from synth_setter.data.vst.generate_vst_dataset import SampleSeed, generate_sample
from synth_setter.data.vst.param_spec_registry import param_specs
from synth_setter.data.vst.writers import make_lance_dataset
from synth_setter.pipeline.schemas.spec import DatasetSpec, RenderConfig
from synth_setter.renderer_factory import make_audio_renderer
from synth_setter.synth_spec import SYNTHS, SynthName


def test_dexed_hydra_config_generates_audio_features_and_parameters() -> None:
    """The real sampling pipeline accepts native audio and encodes its training row."""
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        config = compose(config_name="render/dexed").render
    with open_dict(config):
        config.synth = SYNTHS[SynthName("dexed_py")].model_dump()
    render = RenderConfig.model_validate(OmegaConf.to_container(config, resolve=True))
    sample = generate_sample(
        make_audio_renderer(render),
        render.velocity,
        render.min_loudness,
        param_specs["dexed_py"],
        fixed_note_params={"pitch": 60, "note_start_and_end": (0.0, 1.0)},
        seed=SampleSeed(master_seed=42, max_attempts=20),
        audio_dtype=render.audio_dtype,
    )
    assert sample.audio.shape == (176400, 2)
    assert np.isfinite(sample.audio).all()
    assert 0.001 < np.max(np.abs(sample.audio)) <= 1.0
    assert sample.param_array.shape == (148,)
    assert np.all((sample.param_array >= 0) & (sample.param_array <= 1))
    assert sample.mel_spec.shape == (2, 128, 401)
    assert np.isfinite(sample.mel_spec).all()
    assert len(sample.audio_mp3) > 1000


def test_dexed_2m_experiment_shard_writes_lance_row(tmp_path: Path) -> None:
    """A production 2M shard's render config writes a real native Lance row.

    :param tmp_path: Per-test output directory.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="dataset",
            overrides=["experiment=generate_dataset/dexed-py-lance-2m-40k-10k"],
        )
    spec = DatasetSpec.from_hydra_cfg(cfg)
    shard = spec.shards[0]
    render_cfg = spec.render_for_shard(shard).model_copy(update={"samples_per_shard": 1})

    make_lance_dataset(tmp_path / "shard-000000.lance", render_cfg, shard_id=0)

    assert lance.dataset(tmp_path / "shard-000000.lance").count_rows() == 1


def test_non_finite_native_render_is_counted_rejection_not_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A NaN native render is a counted per-draw rejection; the next draw is accepted.

    :param monkeypatch: Injects one non-finite native render; the engine is fixed-point and cannot
        be driven to NaN by any patch.
    """
    renderer = DexedRenderer(
        plugin_path="dexed", sample_rate=44100, channels=2, signal_duration_seconds=4.0
    )
    real_render = DexedSynth.render
    calls = iter([True])

    def render_nan_once(
        self: DexedSynth,
        midi_note: int,
        velocity: int,
        note_duration: float,
        render_duration: float,
    ) -> np.ndarray:
        audio = real_render(self, midi_note, velocity, note_duration, render_duration)
        return np.full_like(audio, np.nan) if next(calls, False) else audio

    monkeypatch.setattr(DexedSynth, "render", render_nan_once)

    sample = generate_sample(
        renderer,
        100,
        -55.0,
        param_specs["dexed_py"],
        fixed_note_params={"pitch": 60, "note_start_and_end": (0.0, 1.0)},
        seed=SampleSeed(master_seed=42, max_attempts=20),
    )

    assert sample.non_finite_rejections == 1
    assert np.isfinite(sample.audio).all()
