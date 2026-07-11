"""Deterministic retry-loop semantics for ``generate_sample`` (#884).

Only the VST render is faked (silent vs. loud audio); the real parameter sampler
and loudness meter run, so these pin the actual seeded-retry behavior: the
``attempt`` is part of the seed input, and the accepted row is reproducible.
"""

import numpy as np
import pytest

from synth_setter.data.vst import generate_vst_dataset, param_specs
from synth_setter.data.vst.generate_vst_dataset import SampleSeed
from synth_setter.data.vst.seeding import rng_for_sample

_SPEC_NAME = "surge_xt"
_PLUGIN_PATH = "plugins/Surge XT.vst3"
_PRESET_PATH = "presets/surge-base.vstpreset"
_SAMPLE_RATE = 44100.0
_CHANNELS = 2
_DURATION = 1.0
_VELOCITY = 100
_MIN_LOUDNESS = -55.0
_MASTER_SEED = 4242
_SAMPLE_IDX = 7


def _silent_audio() -> np.ndarray:
    return np.zeros((_CHANNELS, int(_SAMPLE_RATE * _DURATION)), dtype=np.float32)


def _loud_audio() -> np.ndarray:
    n = int(_SAMPLE_RATE * _DURATION)
    t = np.arange(n) / _SAMPLE_RATE
    sine = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    return np.stack([sine, sine], axis=0)


def _patch_render(monkeypatch: pytest.MonkeyPatch, outputs: list[np.ndarray]) -> None:
    stream = iter(outputs)
    monkeypatch.setattr(generate_vst_dataset, "render_params", lambda *a, **kw: next(stream))


def _generate(
    *,
    master_seed: int | None = _MASTER_SEED,
    sample_idx: int = _SAMPLE_IDX,
    max_attempts: int = 5,
) -> generate_vst_dataset.VSTDataSample:
    seed = (
        SampleSeed(master_seed=master_seed, sample_idx=sample_idx, max_attempts=max_attempts)
        if master_seed is not None
        else None
    )
    return generate_vst_dataset.generate_sample(
        plugin_path=_PLUGIN_PATH,
        velocity=_VELOCITY,
        signal_duration_seconds=_DURATION,
        sample_rate=_SAMPLE_RATE,
        channels=_CHANNELS,
        min_loudness=_MIN_LOUDNESS,
        param_spec=param_specs[_SPEC_NAME],
        preset_path=_PRESET_PATH,
        seed=seed,
    )


def test_silent_first_attempt_uses_attempt_1_rng(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_render(monkeypatch, [_silent_audio(), _loud_audio()])
    expected_synth, _ = param_specs[_SPEC_NAME].sample(
        rng_for_sample(_MASTER_SEED, _SAMPLE_IDX, 1)
    )

    sample = _generate()

    assert sample.synth_params == expected_synth
    assert sample.attempt == 1


def test_generate_sample_twice_yields_identical_params_and_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_render(monkeypatch, [_silent_audio(), _loud_audio()])
    first = _generate()
    _patch_render(monkeypatch, [_silent_audio(), _loud_audio()])
    second = _generate()

    assert first.synth_params == second.synth_params
    assert first.note_params == second.note_params
    assert first.attempt == second.attempt


def test_generate_sample_records_accepted_attempt_zero_when_first_loud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_render(monkeypatch, [_loud_audio()])
    assert _generate().attempt == 0


def test_generate_sample_all_attempts_silent_raises_runtimeerror_naming_sample_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_render(monkeypatch, [_silent_audio()] * 3)
    with pytest.raises(RuntimeError, match=f"sample {_SAMPLE_IDX}"):
        _generate(max_attempts=3)


@pytest.mark.parametrize("bad_attempts", [0, -1])
def test_generate_sample_rejects_nonpositive_seed_attempt_budget(
    bad_attempts: int,
) -> None:
    """Invalid ``SampleSeed.max_attempts`` fails before rendering.

    :param bad_attempts: Invalid attempt budget value.
    """
    with pytest.raises(ValueError, match="max_attempts must be >= 1"):
        _generate(max_attempts=bad_attempts)


def test_generate_sample_last_attempt_audible_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_render(monkeypatch, [_silent_audio(), _silent_audio(), _loud_audio()])
    assert _generate(max_attempts=3).attempt == 2


def test_generate_sample_without_master_seed_samples_nondeterministically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_render(monkeypatch, [_loud_audio()])
    first = _generate(master_seed=None)
    _patch_render(monkeypatch, [_loud_audio()])
    second = _generate(master_seed=None)

    assert first.synth_params != second.synth_params
    assert first.attempt == second.attempt == 0
