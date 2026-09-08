"""Behaviour tests for the renderer-backed reward that scores sampled rows against target rows."""

from pathlib import Path

import numpy as np
import pytest
import torch

from synth_setter.data.vst.param_spec_registry import param_specs
from synth_setter.models.components.audio_distance import MultiScaleSpectralDistance
from synth_setter.models.components.rendered_reward import SynthRenderedReward
from synth_setter.pipeline.schemas.spec import RenderConfig
from synth_setter.synth_spec import SYNTHS

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SAMPLE_RATE = 16_000
_SIGNAL_DURATION_SECONDS = 1.0


def _surgepy_synth() -> dict[str, object]:
    """Return the surge_simple surgepy identity with its preset pinned to this checkout.

    The registry row names the preset relative to the operator workspace, which another test in the
    same worker may have pointed elsewhere; an absolute path keeps this module independent of that
    process-wide state.

    :returns: SynthSpec fields for the surgepy surge_simple row.
    """
    synth = next(spec for spec in SYNTHS.values() if spec.name == "surge_simple_surgepy")
    values = synth.model_dump()
    values["plugin_state_path"] = str(_REPO_ROOT / synth.plugin_state_path)
    return values


def _reward() -> SynthRenderedReward:
    render = RenderConfig.model_validate(
        {
            "synth": _surgepy_synth(),
            "renderer_backend": "surgepy",
            "sample_rate": _SAMPLE_RATE,
            "channels": 1,
            "velocity": 100,
            "signal_duration_seconds": _SIGNAL_DURATION_SECONDS,
            "min_loudness": -55.0,
            "audio_dtype": "float16",
            "mel_spec_dtype": "float32",
            "samples_per_shard": 8,
            "samples_per_render_batch": 8,
            "max_retries": 0,
            "parallel": False,
            "retain_local_shards": True,
            "plugin_reload_cadence": "render",
            "gui_toggle_cadence": "never",
            "param_sample_cadence": "sample",
        }
    )
    return SynthRenderedReward(
        render_config=render, distance=MultiScaleSpectralDistance(sample_rate=_SAMPLE_RATE)
    )


def _rows(count: int, seed: int) -> torch.Tensor:
    """Draw model-space rows whose note sounds inside the short render buffer.

    :param count: Number of rows.
    :param seed: Seed for the synth columns.
    :returns: Rows shaped ``(count, encoded_width)`` in ``[-1, 1]``.
    """
    spec = param_specs["surge_simple"]
    generator = torch.Generator().manual_seed(seed)
    rows = torch.rand(count, spec.encoded_width, generator=generator) * 2 - 1
    note = spec.encode(
        spec.sample(np.random.default_rng(seed))[0],
        {"pitch": 60, "note_start_and_end": (0.0, _SIGNAL_DURATION_SECONDS)},
    )
    rows[:, spec.synth_columns.stop :] = torch.from_numpy(note[spec.synth_columns.stop :]) * 2 - 1
    return rows


@pytest.mark.requires_surgepy
def test_synth_rendered_reward_prefers_the_row_that_matches_the_target() -> None:
    """The reward is non-positive and highest where sampled and target rows coincide."""
    rows = _rows(2, seed=3)
    target = rows[:1].expand(2, -1)

    rewards = _reward()(rows, target)

    assert rewards.shape == (2,)
    assert rewards[0] > rewards[1]
    assert (rewards <= 0).all()


@pytest.mark.requires_surgepy
def test_synth_rendered_reward_scores_each_row_against_its_own_target() -> None:
    """Swapping one row's target moves that row's reward far more than any other row's.

    Surge randomises oscillator phase per render, so two renders of one row differ by a small
    spectral distance; the swapped row must move by an order of magnitude more than that floor.
    """
    rows = _rows(3, seed=4)
    reward = _reward()

    aligned = reward(rows, rows)
    swapped = reward(rows, torch.stack([rows[0], rows[2], rows[2]]))

    render_floor = (aligned[[0, 2]] - swapped[[0, 2]]).abs().max()
    assert swapped[1] < aligned[1]
    assert (aligned[1] - swapped[1]) > 10 * render_floor


@pytest.mark.requires_surgepy
def test_synth_rendered_reward_renders_out_of_range_rows_without_raising() -> None:
    """Sampled rows can leave ``[-1, 1]``; the reward clamps them into the renderable domain."""
    rows = _rows(2, seed=5) * 3.0

    rewards = _reward()(rows, _rows(2, seed=6))

    assert torch.isfinite(rewards).all()


def test_synth_rendered_reward_declares_params_as_its_target() -> None:
    """The module reads the target from the batch key the reward declares."""
    assert SynthRenderedReward.target_key == "params"
