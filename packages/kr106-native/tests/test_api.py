"""Verify native rendering, control precedence, and output ownership."""

import gc

import kr106_native
import numpy as np
import pytest
from numpy.typing import NDArray


def render(**overrides: object) -> NDArray[np.float32]:
    """Render a short note with optional native API overrides.

    :param **overrides: Keyword arguments replacing the reference note settings.
    :returns: Channel-first stereo samples owned by NumPy.
    """
    arguments = {
        "parameters": {44: 0.35},
        "midi_note": 60,
        "velocity": 100,
        "start_sample": 0,
        "end_sample": 2048,
        "num_samples": 4096,
        "sample_rate": 44_100.0,
    }
    arguments.update(overrides)
    return kr106_native.render_note(**arguments)


def test_metadata_exposes_pinned_engine_and_complete_control_surface() -> None:
    """Expose the pinned revision and native parameter ranges."""
    parameters = kr106_native.get_parameters()

    assert kr106_native.get_version() == "2.5.13"
    assert kr106_native.get_source_revision() == "bc15caee5843ab238a25d0969e68d57db2b1615f"
    assert [parameter["id"] for parameter in parameters] == list(range(58))
    assert parameters[44]["name"] == "Master Volume"
    assert parameters[56]["name"] == "Program"
    assert parameters[56]["maximum"] == 127.0
    assert parameters[57]["name"] == "Bypass"
    assert {parameter["kind"] for parameter in parameters} == {"float", "int", "bool"}


def test_render_note_returns_owned_stereo_float32_audio() -> None:
    """Produce finite, non-silent, contiguous stereo samples."""
    audio = render()

    assert audio.shape == (2, 4096)
    assert audio.dtype == np.float32
    assert audio.flags.c_contiguous
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 1e-5


def test_zero_velocity_renders_only_idle_noise() -> None:
    """MIDI velocity zero is note-off even when Ignore Velocity is enabled."""
    idle = render(parameters={44: 0.35, 20: 1})
    zero_velocity = render(velocity=0)
    np.testing.assert_array_equal(zero_velocity, idle)


def test_non_block_aligned_note_on_starts_at_requested_sample() -> None:
    """The note leaves the idle prefix unchanged and sounds before the next block edge."""
    actual = render(start_sample=513, end_sample=1027, block_size=512)
    idle = render(velocity=0, start_sample=513, end_sample=1027, block_size=512)
    np.testing.assert_array_equal(actual[:, :513], idle[:, :513])
    assert not np.array_equal(actual[:, 513:1024], idle[:, 513:1024])


def test_non_block_aligned_note_off_changes_only_release_samples() -> None:
    """The note-off boundary separates an identical prefix from the hold-enabled reference."""
    released = render(start_sample=513, end_sample=1027, block_size=512)
    held = render(parameters={44: 0.35, 21: 1}, start_sample=513, end_sample=1027, block_size=512)
    np.testing.assert_array_equal(released[:, :1027], held[:, :1027])
    assert not np.array_equal(released[:, 1027:], held[:, 1027:])


def test_render_note_applies_baseline_before_program_and_explicit_controls_after() -> None:
    """Let explicit master volume override the baseline after program loading."""
    from_program = render(parameters={56: 0}, baseline_parameters={44: 1.0})
    overridden = render(parameters={44: 0.0, 56: 0}, baseline_parameters={44: 1.0})

    assert np.max(np.abs(from_program)) > 1e-5
    np.testing.assert_array_equal(overridden, np.zeros((2, 4096), dtype=np.float32))


def test_render_note_program_preserves_live_baseline_controls_and_uses_its_model_bank() -> None:
    """Keep live volume while selecting presets from the baseline model bank."""
    muted = render(parameters={56: 0}, baseline_parameters={43: 1.0, 44: 0.0})
    j60 = render(parameters={56: 0}, baseline_parameters={43: 0.0, 44: 0.35})
    j106 = render(parameters={56: 0}, baseline_parameters={43: 1.0, 44: 0.35})

    np.testing.assert_array_equal(muted, np.zeros((2, 4096), dtype=np.float32))
    assert not np.array_equal(j60, j106)


def test_render_note_honors_baseline_power_and_sync_settings() -> None:
    """Apply baseline power and host synchronization to rendered audio."""
    powered_off = render(baseline_parameters={38: 0.0})
    free_running = render(parameters={5: 1.0}, baseline_parameters={50: 0.0, 54: 0.0})
    host_synced = render(parameters={5: 1.0}, baseline_parameters={50: 1.0, 54: 0.0})

    np.testing.assert_array_equal(powered_off, np.zeros((2, 4096), dtype=np.float32))
    assert not np.array_equal(free_running, host_synced)


def test_explicit_power_on_overrides_powered_off_baseline() -> None:
    """An off baseline cannot mute a later explicit power-on control."""
    restored = render(parameters={38: 1, 44: 0.35}, baseline_parameters={38: 0})
    powered = render(parameters={38: 1, 44: 0.35}, baseline_parameters={38: 1})
    assert np.max(np.abs(restored)) > 0.001
    np.testing.assert_array_equal(restored, powered)


def test_render_note_is_repeatable_across_fresh_engines() -> None:
    """Isolate repeated notes from intervening renders."""
    first = render()
    _ = render(midi_note=67)
    again = render()

    np.testing.assert_array_equal(first, again)


def test_render_note_output_remains_valid_after_next_render_and_collection() -> None:
    """Keep sample storage alive independently of subsequent engine lifetimes."""
    first = render()
    first_copy = first.copy()
    _ = render(midi_note=67)
    gc.collect()

    np.testing.assert_array_equal(first, first_copy)


@pytest.mark.parametrize(
    ("parameter_id", "value"),
    [(41, 12), (46, 4), (47, 0), (55, 0), (56, 127)],
)
def test_render_note_dispatches_engine_specific_controls(parameter_id: int, value: int) -> None:
    """Make engine-specific controls audibly affect the reference note.

    :param parameter_id: Native control requiring engine-specific dispatch.
    :param value: Non-default native value to apply.
    """
    baseline = render()
    changed = render(parameters={44: 0.35, parameter_id: value})

    assert not np.array_equal(baseline, changed)


@pytest.mark.parametrize(
    "changes",
    [
        {"midi_note": -1},
        {"velocity": 128},
        {"start_sample": -1},
        {"end_sample": 4097},
        {"num_samples": 0},
        {"sample_rate": 0.0},
        {"block_size": 0},
        {"parameters": {999: 0.0}},
        {"parameters": {44: 2.0}},
        {"parameters": {56: 128}},
        {"parameters": {45: 6.5}},
        {"parameters": {20: 0.5}},
        {"baseline_parameters": {45: 6.5}},
        {"start_sample": 2048, "end_sample": 2048},
    ],
)
def test_render_note_rejects_invalid_inputs(changes: dict[str, object]) -> None:
    """Reject invalid note intervals and out-of-range native controls.

    :param changes: Invalid overrides for an otherwise valid note.
    """
    with pytest.raises((TypeError, ValueError)):
        render(**changes)
