"""Tests for the full_flush flag in render_params.

Default behavior (full_flush=False) skips pre-render flush+reset cycles
for faster generation. Set FULL_FLUSH=1 or full_flush=True for the
conservative 3-cycle flush+reset behavior.
"""

from unittest.mock import MagicMock

import numpy as np

from src.data.vst.core import render_params

SAMPLE_RATE = 44100.0
CHANNELS = 2
DURATION = 1.0


def _make_mock_plugin():
    """Create a mock plugin that tracks process/reset calls and returns audio."""
    plugin = MagicMock()
    plugin.process.return_value = np.zeros((CHANNELS, int(SAMPLE_RATE * DURATION)))
    plugin.parameters = {}
    return plugin


def test_render_params_default_only_renders():
    """Default (fast mode) should only call process for the render itself.
    pedalboard's process(reset=True) handles state clearing internally."""
    plugin = _make_mock_plugin()

    render_params(
        plugin,
        params={},
        midi_note=60,
        velocity=100,
        note_start_and_end=(0.0, 0.5),
        signal_duration_seconds=DURATION,
        sample_rate=SAMPLE_RATE,
        channels=CHANNELS,
    )

    # 1 process call: render only (no manual flush cycles)
    assert plugin.process.call_count == 1
    # 0 reset calls: pedalboard handles reset via process(reset=True)
    assert plugin.reset.call_count == 0


def test_render_params_full_flush_calls_all_cycles():
    """full_flush=True should call all 3 flush+reset cycles."""
    plugin = _make_mock_plugin()

    render_params(
        plugin,
        params={},
        midi_note=60,
        velocity=100,
        note_start_and_end=(0.0, 0.5),
        signal_duration_seconds=DURATION,
        sample_rate=SAMPLE_RATE,
        channels=CHANNELS,
        full_flush=True,
    )

    # 4 process calls: post-load flush, post-param flush, render, post-render flush
    assert plugin.process.call_count == 4
    # 3 reset calls: post-load, post-param, post-render
    assert plugin.reset.call_count == 3


def test_render_params_full_flush_reads_from_env(monkeypatch):
    """FULL_FLUSH=1 env var should enable all flush+reset cycles."""
    monkeypatch.setenv("FULL_FLUSH", "1")
    plugin = _make_mock_plugin()

    render_params(
        plugin,
        params={},
        midi_note=60,
        velocity=100,
        note_start_and_end=(0.0, 0.5),
        signal_duration_seconds=DURATION,
        sample_rate=SAMPLE_RATE,
        channels=CHANNELS,
    )

    assert plugin.process.call_count == 4
    assert plugin.reset.call_count == 3


def test_render_params_explicit_flag_overrides_env(monkeypatch):
    """Explicit full_flush=False should override FULL_FLUSH=1 env var."""
    monkeypatch.setenv("FULL_FLUSH", "1")
    plugin = _make_mock_plugin()

    render_params(
        plugin,
        params={},
        midi_note=60,
        velocity=100,
        note_start_and_end=(0.0, 0.5),
        signal_duration_seconds=DURATION,
        sample_rate=SAMPLE_RATE,
        channels=CHANNELS,
        full_flush=False,
    )

    # Explicit False overrides env: fast mode (render only)
    assert plugin.process.call_count == 1
    assert plugin.reset.call_count == 0


def test_render_params_returns_audio_in_both_modes():
    """Both modes should return the rendered audio output."""
    plugin = _make_mock_plugin()

    result_fast = render_params(
        plugin, {}, 60, 100, (0.0, 0.5), DURATION, SAMPLE_RATE, CHANNELS,
    )
    result_full = render_params(
        plugin, {}, 60, 100, (0.0, 0.5), DURATION, SAMPLE_RATE, CHANNELS, full_flush=True,
    )

    assert isinstance(result_fast, np.ndarray)
    assert isinstance(result_full, np.ndarray)
