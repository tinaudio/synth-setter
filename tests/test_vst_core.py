"""Tests for the VST3 data generation pipeline (src/data/vst/core.py).

Unit tests use mocked plugins and need no external dependencies.
Integration tests (marked ``slow`` and ``vst``) require the Surge XT
VST3 plugin to be available at ``plugins/Surge XT.vst3``.
"""

import time
from typing import Tuple, TypedDict
from unittest.mock import MagicMock, patch

import librosa
import numpy as np
import pytest
import rootutils
from loguru import logger
from scipy.spatial.distance import cosine as cosine_distance

from src.data.vst.core import (
    _call_show_editor_for,
    load_plugin,
    make_midi_events,
    render_params,
    set_params,
)
from src.data.vst.surge_xt_param_spec import SURGE_SIMPLE_PARAM_SPEC

PROJECT_ROOT = rootutils.find_root(indicator=".project-root")
PLUGIN_PATH = str(PROJECT_ROOT / "plugins" / "Surge XT.vst3")
PRESET_BASE = str(PROJECT_ROOT / "presets" / "surge-base.vstpreset")
PRESET_SIMPLE = str(PROJECT_ROOT / "presets" / "surge-simple.vstpreset")

requires_plugin = pytest.mark.skipif(
    not (PROJECT_ROOT / "plugins" / "Surge XT.vst3").exists(),
    reason="Surge XT plugin not found",
)


class RenderConfig(TypedDict):
    """Typed dict matching the keyword arguments of ``render_params``."""

    midi_note: int
    velocity: int
    note_start_and_end: Tuple[float, float]
    signal_duration_seconds: float
    sample_rate: float
    channels: int


RENDER_CFG = RenderConfig(
    midi_note=60,
    velocity=100,
    note_start_and_end=(0.0, 1.5),
    signal_duration_seconds=2.0,
    sample_rate=44100.0,
    channels=2,
)
SR = RENDER_CFG["sample_rate"]


# ---------------------------------------------------------------------------
# Param helpers — SURGE_SIMPLE_PARAM_SPEC contains parameter names that may
# not exist in every pedalboard-exposed Surge XT build, so every sampler
# filters to ``_valid_names`` first.
# ---------------------------------------------------------------------------


def _valid_names(plugin) -> set[str]:
    """Return the set of parameter names the plugin actually exposes."""
    return set(plugin.parameters.keys())


def _sample_params(param_spec, plugin) -> dict[str, float]:
    """Randomly sample params, keeping only those the plugin recognises."""
    synth_params, _ = param_spec.sample()
    valid = _valid_names(plugin)
    return {k: v for k, v in synth_params.items() if k in valid}


def _default_params(param_spec, plugin) -> dict[str, float]:
    """All valid params set to 0.5 — guaranteed to produce audible output."""
    valid = _valid_names(plugin)
    return {p.name: 0.5 for p in param_spec.synth_params if p.name in valid}


def _contrasting_params(param_spec, plugin) -> Tuple[dict[str, float], dict[str, float]]:
    """Two param dicts at opposite extremes (spec min vs spec max)."""
    valid = _valid_names(plugin)
    lo, hi = {}, {}
    for p in param_spec.synth_params:
        if p.name in valid:
            lo[p.name] = getattr(p, "min", 0.0)
            hi[p.name] = getattr(p, "max", 1.0)
    return lo, hi


# ---------------------------------------------------------------------------
# Render / audio helpers
# ---------------------------------------------------------------------------


def _render(plugin, params, preset=PRESET_SIMPLE, cfg=RENDER_CFG) -> np.ndarray:
    """Render audio through the plugin with the given params and preset."""
    return render_params(plugin, params, preset_path=preset, **cfg)


def _mfcc_sim(audio_a, audio_b, sr=SR, n_mfcc=13) -> float:
    """Cosine similarity between mean-MFCC fingerprints (mono-mixed)."""
    mono_a = np.mean(audio_a, axis=0) if audio_a.ndim > 1 else audio_a
    mono_b = np.mean(audio_b, axis=0) if audio_b.ndim > 1 else audio_b
    mfcc_a = librosa.feature.mfcc(y=mono_a, sr=sr, n_mfcc=n_mfcc).mean(axis=1)
    mfcc_b = librosa.feature.mfcc(y=mono_b, sr=sr, n_mfcc=n_mfcc).mean(axis=1)
    return 1.0 - cosine_distance(mfcc_a, mfcc_b)


def _assert_similar(a, b, *, min_sim=0.90, label=""):
    """Assert MFCC similarity between two audio signals exceeds *min_sim*."""
    sim = _mfcc_sim(a, b)
    logger.info(f"{label} MFCC similarity: {sim:.6f}")
    assert sim > min_sim, f"Expected sim > {min_sim}, got {sim:.6f}"


def _assert_different(a, b, *, max_sim=0.999, label=""):
    """Assert MFCC similarity between two audio signals is below *max_sim*."""
    sim = _mfcc_sim(a, b)
    logger.info(f"{label} MFCC similarity: {sim:.6f}")
    assert sim < max_sim, f"Expected sim < {max_sim}, got {sim:.6f}"


# ---------------------------------------------------------------------------
# Mock / timing helpers
# ---------------------------------------------------------------------------


def _mock_plugin(audio_shape=(2, 88200)):
    """Return a ``MagicMock`` that behaves like a minimal ``VST3Plugin``."""
    plugin = MagicMock()
    plugin.parameters = {}
    plugin.process.return_value = np.zeros(audio_shape)
    return plugin


def _elapsed(fn):
    """Time *fn()* and return ``(seconds, result)``."""
    t0 = time.perf_counter()
    result = fn()
    return time.perf_counter() - t0, result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def surge_plugin():
    """Plugin loaded via ``load_plugin`` (includes editor warmup)."""
    if not (PROJECT_ROOT / "plugins" / "Surge XT.vst3").exists():
        pytest.skip("Plugin not found")
    try:
        return load_plugin(PLUGIN_PATH)
    except Exception as e:
        pytest.skip(f"Could not load plugin (headless?): {e}")


@pytest.fixture(scope="session")
def surge_plugin_raw_state():
    """Plugin loaded with the ``raw_state`` workaround (no editor warmup)."""
    if not (PROJECT_ROOT / "plugins" / "Surge XT.vst3").exists():
        pytest.skip("Plugin not found")
    try:
        from pedalboard import VST3Plugin

        p = VST3Plugin(PLUGIN_PATH)
        p.raw_state = p.raw_state
        return p
    except Exception as e:
        pytest.skip(f"Could not load plugin: {e}")


@pytest.fixture(scope="module")
def param_spec_simple():
    """The ``SURGE_SIMPLE_PARAM_SPEC`` parameter specification."""
    return SURGE_SIMPLE_PARAM_SPEC


# ===========================================================================
# Unit tests — no plugin required
# ===========================================================================


class TestMakeMidiEvents:
    """Verify ``make_midi_events`` produces correct note-on / note-off tuples."""

    def test_basic(self):
        """Standard pitch/velocity/timing values."""
        events = make_midi_events(pitch=60, velocity=100, note_start=0.0, note_end=1.0)
        assert events == (([0x90, 60, 100], 0.0), ([0x80, 60, 100], 1.0))

    def test_boundary_values(self):
        """Extremes: pitch 0/127, velocity 0/127."""
        lo = make_midi_events(pitch=0, velocity=0, note_start=0.0, note_end=0.5)
        assert lo[0][0] == [0x90, 0, 0]
        assert lo[1][0] == [0x80, 0, 0]

        hi = make_midi_events(pitch=127, velocity=127, note_start=0.1, note_end=2.0)
        assert hi == (([0x90, 127, 127], 0.1), ([0x80, 127, 127], 2.0))


class TestSetParams:
    """Verify ``set_params`` writes ``raw_value`` on each parameter."""

    def test_calls_raw_value(self):
        plugin = _mock_plugin()
        for name in ["param_a", "param_b", "param_c"]:
            plugin.parameters[name] = MagicMock()

        params = {"param_a": 0.1, "param_b": 0.5, "param_c": 0.9}
        set_params(plugin, params)

        for name, val in params.items():
            assert plugin.parameters[name].raw_value == val


class TestCallShowEditorFor:
    """Verify ``_call_show_editor_for`` blocks for the requested warmup time."""

    def test_blocks_for_duration(self):
        plugin = MagicMock()
        plugin.show_editor = lambda event: event.wait()

        warmup_s = 0.2
        elapsed, _ = _elapsed(lambda: _call_show_editor_for(plugin, warmup_s=warmup_s))

        assert warmup_s * 0.9 <= elapsed < warmup_s + 1.0


class TestRenderParams:
    """Verify ``render_params`` calls preset loading and flushes in the right order."""

    _CFG = RenderConfig(
        midi_note=60,
        velocity=100,
        note_start_and_end=(0.0, 1.0),
        signal_duration_seconds=2.0,
        sample_rate=44100.0,
        channels=2,
    )

    def _patched_render(self, plugin, preset_path=None):
        """Run ``render_params`` with load_preset, set_params and make_midi_events mocked."""
        with patch("src.data.vst.core.load_preset") as mock_load, \
             patch("src.data.vst.core.set_params"), \
             patch("src.data.vst.core.make_midi_events", return_value=()):
            render_params(plugin, {}, preset_path=preset_path, **self._CFG)
        return mock_load

    def test_calls_load_preset_when_given(self):
        """``load_preset`` is called when a preset path is provided."""
        plugin = _mock_plugin()
        mock_load = self._patched_render(plugin, preset_path="/fake.vstpreset")
        mock_load.assert_called_once_with(plugin, "/fake.vstpreset")

    def test_skips_load_preset_when_none(self):
        """``load_preset`` is *not* called when ``preset_path`` is None."""
        plugin = _mock_plugin()
        mock_load = self._patched_render(plugin, preset_path=None)
        mock_load.assert_not_called()

    def test_flush_sequence(self):
        """Process/reset calls follow: flush→reset→flush→reset→render→flush→reset."""
        plugin = _mock_plugin()
        midi_events = (([0x90, 60, 100], 0.0), ([0x80, 60, 100], 1.0))

        with patch("src.data.vst.core.set_params"), \
             patch("src.data.vst.core.make_midi_events", return_value=midi_events):
            render_params(plugin, {"p": 0.5}, preset_path=None, **self._CFG)

        calls = [c[0] for c in plugin.method_calls if c[0] in ("process", "reset")]
        assert calls == ["process", "reset", "process", "reset", "process", "process", "reset"]


# ===========================================================================
# Integration tests — correctness
# ===========================================================================


@requires_plugin
@pytest.mark.slow
@pytest.mark.vst
class TestPluginCorrectness:
    """End-to-end correctness of preset loading, parameter setting, and rendering."""

    def test_plugin_loads_successfully(self, surge_plugin):
        """Loaded plugin exposes a non-empty parameter dict."""
        assert hasattr(surge_plugin, "parameters")
        assert len(surge_plugin.parameters) > 0

    def test_preset_load_changes_audio(self, surge_plugin):
        """Different presets produce audibly different output."""
        audio_base = _render(surge_plugin, {}, PRESET_BASE)
        audio_simple = _render(surge_plugin, {}, PRESET_SIMPLE)
        _assert_different(audio_base, audio_simple, label="Different presets")

    def test_parameter_readback_after_set(self, surge_plugin, param_spec_simple):
        """``set_params`` values can be read back within 1e-3 tolerance."""
        surge_plugin.load_preset(PRESET_SIMPLE)
        params = _sample_params(param_spec_simple, surge_plugin)
        set_params(surge_plugin, params)

        for name, expected in params.items():
            actual = surge_plugin.parameters[name].raw_value
            assert abs(actual - expected) < 1e-3, f"{name}: {expected} != {actual}"

    def test_render_determinism(self, surge_plugin, param_spec_simple):
        """Same preset+params rendered twice are spectrally similar (>0.90).

        Threshold is relaxed because Surge XT has free-running LFOs.
        """
        params = _default_params(param_spec_simple, surge_plugin)
        a1 = _render(surge_plugin, params)
        a2 = _render(surge_plugin, params)
        _assert_similar(a1, a2, min_sim=0.90, label="Render determinism")

    def test_different_params_different_audio(self, surge_plugin, param_spec_simple):
        """Min-range vs max-range params produce spectrally different audio."""
        lo, hi = _contrasting_params(param_spec_simple, surge_plugin)
        a_lo = _render(surge_plugin, lo)
        a_hi = _render(surge_plugin, hi)
        _assert_different(a_lo, a_hi, max_sim=0.95, label="Contrasting params")

    def test_render_output_shape_and_dtype(self, surge_plugin, param_spec_simple):
        """Output array is ``(channels, sr * duration)`` with a float dtype."""
        params = _sample_params(param_spec_simple, surge_plugin)
        audio = _render(surge_plugin, params)

        expected_samples = int(RENDER_CFG["sample_rate"] * RENDER_CFG["signal_duration_seconds"])
        assert audio.shape == (RENDER_CFG["channels"], expected_samples)
        assert np.issubdtype(audio.dtype, np.floating)

    def test_render_produces_nonsilent_audio(self, surge_plugin, param_spec_simple):
        """Default (mid-range) params produce audio with RMS > 1e-4."""
        params = _default_params(param_spec_simple, surge_plugin)
        audio = _render(surge_plugin, params)

        rms = np.sqrt(np.mean(audio**2))
        logger.info(f"Rendered audio RMS: {rms:.6f}")
        assert rms > 1e-4, f"Audio appears silent: RMS={rms}"


# ===========================================================================
# Integration tests — raw_state workaround (pedalboard #394)
# ===========================================================================


@requires_plugin
@pytest.mark.slow
@pytest.mark.vst
class TestRawStateWorkaround:
    """Validate the ``raw_state = raw_state`` workaround as an alternative to editor warmup."""

    def test_raw_state_preset_loads_correctly(self, surge_plugin_raw_state, param_spec_simple):
        """After the workaround, set params are finite and in [0, 1]."""
        surge_plugin_raw_state.load_preset(PRESET_SIMPLE)
        surge_plugin_raw_state.raw_state = surge_plugin_raw_state.raw_state

        params = _sample_params(param_spec_simple, surge_plugin_raw_state)
        set_params(surge_plugin_raw_state, params)

        for name in params:
            actual = surge_plugin_raw_state.parameters[name].raw_value
            assert np.isfinite(actual), f"{name} is not finite: {actual}"
            assert 0.0 <= actual <= 1.0, f"{name} out of range: {actual}"

    def test_raw_state_render_matches_editor_warmup(
        self, surge_plugin, surge_plugin_raw_state, param_spec_simple
    ):
        """raw_state plugin renders match editor-warmed plugin (MFCC sim > 0.99)."""
        params = _sample_params(param_spec_simple, surge_plugin)
        audio_editor = _render(surge_plugin, params)
        audio_raw = _render(surge_plugin_raw_state, params)
        _assert_similar(audio_editor, audio_raw, min_sim=0.99, label="Editor vs raw_state")

    def test_raw_state_deterministic(self, surge_plugin_raw_state, param_spec_simple):
        """Two identical renders via raw_state plugin are spectrally similar (>0.90)."""
        params = _default_params(param_spec_simple, surge_plugin_raw_state)
        a1 = _render(surge_plugin_raw_state, params)
        a2 = _render(surge_plugin_raw_state, params)
        _assert_similar(a1, a2, min_sim=0.90, label="raw_state determinism")


# ===========================================================================
# Integration tests — throughput
# ===========================================================================


@requires_plugin
@pytest.mark.slow
@pytest.mark.vst
class TestThroughput:
    """Benchmark init and per-render times for both warmup approaches."""

    def test_throughput_init_editor_warmup(self):
        """``load_plugin`` (with editor warmup) completes in < 5 s."""
        elapsed, _ = _elapsed(lambda: load_plugin(PLUGIN_PATH))
        logger.info(f"Editor warmup init time: {elapsed:.3f}s")
        assert elapsed < 5.0

    def test_throughput_init_raw_state(self):
        """``VST3Plugin`` + raw_state trick completes in < 5 s."""
        from pedalboard import VST3Plugin

        def init():
            p = VST3Plugin(PLUGIN_PATH)
            p.raw_state = p.raw_state

        elapsed, _ = _elapsed(init)
        logger.info(f"raw_state init time: {elapsed:.3f}s")
        assert elapsed < 5.0

    def test_throughput_per_render(self, surge_plugin, param_spec_simple):
        """Mean render time (20 runs, first 2 discarded) is < 3 s for 2 s audio."""
        params = _sample_params(param_spec_simple, surge_plugin)
        n_warmup = 2

        times = [_elapsed(lambda: _render(surge_plugin, params))[0] for _ in range(20)]

        measured = times[n_warmup:]
        mean_t, std_t = np.mean(measured), np.std(measured)
        logger.info(f"Per-render ({len(measured)} runs): {mean_t:.3f}s +/- {std_t:.3f}s")
        assert mean_t < 3.0

    def test_throughput_end_to_end_comparison(self, param_spec_simple):
        """Init + 10 renders: raw_state approach is no slower than editor warmup."""
        from pedalboard import VST3Plugin

        n_renders = 10

        t0 = time.perf_counter()
        plugin_editor = load_plugin(PLUGIN_PATH)
        params = _sample_params(param_spec_simple, plugin_editor)
        for _ in range(n_renders):
            _render(plugin_editor, params)
        editor_total = time.perf_counter() - t0

        t0 = time.perf_counter()
        plugin_raw = VST3Plugin(PLUGIN_PATH)
        plugin_raw.raw_state = plugin_raw.raw_state
        for _ in range(n_renders):
            _render(plugin_raw, params)
        raw_total = time.perf_counter() - t0

        logger.info(
            f"End-to-end ({n_renders} renders) — "
            f"editor: {editor_total:.3f}s, raw_state: {raw_total:.3f}s"
        )
        assert raw_total <= editor_total * 1.1
