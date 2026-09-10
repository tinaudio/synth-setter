"""Boundary contracts for the production FaustWasm renderer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from synth_setter.data.vst.faust_param_spec import resolve_faust_param_spec
from synth_setter.data.vst.faustwasm_renderer import (
    FAUSTWASM_VERSION,
    FaustWasmRenderer,
    _quantize_note_window,
)
from synth_setter.data.vst.param_spec import CategoricalParameter, ContinuousParameter
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.synth_spec import SYNTHS, SynthName

_ROOT = Path(__file__).parents[3]
_NODE_MODULE = _ROOT / "node_modules/@grame/faustwasm/package.json"


def _patch(identity: str) -> dict[str, float]:
    """Build a complete in-domain patch from the production parameter registry.

    :param identity: Registered Faust synth identity.
    :returns: Complete canonical scalar patch.
    :raises TypeError: The registry contains an unsupported parameter kind.
    """
    patch: dict[str, float] = {}
    for parameter in resolve_faust_param_spec(ParamSpecName(identity)).synth_params:
        if isinstance(parameter, ContinuousParameter):
            patch[parameter.name] = (parameter.min + parameter.max) / 2.0
        elif isinstance(parameter, CategoricalParameter):
            patch[parameter.name] = float(parameter.raw_values[0])
        else:
            raise TypeError(type(parameter).__name__)
    return patch


def _renderer(identity: str, *, sample_rate: float, duration: float, channels: int) -> FaustWasmRenderer:
    """Build a production renderer with explicit boundary dimensions.

    :param identity: Registered Faust synth identity.
    :param sample_rate: Render sample rate in Hz.
    :param duration: Fixed output duration in seconds.
    :param channels: Expected native output count.
    :returns: Compiled production renderer.
    """
    synth = SYNTHS[SynthName(identity)]
    return FaustWasmRenderer(
        plugin_path=synth.plugin_path,
        plugin_state_path=synth.plugin_state_path,
        sample_rate=sample_rate,
        channels=channels,
        signal_duration_seconds=duration,
        param_spec_name=synth.param_spec_name,
        source_sha256=synth.source_sha256 or "",
        backend_version=FAUSTWASM_VERSION,
        block_size=1,
    )


def test_quantize_note_window_fractional_duration_clamps_to_output_frames() -> None:
    """A fractional final frame never extends beyond the fixed output."""
    assert _quantize_note_window(
        (0.0, 0.15),
        sample_rate=10,
        frames=1,
        signal_duration_seconds=0.15,
    ) == (0, 1)


def test_quantize_note_window_subsample_interval_remains_representable() -> None:
    """A positive sub-sample interval covers every intersecting output frame."""
    sample_rate = 44_100

    assert _quantize_note_window(
        (1.5 / sample_rate, 2.5 / sample_rate),
        sample_rate=sample_rate,
        frames=4,
        signal_duration_seconds=4 / sample_rate,
    ) == (1, 3)


@pytest.mark.parametrize(
    "window",
    [
        (float("nan"), 0.1),
        (0.0, float("inf")),
        (-0.1, 0.1),
        (0.1, 0.1),
        (0.2, 0.1),
        (0.0, 0.21),
        (0.11, 0.15),
    ],
)
def test_quantize_note_window_invalid_or_unrepresentable_window_rejected(
    window: tuple[float, float],
) -> None:
    """Malformed, out-of-range, and discarded-tail windows fail before Node.

    :param window: Invalid note window under test.
    """
    with pytest.raises(ValueError, match="note times"):
        _quantize_note_window(
            window,
            sample_rate=10,
            frames=2 if window[-1] > 0.15 else 1,
            signal_duration_seconds=0.2 if window[-1] > 0.15 else 0.15,
        )


@pytest.mark.parametrize("block_size", [0, -1, 1.5, True])
def test_faustwasm_renderer_invalid_block_size_fails_before_compile(block_size: object) -> None:
    """Invalid loop increments fail before starting the compiler.

    :param block_size: Non-positive or non-integral block size under test.
    """
    synth = SYNTHS[SynthName("faust_filter_osc")]

    with pytest.raises(ValueError, match="block_size must be a positive integer"):
        FaustWasmRenderer(
            plugin_path=synth.plugin_path,
            plugin_state_path=synth.plugin_state_path,
            sample_rate=44_100,
            channels=1,
            signal_duration_seconds=0.1,
            param_spec_name=synth.param_spec_name,
            source_sha256=synth.source_sha256 or "",
            backend_version=FAUSTWASM_VERSION,
            block_size=block_size,  # type: ignore[arg-type]
        )


@pytest.mark.skipif(not _NODE_MODULE.is_file(), reason="run `npm ci` to install @grame/faustwasm")
def test_faustwasm_real_render_clamps_fractional_duration_boundary() -> None:
    """The real runtime renders a final fractional duration into fixed frames."""
    renderer = _renderer("faust_filter_osc", sample_rate=10, duration=0.15, channels=1)

    audio = renderer.render(_patch("faust_filter_osc"), 60, 100, (0.0, 0.15))

    assert audio.shape == (1, 1)
    assert audio.dtype == np.float32
    assert np.isfinite(audio).all()


@pytest.mark.skipif(not _NODE_MODULE.is_file(), reason="run `npm ci` to install @grame/faustwasm")
@pytest.mark.parametrize(
    ("identity", "address"),
    [
        ("faust_bubble", "/bubble/drop"),
        ("faust_church_organ", "/churchOrgan/gate"),
    ],
)
def test_faustwasm_discrete_patch_rejects_fractional_value(
    identity: str,
    address: str,
) -> None:
    """Real button controls reject values between registered states.

    :param identity: Registered synth with a canonical button.
    :param address: Canonical button address under test.
    """
    renderer = _renderer(identity, sample_rate=44_100, duration=0.01, channels=2)
    patch = _patch(identity)
    patch[address] = 0.5

    with pytest.raises(ValueError, match="outside discrete native domain"):
        renderer.render(patch, 60, 100, (0.0, 0.005))


@pytest.mark.skipif(not _NODE_MODULE.is_file(), reason="run `npm ci` to install @grame/faustwasm")
@pytest.mark.parametrize("raw_value", [0.0, 1.0])
def test_faustwasm_discrete_patch_accepts_canonical_endpoints(raw_value: float) -> None:
    """Every registered button state reaches the real runtime.

    :param raw_value: Exact canonical button state under test.
    """
    renderer = _renderer("faust_bubble", sample_rate=44_100, duration=0.01, channels=2)
    patch = _patch("faust_bubble")
    patch["/bubble/drop"] = raw_value

    audio = renderer.render(patch, 60, 100, (0.0, 0.005))

    assert audio.shape == (2, 441)
    assert np.isfinite(audio).all()
