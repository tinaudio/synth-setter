"""Boundary contracts for the production FaustWasm renderer."""

from __future__ import annotations

import shutil

import numpy as np
import pytest
from hydra import compose, initialize_config_module

from synth_setter.data.vst.faust_param_spec import resolve_faust_param_spec
from synth_setter.data.vst.faustwasm_renderer import FaustWasmRenderer, _quantize_note_window
from synth_setter.data.vst.param_spec import CategoricalParameter, ContinuousParameter
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.synth_spec import SYNTHS, SynthName

_NODE_UNAVAILABLE = shutil.which("node") is None


def _configured_backend_version() -> str:
    """Return the authored FaustWasm package pin.

    :returns: Configured backend version.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        return str(compose(config_name="render/faustwasm").render.backend_version)


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


def _renderer(
    identity: str,
    *,
    sample_rate: float,
    duration: float,
    channels: int,
    plugin_path: str | None = None,
) -> FaustWasmRenderer:
    """Build a production renderer with explicit boundary dimensions.

    :param identity: Registered Faust synth identity.
    :param sample_rate: Render sample rate in Hz.
    :param duration: Fixed output duration in seconds.
    :param channels: Expected native output count.
    :param plugin_path: Source reference override; the registered value is used when omitted.
    :returns: Compiled production renderer.
    """
    synth = SYNTHS[SynthName(identity)]
    return FaustWasmRenderer(
        plugin_path=synth.plugin_path if plugin_path is None else plugin_path,
        plugin_state_path=synth.plugin_state_path,
        sample_rate=sample_rate,
        channels=channels,
        signal_duration_seconds=duration,
        param_spec_name=synth.param_spec_name,
        source_sha256=synth.source_sha256 or "",
        backend_version=_configured_backend_version(),
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
    ("window", "frames", "duration"),
    [
        ((float("nan"), 0.1), 1, 0.15),
        ((0.0, float("inf")), 1, 0.15),
        ((-0.1, 0.1), 1, 0.15),
        ((0.1, 0.1), 1, 0.15),
        ((0.2, 0.1), 1, 0.15),
        ((0.0, 0.21), 2, 0.2),
        ((0.11, 0.15), 1, 0.15),
    ],
)
def test_quantize_note_window_invalid_or_unrepresentable_window_rejected(
    window: tuple[float, float],
    frames: int,
    duration: float,
) -> None:
    """Malformed, out-of-range, and discarded-tail windows fail before Node.

    :param window: Invalid note window under test.
    :param frames: Fixed output frame count.
    :param duration: Configured output duration in seconds.
    """
    with pytest.raises(ValueError, match="note times"):
        _quantize_note_window(
            window,
            sample_rate=10,
            frames=frames,
            signal_duration_seconds=duration,
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
            backend_version=_configured_backend_version(),
            block_size=block_size,  # type: ignore[arg-type]
        )


def test_faustwasm_renderer_rejects_registered_source_channel_mismatch() -> None:
    """Direct construction rejects geometry before compiling the source."""
    with pytest.raises(ValueError, match="FaustWasm source requires channels=1"):
        _renderer("faust_filter_osc", sample_rate=44_100, duration=4.0, channels=2)


@pytest.mark.parametrize("channels", [1.0, True])
def test_faustwasm_renderer_rejects_non_integer_channels(channels: object) -> None:
    """Direct construction rejects non-integer geometry before compilation.

    :param channels: Value equal to one without the required integer type.
    """
    with pytest.raises(ValueError, match="channels must be a positive integer"):
        _renderer(
            "faust_filter_osc",
            sample_rate=44_100,
            duration=4.0,
            channels=channels,  # type: ignore[arg-type]
        )


@pytest.mark.skipif(_NODE_UNAVAILABLE, reason="Node.js is unavailable")
def test_faustwasm_registry_reference_renders_real_source() -> None:
    """A canonical registry URI selects source consumed by the real runtime."""
    renderer = _renderer(
        "faust_filter_osc",
        sample_rate=44_100,
        duration=0.01,
        channels=1,
        plugin_path="registry://faust/faust_filter_osc",
    )

    audio = renderer.render(_patch("faust_filter_osc"), 60, 100, (0.0, 0.005))

    assert audio.shape == (1, 441)
    assert np.isfinite(audio).all()


def test_faustwasm_malformed_registry_reference_fails_before_compile() -> None:
    """A malformed source URI fails at the renderer boundary."""
    with pytest.raises(ValueError, match="registry://faust/<registered-source-name>"):
        _renderer(
            "faust_filter_osc",
            sample_rate=44_100,
            duration=0.01,
            channels=1,
            plugin_path="registry://faust/faust_filter_osc/extra",
        )


def test_faustwasm_unknown_registry_reference_fails_before_compile() -> None:
    """An unknown source identity fails at the renderer boundary."""
    with pytest.raises(ValueError, match="Faust source 'not_registered' is not registered"):
        _renderer(
            "faust_filter_osc",
            sample_rate=44_100,
            duration=0.01,
            channels=1,
            plugin_path="registry://faust/not_registered",
        )


def test_faustwasm_mismatched_registry_reference_fails_before_compile() -> None:
    """A source URI cannot select a different parameter identity."""
    with pytest.raises(ValueError, match="selects 'faust_bubble'.*'faust_filter_osc'"):
        _renderer(
            "faust_filter_osc",
            sample_rate=44_100,
            duration=0.01,
            channels=1,
            plugin_path="registry://faust/faust_bubble",
        )


@pytest.mark.skipif(_NODE_UNAVAILABLE, reason="Node.js is unavailable")
def test_faustwasm_blank_plugin_path_retains_legacy_compatibility() -> None:
    """A blank source path retains the pathless legacy render contract."""
    renderer = _renderer(
        "faust_filter_osc",
        sample_rate=44_100,
        duration=0.01,
        channels=1,
        plugin_path="",
    )

    audio = renderer.render(_patch("faust_filter_osc"), 60, 100, (0.0, 0.005))

    assert audio.shape == (1, 441)
    assert np.isfinite(audio).all()


@pytest.mark.skipif(_NODE_UNAVAILABLE, reason="Node.js is unavailable")
def test_faustwasm_real_render_clamps_fractional_duration_boundary() -> None:
    """The real runtime renders a final fractional duration into fixed frames."""
    renderer = _renderer("faust_filter_osc", sample_rate=10, duration=0.15, channels=1)

    audio = renderer.render(_patch("faust_filter_osc"), 60, 100, (0.0, 0.15))

    assert audio.shape == (1, 1)
    assert audio.dtype == np.float32
    assert np.isfinite(audio).all()


@pytest.mark.skipif(_NODE_UNAVAILABLE, reason="Node.js is unavailable")
@pytest.mark.parametrize(
    ("identity", "address", "channels"),
    [
        ("faust_bubble", "/bubble/drop", 2),
        ("faust_church_organ", "/churchOrgan/gate", 2),
        ("faust_kronecker_fdn", "/kroneckerFDN/Kernel/r0", 1),
    ],
)
def test_faustwasm_discrete_patch_rejects_fractional_value(
    identity: str,
    address: str,
    channels: int,
) -> None:
    """Real button controls reject values between registered states.

    :param identity: Registered synth with a canonical button.
    :param address: Canonical button address under test.
    :param channels: Native output channel count.
    """
    renderer = _renderer(identity, sample_rate=44_100, duration=0.01, channels=channels)
    patch = _patch(identity)
    patch[address] = 0.5

    with pytest.raises(ValueError, match="outside discrete native domain"):
        renderer.render(patch, 60, 100, (0.0, 0.005))


@pytest.mark.skipif(_NODE_UNAVAILABLE, reason="Node.js is unavailable")
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
