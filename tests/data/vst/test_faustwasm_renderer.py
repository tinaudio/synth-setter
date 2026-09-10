"""FaustWasm artifact and production renderer contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from synth_setter.data.vst.faust_param_spec import resolve_faust_param_spec
from synth_setter.data.vst.faustwasm_contract import faustwasm_parameter_contract
from synth_setter.data.vst.core import extract_backend_version
from synth_setter.data.vst.faustwasm_renderer import FaustWasmRenderer
from synth_setter.data.vst.param_spec import CategoricalParameter, ContinuousParameter
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.pipeline.schemas.spec import RenderConfig
from synth_setter.renderer_factory import make_audio_renderer
from synth_setter.synth_spec import SYNTHS, SynthName

_FAUSTWASM_VERSION = "0.18.3"
_ROOT = Path(__file__).parents[3]
_NODE_MODULE = _ROOT / "node_modules/@grame/faustwasm/package.json"


def _config(identity: str = "faust_bright_organ", channels: int = 2) -> RenderConfig:
    return RenderConfig(
        synth=SYNTHS[SynthName(identity)],
        renderer_backend="faustwasm",
        backend_version=_FAUSTWASM_VERSION,
        render_contract_version=2,
        sample_rate=44_100,
        channels=channels,
        velocity=100,
        signal_duration_seconds=0.5,
        min_loudness=-100.0,
        samples_per_render_batch=1,
        samples_per_shard=1,
        plugin_reload_cadence="render",
        gui_toggle_cadence="never",
    )


def _midpoint_patch(identity: str) -> dict[str, float]:
    spec = resolve_faust_param_spec(ParamSpecName(identity))
    patch: dict[str, float] = {}
    for parameter in spec.synth_params:
        if isinstance(parameter, ContinuousParameter):
            patch[parameter.name] = (parameter.min + parameter.max) / 2.0
        elif isinstance(parameter, CategoricalParameter):
            patch[parameter.name] = float(parameter.raw_values[-1])
        else:
            raise TypeError(type(parameter).__name__)
    return patch


def test_faustwasm_contract_maps_host_specific_addresses_exactly() -> None:
    bright = faustwasm_parameter_contract(ParamSpecName("faust_bright_organ"))
    church = faustwasm_parameter_contract(ParamSpecName("faust_church_organ"))

    assert bright[0].canonical_address == "/Sequencer/DSP1/brightOrgan/Main/volume"
    assert bright[0].wasm_address == "/brightOrgan/Main/volume"
    assert bright[4].wasm_address == "/brightOrgan/Stops/Fifteenth_2-"
    assert church[0].canonical_address == "/churchOrgan/Zita_Light/Dry/Wet_Mix"
    assert church[0].wasm_address == "/churchOrgan/Zita_Light/Wet_Dry_Mix"


def test_faustwasm_contract_covers_every_canonical_parameter_once() -> None:
    for identity in ("faust_bright_organ", "faust_bubble", "faust_church_organ", "faust_filter_osc"):
        spec = resolve_faust_param_spec(ParamSpecName(identity))
        contract = faustwasm_parameter_contract(ParamSpecName(identity))
        assert [item.canonical_address for item in contract] == spec.synth_param_names


@pytest.mark.skipif(not _NODE_MODULE.is_file(), reason="run `npm ci` to install @grame/faustwasm")
def test_faustwasm_backend_version_reads_pinned_node_package() -> None:
    assert extract_backend_version("faustwasm") == "0.18.3"


def test_faustwasm_legacy_digest_projection_is_rejected() -> None:
    values = _config().model_dump()
    values["render_contract_version"] = 1
    with pytest.raises(ValueError, match="render_contract_version=1"):
        RenderConfig.model_validate(values)


@pytest.mark.skipif(not _NODE_MODULE.is_file(), reason="run `npm ci` to install @grame/faustwasm")
@pytest.mark.parametrize(
    ("identity", "channels"),
    [
        ("faust_bright_organ", 2),
        ("faust_bubble", 2),
        ("faust_church_organ", 2),
        ("faust_filter_osc", 1),
    ],
)
def test_faustwasm_factory_renders_real_source(identity: str, channels: int) -> None:
    renderer = make_audio_renderer(_config(identity, channels))
    params = _midpoint_patch(identity)
    if identity == "faust_bubble":
        params["/bubble/drop"] = 1.0
    if identity == "faust_church_organ":
        params["/churchOrgan/gate"] = 1.0
        params["/churchOrgan/gain"] = 0.2

    audio = renderer.render(params, 60, 100, (0.05, 0.3))

    assert isinstance(renderer, FaustWasmRenderer)
    assert audio.shape == (channels, 22_050)
    assert audio.dtype == np.float32
    assert np.isfinite(audio).all()
    assert float(np.max(np.abs(audio))) > 1e-4


@pytest.mark.skipif(not _NODE_MODULE.is_file(), reason="run `npm ci` to install @grame/faustwasm")
def test_faustwasm_note_boundary_and_state_isolation() -> None:
    renderer = make_audio_renderer(_config())
    params = _midpoint_patch("faust_bright_organ")

    first = renderer.render(params, 60, 100, (0.1, 0.25))
    second = renderer.render(params, 60, 100, (0.1, 0.25))

    assert np.max(np.abs(first[:, :4_410])) == 0.0
    assert np.array_equal(first, second)


@pytest.mark.skipif(not _NODE_MODULE.is_file(), reason="run `npm ci` to install @grame/faustwasm")
def test_faustwasm_incomplete_patch_is_rejected() -> None:
    renderer = make_audio_renderer(_config())
    params = _midpoint_patch("faust_bright_organ")
    params.pop(next(iter(params)))

    with pytest.raises(ValueError, match="missing canonical parameter"):
        renderer.render(params, 60, 100, (0.05, 0.3))


@pytest.mark.skipif(not _NODE_MODULE.is_file(), reason="run `npm ci` to install @grame/faustwasm")
def test_faustwasm_out_of_domain_patch_is_rejected() -> None:
    renderer = make_audio_renderer(_config())
    params = _midpoint_patch("faust_bright_organ")
    params["/Sequencer/DSP1/brightOrgan/Main/volume"] = 2.0

    with pytest.raises(ValueError, match="outside native domain"):
        renderer.render(params, 60, 100, (0.05, 0.3))


@pytest.mark.skipif(not _NODE_MODULE.is_file(), reason="run `npm ci` to install @grame/faustwasm")
def test_faustwasm_and_dawdreamer_share_bright_organ_invariants() -> None:
    params = _midpoint_patch("faust_bright_organ")
    wasm = make_audio_renderer(_config()).render(params, 60, 100, (0.1, 0.25))
    daw_config = _config().model_copy(
        update={"renderer_backend": "dawdreamer", "backend_version": "0.8.3"}
    )
    daw = make_audio_renderer(daw_config).render(params, 60, 100, (0.1, 0.25))

    wasm_rms = float(np.sqrt(np.mean(np.square(wasm[:, 4_410:11_025]))))
    daw_rms = float(np.sqrt(np.mean(np.square(daw[:, 4_410:11_025]))))
    assert np.max(np.abs(wasm[:, :4_410])) == 0.0
    assert np.max(np.abs(daw[:, :4_410])) == 0.0
    assert 0.5 < wasm_rms / daw_rms < 2.0


@pytest.mark.skipif(not _NODE_MODULE.is_file(), reason="run `npm ci` to install @grame/faustwasm")
def test_faustwasm_canonical_volume_has_causal_effect() -> None:
    quiet = _midpoint_patch("faust_bright_organ")
    loud = dict(quiet)
    quiet["/Sequencer/DSP1/brightOrgan/Main/volume"] = 0.0
    loud["/Sequencer/DSP1/brightOrgan/Main/volume"] = 1.0
    renderer = make_audio_renderer(_config())

    quiet_audio = renderer.render(quiet, 60, 100, (0.05, 0.3))
    loud_audio = renderer.render(loud, 60, 100, (0.05, 0.3))

    assert float(np.max(np.abs(quiet_audio))) < 1e-4
    assert float(np.max(np.abs(loud_audio))) > 1e-4
