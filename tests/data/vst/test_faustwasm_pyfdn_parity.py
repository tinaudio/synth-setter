"""The parametric Faust FDN renders pyFDN householder rows on the FaustWasm host."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_module

from synth_setter.data.pyfdn_param_spec import (
    PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC,
    householder_feedback_matrix,
)
from synth_setter.data.vst.faust_param_spec import resolve_faust_param_spec
from synth_setter.data.vst.faustwasm_renderer import FaustWasmRenderer
from synth_setter.data.vst.faustwasm_contract import (
    faustwasm_parameter_contract,
    flatten_canonical_patch,
)
from synth_setter.data.vst.param_spec import require_note_params
from synth_setter.data.vst.param_spec_registry import resolve_param_spec
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.pipeline.schemas.spec import RenderConfig
from synth_setter.renderer_factory import make_audio_renderer
from synth_setter.resources import faustwasm_dir
from synth_setter.synth_spec import SYNTHS, SynthName
from synth_setter.tools.export_faustwasm import main as export_main

_FDN = ParamSpecName("faust_fdn_n8_mono_householder")
_PYFDN = ParamSpecName("pyfdn_n8_mono_householder")
_NODE_UNAVAILABLE = shutil.which("node") is None
_RENDER_SAMPLES = 176_400
# DawDreamer float32 renders of the same source sit ~2e-7 from pyFDN's float64 response.
_PARITY_ATOL = 1e-5
_BROWSER_EXPORTER = Path(__file__).parents[3] / "scripts" / "faustwasm" / "browser"
_BROWSER_DEPS_UNAVAILABLE = not (_BROWSER_EXPORTER / "node_modules" / "@grame" / "faustwasm").is_dir()


def _backend_version() -> str:
    """Return the authored FaustWasm package pin.

    :returns: Configured backend version.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        return str(compose(config_name="render/faustwasm").render.backend_version)


def _pyfdn_config() -> RenderConfig:
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        render = compose(config_name="render/pyfdn").render
    return RenderConfig(synth=SYNTHS[SynthName(_PYFDN)], **render)


def _faustwasm_config() -> RenderConfig:
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        render = compose(config_name="render/faustwasm_fdn").render
    return RenderConfig(synth=SYNTHS[SynthName(_FDN)], **render)


def test_fdn_synth_shares_the_pyfdn_householder_param_spec() -> None:
    """The Faust identity decodes rows exactly as pyFDN does, from its own spec instance."""
    faust_spec = resolve_param_spec(_FDN)
    row = np.linspace(0.05, 0.95, faust_spec.encoded_width)

    faust_params, faust_notes = faust_spec.decode(row)
    pyfdn_params, pyfdn_notes = PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC.decode(row)

    assert faust_spec is not PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC
    assert faust_spec.encoded_width == 27
    assert faust_notes == pyfdn_notes
    assert faust_params.keys() == pyfdn_params.keys()
    for name in faust_params:
        np.testing.assert_array_equal(faust_params[name], pyfdn_params[name])
    assert SYNTHS[SynthName(_FDN)].format == "faust"


def test_fdn_synth_resolves_a_fresh_spec_per_call() -> None:
    """Mutating one resolved spec cannot leak into the next resolution."""
    changed = resolve_faust_param_spec(_FDN)
    changed.synth_params[0].name = "changed"

    assert resolve_faust_param_spec(_FDN).synth_params[0].name == "delays"


def test_fdn_identity_rejects_hosts_that_cannot_flatten_array_fields() -> None:
    """DawDreamer and Faust C++ address scalar sliders only, so the FDN is FaustWasm-only."""
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        render = compose(config_name="render/faust").render

    with pytest.raises(ValueError, match="array-valued"):
        RenderConfig(synth=SYNTHS[SynthName(_FDN)], **{**render, "channels": 1})


def test_fdn_contract_flattens_every_array_element_to_one_slider() -> None:
    """Twenty-seven canonical coordinates map onto twenty-seven compiled sliders."""
    contract = faustwasm_parameter_contract(_FDN)

    assert [item.canonical_address for item in contract][:9] == [
        *(f"delays.{index}" for index in range(8)),
        "input_matrix.0.0",
    ]
    assert [item.wasm_address for item in contract][-3:] == [
        "/fdnHouseholder/direct",
        "/fdnHouseholder/rt_dc_seconds",
        "/fdnHouseholder/rt_nyquist_seconds",
    ]
    assert len(contract) == 27
    assert {item.kind for item in contract} == {"continuous"}
    assert contract[0].minimum == 400.0 and contract[0].maximum == 1200.0


def test_flatten_canonical_patch_drops_the_fixed_householder_feedback() -> None:
    """A decoded pyFDN row flattens to scalar sliders once its derived matrix is verified."""
    synth_params, _ = PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC.sample(np.random.default_rng(0))

    flat = flatten_canonical_patch(_FDN, synth_params)

    assert len(flat) == 27
    assert flat["delays.3"] == float(np.asarray(synth_params["delays"])[3])
    assert flat["output_matrix.0.7"] == float(np.asarray(synth_params["output_matrix"])[0, 7])
    assert flat["post_delay.rt_dc_seconds"] == synth_params["post_delay.rt_dc_seconds"]


def test_flatten_canonical_patch_rejects_a_feedback_matrix_the_source_cannot_render() -> None:
    """The DSP hardcodes Householder(ones); any other feedback matrix is an error."""
    synth_params, _ = PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC.sample(np.random.default_rng(0))
    synth_params["feedback_matrix"] = householder_feedback_matrix(np.arange(1.0, 9.0))

    with pytest.raises(ValueError, match="feedback_matrix"):
        flatten_canonical_patch(_FDN, synth_params)


def test_flatten_canonical_patch_rejects_a_fractional_delay() -> None:
    """A fractional delay would be truncated by the DSP's integer slider, so it fails loudly."""
    synth_params, _ = PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC.sample(np.random.default_rng(0))
    synth_params["delays"] = np.asarray(synth_params["delays"], dtype=np.float64) + 0.7

    with pytest.raises(ValueError, match="delays must contain only integer values"):
        flatten_canonical_patch(_FDN, synth_params)


def test_flatten_canonical_patch_rejects_unknown_fields() -> None:
    """Fields outside the spec and its fixed derived values never reach the compiler."""
    synth_params, _ = PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC.sample(np.random.default_rng(0))
    synth_params["post_matrix"] = 1.0

    with pytest.raises(KeyError, match="post_matrix"):
        flatten_canonical_patch(_FDN, synth_params)


@pytest.fixture(scope="module")
def faustwasm_fdn_renderer() -> FaustWasmRenderer:
    """Compile the FDN artifact once for every parity row.

    :returns: Production FaustWasm renderer for the FDN identity.
    """
    if _NODE_UNAVAILABLE:
        pytest.skip("Node.js is unavailable")
    renderer = make_audio_renderer(_faustwasm_config())
    assert isinstance(renderer, FaustWasmRenderer)
    return renderer


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_faustwasm_fdn_matches_pyfdn_impulse_response(
    faustwasm_fdn_renderer: FaustWasmRenderer,
    seed: int,
) -> None:
    """A sampled pyFDN row renders the same four-second response on both hosts.

    :param faustwasm_fdn_renderer: Module-scoped compiled FaustWasm renderer.
    :param seed: Row sampling seed.
    """
    spec = resolve_param_spec(_PYFDN)
    synth_params, _ = spec.sample(np.random.default_rng(seed))
    reference = make_audio_renderer(_pyfdn_config()).render(synth_params, 0, 0, (0.0, 0.0))

    audio = faustwasm_fdn_renderer.render(synth_params, 60, 100, (0.0, 4.0))

    assert audio.shape == reference.shape == (1, _RENDER_SAMPLES)
    assert float(np.max(np.abs(reference))) > 0.1
    np.testing.assert_allclose(audio, reference, rtol=0.0, atol=_PARITY_ATOL)


@pytest.mark.slow
def test_faustwasm_fdn_renders_the_pyfdn_note_stub_window(
    faustwasm_fdn_renderer: FaustWasmRenderer,
) -> None:
    """The zero-length note window pyFDN rows carry does not gate a mono impulse render.

    :param faustwasm_fdn_renderer: Module-scoped compiled FaustWasm renderer.
    """
    synth_params, note_params = resolve_param_spec(_FDN).sample(np.random.default_rng(7))
    notes = require_note_params(note_params)

    audio = faustwasm_fdn_renderer.render(
        synth_params, notes["pitch"], 0, notes["note_start_and_end"]
    )

    assert notes["note_start_and_end"] == (0.0, 0.0)
    assert audio.shape == (1, _RENDER_SAMPLES)
    assert float(np.max(np.abs(audio))) > 0.1


@pytest.mark.slow
@pytest.mark.skipif(_NODE_UNAVAILABLE, reason="Node.js is unavailable")
@pytest.mark.skipif(
    _BROWSER_DEPS_UNAVAILABLE, reason="run npm ci --prefix scripts/faustwasm/browser"
)
def test_fdn_artifact_exports_and_bundles_for_the_browser(tmp_path: Path) -> None:
    """The public export CLI and browser bundler accept the FDN identity.

    :param tmp_path: Isolated artifact and site destinations.
    """
    artifact = tmp_path / "artifact"
    export_main(["--synth", _FDN, "--output", str(artifact)])
    manifest = json.loads((artifact / "manifest.json").read_text())
    assert manifest["mode"] == "mono" and manifest["outputs"] == 1
    assert len(manifest["parameters"]) == 27
    assert manifest["faustwasmVersion"] == _backend_version()

    site = tmp_path / "site"
    subprocess.run(  # noqa: S603
        [
            "node",
            str(_BROWSER_EXPORTER / "export-browser.mjs"),
            "--artifact",
            str(artifact),
            "--output",
            str(site),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert (site / "manifest.json").is_file()
    assert (site / "index.html").is_file()
    assert (faustwasm_dir() / "runtime.mjs").is_file()
