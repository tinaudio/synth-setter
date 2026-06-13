"""End-to-end ``synth-setter-introspect-plugin`` run against the real Surge XT.

Drives the real CLI entrypoint (real plugin load, drafting, emission, preset
capture) and asserts on the produced artifacts — the real-dependency
counterpart to the fake-plugin CLI tests in ``test_introspect_cli.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from synth_setter.cli.introspect_plugin import main
from synth_setter.data.vst.core import load_plugin, load_preset
from synth_setter.data.vst.param_spec import (
    CategoricalParameter,
    ContinuousParameter,
    ParamSpec,
)
from tests._vst import PLUGIN_PATH
from tests.data.vst._introspect_fakes import assert_ruff_format_clean, exec_module

requires_vst = pytest.mark.requires_vst


@requires_vst
@pytest.mark.slow
@pytest.mark.usefixtures("seeded_rng")
def test_introspect_cli_surge_xt_emits_usable_spec_and_vstpreset(tmp_path: Path) -> None:
    """Real Surge XT introspection yields a usable spec and a loadable ``.vstpreset``.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    spec_path = tmp_path / "draft_param_spec.py"
    preset_path = tmp_path / "draft-base.vstpreset"

    result = CliRunner().invoke(
        main,
        [
            "--plugin-path",
            PLUGIN_PATH,
            "--spec-name",
            "draft",
            "--out-spec",
            str(spec_path),
            "--out-preset",
            str(preset_path),
            "--out-csv",
            str(tmp_path / "draft_params.csv"),
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    # The captured baseline must round-trip through the pipeline's real consumer.
    assert preset_path.read_bytes().startswith(b"VST3")
    load_preset(load_plugin(PLUGIN_PATH), str(preset_path))

    spec_text = spec_path.read_text()
    assert_ruff_format_clean(spec_text)

    spec = exec_module(spec_text)["DRAFT_PARAM_SPEC"]
    assert isinstance(spec, ParamSpec)

    # Surge XT's parameter sheet: the draft must classify its known surface.
    assert "a_amp_eg_attack" in spec.synth_param_names
    filter_type = next(p for p in spec.synth_params if p.name == "a_filter_1_type")
    assert isinstance(filter_type, CategoricalParameter)
    assert "LP 12 dB" in filter_type.values
    # Cardinality boundaries: the 43-label waveshaper set survives as a
    # categorical while the 128-note keytrack selector tips to continuous.
    waveshaper = next(p for p in spec.synth_params if p.name == "a_waveshaper_type")
    assert isinstance(waveshaper, CategoricalParameter)
    keytrack = next(p for p in spec.synth_params if p.name == "a_keytrack_root_key")
    assert isinstance(keytrack, ContinuousParameter)

    # The drafted spec is immediately usable by the sampling pipeline.
    synth_params, note_params = spec.sample()
    encoded = spec.encode(synth_params, note_params)
    assert len(encoded) == len(spec)
    assert all(0.0 <= v <= 1.0 for v in synth_params.values())
