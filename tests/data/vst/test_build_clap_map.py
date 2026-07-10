"""Tests for the pure map-assembly core of ``tools/build_clap_map.py``."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from synth_setter.data.vst.clap_introspect import (
    SURGE_XT_CLAP_PATH,
    ClapParamInfo,
    ClapPluginInfo,
)
from synth_setter.data.vst.param_spec import (
    CategoricalParameter,
    ContinuousParameter,
    ParamSpec,
)
from synth_setter.tools.build_clap_map import (
    _read_display_names,
    build_format_map,
    init_order_errors,
)
from synth_setter.tools.build_clap_map import main as build_clap_map_main
from tests.data.vst._clap import SURGE_XT_CLAP_PARAM_COUNT


def _info(**overrides: object) -> ClapParamInfo:
    """Build a ClapParamInfo with test defaults, overridable per test.

    :param **overrides: Field overrides merged over the defaults.
    :returns: The constructed info.
    """
    fields = {
        "id": 1000,
        "name": "A Amp EG Attack",
        "module": "/A Envelopes/",
        "min_value": 0.0,
        "max_value": 1.0,
        "default_value": 0.0,
        "flags": 32,
        "is_stepped": False,
    }
    fields.update(overrides)
    return ClapParamInfo(**fields)


def _plugin_info(params: list[ClapParamInfo]) -> ClapPluginInfo:
    """Wrap params in a Surge-shaped ClapPluginInfo.

    :param params: Dump entries in enumeration order.
    :returns: The dump object.
    """
    return ClapPluginInfo(
        plugin_id="org.surge-synth-team.surge-xt",
        plugin_name="Surge XT",
        vendor="Surge Synth Team",
        version="1.3.4",
        clap_version="1.2.2",
        params=params,
    )


class TestBuildFormatMap:
    """Pure map-assembly behavior of build_format_map."""

    def test_maps_every_spec_param_through_its_preset_index(self):
        """Each spec param resolves through its preset index to the right CLAP entry."""
        clap_info = _plugin_info(
            [
                _info(name="A Amp EG Attack", id=11),
                _info(name="A Amp EG Envelope Mode", id=22),
            ]
        )
        spec = ParamSpec(
            [
                ContinuousParameter(name="a_amp_eg_attack", max=0.77),
                CategoricalParameter(
                    name="a_amp_eg_envelope_mode",
                    values=["Digital", "Analog"],
                    raw_values=[0.25, 0.75],
                    encoding="onehot",
                ),
            ],
            [],
        )
        indices = {"a_amp_eg_attack": 0, "a_amp_eg_envelope_mode": 1}

        result = build_format_map(clap_info, indices, spec, {})

        assert result.plugin == "Surge XT"
        assert result.version == "1.3.4"
        assert set(result.params) == {"a_amp_eg_attack", "a_amp_eg_envelope_mode"}
        attack = result.params["a_amp_eg_attack"]
        assert attack.clap_param_id == 11
        assert attack.clap_name == "A Amp EG Attack"
        assert attack.clap_module_name == "/A Envelopes/"
        assert (attack.min_value, attack.max_value) == (0.0, 1.0)
        assert not attack.is_stepped

    def test_spec_param_without_preset_index_raises_listing_all_missing(self):
        """All unmapped spec params are listed in one hard error."""
        clap_info = _plugin_info([_info(name="A Amp EG Attack")])
        spec = ParamSpec(
            [
                ContinuousParameter(name="a_amp_eg_attack"),
                ContinuousParameter(name="ghost_param"),
                ContinuousParameter(name="phantom_param"),
            ],
            [],
        )

        with pytest.raises(ValueError, match="(?s)ghost_param.*phantom_param"):
            build_format_map(clap_info, {"a_amp_eg_attack": 0}, spec, {})

    def test_preset_index_beyond_dump_raises_value_error(self):
        """An index past the dump end is a hard error naming the param."""
        clap_info = _plugin_info([_info(name="A Amp EG Attack")])
        spec = ParamSpec([ContinuousParameter(name="a_amp_eg_attack")], [])

        with pytest.raises(ValueError, match="a_amp_eg_attack"):
            build_format_map(clap_info, {"a_amp_eg_attack": 5}, spec, {})

    def test_display_name_cross_check_mismatch_raises_value_error(self):
        """A surge_params.csv name disagreeing with the bridged entry is a hard error."""
        clap_info = _plugin_info([_info(name="B LFO 4 Deform")])
        spec = ParamSpec([ContinuousParameter(name="a_amp_eg_attack")], [])
        display_names = {"a_amp_eg_attack": "A Amp Eg Attack"}

        with pytest.raises(ValueError, match="a_amp_eg_attack"):
            build_format_map(clap_info, {"a_amp_eg_attack": 0}, spec, display_names)

    def test_display_name_cross_check_is_case_insensitive(self):
        """Display-name comparison tolerates case differences (EG vs Eg)."""
        clap_info = _plugin_info([_info(name="A Amp EG Attack")])
        spec = ParamSpec([ContinuousParameter(name="a_amp_eg_attack")], [])
        display_names = {"a_amp_eg_attack": "A Amp Eg Attack"}

        result = build_format_map(clap_info, {"a_amp_eg_attack": 0}, spec, display_names)

        assert result.params["a_amp_eg_attack"].clap_name == "A Amp EG Attack"

    def test_stepped_ref_with_incompatible_categorical_grid_raises(self):
        """raw_values that do not lerp onto consecutive steps are rejected."""
        # Two native steps cannot represent a 3-value categorical: index 2's
        # raw_value 1.0 lands on native step 1, not 2.
        clap_info = _plugin_info([_info(name="Filter Type", is_stepped=True, max_value=1.0)])
        spec = ParamSpec(
            [
                CategoricalParameter(
                    name="filter_type",
                    values=["a", "b", "c"],
                    raw_values=[0.0, 0.5, 1.0],
                    encoding="onehot",
                )
            ],
            [],
        )

        with pytest.raises(ValueError, match="filter_type"):
            build_format_map(clap_info, {"filter_type": 0}, spec, {})

    def test_stepped_ref_with_compatible_categorical_grid_builds(self):
        """A grid-consistent stepped categorical passes and keeps is_stepped."""
        clap_info = _plugin_info([_info(name="Env Mode", is_stepped=True, max_value=1.0)])
        spec = ParamSpec(
            [
                CategoricalParameter(
                    name="env_mode",
                    values=["Digital", "Analog"],
                    raw_values=[0.25, 0.75],
                    encoding="onehot",
                )
            ],
            [],
        )

        result = build_format_map(clap_info, {"env_mode": 0}, spec, {})

        assert result.params["env_mode"].is_stepped


def test_read_display_names_maps_pyname_to_name(tmp_path: Path) -> None:
    """The cross-check table reads pyname -> display name from the CSV columns.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    csv_path = tmp_path / "params.csv"
    csv_path.write_text(',pyname,name,range\n0,a_x,A X,"(None, None, None)"\n')

    assert _read_display_names(csv_path) == {"a_x": "A X"}


def test_read_display_names_missing_column_raises_key_error(tmp_path: Path) -> None:
    """A CSV without the expected columns fails loudly rather than mapping nothing.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    csv_path = tmp_path / "params.csv"
    csv_path.write_text("a,b\n1,2\n")

    with pytest.raises(KeyError):
        _read_display_names(csv_path)


@pytest.mark.slow
@pytest.mark.skipif(
    not SURGE_XT_CLAP_PATH.exists(),
    reason=f"Surge XT CLAP not found at {SURGE_XT_CLAP_PATH} (install the surge-xt package)",
)
def test_dump_command_writes_full_dump_to_requested_path(tmp_path: Path) -> None:
    """The dump command writes a validated full dump to --out.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    out_path = tmp_path / "dump.json"

    result = CliRunner().invoke(
        build_clap_map_main, ["dump", "--out", str(out_path)], catch_exceptions=False
    )

    assert result.exit_code == 0, result.output
    dumped = ClapPluginInfo.model_validate_json(out_path.read_text())
    assert len(dumped.params) == SURGE_XT_CLAP_PARAM_COUNT


class TestInitOrderErrors:
    """The index-bridge soundness comparison behind _assert_init_order_matches."""

    def test_matching_orders_yield_no_errors(self):
        """Identical name sequences (case-insensitive) pass."""
        info = _plugin_info([_info(name="A Amp EG Attack"), _info(name="Scene Mode")])

        assert init_order_errors(["A Amp Eg Attack", "Scene Mode"], info) == []

    def test_count_mismatch_is_reported(self):
        """A dropped param surfaces as a count divergence."""
        info = _plugin_info([_info(name="A Amp EG Attack"), _info(name="Scene Mode")])

        errors = init_order_errors(["A Amp EG Attack"], info)

        assert len(errors) == 1
        assert "1 params" in errors[0]

    def test_every_positional_name_mismatch_is_listed(self):
        """Each diverging index is reported, not just the first."""
        info = _plugin_info([_info(name="Alpha"), _info(name="Beta"), _info(name="Gamma")])

        errors = init_order_errors(["Alpha", "Wrong", "Also Wrong"], info)

        assert len(errors) == 2
        assert "index 1" in errors[0]
        assert "index 2" in errors[1]
