"""Tests for the CLAP param map models and prediction→CLAP-value conversion."""

import json
from pathlib import Path

import pydantic
import pytest

from synth_setter.data.vst.clap_map import (
    ClapParamRef,
    PluginFormatMap,
    load_clap_map,
    synth_params_to_clap_rows,
)
from synth_setter.data.vst.param_spec import (
    CategoricalParameter,
    ContinuousParameter,
    Parameter,
    ParamSpec,
)


def _ref(**overrides: object) -> ClapParamRef:
    """Build a ClapParamRef with test defaults, overridable per test.

    :param **overrides: Field overrides merged over the defaults.
    :returns: The constructed ref.
    """
    fields = {
        "clap_param_id": 101,
        "clap_name": "A Amp EG Attack",
        "clap_module_name": "/A Envelopes/",
        "min_value": 0.0,
        "max_value": 1.0,
        "is_stepped": False,
    }
    fields.update(overrides)
    return ClapParamRef(**fields)


def _map(params: dict[str, ClapParamRef]) -> PluginFormatMap:
    """Wrap params in a minimal PluginFormatMap.

    :param params: pyname -> ref entries.
    :returns: The map.
    """
    return PluginFormatMap(plugin="Surge XT", version="1.3.4", params=params)


def _spec(params: list[Parameter]) -> ParamSpec:
    """Build a synth-only ParamSpec.

    :param params: Synth parameters.
    :returns: The spec.
    """
    return ParamSpec(list(params), [])


class TestClapMapModels:
    """Model validation and JSON round-trip behavior."""

    def test_clap_param_ref_extra_field_raises_validation_error(self):
        """An unknown field is rejected (extra=forbid at the trust boundary)."""
        with pytest.raises(pydantic.ValidationError):
            ClapParamRef.model_validate(
                {
                    "clap_param_id": 1,
                    "clap_name": "X",
                    "clap_module_name": "/M/",
                    "min_value": 0.0,
                    "max_value": 1.0,
                    "is_stepped": False,
                    "surprise": "nope",
                }
            )

    def test_load_clap_map_valid_file_round_trips(self, tmp_path: Path):
        """A dumped map file loads back equal.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        source = _map({"a_amp_eg_attack": _ref()})
        map_path = tmp_path / "map.json"
        map_path.write_text(source.model_dump_json(indent=2))

        loaded = load_clap_map(map_path)

        assert loaded == source

    def test_load_clap_map_missing_required_key_raises_validation_error(self, tmp_path: Path):
        """A document missing a required key fails validation.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        map_path = tmp_path / "map.json"
        map_path.write_text(json.dumps({"plugin": "Surge XT", "params": {}}))

        with pytest.raises(pydantic.ValidationError):
            load_clap_map(map_path)


class TestSynthParamsToClapRows:
    """Decoded-params -> native CLAP row conversion."""

    def test_continuous_param_midpoint_lerps_to_native_range(self):
        """0.5 lerps to the middle of the native range with full identity fields."""
        fmt = _map(
            {"gain": _ref(clap_param_id=7, clap_name="Gain", min_value=-60.0, max_value=0.0)}
        )
        spec = _spec([ContinuousParameter(name="gain")])

        rows = synth_params_to_clap_rows({"gain": 0.5}, spec, fmt)

        assert len(rows) == 1
        assert rows[0].pb_name == "gain"
        assert rows[0].clap_name == "Gain"
        assert rows[0].clap_module_name == "/A Envelopes/"
        assert rows[0].clap_param_id == 7
        assert rows[0].clap_value == pytest.approx(-30.0)

    def test_continuous_param_extremes_map_to_min_and_max(self):
        """0.0 and 1.0 map exactly onto the native bounds."""
        fmt = _map(
            {
                "lo": _ref(clap_param_id=1, min_value=2.0, max_value=10.0),
                "hi": _ref(clap_param_id=2, min_value=2.0, max_value=10.0),
            }
        )
        spec = _spec([ContinuousParameter(name="lo"), ContinuousParameter(name="hi")])

        rows = synth_params_to_clap_rows({"lo": 0.0, "hi": 1.0}, spec, fmt)

        assert rows[0].clap_value == pytest.approx(2.0)
        assert rows[1].clap_value == pytest.approx(10.0)

    def test_stepped_categorical_maps_nearest_raw_value_to_index_offset(self):
        """A stepped categorical emits min_value + nearest raw_values position."""
        fmt = _map({"mode": _ref(clap_param_id=9, is_stepped=True, min_value=0.0, max_value=1.0)})
        spec = _spec(
            [
                CategoricalParameter(
                    name="mode",
                    values=["Digital", "Analog"],
                    raw_values=[0.25, 0.75],
                    encoding="onehot",
                )
            ]
        )

        # 0.7 is nearest raw_value 0.75 (index 1), not the lerped native 0.7.
        rows = synth_params_to_clap_rows({"mode": 0.7}, spec, fmt)

        assert rows[0].clap_value == 1.0

    def test_stepped_categorical_nonzero_min_offsets_index(self):
        """A nonzero native minimum offsets the emitted step index."""
        fmt = _map({"filt": _ref(clap_param_id=3, is_stepped=True, min_value=2.0, max_value=5.0)})
        spec = _spec(
            [
                CategoricalParameter(
                    name="filt",
                    values=["a", "b", "c", "d"],
                    raw_values=[0.0, 1 / 3, 2 / 3, 1.0],
                    encoding="onehot",
                )
            ]
        )

        rows = synth_params_to_clap_rows({"filt": 0.6}, spec, fmt)

        # nearest raw_value is 2/3 -> index 2 -> 2.0 + 2.
        assert rows[0].clap_value == 4.0

    def test_stepped_categorical_wide_ref_uses_index_not_lerp(self):
        """Inputs where the two stepped formulas diverge pick the categorical index.

        raw_value 0.75 over a [0, 10] ref: index gives 1.0, round(lerp) would give 8.0 — a mutant
        collapsing the categorical branch fails here.
        """
        fmt = _map({"mode": _ref(clap_param_id=9, is_stepped=True, min_value=0.0, max_value=10.0)})
        spec = _spec(
            [
                CategoricalParameter(
                    name="mode",
                    values=["Digital", "Analog"],
                    raw_values=[0.25, 0.75],
                    encoding="onehot",
                )
            ]
        )

        rows = synth_params_to_clap_rows({"mode": 0.75}, spec, fmt)

        assert rows[0].clap_value == 1.0

    def test_stepped_param_with_continuous_spec_rounds_lerped_value(self):
        """A stepped ref without a categorical spec param rounds the lerped value."""
        fmt = _map(
            {"steps": _ref(clap_param_id=4, is_stepped=True, min_value=0.0, max_value=10.0)}
        )
        spec = _spec([ContinuousParameter(name="steps")])

        rows = synth_params_to_clap_rows({"steps": 0.68}, spec, fmt)

        assert rows[0].clap_value == 7.0

    def test_restricted_range_continuous_param_passes_through_unit_ref(self):
        """A spec param sampled on [0, 0.77] still passes its decoded value through a [0, 1] ref."""
        fmt = _map({"attack": _ref(clap_param_id=5)})
        spec = _spec([ContinuousParameter(name="attack", min=0.0, max=0.77)])

        rows = synth_params_to_clap_rows({"attack": 0.6}, spec, fmt)

        assert rows[0].clap_value == pytest.approx(0.6)

    def test_categorical_with_continuous_ref_lerps_like_surge(self):
        """The production Surge path: categorical spec param, unstepped [0, N] ref."""
        fmt = _map({"mode": _ref(clap_param_id=9, min_value=0.0, max_value=10.0)})
        spec = _spec(
            [
                CategoricalParameter(
                    name="mode",
                    values=["Digital", "Analog"],
                    raw_values=[0.25, 0.75],
                    encoding="onehot",
                )
            ]
        )

        rows = synth_params_to_clap_rows({"mode": 0.75}, spec, fmt)

        assert rows[0].clap_value == pytest.approx(7.5)

    def test_param_missing_from_map_raises_listing_every_missing_name(self):
        """Every missing pyname is listed in the hard error."""
        fmt = _map({"present": _ref()})
        spec = _spec(
            [
                ContinuousParameter(name="present"),
                ContinuousParameter(name="lost_1"),
                ContinuousParameter(name="lost_2"),
            ]
        )

        with pytest.raises(ValueError, match="lost_1.*lost_2"):
            synth_params_to_clap_rows({"present": 0.5, "lost_1": 0.5, "lost_2": 0.5}, spec, fmt)

    def test_rows_preserve_synth_param_iteration_order(self):
        """Rows come out in synth_params iteration order."""
        fmt = _map(
            {
                "first": _ref(clap_param_id=1),
                "second": _ref(clap_param_id=2),
            }
        )
        spec = _spec([ContinuousParameter(name="first"), ContinuousParameter(name="second")])

        rows = synth_params_to_clap_rows({"first": 0.0, "second": 1.0}, spec, fmt)

        assert [r.pb_name for r in rows] == ["first", "second"]
