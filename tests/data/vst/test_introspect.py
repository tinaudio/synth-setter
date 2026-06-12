"""Behavior tests for ``synth_setter.data.vst.introspect`` (issue #1596)."""

from __future__ import annotations

from pathlib import Path

import pytest

from synth_setter.data.vst.introspect import (
    SkippedParameter,
    capture_preset,
    draft_synth_params,
    render_param_spec_module,
)
from synth_setter.data.vst.param_spec import (
    CategoricalParameter,
    ContinuousParameter,
    DiscreteLiteralParameter,
    ParamSpec,
)
from tests.data.vst._introspect_fakes import (
    IntrospectFakeParameter,
    IntrospectFakePlugin,
    assert_ruff_format_clean,
    exec_module,
)

_CONTINUOUS = IntrospectFakeParameter(float, [0.0, 0.5, 1.0])


def test_draft_float_parameter_yields_full_range_continuous() -> None:
    """A float-typed plugin parameter drafts as a full-range ``ContinuousParameter``."""
    plugin = IntrospectFakePlugin({"cutoff": _CONTINUOUS})

    params, skipped = draft_synth_params(plugin)

    assert skipped == []
    assert len(params) == 1
    (param,) = params
    assert isinstance(param, ContinuousParameter)
    assert param.name == "cutoff"
    assert param.min == 0.0
    assert param.max == 1.0


def test_draft_int_typed_parameter_falls_through_to_continuous() -> None:
    """A parameter of any non-str/bool type drafts as a continuous fallback."""
    plugin = IntrospectFakePlugin({"voices": IntrospectFakeParameter(int, [1, 2, 4, 8])})

    params, skipped = draft_synth_params(plugin)

    assert skipped == []
    (param,) = params
    assert isinstance(param, ContinuousParameter)
    assert param.name == "voices"


def test_draft_str_parameter_yields_onehot_categorical_with_host_raw_values() -> None:
    """A str-typed parameter drafts as an onehot categorical with host-asked raw values."""
    plugin = IntrospectFakePlugin(
        {
            "filter_type": IntrospectFakeParameter(
                str, ["LP", "HP", "BP"], raw_values=[0.0, 0.4, 0.8]
            )
        }
    )

    params, skipped = draft_synth_params(plugin)

    assert skipped == []
    (param,) = params
    assert isinstance(param, CategoricalParameter)
    assert param.name == "filter_type"
    assert param.values == ["LP", "HP", "BP"]
    assert param.raw_values == [0.0, 0.4, 0.8]
    assert param.encoding == "onehot"


def test_draft_bool_parameter_yields_two_value_categorical() -> None:
    """A bool-typed parameter drafts as a two-value onehot categorical."""
    plugin = IntrospectFakePlugin(
        {"retrigger": IntrospectFakeParameter(bool, [False, True], raw_values=[0.0, 1.0])}
    )

    params, skipped = draft_synth_params(plugin)

    assert skipped == []
    (param,) = params
    assert isinstance(param, CategoricalParameter)
    assert param.values == [False, True]
    assert param.raw_values == [0.0, 1.0]
    assert param.encoding == "onehot"


def test_draft_degenerate_parameter_is_skipped_with_reason() -> None:
    """A parameter with a single valid value is skipped and the reason names the count."""
    plugin = IntrospectFakePlugin(
        {
            "m1": IntrospectFakeParameter(float, [0.0]),
            "cutoff": _CONTINUOUS,
        }
    )

    params, skipped = draft_synth_params(plugin)

    assert [p.name for p in params] == ["cutoff"]
    (skip,) = skipped
    assert skip.name == "m1"
    assert "1 valid value" in skip.reason


def test_draft_parameter_with_failing_metadata_is_skipped_not_fatal() -> None:
    """A parameter whose metadata lookup raises is skipped; the rest still draft."""

    class _ExplodingParameter:
        type = str
        valid_values = ["A", "B"]

        def get_raw_value_for(self, value: str) -> float:
            """Fail unconditionally, simulating a plugin that rejects value lookups.

            :param value: Ignored.
            :returns: Never returns.
            :raises RuntimeError: Always.
            """
            raise RuntimeError("host rejected lookup")

    plugin = IntrospectFakePlugin(
        {"weird": _ExplodingParameter(), "cutoff": _CONTINUOUS}  # type: ignore[dict-item]
    )

    params, skipped = draft_synth_params(plugin)

    assert [p.name for p in params] == ["cutoff"]
    (skip,) = skipped
    assert skip.name == "weird"
    assert "host rejected lookup" in skip.reason


def test_draft_preserves_plugin_parameter_order() -> None:
    """Drafted parameters keep the plugin's own parameter order."""
    plugin = IntrospectFakePlugin(
        {
            "b_param": IntrospectFakeParameter(float, [0.0, 1.0]),
            "a_param": IntrospectFakeParameter(float, [0.0, 1.0]),
        }
    )

    params, skipped = draft_synth_params(plugin)

    assert skipped == []
    assert [p.name for p in params] == ["b_param", "a_param"]


def test_rendered_module_execs_into_registered_name_param_spec() -> None:
    """Emitted source executes into ``<SPEC_NAME>_PARAM_SPEC`` with default note params."""
    plugin = IntrospectFakePlugin(
        {
            "cutoff": _CONTINUOUS,
            "filter_type": IntrospectFakeParameter(str, ["LP", "HP"], raw_values=[0.0, 1.0]),
        }
    )
    params, skipped = draft_synth_params(plugin)

    source = render_param_spec_module(
        "my_synth", plugin_name=plugin.name, params=params, skipped=skipped
    )
    spec = exec_module(source)["MY_SYNTH_PARAM_SPEC"]

    assert isinstance(spec, ParamSpec)
    assert spec.synth_param_names == ["cutoff", "filter_type"]
    assert spec.note_param_names == ["pitch", "note_start_and_end"]


@pytest.mark.usefixtures("seeded_rng")
def test_rendered_spec_sample_encode_decode_round_trips() -> None:
    """A spec built from emitted source supports the sample/encode/decode pipeline."""
    plugin = IntrospectFakePlugin(
        {
            "cutoff": _CONTINUOUS,
            "filter_type": IntrospectFakeParameter(str, ["LP", "HP"], raw_values=[0.0, 1.0]),
            "retrigger": IntrospectFakeParameter(bool, [False, True], raw_values=[0.0, 1.0]),
        }
    )
    params, skipped = draft_synth_params(plugin)

    source = render_param_spec_module(
        "my_synth", plugin_name=plugin.name, params=params, skipped=skipped
    )
    spec = exec_module(source)["MY_SYNTH_PARAM_SPEC"]

    synth_params, note_params = spec.sample()
    encoded = spec.encode(synth_params, note_params)
    decoded_synth, decoded_note = spec.decode(encoded)

    assert len(encoded) == len(spec)
    assert decoded_synth.keys() == synth_params.keys()
    assert decoded_synth["filter_type"] in (0.0, 1.0)
    assert 48 <= decoded_note["pitch"] <= 72


def test_rendered_module_lists_skipped_parameters_as_comments() -> None:
    """Skipped parameters appear in the emitted module only as comment lines."""
    skipped = [SkippedParameter("m1", "degenerate: 1 valid value(s)")]

    source = render_param_spec_module(
        "my_synth", plugin_name="Fake Synth", params=[], skipped=skipped
    )

    skip_lines = [line for line in source.splitlines() if "m1" in line]
    assert skip_lines, "skipped parameter must be mentioned"
    assert all(line.lstrip().startswith("#") for line in skip_lines)
    assert "degenerate: 1 valid value(s)" in source
    assert isinstance(exec_module(source)["MY_SYNTH_PARAM_SPEC"], ParamSpec)


def test_rendered_module_collapses_multiline_skip_reasons_into_one_comment() -> None:
    """A multiline skip reason cannot break out of its comment line in the emitted module."""
    skipped = [SkippedParameter("weird", "metadata error: line one\n    line two")]

    source = render_param_spec_module(
        "my_synth", plugin_name="Fake Synth", params=[], skipped=skipped
    )

    assert "metadata error: line one line two" in source
    assert isinstance(exec_module(source)["MY_SYNTH_PARAM_SPEC"], ParamSpec)


def test_rendered_module_survives_quotes_in_categorical_values() -> None:
    """Categorical values containing quotes survive emission and re-execution."""
    plugin = IntrospectFakePlugin(
        {
            "shape": IntrospectFakeParameter(
                str, ['Saw "bright"', "Sine's", "Square"], raw_values=[0.0, 0.5, 1.0]
            )
        }
    )
    params, skipped = draft_synth_params(plugin)

    source = render_param_spec_module(
        "my_synth", plugin_name=plugin.name, params=params, skipped=skipped
    )
    spec = exec_module(source)["MY_SYNTH_PARAM_SPEC"]

    assert spec.synth_params[0].values == ['Saw "bright"', "Sine's", "Square"]
    assert "ContinuousParameter" not in source


def test_rendered_module_imports_only_parameter_classes_it_uses() -> None:
    """A continuous-only draft does not import ``CategoricalParameter`` (would fail F401)."""
    plugin = IntrospectFakePlugin({"cutoff": _CONTINUOUS})
    params, skipped = draft_synth_params(plugin)

    source = render_param_spec_module(
        "my_synth", plugin_name=plugin.name, params=params, skipped=skipped
    )

    assert "CategoricalParameter" not in source
    assert isinstance(exec_module(source)["MY_SYNTH_PARAM_SPEC"], ParamSpec)


def test_rendered_module_header_names_plugin_tool_and_provenance() -> None:
    """The emitted header names the plugin, the generating tool, and any provenance."""
    source = render_param_spec_module(
        "my_synth",
        plugin_name="Odin 2",
        params=[],
        skipped=[],
        provenance="plugin: Odin2.vst3 (version 2.4)",
    )

    header = source.splitlines()[0]
    assert "Odin 2" in header
    assert "synth-setter-introspect-plugin" in source
    assert "plugin: Odin2.vst3 (version 2.4)" in source
    assert isinstance(exec_module(source)["MY_SYNTH_PARAM_SPEC"], ParamSpec)


def test_rendered_module_rejects_undraftable_parameter_type() -> None:
    """Rendering a parameter type the draft never produces raises ``TypeError``."""
    foreign = DiscreteLiteralParameter(name="pitch", min=0, max=1)

    with pytest.raises(TypeError, match="DiscreteLiteralParameter"):
        render_param_spec_module(
            "my_synth", plugin_name="Fake Synth", params=[foreign], skipped=[]
        )


def test_rendered_module_is_ruff_format_clean() -> None:
    """The emitted draft survives ``ruff format --check`` — it is committed as-is.

    Uses a realistic surface: a long categorical (rewrap risk), quotes in
    values (quote-style normalization risk), bools, and a skipped parameter.
    """
    plugin = IntrospectFakePlugin(
        {
            "cutoff": _CONTINUOUS,
            "filter_type": IntrospectFakeParameter(
                str,
                ["Off", "LP 12 dB", "LP 24 dB", "LP Legacy Ladder", "HP 12 dB", "HP 24 dB"],
            ),
            "shape": IntrospectFakeParameter(
                str, ['Saw "bright"', "Sine's", "12\" Speaker's Cab"]
            ),
            "retrigger": IntrospectFakeParameter(bool, [False, True]),
            # long name exercises the wrapped multi-line ContinuousParameter form
            "a_scene_voice_filter_2_keytrack_response_depth_modulation_amount_extended": (
                IntrospectFakeParameter(float, [0.0, 1.0])
            ),
            "m1": IntrospectFakeParameter(float, [0.0]),
        }
    )
    params, skipped = draft_synth_params(plugin)

    source = render_param_spec_module(
        "my_synth",
        plugin_name=plugin.name,
        params=params,
        skipped=skipped,
        provenance="plugin: fake.vst3 (version 1.0), preset: none",
    )

    assert_ruff_format_clean(source)


def test_rendered_module_survives_hostile_plugin_name() -> None:
    """Quotes, backslashes, and newlines in the plugin name cannot break the module."""
    hostile = 'Evil """Synth\\\nName'

    source = render_param_spec_module("my_synth", plugin_name=hostile, params=[], skipped=[])

    assert isinstance(exec_module(source)["MY_SYNTH_PARAM_SPEC"], ParamSpec)
    assert_ruff_format_clean(source)


def test_rendered_module_wraps_long_skip_reasons_within_line_limit() -> None:
    """A very long skip reason wraps to multiple comment lines, each within 99 chars."""
    skipped = [SkippedParameter("weird", "metadata error: " + "x" * 70 + " " + "y" * 70)]

    source = render_param_spec_module(
        "my_synth", plugin_name="Fake Synth", params=[], skipped=skipped
    )

    comment_lines = [line for line in source.splitlines() if line.lstrip().startswith("#")]
    assert len(comment_lines) >= 2
    assert all(len(line) <= 99 for line in source.splitlines())
    assert isinstance(exec_module(source)["MY_SYNTH_PARAM_SPEC"], ParamSpec)


def test_rendered_skipped_only_module_is_ruff_format_clean() -> None:
    """A draft whose synth-params list holds only skip comments stays format-clean."""
    skipped = [SkippedParameter("m1", "degenerate: 1 valid value(s)")]

    source = render_param_spec_module(
        "my_synth", plugin_name="Fake Synth", params=[], skipped=skipped
    )

    assert_ruff_format_clean(source)


def test_capture_preset_writes_plugin_preset_data_bytes(tmp_path: Path) -> None:
    """``capture_preset`` writes the plugin's ``preset_data`` bytes verbatim.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    plugin = IntrospectFakePlugin({}, preset_data=b"VST3\x01\x00fake-state")
    out = tmp_path / "fake-base.vstpreset"

    capture_preset(plugin, out)

    assert out.read_bytes() == b"VST3\x01\x00fake-state"
