"""Behavioral tests for cross-host VST parameter-map construction."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from synth_setter.data.vst.clap_introspect import ClapParamInfo, ClapPluginInfo
from synth_setter.data.vst.param_map import load_param_map
from synth_setter.data.vst.param_spec import CategoricalParameter, ContinuousParameter, ParamSpec
from synth_setter.tools import build_param_map
from synth_setter.tools.build_param_map import HostDump, HostParam, join_param_map


def _host_dump(*params: HostParam, plugin: str = "Test Synth", version: str = "1.0") -> HostDump:
    """Build a host dump.

    :param *params: Enumerated host parameters.
    :param plugin: Host plugin name.
    :param version: Host plugin version.
    :returns: Provenance-bearing host dump.
    """
    return HostDump(
        plugin=plugin,
        plugin_version=version,
        preset_resource="presets/test.vstpreset",
        preset_sha256="a" * 64,
        params=list(params),
    )


def _clap(
    *params: ClapParamInfo, plugin: str = "Test Synth", version: str = "1.0"
) -> ClapPluginInfo:
    """Build a CLAP dump.

    :param *params: Enumerated CLAP parameters.
    :param plugin: CLAP plugin name.
    :param version: CLAP plugin version.
    :returns: CLAP metadata dump.
    """
    return ClapPluginInfo(
        plugin_id="com.example.test",
        plugin_name=plugin,
        vendor="Example",
        version=version,
        clap_version="1.2.0",
        params=list(params),
    )


def _clap_param(index: int, name: str, *, stepped: bool = False) -> ClapParamInfo:
    """Build one CLAP parameter.

    :param index: CLAP parameter id.
    :param name: CLAP parameter name.
    :param stepped: Whether the parameter is categorical.
    :returns: CLAP parameter metadata.
    """
    return ClapParamInfo(
        id=index,
        name=name,
        module="/",
        min_value=0.0,
        max_value=2.0 if stepped else 1.0,
        default_value=0.0,
        flags=1 if stepped else 0,
        is_stepped=stepped,
    )


@pytest.fixture()
def registry(monkeypatch: pytest.MonkeyPatch) -> dict[str, ParamSpec]:
    """Install a minimal builder registry.

    :param monkeypatch: Pytest monkeypatch fixture.
    :returns: Installed parameter-spec registry.
    """
    result = {"test": ParamSpec([ContinuousParameter("cutoff", 0.0, 1.0)], [])}
    monkeypatch.setattr(build_param_map, "param_specs", result)
    return result


def _valid_inputs() -> tuple[HostDump, ClapPluginInfo, HostDump]:
    """Build valid inputs for a three-host join.

    :returns: Pedalboard, CLAP, and DawDreamer dumps.
    """
    return (
        _host_dump(HostParam(index=0, key="cutoff", name="Cutoff")),
        _clap(_clap_param(7, "Cutoff")),
        _host_dump(HostParam(index=11, name="Cutoff")),
    )


def test_join_param_map_preserves_verified_host_identities(registry: dict[str, ParamSpec]) -> None:
    """A valid join preserves each host identity.

    :param registry: Minimal builder registry.
    """
    pedalboard, clap, dawdreamer = _valid_inputs()

    result = join_param_map("test", pedalboard, clap, dawdreamer)

    identity = result.params["cutoff"]
    assert identity.pedalboard.index == 0
    assert identity.pedalboard.name == "Cutoff"
    assert identity.clap is not None
    assert identity.clap.clap_param_id == 7
    assert identity.dawdreamer.index == 11
    assert identity.dawdreamer.name == "Cutoff"


def test_join_param_map_resolves_fx_slots_by_anchor_position(
    registry: dict[str, ParamSpec],
) -> None:
    """FX slots resolve from an anchored DawDreamer position.

    :param registry: Minimal builder registry.
    """
    registry["test"] = ParamSpec([ContinuousParameter("fx_a1_delay_time", 0.0, 1.0)], [])
    pedalboard = _host_dump(HostParam(index=0, key="fx_a1_delay_time", name="FX A1 Delay - Time"))
    clap = _clap(_clap_param(7, "FX A1 Param 1"))
    dawdreamer = _host_dump(
        HostParam(index=40, name="FX A1 FX Type"),
        HostParam(index=41, name="FX A1 Param 1"),
    )

    result = join_param_map("test", pedalboard, clap, dawdreamer)

    assert result.params["fx_a1_delay_time"].dawdreamer.index == 41
    assert result.params["fx_a1_delay_time"].dawdreamer.name == "FX A1 Param 1"


def test_join_param_map_rejects_unanchored_fx_slot(registry: dict[str, ParamSpec]) -> None:
    """An FX slot without its anchor is rejected.

    :param registry: Minimal builder registry.
    """
    registry["test"] = ParamSpec([ContinuousParameter("fx_a1_delay_time", 0.0, 1.0)], [])
    pedalboard = _host_dump(HostParam(index=0, key="fx_a1_delay_time", name="FX A1 Delay - Time"))
    clap = _clap(_clap_param(7, "FX A1 Param 1"))
    dawdreamer = _host_dump(HostParam(index=41, name="FX A1 Param 1"))

    with pytest.raises(
        ValueError, match="DawDreamer name 'FX A1 Param 1' is missing or ambiguous"
    ):
        join_param_map("test", pedalboard, clap, dawdreamer)


@pytest.mark.parametrize(
    "mutation, expected",
    [
        (
            lambda pedalboard, clap, dawdreamer: (
                pedalboard.model_copy(update={"plugin": "Other"}),
                clap,
                dawdreamer,
            ),
            "plugin identities disagree",
        ),
        (
            lambda pedalboard, clap, dawdreamer: (
                pedalboard.model_copy(update={"plugin_version": "2.0"}),
                clap,
                dawdreamer,
            ),
            "host plugin versions disagree",
        ),
        (
            lambda pedalboard, clap, dawdreamer: (
                pedalboard.model_copy(update={"preset_resource": "other.vstpreset"}),
                clap,
                dawdreamer,
            ),
            "preset resources disagree",
        ),
        (
            lambda pedalboard, clap, dawdreamer: (
                pedalboard.model_copy(update={"preset_sha256": "b" * 64}),
                clap,
                dawdreamer,
            ),
            "preset hashes disagree",
        ),
    ],
    ids=["plugin", "version", "preset-resource", "preset-hash"],
)
def test_join_param_map_rejects_provenance_drift(
    registry: dict[str, ParamSpec],
    mutation: Callable[
        [HostDump, ClapPluginInfo, HostDump], tuple[HostDump, ClapPluginInfo, HostDump]
    ],
    expected: str,
) -> None:
    """Provenance drift is rejected.

    :param registry: Minimal builder registry.
    :param mutation: One invalid input transformation.
    :param expected: Required diagnostic text.
    """
    pedalboard, clap, dawdreamer = mutation(*_valid_inputs())

    with pytest.raises(ValueError, match=expected):
        join_param_map("test", pedalboard, clap, dawdreamer)


@pytest.mark.parametrize(
    "pedalboard, clap, dawdreamer, expected",
    [
        (
            _host_dump(
                HostParam(index=0, key="cutoff", name="Cutoff"),
                HostParam(index=1, key="cutoff", name="Duplicate"),
            ),
            _clap(_clap_param(7, "Cutoff")),
            _host_dump(HostParam(index=11, name="Cutoff")),
            "duplicate Pedalboard keys",
        ),
        (
            _host_dump(
                HostParam(index=0, key="cutoff", name="Cutoff"),
                HostParam(index=0, key="other", name="Other"),
            ),
            _clap(_clap_param(7, "Cutoff")),
            _host_dump(HostParam(index=11, name="Cutoff")),
            "duplicate Pedalboard indices",
        ),
        (
            _host_dump(HostParam(index=0, key="cutoff", name="Cutoff")),
            _clap(_clap_param(7, "Cutoff")),
            _host_dump(HostParam(index=11, name="Cutoff"), HostParam(index=11, name="Alias")),
            "duplicate DawDreamer index 11",
        ),
    ],
    ids=["pedalboard-key", "pedalboard-index", "dawdreamer-index"],
)
def test_join_param_map_rejects_duplicate_host_identities(
    registry: dict[str, ParamSpec],
    pedalboard: HostDump,
    clap: ClapPluginInfo,
    dawdreamer: HostDump,
    expected: str,
) -> None:
    """Duplicate host identities are rejected.

    :param registry: Minimal builder registry.
    :param pedalboard: Pedalboard dump under test.
    :param clap: CLAP dump under test.
    :param dawdreamer: DawDreamer dump under test.
    :param expected: Required diagnostic text.
    """
    with pytest.raises(ValueError, match=expected):
        join_param_map("test", pedalboard, clap, dawdreamer)


@pytest.mark.parametrize(
    "pedalboard, clap, dawdreamer, expected",
    [
        (
            _host_dump(),
            _clap(_clap_param(7, "Cutoff")),
            _host_dump(HostParam(index=11, name="Cutoff")),
            "missing Pedalboard identity",
        ),
        (
            _host_dump(HostParam(index=2, key="cutoff", name="Cutoff")),
            _clap(_clap_param(7, "Cutoff")),
            _host_dump(HostParam(index=11, name="Cutoff")),
            "outside CLAP dump",
        ),
        (
            _host_dump(HostParam(index=0, key="cutoff", name="Cutoff")),
            _clap(_clap_param(7, "Cutoff")),
            _host_dump(),
            "DawDreamer name 'Cutoff' is missing or ambiguous",
        ),
        (
            _host_dump(HostParam(index=0, key="cutoff", name="Cutoff")),
            _clap(_clap_param(7, "Cutoff")),
            _host_dump(HostParam(index=11, name="Cutoff"), HostParam(index=12, name="Cutoff")),
            "DawDreamer name 'Cutoff' is missing or ambiguous",
        ),
    ],
    ids=["missing-pedalboard", "clap-index", "missing-dawdreamer", "ambiguous-dawdreamer"],
)
def test_join_param_map_rejects_unresolvable_parameter_identities(
    registry: dict[str, ParamSpec],
    pedalboard: HostDump,
    clap: ClapPluginInfo,
    dawdreamer: HostDump,
    expected: str,
) -> None:
    """Unresolvable parameter identities are rejected.

    :param registry: Minimal builder registry.
    :param pedalboard: Pedalboard dump under test.
    :param clap: CLAP dump under test.
    :param dawdreamer: DawDreamer dump under test.
    :param expected: Required diagnostic text.
    """
    with pytest.raises(ValueError, match=expected):
        join_param_map("test", pedalboard, clap, dawdreamer)


def test_join_param_map_aggregates_independent_errors(registry: dict[str, ParamSpec]) -> None:
    """Independent input defects are aggregated.

    :param registry: Minimal builder registry.
    """
    registry["test"] = ParamSpec(
        [ContinuousParameter("cutoff", 0.0, 1.0), ContinuousParameter("resonance", 0.0, 1.0)],
        [],
    )
    pedalboard = _host_dump(HostParam(index=0, key="cutoff", name="Cutoff"), plugin="Other")
    clap = _clap(_clap_param(7, "Cutoff"))
    dawdreamer = _host_dump()

    with pytest.raises(ValueError) as caught:
        join_param_map("test", pedalboard, clap, dawdreamer)

    assert "plugin identities disagree" in str(caught.value)
    assert "DawDreamer name 'Cutoff' is missing or ambiguous" in str(caught.value)
    assert "resonance: missing Pedalboard identity" in str(caught.value)


def test_join_param_map_rejects_invalid_categorical_grid(registry: dict[str, ParamSpec]) -> None:
    """A categorical grid inconsistent with CLAP steps is rejected.

    :param registry: Minimal builder registry.
    """
    registry["test"] = ParamSpec(
        [CategoricalParameter("mode", ["a", "b", "c"], raw_values=[0.0, 0.2, 1.0])], []
    )
    pedalboard = _host_dump(HostParam(index=0, key="mode", name="Mode"))
    clap = _clap(_clap_param(7, "Mode", stepped=True))
    dawdreamer = _host_dump(HostParam(index=11, name="Mode"))

    with pytest.raises(ValueError, match="categorical grid does not match CLAP steps"):
        join_param_map("test", pedalboard, clap, dawdreamer)


def test_join_param_map_accepts_matching_categorical_grid(registry: dict[str, ParamSpec]) -> None:
    """A categorical grid consistent with CLAP steps is accepted.

    :param registry: Minimal builder registry.
    """
    registry["test"] = ParamSpec(
        [CategoricalParameter("mode", ["a", "b", "c"], raw_values=[0.0, 0.5, 1.0])], []
    )
    pedalboard = _host_dump(HostParam(index=0, key="mode", name="Mode"))
    clap = _clap(_clap_param(7, "Mode", stepped=True))
    dawdreamer = _host_dump(HostParam(index=11, name="Mode"))

    clap_reference = join_param_map("test", pedalboard, clap, dawdreamer).params["mode"].clap
    assert clap_reference is not None
    assert clap_reference.is_stepped


def test_build_command_writes_map_consumable_by_runtime(
    registry: dict[str, ParamSpec], tmp_path: Path
) -> None:
    """The build CLI emits a runtime-consumable map.

    :param registry: Minimal builder registry.
    :param tmp_path: Temporary CLI workspace.
    """
    pedalboard, clap, dawdreamer = _valid_inputs()
    pedalboard_path = tmp_path / "pedalboard.json"
    clap_path = tmp_path / "clap.json"
    dawdreamer_path = tmp_path / "dawdreamer.json"
    output_path = tmp_path / "map.json"
    pedalboard_path.write_text(pedalboard.model_dump_json(), encoding="utf-8")
    clap_path.write_text(clap.model_dump_json(), encoding="utf-8")
    dawdreamer_path.write_text(dawdreamer.model_dump_json(), encoding="utf-8")

    result = CliRunner().invoke(
        build_param_map.main,
        [
            "build",
            "--pedalboard-dump",
            str(pedalboard_path),
            "--clap-dump",
            str(clap_path),
            "--dawdreamer-dump",
            str(dawdreamer_path),
            "--param-spec-name",
            "test",
            "--out",
            str(output_path),
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert load_param_map(output_path).params["cutoff"].dawdreamer.index == 11


def test_dump_dawdreamer_writes_raw_host_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The DawDreamer dump persists raw host labels.

    :param tmp_path: Temporary command workspace.
    :param monkeypatch: Pytest monkeypatch fixture.
    """
    plugin_path = tmp_path / "plugin.vst3"
    preset_path = tmp_path / "preset.vstpreset"
    output_path = tmp_path / "dawdreamer.json"
    plugin_path.touch()
    preset_path.write_bytes(b"preset")

    class Processor:
        """Minimal DawDreamer processor fake."""

        def load_vst3_preset(self, path: str) -> None:
            """Accept the preset supplied by the command.

            :param path: Preset path.
            """
            del path

        def get_parameters_description(self) -> list[dict[str, object]]:
            """Return the raw host identity.

            :returns: One DawDreamer parameter description.
            """
            return [{"index": 20, "name": "FX A1 Param 1"}]

    class Engine:
        """Minimal DawDreamer engine fake."""

        def __init__(self, sample_rate: int, block_size: int) -> None:
            """Accept the command's fixed engine configuration.

            :param sample_rate: Render sample rate.
            :param block_size: Render block size.
            """
            del sample_rate, block_size

        def make_plugin_processor(self, name: str, path: str) -> Processor:
            """Create the preset-capable processor.

            :param name: Graph processor name.
            :param path: VST3 plugin path.
            :returns: Fake plugin processor.
            """
            del name, path
            return Processor()

    monkeypatch.setattr(
        build_param_map, "import_module", lambda _: SimpleNamespace(RenderEngine=Engine)
    )
    result = CliRunner().invoke(
        build_param_map.main,
        [
            "dump-dawdreamer",
            "--plugin",
            str(plugin_path),
            "--plugin-name",
            "Test Synth",
            "--plugin-version",
            "1.0",
            "--preset",
            str(preset_path),
            "--preset-resource",
            "presets/test.vstpreset",
            "--out",
            str(output_path),
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert json.loads(output_path.read_text(encoding="utf-8"))["params"] == [
        {"index": 20, "key": None, "name": "FX A1 Param 1"}
    ]


def test_dump_clap_writes_enumeration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLAP dump serializes the host enumeration.

    :param tmp_path: Temporary command workspace.
    :param monkeypatch: Pytest monkeypatch fixture.
    """
    plugin_path = tmp_path / "plugin.clap"
    output_path = tmp_path / "clap.json"
    plugin_path.touch()
    expected = _clap(_clap_param(7, "Cutoff"))
    monkeypatch.setattr(build_param_map, "dump_clap_plugin", lambda _: expected)

    result = CliRunner().invoke(
        build_param_map.main,
        ["dump-clap", "--plugin", str(plugin_path), "--out", str(output_path)],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert ClapPluginInfo.model_validate_json(output_path.read_text(encoding="utf-8")) == expected


def test_dump_pedalboard_writes_flushed_preset_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Pedalboard dump flushes its preset before serializing metadata.

    :param tmp_path: Temporary command workspace.
    :param monkeypatch: Pytest monkeypatch fixture.
    """
    plugin_path = tmp_path / "plugin.vst3"
    preset_path = tmp_path / "preset.vstpreset"
    output_path = tmp_path / "pedalboard.json"
    plugin_path.touch()
    preset_path.write_bytes(b"preset")
    calls: list[str] = []

    class Plugin:
        """Minimal Pedalboard plugin fake.

        .. attribute :: name

           Plugin name.

        .. attribute :: version

           Plugin version.

        .. attribute :: parameters

           Keyed host parameter metadata.
        """

        name = "Test Synth"
        version = "1.0"
        parameters = {"cutoff": SimpleNamespace(index=0, name="Cutoff")}

        def process(self, *args: object) -> None:
            """Record the required preset flush.

            :param *args: Pedalboard process arguments.
            """
            del args
            calls.append("process")

        def reset(self) -> None:
            """Record the post-flush reset."""
            calls.append("reset")

    plugin = Plugin()
    monkeypatch.setattr("synth_setter.data.vst.core.load_plugin", lambda _: plugin)
    monkeypatch.setattr(
        "synth_setter.data.vst.core.load_preset", lambda *_: calls.append("preset")
    )
    result = CliRunner().invoke(
        build_param_map.main,
        [
            "dump-pedalboard",
            "--plugin",
            str(plugin_path),
            "--preset",
            str(preset_path),
            "--preset-resource",
            "presets/test.vstpreset",
            "--out",
            str(output_path),
        ],
        catch_exceptions=False,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert result.exit_code == 0, result.output
    assert calls == ["preset", "process", "reset"]
    assert (
        payload["preset_sha256"]
        == "d410850fd5f4e0a3cbffa317eb15d8e3c8fe4bcdb7d77433d8618e0ddaca25cb"
    )
    assert payload["params"] == [{"index": 0, "key": "cutoff", "name": "Cutoff"}]
