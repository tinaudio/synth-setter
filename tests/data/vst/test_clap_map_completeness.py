"""Completeness and freshness tests for the committed Surge CLAP maps (#1787 §4)."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from synth_setter.data.vst.clap_introspect import (
    SURGE_XT_CLAP_PATH,
    ClapPluginInfo,
    dump_clap_plugin,
)
from synth_setter.data.vst.clap_map import PluginFormatMap, load_clap_map
from synth_setter.data.vst.param_spec import CategoricalParameter
from synth_setter.data.vst.param_spec_registry import param_specs
from synth_setter.resources import as_file, clap_map
from synth_setter.tools.build_clap_map import main as build_clap_map_main
from tests.data.vst._clap import SURGE_XT_CLAP_PARAM_COUNT

# Every packaged per-spec map; each is checked against its own spec.
_MAPPED_SPECS = ("surge_xt", "surge_simple", "surge_4")

# Repo-tree artifact (deliberately not packaged): anchor on this file, not the cwd.
_CLAP_INFO_PATH = (
    Path(__file__).resolve().parents[3] / "src/synth_setter/data/vst/surge_xt_clap_info.json"
)


@pytest.fixture(scope="module", params=_MAPPED_SPECS)
def spec_name(request: pytest.FixtureRequest) -> str:
    """Parameterize the module over every packaged spec map.

    :param request: Pytest fixture request carrying the spec-name param.
    :returns: ``param_specs`` registry key.
    """
    return request.param


@pytest.fixture(scope="module")
def committed_map(spec_name: str) -> PluginFormatMap:
    """Load the packaged committed map for one spec once per module.

    :param spec_name: ``param_specs`` registry key.
    :returns: The committed map.
    """
    with as_file(clap_map(spec_name)) as path:
        return load_clap_map(path)


class TestCommittedMapCompleteness:
    """Issue #1787 §4 completeness checks over the committed map."""

    def test_every_spec_param_has_an_entry(self, committed_map: PluginFormatMap, spec_name: str):
        """Every spec synth param is mapped.

        :param committed_map: Packaged committed map fixture.
        :param spec_name: Registry key of the spec under test.
        """
        spec_names = {p.name for p in param_specs[spec_name].synth_params}

        missing = spec_names - set(committed_map.params)

        assert missing == set(), f"unmapped spec params: {sorted(missing)}"

    def test_map_carries_only_spec_params(self, committed_map: PluginFormatMap, spec_name: str):
        """The map has no entries beyond the spec.

        :param committed_map: Packaged committed map fixture.
        :param spec_name: Registry key of the spec under test.
        """
        spec_names = {p.name for p in param_specs[spec_name].synth_params}

        extra = set(committed_map.params) - spec_names

        assert extra == set(), f"map entries with no spec param: {sorted(extra)}"

    def test_map_metadata_committed_map_is_populated(self, committed_map: PluginFormatMap):
        """Plugin name and version are populated.

        :param committed_map: Packaged committed map fixture.
        """
        assert committed_map.plugin == "Surge XT"
        assert committed_map.version != ""

    def test_every_entry_has_a_sane_range(self, committed_map: PluginFormatMap):
        """Every entry satisfies min < max.

        :param committed_map: Packaged committed map fixture.
        """
        violations = [
            name for name, ref in committed_map.params.items() if not ref.min_value < ref.max_value
        ]

        assert violations == []

    def test_every_entry_spans_the_full_normalized_range(self, committed_map: PluginFormatMap):
        """All committed refs are exactly [0, 1] — the lerp must stay an identity.

        Surge's CLAP publishes normalized values, so pedalboard raw values pass through unscaled; a
        regenerated map with any other range would silently mis-scale restricted-range spec params
        (#1787).

        :param committed_map: Packaged committed map fixture.
        """
        non_unit = [
            name
            for name, ref in committed_map.params.items()
            if (ref.min_value, ref.max_value) != (0.0, 1.0)
        ]

        assert non_unit == []

    def test_clap_param_ids_committed_map_are_unique(self, committed_map: PluginFormatMap):
        """CLAP ids are unique across the map.

        :param committed_map: Packaged committed map fixture.
        """
        ids = [ref.clap_param_id for ref in committed_map.params.values()]

        assert len(ids) == len(set(ids))

    def test_stepped_entries_are_consistent_with_spec_categoricals(
        self, committed_map: PluginFormatMap, spec_name: str
    ):
        """Stepped entries' raw_values lerp onto consecutive native steps.

        The CLI converts stepped params via ``min_value`` + raw_values position,
        which is only correct on that grid.

        :param committed_map: Packaged committed map fixture.
        :param spec_name: Registry key of the spec under test.
        """
        spec_params = {p.name: p for p in param_specs[spec_name].synth_params}

        violations = []
        for name, ref in committed_map.params.items():
            param = spec_params[name]
            if not (ref.is_stepped and isinstance(param, CategoricalParameter)):
                continue
            span = ref.max_value - ref.min_value
            violations += [
                f"{name}[{i}]" for i, raw in enumerate(param.raw_values) if round(raw * span) != i
            ]

        assert violations == []


class TestCommittedDumpCoherence:
    """Committed dump <-> committed map provenance checks."""

    def test_committed_dump_matches_map_provenance(self, committed_map: PluginFormatMap):
        """Dump and map agree on plugin identity and the full param count.

        :param committed_map: Packaged committed map fixture.
        """
        info = ClapPluginInfo.model_validate_json(_CLAP_INFO_PATH.read_text())

        assert info.plugin_name == committed_map.plugin
        assert info.version == committed_map.version
        assert len(info.params) == SURGE_XT_CLAP_PARAM_COUNT

    def test_every_map_entry_exists_in_the_committed_dump(self, committed_map: PluginFormatMap):
        """Each map entry's id and name exist in the dump.

        :param committed_map: Packaged committed map fixture.
        """
        info = ClapPluginInfo.model_validate_json(_CLAP_INFO_PATH.read_text())
        by_id = {p.id: p for p in info.params}

        stale = [
            name
            for name, ref in committed_map.params.items()
            if by_id.get(ref.clap_param_id) is None
            or by_id[ref.clap_param_id].name != ref.clap_name
        ]

        assert stale == []


@pytest.mark.requires_vst
@pytest.mark.slow
def test_build_command_reproduces_committed_map(
    tmp_path: Path, committed_map: PluginFormatMap, spec_name: str
) -> None:
    """Real rebuild from the committed dump matches the committed map exactly.

    Runs the full builder — pedalboard double-load, init-order validation, csv cross-check — so
    drift between the plugin, the spec, and the committed artifacts fails loudly here rather than
    at bridge runtime.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    :param committed_map: The committed map to compare against.
    :param spec_name: Registry key of the spec under test.
    """
    out_path = tmp_path / "rebuilt_map.json"

    result = CliRunner().invoke(
        build_clap_map_main,
        [
            "build",
            "--clap-info",
            str(_CLAP_INFO_PATH),
            "--out",
            str(out_path),
            "--param-spec-name",
            spec_name,
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert load_clap_map(out_path) == committed_map


@pytest.mark.slow
@pytest.mark.skipif(
    not SURGE_XT_CLAP_PATH.exists(),
    reason=f"Surge XT CLAP not found at {SURGE_XT_CLAP_PATH} (install the surge-xt package)",
)
def test_committed_dump_matches_live_plugin_when_versions_agree() -> None:
    """A fresh live dump is identical to the committed one for the same Surge version.

    A silent rename/reorder in a Surge point release is the index bridge's highest-consequence
    failure mode; this pins it whenever CI's installed Surge matches the committed dump's version.
    """
    committed = ClapPluginInfo.model_validate_json(_CLAP_INFO_PATH.read_text())
    live = dump_clap_plugin(SURGE_XT_CLAP_PATH)
    if live.version != committed.version:
        pytest.skip(f"installed Surge {live.version} != committed dump {committed.version}")

    assert live == committed
