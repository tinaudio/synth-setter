"""Integration tests for the ctypes CLAP introspector against a real Surge XT.clap."""

from pathlib import Path

import pytest

from synth_setter.data.vst.clap_introspect import (
    SURGE_XT_CLAP_PATH,
    ClapPluginInfo,
    dump_clap_plugin,
)
from tests.data.vst._clap import SURGE_XT_CLAP_PARAM_COUNT

# Only the live-plugin tests need the installed CLAP; the error-path tests below
# are pure and must run everywhere.
requires_surge_clap = pytest.mark.skipif(
    not SURGE_XT_CLAP_PATH.exists(),
    reason=f"Surge XT CLAP not found at {SURGE_XT_CLAP_PATH} (install the surge-xt package)",
)


@pytest.fixture(scope="module")
def surge_dump() -> ClapPluginInfo:
    """Dump the installed Surge XT CLAP once per module.

    :returns: The full param dump.
    """
    return dump_clap_plugin(SURGE_XT_CLAP_PATH)


@requires_surge_clap
@pytest.mark.slow
class TestDumpClapPluginSurgeXt:
    """Live-plugin introspection behavior against the installed Surge XT CLAP."""

    def test_descriptor_identifies_surge_instrument_with_version(self, surge_dump: ClapPluginInfo):
        """The descriptor names the Surge instrument and carries a version.

        :param surge_dump: Module-scoped dump fixture.
        """
        assert surge_dump.plugin_name == "Surge XT"
        assert surge_dump.plugin_id == "org.surge-synth-team.surge-xt"
        assert surge_dump.version != ""

    def test_enumerates_full_parameter_set(self, surge_dump: ClapPluginInfo):
        """The dump exposes the full parameter set.

        :param surge_dump: Module-scoped dump fixture.
        """
        assert len(surge_dump.params) == SURGE_XT_CLAP_PARAM_COUNT

    def test_known_continuous_param_carries_name_module_and_range(
        self, surge_dump: ClapPluginInfo
    ):
        """A known param carries name, module, and a sane range.

        :param surge_dump: Module-scoped dump fixture.
        """
        attacks = [p for p in surge_dump.params if p.name == "A Amp EG Attack"]

        assert len(attacks) == 1
        assert attacks[0].module == "/A Envelopes/"
        assert attacks[0].min_value < attacks[0].max_value
        assert not attacks[0].is_stepped

    def test_surge_exposes_uniform_normalized_ranges(self, surge_dump: ClapPluginInfo):
        """Surge's CLAP surface is uniformly continuous [0, 1].

        :param surge_dump: Module-scoped dump fixture.
        """
        # An upgrade flagging stepped params here means rebuilding the map (#1787).
        assert all(p.min_value == 0.0 and p.max_value == 1.0 for p in surge_dump.params)
        assert not any(p.is_stepped for p in surge_dump.params)

    def test_out_of_range_plugin_index_raises_runtime_error(self):
        """A factory index past the plugin count is a hard error naming the index."""
        with pytest.raises(RuntimeError, match="out of range"):
            dump_clap_plugin(SURGE_XT_CLAP_PATH, plugin_index=999)

    def test_param_ids_are_unique(self, surge_dump: ClapPluginInfo):
        """Param ids are unique across the dump.

        :param surge_dump: Module-scoped dump fixture.
        """
        ids = [p.id for p in surge_dump.params]

        assert len(ids) == len(set(ids))


def test_dump_clap_plugin_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    """A nonexistent plugin path raises FileNotFoundError.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    with pytest.raises(FileNotFoundError):
        dump_clap_plugin(tmp_path / "nope.clap")


def test_dump_clap_plugin_bundle_without_binary_raises_file_not_found(tmp_path: Path) -> None:
    """A macOS-style bundle dir missing Contents/MacOS/<stem> raises FileNotFoundError.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    bundle = tmp_path / "Hollow.clap"
    (bundle / "Contents" / "MacOS").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="no binary"):
        dump_clap_plugin(bundle)
