"""Completeness and provenance checks for committed joint parameter maps."""

import hashlib
from pathlib import Path

import pytest

from synth_setter.data.vst.param_map import SynthParamMap, load_param_map
from synth_setter.data.vst.param_spec_registry import param_specs
from synth_setter.resources import as_file, param_map


@pytest.fixture(params=("surge_xt", "surge_simple", "surge_4"))
def spec_name(request: pytest.FixtureRequest) -> str:
    """Select each supported Surge parameter spec.

    :param request: Parameterized fixture request.
    :returns: Spec name.
    """
    return str(request.param)


@pytest.fixture
def committed_map(spec_name: str) -> SynthParamMap:
    """Load one packaged joint map.

    :param spec_name: Registered spec name.
    :returns: Packaged map.
    """
    with as_file(param_map(spec_name)) as path:
        return load_param_map(path)


def test_committed_map_exactly_covers_spec(
    committed_map: SynthParamMap, spec_name: str
) -> None:
    """Every map contains exactly its ParamSpec keys.

    :param committed_map: Packaged map.
    :param spec_name: Registered spec name.
    """
    expected = {param.name for param in param_specs[spec_name].synth_params}
    assert set(committed_map.params) == expected


def test_committed_map_preset_hash_is_fresh(committed_map: SynthParamMap) -> None:
    """Committed preset hashes match repository resources.

    :param committed_map: Packaged map.
    """
    preset = Path(committed_map.preset_resource)
    assert hashlib.sha256(preset.read_bytes()).hexdigest() == committed_map.preset_sha256


def test_committed_map_host_indices_are_in_snapshot_bounds(
    committed_map: SynthParamMap,
) -> None:
    """Every stored host index lies within its snapshot.

    :param committed_map: Packaged map.
    """
    for identity in committed_map.params.values():
        assert 0 <= identity.pedalboard.index < committed_map.pedalboard.parameter_count
        assert 0 <= identity.dawdreamer.index < committed_map.dawdreamer.parameter_count


def test_committed_map_clap_projection_is_complete(committed_map: SynthParamMap) -> None:
    """Every committed identity has a CLAP projection.

    :param committed_map: Packaged map.
    """
    assert set(committed_map.clap_projection().params) == set(committed_map.params)
