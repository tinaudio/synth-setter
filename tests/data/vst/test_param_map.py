"""Tests for the unified cross-host parameter map contract."""

import pytest
from pydantic import ValidationError

from synth_setter.data.vst.clap_map import ClapParamRef
from synth_setter.data.vst.param_map import (
    BackendSnapshot,
    DawDreamerParamRef,
    ParamIdentity,
    PedalboardParamRef,
    SynthParamMap,
)


def _identity(index: int) -> ParamIdentity:
    """Build one complete fake identity.

    :param index: Fake host index.
    :returns: Complete identity.
    """
    return ParamIdentity(
        pedalboard=PedalboardParamRef(index=index, name=f"Param {index}"),
        clap=ClapParamRef(
            clap_param_id=index,
            clap_name=f"Param {index}",
            clap_module_name="/",
            min_value=0.0,
            max_value=1.0,
            is_stepped=False,
        ),
        dawdreamer=DawDreamerParamRef(index=index + 10, name=f"DD {index}"),
    )


def _param_map(params: dict[str, ParamIdentity]) -> SynthParamMap:
    """Build a joint map around fake identities.

    :param params: Identities keyed by pyname.
    :returns: Fake joint map.
    """
    snapshot = BackendSnapshot(plugin_version="1.2.3", parameter_count=20)
    return SynthParamMap(
        plugin="Synth",
        param_spec_name="synth",
        preset_resource="presets/base.vstpreset",
        preset_sha256="a" * 64,
        pedalboard=snapshot,
        clap=snapshot,
        dawdreamer=snapshot,
        params=params,
    )


def test_param_map_projections_preserve_host_identities() -> None:
    """Host projections retain their committed indices."""
    joint = _param_map({"cutoff": _identity(2)})

    assert joint.dawdreamer_indices() == {"cutoff": 12}
    assert joint.clap_projection().params["cutoff"].clap_param_id == 2


def test_param_map_duplicate_dawdreamer_indices_raise() -> None:
    """Two pynames cannot target one DawDreamer index."""
    first = _identity(1)
    second = _identity(2).model_copy(
        update={"dawdreamer": DawDreamerParamRef(index=11, name="Alias")}
    )

    with pytest.raises(ValidationError, match="duplicate dawdreamer"):
        _param_map({"first": first, "second": second})
