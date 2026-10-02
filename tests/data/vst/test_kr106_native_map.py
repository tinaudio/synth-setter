"""KR106 native identities must remain complete, unique, and source-backed."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from synth_setter.data.vst.param_map import SynthParamMap, load_param_map


@pytest.fixture
def map_path() -> Path:
    """Locate the committed full KR106 cross-backend map.

    :returns: Parameter-map artifact path.
    """
    return (
        Path(__file__).resolve().parents[3]
        / "src/synth_setter/data/vst/ultramaster_kr106_param_map.json"
    )


def test_native_map_resolves_dsp_not_host_order(map_path: Path) -> None:
    """The host's reordered Bender LFO control resolves to the DSP's real ID.

    :param map_path: Committed joint map.
    """
    mapping = load_param_map(map_path)
    assert mapping.params["bender_lfo"].dawdreamer.index == 2
    assert mapping.kr106_native_params()["bender_lfo"].native_id == 42
    assert mapping.kr106_native_params()["adsr_mode"].native_id == 43


def test_native_map_rejects_missing_identity(map_path: Path) -> None:
    """Partial native coverage cannot silently drop sampled controls.

    :param map_path: Committed joint map.
    """
    data = json.loads(map_path.read_text())
    data["params"]["attack"]["kr106_native"] = None
    with pytest.raises(ValidationError, match="complete"):
        SynthParamMap.model_validate(data)


def test_native_map_rejects_duplicate_identity(map_path: Path) -> None:
    """Two repository names cannot alias the same native control.

    :param map_path: Committed joint map.
    """
    data = json.loads(map_path.read_text())
    data["params"]["attack"]["kr106_native"] = data["params"]["decay"]["kr106_native"]
    with pytest.raises(ValidationError, match="duplicate KR106-native"):
        SynthParamMap.model_validate(data)


def test_native_map_rejects_missing_baseline_provenance(map_path: Path) -> None:
    """A native map must pin the baseline used for rendering.

    :param map_path: Committed joint map.
    """
    data = json.loads(map_path.read_text())
    data["kr106_native_preset_sha256"] = None
    with pytest.raises(ValidationError, match="complete"):
        SynthParamMap.model_validate(data)
