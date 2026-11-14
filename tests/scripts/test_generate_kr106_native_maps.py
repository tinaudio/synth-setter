"""The baseline extractor preserves DSP indices and rejects corrupt preset containers."""

import json
import struct
from pathlib import Path

import pytest

from scripts.generate_kr106_native_maps import component_values

_ROOT = Path(__file__).resolve().parents[2]
_PRESET = _ROOT / "presets/ultramaster_kr106-base.vstpreset"


def test_committed_preset_extracts_exact_native_baseline() -> None:
    """Extracted native values match the baseline used by workers."""
    expected = json.loads((_ROOT / "presets/ultramaster_kr106-native.json").read_text())
    assert component_values(_PRESET) == expected["parameters"]


@pytest.mark.parametrize(
    "corruption", ["magic", "table_offset", "table_count", "missing_component"]
)
def test_corrupt_preset_container_is_rejected(tmp_path: Path, corruption: str) -> None:
    """Invalid container metadata cannot produce a baseline.

    :param tmp_path: Directory for the malformed preset.
    :param corruption: Container field to damage.
    """
    data = bytearray(_PRESET.read_bytes())
    table = struct.unpack_from("<Q", data, 40)[0]
    if corruption == "magic":
        data[:4] = b"FAIL"
    elif corruption == "table_offset":
        struct.pack_into("<Q", data, 40, len(data) + 1)
    elif corruption == "table_count":
        struct.pack_into("<I", data, table + 4, 0xFFFF)
    else:
        struct.pack_into("<I", data, table + 4, 0)
    path = tmp_path / "bad.vstpreset"
    path.write_bytes(data)
    with pytest.raises(ValueError):
        component_values(path)
