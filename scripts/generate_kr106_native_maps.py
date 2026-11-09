"""Regenerate KR106-native baselines and joint maps from the pinned extension."""

import hashlib
import json
import struct
from pathlib import Path

from synth_setter.data.vst.kr106_native_runtime import import_kr106_native


def component_values(path: Path) -> dict[str, float]:
    """Extract the 56 native controls from a KR106 JUCE VST3 preset.

    :param path: Committed VST3 preset container.
    :returns: DSP-enum-indexed native values.
    :raises ValueError: The container or component bounds are invalid.
    """
    data = path.read_bytes()
    if len(data) < 48 or data[:4] != b"VST3":
        raise ValueError("invalid VST3 preset")
    table = struct.unpack_from("<Q", data, 40)[0]
    if table + 8 > len(data) or data[table : table + 4] != b"List":
        raise ValueError("invalid VST3 chunk table")
    count = struct.unpack_from("<I", data, table + 4)[0]
    if table + 8 + count * 20 > len(data):
        raise ValueError("truncated VST3 chunk table")
    for index in range(count):
        entry = table + 8 + index * 20
        if data[entry : entry + 4] != b"Comp":
            continue
        offset, length = struct.unpack_from("<QQ", data, entry + 4)
        if length < 228 or offset + length > len(data):
            raise ValueError("invalid KR106 component bounds")
        if struct.unpack_from("<I", data, offset)[0] != 56:
            raise ValueError("KR106 component does not contain the pinned control count")
        values = struct.unpack_from("<56f", data, offset + 4)
        return {str(index): value for index, value in enumerate(values)}
    raise ValueError("VST3 preset has no component state")


def main() -> None:
    """Write baseline artifacts and native identities for all existing KR106 specs."""
    root = Path(__file__).resolve().parents[1]
    native = import_kr106_native()
    metadata = native.get_parameters()
    by_name = {item["name"]: item for item in metadata}
    for identity in (
        "ultramaster_kr106",
        "ultramaster_kr106_onehot",
        "ultramaster_kr106_single_note",
    ):
        map_path = root / f"src/synth_setter/data/vst/{identity}_param_map.json"
        mapping = json.loads(map_path.read_text())
        preset = root / mapping["preset_resource"]
        resource = mapping["preset_resource"].replace("-base.vstpreset", "-native.json")
        baseline = {
            "schema_version": 1,
            "source_revision": native.get_source_revision(),
            "source_vstpreset_sha256": hashlib.sha256(preset.read_bytes()).hexdigest(),
            "parameters": component_values(preset),
        }
        baseline_bytes = (json.dumps(baseline, indent=2) + "\n").encode()
        (root / resource).write_bytes(baseline_bytes)
        mapping["kr106_native"] = {
            "plugin_version": native.get_version(),
            "parameter_count": len(metadata),
        }
        mapping["kr106_native_preset_resource"] = resource
        mapping["kr106_native_preset_sha256"] = hashlib.sha256(baseline_bytes).hexdigest()
        for refs in mapping["params"].values():
            parameter = by_name[refs["dawdreamer"]["name"]]
            refs["kr106_native"] = {"native_id": parameter["id"], "name": parameter["name"]}
        map_path.write_text(json.dumps(mapping, indent=2) + "\n")


if __name__ == "__main__":
    main()
