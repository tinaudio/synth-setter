from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from synth_setter.data.vst.dawdreamer_introspect import dump_dawdreamer_plugin
from synth_setter.data.vst.dawdreamer_map import build_dawdreamer_map, load_dawdreamer_map
from tests._vst import PLUGIN_PATH


def test_dump_dawdreamer_plugin_maps_description_fields(monkeypatch) -> None:
    class Processor:
        def get_parameters_description(self) -> list[dict[str, object]]:
            return [
                {
                    "index": 3,
                    "name": "Cutoff",
                    "label": "Hz",
                    "category": "filter",
                    "defaultValue": 0.25,
                    "valueStrings": ["Low", "High"],
                }
            ]

    class Engine:
        def __init__(self, sample_rate: float, block_size: int) -> None:
            assert (sample_rate, block_size) == (48000, 512)

        def make_plugin_processor(self, name: str, path: str) -> Processor:
            assert name == "introspect"
            assert path == str(Path("plugin.vst3").resolve())
            return Processor()

    monkeypatch.setitem(sys.modules, "dawdreamer", types.SimpleNamespace(RenderEngine=Engine))

    result = dump_dawdreamer_plugin(Path("plugin.vst3"), sample_rate=48000, block_size=512)

    assert result.plugin == str(Path("plugin.vst3").resolve())
    assert result.params["cutoff"].index == 3
    assert result.params["cutoff"].value_strings == ("Low", "High")


def test_dawdreamer_map_round_trips_as_json(tmp_path: Path) -> None:
    path = tmp_path / "map.json"
    path.write_text(
        json.dumps(
            {
                "plugin": "/tmp/plugin.vst3",
                "params": {
                    "cutoff": {
                        "index": 3,
                        "name": "Cutoff",
                        "label": "Hz",
                        "category": "filter",
                        "default_value": 0.25,
                        "value_strings": ["Low", "High"],
                    }
                },
            }
        )
    )

    loaded = load_dawdreamer_map(path)

    assert loaded.params["cutoff"].default_value == 0.25


def test_build_dawdreamer_map_normalized_name_collision_raises() -> None:
    """Distinct display names cannot silently collapse to one normalized key."""
    descriptions = [
        {"index": 0, "name": "Foo-Bar"},
        {"index": 1, "name": "Foo Bar"},
    ]

    with pytest.raises(
        ValueError,
        match=r"foo_bar.*Foo-Bar.*Foo Bar",
    ):
        build_dawdreamer_map(Path("plugin.vst3"), descriptions)


@pytest.mark.requires_vst
@pytest.mark.slow
def test_dump_dawdreamer_plugin_maps_real_surge_parameters() -> None:
    """Real Surge introspection produces a populated map without false collisions."""
    result = dump_dawdreamer_plugin(Path(PLUGIN_PATH))

    assert result.plugin == str(Path(PLUGIN_PATH).resolve())
    assert len(result.params) > 2_000
