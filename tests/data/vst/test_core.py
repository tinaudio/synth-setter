from __future__ import annotations

import plistlib
from pathlib import Path

import pytest

from synth_setter.data.vst.core import extract_renderer_version


class TestExtractRendererVersion:
    """Static-metadata + pedalboard-fallback VST3 plugin version extractor."""

    def test_extracts_version_from_linux_moduleinfo_json(self, tmp_path: Path) -> None:
        """Linux moduleinfo.json with Version key returns the version string."""
        plugin = tmp_path / "Plugin.vst3"
        contents = plugin / "Contents"
        contents.mkdir(parents=True)
        (contents / "moduleinfo.json").write_text('{"Version": "1.3.4"}')
        assert extract_renderer_version(plugin) == "1.3.4"

    def test_extracts_version_from_macos_info_plist(self, tmp_path: Path) -> None:
        """MacOS Info.plist with CFBundleShortVersionString returns the version."""
        plugin = tmp_path / "Plugin.vst3"
        contents = plugin / "Contents"
        contents.mkdir(parents=True)
        plist_data = {"CFBundleShortVersionString": "1.3.4"}
        (contents / "Info.plist").write_bytes(plistlib.dumps(plist_data))
        assert extract_renderer_version(plugin) == "1.3.4"

    def test_prefers_moduleinfo_json_when_both_exist(self, tmp_path: Path) -> None:
        """When both static-metadata files exist, moduleinfo.json takes precedence."""
        plugin = tmp_path / "Plugin.vst3"
        contents = plugin / "Contents"
        contents.mkdir(parents=True)
        (contents / "moduleinfo.json").write_text('{"Version": "2.0.0"}')
        plist_data = {"CFBundleShortVersionString": "1.0.0"}
        (contents / "Info.plist").write_bytes(plistlib.dumps(plist_data))
        assert extract_renderer_version(plugin) == "2.0.0"

    def test_raises_file_not_found_when_plugin_path_does_not_exist(self, tmp_path: Path) -> None:
        """Nonexistent plugin path raises FileNotFoundError."""
        plugin = tmp_path / "nonexistent.vst3"
        with pytest.raises(FileNotFoundError, match="Plugin path does not exist"):
            extract_renderer_version(plugin)

    def test_raises_key_error_when_version_field_missing(self, tmp_path: Path) -> None:
        """moduleinfo.json without a Version key raises KeyError before pedalboard fallback."""
        plugin = tmp_path / "Plugin.vst3"
        contents = plugin / "Contents"
        contents.mkdir(parents=True)
        (contents / "moduleinfo.json").write_text('{"Name": "TestPlugin"}')
        with pytest.raises(KeyError):
            extract_renderer_version(plugin)
