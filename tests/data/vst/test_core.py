from __future__ import annotations

import plistlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from synth_setter.data.vst import core
from synth_setter.data.vst.core import (
    extract_renderer_version,
    load_plugin,
    render_params,
)


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


class TestLoadPluginOpenGui:
    """``load_plugin`` honours the ``open_gui`` kwarg on non-Darwin hosts."""

    def test_open_gui_true_invokes_show_editor_warmup_on_non_darwin(  # noqa: DOC101,DOC103
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Default ``open_gui=True`` runs the ``show_editor`` warm-up on non-Darwin."""
        fake_plugin = MagicMock()
        monkeypatch.setattr(core, "VST3Plugin", lambda _path: fake_plugin)
        monkeypatch.setattr(sys, "platform", "linux")

        load_plugin("plugins/Surge XT.vst3")

        fake_plugin.show_editor.assert_called_once()

    def test_open_gui_false_skips_show_editor_warmup_on_non_darwin(  # noqa: DOC101,DOC103
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``open_gui=False`` skips the warm-up even on non-Darwin hosts."""
        fake_plugin = MagicMock()
        monkeypatch.setattr(core, "VST3Plugin", lambda _path: fake_plugin)
        monkeypatch.setattr(sys, "platform", "linux")

        load_plugin("plugins/Surge XT.vst3", open_gui=False)

        fake_plugin.show_editor.assert_not_called()

    def test_open_gui_true_is_still_skipped_on_darwin(  # noqa: DOC101,DOC103
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Even ``open_gui=True`` does not call ``show_editor`` on Darwin (#714 guardrail)."""
        fake_plugin = MagicMock()
        monkeypatch.setattr(core, "VST3Plugin", lambda _path: fake_plugin)
        monkeypatch.setattr(sys, "platform", "darwin")

        load_plugin("plugins/Surge XT.vst3", open_gui=True)

        fake_plugin.show_editor.assert_not_called()


class TestRenderParamsPreloadedPlugin:
    """``render_params`` accepts a pre-loaded plugin and skips load/preset on that path."""

    @staticmethod
    def _fake_plugin(audio_shape: tuple[int, int]) -> MagicMock:
        fake = MagicMock()
        # process() is called multiple times (flushes + render); only the render
        # call's return value matters for the assertion below, so return a fresh
        # zero array each call.
        fake.process.side_effect = lambda *a, **kw: np.zeros(audio_shape, dtype=np.float32)
        return fake

    def test_preloaded_plugin_bypasses_load_and_preset(  # noqa: DOC101,DOC103
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When ``plugin`` is supplied, ``load_plugin`` and ``load_preset`` are not called."""
        load_calls: list[str] = []
        preset_calls: list[tuple[object, str]] = []
        monkeypatch.setattr(
            core,
            "load_plugin",
            lambda path, **_kw: load_calls.append(path) or MagicMock(),
        )
        monkeypatch.setattr(
            core,
            "load_preset",
            lambda plugin, path: preset_calls.append((plugin, path)),
        )

        preloaded = self._fake_plugin(audio_shape=(2, 16))

        render_params(
            "plugins/Surge XT.vst3",
            params={},
            midi_note=60,
            velocity=100,
            note_start_and_end=(0.0, 1.0),
            signal_duration_seconds=1.0,
            sample_rate=16000,
            channels=2,
            preset_path="presets/surge-base.vstpreset",
            plugin=preloaded,
        )

        assert load_calls == []
        assert preset_calls == []
        # The pre-loaded plugin is what ran the render.
        assert preloaded.process.called

    def test_no_plugin_kwarg_reloads_per_call(  # noqa: DOC101,DOC103
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without ``plugin``, ``render_params`` still loads the plugin and preset per call."""
        fake_plugin = self._fake_plugin(audio_shape=(2, 16))
        load_calls: list[str] = []

        def _capture_load(path: str, **_kw: object) -> MagicMock:
            load_calls.append(path)
            return fake_plugin

        preset_calls: list[tuple[object, str]] = []
        monkeypatch.setattr(core, "load_plugin", _capture_load)
        monkeypatch.setattr(
            core,
            "load_preset",
            lambda plugin, path: preset_calls.append((plugin, path)),
        )

        render_params(
            "plugins/Surge XT.vst3",
            params={},
            midi_note=60,
            velocity=100,
            note_start_and_end=(0.0, 1.0),
            signal_duration_seconds=1.0,
            sample_rate=16000,
            channels=2,
            preset_path="presets/surge-base.vstpreset",
        )

        assert load_calls == ["plugins/Surge XT.vst3"]
        assert preset_calls == [(fake_plugin, "presets/surge-base.vstpreset")]

    def test_open_gui_flag_threads_into_load_plugin(  # noqa: DOC101,DOC103
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``open_gui`` kwarg is forwarded to ``load_plugin`` when no plugin is supplied."""
        captured: dict[str, object] = {}

        def _capture_load(path: str, **kwargs: object) -> MagicMock:
            captured["path"] = path
            captured.update(kwargs)
            return self._fake_plugin(audio_shape=(2, 16))

        monkeypatch.setattr(core, "load_plugin", _capture_load)
        monkeypatch.setattr(core, "load_preset", lambda *_a, **_kw: None)

        render_params(
            "plugins/Surge XT.vst3",
            params={},
            midi_note=60,
            velocity=100,
            note_start_and_end=(0.0, 1.0),
            signal_duration_seconds=1.0,
            sample_rate=16000,
            channels=2,
            preset_path="presets/surge-base.vstpreset",
            open_gui=False,
        )

        assert captured == {"path": "plugins/Surge XT.vst3", "open_gui": False}
