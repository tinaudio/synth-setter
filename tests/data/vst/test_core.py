from __future__ import annotations

import plistlib
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from synth_setter.data.vst import core
from synth_setter.data.vst.core import (
    extract_renderer_version,
    load_plugin,
    render_params,
    warmup_plugin,
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


class TestLoadPluginNoWarmup:
    """``load_plugin`` is a pure loader — never calls ``show_editor`` by itself."""

    def test_load_plugin_does_not_call_show_editor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``load_plugin`` only constructs ``VST3Plugin``; warm-up lives in ``warmup_plugin``.

        :param monkeypatch: Pytest fixture used to patch attributes / env / argv.
        """
        fake_plugin = MagicMock()
        monkeypatch.setattr(core, "VST3Plugin", lambda _path: fake_plugin)

        load_plugin("plugins/Surge XT.vst3")

        fake_plugin.show_editor.assert_not_called()


class TestWarmupPlugin:
    """``warmup_plugin`` runs ``show_editor`` on the passed-in plugin once."""

    def test_warmup_plugin_calls_show_editor(self) -> None:
        """``warmup_plugin`` invokes ``show_editor`` exactly once on the supplied plugin."""
        fake_plugin = MagicMock()

        warmup_plugin(fake_plugin)

        fake_plugin.show_editor.assert_called_once()


class TestEditorHeldOpen:
    """``editor_held_open`` opens the plugin editor on a background thread for the ``with``
    body."""

    def test_opens_once_and_closes_on_exit(self) -> None:
        """``show_editor`` is called once on a background thread; close_event set on
        ``__exit__``."""
        fake_plugin = MagicMock()
        captured_event: list[threading.Event] = []

        def record_event(event: threading.Event) -> None:
            captured_event.append(event)
            event.wait(timeout=5.0)

        fake_plugin.show_editor.side_effect = record_event

        with core.editor_held_open(fake_plugin):
            pass

        fake_plugin.show_editor.assert_called_once()
        assert len(captured_event) == 1
        assert captured_event[0].is_set()

    def test_logs_and_reraises_thread_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An exception in ``show_editor`` is logged immediately and re-raised at ``__exit__``.

        :param monkeypatch: Stubs ``core.logger`` so the log call can be observed
            (loguru does not propagate to ``caplog``'s stdlib handler).
        """
        fake_plugin = MagicMock()
        fake_plugin.show_editor.side_effect = RuntimeError("xserver gone")
        fake_logger = MagicMock()
        monkeypatch.setattr(core, "logger", fake_logger)

        with pytest.raises(RuntimeError, match="xserver gone"):
            with core.editor_held_open(fake_plugin):
                time.sleep(0.05)  # let the editor thread run + raise

        assert fake_logger.exception.call_count == 1
        assert "vst-editor-window crashed" in fake_logger.exception.call_args.args[0]

    def test_body_exception_wins_over_editor_exception(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If both the ``with`` body and the editor thread raise, the body's exception propagates.

        Re-raising the captured editor exception inside the ``finally`` clause
        would otherwise mask the body exception (raise-in-finally wins). The
        editor crash still gets a structured error log so it is not lost.

        :param monkeypatch: Stubs ``core.logger`` so the editor crash log can
            be observed.
        :raises ValueError: Intentionally raised inside the ``with`` body to
            exercise the body-wins precedence path under test.
        """
        fake_plugin = MagicMock()
        fake_plugin.show_editor.side_effect = RuntimeError("editor crashed")
        fake_logger = MagicMock()
        monkeypatch.setattr(core, "logger", fake_logger)

        with pytest.raises(ValueError, match="body crashed"):
            with core.editor_held_open(fake_plugin):
                time.sleep(0.05)  # let the editor thread run + raise
                raise ValueError("body crashed")

        # editor crash still surfaces via the structured error log so it is
        # not lost when the body exception takes precedence.
        assert any(
            "also crashed during body exception" in str(call.args[0])
            for call in fake_logger.error.call_args_list
        )

    def test_waits_for_editor_thread_start_before_yield(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``editor_started`` is set before the ``with`` body runs — see #1198.

        Without the handshake the body can begin before the editor thread
        enters ``show_editor``. We delay the thread's start so a body that
        ran immediately would observe ``inside_show_editor`` unset; the
        handshake forces the wait, so the assertion sees it set.

        :param monkeypatch: Wraps ``threading.Thread`` to delay ``start`` and
            expose the race the handshake closes.
        """
        fake_plugin = MagicMock()
        inside_show_editor = threading.Event()

        def block_inside_show_editor(event: threading.Event) -> None:
            inside_show_editor.set()
            event.wait(timeout=5.0)

        fake_plugin.show_editor.side_effect = block_inside_show_editor

        real_thread_cls = threading.Thread

        def make_delayed_thread(*args: object, **kwargs: object) -> threading.Thread:
            t = real_thread_cls(*args, **kwargs)  # type: ignore[arg-type]
            original_start = t.start

            def delayed_start() -> None:
                def deferred() -> None:
                    time.sleep(0.05)
                    original_start()

                real_thread_cls(target=deferred, daemon=True).start()

            t.start = delayed_start  # type: ignore[method-assign]
            return t

        monkeypatch.setattr(threading, "Thread", make_delayed_thread)

        with core.editor_held_open(fake_plugin):
            assert inside_show_editor.is_set()

    def test_slow_start_warns_and_proceeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Editor thread that misses the start handshake logs a warning and the body still runs.

        Simulates a thread that delays starting past
        ``_EDITOR_START_TIMEOUT_SECONDS``. The context manager must not raise;
        it warns and proceeds so the render still happens — best-effort
        handshake (see #1198).

        :param monkeypatch: Tightens ``_EDITOR_START_TIMEOUT_SECONDS`` and stubs
            ``core.logger`` so the slow-start warning assertion is observable.
        """
        monkeypatch.setattr(core, "_EDITOR_START_TIMEOUT_SECONDS", 0.05)
        fake_logger = MagicMock()
        monkeypatch.setattr(core, "logger", fake_logger)

        fake_plugin = MagicMock()
        show_editor_entered = threading.Event()

        def show_editor_blocks(event: threading.Event) -> None:
            show_editor_entered.set()
            event.wait(timeout=5.0)

        fake_plugin.show_editor.side_effect = show_editor_blocks

        real_thread_cls = threading.Thread
        release_real_start = threading.Event()

        def make_slow_thread(*args: object, **kwargs: object) -> threading.Thread:
            t = real_thread_cls(*args, **kwargs)  # type: ignore[arg-type]
            original_start = t.start

            def slow_start() -> None:
                # Defer real start until the handshake timeout has been
                # observed; an order of magnitude under any join timeout.
                def deferred() -> None:
                    release_real_start.wait(timeout=1.0)
                    original_start()

                real_thread_cls(target=deferred, daemon=True).start()

            t.start = slow_start  # type: ignore[method-assign]
            return t

        monkeypatch.setattr(threading, "Thread", make_slow_thread)

        body_ran = False
        with core.editor_held_open(fake_plugin):
            body_ran = True
            # Let the deferred real start fire and the editor thread reach
            # ``show_editor`` so the finally-clause join sees a started
            # thread; production threads start synchronously — the delay is
            # a test-only artifact of the slow-start stub.
            release_real_start.set()
            assert show_editor_entered.wait(timeout=1.0)

        assert body_ran
        assert any(
            "did not signal start" in str(call.args[0])
            for call in fake_logger.warning.call_args_list
        )

    def test_join_timeout_does_not_deadlock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If ``show_editor`` ignores the close event, ``__exit__`` returns within the timeout.

        :param monkeypatch: Tightens ``_EDITOR_JOIN_TIMEOUT_SECONDS`` and stubs
            ``core.logger`` so the leak-warning assertion is observable.
        """
        monkeypatch.setattr(core, "_EDITOR_JOIN_TIMEOUT_SECONDS", 0.1)
        fake_logger = MagicMock()
        monkeypatch.setattr(core, "logger", fake_logger)
        fake_plugin = MagicMock()
        fake_plugin.show_editor.side_effect = lambda _event: time.sleep(2.0)

        start = time.monotonic()
        with core.editor_held_open(fake_plugin):
            pass
        elapsed = time.monotonic() - start

        # 1s slack over the 0.1s timeout to absorb CI scheduler jitter; still
        # an order of magnitude under the 2.0s `show_editor` sleep so a
        # regression to "wait for the thread" would fail this assertion.
        assert elapsed < 1.0
        assert fake_logger.warning.call_count == 1
        assert "did not drain" in fake_logger.warning.call_args.args[0]


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

    def test_preloaded_plugin_bypasses_load_and_preset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When ``plugin`` is supplied, ``load_plugin`` and ``load_preset`` are not called.

        :param monkeypatch: Pytest fixture used to patch attributes / env / argv.
        """
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

    def test_no_plugin_kwarg_reloads_per_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without ``plugin``, ``render_params`` still loads the plugin and preset per call.

        :param monkeypatch: Pytest fixture used to patch attributes / env / argv.
        """
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

    def test_warmup_kwarg_runs_warmup_plugin_after_load(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``warmup=True`` calls ``warmup_plugin`` on the freshly-loaded plugin.

        :param monkeypatch: Pytest fixture used to patch attributes / env / argv.
        """
        fake_plugin = self._fake_plugin(audio_shape=(2, 16))
        warmup_calls: list[object] = []

        monkeypatch.setattr(core, "load_plugin", lambda _path: fake_plugin)
        monkeypatch.setattr(core, "load_preset", lambda *_a, **_kw: None)
        monkeypatch.setattr(core, "warmup_plugin", lambda plugin: warmup_calls.append(plugin))

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
            warmup=True,
        )

        assert warmup_calls == [fake_plugin]

    def test_warmup_kwarg_runs_warmup_on_supplied_plugin(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``warmup=True`` with a cached plugin warms the cached instance, not a fresh load.

        :param monkeypatch: Pytest fixture used to patch attributes / env / argv.
        """
        cached = self._fake_plugin(audio_shape=(2, 16))
        warmup_calls: list[object] = []

        monkeypatch.setattr(core, "warmup_plugin", lambda plugin: warmup_calls.append(plugin))

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
            plugin=cached,
            warmup=True,
        )

        assert warmup_calls == [cached]
