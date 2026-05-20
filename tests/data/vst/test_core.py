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
        """``with`` body entry is gated on the start handshake — see #1198.

        The fake editor thread blocks behind ``release_start`` before
        running ``_run_editor``, so without the handshake the body would
        enter while ``editor_started`` is still unset. The body's entry
        timestamp must therefore land after ``release_start`` was set,
        which is the only signal that unblocks ``editor_started.wait``.

        :param monkeypatch: Wraps ``threading.Thread`` so the editor
            thread's target blocks on a gate; isolates the timing
            contract from scheduler luck.
        """
        fake_plugin = MagicMock()
        fake_plugin.show_editor.side_effect = lambda event: event.wait(timeout=5.0)

        release_start = threading.Event()
        real_thread_cls = threading.Thread

        def make_gated_thread(*args: object, **kwargs: object) -> threading.Thread:
            user_target = kwargs.pop("target", None)

            def gated_target() -> None:
                release_start.wait(timeout=5.0)
                if callable(user_target):
                    user_target()

            return real_thread_cls(*args, target=gated_target, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(threading, "Thread", make_gated_thread)

        release_at: list[float] = []

        def release_after_delay() -> None:
            time.sleep(0.1)
            release_at.append(time.monotonic())
            release_start.set()

        releaser = real_thread_cls(target=release_after_delay, daemon=True)
        releaser.start()

        with core.editor_held_open(fake_plugin):
            t_body_entered = time.monotonic()

        releaser.join(timeout=1.0)
        assert release_at, "releaser did not run"
        assert t_body_entered >= release_at[0], (
            "with-body entered before the start handshake fired"
        )

    def test_without_handshake_body_enters_before_editor_thread_starts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Inverse of the positive test: with the handshake disabled, the body wins the race.

        Pins the negative side of #1198. Stubs the handshake wait so it
        returns ``True`` without firing the raise-on-miss path (#1204) — the
        body must then enter before the gated editor thread fires, proving
        the positive test would fail if the handshake were removed.

        :param monkeypatch: Wraps ``threading.Event.__init__`` so the second
            event created (the handshake event in ``editor_held_open``) has
            its ``wait`` short-circuited; wraps ``threading.Thread`` so the
            editor thread blocks on a gate.
        """
        fake_plugin = MagicMock()
        fake_plugin.show_editor.side_effect = lambda event: event.wait(timeout=5.0)

        release_start = threading.Event()
        real_thread_cls = threading.Thread

        def make_gated_thread(*args: object, **kwargs: object) -> threading.Thread:
            user_target = kwargs.pop("target", None)

            def gated_target() -> None:
                release_start.wait(timeout=5.0)
                if callable(user_target):
                    user_target()

            return real_thread_cls(*args, target=gated_target, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(threading, "Thread", make_gated_thread)

        # Short-circuit the handshake by patching ``wait`` on the second
        # Event created in ``editor_held_open`` (``close_editor`` is first,
        # ``editor_started`` is second). Targeting the instance keeps the
        # close_editor / release_start waits running real code.
        events_created: list[threading.Event] = []
        original_init = threading.Event.__init__

        def tracking_init(self: threading.Event, *args: object, **kwargs: object) -> None:
            original_init(self, *args, **kwargs)  # type: ignore[arg-type]
            events_created.append(self)
            if len(events_created) == 2:
                self.wait = lambda timeout=None: True  # type: ignore[method-assign,assignment]

        monkeypatch.setattr(threading.Event, "__init__", tracking_init)

        with core.editor_held_open(fake_plugin):
            # Body entered with the editor thread still gated — the handshake
            # (not scheduler luck) is what makes the positive test pass.
            assert not release_start.is_set()
            release_start.set()

    def test_slow_start_raises_editor_start_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Editor thread that misses the start handshake raises ``EditorStartTimeout``.

        Simulates a thread that delays starting past
        ``_EDITOR_START_TIMEOUT_SECONDS``. The context manager must raise
        before yielding so the body never runs — a slow editor bring-up is a
        loud, traceable failure rather than a warning lost in noise (#1204).

        :param monkeypatch: Tightens ``_EDITOR_START_TIMEOUT_SECONDS`` and
            wraps ``threading.Thread`` so the editor thread's real start is
            deferred past the handshake deadline.
        """
        monkeypatch.setattr(core, "_EDITOR_START_TIMEOUT_SECONDS", 0.05)

        fake_plugin = MagicMock()
        fake_plugin.show_editor.side_effect = lambda event: event.wait(timeout=5.0)

        # Defer ``editor_started.set()`` by 0.2s — the second ``Event`` created
        # by ``editor_held_open`` (``close_editor`` is first) — so the handshake
        # misses its 0.05s deadline while the real thread still starts and
        # ``show_editor`` honours ``close_editor``, letting the finally-clause
        # teardown join a real started daemon under ``_EDITOR_JOIN_TIMEOUT_SECONDS``
        # (#1204).
        events_created: list[threading.Event] = []
        original_event_init = threading.Event.__init__

        def init_capture(self: threading.Event) -> None:
            original_event_init(self)
            events_created.append(self)
            if len(events_created) == 2:
                original_set = self.set

                def deferred_set() -> None:
                    time.sleep(0.2)
                    original_set()

                self.set = deferred_set  # type: ignore[method-assign]

        monkeypatch.setattr(threading.Event, "__init__", init_capture)

        body_ran = False
        with pytest.raises(core.EditorStartTimeout, match="did not signal start"):
            with core.editor_held_open(fake_plugin):
                body_ran = True

        assert not body_ran, "body must not run when start handshake misses"

    def test_slow_start_joins_editor_thread_before_raising(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Timeout path runs the same teardown (close + join + leak warn) as the success path.

        Without the shared finalization the editor thread can start late and
        call ``show_editor`` after the context manager has already raised,
        leaking work past the failed context-entry. Captures the real
        ``threading.Thread`` instance handed to ``editor_held_open`` and
        asserts ``is_alive()`` is ``False`` once ``EditorStartTimeout``
        propagates — proving the timeout path joined the daemon under the
        same bounded teardown as the success path (#1204).

        :param monkeypatch: Tightens ``_EDITOR_START_TIMEOUT_SECONDS``, defers
            the ``editor_started.set()`` call past the handshake deadline,
            and captures the real thread instance so ``is_alive()`` is
            observable post-raise.
        """
        monkeypatch.setattr(core, "_EDITOR_START_TIMEOUT_SECONDS", 0.05)

        fake_plugin = MagicMock()
        fake_plugin.show_editor.side_effect = lambda event: event.wait(timeout=5.0)

        events_created: list[threading.Event] = []
        original_event_init = threading.Event.__init__

        def init_capture(self: threading.Event) -> None:
            original_event_init(self)
            events_created.append(self)
            if len(events_created) == 2:
                original_set = self.set

                def deferred_set() -> None:
                    time.sleep(0.2)
                    original_set()

                self.set = deferred_set  # type: ignore[method-assign]

        monkeypatch.setattr(threading.Event, "__init__", init_capture)

        real_thread_cls = threading.Thread
        captured_threads: list[threading.Thread] = []

        def capture_thread(*args: object, **kwargs: object) -> threading.Thread:
            t = real_thread_cls(*args, **kwargs)  # type: ignore[arg-type]
            captured_threads.append(t)
            return t

        monkeypatch.setattr(threading, "Thread", capture_thread)

        with pytest.raises(core.EditorStartTimeout):
            with core.editor_held_open(fake_plugin):
                pass

        assert captured_threads, "editor thread was not captured"
        editor_thread = captured_threads[0]
        assert not editor_thread.is_alive(), (
            "editor thread leaked past EditorStartTimeout — join missing on timeout path"
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
