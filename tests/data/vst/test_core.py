from __future__ import annotations

import plistlib
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import numpy as np
import pytest

from synth_setter.data.vst import core
from synth_setter.data.vst.core import (
    RenderWorkerLeaked,
    extract_renderer_version,
    load_plugin,
    render_params,
    warmup_plugin,
)
from tests.data.vst._fake_plugin import FakeVST3Plugin

if TYPE_CHECKING:
    from pedalboard import VST3Plugin


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
    """``warmup_plugin`` runs ``show_editor`` once and sets the close event so it returns."""

    def test_warmup_plugin_opens_editor_once_and_sets_close_event(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``warmup_plugin`` opens the editor once and signals close so the call unblocks.

        Records the close event and waits on it with a bounded timeout, so a
        regressed warmup that never sets the event fails fast instead of
        deadlocking the suite. ``warmup_plugin`` returning proves it set the
        event; the recorded event's ``is_set()`` pins that observable effect
        rather than merely asserting the method was invoked.

        :param monkeypatch: Shrinks ``_EDITOR_INIT_DELAY_SECONDS`` so the
            production close-timer fires promptly — the delay value is not under
            test, only that the event ends up set.
        """
        monkeypatch.setattr(core, "_EDITOR_INIT_DELAY_SECONDS", 0.0)
        opened: list[threading.Event] = []

        class _RecordingPlugin(FakeVST3Plugin):
            def show_editor(self, close_event: threading.Event) -> None:
                """Record the close event and wait on it with a bounded timeout.

                :param close_event: Event ``warmup_plugin`` must set to unblock.
                """
                opened.append(close_event)
                # Bounded wait: a warmup that never sets the event fails fast
                # rather than hanging on FakeVST3Plugin's unbounded wait().
                assert close_event.wait(timeout=5.0), "warmup_plugin did not set the close event"

        warmup_plugin(cast("VST3Plugin", _RecordingPlugin("plugins/Surge XT.vst3")))

        assert len(opened) == 1
        assert opened[0].is_set()


class TestRunWithEditorHeldOpen:
    """``run_with_editor_held_open`` runs ``body()`` on a worker while the caller blocks in
    ``show_editor``."""

    def test_body_runs_on_worker_thread_and_returns_value(self) -> None:
        """``body()`` runs off the caller thread; its return value propagates.

        Captures the thread identity inside ``body`` and asserts it differs
        from the calling thread — pedalboard requires ``show_editor`` on the
        caller (main) thread, so the render work MUST execute elsewhere.
        """
        fake_plugin = MagicMock()
        fake_plugin.show_editor.side_effect = lambda event: event.wait(timeout=5.0)

        caller_ident = threading.get_ident()
        body_ident: list[int] = []

        def body() -> str:
            body_ident.append(threading.get_ident())
            return "done"

        result = core.run_with_editor_held_open(fake_plugin, body)

        assert result == "done"
        assert body_ident and body_ident[0] != caller_ident
        fake_plugin.show_editor.assert_called_once()

    def test_close_event_is_set_after_body_returns(self) -> None:
        """The event passed to ``show_editor`` is set so the main thread unblocks."""
        fake_plugin = MagicMock()
        captured_event: list[threading.Event] = []

        def record_event(event: threading.Event) -> None:
            captured_event.append(event)
            event.wait(timeout=5.0)

        fake_plugin.show_editor.side_effect = record_event

        core.run_with_editor_held_open(fake_plugin, lambda: None)

        assert len(captured_event) == 1
        assert captured_event[0].is_set()

    def test_body_exception_propagates_after_show_editor_returns(self) -> None:
        """An exception raised inside ``body`` propagates to the caller."""
        fake_plugin = MagicMock()
        fake_plugin.show_editor.side_effect = lambda event: event.wait(timeout=5.0)

        def body() -> None:
            raise ValueError("render failed")

        with pytest.raises(ValueError, match="render failed"):
            core.run_with_editor_held_open(fake_plugin, body)

        fake_plugin.show_editor.assert_called_once()

    def test_show_editor_exception_propagates_through_finally(self) -> None:
        """A ``show_editor`` failure on the caller thread surfaces; the worker is joined."""
        fake_plugin = MagicMock()
        fake_plugin.show_editor.side_effect = RuntimeError("xserver gone")

        worker_ran = threading.Event()

        def body() -> None:
            worker_ran.set()

        with pytest.raises(RuntimeError, match="xserver gone"):
            core.run_with_editor_held_open(fake_plugin, body)

        assert worker_ran.is_set()

    def test_run_with_editor_held_open_raises_on_worker_leak(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A worker that outlives the join window after ``show_editor`` returns raises.

        Simulates the leak case: ``show_editor`` returns immediately and the
        worker body keeps running past ``_EDITOR_JOIN_TIMEOUT_SECONDS``. The
        helper must raise ``RenderWorkerLeaked`` rather than logging-and-
        returning, so callers like ``_render_in_batches`` cannot continue
        while renders are still in flight on the background thread.

        :param monkeypatch: Tightens ``_EDITOR_JOIN_TIMEOUT_SECONDS`` so the
            slow body outlives the join window deterministically.
        """
        monkeypatch.setattr(core, "_EDITOR_JOIN_TIMEOUT_SECONDS", 0.1)
        fake_plugin = MagicMock()
        fake_plugin.show_editor.return_value = None

        worker_release = threading.Event()

        def body() -> None:
            worker_release.wait(timeout=5.0)

        start = time.monotonic()
        try:
            with pytest.raises(RenderWorkerLeaked, match="still alive"):
                core.run_with_editor_held_open(fake_plugin, body)
        finally:
            worker_release.set()
        elapsed = time.monotonic() - start

        # 1s slack over the 0.1s timeout to absorb CI scheduler jitter.
        assert elapsed < 1.0

    def test_run_with_editor_held_open_propagates_body_exception_even_on_slow_finish(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Body exceptions take precedence over the leak signal.

        A body that raises must surface its own exception even when the
        worker takes long enough to be at risk of tripping the leak path —
        the body reached a terminal state, so the caller sees the real
        failure rather than a synthetic ``RenderWorkerLeaked``.

        :param monkeypatch: Tightens ``_EDITOR_JOIN_TIMEOUT_SECONDS`` so the
            test exercises the precedence rule near the join boundary.
        """
        monkeypatch.setattr(core, "_EDITOR_JOIN_TIMEOUT_SECONDS", 1.0)
        fake_plugin = MagicMock()

        # Two-event handshake pins the ordering deterministically (no sleep):
        # worker sets ``body_started`` → ``show_editor`` returns and sets
        # ``editor_returned`` → only then does the body raise. The body thus
        # reaches its terminal state strictly *after* ``show_editor`` returns,
        # while the caller is in ``worker.join`` — the "slow finish" case that
        # exercises body-exception precedence over the leak path.
        body_started = threading.Event()
        editor_returned = threading.Event()

        def show_editor(_event: threading.Event) -> None:
            """Return once the worker has started, signalling the body to raise.

            :param _event: Close event from the helper; unused by this stub.
            """
            body_started.wait(timeout=5.0)
            editor_returned.set()

        fake_plugin.show_editor.side_effect = show_editor

        def body() -> None:
            """Raise only after ``show_editor`` has returned, mid-join.

            :raises ValueError: Always, once the editor has been released.
            """
            body_started.set()
            editor_returned.wait(timeout=5.0)
            raise ValueError("body slow-failed")

        with pytest.raises(ValueError, match="body slow-failed"):
            core.run_with_editor_held_open(fake_plugin, body)

    def test_run_with_editor_held_open_no_silent_none_return(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``None`` from a clean body returns; an empty-result clean exit raises distinctly.

        Two paths share the ``not result`` shape: a body that returns
        explicit ``None`` (legitimate) and a worker that exited without
        producing a value or capturing an exception (a bug — the body must
        either return or raise). The first must propagate ``None``; the
        second must raise ``RenderWorkerLeaked`` with a message that
        distinguishes it from the join-timeout case.

        :param monkeypatch: Stubs ``threading.Thread`` for the empty-result
            branch so the helper observes ``not result`` with a dead worker.
        """
        fake_plugin = MagicMock()
        fake_plugin.show_editor.side_effect = lambda event: event.wait(timeout=5.0)

        assert core.run_with_editor_held_open(fake_plugin, lambda: None) is None

        class _DeadThread:
            def __init__(self, *_args: object, **_kwargs: object) -> None: ...

            def start(self) -> None: ...

            def join(self, timeout: float | None = None) -> None: ...

            def is_alive(self) -> bool:
                return False

        monkeypatch.setattr(core.threading, "Thread", _DeadThread)
        fake_plugin_empty = MagicMock()
        fake_plugin_empty.show_editor.return_value = None

        with pytest.raises(RenderWorkerLeaked, match="without producing a result"):
            core.run_with_editor_held_open(fake_plugin_empty, lambda: "ignored")


class TestRenderParamsPreloadedPlugin:
    """``render_params`` accepts a pre-loaded plugin and skips load/preset on that path."""

    @staticmethod
    def _fake_plugin() -> FakeVST3Plugin:
        return FakeVST3Plugin("plugins/Surge XT.vst3")

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

        preloaded = self._fake_plugin()

        output = render_params(
            "plugins/Surge XT.vst3",
            params={},
            midi_note=60,
            velocity=100,
            note_start_and_end=(0.0, 1.0),
            signal_duration_seconds=1.0,
            sample_rate=44100,
            channels=2,
            preset_path="presets/surge-base.vstpreset",
            plugin=cast("VST3Plugin", preloaded),
        )

        assert load_calls == []
        assert preset_calls == []
        # Non-silent audio proves the pre-loaded plugin ran the note-on render.
        assert np.any(output)

    def test_no_plugin_kwarg_reloads_per_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without ``plugin``, ``render_params`` still loads the plugin and preset per call.

        :param monkeypatch: Pytest fixture used to patch attributes / env / argv.
        """
        fake_plugin = self._fake_plugin()
        load_calls: list[str] = []

        def _capture_load(path: str, **_kw: object) -> FakeVST3Plugin:
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
            sample_rate=44100,
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
        fake_plugin = self._fake_plugin()
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
            sample_rate=44100,
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
        cached = self._fake_plugin()
        warmup_calls: list[object] = []

        monkeypatch.setattr(core, "warmup_plugin", lambda plugin: warmup_calls.append(plugin))

        render_params(
            "plugins/Surge XT.vst3",
            params={},
            midi_note=60,
            velocity=100,
            note_start_and_end=(0.0, 1.0),
            signal_duration_seconds=1.0,
            sample_rate=44100,
            channels=2,
            preset_path="presets/surge-base.vstpreset",
            plugin=cast("VST3Plugin", cached),
            warmup=True,
        )

        assert warmup_calls == [cached]
