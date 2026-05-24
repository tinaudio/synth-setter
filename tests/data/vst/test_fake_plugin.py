"""Contract tests for ``FakeVST3Plugin`` — the duck-typed E2E test double."""

import threading

import numpy as np

from synth_setter.data.vst import core
from tests.data.vst._fake_plugin import FakeVST3Plugin


def test_constructor_and_version_attribute() -> None:
    """A freshly constructed fake exposes a non-empty ``version`` string."""
    plugin = FakeVST3Plugin("plugins/anything.vst3")
    assert isinstance(plugin.version, str) and plugin.version


def test_parameters_subscript_assignment_round_trips() -> None:
    """Writing ``parameters[k].raw_value`` is observable on subsequent reads."""
    plugin = FakeVST3Plugin("plugins/anything.vst3")
    plugin.parameters["cutoff"].raw_value = 0.5
    assert plugin.parameters["cutoff"].raw_value == 0.5


def test_parameters_auto_create_on_first_access() -> None:
    """Unknown parameter keys materialise on first access with raw_value ``0.0``."""
    plugin = FakeVST3Plugin("plugins/anything.vst3")
    assert plugin.parameters["never_seen"].raw_value == 0.0


def test_load_preset_is_a_noop() -> None:
    """``load_preset`` accepts any string and does not touch the filesystem."""
    plugin = FakeVST3Plugin("plugins/anything.vst3")
    plugin.load_preset("presets/anything.vstpreset")


def test_reset_is_a_noop() -> None:
    """``reset`` is callable on a fresh fake and never raises."""
    plugin = FakeVST3Plugin("plugins/anything.vst3")
    plugin.reset()


def test_process_with_empty_midi_returns_zero_length_float32_buffer() -> None:
    """Flush calls (``midi_events=[]``) return a zero-length float32 buffer.

    The production pipeline discards the flush return value, so allocating a full-duration zero
    array would waste ~11MB per call at 44.1 kHz / 32 s.
    """
    plugin = FakeVST3Plugin("plugins/anything.vst3")
    out = plugin.process([], 32.0, 44100.0, 2, 2048, True)
    assert out.shape == (2, 0)
    assert out.dtype == np.float32


def test_process_with_note_returns_audio_above_loudness_gate() -> None:
    """Rendered audio must clear the dataset pipeline's ``_MIN_LOUDNESS=-55 dB`` gate.

    Otherwise the loudness filter in ``test_generate_vst_dataset.py`` rejects
    every sample and the E2E shard write fails.
    """
    plugin = FakeVST3Plugin("plugins/anything.vst3")
    note_on = (bytes((0x90, 60, 100)), 0.0)
    note_off = (bytes((0x80, 60, 0)), 1.0)
    out = plugin.process((note_on, note_off), 1.0, 44100.0, 2, 2048, True)

    assert out.shape == (2, 44100)
    assert out.dtype == np.float32
    rms_db = 20.0 * np.log10(np.sqrt(np.mean(out**2)) + 1e-12)
    assert rms_db > -55.0, f"rms {rms_db:.2f} dB below loudness gate"
    assert np.max(np.abs(out)) <= 1.0


def test_process_is_deterministic_across_calls() -> None:
    """Same inputs to ``process`` must produce bit-identical outputs."""
    plugin = FakeVST3Plugin("plugins/anything.vst3")
    note_on = (bytes((0x90, 67, 80)), 0.0)
    args = ((note_on,), 0.25, 44100.0, 2, 2048, True)
    first = plugin.process(*args)
    second = plugin.process(*args)
    assert np.array_equal(first, second)


def test_show_editor_blocks_until_close_event_is_set() -> None:
    """``show_editor`` returns only after the host signals ``close_event.set()``."""
    plugin = FakeVST3Plugin("plugins/anything.vst3")
    close_event = threading.Event()
    entered_wait = threading.Event()
    returned = threading.Event()

    # Wrap ``close_event.wait`` so the test observes the exact moment the
    # worker enters the blocking call — avoids any timing-based race.
    original_wait = close_event.wait

    def _instrumented_wait(timeout: float | None = None) -> bool:
        entered_wait.set()
        return original_wait(timeout)

    close_event.wait = _instrumented_wait  # type: ignore[method-assign]

    def _run() -> None:
        plugin.show_editor(close_event)
        returned.set()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    assert entered_wait.wait(timeout=1.0), "worker thread never entered show_editor"
    assert not returned.is_set(), "show_editor returned before close_event was set"
    close_event.set()
    assert returned.wait(timeout=1.0), "show_editor did not return after close_event"


def test_install_fake_plugin_redirects_core_load_plugin(
    install_fake_plugin: FakeVST3Plugin,
) -> None:
    """After ``install_fake_plugin``, ``core.load_plugin`` returns the fake instance.

    :param install_fake_plugin: Fixture that monkeypatches the loader and yields the instance it
        now returns; identity check pins the contract.
    """

    assert core.load_plugin("plugins/does-not-exist.vst3") is install_fake_plugin
