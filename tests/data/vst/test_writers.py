"""CPU-only tests for ``synth_setter.data.vst.writers``.

Covers the writer module's pure helpers and the CLI dispatcher in
``generate_vst_dataset.main`` — the VST-dependent end-to-end writer tests
live alongside the legacy HDF5 tests in ``test_generate_vst_dataset.py`` and
the new wds e2e tests in ``test_writers_wds_e2e.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from synth_setter.data.vst import writers
from synth_setter.data.vst.generate_vst_dataset import VSTDataSample
from synth_setter.data.vst.writers import _render_in_batches, _shard_metadata_from_render
from synth_setter.pipeline.schemas.shard_metadata import ShardMetadata
from synth_setter.pipeline.schemas.spec import RenderConfig


def _smoke_render_cfg(**overrides: object) -> RenderConfig:  # noqa: DOC101,DOC103,DOC201,DOC203
    """Build a syntactically-valid ``RenderConfig`` for CPU-only tests.

    No I/O happens against ``plugin_path`` or ``preset_path`` in these tests
    — they only need to be non-blank strings.
    """
    kwargs: dict[str, object] = {
        "plugin_path": "plugins/Surge XT.vst3",
        "preset_path": "presets/surge-base.vstpreset",
        "param_spec_name": "surge_simple",
        "renderer_version": "1.3.4",
        "sample_rate": 16000,
        "channels": 2,
        "velocity": 100,
        "signal_duration_seconds": 4.0,
        "min_loudness": -55.0,
        "samples_per_render_batch": 2,
        "samples_per_shard": 4,
        # Darwin-portable (#714).
        "gui_toggle_cadence": "never",
    }
    kwargs.update(overrides)
    return RenderConfig(**kwargs)  # type: ignore[arg-type]


def test_shard_metadata_from_render_projects_five_fields() -> None:
    """``_shard_metadata_from_render`` returns a strict ``ShardMetadata`` with renderer values."""
    render_cfg = _smoke_render_cfg(
        velocity=64,
        signal_duration_seconds=2.5,
        sample_rate=22050,
        channels=1,
        min_loudness=-40.0,
    )

    meta = _shard_metadata_from_render(render_cfg)

    assert isinstance(meta, ShardMetadata)
    assert meta.velocity == 64
    assert meta.signal_duration_seconds == 2.5
    assert meta.sample_rate == 22050
    assert meta.channels == 1
    assert meta.min_loudness == -40.0


def test_shard_metadata_from_render_round_trips_through_json() -> None:
    """The projected metadata serializes and re-validates as a strict ``ShardMetadata``.

    Pinning JSON round-trip is what the wds tar's ``metadata.json`` member
    relies on: a writer-side projection that can't be re-read isn't useful.
    """
    render_cfg = _smoke_render_cfg()

    meta = _shard_metadata_from_render(render_cfg)
    rehydrated = ShardMetadata.model_validate_json(meta.model_dump_json())

    assert rehydrated == meta


def _run_main_with_argv(argv: list[str]) -> None:  # noqa: DOC101,DOC103
    """Invoke ``generate_vst_dataset.main`` with ``argv`` patched in.

    The pydantic-settings CLI reads ``sys.argv`` directly via ``CliApp.run``,
    so tests need to swap the process argv around the call. Imports the entry
    inside the helper so a single import failure doesn't poison the module.
    """
    from synth_setter.data.vst.generate_vst_dataset import main

    with patch.object(sys, "argv", argv):
        main()


# Shared CLI argv prefix for the dispatcher tests below. Built from the same
# ``RenderConfig`` field set the CLI binding inherits, so adding a render-config
# field auto-extends the prefix.
def _cli_argv(data_file: str) -> list[str]:  # noqa: DOC101,DOC103,DOC201,DOC203
    """Build a CLI argv that parses cleanly into a ``RenderConfig`` + ``data_file``.

    All values mirror ``_smoke_render_cfg`` so the parsed config is round-trip
    equal to it. The ``argv[0]`` is a stand-in program name (not used).
    """
    return [
        "generate_vst_dataset",
        data_file,
        "--plugin_path",
        "plugins/Surge XT.vst3",
        "--preset_path",
        "presets/surge-base.vstpreset",
        "--param_spec_name",
        "surge_simple",
        "--renderer_version",
        "1.3.4",
        "--sample_rate",
        "16000",
        "--channels",
        "2",
        "--velocity",
        "100",
        "--signal_duration_seconds",
        "4.0",
        "--min_loudness",
        "-55.0",
        "--samples_per_render_batch",
        "2",
        "--samples_per_shard",
        "4",
        # Mirror _smoke_render_cfg (#714).
        "--gui_toggle_cadence",
        "never",
    ]


def test_main_dispatches_h5_suffix_to_make_hdf5_dataset(tmp_path: Path) -> None:  # noqa: DOC101,DOC103
    """``data_file=foo.h5`` routes to ``make_hdf5_dataset`` (not the wds writer)."""
    data_file = tmp_path / "shard-000000.h5"

    with (
        patch("synth_setter.data.vst.writers.make_hdf5_dataset") as mock_h5,
        patch("synth_setter.data.vst.writers.make_wds_dataset") as mock_wds,
    ):
        _run_main_with_argv(_cli_argv(str(data_file)))

    mock_h5.assert_called_once()
    mock_wds.assert_not_called()
    # First positional arg is the data_file path.
    h5_args, _h5_kwargs = mock_h5.call_args
    assert h5_args[0] == str(data_file)


def test_main_dispatches_tar_suffix_to_make_wds_dataset(tmp_path: Path) -> None:  # noqa: DOC101,DOC103
    """``data_file=foo.tar`` routes to ``make_wds_dataset`` (not the h5 writer)."""
    data_file = tmp_path / "shard-000000.tar"

    with (
        patch("synth_setter.data.vst.writers.make_hdf5_dataset") as mock_h5,
        patch("synth_setter.data.vst.writers.make_wds_dataset") as mock_wds,
    ):
        _run_main_with_argv(_cli_argv(str(data_file)))

    mock_wds.assert_called_once()
    mock_h5.assert_not_called()
    wds_args, _wds_kwargs = mock_wds.call_args
    assert wds_args[0] == str(data_file)


def test_main_rejects_unknown_suffix(tmp_path: Path) -> None:  # noqa: DOC101,DOC103
    """``data_file=foo.bin`` raises ``SystemExit`` rather than silently picking a writer."""
    data_file = tmp_path / "shard-000000.bin"

    with (
        patch("synth_setter.data.vst.writers.make_hdf5_dataset") as mock_h5,
        patch("synth_setter.data.vst.writers.make_wds_dataset") as mock_wds,
        pytest.raises(SystemExit, match=r"data_file must end in one of"),
    ):
        _run_main_with_argv(_cli_argv(str(data_file)))

    mock_h5.assert_not_called()
    mock_wds.assert_not_called()


def _stub_render_dependencies(  # noqa: DOC101,DOC103,DOC201,DOC203
    monkeypatch: pytest.MonkeyPatch,
    *,
    load_plugin_calls: list[dict[str, object]],
    load_preset_calls: list[dict[str, object]],
    cached_plugin_holder: list[object] | None = None,
) -> list[dict[str, object]]:
    """Patch ``load_plugin``, ``load_preset``, and ``generate_sample`` for the writer loop.

    Returns the kwargs captured from each ``generate_sample`` call. If
    ``cached_plugin_holder`` is supplied, the MagicMock returned by the fake
    ``load_plugin`` is appended to it so tests can assert identity-equality
    against the instance threaded into per-render calls.
    """
    captured: list[dict[str, object]] = []

    def _fake_load_plugin(path: str) -> MagicMock:
        plugin = MagicMock(name="cached_plugin")
        load_plugin_calls.append({"path": path})
        if cached_plugin_holder is not None:
            cached_plugin_holder.append(plugin)
        return plugin

    def _fake_load_preset(plugin: object, preset: str) -> None:
        load_preset_calls.append({"plugin": plugin, "preset": preset})

    def _fake_generate_sample(_plugin_path: str, **kwargs: object) -> object:
        captured.append(dict(kwargs))
        return MagicMock(name="vst_sample")

    monkeypatch.setattr(writers, "load_plugin", _fake_load_plugin)
    monkeypatch.setattr(writers, "load_preset", _fake_load_preset)
    monkeypatch.setattr(writers, "generate_sample", _fake_generate_sample)
    return captured


def test_render_in_batches_caches_plugin_when_reload_cadence_is_once(  # noqa: DOC101,DOC103
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``plugin_reload_cadence="once"`` loads the plugin once and reuses the same instance."""
    n = 4
    render_cfg = _smoke_render_cfg(
        samples_per_shard=n,
        samples_per_render_batch=n,
        plugin_reload_cadence="once",
        gui_toggle_cadence="never",
    )
    load_plugin_calls: list[dict[str, object]] = []
    load_preset_calls: list[dict[str, object]] = []
    cached_plugin_holder: list[object] = []
    captured = _stub_render_dependencies(
        monkeypatch,
        load_plugin_calls=load_plugin_calls,
        load_preset_calls=load_preset_calls,
        cached_plugin_holder=cached_plugin_holder,
    )

    flushed: list[tuple[list[VSTDataSample], int]] = []
    _render_in_batches(
        render_cfg=render_cfg,
        param_spec=MagicMock(name="param_spec"),
        start_idx=0,
        fixed_synth_params_list=None,
        fixed_note_params_list=None,
        flush_batch=lambda batch, start: flushed.append((batch, start)),
    )

    assert len(load_plugin_calls) == 1
    assert load_plugin_calls[0]["path"] == render_cfg.plugin_path
    assert len(load_preset_calls) == 1
    assert len(captured) == n
    cached = cached_plugin_holder[0]
    for call_kwargs in captured:
        assert call_kwargs["plugin"] is cached
    assert sum(len(batch) for batch, _ in flushed) == n


def test_render_in_batches_reloads_plugin_per_render_when_reload_cadence_is_render(  # noqa: DOC101,DOC103
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``plugin_reload_cadence="render"`` (default) leaves the plugin to be loaded per call."""
    n = 3
    render_cfg = _smoke_render_cfg(
        samples_per_shard=n,
        samples_per_render_batch=n,
        plugin_reload_cadence="render",
        gui_toggle_cadence="never",
    )
    load_plugin_calls: list[dict[str, object]] = []
    load_preset_calls: list[dict[str, object]] = []
    captured = _stub_render_dependencies(
        monkeypatch,
        load_plugin_calls=load_plugin_calls,
        load_preset_calls=load_preset_calls,
    )

    _render_in_batches(
        render_cfg=render_cfg,
        param_spec=MagicMock(name="param_spec"),
        start_idx=0,
        fixed_synth_params_list=None,
        fixed_note_params_list=None,
        flush_batch=lambda batch, _start: None,
    )

    # Per-render reload hides behind the generate_sample stub, so writers sees no load.
    assert load_plugin_calls == []
    assert load_preset_calls == []
    assert len(captured) == n
    for call_kwargs in captured:
        assert call_kwargs["plugin"] is None
        assert call_kwargs["warmup"] is False


def test_render_in_batches_warmup_once_runs_first_render_only(  # noqa: DOC101,DOC103
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``gui_toggle_cadence="once"`` sets warmup=True on the first render only."""
    n = 4
    render_cfg = _smoke_render_cfg(
        samples_per_shard=n,
        samples_per_render_batch=n,
        plugin_reload_cadence="render",
        gui_toggle_cadence="once",
    )
    captured = _stub_render_dependencies(
        monkeypatch,
        load_plugin_calls=[],
        load_preset_calls=[],
    )

    _render_in_batches(
        render_cfg=render_cfg,
        param_spec=MagicMock(name="param_spec"),
        start_idx=0,
        fixed_synth_params_list=None,
        fixed_note_params_list=None,
        flush_batch=lambda _batch, _start: None,
    )

    warmup_flags = [c["warmup"] for c in captured]
    assert warmup_flags == [True, False, False, False]


def test_render_in_batches_warmup_render_runs_every_render(  # noqa: DOC101,DOC103
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``gui_toggle_cadence="render"`` sets warmup=True on every render."""
    # The Darwin validator rejects gui_toggle_cadence="render" (SIGTRAP, #714);
    # force the non-Darwin path so the schema constructs on macOS CI hosts too.
    monkeypatch.setattr(
        "synth_setter.pipeline.schemas.spec._current_platform", lambda: "linux"
    )
    n = 3
    render_cfg = _smoke_render_cfg(
        samples_per_shard=n,
        samples_per_render_batch=n,
        plugin_reload_cadence="render",
        gui_toggle_cadence="render",
    )
    captured = _stub_render_dependencies(
        monkeypatch,
        load_plugin_calls=[],
        load_preset_calls=[],
    )

    _render_in_batches(
        render_cfg=render_cfg,
        param_spec=MagicMock(name="param_spec"),
        start_idx=0,
        fixed_synth_params_list=None,
        fixed_note_params_list=None,
        flush_batch=lambda _batch, _start: None,
    )

    assert all(c["warmup"] is True for c in captured)


def test_render_in_batches_warmup_never_skips_all_renders(  # noqa: DOC101,DOC103
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``gui_toggle_cadence="never"`` keeps warmup=False on every render."""
    n = 3
    render_cfg = _smoke_render_cfg(
        samples_per_shard=n,
        samples_per_render_batch=n,
        plugin_reload_cadence="render",
        gui_toggle_cadence="never",
    )
    captured = _stub_render_dependencies(
        monkeypatch,
        load_plugin_calls=[],
        load_preset_calls=[],
    )

    _render_in_batches(
        render_cfg=render_cfg,
        param_spec=MagicMock(name="param_spec"),
        start_idx=0,
        fixed_synth_params_list=None,
        fixed_note_params_list=None,
        flush_batch=lambda _batch, _start: None,
    )

    assert all(c["warmup"] is False for c in captured)


def test_render_in_batches_once_once_warms_once_and_caches_plugin(  # noqa: DOC101,DOC103
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``("once","once")`` — load + warm once, reuse the same plugin instance throughout."""
    n = 4
    render_cfg = _smoke_render_cfg(
        samples_per_shard=n,
        samples_per_render_batch=n,
        plugin_reload_cadence="once",
        gui_toggle_cadence="once",
    )
    load_plugin_calls: list[dict[str, object]] = []
    load_preset_calls: list[dict[str, object]] = []
    cached_plugin_holder: list[object] = []
    captured = _stub_render_dependencies(
        monkeypatch,
        load_plugin_calls=load_plugin_calls,
        load_preset_calls=load_preset_calls,
        cached_plugin_holder=cached_plugin_holder,
    )

    _render_in_batches(
        render_cfg=render_cfg,
        param_spec=MagicMock(name="param_spec"),
        start_idx=0,
        fixed_synth_params_list=None,
        fixed_note_params_list=None,
        flush_batch=lambda _batch, _start: None,
    )

    assert len(load_plugin_calls) == 1
    assert len(load_preset_calls) == 1
    cached = cached_plugin_holder[0]
    assert sum(1 for c in captured if c["warmup"] is True) == 1
    for call_kwargs in captured:
        assert call_kwargs["plugin"] is cached


def test_render_in_batches_once_render_warms_every_render_with_cached_plugin(  # noqa: DOC101,DOC103
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``("once","render")`` — cached plugin, warmup on every render across the shard."""
    monkeypatch.setattr(
        "synth_setter.pipeline.schemas.spec._current_platform", lambda: "linux"
    )
    n = 3
    render_cfg = _smoke_render_cfg(
        samples_per_shard=n,
        samples_per_render_batch=n,
        plugin_reload_cadence="once",
        gui_toggle_cadence="render",
    )
    load_plugin_calls: list[dict[str, object]] = []
    load_preset_calls: list[dict[str, object]] = []
    cached_plugin_holder: list[object] = []
    captured = _stub_render_dependencies(
        monkeypatch,
        load_plugin_calls=load_plugin_calls,
        load_preset_calls=load_preset_calls,
        cached_plugin_holder=cached_plugin_holder,
    )

    _render_in_batches(
        render_cfg=render_cfg,
        param_spec=MagicMock(name="param_spec"),
        start_idx=0,
        fixed_synth_params_list=None,
        fixed_note_params_list=None,
        flush_batch=lambda _batch, _start: None,
    )

    assert len(load_plugin_calls) == 1
    cached = cached_plugin_holder[0]
    assert sum(1 for c in captured if c["warmup"] is True) == n
    for call_kwargs in captured:
        assert call_kwargs["plugin"] is cached


# Writer-level cross-cut: the previous tests stub ``generate_sample`` itself,
# so they observe the ``warmup=`` kwarg the writer hands in but not what
# happens inside ``generate_sample``'s loudness-retry loop. The two tests
# below let the real ``generate_sample`` execute and stub one level deeper —
# at ``render_params`` — so they catch a regression in the retry-loop's
# ``warmup = False`` reset that the writer-level kwarg checks would miss.


_RENDERER_FAKE_AUDIO_SHAPE = (2, 16000 * 4)


def _silent_render() -> object:  # noqa: DOC201,DOC203
    """Return a silent stereo render shaped for the test ``_smoke_render_cfg``."""
    import numpy as np

    return np.zeros(_RENDERER_FAKE_AUDIO_SHAPE, dtype=np.float32)


def _loud_render() -> object:  # noqa: DOC201,DOC203
    """Return a loud stereo render shaped for the test ``_smoke_render_cfg``.

    A 440 Hz sine at half-scale comfortably clears the ``-55 dB`` loudness gate.
    """
    import numpy as np

    n = _RENDERER_FAKE_AUDIO_SHAPE[1]
    t = np.arange(n) / 16000.0
    sine = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    return np.stack([sine, sine], axis=0)


def _install_writer_level_fakes(  # noqa: DOC203
    monkeypatch: pytest.MonkeyPatch,
    *,
    retry_on_first_sample: int,
) -> tuple[MagicMock, MagicMock]:
    """Patch the renderer's seams so the writer loop runs without a real plugin.

    The fake ``render_params`` returns silent audio for the first
    ``retry_on_first_sample`` calls (driving the loudness-retry loop in
    ``generate_sample``) then loud audio thereafter; whenever it receives
    ``warmup=True`` it invokes a tracked ``warmup_plugin`` mock so callers
    can assert on the observable side effect rather than the kwarg
    passthrough.

    Returns a ``(warmup_mock, fake_spec)`` pair — the spec is meant to be
    handed directly to ``_render_in_batches`` so the test doesn't depend on
    the registry lookup that ``make_hdf5_dataset`` would normally perform.

    :param monkeypatch: Active monkeypatch fixture from the calling test.
    :param retry_on_first_sample: Number of silent renders before the first
        loud render — drives the loudness-retry loop inside the first
        ``generate_sample`` call.
    :returns: ``(warmup_mock, fake_spec)``.
    """
    import numpy as np

    from synth_setter.data.vst import generate_vst_dataset

    warmup_mock = MagicMock(name="warmup_plugin")
    silent_remaining = [retry_on_first_sample]

    def _fake_render_params(*_args: object, **kwargs: object) -> object:
        if kwargs.get("warmup"):
            warmup_mock()
        if silent_remaining[0] > 0:
            silent_remaining[0] -= 1
            return _silent_render()
        return _loud_render()

    fake_spec_payload = (
        {"a_amp_eg_attack": 0.5},
        {"pitch": 64, "note_start_and_end": (0.1, 0.9)},
    )
    fake_spec = MagicMock(name="param_spec")
    fake_spec.sample.return_value = fake_spec_payload
    fake_spec.encode.return_value = np.zeros((4,), dtype=np.float32)

    monkeypatch.setattr(generate_vst_dataset, "render_params", _fake_render_params)
    monkeypatch.setattr(
        generate_vst_dataset,
        "make_spectrogram",
        lambda *_a, **_kw: np.zeros((2, 128, 401), dtype=np.float32),
    )
    return warmup_mock, fake_spec


def test_render_in_batches_once_cadence_survives_intra_sample_retries(  # noqa: DOC101,DOC103
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``gui_toggle_cadence="once"`` warms exactly once even when sample 0 retries twice.

    Cross-cut for #714: the writer's ``warmup_done`` flag only flips after a
    successful render attempt, but ``generate_sample``'s loudness retry can
    issue multiple ``render_params`` calls per attempt. Without the
    ``warmup = False`` reset inside the retry loop, those internal retries
    would re-warm and silently blow past the per-shard budget. Asserts the
    observable side effect (``warmup_plugin`` call count) across the full
    shard, not the kwarg the writer hands to ``generate_sample``.
    """
    n = 3
    retries = 2
    render_cfg = _smoke_render_cfg(
        samples_per_shard=n,
        samples_per_render_batch=n,
        plugin_reload_cadence="render",
        gui_toggle_cadence="once",
    )
    warmup_mock, fake_spec = _install_writer_level_fakes(
        monkeypatch, retry_on_first_sample=retries
    )

    _render_in_batches(
        render_cfg=render_cfg,
        param_spec=fake_spec,
        start_idx=0,
        fixed_synth_params_list=None,
        fixed_note_params_list=None,
        flush_batch=lambda _batch, _start: None,
    )

    assert warmup_mock.call_count == 1


def test_render_in_batches_render_cadence_warms_once_per_generate_sample_call(  # noqa: DOC101,DOC103
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``gui_toggle_cadence="render"`` warms once per ``generate_sample`` call, not per retry.

    Symmetric pin to ``..._once_cadence_survives_intra_sample_retries``: under
    ``"render"`` cadence the writer hands ``warmup=True`` to every
    ``generate_sample`` call, so warm-up fires once per sample. ``generate_sample``'s
    internal ``warmup = False`` reset (line 129 of ``generate_vst_dataset.py``)
    is cadence-agnostic, so sample 0's silent retries do NOT re-warm — total
    warm-ups equals the sample count, not the render-attempt count.
    """
    monkeypatch.setattr(
        "synth_setter.pipeline.schemas.spec._current_platform", lambda: "linux"
    )
    n = 3
    retries = 2
    render_cfg = _smoke_render_cfg(
        samples_per_shard=n,
        samples_per_render_batch=n,
        plugin_reload_cadence="render",
        gui_toggle_cadence="render",
    )
    warmup_mock, fake_spec = _install_writer_level_fakes(
        monkeypatch, retry_on_first_sample=retries
    )

    _render_in_batches(
        render_cfg=render_cfg,
        param_spec=fake_spec,
        start_idx=0,
        fixed_synth_params_list=None,
        fixed_note_params_list=None,
        flush_batch=lambda _batch, _start: None,
    )

    # One warm per sample (n) — the retry-loop reset drops warmup to False
    # after the first attempt of each ``generate_sample`` call, so sample 0's
    # silent retries do NOT add to the count.
    assert warmup_mock.call_count == n
