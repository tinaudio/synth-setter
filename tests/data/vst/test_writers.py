"""CPU-only tests for ``synth_setter.data.vst.writers``.

Covers the writer module's pure helpers and the CLI dispatcher in
``generate_vst_dataset.main`` — the VST-dependent end-to-end Lance writer tests
live in ``test_generate_vst_dataset.py`` and ``test_fake_plugin_e2e.py``.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from synth_setter.data.vst import writers
from synth_setter.data.vst.generate_vst_dataset import VSTDataSample
from synth_setter.data.vst.param_spec import NoteParams, ParamSpec
from synth_setter.data.vst.writers import _render_in_batches
from synth_setter.pipeline.schemas.render_metrics import (
    RenderRejectionMetrics,
    render_metrics_path,
)
from synth_setter.pipeline.schemas.shard_metadata import ShardMetadata
from synth_setter.pipeline.schemas.spec import RenderConfig


def _smoke_render_cfg(**overrides: object) -> RenderConfig:
    """Build a syntactically-valid ``RenderConfig`` for CPU-only tests.

    No I/O happens against ``plugin_path`` or ``plugin_state_path`` in these tests
    — they only need to be non-blank strings.

    :param \\*\\*overrides: Per-call overrides merged into the default kwargs.
    :return: A ``RenderConfig`` ready for the writer tests.
    """
    kwargs: dict[str, object] = {
        "plugin_path": "plugins/Surge XT.vst3",
        "plugin_state_path": "presets/surge-base.vstpreset",
        "param_spec_name": "surge_simple",
        "renderer_version": "1.3.4",
        "sample_rate": 44100,
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


def test_render_config_shard_metadata_projects_render_provenance_fields() -> None:
    """``RenderConfig.shard_metadata`` returns a strict ``ShardMetadata`` with renderer values."""
    render_cfg = _smoke_render_cfg(
        velocity=64,
        signal_duration_seconds=2.5,
        sample_rate=22050,
        channels=1,
        min_loudness=-40.0,
        base_seed=7,
        attempts_per_sample=9,
    )

    meta = render_cfg.shard_metadata()

    assert isinstance(meta, ShardMetadata)
    assert meta.velocity == 64
    assert meta.signal_duration_seconds == 2.5
    assert meta.sample_rate == 22050
    assert meta.channels == 1
    assert meta.min_loudness == -40.0
    assert meta.base_seed == 7
    assert meta.attempts_per_sample == 9


def test_render_config_shard_metadata_round_trips_through_json() -> None:
    """The projected metadata serializes and re-validates as a strict ``ShardMetadata``.

    Pinning JSON round-trip is what the Lance schema-metadata payload relies on:
    a writer-side projection that can't be re-read isn't useful.
    """
    render_cfg = _smoke_render_cfg()

    meta = render_cfg.shard_metadata()
    rehydrated = ShardMetadata.model_validate_json(meta.model_dump_json())

    assert rehydrated == meta


def _run_main_with_argv(argv: list[str]) -> None:
    """Invoke ``generate_vst_dataset.main`` with ``argv`` patched in.

    The pydantic-settings CLI reads ``sys.argv`` directly via ``CliApp.run``,
    so tests need to swap the process argv around the call. Imports the entry
    inside the helper so a single import failure doesn't poison the module.

    :param argv: Parametrized ``argv`` value under test.
    """
    from synth_setter.data.vst.generate_vst_dataset import main

    with patch.object(sys, "argv", argv):
        main()


# Shared CLI argv prefix for the dispatcher tests below. Built from the same
# ``RenderConfig`` field set the CLI binding inherits, so adding a render-config
# field auto-extends the prefix.
def _cli_argv(data_file: str) -> list[str]:
    """Build a CLI argv that parses cleanly into a ``RenderConfig`` + ``data_file``.

    All values mirror ``_smoke_render_cfg`` so the parsed config is round-trip
    equal to it. The ``argv[0]`` is a stand-in program name (not used).

    :param data_file: Path threaded into argv as the positional data_file arg.
    :return: A list of argv tokens suitable for ``_run_main_with_argv``.
    """
    return [
        "generate_vst_dataset",
        data_file,
        "--shard_id",
        "7",
        "--plugin_path",
        "plugins/Surge XT.vst3",
        "--plugin_state_path",
        "presets/surge-base.vstpreset",
        "--param_spec_name",
        "surge_simple",
        "--renderer_version",
        "1.3.4",
        "--sample_rate",
        "44100",
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


def test_main_dispatches_lance_suffix_to_make_lance_dataset(tmp_path: Path) -> None:
    """``data_file=foo.lance`` routes to ``make_lance_dataset``.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    data_file = tmp_path / "shard-000000.lance"

    with patch(
        "synth_setter.data.vst.writers.make_lance_dataset",
        return_value=RenderRejectionMetrics(clipped=2, silent=3),
    ) as mock_lance:
        _run_main_with_argv(_cli_argv(str(data_file)))

    mock_lance.assert_called_once()
    # First positional arg is the data_file path.
    lance_args, lance_kwargs = mock_lance.call_args
    assert lance_args[0] == str(data_file)
    assert lance_kwargs["shard_id"] == 7
    assert RenderRejectionMetrics.model_validate_json(
        render_metrics_path(data_file).read_text()
    ) == RenderRejectionMetrics(clipped=2, silent=3)


def test_main_rejects_unknown_suffix(tmp_path: Path) -> None:
    """``data_file=foo.bin`` raises ``SystemExit`` rather than silently picking a writer.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    data_file = tmp_path / "shard-000000.bin"

    with (
        patch("synth_setter.data.vst.writers.make_lance_dataset") as mock_lance,
        pytest.raises(SystemExit, match=r"data_file must end in .lance"),
    ):
        _run_main_with_argv(_cli_argv(str(data_file)))

    mock_lance.assert_not_called()


def test_main_rejects_known_non_lance_suffix(tmp_path: Path) -> None:
    """A known legacy format suffix cannot reach the Lance writer.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    data_file = tmp_path / "shard-000000.h5"

    with (
        patch("synth_setter.data.vst.writers.make_lance_dataset") as mock_lance,
        pytest.raises(SystemExit, match=r"data_file must end in .lance"),
    ):
        _run_main_with_argv(_cli_argv(str(data_file)))

    mock_lance.assert_not_called()


class _FakePlugin:
    """Stand-in for a loaded VST plugin handle.

    Carries no behaviour — the writer loop only threads it through and tests
    assert identity (``is``), so a bare object with a debug ``repr`` suffices.
    """

    def __repr__(self) -> str:
        return "_FakePlugin()"


def _stub_plugin_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep writer tests independent of an installed VST.

    :param monkeypatch: Pytest fixture used to patch module-level callables.
    """
    monkeypatch.setattr(writers, "load_plugin", lambda _path: _FakePlugin())
    monkeypatch.setattr(writers, "load_preset", lambda _plugin, _preset: None)


def _stub_plugin_load_seams(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep writer tests plugin-free under the eager ``"once"`` reload default.

    :param monkeypatch: Caller's fixture, so the stubs revert at that test's teardown.
    """
    monkeypatch.setattr(writers, "load_plugin", lambda _path: _FakePlugin())
    monkeypatch.setattr(writers, "load_preset", lambda _plugin, _path: None)


def _stub_render_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    load_plugin_calls: list[dict[str, object]],
    load_preset_calls: list[dict[str, object]],
    cached_plugin_holder: list[object] | None = None,
    clipped_rejections: int = 0,
    silent_rejections: int = 0,
) -> list[dict[str, object]]:
    """Patch ``load_plugin``, ``load_preset``, and ``generate_sample`` for the writer loop.

    Returns the kwargs captured from each ``generate_sample`` call. If
    ``cached_plugin_holder`` is supplied, the ``_FakePlugin`` returned by the
    fake ``load_plugin`` is appended to it so tests can assert identity-equality
    against the instance threaded into per-render calls.

    :param monkeypatch: Pytest fixture used to patch module-level callables.
    :param load_plugin_calls: List receiving the path argument of each fake ``load_plugin`` call.
    :param load_preset_calls: List receiving the kwargs of each fake ``load_preset`` call.
    :param cached_plugin_holder: When supplied, receives the fake plugin instance.
    :param clipped_rejections: Clipped count returned by each fake sample.
    :param silent_rejections: Silent count returned by each fake sample.
    :return: Kwargs captured from each ``generate_sample`` invocation.
    """
    captured: list[dict[str, object]] = []

    def _fake_load_plugin(path: str) -> _FakePlugin:
        plugin = _FakePlugin()
        load_plugin_calls.append({"path": path})
        if cached_plugin_holder is not None:
            cached_plugin_holder.append(plugin)
        return plugin

    def _fake_load_preset(plugin: object, preset: str) -> None:
        load_preset_calls.append({"plugin": plugin, "preset": preset})

    def _fake_generate_sample(**kwargs: object) -> SimpleNamespace:
        captured.append(dict(kwargs))
        return SimpleNamespace(
            clipped_rejections=clipped_rejections,
            silent_rejections=silent_rejections,
        )

    monkeypatch.setattr(writers, "load_plugin", _fake_load_plugin)
    monkeypatch.setattr(writers, "load_preset", _fake_load_preset)
    monkeypatch.setattr(writers, "generate_sample", _fake_generate_sample)
    return captured


def test_render_in_batches_shard_cadence_reuses_first_sample_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``param_sample_cadence="shard"`` draws once, then fixes every later render to it.

    Sample 0 is rendered with no fixed params (the normal loudness-gated draw);
    samples 1..N receive that first sample's ``synth_params`` / ``note_params``
    as the fixed override, so the whole shard is one identical patch.

    :param monkeypatch: Pytest fixture used to patch module-level callables.
    """
    n = 4
    render_cfg = _smoke_render_cfg(
        samples_per_shard=n,
        samples_per_render_batch=n,
        param_sample_cadence="shard",
    )

    returned: list[MagicMock] = []
    captured: list[dict[str, object]] = []

    def _fake_generate_sample(**kwargs: object) -> MagicMock:
        captured.append(dict(kwargs))
        sample = MagicMock(name=f"sample_{len(returned)}")
        sample.clipped_rejections = 0
        sample.silent_rejections = 0
        returned.append(sample)
        return sample

    _stub_plugin_loading(monkeypatch)
    monkeypatch.setattr(writers, "generate_sample", _fake_generate_sample)
    _stub_plugin_load_seams(monkeypatch)

    _render_in_batches(
        render_cfg=render_cfg,
        param_spec=MagicMock(name="param_spec"),
        start_idx=0,
        fixed_synth_params_list=None,
        fixed_note_params_list=None,
        flush_batch=lambda _batch, _start: None,
    )

    assert len(captured) == n
    assert captured[0]["fixed_synth_params"] is None
    assert captured[0]["fixed_note_params"] is None
    first = returned[0]
    for call_kwargs in captured[1:]:
        assert call_kwargs["fixed_synth_params"] is first.synth_params
        assert call_kwargs["fixed_note_params"] is first.note_params


def test_render_in_batches_sample_cadence_draws_fresh_params_every_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``param_sample_cadence="sample"`` (default) never fixes params from a prior render.

    :param monkeypatch: Pytest fixture used to patch module-level callables.
    """
    n = 3
    render_cfg = _smoke_render_cfg(
        samples_per_shard=n,
        samples_per_render_batch=n,
        param_sample_cadence="sample",
    )
    captured = _stub_render_dependencies(monkeypatch, load_plugin_calls=[], load_preset_calls=[])

    _render_in_batches(
        render_cfg=render_cfg,
        param_spec=MagicMock(name="param_spec"),
        start_idx=0,
        fixed_synth_params_list=None,
        fixed_note_params_list=None,
        flush_batch=lambda _batch, _start: None,
    )

    assert len(captured) == n
    for call_kwargs in captured:
        assert call_kwargs["fixed_synth_params"] is None
        assert call_kwargs["fixed_note_params"] is None


def test_render_in_batches_shard_cadence_seeds_single_patch_from_caller_row_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shard-cadence copy seeds the shard's single patch from source row 0, then reuses it.

    :param monkeypatch: Pytest fixture used to patch module-level callables.
    """
    n = 3
    render_cfg = _smoke_render_cfg(
        samples_per_shard=n,
        samples_per_render_batch=n,
        param_sample_cadence="shard",
    )
    captured: list[dict[str, object]] = []

    def _fake_generate_sample(**kwargs: object) -> MagicMock:
        captured.append(dict(kwargs))
        sample = MagicMock(name=f"sample_{len(captured)}")
        sample.clipped_rejections = 0
        sample.silent_rejections = 0
        # Mirror the real renderer: the sample reports the params it rendered with, so shard
        # cadence reuses concrete row-0 values rather than MagicMock placeholder attributes.
        sample.synth_params = kwargs["fixed_synth_params"]
        sample.note_params = kwargs["fixed_note_params"]
        return sample

    _stub_plugin_loading(monkeypatch)
    monkeypatch.setattr(writers, "generate_sample", _fake_generate_sample)
    _stub_plugin_load_seams(monkeypatch)

    synth_rows = [{"a": 0.1}, {"a": 0.2}, {"a": 0.3}]
    note_rows: list[NoteParams] = [
        {"pitch": 60 + i, "note_start_and_end": (0.0, 1.0)} for i in range(n)
    ]
    # Source rows must differ so "only row 0 is used" is a real assertion, not a tautology.
    assert synth_rows[0] != synth_rows[1]
    _render_in_batches(
        render_cfg=render_cfg,
        param_spec=MagicMock(name="param_spec"),
        start_idx=0,
        fixed_synth_params_list=synth_rows,
        fixed_note_params_list=note_rows,
        flush_batch=lambda _batch, _start: None,
    )

    # Sample 0 seeds the patch from row 0; every later render reuses it, so all renders use row 0.
    assert len(captured) == n
    for call_kwargs in captured:
        assert call_kwargs["fixed_synth_params"] == synth_rows[0]
        assert call_kwargs["fixed_note_params"] == note_rows[0]


def test_render_in_batches_caches_plugin_when_reload_cadence_is_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``plugin_reload_cadence="once"`` loads the plugin once and reuses the same instance.

    :param monkeypatch: Pytest fixture used to patch attributes / env / argv.
    """
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
        assert getattr(call_kwargs["renderer"], "plugin") is cached
    assert sum(len(batch) for batch, _ in flushed) == n


@pytest.mark.parametrize(
    ("cadence", "reload_each_render"),
    [("once", False), ("render", True)],
)
def test_make_renderer_maps_dawdreamer_reload_cadence(
    monkeypatch: pytest.MonkeyPatch,
    cadence: str,
    reload_each_render: bool,
) -> None:
    """DawDreamer receives the requested plugin lifecycle policy.

    :param monkeypatch: Replaces renderer construction with a capture seam.
    :param cadence: Public reload cadence under test.
    :param reload_each_render: Expected renderer lifecycle flag.
    """
    captured: dict[str, object] = {}

    def capture_renderer(**kwargs: object) -> MagicMock:
        captured.update(kwargs)
        return MagicMock()

    monkeypatch.setattr(writers, "DawDreamerRenderer", capture_renderer)
    render_cfg = _smoke_render_cfg(
        renderer_backend="dawdreamer",
        plugin_reload_cadence=cadence,
        gui_toggle_cadence="never",
    )

    writers._make_renderer(render_cfg)

    assert captured["reload_plugin_each_render"] is reload_each_render


def test_make_renderer_torchsynth_backend_builds_in_process_renderer() -> None:
    """The torchsynth backend dispatches to the in-process renderer with the audio geometry."""
    from synth_setter.data.vst.renderers import TorchSynthRenderer

    render_cfg = _smoke_render_cfg(
        renderer_backend="torchsynth",
        plugin_path="torchsynth",
        plugin_state_path="",
        param_spec_name="torchsynth_adsr",
        renderer_version="1.0.2",
        sample_rate=22050,
        signal_duration_seconds=0.5,
        gui_toggle_cadence="never",
    )

    renderer = writers._make_renderer(render_cfg)

    assert isinstance(renderer, TorchSynthRenderer)
    assert (renderer.sample_rate, renderer.channels) == (22050, 2)
    assert renderer.signal_duration_seconds == 0.5


def test_render_in_batches_reloads_plugin_per_render_when_reload_cadence_is_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit ``plugin_reload_cadence="render"`` (non-default, #1999) loads the plugin per call.

    :param monkeypatch: Pytest fixture used to patch attributes / env / argv.
    """
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
        assert getattr(call_kwargs["renderer"], "plugin") is None
        assert call_kwargs["warmup"] is False


def test_render_in_batches_warmup_once_runs_first_render_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``gui_toggle_cadence="once"`` sets warmup=True on the first render only.

    :param monkeypatch: Pytest fixture used to patch attributes / env / argv.
    """
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


def test_render_in_batches_warmup_render_runs_every_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``gui_toggle_cadence="render"`` sets warmup=True on every render.

    :param monkeypatch: Pytest fixture used to patch attributes / env / argv.
    """
    # The Darwin validator rejects gui_toggle_cadence="render" (SIGTRAP, #714);
    # force the non-Darwin path so the schema constructs on macOS CI hosts too.
    monkeypatch.setattr("synth_setter.pipeline.schemas.spec._current_platform", lambda: "linux")
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


def test_render_in_batches_warmup_never_skips_all_renders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``gui_toggle_cadence="never"`` keeps warmup=False on every render.

    :param monkeypatch: Pytest fixture used to patch attributes / env / argv.
    """
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


def test_render_in_batches_always_on_runs_loop_via_run_with_editor_held_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``gui_toggle_cadence="always_on"`` hands the render loop to ``run_with_editor_held_open``.

    Asserts the helper is invoked once with the cached plugin, the body
    callable executes, the loop runs for all samples, and ``warmup_plugin``
    never fires per render.

    :param monkeypatch: Pytest fixture used to patch attributes / env / argv.
    """
    n = 3
    render_cfg = _smoke_render_cfg(
        samples_per_shard=n,
        samples_per_render_batch=n,
        plugin_reload_cadence="once",
        gui_toggle_cadence="always_on",
    )
    cached_plugin_holder: list[object] = []
    captured = _stub_render_dependencies(
        monkeypatch,
        load_plugin_calls=[],
        load_preset_calls=[],
        cached_plugin_holder=cached_plugin_holder,
        clipped_rejections=1,
        silent_rejections=2,
    )
    held_open_plugins: list[object] = []

    def fake_run(plugin: object, body: Callable[[], object]) -> object:
        held_open_plugins.append(plugin)
        return body()

    monkeypatch.setattr(writers, "run_with_editor_held_open", fake_run)

    metrics = _render_in_batches(
        render_cfg=render_cfg,
        param_spec=MagicMock(name="param_spec"),
        start_idx=0,
        fixed_synth_params_list=None,
        fixed_note_params_list=None,
        flush_batch=lambda _batch, _start: None,
    )

    assert metrics == RenderRejectionMetrics(clipped=3, silent=6)
    assert len(held_open_plugins) == 1
    assert held_open_plugins[0] is cached_plugin_holder[0]
    assert len(captured) == n
    assert all(c["warmup"] is False for c in captured)


def test_render_in_batches_always_on_propagates_worker_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any exception raised by ``run_with_editor_held_open`` escapes ``_render_in_batches``.

    Mirrors the intent of the deleted ``EditorStartTimeout`` propagation pin:
    the upstream loop must not swallow exceptions from the held-open scope,
    or a failed editor bring-up would silently degrade to no-editor renders
    and defeat the always_on contract.

    :param monkeypatch: Pytest fixture used to patch attributes / env / argv.
    """
    n = 3
    render_cfg = _smoke_render_cfg(
        samples_per_shard=n,
        samples_per_render_batch=n,
        plugin_reload_cadence="once",
        gui_toggle_cadence="always_on",
    )
    captured = _stub_render_dependencies(
        monkeypatch,
        load_plugin_calls=[],
        load_preset_calls=[],
    )

    def raising_run(_plugin: object, _body: Callable[[], object]) -> object:
        raise RuntimeError("show_editor failed")

    monkeypatch.setattr(writers, "run_with_editor_held_open", raising_run)

    with pytest.raises(RuntimeError, match="show_editor failed"):
        _render_in_batches(
            render_cfg=render_cfg,
            param_spec=MagicMock(name="param_spec"),
            start_idx=0,
            fixed_synth_params_list=None,
            fixed_note_params_list=None,
            flush_batch=lambda _batch, _start: None,
        )

    assert captured == []


@pytest.mark.parametrize("legacy_cadence", ["never", "once", "render"])
def test_render_in_batches_non_always_on_skips_run_with_editor_held_open(
    monkeypatch: pytest.MonkeyPatch, legacy_cadence: str
) -> None:
    """Every legacy ``gui_toggle_cadence`` leaves ``run_with_editor_held_open`` untouched.

    :param monkeypatch: Pins ``_current_platform`` to ``"linux"`` so the
        ``"render"`` cadence is accepted on Darwin runners (the schema's #714
        gate rejects it there), and patches the writer's render dependencies.
    :param legacy_cadence: Parametrized over each non-``always_on`` cadence so
        all three inline paths are pinned, not just ``never``.
    """
    monkeypatch.setattr("synth_setter.pipeline.schemas.spec._current_platform", lambda: "linux")
    render_cfg = _smoke_render_cfg(
        samples_per_shard=1,
        samples_per_render_batch=1,
        plugin_reload_cadence="once",
        gui_toggle_cadence=legacy_cadence,
    )
    _stub_render_dependencies(monkeypatch, load_plugin_calls=[], load_preset_calls=[])
    held_open_calls: list[object] = []

    def fake_run(plugin: object, body: Callable[[], object]) -> object:
        held_open_calls.append(plugin)
        return body()

    monkeypatch.setattr(writers, "run_with_editor_held_open", fake_run)

    _render_in_batches(
        render_cfg=render_cfg,
        param_spec=MagicMock(name="param_spec"),
        start_idx=0,
        fixed_synth_params_list=None,
        fixed_note_params_list=None,
        flush_batch=lambda _batch, _start: None,
    )

    assert held_open_calls == []


def test_render_in_batches_once_once_warms_once_and_caches_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``("once","once")`` — load + warm once, reuse the same plugin instance throughout.

    :param monkeypatch: Pytest fixture used to patch attributes / env / argv.
    """
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
        assert getattr(call_kwargs["renderer"], "plugin") is cached


def test_render_in_batches_once_render_warms_every_render_with_cached_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``("once","render")`` — cached plugin, warmup on every render across the shard.

    :param monkeypatch: Pytest fixture used to patch attributes / env / argv.
    """
    monkeypatch.setattr("synth_setter.pipeline.schemas.spec._current_platform", lambda: "linux")
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
        assert getattr(call_kwargs["renderer"], "plugin") is cached


# Writer-level cross-cut: the previous tests stub ``generate_sample`` itself,
# so they observe the ``warmup=`` kwarg the writer hands in but not what
# happens inside ``generate_sample``'s loudness-retry loop. The two tests
# below let the real ``generate_sample`` execute and stub one level deeper —
# at ``render_params`` — so they catch a regression in the retry-loop's
# ``warmup = False`` reset that the writer-level kwarg checks would miss.


_RENDERER_FAKE_SAMPLE_RATE = 44100
_RENDERER_FAKE_AUDIO_SHAPE = (2, _RENDERER_FAKE_SAMPLE_RATE * 4)


def _silent_render() -> object:
    """Return a silent stereo render shaped for the test ``_smoke_render_cfg``.

    :return: Zero-filled stereo audio array shaped like a real render.
    """
    import numpy as np

    return np.zeros(_RENDERER_FAKE_AUDIO_SHAPE, dtype=np.float32)


def _loud_render() -> object:
    """Return a loud stereo render shaped for the test ``_smoke_render_cfg``.

    A 440 Hz sine at half-scale comfortably clears the ``-55 dB`` loudness gate.

    :return: 440 Hz sine wave stereo audio array shaped like a real render.
    """
    import numpy as np

    n = _RENDERER_FAKE_AUDIO_SHAPE[1]
    t = np.arange(n) / _RENDERER_FAKE_SAMPLE_RATE
    sine = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    return np.stack([sine, sine], axis=0)


def _install_writer_level_fakes(
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
    the registry lookup that ``make_lance_dataset`` would normally perform.

    :param monkeypatch: Active monkeypatch fixture from the calling test.
    :param retry_on_first_sample: Number of silent renders before the first
        loud render — drives the loudness-retry loop inside the first
        ``generate_sample`` call.
    :returns: ``(warmup_mock, fake_spec)``.
    """
    import numpy as np

    from synth_setter.data.vst import generate_vst_dataset

    _stub_plugin_loading(monkeypatch)
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
    fake_spec = MagicMock(spec_set=ParamSpec, name="param_spec")
    fake_spec.sample.return_value = fake_spec_payload
    fake_spec.encode.return_value = np.zeros((4,), dtype=np.float32)

    monkeypatch.setattr("synth_setter.data.vst.core.render_params", _fake_render_params)
    monkeypatch.setattr(
        generate_vst_dataset,
        "make_spectrogram",
        lambda *_a, **_kw: np.zeros((2, 128, 401), dtype=np.float32),
    )
    _stub_plugin_load_seams(monkeypatch)
    return warmup_mock, fake_spec


def test_render_in_batches_aggregates_rejections_by_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shard rendering sums silent and clipped rejections independently.

    :param monkeypatch: Pytest fixture used to patch renderer dependencies.
    """
    render_cfg = _smoke_render_cfg(samples_per_shard=2, samples_per_render_batch=1)
    _, fake_spec = _install_writer_level_fakes(monkeypatch, retry_on_first_sample=2)

    metrics = _render_in_batches(
        render_cfg=render_cfg,
        param_spec=fake_spec,
        start_idx=0,
        fixed_synth_params_list=None,
        fixed_note_params_list=None,
        flush_batch=lambda _batch, _start: None,
    )

    assert metrics.silent == 2
    assert metrics.clipped == 0


def test_render_in_batches_once_cadence_survives_intra_sample_retries(
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

    :param monkeypatch: Pytest fixture used to patch attributes / env / argv.
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


def test_render_in_batches_render_cadence_warms_once_per_generate_sample_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``gui_toggle_cadence="render"`` warms once per ``generate_sample`` call, not per retry.

    Symmetric pin to ``..._once_cadence_survives_intra_sample_retries``: under
    ``"render"`` cadence the writer hands ``warmup=True`` to every
    ``generate_sample`` call, so warm-up fires once per sample. ``generate_sample``'s
    internal ``warmup = False`` reset (line 129 of ``generate_vst_dataset.py``)
    is cadence-agnostic, so sample 0's silent retries do NOT re-warm — total
    warm-ups equals the sample count, not the render-attempt count.

    :param monkeypatch: Pytest fixture used to patch attributes / env / argv.
    """
    monkeypatch.setattr("synth_setter.pipeline.schemas.spec._current_platform", lambda: "linux")
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


def test_render_in_batches_shard_cadence_draws_param_spec_once_for_whole_shard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``param_sample_cadence="shard"`` calls ``param_spec.sample()`` once and reuses the patch.

    Drives the real ``generate_sample`` (only ``render_params`` /
    ``make_spectrogram`` are faked), so this pins observable behaviour through
    the production code path: exactly one draw for the shard, and every flushed
    sample carries that same patch.

    :param monkeypatch: Pytest fixture used to patch attributes / env / argv.
    """
    n = 4
    render_cfg = _smoke_render_cfg(
        samples_per_shard=n,
        samples_per_render_batch=n,
        param_sample_cadence="shard",
    )
    _warmup_mock, fake_spec = _install_writer_level_fakes(monkeypatch, retry_on_first_sample=0)

    flushed: list[VSTDataSample] = []
    _render_in_batches(
        render_cfg=render_cfg,
        param_spec=fake_spec,
        start_idx=0,
        fixed_synth_params_list=None,
        fixed_note_params_list=None,
        flush_batch=lambda batch, _start: flushed.extend(batch),
    )

    assert fake_spec.sample.call_count == 1
    assert len(flushed) == n
    assert all(s.synth_params == {"a_amp_eg_attack": 0.5} for s in flushed)
    assert all(s.note_params == {"pitch": 64, "note_start_and_end": (0.1, 0.9)} for s in flushed)


def test_render_in_batches_sample_cadence_draws_param_spec_per_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``param_sample_cadence="sample"`` draws a fresh patch for every sample in the shard.

    The contrast case to the shard-cadence draw-once test: the default cadence
    calls ``param_spec.sample()`` once per sample through the real
    ``generate_sample`` path.

    :param monkeypatch: Pytest fixture used to patch attributes / env / argv.
    """
    n = 4
    render_cfg = _smoke_render_cfg(
        samples_per_shard=n,
        samples_per_render_batch=n,
        param_sample_cadence="sample",
    )
    _warmup_mock, fake_spec = _install_writer_level_fakes(monkeypatch, retry_on_first_sample=0)

    _render_in_batches(
        render_cfg=render_cfg,
        param_spec=fake_spec,
        start_idx=0,
        fixed_synth_params_list=None,
        fixed_note_params_list=None,
        flush_batch=lambda _batch, _start: None,
    )

    assert fake_spec.sample.call_count == n
