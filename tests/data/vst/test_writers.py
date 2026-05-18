"""CPU-only tests for ``synth_setter.data.vst.writers``.

Covers the writer module's pure helpers and the CLI dispatcher in
``generate_vst_dataset.main`` — the VST-dependent end-to-end writer tests
live alongside the legacy HDF5 tests in ``test_generate_vst_dataset.py`` and
the new wds e2e tests in ``test_writers_wds_e2e.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from synth_setter.data.vst import writers
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
        "open_gui_every_render": False,
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
        "--open_gui_every_render",
        "False",
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
    monkeypatch: pytest.MonkeyPatch, *, load_plugin_calls: list[Any], load_preset_calls: list[Any]
) -> list[dict[str, Any]]:
    """Patch ``load_plugin``, ``load_preset``, and ``generate_sample`` for the writer loop.

    Returns the kwargs captured from each ``generate_sample`` call so tests can
    assert which path was taken (plugin supplied vs. None).
    """
    captured: list[dict[str, Any]] = []

    def _fake_load_plugin(path: str, **kwargs: object) -> MagicMock:
        load_plugin_calls.append({"path": path, **kwargs})
        return MagicMock(name="cached_plugin")

    def _fake_load_preset(plugin: object, preset: str) -> None:
        load_preset_calls.append({"plugin": plugin, "preset": preset})

    def _fake_generate_sample(_plugin_path: str, **kwargs: object) -> object:
        captured.append(dict(kwargs))
        return MagicMock(name="vst_sample")

    monkeypatch.setattr(writers, "load_plugin", _fake_load_plugin)
    monkeypatch.setattr(writers, "load_preset", _fake_load_preset)
    monkeypatch.setattr(writers, "generate_sample", _fake_generate_sample)
    return captured


def test_render_in_batches_caches_plugin_when_reload_is_false(  # noqa: DOC101,DOC103
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``reload_plugin_every_render=False`` loads the plugin once and reuses it for every
    sample."""
    n = 4
    render_cfg = _smoke_render_cfg(
        samples_per_shard=n,
        samples_per_render_batch=n,
        reload_plugin_every_render=False,
        open_gui_every_render=False,
    )
    load_plugin_calls: list[Any] = []
    load_preset_calls: list[Any] = []
    captured = _stub_render_dependencies(
        monkeypatch,
        load_plugin_calls=load_plugin_calls,
        load_preset_calls=load_preset_calls,
    )

    flushed: list[tuple[list[Any], int]] = []
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
    assert load_plugin_calls[0]["open_gui"] is False
    assert len(load_preset_calls) == 1
    assert len(captured) == n
    for call_kwargs in captured:
        assert call_kwargs["plugin"] is not None


def test_render_in_batches_reloads_plugin_per_render_when_reload_is_true(  # noqa: DOC101,DOC103
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``reload_plugin_every_render=True`` (default) leaves the plugin to be loaded per call."""
    n = 3
    render_cfg = _smoke_render_cfg(
        samples_per_shard=n,
        samples_per_render_batch=n,
        reload_plugin_every_render=True,
        open_gui_every_render=False,
    )
    load_plugin_calls: list[Any] = []
    load_preset_calls: list[Any] = []
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

    # No shard-level load when reload_plugin_every_render=True; each render_params
    # call reloads on its own (which is mocked out via the generate_sample stub).
    assert load_plugin_calls == []
    assert load_preset_calls == []
    assert len(captured) == n
    for call_kwargs in captured:
        assert call_kwargs["plugin"] is None
        assert call_kwargs["open_gui"] is False
