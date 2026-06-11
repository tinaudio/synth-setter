"""End-to-end shard-write proven against a duck-typed ``FakeVST3Plugin``.

No real ``.vst3`` bundle, no X11, no Surge XT — ``install_fake_plugin``
swaps the loader for the fake, so the whole ``make_hdf5_dataset`` path
(batch loop, held-open editor, HDF5 writer, mel-spec computation) runs
on every PR. The real-plugin counterpart at
``test_always_on_integration.py`` stays as the "does Surge XT still
work" gate; this is the "does our pipeline still work" gate.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  side-effect: registers Blosc2 filter for h5py reads
import numpy as np
import pytest

from synth_setter.data.vst import core
from synth_setter.data.vst.generate_vst_dataset import fixed_params_from_dataset
from synth_setter.data.vst.param_spec_registry import param_specs
from synth_setter.data.vst.shapes import AUDIO_FIELD
from synth_setter.data.vst.writers import make_hdf5_dataset, make_wds_dataset
from synth_setter.pipeline.schemas.spec import RenderConfig

_ = hdf5plugin  # keep type checkers from flagging the side-effect import

from tests.data.vst._fake_plugin import FakeVST3Plugin  # noqa: E402
from tests.data.vst.test_generate_vst_dataset import (  # noqa: E402  pinned canonical patch
    _HARDCODED_NOTE_PARAMS,
    _HARDCODED_SYNTH_PARAMS,
    _SPEC_NAME,
    _render_cfg,
)
from tests.helpers.logger_assertions import assert_no_logger_exceptions  # noqa: E402

_PLUGIN_PATH = "plugins/fake.vst3"  # never touched on disk — load_plugin is patched
_PRESET_PATH = "presets/fake.vstpreset"
_RENDERER_VERSION = "fake-0.0.0"


def _fake_render_cfg(**overrides: object) -> RenderConfig:
    """Build a ``RenderConfig`` pointing at the fake plugin paths.

    Wraps the canonical ``_render_cfg`` and rebinds ``plugin_path`` / ``preset_path`` /
    ``renderer_version`` to the never-touched fake-plugin strings so the writer runs
    entirely under ``install_fake_plugin``.

    :param \\*\\*overrides: Passed through to ``_render_cfg`` (e.g. ``num_samples``,
        ``samples_per_render_batch``, ``param_sample_cadence`` via ``model_copy``).
    :returns: A ``RenderConfig`` wired for the fake plugin.
    """
    num_samples = overrides.pop("num_samples")
    cadence = overrides.pop("param_sample_cadence", None)
    cfg = _render_cfg(num_samples=num_samples, **overrides)  # type: ignore[arg-type]
    update: dict[str, object] = {
        "plugin_path": _PLUGIN_PATH,
        "preset_path": _PRESET_PATH,
        "renderer_version": _RENDERER_VERSION,
    }
    if cadence is not None:
        update["param_sample_cadence"] = cadence
    return cfg.model_copy(update=update)


def _count_wds_audio_rows(tar_path: Path) -> int:
    """Sum the first-axis length of every ``*.audio.npy`` member in a wds tar.

    :param tar_path: Path to a webdataset ``.tar`` shard.
    :returns: Total rendered-sample row count across all audio members.
    """
    rows = 0
    with tarfile.open(tar_path, mode="r") as tar:
        for member in tar.getmembers():
            if not member.name.endswith(f".{AUDIO_FIELD}.npy"):
                continue
            extracted = tar.extractfile(member)
            assert extracted is not None, f"unreadable tar member {member.name}"
            rows += int(np.load(io.BytesIO(extracted.read()), allow_pickle=False).shape[0])
    return rows


@pytest.mark.fake_vst
def test_make_hdf5_dataset_writes_valid_shard_under_fake_plugin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    install_fake_plugin: FakeVST3Plugin,
) -> None:
    """The dataset pipeline produces a valid shard with no real VST3 or X11.

    Four samples in two batches force a mid-shard flush inside the
    held-open editor scope. Asserts shape/dtype/finiteness and that the
    editor-thread crash log never fires — the fake's ``show_editor``
    just blocks on the close event, so there is no realistic failure
    mode here, but keeping the assertion pins the contract for when
    the held-open editor surface evolves.

    :param tmp_path: Destination directory for the shard HDF5 file under test.
    :param monkeypatch: Stubs ``core.logger`` so the crash-gate assertion
        is observable (loguru does not propagate to ``caplog``).
    :param install_fake_plugin: Swaps ``core.load_plugin`` /
        ``core.VST3Plugin`` for the fake before the writer fires.
    """
    num_samples = 4
    render_cfg = _render_cfg(
        num_samples=num_samples,
        samples_per_render_batch=2,
        plugin_reload_cadence="once",
        gui_toggle_cadence="always_on",
    ).model_copy(
        update={
            "plugin_path": _PLUGIN_PATH,
            "preset_path": _PRESET_PATH,
            "renderer_version": _RENDERER_VERSION,
        }
    )
    out = tmp_path / "shard-000000.h5"
    fixed_synth = [_HARDCODED_SYNTH_PARAMS] * num_samples
    fixed_note = [_HARDCODED_NOTE_PARAMS] * num_samples

    with assert_no_logger_exceptions(monkeypatch, core):
        make_hdf5_dataset(
            hdf5_file=out,
            render_cfg=render_cfg,
            fixed_synth_params_list=fixed_synth,
            fixed_note_params_list=fixed_note,
        )

    assert out.exists()
    with h5py.File(out, "r") as f:
        for key in ("audio", "mel_spec", "param_array"):
            assert key in f, f"missing expected dataset: {key}"
        audio_ds = f["audio"]
        assert isinstance(audio_ds, h5py.Dataset)
        assert audio_ds.shape[0] == num_samples
        audio = audio_ds[...]
    assert np.isfinite(audio).all(), "rendered audio contains NaN/Inf"
    assert (np.abs(audio) <= 1.0).all(), "rendered audio exceeds [-1, 1] bounds"


@pytest.mark.fake_vst
def test_make_hdf5_dataset_shard_cadence_writes_one_identical_patch_per_shard(
    tmp_path: Path,
    install_fake_plugin: FakeVST3Plugin,
) -> None:
    """``param_sample_cadence="shard"`` writes a shard whose ``param_array`` rows are identical.

    The on-disk counterpart to the writer-level draws-once pins in
    ``test_writers.py``: drives the real ``make_hdf5_dataset`` writer end-to-end
    under the fake plugin (no fixed-params list — shard cadence draws its own
    single patch) and reads the produced shard back to confirm every encoded
    parameter row equals row 0. This is the #489 variance probe's one-patch-per-
    shard invariant, asserted on the fast CPU loop rather than only under real R2.

    :param tmp_path: Destination directory for the shard HDF5 file under test.
    :param install_fake_plugin: Swaps ``core.load_plugin`` / ``core.VST3Plugin``
        for the fake so the per-render draw runs without a real VST3 or X11.
    """
    num_samples = 4
    render_cfg = _fake_render_cfg(
        num_samples=num_samples,
        samples_per_render_batch=2,
        param_sample_cadence="shard",
    )
    out = tmp_path / "shard-000000.h5"

    make_hdf5_dataset(hdf5_file=out, render_cfg=render_cfg)

    with h5py.File(out, "r") as f:
        param_ds = f["param_array"]
        assert isinstance(param_ds, h5py.Dataset)
        params = param_ds[...]
    assert params.shape[0] == num_samples
    assert np.array_equal(params, np.broadcast_to(params[0], params.shape)), (
        "shard-cadence shard has non-identical param rows"
    )


@pytest.mark.fake_vst
def test_make_hdf5_dataset_shard_cadence_copy_seeds_whole_shard_from_source_row_zero(
    tmp_path: Path,
    install_fake_plugin: FakeVST3Plugin,
) -> None:
    """Shard cadence + a copy source renders the source's row-0 patch across the whole shard.

    The wired #489 single-preset path end to end under the fake plugin: a sample-cadence source
    shard (distinct per-row patches) is decoded by ``fixed_params_from_dataset`` and replayed under
    ``param_sample_cadence="shard"`` via ``make_hdf5_dataset``. Every output row must equal the
    source's first row — proving the shard's single patch is seeded from the copy source (not drawn
    fresh) and reused, with later source rows ignored.

    :param tmp_path: Destination directory for the source and output shards.
    :param install_fake_plugin: Swaps the plugin loader for the fake so renders need no VST3/X11.
    """
    num_samples = 3
    spec = param_specs[_SPEC_NAME]

    source = tmp_path / "shard-000000.h5"
    make_hdf5_dataset(
        hdf5_file=source,
        render_cfg=_fake_render_cfg(num_samples=num_samples, param_sample_cadence="sample"),
    )
    synth_list, note_list = fixed_params_from_dataset(source, spec)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out = out_dir / "shard-000000.h5"
    make_hdf5_dataset(
        hdf5_file=out,
        render_cfg=_fake_render_cfg(num_samples=num_samples, param_sample_cadence="shard"),
        fixed_synth_params_list=synth_list,
        fixed_note_params_list=note_list,
    )

    with h5py.File(source, "r") as f:
        src_ds = f["param_array"]
        assert isinstance(src_ds, h5py.Dataset)
        src_params = src_ds[...]
    with h5py.File(out, "r") as f:
        out_ds = f["param_array"]
        assert isinstance(out_ds, h5py.Dataset)
        out_params = out_ds[...]

    # Source rows must differ so "seeded from row 0" is distinguishable from any other row.
    assert not np.array_equal(src_params[0], src_params[1]), (
        "source rows must differ for this test"
    )
    assert out_params.shape[0] == num_samples
    assert np.array_equal(out_params, np.broadcast_to(out_params[0], out_params.shape)), (
        "shard-cadence copy shard has non-identical param rows"
    )
    # atol absorbs the float32 decode->re-encode round trip through fixed_params_from_dataset.
    assert np.allclose(out_params[0], src_params[0], atol=1e-4), (
        "shard-cadence copy did not seed the shard from the source's first row"
    )


@pytest.mark.fake_vst
def test_make_wds_dataset_rerun_overwrites_rather_than_appends(
    tmp_path: Path,
    install_fake_plugin: FakeVST3Plugin,
) -> None:
    """Re-running the wds writer on an existing path overwrites it (non-resumable).

    ``make_wds_dataset`` pins ``start_idx=0`` and opens the path with
    ``wds.TarWriter`` in write mode, so a re-run must truncate — a second pass
    yields exactly ``samples_per_shard`` rows, not double. Pins the
    non-resumability contract on the fast loop (HDF5 resume is covered in
    ``test_generate_vst_dataset.py``; this is its wds counterpart).

    :param tmp_path: Destination directory for the wds tar shard under test.
    :param install_fake_plugin: Swaps the plugin loader for the fake so the
        render runs without a real VST3 or X11.
    """
    num_samples = 4
    render_cfg = _fake_render_cfg(num_samples=num_samples, samples_per_render_batch=2)
    out = tmp_path / "shard-000000.tar"

    make_wds_dataset(wds_file=out, render_cfg=render_cfg)
    assert _count_wds_audio_rows(out) == num_samples

    make_wds_dataset(wds_file=out, render_cfg=render_cfg)
    assert _count_wds_audio_rows(out) == num_samples, (
        "wds re-run appended instead of overwriting the shard"
    )
