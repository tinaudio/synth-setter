"""End-to-end shard-write proven against a duck-typed ``FakeVST3Plugin``.

No real ``.vst3`` bundle, no X11, no Surge XT — ``install_fake_plugin``
swaps the loader for the fake, so the whole ``make_hdf5_dataset`` path
(batch loop, held-open editor, HDF5 writer, mel-spec computation) runs
on every PR. The real-plugin counterpart at
``test_always_on_integration.py`` stays as the "does Surge XT still
work" gate; this is the "does our pipeline still work" gate.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import h5py
import hdf5plugin  # noqa: F401  side-effect: registers Blosc2 filter for h5py reads
import numpy as np
import pytest

from synth_setter.data.vst import core
from synth_setter.data.vst.writers import make_hdf5_dataset

_ = hdf5plugin  # keep type checkers from flagging the side-effect import

from tests.data.vst._fake_plugin import FakeVST3Plugin  # noqa: E402
from tests.data.vst.test_generate_vst_dataset import (  # noqa: E402  pinned canonical patch
    _HARDCODED_NOTE_PARAMS,
    _HARDCODED_SYNTH_PARAMS,
    _render_cfg,
)

_PLUGIN_PATH = "plugins/fake.vst3"  # never touched on disk — load_plugin is patched
_PRESET_PATH = "presets/fake.vstpreset"
_RENDERER_VERSION = "fake-0.0.0"


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
    fake_logger = MagicMock(wraps=core.logger)
    monkeypatch.setattr(core, "logger", fake_logger)

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

    assert fake_logger.exception.call_count == 0, (
        f"unexpected logger.exception calls: {fake_logger.exception.call_args_list}"
    )


@pytest.mark.fake_vst
def test_same_seed_produces_identical_param_arrays(
    tmp_path: Path,
    install_fake_plugin: FakeVST3Plugin,
) -> None:
    """Two ``make_hdf5_dataset`` runs with the same seed write identical ``param_array`` datasets.

    Exercises the full shard-render stack — RNG seeding → ``param_spec.sample()``
    → HDF5 write — so the per-shard reproducibility guarantee is validated
    end-to-end.

    :param tmp_path: Destination directory for the two output shards.
    :param install_fake_plugin: Swaps the real loader for the fake so no VST binary is needed.
    """
    num_samples = 4
    render_cfg = _render_cfg(
        num_samples=num_samples,
        samples_per_render_batch=2,
        plugin_reload_cadence="once",
        gui_toggle_cadence="never",
    ).model_copy(
        update={
            "plugin_path": _PLUGIN_PATH,
            "preset_path": _PRESET_PATH,
            "renderer_version": _RENDERER_VERSION,
            "seed": 77,
        }
    )

    out_a = tmp_path / "shard_a.h5"
    out_b = tmp_path / "shard_b.h5"
    make_hdf5_dataset(hdf5_file=out_a, render_cfg=render_cfg)
    make_hdf5_dataset(hdf5_file=out_b, render_cfg=render_cfg)

    with h5py.File(out_a, "r") as fa, h5py.File(out_b, "r") as fb:
        params_a = np.asarray(fa["param_array"])
        params_b = np.asarray(fb["param_array"])

    np.testing.assert_array_equal(params_a, params_b)


@pytest.mark.fake_vst
def test_different_seeds_produce_different_param_arrays(
    tmp_path: Path,
    install_fake_plugin: FakeVST3Plugin,
) -> None:
    """Two ``make_hdf5_dataset`` runs with different seeds write different ``param_array``
    datasets.

    Confirms the seed actually changes sampled params — a no-op seeding would make reproducibility
    meaningless.

    :param tmp_path: Destination directory for the two output shards.
    :param install_fake_plugin: Swaps the real loader for the fake so no VST binary is needed.
    """
    num_samples = 4

    def _cfg(seed: int):
        return _render_cfg(
            num_samples=num_samples,
            samples_per_render_batch=2,
            plugin_reload_cadence="once",
            gui_toggle_cadence="never",
        ).model_copy(
            update={
                "plugin_path": _PLUGIN_PATH,
                "preset_path": _PRESET_PATH,
                "renderer_version": _RENDERER_VERSION,
                "seed": seed,
            }
        )

    out_1 = tmp_path / "shard_seed1.h5"
    out_2 = tmp_path / "shard_seed2.h5"
    make_hdf5_dataset(hdf5_file=out_1, render_cfg=_cfg(seed=1))
    make_hdf5_dataset(hdf5_file=out_2, render_cfg=_cfg(seed=2))

    with h5py.File(out_1, "r") as f1, h5py.File(out_2, "r") as f2:
        params_1 = np.asarray(f1["param_array"])
        params_2 = np.asarray(f2["param_array"])

    assert not np.array_equal(params_1, params_2), (
        "Different seeds must produce different param arrays"
    )
