"""End-to-end shard-write proven against a duck-typed ``FakeVST3Plugin``.

No real ``.vst3`` bundle, no X11, no Surge XT — ``install_fake_plugin``
swaps the loader for the fake, so the whole ``make_lance_dataset`` path
(batch loop, held-open editor, Lance writer, mel-spec computation) runs
on every PR. The real-plugin counterpart at
``test_always_on_integration.py`` stays as the "does Surge XT still
work" gate; this is the "does our pipeline still work" gate.
"""

from __future__ import annotations

from pathlib import Path

import lance
import numpy as np
import pytest

from synth_setter.data.vst import core
from synth_setter.data.vst.shapes import PARAM_ARRAY_FIELD
from synth_setter.data.vst.writers import make_lance_dataset
from synth_setter.pipeline.ci.validate_shard import validate_shard
from synth_setter.pipeline.data.lance_shard import iter_lance_column_rows, read_shard_metadata
from synth_setter.pipeline.schemas.shard_metadata import ShardMetadata
from synth_setter.pipeline.schemas.spec import DatasetSpec, RenderConfig
from tests.data.vst._fake_plugin import FakeVST3Plugin  # noqa: E402
from tests.data.vst.test_generate_vst_dataset import (  # noqa: E402  pinned canonical patch
    _HARDCODED_NOTE_PARAMS,
    _HARDCODED_SYNTH_PARAMS,
    _render_cfg,
)
from tests.helpers.finalize_shards import build_lance_smoke_spec  # noqa: E402
from tests.helpers.logger_assertions import assert_no_logger_exceptions  # noqa: E402

_PLUGIN_PATH = "plugins/fake.vst3"  # never touched on disk — load_plugin is patched
_PRESET_PATH = "presets/fake.vstpreset"
_RENDERER_VERSION = "fake-0.0.0"


def _fake_render_cfg(**overrides: object) -> RenderConfig:
    """Build a ``RenderConfig`` pointing at the fake plugin paths.

    Wraps the canonical ``_render_cfg`` and rebinds ``plugin_path`` / ``plugin_state_path`` /
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
        "plugin_state_path": _PRESET_PATH,
        "renderer_version": _RENDERER_VERSION,
    }
    if cadence is not None:
        update["param_sample_cadence"] = cadence
    return cfg.model_copy(update=update)


def _lance_spec_for(render_cfg: RenderConfig) -> DatasetSpec:
    """Build a one-shard lance ``DatasetSpec`` around ``render_cfg``.

    Gives ``validate_shard`` a spec whose render config matches what the
    writer under test actually rendered.

    :param render_cfg: The fake-plugin render config driving the writer.
    :returns: A frozen lance ``DatasetSpec`` with a single train shard.
    """
    return build_lance_smoke_spec(
        task_name="fake-plugin-lance-e2e",
        train_val_test_sizes=(render_cfg.samples_per_shard, 0, 0),
        render=render_cfg,
        base_seed=render_cfg.base_seed,
    )


@pytest.mark.fake_vst
def test_make_lance_dataset_always_on_writes_valid_shard_under_fake_plugin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    install_fake_plugin: FakeVST3Plugin,
) -> None:
    """The dataset pipeline produces a valid shard with no real VST3 or X11.

    Four samples in two batches force a mid-shard flush inside the
    held-open editor scope. Asserts row count/finiteness and that the
    editor-thread crash log never fires — the fake's ``show_editor``
    just blocks on the close event, so there is no realistic failure
    mode here, but keeping the assertion pins the contract for when
    the held-open editor surface evolves.

    :param tmp_path: Destination directory for the shard Lance dataset under test.
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
            "plugin_state_path": _PRESET_PATH,
            "renderer_version": _RENDERER_VERSION,
        }
    )
    out = tmp_path / "shard-000000.lance"
    fixed_synth = [_HARDCODED_SYNTH_PARAMS] * num_samples
    fixed_note = [_HARDCODED_NOTE_PARAMS] * num_samples

    with assert_no_logger_exceptions(monkeypatch, core):
        make_lance_dataset(
            out,
            render_cfg,
            fixed_synth_params_list=fixed_synth,
            fixed_note_params_list=fixed_note,
        )

    assert out.exists()
    dataset = lance.dataset(str(out))
    for key in ("audio", "mel_spec", "param_array"):
        assert key in dataset.schema.names, f"missing expected column: {key}"
    assert dataset.count_rows() == num_samples
    audio = np.stack(list(iter_lance_column_rows(out, "audio")), axis=0).astype(np.float32)
    assert np.isfinite(audio).all(), "rendered audio contains NaN/Inf"
    assert (np.abs(audio) <= 1.0).all(), "rendered audio exceeds [-1, 1] bounds"


@pytest.mark.fake_vst
def test_make_lance_dataset_shard_cadence_writes_one_identical_patch_per_shard(
    tmp_path: Path,
    install_fake_plugin: FakeVST3Plugin,
) -> None:
    """``param_sample_cadence="shard"`` writes a shard whose ``param_array`` rows are identical.

    The on-disk counterpart to the writer-level draws-once pins in
    ``test_writers.py``: drives the real ``make_lance_dataset`` writer end-to-end
    under the fake plugin (no fixed-params list — shard cadence draws its own
    single patch) and reads the produced shard back to confirm every encoded
    parameter row equals row 0. This is the #489 variance probe's one-patch-per-
    shard invariant, asserted on the fast CPU loop.

    :param tmp_path: Destination directory for the shard Lance dataset under test.
    :param install_fake_plugin: Swaps ``core.load_plugin`` / ``core.VST3Plugin``
        for the fake so the per-render draw runs without a real VST3 or X11.
    """
    num_samples = 4
    render_cfg = _fake_render_cfg(
        num_samples=num_samples,
        samples_per_render_batch=2,
        param_sample_cadence="shard",
    )
    out = tmp_path / "shard-000000.lance"

    make_lance_dataset(out, render_cfg)

    params = np.stack(list(iter_lance_column_rows(out, PARAM_ARRAY_FIELD)), axis=0)
    assert params.shape[0] == num_samples
    assert np.array_equal(params, np.broadcast_to(params[0], params.shape)), (
        "shard-cadence shard has non-identical param rows"
    )


@pytest.mark.fake_vst
def test_make_lance_dataset_writes_validator_passing_shard_under_fake_plugin(
    tmp_path: Path,
    install_fake_plugin: FakeVST3Plugin,
) -> None:
    """The real Lance writer produces a shard ``validate_shard`` accepts.

    Drives ``make_lance_dataset`` end-to-end (batch loop, per-batch flushes via
    a ``samples_per_render_batch`` below ``num_samples``, schema construction,
    writer close) and
    checks the produced file through the production validator — schema,
    dtypes, inner shapes, row count — plus a whole-model ``ShardMetadata``
    round-trip against the render config.

    :param tmp_path: Destination directory for the Lance shard under test.
    :param install_fake_plugin: Swaps the plugin loader for the fake so the
        render runs without a real VST3 or X11.
    """
    num_samples = 4
    render_cfg = _fake_render_cfg(num_samples=num_samples, samples_per_render_batch=2)
    spec = _lance_spec_for(render_cfg)
    out = tmp_path / spec.shards[0].filename

    make_lance_dataset(
        out,
        render_cfg,
        fixed_synth_params_list=[_HARDCODED_SYNTH_PARAMS] * num_samples,
        fixed_note_params_list=[_HARDCODED_NOTE_PARAMS] * num_samples,
    )

    assert validate_shard(out, spec) == []
    meta = read_shard_metadata(lance.dataset(str(out)).schema)
    # Whole-model equality: a new ShardMetadata field fails construction here,
    # forcing this round-trip pin to cover it.
    assert meta == ShardMetadata(
        velocity=render_cfg.velocity,
        signal_duration_seconds=render_cfg.signal_duration_seconds,
        sample_rate=render_cfg.sample_rate,
        channels=render_cfg.channels,
        min_loudness=render_cfg.min_loudness,
        base_seed=render_cfg.base_seed,
        attempts_per_sample=render_cfg.attempts_per_sample,
    )


@pytest.mark.fake_vst
def test_make_lance_dataset_rerun_overwrites_rather_than_appends(
    tmp_path: Path,
    install_fake_plugin: FakeVST3Plugin,
) -> None:
    """Re-running the Lance writer on an existing path overwrites it (non-resumable).

    ``make_lance_dataset`` pins ``start_idx=0`` and commits with overwrite
    semantics, so a second pass yields exactly ``samples_per_shard`` rows, not
    double.

    :param tmp_path: Destination directory for the Lance shard under test.
    :param install_fake_plugin: Swaps the plugin loader for the fake so the
        render runs without a real VST3 or X11.
    """
    num_samples = 4
    render_cfg = _fake_render_cfg(num_samples=num_samples, samples_per_render_batch=2)
    fixed_synth = [_HARDCODED_SYNTH_PARAMS] * num_samples
    fixed_note = [_HARDCODED_NOTE_PARAMS] * num_samples
    out = tmp_path / "shard-000000.lance"

    make_lance_dataset(
        out,
        render_cfg,
        fixed_synth_params_list=fixed_synth,
        fixed_note_params_list=fixed_note,
    )
    assert lance.dataset(str(out)).count_rows() == num_samples
    first_run_params = np.stack(list(iter_lance_column_rows(out, PARAM_ARRAY_FIELD)), axis=0)

    make_lance_dataset(
        out,
        render_cfg,
        fixed_synth_params_list=fixed_synth,
        fixed_note_params_list=fixed_note,
    )
    assert lance.dataset(str(out)).count_rows() == num_samples, (
        "lance re-run appended instead of overwriting the shard"
    )
    rerun_params = np.stack(list(iter_lance_column_rows(out, PARAM_ARRAY_FIELD)), axis=0)
    np.testing.assert_array_equal(rerun_params, first_run_params)
