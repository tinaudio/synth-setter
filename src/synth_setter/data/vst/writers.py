"""Per-shard Lance dataset writer.

``make_lance_dataset`` is the sole entrypoint dispatched by the renderer CLI on
the output suffix; it writes a Lance dataset directory, one fragment per render
batch, committed as one dataset and compacted to a single fragment at the end.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path

import numpy as np
from loguru import logger
from pedalboard import VST3Plugin
from tqdm import trange

from synth_setter.data.vst.core import load_plugin, load_preset, run_with_editor_held_open
from synth_setter.data.vst.generate_vst_dataset import (
    SampleSeed,
    VSTDataSample,
    generate_sample,
)
from synth_setter.data.vst.param_spec import NoteParams, ParamSpec
from synth_setter.data.vst.param_spec_registry import resolve_param_spec
from synth_setter.data.vst.renderers import (
    AudioRenderer,
    DawDreamerRenderer,
    PedalboardRenderer,
    TorchSynthRenderer,
)
from synth_setter.data.vst.shapes import DATASET_FIELD_NAMES, dataset_field_shapes
from synth_setter.pipeline.schemas.render_metrics import RenderRejectionMetrics
from synth_setter.pipeline.schemas.spec import RenderConfig


def _sample_batch_arrays(samples: list[VSTDataSample]) -> dict[str, np.ndarray]:
    """Stack rendered samples into writer-field arrays.

    :param samples: Rendered samples in row order.
    :returns: Mapping keyed by ``DATASET_FIELD_NAMES``.
    :rtype: dict[str, np.ndarray]
    """
    audio_name, mel_name, param_name = DATASET_FIELD_NAMES
    return {
        audio_name: np.stack([s.audio.T for s in samples], axis=0).astype(np.float16),
        mel_name: np.stack([s.mel_spec for s in samples], axis=0),
        param_name: np.stack([s.param_array for s in samples], axis=0),
    }


def _validate_fixed_params_lengths(
    *,
    num_samples: int,
    fixed_synth_params_list: list[dict[str, float]] | None,
    fixed_note_params_list: list[NoteParams] | None,
) -> None:
    """Raise ``ValueError`` unless each fixed-params list spans the whole shard.

    Fixed params are indexed by absolute row ``i`` (see the loop in
    ``_render_in_batches``), so each list holds one entry per shard row —
    ``num_samples`` entries — even on a resumed run that only re-renders the tail
    (the already-written rows keep their list slots). We require exact equality
    (not ``>=``) so a mismatched source — e.g. a dataset copy whose shard row
    count differs from ``samples_per_shard`` — is caught here instead of silently
    truncated.

    :param num_samples: Number of samples this shard holds (``samples_per_shard``).
    :param fixed_synth_params_list: Optional pre-set synth params, one dict per shard row.
    :param fixed_note_params_list: Optional pre-set note params, one dict per shard row.
    :raises ValueError: If either list's length is not exactly ``num_samples``.
    """
    for name, lst in [
        ("fixed_synth_params_list", fixed_synth_params_list),
        ("fixed_note_params_list", fixed_note_params_list),
    ]:
        if lst is not None and len(lst) != num_samples:
            raise ValueError(
                f"{name} has length {len(lst)}, expected exactly "
                f"num_samples = {num_samples} (one entry per shard row); "
                "for a dataset copy, the source shard's row count must equal "
                "samples_per_shard"
            )


def _make_renderer(render_cfg: RenderConfig, plugin: VST3Plugin | None = None) -> AudioRenderer:
    """Construct the configured audio renderer for one shard or render.

    :param render_cfg: Render settings that identify the backend and audio shape.
    :param plugin: Preloaded pedalboard plugin for ``plugin_reload_cadence="once"``.
    :returns: Renderer configured for the requested backend.
    """
    if render_cfg.renderer_backend == "dawdreamer":
        from synth_setter.data.vst.param_map import load_param_map
        from synth_setter.resources import as_file, param_map

        with as_file(param_map(render_cfg.param_spec_name)) as path:
            joint_map = load_param_map(path)
        return DawDreamerRenderer(
            plugin_path=render_cfg.plugin_path,
            sample_rate=render_cfg.sample_rate,
            channels=render_cfg.channels,
            signal_duration_seconds=render_cfg.signal_duration_seconds,
            plugin_state_path=render_cfg.plugin_state_path,
            parameter_map=joint_map,
            reload_plugin_each_render=render_cfg.plugin_reload_cadence == "render",
        )
    if render_cfg.renderer_backend == "torchsynth":
        return TorchSynthRenderer(
            plugin_path=render_cfg.plugin_path,
            sample_rate=render_cfg.sample_rate,
            channels=render_cfg.channels,
            signal_duration_seconds=render_cfg.signal_duration_seconds,
        )
    return PedalboardRenderer(
        plugin_path=render_cfg.plugin_path,
        sample_rate=render_cfg.sample_rate,
        channels=render_cfg.channels,
        signal_duration_seconds=render_cfg.signal_duration_seconds,
        plugin_state_path=render_cfg.plugin_state_path,
        plugin=plugin,
    )


def _render_in_batches(
    *,
    render_cfg: RenderConfig,
    param_spec: ParamSpec,
    start_idx: int,
    fixed_synth_params_list: list[dict[str, float]] | None,
    fixed_note_params_list: list[NoteParams] | None,
    flush_batch: Callable[[list[VSTDataSample], int], None],
) -> RenderRejectionMetrics:
    """Render samples from ``start_idx`` to ``render_cfg.samples_per_shard`` in fixed-size batches.

    The flush callback is invoked once per full ``samples_per_render_batch``
    batch plus once for the trailing remainder, with the batch and its starting
    row index.

    :param render_cfg: Per-shard renderer config from the dataset spec.
    :param param_spec: Resolved parameter spec for the render.
    :param start_idx: First shard-local row this run renders (non-zero on resume).
    :param fixed_synth_params_list: Pre-set synth params (or ``None``), indexed by absolute row.
        Under shard cadence the shard's single patch is seeded from row ``start_idx`` and reused;
        callers pin ``start_idx=0`` for shard cadence, so that seed is row 0 and the
        remaining rows go unused.
    :param fixed_note_params_list: Pre-set note params (or ``None``), indexed by absolute row;
        shares the synth list's shard-cadence seed-from-``start_idx``-and-reuse behavior.
    :param flush_batch: Called with ``(batch, batch_start_idx)`` to persist each batch.
    :returns: Counts of silent and clipped draws rejected across the shard.
    :raises RuntimeError: ``gui_toggle_cadence="always_on"`` reaches the
        renderer without ``plugin_reload_cadence="once"`` (validator regression).
    """
    num_samples = render_cfg.samples_per_shard
    share_params = render_cfg.param_sample_cadence == "shard"
    clipped_rejections = 0
    silent_rejections = 0

    # "once" reuses one renderer per shard; "render" reloads for each attempt (see #705).
    cached_plugin: VST3Plugin | None = None
    cached_renderer: AudioRenderer | None = None
    if render_cfg.plugin_reload_cadence == "once":
        if render_cfg.renderer_backend == "pedalboard":
            cached_plugin = load_plugin(render_cfg.plugin_path)
            load_preset(cached_plugin, render_cfg.plugin_state_path)
        cached_renderer = _make_renderer(render_cfg, cached_plugin)

    def _render_loop() -> None:
        nonlocal clipped_rejections, silent_rejections
        sample_batch: list[VSTDataSample] = []
        sample_batch_start = start_idx
        warmup_done = False
        # param_sample_cadence="shard": the first rendered row (start_idx) sets the shard's single
        # patch (drawn fresh, or copied from the source's same row); later renders reuse it (#489).
        shared_synth: dict[str, float] | None = None
        shared_note: NoteParams | None = None
        for i in trange(start_idx, num_samples):
            logger.info(f"Making sample {i}")
            warmup_this_render = render_cfg.gui_toggle_cadence == "render" or (
                render_cfg.gui_toggle_cadence == "once" and not warmup_done
            )
            # Fixed params are indexed by absolute row ``i`` (full-shard lists),
            # so a resumed run still reads the source row matching each output row.
            fixed_synth: dict[str, float] | None
            fixed_note: NoteParams | None
            if share_params and shared_synth is not None:
                fixed_synth, fixed_note = shared_synth, shared_note
            else:
                fixed_synth = (
                    fixed_synth_params_list[i] if fixed_synth_params_list is not None else None
                )
                fixed_note = (
                    fixed_note_params_list[i] if fixed_note_params_list is not None else None
                )
            renderer = cached_renderer or _make_renderer(render_cfg)
            sample = generate_sample(
                renderer=renderer,
                velocity=render_cfg.velocity,
                min_loudness=render_cfg.min_loudness,
                param_spec=param_spec,
                fixed_synth_params=fixed_synth,
                fixed_note_params=fixed_note,
                warmup=warmup_this_render,
                # The split-local index stays stable across shard layouts and resumes (#884).
                seed=SampleSeed(
                    master_seed=render_cfg.base_seed,
                    sample_idx=render_cfg.sample_offset + i,
                    max_attempts=render_cfg.attempts_per_sample,
                ),
            )
            if share_params and shared_synth is None:
                shared_synth = sample.synth_params
                shared_note = sample.note_params
            clipped_rejections += sample.clipped_rejections
            silent_rejections += sample.silent_rejections
            sample_batch.append(sample)
            if warmup_this_render and render_cfg.gui_toggle_cadence == "once":
                warmup_done = True
            if len(sample_batch) == render_cfg.samples_per_render_batch:
                flush_batch(sample_batch, sample_batch_start)
                sample_batch = []
                sample_batch_start += render_cfg.samples_per_render_batch

        if sample_batch:
            flush_batch(sample_batch, sample_batch_start)

    # always_on: main thread blocks in ``show_editor`` while ``_render_loop`` runs
    # on a worker (pedalboard requires show_editor on the main thread, #1204).
    # RenderConfig validator pairs always_on with plugin_reload_cadence="once"
    # so cached_plugin is non-None on this branch (#1187).
    if render_cfg.gui_toggle_cadence == "always_on":
        if cached_plugin is None:
            raise RuntimeError(
                "always_on reached the renderer without a cached plugin; "
                "RenderConfig._always_on_requires_plugin_reload_once validator "
                "should have rejected this combination."
            )
        run_with_editor_held_open(cached_plugin, _render_loop)
    else:
        _render_loop()

    return RenderRejectionMetrics(
        clipped=clipped_rejections,
        silent=silent_rejections,
    )


def make_lance_dataset(
    lance_dir: Path | str,
    render_cfg: RenderConfig,
    *,
    shard_id: int | None = None,
    fixed_synth_params_list: list[dict[str, float]] | None = None,
    fixed_note_params_list: list[NoteParams] | None = None,
) -> RenderRejectionMetrics:
    """Render ``render_cfg.samples_per_shard`` samples to a Lance dataset directory.

    Not resumable: any dataset already at ``lance_dir`` is overwritten on each
    run. Audio is stored as ``float16``; ``mel_spec`` and ``param_array`` stay
    ``float32``. The shard metadata is embedded in Arrow schema metadata so
    validation and finalize recover the sidecar payload at read time. Each
    render batch becomes one Lance fragment, committed as one dataset at the
    end, then compacted to a single fragment with pre-compaction manifests and
    data files removed.

    :param lance_dir: Destination ``.lance`` dataset directory.
    :param render_cfg: Per-shard renderer config from the dataset spec.
    :param shard_id: Logical shard number stored in row debug documents; ``None`` for ad hoc renders.
    :param fixed_synth_params_list: Optional pre-set synth params, one dict per
        shard row. Must have length ``samples_per_shard``. Under
        ``param_sample_cadence="shard"`` only row 0 is consumed (it seeds the
        shard's single patch); rows 1..N are required but unused.
    :param fixed_note_params_list: Optional pre-set note params; same full-shard
        contract as ``fixed_synth_params_list``.
    :returns: Counts of silent and clipped draws rejected across the shard.
    """
    # Function-local so importing this module (e.g. from the launcher) never
    # pays the `lance` import cost.
    import lance

    from synth_setter.pipeline.data.lance_shard import (
        commit_lance_dataset,
        lance_fragment,
        lance_schema,
        record_batch_from_arrays,
        seed_debug_array,
    )

    param_spec = resolve_param_spec(render_cfg.param_spec_name)
    meta = render_cfg.shard_metadata()
    start_idx = 0
    lance_path = Path(lance_dir)

    _validate_fixed_params_lengths(
        num_samples=render_cfg.samples_per_shard,
        fixed_synth_params_list=fixed_synth_params_list,
        fixed_note_params_list=fixed_note_params_list,
    )
    schema = lance_schema(dataset_field_shapes(render_cfg, param_spec.encoded_width), meta)
    lance_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = Path(tempfile.mkdtemp(dir=lance_path.parent, prefix=f".{lance_path.name}.tmp-"))

    try:
        fragments: list[lance.fragment.FragmentMetadata] = []
        shard_parameter_attempt: int | None = None
        fixed_synth = fixed_synth_params_list is not None
        fixed_note = fixed_note_params_list is not None
        parameter_source = (
            "fixed" if fixed_synth and fixed_note else "mixed" if fixed_synth or fixed_note else "sampled"
        )

        def _flush(batch: list[VSTDataSample], batch_start: int) -> None:
            nonlocal shard_parameter_attempt
            sample_indices = [
                render_cfg.sample_offset + batch_start + row for row in range(len(batch))
            ]
            sampled_shard_parameters = (
                render_cfg.param_sample_cadence == "shard" and parameter_source != "fixed"
            )
            if sampled_shard_parameters and shard_parameter_attempt is None:
                shard_parameter_attempt = batch[0].attempt
            parameter_sample_idx = render_cfg.sample_offset if sampled_shard_parameters else None
            debug = seed_debug_array(
                render_cfg.base_seed,
                sample_indices,
                [sample.attempt for sample in batch],
                shard_id=shard_id,
                parameter_sample_idx=parameter_sample_idx,
                parameter_attempt=shard_parameter_attempt,
                parameter_source=parameter_source,
            )
            record_batch = record_batch_from_arrays(
                _sample_batch_arrays(batch), schema, debug=debug
            )
            fragments.append(lance_fragment(staging_path, schema, record_batch))

        # Commit only after a clean render: orphaned fragment data files from a failed
        # run stay uncommitted (no dataset manifest references them).
        metrics = _render_in_batches(
            render_cfg=render_cfg,
            param_spec=param_spec,
            start_idx=start_idx,
            fixed_synth_params_list=fixed_synth_params_list,
            fixed_note_params_list=fixed_note_params_list,
            flush_batch=_flush,
        )
        commit_lance_dataset(staging_path, schema, fragments)
        # Compact per-batch fragments into one, then drop the pre-compaction manifest and
        # its data files; delete_unverified is safe — the staging dir is exclusively ours.
        dataset = lance.dataset(str(staging_path))
        dataset.optimize.compact_files(target_rows_per_fragment=render_cfg.samples_per_shard)
        dataset.cleanup_old_versions(older_than=timedelta(0), delete_unverified=True)
        if lance_path.exists():
            shutil.rmtree(lance_path)
        staging_path.rename(lance_path)
        return metrics
    finally:
        if staging_path.exists():
            shutil.rmtree(staging_path, ignore_errors=True)
