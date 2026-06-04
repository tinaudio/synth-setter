"""Per-shard dataset writers — HDF5 (resumable) and webdataset tar (new).

Consist of two entrypoints dispatched by the renderer CLI on the output file's suffix:
``make_hdf5_dataset`` keeps the resumable HDF5 path (signature takes a path and
opens the file internally), and ``make_wds_dataset`` is the new tar-shard
writer using ``webdataset.TarWriter``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, cast

import h5py
import numpy as np
import webdataset as wds
from loguru import logger
from pedalboard import VST3Plugin
from tqdm import trange

from synth_setter.data.vst import param_specs
from synth_setter.data.vst.core import load_plugin, load_preset, run_with_editor_held_open
from synth_setter.data.vst.generate_vst_dataset import (
    VSTDataSample,
    create_datasets_and_get_start_idx,
    generate_sample,
)
from synth_setter.data.vst.param_spec import ParamSpec
from synth_setter.data.vst.shapes import DATASET_FIELD_NAMES
from synth_setter.pipeline.schemas.shard_metadata import ShardMetadata
from synth_setter.pipeline.schemas.spec import RenderConfig


class _WdsTarSink(Protocol):
    """Minimal surface from ``wds.TarWriter`` used by the wds writer path.

    The webdataset library lacks PEP 561 type stubs, so direct ``wds.TarWriter``
    references trigger ``reportAttributeAccessIssue`` under pyright. Typing the
    helper signatures against this Protocol keeps the call surface narrow and
    confines the type-ignore to the single ``wds.TarWriter(...)`` instantiation.
    """

    def write(self, sample: dict[str, Any]) -> None:
        """Write a single sample dict to the underlying tar stream.

        :param sample: Mapping containing a ``__key__`` plus member-name → value entries.
        """
        ...

    def close(self) -> None:
        """Close the underlying tar stream."""
        ...

    def __enter__(self) -> _WdsTarSink:
        """Enter the context manager and return ``self``.

        :returns: This sink, for use inside a ``with`` block.
        :rtype: _WdsTarSink
        """
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the context manager, closing the underlying tar stream.

        :param exc_type: Exception type if raised inside the block, else ``None``.
        :param exc_val: Exception instance if raised inside the block, else ``None``.
        :param exc_tb: Traceback if an exception was raised inside the block, else ``None``.
        """
        ...


def save_hdf5_samples(
    samples: list[VSTDataSample],
    audio_dataset: h5py.Dataset,
    mel_dataset: h5py.Dataset,
    param_dataset: h5py.Dataset,
    start_idx: int,
) -> None:
    """Append a batch of rendered samples to the three HDF5 datasets in place.

    :param samples: Rendered samples in row order; the first lands at ``start_idx``.
    :param audio_dataset: Pre-created HDF5 audio dataset (shape ``(N, C, T)``).
    :param mel_dataset: Pre-created HDF5 mel-spectrogram dataset (shape ``(N, C, M, F)``).
    :param param_dataset: Pre-created HDF5 parameter-array dataset (shape ``(N, P)``).
    :param start_idx: Row at which the batch's first sample is written.
    """
    logger.info(f"Saving {len(samples)} samples to hdf5...")
    audios = np.stack([s.audio.T for s in samples], axis=0)
    mel_specs = np.stack([s.mel_spec for s in samples], axis=0)
    param_arrays = np.stack([s.param_array for s in samples], axis=0)

    end = start_idx + len(samples)
    audio_dataset[start_idx:end, :, :] = audios
    mel_dataset[start_idx:end, :, :] = mel_specs
    param_dataset[start_idx:end, :] = param_arrays

    logger.info(f"{len(samples)} hdf5 samples written!")


def save_wds_samples(
    samples: list[VSTDataSample],
    sink: _WdsTarSink,
    start_idx: int,
) -> None:
    """Write a batch of rendered samples as a single tar entry keyed by ``start_idx``.

    Audio is cast to ``float16`` to match the h5 path's storage precision so
    consumers see the same dtype regardless of which writer produced the shard;
    ``mel_spec`` and ``param_array`` stay ``float32``.

    :param samples: Rendered samples in row order; all land under the same tar key.
    :param sink: An open ``wds.TarWriter``-compatible sink (Protocol typed).
    :param start_idx: Logical row index of the batch's first sample — also the tar key.
    """
    logger.info(f"Saving {len(samples)} samples to wds...")
    audios = np.stack([s.audio.T for s in samples], axis=0).astype(np.float16)
    mel_specs = np.stack([s.mel_spec for s in samples], axis=0)
    param_arrays = np.stack([s.param_array for s in samples], axis=0)

    audio_name, mel_name, param_name = DATASET_FIELD_NAMES
    sink.write(
        {
            "__key__": f"{start_idx:08d}",
            f"{audio_name}.npy": audios,
            f"{mel_name}.npy": mel_specs,
            f"{param_name}.npy": param_arrays,
        }
    )

    logger.info(f"{len(samples)} wds samples written!")


def _validate_fixed_params_lengths(
    *,
    num_samples: int,
    fixed_synth_params_list: list[dict[str, float]] | None,
    fixed_note_params_list: list[dict[str, int | tuple[float, float]]] | None,
) -> None:
    """Raise ``ValueError`` unless each fixed-params list spans the whole shard.

    Fixed params are indexed by absolute row ``i`` (see
    ``_generate_sample_for_index``), so each list holds one entry per shard row —
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


def _generate_sample_for_index(
    i: int,
    *,
    plugin_path: str,
    preset_path: str,
    velocity: int,
    signal_duration_seconds: float,
    sample_rate: float,
    channels: int,
    min_loudness: float,
    param_spec: ParamSpec,
    fixed_synth_params_list: list[dict[str, float]] | None,
    fixed_note_params_list: list[dict[str, int | tuple[float, float]]] | None,
    plugin: VST3Plugin | None = None,
    warmup: bool = False,
) -> VSTDataSample:
    """Render the ``i``-th sample, picking up the ``i``-th fixed-params entry.

    Fixed params are indexed by absolute row ``i`` so a resumed run (which only
    re-renders rows ``[start_idx, num_samples)``) still reads the source row that
    matches each output row.

    :param i: Absolute row index this call is rendering, and the fixed-params index.
    :param plugin_path: Path to the VST3 bundle to load (ignored when ``plugin`` is supplied).
    :param preset_path: Path to the ``.vstpreset`` to apply (ignored when ``plugin`` is supplied).
    :param velocity: MIDI velocity in ``[0, 127]``.
    :param signal_duration_seconds: Duration of the rendered clip in seconds.
    :param sample_rate: Sample rate of the rendered clip in Hz.
    :param channels: Number of audio channels rendered.
    :param min_loudness: Loudness gate threshold in LUFS.
    :param param_spec: Parameter spec used to sample/encode parameters.
    :param fixed_synth_params_list: Optional pre-set synth params, indexed by ``i``.
    :param fixed_note_params_list: Optional pre-set note params, indexed by ``i``.
    :param plugin: Optional pre-loaded plugin reused across the shard's renders.
    :param warmup: Run ``warmup_plugin`` on the plugin used for this render.
    :returns: The freshly rendered sample.
    :rtype: VSTDataSample
    """
    return generate_sample(
        plugin_path,
        velocity=velocity,
        signal_duration_seconds=signal_duration_seconds,
        sample_rate=sample_rate,
        channels=channels,
        min_loudness=min_loudness,
        param_spec=param_spec,
        preset_path=preset_path,
        fixed_synth_params=(
            fixed_synth_params_list[i] if fixed_synth_params_list is not None else None
        ),
        fixed_note_params=(
            fixed_note_params_list[i] if fixed_note_params_list is not None else None
        ),
        plugin=plugin,
        warmup=warmup,
    )


def _render_in_batches(
    *,
    render_cfg: RenderConfig,
    param_spec: ParamSpec,
    start_idx: int,
    fixed_synth_params_list: list[dict[str, float]] | None,
    fixed_note_params_list: list[dict[str, int | tuple[float, float]]] | None,
    flush_batch: Callable[[list[VSTDataSample], int], None],
) -> None:
    """Render samples from ``start_idx`` to ``render_cfg.samples_per_shard`` in fixed-size batches.

    The h5 and wds writers share this loop verbatim: only the per-batch flush
    differs (HDF5 dataset slice assignment vs. tar member write). The flush
    callback is invoked once per full ``samples_per_render_batch`` batch plus once
    for the trailing remainder, with the batch and its starting row index.

    :param render_cfg: Per-shard renderer config from the dataset spec.
    :param param_spec: Resolved parameter spec for the render.
    :param start_idx: First absolute row index this run renders (non-zero on resume).
    :param fixed_synth_params_list: Optional pre-set synth params, indexed by absolute row.
    :param fixed_note_params_list: Optional pre-set note params, indexed by absolute row.
    :param flush_batch: Called with ``(batch, batch_start_idx)`` to persist each batch.
    :raises RuntimeError: ``gui_toggle_cadence="always_on"`` reaches the
        renderer without ``plugin_reload_cadence="once"`` (validator regression).
    """
    num_samples = render_cfg.samples_per_shard
    # plugin_reload_cadence="once": load + preset once per shard, reuse instance (#705).
    # "render" (default): cached_plugin stays None; each render reloads (#489 historical).
    cached_plugin: VST3Plugin | None = None
    if render_cfg.plugin_reload_cadence == "once":
        cached_plugin = load_plugin(render_cfg.plugin_path)
        load_preset(cached_plugin, render_cfg.preset_path)

    def _render_loop() -> None:
        sample_batch: list[VSTDataSample] = []
        sample_batch_start = start_idx
        warmup_done = False
        for i in trange(start_idx, num_samples):
            logger.info(f"Making sample {i}")
            warmup_this_render = render_cfg.gui_toggle_cadence == "render" or (
                render_cfg.gui_toggle_cadence == "once" and not warmup_done
            )
            sample_batch.append(
                _generate_sample_for_index(
                    i,
                    plugin_path=render_cfg.plugin_path,
                    preset_path=render_cfg.preset_path,
                    velocity=render_cfg.velocity,
                    signal_duration_seconds=render_cfg.signal_duration_seconds,
                    sample_rate=render_cfg.sample_rate,
                    channels=render_cfg.channels,
                    min_loudness=render_cfg.min_loudness,
                    param_spec=param_spec,
                    fixed_synth_params_list=fixed_synth_params_list,
                    fixed_note_params_list=fixed_note_params_list,
                    plugin=cached_plugin,
                    warmup=warmup_this_render,
                )
            )
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


def _shard_metadata_from_render(render_cfg: RenderConfig) -> ShardMetadata:
    """Project a ``RenderConfig`` onto the per-shard sidecar metadata fields.

    Single source of truth for the five render-derived attrs that both the
    HDF5 ``audio.attrs`` sidecar and the wds ``metadata.json`` tar member
    expose. Keeping projection here means the two writers can never drift.

    :param render_cfg: Per-shard renderer config from the dataset spec.
    :returns: Strict ``ShardMetadata`` with the five render-derived fields filled.
    :rtype: ShardMetadata
    """
    return ShardMetadata(
        velocity=render_cfg.velocity,
        signal_duration_seconds=render_cfg.signal_duration_seconds,
        sample_rate=render_cfg.sample_rate,
        channels=render_cfg.channels,
        min_loudness=render_cfg.min_loudness,
    )


def make_hdf5_dataset(
    hdf5_file: Path | str,
    render_cfg: RenderConfig,
    *,
    fixed_synth_params_list: list[dict[str, float]] | None = None,
    fixed_note_params_list: list[dict[str, int | tuple[float, float]]] | None = None,
) -> None:
    """Render ``render_cfg.samples_per_shard`` samples to an HDF5 file at ``hdf5_file``.

    Resumable: a partially-written file picks up at the first all-zero row, so
    a crashed worker can re-run with the same args and only the missing tail is
    rendered. Audio is stored as ``float16`` (Blosc2-compressed); ``mel_spec``
    and ``param_array`` are ``float32``. The five sidecar attrs (velocity,
    signal duration, sample rate, channels, min_loudness) are written to
    ``audio.attrs`` from a single ``ShardMetadata`` instance — the same
    instance ``make_wds_dataset`` uses for its ``metadata.json`` member, so
    both formats expose identical metadata.

    :param hdf5_file: Destination HDF5 path; opened in append mode so partial
        files can resume.
    :param render_cfg: Per-shard renderer config from the dataset spec.
    :param fixed_synth_params_list: Optional pre-set synth params, one dict per
        shard row. Must have length ``samples_per_shard`` (the full shard); rows
        are indexed absolutely, so a resumed run re-renders only its tail while
        still reading the matching source row.
    :param fixed_note_params_list: Optional pre-set note params; same full-shard
        contract as ``fixed_synth_params_list``.
    """
    param_spec = param_specs[render_cfg.param_spec_name]
    meta = _shard_metadata_from_render(render_cfg)
    # Validate before opening the file so a bad fixed-params list (e.g. a copy
    # source whose row count != samples_per_shard) fails without leaving an
    # empty output shard on disk.
    _validate_fixed_params_lengths(
        num_samples=render_cfg.samples_per_shard,
        fixed_synth_params_list=fixed_synth_params_list,
        fixed_note_params_list=fixed_note_params_list,
    )
    with h5py.File(hdf5_file, "a") as h5:
        audio_dataset, mel_dataset, param_dataset, start_idx = create_datasets_and_get_start_idx(
            hdf5_file=h5,
            num_samples=render_cfg.samples_per_shard,
            channels=render_cfg.channels,
            sample_rate=render_cfg.sample_rate,
            signal_duration_seconds=render_cfg.signal_duration_seconds,
            num_params=len(param_spec),
        )

        for k, v in meta.model_dump().items():
            audio_dataset.attrs[k] = v

        def _flush(batch: list[VSTDataSample], batch_start: int) -> None:
            save_hdf5_samples(batch, audio_dataset, mel_dataset, param_dataset, batch_start)

        _render_in_batches(
            render_cfg=render_cfg,
            param_spec=param_spec,
            start_idx=start_idx,
            fixed_synth_params_list=fixed_synth_params_list,
            fixed_note_params_list=fixed_note_params_list,
            flush_batch=_flush,
        )


def make_wds_dataset(
    wds_file: Path | str,
    render_cfg: RenderConfig,
    *,
    fixed_synth_params_list: list[dict[str, float]] | None = None,
    fixed_note_params_list: list[dict[str, int | tuple[float, float]]] | None = None,
) -> None:
    """Render ``render_cfg.samples_per_shard`` samples to a webdataset tar at ``wds_file``.

    Not resumable: ``start_idx`` is pinned to 0 and the file is opened by
    ``wds.TarWriter`` in write mode, so re-running overwrites. Audio is cast to
    ``float16`` to match the h5 path's storage precision; consumers can upcast
    on read if higher precision is needed. The shard's ``metadata.json`` member
    is built from the same ``ShardMetadata`` instance ``make_hdf5_dataset``
    uses for its ``audio.attrs``, so both formats expose identical metadata.

    :param wds_file: Destination tar path passed to ``webdataset.TarWriter``.
    :param render_cfg: Per-shard renderer config from the dataset spec.
    :param fixed_synth_params_list: Optional pre-set synth params, one dict per
        shard row. Must have length ``samples_per_shard``; ``list[0]`` lands at
        row 0 (the wds path is non-resumable, ``start_idx = 0``).
    :param fixed_note_params_list: Optional pre-set note params; same full-shard
        contract as ``fixed_synth_params_list``.
    """
    param_spec = param_specs[render_cfg.param_spec_name]
    meta = _shard_metadata_from_render(render_cfg)
    start_idx = 0
    _validate_fixed_params_lengths(
        num_samples=render_cfg.samples_per_shard,
        fixed_synth_params_list=fixed_synth_params_list,
        fixed_note_params_list=fixed_note_params_list,
    )
    with cast(
        _WdsTarSink,
        wds.TarWriter(str(wds_file)),  # pyright: ignore[reportAttributeAccessIssue]
    ) as sink:

        def _flush(batch: list[VSTDataSample], batch_start: int) -> None:
            save_wds_samples(batch, sink, batch_start)

        _render_in_batches(
            render_cfg=render_cfg,
            param_spec=param_spec,
            start_idx=start_idx,
            fixed_synth_params_list=fixed_synth_params_list,
            fixed_note_params_list=fixed_note_params_list,
            flush_batch=_flush,
        )

        sink.write({"__key__": "metadata", "json": meta.model_dump()})
