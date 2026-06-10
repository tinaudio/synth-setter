"""Spec-driven generate_dataset runner.

Two console-script surfaces:

- ``synth-setter-generate-dataset`` → :func:`main` — operator entry; runs the
  spec in-process or dispatches it to SkyPilot based on
  ``cfg.skypilot_launch.compute_template``.
- ``synth-setter-generate-dataset-from-hydra`` → :func:`from_hydra` — worker
  entry; pure ``@hydra.main`` re-compose so launcher/worker share argv.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import hydra
import wandb
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.loggers import Logger
from lightning.pytorch.loggers.wandb import WandbLogger
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

from synth_setter.cli.finalize_dataset import finalize_from_spec
from synth_setter.data.vst.core import extract_renderer_version
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.constants import (
    INPUT_SPEC_FILENAME,
    STATS_NPZ_FILENAME,
    WORKER_SPEC_URI_ENV,
)
from synth_setter.pipeline.partitioning import (
    available_cpus,
    get_my_shards,
    read_rank_world_from_env,
)
from synth_setter.pipeline.schemas.skypilot_launch import SkypilotLaunchConfig
from synth_setter.pipeline.schemas.spec import DatasetSpec, RenderConfig, ShardSpec
from synth_setter.pipeline.spec_io import (
    load_spec_from_root,
    upload_spec,
    write_spec_locally,
)
from synth_setter.resources import as_file, vst_headless_wrapper
from synth_setter.utils import extras, log_wandb_provenance, pin_wandb_run_id, register_resolvers
from synth_setter.utils.instantiators import close_loggers, instantiate_loggers
from synth_setter.workspace import operator_workspace

# Side effect only: publish ``PROJECT_ROOT`` so ``${oc.env:PROJECT_ROOT}``
# in ``configs/paths/default.yaml`` resolves under @hydra.main compose.
operator_workspace()

# Defensive parity with train.py/eval.py. The dataset compose path uses no
# ``${mul:}``/``${div:}`` resolvers today, so this is parity-only, not required.
register_resolvers()

# Worker-side checkout path — baked WORKDIR of the dev-snapshot image, not the
# launcher's workspace (which may not exist on the worker filesystem).
_WORKER_REPO_ROOT = "/home/build/synth-setter"

# Smoke-shard-sized; longer eval runs belong on the dispatch path, not inline.
_ORACLE_EVAL_TIMEOUT_SECONDS = 600

# Finalized artifacts the eval datamodule opens; all must sit in dataset_root.
_ORACLE_EVAL_REQUIRED_ARTIFACTS = ("train.h5", "val.h5", "test.h5", STATS_NPZ_FILENAME)


def _check_call_streamed(args: Sequence[str], *, timeout: float | None = None) -> None:
    """Run ``args``, teeing the child's merged stdout+stderr through ``sys.stderr``.

    wandb ``console=wrap`` captures only Python-level writes in this process, so
    child output must be re-written through the parent's ``sys.stderr`` to reach
    the run's server-side Logs tab (#1465); inherited fds would bypass it.
    ``PYTHONUNBUFFERED=1`` keeps Python children line-buffered on the pipe so a
    hang still leaves its last lines visible (the #735 diagnosis property).

    :param args: Child argv, run unquoted with no shell, so callers pre-validate it.
    :param timeout: Wall-clock seconds before the child's process group is
        killed; ``None`` means no limit.
    :raises subprocess.CalledProcessError: Child exited non-zero.
    :raises subprocess.TimeoutExpired: Child outlived ``timeout`` and was killed.
    """
    with subprocess.Popen(  # noqa: S603 — argv built from validated specs by callers
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        # Child leads its own process group so the kill paths reap grandchildren
        # (e.g. the headless-VST wrapper tree) holding the pipe open past the timeout.
        start_new_session=True,
    ) as proc:
        timed_out = threading.Event()

        def _kill_group() -> None:
            if proc.returncode is not None:
                # Already reaped — killpg here could hit a recycled pgid.
                return
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                # Group already gone — nothing left to reap.
                pass

        def _kill_on_timeout() -> None:
            timed_out.set()
            _kill_group()

        timer = threading.Timer(timeout, _kill_on_timeout) if timeout is not None else None
        if timer is not None:
            timer.start()
        try:
            assert proc.stdout is not None  # noqa: S101 — guaranteed by stdout=PIPE
            for line in proc.stdout:
                # Resolved at write time so wandb's wrapped stream sees every line.
                sys.stderr.write(line)
            returncode = proc.wait()
        finally:
            if timer is not None:
                timer.cancel()
            # Reap the group on any abrupt exit (KeyboardInterrupt, write failure)
            # so a pipe-holding tree is never orphaned; ``__exit__`` then waits.
            if proc.poll() is None:
                _kill_group()
    # A timer kill surfaces as a signal exit (negative returncode); a child that
    # beat the timer to a natural exit keeps its truthful exception type.
    if timeout is not None and timed_out.is_set() and returncode < 0:
        raise subprocess.TimeoutExpired(args, timeout)
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, args)


def _run_oracle_eval_subprocess(
    dataset_root: Path,
    run_dir: Path,
    run_id: str,
    *,
    render: RenderConfig,
    num_workers: int,
    predict_file: Path,
    metric_prefix: str = "",
) -> None:
    """Run the fake-oracle eval over one split of ``dataset_root``.

    ``_check_call_streamed`` raises on a non-zero eval exit or wall-clock
    timeout, so either propagates to the caller.

    :param dataset_root: Dir holding the finalized HDF5 splits, their source
        shards, and ``stats.npz``. The splits are virtual datasets that
        reference the shards by basename, so they resolve only when read in
        place beside those shards.
    :param run_dir: Hydra per-run dir for the eval's own outputs (predictions,
        ``metrics/metrics.json``), kept separate from ``dataset_root`` so eval
        artifacts don't mix with the dataset files.
    :param run_id: Canonical ``spec.run_id``; the eval resumes this wandb run
        so its ``audio/*`` metrics land on the generate phase's run.
    :param render: The generation ``RenderConfig``. The eval re-renders
        predictions via ``predict_vst_audio``; every render field it renders with
        (param spec, preset, plugin, sample rate, channels, velocity, signal
        duration) is overridden from this so the re-render matches generation
        exactly rather than falling back to the render group / CLI defaults.
    :param num_workers: Predict DataLoader worker count, forwarded verbatim from
        the generate run's ``datamodule`` config — no platform guard. On
        spawn-start-method platforms (Darwin) the caller must configure ``0``:
        workers pickle the dataset, but ``SurgeXTDataset`` holds an open h5py
        handle that cannot be pickled.
    :param predict_file: HDF5 split file for the datamodule's predict dataloader
        (e.g. ``dataset_root / "train.h5"``).
    :param metric_prefix: Prepended to every audio metric key the eval logs
        (both ``audio/*`` and ``shuffled_audio/*``). All splits resume one wandb
        run, so a bare key is overwritten by the last split; pass ``"<split>/"``
        to namespace it. Empty (the default) leaves keys bare — used for the
        canonical ``test`` split.
    :raises FileNotFoundError: ``dataset_root`` is missing any finalized split
        or ``stats.npz`` — e.g. a resume where ``finalize_from_spec``
        short-circuited on an existing R2 marker without repopulating it.
        Also raised when ``predict_file`` itself does not exist.
    """
    missing = [n for n in _ORACLE_EVAL_REQUIRED_ARTIFACTS if not (dataset_root / n).is_file()]
    if missing:
        raise FileNotFoundError(
            f"inline oracle-eval expects the finalized splits + stats in {dataset_root}, "
            f"but {missing} are absent. finalize_from_spec short-circuits when R2 already "
            f"holds the dataset.complete marker, leaving output_dir unpopulated on a resume; "
            f"rerun with a fresh paths.output_dir."
        )
    if not predict_file.is_file():
        raise FileNotFoundError(
            f"predict_file {predict_file} not found; "
            f"ensure the split HDF5 exists in {dataset_root} before shelling out."
        )
    argv = [
        sys.executable,
        "-m",
        "synth_setter.cli.eval",
        "experiment=surge/fake_oracle",
        f"datamodule.dataset_root={dataset_root}",
        f"hydra.run.dir={run_dir}",
        "ckpt_path=null",
        "logger=wandb",
        # eval.yaml leaves render null; render_vst=true re-renders predicted
        # params. Take the surge_simple group for structure, then override every
        # render field predict_vst_audio renders with to the generation render so
        # the round-trip matches it exactly (not the group / CLI defaults).
        "render=surge_simple",
        f"render.param_spec_name={render.param_spec_name}",
        f"render.preset_path={render.preset_path}",
        f"render.plugin_path={render.plugin_path}",
        f"render.sample_rate={render.sample_rate}",
        f"render.channels={render.channels}",
        f"render.velocity={render.velocity}",
        f"render.signal_duration_seconds={render.signal_duration_seconds}",
        # id already exists in logger/wandb.yaml (id: null) so a plain
        # override suffices; resume is absent there and needs +append.
        f"logger.wandb.id={run_id}",
        "+logger.wandb.resume=must",
        # SurgeXTDataset floors len to samples // batch_size; the 128 default
        # would yield zero batches on the smoke-sized test split (4 samples),
        # so predict_step never runs and no audio/* metric is logged — see #1331.
        "datamodule.batch_size=1",
        # Forwarded from the generate run so the eval honours the same worker
        # count; pass 0 on Darwin where the open-h5py dataset can't be pickled.
        f"datamodule.num_workers={num_workers}",
        # Override the datamodule's default predict_file (test.h5) so the caller
        # can route each invocation to a specific split independently.
        f"datamodule.predict_file={predict_file}",
        "mode=predict",
    ]
    # +append: metric_prefix is absent from eval.yaml's evaluation group. Empty
    # (test split) leaves keys bare so existing sweeps/dashboards keep resolving.
    if metric_prefix:
        argv.append(f"+evaluation.metric_prefix={metric_prefix}")
    logger.info(f"oracle_eval_inline subprocess: {argv}")
    _check_call_streamed(argv, timeout=_ORACLE_EVAL_TIMEOUT_SECONDS)


def _unsupported_cadence_reason(render_cfg: DictConfig) -> str | None:
    """Return the reason if ``gui_toggle_cadence=always_on`` lacks ``plugin_reload_cadence=once``.

    Covers only that one combination — it mirrors
    ``RenderConfig._always_on_requires_plugin_reload_once`` so ``main`` can skip the
    grid cell that validator would raise on. Other ``RenderConfig`` validation errors
    are not pre-empted here. Kept in sync with that validator by hand.

    :param render_cfg: Composed ``render`` group; reads ``gui_toggle_cadence`` and
        ``plugin_reload_cadence``.
    :returns: A reason string when the combination is unsupported, else ``None``.
    """
    if (
        render_cfg.get("gui_toggle_cadence") == "always_on"
        and render_cfg.get("plugin_reload_cadence") != "once"
    ):
        reload_cadence = render_cfg.get("plugin_reload_cadence")
        return (
            f"gui_toggle_cadence=always_on requires plugin_reload_cadence=once, "
            f"got plugin_reload_cadence={reload_cadence!r}"
        )
    return None


def _rclone_copy(src: str, dest: str) -> None:
    """Upload a file to R2 via rclone with checksum verification.

    Connection-level timeouts and retries are rclone's job, not ours:
      --contimeout=30s   bound the TCP connect phase
      --timeout=300s     bound any single HTTP request (PUT, list, etc.)
      --retries=3        retry the whole copy on transient failure
      -vv                emit per-request debug log so a failure leaves
                         actionable evidence in the worker stdout
    A non-zero rclone exit raises CalledProcessError and the run fails — we
    do not silently accept partial uploads behind a Python wall-clock.
    """
    args = [  # noqa: S607 — rclone resolved by the image's PATH
        "rclone",
        "copy",
        "-vv",
        "--checksum",
        "--contimeout=30s",
        "--timeout=300s",
        "--retries=3",
        src,
        dest,
    ]
    _check_call_streamed(args)
    # Distinct sentinel so we can grep CI logs for "rclone returned" and tell
    # at a glance whether the rclone subprocess actually exited (vs. hanging
    # post-upload — see #735). If the upload itself failed, _check_call_streamed
    # already raised before we got here.
    logger.info(f"rclone returned cleanly: {src} -> {dest}")


def build_generate_args(spec: DatasetSpec, shard: ShardSpec, output_dir: Path) -> list[str]:
    """Build CLI args for ``generate_vst_dataset.py`` from a spec and shard.

    The flag set is derived from ``RenderConfig.model_fields`` so every renderer
    config field surfaces as a ``--<field>`` option automatically; adding a
    field on the model auto-extends the CLI invocation. The writer is dispatched
    on ``shard.filename``'s suffix inside the subprocess via
    ``OutputFormat.from_extension``. When ``spec.copy_dataset_root_uri`` is set,
    it is forwarded as ``--copy_dataset_root_uri`` so the subprocess re-renders
    the same-named source shard's params instead of sampling fresh ones.
    """
    output_path = output_dir / shard.filename
    args = [
        sys.executable,
        "src/synth_setter/data/vst/generate_vst_dataset.py",
        str(output_path),
    ]
    for key, value in spec.render.model_dump().items():
        args.extend([f"--{key}", str(value)])
    if spec.copy_dataset_root_uri is not None:
        args.extend(["--copy_dataset_root_uri", spec.copy_dataset_root_uri])

    return args


def _validate_copy_source(spec: DatasetSpec) -> None:
    """Preflight a dataset-copy run against the source's persisted spec.

    No-op unless ``spec.copy_dataset_root_uri`` is set. Otherwise loads the
    source's ``input_spec.json`` from under the root URI (the spec sits beside
    the shards at the dataset prefix root) and delegates to
    :meth:`DatasetSpec.validate_copy_source`, so a source that disagrees on any
    copy-relevant value fails once at launch rather than per-shard mid-render.
    The root URI may be a bare path, ``file://`` URI, or ``r2://`` URI.

    A genuinely absent spec (``FileNotFoundError``) and an object-store access
    failure (``CalledProcessError`` from the ``r2://`` rclone fetch — auth,
    network, config) get distinct messages: conflating them would point the
    operator at "sync the spec" when the real fault is credentials or
    connectivity.

    :param spec: The target dataset spec about to be rendered.
    :raises ValueError: the copy root URI holds no ``input_spec.json``; the
        source spec could not be fetched from the object store; the source spec
        JSON is malformed or stale; or the source spec mismatches ``spec`` on a
        copy-relevant value.
    """
    if spec.copy_dataset_root_uri is None:
        return
    try:
        source = load_spec_from_root(spec.copy_dataset_root_uri)
    except FileNotFoundError as exc:
        raise ValueError(
            f"dataset-copy source has no {INPUT_SPEC_FILENAME} under "
            f"{spec.copy_dataset_root_uri!r}; sync the source dataset's spec alongside its "
            "shards (it lives beside the shards at the dataset prefix root) so the copy "
            "can be validated."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise ValueError(
            f"dataset-copy source spec under {spec.copy_dataset_root_uri!r} could not be "
            f"fetched: rclone command {exc.cmd!r} exited {exc.returncode}. This is an "
            "object-store access failure (auth/network/config), not a missing spec — check "
            "R2 credentials and connectivity."
        ) from exc
    except ValidationError as exc:
        raise ValueError(
            f"dataset-copy source spec under {spec.copy_dataset_root_uri!r} is malformed or "
            "stale; re-materialize it from the source dataset's current spec."
        ) from exc
    spec.validate_copy_source(source)
    logger.info(f"dataset-copy source OK: {spec.copy_dataset_root_uri} matches the target spec")


def generate(spec: DatasetSpec, work_dir: Path, loggers: list[Logger]) -> None:  # noqa: DOC503
    """Render+upload each owned shard; writes shards under ``work_dir``.

    Subprocesses fail-fast: later shards are not attempted on subprocess error.
    Before each render, R2 is probed for the shard's destination object: if it
    already exists with non-zero size, the shard is skipped (resumability MVP
    — see #750). The probe uses ``check=True``, so a non-zero rclone exit
    (auth, network) propagates as a hard failure rather than degrading silently
    into a re-render.

    The launcher builds the spec interpreter-only (no pedalboard / X11) trusting
    ``configs/render/<spec>.yaml``; the worker — which has pedalboard — verifies
    the plugin and pinned ``renderer_version`` agree.

    The spec is pushed to every logger as hyperparameters and, when a
    ``WandbLogger`` is present in ``loggers``, uploaded as a
    ``<task_name>-input-spec`` artifact before any render begins. The wandb
    run is bracketed by the function: ``finalize(status)`` and ``wandb.finish()``
    fire in ``finally`` so the run is closed (and offline binaries flushed) on
    both success and failure.

    :param spec: Validated dataset spec; rank/world env partitions ``spec.shards``
        across worker pods, and ``spec.render.renderer_version`` is cross-checked
        against the loaded plugin.
    :param work_dir: Hydra per-run output dir supplied by the caller; created
        if missing. Shards are written here before the rclone upload.
    :param loggers: Lightning loggers instantiated by ``instantiate_loggers`` —
        typically a single ``WandbLogger`` whose ``id`` was pinned to
        ``spec.run_id`` by the caller. May be empty (logger group disabled).
    :raises RuntimeError: If the worker's plugin version disagrees with
        ``spec.render.renderer_version``.
    """
    status = "success"
    try:
        # Inside the try so a helper failure (e.g. tempfile creation in
        # ``_log_spec_artifact``) still triggers ``finalize("failed")`` +
        # ``wandb.finish()`` in the ``finally`` — otherwise the wandb run
        # leaks un-closed on the helper's exception path.
        _log_hyperparams(loggers, spec)
        # Provenance mutates the process-global ``wandb.run``; only stamp it when
        # a ``WandbLogger`` here owns the run, mirroring ``close_loggers`` — else
        # an empty-logger run would stamp a foreign run started elsewhere.
        if any(isinstance(lg, WandbLogger) for lg in loggers):
            log_wandb_provenance()
        _log_spec_artifact(loggers, spec)
        # Fail a misconfigured dataset-copy at launch, before the first render.
        _validate_copy_source(spec)
        render = spec.render
        actual_renderer_version = extract_renderer_version(Path(render.plugin_path))
        if actual_renderer_version != render.renderer_version:
            raise RuntimeError(
                f"Renderer version mismatch: spec pins {render.renderer_version!r} but "
                f"plugin at {render.plugin_path} reports {actual_renderer_version!r}. "
                "Rebuild the image against the matching SURGE_GIT_REF, or bump "
                "renderer_version in the dataset config that produced this spec."
            )
        logger.info(
            f"renderer_version OK: plugin at {render.plugin_path} == {render.renderer_version}"
        )

        rank, world = read_rank_world_from_env()
        my_range = get_my_shards(spec.num_shards, rank=rank, world=world)
        logger.info(
            f"shard partition: rank={rank}/{world} owns shard_ids "
            f"[{my_range.start}, {my_range.stop}) "
            f"({len(my_range)} of {spec.num_shards} shards)"
        )

        r2_dest_prefix = spec.r2.rclone_prefix()
        work_dir.mkdir(parents=True, exist_ok=True)

        # ``start`` brackets only the dispatch call so the rate still includes the
        # in-loop R2 skip probes (observable cost of the resumability MVP, #750).
        start = time.perf_counter()
        if spec.render.parallel and len(my_range) > 0:
            rendered, skipped = _dispatch_shards_parallel(
                spec, my_range, work_dir, r2_dest_prefix, loggers
            )
        else:
            rendered, skipped = _dispatch_shards_serial(
                spec, my_range, work_dir, r2_dest_prefix, loggers
            )
        elapsed_s = time.perf_counter() - start
        samples = rendered * spec.render.samples_per_shard
        rate = samples / elapsed_s if elapsed_s > 0 else 0.0
        logger.info(
            f"shard summary: rendered={rendered} skipped={skipped} of {len(my_range)} assigned"
        )
        logger.info(
            f"generation speed: {samples} samples in {elapsed_s:.3f}s "
            f"= {rate:.3f} samples/s (wallclock includes R2 skip probes)"
        )
        _log_summary(loggers, rendered, skipped, len(my_range), elapsed_s, samples, rate)
    except BaseException:
        status = "failed"
        raise
    finally:
        close_loggers(loggers, status)


def _log_hyperparams(loggers: list[Logger], spec: DatasetSpec) -> None:
    """Push the spec onto each logger as hyperparameters.

    :param loggers: Lightning loggers — empty list is a no-op.
    :param spec: Serialized via ``model_dump(mode="json")`` so nested
        ``R2Location`` / ``ShardSpec`` entries round-trip through wandb's
        config flattener.
    """
    payload = spec.model_dump(mode="json")
    for lg in loggers:
        try:
            lg.log_hyperparams(payload)
        except Exception as exc:  # noqa: BLE001 — third-party logger failures must not abort the run
            logger.warning(f"log_hyperparams failed on {type(lg).__name__}: {exc}")


def _log_spec_artifact(loggers: list[Logger], spec: DatasetSpec) -> None:
    """Upload the spec as a wandb artifact when a ``WandbLogger`` is present.

    Writes the spec JSON to a tempfile; the wandb client copies the payload
    into its own store before this function returns.

    :param loggers: Lightning loggers; non-``WandbLogger`` entries are skipped.
    :param spec: ``spec.task_name`` names the artifact (``<task_name>-input-spec``).
    """
    wandb_lgs = [lg for lg in loggers if isinstance(lg, WandbLogger)]
    if not wandb_lgs:
        return
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        f.write(spec.model_dump_json(indent=2))
        tmp = Path(f.name)
    try:
        for lg in wandb_lgs:
            try:
                art = wandb.Artifact(name=f"{spec.task_name}-input-spec", type="dataset-spec")
                art.add_file(str(tmp))
                lg.experiment.log_artifact(art)
            except Exception as exc:  # noqa: BLE001 — wandb artifact failure must not abort the run
                logger.warning(f"log_spec_artifact failed on {type(lg).__name__}: {exc}")
    finally:
        # Best-effort tempfile cleanup; wandb has already copied the payload
        # into its own store by this point. A leaked tempfile under the OS
        # temp dir is preferable to letting a deletion failure (Windows lock,
        # antivirus scan, prior removal) abort dataset generation.
        try:
            tmp.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning(f"tempfile cleanup failed for {tmp}: {exc}")


def _log_shard_metrics(
    loggers: list[Logger], shard_id: int, byte_size: int, render_seconds: float
) -> None:
    """Emit per-shard byte size + render duration as a wandb history row.

    :param loggers: Lightning loggers — empty list is a no-op.
    :param shard_id: Passed as ``step`` so wandb's x-axis aligns with shard order.
    :param byte_size: Local shard file size in bytes; stable because shards are
        retained at ``work_dir``.
    :param render_seconds: Wall-clock seconds from subprocess invoke through
        upload-end; ``0.0`` on the R2-skip branch.
    """
    payload = {"shard/bytes": byte_size, "shard/render_seconds": render_seconds}
    for lg in loggers:
        try:
            lg.log_metrics(payload, step=shard_id)
        except Exception as exc:  # noqa: BLE001 — third-party logger failures must not abort the run
            logger.warning(f"log_metrics(shard) failed on {type(lg).__name__}: {exc}")


def _log_summary(
    loggers: list[Logger],
    rendered: int,
    skipped: int,
    total: int,
    elapsed_s: float,
    samples: int,
    rate: float,
) -> None:
    """Emit the run-level shard counters + e2e generation triple.

    Mirrors the loguru summary line at ``generate()`` so wandb history and
    stdout agree on the rate that the resumability MVP (#750, #1304) reports.

    :param loggers: Lightning loggers — empty list is a no-op.
    :param rendered: Shards this rank actually rendered.
    :param skipped: Shards short-circuited by the R2-skip probe.
    :param total: ``len(my_range)`` — owned shard count for this rank.
    :param elapsed_s: Wall-clock seconds bracketing the dispatcher (includes
        the R2 skip probes by design).
    :param samples: ``rendered * spec.render.samples_per_shard``.
    :param rate: ``samples / elapsed_s`` (``0.0`` when ``elapsed_s == 0``).
    """
    payload = {
        "shards/rendered": rendered,
        "shards/skipped": skipped,
        "shards/total": total,
        "generation/elapsed_seconds": elapsed_s,
        "generation/samples": samples,
        "generation/samples_per_second": rate,
    }
    for lg in loggers:
        try:
            lg.log_metrics(payload)
        except Exception as exc:  # noqa: BLE001 — third-party logger failures must not abort the run
            logger.warning(f"log_metrics(summary) failed on {type(lg).__name__}: {exc}")


def _dispatch_shards_serial(
    spec: DatasetSpec,
    my_range: range,
    work_dir: Path,
    r2_dest_prefix: str,
    loggers: list[Logger],
) -> tuple[int, int]:
    """Render+upload owned shards in order; fail-fast on first error.

    :param spec: Validated dataset spec.
    :param my_range: Contiguous range of shard IDs owned by this rank.
    :param work_dir: Hydra per-run output dir; shards land here before upload.
    :param r2_dest_prefix: ``spec.r2.rclone_prefix()``.
    :param loggers: Forwarded to ``_render_one_owned_shard`` so per-shard
        byte size + render duration land in wandb history.
    :returns: ``(rendered, skipped)`` summary counts over ``my_range``.
    """
    rendered = 0
    skipped = 0
    for shard_id in my_range:
        r, s = _render_one_owned_shard(spec, shard_id, work_dir, r2_dest_prefix, loggers)
        rendered += int(r)
        skipped += int(s)
    return rendered, skipped


def _dispatch_shards_parallel(
    spec: DatasetSpec,
    my_range: range,
    work_dir: Path,
    r2_dest_prefix: str,
    loggers: list[Logger],
) -> tuple[int, int]:
    """Render+upload owned shards concurrently via a ``ThreadPoolExecutor``.

    Pool size is ``min(max(1, available_cpus() // 2), len(my_range))``. The
    heuristic halves the CPU count to leave headroom for each renderer
    subprocess's own intra-process threading (pedalboard / librosa / BLAS).

    Producer/consumer dispatch: at most ``workers`` shards are submitted at
    any time, and a new shard is submitted only after every completion in
    the current batch has been observed without error. The first worker
    exception is re-raised by ``fut.result()`` and the loop exits without
    submitting more work; ``shutdown(cancel_futures=True)`` then aborts
    any not-yet-started shards while in-flight peers run to completion
    (Python has no thread interruption) — matches the
    no-interruption-mid-render reality of the serial path. Submitting in
    waves rather than all shards up-front guarantees no shard beyond the
    in-flight set can start once a failure surfaces, even if a worker
    thread is freed before the main thread observes the exception.

    :param spec: Validated dataset spec.
    :param my_range: Non-empty contiguous range of shard IDs owned by this rank.
    :param work_dir: Hydra per-run output dir; every owned shard lands here.
    :param r2_dest_prefix: ``spec.r2.rclone_prefix()``.
    :param loggers: Forwarded to ``_render_one_owned_shard`` so per-shard
        byte size + render duration land in wandb history.
    :returns: ``(rendered, skipped)`` summary counts over ``my_range``.
    """
    workers = min(max(1, available_cpus() // 2), len(my_range))
    logger.info(f"parallel dispatch: workers={workers} shards={len(my_range)}")
    rendered = 0
    skipped = 0
    pending = iter(my_range)
    in_flight: set[Future[tuple[bool, bool]]] = set()
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        for _ in range(workers):
            sid = next(pending, None)
            if sid is None:
                break
            in_flight.add(
                pool.submit(_render_one_owned_shard, spec, sid, work_dir, r2_dest_prefix, loggers)
            )
        while in_flight:
            done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            for fut in done:
                r, s = fut.result()
                rendered += int(r)
                skipped += int(s)
            for _ in range(len(done)):
                sid = next(pending, None)
                if sid is None:
                    break
                in_flight.add(
                    pool.submit(
                        _render_one_owned_shard, spec, sid, work_dir, r2_dest_prefix, loggers
                    )
                )
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return rendered, skipped


def _render_one_owned_shard(
    spec: DatasetSpec,
    shard_id: int,
    work_dir: Path,
    r2_dest_prefix: str,
    loggers: list[Logger],
) -> tuple[bool, bool]:
    """Render+upload one owned shard, or skip if R2 already has it.

    Encapsulates the R2 skip-probe + ``_render_and_upload_shard`` invocation
    so the serial and parallel dispatch arms share one callable. Emits one
    ``shard/bytes`` + ``shard/render_seconds`` history row per call —
    ``render_seconds == 0.0`` on the skip branch, wall-clock from subprocess
    invoke through upload-end on the render branch.

    :param spec: Validated dataset spec; ``spec.shards[shard_id]`` is fetched.
    :param shard_id: Index into ``spec.shards``; also the ``step`` for the row.
    :param work_dir: Hydra per-run output dir; shards land here before upload.
    :param r2_dest_prefix: ``spec.r2.rclone_prefix()``.
    :param loggers: Forwarded to ``_log_shard_metrics``.
    :returns: ``(rendered, skipped)`` — exactly one is ``True``.
    """
    shard = spec.shards[shard_id]
    existing_size = r2_io.object_size(spec.r2.shard_uri(shard))
    if existing_size is not None and existing_size > 0:
        logger.info(
            f"skipping shard {shard.shard_id} — already in R2 "
            f"({existing_size} bytes): {shard.filename}"
        )
        _log_shard_metrics(loggers, shard_id, byte_size=existing_size, render_seconds=0.0)
        return False, True
    t0 = time.monotonic()
    byte_size = _render_and_upload_shard(spec, shard, work_dir, r2_dest_prefix)
    _log_shard_metrics(
        loggers, shard_id, byte_size=byte_size, render_seconds=time.monotonic() - t0
    )
    return True, False


def _render_and_upload_shard(
    spec: DatasetSpec,
    shard: ShardSpec,
    work_dir: Path,
    r2_dest_prefix: str,
) -> int:
    """Render a single shard and upload it to R2; shards are retained at ``work_dir``.

    Rendered shards stay on disk under ``work_dir`` for post-mortem inspection
    (``finalize_dataset`` re-downloads from R2 — launcher and worker pods do not
    share a filesystem). Peak local disk per rank scales with the number of owned
    shards. The renderer subprocess is wrapped in a retry loop bounded by
    ``spec.render.max_retries`` (default 0 = strict fail-fast); rclone is outside
    the loop because it already retries via ``--retries=3``.

    :returns: Local shard file size in bytes; stable for the caller because
        shards are retained at ``work_dir``.
    :raises subprocess.CalledProcessError: Renderer (or rclone) subprocess exited non-zero after
        exhausting the retry budget.
    :raises RuntimeError: Renderer exited 0 without writing the expected shard file.
    """
    # Zipped wheels extract the wrapper to a temp file that only lives while
    # ``as_file()`` is open; ``ExitStack`` keeps it on disk across the retry
    # loop, and skips materialization on non-Linux.
    with ExitStack() as stack:
        if sys.platform == "linux":
            wrapper_path = stack.enter_context(as_file(vst_headless_wrapper()))
            args = [str(wrapper_path)]
        else:
            args = []
        args += build_generate_args(spec, shard, work_dir)
        logger.info(f"rendering shard {shard.shard_id} -> {shard.filename}")
        max_attempts = spec.render.max_retries + 1
        for attempt in range(max_attempts):
            try:
                _check_call_streamed(args)
                break
            except subprocess.CalledProcessError:
                if attempt + 1 == max_attempts:
                    raise
                logger.warning(
                    f"shard {shard.shard_id} render failed on attempt "
                    f"{attempt + 1}/{max_attempts}; retrying"
                )
    shard_path = work_dir / shard.filename
    # Surface a generator that exited 0 without writing output here, not as a
    # downstream rclone "source not found".
    if not shard_path.is_file():
        raise RuntimeError(
            f"generate_vst_dataset.py exited 0 but did not write expected shard file: {shard_path}"
        )
    byte_size = shard_path.stat().st_size
    logger.info(f"shard rendered: {shard_path} ({byte_size} bytes)")
    _rclone_copy(str(shard_path), r2_dest_prefix)
    logger.info(f"shard uploaded: {shard.filename} -> {r2_dest_prefix}")
    return byte_size


def spec_from_cfg(cfg: DictConfig) -> DatasetSpec:
    """Build a DatasetSpec from a Hydra-composed cfg."""
    return DatasetSpec.from_hydra_cfg(cfg)


def _sky_cfg_from_dataset_cfg(cfg: DictConfig) -> SkypilotLaunchConfig:
    """Validate the ``skypilot_launch`` sub-tree of a composed dataset cfg.

    Rejects operator-supplied ``cmd`` — it is launcher-internal (built from
    argv by :func:`main`), so a CLI override would shadow that construction
    and bypass the trust boundary.

    :param cfg: Composed dataset cfg.
    :return: Validated config with ``cmd`` unset; the entrypoint populates it
        later via ``model_copy(update=...)``.
    :raises TypeError: ``cfg.skypilot_launch`` did not resolve to a mapping.
    :raises ValueError: operator supplied ``skypilot_launch.cmd`` from Hydra.
    """
    raw: object = OmegaConf.to_container(cfg.skypilot_launch, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError(f"cfg.skypilot_launch must compose to a mapping; got {type(raw).__name__}")
    sky_kwargs: dict[str, Any] = {k: v for k, v in raw.items() if isinstance(k, str)}
    if sky_kwargs.get("cmd") is not None:
        raise ValueError(
            "skypilot_launch.cmd is launcher-internal and cannot be set from Hydra; "
            "the worker-side bash one-liner is built from argv by main()."
        )
    if sky_kwargs.get("extra_envs"):
        raise ValueError(
            "skypilot_launch.extra_envs is launcher-internal and cannot be set from Hydra; "
            "main() injects dataset-specific worker envs (e.g. WORKER_SPEC_URI)."
        )
    return SkypilotLaunchConfig(**sky_kwargs)


def _smoke_job_name(spec: DatasetSpec) -> str:
    """Build the dataset-flavored SkyPilot job-name stem from ``spec.task_name``.

    The first 8 chars of ``task_name`` are interpolated into the
    ``synth-setter-smoke-<…>`` stem. Validated against the launcher's
    k8s-label-subset grammar so a malformed ``task_name`` raises here with a
    domain-specific message, not later from inside ``dispatch_via_skypilot``
    where the spec is no longer in scope.

    :param spec: Validated dataset spec.
    :return: Job-name stem matching the launcher's ``_JOB_NAME_RE`` grammar.
    :raises ValueError: derived stem violates ``_JOB_NAME_RE``.
    """
    from synth_setter.pipeline.skypilot_launch import _JOB_NAME_RE

    stem = f"synth-setter-smoke-{spec.task_name[:8]}"
    if not _JOB_NAME_RE.fullmatch(stem):
        raise ValueError(
            f"derived job-name stem {stem!r} contains characters outside "
            f"{_JOB_NAME_RE.pattern}; fix spec.task_name or pin "
            "skypilot_launch.job_name explicitly."
        )
    return stem


def _build_worker_cmd(overrides: list[str], spec: DatasetSpec) -> str:
    """Reconstruct the worker-side bash command that re-enters Hydra via from_hydra.

    Each override is shell-quoted individually so spaces/metachars survive bash
    interpretation. ``sync_worker_checkout.sh`` runs between cd and exec for
    the PR-CI bake-lag bypass (see #735 / #841).

    ``spec.created_at`` is pinned as a Hydra override so the worker's
    re-compose lands on the same ``r2.prefix`` as the launcher (the
    ``default_factory`` that produces it would otherwise fire twice and the
    derived ``run_id`` / ``r2.prefix`` would diverge — see
    ``_default_run_id`` / ``_fill_default_r2_prefix`` in
    ``synth_setter.pipeline.schemas.spec``).

    :param overrides: Operator's Hydra overrides (``HydraConfig.get().overrides.task``).
    :param spec: Launcher's ``DatasetSpec``; runtime fields are pinned into
        the worker overrides for compose determinism.
    :return: Bash one-liner suitable for use as a ``sky.Task`` ``run:`` block.
    """
    pinned_overrides = [f"+created_at={spec.created_at.isoformat()}"]
    all_overrides = list(overrides) + pinned_overrides
    parts = [
        f"cd {shlex.quote(_WORKER_REPO_ROOT)}",
        "bash scripts/sync_worker_checkout.sh",
        "exec synth-setter-generate-dataset-from-hydra "
        + " ".join(shlex.quote(o) for o in all_overrides),
    ]
    return " && ".join(parts)


def _loggers_pinned_to_spec(cfg: DictConfig, spec: DatasetSpec) -> list[Logger]:
    """Pin the wandb run id to ``spec.run_id`` and instantiate ``cfg.logger``.

    :param cfg: Composed dataset cfg; ``cfg.logger.wandb.id`` is overridden
        in place so the wandb run identity stays in lockstep with the spec's
        ``make_dataset_wandb_run_id`` derivation.
    :param spec: ``spec.run_id`` is the canonical ``wandb.run.id`` for this
        dataset.
    :returns: Loggers list — empty when ``cfg.logger`` is omitted/null.
    """
    # spec.run_id (not a fresh stamp) keeps the wandb run in lockstep with the R2 prefix.
    pin_wandb_run_id(cfg, spec.run_id, "data-generation")
    return instantiate_loggers(cfg.get("logger"))


@hydra.main(version_base="1.3", config_path="pkg://synth_setter.configs", config_name="dataset")
def from_hydra(cfg: DictConfig) -> None:
    """Worker-side @hydra.main entry: build the spec and render it in-process.

    :param cfg: Composed Hydra dataset cfg supplied by ``@hydra.main`` from
        the worker's argv overrides.
    """
    extras(cfg)
    spec = spec_from_cfg(cfg)
    loggers = _loggers_pinned_to_spec(cfg, spec)
    generate(spec, Path(cfg.paths.output_dir), loggers)


@hydra.main(version_base="1.3", config_path="pkg://synth_setter.configs", config_name="dataset")
def main(cfg: DictConfig) -> None:
    """Operator CLI: run the composed dataset spec locally or dispatch to SkyPilot.

    Worker-side overrides are replayed verbatim under ``_build_worker_cmd`` so
    the launcher/worker composition matches byte-for-byte; ``HydraConfig`` is
    the authoritative source for the operator's task overrides.

    The one schema-invalid cadence cell ``gui_toggle_cadence=always_on`` +
    ``plugin_reload_cadence!=once`` (see :func:`_unsupported_cadence_reason`) is a
    logged no-op rather than a raise, so a wandb grid sweep can enumerate it without
    failing the trial; other ``RenderConfig`` validation errors still raise.

    :param cfg: Hydra-composed dataset cfg.
    :raises ValueError: ``oracle_eval_inline=true`` without
        ``finalize_inline=true``, with ``output_format!=hdf5``, or with
        a zero-size train / val / test split (the eval datamodule opens
        all three split files unconditionally).
    """
    extras(cfg)
    render_cfg = cfg.get("render")
    skip_reason = None if render_cfg is None else _unsupported_cadence_reason(render_cfg)
    if skip_reason is not None:
        logger.warning(
            f"skipping run: unsupported render cadence combination ({skip_reason}). "
            "No dataset generated — a no-op so a wandb grid sweep can enumerate this "
            "cell without failing the trial."
        )
        return

    overrides = list(HydraConfig.get().overrides.task)
    spec = spec_from_cfg(cfg)
    sky_cfg = _sky_cfg_from_dataset_cfg(cfg)

    if sky_cfg.compute_template is None and cfg.oracle_eval_inline:
        if not cfg.finalize_inline:
            raise ValueError(
                "oracle_eval_inline=true requires finalize_inline=true; "
                "the inline oracle eval reads the {train,val,test}.h5 files "
                "finalize uploads to R2."
            )
        if cfg.output_format != "hdf5":
            raise ValueError(
                "oracle_eval_inline=true only supports output_format=hdf5; "
                f"got {cfg.output_format!r}."
            )
        if any(size == 0 for size in spec.train_val_test_sizes):
            raise ValueError(
                "oracle_eval_inline=true requires all of "
                f"train_val_test_sizes > 0; got {tuple(spec.train_val_test_sizes)}. "
                "SurgeDataModule opens train.h5 / val.h5 / test.h5 unconditionally."
            )

    spec_path = write_spec_locally(spec, Path(cfg.paths.output_dir))
    logger.info(f"wrote local spec to {spec_path}")

    # Load creds once and upload the canonical spec here so workers boot pointing at it.
    env_file = Path(sky_cfg.env_file).expanduser() if sky_cfg.env_file else None
    r2_io.ensure_r2_env_loaded(env_file)
    r2_uri = upload_spec(spec)
    logger.info(f"spec uploaded -> {r2_uri}")

    # ``input_spec_uri()`` includes the run prefix so workers read the same object just uploaded.
    spec_uri = spec.r2.input_spec_uri()

    if sky_cfg.compute_template is None:
        loggers = _loggers_pinned_to_spec(cfg, spec)
        # finalize runs outside the wandb-tracked region — see #1289.
        generate(spec, Path(cfg.paths.output_dir), loggers)
        if cfg.finalize_inline:
            finalize_from_spec(spec, Path(cfg.paths.output_dir))
        if cfg.oracle_eval_inline:
            # generate + finalize already wrote the shards, VDS splits, and
            # stats.npz into output_dir; the splits reference the shards by
            # basename, so read them in place — no R2 round-trip.
            output_dir = Path(cfg.paths.output_dir)
            for split in ("train", "val", "test"):
                # test stays bare; train/val are namespaced so the shared run
                # keeps one summary key per split (see _run_oracle_eval_subprocess).
                metric_prefix = "" if split == "test" else f"{split}/"
                _run_oracle_eval_subprocess(
                    output_dir,
                    output_dir / "oracle_eval" / split / spec.run_id,
                    spec.run_id,
                    render=spec.render,
                    num_workers=cfg.datamodule.num_workers,
                    predict_file=output_dir / f"{split}.h5",
                    metric_prefix=metric_prefix,
                )
        return

    if cfg.finalize_inline or cfg.oracle_eval_inline:
        logger.info(
            f"finalize_inline={cfg.finalize_inline}, "
            f"oracle_eval_inline={cfg.oracle_eval_inline} ignored: "
            f"skypilot_launch.compute_template={sky_cfg.compute_template!r} "
            "dispatches to a worker; finalize runs out-of-band via the "
            "finalize-dataset workflow."
        )

    # Deferred import — SkyPilot pulls heavy provider SDKs on import.
    from synth_setter.pipeline.skypilot_launch import dispatch_via_skypilot

    sky_cfg = sky_cfg.model_copy(
        update={
            "cmd": _build_worker_cmd(overrides, spec),
            "job_name": sky_cfg.job_name or _smoke_job_name(spec),
            "extra_envs": {WORKER_SPEC_URI_ENV: spec_uri},
        }
    )
    dispatch_via_skypilot(sky_cfg)


if __name__ == "__main__":
    main()
