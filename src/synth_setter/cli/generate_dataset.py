"""Spec-driven generate_dataset runner.

Two console-script surfaces:

- ``synth-setter-generate-dataset`` → :func:`main` — operator entry; runs the
  spec in-process or dispatches it to SkyPilot based on
  ``cfg.skypilot_launch.compute_template``.
- ``synth-setter-generate-dataset-from-hydra`` → :func:`from_hydra` — worker
  entry; pure ``@hydra.main`` re-compose so launcher/worker share argv.

``synth-setter-generate-dataset-from-spec-uri`` (see
:mod:`synth_setter.cli.generate_dataset_from_spec_uri`) is the spec-first
counterpart: it renders an already-materialized ``input_spec.json`` by URI
instead of composing one.
"""

from __future__ import annotations

import platform
import re
import shlex
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import ExitStack
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

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
from synth_setter.data.vst.dawdreamer_runtime import ensure_dawdreamer_runtime
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.ci.validate_shard import validate_shard
from synth_setter.pipeline.constants import (
    STATS_NPZ_FILENAME,
    WORKER_SPEC_URI_ENV,
)
from synth_setter.pipeline.data.lance_staging import (
    shard_has_complete_attempt,
    stage_lance_shard_attempt,
    write_rendering_marker,
)
from synth_setter.pipeline.partitioning import (
    available_cpus,
    get_my_shards,
    read_rank_world_from_env,
)
from synth_setter.pipeline.schemas.render_metrics import (
    RenderRejectionMetrics,
    render_metrics_path,
)
from synth_setter.pipeline.schemas.skypilot_launch import SkypilotLaunchConfig
from synth_setter.pipeline.schemas.spec import DatasetSpec, RenderConfig, ShardSpec, Split
from synth_setter.pipeline.shard_claims import ShardClaims
from synth_setter.pipeline.spec_io import (
    upload_spec,
    write_spec_locally,
)

# Imported under the module-local name tests already patch as the render /
# rclone / eval subprocess seam (see tests/helpers/render_subprocess.py).
from synth_setter.pipeline.subprocess_stream import check_call_streamed as _check_call_streamed
from synth_setter.pipeline.subprocess_stream import scaled_timeout
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
_WORKER_VENV = "/venv/main"

# The inline eval (predict + re-render + metrics over a whole split) scales its
# timeout with that split's sample count; per-sample covers all three. See scaled_timeout.
_ORACLE_EVAL_TIMEOUT_OVERHEAD_SECONDS = 600.0
_ORACLE_EVAL_TIMEOUT_PER_SAMPLE_SECONDS = 120.0

# Finalized artifacts the eval datamodule opens; all must sit in dataset_root.
_ORACLE_EVAL_REQUIRED_ARTIFACTS = ("train.lance", "val.lance", "test.lance", STATS_NPZ_FILENAME)


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

    :param dataset_root: Dir holding the finalized Lance split datasets and
        ``stats.npz``.
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
        workers pickle the dataset, but the Lance shard handle is not fork-safe.
    :param predict_file: Lance split dataset directory for the datamodule's
        predict dataloader (e.g. ``dataset_root / "train.lance"``).
    :param metric_prefix: Prepended to every audio metric key the eval logs
        (both ``audio/*`` and ``shuffled_audio/*``). All splits resume one wandb
        run, so a bare key is overwritten by the last split; pass ``"<split>/"``
        to namespace it. Empty (the default) leaves keys bare — used for the
        canonical ``test`` split.
    :raises FileNotFoundError: ``dataset_root`` is missing any finalized split
        or ``stats.npz`` — e.g. a resume where ``finalize_from_spec``
        short-circuited on an existing R2 marker without staging ``stats.npz``
        locally. Also raised when ``predict_file`` itself does not exist.
    """
    missing = [n for n in _ORACLE_EVAL_REQUIRED_ARTIFACTS if not (dataset_root / n).exists()]
    if missing:
        raise FileNotFoundError(
            f"inline oracle-eval expects the finalized splits + stats in {dataset_root}, "
            f"but {missing} are absent. finalize_from_spec short-circuits when R2 already "
            f"holds the dataset.complete marker, so stats.npz is never staged locally on "
            f"a resume; rerun with a fresh paths.output_dir."
        )
    if not predict_file.exists():
        raise FileNotFoundError(
            f"predict_file {predict_file} not found; "
            f"ensure the Lance split exists in {dataset_root} before shelling out."
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
        # ``+`` adds identity keys absent from ``render/vst.yaml``; generic knobs override normally.
        "render=vst",
        f"+render.param_spec_name={render.param_spec_name}",
        f"+render.plugin_state_path={render.plugin_state_path}",
        f"+render.plugin_path={render.plugin_path}",
        f"+render.renderer_version={render.renderer_version}",
        f"render.sample_rate={render.sample_rate}",
        f"render.channels={render.channels}",
        f"render.velocity={render.velocity}",
        f"render.signal_duration_seconds={render.signal_duration_seconds}",
        # id already exists in logger/wandb.yaml (id: null) so a plain
        # override suffices; resume is absent there and needs +append.
        f"logger.wandb.id={run_id}",
        "+logger.wandb.resume=must",
        # VSTDataset floors len to samples // batch_size; the 128 default
        # would yield zero batches on the smoke-sized test split (4 samples),
        # so predict_step never runs and no audio/* metric is logged — see #1331.
        "datamodule.batch_size=1",
        # Forwarded from the generate run so the eval honours the same worker
        # count; pass 0 on Darwin where the Lance shard handle isn't fork-safe.
        f"datamodule.num_workers={num_workers}",
        # Override the datamodule's default predict_file (test.lance) so the caller
        # can route each invocation to a specific split independently.
        f"datamodule.predict_file={predict_file}",
        "mode=predict",
    ]
    # +append: metric_prefix is absent from eval.yaml's evaluation group. Empty
    # (test split) leaves keys bare so existing sweeps/dashboards keep resolving.
    if metric_prefix:
        argv.append(f"+evaluation.metric_prefix={metric_prefix}")
    # Function-local so the heavy ``lance`` import is only paid when this
    # oracle-eval CLI path actually runs.
    import lance

    # Budget scales with the split's sample count (predict + re-render + metrics
    # all run over it); the finalized Lance split exposes it as its row count.
    num_samples = int(lance.dataset(str(predict_file)).count_rows())
    logger.info(f"oracle_eval_inline subprocess: {argv}")
    _check_call_streamed(
        argv,
        timeout=scaled_timeout(
            num_samples,
            overhead_seconds=_ORACLE_EVAL_TIMEOUT_OVERHEAD_SECONDS,
            per_sample_seconds=_ORACLE_EVAL_TIMEOUT_PER_SAMPLE_SECONDS,
        ),
    )


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


def build_generate_args(spec: DatasetSpec, shard: ShardSpec, output_dir: Path) -> list[str]:
    """Build CLI args for ``generate_vst_dataset.py`` from a spec and shard.

    The flag set is derived from ``RenderConfig.model_fields`` so every renderer
    config field surfaces as a ``--<field>`` option automatically; adding a
    field on the model auto-extends the CLI invocation.
    """
    output_path = output_dir / shard.filename
    args = [
        sys.executable,
        "src/synth_setter/data/vst/generate_vst_dataset.py",
        str(output_path),
    ]
    render_args = spec.render_for_shard(shard).model_dump()
    for key, value in render_args.items():
        args.extend([f"--{key}", str(value)])

    return args


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
    :raises RuntimeError: If DawDreamer is unavailable on this worker or the
        plugin version disagrees with ``spec.render.renderer_version``.
    """
    ensure_dawdreamer_runtime(spec.render.renderer_backend)
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

        work_dir.mkdir(parents=True, exist_ok=True)

        # ``start`` brackets only the dispatch call so the rate still includes the
        # in-loop R2 skip probes (observable cost of the resumability MVP, #750).
        start = time.perf_counter()
        rendered, skipped, assigned, rejections = _dispatch_shards(
            spec,
            work_dir=work_dir,
            loggers=loggers,
        )
        elapsed_s = time.perf_counter() - start
        samples = rendered * spec.render.samples_per_shard
        rate = samples / elapsed_s if elapsed_s > 0 else 0.0
        logger.info(
            "shard summary: rendered={rendered} skipped={skipped} {assignment}",
            rendered=rendered,
            skipped=skipped,
            assignment=(
                f"({assigned} shard claims won by this machine)"
                if spec.use_shard_queue
                else f"of {assigned} assigned"
            ),
        )
        logger.info(
            "generation speed: {samples} samples in {elapsed_s:.3f}s "
            "= {rate:.3f} samples/s (wallclock includes R2 skip probes)",
            samples=samples,
            elapsed_s=elapsed_s,
            rate=rate,
        )
        _log_summary(
            loggers,
            rendered=rendered,
            skipped=skipped,
            total=assigned,
            elapsed_s=elapsed_s,
            samples=samples,
            rate=rate,
            rejections=rejections,
        )
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
    loggers: list[Logger],
    *,
    shard_id: int,
    byte_size: int,
    render_seconds: float,
    rejections: RenderRejectionMetrics,
) -> None:
    """Emit per-shard byte size + render duration as a wandb history row.

    :param loggers: Lightning loggers — empty list is a no-op.
    :param shard_id: Passed as ``step`` so wandb's x-axis aligns with shard order.
    :param byte_size: Local shard file size in bytes; stable because shards are
        retained at ``work_dir``.
    :param render_seconds: Wall-clock seconds from subprocess invoke through
        upload-end; ``0.0`` on the R2-skip branch.
    :param rejections: Silent and clipped sampled draws rejected by the renderer.
    """
    payload = {
        "shard/bytes": byte_size,
        "shard/samples_rejected_clipped": rejections.clipped,
        "shard/samples_rejected_silent": rejections.silent,
        "shard/render_seconds": render_seconds,
    }
    for lg in loggers:
        try:
            lg.log_metrics(payload, step=shard_id)
        except Exception as exc:  # noqa: BLE001 — third-party logger failures must not abort the run
            logger.warning(f"log_metrics(shard) failed on {type(lg).__name__}: {exc}")


def _log_summary(
    loggers: list[Logger],
    *,
    rendered: int,
    skipped: int,
    total: int,
    elapsed_s: float,
    samples: int,
    rate: float,
    rejections: RenderRejectionMetrics,
) -> None:
    """Emit the run-level shard counters + e2e generation triple.

    Mirrors the loguru summary line at ``generate()`` so wandb history and
    stdout agree on the rate that the resumability MVP (#750, #1304) reports.

    :param loggers: Lightning loggers — empty list is a no-op.
    :param rendered: Shards this rank actually rendered.
    :param skipped: Shards short-circuited by the R2-skip probe.
    :param total: Owned shard count for this rank (``len(my_range)``); in
        claims mode, the claims this machine won (``rendered + skipped``).
    :param elapsed_s: Wall-clock seconds bracketing the dispatcher (includes
        the R2 skip probes by design).
    :param samples: ``rendered * spec.render.samples_per_shard``.
    :param rate: ``samples / elapsed_s`` (``0.0`` when ``elapsed_s == 0``).
    :param rejections: Silent and clipped sampled draws rejected across rendered shards.
    """
    payload = {
        "shards/rendered": rendered,
        "shards/skipped": skipped,
        "shards/total": total,
        "generation/elapsed_seconds": elapsed_s,
        "generation/samples": samples,
        "generation/samples_per_second": rate,
        "generation/samples_rejected_clipped": rejections.clipped,
        "generation/samples_rejected_silent": rejections.silent,
    }
    for lg in loggers:
        try:
            lg.log_metrics(payload)
        except Exception as exc:  # noqa: BLE001 — third-party logger failures must not abort the run
            logger.warning(f"log_metrics(summary) failed on {type(lg).__name__}: {exc}")


def _sum_rejections(
    left: RenderRejectionMetrics,
    right: RenderRejectionMetrics,
) -> RenderRejectionMetrics:
    """Sum rejection counts without mutating either validated report.

    :param left: Counts accumulated from earlier shards.
    :param right: Counts from the next completed shard.
    :returns: Field-wise total without modifying either input.
    """
    return RenderRejectionMetrics(
        clipped=left.clipped + right.clipped,
        silent=left.silent + right.silent,
    )


def _dispatch_shards_serial(
    spec: DatasetSpec,
    my_range: range,
    work_dir: Path,
    loggers: list[Logger],
) -> tuple[int, int, RenderRejectionMetrics]:
    """Render+upload owned shards in order; fail-fast on first error.

    :param spec: Validated dataset spec.
    :param my_range: Contiguous range of shard IDs owned by this rank.
    :param work_dir: Hydra per-run output dir; shards land here before upload.
    :param loggers: Forwarded to ``_render_one_owned_shard`` so per-shard
        byte size + render duration land in wandb history.
    :returns: Rendered/skipped shard counts and rejection totals over ``my_range``.
    """
    rendered = 0
    skipped = 0
    rejections = RenderRejectionMetrics()
    for shard_id in my_range:
        did_render, did_skip, shard_rejections = _render_one_owned_shard(
            spec, shard_id, work_dir, loggers
        )
        rendered += int(did_render)
        skipped += int(did_skip)
        rejections = _sum_rejections(rejections, shard_rejections)
    return rendered, skipped, rejections


def _dispatch_shards_parallel(
    spec: DatasetSpec,
    my_range: range,
    work_dir: Path,
    loggers: list[Logger],
) -> tuple[int, int, RenderRejectionMetrics]:
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
    :param loggers: Forwarded to ``_render_one_owned_shard`` so per-shard
        byte size + render duration land in wandb history.
    :returns: Rendered/skipped shard counts and rejection totals over ``my_range``.
    """
    workers = min(max(1, available_cpus() // 2), len(my_range))
    logger.info(f"parallel dispatch: workers={workers} shards={len(my_range)}")
    rendered = 0
    skipped = 0
    rejections = RenderRejectionMetrics()
    pending = iter(my_range)
    in_flight: set[Future[tuple[bool, bool, RenderRejectionMetrics]]] = set()
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        for _ in range(workers):
            sid = next(pending, None)
            if sid is None:
                break
            in_flight.add(pool.submit(_render_one_owned_shard, spec, sid, work_dir, loggers))
        while in_flight:
            done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            for fut in done:
                did_render, did_skip, shard_rejections = fut.result()
                rendered += int(did_render)
                skipped += int(did_skip)
                rejections = _sum_rejections(rejections, shard_rejections)
            for _ in range(len(done)):
                sid = next(pending, None)
                if sid is None:
                    break
                in_flight.add(pool.submit(_render_one_owned_shard, spec, sid, work_dir, loggers))
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return rendered, skipped, rejections


def _dispatch_shards(
    spec: DatasetSpec,
    *,
    work_dir: Path,
    loggers: list[Logger],
) -> tuple[int, int, int, RenderRejectionMetrics]:
    """Dispatch shards by dynamic claim-table wins or static rank/world ownership.

    :param spec: Validated dataset spec.
    :param work_dir: Directory where shards are rendered before upload.
    :param loggers: Loggers receiving per-shard metrics.
    :returns: Rendered/skipped/assigned counts plus rejection totals; ``assigned``
        is the static rank's shard count or the claims won by this machine.
    """
    if spec.use_shard_queue:
        logger.info(
            "shard distribution: dynamic claims over {num_shards} shards "
            "(use_shard_queue=true; rank/world partitioning bypassed)",
            num_shards=spec.num_shards,
        )
        if spec.render.parallel:
            logger.warning(
                "render.parallel=true is ignored with use_shard_queue=true: "
                "claims mode renders one claim at a time per machine"
            )
        rendered, skipped, rejections = _dispatch_shards_from_claims(
            _shard_claims_for_spec(spec),
            spec,
            work_dir=work_dir,
            loggers=loggers,
        )
        return rendered, skipped, rendered + skipped, rejections

    rank, world = read_rank_world_from_env()
    my_range = get_my_shards(spec.num_shards, rank=rank, world=world)
    logger.info(
        "shard partition: rank={rank}/{world} owns shard_ids "
        "[{start}, {stop}) ({owned} of {total} shards)",
        rank=rank,
        world=world,
        start=my_range.start,
        stop=my_range.stop,
        owned=len(my_range),
        total=spec.num_shards,
    )
    if spec.render.parallel and len(my_range) > 0:
        rendered, skipped, rejections = _dispatch_shards_parallel(
            spec, my_range, work_dir, loggers
        )
    else:
        rendered, skipped, rejections = _dispatch_shards_serial(spec, my_range, work_dir, loggers)
    return rendered, skipped, len(my_range), rejections


def _shard_claims_for_spec(spec: DatasetSpec) -> ShardClaims:
    """Build the run's shard-claims table from process-env storage settings.

    Callers run after ``ensure_r2_env_loaded`` (operator) or the worker env
    bootstrap, so the canonical ``SYNTH_SETTER_STORAGE_*`` keys are present.

    :param spec: Validated dataset spec; ``spec.r2`` locates the claims table.
    :returns: Claims over the run's ``metadata/shard-claims.lance`` table.
    """
    return ShardClaims.for_run(*r2_io.lance_target(spec.r2.shard_claims_uri()))


def _dispatch_shards_from_claims(
    claims: ShardClaims,
    spec: DatasetSpec,
    *,
    work_dir: Path,
    loggers: list[Logger],
) -> tuple[int, int, RenderRejectionMetrics]:
    """Claim, render, and complete shards until nothing is claimable.

    A failed render propagates with its claim left held: peers cannot pile
    onto the poison shard until the lease lapses, after which it is re-tried
    by whichever worker (or relaunch) claims it next.

    :param claims: Claims table from which this worker takes shards.
    :param spec: Validated dataset spec.
    :param work_dir: Directory where shards are rendered before upload.
    :param loggers: Loggers receiving per-shard metrics.
    :returns: Rendered/skipped counts and rejection totals over won claims.
    :raises ValueError: A claimed row's shard ID falls outside the spec —
        the persisted table drifted from the spec (negative IDs would
        otherwise silently render the wrong shard via Python indexing).
    """
    rendered = 0
    skipped = 0
    rejections = RenderRejectionMetrics()
    while (claimed := claims.claim()) is not None:
        if not 0 <= claimed.shard_id < spec.num_shards:
            raise ValueError(
                f"claimed shard_id {claimed.shard_id} is outside [0, {spec.num_shards})"
            )
        logger.info(
            "claimed shard {shard_id} (generation {claim_gen})",
            shard_id=claimed.shard_id,
            claim_gen=claimed.claim_gen,
        )
        did_render, did_skip, shard_rejections = _render_one_owned_shard(
            spec, claimed.shard_id, work_dir, loggers
        )
        claims.complete(claimed)
        rendered += int(did_render)
        skipped += int(did_skip)
        rejections = _sum_rejections(rejections, shard_rejections)
    return rendered, skipped, rejections


def _render_one_owned_shard(
    spec: DatasetSpec,
    shard_id: int,
    work_dir: Path,
    loggers: list[Logger],
) -> tuple[bool, bool, RenderRejectionMetrics]:
    """Render+stage one owned shard, or skip if it is already staged.

    Encapsulates the staging skip-probe + ``_render_and_upload_shard`` invocation
    so the serial and parallel dispatch arms share one callable. Emits one
    ``shard/bytes`` + ``shard/render_seconds`` history row per call —
    ``render_seconds == 0.0`` on the skip branch, wall-clock from subprocess
    invoke through upload-end on the render branch.

    :param spec: Validated dataset spec; ``spec.shards[shard_id]`` is fetched.
    :param shard_id: Index into ``spec.shards``; also the ``step`` for the row.
    :param work_dir: Hydra per-run output dir; shards land here before staging.
    :param loggers: Forwarded to ``_log_shard_metrics``.
    :returns: Rendered/skipped flags plus this call's rejection counts.
    """
    shard = spec.shards[shard_id]
    # A Lance shard is staged iff a complete attempt set (sidecar + stats +
    # .valid) exists; orphaned fragment data from a crash must not skip (#1776).
    if shard_has_complete_attempt(spec, shard.shard_id):
        logger.info("skipping shard {} — already staged: {}", shard.shard_id, shard.filename)
        # Staged fragment size isn't probed on the skip path; the metrics row
        # deliberately logs 0 bytes for an already-staged lance shard.
        _log_shard_metrics(
            loggers,
            shard_id=shard_id,
            byte_size=0,
            render_seconds=0.0,
            rejections=RenderRejectionMetrics(),
        )
        return False, True, RenderRejectionMetrics()
    t0 = time.monotonic()
    byte_size, rejections = _render_and_upload_shard(spec, shard, work_dir)
    logger.info(
        "shard {} render rejections: silent={} clipped={}",
        shard_id,
        rejections.silent,
        rejections.clipped,
    )
    _log_shard_metrics(
        loggers,
        shard_id=shard_id,
        byte_size=byte_size,
        render_seconds=time.monotonic() - t0,
        rejections=rejections,
    )
    return True, False, rejections


def _worker_id() -> str:
    """Return this worker's staging identity, sanitized for object-key use.

    The hostname (RunPod pod name locally unique per worker) identifies which
    infrastructure produced an attempt; it appears only in staging filenames
    and never in canonical paths (design §7.3).

    :returns: Hostname with any character outside ``[A-Za-z0-9._-]`` replaced by ``-``.
    """
    return re.sub(r"[^A-Za-z0-9._-]", "-", platform.node() or "worker")


def _load_render_rejections(metrics_path: Path, shard_id: int) -> RenderRejectionMetrics:
    """Validate a renderer sidecar and add shard context to contract failures.

    :param metrics_path: Renderer sidecar expected after a successful subprocess exit.
    :param shard_id: Logical shard used to identify a failed report.
    :returns: Strictly validated rejection counts.
    :raises RuntimeError: The sidecar is missing, unreadable, or invalid.
    """
    try:
        return RenderRejectionMetrics.model_validate_json(metrics_path.read_text())
    except (OSError, UnicodeError, ValidationError) as exc:
        raise RuntimeError(
            f"invalid render metrics for shard {shard_id}: {metrics_path}: {exc}"
        ) from exc


def _render_and_upload_shard(
    spec: DatasetSpec,
    shard: ShardSpec,
    work_dir: Path,
) -> tuple[int, RenderRejectionMetrics]:
    """Render a single shard and stage it to R2; shards are retained at ``work_dir``.

    Rendered shards stay on disk under ``work_dir`` for post-mortem inspection
    (``finalize_dataset`` re-downloads from R2 — launcher and worker pods do not
    share a filesystem). Peak local disk per rank scales with the number of owned
    shards. The renderer subprocess is wrapped in a retry loop bounded by
    ``spec.render.max_retries`` (default 0 = strict fail-fast); staging is outside
    the loop because its rclone transport already retries via ``--retries=3``.

    :param spec: Validated dataset spec; provides the render config and R2 URIs.
    :param shard: Shard to render; names the output dataset and seeds the renderer.
    :param work_dir: Hydra per-run output dir the shard is written under.
    :returns: Local shard byte size and validated renderer rejection counts.
    :raises subprocess.CalledProcessError: Renderer (or rclone) subprocess exited non-zero after
        exhausting the retry budget.
    :raises RuntimeError: Renderer output or metrics are missing/invalid, or
        the rendered Lance shard failed local validation.
    """
    # One identity threads the start marker and the staged attempt below.
    worker_id = _worker_id()
    attempt_uuid = uuid4().hex
    # Attempt start marker — append-only; orphaned without a .valid it is
    # the observable evidence of a crashed attempt (#1776).
    write_rendering_marker(spec, shard.shard_id, worker_id=worker_id, attempt_uuid=attempt_uuid)
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
        logger.info("rendering shard {} -> {}", shard.shard_id, shard.filename)
        max_attempts = spec.render.max_retries + 1
        metrics_path = render_metrics_path(work_dir / shard.filename)
        for attempt in range(max_attempts):
            metrics_path.unlink(missing_ok=True)
            try:
                _check_call_streamed(args)
                break
            except subprocess.CalledProcessError:
                if attempt + 1 == max_attempts:
                    raise
                logger.warning(
                    "shard {} render failed on attempt {}/{}; retrying",
                    shard.shard_id,
                    attempt + 1,
                    max_attempts,
                )
    shard_path = work_dir / shard.filename
    # Surface a generator that exited 0 without writing output here, not as a
    # downstream rclone "source not found". Lance shards are directories.
    if not shard_path.is_dir():
        raise RuntimeError(
            "generate_vst_dataset.py exited 0 but did not write expected shard "
            f"dataset: {shard_path}"
        )
    rejections = _load_render_rejections(metrics_path, shard.shard_id)
    byte_size = sum(p.stat().st_size for p in shard_path.rglob("*") if p.is_file())
    logger.info("shard rendered: {} ({} bytes)", shard_path, byte_size)
    # Worker-side validation gates staging — corrupt renders never earn a
    # .valid marker (design §7.3 shard write protocol).
    shard_errors = validate_shard(shard_path, spec)
    if shard_errors:
        raise RuntimeError(
            f"shard {shard.filename} failed local validation: {'; '.join(shard_errors)}"
        )
    stage_lance_shard_attempt(
        spec, shard, shard_path, worker_id=worker_id, attempt_uuid=attempt_uuid
    )
    logger.info(
        "shard staged: {} -> {}",
        shard.filename,
        spec.r2.shard_staging_dir_uri(shard.shard_id),
    )
    return byte_size, rejections


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


def _worker_python_bootstrap_cmd() -> str:
    """Build a worker-Python repair command that survives a stale checkout.

    :return: Bash command that sources the helper when present and otherwise performs the same
        guarded repair inline.
    """
    worker_venv = shlex.quote(_WORKER_VENV)
    return (
        f"{{ worker_venv={worker_venv}; "
        "if [[ -f scripts/ensure_worker_python.sh ]]; then "
        'source scripts/ensure_worker_python.sh "$worker_venv"; '
        "else "
        'worker_python="$worker_venv/bin/python"; '
        "export SYNTH_SETTER_WORKER_PYTHON_RECREATED=0; "
        'if [[ -n "${VIRTUAL_ENV:-}" && "$VIRTUAL_ENV" != "$worker_venv" ]]; then '
        'echo "ERROR: worker VIRTUAL_ENV must be $worker_venv (got $VIRTUAL_ENV)" >&2; '
        "false; "
        "else "
        'if [[ ! -x "$worker_python" ]] || ! '
        '"$worker_python" -c \'import sys; '
        "raise SystemExit(sys.version_info[:3] != (3, 12, 13))'; then "
        'echo "Recreating $worker_venv with Python 3.12.13"; '
        'rm -rf -- "$worker_venv"; '
        'uv venv --python 3.12.13 "$worker_venv"; '
        "export SYNTH_SETTER_WORKER_PYTHON_RECREATED=1; "
        "fi; "
        'export VIRTUAL_ENV="$worker_venv"; '
        'export PATH="$worker_venv/bin:$PATH"; '
        "fi; "
        "fi; }"
    )


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
        _worker_python_bootstrap_cmd(),
        "bash scripts/sync_worker_checkout.sh --python-ready",
        'if [[ "${SYNTH_SETTER_WORKER_PYTHON_RECREATED:-0}" == "1" && '
        '-z "${WORKER_GIT_REF:-}" ]]; then uv pip install --group runtime -e .; fi',
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
        ``finalize_inline=true``, or with a zero-size train / val / test split
        (the eval datamodule opens all three split files unconditionally).
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

    if sky_cfg.compute_template is None:
        ensure_dawdreamer_runtime(spec.render.renderer_backend)

    if sky_cfg.compute_template is None and cfg.oracle_eval_inline:
        if not cfg.finalize_inline:
            raise ValueError(
                "oracle_eval_inline=true requires finalize_inline=true; "
                "the inline oracle eval reads the {train,val,test}.lance datasets "
                "finalize produces."
            )
        if any(size == 0 for size in spec.train_val_test_sizes):
            raise ValueError(
                "oracle_eval_inline=true requires all of "
                f"train_val_test_sizes > 0; got {tuple(spec.train_val_test_sizes)}. "
                "VSTDataModule opens train.lance / val.lance / test.lance unconditionally."
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

    if spec.use_shard_queue:
        # Before any worker starts (local or SkyPilot) so no claim can race a
        # missing table; a relaunch only tops up rows absent from the table.
        inserted = _shard_claims_for_spec(spec).populate(shard.shard_id for shard in spec.shards)
        logger.info(
            "shard claims populated: inserted {inserted} of {num_shards} claim rows",
            inserted=inserted,
            num_shards=spec.num_shards,
        )

    if sky_cfg.compute_template is None:
        loggers = _loggers_pinned_to_spec(cfg, spec)
        # finalize runs outside the wandb-tracked region — see #1289.
        generate(spec, Path(cfg.paths.output_dir), loggers)
        if cfg.finalize_inline:
            finalize_from_spec(spec, Path(cfg.paths.output_dir))
        if cfg.oracle_eval_inline:
            output_dir = Path(cfg.paths.output_dir)
            splits: tuple[Split, ...] = ("train", "val", "test")
            # finalize writes each split dataset straight to R2 and stages only
            # stats.npz locally; materialize the splits for the local eval.
            for split in splits:
                r2_io.download_dir_no_overwrite(
                    spec.r2.split_lance_uri(split), output_dir / f"{split}.lance"
                )
            for split in splits:
                # test stays bare; train/val are namespaced so the shared run
                # keeps one summary key per split (see _run_oracle_eval_subprocess).
                metric_prefix = "" if split == "test" else f"{split}/"
                _run_oracle_eval_subprocess(
                    output_dir,
                    output_dir / "oracle_eval" / split / spec.run_id,
                    spec.run_id,
                    render=spec.render,
                    num_workers=cfg.datamodule.num_workers,
                    predict_file=output_dir / f"{split}.lance",
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
    # hydra.main types its wrapper as Any, so pyright sees the undecorated
    # one-arg signature; the wrapper itself takes no positional args.
    cast("Callable[[], None]", main)()
