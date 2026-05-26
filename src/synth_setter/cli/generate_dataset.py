"""Spec-driven generate_dataset runner.

Two console-script surfaces:

- ``synth-setter-generate-dataset`` → :func:`main` — operator entry; runs the
  spec in-process or dispatches it to SkyPilot based on
  ``cfg.skypilot_launch.compute_template``.
- ``synth-setter-generate-dataset-from-hydra`` → :func:`from_hydra` — worker
  entry; pure ``@hydra.main`` re-compose so launcher/worker share argv.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
import tempfile
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import AbstractContextManager, ExitStack, nullcontext
from pathlib import Path
from typing import Any

import hydra
from hydra import compose, initialize_config_module
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from synth_setter.data.vst.core import extract_renderer_version
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.partitioning import (
    available_cpus,
    get_my_shards,
    read_rank_world_from_env,
)
from synth_setter.pipeline.schemas.skypilot_launch import SkypilotLaunchConfig
from synth_setter.pipeline.schemas.spec import DatasetSpec, ShardSpec
from synth_setter.pipeline.spec_io import (
    upload_spec,
    write_spec_locally,
)
from synth_setter.resources import as_file, vst_headless_wrapper
from synth_setter.workspace import operator_workspace

# Composed-config keys that aren't DatasetSpec fields (interpolation sources, Hydra
# runtime, dispatch-mode sub-trees). See dataset.yaml in the shipped configs/
# tree for the live set. ``r2`` is *not* listed: it composes from
# ``r2/default.yaml`` directly into the spec's nested ``R2Location`` field.
_NON_SPEC_KEYS: tuple[str, ...] = (
    "data",
    "paths",
    "hydra",
    "run_name",
    "skypilot_launch",
)

# Anchor for ``cfg.paths.*`` interpolations and the local spec mirror.
# See :func:`synth_setter.workspace.operator_workspace` for resolution.
_OPERATOR_WORKSPACE = operator_workspace()

# Worker-side checkout path — baked WORKDIR of the dev-snapshot image, not the
# launcher's workspace (which may not exist on the worker filesystem).
_WORKER_REPO_ROOT = "/home/build/synth-setter"


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
    subprocess.check_call(args)  # noqa: S603 — args from validated spec
    # Distinct sentinel so we can grep CI logs for "rclone returned" and tell
    # at a glance whether the rclone subprocess actually exited (vs. hanging
    # post-upload — see #735). If the upload itself failed, check_call already
    # raised before we got here.
    logger.info(f"rclone returned cleanly: {src} -> {dest}")


def build_generate_args(spec: DatasetSpec, shard: ShardSpec, output_dir: Path) -> list[str]:
    """Build CLI args for ``generate_vst_dataset.py`` from a spec and shard.

    The flag set is derived from ``RenderConfig.model_fields`` so every renderer
    config field surfaces as a ``--<field>`` option automatically; adding a
    field on the model auto-extends the CLI invocation. The writer is dispatched
    on ``shard.filename``'s suffix inside the subprocess via
    ``EXTENSION_TO_OUTPUT_FORMAT``.
    """
    output_path = output_dir / shard.filename
    args = [
        sys.executable,
        "src/synth_setter/data/vst/generate_vst_dataset.py",
        str(output_path),
    ]
    for key, value in spec.render.model_dump().items():
        args.extend([f"--{key}", str(value)])

    return args


def run(spec: DatasetSpec) -> None:
    """Render+upload each owned shard in turn.

    Spec upload no longer happens here — ``main()`` writes the canonical R2
    copy once on the launcher host before either calling ``run(spec)`` inline
    (local-run) or dispatching to a SkyPilot worker pod that re-enters via
    ``from_hydra`` → ``run(spec)``. Each shard is rendered, uploaded, and
    (when ``spec.output_dir`` is ``None``) unlinked before moving on, bounding
    local disk to one shard at a time. With ``spec.output_dir`` set, shards
    are staged there and kept on disk after upload; peak local disk grows
    with the rank's owned-shard count. Subprocesses fail-fast: later shards
    are not attempted on subprocess error.

    Before each render, R2 is probed for the shard's destination object: if it already exists with
    non-zero size, the shard is skipped (resumability MVP — see #750). The probe uses
    ``check=True``, so a non-zero rclone exit (auth, network) propagates as a hard failure rather
    than degrading silently into a re-render.

    The launcher builds the spec interpreter-only (no pedalboard / X11) trusting
    ``configs/render/<spec>.yaml``; this is where the worker — which has pedalboard
    — verifies the plugin and pinned ``renderer_version`` agree.

    :raises RuntimeError: If the worker's plugin version disagrees with
        ``spec.render.renderer_version``.
    """
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

    work_dir_cm: AbstractContextManager[str]
    if spec.output_dir is None:
        work_dir_cm = tempfile.TemporaryDirectory()
    else:
        work_dir = Path(spec.output_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"staging shards at operator-supplied output_dir: {work_dir}")
        work_dir_cm = nullcontext(str(work_dir))

    with work_dir_cm as work_dir_str:
        work_dir = Path(work_dir_str)
        if spec.render.parallel and len(my_range) > 0:
            rendered, skipped = _dispatch_shards_parallel(spec, my_range, work_dir, r2_dest_prefix)
        else:
            rendered, skipped = _dispatch_shards_serial(spec, my_range, work_dir, r2_dest_prefix)
        logger.info(
            f"shard summary: rendered={rendered} skipped={skipped} of {len(my_range)} assigned"
        )


def _dispatch_shards_serial(
    spec: DatasetSpec,
    my_range: range,
    work_dir: Path,
    r2_dest_prefix: str,
) -> tuple[int, int]:
    """Render+upload owned shards in order; fail-fast on first error.

    :param spec: Validated dataset spec.
    :param my_range: Contiguous range of shard IDs owned by this rank.
    :param work_dir: Per-run tempdir owned by ``run()``.
    :param r2_dest_prefix: ``spec.r2.rclone_prefix()``.
    :returns: ``(rendered, skipped)`` summary counts over ``my_range``.
    """
    rendered = 0
    skipped = 0
    for shard_id in my_range:
        r, s = _render_one_owned_shard(spec, shard_id, work_dir, r2_dest_prefix)
        rendered += int(r)
        skipped += int(s)
    return rendered, skipped


def _dispatch_shards_parallel(
    spec: DatasetSpec,
    my_range: range,
    work_dir: Path,
    r2_dest_prefix: str,
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
    :param work_dir: Per-run staging dir owned by ``run()`` (tempdir, or
        ``spec.output_dir`` when set); with a tempdir peak local disk scales
        with the pool size (one in-flight shard per worker thread), with
        ``spec.output_dir`` the unlink is skipped so peak disk scales with
        ``my_range``.
    :param r2_dest_prefix: ``spec.r2.rclone_prefix()``.
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
                pool.submit(_render_one_owned_shard, spec, sid, work_dir, r2_dest_prefix)
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
                    pool.submit(_render_one_owned_shard, spec, sid, work_dir, r2_dest_prefix)
                )
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return rendered, skipped


def _render_one_owned_shard(
    spec: DatasetSpec,
    shard_id: int,
    work_dir: Path,
    r2_dest_prefix: str,
) -> tuple[bool, bool]:
    """Render+upload one owned shard, or skip if R2 already has it.

    Encapsulates the R2 skip-probe + ``_render_and_upload_shard`` invocation
    so the serial and parallel dispatch arms share one callable.

    :param spec: Validated dataset spec; ``spec.shards[shard_id]`` is fetched.
    :param shard_id: Index into ``spec.shards``.
    :param work_dir: Per-run tempdir owned by ``run()``.
    :param r2_dest_prefix: ``spec.r2.rclone_prefix()``.
    :returns: ``(rendered, skipped)`` — exactly one is ``True``.
    """
    shard = spec.shards[shard_id]
    existing_size = r2_io.object_size(spec.r2.shard_uri(shard))
    if existing_size is not None and existing_size > 0:
        logger.info(
            f"skipping shard {shard.shard_id} — already in R2 "
            f"({existing_size} bytes): {shard.filename}"
        )
        return False, True
    _render_and_upload_shard(spec, shard, work_dir, r2_dest_prefix)
    return True, False


def _render_and_upload_shard(
    spec: DatasetSpec,
    shard: ShardSpec,
    work_dir: Path,
    r2_dest_prefix: str,
) -> None:
    """Render a single shard, upload it to R2, then unlink (unless ``spec.output_dir`` is set).

    With ``spec.output_dir is None``, unlinking after upload bounds local disk to one
    in-flight shard per caller — necessary on disk-constrained workers running multi-shard
    partitions. Under ``render.parallel=True`` this bound applies per worker thread, so peak
    local disk scales with the dispatcher's pool size (see ``_dispatch_shards_parallel`` and
    ``RenderConfig.parallel``). When ``spec.output_dir`` is set the unlink is skipped so the
    operator can inspect/reuse the shards; the disk bound then no longer holds and peak local
    disk scales with the rank's full assigned range (or the entire dataset for a single-worker
    run). The renderer subprocess is wrapped in a retry loop bounded by ``spec.render.max_retries``
    (default 0 = strict fail-fast); rclone is outside the loop because it already retries via
    ``--retries=3``.

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
                subprocess.check_call(args)  # noqa: S603 — args built from validated spec
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
    logger.info(f"shard rendered: {shard_path} ({shard_path.stat().st_size} bytes)")
    _rclone_copy(str(shard_path), r2_dest_prefix)
    logger.info(f"shard uploaded: {shard.filename} -> {r2_dest_prefix}")
    if spec.output_dir is None:
        shard_path.unlink()
        logger.info(f"shard removed locally: {shard_path}")
    else:
        logger.info(f"shard kept locally (output_dir set): {shard_path}")


def spec_from_cfg(cfg: DictConfig) -> DatasetSpec:
    """Build a DatasetSpec from a Hydra-composed cfg.

    Resolves all interpolations, drops the non-DatasetSpec sub-trees, and constructs the model.
    Raises if the composed config is not a mapping.
    """
    raw: object = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError(f"composed config is not a mapping: {type(raw).__name__}")
    spec_kwargs: dict[str, Any] = {
        k: v for k, v in raw.items() if isinstance(k, str) and k not in _NON_SPEC_KEYS
    }
    return DatasetSpec(**spec_kwargs)


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
    return SkypilotLaunchConfig(**sky_kwargs)


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

    :param overrides: Operator's Hydra overrides (launcher's ``sys.argv[1:]``).
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


@hydra.main(version_base="1.3", config_path="pkg://synth_setter.configs", config_name="dataset")
def from_hydra(cfg: DictConfig) -> None:
    """Worker-side @hydra.main entry: build the spec and render it in-process.

    :param cfg: Composed Hydra dataset cfg supplied by ``@hydra.main`` from
        the worker's argv overrides.
    """
    run(spec_from_cfg(cfg))


def main() -> None:
    """Operator CLI: compose dataset cfg from argv, then run locally or dispatch to SkyPilot.

    Uses programmatic compose (not ``@hydra.main``) so the dispatch branch can
    be picked from ``cfg.skypilot_launch.compute_template`` before Hydra would
    hand the cfg straight to the body. Overrides are replayed verbatim on the
    worker so the launcher/worker composition matches byte-for-byte.
    """
    overrides = list(sys.argv[1:])

    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(config_name="dataset", overrides=overrides)

    # Programmatic compose leaves ${hydra:runtime.output_dir} unset; pin paths.*
    # so spec_from_cfg's resolve step doesn't trip on the unresolved interpolation.
    cfg.paths.root_dir = str(_OPERATOR_WORKSPACE)
    cfg.paths.output_dir = str(_OPERATOR_WORKSPACE)
    cfg.paths.work_dir = str(_OPERATOR_WORKSPACE)

    spec = spec_from_cfg(cfg)
    sky_cfg = _sky_cfg_from_dataset_cfg(cfg)

    # ``_OPERATOR_WORKSPACE`` (not cfg.paths.output_dir) is the anchor: the
    # paths.* pins above are defensive shims for
    # ${hydra:runtime.output_dir} resolution, not the operator-side
    # artifact root.
    spec_path = write_spec_locally(spec, _OPERATOR_WORKSPACE)
    logger.info(f"wrote local spec to {spec_path}")

    # Load + validate R2 creds once for the whole run, then upload the
    # canonical spec from this single launcher-side site. Both branches benefit:
    # local-run uses the just-loaded env for run()'s shard uploads, and the
    # dispatch branch lets the worker boot already pointing at an existing
    # canonical object (no per-rank re-write).
    env_file = Path(sky_cfg.env_file).expanduser() if sky_cfg.env_file else None
    r2_io.ensure_r2_env_loaded(env_file)
    r2_uri = upload_spec(spec)
    logger.info(f"spec uploaded -> {r2_uri}")

    if sky_cfg.compute_template is None:
        run(spec)
        return

    # Deferred import — SkyPilot pulls heavy provider SDKs on import.
    from synth_setter.pipeline.skypilot_launch import dispatch_via_skypilot

    sky_cfg = sky_cfg.model_copy(update={"cmd": _build_worker_cmd(overrides, spec)})
    # ``input_spec_uri()`` (not ``uri(INPUT_SPEC_FILENAME)``) — the former
    # includes the run's prefix so the worker reads the same canonical object
    # ``main()`` just uploaded.
    dispatch_via_skypilot(spec, sky_cfg, spec_uri=spec.r2.input_spec_uri())


if __name__ == "__main__":
    main()
