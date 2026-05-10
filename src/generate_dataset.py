"""Spec-driven generate_dataset runner.

Public API:
    compose_dataset_spec(experiment, overrides): Build a DatasetSpec by composing
        ``configs/dataset.yaml`` with the named experiment override.
    load_spec_from_uri(uri): Parse a DatasetSpec from a local path or r2:// URI.
    run(spec): Full flow — upload spec to R2 once, then loop over
        ``spec.shards`` rendering and uploading each.
    build_generate_args(spec, shard, output_dir): Build CLI args for
        src/data/vst/generate_vst_dataset.py.
    main(cfg): @hydra.main entrypoint — composes via ``configs/dataset.yaml``,
        constructs DatasetSpec, and calls run(spec) when invoked directly.

The container's runtime CLI (``scripts/docker_entrypoint.py generate_dataset
--spec <path-or-uri>``) parses an already-materialized spec and calls
``run(spec)`` in-process; that path skips Hydra composition because workers
receive the spec JSON via R2.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from src.data.vst.core import extract_renderer_version
from src.pipeline.constants import INPUT_SPEC_FILENAME
from src.pipeline.partitioning import get_my_shards, read_rank_world_from_env
from src.pipeline.r2_io import downloaded_to_tempfile, is_r2_uri
from src.pipeline.schemas.spec import DatasetSpec, ShardSpec

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONFIGS_DIR = _REPO_ROOT / "configs"


def compose_dataset_spec(experiment: str, *, overrides: list[str] | None = None) -> DatasetSpec:
    """Compose ``configs/dataset.yaml`` with ``experiment=<name>`` and any extra Hydra overrides.

    Returns a fully-validated ``DatasetSpec``. Group sub-trees on the composed
    config that aren't fields on ``DatasetSpec`` (e.g. ``data``, ``hydra``) are
    filtered out via positive selection in ``_dataset_spec_from_cfg``.

    Hydra's ``GlobalHydra`` is a process-global singleton: any prior
    initialization is cleared on entry, and the singleton is cleared again
    in ``finally`` so a composition failure leaves the process in a clean
    uninitialized state rather than half-set. We do not save and restore the
    caller's prior tree — callers that need their own Hydra context must
    re-initialize after this returns.
    """
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    overrides_list = [f"experiment={experiment}"] + (overrides or [])
    try:
        with initialize_config_dir(version_base="1.3", config_dir=str(_CONFIGS_DIR)):
            cfg = compose(config_name="dataset", overrides=overrides_list)
        return _dataset_spec_from_cfg(cfg)
    finally:
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()


def _dataset_spec_from_cfg(cfg: DictConfig) -> DatasetSpec:
    """Convert a composed Hydra config to a validated ``DatasetSpec``.

    Selects only keys that match ``DatasetSpec.model_fields``; group sub-trees
    like ``data`` and ``hydra`` are loaded by Hydra for composition's sake but
    aren't fields on the spec, so they get filtered out. Adding a new Hydra
    group does not silently break this path.
    """
    raw: object = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError(f"composed Hydra config is not a mapping: {type(raw).__name__}")
    spec_kwargs: dict[str, Any] = {
        k: v for k, v in raw.items() if isinstance(k, str) and k in DatasetSpec.model_fields
    }
    return DatasetSpec(**spec_kwargs)


def load_spec_from_uri(spec_uri: str) -> DatasetSpec:
    """Load a DatasetSpec from a local path or `r2://bucket/key` URI.

    Local paths are read directly. R2 URIs are downloaded via the shared
    ``src.pipeline.r2_io.downloaded_to_tempfile`` helper (rclone copyto to a
    fully-specified destination — robust to keys with subdirectory components
    that ``rclone copy`` would mirror under the tmpdir). The standard
    ``RCLONE_CONFIG_R2_*`` env vars must be set in the caller's environment.

    The R2-URI path exists because SkyPilot's RunPod backend rejects
    programmatic `task.update_file_mounts(...)` with a public-key-overflow
    error (see #749), so the launcher ships the spec via R2 instead of
    file_mounts.
    """
    if is_r2_uri(spec_uri):
        with downloaded_to_tempfile(spec_uri) as local_path:
            spec_text = local_path.read_text()
    else:
        spec_text = Path(spec_uri).read_text()
    return DatasetSpec.model_validate_json(spec_text)


# Bootstraps Xvfb + xsettingsd + dbus for VST3 plugin init; resolved relative
# to the container WORKDIR (``/home/build/synth-setter``) baked in the image.
# X11 wrapping lives at the audio-rendering boundary (this subprocess call),
# not at the container entrypoint — the click CLI stays X11-agnostic so idle
# and passthrough don't pay the Xvfb startup cost.
VST_HEADLESS_WRAPPER = "scripts/run-linux-vst-headless.sh"


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
    """Build CLI args for generate_vst_dataset.py from a spec and shard.

    The renderer takes a single ``--render-cfg-json`` arg holding the JSON-
    serialized ``RenderConfig`` sub-model; ``shard.filename``'s suffix is
    what selects the writer (``.h5`` → hdf5, ``.tar`` → wds).
    """
    output_path = output_dir / shard.filename
    return [
        sys.executable,
        "src/data/vst/generate_vst_dataset.py",
        str(output_path),
        "--render-cfg-json",
        spec.render.model_dump_json(),
    ]


def run(spec: DatasetSpec) -> None:
    """Upload the spec to R2 once, then render+upload each owned shard in turn.

    Spec serialization, spec upload, and the renderer-version constraint check
    happen once pre-loop. Each shard is rendered, uploaded, and unlinked before
    moving on — bounding local disk to one shard at a time. Subprocesses
    fail-fast: later shards are not attempted on subprocess error.

    The launcher builds the spec interpreter-only (no pedalboard / X11) trusting
    ``configs/render/<spec>.yaml``; this is where the worker — which has
    pedalboard — verifies the plugin and pinned ``renderer_version`` agree.

    Raises:
        RuntimeError: If the worker's plugin version disagrees with
            ``spec.renderer_version``.
    """
    render = spec.render
    actual_renderer_version = extract_renderer_version(Path(render.plugin_path))
    if actual_renderer_version != render.renderer_version:
        raise RuntimeError(
            f"Renderer version mismatch: spec pins {render.renderer_version!r} but "
            f"plugin at {render.plugin_path} reports {actual_renderer_version!r}. "
            "Rebuild the image against the matching SURGE_GIT_REF, or bump "
            "renderer_version in configs/render/surge_xt.yaml."
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

    r2_dest_prefix = f"r2:{spec.r2_bucket}/{spec.r2_prefix}"

    with tempfile.TemporaryDirectory() as work_dir_str:
        work_dir = Path(work_dir_str)
        # rclone copy preserves the source basename, and the local file is already
        # named INPUT_SPEC_FILENAME — so the prefix-directory destination lands at
        # `{prefix}{INPUT_SPEC_FILENAME}` without a double-name.
        spec_path = work_dir / INPUT_SPEC_FILENAME
        spec_path.write_text(spec.model_dump_json(indent=2))
        logger.info(f"spec written: {spec_path}")
        _rclone_copy(str(spec_path), r2_dest_prefix)
        logger.info(f"spec uploaded -> {r2_dest_prefix}")

        for shard_id in my_range:
            _render_and_upload_shard(spec, spec.shards[shard_id], work_dir, r2_dest_prefix)


def _render_and_upload_shard(
    spec: DatasetSpec,
    shard: ShardSpec,
    work_dir: Path,
    r2_dest_prefix: str,
) -> None:
    """Render a single shard, upload it to R2, then unlink the local file.

    Unlinking after upload bounds local disk to one shard at a time — necessary on disk-constrained
    workers running multi-shard partitions.
    """
    args = [VST_HEADLESS_WRAPPER] if sys.platform == "linux" else []
    args += build_generate_args(spec, shard, work_dir)
    logger.info(f"rendering shard {shard.shard_id} -> {shard.filename}")
    subprocess.check_call(args)  # noqa: S603 — args built from validated spec
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
    shard_path.unlink()
    logger.info(f"shard removed locally: {shard_path}")


@hydra.main(version_base="1.3", config_path=str(_CONFIGS_DIR), config_name="dataset")
def main(cfg: DictConfig) -> None:
    """Hydra-driven entrypoint: compose ``configs/dataset.yaml``, materialize, run.

    Constructs a ``DatasetSpec`` from the composed config on line 1 of the body
    and proceeds to ``run(spec)``. Operators invoke this via
    ``python -m src.generate_dataset experiment=<name> [+overrides...]``;
    workers in containerized clusters bypass this path and consume an
    already-materialized spec via ``scripts/docker_entrypoint.py generate_dataset --spec <uri>``.
    """
    spec = _dataset_spec_from_cfg(cfg)
    run(spec)


if __name__ == "__main__":
    main()
