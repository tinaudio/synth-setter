"""Spec-driven generate_dataset runner.

``main(cfg)`` is the Hydra-composed CLI entry, invoked via
``python -m synth_setter.cli.generate_dataset experiment=<id>`` (or the
``synth-setter-generate-dataset`` console script).

The click CLI in ``src/synth_setter/tools/docker_entrypoint.py`` is the SkyPilot-worker entry that reads a
pre-materialized spec from R2 via ``load_spec_from_uri``.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import hydra
import rootutils
from loguru import logger
from omegaconf import DictConfig, OmegaConf

# Bootstrap PROJECT_ROOT + sys.path — see https://github.com/ashleve/rootutils.
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from synth_setter.data.vst.core import extract_renderer_version  # noqa: E402
from synth_setter.pipeline import r2_io  # noqa: E402
from synth_setter.pipeline.constants import INPUT_SPEC_FILENAME  # noqa: E402
from synth_setter.pipeline.partitioning import (  # noqa: E402
    get_my_shards,
    read_rank_world_from_env,
)
from synth_setter.pipeline.schemas.spec import DatasetSpec, ShardSpec  # noqa: E402

# Composed-config keys that aren't DatasetSpec fields: ``data`` / ``r2`` are interpolation
# sources for top-level keys; ``paths`` / ``hydra`` exist only for Hydra runtime; ``run_name``
# is a Hydra-output-dir interpolation source (see ``configs/dataset.yaml``);
# ``compute_template`` is a dispatch knob consumed by ``main`` before DatasetSpec construction.
_NON_SPEC_KEYS: tuple[str, ...] = (
    "data",
    "r2",
    "paths",
    "hydra",
    "run_name",
    "compute_template",
)

# Resolves ``cfg.compute_template`` (a name) to a YAML under ``configs/compute/`` when the
# Hydra entry decides to dispatch via SkyPilot — kept relative to this file so the package
# remains importable regardless of CWD.
_COMPUTE_DIR = Path(__file__).resolve().parents[3] / "configs" / "compute"


def load_spec_from_uri(spec_uri: str) -> DatasetSpec:
    """Load a DatasetSpec from a local path or `r2://bucket/key` URI.

    Local paths are read directly. R2 URIs are downloaded via rclone (which
    requires the standard `RCLONE_CONFIG_R2_*` env vars to be set in the
    caller's environment) into a tmpdir and parsed.

    The R2-URI path exists because SkyPilot's RunPod backend rejects
    programmatic `task.update_file_mounts(...)` with a public-key-overflow
    error (see #749), so the launcher ships the spec via R2 instead of
    file_mounts.
    """
    if r2_io.is_r2_uri(spec_uri):
        with r2_io.downloaded_to_tempfile(spec_uri) as local_path:
            spec_text = local_path.read_text()
    else:
        spec_text = Path(spec_uri).read_text()
    return DatasetSpec.model_validate_json(spec_text)


# Bootstraps Xvfb + xsettingsd + dbus for VST3 plugin init; resolved relative
# to the container WORKDIR (``/home/build/synth-setter``) baked in the image.
# X11 wrapping lives at the audio-rendering boundary (this subprocess call),
# not at the container entrypoint — the click CLI stays X11-agnostic so idle
# and passthrough don't pay the Xvfb startup cost.
VST_HEADLESS_WRAPPER = "docker/ubuntu22_04/run-linux-vst-headless.sh"


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
    """Upload the spec to R2 once, then render+upload each owned shard in turn.

    Spec serialization, spec upload, and the renderer-version constraint check
    happen once pre-loop. Each shard is rendered, uploaded, and unlinked before
    moving on — bounding local disk to one shard at a time. Subprocesses
    fail-fast: later shards are not attempted on subprocess error.

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

        rendered = 0
        skipped = 0
        for shard_id in my_range:
            shard = spec.shards[shard_id]
            shard_object_uri = r2_io.shard_uri(spec.r2_bucket, spec.r2_prefix, shard.filename)
            existing_size = r2_io.object_size(shard_object_uri)
            if existing_size is not None and existing_size > 0:
                logger.info(
                    f"skipping shard {shard.shard_id} — already in R2 "
                    f"({existing_size} bytes): {shard.filename}"
                )
                skipped += 1
                continue
            _render_and_upload_shard(spec, shard, work_dir, r2_dest_prefix)
            rendered += 1
        logger.info(
            f"shard summary: rendered={rendered} skipped={skipped} of {len(my_range)} assigned"
        )


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


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="dataset")
def main(cfg: DictConfig) -> None:
    """Hydra-composed CLI entry: ``python -m synth_setter.cli.generate_dataset experiment=<id>``.

    When ``cfg.compute_template`` is non-empty, the run is dispatched to the named SkyPilot
    template (one of ``configs/compute/<name>.yaml``) via ``dispatch_via_skypilot`` — same
    code path as the standalone launcher CLI. Left null (the default), the entrypoint
    renders shards in-process via ``run``.
    """
    spec = spec_from_cfg(cfg)
    template_name = cfg.get("compute_template")
    if not template_name:
        run(spec)
        return

    # Deferred so the local (no-compute_template) path doesn't pay the SkyPilot import cost.
    from synth_setter.pipeline.schemas.compute import compute_config_from_cfg
    from synth_setter.pipeline.skypilot_launch import dispatch_via_skypilot

    compute_config = compute_config_from_cfg(cfg, compute_dir=_COMPUTE_DIR)
    dispatch_via_skypilot(spec, compute_config)


if __name__ == "__main__":
    main()
