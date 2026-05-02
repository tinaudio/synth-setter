"""Spec-driven generate_dataset runner.

Public API:
    load_spec_from_uri(uri): Parse a DatasetPipelineSpec from a local path or r2:// URI.
    run(spec): Full flow — upload spec to R2, generate shard, upload shard to R2.
    run(spec): Full flow — upload spec to R2 once, then loop over
        ``spec.shards`` rendering and uploading each.
    build_generate_args(spec, shard, output_dir): Build CLI args for
        src/data/vst/generate_vst_dataset.py.

This module is no longer invocable via ``python -m``; the container's CLI
entrypoint (``scripts/docker_entrypoint.py generate_dataset --spec <path-or-uri>``)
parses the spec and calls ``run(spec)`` in-process.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from loguru import logger

from pipeline.constants import INPUT_SPEC_FILENAME
from pipeline.schemas.spec import DatasetPipelineSpec, ShardSpec
from src.data.vst.core import extract_renderer_version

_R2_URI_SCHEME = "r2://"


def load_spec_from_uri(spec_uri: str) -> DatasetPipelineSpec:
    """Load a DatasetPipelineSpec from a local path or `r2://bucket/key` URI.

    Local paths are read directly. R2 URIs are downloaded via rclone (which
    requires the standard `RCLONE_CONFIG_R2_*` env vars to be set in the
    caller's environment) into a tmpdir and parsed.

    The R2-URI path exists because SkyPilot's RunPod backend rejects
    programmatic `task.update_file_mounts(...)` with a public-key-overflow
    error (see #749), so the launcher ships the spec via R2 instead of
    file_mounts.
    """
    if spec_uri.startswith(_R2_URI_SCHEME):
        rclone_path = "r2:" + spec_uri[len(_R2_URI_SCHEME) :]
        with tempfile.TemporaryDirectory() as tmpdir:
            args = [  # noqa: S607 — rclone resolved by image's PATH
                "rclone",
                "copy",
                "--checksum",
                rclone_path,
                tmpdir,
            ]
            subprocess.check_call(args)  # noqa: S603 — args from validated spec URI
            local_path = Path(tmpdir) / Path(spec_uri).name
            spec_text = local_path.read_text()
    else:
        spec_text = Path(spec_uri).read_text()
    return DatasetPipelineSpec.model_validate_json(spec_text)


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


def build_generate_args(
    spec: DatasetPipelineSpec, shard: ShardSpec, output_dir: Path
) -> list[str]:
    """Build CLI args for generate_vst_dataset.py from a spec and shard.

    Args:
        spec: Materialized pipeline spec (dataset-level parameters).
        shard: The specific shard to generate (owns filename).
        output_dir: Directory for the output HDF5 file.

    Returns:
        List of CLI arguments for generate_vst_dataset.py.
    """
    output_file = output_dir / shard.filename

    options = {
        "plugin_path": spec.plugin_path,
        "preset_path": spec.preset_path,
        "sample_rate": spec.sample_rate,
        "channels": spec.channels,
        "velocity": spec.velocity,
        "signal_duration_seconds": spec.signal_duration_seconds,
        "min_loudness": spec.min_loudness,
        "param_spec": spec.param_spec,
        "sample_batch_size": spec.sample_batch_size,
    }

    args = [
        sys.executable,
        "src/data/vst/generate_vst_dataset.py",
        str(output_file),
        str(spec.shard_size),
    ]
    for key, value in options.items():
        args.extend([f"--{key}", str(value)])

    return args


def run(spec: DatasetPipelineSpec) -> None:
    """Materialized-spec driven run: upload spec to R2 once, then render+upload each shard.

    Spec serialization, spec upload, and the renderer-version constraint check
    happen once per run (pre-loop). Each shard is then rendered, uploaded, and
    its local HDF5 file unlinked before moving on — so local disk usage is
    bounded to one shard's worth at a time. Subprocess failures propagate
    immediately (fail-fast); later shards are not attempted.

    HDF5-only. Other output formats (e.g. WebDataset) are not yet wired into
    the downstream generator.

    Args:
        spec: Pre-materialized DatasetPipelineSpec.

    Raises:
        ValueError: If ``spec.output_format != "hdf5"``.
        RuntimeError: If the worker's plugin version disagrees with
            ``spec.renderer_version``.
    """
    if spec.output_format != "hdf5":
        raise ValueError(
            f"generate_vst_dataset.py only supports hdf5 output, got: {spec.output_format}"
        )

    # Constraint check: the plugin actually present on this worker must match the
    # renderer_version pinned into the spec at materialize time. The launcher trusts
    # SURGE_XT_RENDERER_VERSION blindly so its code path stays interpreter-only;
    # the worker is where pedalboard is available, so this is where we verify.
    actual_renderer_version = extract_renderer_version(Path(spec.plugin_path))
    if actual_renderer_version != spec.renderer_version:
        raise RuntimeError(
            f"Renderer version mismatch: spec pins {spec.renderer_version!r} but "
            f"plugin at {spec.plugin_path} reports {actual_renderer_version!r}. "
            "Rebuild the image against the matching SURGE_GIT_REF, or bump "
            "SURGE_XT_RENDERER_VERSION in pipeline.schemas.spec."
        )
    logger.info(f"renderer_version OK: plugin at {spec.plugin_path} == {spec.renderer_version}")

    r2_dest_prefix = f"r2:{spec.r2_bucket}/{spec.r2_prefix}"

    with tempfile.TemporaryDirectory() as work_dir_str:
        work_dir = Path(work_dir_str)

        # `rclone copy` treats the destination as a directory and preserves
        # the source's basename. The local spec file is already named
        # INPUT_SPEC_FILENAME, so uploading to the prefix directory lands
        # the object at `{prefix}{INPUT_SPEC_FILENAME}` without the
        # double-name issue a full object-key destination would cause.
        spec_path = work_dir / INPUT_SPEC_FILENAME
        spec_path.write_text(spec.model_dump_json(indent=2))
        logger.info(f"spec written: {spec_path}")
        _rclone_copy(str(spec_path), r2_dest_prefix)
        logger.info(f"spec uploaded -> {r2_dest_prefix}")

        for shard in spec.shards:
            _render_and_upload_shard(spec, shard, work_dir, r2_dest_prefix)


def _render_and_upload_shard(
    spec: DatasetPipelineSpec,
    shard: ShardSpec,
    work_dir: Path,
    r2_dest_prefix: str,
) -> None:
    """Render a single shard, upload it to R2, then unlink the local file.

    Unlinking after upload bounds local disk to one shard's HDF5 at a time — necessary for multi-
    shard runs on disk-constrained workers.
    """
    args = [VST_HEADLESS_WRAPPER, *build_generate_args(spec, shard, work_dir)]
    logger.info(f"rendering shard {shard.shard_id} -> {shard.filename}")
    subprocess.check_call(args)  # noqa: S603 — args built from validated spec
    shard_path = work_dir / shard.filename
    # Catches a generator that exits 0 without writing output — surfaces a clear error
    # at the rendering boundary rather than a downstream rclone "source not found".
    if not shard_path.is_file():
        raise RuntimeError(
            f"generate_vst_dataset.py exited 0 but did not write expected shard file: {shard_path}"
        )
    logger.info(f"shard rendered: {shard_path} ({shard_path.stat().st_size} bytes)")
    _rclone_copy(str(shard_path), r2_dest_prefix)
    logger.info(f"shard uploaded: {shard.filename} -> {r2_dest_prefix}")
    shard_path.unlink()
    logger.info(f"shard removed locally: {shard_path}")


if __name__ == "__main__":
    raise SystemExit(
        "pipeline.entrypoints.generate_dataset is no longer invocable via `python -m`. "
        "Use `scripts/docker_entrypoint.py generate_dataset --spec <path>` instead."
    )
