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
from pathlib import Path
from typing import Any

import hydra
import rootutils
from hydra import compose, initialize_config_dir
from loguru import logger
from omegaconf import DictConfig, OmegaConf

# Bootstrap PROJECT_ROOT + sys.path — see https://github.com/ashleve/rootutils.
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from synth_setter.data.vst.core import extract_renderer_version  # noqa: E402
from synth_setter.pipeline import r2_io  # noqa: E402
from synth_setter.pipeline.partitioning import (  # noqa: E402
    get_my_shards,
    read_rank_world_from_env,
)
from synth_setter.pipeline.schemas.skypilot_launch import SkypilotLaunchConfig  # noqa: E402
from synth_setter.pipeline.schemas.spec import DatasetSpec, ShardSpec  # noqa: E402
from synth_setter.pipeline.spec_io import (  # noqa: E402
    upload_spec,
    write_spec_locally,
)

# Composed-config keys that aren't DatasetSpec fields (interpolation sources, Hydra
# runtime, dispatch-mode sub-trees). See configs/dataset.yaml for the live set.
# ``r2`` is *not* listed: it composes from ``configs/r2/default.yaml`` directly
# into the spec's nested ``R2Location`` field after the migration.
_NON_SPEC_KEYS: tuple[str, ...] = (
    "data",
    "paths",
    "hydra",
    "run_name",
    "skypilot_launch",
)

# Resolve repo root from this file so the entry-point is cwd-independent.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIR = _REPO_ROOT / "configs"

# Worker-side checkout path — baked WORKDIR of the dev-snapshot image, not the
# launcher's _REPO_ROOT (which may not exist on the worker filesystem).
_WORKER_REPO_ROOT = "/home/build/synth-setter"


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

    r2_dest_prefix = spec.r2.rclone_prefix()

    with tempfile.TemporaryDirectory() as work_dir_str:
        work_dir = Path(work_dir_str)
        r2_uri = upload_spec(spec)
        logger.info(f"spec uploaded -> {r2_uri}")

        rendered = 0
        skipped = 0
        for shard_id in my_range:
            shard = spec.shards[shard_id]
            shard_object_uri = spec.r2.shard_uri(shard)
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


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="dataset")
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

    with initialize_config_dir(version_base="1.3", config_dir=str(_CONFIG_DIR)):
        cfg = compose(config_name="dataset", overrides=overrides)

    # Programmatic compose leaves ${hydra:runtime.output_dir} unset; pin paths.*
    # so spec_from_cfg's resolve step doesn't trip on the unresolved interpolation.
    cfg.paths.root_dir = str(_REPO_ROOT)
    cfg.paths.output_dir = str(_REPO_ROOT)
    cfg.paths.work_dir = str(_REPO_ROOT)

    spec = spec_from_cfg(cfg)
    sky_cfg = _sky_cfg_from_dataset_cfg(cfg)

    # _REPO_ROOT (not cfg.paths.output_dir) is the anchor: the paths.* pins
    # above are defensive shims for ${hydra:runtime.output_dir} resolution,
    # not the operator-side artifact root.
    spec_path = write_spec_locally(spec, _REPO_ROOT)
    logger.info(f"wrote local spec to {spec_path}")

    if sky_cfg.compute_template is None:
        run(spec)
        return

    # Runner-side R2 upload deferred — rclone env loads inside
    # dispatch_via_skypilot, after this point. See workflow-spec-upload-delegation series.

    # Deferred import — SkyPilot pulls heavy provider SDKs on import.
    from synth_setter.pipeline.skypilot_launch import dispatch_via_skypilot

    sky_cfg = sky_cfg.model_copy(update={"cmd": _build_worker_cmd(overrides, spec)})
    dispatch_via_skypilot(spec, sky_cfg)


if __name__ == "__main__":
    main()
