"""Programmatic ``sky.Task`` construction from a validated ``ComputeConfig``.

Replaces the legacy raw-YAML compute templates: option YAMLs live in the
``skypilot_launch/compute`` Hydra group, bash lives in ``compute/scripts/*.sh``
package data, and this module assembles the task via ``sky.Task`` /
``sky.Resources`` constructors (no ``from_yaml_config`` on a raw file).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from functools import cache
from importlib.abc import Traversable
from typing import TYPE_CHECKING

from synth_setter.pipeline.schemas.compute import ComputeConfig, ComputeResources
from synth_setter.resources import configs_dir

if TYPE_CHECKING:
    import sky

WORKER_CMD_SENTINEL = "${WORKER_CMD}"

_COMPUTE_GROUP = "skypilot_launch/compute"


@cache
def load_compute_script(name: str) -> str:
    """Load one packaged compute script with its shebang/header comments stripped.

    Header stripping keeps the shipped setup/run text identical to the legacy
    inline YAML blocks while letting the ``.sh`` files carry a shebang (for
    shellcheck) and rationale comments.

    :param name: Script filename under ``configs/skypilot_launch/compute/scripts/``.
    :returns: Script body starting at the first non-comment line.
    :raises FileNotFoundError: No packaged script has this name.
    """
    ref = configs_dir() / "skypilot_launch" / "compute" / "scripts" / name
    if not ref.is_file():
        raise FileNotFoundError(
            f"compute script not found: {name} (expected under {_COMPUTE_GROUP}/scripts/)"
        )
    lines = ref.read_text(encoding="utf-8").splitlines(keepends=True)
    body_start = 0
    for i, line in enumerate(lines):
        if not line.startswith("#"):
            body_start = i
            break
    return "".join(lines[body_start:])


def compute_option_names() -> list[str]:
    """List every checked-in ``skypilot_launch/compute`` option name.

    :returns: Sorted option names (``runpod/smoke`` style) discovered from the
        packaged config tree.
    """

    def walk(node: Traversable, prefix: str) -> Iterator[str]:
        for child in node.iterdir():
            if child.is_dir() and child.name != "scripts":
                yield from walk(child, f"{prefix}{child.name}/")
            elif child.name.endswith(".yaml"):
                yield f"{prefix}{child.name.removesuffix('.yaml')}"

    return sorted(walk(configs_dir() / "skypilot_launch" / "compute", ""))


def load_compute_option(name: str) -> ComputeConfig:
    """Compose one ``skypilot_launch/compute`` option into a validated model.

    Uses the Hydra Compose API (not ``yaml.safe_load``) because debug options
    inherit their pool from ``runpod/smoke`` via a defaults list.

    :param name: Option name relative to the group (e.g. ``runpod/smoke``).
    :returns: Validated compute option.
    :raises ValueError: The option does not exist or fails validation.
    """
    from hydra import compose, initialize_config_module
    from hydra.core.global_hydra import GlobalHydra
    from hydra.errors import MissingConfigException
    from omegaconf import OmegaConf

    GlobalHydra.instance().clear()
    try:
        with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
            try:
                cfg = compose(overrides=[f"+{_COMPUTE_GROUP}={name}"])
            except MissingConfigException as exc:
                raise ValueError(
                    f"unknown {_COMPUTE_GROUP} option {name!r}; "
                    f"available: {', '.join(compute_option_names())}"
                ) from exc
    finally:
        GlobalHydra.instance().clear()
    raw = OmegaConf.to_container(cfg.skypilot_launch.compute, resolve=True)
    if not isinstance(raw, dict):
        raise ValueError(f"compute option {name!r} must compose to a mapping")
    return ComputeConfig(**{str(k): v for k, v in raw.items()})


def resolve_run_block(compute: ComputeConfig, cmd: str | None) -> str:
    """Resolve the task ``run`` block from the option's run source and ``cmd``.

    :param compute: Validated compute option.
    :param cmd: Launcher-injected worker command, when the caller has one.
    :returns: The final ``run`` block text.
    :raises ValueError: ``cmd`` is set alongside a ``run_script`` (it would be
        silently dropped), or missing where the option requires it.
    """
    if compute.run_script is not None:
        if cmd is not None:
            raise ValueError(
                f"compute option {compute.name!r} carries run_script="
                f"{compute.run_script!r}, but cmd is also set — cmd cannot be "
                "silently dropped. Pick a compute option without a run_script "
                "to opt into the cmd-injection flow."
            )
        return load_compute_script(compute.run_script)
    if cmd is None:
        raise ValueError(
            f"compute option {compute.name!r} takes an injected worker cmd, but none was given"
        )
    if compute.run_wrapper is not None:
        return load_compute_script(compute.run_wrapper).replace(WORKER_CMD_SENTINEL, cmd)
    return cmd


def resolve_volumes(compute: ComputeConfig, network_volume: str | None) -> dict[str, str]:
    """Resolve the task ``volumes`` mapping with both-or-neither validation.

    :param compute: Validated compute option.
    :param network_volume: SkyPilot volume name to mount, or ``None``.
    :returns: Task-level volumes mapping (empty when no mount is configured).
    :raises ValueError: The option mounts a volume without a configured name,
        or a name is configured with nowhere to land.
    """
    if compute.mount_network_volume is not None and network_volume is None:
        raise ValueError(
            f"compute option {compute.name!r} mounts a network volume at "
            f"{compute.mount_network_volume} but the launch config does not set "
            "network_volume; set it to the SkyPilot volume name for the target "
            "data center."
        )
    if network_volume is not None and compute.mount_network_volume is None:
        raise ValueError(
            f"launch config sets network_volume={network_volume!r} but compute "
            f"option {compute.name!r} has no mount_network_volume path to mount "
            "it at."
        )
    if compute.mount_network_volume is None or network_volume is None:
        return {}
    return {compute.mount_network_volume: network_volume}


def _sky_resources(entry: ComputeResources, image_id: str | None) -> list[sky.Resources]:
    """Expand one resources entry into constructor-built ``sky.Resources``.

    A multi-key ``accelerators`` mapping expands into one ``sky.Resources``
    per accelerator — mirroring ``sky.Resources.from_yaml_config``, which
    treats such a mapping as any-of (the bare constructor would keep the dict
    on a single Resources with different launch semantics).

    :param entry: Validated resources entry.
    :param image_id: Resolved ``docker:<image>`` pin, or ``None``.
    :returns: One ``sky.Resources`` per accelerator alternative.
    """
    import sky
    from sky.utils.registry import CLOUD_REGISTRY

    cloud = CLOUD_REGISTRY.from_str(entry.cloud)
    accelerator_options: list[dict[str, int | float] | None] = (
        [{accel: count} for accel, count in entry.accelerators.items()]
        if entry.accelerators
        else [None]
    )
    overrides = (
        {"kubernetes": {"pod_config": entry.kubernetes_pod_config}}
        if entry.kubernetes_pod_config is not None
        else None
    )
    return [
        sky.Resources(
            cloud=cloud,
            accelerators=accelerators,
            instance_type=entry.instance_type,
            cpus=entry.cpus,
            memory=entry.memory,
            disk_size=entry.disk_size,
            use_spot=entry.use_spot,
            image_id=image_id,
            _cluster_config_overrides=overrides,
        )
        for accelerators in accelerator_options
    ]


def build_sky_task(
    compute: ComputeConfig,
    *,
    cmd: str | None,
    worker_image: str | None,
    envs: Mapping[str, str] | None = None,
    network_volume: str | None = None,
) -> sky.Task:
    """Build a ``sky.Task`` from a compute option via SDK constructors.

    :param compute: Validated compute option.
    :param cmd: Launcher-injected worker command; forbidden with a
        ``run_script`` option, required otherwise.
    :param worker_image: ``repo:tag`` image pinned as ``docker:<image>`` into
        every resources entry (skipped for ``docker-in-run`` delivery, where
        the run wrapper consumes ``WORKER_IMAGE`` from env); ``None`` keeps
        any static ``image_id`` pins.
    :param envs: Base task environment (per-rank keys land later via
        ``task.update_envs``).
    :param network_volume: SkyPilot volume name mounted at the option's
        ``mount_network_volume`` path.
    :returns: Fully constructed task ready for ``sky.jobs.launch``.
    """
    import sky

    run = resolve_run_block(compute, cmd)
    volumes = resolve_volumes(compute, network_volume)
    setup = "".join(load_compute_script(script) for script in compute.setup_scripts)

    def image_id_for(entry: ComputeResources) -> str | None:
        if compute.image_delivery == "docker-in-run":
            return None
        return f"docker:{worker_image}" if worker_image is not None else entry.image_id

    resources_list = [
        res for entry in compute.resources for res in _sky_resources(entry, image_id_for(entry))
    ]

    task = sky.Task(
        run=run,
        setup=setup or None,
        envs=dict(envs) if envs else None,
        file_mounts=dict(compute.file_mounts) if compute.file_mounts else None,
        volumes=dict(volumes) if volumes else None,
    )
    # A set is SkyPilot's any-of (a list would be an ordered preference).
    task.set_resources(resources_list[0] if len(resources_list) == 1 else set(resources_list))
    return task
