"""Build SkyPilot task documents from validated Hydra compute options."""

from __future__ import annotations

from collections.abc import Iterator
from functools import cache
from importlib.abc import Traversable

from synth_setter.pipeline.schemas.compute import ComputeConfig
from synth_setter.resources import configs_dir

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


def build_task_doc(
    compute: ComputeConfig,
    *,
    cmd: str | None,
    network_volume: str | None = None,
) -> dict[str, object]:
    """Build a SkyPilot task document with scripts and volume names resolved.

    :param compute: Validated Hydra compute option.
    :param cmd: Worker command required by injected-command options.
    :param network_volume: Volume name paired with the option's mount path.
    :returns: Mapping accepted by ``sky.Task.from_yaml_config``.
    """
    task_doc = compute.model_dump(exclude_none=True)
    resources = task_doc.pop("resources")
    assert isinstance(resources, list)

    task_doc.pop("mount_network_volume", None)
    image_delivery = task_doc.pop("image_delivery")
    setup_scripts = task_doc.pop("setup_scripts")
    task_doc.pop("run_wrapper", None)
    task_doc.pop("run_script", None)

    if image_delivery == "docker-in-run":
        for resource in resources:
            resource.pop("image_id", None)
    task_doc["resources"] = resources[0] if len(resources) == 1 else {"any_of": resources}

    setup = "".join(load_compute_script(script) for script in setup_scripts)
    if setup:
        task_doc["setup"] = setup
    task_doc["run"] = resolve_run_block(compute, cmd)

    volumes = resolve_volumes(compute, network_volume)
    if volumes:
        task_doc["volumes"] = volumes
    return task_doc
