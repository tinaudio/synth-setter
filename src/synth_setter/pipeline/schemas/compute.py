"""SkyPilot compute config schema and Hydra-compose entrypoint.

A ``ComputeConfig`` validates a SkyPilot Task YAML at the launcher's trust boundary,
mirroring ``DatasetSpec`` for dataset generation and ``ImageConfig`` for image builds.
The flow:

1. Hydra composes ``configs/skypilot_launch.yaml`` and resolves a ``compute_template``
   *name* (a string field, not a sub-tree — OmegaConf's interpolation grammar can't parse
   the literal ``${VAR}`` bash expansions inside the compute YAMLs' ``setup:`` / ``run:``
   blocks).
2. ``compute_config_from_cfg`` reads that name and loads
   ``configs/compute/<name>.yaml`` directly via ``yaml.safe_load`` — sidestepping
   OmegaConf.
3. ``ComputeConfig`` validates the parsed dict at the trust boundary.
4. The launcher dispatches per-rank via ``sky.Task.from_yaml_config(model_dump())``.

The model intentionally stays permissive (``extra="allow"``) because SkyPilot's Task schema
has many optional fields (``num_nodes``, ``file_mounts``, ``workdir``, ``service``, ...) and
the goal of this layer is to (a) prove required fields are present at launcher time, (b)
make compute templates Hydra-selectable, and (c) round-trip cleanly to SkyPilot — not to
re-implement SkyPilot's own schema validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig
from pydantic import BaseModel, ConfigDict, Field


class ComputeConfig(BaseModel):  # noqa: DOC601,DOC603
    """Validated SkyPilot Task config — the YAML loaded by the launcher per ``--template``.

    Required fields are those the launcher reads or that SkyPilot rejects when absent for our
    workload (``resources``, ``run``). ``setup`` and ``envs`` are optional — not every template
    needs a pre-run install step, and ``task.update_envs`` can add keys at launch time.
    Extra keys pass through unchanged so callers can extend templates with any
    SkyPilot-supported field (e.g. ``num_nodes``, ``file_mounts``, ``workdir``) without
    touching this schema.
    """

    # ``strict=False`` (not strict=True like DatasetSpec/ImageConfig) because SkyPilot Task
    # YAMLs carry mixed-type values inside ``resources`` (e.g. ``disk_size: 50`` ints, ``cpus: 1+``
    # strings, accelerators flow-mapping idioms); ``yaml.safe_load`` already normalizes scalars
    # by syntax, and SkyPilot owns the final type contract via ``from_yaml_config``.
    model_config = ConfigDict(strict=False, frozen=True, extra="allow")

    resources: dict[str, Any]
    envs: dict[str, Any] = Field(default_factory=dict)
    setup: str | None = None
    run: str


def load_compute_config_yaml(path: Path) -> ComputeConfig:  # noqa: DOC203
    """Load a SkyPilot Task YAML and validate it as a ``ComputeConfig``.

    Used both by tests (to pin every shipped template) and by the launcher itself: the Click
    CLI reads ``--template <path>`` directly so CI workflows that pass full paths keep working,
    and Hydra-composed callers funnel through ``compute_config_from_cfg`` (which delegates here
    after resolving the name).

    :param path: Path to a YAML file under ``configs/compute/`` (or any SkyPilot Task YAML).
    :returns: Validated ``ComputeConfig`` populated from the YAML contents.
    :raises FileNotFoundError: ``path`` does not exist or is not a file.
    :raises ValueError: top-level YAML is not a mapping.
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        # Re-raise as ValueError so callers (notably ``skypilot_launch.py``) can catch
        # parse and validation failures with the same ``except`` clause without taking a
        # direct dependency on PyYAML's exception hierarchy.
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        # Normalize ``None`` to ``{}`` above (empty YAML file is acceptable — Pydantic
        # will surface the missing required fields). Other non-mapping top-levels
        # (lists, scalars) get a clear error rather than a confusing
        # ``ComputeConfig.__init__()`` complaint about missing required fields.
        raise ValueError(f"Top-level YAML in {path} must be a mapping, got {type(raw).__name__}")
    return ComputeConfig(**raw)


def compute_config_from_cfg(  # noqa: DOC203
    cfg: DictConfig, *, compute_dir: Path
) -> ComputeConfig:
    """Build a ``ComputeConfig`` by resolving ``cfg.compute_template`` (a name) to a YAML file.

    The launcher's top-level Hydra entrypoint (``configs/skypilot_launch.yaml``) declares
    ``compute_template: runpod-template`` — a string naming a file under ``compute_dir``.
    Sub-tree composition is avoided here because the compute YAMLs contain literal ``${VAR}``
    bash expansions in ``setup:`` / ``run:`` that fail OmegaConf's interpolation grammar at
    DictConfig-load time. Loading the YAML directly with ``yaml.safe_load`` (via
    ``load_compute_config_yaml``) sidesteps that conflict.

    The field is ``compute_template`` (not ``compute``) so Hydra doesn't treat
    ``compute=X`` as a defaults-list override against the ``configs/compute/`` group.

    A trailing ``.yaml`` on the name is tolerated (and stripped) so users can write either
    ``compute_template=runpod-template`` or ``compute_template=runpod-template.yaml``. Path
    separators (``/``, ``\\``) and leading dots are rejected so the name can't escape
    ``compute_dir`` or resolve to a hidden file.

    :param cfg: Hydra DictConfig with a string ``compute_template`` field naming the file.
    :param compute_dir: Directory containing ``<name>.yaml`` compute templates.
    :returns: Validated ``ComputeConfig`` populated from
        ``compute_dir/<cfg.compute_template>.yaml``.
    :raises KeyError: ``cfg`` has no ``compute_template`` key.
    :raises ValueError: ``cfg.compute_template`` is not a non-empty filename stem
        (e.g. contains path separators or a leading dot).
    """
    if "compute_template" not in cfg:
        raise KeyError(
            "Hydra cfg has no `compute_template` field; expected a template name string."
        )
    name = cfg.compute_template
    if not isinstance(name, str) or not name:
        raise ValueError(
            f"cfg.compute_template must be a non-empty string template name, got {name!r}"
        )
    stem = name[: -len(".yaml")] if name.endswith(".yaml") else name
    if not stem or stem.startswith(".") or any(sep in stem for sep in ("/", "\\")):
        raise ValueError(
            f"cfg.compute_template must be a simple filename stem (no path separators or "
            f"leading dots), got {name!r}"
        )
    return load_compute_config_yaml(compute_dir / f"{stem}.yaml")
