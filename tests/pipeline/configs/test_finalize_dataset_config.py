"""Config-layer tests that compose the real ``finalize_dataset.yaml`` Hydra config.

The peer entrypoints (eval, generate_dataset) compose their shipped YAML in
config-layer tests, but the finalize suite builds every spec by hand via
``OmegaConf.create`` and never composes ``finalize_dataset.yaml`` — so a break
in its ``defaults`` / ``paths`` / ``logger`` composition passes the whole
finalize suite and only fails in production. These bare-compose tests pin the
load-bearing fields ``finalize()`` reads: ``dataset_root_uri``,
``paths.output_dir``, the ``logger: wandb`` group, and the ``hydra`` run.dir /
job_logging interpolations.

Runtime-only interpolations (``paths.output_dir`` → ``${hydra:runtime.output_dir}``
and ``job_logging``'s ``${hydra.runtime.output_dir}``) populate only under a live
``@hydra.main`` run, so these tests assert the raw templates; the entrypoint
suite drives the decorated ``main()`` for their resolution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

# A finalize cfg needs only ``dataset_root_uri`` (the sole ``???`` field).  The
# value's scheme is never dereferenced at the config layer.
_DATASET_ROOT_URI = "r2://bucket/run/"


def _raw(node: DictConfig) -> dict[str, Any]:
    """Return ``node`` as a plain dict with interpolations left as raw templates.

    :param node: A composed config node to unwrap without resolving.
    :returns: The node's contents as nested plain Python with ``${…}`` strings intact.
    """
    return cast("dict[str, Any]", OmegaConf.to_container(node, resolve=False))


def _compose_finalize() -> DictConfig:
    """Compose ``finalize_dataset.yaml`` with the required ``dataset_root_uri``.

    Uses ``return_hydra_config=True`` so the ``hydra.*`` sub-tree (carrying the
    run.dir / job_logging interpolations finalize relies on) is present in the
    composed cfg.

    :returns: The composed cfg; the caller must clear ``GlobalHydra``.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        return compose(
            config_name="finalize_dataset",
            return_hydra_config=True,
            overrides=[f"dataset_root_uri={_DATASET_ROOT_URI}"],
        )


def test_finalize_config_surfaces_dataset_root_uri_override() -> None:
    """The composed cfg carries the ``dataset_root_uri`` override finalize loads the spec from.

    ``finalize()`` reads ``cfg.dataset_root_uri`` (a ``???`` field) and passes it
    to ``load_spec_from_root``; a passing compose proves the field is surfaced and
    the override lands.
    """
    cfg = _compose_finalize()
    try:
        assert cfg.dataset_root_uri == _DATASET_ROOT_URI
    finally:
        GlobalHydra.instance().clear()


def test_finalize_config_paths_output_dir_is_runtime_output_template() -> None:
    """``paths.output_dir`` is the runtime-output template finalize passes as the work dir.

    ``finalize()`` reads ``cfg.paths.output_dir`` as the scratch dir; the
    ``paths: default`` group binds it to ``${hydra:runtime.output_dir}``, which
    only resolves under a live ``@hydra.main`` run, so the config layer pins the
    raw template.
    """
    cfg = _compose_finalize()
    try:
        assert _raw(cfg.paths)["output_dir"] == "${hydra:runtime.output_dir}"
    finally:
        GlobalHydra.instance().clear()


def test_finalize_config_composes_wandb_logger_group() -> None:
    """The ``logger: wandb`` default composes the ``WandbLogger`` finalize logs the artifact to.

    ``finalize()`` reads ``cfg.logger`` (``instantiate_loggers`` + the
    ``logger.wandb.resume`` update), so the composed group must expose the
    ``WandbLogger`` target and bind ``save_dir`` to ``paths.output_dir``.
    """
    cfg = _compose_finalize()
    try:
        assert list(cfg.logger.keys()) == ["wandb"]
        assert cfg.logger.wandb._target_ == "lightning.pytorch.loggers.wandb.WandbLogger"
        assert _raw(cfg.logger.wandb)["save_dir"] == "${paths.output_dir}"
    finally:
        GlobalHydra.instance().clear()


def test_finalize_config_run_dir_resolves_under_pinned_log_dir(
    cfg_finalize: DictConfig, tmp_path: Path
) -> None:
    """The overridden ``hydra.run.dir`` resolves under the fixture's pinned ``paths.log_dir``.

    ``finalize_dataset.yaml`` overrides ``run.dir`` to
    ``${paths.log_dir}/finalize_dataset/${now:…}`` because the shared group
    default references ``${run_name}``, which this cfg does not surface. The
    ``cfg_finalize`` fixture pins ``paths.log_dir`` to ``tmp_path``, so the
    override resolves without that unbound key — also exercising the fixture.

    :param cfg_finalize: Function-scoped finalize cfg with paths pinned to ``tmp_path``.
    :param tmp_path: Per-test path the fixture binds ``paths.log_dir`` to.
    """
    run_dir = OmegaConf.select(cfg_finalize, "hydra.run.dir")
    assert run_dir.startswith(f"{tmp_path}/finalize_dataset/")


def test_finalize_config_job_logging_filename_interpolates_task_name() -> None:
    """``job_logging``'s log filename carries the ``${task_name}`` the cfg surfaces.

    The shared ``hydra/default.yaml`` interpolates ``${task_name}`` into
    ``job_logging.handlers.file.filename``; ``finalize_dataset.yaml`` surfaces
    ``task_name: finalize_dataset`` precisely so that template resolves. The
    filename's other half (``${hydra.runtime.output_dir}``) is runtime-only, so
    the config layer pins the raw template and the surfaced ``task_name``.
    """
    cfg = _compose_finalize()
    try:
        assert cfg.task_name == "finalize_dataset"
        filename = _raw(cfg.hydra.job_logging.handlers.file)["filename"]
        assert filename == "${hydra.runtime.output_dir}/${task_name}.log"
    finally:
        GlobalHydra.instance().clear()
