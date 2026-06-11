"""General-purpose utilities: OmegaConf resolvers, task wrappers, and gradient watching."""

import hashlib
import re
import shutil
import warnings
from collections.abc import Callable
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import torch
from lightning import LightningModule
from lightning.pytorch.loggers import Logger, WandbLogger
from omegaconf import DictConfig, OmegaConf

from synth_setter.utils import pylogger, rich_utils

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

# Cap the readable cache-key slug so "<slug>-<sha256[:12]>" stays within the
# common 255-byte filename limit; the hash suffix preserves uniqueness.
_MAX_SLUG_LEN = 200


def register_resolvers() -> None:
    # Avoid double-registration when modules are imported multiple times in tests
    if not OmegaConf.has_resolver("mul"):
        OmegaConf.register_new_resolver("mul", lambda x, y: x * y)
    if not OmegaConf.has_resolver("div"):
        OmegaConf.register_new_resolver("div", lambda x, y: int(x) // int(y))
    if not OmegaConf.has_resolver("wandb"):
        OmegaConf.register_new_resolver("wandb", _resolve_wandb_checkpoint)


def _resolve_wandb_checkpoint(ref: str) -> str:
    """Resolve a W&B model-artifact reference to a local cached checkpoint path.

    Downloads the artifact under ``$PROJECT_ROOT/.cache/checkpoints/<key>`` once
    and reuses it on later resolutions, so re-resolving the same ``ref`` never
    re-downloads — unless the cached dir holds no ``.ckpt`` (a partial download),
    in which case it is refetched. The cache key (see :func:`_cache_key`) is a
    path-safe slug plus a hash, so a hostile ref (``..``, ``:``) can never escape
    the cache root and distinct refs never collide. ``wandb`` is imported lazily
    so importing this module never requires it.

    :param ref: Artifact ref such as ``entity/project/model-x:alias`` or
        ``model-x:latest``.
    :returns: Absolute path to the checkpoint inside the downloaded artifact —
        ``model.ckpt`` if present, else the sole ``.ckpt`` (see
        :func:`_select_checkpoint` for the no-ckpt / ambiguous-ckpt errors).
    :raises ModuleNotFoundError: If the optional ``wandb`` package is not installed.
    """
    if not find_spec("wandb"):
        raise ModuleNotFoundError(
            f"Resolving ${{wandb:{ref}}} requires the optional 'wandb' package; "
            "install the 'util' dependency group (aggregated into 'runtime'/'dev') "
            "to use this resolver."
        )
    import wandb
    from synth_setter.workspace import operator_workspace

    cache_dir = operator_workspace() / ".cache" / "checkpoints" / _cache_key(ref)
    checkpoints = sorted(cache_dir.glob("**/*.ckpt"))
    # A cached dir with no .ckpt is a partial download — refetch rather than trust it.
    if not checkpoints:
        cache_dir.mkdir(parents=True, exist_ok=True)
        _download_artifact_to_cache(wandb.Api().artifact(ref), cache_dir)
        checkpoints = sorted(cache_dir.glob("**/*.ckpt"))
    return _select_checkpoint(ref, checkpoints)


def _download_artifact_to_cache(artifact: Any, cache_dir: Path) -> None:
    """Materialize a model artifact's checkpoint(s) into ``cache_dir``.

    A reference-only artifact records the checkpoint as ``s3://`` manifest
    entries; W&B's native ``download`` cannot reach R2's custom endpoint, so each
    reference is rewritten to ``r2://`` and rclone-pulled instead. An artifact
    with no ``s3://`` references (a legacy file-upload artifact) falls back to the
    native ``download``. ``r2_io`` is imported lazily so the module stays cheap to
    import for callers that never resolve a reference.

    :param artifact: A ``wandb`` artifact exposing ``manifest.entries`` and ``download``.
    :param cache_dir: Destination cache directory for the resolved checkpoint(s).
    :raises ValueError: A reference's basename is unsafe (``.``/``..``/empty),
        which would escape ``cache_dir``.
    :raises RuntimeError: The ``rclone`` binary needed to fetch an R2 reference
        is not on ``PATH``.
    """
    from synth_setter.pipeline import r2_io

    refs = [
        entry.ref
        for entry in artifact.manifest.entries.values()
        if getattr(entry, "ref", None) and entry.ref.startswith("s3://")
    ]
    if not refs:
        artifact.download(root=str(cache_dir))
        return
    # Reference artifacts rclone-pull from R2. Check rclone up front so a missing
    # binary surfaces an actionable message instead of a bare FileNotFoundError
    # from deep inside ensure_r2_env_loaded's auth ping.
    if shutil.which("rclone") is None:
        raise RuntimeError(
            "Resolving an R2 reference checkpoint requires the 'rclone' binary on PATH."
        )
    # Populate the structural RCLONE_CONFIG_R2_* defaults (so an env wiring only the
    # secret keys resolves the r2: remote) and fail loud here if creds are absent —
    # a checkpoint that can't be fetched must error at resolve time, not yield an
    # empty cache.
    r2_io.ensure_r2_env_loaded()
    for s3_ref in refs:
        r2_uri = r2_io.from_s3_uri(s3_ref)
        name = Path(r2_uri).name
        if name in ("", ".", ".."):
            raise ValueError(
                f"W&B reference {s3_ref!r} has an unsafe checkpoint basename {name!r}"
            )
        r2_io.download_to_path(r2_uri, cache_dir / name)


def _cache_key(ref: str) -> str:
    """Build a collision-free, path-safe cache-dir name for a W&B artifact ref.

    The readable slug excludes ``.`` so no ``ref`` can produce a ``.`` / ``..``
    path segment that escapes the cache root; a short ``sha256`` suffix keeps
    distinct refs that slug identically (e.g. ``a/b`` vs ``a:b``) from colliding
    onto the same cache dir and returning the wrong checkpoint. The slug is
    capped so the name stays under the common 255-byte filename limit; the hash
    suffix preserves uniqueness across refs that share a truncated slug.

    :param ref: Artifact ref such as ``entity/project/model-x:alias``.
    :returns: ``<slug>-<sha256[:12]>``, safe as a single path component.
    """
    slug = re.sub(r"[^A-Za-z0-9_-]", "_", ref)[:_MAX_SLUG_LEN]
    digest = hashlib.sha256(ref.encode()).hexdigest()[:12]
    return f"{slug}-{digest}"


def _select_checkpoint(ref: str, checkpoints: list[Path]) -> str:
    """Pick the checkpoint a ``ref`` resolves to, erroring on an ambiguous artifact.

    :param ref: Originating artifact ref, used only for error messages.
    :param checkpoints: ``.ckpt`` paths found in the downloaded artifact.
    :returns: The sole ``model.ckpt`` if present, else the sole checkpoint.
    :raises FileNotFoundError: If ``checkpoints`` is empty.
    :raises ValueError: If several ``model.ckpt`` files, or several
        non-``model.ckpt`` checkpoints, make the selection ambiguous.
    """
    if not checkpoints:
        raise FileNotFoundError(f"W&B artifact {ref!r} contains no .ckpt")
    preferred = [p for p in checkpoints if p.name == "model.ckpt"]
    if len(preferred) > 1:
        paths = ", ".join(str(p) for p in preferred)
        raise ValueError(f"W&B artifact {ref!r} has ambiguous model.ckpt files ({paths})")
    if preferred:
        return str(preferred[0])
    if len(checkpoints) > 1:
        names = ", ".join(p.name for p in checkpoints)
        raise ValueError(
            f"W&B artifact {ref!r} has ambiguous checkpoints ({names}); none named model.ckpt"
        )
    return str(checkpoints[0])


def extras(cfg: DictConfig) -> None:
    """Apply optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)

    if precision := cfg.extras.get("float32_matmul_precision", False):
        log.info("Enabling float32 matmul precision! <cfg.extras.float32_matmul_precision=True>")
        torch.set_float32_matmul_precision(precision)


def task_wrapper(task_func: Callable) -> Callable:
    """Wrap a task function to control its failure behavior.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    .. code-block:: python

        @utils.task_wrapper
        def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            ...
            return metric_dict, object_dict

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: dict[str, Any], metric_name: str | None) -> float | None:
    """Safely retrieves value of the metric logged in LightningModule.

    :param metric_dict: A dict containing metric values.
    :param metric_name: If provided, the name of the metric to retrieve.
    :return: If a metric name was provided, the value of the metric.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


def watch_gradients(model: LightningModule, loggers: list[Logger]) -> None:
    """Watches gradients during training.

    :param model: The model to watch gradients for.
    :param loggers: A list of loggers to search for a WandbLogger.
    """
    for logger in loggers:
        if isinstance(logger, WandbLogger):
            logger.watch(model, log="gradients")
            return

    warnings.warn("WandbLogger not found in loggers! Skipping gradient watching...")
