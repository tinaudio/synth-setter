"""Hydra entrypoint for evaluating a trained model on a datamodule's test split."""

import json
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
import wandb
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import DictConfig, OmegaConf

from synth_setter.pipeline import r2_io
from synth_setter.pipeline.schemas.spec import _get_git_sha
from synth_setter.resources import as_file, vst_headless_wrapper
from synth_setter.run_id import make_wandb_run_id
from synth_setter.utils import (
    RankedLogger,
    extras,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    log_wandb_provenance,
    pin_wandb_run_id,
    register_resolvers,
    resolve_run_config_id,
    task_wrapper,
)
from synth_setter.workspace import operator_workspace

_PREDICT_VST_AUDIO_MODULE = "synth_setter.evaluation.predict_vst_audio"
_COMPUTE_AUDIO_METRICS_MODULE = "synth_setter.evaluation.compute_audio_metrics"
_SUBPROCESS_TIMEOUT_SECONDS = 600
_AGGREGATED_METRICS_FILENAME = "aggregated_metrics.csv"
_AGGREGATED_METRICS_SHUFFLED_FILENAME = "aggregated_metrics_shuffled.csv"
_AGGREGATED_METRICS_STATS: tuple[str, ...] = ("mean", "std")

# Resolve workspace at import so ``${oc.env:PROJECT_ROOT}`` in
# ``configs/paths/default.yaml`` interpolates under any install layout.
operator_workspace()

register_resolvers()

log = RankedLogger(__name__, rank_zero_only=True)


def _load_audio_metrics(metrics_dir: Path) -> dict[str, float]:
    """Flatten ``aggregated_metrics.csv`` into ``{"audio/<name>_<stat>": value}``.

    :param metrics_dir: Directory containing the ``aggregated_metrics.csv`` produced by
        :mod:`synth_setter.evaluation.compute_audio_metrics`; rows are metric names,
        columns are :data:`_AGGREGATED_METRICS_STATS`.
    :returns: One entry per ``(metric, stat)`` cell of the CSV.
    :raises FileNotFoundError: when the producing subprocess returned 0 without writing the
        CSV; surfaced so the silent-success failure mode is loud.
    :raises ValueError: when the CSV is missing a required stat column.
    """
    csv_path = metrics_dir / _AGGREGATED_METRICS_FILENAME
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"{_AGGREGATED_METRICS_FILENAME} missing at {csv_path} — the compute_audio_metrics "
            "subprocess returned 0 but did not write the aggregated CSV."
        )
    df = pd.read_csv(csv_path, index_col=0)
    missing = [stat for stat in _AGGREGATED_METRICS_STATS if stat not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path} missing required stat columns {missing}; got {list(df.columns)}."
        )
    result: dict[str, float] = {
        f"audio/{metric}_{stat}": float(df.at[metric, stat])
        for metric in df.index
        for stat in _AGGREGATED_METRICS_STATS
    }
    shuffled_path = metrics_dir / _AGGREGATED_METRICS_SHUFFLED_FILENAME
    if shuffled_path.is_file():
        shuffled_df = pd.read_csv(shuffled_path, index_col=0)
        missing_s = [s for s in _AGGREGATED_METRICS_STATS if s not in shuffled_df.columns]
        if missing_s:
            raise ValueError(
                f"{shuffled_path} missing required stat columns {missing_s}; "
                f"got {list(shuffled_df.columns)}."
            )
        result.update(
            {
                f"shuffled_audio/{metric}_{stat}": float(shuffled_df.at[metric, stat])
                for metric in shuffled_df.index
                for stat in _AGGREGATED_METRICS_STATS
            }
        )
    return result


def _log_audio_metrics_to_wandb(audio_metrics: dict[str, float]) -> None:
    """No-op when ``wandb.run`` is unset; otherwise log to it, swallowing wandb errors.

    :param audio_metrics: Forwarded verbatim to ``wandb.run.log``.
    """
    if wandb.run is None:
        return
    try:
        wandb.run.log(audio_metrics)
    except Exception as exc:
        log.warning(f"wandb.run.log raised {type(exc).__name__}: {exc}; metrics still returned.")


def _run_predict_postprocessing(cfg: DictConfig) -> dict[str, float]:  # noqa: DOC502,DOC503
    """Render VST audio, compute audio metrics, and return their aggregated values.

    The VST render subprocess is prefixed with the headless wrapper on Linux so
    the VST3 plugin gets an Xvfb display before pedalboard imports it; the
    metrics subprocess is CPU-only and needs no wrapper. ``--shuffle_seed`` is
    always forwarded; the render-order probe (#489) runs automatically inside
    ``compute_audio_metrics`` when all sample dirs have identical params.

    :param cfg: Reads ``cfg.evaluation`` (gates + ``num_workers`` + ``shuffle_seed``
        + optional ``metric_prefix``), ``cfg.render`` (param spec, preset, optional
        plugin path), and ``cfg.paths.output_dir`` (base for ``predictions/``,
        ``audio/``, ``metrics/``).
    :returns: ``{"<metric_prefix>audio/<name>_<stat>": value}`` when ``compute_metrics``
        ran (``metric_prefix`` empty by default); empty dict otherwise. Always
        rank-zero — the caller gates DDP duplication.
    :raises ValueError: if ``evaluation.render_vst`` is enabled but ``cfg.render`` is
        unset, or the expected input directory for a stage is missing.
    :raises subprocess.CalledProcessError: propagated from a non-zero subprocess exit.
    :raises subprocess.TimeoutExpired: propagated when a subprocess exceeds
        :data:`_SUBPROCESS_TIMEOUT_SECONDS`.
    """
    output_dir = Path(cfg.paths.output_dir)
    predictions_dir = output_dir / "predictions"
    audio_dir = output_dir / "audio"
    metrics_dir = output_dir / "metrics"

    if cfg.evaluation.render_vst:
        if cfg.get("render") is None:
            raise ValueError(
                "evaluation.render_vst=true requires a render config group "
                "(e.g. `render=surge_xt`); cfg.render is unset."
            )
        if not predictions_dir.is_dir():
            raise ValueError(
                f"evaluation.render_vst=true expects predictions at {predictions_dir} "
                "— configure a PredictionWriter callback (e.g. `callbacks=prediction_writer`) "
                "so trainer.predict writes one params CSV per sample before rendering."
            )
        with ExitStack() as stack:
            args: list[str] = []
            if sys.platform == "linux":
                wrapper_path = Path(stack.enter_context(as_file(vst_headless_wrapper())))
                args.append(str(wrapper_path))
            args += [
                sys.executable,
                "-m",
                _PREDICT_VST_AUDIO_MODULE,
                str(predictions_dir),
                str(audio_dir),
                "--param_spec",
                cfg.render.param_spec_name,
                "--preset_path",
                cfg.render.preset_path,
            ]
            if cfg.render.get("plugin_path"):
                args += ["--plugin_path", cfg.render.plugin_path]
            # Forward the remaining render fields predict_vst_audio renders with so the
            # re-render matches the dataset's generation render rather than this module's
            # CLI defaults. Gated like plugin_path so a partial render cfg still works.
            for flag, key in (
                ("--sample_rate", "sample_rate"),
                ("--channels", "channels"),
                ("--velocity", "velocity"),
                ("--signal_duration_seconds", "signal_duration_seconds"),
            ):
                value = cfg.render.get(key)
                if value is not None:
                    args += [flag, str(value)]
            if cfg.evaluation.rerender_target:
                args.append("-t")
            log.info(f"Rendering predicted audio: {args}")
            subprocess.run(  # noqa: S603
                args,
                check=True,
                timeout=_SUBPROCESS_TIMEOUT_SECONDS,
            )

    if cfg.evaluation.compute_metrics:
        if not audio_dir.is_dir():
            raise ValueError(
                f"evaluation.compute_metrics=true expects rendered audio at {audio_dir} "
                "— enable evaluation.render_vst or point cfg.paths.output_dir at a "
                "directory containing an `audio/` subdirectory."
            )
        args = [
            sys.executable,
            "-m",
            _COMPUTE_AUDIO_METRICS_MODULE,
            str(audio_dir),
            str(metrics_dir),
            "--shuffle_seed",
            str(cfg.evaluation.get("shuffle_seed", 0)),
            "-w",
            str(cfg.evaluation.num_workers),
        ]
        log.info(f"Computing audio metrics: {args}")
        subprocess.run(  # noqa: S603
            args,
            check=True,
            timeout=_SUBPROCESS_TIMEOUT_SECONDS,
        )
        audio_metrics = _load_audio_metrics(metrics_dir)
        # Namespace every key (audio/* and shuffled_audio/*) per caller — e.g. one
        # wandb run shared across splits — so passes don't overwrite each other.
        prefix = cfg.evaluation.get("metric_prefix", "")
        if prefix:
            audio_metrics = {f"{prefix}{key}": value for key, value in audio_metrics.items()}
        _log_audio_metrics_to_wandb(audio_metrics)
        return audio_metrics

    return {}


@task_wrapper
def evaluate(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate the given checkpoint on a datamodule testset.

    Wrapped in ``@task_wrapper`` so crashes still flush the run dir.

    :param cfg: Hydra-composed cfg; ``cfg.ckpt_path=None`` is allowed and
        evaluates the in-memory model (Lightning's documented no-op).
    :return: ``(metric_dict, object_dict)``. ``metric_dict`` merges
        ``trainer.callback_metrics`` (``torch.Tensor`` values) with audio
        metrics from :func:`_run_predict_postprocessing` (Python ``float``),
        so callers iterating values must handle both.
    """
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    pin_wandb_run_id(cfg, make_wandb_run_id(resolve_run_config_id(cfg)), "evaluation")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger, callbacks=callbacks)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
        "callbacks": callbacks,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)
        log_wandb_provenance()

    mode = cfg.get("mode", "test")

    audio_metrics: dict[str, float] = {}
    if mode == "test":
        log.info("Starting testing!")
        trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.ckpt_path,
            weights_only=False,
        )
    # Accept both spellings for backwards compatibility with older configs.
    elif mode == "val" or mode == "validate":
        log.info("Starting validating!")
        trainer.validate(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.ckpt_path,
            weights_only=False,
        )
    elif mode == "predict":
        trainer.predict(
            model=model,
            dataloaders=datamodule,
            ckpt_path=cfg.ckpt_path,
            return_predictions=False,
            weights_only=False,
        )
        # Rank-zero gate: trainer.predict runs on every rank in DDP/multi-device
        # setups, but the postprocessing subprocesses share one output_dir.
        if trainer.is_global_zero:
            audio_metrics = _run_predict_postprocessing(cfg)

    metric_dict: dict[str, Any] = dict(trainer.callback_metrics)
    metric_dict.update(audio_metrics)

    # Persist + publish results here, not after evaluate() returns: @task_wrapper's
    # finally closes the W&B run on return, so the eval-results artifact has to be
    # logged while the run is still open or it would attach to nothing.
    # All ranks share one output_dir, so gate the dump on global-zero (as with the
    # upload + artifact log below) to avoid concurrent writers corrupting metrics.json.
    if trainer.is_global_zero:
        _dump_metric_dict(metric_dict, Path(cfg.paths.output_dir))
    _maybe_upload_output_dir(cfg, trainer.is_global_zero)
    upload_uri = _upload_output_dir_uri(cfg)
    # _get_git_sha() shells out, so only invoke it on the path that actually logs
    # the artifact (global-zero with a configured R2 prefix).
    if trainer.is_global_zero and upload_uri:
        _log_eval_results_artifact(logger, cfg, upload_uri, metric_dict, _get_git_sha())

    return metric_dict, object_dict


def _dump_metric_dict(metric_dict: dict[str, Any], output_dir: Path) -> Path:
    """Serialize ``metric_dict`` to ``<output_dir>/metrics/metrics.json``; return the path.

    Lightning tensors and numpy arrays are coerced to native floats / lists
    so downstream gates (workflow asserters, CSV joiners) can parse without
    importing torch. Callers that want the structured artifact for a
    pass/fail gate read from the returned path.

    :param metric_dict: Return value of :func:`evaluate`'s first tuple element.
    :param output_dir: Hydra per-run output dir; the ``metrics/`` subdir is
        created if missing.
    :returns: Absolute path to the written ``metrics.json``.
    """
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    serializable: dict[str, Any] = {}
    for key, value in metric_dict.items():
        if hasattr(value, "item") and not hasattr(value, "__len__"):
            serializable[key] = value.item()
        elif hasattr(value, "tolist"):
            serializable[key] = value.tolist()
        else:
            serializable[key] = value
    out_path = metrics_dir / "metrics.json"
    out_path.write_text(json.dumps(serializable, indent=2, sort_keys=True))
    return out_path


def _eval_summary_metrics(metric_dict: dict[str, Any]) -> dict[str, float]:
    """Reduce ``metric_dict`` to JSON-safe scalar floats for artifact metadata.

    Keeps ``artifact.metadata`` small per ``storage-provenance-spec.md`` §6:
    single-element tensors (``ndim == 0``) are coerced via ``.item()``; Python
    ``float`` / ``int`` pass through; vectors and other non-scalars are dropped.
    ``bool`` is excluded — it subclasses ``int`` but is not a metric value.

    :param metric_dict: :func:`evaluate`'s first tuple element (tensors + floats).
    :returns: ``{name: float}`` over the scalar entries only.
    """
    summary: dict[str, float] = {}
    for key, value in metric_dict.items():
        if hasattr(value, "item") and getattr(value, "ndim", 0) == 0:
            summary[key] = float(value.item())
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            summary[key] = float(value)
    return summary


def build_eval_results_artifact(
    cfg: DictConfig,
    output_dir_uri: str,
    metric_dict: dict[str, Any],
    git_sha: str,
) -> wandb.Artifact:
    """Build the canonical ``eval-results`` W&B artifact for an eval run.

    Names the artifact ``eval-{config_id}`` (type ``eval-results``) per
    ``storage-provenance-spec.md`` §4, where ``config_id`` is
    :func:`resolve_run_config_id`. References the R2 output prefix as an
    ``s3://`` URI and records the scalar summary metrics plus ``git_sha`` in
    ``artifact.metadata`` per §6. The reference uses ``checksum=False`` because
    R2's custom S3 endpoint is not reachable by W&B's default reference handler
    — the URI records lineage, not a content hash.

    :param cfg: Hydra-composed cfg; resolves the config_id for the name.
    :param output_dir_uri: The ``r2://`` prefix the eval output dir was mirrored
        to; rewritten to ``s3://`` for the reference.
    :param metric_dict: :func:`evaluate`'s first tuple element; reduced to a
        scalar summary for metadata.
    :param git_sha: Commit SHA recorded in metadata for lineage.
    :returns: An unlogged ``wandb.Artifact`` ready for ``log_artifact``.
    """
    artifact = wandb.Artifact(
        name=f"eval-{resolve_run_config_id(cfg)}",
        type="eval-results",
        metadata={**_eval_summary_metrics(metric_dict), "git_sha": git_sha},
    )
    artifact.add_reference(r2_io.to_s3_uri(output_dir_uri), checksum=False)
    return artifact


def _log_eval_results_artifact(
    loggers: list[Logger],
    cfg: DictConfig,
    output_dir_uri: str | None,
    metric_dict: dict[str, Any],
    git_sha: str,
) -> None:
    """Log the ``eval-results`` artifact to each ``WandbLogger`` in ``loggers``.

    Mirrors ``finalize_dataset._log_dataset_artifact``: a wandb failure warns
    and is swallowed so artifact logging never aborts a completed eval — the R2
    outputs are already mirrored. No-op when ``output_dir_uri`` is null (nothing
    to reference) or when no entry is a ``WandbLogger``.

    :param loggers: Lightning loggers; only ``WandbLogger`` entries log.
    :param cfg: Forwarded to :func:`build_eval_results_artifact` for the name.
    :param output_dir_uri: The ``r2://`` prefix the output dir was mirrored to,
        or null to skip logging.
    :param metric_dict: Forwarded to :func:`build_eval_results_artifact`.
    :param git_sha: Forwarded to :func:`build_eval_results_artifact`.
    """
    if not output_dir_uri:
        return
    for lg in loggers:
        if not isinstance(lg, WandbLogger):
            continue
        try:
            lg.experiment.log_artifact(
                build_eval_results_artifact(cfg, output_dir_uri, metric_dict, git_sha)
            )
        except Exception as exc:  # noqa: BLE001 — wandb artifact failure must not abort eval
            log.warning(f"_log_eval_results_artifact failed on {type(lg).__name__}: {exc}")


def _upload_output_dir_uri(cfg: DictConfig) -> str | None:
    """Resolve ``cfg.evaluation.upload_output_dir_uri``, tolerating an absent section.

    The R2 mirror is opt-in, so a missing ``evaluation`` group or unset key means
    "no upload" rather than a misconfiguration. ``OmegaConf.select`` returns
    ``None`` for any missing path segment without raising, unlike attribute access.

    :param cfg: Hydra-composed eval cfg.
    :returns: The configured destination URI, or ``None`` when unset/absent.
    """
    return OmegaConf.select(cfg, "evaluation.upload_output_dir_uri")


def _maybe_upload_output_dir(cfg: DictConfig, is_global_zero: bool) -> None:
    """Mirror the whole Hydra run dir to R2 when ``evaluation.upload_output_dir_uri`` is set.

    Opt-in: a null URI is a no-op. Runs last so every artifact — metrics,
    predictions, rendered audio, config logs — is on disk before the copy. The
    configured URI is the exact destination prefix; the run dir's contents land
    directly beneath it. Credential validation is delegated to
    :func:`r2_io.ensure_r2_env_loaded`, matching the datamodule's R2 prefetch.

    Only the global-zero rank uploads: under DDP ``main`` runs on every rank
    against the one shared ``output_dir``, so an ungated copy would race N
    redundant uploads — the same rank gate :func:`evaluate` puts on predict
    postprocessing.

    :param cfg: Reads ``cfg.evaluation.upload_output_dir_uri`` (``r2://`` prefix or
        null) and ``cfg.paths.output_dir`` (the local tree to copy).
    :param is_global_zero: Whether this is the global-zero rank; non-zero ranks
        return without touching R2.
    :raises ValueError: ``upload_output_dir_uri`` is set but not an ``r2://`` URI;
        checked before the credential ping so a misconfigured destination is
        attributed to the URI rather than surfacing as an auth failure.
    """
    if not is_global_zero:
        return
    dest_uri = _upload_output_dir_uri(cfg)
    if not dest_uri:
        return
    if not r2_io.is_r2_uri(dest_uri):
        raise ValueError(
            f"evaluation.upload_output_dir_uri must be an r2:// URI; got {dest_uri!r}."
        )
    output_dir = Path(cfg.paths.output_dir)
    log.info(f"Uploading eval output dir {output_dir} to {dest_uri}")
    r2_io.ensure_r2_env_loaded()
    r2_io.upload_dir(output_dir, dest_uri)


@hydra.main(version_base="1.3", config_path="pkg://synth_setter.configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Run the evaluation entrypoint.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # evaluate() persists metrics, mirrors the output dir to R2, and logs the
    # eval-results artifact internally (before @task_wrapper closes the run).
    evaluate(cfg)


if __name__ == "__main__":
    main()
