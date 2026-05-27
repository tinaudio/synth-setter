"""Hydra entrypoint for evaluating a trained model on a datamodule's test split."""

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
from omegaconf import DictConfig

from synth_setter.resources import as_file, vst_headless_wrapper
from synth_setter.utils import (
    RankedLogger,
    extras,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    log_wandb_provenance,
    register_resolvers,
    task_wrapper,
)
from synth_setter.workspace import operator_workspace

_PREDICT_VST_AUDIO_MODULE = "synth_setter.evaluation.predict_vst_audio"
_COMPUTE_AUDIO_METRICS_MODULE = "synth_setter.evaluation.compute_audio_metrics"
_SUBPROCESS_TIMEOUT_SECONDS = 600
_AGGREGATED_METRICS_FILENAME = "aggregated_metrics.csv"
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
    return {
        f"audio/{metric}_{stat}": float(df.at[metric, stat])
        for metric in df.index
        for stat in _AGGREGATED_METRICS_STATS
    }


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
    metrics subprocess is CPU-only and needs no wrapper.

    :param cfg: Reads ``cfg.evaluation`` (gates + ``num_workers``), ``cfg.render``
        (param spec, preset, optional plugin path), and ``cfg.paths.output_dir``
        (base for ``predictions/``, ``audio/``, ``metrics/``).
    :returns: ``{"audio/<name>_<stat>": value}`` when ``compute_metrics`` ran;
        empty dict otherwise. Always rank-zero — the caller gates DDP duplication.
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
                "(e.g. `+render=surge_xt`); cfg.render is unset."
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
        _log_audio_metrics_to_wandb(audio_metrics)
        return audio_metrics

    return {}


@task_wrapper
def evaluate(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate the given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: ``(metric_dict, object_dict)``. ``metric_dict`` is the
        ``trainer.callback_metrics`` copy merged with any audio metrics from
        :func:`_run_predict_postprocessing`; Lightning entries are ``torch.Tensor``
        while audio entries are Python ``float``, so callers iterating values
        must handle both.
    """
    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
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

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="pkg://synth_setter.configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Run the evaluation entrypoint.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
