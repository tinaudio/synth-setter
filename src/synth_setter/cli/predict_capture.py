"""``synth-setter-predict-capture`` — single-capture sound-match inference (#1787).

Python half of the live sound-match bridge: a CLAP host plugin captures 4 s of
audio to ``capture-sample-dir/<uuid>.wav`` and spawns this CLI; we predict the
Surge patch that best matches the sound and write
``param-prediction-dir/<uuid>/params.csv`` (plus ``pred-0.pt`` as a debugging
aid). Values in ``params.csv`` are already in each parameter's native CLAP
domain per the committed per-spec map (:func:`synth_setter.resources.clap_map`).

Failure semantics: any error exits nonzero and ``params.csv`` is written via a
``.tmp`` + atomic rename, so its absence *is* the failure signal — the C++
side never sees a partial file.
"""

from __future__ import annotations

import csv
import logging
import math
import os
from collections.abc import Sequence
from pathlib import Path

import click
import torch

from synth_setter.data.audio_datamodule import AudioFolderDataset
from synth_setter.data.vst.clap_map import (
    ClapCsvRow,
    PluginFormatMap,
    load_clap_map,
    synth_params_to_clap_rows,
)
from synth_setter.data.vst.param_spec import ParamSpec, decode_model_output
from synth_setter.data.vst.param_spec_registry import param_specs
from synth_setter.models.surge_ff_module import VSTFeedForwardModule
from synth_setter.models.surge_flow_matching_module import VSTFlowMatchingModule
from synth_setter.resources import as_file, clap_map

# SET ME: deployment checkpoint — use an absolute path (this placeholder is
# repo-relative); the C++ bridge passes no --checkpoint (#1787).
_DEFAULT_CHECKPOINT = Path("checkpoints/sound-match-bridge.ckpt")

# SET ME: deployment log dir — use an absolute path (this placeholder is
# repo-relative); the C++ bridge passes no --log-dir (#1787).
_DEFAULT_LOG_DIR = Path("logs/sound-match-bridge")

_MODEL_CLASSES: dict[str, type[VSTFlowMatchingModule] | type[VSTFeedForwardModule]] = {
    "flow": VSTFlowMatchingModule,
    "ff": VSTFeedForwardModule,
}

_CSV_HEADER = ("pb_name", "clap_name", "clap_module_name", "clap_param_id", "clap_value")


def _open_run_logger(log_path: Path) -> logging.Logger:
    """Open a file-only logger for one bridge run, appending to ``log_path``.

    File-handler-only with ``propagate = False``: any stream handler would
    break repeated in-process CliRunner invocations under pytest's ``log_cli``
    (closed-stream errors), and the console already gets ``click.echo``.

    :param log_path: Destination ``<uuid>.log``; its parent must exist.
    :returns: Logger exclusive to this run; close via :func:`_close_run_logger`.
    """
    logger = logging.getLogger(f"synth_setter.cli.predict_capture.{log_path.stem}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def _close_run_logger(logger: logging.Logger) -> None:
    """Detach and close the run logger's handlers so retries never double-log.

    :param logger: Logger returned by :func:`_open_run_logger`.
    """
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def detect_model_class(checkpoint: Path) -> str:
    """Infer which ``_MODEL_CLASSES`` module produced a checkpoint.

    The two modules have disjoint child-module layouts (``net`` vs
    ``encoder``/``vector_field``), so the state-dict key prefixes identify the
    class without any config input — the C++ bridge passes no ``--model-class``.

    :param checkpoint: Lightning checkpoint file.
    :returns: ``_MODEL_CLASSES`` key (``"ff"`` or ``"flow"``).
    :raises ValueError: when the state dict matches neither module's layout.
    """
    # weights_only=False for parity with the load_from_checkpoint trust stance.
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    prefixes = {key.split(".", 1)[0] for key in state_dict}
    if "net" in prefixes and not {"encoder", "vector_field"} & prefixes:
        return "ff"
    if {"encoder", "vector_field"} <= prefixes and "net" not in prefixes:
        return "flow"
    raise ValueError(f"cannot infer model class from state-dict prefixes {sorted(prefixes)}")


def compute_capture_mel(wav_path: Path, stats_file: Path | None = None) -> torch.Tensor:
    """Compute the model-input mel for one capture via the training data path.

    Reuses :class:`AudioFolderDataset` (resample to 44 100 Hz, mono→stereo
    up-mix, pad/truncate to 4.0 s, ×0.5 amplitude, training mel) so the
    transform can never drift from the training contract.

    :param wav_path: Capture WAV; any sample rate/channel-count the dataset accepts.
    :param stats_file: Optional ``.npz`` with the training run's saved mel
        ``mean``/``std``; required whenever the served checkpoint trained with
        ``use_saved_mean_and_variance`` so serve-time input matches training.
    :returns: Mel spectrogram of shape ``(2, 128, frames)``.
    """
    dataset = AudioFolderDataset(
        root=str(wav_path.parent),
        reference_stats_file=None if stats_file is None else str(stats_file),
        files=[wav_path],
    )
    return dataset[0]["mel_spec"]


def decode_and_convert(  # noqa: DOC502 — ValueError propagates from synth_params_to_clap_rows
    prediction: torch.Tensor,
    spec: ParamSpec,
    format_map: PluginFormatMap,
) -> list[ClapCsvRow]:
    """Decode a raw prediction row and convert it to native-domain CLAP rows.

    Applies the model-output transform ``(x + 1) / 2`` then clips to ``[0, 1]``
    (the inverse scale documented in ``predict_vst_audio``), decodes via the
    spec, and discards ``note_params`` — the bridge only applies synth params.

    :param prediction: Tensor of shape ``(1, len(spec))`` with values in ``[-1, 1]``.
    :param spec: Spec the model was trained against.
    :param format_map: Committed pyname → CLAP identity map.
    :returns: One row per decoded synth parameter.
    :raises ValueError: when any decoded param is missing from ``format_map``.
    """
    row = prediction[0].detach().cpu().float().numpy()
    synth_params, _ = decode_model_output(row, spec)
    return synth_params_to_clap_rows(synth_params, spec, format_map)


# DOC503: the bare re-raise after .tmp cleanup is not a new exception type.
def write_params_csv(rows: Sequence[ClapCsvRow], dest: Path) -> None:  # noqa: DOC503
    """Write the bridge CSV atomically (``.tmp`` then rename within one filesystem).

    :param rows: Converted rows; ``clap_value`` is formatted with ``%.9g``
        (float32-faithful, no trailing noise digits).
    :param dest: Final ``params.csv`` path; the ``.tmp`` sibling is transient.
    :raises ValueError: when any ``clap_value`` is NaN/Inf — a literal "nan" in
        the CSV would poison the live CLAP host, so it must fail loudly instead.
    """
    non_finite = [row.pb_name for row in rows if not math.isfinite(row.clap_value)]
    if non_finite:
        raise ValueError(f"non-finite clap_value for: {', '.join(non_finite)}")

    tmp_path = dest.with_suffix(".csv.tmp")
    try:
        with tmp_path.open("w", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(_CSV_HEADER)
            for row in rows:
                writer.writerow(
                    (
                        row.pb_name,
                        row.clap_name,
                        row.clap_module_name,
                        row.clap_param_id,
                        f"{row.clap_value:.9g}",
                    )
                )
        os.replace(tmp_path, dest)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _predict_raw_params(
    mel_spec: torch.Tensor, model: VSTFlowMatchingModule | VSTFeedForwardModule
) -> torch.Tensor:
    """Run one mel through the module's real predict path.

    :param mel_spec: Mel of shape ``(2, 128, frames)``.
    :param model: Loaded module in eval mode.
    :returns: Raw prediction tensor of shape ``(1, num_params)``, values in ``[-1, 1]``.
    """
    batch = {"mel_spec": mel_spec.unsqueeze(0).to(model.device)}
    with torch.no_grad():
        # The ff module annotates batch as a tuple but reads dict keys; both
        # modules consume {'mel_spec': ...} at runtime.
        prediction, _ = model.predict_step(batch, 0)  # pyright: ignore[reportArgumentType]
    return prediction.detach().cpu()


@click.command()
@click.argument("wav_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--prediction-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Bridge param-prediction-dir; the <uuid>/ subdir is created here.",
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_CHECKPOINT,
    show_default=True,
    help="Lightning checkpoint to load.",
)
@click.option(
    "--map",
    "map_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="CLAP param map JSON [default: the packaged map matching --param-spec-name].",
)
@click.option(
    "--model-class",
    type=click.Choice(sorted(_MODEL_CLASSES)),
    default=None,
    help="LightningModule the checkpoint was trained with "
    "[default: detected from the checkpoint's state dict].",
)
@click.option(
    "--stats-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Mel mean/std .npz from training; set when the checkpoint trained normalized.",
)
@click.option("--param-spec-name", default="surge_xt", show_default=True)
@click.option("--device", default="cpu", show_default=True)
@click.option(
    "--log-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=_DEFAULT_LOG_DIR,
    show_default=True,
    help="Run logs land here as <uuid>.log (appended across retries).",
)
# DOC501/DOC503: the bare re-raise after logging is not a new exception type.
def main(  # noqa: DOC501, DOC503
    wav_path: Path,
    prediction_dir: Path,
    checkpoint: Path,
    map_path: Path | None,
    stats_file: Path | None,
    model_class: str | None,
    param_spec_name: str,
    device: str,
    log_dir: Path,
) -> None:
    """Predict Surge params for one capture WAV and write the bridge CSV.

    Every run — including any crash — is recorded in ``<log-dir>/<uuid>.log``;
    the console mirror stays on stderr via ``click.echo``.

    :param wav_path: Capture file; its stem is the bridge uuid.
    :param prediction_dir: Where the ``<uuid>/`` output dir is created.
    :param checkpoint: Checkpoint file to run.
    :param map_path: Map override; ``None`` resolves the packaged map.
    :param stats_file: Saved mel stats to normalize with; ``None`` skips normalization.
    :param model_class: ``_MODEL_CLASSES`` key selecting the module class;
        ``None`` detects it from the checkpoint's state dict.
    :param param_spec_name: ``param_specs`` registry key.
    :param device: torch device for inference.
    :param log_dir: Directory receiving the per-uuid run log.
    """
    capture_uuid = wav_path.stem
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = _open_run_logger(log_dir / f"{capture_uuid}.log")
    try:
        logger.info(
            "predict_capture start: wav=%s checkpoint=%s spec=%s device=%s map=%s stats=%s",
            wav_path,
            checkpoint,
            param_spec_name,
            device,
            map_path or "packaged",
            stats_file or "none",
        )
        _run(
            wav_path=wav_path,
            prediction_dir=prediction_dir,
            checkpoint=checkpoint,
            map_path=map_path,
            stats_file=stats_file,
            model_class=model_class,
            param_spec_name=param_spec_name,
            device=device,
            logger=logger,
        )
    except Exception:
        # Absence of params.csv stays the bridge's failure signal; the log
        # carries the reason so a crashed spawn is diagnosable after the fact.
        logger.exception("predict_capture failed")
        raise
    finally:
        _close_run_logger(logger)


def _say(logger: logging.Logger, message: str) -> None:
    """Mirror one milestone to the run log and the stderr console.

    :param logger: The per-run file logger.
    :param message: Milestone text.
    """
    logger.info("%s", message)
    click.echo(message, err=True)


def _run(
    wav_path: Path,
    prediction_dir: Path,
    checkpoint: Path,
    map_path: Path | None,
    stats_file: Path | None,
    model_class: str | None,
    param_spec_name: str,
    device: str,
    logger: logging.Logger,
) -> None:
    """Execute one bridge prediction under an open run logger.

    :param wav_path: Capture file; its stem is the bridge uuid.
    :param prediction_dir: Where the ``<uuid>/`` output dir is created.
    :param checkpoint: Checkpoint file to run.
    :param map_path: Map override; ``None`` resolves the packaged map.
    :param stats_file: Saved mel stats to normalize with; ``None`` skips normalization.
    :param model_class: ``_MODEL_CLASSES`` key; ``None`` detects from the checkpoint.
    :param param_spec_name: ``param_specs`` registry key.
    :param device: torch device for inference.
    :param logger: Per-run file logger from :func:`_open_run_logger`.
    """
    capture_uuid = wav_path.stem
    uuid_dir = prediction_dir / capture_uuid
    # A retried uuid must never expose the previous run's result when this run
    # fails — absence of params.csv IS the failure signal.
    (uuid_dir / "params.csv").unlink(missing_ok=True)

    if map_path is not None:
        format_map = load_clap_map(map_path)
    else:
        with as_file(clap_map(param_spec_name)) as packaged:
            format_map = load_clap_map(packaged)
    spec = param_specs[param_spec_name]

    if model_class is None:
        model_class = detect_model_class(checkpoint)
        _say(logger, f"detected model class {model_class} from the checkpoint")
    _say(logger, f"loading {model_class} checkpoint {checkpoint}")
    # weights_only=False unpickles the module graph; checkpoint provenance is
    # deployment-controlled (same trust stance as train/eval).
    model = _MODEL_CLASSES[model_class].load_from_checkpoint(
        checkpoint, map_location=device, weights_only=False
    )
    # map_location only remaps storages; move the module so model.device (and
    # the batch _predict_raw_params sends) actually follow --device.
    model.to(device)
    model.eval()

    if stats_file is None:
        # Serving a stats-normalized checkpoint without --stats-file is silent
        # train/serve skew, so the omission is at least loud in the log.
        _say(logger, "warning: no --stats-file — mel is unnormalized")
    mel_spec = compute_capture_mel(wav_path, stats_file)
    logger.info("mel computed: shape=%s", tuple(mel_spec.shape))

    prediction = _predict_raw_params(mel_spec, model)
    uuid_dir.mkdir(parents=True, exist_ok=True)
    torch.save(prediction, uuid_dir / "pred-0.pt")
    logger.info("saved raw prediction: %s", uuid_dir / "pred-0.pt")

    rows = decode_and_convert(prediction, spec, format_map)
    write_params_csv(rows, uuid_dir / "params.csv")
    _say(
        logger,
        f"wrote {len(rows)} params to {uuid_dir / 'params.csv'} "
        f"(checkpoint={checkpoint} map={map_path or f'packaged {param_spec_name}_clap_map.json'} "
        f"spec={param_spec_name})",
    )


if __name__ == "__main__":
    main()
