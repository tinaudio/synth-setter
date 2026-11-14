"""``synth-setter-predict-capture`` — single-capture sound-match inference (#1787).

Python half of the live sound-match bridge: a CLAP host plugin captures 4 s of
audio to ``capture-sample-dir/<uuid>.wav`` and spawns this CLI; an operator may
instead supply a Surge FXP target. We predict the synth patch that best matches
the sound and write ``param-prediction-dir/<input-stem>/params.csv`` (plus
``pred-0.pt`` as a debugging aid). Values in ``params.csv`` are already in each parameter's native CLAP
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
import numpy as np
import torch

from synth_setter.cli.clap_render import resolve_inverse_checkpoint
from synth_setter.data.audio_datamodule import AudioFolderDataset
from synth_setter.data.vst.clap_map import (
    ClapCsvRow,
    PluginFormatMap,
    load_clap_map,
    synth_params_to_clap_rows,
)
from synth_setter.data.vst.core import write_wav
from synth_setter.data.vst.param_map import SynthParamMap, load_param_map
from synth_setter.data.vst.param_spec import (
    ParamSpec,
    decode_model_output,
    require_scalar_synth_params,
)
from synth_setter.data.vst.param_spec_registry import param_specs
from synth_setter.data.vst.renderers import SurgePyRenderer
from synth_setter.data.vst.surgepy_runtime import import_surgepy
from synth_setter.models.vst_ff_module import VSTFeedForwardModule
from synth_setter.models.vst_flow_matching_module import VSTFlowMatchingModule
from synth_setter.pipeline import r2_io
from synth_setter.resources import as_file, param_map
from synth_setter.synth_spec import SynthName, resolve_synth
from synth_setter.workspace import operator_workspace

# SET ME: deployment checkpoint — use an absolute path (this placeholder is
# repo-relative); the C++ bridge passes no --checkpoint (#1787).
_DEFAULT_CHECKPOINT = Path("checkpoints/sound-match-bridge.ckpt")

# SET ME: deployment log dir — use an absolute path (this placeholder is
# repo-relative); the C++ bridge passes no --log-dir (#1787).
_DEFAULT_LOG_DIR = Path("logs/sound-match-bridge")

# SET ME (optional): pin to "flow"/"ff" to skip the detection load of the
# checkpoint — the C++ bridge passes no --model-class (#1787).
_DEFAULT_MODEL_CLASS: str | None = None

# Flow sampling draws noise; a fixed seed keeps serving and retries reproducible.
_SERVING_SEED = 0
_FXP_CHANNELS = 2
_FXP_MIDI_NOTE = 60
_FXP_NOTE_SECONDS = 2.0
_FXP_SAMPLE_RATE = 44_100
_FXP_SIGNAL_SECONDS = 4.0
_FXP_VELOCITY = 100

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
    handler = logging.FileHandler(log_path, encoding="utf-8")
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


def render_fxp_target(fxp_path: Path) -> np.ndarray:
    """Render an arbitrary Surge FXP as the fixed prediction target.

    The note is MIDI 60 at velocity 100 for two seconds, followed by a two-second release tail at
    44.1 kHz stereo.

    :param fxp_path: Existing Surge FXP patch.
    :returns: Channel-leading float32 audio with exactly 176,400 samples.
    :raises RuntimeError: If Surge cannot load the patch.
    """
    surgepy = import_surgepy()
    synth = surgepy.createSurge(_FXP_SAMPLE_RATE)
    if not synth.loadPatch(str(fxp_path.resolve())):
        raise RuntimeError(f"SurgePy could not load patch {fxp_path}")

    sample_count = int(_FXP_SAMPLE_RATE * _FXP_SIGNAL_SECONDS)
    block_size = synth.getBlockSize()
    block_count = math.ceil(sample_count / block_size)
    note_blocks = math.ceil(_FXP_SAMPLE_RATE * _FXP_NOTE_SECONDS / block_size)
    audio = synth.createMultiBlock(block_count)
    try:
        synth.playNote(0, _FXP_MIDI_NOTE, _FXP_VELOCITY)
        synth.processMultiBlock(audio, 0, note_blocks)
        synth.releaseNote(0, _FXP_MIDI_NOTE)
        synth.processMultiBlock(audio, note_blocks, block_count - note_blocks)
    finally:
        synth.allNotesOff()
    return np.asarray(audio[:, :sample_count], dtype=np.float32)


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
    return dataset[0]["mel"]


def _decode_synth_params(prediction: torch.Tensor, spec: ParamSpec) -> dict[str, float]:
    """Decode one raw model output row into scalar renderer parameters.

    :param prediction: Tensor of shape ``(1, len(spec))`` in the model-output domain.
    :param spec: Spec the model was trained against.
    :returns: Renderer-native scalar synth parameters; note parameters are discarded.
    """
    row = prediction[0].detach().cpu().float().numpy()
    synth_values, _ = decode_model_output(row, spec)
    return require_scalar_synth_params(synth_values)


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
    return synth_params_to_clap_rows(_decode_synth_params(prediction, spec), spec, format_map)


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
        with tmp_path.open("w", newline="", encoding="utf-8") as f:
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
    mel: torch.Tensor, model: VSTFlowMatchingModule | VSTFeedForwardModule
) -> torch.Tensor:
    """Run one mel through the module's real predict path.

    :param mel: Mel of shape ``(2, 128, frames)``.
    :param model: Loaded module in eval mode.
    :returns: Raw prediction tensor of shape ``(1, num_params)``, values in ``[-1, 1]``.
    """
    batch = {"mel": mel.unsqueeze(0).to(model.device)}
    with torch.no_grad():
        # The ff module annotates batch as a tuple but reads dict keys; both
        # modules consume {'mel': ...} at runtime.
        prediction, _ = model.predict_step(batch, 0)  # pyright: ignore[reportArgumentType]
    return prediction.detach().cpu()


@click.command()
@click.argument(
    "wav_path",
    required=False,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--fxp",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Render this Surge FXP as the input target (MIDI 60, velocity 100, 2 s note + 2 s tail).",
)
@click.option(
    "--prediction-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Bridge param-prediction-dir; the <input-stem>/ subdir is created here.",
)
@click.option(
    "--checkpoint",
    type=str,
    default=str(_DEFAULT_CHECKPOINT),
    show_default=True,
    help="Local path or r2:// URI of the Lightning checkpoint to load.",
)
@click.option(
    "--checkpoint-sha256",
    default=None,
    help="Optional SHA-256 required for the local or downloaded checkpoint.",
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
    default=_DEFAULT_MODEL_CLASS,
    help="LightningModule the checkpoint was trained with "
    "[default: the deployment constant, else detected from the checkpoint's state dict].",
)
@click.option(
    "--stats-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Mel mean/std .npz from training; set when the checkpoint trained normalized.",
)
@click.option("--param-spec-name", default="surge_xt", show_default=True)
@click.option(
    "--render-audio",
    is_flag=True,
    help="Render predicted parameters to pred.wav with the native Surge engine.",
)
@click.option("--device", default="cpu", show_default=True)
@click.option(
    "--log-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=_DEFAULT_LOG_DIR,
    show_default=True,
    help="Run logs land here as <input-stem>.log (appended across retries).",
)
# DOC501/DOC503: the bare re-raise after logging is not a new exception type.
def main(  # noqa: DOC501, DOC503
    wav_path: Path | None,
    fxp: Path | None,
    prediction_dir: Path,
    checkpoint: str,
    checkpoint_sha256: str | None,
    map_path: Path | None,
    stats_file: Path | None,
    model_class: str | None,
    param_spec_name: str,
    render_audio: bool,
    device: str,
    log_dir: Path,
) -> None:
    """Predict synth parameters for one WAV or rendered FXP target.

    Every run — including any crash — is recorded in
    ``<log-dir>/<input-stem>.log``; the console mirror stays on stderr.

    :param wav_path: Capture file, mutually exclusive with ``fxp``.
    :param fxp: Surge patch rendered as the target, mutually exclusive with ``wav_path``.
    :param prediction_dir: Where the ``<input-stem>/`` output dir is created.
    :param checkpoint: Local checkpoint path or R2 object URI.
    :param checkpoint_sha256: Optional required checkpoint digest.
    :param map_path: Map override; ``None`` resolves the packaged map.
    :param stats_file: Saved mel stats to normalize with; ``None`` skips normalization.
    :param model_class: ``_MODEL_CLASSES`` key selecting the module class;
        ``None`` detects it from the checkpoint's state dict.
    :param param_spec_name: ``param_specs`` registry key.
    :param render_audio: Whether to render the prediction through native SurgePy.
    :param device: torch device for inference.
    :param log_dir: Directory receiving the per-input run log.
    """
    if (wav_path is None) == (fxp is None):
        raise click.UsageError("exactly one of WAV_PATH or --fxp is required")
    input_path = wav_path if wav_path is not None else fxp
    assert input_path is not None
    input_stem = input_path.stem
    output_dir = prediction_dir / input_stem
    (output_dir / "params.csv").unlink(missing_ok=True)
    (output_dir / "pred.wav").unlink(missing_ok=True)

    log_dir.mkdir(parents=True, exist_ok=True)
    logger = _open_run_logger(log_dir / f"{input_stem}.log")
    try:
        logger.info(
            "predict_capture start: target=%s checkpoint=%s spec=%s device=%s map=%s stats=%s",
            input_path,
            checkpoint,
            param_spec_name,
            device,
            map_path or "packaged",
            stats_file or "none",
        )
        if r2_io.is_r2_uri(checkpoint):
            r2_io.ensure_r2_env_loaded()
        resolved_checkpoint = resolve_inverse_checkpoint(checkpoint, checkpoint_sha256)
        target_wav = wav_path
        if fxp is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            target_wav = output_dir / "target.wav"
            write_wav(
                render_fxp_target(fxp),
                str(target_wav),
                _FXP_SAMPLE_RATE,
                _FXP_CHANNELS,
            )
            _say(logger, f"rendered FXP target to {target_wav}")
        assert target_wav is not None
        _run(
            wav_path=target_wav,
            output_dir=output_dir,
            checkpoint=resolved_checkpoint,
            map_path=map_path,
            stats_file=stats_file,
            model_class=model_class,
            param_spec_name=param_spec_name,
            render_audio=render_audio,
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


def _prediction_renderer(param_spec_name: str, joint_map: SynthParamMap) -> SurgePyRenderer:
    """Build the registered native renderer for a selected parameter spec.

    :param param_spec_name: Selected ``param_specs`` registry key.
    :param joint_map: Packaged joint map matching the selected spec.
    :returns: Renderer pinned to the map's registered FXP baseline.
    :raises ValueError: If the spec or map cannot drive native SurgePy rendering.
    """
    try:
        synth = resolve_synth(SynthName(f"{param_spec_name}_surgepy"))
    except KeyError:
        raise ValueError(
            f"parameter spec {param_spec_name!r} does not support SurgePy prediction rendering"
        ) from None
    if joint_map.surgepy is None or joint_map.surgepy_preset_resource is None:
        raise ValueError(
            f"parameter spec {param_spec_name!r} does not support SurgePy prediction rendering"
        )
    if (
        str(joint_map.param_spec_name) != param_spec_name
        or str(synth.param_spec_name) != param_spec_name
        or synth.plugin_state_path != joint_map.surgepy_preset_resource
    ):
        raise ValueError(f"SurgePy registry and parameter map disagree for {param_spec_name!r}")
    return SurgePyRenderer(
        plugin_path=synth.plugin_path,
        sample_rate=_FXP_SAMPLE_RATE,
        channels=_FXP_CHANNELS,
        signal_duration_seconds=_FXP_SIGNAL_SECONDS,
        plugin_state_path=str(operator_workspace() / synth.plugin_state_path),
        parameter_map=joint_map,
    )


def _run(
    wav_path: Path,
    output_dir: Path,
    checkpoint: Path,
    map_path: Path | None,
    stats_file: Path | None,
    model_class: str | None,
    param_spec_name: str,
    render_audio: bool,
    device: str,
    logger: logging.Logger,
) -> None:
    """Execute one bridge prediction under an open run logger.

    :param wav_path: Capture WAV or persisted FXP target render.
    :param output_dir: Directory receiving prediction artifacts.
    :param checkpoint: Checkpoint file to run.
    :param map_path: Map override; ``None`` resolves the packaged map.
    :param stats_file: Saved mel stats to normalize with; ``None`` skips normalization.
    :param model_class: ``_MODEL_CLASSES`` key; ``None`` detects from the checkpoint.
    :param param_spec_name: ``param_specs`` registry key.
    :param render_audio: Whether to render the prediction through native SurgePy.
    :param device: torch device for inference.
    :param logger: Per-run file logger from :func:`_open_run_logger`.
    """
    joint_map: SynthParamMap | None = None
    renderer: SurgePyRenderer | None = None
    if render_audio:
        with as_file(param_map(param_spec_name)) as packaged:
            joint_map = load_param_map(packaged)
        renderer = _prediction_renderer(param_spec_name, joint_map)

    if map_path is not None:
        format_map = load_clap_map(map_path)
    else:
        if joint_map is None:
            with as_file(param_map(param_spec_name)) as packaged:
                joint_map = load_param_map(packaged)
        format_map = joint_map.clap_projection()
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
    mel = compute_capture_mel(wav_path, stats_file)
    logger.info("mel computed: shape=%s", tuple(mel.shape))

    # The flow module's predict_step samples noise; without this the same
    # capture yields a different patch on every spawn.
    torch.manual_seed(_SERVING_SEED)
    prediction = _predict_raw_params(mel, model)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(prediction, output_dir / "pred-0.pt")
    logger.info("saved raw prediction: %s", output_dir / "pred-0.pt")

    synth_params = _decode_synth_params(prediction, spec)
    rows = synth_params_to_clap_rows(synth_params, spec, format_map)
    if renderer is not None:
        audio = renderer.render(
            synth_params,
            midi_note=_FXP_MIDI_NOTE,
            velocity=_FXP_VELOCITY,
            note_start_and_end=(0.0, _FXP_NOTE_SECONDS),
        )
        write_wav(audio, str(output_dir / "pred.wav"), _FXP_SAMPLE_RATE, _FXP_CHANNELS)
        _say(logger, f"rendered predicted audio to {output_dir / 'pred.wav'}")
    write_params_csv(rows, output_dir / "params.csv")
    _say(
        logger,
        f"wrote {len(rows)} params to {output_dir / 'params.csv'} "
        f"(checkpoint={checkpoint} map={map_path or f'packaged {param_spec_name}_param_map.json'} "
        f"spec={param_spec_name})",
    )


if __name__ == "__main__":
    main()
