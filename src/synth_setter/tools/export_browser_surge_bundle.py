"""Export the music-sketch Surge flow as a self-contained browser bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from synth_setter.cli.clap_render import resolve_inverse_checkpoint
from synth_setter.cli.sketch_render import (
    _load_settings,
    _producer_revision,
    _resolve_stats,
    load_render_config,
)
from synth_setter.conditioning import SketchControlSpec, resolve_sketch_controls
from synth_setter.data.vst.param_map import SurgePyParamRef, SynthParamMap, load_param_map
from synth_setter.data.vst.param_spec import (
    CategoricalParameter,
    ContinuousParameter,
    DiscreteLiteralParameter,
    NoteDurationParameter,
    Parameter,
    ParamSpec,
)
from synth_setter.data.vst.param_spec_registry import resolve_param_spec
from synth_setter.models.flow_onnx import export_flow_onnx
from synth_setter.models.music_frontend import (
    MusicSketchFrontend,
    StereoMelFrontend,
    export_music_sketch_onnx,
    export_stereo_mel_onnx,
)
from synth_setter.models.vst_flow_matching_module import VSTFlowMatchingModule
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.resources import as_file, param_map

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1
_SAMPLE_RATE = 44_100
_SAMPLE_FRAMES = 176_400
_CHANNELS = 2
_PARAM_SPEC_NAME = "surge_simple"
_ENCODED_WIDTH = 92
_NUM_MUSIC_CONTROLS = 386
_GRAPH_FILES = (
    "conditioning.onnx",
    "frontend.onnx",
    "preset.fxp",
    "sketch.onnx",
    "velocity.onnx",
)
_NATIVE_REFERENCE_VERSION = "1.3.4"
_WASM_TARGET_VERSION = "1.4.0"


def _sha256(path: Path) -> str:
    """Return one file's lowercase SHA-256.

    :param path: File to hash.
    :returns: Hex digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parameter_descriptor(
    parameter: Parameter,
    span: slice,
    target: str,
    identities: dict[str, SurgePyParamRef],
) -> dict[str, object]:
    """Describe one encoded field without duplicating decoder logic in JavaScript.

    :param parameter: Registered logical parameter.
    :param span: Columns occupied in the model row.
    :param target: ``synth`` or ``note`` destination.
    :param identities: Surge native identities keyed by synth parameter name.
    :returns: JSON-compatible decoder descriptor.
    :raises TypeError: The registered parameter type has no browser decoder contract.
    """
    descriptor: dict[str, object] = {
        "name": parameter.name,
        "offset": span.start,
        "width": span.stop - span.start,
        "target": target,
    }
    if target == "synth":
        identity = identities[parameter.name]
        descriptor.update({"id": identity.synth_side_id, "nativeName": identity.name})
    if isinstance(parameter, ContinuousParameter):
        descriptor.update({"kind": "continuous", "min": parameter.min, "max": parameter.max})
    elif isinstance(parameter, CategoricalParameter):
        if parameter.encoding == "scalar":
            descriptor.update({"kind": "continuous", "min": 0.0, "max": 1.0})
        else:
            descriptor.update({"kind": "onehot", "values": list(parameter.raw_values)})
    elif isinstance(parameter, DiscreteLiteralParameter):
        if parameter.encoding == "scalar":
            descriptor.update({"kind": "integer", "min": parameter.min, "max": parameter.max})
        else:
            descriptor.update(
                {"kind": "onehot", "values": list(range(parameter.min, parameter.max + 1))}
            )
    elif isinstance(parameter, NoteDurationParameter):
        descriptor.update({"kind": "window", "max": parameter.max_note_duration_seconds})
    else:
        raise TypeError(f"unsupported browser decoder parameter {type(parameter).__name__}")
    return descriptor


def _decoder_parameters(spec: ParamSpec, parameter_map: SynthParamMap) -> list[dict[str, object]]:
    """Derive ordered model-column descriptors from the registered spec and Surge map.

    :param spec: Registered model encoding contract.
    :param parameter_map: Proven native Surge identities.
    :returns: One descriptor per logical parameter in encoded order.
    """
    identities = parameter_map.surgepy_params()
    synth_count = len(spec.synth_params)
    return [
        _parameter_descriptor(
            parameter, span, "synth" if index < synth_count else "note", identities
        )
        for index, (parameter, span) in enumerate(spec.encoded_slices())
    ]


def _validate_model(model: VSTFlowMatchingModule) -> SketchControlSpec:
    """Validate the fixed browser Surge flow contract.

    :param model: Candidate CPU evaluation checkpoint.
    :returns: Resolved music sketch configuration.
    :raises ValueError: The checkpoint is incompatible with the browser app.
    """
    sketch = resolve_sketch_controls(model.hparams["sketch_controls"])
    if model.hparams.get("param_spec") != _PARAM_SPEC_NAME:
        raise ValueError("browser Surge bundles require the surge_simple parameter spec")
    if model.hparams.get("conditioning") != "mel":
        raise ValueError("browser Surge bundles require mel content conditioning")
    if model.hparams.get("num_params") != _ENCODED_WIDTH:
        raise ValueError("browser Surge bundles require 92 model outputs")
    if (
        sketch is None
        or sketch.profile != "music"
        or sketch.layout.num_controls != _NUM_MUSIC_CONTROLS
        or sketch.num_frames != 32
    ):
        raise ValueError("browser Surge bundles require music sketch controls shaped (386, 32)")
    return sketch


def export_browser_surge_bundle(
    model: VSTFlowMatchingModule,
    stats_path: Path,
    output: Path,
    *,
    checkpoint_sha256: str,
    git_revision: str,
) -> dict[str, object]:
    """Atomically publish browser front ends, flow graphs, preset, and manifest.

    :param model: CPU evaluation checkpoint on the Surge music-sketch contract.
    :param stats_path: Two-channel training mel statistics.
    :param output: Absent destination directory.
    :param checkpoint_sha256: Verified source checkpoint digest.
    :param git_revision: Producing source revision.
    :returns: Written manifest.
    :raises FileExistsError: The destination exists.
    :raises RuntimeError: Preset provenance differs from the committed Surge map.
    :raises ValueError: The checkpoint, render grid, or statistics are incompatible.
    """
    if output.exists():
        raise FileExistsError(f"bundle destination already exists: {output}")
    sketch = _validate_model(model)
    render = load_render_config()
    if (
        render.param_spec_name != _PARAM_SPEC_NAME
        or render.sample_rate != _SAMPLE_RATE
        or render.channels != _CHANNELS
        or int(render.sample_rate * render.signal_duration_seconds) != _SAMPLE_FRAMES
    ):
        raise ValueError("sketch_render must select the fixed stereo Surge browser grid")
    with as_file(param_map(render.param_spec_name)) as path:
        parameter_map = load_param_map(path)
    preset = Path(render.plugin_state_path)
    if _sha256(preset) != parameter_map.surgepy_preset_sha256:
        raise RuntimeError("Surge preset SHA-256 does not match the parameter map")
    with np.load(stats_path) as stats:
        mel_frontend = StereoMelFrontend(
            sample_rate=_SAMPLE_RATE, mean=stats["mean"], std=stats["std"]
        ).eval()
    sketch_frontend = MusicSketchFrontend(
        sample_rate=_SAMPLE_RATE,
        output_frames=sketch.num_frames,
        pitch_zero_threshold=sketch.pitch_zero_threshold,
    ).eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}."))
    try:
        manifest = _write_bundle(
            model,
            mel_frontend,
            sketch_frontend,
            sketch,
            resolve_param_spec(ParamSpecName(_PARAM_SPEC_NAME)),
            parameter_map,
            preset,
            stats_path,
            staging,
            checkpoint_sha256,
            git_revision,
        )
        os.rename(staging, output)
    finally:
        if staging.exists():
            try:
                shutil.rmtree(staging)
            except OSError:
                logger.warning("failed to remove bundle staging directory %s", staging)
    return manifest


def _write_bundle(
    model: VSTFlowMatchingModule,
    mel_frontend: StereoMelFrontend,
    sketch_frontend: MusicSketchFrontend,
    sketch: SketchControlSpec,
    spec: ParamSpec,
    parameter_map: SynthParamMap,
    preset: Path,
    stats_path: Path,
    staging: Path,
    checkpoint_sha256: str,
    git_revision: str,
) -> dict[str, object]:
    """Populate an unpublished staging directory with the complete bundle.

    :param model: Validated flow model.
    :param mel_frontend: Statistics-bound stereo mel graph.
    :param sketch_frontend: PESTO music-sketch graph.
    :param sketch: Checkpoint sketch contract.
    :param spec: Registered output decoder contract.
    :param parameter_map: Native Surge identities.
    :param preset: Proven baseline patch.
    :param stats_path: Source statistics archive.
    :param staging: Empty unpublished directory.
    :param checkpoint_sha256: Verified checkpoint digest.
    :param git_revision: Producing source revision.
    :returns: Manifest written beside the artifacts.
    """
    waveform = torch.zeros(1, _CHANNELS, _SAMPLE_FRAMES)
    with torch.no_grad():
        batch = {"mel": mel_frontend(waveform), "sketch_ctrl": sketch_frontend(waveform)}
    export_flow_onnx(model, batch, staging / "flow")
    for name in ("conditioning.onnx", "velocity.onnx"):
        (staging / "flow" / name).rename(staging / name)
    (staging / "flow").rmdir()
    export_stereo_mel_onnx(mel_frontend, staging / "frontend.onnx")
    export_music_sketch_onnx(sketch_frontend, staging / "sketch.onnx")
    shutil.copyfile(preset, staging / "preset.fxp")
    manifest: dict[str, Any] = {
        "schemaVersion": _SCHEMA_VERSION,
        "paramSpecName": _PARAM_SPEC_NAME,
        "encodedWidth": _ENCODED_WIDTH,
        "parameters": _decoder_parameters(spec, parameter_map),
        "sketch": {
            "profile": sketch.profile,
            "numControls": sketch.layout.num_controls,
            "numFrames": sketch.num_frames,
            "pitchZeroThreshold": sketch.pitch_zero_threshold,
        },
        "sampleRate": _SAMPLE_RATE,
        "frames": _SAMPLE_FRAMES,
        "channels": _CHANNELS,
        "sampling": {
            "steps": model.hparams["test_sample_steps"],
            "contentCfg": model.hparams["test_cfg_strength"],
            "sketchCfg": model.hparams["test_sketch_cfg_strength"],
        },
        "surge": {
            "nativeReferenceVersion": _NATIVE_REFERENCE_VERSION,
            "wasmTargetVersion": _WASM_TARGET_VERSION,
        },
        "files": {
            name: {"sha256": _sha256(staging / name), "bytes": (staging / name).stat().st_size}
            for name in _GRAPH_FILES
        },
        "checkpointSha256": checkpoint_sha256,
        "statsSha256": _sha256(stats_path),
        "gitRevision": git_revision,
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    return manifest


def _parser() -> argparse.ArgumentParser:
    """Build the digest-pinned browser bundle parser.

    :returns: Parser accepting explicit sources or sketch-render defaults.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", help="Local path or R2 URI")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--stats", help="Local path or R2 URI of stats.npz")
    parser.add_argument("--stats-sha256")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Verify sources, load the real checkpoint on CPU, and publish the bundle.

    :param argv: Optional arguments excluding the executable name.
    :raises ValueError: A source and its digest are not specified together.
    """
    args = _parser().parse_args(argv)
    defaults = _load_settings()
    checkpoint_source = args.checkpoint or defaults.checkpoint
    checkpoint_digest = args.checkpoint_sha256 or (
        defaults.checkpoint_sha256 if args.checkpoint is None else None
    )
    stats_source = args.stats or defaults.stats
    stats_digest = args.stats_sha256 or (defaults.stats_sha256 if args.stats is None else None)
    if checkpoint_digest is None or stats_digest is None:
        raise ValueError("explicit checkpoint and stats sources require their SHA-256 digests")
    checkpoint = resolve_inverse_checkpoint(checkpoint_source, checkpoint_digest)
    stats = _resolve_stats(stats_source, stats_digest)
    model = VSTFlowMatchingModule.load_from_checkpoint(
        checkpoint, map_location="cpu", weights_only=False
    ).eval()
    export_browser_surge_bundle(
        model,
        stats,
        args.output.expanduser().resolve(),
        checkpoint_sha256=checkpoint_digest,
        git_revision=_producer_revision(),
    )


if __name__ == "__main__":
    main()
