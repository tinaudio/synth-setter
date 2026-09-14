"""Export a pyFDN sketch flow checkpoint as a self-contained browser inference bundle."""

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

import numpy as np
import torch

from synth_setter.cli.clap_render import resolve_inverse_checkpoint
from synth_setter.cli.sketch_render import _producer_revision, _resolve_stats
from synth_setter.conditioning import SketchControlSpec, resolve_sketch_controls
from synth_setter.data.pyfdn_instrument import PyFDNRenderer
from synth_setter.data.vst.param_spec_registry import resolve_param_spec
from synth_setter.features.pyfdn_controls import extract_reverb_sketch
from synth_setter.models.flow_onnx import export_flow_onnx
from synth_setter.models.mel_frontend import NormalizedMelFrontend, export_frontend_onnx
from synth_setter.models.vst_flow_matching_module import VSTFlowMatchingModule
from synth_setter.param_spec_name import ParamSpecName

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1
_SAMPLE_RATE = 44_100
_DURATION_SECONDS = 4.0
_BRANCH_ORDER = ("unconditional", "sketch_only", "content_only", "full")
_GRAPHS = ("frontend.onnx", "conditioning.onnx", "velocity.onnx")
# Seed for the representative pyFDN render that shapes the exported graphs.
_EXAMPLE_SEED = 3


def _sha256(path: Path) -> str:
    """Return one file's hex digest.

    :param path: File to hash.
    :returns: Lowercase SHA-256.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _example_impulse_response(param_spec_name: ParamSpecName) -> np.ndarray:
    """Render one real impulse response from the checkpoint's own synth.

    :param param_spec_name: Registered pyFDN parameter spec.
    :returns: Channel-first float32 audio on the production grid.
    """
    spec = resolve_param_spec(param_spec_name)
    renderer = PyFDNRenderer(
        excitation="impulse",
        param_spec_name=param_spec_name,
        plugin_path="pyfdn",
        sample_rate=_SAMPLE_RATE,
        channels=1,
        signal_duration_seconds=_DURATION_SECONDS,
        plugin_state_path="",
    )
    synth_params, _ = spec.sample(np.random.default_rng(_EXAMPLE_SEED))
    return renderer.render(synth_params, 60, 100, (0.0, _DURATION_SECONDS))


def export_browser_fdn_bundle(
    model: VSTFlowMatchingModule,
    stats_path: Path,
    output: Path,
    *,
    checkpoint_sha256: str,
    git_revision: str,
) -> dict[str, object]:
    """Atomically publish the three ONNX graphs and a provenance manifest.

    :param model: CPU evaluation checkpoint with mel and ``pyfdn_reverb`` sketch conditioning.
    :param stats_path: Training mel statistics archive with ``mean`` and ``std``.
    :param output: Absent destination directory.
    :param checkpoint_sha256: Verified digest of the source checkpoint.
    :param git_revision: Producing source revision.
    :returns: The written manifest.
    :raises FileExistsError: The destination already exists.
    :raises ValueError: The checkpoint is not a pyFDN reverb-sketch mel flow.
    """
    if output.exists():
        raise FileExistsError(f"bundle destination already exists: {output}")
    sketch = resolve_sketch_controls(model.hparams["sketch_controls"])
    if sketch is None or sketch.profile != "pyfdn_reverb":
        raise ValueError("browser FDN bundles require the pyfdn_reverb sketch profile")
    if model.hparams["conditioning"] != "mel":
        raise ValueError("browser FDN bundles require mel content conditioning")
    with np.load(stats_path) as stats:
        frontend = NormalizedMelFrontend(
            sample_rate=_SAMPLE_RATE, mean=stats["mean"], std=stats["std"]
        ).eval()
    audio = _example_impulse_response(ParamSpecName(model.hparams["param_spec"]))
    with torch.no_grad():
        mel = frontend(torch.from_numpy(audio))
    batch = {
        "mel": mel,
        "sketch_ctrl": torch.from_numpy(extract_reverb_sketch(audio[0], _SAMPLE_RATE)).unsqueeze(
            0
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}."))
    try:
        manifest = _write_bundle(
            model, frontend, batch, staging, sketch, checkpoint_sha256, stats_path, git_revision
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
    frontend: NormalizedMelFrontend,
    batch: dict[str, torch.Tensor],
    staging: Path,
    sketch: SketchControlSpec,
    checkpoint_sha256: str,
    stats_path: Path,
    git_revision: str,
) -> dict[str, object]:
    """Write the three graphs and the manifest into an unpublished staging directory.

    :param model: Validated CPU evaluation checkpoint.
    :param frontend: Statistics-bound mel front end.
    :param batch: Representative normalized mel and reverb-sketch pair.
    :param staging: Empty staging directory.
    :param sketch: Checkpoint sketch-control contract.
    :param checkpoint_sha256: Verified digest of the source checkpoint.
    :param stats_path: Training statistics archive.
    :param git_revision: Producing source revision.
    :returns: The manifest that was written.
    """
    export_flow_onnx(model, batch, staging / "flow")
    for name in ("conditioning.onnx", "velocity.onnx"):
        (staging / "flow" / name).rename(staging / name)
    (staging / "flow").rmdir()
    export_frontend_onnx(frontend, staging / "frontend.onnx")
    manifest: dict[str, object] = {
        "schemaVersion": _SCHEMA_VERSION,
        "paramSpecName": model.hparams["param_spec"],
        "encodedWidth": model.hparams["num_params"],
        "sketch": {
            "profile": sketch.profile,
            "column": sketch.column,
            "numControls": sketch.layout.num_controls,
            "numFrames": sketch.num_frames,
            "numControlTokens": sketch.num_control_tokens,
        },
        "sampleRate": _SAMPLE_RATE,
        "frames": frontend.frames,
        "melShape": list(batch["mel"].shape),
        "sampling": {
            "steps": model.hparams["test_sample_steps"],
            "contentCfg": model.hparams["test_cfg_strength"],
            "sketchCfg": model.hparams["test_sketch_cfg_strength"],
        },
        "branchOrder": list(_BRANCH_ORDER),
        "files": {
            name: {"sha256": _sha256(staging / name), "bytes": (staging / name).stat().st_size}
            for name in _GRAPHS
        },
        "checkpointSha256": checkpoint_sha256,
        "statsSha256": _sha256(stats_path),
        "gitRevision": git_revision,
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    return manifest


def _parser() -> argparse.ArgumentParser:
    """Build the digest-pinned export parser.

    :returns: Parser requiring checkpoint, statistics, and an absent destination.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Local path or R2 URI")
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--stats", required=True, help="Local path or R2 URI of stats.npz")
    parser.add_argument("--stats-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Verify inputs, load the checkpoint on CPU, and publish the bundle.

    :param argv: Optional command-line arguments excluding the executable name.
    """
    args = _parser().parse_args(argv)
    checkpoint = resolve_inverse_checkpoint(args.checkpoint, args.checkpoint_sha256)
    stats = _resolve_stats(args.stats, args.stats_sha256)
    model = VSTFlowMatchingModule.load_from_checkpoint(
        checkpoint, map_location="cpu", weights_only=False
    ).eval()
    export_browser_fdn_bundle(
        model,
        stats,
        args.output.expanduser().resolve(),
        checkpoint_sha256=args.checkpoint_sha256,
        git_revision=_producer_revision(),
    )


if __name__ == "__main__":
    main()
