"""The Surge browser exporter publishes a self-describing executable bundle."""

import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch
from hydra.utils import instantiate
from lightning import Trainer

from synth_setter.cli.sketch_render import _resolve_stats, load_render_config
from synth_setter.models.components.transformer import ApproxEquivTransformer, LearntProjection
from synth_setter.models.flow_onnx import branch_weights
from synth_setter.models.vst_flow_matching_module import VSTFlowMatchingModule
from synth_setter.renderer_factory import make_audio_renderer
from synth_setter.tools.export_browser_surge_bundle import (
    _load_settings,
    export_browser_surge_bundle,
    main,
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.integration_r2,
    pytest.mark.r2,
    pytest.mark.requires_surgepy,
]

_FRAMES = 176_400


def _sha256(path: Path) -> str:
    """Return one file's lowercase SHA-256.

    :param path: File to hash.
    :returns: Hex digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("source_field", ["checkpoint", "stats"])
def test_cli_defaults_pin_content_addressed_artifacts(source_field: str) -> None:
    """Browser defaults name the digest, not a mutable training object.

    :param source_field: Artifact whose URI must contain its declared digest.
    """
    settings = _load_settings()
    source = getattr(settings, source_field)
    digest = getattr(settings, f"{source_field}_sha256")
    assert source.rsplit("/", 2)[-2] == digest


@pytest.fixture(scope="module")
def stats_path() -> Path:
    """Resolve the production two-channel mel statistics.

    :returns: Digest-verified local archive.
    """
    settings = _load_settings()
    return _resolve_stats(settings.stats, settings.stats_sha256)


@pytest.fixture(scope="module")
def tiny_music_model() -> VSTFlowMatchingModule:
    """Build a small flow while retaining the real Surge decoder contract.

    :returns: Evaluation model with stereo mel, music sketch, and 92 outputs.
    """
    torch.manual_seed(7)
    model = instantiate(
        {"_target_": "synth_setter.models.vst_flow_matching_module.VSTFlowMatchingModule"},
        encoder=instantiate(
            {"_target_": "synth_setter.models.components.transformer.AudioSpectrogramTransformer"},
            d_model=16,
            n_heads=2,
            n_layers=1,
            n_conditioning_outputs=1,
            patch_size=16,
            patch_stride=10,
            input_channels=2,
            spec_shape=(128, 401),
        ),
        vector_field=ApproxEquivTransformer(
            projection=LearntProjection(d_model=16, d_token=16, num_params=92, num_tokens=4),
            d_model=16,
            conditioning_dim=16,
            num_heads=2,
            num_layers=1,
            d_ff=16,
            zero_init=False,
        ),
        optimizer=torch.optim.Adam,
        scheduler=None,
        num_params=92,
        param_spec="surge_simple",
        sketch_controls={"profile": "music", "num_frames": 32, "num_control_tokens": 32},
        test_sample_steps=9,
        test_cfg_strength=1.25,
        test_sketch_cfg_strength=2.0,
    ).eval()
    assert isinstance(model, VSTFlowMatchingModule)
    return model


def test_export_bundle_manifest_describes_graphs_preset_and_decoder(
    tmp_path: Path, tiny_music_model: VSTFlowMatchingModule, stats_path: Path
) -> None:
    """Published metadata derives decoder fields from the registered Surge spec and map.

    :param tmp_path: Bundle destination parent.
    :param tiny_music_model: Supported small checkpoint.
    :param stats_path: Production mel statistics.
    """
    bundle = tmp_path / "bundle"
    manifest = export_browser_surge_bundle(
        tiny_music_model,
        stats_path,
        bundle,
        checkpoint_sha256="a" * 64,
        git_revision="deadbeef",
    )
    assert json.loads((bundle / "manifest.json").read_text()) == manifest
    assert manifest["schemaVersion"] == 1
    assert manifest["paramSpecName"] == "surge_simple"
    assert manifest["encodedWidth"] == 92
    assert manifest["sketch"] == {
        "profile": "music",
        "numControls": 386,
        "numFrames": 32,
        "pitchZeroThreshold": 0.1,
    }
    assert manifest["sampleRate"] == 44_100
    assert manifest["frames"] == _FRAMES
    assert manifest["channels"] == 2
    assert manifest["sampling"] == {"steps": 9, "contentCfg": 1.25, "sketchCfg": 2.0}
    assert manifest["checkpointSha256"] == "a" * 64
    assert manifest["statsSha256"] == _sha256(stats_path)
    assert manifest["gitRevision"] == "deadbeef"
    parameters = manifest["parameters"]
    assert isinstance(parameters, list)
    assert parameters[0].keys() >= {
        "name",
        "kind",
        "offset",
        "width",
        "target",
        "id",
        "nativeName",
    }
    assert parameters[0]["target"] == "synth"
    assert parameters[-2] == {
        "name": "pitch",
        "kind": "integer",
        "offset": 89,
        "width": 1,
        "target": "note",
        "min": 48,
        "max": 72,
    }
    assert parameters[-1] == {
        "name": "note_start_and_end",
        "kind": "window",
        "offset": 90,
        "width": 2,
        "target": "note",
        "max": 4.0,
    }
    files = manifest["files"]
    assert isinstance(files, dict)
    for name in (
        "frontend.onnx",
        "sketch.onnx",
        "conditioning.onnx",
        "velocity.onnx",
        "preset.fxp",
    ):
        assert files[name] == {
            "sha256": _sha256(bundle / name),
            "bytes": (bundle / name).stat().st_size,
        }


def test_export_bundle_graphs_chain_on_onnx_runtime(
    tmp_path: Path, tiny_music_model: VSTFlowMatchingModule, stats_path: Path
) -> None:
    """A stereo waveform flows through both front ends and conditioning graph.

    :param tmp_path: Bundle destination parent.
    :param tiny_music_model: Supported small checkpoint.
    :param stats_path: Production mel statistics.
    """
    bundle = tmp_path / "bundle"
    export_browser_surge_bundle(
        tiny_music_model, stats_path, bundle, checkpoint_sha256="b" * 64, git_revision="x"
    )
    waveform = np.zeros((1, 2, _FRAMES), dtype=np.float32)
    (mel,) = ort.InferenceSession(str(bundle / "frontend.onnx")).run(None, {"waveform": waveform})
    (sketch,) = ort.InferenceSession(str(bundle / "sketch.onnx")).run(None, {"waveform": waveform})
    assert isinstance(mel, np.ndarray)
    assert isinstance(sketch, np.ndarray)
    outputs = ort.InferenceSession(str(bundle / "conditioning.onnx")).run(
        None, {"mel": mel, "sketch_ctrl": sketch}
    )
    assert mel.shape == (1, 2, 128, 401)
    assert sketch.shape == (1, 386, 32)
    assert all(isinstance(output, np.ndarray) and np.isfinite(output).all() for output in outputs)


def test_export_bundle_existing_destination_is_untouched(
    tmp_path: Path, tiny_music_model: VSTFlowMatchingModule, stats_path: Path
) -> None:
    """Refusing overwrite preserves caller-owned content.

    :param tmp_path: Existing destination parent.
    :param tiny_music_model: Supported small checkpoint.
    :param stats_path: Production mel statistics.
    """
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "owned.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        export_browser_surge_bundle(
            tiny_music_model, stats_path, bundle, checkpoint_sha256="a" * 64, git_revision="x"
        )
    assert (bundle / "owned.txt").read_text() == "keep"
    assert sorted(path.name for path in bundle.iterdir()) == ["owned.txt"]


@pytest.mark.slow
def test_cli_default_real_checkpoint_runs_native_audio_through_flow(tmp_path: Path) -> None:
    """Default pinned production artifacts export and execute freshly rendered audio.

    :param tmp_path: Bundle destination parent.
    """
    native = make_audio_renderer(load_render_config())
    content_audio = native.render({}, midi_note=72, velocity=80, note_start_and_end=(0.5, 1.5))
    sketch_audio = native.render({}, midi_note=60, velocity=100, note_start_and_end=(0.0, 2.0))
    bundle = tmp_path / "real-bundle"
    main(["--output", str(bundle)])
    manifest = json.loads((bundle / "manifest.json").read_text())
    frontend = ort.InferenceSession(str(bundle / "frontend.onnx"))
    (mel,) = frontend.run(None, {frontend.get_inputs()[0].name: content_audio[None]})
    sketch_graph = ort.InferenceSession(str(bundle / "sketch.onnx"))
    (sketch,) = sketch_graph.run(None, {sketch_graph.get_inputs()[0].name: sketch_audio[None]})
    conditioning, controls, null_controls = ort.InferenceSession(
        str(bundle / "conditioning.onnx")
    ).run(None, {"mel": mel, "sketch_ctrl": sketch})
    assert isinstance(conditioning, np.ndarray)
    assert isinstance(controls, np.ndarray)
    assert isinstance(null_controls, np.ndarray)
    weights = np.asarray(
        branch_weights(
            "both", manifest["sampling"]["contentCfg"], manifest["sampling"]["sketchCfg"]
        ),
        dtype=np.float32,
    )
    (velocity,) = ort.InferenceSession(str(bundle / "velocity.onnx")).run(
        None,
        {
            "x": np.zeros((1, 92), dtype=np.float32),
            "t": np.zeros((1, 1), dtype=np.float32),
            "conditioning": conditioning,
            "controls": controls,
            "null_controls": null_controls,
            "branch_weights": weights,
        },
    )
    assert isinstance(velocity, np.ndarray)
    assert velocity.shape == (1, 92)
    assert np.isfinite(velocity).all()


def test_cli_tiny_checkpoint_publishes_bundle(
    tmp_path: Path, tiny_music_model: VSTFlowMatchingModule, stats_path: Path
) -> None:
    """The public CLI loads a real Lightning checkpoint and publishes all artifacts.

    :param tmp_path: Checkpoint and output parent.
    :param tiny_music_model: Model saved through Lightning.
    :param stats_path: Production mel statistics.
    """
    checkpoint = tmp_path / "model.ckpt"
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.strategy.connect(tiny_music_model)
    trainer.save_checkpoint(checkpoint)
    bundle = tmp_path / "bundle"
    main(
        [
            "--checkpoint",
            str(checkpoint),
            "--checkpoint-sha256",
            _sha256(checkpoint),
            "--stats",
            str(stats_path),
            "--stats-sha256",
            _sha256(stats_path),
            "--output",
            str(bundle),
        ]
    )
    assert json.loads((bundle / "manifest.json").read_text())["encodedWidth"] == 92
    assert (bundle / "sketch.onnx").is_file()
