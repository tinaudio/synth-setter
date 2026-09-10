"""The browser bundle exporter publishes graphs the ONNX runtime can drive end to end."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch
from hydra.utils import instantiate

from synth_setter.models.components.transformer import ApproxEquivTransformer, LearntProjection
from synth_setter.models.flow_onnx import branch_weights
from synth_setter.models.vst_flow_matching_module import VSTFlowMatchingModule
from synth_setter.tools.export_browser_fdn_bundle import export_browser_fdn_bundle, main

_FRAMES = 176_400
_REAL_CHECKPOINT = os.environ.get("SYNTH_SETTER_FDN_SKETCH_CHECKPOINT")
_REAL_STATS = os.environ.get("SYNTH_SETTER_FDN_SKETCH_STATS")


@pytest.fixture
def tiny_fdn_model() -> VSTFlowMatchingModule:
    """Return a small randomly initialised model on the pyFDN sketch contract.

    :returns: Evaluation model with 27 outputs and reverb-sketch tokens.
    """
    torch.manual_seed(5)
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
            input_channels=1,
            spec_shape=(128, 401),
        ),
        vector_field=ApproxEquivTransformer(
            projection=LearntProjection(d_model=16, d_token=16, num_params=27, num_tokens=4),
            d_model=16,
            conditioning_dim=16,
            num_heads=2,
            num_layers=1,
            d_ff=16,
            zero_init=False,
        ),
        optimizer=torch.optim.Adam,
        scheduler=None,
        num_params=27,
        param_spec="pyfdn_n8_mono_householder",
        sketch_controls={
            "profile": "pyfdn_reverb",
            "column": "pyfdn_sketch",
            "num_frames": 32,
            "num_control_tokens": 32,
        },
        test_sample_steps=7,
        test_cfg_strength=1.5,
        test_sketch_cfg_strength=2.5,
    ).eval()
    assert isinstance(model, VSTFlowMatchingModule)
    return model


@pytest.fixture
def stats_path(tmp_path: Path) -> Path:
    """Write finite positive mel statistics on the production grid.

    :param tmp_path: Statistics location.
    :returns: Path to ``stats.npz``.
    """
    rng = np.random.default_rng(2)
    path = tmp_path / "stats.npz"
    np.savez(
        path,
        mean=rng.normal(-30.0, 5.0, (1, 128, 401)).astype(np.float32),
        std=rng.uniform(1.0, 9.0, (1, 128, 401)).astype(np.float32),
    )
    return path


def _sha256(path: Path) -> str:
    """Return one file's hex digest.

    :param path: File to hash.
    :returns: Lowercase SHA-256.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_bundle(bundle: Path, waveform: np.ndarray, mode: str) -> np.ndarray:
    """Drive the published graphs like the browser: front end, conditioning, one velocity.

    :param bundle: Published bundle directory.
    :param waveform: Mono float32 waveform shaped ``(1, 176400)``.
    :param mode: Conditioning mode for the branch weights.
    :returns: One guided velocity row.
    """
    manifest = json.loads((bundle / "manifest.json").read_text())
    frontend = ort.InferenceSession(str(bundle / "frontend.onnx"))
    (mel,) = frontend.run(None, {"waveform": waveform})
    sketch = np.zeros(
        (1, manifest["sketch"]["numControls"], manifest["sketch"]["numFrames"]), np.float32
    )
    encoder = ort.InferenceSession(str(bundle / "conditioning.onnx"))
    conditioning, controls, null_controls = encoder.run(None, {"mel": mel, "sketch_ctrl": sketch})
    velocity = ort.InferenceSession(str(bundle / "velocity.onnx"))
    weights = branch_weights(
        mode, manifest["sampling"]["contentCfg"], manifest["sampling"]["sketchCfg"]
    )
    (row,) = velocity.run(
        None,
        {
            "x": np.zeros((1, manifest["encodedWidth"]), np.float32),
            "t": np.zeros((1, 1), np.float32),
            "conditioning": conditioning,
            "controls": controls,
            "null_controls": null_controls,
            "branch_weights": np.asarray(weights, np.float32),
        },
    )
    assert isinstance(row, np.ndarray)
    return row


def test_export_bundle_graphs_drive_end_to_end_on_onnx_runtime(
    tmp_path: Path, tiny_fdn_model: VSTFlowMatchingModule, stats_path: Path
) -> None:
    """Front end, conditioning, and velocity graphs chain on the CPU runtime.

    :param tmp_path: Bundle parent directory.
    :param tiny_fdn_model: Small model on the pyFDN sketch contract.
    :param stats_path: Mel statistics.
    """
    bundle = tmp_path / "bundle"
    manifest = export_browser_fdn_bundle(
        tiny_fdn_model, stats_path, bundle, checkpoint_sha256="a" * 64, git_revision="deadbeef"
    )
    assert json.loads((bundle / "manifest.json").read_text()) == manifest
    assert manifest["schemaVersion"] == 1
    assert manifest["paramSpecName"] == "pyfdn_n8_mono_householder"
    assert manifest["encodedWidth"] == 27
    assert manifest["sketch"] == {
        "profile": "pyfdn_reverb",
        "column": "pyfdn_sketch",
        "numControls": 10,
        "numFrames": 32,
        "numControlTokens": 32,
    }
    assert manifest["sampleRate"] == 44_100
    assert manifest["frames"] == _FRAMES
    assert manifest["melShape"] == [1, 1, 128, 401]
    assert manifest["sampling"] == {"steps": 7, "contentCfg": 1.5, "sketchCfg": 2.5}
    assert manifest["branchOrder"] == ["unconditional", "sketch_only", "content_only", "full"]
    assert manifest["checkpointSha256"] == "a" * 64
    assert manifest["statsSha256"] == _sha256(stats_path)
    assert manifest["gitRevision"] == "deadbeef"
    files = manifest["files"]
    assert isinstance(files, dict)
    for name in ("frontend.onnx", "conditioning.onnx", "velocity.onnx"):
        assert files[name]["sha256"] == _sha256(bundle / name)
        assert files[name]["bytes"] == (bundle / name).stat().st_size
    rng = np.random.default_rng(0)
    waveform = (rng.standard_normal((1, _FRAMES)) * 0.1).astype(np.float32)
    velocity = _run_bundle(bundle, waveform, "both")
    assert velocity.shape == (1, 27)
    assert np.isfinite(velocity).all()
    assert not np.array_equal(velocity, _run_bundle(bundle, waveform, "unconditional"))


def test_export_bundle_existing_destination_left_untouched(
    tmp_path: Path, tiny_fdn_model: VSTFlowMatchingModule, stats_path: Path
) -> None:
    """Refusing to overwrite keeps caller-owned files intact.

    :param tmp_path: Existing destination.
    :param tiny_fdn_model: Supported model.
    :param stats_path: Mel statistics.
    """
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "owned.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        export_browser_fdn_bundle(
            tiny_fdn_model, stats_path, bundle, checkpoint_sha256="a" * 64, git_revision="x"
        )
    assert (bundle / "owned.txt").read_text() == "keep"
    assert not (bundle / "manifest.json").exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["bundle", "stats.npz"]


def test_export_bundle_music_profile_checkpoint_rejected(tmp_path: Path, stats_path: Path) -> None:
    """A music-sketch checkpoint has no reverb sketch and must not export as an FDN bundle.

    :param tmp_path: Absent destination parent.
    :param stats_path: Mel statistics.
    """
    torch.manual_seed(1)
    model = instantiate(
        {"_target_": "synth_setter.models.vst_flow_matching_module.VSTFlowMatchingModule"},
        encoder=instantiate(
            {"_target_": "synth_setter.models.components.transformer.AudioSpectrogramTransformer"},
            d_model=16,
            n_heads=2,
            n_layers=1,
            n_conditioning_outputs=1,
            patch_size=4,
            patch_stride=2,
            spec_shape=(8, 8),
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
        sketch_controls={"num_frames": 4, "num_control_tokens": 4},
    ).eval()
    with pytest.raises(ValueError, match="pyfdn_reverb"):
        export_browser_fdn_bundle(
            model, stats_path, tmp_path / "bundle", checkpoint_sha256="a" * 64, git_revision="x"
        )
    assert not (tmp_path / "bundle").exists()


def test_cli_wrong_checkpoint_digest_fails_before_writing(
    tmp_path: Path, stats_path: Path
) -> None:
    """An unverified checkpoint is never deserialized or exported.

    :param tmp_path: Destination parent.
    :param stats_path: Mel statistics.
    """
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"not a checkpoint")
    with pytest.raises(RuntimeError, match="SHA-256"):
        main(
            [
                "--checkpoint",
                str(checkpoint),
                "--checkpoint-sha256",
                "0" * 64,
                "--stats",
                str(stats_path),
                "--stats-sha256",
                _sha256(stats_path),
                "--output",
                str(tmp_path / "bundle"),
            ]
        )
    assert not (tmp_path / "bundle").exists()


@pytest.mark.slow
@pytest.mark.skipif(
    not (_REAL_CHECKPOINT and Path(_REAL_CHECKPOINT).is_file() and _REAL_STATS),
    reason="set SYNTH_SETTER_FDN_SKETCH_CHECKPOINT and SYNTH_SETTER_FDN_SKETCH_STATS",
)
def test_cli_real_checkpoint_exports_bundle_matching_native_velocity(tmp_path: Path) -> None:
    """The real flow_sketch checkpoint exports and its velocity graph matches PyTorch.

    :param tmp_path: Bundle destination parent.
    """
    assert _REAL_CHECKPOINT is not None and _REAL_STATS is not None
    checkpoint, stats = Path(_REAL_CHECKPOINT), Path(_REAL_STATS)
    bundle = tmp_path / "bundle"
    main(
        [
            "--checkpoint",
            str(checkpoint),
            "--checkpoint-sha256",
            _sha256(checkpoint),
            "--stats",
            str(stats),
            "--stats-sha256",
            _sha256(stats),
            "--output",
            str(bundle),
        ]
    )
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["encodedWidth"] == 27
    model = VSTFlowMatchingModule.load_from_checkpoint(
        checkpoint, map_location="cpu", weights_only=False
    ).eval()
    rng = np.random.default_rng(0)
    waveform = (rng.standard_normal((1, _FRAMES)) * 0.1).astype(np.float32)
    actual = _run_bundle(bundle, waveform, "both")
    (mel,) = ort.InferenceSession(str(bundle / "frontend.onnx")).run(None, {"waveform": waveform})
    with torch.no_grad():
        expected = model._velocity_field(
            model.encoder(torch.from_numpy(mel)),
            manifest["sampling"]["contentCfg"],
            model._control_token_branches_from_batch({"sketch_ctrl": torch.zeros(1, 10, 32)}),
            sketch_cfg_strength=manifest["sampling"]["sketchCfg"],
        )(torch.zeros(1, 27), torch.zeros(1, 1))
    np.testing.assert_allclose(actual, expected.numpy(), rtol=2e-4, atol=2e-4)
