"""Four-model publication retains executable, distinct checkpoint bundles."""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

from synth_setter.tools.export_browser_surge_comparison import main


@pytest.mark.slow
@pytest.mark.integration_r2
@pytest.mark.r2
def test_comparison_cli_four_checkpoints_produces_executable_catalog(tmp_path: Path) -> None:
    """Each catalog entry supplies a real conditioning and velocity graph.

    :param tmp_path: Isolated output directory.
    """
    destination = tmp_path / "models"
    main(["--output", str(destination)])
    catalog = json.loads((destination / "catalog.json").read_text())
    assert len({row["checkpointSha256"] for row in catalog["models"]}) == 4
    for row in catalog["models"]:
        root = destination / row["id"]
        manifest = json.loads((root / "manifest.json").read_text())
        (mel,) = ort.InferenceSession(str(root / "frontend.onnx")).run(
            None, {"waveform": np.zeros((1, 2, 176400), dtype=np.float32)}
        )
        feeds = {"mel": mel}
        if row["sketch"]:
            (sketch,) = ort.InferenceSession(str(root / "sketch.onnx")).run(
                None, {"waveform": np.zeros((1, 2, 176400), dtype=np.float32)}
            )
            feeds["sketch_ctrl"] = sketch
        conditioning = ort.InferenceSession(str(root / "conditioning.onnx"))
        encoded = conditioning.run(None, feeds)
        velocity_feeds = dict(
            zip([out.name for out in conditioning.get_outputs()], encoded, strict=True)
        )
        width = manifest["encodedWidth"]
        velocity_feeds.update(
            {
                "x": np.zeros((1, width), dtype=np.float32),
                "t": np.array([[0.5]], dtype=np.float32),
                "branch_weights": np.array(
                    [-1, 0, 0, 2] if row["sketch"] else [-1, 2], dtype=np.float32
                ),
            }
        )
        (velocity,) = ort.InferenceSession(str(root / "velocity.onnx")).run(None, velocity_feeds)
        assert isinstance(velocity, np.ndarray)
        assert velocity.shape == (1, width)
        assert np.isfinite(velocity).all()
        assert np.abs(velocity).max() > 0


def test_comparison_cli_existing_output_preserves_owner_files(tmp_path: Path) -> None:
    """Refusing publication does not modify existing output.

    :param tmp_path: Existing destination.
    """
    sentinel = tmp_path / "owned.txt"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        main(["--output", str(tmp_path)])
    assert sentinel.read_text() == "keep"
