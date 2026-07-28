"""Data-path tests for the sketch control column plumbing."""

from pathlib import Path

import numpy as np
import torch

from synth_setter.conditioning import EmbeddingConditioningSpec, SketchControlSpec
from synth_setter.data.lance_datamodule import LanceVSTDataModule
from synth_setter.data.vst_datamodule import RawBatch, prepare_batch
from synth_setter.param_spec_name import ParamSpecName

_NUM_FRAMES = 11
_M2L_SHAPE = (6, 7)


def _sketch_module(root: Path, *, fake: bool) -> LanceVSTDataModule:
    """Build a datamodule configured for m2l plus sketch controls.

    :param root: Dataset root (unused in fake mode).
    :param fake: Whether to synthesize batches.
    :returns: Unset-up datamodule.
    """
    return LanceVSTDataModule(
        dataset_root=root,
        batch_size=2,
        conditioning=EmbeddingConditioningSpec(
            column="music2latent", input_shape=_M2L_SHAPE
        ),
        sketch=SketchControlSpec(num_frames=_NUM_FRAMES),
        fake=fake,
        use_saved_mean_and_variance=False,
        num_workers=0,
        pin_memory=False,
        param_spec_name=ParamSpecName("surge_xt"),
    )


def test_sketch_spec_adds_column_to_loader_projection(tmp_path: Path) -> None:
    """A configured sketch spec projects its stored column for every split.

    :param tmp_path: Per-test dataset root.
    """
    module = _sketch_module(tmp_path, fake=True)

    columns = module._loader_columns(read_audio=False)  # noqa: SLF001

    assert "sketch_ctrl" in columns


def test_fake_mode_with_sketch_spec_yields_sketch_batch_key(tmp_path: Path) -> None:
    """Fake batches expose sketch controls with the configured shape.

    :param tmp_path: Per-test dataset root.
    """
    module = _sketch_module(tmp_path, fake=True)

    module.setup("validate")
    try:
        batch = next(iter(module.val_dataloader()))
    finally:
        module.teardown()

    sketch = batch["sketch_ctrl"]
    assert sketch is not None
    assert sketch.shape == (2, 3, _NUM_FRAMES)


def test_prepare_batch_ot_keeps_sketch_aligned_with_params() -> None:
    """OT row permutation moves sketch rows together with their parameters."""
    row_ids = np.arange(6, dtype=np.float32)
    raw: RawBatch = {
        "param_array": np.stack([row_ids / 8, row_ids / 8], axis=1),
        "sketch_ctrl": np.repeat(row_ids[:, None, None], 3, axis=1).repeat(4, axis=2),
    }

    batch = prepare_batch(
        raw,
        mean=None,
        std=None,
        rescale_params=False,
        ot=True,
        generator=torch.Generator().manual_seed(3),
    )

    sketch = batch["sketch_ctrl"]
    params = batch["params"]
    assert sketch is not None
    assert params is not None
    torch.testing.assert_close(sketch[:, 0, 0] / 8, params[:, 0])
