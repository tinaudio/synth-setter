"""Compatibility tests for validation-aligned checkpoint selection."""

from lightning.pytorch.callbacks import ModelCheckpoint

from synth_setter.utils.callbacks import ValidationAlignedModelCheckpoint


def test_checkpoint_state_key_remains_resume_compatible() -> None:
    """Existing ModelCheckpoint state restores into the aligned callback."""
    kwargs = {"monitor": "val/score", "mode": "min", "every_n_train_steps": 10}
    existing = ModelCheckpoint(**kwargs)
    aligned = ValidationAlignedModelCheckpoint(**kwargs)

    assert aligned.state_key == existing.state_key
