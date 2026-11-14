"""Record ``SketchControlTokens`` outputs at a historical commit for equivalence tests.

Run from a worktree checked out at the reference commit (``tests/fixtures/sketch/README.md``
lists them); the module under test is whatever ``synth_setter`` resolves to there, so only
constructor arguments every revision accepts are used.
"""

import argparse
import logging
from pathlib import Path
from typing import cast

import torch
from torch import nn

from synth_setter.conditioning import NUM_SKETCH_CONTROLS, NUM_SKETCH_TRACK_ROWS
from synth_setter.models.components.sketch_tokens import SketchControlTokens

_BATCH = 2
_D_MODEL = 16
_NUM_CONTROL_TOKENS = 32
_SEED = 2613

logger = logging.getLogger(__name__)


def _controls(frames: int, generator: torch.Generator) -> torch.Tensor:
    """Draw a stored-sketch-like batch: signed track rows and near one-hot pitch rows.

    :param frames: Temporal grid length.
    :param generator: Seeded RNG shared across the recorded tensors.
    :returns: ``(_BATCH, NUM_SKETCH_CONTROLS, frames)`` controls with silent pitch frames.
    """
    controls = torch.rand((_BATCH, NUM_SKETCH_CONTROLS, frames), generator=generator)
    controls[:, :NUM_SKETCH_TRACK_ROWS] = controls[:, :NUM_SKETCH_TRACK_ROWS] * 2 - 1
    pitch = torch.zeros_like(controls[:, NUM_SKETCH_TRACK_ROWS:])
    bins = torch.randint(0, pitch.shape[1], (_BATCH, frames), generator=generator)
    pitch.scatter_(1, bins[:, None, :], 1.0)
    pitch[:, :, ::7] = 0.0
    controls[:, NUM_SKETCH_TRACK_ROWS:] = pitch
    return controls


def main() -> None:
    """Record train-mode tokens for each requested frame grid and save them with the inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--commit", required=True, help="full SHA of the checked-out reference")
    args = parser.parse_args()

    generator = torch.Generator().manual_seed(_SEED)
    module = SketchControlTokens(
        d_model=_D_MODEL, num_control_tokens=_NUM_CONTROL_TOKENS, profile="music"
    )
    with torch.no_grad():
        for projection in module.projections.values():
            linear = cast(nn.Linear, projection)
            linear.weight.copy_(torch.randn(linear.weight.shape, generator=generator) * 0.1)
    keep = torch.tensor([[True, True, True], [True, False, True]])
    fixture: dict[str, object] = {
        "commit": args.commit,
        "torch": str(torch.__version__),
        "state_dict": module.state_dict(),
        "keep": keep,
    }
    module.train()
    with torch.no_grad():
        for frames in args.frames:
            controls = _controls(frames, generator)
            fixture[f"controls_{frames}"] = controls
            fixture[f"tokens_{frames}"] = module(controls, keep)
    torch.save(fixture, args.output)
    logger.info("saved %s from %s", args.output, args.commit)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
