"""Regenerate the SAME-S golden latents fixture for the equivalence test.

Run from a checkout with the ``same`` extra installed
(``PYTHONPATH=. uv run python tests/pipeline/data/generate_same_golden.py``);
encodes the deterministic sine-sweep fixture on CPU with the public
``stabilityai/SAME-S`` checkpoint and overwrites
``tests/fixtures/same/same_s_golden_latents.npz``. Record the lock's torch
version in the equivalence test's docstring whenever the golden is refreshed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def main() -> None:
    """Encode the shared fixture with real SAME-S weights and write the golden."""
    import torch

    from synth_setter.pipeline.data.add_embeddings import load_same_audio_encoder
    from tests.pipeline.data.test_same_encoder_e2e import _GOLDEN_PATH, sine_sweep_fixture

    encode = load_same_audio_encoder("stabilityai/SAME-S", device="cpu")
    latents = encode(sine_sweep_fixture())
    _GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(_GOLDEN_PATH, latents=latents)
    print(f"wrote {Path(_GOLDEN_PATH)} shape={latents.shape} torch={torch.__version__}")


if __name__ == "__main__":
    main()
