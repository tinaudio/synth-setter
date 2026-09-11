#!/usr/bin/env python
"""Encode a fixed prompt set with SA3's prompt conditioner and save the embeddings.

Deliberately imports nothing from ``synth_setter`` so the same script runs inside a
scratch venv pinned to upstream's ``torch==2.7.1``, isolating the torch version as the
only variable. Run it once per environment, then diff the two ``.npy`` files.

    uv run python scripts/dev/sa3_t5gemma_parity.py --checkpoint DIR --out locked.npy
"""

import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np
import torch
from safetensors import safe_open
from stable_audio_3.models.conditioners import T5GemmaConditioner

PADDING_EMBEDDING_KEY = "conditioner.conditioners.prompt.padding_embedding"
# Short, exactly-at-budget, over-budget, empty, and non-ASCII prompts.
PROMPTS: list[str] = [
    "",
    " ",
    "warm analog pad",
    "bright plucked lead with fast decay",
    "cutoff, resonance, a_amp_eg_attack, a_amp_eg_decay",
    "808 sub bass, heavy distortion",
    "ambient drone, slow evolving texture",
    "staccato marimba, dry room",
    "granular wash with long reverb tail",
    "detuned saw stack, wide chorus",
    "acid squelch, high resonance sweep",
    "sine bell, metallic overtones",
    "noise sweep riser",
    "vocal formant pad",
    "percussive wooden click",
    "orchestral string swell",
    "chiptune square arpeggio",
    "ばりばりのベース音",
    "clavier électrique lumineux",
    "a, " * 300,
    "cutoff, resonance, " * 200,
    "x" * 4000,
]


def load_conditioner(checkpoint_dir: Path, device: str, dtype: str) -> T5GemmaConditioner:
    """Build SA3's prompt conditioner with its checkpoint's learned padding embedding.

    :param checkpoint_dir: Directory holding ``model_config.json`` and ``model.safetensors``.
    :param device: Torch device.
    :param dtype: Torch dtype name the encoder runs in.
    :returns: Conditioner ready for inference.
    """
    conditioning = json.loads((checkpoint_dir / "model_config.json").read_text())["model"][
        "conditioning"
    ]
    config = next(c for c in conditioning["configs"] if c["id"] == "prompt")["config"]
    conditioner = T5GemmaConditioner(
        output_dim=conditioning["cond_dim"],
        model_name="google/t5gemma-b-b-ul2",
        max_length=config["max_length"],
        padding_mode=config["padding_mode"],
        model_path=str(checkpoint_dir),
        subfolder=config["subfolder"],
    )
    with safe_open(checkpoint_dir / "model.safetensors", framework="pt", device=device) as weights:
        conditioner.padding_embedding.data.copy_(weights.get_tensor(PADDING_EMBEDDING_KEY))
    # SA3 stashes the frozen encoder in __dict__, so it reads as the class, not an instance.
    cast("torch.nn.Module", conditioner.model).to(getattr(torch, dtype))
    return conditioner.to(device).eval().requires_grad_(False)


def main() -> None:
    """Encode the fixed prompt set and write the embeddings to disk."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    conditioner = load_conditioner(args.checkpoint, args.device, args.dtype)
    with torch.no_grad():
        embeddings, _ = conditioner(PROMPTS, args.device)
    array = embeddings.float().cpu().numpy()
    np.save(args.out, array)
    print(  # noqa: T201 — CLI tool: stdout is its product, not a debug print
        f"torch={torch.__version__} dtype={args.dtype} prompts={len(PROMPTS)} shape={array.shape} -> {args.out}"
    )


if __name__ == "__main__":
    main()
