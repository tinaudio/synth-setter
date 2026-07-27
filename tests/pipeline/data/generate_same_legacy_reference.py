"""Generate SAME parity references with the pinned legacy runtime.

Run at commit ``82648af075f677cedc7ad3f2bcd562bdfe9297d1`` with
``stable-audio-tools==0.0.20`` and ``torch==2.12.0+cpu``. Each archive embeds
its runtime and immutable HuggingFace checkpoint provenance.
"""

from __future__ import annotations

import json
import logging
import platform
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from synth_setter.pipeline.data.add_embeddings import SAME_SAMPLE_RATE
from tests.helpers.same_reference import (
    SAME_HF_CHECKPOINTS,
    SAME_REFERENCE_DIR,
    SAME_REFERENCE_RANDOM_SEED,
    same_reference_audio,
)

logger = logging.getLogger(__name__)


def _encode_reference(repo_id: str, revision: str) -> np.ndarray:
    """Encode the deterministic fixture through the strict legacy loader path.

    :param repo_id: Public HuggingFace checkpoint repository.
    :param revision: Immutable checkpoint commit.
    :returns: Legacy float32 latents.
    """
    from stable_audio_tools.models.factory import create_model_from_config
    checkpoint_dir = Path(snapshot_download(repo_id, revision=revision))
    config = json.loads((checkpoint_dir / "model_config.json").read_text())
    model = create_model_from_config(config)
    model.load_state_dict(load_file(checkpoint_dir / "model.safetensors"), strict=True)
    model = model.to("cpu").eval().requires_grad_(False)
    torch.manual_seed(SAME_REFERENCE_RANDOM_SEED)
    with torch.no_grad():
        latents = model.encode(torch.from_numpy(same_reference_audio(SAME_SAMPLE_RATE)))
    return latents.float().cpu().numpy()


def main() -> None:
    """Write one compressed reference archive per immutable SAME checkpoint."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for model_name, (repo_id, revision) in SAME_HF_CHECKPOINTS.items():
        latents = _encode_reference(repo_id, revision)
        output = SAME_REFERENCE_DIR / f"{model_name}_legacy_reference.npz"
        np.savez_compressed(
            output,
            latents=latents,
            hf_repo=np.array(repo_id),
            hf_revision=np.array(revision),
            reference_runtime=np.array("stable-audio-tools==0.0.20"),
            torch_version=np.array(torch.__version__),
            platform_system=np.array(platform.system()),
            platform_machine=np.array(platform.machine()),
            random_seed=np.array(SAME_REFERENCE_RANDOM_SEED),
        )
        logger.info("wrote %s shape=%s torch=%s", output, latents.shape, torch.__version__)


if __name__ == "__main__":
    main()
