"""Attribute a parity mismatch to the stage whose recorded digest diverged.

Two paths that compute the same latents from the same checkpoint can only disagree because they
read different weights, fed a different mel, or ran different kernels. Comparing the arrays alone
says none of that, and the failing lane's logs expire before anyone can ask (#3692), so each path
records a digest per stage and the mismatch is reported against them.
"""

from __future__ import annotations

from collections.abc import Mapping

# Ordered from earliest stage to latest, so a summary reads as a pipeline.
DIGEST_STAGES = ("checkpoint_file", "encoder_weights", "mel")
_MISSING = "missing"

# Injected into both probe programs, so neither path can digest a stage its own way.
DIGEST_PRELUDE = """
import hashlib
import json


def digest_tensor(tensor):
    array = tensor.detach().to("cpu", torch.float32).contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()[:16]


def digest_weights(module):
    running = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        running.update(name.encode())
        running.update(digest_tensor(tensor).encode())
    return running.hexdigest()[:16]


def digest_file(path):
    running = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            running.update(block)
    return running.hexdigest()[:16]


def record_digests(path, checkpoint, vae, mel):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "checkpoint_file": digest_file(checkpoint),
                "encoder_weights": digest_weights(vae),
                "mel": digest_tensor(mel),
            },
            handle,
        )
"""


def describe_digest_divergence(direct: Mapping[str, str], adapter: Mapping[str, str]) -> str:
    """Return which recorded stages differ between the two paths.

    :param direct: Stage digests recorded by the direct upstream path.
    :param adapter: Stage digests recorded by the adapter path.
    :returns: One line per stage, differing stages first, each naming both digests.
    :raises ValueError: If neither path recorded any stage.
    """
    if not direct and not adapter:
        raise ValueError("no digests recorded by either path")

    differing = []
    agreeing = []
    for stage in DIGEST_STAGES:
        left, right = direct.get(stage, _MISSING), adapter.get(stage, _MISSING)
        if left == right and left != _MISSING:
            agreeing.append(stage)
        else:
            differing.append(f"{stage}: direct={left} adapter={right}")

    if not differing:
        return f"every recorded stage agrees ({', '.join(agreeing)}); divergence is downstream"
    return "\n".join([*differing, f"agreeing stages: {', '.join(agreeing) or 'none'}"])
