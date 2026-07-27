"""Deterministic seed derivation for online synthetic datasets.

Derive a seed for each logical row before resetting its ``torch.Generator``::

    generator.manual_seed(derive_sample_seed(dataset_seed, index))
"""

import hashlib

_PERSONALIZATION = b"synth-sample"


def derive_sample_seed(base_seed: int, index: int) -> int:
    """Derive an RNG seed from a dataset seed and sample index.

    :param base_seed: Seed identifying the dataset split.
    :param index: Logical sample index within the split.
    :returns: Deterministic 64-bit seed accepted by ``torch.Generator``.
    """
    payload = f"{base_seed}:{index}".encode("ascii")
    digest = hashlib.blake2b(payload, digest_size=8, person=_PERSONALIZATION).digest()
    return int.from_bytes(digest, "little")
