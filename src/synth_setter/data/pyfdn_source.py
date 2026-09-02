"""Canonical procedural excitation for the pyFDN instrument."""

import hashlib
from importlib.metadata import version

import librosa
import numpy as np
from jaxtyping import Float32

_SOURCE_IDENTITY = "librosa_log_chirp_v1"
_FMIN_HZ = 20.0
_FMAX_HZ = 20_000.0
_SAMPLE_RATE_HZ = 48_000
_CHIRP_FRAMES = 48_000
_TOTAL_FRAMES = 192_000
_LINEAR = False
_PHI = None
_AMPLITUDE = 0.1

def generate_canonical_pyfdn_source() -> Float32[np.ndarray, "1 192000"]:
    """Generate the immutable canonical F1 pyFDN excitation.

    :returns: Read-only contiguous float32 audio shaped ``(1, 192000)``.
    """
    chirp = librosa.chirp(
        fmin=_FMIN_HZ,
        fmax=_FMAX_HZ,
        sr=_SAMPLE_RATE_HZ,
        length=_CHIRP_FRAMES,
        linear=_LINEAR,
    )
    source = np.zeros((1, _TOTAL_FRAMES), dtype=np.float32)
    source[0, :_CHIRP_FRAMES] = _AMPLITUDE * chirp
    immutable_bytes = source.tobytes(order="C")
    return np.frombuffer(immutable_bytes, dtype=np.float32).reshape(1, _TOTAL_FRAMES)


def canonical_pyfdn_source_provenance() -> dict[
    str, str | float | int | bool | None
]:
    """Describe the canonical source and fingerprint its generated float32 bytes.

    :returns: Complete sound-affecting source fields, installed librosa version, and SHA-256.
    """
    source = generate_canonical_pyfdn_source()
    return {
        "identity": _SOURCE_IDENTITY,
        "implementation": "librosa.chirp",
        "fmin_hz": _FMIN_HZ,
        "fmax_hz": _FMAX_HZ,
        "sample_rate_hz": _SAMPLE_RATE_HZ,
        "chirp_frames": _CHIRP_FRAMES,
        "total_frames": _TOTAL_FRAMES,
        "linear": _LINEAR,
        "phi": _PHI,
        "amplitude": _AMPLITUDE,
        "tail_frames": _TOTAL_FRAMES - _CHIRP_FRAMES,
        "channels": 1,
        "dtype": "float32",
        "layout": "channel_first",
        "librosa_version": version("librosa"),
        "sha256": hashlib.sha256(source.tobytes(order="C")).hexdigest(),
    }
