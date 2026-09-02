"""Canonical procedural excitation for the pyFDN instrument.

Example:
    ``source = generate_canonical_pyfdn_source()`` returns the fixed F1 waveform.
"""

import hashlib
from importlib.metadata import version

import librosa
import numpy as np
from jaxtyping import Float32

_SOURCE_IDENTITY = "librosa_log_chirp_v1"
_FMIN_HZ = 20.0
_FMAX_HZ = 20_000.0
PYFDN_SOURCE_SAMPLE_RATE_HZ = 48_000
PYFDN_SOURCE_CHANNELS = 1
PYFDN_SOURCE_TOTAL_FRAMES = 192_000
_CHIRP_FRAMES = 48_000
_LINEAR = False
_PHI = None
_AMPLITUDE = 0.1

type PyFDNSourceProvenance = dict[str, str | float | int | bool | None]


def generate_canonical_pyfdn_source() -> Float32[np.ndarray, "1 192000"]:
    """Generate the immutable canonical F1 pyFDN excitation.

    :returns: Read-only contiguous float32 audio shaped ``(1, 192000)``.
    :raises ValueError: Librosa returns wrong-shape, non-finite, or out-of-range samples.
    """
    chirp = librosa.chirp(
        fmin=_FMIN_HZ,
        fmax=_FMAX_HZ,
        sr=PYFDN_SOURCE_SAMPLE_RATE_HZ,
        length=_CHIRP_FRAMES,
        linear=_LINEAR,
    )
    chirp_array = np.asarray(chirp)
    if chirp_array.shape != (_CHIRP_FRAMES,):
        raise ValueError(
            f"librosa chirp must have shape {(_CHIRP_FRAMES,)}, got {chirp_array.shape}"
        )
    if not np.isfinite(chirp_array).all():
        raise ValueError("librosa chirp must contain only finite values")
    if np.max(np.abs(chirp_array)) > 1.0:
        raise ValueError("librosa chirp amplitude must not exceed 1.0")

    source = np.zeros(
        (PYFDN_SOURCE_CHANNELS, PYFDN_SOURCE_TOTAL_FRAMES), dtype=np.float32
    )
    source[0, :_CHIRP_FRAMES] = _AMPLITUDE * chirp_array
    immutable_bytes = source.tobytes(order="C")
    return np.frombuffer(immutable_bytes, dtype=np.float32).reshape(
        PYFDN_SOURCE_CHANNELS, PYFDN_SOURCE_TOTAL_FRAMES
    )


def _canonical_pyfdn_source_provenance(
    source: Float32[np.ndarray, "1 192000"],
) -> PyFDNSourceProvenance:
    """Build provenance for already-generated canonical source bytes.

    :param source: Canonical contiguous float32 audio shaped ``(1, 192000)``.
    :returns: Complete source fields and the supplied source's SHA-256.
    """
    return {
        "identity": _SOURCE_IDENTITY,
        "implementation": "librosa.chirp",
        "fmin_hz": _FMIN_HZ,
        "fmax_hz": _FMAX_HZ,
        "sample_rate_hz": PYFDN_SOURCE_SAMPLE_RATE_HZ,
        "chirp_frames": _CHIRP_FRAMES,
        "total_frames": PYFDN_SOURCE_TOTAL_FRAMES,
        "linear": _LINEAR,
        "phi": _PHI,
        "amplitude": _AMPLITUDE,
        "tail_frames": PYFDN_SOURCE_TOTAL_FRAMES - _CHIRP_FRAMES,
        "channels": PYFDN_SOURCE_CHANNELS,
        "dtype": "float32",
        "layout": "channel_first",
        "librosa_version": version("librosa"),
        "sha256": hashlib.sha256(source.tobytes(order="C")).hexdigest(),
    }


def canonical_pyfdn_source_provenance() -> PyFDNSourceProvenance:
    """Describe the canonical source and fingerprint its generated float32 bytes.

    :returns: Complete sound-affecting source fields, installed librosa version, and SHA-256.
    """
    return _canonical_pyfdn_source_provenance(generate_canonical_pyfdn_source())
