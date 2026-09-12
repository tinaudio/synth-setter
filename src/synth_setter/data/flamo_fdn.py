"""A complete vanilla FDN shared by the offline and FLAMO render paths."""

from dataclasses import dataclass

import numpy as np
import torch
from flamo.processor.system import Shell
from pyFDN import (
    FDNBuild,
    build_to_impz,
    dss_to_flamo,
    fdn_build_from_dict,
    fdn_build_to_dict,
)


@dataclass(frozen=True)
class FlamoFDN:
    """Own a validated FDN build with no processing outside its declared fields.

    .. attribute :: build

        Canonical snapshot of the supplied matrices, delays, sample rate and SOS hooks.
    """

    build: FDNBuild

    def __post_init__(self) -> None:
        """Reject unsupported build contents before either renderer is constructed.

        :raises TypeError: The input is not a pyFDN build or contains nonnumeric hooks.
        :raises ValueError: Geometry or SOS coefficients violate the FLAMO FDN contract.
        """
        if not isinstance(self.build, FDNBuild):
            raise TypeError("FlamoFDN requires a constructed pyFDN.FDNBuild")
        build = fdn_build_from_dict(fdn_build_to_dict(self.build))
        if any(dimension == 0 for array in (build.A, build.B, build.C) for dimension in array.shape):
            raise ValueError("FDN matrices must have nonempty channel dimensions")
        for name in ("post_delay", "post_matrix", "post_output"):
            sos = getattr(build, name)
            if sos is None:
                continue
            if not np.all(sos[:, 3, :] == 1.0):
                raise ValueError(f"{name} must be normalized with a0=1 for offline rendering")
        object.__setattr__(self, "build", build)

    def impulse_response(self, length: int) -> np.ndarray:
        """Render the complete build through pyFDN's time-domain implementation.

        :param length: Positive output length in samples.
        :returns: Response shaped ``(samples, outputs, inputs)`` without FFT tail aliasing.
        :raises ValueError: The requested output length is not positive.
        """
        if length <= 0:
            raise ValueError("impulse-response length must be positive")
        return build_to_impz(self.build, ir_len=length)

    def to_flamo(
        self,
        *,
        nfft: int = 2**16,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Shell:
        """Construct the corresponding upstream FLAMO graph, including every filter hook.

        :param nfft: Positive FFT period; longer periods reduce circular-tail aliasing.
        :param device: FLAMO device, or its default when omitted.
        :param dtype: FLAMO floating-point precision, or its default when omitted.
        :returns: FFT/FDN/inverse-FFT Shell; network parameter binding is a separate concern.
        :raises ValueError: The FFT period is not positive.
        """
        if nfft <= 0:
            raise ValueError("nfft must be positive")
        build = self.build
        return dss_to_flamo(
            build.A,
            build.B,
            build.C,
            build.D,
            build.delays,
            build.fs,
            nfft=nfft,
            device=device,
            dtype=dtype,
            post_delay=build.post_delay,
            post_matrix=build.post_matrix,
            post_output=build.post_output,
        )
