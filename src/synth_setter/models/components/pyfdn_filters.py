"""Tensor-native filter design for the registered pyFDN attenuation controls."""

import math

from beartype import beartype
from jaxtyping import Float, jaxtyped
from pyFDN import decay_to_first_order_shelf
from pyFDN.eq import (
    BANDWIDTH_R,
    CENTER_FREQUENCIES,
    SHELVING_CROSSOVER,
)
from pyFDN.eq.graphic_eq import _geq_sections
from torch import Tensor

from synth_setter.data.pyfdn_param_spec import PYFDN_RT_CROSSOVER_HZ

_ORDER = "order"


@jaxtyped(typechecker=beartype)
def attenuation_sos(
    delays: Float[Tensor, _ORDER],
    rt_dc: Float[Tensor, ""],
    rt_nyquist: Float[Tensor, ""],
    *,
    sample_rate: int,
) -> Float[Tensor, "1 6 order"]:
    """Design the same first-order shelf as pyFDN's offline decay builder.

    :param delays: Delay-line lengths in samples.
    :param rt_dc: Low-frequency RT60 in seconds.
    :param rt_nyquist: Nyquist RT60 in seconds.
    :param sample_rate: Processing rate in Hz.
    :returns: Normalized first-order shelf SOS bank.
    """
    return decay_to_first_order_shelf(
        rt_dc, rt_nyquist, PYFDN_RT_CROSSOVER_HZ, delays, sample_rate
    )


@jaxtyped(typechecker=beartype)
def command_geq_sos(
    command_gain_db: Float[Tensor, "11 channels"], *, sample_rate: int
) -> Float[Tensor, "11 6 channels"]:
    """Assemble Götz command gains using pyFDN's tensor-capable biquad designs.

    :param command_gain_db: Flat, low-shelf, eight peak, and high-shelf gains in dB.
    :param sample_rate: Processing rate in Hz.
    :returns: Normalized SOS bank preserving gradients to every command gain.
    """
    omega_scale = 2.0 * math.pi / sample_rate
    # pyFDN exposes command-gain assembly only through its internal GEQ builder.
    sos = _geq_sections(
        CENTER_FREQUENCIES * omega_scale,
        SHELVING_CROSSOVER * omega_scale,
        BANDWIDTH_R,
        command_gain_db,
    )
    return sos / sos[:, 3:4, :]
