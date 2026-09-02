"""Click validation for runtime classifier-free guidance options."""

import math

import click


def validate_cfg_strength(
    _context: click.Context, parameter: click.Parameter, value: float | None
) -> float | None:
    """Accept an omitted or finite nonnegative guidance strength.

    :param _context: Active Click context.
    :param parameter: Option being validated.
    :param value: Parsed override, or ``None`` when omitted.
    :returns: ``None`` when omitted; otherwise a finite nonnegative value.
    :raises click.BadParameter: The override is negative or non-finite.
    """
    if value is not None and (not math.isfinite(value) or value < 0.0):
        raise click.BadParameter(
            "must be finite and greater than or equal to zero", param=parameter
        )
    return value
