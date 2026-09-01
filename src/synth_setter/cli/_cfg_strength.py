"""Shared runtime CFG override handling for CLAP render modes."""

import math
from collections.abc import MutableMapping

import click


def validate_cfg_strength(
    _context: click.Context, parameter: click.Parameter, value: float | None
) -> float | None:
    """Accept an omitted or finite nonnegative guidance strength.

    :param _context: Active Click context.
    :param parameter: Option being validated.
    :param value: Parsed override, or ``None`` when omitted.
    :returns: Validated override.
    :raises click.BadParameter: The override is negative or non-finite.
    """
    if value is not None and (not math.isfinite(value) or value < 0.0):
        raise click.BadParameter(
            "must be finite and greater than or equal to zero", param=parameter
        )
    return value


def apply_cfg_strength_overrides(
    hparams: MutableMapping[str, object],
    content_cfg_strength: float | None,
    sketch_cfg_strength: float | None,
) -> tuple[float, float]:
    """Apply nullable CLI overrides to prediction-time model hyperparameters.

    :param hparams: Mutable Lightning model hyperparameters.
    :param content_cfg_strength: Optional content guidance override.
    :param sketch_cfg_strength: Optional sketch guidance override.
    :returns: Effective content and sketch strengths.
    :raises ValueError: A saved checkpoint strength is invalid.
    """
    saved_content = hparams.get("test_cfg_strength")
    if not isinstance(saved_content, (int, float)) or isinstance(saved_content, bool):
        raise ValueError("checkpoint test_cfg_strength must be numeric")
    effective_content = (
        float(saved_content) if content_cfg_strength is None else content_cfg_strength
    )

    saved_sketch = hparams.get("test_sketch_cfg_strength")
    if sketch_cfg_strength is not None:
        effective_sketch = sketch_cfg_strength
    elif saved_sketch is None:
        effective_sketch = effective_content
    elif isinstance(saved_sketch, (int, float)) and not isinstance(saved_sketch, bool):
        effective_sketch = float(saved_sketch)
    else:
        raise ValueError("checkpoint test_sketch_cfg_strength must be numeric or null")

    for name, strength in (
        ("test_cfg_strength", effective_content),
        ("test_sketch_cfg_strength", effective_sketch),
    ):
        if not math.isfinite(strength) or strength < 0.0:
            raise ValueError(f"checkpoint {name} must be finite and nonnegative")
        hparams[name] = strength
    return effective_content, effective_sketch
