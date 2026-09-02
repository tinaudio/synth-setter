"""Shared runtime CFG override handling for CLAP render modes."""

import math
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from dataclasses import dataclass

import click


@dataclass(frozen=True)
class CfgStrengths[StrengthValue]:
    """Named content and sketch classifier-free guidance strengths.

    .. attribute :: content

        Content-conditioning strength or requested override.

    .. attribute :: sketch

        Sketch-conditioning strength or requested override.
    """

    content: StrengthValue
    sketch: StrengthValue


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


@contextmanager
def temporary_cfg_strength_overrides(
    hparams: MutableMapping[str, object],
    requested: CfgStrengths[float | None],
) -> Iterator[CfgStrengths[float]]:
    """Apply CFG overrides only while one prediction is running.

    :yield CfgStrengths[float]: Effective content and sketch strengths.
    :param hparams: Mutable Lightning model hyperparameters.
    :param requested: Optional content and sketch guidance overrides.
    :raises ValueError: A saved checkpoint strength is invalid.
    """
    saved_content = hparams.get("test_cfg_strength")
    if not isinstance(saved_content, (int, float)) or isinstance(saved_content, bool):
        raise ValueError("checkpoint test_cfg_strength must be numeric")
    effective_content = float(saved_content) if requested.content is None else requested.content

    saved_sketch = hparams.get("test_sketch_cfg_strength")
    if requested.sketch is not None:
        effective_sketch = requested.sketch
    elif saved_sketch is None:
        effective_sketch = effective_content
    elif isinstance(saved_sketch, (int, float)) and not isinstance(saved_sketch, bool):
        effective_sketch = float(saved_sketch)
    else:
        raise ValueError("checkpoint test_sketch_cfg_strength must be numeric or null")

    effective = CfgStrengths(content=effective_content, sketch=effective_sketch)
    names_and_strengths = (
        ("test_cfg_strength", effective.content),
        ("test_sketch_cfg_strength", effective.sketch),
    )
    for name, strength in names_and_strengths:
        if not math.isfinite(strength) or strength < 0.0:
            raise ValueError(f"checkpoint {name} must be finite and nonnegative")

    original = {name: (name in hparams, hparams.get(name)) for name, _ in names_and_strengths}
    try:
        for name, strength in names_and_strengths:
            hparams[name] = strength
        yield effective
    finally:
        for name, _ in names_and_strengths:
            was_present, value = original[name]
            if was_present:
                hparams[name] = value
            else:
                hparams.pop(name, None)
