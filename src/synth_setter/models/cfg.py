"""Classifier-free guidance values shared by model inference callers."""

import math
from dataclasses import dataclass

from beartype import beartype
from jaxtyping import jaxtyped

_MISSING = object()


@dataclass(frozen=True)
class CfgStrengths[StrengthValue]:
    """Named content and sketch classifier-free guidance strengths.

    .. attribute :: content

        Content-conditioning guidance strength.

    .. attribute :: sketch

        Sketch-conditioning guidance strength.
    """

    content: StrengthValue
    sketch: StrengthValue


@jaxtyped(typechecker=beartype)
def resolve_cfg_strengths(
    requested: CfgStrengths[float | None],
    *,
    default_content: object,
    default_sketch: object = _MISSING,
) -> CfgStrengths[float]:
    """Resolve optional guidance values without mutating checkpoint metadata.

    Missing and null legacy sketch defaults follow the effective content value.

    :param requested: Optional runtime content and sketch overrides.
    :param default_content: Checkpoint content guidance value.
    :param default_sketch: Checkpoint sketch guidance value, if present.
    :returns: Finite nonnegative effective guidance strengths.
    """
    content = (
        _validated_strength(default_content, "content")
        if requested.content is None
        else _validated_strength(requested.content, "content")
    )
    sketch = (
        content
        if requested.sketch is None and default_sketch in (_MISSING, None)
        else _validated_strength(
            default_sketch if requested.sketch is None else requested.sketch,
            "sketch",
        )
    )
    return CfgStrengths(content=content, sketch=sketch)


@jaxtyped(typechecker=beartype)
def _validated_strength(value: object, branch: str) -> float:
    """Return one valid guidance value.

    :param value: Candidate numeric guidance value.
    :param branch: Branch name included in validation errors.
    :returns: Finite nonnegative guidance strength.
    :raises ValueError: The value is nonnumeric, negative, or non-finite.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{branch} CFG strength must be numeric")
    strength = float(value)
    if not math.isfinite(strength) or strength < 0.0:
        raise ValueError(f"{branch} CFG strength must be finite and nonnegative")
    return strength
