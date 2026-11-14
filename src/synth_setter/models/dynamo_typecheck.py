"""Let ``torch.compile`` trace through jaxtyping's runtime shape checks.

``jaxtyped`` binds axis sizes in a thread-local memo stack it pushes on entry and pops in a
``finally`` on exit. Dynamo re-executes a traced frame after a graph break without unwinding
that stack, so the next pass type-checks against a desynchronized memo and rejects arguments
that satisfy their own annotation — see
https://github.com/tinaudio/synth-setter/issues/3572. Checking during tracing buys nothing
regardless: Dynamo already guards the compiled graph on dtype and shape.

Every ``jaxtyped`` function reached while tracing has to bypass, so this flips jaxtyping's own
``jaxtyping_disable`` switch — which its wrapper reads per call — rather than decorating call
sites. Eager execution keeps full type checking.
"""

from __future__ import annotations

from typing import Final

import torch
from beartype import beartype
from jaxtyping import _config as jaxtyping_config
from jaxtyping import jaxtyped

_FLAG: Final = "jaxtyping_disable"
_EXPLICIT_FLAG: Final = "_synth_setter_explicit_jaxtyping_disable"


@jaxtyped(typechecker=beartype)
def install_dynamo_typecheck_bypass() -> None:
    """Make jaxtyping's disable switch read ``True`` for the duration of Dynamo tracing.

    Idempotent, and leaves any explicit ``JAXTYPING_DISABLE`` setting readable and writable
    through the same attribute.

    :raises RuntimeError: If jaxtyping no longer stores the switch on its config instance.
    """
    config = jaxtyping_config.config
    config_type = type(config)
    if isinstance(vars(config_type).get(_FLAG), property):
        return
    if _FLAG not in vars(config):
        raise RuntimeError(
            f"jaxtyping no longer stores {_FLAG!r} on its config instance, so the Dynamo "
            "type-check bypass cannot be installed"
        )

    setattr(config, _EXPLICIT_FLAG, vars(config).pop(_FLAG))
    setattr(
        config_type,
        _FLAG,
        property(
            lambda self: torch.compiler.is_compiling() or getattr(self, _EXPLICIT_FLAG),
            lambda self, value: setattr(self, _EXPLICIT_FLAG, value),
        ),
    )
