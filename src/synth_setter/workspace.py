"""Operator-side workspace anchor for launcher modules.

Resolves where launchers write the spec mirror and what
``${oc.env:PROJECT_ROOT}`` points at in ``configs/paths/default.yaml``;
the precedence is documented on :func:`operator_workspace`. Distinct
from :mod:`synth_setter.resources`, which locates package-shipped assets
via :mod:`importlib.resources`.
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path

from loguru import logger

__all__ = ["operator_workspace"]

# .project-root marker, tracked at the checkout root so editable installs
# resolve via the parents[]-walk below.
_CHECKOUT_MARKER = ".project-root"

_WORKSPACE_ENV = "SYNTH_SETTER_WORKSPACE"


def _checkout_root() -> Path | None:
    """Return the first ancestor of this file containing ``.project-root``."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / _CHECKOUT_MARKER).is_file():
            return candidate
    return None


@cache
def operator_workspace() -> Path:
    """Resolve the operator workspace; publish ``$PROJECT_ROOT`` if unset.

    Order: ``$SYNTH_SETTER_WORKSPACE`` → checkout reachable from
    ``__file__`` (editable / in-repo) → ``Path.cwd()`` (packaged install,
    logs a warning since the path is likely unintended). A pre-set
    ``$PROJECT_ROOT`` is preserved — operators who set both keep theirs.

    The ``$PROJECT_ROOT`` publication happens once per process via
    ``@cache``; subsequent callers who ``del os.environ["PROJECT_ROOT"]``
    will not see it re-published. Launchers therefore call this once at
    module import, before any Hydra compose.

    :returns: Absolute, resolved workspace path.
    """
    override = os.environ.get(_WORKSPACE_ENV)
    if override:
        workspace = Path(override).resolve()
    else:
        checkout = _checkout_root()
        if checkout is not None:
            workspace = checkout
        else:
            workspace = Path.cwd().resolve()
            logger.warning(
                "no checkout marker reachable; using cwd as workspace: {}. Set ${} to override.",
                workspace,
                _WORKSPACE_ENV,
            )
    os.environ.setdefault("PROJECT_ROOT", str(workspace))
    return workspace
