"""Operator-side workspace anchor for launcher modules.

The console-script entries (``synth-setter-generate-dataset``,
``synth-setter-finalize-dataset``, ``synth-setter-train``,
``synth-setter-eval``) need *somewhere* to write the launcher-side spec
mirror and to resolve ``configs/paths/default.yaml``'s
``${oc.env:PROJECT_ROOT}`` interpolation. Under a checkout this was the
repo root; under a wheel install no such root exists, so we resolve
explicitly and document the fallback.

Distinct from :mod:`synth_setter.resources` — that module locates
*package-shipped* assets (configs, scripts) via ``importlib.resources``.
This module locates the *operator's* workspace (where artifacts get
written, where ``.env`` lives, where local spec mirrors land).
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path

from loguru import logger

__all__ = ["operator_workspace"]

# Marker file used by ``rootutils`` to identify a checkout root. Kept
# tracked at the repo root so the checkout-detection branch below stays
# truthy under ``pip install -e .``.
_CHECKOUT_MARKER = ".project-root"

# Override env var. Operators on a wheel install (or anyone who wants the
# workspace to live somewhere other than CWD / the checkout) sets this.
_WORKSPACE_ENV = "SYNTH_SETTER_WORKSPACE"


def _checkout_root() -> Path | None:
    """Return the synth-setter checkout root if reachable from this file.

    ``parents[2]`` from ``src/synth_setter/workspace.py`` is the checkout;
    we walk up regardless and stop at the first ``.project-root`` so
    namespace-package layouts and rare reorganizations still resolve.
    """
    for candidate in Path(__file__).resolve().parents:
        if (candidate / _CHECKOUT_MARKER).is_file():
            return candidate
    return None


@cache
def operator_workspace() -> Path:
    """Resolve the operator's workspace directory.

    Resolution order:

    1. ``$SYNTH_SETTER_WORKSPACE`` if set (explicit operator override).
    2. The synth-setter checkout root if reachable from ``__file__``
       (editable install or in-repo invocation).
    3. ``Path.cwd()`` as the last resort (packaged install with no
       override).

    Sets ``os.environ["PROJECT_ROOT"]`` to the resolved path as a side
    effect so ``configs/paths/default.yaml``'s ``${oc.env:PROJECT_ROOT}``
    interpolation resolves under any install layout. Existing
    ``PROJECT_ROOT`` values are preserved — operators who set both win.

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
            logger.info(
                "no checkout marker reachable; using cwd as workspace: {}. Set ${} to override.",
                workspace,
                _WORKSPACE_ENV,
            )
    os.environ.setdefault("PROJECT_ROOT", str(workspace))
    return workspace
