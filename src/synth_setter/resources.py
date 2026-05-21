"""Locate data files shipped inside the ``synth_setter`` package.

All callsites must use these helpers — never ``Path(__file__).parents[N]``
or repo-relative strings. ``importlib.resources`` keeps the lookup correct
under editable installs, wheels, and zip imports.

:func:`configs_dir` and :func:`vst_headless_wrapper` return ``Traversable``
objects, not real ``Path``\\ s. Callers that hand the result to Hydra,
``subprocess``, or any other API that needs a concrete filesystem path must
wrap the result in :func:`as_file` (re-exported here) so it stays valid
under zipped wheels. ``str(traversable)`` happens to resolve to a real
on-disk path under unpacked-wheel and editable installs, but the
``as_file`` context is the only install-layout-safe way to materialize.

Hydra entrypoints use the ``pkg://`` URI scheme directly
(``@hydra.main(config_path="pkg://synth_setter.configs", ...)``,
``initialize_config_module(config_module="synth_setter.configs")``), so
``configs_dir`` itself is only needed for the small set of helpers that
want a ``Traversable`` handle on the tree.
"""

from __future__ import annotations

from importlib.abc import Traversable
from importlib.resources import as_file, files

__all__ = ["as_file", "configs_dir", "vst_headless_wrapper"]


def configs_dir() -> Traversable:
    """Return the ``synth_setter/configs`` directory as a Traversable.

    Hydra itself uses the ``pkg://synth_setter.configs`` URI scheme — this
    helper is for callers that want a ``Traversable`` handle on the tree
    for iteration or YAML loading. Wrap in :func:`as_file` when a real
    filesystem path is required.

    :returns: Traversable pointing at the shipped Hydra config tree.
    """
    return files("synth_setter") / "configs"


def vst_headless_wrapper() -> Traversable:
    """Return the Xvfb + xsettingsd + dbus wrapper for headless VST init.

    Linux callers materialize this via :func:`as_file` and prepend the
    yielded ``Path`` to the renderer ``argv`` so the VST3 plugin gets a
    display before pedalboard imports it. ``str(vst_headless_wrapper())``
    only works under unpacked-wheel installs — use :func:`as_file` to
    survive a zipped wheel.

    :returns: Traversable pointing at ``run-linux-vst-headless.sh``.
    """
    return files("synth_setter") / "scripts" / "run-linux-vst-headless.sh"
