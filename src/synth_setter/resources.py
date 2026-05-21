"""Locate data files shipped inside the ``synth_setter`` package.

All callsites must use these helpers — never ``Path(__file__).parents[N]``
or repo-relative strings. ``importlib.resources`` keeps the lookup correct
under editable installs, wheels, and zip imports.

The :func:`configs_dir` and :func:`vst_headless_wrapper` helpers return
``Traversable`` objects. ``str()`` works directly for modern pip layouts
that unpack wheels to the filesystem; wrap in :func:`as_file` (re-exported
here) when handing a path to a subprocess that must open a real file.
"""

from __future__ import annotations

from importlib.abc import Traversable
from importlib.resources import as_file, files

__all__ = ["as_file", "configs_dir", "vst_headless_wrapper"]


def configs_dir() -> Traversable:
    """Return the ``synth_setter/configs`` directory as a Traversable.

    Hydra wants a real filesystem path (``initialize_config_dir`` /
    ``config_path=``); wrap the result in :func:`as_file` if running from
    a zipped wheel, or call ``str()`` for the common unpacked-wheel case.

    :returns: Traversable pointing at the shipped Hydra config tree.
    """
    return files("synth_setter") / "configs"


def vst_headless_wrapper() -> Traversable:
    """Return the Xvfb + xsettingsd + dbus wrapper for headless VST init.

    Linux callers prepend ``str(vst_headless_wrapper())`` (or the path
    yielded by :func:`as_file`) to the renderer ``argv`` so the VST3
    plugin gets a display before pedalboard imports it.

    :returns: Traversable pointing at ``run-linux-vst-headless.sh``.
    """
    return files("synth_setter") / "scripts" / "run-linux-vst-headless.sh"
