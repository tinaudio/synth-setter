"""Locate data files shipped inside the ``synth_setter`` package.

All callsites must use these helpers — never ``Path(__file__).parents[N]``,
``str(traversable)`` cast to ``Path``, or repo-relative strings.
``importlib.resources`` keeps the lookup correct under every install layout
Python's import machinery accepts: editable installs, unpacked wheels, zipped
wheels / zipapps, and namespace-package multi-sources.

The helpers return ``Traversable`` objects, not real ``Path``\\ s. The
``Traversable`` protocol promises ``iterdir``, ``is_file``, ``is_dir``,
``joinpath``, ``/``, ``name``, ``open``, ``read_bytes``, ``read_text`` — and
nothing else. In particular it does NOT promise ``__fspath__``, ``glob``,
``stem``, or ``suffix`` — those exist only on the concrete ``Path`` subclass
that backs unpacked installs.

Two patterns survive every layout:

1. **Stay on the Traversable API.** For directory iteration use
   ``iterdir()`` + a filter on ``name``; for reading a packaged YAML use
   ``read_text()`` straight into the parser
   (``OmegaConf.create(traversable.read_text())``).

2. **Materialize via** :func:`as_file` **(re-exported here).** When a third
   party API only accepts a filesystem path — ``subprocess``, anything that
   does its own ``open(path)`` — wrap the Traversable in ``as_file`` and hold
   the resulting context open across the API call::

       with as_file(vst_headless_wrapper()) as wrapper_path:
           subprocess.check_call([str(wrapper_path), ...])

   Under zipped installs ``as_file`` extracts to a tempfile and cleans up on
   exit; under unpacked installs it's a no-op that returns the on-disk path.
   :class:`contextlib.ExitStack` is the right tool when multiple resources
   need to outlive a single ``with`` block.

Hydra entrypoints use the ``pkg://`` URI scheme directly
(``@hydra.main(config_path="pkg://synth_setter.configs", ...)``,
``initialize_config_module(config_module="synth_setter.configs")``), so
:func:`configs_dir` itself is only needed for the small set of helpers that
want a ``Traversable`` handle on the tree.
"""

from __future__ import annotations

from importlib.abc import Traversable
from importlib.resources import as_file, files

__all__ = [
    "as_file",
    "configs_dir",
    "generate_vst_dataset_script",
    "vst_headless_wrapper",
]


def configs_dir() -> Traversable:
    """Return the ``synth_setter/configs`` directory as a Traversable.

    Hydra itself uses the ``pkg://synth_setter.configs`` URI scheme — this
    helper is for callers that want a ``Traversable`` handle on the tree for
    iteration (``iterdir()``) or text loading (``read_text()``). Wrap in
    :func:`as_file` when a real filesystem path is required.

    :returns: Traversable pointing at the shipped Hydra config tree.
    """
    return files("synth_setter") / "configs"


def vst_headless_wrapper() -> Traversable:
    """Return the Xvfb + xsettingsd + dbus wrapper for headless VST init.

    Linux callers materialize this via :func:`as_file` and prepend the yielded
    ``Path`` to the renderer ``argv`` so the VST3 plugin gets a display before
    pedalboard imports it.

    :returns: Traversable pointing at ``run-linux-vst-headless.sh``.
    """
    return files("synth_setter") / "scripts" / "run-linux-vst-headless.sh"


def generate_vst_dataset_script() -> Traversable:
    """Return the ``generate_vst_dataset.py`` worker script.

    Invoked as a subprocess by :func:`synth_setter.cli.generate_dataset.run`
    once per shard; callers materialize via :func:`as_file` and pass the
    yielded ``Path`` as the ``python <script>`` argument. The previous
    cwd-relative invocation (``"src/synth_setter/data/vst/..."``) only worked
    when the launcher ran from a repo checkout.

    :returns: Traversable pointing at ``data/vst/generate_vst_dataset.py``.
    """
    return files("synth_setter") / "data" / "vst" / "generate_vst_dataset.py"
