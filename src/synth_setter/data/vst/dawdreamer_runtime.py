"""Execution-time capability checks for the optional DawDreamer backend."""

from __future__ import annotations

import platform
import sys
from importlib import import_module
from importlib.metadata import version as distribution_version
from typing import Protocol

from synth_setter.renderer_backend import RendererBackend as RendererBackend

# Eight callbacks cover Surge's observed two-stage identity publication across tested block sizes.
_PRESET_SETTLE_CYCLES = 8
_SUPPORTED_PYTHON_MINOR = (3, 12)
_SUPPORTED_TARGETS = {
    ("Darwin", "arm64"),
    ("Darwin", "x86_64"),
    ("Linux", "x86_64"),
    ("Windows", "AMD64"),
}


class DawDreamerEngine(Protocol):
    """Engine operation needed to settle asynchronous VST preset state."""

    def render(self, duration: float) -> object:
        """Process audio for a duration.

        :param duration: Processing duration in seconds.
        :returns: Backend-specific render status.
        """
        ...


def settle_dawdreamer_preset(
    engine: DawDreamerEngine,
    *,
    sample_rate: float,
    block_size: int,
) -> None:
    """Activate preset-dependent parameter identities before parameter writes.

    :param engine: DawDreamer engine whose graph contains the preset-loaded plugin.
    :param sample_rate: Engine sample rate in Hz.
    :param block_size: Engine processing block size in samples.
    """
    block_duration = block_size / sample_rate
    for _ in range(_PRESET_SETTLE_CYCLES):
        engine.render(block_duration)


def ensure_dawdreamer_runtime(
    renderer_backend: RendererBackend,
    renderer_version: str | None = None,
) -> None:
    """Fail early when a DawDreamer worker cannot load the pinned package.

    The check belongs on the process that renders audio. A launcher may run a newer Python while
    dispatching to a compatible worker interpreter.

    :param renderer_backend: Backend selected for the current render process.
    :param renderer_version: Optional required DawDreamer distribution version.
    :raises RuntimeError: DawDreamer is unsupported or unavailable on this worker.
    :raises ValueError: The required renderer version differs from the installed package.
    """
    if renderer_backend not in ("dawdreamer", "dawdreamer_faust"):
        return

    python_minor = sys.version_info[:2]
    if sys.implementation.name != "cpython" or python_minor != _SUPPORTED_PYTHON_MINOR:
        detected = f"{sys.implementation.name} {python_minor[0]}.{python_minor[1]}"
        raise RuntimeError(
            "DawDreamer 0.8.3 requires CPython 3.12 on the render worker; "
            f"detected {detected}. Dispatch to a supported worker or recreate this "
            "environment with Python 3.12."
        )

    target = (platform.system(), platform.machine())
    if target not in _SUPPORTED_TARGETS:
        supported = "Darwin/arm64, Darwin/x86_64, Linux/x86_64, or Windows/AMD64"
        raise RuntimeError(
            f"DawDreamer 0.8.3 has no wheel for {target[0]}/{target[1]}; supported "
            f"worker targets are {supported}."
        )

    try:
        import_module("dawdreamer")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "DawDreamer could not be loaded on this render worker. Run `uv sync --frozen` "
            "under CPython 3.12 and verify the worker matches a supported wheel target."
        ) from exc

    if renderer_version is not None:
        installed_version = distribution_version("dawdreamer")
        if installed_version != renderer_version:
            raise ValueError(
                f"DawDreamer renderer version {renderer_version!r} does not match "
                f"installed version {installed_version!r}"
            )
