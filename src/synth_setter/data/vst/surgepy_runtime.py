"""Typed lazy boundary for the native SurgePy extension.

Usage::

    surgepy = import_surgepy()
    synth = surgepy.createSurge(44_100)
"""

from __future__ import annotations

from collections.abc import Generator
from importlib import import_module
from pathlib import Path
from struct import unpack_from
from typing import Protocol, cast

import numpy as np

from synth_setter.renderer_backend import RendererBackend

# git never abbreviates below seven digits, so a shorter stamp cannot name a commit.
_MIN_ABBREVIATED_COMMIT = 7
_HEX_DIGITS = frozenset("0123456789abcdef")


class SurgePyParameterId(Protocol):
    """Stable synth-side parameter identifier exposed by SurgePy."""

    def getSynthSideId(self) -> int: ...


class SurgePyNamedParam(Protocol):
    """Named Surge parameter handle accepted by the native engine."""

    def getId(self) -> SurgePyParameterId: ...

    def getName(self) -> str: ...


class SurgePySynth(Protocol):
    """SurgePy synthesizer surface used by the renderer."""

    def getBlockSize(self) -> int: ...

    def getPatch(self) -> dict[str, object]: ...

    def loadPatch(self, path: str) -> bool: ...

    def getParamMax(self, parameter: SurgePyNamedParam) -> float: ...

    def getParamMin(self, parameter: SurgePyNamedParam) -> float: ...

    def getParamValType(self, parameter: SurgePyNamedParam) -> str: ...

    def setParamVal(self, parameter: SurgePyNamedParam, value: float) -> None: ...

    def playNote(
        self,
        channel: int,
        midiNote: int,
        velocity: int,
        detune: int = 0,
    ) -> None: ...

    def releaseNote(
        self,
        channel: int,
        midiNote: int,
        releaseVelocity: int = 0,
    ) -> None: ...

    def allNotesOff(self) -> None: ...

    def createMultiBlock(self, blockCapacity: int) -> np.ndarray: ...

    def processMultiBlock(
        self,
        val: np.ndarray,
        startBlock: int = 0,
        nBlocks: int = -1,
    ) -> None: ...


class SurgePyModule(Protocol):
    """Module-level SurgePy constructors used by production code."""

    def createSurge(self, sampleRate: float) -> SurgePySynth: ...

    def getVersion(self) -> str: ...


def surge_component_state(path: Path) -> bytes:
    """Extract native Surge state from a VST3-preset or FXP container.

    :param path: Surge preset container.
    :returns: Native ``sub3`` component bytes shared across formats.
    :raises ValueError: If the container does not hold Surge component state.
    """
    data = path.read_bytes()
    if path.suffix.casefold() == ".fxp":
        if len(data) < 60 or data[:4] != b"CcnK" or data[8:12] != b"FPCh":
            raise ValueError(f"{path} is not an FXP chunk container")
        chunk_size = unpack_from(">I", data, 56)[0]
        component = data[60:]
        if chunk_size != len(component) or not component.startswith(b"sub3"):
            raise ValueError(f"{path} has invalid FXP component bounds")
        return component

    if len(data) < 48 or data[:4] != b"VST3":
        raise ValueError(f"{path} is not a VST3 preset container")
    list_offset = unpack_from("<Q", data, 40)[0]
    if list_offset + 8 > len(data) or data[list_offset : list_offset + 4] != b"List":
        raise ValueError(f"{path} has no valid VST3 chunk list")
    entry_count = unpack_from("<I", data, list_offset + 4)[0]
    if list_offset + 8 + entry_count * 20 > len(data):
        raise ValueError(f"{path} has truncated VST3 chunk entries")
    for index in range(entry_count):
        entry_offset = list_offset + 8 + index * 20
        if data[entry_offset : entry_offset + 4] != b"Comp":
            continue
        offset, size = unpack_from("<QQ", data, entry_offset + 4)
        component = data[offset : offset + size]
        terminator = b"JUCEPrivateData"
        if offset + size > len(data) or not component.endswith(terminator):
            raise ValueError(f"{path} has invalid VST3 component bounds")
        component = component[: -len(terminator)]
        if not component.startswith(b"sub3"):
            raise ValueError(f"{path} has no Surge component state")
        return component
    raise ValueError(f"{path} has no VST3 component chunk")


def import_surgepy() -> SurgePyModule:
    """Import the optional native extension only when the backend is selected.

    :returns: Typed SurgePy module.
    :raises RuntimeError: If the native extension cannot be imported.
    """
    try:
        return cast(SurgePyModule, import_module("surgepy"))
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "SurgePy backend requires the pinned surgepy native extension"
        ) from exc


def _is_abbreviated_commit(component: str) -> bool:
    """Judge whether a version component identifies a commit.

    :param component: Trailing dot-separated component of a version stamp.
    :returns: Whether the component is a lowercase hex abbreviation git would emit.
    """
    return len(component) >= _MIN_ABBREVIATED_COMMIT and all(
        character in _HEX_DIGITS for character in component
    )


def surgepy_version_matches(actual: str, configured: str) -> bool:
    """Judge whether two SurgePy stamps name the same build.

    Surge bakes ``git rev-parse --short`` into the extension at build time, and that
    abbreviation widens with the object count of whichever clone produced the wheel,
    so one pinned commit reaches us as both ``f7b97c68`` and ``f7b97c682``.

    :param actual: Stamp reported by the installed extension.
    :param configured: Stamp pinned by the synth identity group.
    :returns: Whether both stamps identify the same Surge build.
    """
    if actual == configured:
        return True
    actual_release, _, actual_commit = actual.rpartition(".")
    configured_release, _, configured_commit = configured.rpartition(".")
    if not actual_release or actual_release != configured_release:
        return False
    if not (_is_abbreviated_commit(actual_commit) and _is_abbreviated_commit(configured_commit)):
        return False
    shorter, longer = sorted((actual_commit, configured_commit), key=len)
    return longer.startswith(shorter)


def ensure_surgepy_runtime(
    renderer_backend: RendererBackend,
    renderer_version: str | None = None,
) -> None:
    """Validate the selected SurgePy runtime and optional version stamp.

    :param renderer_backend: Renderer selected by the validated render config.
    :param renderer_version: Expected live engine version, when supplied.
    :raises RuntimeError: If SurgePy is unavailable or its version differs.
    """
    if renderer_backend != "surgepy":
        return
    actual = import_surgepy().getVersion()
    if renderer_version and not surgepy_version_matches(actual, renderer_version):
        raise RuntimeError(
            f"SurgePy version {actual!r} does not match configured {renderer_version!r}"
        )


def iter_surgepy_named_params(
    value: object,
) -> Generator[SurgePyNamedParam, None, None]:
    """Yield every native parameter handle from SurgePy's nested patch tree.

    :param value: Patch node returned by ``SurgeSynthesizer.getPatch()``.
    :yields: Parameter handles in patch order.
    :ytype: SurgePyNamedParam
    """
    if hasattr(value, "getId") and hasattr(value, "getName"):
        yield cast(SurgePyNamedParam, value)
        return
    if isinstance(value, dict):
        for child in value.values():
            yield from iter_surgepy_named_params(child)
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            yield from iter_surgepy_named_params(child)
