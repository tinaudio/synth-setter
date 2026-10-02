"""Typed lazy boundary for the pinned KR106 native extension."""

from importlib import import_module
from importlib.metadata import version
from typing import Literal, Protocol, cast

import numpy as np
from pydantic import BaseModel, ConfigDict

KR106_SOURCE_REVISION = "bc15caee5843ab238a25d0969e68d57db2b1615f"
KR106_SYNTH_VERSION = "2.5.13"
KR106_BACKEND_VERSION = "0.1.0"


class NativeParameter(BaseModel):
    """One control descriptor exported by the native engine.

    .. attribute :: model_config
        Strict immutable metadata schema.
    .. attribute :: id
        Engine dispatch identifier.
    .. attribute :: name
        Stable engine control name.
    .. attribute :: minimum
        Inclusive native lower bound.
    .. attribute :: maximum
        Inclusive native upper bound.
    .. attribute :: kind
        Continuous, integer, or switch conversion policy.
    .. attribute :: default
        Native default before baseline application.
    """

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    id: int
    name: str
    minimum: float
    maximum: float
    kind: Literal["float", "int", "bool"]
    default: float


class NativeBaseline(BaseModel):
    """Versioned native parameters extracted from a committed VST preset.

    .. attribute :: model_config
        Strict immutable baseline schema.
    .. attribute :: schema_version
        Baseline serialization version.
    .. attribute :: source_revision
        Pinned upstream DSP revision.
    .. attribute :: source_vstpreset_sha256
        Digest of the original VST preset.
    .. attribute :: parameters
        DSP-indexed native baseline values.
    """

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    schema_version: Literal[1]
    source_revision: Literal["bc15caee5843ab238a25d0969e68d57db2b1615f"]
    source_vstpreset_sha256: str
    parameters: dict[str, float]


class KR106NativeModule(Protocol):
    """Small extension surface used by the dataset and evaluation adapter."""

    def get_version(self) -> str:
        """Return the embedded synth version.

        :returns: KR106 source version.
        """
        ...

    def get_source_revision(self) -> str:
        """Return the exact source revision compiled into the extension.

        :returns: Full upstream git commit hash.
        """
        ...

    def get_parameters(self) -> list[dict[str, object]]:
        """Describe the extension's native controls.

        :returns: Identity, range, type and default metadata for each control.
        """
        ...

    def render_note(
        self,
        parameters: dict[int, float],
        midi_note: int,
        velocity: int,
        start_sample: int,
        end_sample: int,
        num_samples: int,
        sample_rate: float,
        block_size: int,
        *,
        baseline_parameters: dict[int, float],
    ) -> np.ndarray:
        """Render one isolated clip into a NumPy-owned buffer.

        :param parameters: Explicit DSP-native control overrides.
        :param midi_note: MIDI pitch.
        :param velocity: MIDI velocity.
        :param start_sample: Inclusive note-on frame.
        :param end_sample: Note-off frame.
        :param num_samples: Exact clip length in frames.
        :param sample_rate: Sample rate in Hz.
        :param block_size: Maximum DSP processing span.
        :param baseline_parameters: Native baseline applied before program selection and overrides.
        :returns: Channel-leading stereo float32 audio.
        """
        ...


def import_kr106_native() -> KR106NativeModule:
    """Import and verify the source pin before creating any native engine.

    :returns: Extension built from the supported KR106 source revision.
    :raises RuntimeError: The extension is missing or built from another source.
    """
    try:
        native = cast(KR106NativeModule, import_module("kr106_native"))
    except ImportError as exc:
        raise RuntimeError("KR106-native rendering requires the kr106-native package") from exc
    if (
        version("kr106-native") != KR106_BACKEND_VERSION
        or native.get_version() != KR106_SYNTH_VERSION
        or native.get_source_revision() != KR106_SOURCE_REVISION
    ):
        raise RuntimeError("KR106-native extension version/source revision does not match its pin")
    return native
