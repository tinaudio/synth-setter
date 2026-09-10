"""Offline FaustWasm renderer backed by bounded Node subprocesses."""

from __future__ import annotations

import json
import math
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from synth_setter.data.vst.faust_sources import resolve_faust_dsp
from synth_setter.data.vst.faustwasm_artifacts import (
    ArtifactManifest,
    compile_faustwasm_artifact,
    run_faustwasm_render_worker,
)
from synth_setter.data.vst.renderers import (
    AudioRenderer,
    ParameterValue,
    _validate_rendered_audio,
    require_scalar_synth_params,
)
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.synth_spec import (
    SYNTHS,
    SynthName,
    validate_faust_registry_reference,
)


def _quantize_note_window(
    note_start_and_end: tuple[float, float],
    *,
    sample_rate: float,
    frames: int,
    signal_duration_seconds: float,
) -> tuple[int, int]:
    """Map a valid half-open time window to every intersecting output frame.

    :param note_start_and_end: Note-on and note-off times in seconds.
    :param sample_rate: Output sample rate in Hz.
    :param frames: Fixed output frame count.
    :param signal_duration_seconds: Configured render duration in seconds.
    :returns: Half-open integer frame window clamped to the fixed output.
    :raises ValueError: Times are malformed, out of bounds, or only cover a discarded tail.
    """
    start, end = note_start_and_end
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, (int, float))
        or not isinstance(end, (int, float))
        or not math.isfinite(start)
        or not math.isfinite(end)
        or not 0.0 <= start < end <= signal_duration_seconds
    ):
        raise ValueError("note times must satisfy 0 <= start < end <= signal duration")
    if isinstance(frames, bool) or not isinstance(frames, int) or frames < 1:
        raise ValueError("frames must be a positive integer")

    start_frame = min(math.floor(start * sample_rate), frames)
    end_frame = min(math.ceil(end * sample_rate), frames)
    if start_frame >= end_frame:
        raise ValueError("note times do not overlap any output frame")
    return start_frame, end_frame


@dataclass(kw_only=True)
class FaustWasmRenderer(AudioRenderer):
    """Compile a checked-in Faust source once and render isolated Node processes.

    .. attribute :: block_size
       :type: int

       Node offline-processing block size.
    .. attribute :: param_spec_name
       :type: ParamSpecName

       Checked-in Faust source identity.
    .. attribute :: source_sha256
       :type: str

       Required source digest.
    .. attribute :: backend_version
       :type: str

       Required FaustWasm package version.
    """

    block_size: int = field(kw_only=True)
    param_spec_name: ParamSpecName = field(kw_only=True)
    source_sha256: str = field(kw_only=True)
    backend_version: str = field(kw_only=True)
    _temporary_directory: tempfile.TemporaryDirectory[str] = field(init=False, repr=False)
    _manifest: ArtifactManifest = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate renderer dimensions before compiling the shared artifact.

        :raises ValueError: Renderer provenance or dimensions violate the backend contract.
        """
        if self.plugin_state_path:
            raise ValueError("FaustWasm renderer accepts no preset path")
        source_identity = self.param_spec_name
        if self.plugin_path:
            source_identity = validate_faust_registry_reference(
                self.plugin_path,
                self.param_spec_name,
            )
        if isinstance(self.block_size, bool) or not isinstance(self.block_size, int) or self.block_size < 1:
            raise ValueError("block_size must be a positive integer")
        if not math.isfinite(self.sample_rate) or self.sample_rate <= 0:
            raise ValueError("sample_rate must be finite and positive")
        if not math.isfinite(self.signal_duration_seconds) or self.signal_duration_seconds <= 0:
            raise ValueError("signal_duration_seconds must be finite and positive")
        frames = int(self.sample_rate * self.signal_duration_seconds)
        if frames < 1:
            raise ValueError("render duration must contain at least one output frame")

        synth = SYNTHS[SynthName(source_identity)]
        expected_channels = resolve_faust_dsp(source_identity).outputs
        if self.channels != expected_channels:
            raise ValueError(f"FaustWasm source requires channels={expected_channels}")
        if synth.source_sha256 != self.source_sha256:
            raise ValueError("source_sha256 does not match the registered Faust source")
        self._temporary_directory = tempfile.TemporaryDirectory(prefix="synth-setter-faustwasm-")
        self._manifest = compile_faustwasm_artifact(
            synth,
            Path(self._temporary_directory.name),
            backend_version=self.backend_version,
        )

    def _validate_patch(self, params: dict[str, float]) -> None:
        """Reject incomplete, unknown, non-finite, or out-of-domain values.

        :param params: Complete canonical scalar patch.
        :raises KeyError: The patch contains an unknown canonical address.
        :raises ValueError: The patch is incomplete or contains an invalid value.
        """
        parameters = {item.canonicalAddress: item for item in self._manifest.parameters}
        unknown = sorted(params.keys() - parameters.keys())
        if unknown:
            raise KeyError(f"unknown canonical parameter(s): {', '.join(unknown)}")
        missing = sorted(parameters.keys() - params.keys())
        if missing:
            raise ValueError(f"missing canonical parameter(s): {', '.join(missing)}")
        for address, value in params.items():
            parameter = parameters[address]
            if not np.isfinite(value) or not parameter.min <= value <= parameter.max:
                raise ValueError(f"parameter outside native domain: {address}")
            if parameter.kind == "discrete" and (
                parameter.values is None or value not in parameter.values
            ):
                raise ValueError(f"parameter outside discrete native domain: {address}")

    def render(
        self,
        params: Mapping[str, ParameterValue],
        midi_note: int,
        velocity: int,
        note_start_and_end: tuple[float, float],
        *,
        warmup: bool = False,
    ) -> np.ndarray:
        """Render a complete canonical patch into channel-major float32 audio.

        :param params: Complete canonical parameter patch.
        :param midi_note: MIDI note number.
        :param velocity: MIDI velocity.
        :param note_start_and_end: Note-on and note-off times in seconds.
        :param warmup: Ignored because every render starts an isolated processor.
        :returns: Channel-major float32 audio.
        """
        del warmup
        scalar_params = require_scalar_synth_params(params)
        self._validate_patch(scalar_params)
        frames = int(self.sample_rate * self.signal_duration_seconds)
        start_frame, end_frame = _quantize_note_window(
            note_start_and_end,
            sample_rate=self.sample_rate,
            frames=frames,
            signal_duration_seconds=self.signal_duration_seconds,
        )
        request = {
            "expectedFaustWasmVersion": self.backend_version,
            "sampleRate": self.sample_rate,
            "blockSize": self.block_size,
            "frames": frames,
            "note": midi_note,
            "velocity": velocity,
            "startFrame": start_frame,
            "endFrame": end_frame,
            "params": scalar_params,
        }
        directory = Path(self._temporary_directory.name)
        request_path = directory / "render-request.json"
        output_path = directory / "audio.f32"
        request_path.write_text(json.dumps(request))
        run_faustwasm_render_worker(directory, request_path, output_path)
        audio = np.fromfile(output_path, dtype="<f4").reshape(self.channels, frames)
        return _validate_rendered_audio(
            audio,
            channels=self.channels,
            samples=frames,
        )
