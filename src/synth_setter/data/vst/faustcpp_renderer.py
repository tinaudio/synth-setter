"""Offline native C++ renderer for checked-in Faust sources."""

from __future__ import annotations

import hashlib
import math
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from synth_setter.data.vst.core import extract_backend_version
from synth_setter.data.vst.faust_param_spec import resolve_faust_param_spec
from synth_setter.data.vst.faust_sources import resolve_faust_dsp
from synth_setter.data.vst.faustwasm_renderer import _quantize_note_window
from synth_setter.data.vst.param_spec import CategoricalParameter, ContinuousParameter
from synth_setter.data.vst.renderers import (
    AudioRenderer,
    ParameterValue,
    _validate_rendered_audio,
    require_scalar_synth_params,
)
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.synth_spec import validate_faust_registry_reference

_COMPILE_TIMEOUT_SECONDS = 120
_RENDER_TIMEOUT_SECONDS = 60


def _native_address(identity: ParamSpecName, canonical_address: str) -> str:
    """Map a stable dataset address to the native C++ MapUI address.

    :param identity: Checked-in Faust identity.
    :param canonical_address: Stable address used by the parameter specification.
    :returns: Address emitted by the native Faust compiler.
    """
    if identity == "faust_bright_organ":
        return canonical_address.removeprefix("/Sequencer/DSP1")
    return canonical_address


@dataclass(kw_only=True)
class FaustCppRenderer(AudioRenderer):
    """Compile one Faust source to a native executable and isolate every render.

    .. attribute :: block_size

       Native offline processing block size.

    .. attribute :: param_spec_name

       Checked-in Faust source identity.

    .. attribute :: source_sha256

       Required source digest.

    .. attribute :: backend_version

       Required Faust CLI version.
    """

    block_size: int = field(kw_only=True)
    param_spec_name: ParamSpecName = field(kw_only=True)
    source_sha256: str = field(kw_only=True)
    backend_version: str = field(kw_only=True)
    _temporary_directory: tempfile.TemporaryDirectory[str] = field(init=False, repr=False)
    _executable: Path = field(init=False, repr=False)
    _identity: ParamSpecName = field(init=False, repr=False)
    _parameters: dict[str, ContinuousParameter | CategoricalParameter] = field(
        init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Validate provenance and compile the native runner once per renderer.

        :raises ValueError: Renderer configuration or source provenance is invalid.
        :raises RuntimeError: The native toolchain is missing, mismatched, or compilation fails.
        :raises TypeError: The parameter specification contains an unsupported domain.
        """
        if self.plugin_state_path:
            raise ValueError("Faust C++ renderer accepts no preset path")
        self._identity = self.param_spec_name
        if self.plugin_path:
            self._identity = validate_faust_registry_reference(
                self.plugin_path, self.param_spec_name
            )
        if isinstance(self.block_size, bool) or not isinstance(self.block_size, int) or self.block_size < 1:
            raise ValueError("block_size must be a positive integer")
        if (
            not math.isfinite(self.sample_rate)
            or self.sample_rate <= 0
            or not float(self.sample_rate).is_integer()
        ):
            raise ValueError("sample_rate must be a positive whole number")
        if not math.isfinite(self.signal_duration_seconds) or self.signal_duration_seconds <= 0:
            raise ValueError("signal_duration_seconds must be finite and positive")
        if type(self.channels) is not int or self.channels < 1:
            raise ValueError("channels must be a positive integer")

        dsp = resolve_faust_dsp(self._identity)
        if self.channels != dsp.outputs:
            raise ValueError(f"Faust C++ source requires channels={dsp.outputs}")
        if hashlib.sha256(dsp.source.encode()).hexdigest() != self.source_sha256:
            raise ValueError("source_sha256 does not match the registered Faust source")
        installed_version = extract_backend_version("faustcpp")
        if installed_version != self.backend_version:
            raise RuntimeError(
                f"Faust CLI version {installed_version!r} != configured {self.backend_version!r}"
            )
        parameters = resolve_faust_param_spec(self._identity).synth_params
        if not all(
            isinstance(parameter, (ContinuousParameter, CategoricalParameter))
            for parameter in parameters
        ):
            raise TypeError("Faust C++ supports only continuous and categorical parameters")
        self._parameters = {
            parameter.name: parameter
            for parameter in parameters
            if isinstance(parameter, (ContinuousParameter, CategoricalParameter))
        }
        self._temporary_directory = tempfile.TemporaryDirectory(prefix="synth-setter-faustcpp-")
        self._executable = self._compile(dsp.source)

    def _compile(self, source: str) -> Path:
        """Compile the checked-in source and packaged architecture to an executable.

        :param source: Digest-verified Faust source.
        :returns: Native executable path.
        :raises RuntimeError: Faust or C++ compilation fails.
        """
        directory = Path(self._temporary_directory.name)
        source_path = directory / "source.dsp"
        generated_path = directory / "renderer.cpp"
        executable = directory / "renderer"
        source_path.write_text(source)
        architecture = Path(__file__).with_name("faustcpp_architecture.txt")
        try:
            include_dir = subprocess.run(  # noqa: S603
                ["faust", "-includedir"],  # noqa: S607
                check=True,
                capture_output=True,
                text=True,
                timeout=_COMPILE_TIMEOUT_SECONDS,
            ).stdout.strip()
            subprocess.run(  # noqa: S603
                [
                    "faust",
                    "-a",
                    str(architecture),
                    "-cn",
                    "SynthSetterDsp",
                    "-ftz",
                    "0",
                    str(source_path),
                    "-o",
                    str(generated_path),
                ],  # noqa: S607
                check=True,
                capture_output=True,
                text=True,
                timeout=_COMPILE_TIMEOUT_SECONDS,
            )
            subprocess.run(  # noqa: S603
                [
                    "g++",
                    "-O3",
                    "-std=c++17",
                    "-I",
                    include_dir,
                    str(generated_path),
                    "-o",
                    str(executable),
                ],  # noqa: S607
                check=True,
                capture_output=True,
                text=True,
                timeout=_COMPILE_TIMEOUT_SECONDS,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            detail = getattr(error, "stderr", "") or str(error)
            raise RuntimeError(f"Faust C++ compilation failed: {detail}") from error
        return executable

    def _validate_patch(self, params: dict[str, float]) -> None:
        """Require one complete finite patch within every native parameter domain.

        :param params: Canonical native parameter values.
        :raises KeyError: A parameter address is unknown.
        :raises ValueError: The patch is incomplete or contains an invalid value.
        :raises TypeError: A parameter has an unsupported domain.
        """
        expected = self._parameters.keys()
        unknown = sorted(params.keys() - expected)
        if unknown:
            raise KeyError(f"unknown Faust parameter address(es): {', '.join(unknown)}")
        missing = sorted(expected - params.keys())
        if missing:
            raise ValueError(f"missing Faust parameter address(es): {', '.join(missing)}")
        for address, value in params.items():
            parameter = self._parameters[address]
            if not np.isfinite(value):
                raise ValueError(f"parameter outside native domain: {address}")
            if isinstance(parameter, ContinuousParameter):
                if not parameter.min <= value <= parameter.max:
                    raise ValueError(f"parameter outside native domain: {address}")
            elif isinstance(parameter, CategoricalParameter):
                if value not in parameter.raw_values:
                    raise ValueError(f"parameter outside discrete native domain: {address}")
            else:
                raise TypeError(f"unsupported Faust parameter {type(parameter).__name__}")

    def render(
        self,
        params: Mapping[str, ParameterValue],
        midi_note: int,
        velocity: int,
        note_start_and_end: tuple[float, float],
        *,
        warmup: bool = False,
    ) -> np.ndarray:
        """Render a complete canonical patch through a fresh native DSP instance.

        :param params: Complete canonical parameter patch.
        :param midi_note: MIDI note number.
        :param velocity: MIDI velocity in the inclusive range ``[0, 127]``.
        :param note_start_and_end: Note-on and note-off times in seconds.
        :param warmup: Ignored because each executable run creates a fresh DSP.
        :returns: Channel-major float32 audio.
        :raises RuntimeError: The native renderer fails or writes an incomplete buffer.
        :raises ValueError: MIDI or note timing values are invalid.
        """
        del warmup
        scalar_params = require_scalar_synth_params(params)
        self._validate_patch(scalar_params)
        if isinstance(midi_note, bool) or not isinstance(midi_note, int) or not 0 <= midi_note <= 127:
            raise ValueError("midi_note must be an integer in [0, 127]")
        if isinstance(velocity, bool) or not isinstance(velocity, int) or not 0 <= velocity <= 127:
            raise ValueError("velocity must be an integer in [0, 127]")
        frames = int(self.sample_rate * self.signal_duration_seconds)
        start_frame, end_frame = _quantize_note_window(
            note_start_and_end,
            sample_rate=self.sample_rate,
            frames=frames,
            signal_duration_seconds=self.signal_duration_seconds,
        )
        directory = Path(self._temporary_directory.name)
        request_path = directory / "request.txt"
        output_path = directory / "audio.f32"
        lines = [
            f"{frames} {self.block_size} {int(self.sample_rate)} {start_frame} {end_frame} "
            f"{midi_note} {velocity} {int(self._identity == 'faust_bright_organ')} "
            f"{len(scalar_params)}"
        ]
        lines.extend(
            f"{_native_address(self._identity, address)}\t{value:.17g}"
            for address, value in scalar_params.items()
        )
        request_path.write_text("\n".join(lines) + "\n")
        output_path.unlink(missing_ok=True)
        try:
            subprocess.run(  # noqa: S603
                [str(self._executable), str(request_path), str(output_path)],
                check=True,
                capture_output=True,
                text=True,
                timeout=_RENDER_TIMEOUT_SECONDS,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            detail = getattr(error, "stderr", "") or str(error)
            raise RuntimeError(f"Faust C++ render failed: {detail}") from error
        audio = np.fromfile(output_path, dtype=np.float32)
        expected_values = self.channels * frames
        if audio.size != expected_values:
            raise RuntimeError(
                f"Faust C++ renderer wrote {audio.size} samples; expected {expected_values}"
            )
        return _validate_rendered_audio(
            audio.reshape(self.channels, frames), channels=self.channels, samples=frames
        )
