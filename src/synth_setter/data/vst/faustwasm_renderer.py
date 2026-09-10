"""Offline FaustWasm renderer backed by bounded Node subprocesses."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Mapping
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from synth_setter.data.vst.faust_sources import resolve_faust_dsp
from synth_setter.data.vst.faustwasm_contract import (
    faustwasm_parameter_contract,
    faustwasm_reserved_addresses,
)
from synth_setter.data.vst.renderers import (
    AudioRenderer,
    ParameterValue,
    _validate_rendered_audio,
    require_scalar_synth_params,
)
from synth_setter.param_spec_name import ParamSpecName

FAUSTWASM_VERSION = "0.18.3"
_NODE_TIMEOUT_SECONDS = 60


class _ArtifactFile(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    path: str
    sha256: str


class _ArtifactFiles(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    dsp: _ArtifactFile
    mixer: _ArtifactFile | None = None
    effect: _ArtifactFile | None = None


class _ArtifactParameter(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    canonicalAddress: str
    wasmAddress: str
    min: float
    max: float
    kind: str


class _ArtifactManifest(BaseModel):
    model_config = ConfigDict(strict=True, extra="allow")

    schemaVersion: Literal[1]
    identity: str
    faustwasmVersion: str
    libfaustVersion: str
    compileOptions: str
    sourceSha256: str
    mode: Literal["mono", "poly"]
    voices: int
    outputs: int
    parameters: list[_ArtifactParameter]
    files: _ArtifactFiles


def _repository_script(name: str) -> Path:
    """Resolve a checkout-only Node entrypoint with an actionable install error.

    :param name: Entry-point filename under ``scripts/faustwasm``.
    :returns: Absolute entry-point path.
    :raises RuntimeError: The script, dependency, or Node.js executable is unavailable.
    """
    root = Path(__file__).resolve().parents[4]
    script = root / "scripts" / "faustwasm" / name
    package = root / "node_modules" / "@grame" / "faustwasm" / "package.json"
    if not script.is_file():
        raise RuntimeError(
            "FaustWasm Node assets are unavailable outside a synth-setter checkout; "
            "run from the repository containing scripts/faustwasm"
        )
    if not package.is_file():
        raise RuntimeError("FaustWasm runtime is not installed; run `npm ci` at the repository root")
    if shutil.which("node") is None:
        raise RuntimeError("FaustWasm rendering requires Node.js on PATH")
    return script


@dataclass(kw_only=True)
class FaustWasmRenderer(AudioRenderer):
    """Compile a checked-in Faust source once and render isolated Node processes.

    .. attribute :: param_spec_name
       :type: ParamSpecName

       Checked-in Faust source identity.
    .. attribute :: source_sha256
       :type: str

       Required source digest.
    .. attribute :: backend_version
       :type: str

       Required FaustWasm package version.
    .. attribute :: block_size
       :type: int

       Node offline-processing block size.
    """

    param_spec_name: ParamSpecName = field(kw_only=True)
    source_sha256: str = field(kw_only=True)
    backend_version: str = field(kw_only=True)
    block_size: int = 128
    _temporary_directory: tempfile.TemporaryDirectory[str] = field(init=False, repr=False)
    _manifest: _ArtifactManifest = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.backend_version != FAUSTWASM_VERSION:
            raise ValueError(
                f"FaustWasm backend version {self.backend_version!r} does not match "
                f"installed {FAUSTWASM_VERSION!r}"
            )
        if self.plugin_path or self.plugin_state_path:
            raise ValueError("FaustWasm renderer accepts no plugin or preset path")
        self._temporary_directory = tempfile.TemporaryDirectory(prefix="synth-setter-faustwasm-")
        self._compile_artifact()

    def _compile_artifact(self) -> None:
        dsp = resolve_faust_dsp(self.param_spec_name)
        parameters = [
            {
                "canonicalAddress": item.canonical_address,
                "wasmAddress": item.wasm_address,
                "min": item.minimum,
                "max": item.maximum,
                "kind": item.kind,
            }
            for item in faustwasm_parameter_contract(self.param_spec_name)
        ]
        directory = Path(self._temporary_directory.name)
        request_path = directory / "compile-request.json"
        request_path.write_text(
            json.dumps(
                {
                    "identity": self.param_spec_name,
                    "source": dsp.source,
                    "sourceSha256": self.source_sha256,
                    "mode": "poly" if dsp.num_voices else "mono",
                    "voices": dsp.num_voices,
                    "outputs": self.channels,
                    "parameters": parameters,
                    "reservedWasmAddresses": faustwasm_reserved_addresses(
                        self.param_spec_name
                    ),
                }
            )
        )
        subprocess.run(  # noqa: S603
            ["node", str(_repository_script("export-artifacts.mjs")), str(request_path), str(directory)],
            check=True,
            capture_output=True,
            text=True,
            timeout=_NODE_TIMEOUT_SECONDS,
        )
        self._manifest = _ArtifactManifest.model_validate_json(
            (directory / "manifest.json").read_text()
        )
        expected_parameters = [
            (item["canonicalAddress"], item["wasmAddress"], item["min"], item["max"], item["kind"])
            for item in parameters
        ]
        actual_parameters = [
            (item.canonicalAddress, item.wasmAddress, item.min, item.max, item.kind)
            for item in self._manifest.parameters
        ]
        expected_mode = "poly" if dsp.num_voices else "mono"
        if (
            self._manifest.identity != self.param_spec_name
            or self._manifest.faustwasmVersion != self.backend_version
            or self._manifest.sourceSha256 != self.source_sha256
            or self._manifest.mode != expected_mode
            or self._manifest.voices != dsp.num_voices
            or self._manifest.outputs != self.channels
            or actual_parameters != expected_parameters
        ):
            raise ValueError("compiled FaustWasm artifact provenance does not match the render config")

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
        start, end = note_start_and_end
        start_frame = round(start * self.sample_rate)
        end_frame = round(end * self.sample_rate)
        request = {
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
        subprocess.run(  # noqa: S603
            [
                "node",
                str(_repository_script("render-worker.mjs")),
                str(directory / "manifest.json"),
                str(request_path),
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=_NODE_TIMEOUT_SECONDS,
        )
        audio = np.fromfile(output_path, dtype="<f4").reshape(self.channels, frames)
        return _validate_rendered_audio(audio, channels=self.channels, samples=frames)
