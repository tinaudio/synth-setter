"""Compile registry-backed FaustWasm artifacts for rendering or persistent export."""

from __future__ import annotations

import ctypes
import errno
import json
import logging
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from synth_setter.data.vst.faust_sources import resolve_faust_dsp
from synth_setter.data.vst.faustwasm_contract import (
    faustwasm_parameter_contract,
    faustwasm_reserved_addresses,
)
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.resources import as_file, faustwasm_dir
from synth_setter.synth_spec import SYNTHS, SynthName, SynthSpec

_NODE_TIMEOUT_SECONDS = 60
_RENAME_NOREPLACE = 1
_RENAME_EXCL = 4
_AT_FDCWD = -100

logger = logging.getLogger(__name__)


def _rename_without_replace(source: Path, destination: Path) -> None:
    """Atomically rename a directory without replacing an existing destination.

    :param source: Staged directory to publish.
    :param destination: Absent publication destination.
    :raises FileExistsError: The destination already exists.
    :raises OSError: The platform cannot complete the atomic rename.
    """
    if os.name == "nt":
        os.rename(source, destination)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        result = libc.renamex_np(os.fsencode(source), os.fsencode(destination), _RENAME_EXCL)
    else:
        result = libc.renameat2(
            _AT_FDCWD,
            os.fsencode(source),
            _AT_FDCWD,
            os.fsencode(destination),
            _RENAME_NOREPLACE,
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(error_number, os.strerror(error_number), destination)
    raise OSError(error_number, os.strerror(error_number), destination)


class _ArtifactFile(BaseModel):
    """One relative artifact path and its content digest.

    .. attribute :: model_config

        Strict manifest validation settings.
    .. attribute :: path

        Path relative to the manifest.
    .. attribute :: sha256

        Lowercase SHA-256 digest of the file bytes.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    path: str
    sha256: str


class _ArtifactFiles(BaseModel):
    """Compiled DSP modules referenced by a manifest.

    .. attribute :: model_config

        Strict manifest validation settings.
    .. attribute :: dsp

        Required signal processor module.
    .. attribute :: mixer

        Polyphonic voice mixer module when present.
    .. attribute :: effect

        Polyphonic effect module when present.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    dsp: _ArtifactFile
    mixer: _ArtifactFile | None = None
    effect: _ArtifactFile | None = None


class _ArtifactParameter(BaseModel):
    """Canonical parameter identity and exact native domain.

    .. attribute :: model_config

        Strict manifest validation settings.
    .. attribute :: canonicalAddress

        Stable dataset parameter address.
    .. attribute :: wasmAddress

        Address emitted by the pinned Faust compiler.
    .. attribute :: min

        Native lower bound.
    .. attribute :: max

        Native upper bound.
    .. attribute :: kind

        Continuous or discrete domain kind.
    .. attribute :: values

        Exact discrete values, otherwise ``None``.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    canonicalAddress: str
    wasmAddress: str
    min: float
    max: float
    kind: str
    values: list[float] | None


class _ArtifactManifest(BaseModel):
    """Validated manifest consumed by Python and JavaScript runtimes.

    .. attribute :: model_config

        Strict known-field validation with compiler metadata passthrough.
    .. attribute :: schemaVersion

        Artifact schema version.
    .. attribute :: identity

        Registered synth identity.
    .. attribute :: faustwasmVersion

        FaustWasm package version.
    .. attribute :: libfaustVersion

        Native Faust compiler version.
    .. attribute :: compileOptions

        Compiler options used for every module.
    .. attribute :: sourceSha256

        Digest of the compiled source.
    .. attribute :: mode

        Monophonic or polyphonic runtime mode.
    .. attribute :: voices

        Polyphonic voice count, or zero for mono.
    .. attribute :: outputs

        Native output channel count.
    .. attribute :: inputs

        Native input channel count; defaults to zero for schema-v1 synth artifacts.
    .. attribute :: parameters

        Complete canonical parameter map.
    .. attribute :: files

        Compiled module paths and digests.
    """

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
    inputs: int = 0
    parameters: list[_ArtifactParameter]
    files: _ArtifactFiles


type ArtifactManifest = _ArtifactManifest


def _faustwasm_entrypoint(resource_directory: Path, name: str) -> Path:
    """Validate one materialized Node entrypoint before execution.

    :param resource_directory: Materialized packaged FaustWasm directory.
    :param name: Entry-point filename in that directory.
    :returns: Absolute entry-point path.
    :raises RuntimeError: The entrypoint or Node.js executable is unavailable.
    """
    if shutil.which("node") is None:
        raise RuntimeError("FaustWasm rendering requires Node.js on PATH")
    script = resource_directory / name
    if not script.is_file():
        raise RuntimeError(f"packaged FaustWasm entrypoint is unavailable: {name}")
    return script


def _assemble_faustwasm_file(resource_directory: Path, filename: str) -> None:
    """Assemble one compiler binary from its packaged chunks.

    :param resource_directory: Writable materialized FaustWasm directory.
    :param filename: Filename under ``vendor/libfaust-wasm``.
    :raises RuntimeError: Packaged chunks are missing or malformed.
    """
    parent = resource_directory / "vendor" / "libfaust-wasm"
    parts = sorted(parent.glob(f"{filename}.binpart*"))
    if not parts:
        raise RuntimeError(f"packaged FaustWasm binary chunks are unavailable: {filename}")
    with (parent / filename).open("wb") as destination:
        for part in parts:
            with part.open("rb") as source:
                if source.read(1) != b"\0":
                    raise RuntimeError(f"malformed packaged FaustWasm binary chunk: {part.name}")
                shutil.copyfileobj(source, destination)


def _compile_request(synth: SynthSpec, backend_version: str) -> dict[str, object]:
    """Build the single registry-backed request accepted by the Node compiler.

    :param synth: Registered digest-pinned Faust synth identity.
    :param backend_version: Expected FaustWasm package version.
    :returns: JSON-compatible compile request.
    """
    identity = ParamSpecName(synth.param_spec_name)
    dsp = resolve_faust_dsp(identity)
    parameters = [
        {
            "canonicalAddress": item.canonical_address,
            "wasmAddress": item.wasm_address,
            "min": item.minimum,
            "max": item.maximum,
            "kind": item.kind,
            "values": list(item.values) if item.values is not None else None,
        }
        for item in faustwasm_parameter_contract(identity)
    ]
    return {
        "expectedFaustWasmVersion": backend_version,
        "identity": identity,
        "source": dsp.source,
        "sourceSha256": synth.source_sha256,
        "mode": "poly" if dsp.num_voices else "mono",
        "voices": dsp.num_voices,
        "expectedInputs": dsp.inputs,
        "expectedOutputs": dsp.outputs,
        "parameters": parameters,
        "reservedWasmAddresses": faustwasm_reserved_addresses(identity),
    }


def compile_faustwasm_artifact(
    synth: SynthSpec,
    output_directory: Path,
    *,
    backend_version: str,
) -> ArtifactManifest:
    """Compile one registered Faust synth into ``output_directory``.

    :param synth: Registered digest-pinned Faust synth identity.
    :param output_directory: Existing or new directory for compiled modules and manifest.
    :param backend_version: Expected FaustWasm package version.
    :returns: Validated manifest whose paths are relative to ``output_directory``.
    :raises ValueError: The synth or compiled provenance differs from the registry contract.

    Runtime setup errors (``RuntimeError``), compilation timeouts
    (``subprocess.TimeoutExpired``), and failures (``subprocess.CalledProcessError``)
    propagate to the caller.
    """
    if synth.format != "faust" or synth.source_sha256 is None:
        raise ValueError("FaustWasm artifacts require a registered Faust source")
    output_directory.mkdir(parents=True, exist_ok=True)
    request = _compile_request(synth, backend_version)
    request_path = output_directory / ".compile-request.json"
    request_path.write_text(json.dumps(request))
    try:
        with (
            as_file(faustwasm_dir()) as packaged_directory,
            tempfile.TemporaryDirectory(prefix="synth-setter-faustwasm-") as temporary_directory,
        ):
            resource_directory = Path(temporary_directory) / "faustwasm"
            shutil.copytree(packaged_directory, resource_directory)
            # FaustWasm writes a transient ESM shim beside libfaust-wasm.js during compilation.
            for root, _, _ in os.walk(resource_directory):
                directory = Path(root)
                directory.chmod(directory.stat().st_mode | stat.S_IWUSR)
            # Source-control size gates require the compiler binaries to ship as bounded chunks.
            _assemble_faustwasm_file(resource_directory, "libfaust-wasm.data")
            _assemble_faustwasm_file(resource_directory, "libfaust-wasm.wasm")
            script = _faustwasm_entrypoint(resource_directory, "export-artifacts.mjs")
            subprocess.run(  # noqa: S603
                ["node", str(script), str(request_path), str(output_directory)],
                check=True,
                capture_output=True,
                text=True,
                timeout=_NODE_TIMEOUT_SECONDS,
            )
    finally:
        request_path.unlink(missing_ok=True)

    manifest = _ArtifactManifest.model_validate_json(
        (output_directory / "manifest.json").read_text()
    )
    expected_parameters = request["parameters"]
    actual_parameters = [parameter.model_dump() for parameter in manifest.parameters]
    if (
        manifest.identity != synth.param_spec_name
        or manifest.faustwasmVersion != backend_version
        or manifest.sourceSha256 != synth.source_sha256
        or manifest.mode != request["mode"]
        or manifest.voices != request["voices"]
        or manifest.inputs != request["expectedInputs"]
        or manifest.outputs != request["expectedOutputs"]
        or actual_parameters != expected_parameters
    ):
        raise ValueError("compiled FaustWasm artifact provenance does not match the synth registry")
    return manifest


def run_faustwasm_render_worker(
    artifact_directory: Path,
    request_path: Path,
    output_path: Path,
) -> None:
    """Render one request through persisted modules without recompiling Faust.

    :param artifact_directory: Directory containing ``manifest.json`` and its modules.
    :param request_path: JSON render request path.
    :param output_path: Destination for channel-major little-endian float32 samples.
    """
    with as_file(faustwasm_dir()) as resource_directory:
        script = _faustwasm_entrypoint(resource_directory, "render-worker.mjs")
        subprocess.run(  # noqa: S603
            [
                "node",
                str(script),
                str(artifact_directory / "manifest.json"),
                str(request_path),
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=_NODE_TIMEOUT_SECONDS,
        )


def export_faustwasm_artifact(
    synth_name: SynthName,
    output_directory: Path,
    *,
    backend_version: str,
) -> ArtifactManifest:
    """Atomically publish one registry-backed artifact to an absent destination.

    :param synth_name: Key of a registered Faust synth.
    :param output_directory: Destination directory, which must not exist.
    :param backend_version: Expected FaustWasm package version.
    :returns: Validated manifest for the published artifact.
    :raises FileExistsError: The destination already exists.
    """
    if output_directory.exists():
        raise FileExistsError(f"output destination already exists: {output_directory}")
    synth = SYNTHS[synth_name]
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_directory.name}.",
            dir=output_directory.parent,
        )
    )
    try:
        manifest = compile_faustwasm_artifact(
            synth,
            staging,
            backend_version=backend_version,
        )
        _rename_without_replace(staging, output_directory)
        return manifest
    finally:
        if staging.exists():
            try:
                shutil.rmtree(staging)
            except OSError:
                logger.warning("failed to remove FaustWasm staging directory %s", staging)
