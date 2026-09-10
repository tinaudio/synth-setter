"""Fixed-value Faust export for validated pyFDN builds."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Literal

import adac
import torch
from dawdreamer.dawdreamer import RenderEngine
from pydantic import BaseModel, ConfigDict
from pyFDN import fdn_build_from_dict

from synth_setter.data.basic_fdn import BasicFDN

_BLOCK_SIZE = 128

type _Matrix = list[list[float]]
type _SosArray = list[list[list[float]]]


class _FDNBuildDocument(BaseModel):
    """Validate the complete pyFDN v2 JSON document without coercion.

    .. attribute :: format

        Required pyFDN build format marker.

    .. attribute :: version

        Required pyFDN build schema version.

    .. attribute :: feedback_matrix

        Square delay-network feedback matrix.

    .. attribute :: input_matrix

        Mapping from input channels to delay lines.

    .. attribute :: output_matrix

        Mapping from delay lines to output channels.

    .. attribute :: direct_matrix

        Direct mapping from input to output channels.

    .. attribute :: delays

        Delay lengths in samples.

    .. attribute :: sample_rate

        Build sample rate in hertz.

    .. attribute :: post_delay

        Optional SOS filters after the delays.

    .. attribute :: post_matrix

        Optional SOS filters after the feedback matrix.

    .. attribute :: post_output

        Optional SOS filters after the output matrix.

    .. attribute :: model_config

        Strict Pydantic validation settings.
    """

    format: Literal["pyfdn-fdn-build"]
    version: Literal[2]
    feedback_matrix: _Matrix
    input_matrix: _Matrix
    output_matrix: _Matrix
    direct_matrix: _Matrix
    delays: list[int]
    sample_rate: float
    post_delay: _SosArray | None = None
    post_matrix: _SosArray | None = None
    post_output: _SosArray | None = None

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


def _load_basic_fdn(path: Path) -> BasicFDN:
    """Parse one strict v2 document into the canonical BasicFDN boundary.

    :param path: Existing pyFDN build JSON path.
    :returns: Validated basic FDN.
    """
    document = _FDNBuildDocument.model_validate_json(path.read_text(encoding="utf-8"))
    return BasicFDN(fdn_build_from_dict(document.model_dump()))


def _fixed_faust_source(fdn: BasicFDN) -> str:
    """Convert one BasicFDN through FLAMO and ADAC without controls.

    :param fdn: Validated FDN using its declared sample rate.
    :returns: Complete fixed-value Faust source.
    """
    model = fdn.to_flamo(device="cpu", dtype=torch.float32)
    config = adac.flamo_to_json(model, fs=fdn.build.fs, name="BasicFDN")
    return adac.json_to_faust(config)


def _compile_source(source: str, sample_rate: float) -> None:
    """Compile generated source directly with the target DawDreamer consumer.

    :param source: Complete Faust program.
    :param sample_rate: Build sample rate in hertz.
    :raises RuntimeError: DawDreamer rejects the source.
    """
    engine = RenderEngine(sample_rate, _BLOCK_SIZE)
    processor = engine.make_faust_processor("fdn_export")
    processor.num_voices = 0
    if not processor.set_dsp_string(source):
        raise RuntimeError("DawDreamer rejected the generated Faust source")
    if not processor.compile():
        raise RuntimeError("DawDreamer could not compile the generated Faust source")


def _publish_new_file(path: Path, source: str) -> None:
    """Atomically publish complete source without replacing an existing path.

    :param path: Destination that must not exist.
    :param source: Compiled Faust source.
    """
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as temporary_file:
            temporary_file.write(source)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def export_fdn_faust(input_path: Path, output_path: Path) -> None:
    """Export a pyFDN build as compiled, fixed-value Faust source.

    :param input_path: Existing pyFDN v2 build JSON.
    :param output_path: New ``.dsp`` destination; parent directory must exist.
    :raises FileExistsError: The output already exists.
    """
    if output_path.exists():
        raise FileExistsError(output_path)
    fdn = _load_basic_fdn(input_path)
    source = _fixed_faust_source(fdn)
    _compile_source(source, fdn.build.fs)
    _publish_new_file(output_path, source)
