"""Deterministic dummy-shard writers and a renderer stub for launcher tests.

Both the real-R2 launcher roundtrip (``tests/integration/test_local_launcher_roundtrip.py``)
and the fast fake-R2 orchestrator test (``tests/test_generate_dataset.py``) drive
``cli.generate_dataset`` with the Surge VST3 subprocess replaced by a deterministic
stub that writes a validation-passing shard of the right shape. Centralizing the
writers + stub here keeps the two lanes from drifting: a change to the writer's
shard layout updates both at once. The lance writer is imported from
``tests.helpers.finalize_shards``, which owns it for the finalize lanes.
"""

from __future__ import annotations

import io
import tarfile
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np

from synth_setter.data.vst.shapes import (
    AUDIO_FIELD,
    DATASET_FIELD_DTYPES,
    DATASET_FIELD_NAMES,
    MEL_SPEC_FIELD,
    PARAM_ARRAY_FIELD,
    dataset_field_shapes,
)
from synth_setter.pipeline.schemas.spec import DatasetSpec, OutputFormat
from synth_setter.pipeline.subprocess_stream import check_call_streamed
from tests.helpers.finalize_shards import smoke_shard_metadata, write_minimal_lance_shard
from tests.helpers.subprocess_args import find_script_index

# rclone passthrough: bound from the pipeline module, so it bypasses the patched
# cli seam (no recursion) yet runs the real streamed runner on the real binary.
_REAL_CHECK_CALL = check_call_streamed


def write_dummy_h5_shard(output_path: Path, spec: DatasetSpec) -> None:
    """Write a validation-passing HDF5 shard with zeroed datasets.

    Dataset shapes come from ``synth_setter.data.vst.shapes`` so ``validate_shard``
    accepts the output; values are all zeros (the validator checks structure and
    shape, not content), so the shard is deterministic with no RNG.

    :param output_path: Destination ``.h5`` file path; parent dir must exist.
    :param spec: Dataset spec whose ``render`` config and ``num_params`` drive the
        per-field array shapes.
    """
    shapes = dataset_field_shapes(spec.render, spec.num_params)
    with h5py.File(output_path, "w") as f:
        for field, shape in shapes.items():
            f.create_dataset(field, data=np.zeros(shape, dtype=DATASET_FIELD_DTYPES[field]))


def write_dummy_tar_shard(output_path: Path, spec: DatasetSpec) -> None:
    """Write a validation-passing WDS tar shard.

    A single batch keyed by ``00000000`` holds ``samples_per_shard`` rows for every
    writer field; ``metadata.json`` mirrors the ``RenderConfig`` fields ``ShardMetadata``
    requires. Validation is structural, so all-zero arrays are accepted.

    :param output_path: Destination ``.tar`` file path; parent dir must exist.
    :param spec: Dataset spec whose ``render`` config and ``num_params`` drive the
        per-field array shapes and the ``ShardMetadata`` field values.
    :raises ValueError: If a field name is not in ``DATASET_FIELD_NAMES``.
    """
    render = spec.render
    shapes = dataset_field_shapes(render, spec.num_params)
    audio = np.zeros(shapes[AUDIO_FIELD], dtype=DATASET_FIELD_DTYPES[AUDIO_FIELD])
    mel = np.zeros(shapes[MEL_SPEC_FIELD], dtype=DATASET_FIELD_DTYPES[MEL_SPEC_FIELD])
    params = np.zeros(shapes[PARAM_ARRAY_FIELD], dtype=DATASET_FIELD_DTYPES[PARAM_ARRAY_FIELD])
    metadata = smoke_shard_metadata(render)
    with tarfile.open(output_path, mode="w") as tar:
        for field_name, arr in (
            (AUDIO_FIELD, audio),
            (MEL_SPEC_FIELD, mel),
            (PARAM_ARRAY_FIELD, params),
        ):
            if field_name not in DATASET_FIELD_NAMES:
                raise ValueError(f"Unknown field: {field_name!r}")
            buf = io.BytesIO()
            np.save(buf, arr, allow_pickle=False)
            payload = buf.getvalue()
            info = tarfile.TarInfo(name=f"00000000.{field_name}.npy")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        payload = metadata.model_dump_json().encode("utf-8")
        info = tarfile.TarInfo(name="metadata.json")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))


def stub_renderer(spec: DatasetSpec) -> Callable[[list[str]], None]:
    """Return a ``_check_call_streamed`` side effect that writes dummy shards.

    Dispatches on the renderer output path's suffix via ``OutputFormat.from_extension``,
    so the same factory backs hdf5, wds, and lance runs. ``rclone`` invocations
    fall through to the real binary so the R2 upload, the skip-existing probe, and
    any purge hit the configured remote (real R2, or a local-backed fake remote).

    :param spec: Dataset spec the launcher will materialize; threaded into the
        dummy-shard writers so shapes match the validator's expectations.
    :returns: A callable matching ``_check_call_streamed``'s side-effect contract.
    """

    def _side_effect(args: list[str]) -> None:
        if args and args[0] == "rclone":
            _REAL_CHECK_CALL(args)
            return
        script_idx = find_script_index(args)
        output_file = Path(args[script_idx + 1])
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fmt = OutputFormat.from_extension(output_file.suffix)
        if fmt is OutputFormat.HDF5:
            write_dummy_h5_shard(output_file, spec)
        elif fmt is OutputFormat.WDS:
            write_dummy_tar_shard(output_file, spec)
        elif fmt is OutputFormat.LANCE:
            write_minimal_lance_shard(output_file, spec)
        else:
            raise AssertionError(
                f"stubbed renderer cannot write output with suffix {output_file.suffix!r}"
            )

    return _side_effect
