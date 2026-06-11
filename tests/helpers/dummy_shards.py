"""Deterministic dummy-shard writers and a renderer stub for launcher tests.

Both the real-R2 launcher roundtrip (``tests/integration/test_local_launcher_roundtrip.py``)
and the fast fake-R2 orchestrator test (``tests/test_generate_dataset.py``) drive
``cli.generate_dataset`` with the Surge VST3 subprocess replaced by a deterministic
stub that writes a validation-passing shard of the right shape. Centralizing the
writers + stub here keeps the two lanes from drifting: a change to the writer's
shard layout updates both at once.
"""

from __future__ import annotations

import io
import subprocess
import tarfile
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np

from synth_setter.data.vst.shapes import (
    AUDIO_FIELD,
    DATASET_FIELD_NAMES,
    MEL_SPEC_FIELD,
    PARAM_ARRAY_FIELD,
    audio_dataset_shape,
    mel_dataset_shape,
    param_array_dataset_shape,
)
from synth_setter.pipeline.schemas.shard_metadata import ShardMetadata
from synth_setter.pipeline.schemas.spec import DatasetSpec, OutputFormat
from tests.helpers.subprocess_args import find_script_index

# Captured at import so the rclone-passthrough side effect calls the real
# subprocess.check_call without recursing through any patch on that symbol.
_REAL_CHECK_CALL = subprocess.check_call


def write_dummy_h5_shard(output_path: Path, spec: DatasetSpec) -> None:
    """Write a validation-passing HDF5 shard with zeroed datasets.

    Dataset shapes come from ``synth_setter.data.vst.shapes`` so ``validate_shard``
    accepts the output; values are all zeros (the validator checks structure and
    shape, not content), so the shard is deterministic with no RNG.

    :param output_path: Destination ``.h5`` file path; parent dir must exist.
    :param spec: Dataset spec whose ``render`` config and ``num_params`` drive the
        per-field array shapes.
    """
    render = spec.render
    n = render.samples_per_shard
    audio_shape = audio_dataset_shape(
        n, render.channels, render.sample_rate, render.signal_duration_seconds
    )
    mel_shape = mel_dataset_shape(
        n, render.channels, render.sample_rate, render.signal_duration_seconds
    )
    param_shape = param_array_dataset_shape(n, spec.num_params)
    with h5py.File(output_path, "w") as f:
        f.create_dataset(AUDIO_FIELD, data=np.zeros(audio_shape, dtype=np.float16))
        f.create_dataset(MEL_SPEC_FIELD, data=np.zeros(mel_shape, dtype=np.float32))
        f.create_dataset(PARAM_ARRAY_FIELD, data=np.zeros(param_shape, dtype=np.float32))


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
    n = render.samples_per_shard
    audio = np.zeros(
        audio_dataset_shape(
            n, render.channels, render.sample_rate, render.signal_duration_seconds
        ),
        dtype=np.float16,
    )
    mel = np.zeros(
        mel_dataset_shape(n, render.channels, render.sample_rate, render.signal_duration_seconds),
        dtype=np.float32,
    )
    params = np.zeros(param_array_dataset_shape(n, spec.num_params), dtype=np.float32)
    metadata = ShardMetadata(
        velocity=render.velocity,
        signal_duration_seconds=render.signal_duration_seconds,
        sample_rate=render.sample_rate,
        channels=render.channels,
        min_loudness=render.min_loudness,
    )
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
    """Return a ``subprocess.check_call`` side effect that writes dummy shards.

    Dispatches on the renderer output path's suffix via ``OutputFormat.from_extension``,
    so the same factory backs both hdf5 and wds runs. ``rclone`` invocations fall
    through to the real binary so the R2 upload, the skip-existing probe, and any
    purge hit the configured remote (real R2, or a local-backed fake remote).

    :param spec: Dataset spec the launcher will materialize; threaded into the
        dummy-shard writers so shapes match the validator's expectations.
    :returns: A callable matching ``subprocess.check_call``'s side-effect contract.
    """

    def _side_effect(args: list[str]) -> None:
        if args and args[0] == "rclone":
            _REAL_CHECK_CALL(args)  # noqa: S603 — passthrough to real rclone
            return
        script_idx = find_script_index(args)
        output_file = Path(args[script_idx + 1])
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fmt = OutputFormat.from_extension(output_file.suffix)
        if fmt is OutputFormat.HDF5:
            write_dummy_h5_shard(output_file, spec)
        elif fmt is OutputFormat.WDS:
            write_dummy_tar_shard(output_file, spec)
        else:
            raise AssertionError(
                f"stubbed renderer cannot write output with suffix {output_file.suffix!r}"
            )

    return _side_effect
