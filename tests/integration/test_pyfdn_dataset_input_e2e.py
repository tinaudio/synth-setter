"""Dataset audio through the production pyFDN shard-generation CLI."""

from __future__ import annotations

import subprocess
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest

from synth_setter.cli.generate_dataset import build_generate_args
from synth_setter.data.pyfdn_param_spec import (
    PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC,
    pyfdn_param_spec_sha256,
)
from synth_setter.data.vst.shapes import AUDIO_FIELD, DATASET_FIELD_NAMES
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.pipeline.data.lance_shard import (
    SHARD_METADATA_SCHEMA_KEY,
    iter_lance_column_rows,
    tensor_array,
)
from synth_setter.pipeline.schemas.shard_metadata import ShardMetadata
from synth_setter.pipeline.schemas.spec import DatasetSpec, InputAudioSource, RenderConfig
from synth_setter.synth_spec import SynthName, SynthSpec

_FRAMES = 176_400
_SAMPLE_RATE = 44_100


def _write_input_dataset(root: Path) -> str:
    audio = np.zeros((2, 1, _FRAMES), dtype=np.float32)
    audio[0, 0, 0] = 0.05
    audio[1, 0, 100] = 0.05
    metadata = ShardMetadata(
        velocity=0,
        signal_duration_seconds=4.0,
        sample_rate=_SAMPLE_RATE,
        channels=1,
        min_loudness=-100.0,
    )
    field = pa.field(
        AUDIO_FIELD,
        pa.fixed_shape_tensor(pa.float32(), (1, _FRAMES)),
        nullable=False,
    )
    schema = pa.schema(
        [field],
        metadata={SHARD_METADATA_SCHEMA_KEY: metadata.model_dump_json().encode()},
    )
    storage = tensor_array(audio, np.dtype("float32"), (1, _FRAMES)).storage
    table = pa.Table.from_arrays(
        [pa.ExtensionArray.from_storage(field.type, storage)], schema=schema
    )
    dataset = lance.write_dataset(table, str(root / "train.lance"))
    transaction = dataset.read_transaction(dataset.version)
    assert transaction is not None
    return transaction.uuid


def _render_config(source_root: Path, txid: str, sampling_seed: int) -> RenderConfig:
    return RenderConfig(
        synth=SynthSpec(
            name=SynthName("pyfdn_n8_mono_householder"),
            param_spec_name=ParamSpecName("pyfdn_n8_mono_householder"),
            format="pyfdn",
            plugin_path="pyfdn",
            plugin_state_path="",
            synth_version="0.4.2",
            source_sha256=pyfdn_param_spec_sha256(PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC),
        ),
        renderer_backend="pyfdn",
        render_contract_version=2,
        pyfdn_excitation="impulse",
        input_audio_source=InputAudioSource(
            dataset_uri=source_root.as_uri(),
            snapshot_txid=txid,
            sampling_seed=sampling_seed,
        ),
        sample_rate=_SAMPLE_RATE,
        channels=1,
        velocity=0,
        signal_duration_seconds=4.0,
        min_loudness=-100.0,
        audio_dtype="float32",
        mel_spec_dtype="float32",
        samples_per_render_batch=1,
        samples_per_shard=1,
        base_seed=19,
        attempts_per_sample=1,
        param_sample_cadence="sample",
        plugin_reload_cadence="render",
        gui_toggle_cadence="never",
    )


def _run_cli(destination: Path, render: RenderConfig) -> np.ndarray:
    spec = DatasetSpec.model_validate(
        {
            "task_name": "pyfdn-dataset-input-e2e",
            "output_format": "lance",
            "train_val_test_sizes": [1, 0, 0],
            "base_seed": 19,
            "r2": {"bucket": "intermediate-data"},
            "render": render.model_dump(mode="json"),
        }
    )
    destination.mkdir()
    result = subprocess.run(  # noqa: S603 — argv comes from the validated production spec
        build_generate_args(spec, spec.shards[0], destination),
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    dataset = lance.dataset(str(destination / spec.shards[0].filename))
    assert dataset.count_rows() == 1
    assert all(name in dataset.schema.names for name in DATASET_FIELD_NAMES)
    return next(iter_lance_column_rows(destination / spec.shards[0].filename, AUDIO_FIELD))


@pytest.mark.slow
def test_pyfdn_dataset_input_cli_selected_row_changes_output_and_repeats_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise pinned dataset input through the production generation CLI.

    :param tmp_path: Temporary test directory.
    :param monkeypatch: Environment override fixture.
    """
    source_root = tmp_path / "source"
    txid = _write_input_dataset(source_root)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    selected_row_one = _run_cli(tmp_path / "row-one", _render_config(source_root, txid, 0))
    selected_row_zero = _run_cli(tmp_path / "row-zero", _render_config(source_root, txid, 4))
    repeated_row_one = _run_cli(tmp_path / "repeat", _render_config(source_root, txid, 0))

    assert selected_row_one.shape == (1, _FRAMES)
    assert np.isfinite(selected_row_one).all()
    assert not np.array_equal(selected_row_one, selected_row_zero)
    assert selected_row_one.tobytes() == repeated_row_one.tobytes()
