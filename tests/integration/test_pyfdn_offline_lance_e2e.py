"""Real canonical pyFDN render through acceptance, Lance, and model reader."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import lance
import numpy as np
import pytest
import torch

from synth_setter.cli.generate_dataset import build_generate_args
from synth_setter.data.lance_datamodule import LanceVSTDataModule
from synth_setter.data.pyfdn_instrument import PyFDNRenderer
from synth_setter.data.pyfdn_param_spec import PYFDN_N8_MONO_PARAM_SPEC
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.pipeline.data.lance_shard import (
    iter_lance_column_rows,
    read_shard_metadata,
)
from synth_setter.pipeline.schemas.render_metrics import (
    RenderRejectionMetrics,
    render_metrics_path,
)
from synth_setter.pipeline.schemas.seed_debug import SeedDebugDocument
from synth_setter.pipeline.schemas.spec import DatasetSpec, RenderConfig
from synth_setter.synth_spec import SynthName, SynthSpec


@pytest.mark.slow
def test_pyfdn_acceptance_lance_reader_rerender_round_trip(tmp_path: Path) -> None:
    """A real accepted row survives storage and the production model-batch reader.

    :param tmp_path: Isolated destination for the production Lance writer.
    """
    render = RenderConfig(
        synth=SynthSpec(
            name=SynthName("pyfdn_n8_mono"),
            param_spec_name=ParamSpecName("pyfdn_n8_mono"),
            plugin_path="pyfdn",
            plugin_state_path="",
            synth_version="0.4.2",
        ),
        renderer_backend="pyfdn",
        sample_rate=44_100,
        channels=1,
        velocity=0,
        signal_duration_seconds=4.0,
        min_loudness=-100.0,
        audio_dtype="float32",
        mel_spec_dtype="float32",
        samples_per_render_batch=1,
        samples_per_shard=1,
        base_seed=19,
        attempts_per_sample=100,
        param_sample_cadence="sample",
        plugin_reload_cadence="render",
        gui_toggle_cadence="never",
    )
    spec = DatasetSpec.model_validate(
        {
            "task_name": "pyfdn-offline-cli-e2e",
            "output_format": "lance",
            "train_val_test_sizes": [1, 0, 0],
            "base_seed": 19,
            "r2": {"bucket": "intermediate-data"},
            "render": render.model_dump(mode="json"),
        }
    )
    shard = spec.shards[0]
    output = tmp_path / shard.filename
    result = subprocess.run(  # noqa: S603 — production argv from the validated spec
        build_generate_args(spec, shard, tmp_path),
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    rejections = RenderRejectionMetrics.model_validate_json(
        render_metrics_path(output).read_text()
    )

    dataset = lance.dataset(str(output))
    metadata = read_shard_metadata(dataset.schema)
    assert dataset.count_rows() == 1
    mel_rows = list(iter_lance_column_rows(output, "mel_spec"))
    assert len(mel_rows) == 1
    assert mel_rows[0].shape == (1, 128, 401)
    assert mel_rows[0].dtype == np.float32
    assert metadata == render.shard_metadata()
    assert rejections.clipped >= 0
    assert rejections.silent >= 0

    datamodule = LanceVSTDataModule(
        dataset_root=tmp_path,
        predict_file=output,
        use_saved_mean_and_variance=False,
        batch_size=1,
        ot=False,
        num_workers=0,
        conditioning="audio",
        pin_memory=False,
        param_spec_name=ParamSpecName("pyfdn_n8_mono"),
    )
    datamodule.setup("predict")
    batch = next(iter(datamodule.predict_dataloader()))
    audio = batch["audio"]
    params = batch["params"]
    assert audio is not None
    assert params is not None
    assert audio.shape == (1, 1, 176_400)
    assert params.shape == (1, 91)
    assert audio.dtype == params.dtype == torch.float32
    assert torch.isfinite(audio).all()
    assert torch.all((-1.0 <= params) & (params <= 1.0))

    debug_json = dataset.to_table(columns=["debug"])["debug"][0].as_py()
    debug = SeedDebugDocument.model_validate(json.loads(debug_json))
    assert debug.attempt == rejections.clipped + rejections.silent
    assert debug.seed is not None
    native_params, native_notes = PYFDN_N8_MONO_PARAM_SPEC.sample(
        np.random.default_rng(debug.seed)
    )
    assert native_notes == {"pitch": 0, "note_start_and_end": (0.0, 0.0)}
    native_rerender = PyFDNRenderer().render(native_params)
    np.testing.assert_array_equal(native_rerender, audio[0].numpy())

    encoded = PYFDN_N8_MONO_PARAM_SPEC.model_to_encoded(params[0].numpy())
    decoded, note_params = PYFDN_N8_MONO_PARAM_SPEC.decode(encoded)
    decoded_rerender = PyFDNRenderer().render(decoded)
    assert note_params == {"pitch": 0, "note_start_and_end": (0.0, 0.0)}
    np.testing.assert_allclose(decoded_rerender, audio[0].numpy(), rtol=1e-4, atol=2e-5)
