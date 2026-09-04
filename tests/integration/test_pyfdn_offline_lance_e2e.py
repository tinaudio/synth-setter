"""Real canonical pyFDN render through acceptance, Lance, and model reader."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import lance
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from synth_setter.cli.generate_dataset import build_generate_args
from synth_setter.cli.train import train
from synth_setter.data.lance_datamodule import LanceVSTDataModule
from synth_setter.data.pyfdn_instrument import PyFDNRenderer
from synth_setter.data.pyfdn_param_spec import (
    PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC,
    PYFDN_PITCHSHIFT_N8_MONO_HOUSEHOLDER_PARAM_SPEC,
)
from synth_setter.models.vst_flow_matching_module import VSTFlowMatchingModule
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
from synth_setter.workspace import operator_workspace


@pytest.mark.slow
def test_pyfdn_acceptance_lance_reader_rerender_round_trip(tmp_path: Path) -> None:
    """A real accepted row survives storage and the production model-batch reader.

    :param tmp_path: Isolated destination for the production Lance writer.
    """
    synth_name = "pyfdn_n8_mono_householder"
    param_spec = PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC
    render = RenderConfig(
        synth=SynthSpec(
            name=SynthName(synth_name),
            param_spec_name=ParamSpecName(synth_name),
            plugin_path="pyfdn",
            plugin_state_path="",
            synth_version="0.4.2",
        ),
        renderer_backend="pyfdn",
        pyfdn_excitation="impulse",
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
    assert rejections.non_finite >= 0
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
        param_spec_name=ParamSpecName(synth_name),
    )
    datamodule.setup("predict")
    batch = next(iter(datamodule.predict_dataloader()))
    audio = batch["audio"]
    params = batch["params"]
    assert audio is not None
    assert params is not None
    assert audio.shape == (1, 1, 176_400)
    assert params.shape == (1, param_spec.encoded_width)
    assert audio.dtype == params.dtype == torch.float32
    assert torch.isfinite(audio).all()
    assert torch.all((-1.0 <= audio) & (audio <= 1.0))
    assert torch.all((-1.0 <= params) & (params <= 1.0))

    debug_json = dataset.to_table(columns=["debug"])["debug"][0].as_py()
    debug = SeedDebugDocument.model_validate(json.loads(debug_json))
    assert debug.attempt == (rejections.clipped + rejections.non_finite + rejections.silent)
    assert debug.seed is not None
    native_params, native_notes = param_spec.sample(np.random.default_rng(debug.seed))
    assert native_notes == {"pitch": 0, "note_start_and_end": (0.0, 0.0)}
    native_rerender = PyFDNRenderer().render(native_params)
    np.testing.assert_array_equal(native_rerender, audio[0].numpy())

    encoded = param_spec.model_to_encoded(params[0].numpy())
    decoded, note_params = param_spec.decode(encoded)
    decoded_rerender = PyFDNRenderer().render(decoded)
    assert note_params == {"pitch": 0, "note_start_and_end": (0.0, 0.0)}
    np.testing.assert_allclose(decoded_rerender, audio[0].numpy(), rtol=1e-4, atol=2e-5)


@pytest.mark.slow
def test_pyfdn_pitchshift_lance_reader_rerender_round_trip(tmp_path: Path) -> None:
    """A real pitch-shift row survives generation, loading, and native rerender.

    :param tmp_path: Isolated destination for the production Lance writer.
    """
    identity = ParamSpecName("pyfdn_pitchshift_n8_mono_householder")
    render = RenderConfig(
        synth=SynthSpec(
            name=SynthName(identity),
            param_spec_name=identity,
            plugin_path="pyfdn",
            plugin_state_path="",
            synth_version="0.4.2",
        ),
        renderer_backend="pyfdn",
        pyfdn_excitation="impulse",
        sample_rate=44_100,
        channels=1,
        velocity=0,
        signal_duration_seconds=4.0,
        min_loudness=-100.0,
        audio_dtype="float32",
        mel_spec_dtype="float32",
        samples_per_render_batch=1,
        samples_per_shard=1,
        base_seed=23,
        attempts_per_sample=100,
        param_sample_cadence="sample",
        plugin_reload_cadence="render",
        gui_toggle_cadence="never",
    )
    spec = DatasetSpec.model_validate(
        {
            "task_name": "pyfdn-pitchshift-cli-e2e",
            "output_format": "lance",
            "train_val_test_sizes": [1, 0, 0],
            "base_seed": 23,
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
    dataset = lance.dataset(str(output))
    assert dataset.count_rows() == 1
    datamodule = LanceVSTDataModule(
        dataset_root=tmp_path,
        predict_file=output,
        use_saved_mean_and_variance=False,
        batch_size=1,
        ot=False,
        num_workers=0,
        conditioning="audio",
        pin_memory=False,
        param_spec_name=identity,
    )
    datamodule.setup("predict")
    batch = next(iter(datamodule.predict_dataloader()))
    audio = batch["audio"]
    params = batch["params"]
    assert audio is not None
    assert params is not None
    encoded = PYFDN_PITCHSHIFT_N8_MONO_HOUSEHOLDER_PARAM_SPEC.model_to_encoded(params[0].numpy())
    decoded, note_params = PYFDN_PITCHSHIFT_N8_MONO_HOUSEHOLDER_PARAM_SPEC.decode(encoded)
    rerendered = PyFDNRenderer(param_spec_name=identity).render(decoded)

    assert spec.num_params == 45
    assert audio.shape == (1, 1, 176_400)
    assert params.shape == (1, 45)
    assert audio.dtype == params.dtype == torch.float32
    assert torch.isfinite(audio).all()
    assert torch.all((-1.0 <= audio) & (audio <= 1.0))
    assert torch.all((-1.0 <= params) & (params <= 1.0))
    assert note_params == {"pitch": 0, "note_start_and_end": (0.0, 0.0)}
    np.testing.assert_allclose(rerendered, audio[0].numpy(), rtol=1e-4, atol=5e-5)


@pytest.mark.slow
def test_pyfdn_sketch_generation_augmentation_training_sampling_end_to_end(
    tmp_path: Path,
) -> None:
    """Verify persisted reverb sketches survive augmentation, checkpoint loading, and sampling.

    :param tmp_path: Isolated root for generated Lance data and training artifacts.
    """
    identity = ParamSpecName("pyfdn_n8_mono_householder")
    render = RenderConfig(
        synth=SynthSpec(
            name=SynthName(identity),
            param_spec_name=identity,
            plugin_path="pyfdn",
            plugin_state_path="",
            synth_version="0.4.2",
        ),
        renderer_backend="pyfdn",
        pyfdn_excitation="impulse",
        sample_rate=44_100,
        channels=1,
        velocity=0,
        signal_duration_seconds=4.0,
        min_loudness=-100.0,
        audio_dtype="float32",
        mel_spec_dtype="float32",
        samples_per_render_batch=1,
        samples_per_shard=1,
        base_seed=31,
        attempts_per_sample=100,
        param_sample_cadence="sample",
        plugin_reload_cadence="render",
        gui_toggle_cadence="never",
    )
    spec = DatasetSpec.model_validate(
        {
            "task_name": "pyfdn-sketch-training-e2e",
            "output_format": "lance",
            "train_val_test_sizes": [1, 0, 0],
            "base_seed": 31,
            "r2": {"bucket": "intermediate-data"},
            "render": render.model_dump(mode="json"),
        }
    )
    generated_root = tmp_path / "generated"
    generated_root.mkdir()
    shard = spec.shards[0]
    generated_lance = generated_root / shard.filename
    generation = subprocess.run(  # noqa: S603 — argv comes from the validated production spec
        build_generate_args(spec, shard, generated_root),
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert generation.returncode == 0, generation.stderr[-2000:]

    before = lance.dataset(str(generated_lance))
    source_rows = before.to_table(columns=["audio", "param_array"])
    source_version = before.version
    subprocess.run(  # noqa: S603 — module and every Hydra override are test-owned
        [
            sys.executable,
            "-m",
            "synth_setter.pipeline.data.add_embeddings",
            f"lance_uri={generated_lance}",
            "embeddings=[pyfdn_sketch]",
            "batch_size=1",
            "build_index=false",
        ],
        check=True,
        timeout=120,
    )

    augmented = lance.dataset(str(generated_lance))
    assert augmented.count_rows() == 1
    assert augmented.version == source_version + 1
    assert augmented.to_table(columns=["audio", "param_array"]).equals(source_rows)
    sketch_type = augmented.schema.field("pyfdn_sketch").type
    assert [sketch_type.field(index).name for index in range(sketch_type.num_fields)] == [
        "edc",
        "echo_density",
        "spectral_flatness",
    ]

    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    for split in ("train", "val", "test"):
        shutil.copytree(generated_lance, dataset_root / f"{split}.lance")

    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=True,
            overrides=["experiment=pyfdn/flow_sketch", "trainer=cpu"],
        )
        with open_dict(cfg):
            cfg.paths.root_dir = str(operator_workspace())
            cfg.paths.output_dir = str(tmp_path / "training")
            cfg.paths.log_dir = str(tmp_path / "training")
            cfg.logger = None
            cfg.training.val_audio_probe = False
            cfg.test = False
            cfg.trainer.max_epochs = 1
            cfg.trainer.max_steps = 1
            cfg.trainer.limit_train_batches = 1
            cfg.trainer.limit_val_batches = 0
            cfg.trainer.num_sanity_val_steps = 0
            cfg.trainer.log_every_n_steps = 1
            cfg.datamodule.dataset_root = str(dataset_root)
            cfg.datamodule.predict_file = str(dataset_root / "test.lance")
            cfg.datamodule.use_saved_mean_and_variance = False
            cfg.datamodule.batch_size = 1
            cfg.datamodule.ot = False
            cfg.datamodule.num_workers = 0
            cfg.datamodule.pin_memory = False
            cfg.model.compile = False
            cfg.model.scheduler = None
            cfg.model.sketch_dropout_rate = 0.0
            cfg.model.all_conditioning_dropout_rate = 0.0
            cfg.model.encoder.d_model = 16
            cfg.model.encoder.n_heads = 1
            cfg.model.encoder.n_layers = 1
            cfg.model.encoder.n_conditioning_outputs = 1
            cfg.model.vector_field.d_model = 16
            cfg.model.vector_field.num_heads = 1
            cfg.model.vector_field.d_ff = 16
            cfg.model.vector_field.num_layers = 1
            cfg.model.vector_field.projection.num_tokens = 2
            cfg.model.validation_sample_steps = 1
            cfg.model.test_sample_steps = 1
            cfg.callbacks.model_checkpoint.save_top_k = 0
            cfg.callbacks.model_checkpoint.save_last = True
            if "lr_monitor" in cfg.callbacks:
                del cfg.callbacks.lr_monitor
    GlobalHydra.instance().clear()
    HydraConfig().set_config(cfg)

    metrics, objects = train(cfg)
    train_losses = [value for name, value in metrics.items() if name.startswith("train/loss")]
    assert train_losses
    assert all(torch.isfinite(loss).all() for loss in train_losses)

    datamodule = objects["datamodule"]
    assert isinstance(datamodule, LanceVSTDataModule)
    datamodule.setup("fit")
    batch = next(iter(datamodule.train_dataloader()))
    sketch = batch["sketch_ctrl"]
    assert sketch.shape == (1, 10, 32)
    assert sketch.dtype == torch.float32
    assert torch.isfinite(sketch).all()
    assert torch.all((-1.0 <= sketch) & (sketch <= 1.0))
    assert torch.all(torch.diff(sketch[:, :8], dim=-1) <= 1e-6)

    trained_model = objects["model"]
    assert isinstance(trained_model, VSTFlowMatchingModule)
    checkpoint = Path(cfg.paths.output_dir) / "checkpoints" / "last.ckpt"
    model = VSTFlowMatchingModule.load_from_checkpoint(
        checkpoint, map_location="cpu", weights_only=False
    )
    assert model.sketch_tokens is not None
    control_tokens = model.sketch_tokens(sketch, torch.ones((1, 3), dtype=torch.bool))
    assert control_tokens.shape == (1, 32, 16)

    model.train()
    loss = model.training_step(batch, 0)
    loss.backward()
    projection_parameters = list(model.sketch_tokens.projections.parameters())
    assert projection_parameters
    for parameter in projection_parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad)

    model.eval()
    predictions = model.sample_batch(
        batch,
        noise=torch.zeros((1, PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC.encoded_width)),
        content_cfg_strength=1.0,
        sketch_cfg_strength=1.0,
        sample_steps=1,
    )
    assert predictions.shape == (1, PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC.encoded_width)
    assert torch.isfinite(predictions).all()
