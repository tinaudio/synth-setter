"""Hydra contract tests for shipped SLAP training experiments."""

from pathlib import Path

import hydra
import pytest
import torch
from hydra import compose, initialize_config_module
from omegaconf import DictConfig
from torch.utils.data import Dataset

from synth_setter.models.slap_module import SLAPModule
from tests.helpers.run_if import RunIf

_SLAP_EXPERIMENTS = ("surge/slap_ast_audio_vst_ff_param",)
_AST_TARGET = "synth_setter.models.components.transformer.AudioSpectrogramTransformer"


class _FixedPairDataset(Dataset):
    """Keep synthetic modalities and row identities paired through Lightning collation."""

    def __init__(self, batch: dict[str, torch.Tensor]) -> None:
        """Store a fixed paired batch.

        :param batch: Modality tensors and row identities sharing their leading dimension.
        """
        self.batch = batch

    def __len__(self) -> int:
        return len(self.batch["sample_id"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {name: value[index] for name, value in self.batch.items()}


def _compose_slap_experiment(experiment: str) -> DictConfig:
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        return compose(
            config_name="train.yaml",
            overrides=[f"experiment={experiment}"],
        )


def _shrink_ast_layers(cfg: DictConfig) -> None:
    cfg.model.param_encoder.encoder.n_layers = 1
    for arm in (cfg.model.audio_encoder,):
        if "_args_" not in arm.encoder:
            continue
        for layer in arm.encoder._args_:
            if layer.get("_target_") == _AST_TARGET:
                layer.n_layers = 1


@pytest.mark.parametrize("experiment", _SLAP_EXPERIMENTS)
def test_slap_experiment_instantiates_complete_model(experiment: str) -> None:
    """Hydra must resolve mel conditioning and a concrete SLAPModule together.

    :param experiment: Shipped SLAP experiment name.
    """
    cfg = _compose_slap_experiment(experiment)

    model = hydra.utils.instantiate(cfg.model)

    assert isinstance(model, SLAPModule)
    assert cfg.datamodule.conditioning == "mel"
    assert cfg.datamodule.ot is False
    assert cfg.model.audio_input_key == "mel"
    assert cfg.model.retrieval_eval is True
    assert cfg.datamodule.eval_sample_ids is True


@pytest.mark.parametrize("experiment", _SLAP_EXPERIMENTS)
def test_slap_model_accepts_paired_surge_batch(experiment: str) -> None:
    """Compute a finite objective and route gradients into every trainable weight.

    :param experiment: Shipped SLAP experiment name.
    """
    cfg = _compose_slap_experiment(experiment)
    _shrink_ast_layers(cfg)
    model = hydra.utils.instantiate(cfg.model)
    batch = {
        "audio": None,
        "mel": torch.randn(2, 2, 128, 401),
        "params": torch.rand(2, 7),
    }

    loss = model.training_step(batch, batch_idx=0)
    loss.backward()

    assert torch.isfinite(loss)
    assert all(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in model.audio_encoder.parameters()
    )
    assert all(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in model.param_encoder.parameters()
        if parameter.requires_grad
    )


def test_slap_param_arm_frozen_weights_are_only_unused_inverse_projection() -> None:
    """Only the projection's unused token-to-parameter half may skip training."""
    cfg = _compose_slap_experiment("surge/slap_ast_audio_vst_ff_param")
    _shrink_ast_layers(cfg)
    model = hydra.utils.instantiate(cfg.model)

    frozen = [
        name
        for name, parameter in model.param_encoder.named_parameters()
        if not parameter.requires_grad
    ]

    assert len(frozen) == 1
    assert frozen[0].endswith("projection._out_projection")


@pytest.mark.gpu
@RunIf(min_gpus=1)
@pytest.mark.slow
@pytest.mark.parametrize("experiment", _SLAP_EXPERIMENTS)
def test_slap_fixed_batch_lightning_learns_distinct_pairs(experiment: str) -> None:
    """Learn four synthetic pairs with one-layer backbones and production EMA updates.

    This bounded capacity smoke test is not full-model or held-out sound-matching quality.

    :param experiment: Shipped SLAP experiment name.
    """
    from lightning.pytorch import Trainer
    from torch.utils.data import DataLoader

    from synth_setter.evaluation.paired_retrieval import paired_retrieval_metrics

    torch.manual_seed(7)
    cfg = _compose_slap_experiment(experiment)
    _shrink_ast_layers(cfg)
    cfg.model.optimizer.lr = 1e-3
    model = hydra.utils.instantiate(cfg.model).cuda()
    batch = {
        "mel": torch.randn(4, 2, 128, 401),
        "params": 2 * torch.rand(4, 7) - 1,
        "sample_id": torch.arange(4),
    }
    device_batch = {name: value.cuda() for name, value in batch.items()}
    initial_targets = {
        name: value.detach().clone()
        for name, value in model.named_parameters()
        if name.startswith(("audio_ema.", "param_ema."))
    }
    model.eval()
    with torch.inference_mode():
        initial_loss = model._losses(device_batch)["total_loss"]
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        precision="32-true",
        max_steps=1000,
        max_epochs=-1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, train_dataloaders=DataLoader(_FixedPairDataset(batch), batch_size=4))
    model.cuda()
    assert model._ema_optimizer_steps.item() == trainer.global_step == 1000
    current_parameters = dict(model.named_parameters())
    for prefix in ("audio_ema.", "param_ema."):
        assert any(
            not torch.equal(value, current_parameters[name])
            for name, value in initial_targets.items()
            if name.startswith(prefix)
        )
    model.eval()
    with torch.inference_mode():
        loss = model._losses(device_batch)["total_loss"]
        audio = model.audio_encoder(device_batch["mel"])[2]
        params = model.param_encoder(device_batch["params"])[2]
        permutation = torch.tensor([2, 0, 3, 1], device="cuda")
        reordered = model.audio_encoder(device_batch["mel"][permutation])[2]
    torch.testing.assert_close(reordered, audio[permutation], rtol=1e-5, atol=1e-6)
    metrics = paired_retrieval_metrics(audio, params, batch["sample_id"].tolist())
    assert loss < 0.5 * initial_loss, (float(initial_loss), float(loss), metrics)
    assert metrics["audio/embedding_variance"] > 0
    assert metrics["param/embedding_variance"] > 0
    assert metrics["matched_similarity"] > metrics["mismatched_similarity"], metrics
    assert metrics["audio_to_param/recall_at_1"] >= 0.75, metrics
    assert metrics["param_to_audio/recall_at_1"] >= 0.75, metrics


@pytest.mark.gpu
@RunIf(min_gpus=1)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_slap_gpu_ragged_batch_duplicate_predictions_accept_roundoff(dtype: torch.dtype) -> None:
    """The shipped online arms tolerate the same row inferred in different batch shapes.

    :param dtype: Float32 or the shipped bf16 mixed-precision inference mode.
    """
    from synth_setter.evaluation.paired_retrieval import paired_retrieval_metrics

    torch.manual_seed(23)
    cfg = _compose_slap_experiment("surge/slap_ast_audio_vst_ff_param")
    model = hydra.utils.instantiate(cfg.model).cuda().eval()
    mel = torch.randn(3, 2, 128, 401, device="cuda")
    params = torch.rand(3, 7, device="cuda")
    with (
        torch.inference_mode(),
        torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32),
    ):
        audio_full = model.audio_encoder(mel)[2]
        params_full = model.param_encoder(params)[2]
        audio_single = model.audio_encoder(mel[:1])[2]
        params_single = model.param_encoder(params[:1])[2]
    result = paired_retrieval_metrics(
        torch.cat((audio_full, audio_single)),
        torch.cat((params_full, params_single)),
        [0, 1, 2, 0],
    )
    assert result["gallery_size"] == 3


def test_slap_real_lance_fit_sanity_and_checkpoint_test_score_full_gallery(tmp_path: Path) -> None:
    """A real train-produced checkpoint scores all rows after Lance collation.

    :param tmp_path: Local Lance split and checkpoint destination.
    """
    import numpy as np
    from lightning.pytorch import Trainer

    from tests.helpers.lance_fixtures import make_shard_columns, write_lance_shard

    cfg = _compose_slap_experiment("surge/slap_ast_audio_vst_ff_param")
    cfg.datamodule.dataset_root = str(tmp_path)
    cfg.datamodule.download_dataset_root_uri = None
    cfg.datamodule.use_saved_mean_and_variance = False
    cfg.datamodule.batch_size = 2
    cfg.datamodule.num_workers = 0
    cfg.datamodule.val_num_workers = 0
    rng = np.random.default_rng(4)
    for split in ("train", "val", "test"):
        columns = make_shard_columns(3, num_params=7)
        columns["mel_spec"] = rng.standard_normal((3, 2, 128, 401)).astype(np.float32)
        write_lance_shard(tmp_path / f"{split}.lance", columns)
    module = hydra.utils.instantiate(cfg.datamodule)
    model = hydra.utils.instantiate(cfg.model)
    trainer = Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        max_epochs=1,
        num_sanity_val_steps=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, datamodule=module)
    assert trainer.callback_metrics["retrieval/val/gallery_size"] == 3
    assert torch.isfinite(trainer.callback_metrics["retrieval/val/audio_to_param/mrr"])
    checkpoint = tmp_path / "slap.ckpt"
    trainer.save_checkpoint(checkpoint)
    restored = hydra.utils.instantiate(cfg.model)
    result = trainer.test(restored, datamodule=module, ckpt_path=checkpoint)[0]
    assert result["retrieval/test/gallery_size"] == 3
    assert result["retrieval/test/audio/embedding_variance"] > 0
    assert result["retrieval/test/param/embedding_variance"] > 0
    assert "loss/test/total_loss" in result
