"""Full-gallery retrieval through Lightning's actual evaluation lifecycle."""

from functools import partial
from pathlib import Path
from typing import cast

import pytest
import torch
from lightning.pytorch import Trainer
from torch import nn
from torch.utils.data import DataLoader, Dataset

from synth_setter.models.components.slap import BYOLLoss, SiameseArm
from synth_setter.models.slap_module import SLAPModule


def _retrieval_model() -> SLAPModule:
    """Build real Siamese arms with controlled predictor geometry.

    :returns: Retrieval-enabled module.
    """
    arm = SiameseArm(nn.Identity(), nn.Identity(), nn.Linear(3, 3, bias=False))
    with torch.no_grad():
        cast(nn.Linear, arm.transform).weight.copy_(torch.eye(3))
    return SLAPModule(
        arm,
        param_encoder=arm,
        loss_fn=BYOLLoss(),
        optimizer=partial(torch.optim.SGD, lr=0.1),
        retrieval_eval=True,
    )


def _loader(rows: torch.Tensor, batch_size: int = 2) -> DataLoader:
    """Batch paired rows with stable source identities.

    :param rows: Paired input vectors.
    :param batch_size: Rows per forward.
    :returns: Ordered sample loader.
    """
    return DataLoader(
        cast(
            Dataset,
            [
                {"audio": row, "params": row, "sample_id": torch.tensor(i)}
                for i, row in enumerate(rows)
            ],
        ),
        batch_size=batch_size,
    )


def _trainer(tmp_path: Path) -> Trainer:
    """Build a local evaluation trainer without external logging.

    :param tmp_path: Trainer output location.
    :returns: CPU trainer.
    """
    return Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )


@pytest.mark.parametrize("stage", ["validate", "test"])
def test_retrieval_epoch_collapsed_predictions_use_full_gallery(
    tmp_path: Path, stage: str
) -> None:
    """Collapsed predictions expose batch-local scoring as an inflated Recall@1.

    :param tmp_path: Trainer output location.
    :param stage: Lightning evaluation entrypoint.
    """
    result = getattr(_trainer(tmp_path), stage)(_retrieval_model(), _loader(torch.ones(6, 3)))[0]
    split = "val" if stage == "validate" else "test"
    assert result[f"retrieval/{split}/audio_to_param/recall_at_1"] == pytest.approx(1 / 6)
    assert result[f"retrieval/{split}/gallery_size"] == 6
    assert f"loss/{split}/total_loss" in result


def test_retrieval_online_predictor_not_projection_or_ema_is_scored(tmp_path: Path) -> None:
    """Rotating only the online predictor defeats otherwise perfect projection retrieval.

    :param tmp_path: Trainer output location.
    """
    audio_arm = SiameseArm(nn.Identity(), nn.Identity(), nn.Linear(3, 3, bias=False))
    param_arm = SiameseArm(nn.Identity(), nn.Identity(), nn.Linear(3, 3, bias=False))
    with torch.no_grad():
        cast(nn.Linear, audio_arm.transform).weight.copy_(torch.eye(3))
        cast(nn.Linear, param_arm.transform).weight.copy_(torch.eye(3).roll(1, dims=0))
    model = SLAPModule(
        audio_arm,
        param_encoder=param_arm,
        loss_fn=BYOLLoss(),
        optimizer=partial(torch.optim.SGD, lr=0.1),
        retrieval_eval=True,
    )
    result = _trainer(tmp_path).validate(model, _loader(torch.eye(3)))[0]
    assert result["retrieval/val/audio_to_param/recall_at_1"] == 0


def test_retrieval_repeated_evaluation_resets_gallery(tmp_path: Path) -> None:
    """Repeated calls cannot retain old rows or singleton mismatch metrics.

    :param tmp_path: Trainer output location.
    """
    trainer, model = _trainer(tmp_path), _retrieval_model()
    trainer.validate(model, _loader(torch.eye(3)))
    result = trainer.validate(model, _loader(torch.ones(1, 3)))[0]
    assert result["retrieval/val/gallery_size"] == 1
    assert "retrieval/val/mismatched_similarity" not in result


def test_retrieval_multiple_loaders_keep_id_namespaces_separate(tmp_path: Path) -> None:
    """Different splits may reuse the same integer identities without collisions.

    :param tmp_path: Trainer output location.
    """
    results = _trainer(tmp_path).validate(
        _retrieval_model(), [_loader(torch.eye(3)), _loader(torch.ones(2, 3))]
    )
    assert results[0]["retrieval/val/gallery_size/dataloader_idx_0"] == 3
    assert results[1]["retrieval/val/gallery_size/dataloader_idx_1"] == 2


def test_retrieval_enabled_missing_ids_fails_without_inventing_batch_ids(tmp_path: Path) -> None:
    """Enabled retrieval rejects unidentified observations rather than fabricating IDs.

    :param tmp_path: Trainer output location.
    """
    loader = DataLoader(
        cast(Dataset, [{"audio": torch.ones(3), "params": torch.ones(3)}]), batch_size=1
    )
    with pytest.raises(ValueError, match="sample_id"):
        _trainer(tmp_path).validate(_retrieval_model(), loader)


def test_retrieval_lightning_two_cpu_ranks_score_one_global_gallery(tmp_path: Path) -> None:
    """Lightning's actual distributed sampler padding contributes only one candidate.

    :param tmp_path: Trainer output location.
    """
    trainer = Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        devices=2,
        strategy="ddp_spawn",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    result = trainer.validate(_retrieval_model(), _loader(torch.ones(5, 3)))[0]
    assert result["retrieval/val/gallery_size"] == 5
    assert result["retrieval/val/audio_to_param/recall_at_1"] == pytest.approx(0.2)
