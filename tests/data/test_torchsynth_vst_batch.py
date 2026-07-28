"""TorchSynth's online loader must satisfy the VST module family's batch contract."""

import torch

from synth_setter.data.torchsynth_datamodule import (
    TorchSynthDataModule,
    TorchSynthItem,
    collate_audio_dict,
    collate_vst_dict,
)
from synth_setter.models.components.residual_mlp import ConditionalResidualMLP
from synth_setter.models.components.transformer import AudioSpectrogramTransformer
from synth_setter.models.vst_flow_matching_module import VSTFlowMatchingModule

_SAMPLE_RATE = 44_100
_SIGNAL_LENGTH = 4_410
_BATCH = 4
_NUM_PARAMS = 76


def _datamodule() -> TorchSynthDataModule:
    """Build a tiny online datamodule.

    :returns: A datamodule already set up for fitting.
    """
    datamodule = TorchSynthDataModule(
        sample_rate=_SAMPLE_RATE,
        signal_length=_SIGNAL_LENGTH,
        midi_pitch=60,
        train_val_test_sizes=(8, 4, 4),
        train_val_test_seeds=(1, 2, 3),
        batch_size=_BATCH,
        num_workers=0,
    )
    datamodule.setup("fit")
    return datamodule


def test_dict_batch_carries_the_vst_module_keys() -> None:
    """The dict format supplies everything the VST modules index off the batch."""
    batch = next(iter(_datamodule().train_dataloader()))
    assert set(batch) == {"params", "noise", "mel_spec", "audio"}
    assert batch["params"].shape == (_BATCH, _NUM_PARAMS)
    assert batch["noise"].shape == (_BATCH, _NUM_PARAMS)
    assert batch["audio"].shape == (_BATCH, _SIGNAL_LENGTH)
    assert batch["mel_spec"].shape[:2] == (_BATCH, 1)
    assert all(batch[key].dtype == torch.float32 for key in batch)


def test_dict_batch_params_are_in_model_space() -> None:
    """VST batches carry params in ``[-1, 1]``, not the renderer's ``[0, 1]``."""
    params = next(iter(_datamodule().train_dataloader()))["params"]
    assert params.min() >= -1.0
    assert params.max() <= 1.0
    assert params.min() < 0.0


def _known_rows(signal_length: int = 8) -> list[TorchSynthItem]:
    """Return two rows with distinct audio and parameter values.

    :param signal_length: Number of audio samples in each row.
    :returns: Two collator-ready rows.
    """
    renderer = torch.nn.Identity()
    return [
        (
            torch.full((1, signal_length), 0.25, dtype=torch.float32),
            torch.tensor([[0.0, 0.5]], dtype=torch.float32),
            renderer,
        ),
        (
            torch.full((1, signal_length), -0.5, dtype=torch.float32),
            torch.tensor([[1.0, 0.25]], dtype=torch.float32),
            renderer,
        ),
    ]


def test_collate_audio_dict_returns_exact_audio_batch_contract() -> None:
    """Audio-only collation emits three float32 tensors with exact shapes."""
    batch = collate_audio_dict(_known_rows())

    assert set(batch) == {"params", "noise", "audio"}
    assert batch["params"].shape == (2, 2)
    assert batch["noise"].shape == (2, 2)
    assert batch["audio"].shape == (2, 8)
    assert all(value.dtype == torch.float32 for value in batch.values())


def test_collate_audio_dict_maps_known_rows_to_model_space() -> None:
    """Known rows preserve order and use the exact ``2p - 1`` parameter mapping."""
    batch = collate_audio_dict(_known_rows())

    assert torch.equal(
        batch["audio"],
        torch.tensor([[0.25] * 8, [-0.5] * 8], dtype=torch.float32),
    )
    assert torch.equal(batch["params"], torch.tensor([[-1.0, 0.0], [1.0, -0.5]]))


def test_collate_vst_dict_composes_mel_with_audio_batch_contract() -> None:
    """Mel collation adds the Lance-compatible mel tensor to the audio batch."""
    batch = collate_vst_dict(_known_rows(signal_length=4_410), sample_rate=_SAMPLE_RATE)

    assert set(batch) == {"params", "noise", "mel_spec", "audio"}
    assert batch["params"].shape == (2, 2)
    assert batch["noise"].shape == (2, 2)
    assert batch["audio"].shape == (2, 4_410)
    assert batch["mel_spec"].shape == (2, 1, 128, 11)
    assert all(value.dtype == torch.float32 for value in batch.values())


def test_vst_flow_matching_module_trains_on_an_online_torchsynth_batch() -> None:
    """The production flow module consumes the online batch and produces real gradients."""
    torch.manual_seed(0)
    batch = next(iter(_datamodule().train_dataloader()))
    n_mels, n_frames = batch["mel_spec"].shape[-2:]
    num_layers = 4
    vector_field = ConditionalResidualMLP(
        n_params=_NUM_PARAMS, d_model=64, d_enc=64, conditioning_dim=64, num_layers=num_layers
    )
    encoder = AudioSpectrogramTransformer(
        d_model=64,
        n_heads=4,
        n_layers=2,
        n_conditioning_outputs=num_layers,
        patch_size=16,
        patch_stride=10,
        input_channels=1,
        spec_shape=(int(n_mels), int(n_frames)),  # pyright: ignore[reportArgumentType]
    )
    module = VSTFlowMatchingModule(
        encoder=encoder,
        vector_field=vector_field,
        optimizer=torch.optim.Adam,  # pyright: ignore[reportArgumentType]
        scheduler=None,  # pyright: ignore[reportArgumentType]
        num_params=_NUM_PARAMS,
        conditioning="mel",
    )

    loss, _, _, _ = module._train_step(batch)
    gradients = torch.autograd.grad(loss, [p for p in vector_field.parameters()])

    assert torch.isfinite(loss)
    flat = torch.cat([g.flatten() for g in gradients if g is not None])
    assert torch.isfinite(flat).all()
    assert (flat != 0).any()
