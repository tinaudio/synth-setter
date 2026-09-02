"""Deterministic online data contracts for the fixed-source pyFDN instrument."""

import hashlib
import inspect
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from torch.utils.data import RandomSampler, SequentialSampler

from synth_setter.data.pyfdn_datamodule import PyFDNDataModule, PyFDNDataset
from synth_setter.data.pyfdn_instrument import params_to_fdn_build
from synth_setter.data.pyfdn_param_spec import PYFDN_N8_MONO_PARAM_SPEC
from synth_setter.data.sample_seed import derive_sample_seed


@pytest.fixture
def source_file(tmp_path: Path) -> tuple[Path, str]:
    """Write a checksum-pinned lossless source with the production geometry.

    :param tmp_path: Temporary directory owned by pytest.
    :returns: Source path and SHA-256 of its exact stored bytes.
    """
    path = tmp_path / "source.wav"
    time = np.arange(192_000, dtype=np.float64) / 48_000.0
    source = 0.1 * np.sin(2.0 * np.pi * 220.0 * time)
    sf.write(path, source, 48_000, subtype="PCM_16")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_pyfdn_dataset_same_index_is_deterministic(
    source_file: tuple[Path, str],
) -> None:
    """A split seed and row index fix both the patch and real rendered audio.

    :param source_file: Valid fixed source and checksum.
    """
    path, checksum = source_file
    dataset = PyFDNDataset(2, 123, path, checksum)

    audio_a, params_a, _ = dataset[0]
    audio_b, params_b, _ = dataset[0]

    assert torch.equal(params_a, params_b)
    assert torch.equal(audio_a, audio_b)


def test_pyfdn_dataset_row_encodes_the_exact_derived_seed_sample(
    source_file: tuple[Path, str],
) -> None:
    """The stored label is exactly the spec sample for the derived row seed.

    :param source_file: Valid fixed source and checksum.
    """
    path, checksum = source_file
    _, encoded, _ = PyFDNDataset(3, 123, path, checksum)[2]
    params, notes = PYFDN_N8_MONO_PARAM_SPEC.sample(
        np.random.default_rng(derive_sample_seed(123, 2))
    )

    np.testing.assert_array_equal(
        encoded[0].numpy(), PYFDN_N8_MONO_PARAM_SPEC.encode(params, notes)
    )


def test_pyfdn_dataset_different_indices_change_sampled_patch(
    source_file: tuple[Path, str],
) -> None:
    """Distinct derived row seeds produce distinct fixed training patches.

    :param source_file: Valid fixed source and checksum.
    """
    path, checksum = source_file
    dataset = PyFDNDataset(2, 123, path, checksum)

    first = dataset[0][1]
    second = dataset[1][1]

    assert not torch.equal(first, second)


def test_pyfdn_dataset_row_has_exact_online_contract(
    source_file: tuple[Path, str],
) -> None:
    """Rows expose channel-first audio, encoded labels, and the native render callable.

    :param source_file: Valid fixed source and checksum.
    """
    path, checksum = source_file
    audio, encoded, render = PyFDNDataset(1, 123, path, checksum)[0]
    params, notes = PYFDN_N8_MONO_PARAM_SPEC.decode(encoded[0].numpy())

    assert audio.shape == (1, 192_000)
    assert encoded.shape == (1, 89)
    assert audio.dtype == encoded.dtype == torch.float32
    assert notes == {}
    np.testing.assert_allclose(render(params), audio.numpy(), rtol=1e-4, atol=2e-5)


def test_pyfdn_dataset_loads_source_once_per_process(
    source_file: tuple[Path, str],
) -> None:
    """After the first row loads the source, later rows need no filesystem read.

    :param source_file: Valid fixed source and checksum.
    """
    path, checksum = source_file
    dataset = PyFDNDataset(2, 123, path, checksum)
    dataset[0]
    path.unlink()

    audio, params, _ = dataset[1]

    assert audio.shape == (1, 192_000)
    assert params.shape == (1, 89)


def test_pyfdn_datasets_share_one_source_load_within_a_process(
    source_file: tuple[Path, str],
) -> None:
    """Separate splits in one process reuse the validated decoded source.

    :param source_file: Valid fixed source and checksum.
    """
    path, checksum = source_file
    first = PyFDNDataset(1, 123, path, checksum)
    second = PyFDNDataset(1, 456, path, checksum)
    first[0]
    path.unlink()

    audio, _, _ = second[0]

    assert audio.shape == (1, 192_000)


def test_pyfdn_datamodule_uses_fixed_default_split_seeds(
    source_file: tuple[Path, str],
) -> None:
    """Train, validation, and test map indices through stable distinct seed domains.

    :param source_file: Valid fixed source and checksum.
    """
    path, checksum = source_file
    datamodule = PyFDNDataModule(
        path,
        checksum,
        train_val_test_sizes=(1, 1, 1),
        batch_size=1,
        num_workers=0,
    )
    datamodule.setup(None)

    assert (datamodule.train.seed, datamodule.val.seed, datamodule.test.seed) == (
        123,
        456,
        789,
    )


def test_pyfdn_datamodule_training_rows_remain_fixed_across_epochs(
    source_file: tuple[Path, str],
) -> None:
    """Training shuffles only indices; its finite row set does not resample by epoch.

    :param source_file: Valid fixed source and checksum.
    """
    path, checksum = source_file
    datamodule = PyFDNDataModule(
        path,
        checksum,
        train_val_test_sizes=(2, 1, 1),
        batch_size=1,
        num_workers=0,
    )
    datamodule.setup("fit")
    loader = datamodule.train_dataloader()

    first_epoch = {tuple(batch["params"].flatten().tolist()) for batch in loader}
    second_epoch = {tuple(batch["params"].flatten().tolist()) for batch in loader}

    assert first_epoch == second_epoch


def test_pyfdn_datamodule_loaders_shuffle_only_training_rows(
    source_file: tuple[Path, str],
) -> None:
    """Evaluation preserves row order while training may shuffle its fixed row set.

    :param source_file: Valid fixed source and checksum.
    """
    path, checksum = source_file
    datamodule = PyFDNDataModule(
        path,
        checksum,
        train_val_test_sizes=(1, 1, 1),
        num_workers=0,
    )
    datamodule.setup(None)

    assert isinstance(datamodule.train_dataloader().sampler, RandomSampler)
    assert isinstance(datamodule.val_dataloader().sampler, SequentialSampler)
    assert isinstance(datamodule.test_dataloader().sampler, SequentialSampler)


def test_pyfdn_datamodule_batch_matches_audio_conditioned_flow_contract(
    source_file: tuple[Path, str],
) -> None:
    """A real online batch has the keys, model-space labels, and shapes consumed by flow.

    :param source_file: Valid fixed source and checksum.
    """
    path, checksum = source_file
    datamodule = PyFDNDataModule(
        path,
        checksum,
        train_val_test_sizes=(2, 1, 1),
        batch_size=2,
        num_workers=0,
    )
    datamodule.setup("fit")

    batch = next(iter(datamodule.train_dataloader()))

    assert set(batch) == {"audio", "params", "noise"}
    assert batch["audio"].shape == (2, 192_000)
    assert batch["params"].shape == batch["noise"].shape == (2, 89)
    assert all(value.dtype == torch.float32 for value in batch.values())
    assert torch.isfinite(batch["audio"]).all()
    assert torch.all((-1.0 <= batch["params"]) & (batch["params"] <= 1.0))


def test_pyfdn_public_signatures_omit_unsupported_sampling_modes() -> None:
    """Strict F1 APIs expose no sorting, coprime, filter, or epoch-resampling flags."""
    names = set(inspect.signature(PyFDNDataModule).parameters) | set(
        inspect.signature(PyFDNDataset).parameters
    )

    assert names.isdisjoint(
        {
            "resample_train_per_epoch",
            "coprime",
            "sort",
            "filter_hooks",
            "sampling_mode",
        }
    )


def test_pyfdn_production_path_source_to_batch_decode_and_real_rerender(
    source_file: tuple[Path, str],
) -> None:
    """A generated source traverses sampling, real pyFDN, batching, decode, and rerender.

    :param source_file: Valid fixed source and checksum.
    """
    path, checksum = source_file
    datamodule = PyFDNDataModule(
        path,
        checksum,
        train_val_test_sizes=(1, 1, 1),
        batch_size=1,
        num_workers=0,
    )
    datamodule.setup("test")
    item_audio, item_encoded, render = datamodule.test[0]

    batch = next(iter(datamodule.test_dataloader()))
    encoded = PYFDN_N8_MONO_PARAM_SPEC.model_to_encoded(batch["params"][0].numpy())
    decoded, decoded_notes = PYFDN_N8_MONO_PARAM_SPEC.decode(encoded)
    build = params_to_fdn_build(decoded, sample_rate=48_000.0)
    rerendered = render(decoded)

    assert decoded_notes == {}
    assert build.post_delay is build.post_matrix is build.post_output is None
    np.testing.assert_array_equal(batch["audio"][0].numpy(), item_audio[0].numpy())
    np.testing.assert_allclose(rerendered, item_audio.numpy(), rtol=1e-4, atol=2e-5)
    np.testing.assert_allclose(item_encoded[0].numpy(), encoded, atol=2e-8)
