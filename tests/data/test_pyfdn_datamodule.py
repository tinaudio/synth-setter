"""Deterministic online data contracts for the fixed-source pyFDN instrument."""

import inspect
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import SequentialSampler
from torchdata.stateful_dataloader.sampler import RandomSampler

from synth_setter.data.pyfdn_datamodule import PyFDNDataModule, PyFDNDataset
from synth_setter.data.pyfdn_instrument import PyFDNRenderer, params_to_fdn_build
from synth_setter.data.pyfdn_param_spec import PYFDN_N8_MONO_PARAM_SPEC
from synth_setter.data.sample_seed import derive_sample_seed


def test_pyfdn_dataset_same_index_is_deterministic(
    source_file: tuple[Path, str],
) -> None:
    """A split seed and row index fix both the patch and real rendered audio.

    :param source_file: Valid fixed source and checksum.
    """
    path, checksum = source_file
    dataset = PyFDNDataset(path, checksum, num_samples=2, seed=123)

    audio_a, params_a = dataset[0]
    audio_b, params_b = dataset[0]

    assert torch.equal(params_a, params_b)
    assert torch.equal(audio_a, audio_b)


def test_pyfdn_dataset_row_encodes_the_exact_derived_seed_sample(
    source_file: tuple[Path, str],
) -> None:
    """The stored label is exactly the spec sample for the derived row seed.

    :param source_file: Valid fixed source and checksum.
    """
    path, checksum = source_file
    _, encoded = PyFDNDataset(path, checksum, num_samples=3, seed=123)[2]
    params, notes = PYFDN_N8_MONO_PARAM_SPEC.sample(
        np.random.default_rng(derive_sample_seed(123, 2))
    )

    np.testing.assert_array_equal(
        encoded[0].numpy(), PYFDN_N8_MONO_PARAM_SPEC.encode(params, notes)
    )


@pytest.mark.parametrize("index", [-1, 1])
def test_pyfdn_dataset_index_outside_split_raises(
    source_file: tuple[Path, str], index: int
) -> None:
    """Direct access cannot synthesize rows outside the finite split.

    :param source_file: Valid fixed source and checksum.
    :param index: Negative or upper-bound index outside a one-row split.
    """
    path, checksum = source_file
    dataset = PyFDNDataset(path, checksum, num_samples=1, seed=123)

    with pytest.raises(IndexError, match="outside"):
        dataset[index]


def test_pyfdn_dataset_different_indices_change_sampled_patch(
    source_file: tuple[Path, str],
) -> None:
    """Distinct derived row seeds produce distinct fixed training patches.

    :param source_file: Valid fixed source and checksum.
    """
    path, checksum = source_file
    dataset = PyFDNDataset(path, checksum, num_samples=2, seed=123)

    first = dataset[0][1]
    second = dataset[1][1]

    assert not torch.equal(first, second)


def test_pyfdn_dataset_row_has_exact_online_contract(
    source_file: tuple[Path, str],
) -> None:
    """Rows expose only the channel-first audio and encoded labels training consumes.

    :param source_file: Valid fixed source and checksum.
    """
    path, checksum = source_file
    audio, encoded = PyFDNDataset(path, checksum, num_samples=1, seed=123)[0]

    assert audio.shape == (1, 192_000)
    assert encoded.shape == (1, 91)
    assert audio.dtype == encoded.dtype == torch.float32


def test_pyfdn_dataset_loads_source_once_per_process(
    source_file: tuple[Path, str],
) -> None:
    """After the first row loads the source, later rows need no filesystem read.

    :param source_file: Valid fixed source and checksum.
    """
    path, checksum = source_file
    dataset = PyFDNDataset(path, checksum, num_samples=2, seed=123)
    dataset[0]
    path.unlink()

    audio, params = dataset[1]

    assert audio.shape == (1, 192_000)
    assert params.shape == (1, 91)


def test_pyfdn_datasets_share_one_source_load_within_a_process(
    source_file: tuple[Path, str],
) -> None:
    """Separate splits in one process reuse the validated decoded source.

    :param source_file: Valid fixed source and checksum.
    """
    path, checksum = source_file
    first = PyFDNDataset(path, checksum, num_samples=1, seed=123)
    second = PyFDNDataset(path, checksum, num_samples=1, seed=456)
    first[0]
    path.unlink()

    audio, _ = second[0]

    assert audio.shape == (1, 192_000)


def test_pyfdn_datamodule_default_split_seed_domains_are_disjoint() -> None:
    """Every production-default row derives a unique seed across all splits."""
    datamodule = PyFDNDataModule("unused.wav", "0" * 64)
    datamodule.setup(None)

    assert (datamodule.train.seed, datamodule.val.seed, datamodule.test.seed) == (
        123,
        456,
        789,
    )
    splits = (datamodule.train, datamodule.val, datamodule.test)
    derived_seeds = [
        derive_sample_seed(split.seed, index)
        for split in splits
        for index in range(len(split))
    ]
    expected_rows = sum(len(split) for split in splits)
    assert len(derived_seeds) == expected_rows
    assert len(set(derived_seeds)) == expected_rows


def test_pyfdn_datamodule_duplicate_split_seeds_raise(
    source_file: tuple[Path, str],
) -> None:
    """Matching split seeds cannot leak identical indexed rows into evaluation.

    :param source_file: Valid fixed source and checksum.
    """
    path, checksum = source_file

    with pytest.raises(ValueError, match="distinct"):
        PyFDNDataModule(path, checksum, train_val_test_seeds=(123, 123, 789))


def test_pyfdn_datamodule_non_audio_conditioning_raises() -> None:
    """The online batch contract rejects unsupported conditioning modes."""
    with pytest.raises(ValueError, match="conditioning must be 'audio'"):
        PyFDNDataModule("unused.wav", "0" * 64, conditioning="mel")


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


@pytest.mark.parametrize("num_workers", [0, 2])
def test_pyfdn_training_loader_restores_remaining_shuffled_batches(
    source_file: tuple[Path, str], num_workers: int
) -> None:
    """A checkpointed loader resumes after consumed batches without repeats or skips.

    :param source_file: Valid fixed source and checksum.
    :param num_workers: Worker count, including the prefetching production path.
    """
    path, checksum = source_file
    datamodule = PyFDNDataModule(
        path,
        checksum,
        train_val_test_sizes=(4, 1, 1),
        batch_size=1,
        num_workers=num_workers,
        persistent_workers=False,
    )
    datamodule.setup("fit")
    loader = datamodule.train_dataloader()
    iterator = iter(loader)
    next(iterator)
    state = loader.state_dict()
    uninterrupted = [batch["params"] for batch in iterator]

    resumed = datamodule.train_dataloader()
    resumed.load_state_dict(state)

    assert len(uninterrupted) == 3
    assert all(
        torch.equal(actual["params"], expected)
        for actual, expected in zip(resumed, uninterrupted, strict=True)
    )


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
    assert batch["params"].shape == batch["noise"].shape == (2, 91)
    assert all(value.dtype == torch.float32 for value in batch.values())
    assert torch.isfinite(batch["audio"]).all()
    assert torch.all((-1.0 <= batch["params"]) & (batch["params"] <= 1.0))


@pytest.mark.dataloader_multiprocess
@pytest.mark.xdist_group(name="dataloader-multiprocess")
@pytest.mark.slow
def test_pyfdn_datamodule_multiprocess_workers_render_finite_batches(
    source_file: tuple[Path, str],
) -> None:
    """Worker processes load and reuse their own fixed-source renderer.

    :param source_file: Valid fixed source and checksum.
    """
    path, checksum = source_file
    datamodule = PyFDNDataModule(
        path,
        checksum,
        train_val_test_sizes=(4, 1, 1),
        batch_size=2,
        num_workers=2,
        persistent_workers=False,
    )
    datamodule.setup("fit")

    batches = list(datamodule.train_dataloader())

    assert len(batches) == 2
    assert all(torch.isfinite(batch["audio"]).all() for batch in batches)


def test_pyfdn_public_signatures_omit_unsupported_sampling_modes() -> None:
    """Strict F1 APIs expose no sorting, filtering, dropping, or resampling flags."""
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
            "drop_last",
        }
    )


@pytest.mark.parametrize("factory", [PyFDNDataModule, PyFDNDataset])
def test_pyfdn_public_optional_configuration_is_keyword_only(factory: type) -> None:
    """Only the two required source-identity arguments accept positional binding.

    :param factory: Public online data constructor under test.
    """
    parameters = inspect.signature(factory).parameters

    assert parameters["source_audio_path"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameters["source_audio_sha256"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name in tuple(parameters)[2:]:
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


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
    item_audio, item_encoded = datamodule.test[0]

    batch = next(iter(datamodule.test_dataloader()))
    encoded = PYFDN_N8_MONO_PARAM_SPEC.model_to_encoded(batch["params"][0].numpy())
    decoded, decoded_notes = PYFDN_N8_MONO_PARAM_SPEC.decode(encoded)
    build = params_to_fdn_build(decoded, sample_rate=48_000.0)
    rerendered = PyFDNRenderer(path, checksum).render(decoded)

    assert decoded_notes == {}
    post_delay = build.post_delay
    assert post_delay is not None
    assert post_delay.shape == (1, 6, 8)
    assert post_delay.dtype == np.float64
    assert np.isfinite(post_delay).all()
    assert build.post_matrix is build.post_output is None
    np.testing.assert_array_equal(batch["audio"][0].numpy(), item_audio[0].numpy())
    np.testing.assert_allclose(rerendered, item_audio.numpy(), rtol=1e-4, atol=2e-5)
    np.testing.assert_allclose(item_encoded[0].numpy(), encoded, atol=2e-8)
