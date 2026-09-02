"""Deterministic online datasets for the canonical-source pyFDN instrument.

Example:
    ``PyFDNDataModule().setup("fit")`` builds fixed seeded splits.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from functools import cache
from typing import Any, cast

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import RandomSampler

from synth_setter.conditioning import ConditioningMode
from synth_setter.data.pyfdn_instrument import PyFDNRenderer
from synth_setter.data.pyfdn_param_spec import PYFDN_N8_MONO_PARAM_SPEC
from synth_setter.data.pyfdn_source import (
    PYFDN_SOURCE_SAMPLE_RATE_HZ,
    PyFDNSourceProvenance,
)
from synth_setter.data.sample_seed import derive_sample_seed

type PyFDNItem = tuple[torch.Tensor, torch.Tensor]
type PyFDNBatch = dict[str, torch.Tensor]


@cache
def _make_process_renderer(
    synth_version: str,
    _process_id: int,
) -> PyFDNRenderer:
    """Create one renderer per pyFDN version and process.

    :param synth_version: Required installed pyFDN version.
    :param _process_id: Process identity that prevents reuse of an inherited fork cache.
    :returns: Lazily generated renderer shared by datasets in this process.
    """
    return PyFDNRenderer(synth_version=synth_version)


def collate_pyfdn_audio_dict(batch: Sequence[PyFDNItem]) -> PyFDNBatch:
    """Collate fixed-shape pyFDN rows into the audio-conditioned flow contract.

    :param batch: Non-empty rows with float32 audio shaped ``(1, 192_000)`` and
        unit-domain encoded patches shaped ``(1, 91)``.
    :returns: Native-amplitude ``audio`` shaped ``(batch, 192_000)`` and model-domain
        ``params`` and ``noise`` shaped ``(batch, 91)``, all float32.
    """
    audio = torch.cat([row[0] for row in batch], dim=0)
    encoded = torch.cat([row[1] for row in batch], dim=0)
    params = encoded * 2.0 - 1.0
    return {"audio": audio, "params": params, "noise": torch.randn_like(params)}


class PyFDNDataset(Dataset[PyFDNItem]):
    """Sample deterministic pyFDN patches and render the canonical source on demand."""

    def __init__(
        self,
        *,
        num_samples: int,
        seed: int,
        synth_version: str = "0.4.2",
    ) -> None:
        """Bind one deterministic split to the canonical source.

        :param num_samples: Number of logical rows in this split.
        :param seed: Base seed folded with each row index.
        :param synth_version: Required installed pyFDN version.
        """
        self.num_samples = num_samples
        self.seed = seed
        self.synth_version = synth_version

    def __len__(self) -> int:
        """Return the configured split length.

        :returns: Split cardinality used to bound valid row indices.
        """
        return self.num_samples

    def _process_renderer(self) -> PyFDNRenderer:
        """Return the source renderer shared by datasets in this process.

        :returns: Lazily loaded process-local renderer.
        """
        return _make_process_renderer(self.synth_version, os.getpid())

    def __getitem__(self, index: int) -> PyFDNItem:
        """Sample and render one row from its derived deterministic seed.

        :param index: Logical row index.
        :returns: Float32 audio and its unit-domain encoded patch.
        :raises IndexError: The index is outside this finite split.
        """
        if not 0 <= index < self.num_samples:
            raise IndexError(f"index {index} is outside split of length {self.num_samples}")
        rng = np.random.default_rng(derive_sample_seed(self.seed, index))
        params, note_params = PYFDN_N8_MONO_PARAM_SPEC.sample(rng)
        encoded = PYFDN_N8_MONO_PARAM_SPEC.encode(params, note_params)
        renderer = self._process_renderer()
        audio = renderer.render(params)
        return torch.from_numpy(audio), torch.from_numpy(encoded).unsqueeze(0)


class PyFDNDataModule(LightningDataModule):
    """Serve fixed deterministic train, validation, and test pyFDN rows."""

    def __init__(
        self,
        *,
        synth_version: str = "0.4.2",
        sample_rate: int = PYFDN_SOURCE_SAMPLE_RATE_HZ,
        train_val_test_sizes: tuple[int, int, int] = (100_000, 10_000, 10_000),
        train_val_test_seeds: tuple[int, int, int] = (123, 456, 789),
        batch_size: int = 32,
        num_workers: int = 0,
        conditioning: ConditioningMode = "audio",
        persistent_workers: bool = True,
        val_num_workers: int = 0,
    ) -> None:
        """Configure deterministic online pyFDN splits and loaders.

        :param synth_version: Required installed pyFDN version.
        :param sample_rate: Canonical source sample rate; exactly 48000 Hz.
        :param train_val_test_sizes: Row counts for train, validation, and test.
        :param train_val_test_seeds: Base seeds for train, validation, and test.
        :param batch_size: DataLoader batch size.
        :param num_workers: Worker processes for training and test loaders.
        :param conditioning: Model-batch modality; pyFDN supports raw audio only.
        :param persistent_workers: Keep positive-count workers alive between epochs.
        :param val_num_workers: Worker processes for the validation loader.
        :raises ValueError: Sample rate, split seeds, or conditioning violate the contract.
        """
        if len(set(train_val_test_seeds)) != 3:
            raise ValueError("train, validation, and test seeds must be distinct")
        if sample_rate != PYFDN_SOURCE_SAMPLE_RATE_HZ:
            raise ValueError("pyFDN sample_rate must be exactly 48000")
        if conditioning != "audio":
            raise ValueError("pyFDN conditioning must be 'audio'")
        super().__init__()
        self.synth_version = synth_version
        self.sample_rate = sample_rate
        self.train_val_test_sizes = train_val_test_sizes
        self.train_val_test_seeds = train_val_test_seeds
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.conditioning = conditioning
        self.persistent_workers = persistent_workers
        self.val_num_workers = val_num_workers

    @property
    def source_provenance(self) -> PyFDNSourceProvenance:
        """Return the canonical source identity, implementation version, and digest.

        :returns: Fresh provenance suitable for run or dataset metadata.
        """
        return _make_process_renderer(self.synth_version, os.getpid()).source_provenance

    def state_dict(self) -> dict[str, PyFDNSourceProvenance]:
        """Bind Lightning checkpoints to the canonical source used by this process.

        :returns: Canonical source provenance for checkpoint persistence.
        """
        return {"source_provenance": self.source_provenance}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Require checkpoint source provenance to match the current canonical source.

        :param state_dict: Datamodule state restored by Lightning.
        :raises ValueError: Source provenance is absent, malformed, or mismatched.
        """
        checkpoint_provenance = state_dict.get("source_provenance")
        if not isinstance(checkpoint_provenance, dict):
            raise ValueError("checkpoint source provenance is missing or malformed")
        if checkpoint_provenance != self.source_provenance:
            raise ValueError("checkpoint source provenance does not match the canonical source")

    def setup(self, stage: str | None = None) -> None:
        """Build only the deterministic splits required by a Lightning stage.

        :param stage: Lightning stage name, or ``None`` to build every split.
        """

        def dataset(size: int, seed: int) -> PyFDNDataset:
            return PyFDNDataset(
                num_samples=size,
                seed=seed,
                synth_version=self.synth_version,
            )

        train_size, val_size, test_size = self.train_val_test_sizes
        train_seed, val_seed, test_seed = self.train_val_test_seeds
        if stage in (None, "fit"):
            self.train = dataset(train_size, train_seed)
            self.val = dataset(val_size, val_seed)
        elif stage == "validate":
            self.val = dataset(val_size, val_seed)
        if stage in (None, "test"):
            self.test = dataset(test_size, test_seed)

    def _loader(
        self,
        dataset: Dataset[PyFDNItem],
        *,
        num_workers: int,
        sampler: Sampler[int] | None = None,
    ) -> StatefulDataLoader[PyFDNBatch]:
        """Wrap one online split with the fixed flow collator.

        :param dataset: Deterministic online split to load.
        :param num_workers: Worker processes for this split.
        :param sampler: Optional stateful training index order; evaluation defaults to sequential.
        :returns: Loader emitting the audio-conditioned flow batch contract.
        """
        return cast(
            StatefulDataLoader[PyFDNBatch],
            StatefulDataLoader(
                dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=num_workers,
                persistent_workers=self.persistent_workers and num_workers > 0,
                collate_fn=collate_pyfdn_audio_dict,
            ),
        )

    def train_dataloader(self) -> StatefulDataLoader[PyFDNBatch]:
        """Return a shuffled loader over the fixed training rows.

        :returns: Batched deterministic training data.
        """
        return self._loader(
            self.train,
            num_workers=self.num_workers,
            sampler=RandomSampler(self.train),
        )

    def val_dataloader(self) -> StatefulDataLoader[PyFDNBatch]:
        """Return the fixed-order validation loader.

        :returns: Batched deterministic validation data.
        """
        return self._loader(self.val, num_workers=self.val_num_workers)

    def test_dataloader(self) -> StatefulDataLoader[PyFDNBatch]:
        """Return the fixed-order test loader.

        :returns: Batched deterministic test data.
        """
        return self._loader(self.test, num_workers=self.num_workers)
