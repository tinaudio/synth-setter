"""Deterministic online datasets for the fixed-source pyFDN instrument.

Example:
    ``PyFDNDataModule(source_path, sha256).setup("fit")`` builds fixed seeded splits.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import cast

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import RandomSampler

from synth_setter.conditioning import ConditioningMode
from synth_setter.data.pyfdn_instrument import PyFDNRenderer
from synth_setter.data.pyfdn_param_spec import PYFDN_N8_MONO_PARAM_SPEC
from synth_setter.data.sample_seed import derive_sample_seed

type PyFDNItem = tuple[torch.Tensor, torch.Tensor]
type PyFDNBatch = dict[str, torch.Tensor]


@dataclass(frozen=True, slots=True)
class _ProcessRendererKey:
    source_audio_path: Path
    source_audio_sha256: str
    synth_version: str
    sample_rate: int
    channels: int
    signal_duration_seconds: float
    process_id: int


@cache
def _make_process_renderer(key: _ProcessRendererKey) -> PyFDNRenderer:
    """Load one renderer for each source and process identity.

    :param key: Immutable renderer identity, including the current process.
    :returns: Lazily loaded renderer shared by matching datasets in this process.
    """
    return PyFDNRenderer(
        key.source_audio_path,
        key.source_audio_sha256,
        synth_version=key.synth_version,
        sample_rate=key.sample_rate,
        channels=key.channels,
        signal_duration_seconds=key.signal_duration_seconds,
    )


def collate_pyfdn_audio_dict(batch: Sequence[PyFDNItem]) -> PyFDNBatch:
    """Collate fixed-shape pyFDN rows into the audio-conditioned flow contract.

    :param batch: Non-empty rows with float32 audio shaped ``(1, 192_000)`` and
        unit-domain encoded patches shaped ``(1, 89)``.
    :returns: Native-amplitude ``audio`` shaped ``(batch, 192_000)`` and model-domain
        ``params`` and ``noise`` shaped ``(batch, 89)``, all float32.
    """
    audio = torch.cat([row[0] for row in batch], dim=0)
    encoded = torch.cat([row[1] for row in batch], dim=0)
    params = encoded * 2.0 - 1.0
    return {"audio": audio, "params": params, "noise": torch.randn_like(params)}


class PyFDNDataset(Dataset[PyFDNItem]):
    """Sample deterministic pyFDN patches and render one fixed source on demand."""

    def __init__(
        self,
        source_audio_path: str | Path,
        source_audio_sha256: str,
        *,
        num_samples: int,
        seed: int,
        synth_version: str = "0.4.2",
        sample_rate: int = 48_000,
        channels: int = 1,
        signal_duration_seconds: float = 4.0,
    ) -> None:
        """Bind one deterministic split and its checksum-pinned source.

        :param source_audio_path: Path to the fixed lossless source.
        :param source_audio_sha256: Expected SHA-256 of the source bytes.
        :param num_samples: Number of logical rows in this split.
        :param seed: Base seed folded with each row index.
        :param synth_version: Required installed pyFDN version.
        :param sample_rate: Fixed source and build sample rate in Hz.
        :param channels: Fixed source channel count.
        :param signal_duration_seconds: Fixed source duration in seconds.
        """
        self.num_samples = num_samples
        self.seed = seed
        self.source_audio_path = Path(source_audio_path)
        self.source_audio_sha256 = source_audio_sha256
        self.synth_version = synth_version
        self.sample_rate = sample_rate
        self.channels = channels
        self.signal_duration_seconds = signal_duration_seconds

    def __len__(self) -> int:
        """Return the configured split length.

        :returns: Split cardinality used to bound valid row indices.
        """
        return self.num_samples

    def _process_renderer(self) -> PyFDNRenderer:
        """Return the source renderer shared by datasets in this process.

        :returns: Lazily loaded process-local renderer.
        """
        return _make_process_renderer(
            _ProcessRendererKey(
                source_audio_path=self.source_audio_path,
                source_audio_sha256=self.source_audio_sha256,
                synth_version=self.synth_version,
                sample_rate=self.sample_rate,
                channels=self.channels,
                signal_duration_seconds=self.signal_duration_seconds,
                process_id=os.getpid(),
            )
        )

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
        source_audio_path: str | Path,
        source_audio_sha256: str,
        *,
        synth_version: str = "0.4.2",
        sample_rate: int = 48_000,
        channels: int = 1,
        signal_duration_seconds: float = 4.0,
        train_val_test_sizes: tuple[int, int, int] = (100_000, 10_000, 10_000),
        train_val_test_seeds: tuple[int, int, int] = (123, 456, 789),
        batch_size: int = 32,
        num_workers: int = 0,
        conditioning: ConditioningMode = "audio",
        persistent_workers: bool = True,
        val_num_workers: int = 0,
    ) -> None:
        """Configure deterministic online pyFDN splits and loaders.

        :param source_audio_path: Path to the fixed lossless source.
        :param source_audio_sha256: Expected SHA-256 of the source bytes.
        :param synth_version: Required installed pyFDN version.
        :param sample_rate: Fixed source and build sample rate in Hz.
        :param channels: Fixed source channel count.
        :param signal_duration_seconds: Fixed source duration in seconds.
        :param train_val_test_sizes: Row counts for train, validation, and test.
        :param train_val_test_seeds: Base seeds for train, validation, and test.
        :param batch_size: DataLoader batch size.
        :param num_workers: Worker processes for training and test loaders.
        :param conditioning: Model-batch modality; pyFDN supports raw audio only.
        :param persistent_workers: Keep positive-count workers alive between epochs.
        :param val_num_workers: Worker processes for the validation loader.
        :raises ValueError: Split seeds repeat or conditioning does not select raw audio.
        """
        if len(set(train_val_test_seeds)) != 3:
            raise ValueError("train, validation, and test seeds must be distinct")
        if conditioning != "audio":
            raise ValueError("pyFDN conditioning must be 'audio'")
        super().__init__()
        self.source_audio_path = Path(source_audio_path)
        self.source_audio_sha256 = source_audio_sha256
        self.synth_version = synth_version
        self.sample_rate = sample_rate
        self.channels = channels
        self.signal_duration_seconds = signal_duration_seconds
        self.train_val_test_sizes = train_val_test_sizes
        self.train_val_test_seeds = train_val_test_seeds
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.conditioning = conditioning
        self.persistent_workers = persistent_workers
        self.val_num_workers = val_num_workers

    def setup(self, stage: str | None = None) -> None:
        """Build only the deterministic splits required by a Lightning stage.

        :param stage: Lightning stage name, or ``None`` to build every split.
        """

        def dataset(size: int, seed: int) -> PyFDNDataset:
            return PyFDNDataset(
                self.source_audio_path,
                self.source_audio_sha256,
                num_samples=size,
                seed=seed,
                synth_version=self.synth_version,
                sample_rate=self.sample_rate,
                channels=self.channels,
                signal_duration_seconds=self.signal_duration_seconds,
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
