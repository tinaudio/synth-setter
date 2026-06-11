import random
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar, Literal, Protocol

import h5py
import hdf5plugin  # noqa: F401  # side-effect import: registers HDF5 blosc filters for shard I/O
import numpy as np
import torch
from lightning import LightningDataModule

from synth_setter.data.ot import _hungarian_match
from synth_setter.data.vst.param_spec_registry import param_specs
from synth_setter.pipeline import r2_io

# Registry key whose spec width sizes fake-mode batches and seeds the
# datamodule default when no ``param_spec_name`` is configured.
_DEFAULT_PARAM_SPEC_NAME = "surge_xt"


class ShardColumn(Protocol):
    """One named array column of a shard — the read surface of ``h5py.Dataset``."""

    @property
    def shape(self) -> tuple[int, ...]:
        """``(num_rows, *per_row_shape)`` of the column."""
        ...

    def __getitem__(self, idx: slice | Sequence[int] | np.ndarray) -> np.ndarray:
        """Materialize the selected rows as one array.

        :param idx: Slice or per-row integer indices (samplers yield numpy arrays).
        :return: ``(len(idx), *per_row_shape)`` array.
        """
        ...


class ShardFile(Protocol):
    """Read handle over one dataset shard — the surface of ``h5py.File`` the readers use."""

    def __getitem__(self, name: str) -> ShardColumn:
        """Return the named column view.

        :param name: Column name within the shard.
        :return: Read view over that column.
        """
        ...

    def __bool__(self) -> bool:
        """True while open, False after ``close`` — the teardown tests' contract.

        :return: Whether the shard is still open.
        """
        ...

    def close(self) -> None:
        """Release the underlying storage handle."""
        ...


class VSTDataset(torch.utils.data.Dataset):
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    # Declared here because __init__ assigns it on two branches (None in fake mode).
    dataset_file: ShardFile | None

    def __init__(
        self,
        dataset_file: str | Path,
        batch_size: int,
        ot: bool = True,
        read_audio: bool = False,
        read_mel: bool = True,
        read_m2l: bool = False,
        use_saved_mean_and_variance: bool = True,
        rescale_params: bool = True,
        fake: bool = False,
        repeat_first_batch: bool = False,
        num_params: int | None = None,
    ):
        self.batch_size = batch_size
        self.ot = ot

        self.read_audio = read_audio
        self.read_mel = read_mel
        self.read_m2l = read_m2l

        self.rescale_params = rescale_params

        self.fake = fake
        # Fake-mode width only; real mode reads the width from the shard's param_array.
        self.num_params = (
            num_params if num_params is not None else len(param_specs[_DEFAULT_PARAM_SPEC_NAME])
        )
        if fake:
            self.dataset_file = None
            return

        self.repeat_first_batch = repeat_first_batch

        self.dataset_file = self._open(dataset_file)

        if use_saved_mean_and_variance:
            self._load_dataset_statistics(dataset_file)

    def _open(self, dataset_file: str | Path) -> ShardFile:
        """Open the shard read-only; storage-format subclasses override this hook.

        :param dataset_file: Path to the shard on disk.
        :return: Open read handle for the shard.
        """
        return h5py.File(dataset_file, "r")

    def _load_dataset_statistics(self, dataset_file: str | Path):
        # for /path/to/train.h5 we would expect to find /path/to/stats.npz
        # if not, we throw an error
        stats_file = VSTDataset.get_stats_file_path(dataset_file)
        if not stats_file.exists():
            raise FileNotFoundError(
                f"Could not find statistics file {stats_file}. \n"
                "Make sure to first run `src/synth_setter/pipeline/data/stats.py`."
            )

        with np.load(stats_file) as stats:
            self.mean = stats["mean"]
            self.std = stats["std"]

    @staticmethod
    def get_stats_file_path(dataset_file: str | Path) -> Path:
        dataset_file = Path(dataset_file)
        data_dir = dataset_file.parent
        return data_dir / "stats.npz"

    def __len__(self):
        if self.fake:
            return 10000

        return self.dataset_file["audio"].shape[0] // self.batch_size

    def _get_fake_item(self):
        audio = torch.randn(self.batch_size, 2, 44100 * 4) if self.read_audio else None
        mel_spec = torch.randn(self.batch_size, 2, 128, 401) if self.read_mel else None
        m2l = torch.randn(self.batch_size, 128, 42) if self.read_m2l else None
        param_array = torch.rand(self.batch_size, self.num_params)

        if self.rescale_params:
            param_array = param_array * 2 - 1

        noise = torch.randn_like(param_array)

        return dict(
            mel_spec=mel_spec,
            m2l=m2l,
            params=param_array,
            noise=noise,
            audio=audio,
        )

    def _index_dataset(self, ds: ShardColumn, idx: int | Sequence[int] | np.ndarray) -> np.ndarray:
        if self.repeat_first_batch:
            return ds[: self.batch_size]
        if isinstance(idx, int):
            start_idx = idx * self.batch_size
            end_idx = start_idx + self.batch_size

            return ds[start_idx:end_idx]

        elif isinstance(idx, tuple) and len(idx) == 2:
            return ds[idx[0] : idx[1]]

        return ds[idx]

    def __getitem__(self, idx: int | Sequence[int] | np.ndarray):
        if self.fake:
            return self._get_fake_item()

        if self.read_audio:
            audio = self._index_dataset(self.dataset_file["audio"], idx)
            audio = torch.from_numpy(audio).to(dtype=torch.float32)
        else:
            audio = None

        if self.read_mel:
            mel_spec = self._index_dataset(self.dataset_file["mel_spec"], idx)
            if self.mean is not None and self.std is not None:
                mel_spec = (mel_spec - self.mean) / self.std
            mel_spec = torch.from_numpy(mel_spec).to(dtype=torch.float32)
        else:
            mel_spec = None

        if self.read_m2l:
            m2l = self._index_dataset(self.dataset_file["music2latent"], idx)
            m2l = torch.from_numpy(m2l).to(dtype=torch.float32)
        else:
            m2l = None

        param_array = self._index_dataset(self.dataset_file["param_array"], idx)
        if self.rescale_params:
            param_array = param_array * 2 - 1
        param_array = torch.from_numpy(param_array).to(dtype=torch.float32)
        noise = torch.randn_like(param_array)
        if self.ot:
            noise, param_array, mel_spec, audio = _hungarian_match(
                noise, param_array, mel_spec, audio
            )

        return dict(
            mel_spec=mel_spec.contiguous() if mel_spec is not None else None,
            m2l=m2l.contiguous() if m2l is not None else None,
            params=param_array.contiguous(),
            noise=noise.contiguous(),
            audio=audio.contiguous() if audio is not None else None,
        )


class WithinChunkShuffledSampler(torch.utils.data.Sampler):
    """When we have a hdf5 dataset on disk with layout:
        shard1.h5
        shard2.h5
        ...
        shardN.h5
    and each shard is 10,000 samples long, we want to sample items within each block of
    10,000 rather than randomly sampling across the entire dataset, to reduce the
    number of concurrent file handles that h5py has to deal with.
    This is not always exactly possible, but we can minimize the number of inter-shard
    reads to only the boundaries.
    """

    def __init__(self, batch_size: int, num_batches: int, batches_per_group: int):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.batches_per_group = batches_per_group

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        num_groups = self.num_batches // self.batches_per_group
        samples_per_group = self.batches_per_group * self.batch_size
        group_sizes = [samples_per_group] * num_groups

        remaining = self.num_batches % self.batches_per_group
        if remaining > 0:
            num_groups += 1
            group_sizes.append(remaining * self.batch_size)

        indices = [
            np.random.permutation(group_size).reshape(-1, self.batch_size) + i * samples_per_group
            for i, group_size in enumerate(group_sizes)
        ]

        indices = np.concatenate(indices, axis=0)

        # shuffle by rows
        np.random.shuffle(indices)

        for row in indices:
            row.sort()
            yield row.tolist()


class ShuffledSampler(torch.utils.data.Sampler):
    def __init__(self, batch_size: int, num_batches: int):
        self.batch_size = batch_size
        self.num_batches = num_batches

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        samples = np.random.permutation(self.num_batches * self.batch_size)

        for i in range(self.num_batches):
            sample = samples[i * self.batch_size : (i + 1) * self.batch_size]
            sample = np.sort(sample)
            yield sample


class ShiftedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, batch_size: int, num_batches: int):
        self.batch_size = batch_size
        self.num_batches = num_batches

    def __len__(self):
        return self.num_batches - 1

    def __iter__(self):
        offset = random.randint(0, self.batch_size - 1)
        perm = np.random.permutation(self.num_batches - 1)
        for i in perm:
            yield (i * self.batch_size + offset, (i + 1) * self.batch_size + offset)


class VSTDataModule(LightningDataModule):
    # Storage-format extension points: subclasses pair a dataset class with its shard suffix.
    dataset_cls: ClassVar[type[VSTDataset]] = VSTDataset
    shard_suffix: ClassVar[str] = ".h5"

    def __init__(
        self,
        dataset_root: str | Path,
        download_dataset_root_uri: str | None = None,
        use_saved_mean_and_variance: bool = True,
        batch_size: int = 1024,
        ot: bool = True,
        num_workers: int = 0,
        fake: bool = False,
        repeat_first_batch: bool = False,
        predict_file: str | None = None,
        conditioning: Literal["mel", "m2l"] = "mel",
        pin_memory: bool = True,
        param_spec_name: str = _DEFAULT_PARAM_SPEC_NAME,
    ):
        super().__init__()

        self.dataset_root = Path(dataset_root)
        self.download_dataset_root_uri = download_dataset_root_uri
        self.use_saved_mean_and_variance = use_saved_mean_and_variance
        self.batch_size = batch_size
        self.ot = ot
        self.num_workers = num_workers
        self.fake = fake
        self.repeat_first_batch = repeat_first_batch
        self.predict_file = (
            predict_file
            if predict_file is not None
            else self.dataset_root / f"test{self.shard_suffix}"
        )
        self.conditioning = conditioning
        self.pin_memory = pin_memory
        self.param_spec_name = param_spec_name

    def prepare_data(self) -> None:
        """Hydrate ``dataset_root`` from R2 when a download URI is configured.

        Lightning calls this rank-0-only, before ``setup``, so the no-clobber R2
        copy runs once even under DDP. Opt-in: a ``None`` URI is a no-op and the
        local ``dataset_root`` is used as-is.
        """
        if not self.download_dataset_root_uri:
            return
        r2_io.ensure_r2_env_loaded()
        r2_io.download_dir_no_overwrite(self.download_dataset_root_uri, self.dataset_root)

    def setup(self, stage: str | None = None):
        # KeyError here fails fast on an unregistered param_spec_name.
        num_params = len(param_specs[self.param_spec_name])
        self.train_dataset = self.dataset_cls(
            self.dataset_root / f"train{self.shard_suffix}",
            batch_size=self.batch_size,
            ot=self.ot,
            use_saved_mean_and_variance=self.use_saved_mean_and_variance,
            fake=self.fake,
            repeat_first_batch=self.repeat_first_batch,
            read_mel=self.conditioning == "mel",
            read_m2l=self.conditioning == "m2l",
            num_params=num_params,
        )
        self.val_dataset = self.dataset_cls(
            self.dataset_root / f"val{self.shard_suffix}",
            batch_size=self.batch_size,
            ot=False,
            use_saved_mean_and_variance=self.use_saved_mean_and_variance,
            fake=self.fake,
            repeat_first_batch=self.repeat_first_batch,
            read_mel=self.conditioning == "mel",
            read_m2l=self.conditioning == "m2l",
            num_params=num_params,
        )
        self.test_dataset = self.dataset_cls(
            self.dataset_root / f"test{self.shard_suffix}",
            batch_size=self.batch_size,
            ot=False,
            use_saved_mean_and_variance=self.use_saved_mean_and_variance,
            fake=self.fake,
            repeat_first_batch=self.repeat_first_batch,
            read_mel=self.conditioning == "mel",
            read_m2l=self.conditioning == "m2l",
            num_params=num_params,
        )
        self.predict_dataset = self.dataset_cls(
            self.predict_file,
            batch_size=self.batch_size,
            ot=False,
            read_audio=True,
            use_saved_mean_and_variance=self.use_saved_mean_and_variance,
            fake=self.fake,
            read_mel=self.conditioning == "mel",
            read_m2l=self.conditioning == "m2l",
            num_params=num_params,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # sampler=WithinChunkShuffledSampler(
            #     self.batch_size, len(self.train_dataset), 4
            # ),
            sampler=ShiftedBatchSampler(self.batch_size, len(self.train_dataset)),
            # sampler=ShuffledSampler(self.batch_size, len(self.train_dataset)),
            # shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.predict_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def teardown(self, stage: str | None = None):
        # fake mode leaves dataset_file None (no h5 opened), so guard each close.
        for dataset in (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
            self.predict_dataset,
        ):
            if dataset.dataset_file is not None:
                dataset.dataset_file.close()


# Deprecated aliases: archived W&B run configs and external job scripts resolve the
# old ``_target_`` paths, so Hydra must keep finding these names.
SurgeXTDataset = VSTDataset
SurgeDataModule = VSTDataModule
