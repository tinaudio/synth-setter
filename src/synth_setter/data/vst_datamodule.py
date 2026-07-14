import random
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import ClassVar, Literal, NotRequired, Protocol, TypedDict

import h5py
import hdf5plugin  # noqa: F401  # side-effect import: registers HDF5 blosc filters for shard I/O
import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from synth_setter.data.ot import _hungarian_match
from synth_setter.data.vst.param_spec_registry import param_specs
from synth_setter.pipeline import r2_io


# DOC601/DOC603: pydoclint can't read sphinx ``:ivar:`` docs, so TypedDict keys
# are documented in the docstring body instead.
class RawBatch(TypedDict):  # noqa: DOC601, DOC603
    """One batch of read shard columns, the input to :func:`prepare_batch`.

    Shapes are ``(batch, ...)``: ``param_array`` is ``(batch, num_params)`` and
    always present; ``mel_spec`` is ``(batch, channels, n_mels, n_frames)``,
    ``music2latent`` is ``(batch, latent_dim, n_frames)``, ``audio`` is
    ``(batch, channels, samples)``. An optional key whose read flag is off may
    be stored as ``None`` or omitted entirely; both read as unread.
    """

    param_array: np.ndarray
    mel_spec: NotRequired[np.ndarray | None]
    music2latent: NotRequired[np.ndarray | None]
    audio: NotRequired[np.ndarray | None]


def prepare_batch(
    raw: RawBatch,
    *,
    mean: np.ndarray | None,
    std: np.ndarray | None,
    rescale_params: bool,
    ot: bool,
    generator: torch.Generator,
) -> dict[str, torch.Tensor | None]:
    """Turn one batch of read columns into model-ready tensors.

    The single owner of per-batch semantics: mel normalization, param rescale,
    seeded noise, and optional Hungarian matching, in that order. Pure — it
    reads only ``raw`` and the passed args and never writes back into them, so a
    fixed ``generator`` seed makes the whole batch reproducible. The h5 and Lance
    read paths both route here so batch contents stay format-independent.

    :param raw: Read shard columns; see :class:`RawBatch` for keys and shapes.
        ``param_array`` ``(batch, num_params)`` is required; a missing or
        ``None`` optional modality drops to ``None`` in the output.
    :param mean: Mel mean to subtract, broadcasting over the batch axis, or
        ``None`` to skip normalization.
    :param std: Mel std to divide by, broadcasting over the batch axis, or
        ``None`` to skip normalization.
    :param rescale_params: Whether to map params from ``[0, 1]`` to ``[-1, 1]``.
    :param ot: Whether to Hungarian-match noise to params (and permute the
        conditioning rows to follow).
    :param generator: RNG for the noise draw; seed it to pin the batch.
    :returns: ``{"mel_spec", "m2l", "params", "noise", "audio"}`` with
        ``float32`` contiguous tensors, ``None`` for unread modalities; ``params``
        and ``noise`` are ``(batch, num_params)``.
    """
    audio_raw = raw.get("audio")
    audio = torch.from_numpy(audio_raw).to(dtype=torch.float32) if audio_raw is not None else None

    mel_raw = raw.get("mel_spec")
    if mel_raw is not None:
        if mean is not None and std is not None:
            mel_raw = (mel_raw - mean) / std
        mel_spec = torch.from_numpy(mel_raw).to(dtype=torch.float32)
    else:
        mel_spec = None

    m2l_raw = raw.get("music2latent")
    m2l = torch.from_numpy(m2l_raw).to(dtype=torch.float32) if m2l_raw is not None else None

    param_array = raw["param_array"]
    if rescale_params:
        param_array = param_array * 2 - 1
    params = torch.from_numpy(param_array).to(dtype=torch.float32)
    # empty_like(params) matches params' device/dtype/shape and normal_ accepts
    # the generator (randn_like does not), so noise can't land on a foreign device.
    noise = torch.empty_like(params).normal_(generator=generator)
    if ot:
        noise, params, mel_spec, audio = _hungarian_match(noise, params, mel_spec, audio)

    return {
        "mel_spec": mel_spec.contiguous() if mel_spec is not None else None,
        "m2l": m2l.contiguous() if m2l is not None else None,
        "params": params.contiguous(),
        "noise": noise.contiguous(),
        "audio": audio.contiguous() if audio is not None else None,
    }


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


# DOC601/DOC603: pydoclint can't read sphinx ``:ivar:`` docs, so class-level
# annotations are documented in the docstring body instead.
class VSTDataset(torch.utils.data.Dataset):  # noqa: DOC601, DOC603
    """Batch-indexed dataset over one shard of VST renders (HDF5 by default).

    Each ``__getitem__`` returns a whole batch (the dataloader uses ``batch_size=None``)
    of conditioning features, target params, and noise (Hungarian-matched to params when
    ``ot`` is set). Fake mode synthesises
    random tensors of the configured width without opening any file. ``mean`` and ``std``
    hold the optional saved mel statistics applied during reads (``None`` when unloaded);
    ``dataset_file`` is the open :class:`ShardFile` handle (``None`` in fake mode);
    ``generator`` is the RNG threaded into :func:`prepare_batch` for the per-batch
    noise draw — seeded from the global RNG at construction (so Lightning's
    ``seed_everything`` governs it) and re-seeded once per dataloader worker from
    the worker seed (seed it manually to pin a batch); fake mode leaves it at the
    default seed because fake noise never uses it.
    Storage-format subclasses override ``_open`` to read non-HDF5 shards.
    """

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
    ) -> None:
        """Open the shard and configure which features each batch reads.

        :param dataset_file: Path to the HDF5 shard; ignored when ``fake`` is set.
        :param batch_size: Number of samples returned per ``__getitem__``.
        :param ot: Whether to optimal-transport match noise to params per batch.
        :param read_audio: Whether to read the raw audio tensor.
        :param read_mel: Whether to read the mel-spectrogram conditioning tensor.
        :param read_m2l: Whether to read the music2latent conditioning tensor.
        :param use_saved_mean_and_variance: Whether to load and apply saved mel stats.
        :param rescale_params: Whether to rescale params from ``[0, 1]`` to ``[-1, 1]``.
        :param fake: Whether to synthesise random batches instead of reading the shard.
        :param repeat_first_batch: Whether every index returns the first batch.
        :param num_params: Required param width for fake mode; ignored for shard-backed datasets.
        :raises ValueError: If fake mode has no explicit ``num_params``.
        """
        self.batch_size = batch_size
        self.ot = ot

        # Seeded from a global-RNG draw so seed_everything governs noise (a bare
        # Generator() repeats a fixed default seed); workers re-seed on first read.
        # Fake mode never uses it, so it skips the draw to leave the global stream
        # untouched.
        self.generator = torch.Generator()
        if not fake:
            self.generator.manual_seed(int(torch.randint(2**63 - 1, (1,)).item()))
        self._worker_reseed_done = False

        self.read_audio = read_audio
        self.read_mel = read_mel
        self.read_m2l = read_m2l

        self.rescale_params = rescale_params

        self.fake = fake
        if fake:
            if num_params is None:
                raise ValueError("num_params is required when fake=True")
            self.num_params = num_params
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

    def _load_dataset_statistics(self, dataset_file: str | Path) -> None:
        """Load the mel mean and std saved alongside the shard.

        :param dataset_file: Shard path used to locate the sibling ``stats.npz``.
        :raises FileNotFoundError: If the expected ``stats.npz`` is missing.
        """
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
        """Return the ``stats.npz`` path that sits beside the shard.

        :param dataset_file: Shard path whose parent directory holds the stats file.
        :returns: Path to the sibling ``stats.npz``.
        """
        dataset_file = Path(dataset_file)
        data_dir = dataset_file.parent
        return data_dir / "stats.npz"

    def __len__(self) -> int:
        """Return the number of batches in the shard (or a fixed count in fake mode).

        :returns: Batch count for the dataset.
        """
        if self.fake:
            return 10000

        # param_array is the one always-present column; audio is optional on
        # Lance shards, so counting rows off it would KeyError before any read.
        return self.dataset_file["param_array"].shape[0] // self.batch_size

    def _get_fake_item(self) -> dict[str, torch.Tensor | None]:
        """Synthesise one random batch matching the configured feature flags.

        Bypasses :func:`prepare_batch`: fake batches are already tensors and need
        no normalization or OT. Fake noise deliberately draws from the global RNG,
        not ``self.generator`` — under workers, Lightning's ``pl_worker_init_function``
        re-seeds worker global RNGs, so both modes stay ``seed_everything``-reproducible.

        :returns: Mapping of feature names to random tensors (or ``None`` when unread).
        """
        audio = torch.randn(self.batch_size, 2, 44100 * 4) if self.read_audio else None
        mel_spec = torch.randn(self.batch_size, 2, 128, 401) if self.read_mel else None
        m2l = torch.randn(self.batch_size, 128, 42) if self.read_m2l else None
        param_array = torch.rand(self.batch_size, self.num_params)

        if self.rescale_params:
            param_array = param_array * 2 - 1

        noise = torch.randn_like(param_array)

        return {
            "mel_spec": mel_spec,
            "m2l": m2l,
            "params": param_array,
            "noise": noise,
            "audio": audio,
        }

    def _index_dataset(
        self, ds: ShardColumn, idx: int | tuple[int, int] | Sequence[int] | np.ndarray
    ) -> np.ndarray:
        """Slice one batch out of a shard column for the given index.

        :param ds: Shard column to slice.
        :param idx: Batch index, or a ``(start, stop)`` pair, or a sequence of rows.
        :returns: The selected batch as a NumPy array.
        """
        if self.repeat_first_batch:
            return ds[: self.batch_size]
        if isinstance(idx, int):
            start_idx = idx * self.batch_size
            end_idx = start_idx + self.batch_size

            return ds[start_idx:end_idx]

        elif isinstance(idx, tuple) and len(idx) == 2:
            return ds[idx[0] : idx[1]]

        return ds[idx]

    def _reseed_generator_in_worker(self) -> None:
        """Re-seed ``generator`` once per dataloader worker, on first batch access.

        Workers fork after ``__init__``, so every fork inherits identical
        generator state and would draw correlated noise; the DataLoader-assigned
        worker seed (``base_seed + worker_id``) is distinct per worker yet still
        derived from the global RNG, keeping ``seed_everything`` in charge. Done
        lazily instead of via ``worker_init_fn`` so Lightning's auto-added
        ``pl_worker_init_function`` (gated on ``worker_init_fn is None``) stays.
        """
        if self._worker_reseed_done:
            return
        worker_info = torch.utils.data.get_worker_info()
        # Latch only inside a worker: a parent-process read must not mark the
        # flag, or forks made afterwards would inherit it and skip re-seeding.
        if worker_info is None:
            return
        self._worker_reseed_done = True
        self.generator.manual_seed(worker_info.seed)

    def __getitem__(
        self, idx: int | tuple[int, int] | Sequence[int] | np.ndarray
    ) -> dict[str, torch.Tensor | None]:
        """Read one batch of features, params, and matched noise at ``idx``.

        :param idx: Batch index, or a ``(start, stop)`` pair, or a sequence of rows.
        :returns: Mapping of feature names to tensors (or ``None`` when unread).
        """
        if self.fake:
            return self._get_fake_item()
        self._reseed_generator_in_worker()

        # Every key is always present; an off read flag stores ``None`` rather
        # than omitting the key, so prepare_batch never relies on key-absence.
        raw: RawBatch = {
            "param_array": self._index_dataset(self.dataset_file["param_array"], idx),
            "audio": self._read_column("audio", idx, self.read_audio),
            "mel_spec": self._read_column("mel_spec", idx, self.read_mel),
            "music2latent": self._read_column("music2latent", idx, self.read_m2l),
        }
        return prepare_batch(
            raw,
            mean=self.mean,
            std=self.std,
            rescale_params=self.rescale_params,
            ot=self.ot,
            generator=self.generator,
        )

    def _read_column(
        self, name: str, idx: int | tuple[int, int] | Sequence[int] | np.ndarray, read: bool
    ) -> np.ndarray | None:
        """Slice one optional shard column, or skip it when its read flag is off.

        :param name: Column name within the shard.
        :param idx: Batch index, or a ``(start, stop)`` pair, or a sequence of rows.
        :param read: Whether this column's modality is enabled.
        :returns: The selected rows, or ``None`` when ``read`` is false.
        """
        if not read:
            return None
        return self._index_dataset(self.dataset_file[name], idx)


class WithinChunkShuffledSampler(torch.utils.data.Sampler):
    """Shuffle batches within fixed-size groups to limit concurrent HDF5 file handles.

    The dataset is stored as equal-length shards on disk. Sampling within each group of
    ``batches_per_group`` keeps reads local to a shard rather than spanning the whole
    dataset, so h5py only crosses shard boundaries at group edges.
    """

    def __init__(self, batch_size: int, num_batches: int, batches_per_group: int) -> None:
        """Configure the batch geometry the sampler shuffles within.

        :param batch_size: Number of samples per batch.
        :param num_batches: Total number of batches across the dataset.
        :param batches_per_group: Number of batches shuffled together as one group.
        """
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.batches_per_group = batches_per_group

    def __len__(self) -> int:
        """Return the total number of batches.

        :returns: Batch count for the sampler.
        """
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        """Yield shuffled batches of row indices, sorted within each batch.

        :yields list[int]: A batch of row indices as a sorted list.
        """
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

        np.random.shuffle(indices)

        for row in indices:
            row.sort()
            yield row.tolist()


class ShuffledSampler(torch.utils.data.Sampler):
    """Sample batches of randomly permuted row indices across the whole dataset."""

    def __init__(self, batch_size: int, num_batches: int) -> None:
        """Configure the batch geometry the sampler permutes over.

        :param batch_size: Number of samples per batch.
        :param num_batches: Total number of batches across the dataset.
        """
        self.batch_size = batch_size
        self.num_batches = num_batches

    def __len__(self) -> int:
        """Return the total number of batches.

        :returns: Batch count for the sampler.
        """
        return self.num_batches

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield shuffled batches of row indices, sorted within each batch.

        :yields np.ndarray: A batch of row indices as a sorted array.
        """
        samples = np.random.permutation(self.num_batches * self.batch_size)

        for i in range(self.num_batches):
            sample = samples[i * self.batch_size : (i + 1) * self.batch_size]
            sample = np.sort(sample)
            yield sample


class ShiftedBatchSampler(torch.utils.data.BatchSampler):
    """Sample contiguous batches shifted by a random per-epoch offset."""

    def __init__(self, batch_size: int, num_batches: int) -> None:
        """Configure the batch geometry the sampler shifts and permutes.

        :param batch_size: Number of samples per batch.
        :param num_batches: Total number of batches across the dataset.
        """
        self.batch_size = batch_size
        self.num_batches = num_batches

    def __len__(self) -> int:
        """Return the number of batches, minus one to leave room for the shift.

        :returns: Batch count available after the offset shift.
        """
        return self.num_batches - 1

    def __iter__(self) -> Iterator[tuple[int, int]]:
        """Yield ``(start, stop)`` batch bounds shifted by a random offset.

        :yield: A ``(start, stop)`` index pair for one batch.
        :ytype: tuple[int, int]
        """
        offset = random.randint(0, self.batch_size - 1)
        perm = np.random.permutation(self.num_batches - 1)
        for i in perm:
            yield (i * self.batch_size + offset, (i + 1) * self.batch_size + offset)


class VSTDataModule(LightningDataModule):
    """Lightning datamodule wiring VST shards into train/val/test/predict dataloaders.

    Optionally hydrates ``dataset_root`` from R2 in ``prepare_data``, builds a
    :class:`VSTDataset` per split in ``setup``, and closes open shards in ``teardown``.

    .. attribute :: dataset_cls

        Storage-format extension point: the dataset class each split opens.

    .. attribute :: shard_suffix

        Storage-format extension point: the shard filename suffix per split.
    """

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
        predict_file: str | Path | None = None,
        conditioning: Literal["mel", "m2l"] = "mel",
        pin_memory: bool = True,
        *,
        param_spec_name: str,
    ) -> None:
        """Store dataloader and dataset configuration for later ``setup``.

        :param dataset_root: Local directory holding the per-split shard files
            (``shard_suffix`` names the format).
        :param download_dataset_root_uri: R2 URI to hydrate ``dataset_root`` from, or
            ``None`` to use the local directory as-is.
        :param use_saved_mean_and_variance: Whether to load and apply saved mel stats.
        :param batch_size: Number of samples per batch.
        :param ot: Whether to optimal-transport match noise to params per batch.
        :param num_workers: Number of dataloader worker processes.
        :param fake: Whether datasets synthesise random batches instead of reading shards.
        :param repeat_first_batch: Whether every index returns the first batch.
        :param predict_file: Shard used for prediction; defaults to the test split.
        :param conditioning: Conditioning feature, either ``"mel"`` or ``"m2l"``.
        :param pin_memory: Whether dataloaders pin memory for faster host-to-device copies.
        :param param_spec_name: Registry key selecting the param spec width.
        """
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
            Path(predict_file)
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

    def setup(self, stage: str | None = None) -> None:
        """Build a ``dataset_cls`` dataset for each split.

        :param stage: Lightning stage hint (``fit``/``validate``/``test``/``predict``);
            unused because every split dataset is constructed eagerly.
        """
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

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader using the shifted-batch sampler.

        :returns: Dataloader over the training dataset.
        """
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            sampler=ShiftedBatchSampler(self.batch_size, len(self.train_dataset)),
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader.

        :returns: Dataloader over the validation dataset.
        """
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader.

        :returns: Dataloader over the test dataset.
        """
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def predict_dataloader(self) -> DataLoader:
        """Return the prediction dataloader.

        :returns: Dataloader over the prediction dataset.
        """
        return torch.utils.data.DataLoader(
            self.predict_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def teardown(self, stage: str | None = None) -> None:
        """Close every open shard handle.

        :param stage: Lightning stage hint; unused because all splits are closed together.
        """
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
