"""Shared VST datamodule configuration and model-batch preparation."""

from pathlib import Path
from typing import NotRequired, TypedDict

import numpy as np
import torch
from lightning import LightningDataModule

from synth_setter.conditioning import ConditioningMode
from synth_setter.data.ot import _hungarian_match
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.pipeline import r2_io

_SEED_BOUND = torch.iinfo(torch.int64).max


# DOC601/DOC603: pydoclint can't read sphinx ``:ivar:`` docs, so TypedDict keys
# are documented in the docstring body instead.
class RawBatch(TypedDict):  # noqa: DOC601, DOC603
    """One batch of stored VST columns consumed by :func:`prepare_batch`.

    Shapes are ``(batch, ...)``: ``param_array`` is ``(batch, num_params)`` and
    always present; ``mel_spec`` is ``(batch, channels, n_mels, n_frames)``,
    ``music2latent`` is ``(batch, latent_dim, n_frames)``, and ``audio`` is
    ``(batch, channels, samples)``. Optional unread modalities may be absent or
    ``None``.
    """

    param_array: np.ndarray
    mel_spec: NotRequired[np.ndarray | None]
    music2latent: NotRequired[np.ndarray | None]
    audio: NotRequired[np.ndarray | None]


def _raw_batch_validation_error(raw: RawBatch) -> str | None:
    """Return the first stored-value contract violation, if any.

    :param raw: Read shard columns to validate.
    :returns: Validation message, or ``None`` when every stored value is valid.
    """
    arrays = {
        "param_array": raw["param_array"],
        "mel_spec": raw.get("mel_spec"),
        "music2latent": raw.get("music2latent"),
        "audio": raw.get("audio"),
    }
    for column, array in arrays.items():
        if array is not None and not np.isfinite(array).all():
            return f"{column} contains non-finite values"
    params = raw["param_array"]
    if np.any((params < 0) | (params > 1)):
        return "param_array values must be within [0, 1]"
    audio = raw.get("audio")
    if audio is not None and np.any((audio < -1) | (audio > 1)):
        return "audio values must be within [-1, 1]"
    return None


def prepare_batch(
    raw: RawBatch,
    *,
    mean: np.ndarray | None,
    std: np.ndarray | None,
    rescale_params: bool,
    ot: bool,
    generator: torch.Generator,
) -> dict[str, torch.Tensor | None]:
    """Turn one batch of stored columns into model-ready tensors.

    :param raw: Stored columns; see :class:`RawBatch` for keys and shapes.
    :param mean: Mel mean to subtract, or ``None`` to skip normalization.
    :param std: Mel standard deviation, or ``None`` to skip normalization.
    :param rescale_params: Whether to map parameters from ``[0, 1]`` to ``[-1, 1]``.
    :param ot: Whether to Hungarian-match noise to parameters.
    :param generator: RNG for the noise draw.
    :returns: Model batch with float32 contiguous tensors and ``None`` for unread modalities.
    :raises ValueError: If stored or transformed values violate the numeric contract.
    """
    validation_error = _raw_batch_validation_error(raw)
    if validation_error is not None:
        raise ValueError(validation_error)

    audio_raw = raw.get("audio")
    audio = torch.from_numpy(audio_raw).to(dtype=torch.float32) if audio_raw is not None else None

    mel_raw = raw.get("mel_spec")
    if mel_raw is not None:
        if mean is not None and std is not None:
            with np.errstate(over="ignore", invalid="ignore"):
                mel_raw = (mel_raw - mean) / std
            if not np.isfinite(mel_raw).all():
                raise ValueError("mel_spec normalization produced non-finite values")
        mel_spec = torch.from_numpy(mel_raw).to(dtype=torch.float32)
        if not torch.isfinite(mel_spec).all():
            raise ValueError("mel_spec float32 conversion produced non-finite values")
    else:
        mel_spec = None

    m2l_raw = raw.get("music2latent")
    m2l = torch.from_numpy(m2l_raw).to(dtype=torch.float32) if m2l_raw is not None else None

    param_array = raw["param_array"]
    if rescale_params:
        param_array = param_array * 2 - 1
    params = torch.from_numpy(param_array).to(dtype=torch.float32)
    noise = torch.empty_like(params).normal_(generator=generator)
    if ot:
        noise, params, mel_spec, m2l, audio = _hungarian_match(
            noise, params, mel_spec, m2l, audio
        )

    return {
        "mel_spec": mel_spec.contiguous() if mel_spec is not None else None,
        "m2l": m2l.contiguous() if m2l is not None else None,
        "params": params.contiguous(),
        "noise": noise.contiguous(),
        "audio": audio.contiguous() if audio is not None else None,
    }


def draw_generator_seed() -> int:
    """Draw a noise-generator seed from the global PyTorch RNG.

    :returns: Seed for ``torch.Generator.manual_seed``.
    """
    return int(torch.randint(_SEED_BOUND, (1,)).item())


def ranked_generator_seed(base_seed: int, rank: int, num_workers: int = 1) -> int:
    """Namespace a PyTorch generator seed by distributed rank.

    :param base_seed: Process or worker seed before rank namespacing.
    :param rank: Distributed process rank.
    :param num_workers: Worker streams reserved per rank.
    :returns: Rank-specific seed accepted by ``manual_seed``.
    """
    return (base_seed + rank * num_workers) % (2**64)


def load_dataset_statistics(dataset_file: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate mel statistics stored beside a split.

    :param dataset_file: Split path whose parent contains ``stats.npz``.
    :returns: Broadcasting ``(mean, std)`` arrays.
    :raises FileNotFoundError: If ``stats.npz`` is missing.
    :raises ValueError: If values are non-finite or standard deviations are not positive.
    """
    stats_file = Path(dataset_file).parent / "stats.npz"
    if not stats_file.exists():
        raise FileNotFoundError(
            f"Could not find statistics file {stats_file}. \n"
            "Make sure to first run `src/synth_setter/pipeline/data/stats.py`."
        )
    with np.load(stats_file) as stats:
        mean = stats["mean"]
        std = stats["std"]
    if not np.isfinite(mean).all():
        raise ValueError("mean must contain only finite values")
    if not np.isfinite(std).all():
        raise ValueError("std must contain only finite values")
    if np.any(std <= 0):
        raise ValueError("std values must be positive")
    return mean, std


class VSTDataModule(LightningDataModule):
    """Store shared VST loader configuration and optionally hydrate data from R2.

    .. attribute :: shard_suffix

       Filename suffix for each split dataset.
    """

    shard_suffix = ".lance"

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
        conditioning: ConditioningMode = "mel",
        pin_memory: bool = True,
        *,
        param_spec_name: ParamSpecName,
    ) -> None:
        """Store configuration shared by concrete VST datamodules.

        :param dataset_root: Local directory holding per-split datasets.
        :param download_dataset_root_uri: R2 URI used to hydrate ``dataset_root``.
        :param use_saved_mean_and_variance: Whether to apply saved mel statistics.
        :param batch_size: Samples per model batch.
        :param ot: Whether training batches use optimal-transport matching.
        :param num_workers: Worker processes per dataloader.
        :param fake: Whether to synthesize samples instead of reading Lance.
        :param repeat_first_batch: Whether non-predict loaders repeat their first full batch.
        :param predict_file: Prediction split; defaults to ``test.lance``.
        :param conditioning: Conditioning feature, ``"mel"`` or ``"m2l"``.
        :param pin_memory: Whether dataloaders pin returned tensors.
        :param param_spec_name: Registry key selecting parameter width.
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
        self.conditioning: ConditioningMode = conditioning
        self.pin_memory = pin_memory
        self.param_spec_name = param_spec_name

    def prepare_data(self) -> None:
        """Hydrate ``dataset_root`` from R2 when configured."""
        if not self.download_dataset_root_uri:
            return
        r2_io.ensure_r2_env_loaded()
        r2_io.download_dir_no_overwrite(self.download_dataset_root_uri, self.dataset_root)


def __getattr__(name: str) -> object:
    """Resolve archived Surge aliases without creating an import cycle.

    :param name: Requested module attribute.
    :returns: Current Lance-backed compatibility target.
    :raises AttributeError: If ``name`` is not a compatibility alias.
    """
    if name == "SurgeDataModule":
        from synth_setter.data.lance_datamodule import LanceVSTDataModule

        return LanceVSTDataModule
    if name == "SurgeXTDataset":
        from synth_setter.data.lance_torch import LanceTensorMapDataset

        return LanceTensorMapDataset
    raise AttributeError(name)
