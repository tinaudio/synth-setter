"""Shared VST datamodule configuration and model-batch preparation."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NotRequired, Self, TypedDict

import numpy as np
import torch
from lightning import LightningDataModule
from pydantic import BaseModel, ConfigDict, model_validator

from synth_setter.conditioning import (
    SKETCH_CTRL_FIELD,
    SKETCH_PITCH_SLICE,
    Conditioning,
    EmbeddingConditioningSpec,
    SketchControls,
    SketchControlSpec,
    resolve_embedding_conditioning,
    resolve_sketch_controls,
)
from synth_setter.data.ot import _hungarian_match
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.data.lance_materialize import materialize_splits, subset_dirname

_SEED_BOUND = torch.iinfo(torch.int64).max
_MATERIALIZE_SPLITS = ("train", "val", "test")


# DOC601/DOC603: pydoclint can't read sphinx ``:ivar:`` docs, so TypedDict keys
# are documented in the docstring body instead.
class RawBatch(TypedDict):  # noqa: DOC601, DOC603
    """One batch of stored VST columns consumed by :func:`prepare_batch`.

    Shapes are ``(batch, ...)``: ``param_array`` is ``(batch, num_params)`` and
    always present; ``mel_spec`` is ``(batch, channels, n_mels, n_frames)``,
    ``music2latent`` is ``(batch, latent_dim, n_frames)``, ``conditioning`` is
    one configured fixed-shape embedding column, ``sketch_ctrl`` is ``(batch,
    num_sketch_controls, n_frames)``, and ``audio`` is ``(batch, channels,
    samples)``. Optional unread modalities may be absent or ``None``.
    """

    param_array: np.ndarray
    mel_spec: NotRequired[np.ndarray | None]
    music2latent: NotRequired[np.ndarray | None]
    conditioning: NotRequired[np.ndarray | None]
    sketch_ctrl: NotRequired[np.ndarray | None]
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
        "conditioning": raw.get("conditioning"),
        "sketch_ctrl": raw.get("sketch_ctrl"),
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


def validate_channel_statistics(
    mean: np.ndarray, std: np.ndarray, *, label: str
) -> None:
    """Reject non-finite or non-positive normalization statistics.

    :param mean: Mean array to subtract.
    :param std: Standard-deviation array to divide by.
    :param label: Statistic owner named in error messages (e.g. ``"conditioning"``).
    :raises ValueError: If values are non-finite or standard deviations are not positive.
    """
    if not np.isfinite(mean).all():
        raise ValueError(f"{label} mean must contain only finite values")
    if not np.isfinite(std).all():
        raise ValueError(f"{label} std must contain only finite values")
    if np.any(std <= 0):
        raise ValueError(f"{label} std values must be positive")


def prepare_batch(
    raw: RawBatch,
    *,
    mean: np.ndarray | None,
    std: np.ndarray | None,
    conditioning_mean: np.ndarray | None = None,
    conditioning_std: np.ndarray | None = None,
    rescale_params: bool,
    ot: bool,
    generator: torch.Generator,
    sketch_pitch_zero_threshold: float | None = None,
) -> dict[str, torch.Tensor | None]:
    """Turn one batch of stored columns into model-ready tensors.

    :param raw: Stored columns; see :class:`RawBatch` for keys and shapes.
    :param mean: Mel mean to subtract, or ``None`` to skip normalization.
    :param std: Mel standard deviation, or ``None`` to skip normalization.
    :param conditioning_mean: Per-channel conditioning mean, or ``None`` to skip.
    :param conditioning_std: Per-channel conditioning std, or ``None`` to skip.
    :param rescale_params: Whether to map parameters from ``[0, 1]`` to ``[-1, 1]``.
    :param ot: Whether to Hungarian-match noise to parameters.
    :param generator: RNG for the noise draw.
    :param sketch_pitch_zero_threshold: Zero-bin ``sketch_ctrl`` pitch
        activations below this value (#2614), or ``None`` to skip.
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

    conditioning_raw = raw.get("conditioning")
    if conditioning_raw is not None and conditioning_mean is not None and conditioning_std is not None:
        validate_channel_statistics(conditioning_mean, conditioning_std, label="conditioning")
        conditioning_raw = (conditioning_raw - conditioning_mean) / conditioning_std
    conditioning = (
        torch.from_numpy(conditioning_raw).to(dtype=torch.float32)
        if conditioning_raw is not None
        else None
    )
    if conditioning is not None and not torch.isfinite(conditioning).all():
        raise ValueError("conditioning float32 conversion produced non-finite values")

    sketch_raw = raw.get(SKETCH_CTRL_FIELD)
    if sketch_raw is not None:
        sketch = torch.from_numpy(sketch_raw).to(dtype=torch.float32)
        if sketch_pitch_zero_threshold is not None:
            # Clone: from_numpy shares storage, and binning must not mutate the
            # caller's stored batch.
            sketch = sketch.clone()
            pitch = sketch[:, SKETCH_PITCH_SLICE]
            sketch[:, SKETCH_PITCH_SLICE] = pitch.where(
                pitch >= sketch_pitch_zero_threshold, 0.0
            )
    else:
        sketch = None

    param_array = raw["param_array"]
    if rescale_params:
        param_array = param_array * 2 - 1
    params = torch.from_numpy(param_array).to(dtype=torch.float32)
    noise = torch.empty_like(params).normal_(generator=generator)
    if ot:
        noise, params, mel_spec, m2l, conditioning, sketch, audio = _hungarian_match(
            noise, params, mel_spec, m2l, conditioning, sketch, audio
        )

    return {
        "mel_spec": mel_spec.contiguous() if mel_spec is not None else None,
        "m2l": m2l.contiguous() if m2l is not None else None,
        "conditioning": (
            conditioning.contiguous() if conditioning is not None else None
        ),
        SKETCH_CTRL_FIELD: sketch.contiguous() if sketch is not None else None,
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


class _MaterializeConfig(BaseModel):
    """Strict materialization settings parsed at the Hydra boundary.

    .. attribute :: model_config

        Strict frozen-model configuration.

    .. attribute :: download_dataset_root_uri

        Hydration source URI, or ``None``.

    .. attribute :: download_dataset_txids

        Per-split transaction pins, or ``None`` for latest snapshots.

    .. attribute :: download_dataset_row_limit

        First-N row cap per split, or ``None``.
    """

    model_config = ConfigDict(strict=True, frozen=True)

    download_dataset_root_uri: str | None
    download_dataset_txids: dict[str, str] | None
    download_dataset_row_limit: int | None

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        """Reject incomplete or contradictory materialization settings.

        :returns: Validated settings.
        :raises ValueError: Materialization lacks a source or has invalid split keys.
        """
        materialize = (
            self.download_dataset_txids is not None
            or self.download_dataset_row_limit is not None
        )
        if materialize and not self.download_dataset_root_uri:
            raise ValueError(
                "download_dataset_txids and download_dataset_row_limit require "
                "download_dataset_root_uri"
            )
        if self.download_dataset_txids is None:
            return self
        missing = [
            split
            for split in _MATERIALIZE_SPLITS
            if split not in self.download_dataset_txids
        ]
        if missing:
            raise ValueError(f"download_dataset_txids is missing txids for splits: {missing}")
        unknown = sorted(set(self.download_dataset_txids) - set(_MATERIALIZE_SPLITS))
        if unknown:
            raise ValueError(f"download_dataset_txids has unknown split keys: {unknown}")
        return self


class VSTDataModule(LightningDataModule):
    """Store shared VST loader configuration and optionally hydrate data from R2.

    .. attribute :: shard_suffix

       Filename suffix for each split dataset.
    """

    shard_suffix = ".lance"

    # DOC502: the documented ValueError propagates from _MaterializeConfig.
    def __init__(  # noqa: DOC502
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
        conditioning: Conditioning = "mel",
        sketch: SketchControls = None,
        pin_memory: bool = True,
        *,
        param_spec_name: ParamSpecName,
        download_dataset_txids: dict[str, str] | None = None,
        download_dataset_row_limit: int | None = None,
    ) -> None:
        """Store configuration shared by concrete VST datamodules.

        :param dataset_root: Local directory holding per-split datasets. When
            hydrating, the splits land in a request-addressed subdirectory and
            ``self.dataset_root`` becomes that subdirectory.
        :param download_dataset_root_uri: R2 or file URI hydrated as a projected
            column subset; the loaders' read set is the only thing transferred.
        :param use_saved_mean_and_variance: Whether to apply saved mel statistics.
        :param batch_size: Samples per model batch.
        :param ot: Whether training batches use optimal-transport matching.
        :param num_workers: Worker processes per dataloader.
        :param fake: Whether to synthesize samples instead of reading Lance.
        :param repeat_first_batch: Whether non-predict loaders repeat their first full batch.
        :param predict_file: Prediction split; defaults to ``test.lance``. A path
            naming the configured ``dataset_root`` rebases onto the subset directory.
        :param conditioning: Legacy mel/m2l mode or a fixed-shape embedding spec.
        :param sketch: Optional sketch-control spec adding its stored column to
            every split's read set (#2612).
        :param pin_memory: Whether dataloaders pin returned tensors.
        :param param_spec_name: Registry key selecting parameter width.
        :param download_dataset_txids: Per-split transaction uuids pinning the
            source snapshots. Each split has independent transaction history.
        :param download_dataset_row_limit: First-N rows per split at materialization
            time. Without txids, disposable runs use the latest source snapshots.
        :raises ValueError: If the materialization settings are inconsistent —
            fail at construction, never silently hydrate the wrong data.
        """
        materialize_config = _MaterializeConfig(
            download_dataset_txids=(
                dict(download_dataset_txids)
                if download_dataset_txids is not None
                else None
            ),
            download_dataset_row_limit=download_dataset_row_limit,
            download_dataset_root_uri=download_dataset_root_uri,
        )
        super().__init__()
        configured_root = Path(dataset_root)
        self.download_dataset_root_uri = materialize_config.download_dataset_root_uri
        self.use_saved_mean_and_variance = use_saved_mean_and_variance
        self.batch_size = batch_size
        self.ot = ot
        self.num_workers = num_workers
        self.fake = fake
        self.repeat_first_batch = repeat_first_batch
        self.conditioning: Conditioning = conditioning
        self.embedding_conditioning: EmbeddingConditioningSpec | None = (
            resolve_embedding_conditioning(conditioning)
        )
        self.sketch_controls: SketchControlSpec | None = resolve_sketch_controls(sketch)
        self.pin_memory = pin_memory
        self.param_spec_name = param_spec_name
        self.download_dataset_txids = materialize_config.download_dataset_txids
        self.download_dataset_row_limit = materialize_config.download_dataset_row_limit
        predict_split = self._predict_split(predict_file, configured_root)
        self.projection = self._derive_projection(predict_split)
        self.dataset_root = self._resolve_dataset_root(configured_root, self.projection)
        self.predict_file = self._resolve_predict_file(predict_file, predict_split)

    def _conditioning_column(self) -> str:
        """Return the stored column backing the configured conditioning.

        :returns: Legacy mel column or the resolved embedding column.
        """
        spec = self.embedding_conditioning
        return "mel_spec" if spec is None else spec.column

    def _predict_split(
        self, predict_file: str | Path | None, configured_root: Path
    ) -> str | None:
        """Identify which materialized split, if any, also serves prediction.

        Resolved against the configured root rather than the subset directory,
        because the subset's name depends on this answer.

        :param predict_file: Configured prediction split, or ``None`` for the default.
        :param configured_root: Dataset root exactly as configured.
        :returns: Split name serving prediction, or ``None`` when served elsewhere.
        """
        if predict_file is None:
            return "test"
        path = Path(predict_file)
        if path.parent != configured_root:
            return None
        split_of = {f"{split}{self.shard_suffix}": split for split in _MATERIALIZE_SPLITS}
        return split_of.get(path.name)

    def _derive_projection(self, predict_split: str | None) -> dict[str, list[str]]:
        """Derive the columns to materialize for every split.

        :param predict_split: Split that also serves prediction, or ``None``.
        :returns: Columns the loaders read, keyed by split.
        """
        return {
            split: self._loader_columns(read_audio=split == predict_split)
            for split in _MATERIALIZE_SPLITS
        }

    def _resolve_dataset_root(
        self, configured_root: Path, projection: Mapping[str, Sequence[str]]
    ) -> Path:
        """Place the splits in a request-addressed subset directory when hydrating.

        :param configured_root: Dataset root exactly as configured.
        :param projection: Columns to materialize per split.
        :returns: Directory the split loaders read from.
        """
        if not self.download_dataset_root_uri:
            return configured_root
        return configured_root / subset_dirname(
            self._conditioning_column(),
            self.download_dataset_root_uri,
            txids=self.download_dataset_txids,
            projection=projection,
            row_limit=self.download_dataset_row_limit,
        )

    def _resolve_predict_file(
        self, predict_file: str | Path | None, predict_split: str | None
    ) -> Path:
        """Rebase a configured prediction split onto the resolved dataset root.

        Only a path naming a split this datamodule materializes moves; anything
        else is served from wherever it was configured.

        :param predict_file: Configured prediction split, or ``None`` for the default.
        :param predict_split: Split serving prediction, or ``None`` when served elsewhere.
        :returns: Prediction split path; defaults to the test split.
        """
        if predict_file is None:
            return self.dataset_root / f"test{self.shard_suffix}"
        if predict_split is not None:
            return self.dataset_root / f"{predict_split}{self.shard_suffix}"
        return Path(predict_file)

    def _loader_columns(self, *, read_audio: bool) -> list[str]:
        """Derive the stored columns the split loaders read.

        :param read_audio: Whether the split additionally serves prediction audio.
        :returns: Projection for one split — never user-configured.
        """
        columns = ["param_array", self._conditioning_column()]
        if self.sketch_controls is not None:
            columns.append(self.sketch_controls.column)
        if read_audio:
            columns.append("audio")
        return columns

    def prepare_data(self) -> None:
        """Rematerialize the projected splits under ``dataset_root`` when a source is configured."""
        if not self.download_dataset_root_uri:
            return
        if r2_io.is_r2_uri(self.download_dataset_root_uri):
            r2_io.ensure_r2_env_loaded()
        materialize_splits(
            self.download_dataset_root_uri,
            self.dataset_root,
            txids=self.download_dataset_txids,
            projection=self.projection,
            row_limit=self.download_dataset_row_limit,
            shard_suffix=self.shard_suffix,
        )



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
        from synth_setter.data.lance_torch import LanceMapDataset

        return LanceMapDataset
    raise AttributeError(name)
