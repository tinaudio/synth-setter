"""Shared VST datamodule configuration and model-batch preparation."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NotRequired, Self, TypedDict

import numpy as np
import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from lightning import LightningDataModule
from pydantic import BaseModel, ConfigDict, PositiveInt, model_validator

from synth_setter.conditioning import (
    NUM_SKETCH_CONTROLS,
    NUM_SKETCH_TRACK_ROWS,
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
from synth_setter.data.vst.shapes import MEL_N_MELS, make_spectrogram
from synth_setter.features.sketch_controls import extract_sketch_controls
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.data.lance_materialize import materialize_splits, subset_dirname
from synth_setter.sketch import pool_sketch_controls

_SEED_BOUND = torch.iinfo(torch.int64).max
_MATERIALIZE_SPLITS = ("train", "val", "test")


# DOC601/DOC603: pydoclint can't read sphinx ``:ivar:`` docs, so TypedDict keys
# are documented in the docstring body instead.
class RawBatch(TypedDict):  # noqa: DOC601, DOC603
    """One batch of stored VST columns consumed by :func:`prepare_batch`.

    Keys are stored Lance column names, not model-batch keys; ``prepare_batch``
    maps ``mel_spec`` onto the ``mel`` batch entry and ``music2latent`` onto ``m2l``.
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
    return _sketch_range_validation_error(raw.get("sketch_ctrl"))


def _sketch_range_validation_error(sketch: np.ndarray | None) -> str | None:
    """Return the first sketch-control range violation, if any.

    Row-group bounds are the storage contract in :mod:`synth_setter.conditioning`;
    ``_validate_sketch_column`` only samples row 0, so every row is checked here.

    :param sketch: Stored ``sketch_ctrl`` rows, or ``None`` when unread.
    :returns: Validation message, or ``None`` when every row is in range.
    """
    if sketch is None:
        return None
    tracks = sketch[:, :NUM_SKETCH_TRACK_ROWS]
    if np.any((tracks < -1) | (tracks > 1)):
        return "sketch_ctrl loudness/centroid values must be within [-1, 1]"
    pitch = sketch[:, SKETCH_PITCH_SLICE]
    if np.any((pitch < 0) | (pitch > 1)):
        return "sketch_ctrl pitch activations must be within [0, 1]"
    return None


@dataclass(frozen=True)
class PreparedAudioInputs:
    """Model-ready waveforms and features.

    .. attribute :: guide_audio

        Stereo guide waveform.
    .. attribute :: reference_audio

        Stereo reference waveform.
    .. attribute :: ref_mel

        Normalized reference mel.
    .. attribute :: sketch_controls

        Prepared guide controls.
    """

    guide_audio: Float[torch.Tensor, "channels samples"]
    reference_audio: Float[torch.Tensor, "channels samples"]
    ref_mel: Float[torch.Tensor, "channels mel_bins mel_frames"]
    sketch_controls: Float[torch.Tensor, f"{NUM_SKETCH_CONTROLS} sketch_frames"]


@jaxtyped(typechecker=beartype)
def normalize_mel(
    mel: Float[np.ndarray, "*shape"],
    mean: Float[np.ndarray, "*mean_shape"],
    std: Float[np.ndarray, "*std_shape"],
) -> Float[torch.Tensor, "*shape"]:
    """Normalize a mel array and convert it to finite float32 model input.

    :param mel: Finite mel values with any leading batch dimensions.
    :param mean: Finite mean broadcastable to ``mel``.
    :param std: Positive standard deviation broadcastable to ``mel``.
    :returns: Contiguous normalized float32 tensor with ``mel``'s shape.
    :raises ValueError: Normalization or float32 conversion is non-finite.
    """
    if not np.isfinite(mean).all():
        raise ValueError("mean must contain only finite values")
    if not np.isfinite(std).all():
        raise ValueError("std must contain only finite values")
    if np.any(std <= 0):
        raise ValueError("std values must be positive")
    with np.errstate(over="ignore", invalid="ignore"):
        normalized = (mel - mean) / std
    if not np.isfinite(normalized).all():
        raise ValueError("mel_spec normalization produced non-finite values")
    prepared = torch.from_numpy(normalized).to(dtype=torch.float32)
    if not torch.isfinite(prepared).all():
        raise ValueError("mel_spec float32 conversion produced non-finite values")
    return prepared.contiguous()


@jaxtyped(typechecker=beartype)
def prepare_sketch_controls(
    controls: Float[torch.Tensor, f"batch {NUM_SKETCH_CONTROLS} frames"],
    spec: SketchControlSpec,
) -> Float[torch.Tensor, f"batch {NUM_SKETCH_CONTROLS} prepared_frames"]:
    """Fit extracted controls to a sketch spec and apply its pitch zero-bin.

    :param controls: Float controls with track rows in ``[-1, 1]`` and pitch
        rows in ``[0, 1]``.
    :param spec: Resolved storage-frame and pitch-threshold contract.
    :returns: Float32 controls with ``spec.num_frames`` temporal frames.
    :raises ValueError: Controls are non-finite, out of range, or cannot be
        pooled down to the requested frame count.
    """
    if not torch.isfinite(controls).all():
        raise ValueError("sketch_ctrl contains non-finite values")
    tracks = controls[:, :NUM_SKETCH_TRACK_ROWS]
    if torch.any((tracks < -1) | (tracks > 1)):
        raise ValueError("sketch_ctrl loudness/centroid values must be within [-1, 1]")
    pitch = controls[:, SKETCH_PITCH_SLICE]
    if torch.any((pitch < 0) | (pitch > 1)):
        raise ValueError("sketch_ctrl pitch activations must be within [0, 1]")
    if spec.num_frames > controls.shape[-1]:
        raise ValueError(
            f"sketch_ctrl cannot expand from {controls.shape[-1]} to {spec.num_frames} frames"
        )

    prepared = controls.to(dtype=torch.float32)
    if prepared.shape[-1] != spec.num_frames:
        prepared = pool_sketch_controls(prepared, spec.num_frames)
    prepared = prepared.clone()
    pitch = prepared[:, SKETCH_PITCH_SLICE]
    prepared[:, SKETCH_PITCH_SLICE] = pitch.where(
        pitch >= spec.pitch_zero_threshold, 0.0
    )
    return prepared.contiguous()


@jaxtyped(typechecker=beartype)
def prepare_paired_audio_inputs(
    guide_audio: Float[torch.Tensor, "channels samples"],
    reference_audio: Float[torch.Tensor, "channels samples"],
    *,
    sample_rate: int,
    mean: Float[np.ndarray, "*mean_shape"],
    std: Float[np.ndarray, "*std_shape"],
    sketch_spec: SketchControlSpec,
) -> PreparedAudioInputs:
    """Prepare loaded guide/reference waveforms for sketch-conditioned inference.

    :param guide_audio: Stereo float32 guide waveform on the model sample grid.
    :param reference_audio: Stereo float32 reference waveform on the same grid.
    :param sample_rate: Waveform and feature sample rate in hertz.
    :param mean: Reference-mel normalization mean.
    :param std: Reference-mel normalization standard deviation.
    :param sketch_spec: Resolved checkpoint sketch contract.
    :returns: Validated model-ready waveforms, mel, and sketch controls.
    :raises ValueError: Waveforms or prepared features violate shape, dtype, range, or finiteness
        contracts.
    """
    for name, audio in (("guide_audio", guide_audio), ("reference_audio", reference_audio)):
        if audio.dtype != torch.float32:
            raise ValueError(f"{name} must have dtype torch.float32")
        if audio.shape[0] != 2:
            raise ValueError(f"{name} must have two channels, got {audio.shape[0]}")
        if not torch.isfinite(audio).all() or torch.any(audio.abs() > 1):
            raise ValueError(f"{name} must be finite and within [-1, 1]")

    raw_mel = make_spectrogram(reference_audio.detach().cpu().numpy(), sample_rate)
    ref_mel = normalize_mel(raw_mel, mean, std)
    sketch = extract_sketch_controls(guide_audio, sample_rate)
    sketch_controls = prepare_sketch_controls(sketch.unsqueeze(0), sketch_spec)[0]
    if ref_mel.shape[0] != 2 or ref_mel.shape[1] != MEL_N_MELS:
        raise ValueError(
            f"reference mel must have shape (2, {MEL_N_MELS}, frames), got {ref_mel.shape}"
        )
    return PreparedAudioInputs(
        guide_audio=guide_audio.contiguous(),
        reference_audio=reference_audio.contiguous(),
        ref_mel=ref_mel,
        sketch_controls=sketch_controls,
    )


def prepare_batch(
    raw: RawBatch,
    *,
    mean: np.ndarray | None,
    std: np.ndarray | None,
    rescale_params: bool,
    ot: bool,
    generator: torch.Generator,
    sketch_spec: SketchControlSpec | None = None,
) -> dict[str, torch.Tensor | None]:
    """Turn one batch of stored columns into model-ready tensors.

    :param raw: Stored columns; see :class:`RawBatch` for keys and shapes.
    :param mean: Mel mean to subtract, or ``None`` to skip normalization.
    :param std: Mel standard deviation, or ``None`` to skip normalization.
    :param rescale_params: Whether to map parameters from ``[0, 1]`` to ``[-1, 1]``.
    :param ot: Whether to Hungarian-match noise to parameters.
    :param generator: RNG for the noise draw.
    :param sketch_spec: Resolved sketch preparation contract, or ``None`` to
        pass stored controls through unchanged.
    :returns: Model batch with float32 contiguous tensors and ``None`` for unread
        modalities; the stored ``mel_spec`` column is emitted under the ``mel`` key,
        as ``music2latent`` is under ``m2l``.
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
            mel = normalize_mel(mel_raw, mean, std)
        else:
            mel = torch.from_numpy(mel_raw).to(dtype=torch.float32)
            if not torch.isfinite(mel).all():
                raise ValueError("mel_spec float32 conversion produced non-finite values")
    else:
        mel = None

    m2l_raw = raw.get("music2latent")
    m2l = torch.from_numpy(m2l_raw).to(dtype=torch.float32) if m2l_raw is not None else None

    conditioning_raw = raw.get("conditioning")
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
        if sketch_spec is not None:
            sketch = prepare_sketch_controls(sketch, sketch_spec)
    else:
        sketch = None

    param_array = raw["param_array"]
    if rescale_params:
        param_array = param_array * 2 - 1
    params = torch.from_numpy(param_array).to(dtype=torch.float32)
    noise = torch.empty_like(params).normal_(generator=generator)
    if ot:
        noise, params, mel, m2l, conditioning, sketch, audio = _hungarian_match(
            noise, params, mel, m2l, conditioning, sketch, audio
        )

    return {
        "mel": mel.contiguous() if mel is not None else None,
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

    Invalid values propagate :class:`ValueError` from :func:`load_mel_statistics`.
    """
    stats_file = Path(dataset_file).parent / "stats.npz"
    if not stats_file.exists():
        raise FileNotFoundError(
            f"Could not find statistics file {stats_file}. \n"
            "Make sure to first run `src/synth_setter/pipeline/data/stats.py`."
        )
    return load_mel_statistics(stats_file)


def load_mel_statistics(stats_file: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate one mel-statistics ``.npz``.

    :param stats_file: Path to the ``.npz`` holding ``mean`` and ``std``.
    :returns: Broadcasting ``(mean, std)`` arrays.
    :raises ValueError: If values are non-finite or standard deviations are not positive.
    """
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

    .. attribute :: high_memory_materialization

        Whether materialization uses high-memory Lance tuning.
    """

    model_config = ConfigDict(strict=True, frozen=True)

    download_dataset_root_uri: str | None
    download_dataset_txids: dict[str, str] | None
    download_dataset_row_limit: PositiveInt | None
    high_memory_materialization: bool

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
        high_memory_materialization: bool = False,
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
        :param high_memory_materialization: Whether to use high-memory Lance tuning.
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
            high_memory_materialization=high_memory_materialization,
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
        self.high_memory_materialization = (
            materialize_config.high_memory_materialization
        )
        predict_split = self._predict_split(predict_file, configured_root)
        self.projection = self._derive_projection(predict_split)
        self.dataset_root = self._resolve_dataset_root(configured_root, self.projection)
        self.predict_file = self._resolve_predict_file(predict_file, predict_split)

    def _conditioning_column(self) -> str:
        """Return the stored column backing the configured conditioning.

        :returns: Stored column for the raw mode or resolved embedding.
        """
        spec = self.embedding_conditioning
        if spec is not None:
            return spec.column
        return "audio" if self.conditioning == "audio" else "mel_spec"

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
        if read_audio and "audio" not in columns:
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
            high_memory_materialization=self.high_memory_materialization,
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
