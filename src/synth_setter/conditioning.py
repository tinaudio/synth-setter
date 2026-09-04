"""Conditioning contracts shared across data and model layers."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

ConditioningMode = Literal["mel", "m2l", "audio"]
LEGACY_M2L_INPUT_SHAPE = (128, 42)
# Modes read straight from the model-batch entry of the same name. "audio" serves
# online-render synths, which have no stored mel because their audio only exists
# at training time.
RAW_CONDITIONING_MODES: frozenset[str] = frozenset({"mel", "audio"})
# Every cached embedding is collated here, whatever its stored column.
EMBEDDING_BATCH_KEY = "conditioning"

# Sketch-control storage contract (#2612), hosted here (``data.vst.shapes``
# re-exports it) so model modules import it without the VST runtime package.
# In-memory model-batch key; the stored layout nests under SKETCH_STRUCT_FIELD (#2707).
SKETCH_CTRL_FIELD: str = "sketch_ctrl"
# Stored Lance struct column and its child names (#2707); the datamodule
# reassembles the children into the flat SKETCH_CTRL_FIELD batch tensor.
SKETCH_STRUCT_FIELD: str = "sketch"
SKETCH_LOUDNESS_CHILD: str = "loudness"
SKETCH_CENTROID_CHILD: str = "centroid"
SKETCH_PITCH_CHILD: str = "pitch"
SKETCH_VEC_CHILD: str = "vec"
# PESTO mir-1k_g7 activation width: 128 semitones x 3 bins.
SKETCH_PITCH_BINS: int = 384
# Scalar tracks precede the pitch block.
NUM_SKETCH_TRACK_ROWS: int = 2
NUM_SKETCH_CONTROLS: int = NUM_SKETCH_TRACK_ROWS + SKETCH_PITCH_BINS
SKETCH_STORAGE_FRAMES: int = 32
SKETCH_LOUDNESS_ROW: int = 0
SKETCH_CENTROID_ROW: int = 1
SKETCH_PITCH_SLICE: slice = slice(NUM_SKETCH_TRACK_ROWS, NUM_SKETCH_CONTROLS)

PYFDN_SKETCH_STRUCT_FIELD: str = "pyfdn_sketch"
PYFDN_SKETCH_EDC_CHILD: str = "edc"
PYFDN_SKETCH_ECHO_DENSITY_CHILD: str = "echo_density"
PYFDN_SKETCH_SPECTRAL_FLATNESS_CHILD: str = "spectral_flatness"
PYFDN_SKETCH_EDC_BANDS: int = 8
PYFDN_SKETCH_CONTROLS: int = 10

SketchControlProfile = Literal["music", "pyfdn_reverb"]


@dataclass(frozen=True)
class SketchControlLayout:
    """Define channel groups for one sketch-control profile.

    .. attribute :: profile

        Profile discriminator.

    .. attribute :: group_names

        Projection and dropout group names in channel order.

    .. attribute :: group_slices

        Non-overlapping channel slices in group order.
    """

    profile: SketchControlProfile
    group_names: tuple[str, ...]
    group_slices: tuple[slice, ...]

    @property
    def group_widths(self) -> tuple[int, ...]:
        """Return projection input widths in group order."""
        return tuple(group.stop - group.start for group in self.group_slices)

    @property
    def num_controls(self) -> int:
        """Return the required input channel count."""
        return self.group_slices[-1].stop


SKETCH_CONTROL_LAYOUTS: Mapping[SketchControlProfile, SketchControlLayout] = {
    "music": SketchControlLayout(
        profile="music",
        group_names=("loudness", "centroid", "pitch"),
        group_slices=(
            slice(SKETCH_LOUDNESS_ROW, SKETCH_LOUDNESS_ROW + 1),
            slice(SKETCH_CENTROID_ROW, SKETCH_CENTROID_ROW + 1),
            SKETCH_PITCH_SLICE,
        ),
    ),
    "pyfdn_reverb": SketchControlLayout(
        profile="pyfdn_reverb",
        group_names=(
            PYFDN_SKETCH_EDC_CHILD,
            PYFDN_SKETCH_ECHO_DENSITY_CHILD,
            PYFDN_SKETCH_SPECTRAL_FLATNESS_CHILD,
        ),
        group_slices=(
            slice(0, PYFDN_SKETCH_EDC_BANDS),
            slice(PYFDN_SKETCH_EDC_BANDS, PYFDN_SKETCH_EDC_BANDS + 1),
            slice(PYFDN_SKETCH_EDC_BANDS + 1, PYFDN_SKETCH_CONTROLS),
        ),
    ),
}


def sketch_control_layout(profile: SketchControlProfile) -> SketchControlLayout:
    """Return the authoritative channel layout for a sketch profile.

    :param profile: Profile discriminator.
    :returns: Immutable group names and channel slices.
    """
    return SKETCH_CONTROL_LAYOUTS[profile]


class EmbeddingConditioningSpec(BaseModel):
    """Select one fixed-shape Lance embedding column for conditioning.

    .. attribute :: model_config

        Strict immutable Pydantic model configuration.

    .. attribute :: column

        Stored Lance column name.

    .. attribute :: input_shape

        Fixed per-row tensor shape expected from the column.
    """

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    column: str = Field(min_length=1)
    input_shape: tuple[PositiveInt, ...] = Field(min_length=1)


Conditioning = ConditioningMode | EmbeddingConditioningSpec | Mapping[str, object]


def conditioning_batch_key(conditioning: Conditioning) -> str:
    """Return the model-batch key holding the configured conditioning tensor.

    :param conditioning: Configured mode literal, parsed spec, or Hydra mapping.
    :returns: Batch key; a raw mode names its own entry, embeddings share one.
    """
    if resolve_embedding_conditioning(conditioning) is not None:
        return EMBEDDING_BATCH_KEY
    # Only RAW_CONDITIONING_MODES resolve to no embedding, and each names its key.
    return cast(str, conditioning)


def resolve_embedding_conditioning(
    conditioning: Conditioning,
) -> EmbeddingConditioningSpec | None:
    """Resolve generic embedding configuration while leaving raw modes unrouted.

    :param conditioning: Legacy literal, parsed spec, or Hydra mapping.
    :returns: Fixed-shape embedding spec, or ``None`` for a raw observation.
    :raises TypeError: If ``conditioning`` is neither a supported literal nor mapping.
    :raises ValueError: If an unsupported string literal is provided.
    """
    if isinstance(conditioning, str):
        if conditioning in RAW_CONDITIONING_MODES:
            return None
        if conditioning == "m2l":
            return EmbeddingConditioningSpec(
                column="music2latent", input_shape=LEGACY_M2L_INPUT_SHAPE
            )
        raise ValueError(f"unknown conditioning mode {conditioning!r}")
    if isinstance(conditioning, EmbeddingConditioningSpec):
        return conditioning
    if not isinstance(conditioning, Mapping):
        raise TypeError(
            f"conditioning must be 'mel', 'm2l', 'audio', or a mapping, got {conditioning!r}"
        )

    values = dict(conditioning)
    input_shape = values.get("input_shape")
    if isinstance(input_shape, Sequence) and not isinstance(input_shape, (str, tuple)):
        values["input_shape"] = tuple(input_shape)
    return EmbeddingConditioningSpec.model_validate(values)


class SketchControlSpec(BaseModel):
    """Select the stored sketch-control column and its token budget.

    The selected profile resolves to this module's authoritative channel layout.

    .. attribute :: model_config

        Strict immutable Pydantic model configuration.

    .. attribute :: column

        Stored Lance struct column name.

    .. attribute :: profile

        Channel layout and temporal tokenization contract.

    .. attribute :: num_frames

        Mel-grid frames per stored control row.

    .. attribute :: num_control_tokens

        Control tokens the time axis is resampled to.

    .. attribute :: pitch_zero_threshold

        Pitch activations below this zero-bin at batch preparation (#2614).
    """

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    column: str = Field(default=SKETCH_STRUCT_FIELD, min_length=1)
    profile: SketchControlProfile = "music"
    num_frames: PositiveInt
    num_control_tokens: PositiveInt = 32
    # Bounded to the documented [0, 1] activation range: a negative threshold
    # silently disables binning and one above 1 zeroes the whole pitch block.
    pitch_zero_threshold: float = Field(default=0.1, ge=0.0, le=1.0)

    @property
    def layout(self) -> SketchControlLayout:
        """Return the selected profile's channel layout."""
        return sketch_control_layout(self.profile)

    @model_validator(mode="after")
    def temporal_layout_matches_token_budget(self) -> "SketchControlSpec":
        """Reject profile configurations that lose the stored temporal contract.

        :returns: The validated sketch-control specification.
        :raises ValueError: Stored frames and model token counts violate the profile.
        """
        if self.profile == "pyfdn_reverb" and (
            self.num_frames != SKETCH_STORAGE_FRAMES
            or self.num_control_tokens != SKETCH_STORAGE_FRAMES
        ):
            raise ValueError(
                f"pyfdn_reverb sketch requires num_frames=num_control_tokens="
                f"{SKETCH_STORAGE_FRAMES}"
            )
        if (
            self.profile == "music"
            and self.num_frames == SKETCH_STORAGE_FRAMES
            and self.num_control_tokens != SKETCH_STORAGE_FRAMES
        ):
            raise ValueError(
                f"pooled sketch storage requires num_control_tokens={SKETCH_STORAGE_FRAMES}"
            )
        return self


type SketchControls = SketchControlSpec | Mapping[str, object] | None


def resolve_sketch_controls(sketch: SketchControls) -> SketchControlSpec | None:
    """Resolve optional sketch-control configuration from Hydra or code.

    :param sketch: ``None``, a parsed spec, or a Hydra mapping.
    :returns: Parsed spec, or ``None`` when sketch conditioning is off.
    :raises TypeError: If ``sketch`` is neither ``None``, a spec, nor a mapping.
    """
    if sketch is None or isinstance(sketch, SketchControlSpec):
        return sketch
    if not isinstance(sketch, Mapping):
        raise TypeError(f"sketch must be None, a spec, or a mapping, got {sketch!r}")
    return SketchControlSpec.model_validate(dict(sketch))
