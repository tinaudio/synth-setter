from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np
from loguru import logger
from pydantic_settings import BaseSettings, CliApp, CliPositionalArg, SettingsConfigDict
from pyloudnorm import Meter

from synth_setter.data.vst.dawdreamer_runtime import ensure_dawdreamer_runtime
from synth_setter.data.vst.param_spec import NoteParams, ParamSpec
from synth_setter.data.vst.renderers import AudioRenderer
from synth_setter.data.vst.seeding import rng_for_sample
from synth_setter.data.vst.shapes import (
    MEL_N_MELS,
    MEL_WINDOW,
    mel_hop_length,
    mel_n_fft,
)
from synth_setter.pipeline.schemas.shard_metadata import DEFAULT_ATTEMPTS_PER_SAMPLE
from synth_setter.pipeline.schemas.spec import (
    OutputFormat,
    RenderConfig,
)

# Loudness-gate retry ceiling when a caller does not override it (#884).
DEFAULT_MAX_ATTEMPTS = DEFAULT_ATTEMPTS_PER_SAMPLE


@dataclass(frozen=True)
class SampleSeed:
    """Per-sample seeding inputs for ``generate_sample`` (#884).

    .. attribute :: master_seed

        Per-shard master seed (``ShardSpec.seed``), folded into every draw.

    .. attribute :: sample_idx

        Absolute row index, folded into the per-sample seed.

    .. attribute :: max_attempts

        Loudness-gate retry budget before the row fails loudly.
    """

    master_seed: int
    sample_idx: int = 0
    max_attempts: int = DEFAULT_MAX_ATTEMPTS


@dataclass
class VSTDataSample:
    synth_params: dict[str, float]
    note_params: NoteParams

    sample_rate: float
    channels: int

    param_spec: ParamSpec

    audio: np.ndarray
    mel_spec: np.ndarray
    param_array: np.ndarray = field(init=False)
    # Loudness-gate attempt the accepted draw came from (#884).
    attempt: int = 0

    def __post_init__(self) -> None:
        self.param_array = self.param_spec.encode(self.synth_params, self.note_params)


def make_spectrogram(audio: np.ndarray, sample_rate: float) -> np.ndarray:
    """Per-channel mel-spectrogram in dB; STFT params come from module-level constants."""
    spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=MEL_N_MELS,
        n_fft=mel_n_fft(sample_rate),
        hop_length=mel_hop_length(sample_rate),
        window=MEL_WINDOW,
        center=True,
    )
    spec_db = librosa.power_to_db(spec, ref=np.max)
    return spec_db


def generate_sample(
    renderer: AudioRenderer,
    velocity: int,
    min_loudness: float,
    param_spec: ParamSpec,
    fixed_synth_params: dict[str, float] | None = None,
    fixed_note_params: NoteParams | None = None,
    *,
    warmup: bool = False,
    seed: SampleSeed | None = None,
) -> VSTDataSample:
    """Render a single VST sample, retrying silent draws up to the attempt budget.

    When ``fixed_synth_params`` and/or ``fixed_note_params`` are supplied, they take
    precedence over the values drawn from ``param_spec.sample()`` for deterministic
    rendering. When ``fixed_synth_params`` is supplied (with or without
    ``fixed_note_params``), the function raises ``ValueError`` on loudness fail
    rather than retrying — the synth patch is the dominant determinant of loudness,
    so re-sampling note params alone almost never lifts a silent patch above
    ``min_loudness``. When only ``fixed_note_params`` (or nothing) is supplied, the
    synth is re-sampled per attempt and the loop is meaningful.

    With ``seed`` set, sampling draws from
    ``rng_for_sample(seed.master_seed, seed.sample_idx, attempt)`` so a given row is
    reproducible regardless of worker/order/retry history (#884); ``seed=None`` draws
    from a fresh non-deterministic generator and uses the default attempt budget.

    :param renderer: Audio host that renders every sample.
    :param velocity: MIDI velocity applied to each rendered note.
    :param min_loudness: Minimum integrated loudness required to accept a render.
    :param param_spec: Parameter specification used to sample synth and note values.
    :param fixed_synth_params: Optional synth values that replace sampled values.
    :param fixed_note_params: Optional note values that replace sampled values.
    :param warmup: Forwarded to the renderer; runs the backend's optional editor
        warm-up on the plugin used for this render (newly loaded or cached).
        Applied at most once per ``generate_sample`` call — the loudness-gate
        retry loop drops ``warmup`` to ``False`` after the first attempt so a
        retrying sample never exceeds the per-shard cadence budget (#714).
    :param seed: Per-sample seeding inputs; ``None`` samples non-deterministically.
    :returns: The accepted sample, with ``attempt`` set to the winning retry.
    :raises ValueError: If the attempt budget is nonpositive, or a
        ``fixed_synth_params`` render fell below ``min_loudness``.
    :raises RuntimeError: The sampling path stayed silent for the whole attempt budget.
    """
    max_attempts = seed.max_attempts if seed is not None else DEFAULT_MAX_ATTEMPTS
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
    for attempt in range(max_attempts):
        if fixed_synth_params is None or fixed_note_params is None:
            logger.debug("sampling params")
            rng = (
                rng_for_sample(seed.master_seed, seed.sample_idx, attempt)
                if seed is not None
                else None
            )
            sampled_synth, sampled_note = param_spec.sample(rng)
            synth_params = fixed_synth_params if fixed_synth_params is not None else sampled_synth
            note_params = fixed_note_params if fixed_note_params is not None else sampled_note
        else:
            synth_params = fixed_synth_params
            note_params = fixed_note_params

        output = renderer.render(
            synth_params,
            note_params["pitch"],
            velocity,
            note_params["note_start_and_end"],
            warmup=warmup,
        )
        warmup = False

        meter = Meter(renderer.sample_rate)
        loudness = meter.integrated_loudness(output.T)
        logger.debug(f"loudness: {loudness}")
        if loudness < min_loudness:
            if fixed_synth_params is not None:
                raise ValueError(
                    f"fixed_synth_params render produced loudness {loudness:.2f} dB "
                    f"below min_loudness {min_loudness:.2f} dB. The synth patch is "
                    f"held constant and dominates loudness, so retrying is futile "
                    f"(the fully-fixed case has no re-sample input at all; the "
                    f"only-synth-fixed case re-samples note params, which rarely "
                    f"lifts a silent patch above the threshold). Provide a louder "
                    f"patch."
                )
            logger.debug("loudness too low, skipping")
            continue

        logger.debug("making spectrogram")
        spectrogram = make_spectrogram(output, renderer.sample_rate)
        return VSTDataSample(
            synth_params=synth_params,
            note_params=note_params,
            audio=output.T,
            mel_spec=spectrogram,
            sample_rate=renderer.sample_rate,
            channels=renderer.channels,
            param_spec=param_spec,
            attempt=attempt,
        )

    failed_idx = str(seed.sample_idx) if seed is not None else "<unseeded>"
    seed_hint = (
        "Production callers should pass SampleSeed/base_seed for reproducibility; "
        if seed is None
        else ""
    )
    raise RuntimeError(
        f"sample {failed_idx} stayed below min_loudness {min_loudness:.2f} dB "
        f"after {max_attempts} attempts. {seed_hint}Raise the per-sample attempt budget "
        f"(``attempts_per_sample`` / ``SampleSeed.max_attempts``) or lower min_loudness."
    )


class _GenerateCliArgs(RenderConfig, BaseSettings):
    """Pydantic-settings CLI binding for ``generate_vst_dataset.py``.

    Inherits every ``RenderConfig`` field so the CLI flag set tracks the model
    automatically — adding or removing a field on ``RenderConfig`` extends or
    shrinks the CLI surface without a parallel update here. Adds ``data_file``
    as the sole positional arg (the destination shard path; its ``.lance``
    suffix selects the Lance writer via ``OutputFormat.from_extension``).
    """

    model_config = SettingsConfigDict(
        cli_parse_args=True,
        cli_prog_name="generate_vst_dataset",
        cli_kebab_case=False,
        strict=True,
        extra="forbid",
    )

    data_file: CliPositionalArg[str]


def main() -> None:
    """Entry point — parse CLI args into a ``RenderConfig`` and render one shard.

    ``data_file`` must carry the ``.lance`` suffix (validated via
    ``OutputFormat.from_extension``); any other suffix raises ``SystemExit``
    rather than silently producing a half-written file in the wrong format.
    """
    # Import lazily so importing this module for VSTDataSample/generate_sample
    # doesn't pay the heavy ``lance`` import.
    from synth_setter.data.vst.writers import make_lance_dataset

    args = CliApp.run(_GenerateCliArgs)
    render_cfg = RenderConfig(**args.model_dump(exclude={"data_file"}))
    ensure_dawdreamer_runtime(render_cfg.renderer_backend)

    suffix = Path(args.data_file).suffix
    if OutputFormat.from_extension(suffix) is None:
        raise SystemExit(
            f"data_file must end in one of {sorted(f.extension for f in OutputFormat)}, "
            f"got {suffix!r}"
        )

    make_lance_dataset(args.data_file, render_cfg)


if __name__ == "__main__":
    main()
