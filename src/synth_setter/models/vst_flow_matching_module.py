"""Lightning module for flow-matching VST parameter prediction."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping, MutableMapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import torch
from beartype import beartype
from jaxtyping import Bool, Float, Shaped, jaxtyped
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm

from synth_setter.conditioning import (
    Conditioning,
    SketchControls,
    conditioning_batch_key,
    resolve_sketch_controls,
)
from synth_setter.metrics import (
    BestSwapParamMSE,
    NumberGroupSwapParamMSE,
    best_swap_per_param_mse,
    midi_pitch_residuals,
    number_group_swap_per_param_mse,
    supports_midi_pitch_residuals,
)
from synth_setter.models.components.pretrained_encoder import PretrainedConditioningEncoder
from synth_setter.models.components.sketch_tokens import CONTROL_GROUPS, SketchControlTokens

_BATCH_SHAPE = "batch"
_BATCH_ANY_SHAPE = "batch ..."
_BATCH_TIME_SHAPE = "batch 1"
_FROZEN_BACKBONE_PREFIX = "encoder.backbone."
_PARAM_SHAPE = "params"
_SampleBatch = Mapping[str, Shaped[torch.Tensor, _BATCH_ANY_SHAPE] | None]
# Stored outside hyper_parameters because Lightning replaces those with load-time kwargs
# before the hook runs; missing keys retain their legacy velocity/MSE meanings.
_ENDPOINT_LOSS_KEY = "endpoint_loss"
_LEGACY_ENDPOINT_LOSS = "mse"
_PARAMETERIZATION_KEY = "parameterization"
_LEGACY_PARAMETERIZATION = "velocity"

EndpointLoss = Literal["mse", "mixed"]
Parameterization = Literal["velocity", "endpoint"]
_ENDPOINT_LOSSES: frozenset[str] = frozenset(("mixed", "mse"))
_PARAMETERIZATIONS: frozenset[str] = frozenset(("endpoint", "velocity"))
_EVAL_BATCH_SEED_STRIDE = 2**16
_EVAL_SEED_MODULUS = 2**63 - 1
_EVAL_TEST_SEED_OFFSET = 1_000_003

if TYPE_CHECKING:
    from synth_setter.models.components.audio_feedback import (
        AudioFeedbackLoss,
        GradientBalance,
    )


@dataclass(frozen=True)
class ConditioningKeepMasks:
    """Positive keep state for every identity-bearing conditioning stream.

    .. attribute :: content

       Content-conditioning keep state shaped ``(batch,)``.

    .. attribute :: sketch_groups

       Per-sketch-group keep state shaped ``(batch, 3)``.
    """

    content: Bool[torch.Tensor, _BATCH_SHAPE]
    sketch_groups: Bool[torch.Tensor, "batch groups"]

    @classmethod
    @jaxtyped(typechecker=beartype)
    def content_only(cls, content: Bool[torch.Tensor, _BATCH_SHAPE]) -> ConditioningKeepMasks:
        """Build keep state for a run with no sketch controls configured.

        :param content: Content-conditioning keep state shaped ``(batch,)``.
        :returns: Keep state whose sketch groups are all absent.
        """
        return cls(
            content=content,
            sketch_groups=torch.zeros(
                content.shape[0],
                len(CONTROL_GROUPS),
                dtype=torch.bool,
                device=content.device,
            ),
        )

    @property
    @jaxtyped(typechecker=beartype)
    def identity_keep(self) -> Bool[torch.Tensor, _BATCH_SHAPE]:
        """Return rows retaining at least one identity-bearing stream.

        :returns: Positive identity keep state shaped ``(batch,)``.
        """
        return self.content | self.sketch_groups.any(dim=-1)


@dataclass(frozen=True)
class ControlTokenBranches:
    """Full-sketch and PE-only control-token states for CFG sampling.

    .. attribute :: conditional

       Full sketch-control tokens used by sketch-only and content-plus-sketch branches.

    .. attribute :: unconditional

       PE-only tokens used by the unconditional branch.
    """

    conditional: Float[torch.Tensor, "batch tokens d_model"]
    unconditional: Float[torch.Tensor, "batch tokens d_model"]


@dataclass(frozen=True)
class TrainStepOutputs:
    """Loss terms produced by one training step.

    .. attribute :: loss

       Flow-matching loss; the only term every configuration produces.

    .. attribute :: per_param_flow_mse

       Weighted model-space MSE diagnostic for each encoded parameter column.

    .. attribute :: audio_term

       Weighted audio-feedback loss, or ``None`` without an attached audio loss.

    .. attribute :: penalty

       Vector-field regularization penalty, or ``None`` for fields that define none.

    .. attribute :: grad_balance

       Gradient diagnostics for the audio term, or ``None`` off the probe cadence.

    .. attribute :: t

       Flow time per row, shaped ``(batch, 1)``.

    .. attribute :: conditioning_keep

       Keep state sampled for this step's conditioning streams.
    """

    loss: torch.Tensor
    per_param_flow_mse: Float[torch.Tensor, _PARAM_SHAPE]
    audio_term: torch.Tensor | None
    penalty: torch.Tensor | None
    grad_balance: GradientBalance | None
    t: torch.Tensor
    conditioning_keep: ConditioningKeepMasks


type _FieldTransform = Callable[
    [Shaped[torch.Tensor, "batch ..."]], Shaped[torch.Tensor, "batch ..."]
]
type _TimeField = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@runtime_checkable
class _ParamSpecLike(Protocol):
    """Parameter layout contract needed by mixed endpoint objectives."""

    @jaxtyped(typechecker=beartype)
    def encoded_slices(self) -> Iterator[tuple[object, slice]]:
        """Yield each logical parameter with its encoded span.

        :returns: Logical parameters paired with their encoded slices.
        """
        ...


@jaxtyped(typechecker=beartype)
def _uses_onehot_classification(parameter: object) -> bool:
    """Return whether a parameter span represents one categorical draw.

    :param parameter: Typed logical parameter.
    :returns: True only for one-hot categorical or integer-literal spans.
    """
    return getattr(parameter, "encoding", None) == "onehot"


@jaxtyped(typechecker=beartype)
def endpoint_prediction_to_model(
    prediction: Float[torch.Tensor, "batch params"],
    param_spec: _ParamSpecLike,
) -> Float[torch.Tensor, "batch params"]:
    """Convert one-hot logits to model-space probabilities without changing numerical spans.

    :param prediction: Raw endpoint output; one-hot spans are logits.
    :param param_spec: Parameter layout defining typed encoded spans.
    :returns: Endpoint in model space, with categorical probabilities mapped to ``[-1, 1]``.
    """
    endpoint = prediction.clone()
    for parameter, span in param_spec.encoded_slices():
        if _uses_onehot_classification(parameter):
            endpoint[:, span] = 2 * torch.softmax(prediction[:, span], dim=-1) - 1
    return endpoint


@jaxtyped(typechecker=beartype)
def mixed_endpoint_row_loss(
    prediction: Float[torch.Tensor, "batch params"],
    target: Float[torch.Tensor, "batch params"],
    param_spec: _ParamSpecLike,
) -> Float[torch.Tensor, "batch 1"]:
    """Average one classification or regression term per logical parameter and row.

    :param prediction: Raw endpoint output; one-hot spans are logits.
    :param target: Clean model-space endpoint.
    :param param_spec: Parameter layout defining typed encoded spans.
    :returns: Unweighted per-row mixed objective shaped ``(batch, 1)``.
    """
    terms: list[Float[torch.Tensor, _BATCH_SHAPE]] = []
    for parameter, span in param_spec.encoded_slices():
        predicted_span = prediction[:, span]
        target_span = target[:, span]
        if _uses_onehot_classification(parameter):
            term = torch.nn.functional.cross_entropy(
                predicted_span, target_span.argmax(dim=-1), reduction="none"
            )
        else:
            term = (predicted_span - target_span).square().mean(dim=-1)
        terms.append(term)
    return torch.stack(terms, dim=-1).mean(dim=-1, keepdim=True)


@jaxtyped(typechecker=beartype)
def joint_cfg_velocity(
    conditional_field: _TimeField,
    unconditional_field: _TimeField,
    cfg_strength: float,
) -> _TimeField:
    """Build one joint two-branch classifier-free-guidance velocity.

    :param conditional_field: Content-plus-sketch conditional time field.
    :param unconditional_field: Unconditional time field.
    :param cfg_strength: Joint classifier-free-guidance scale.
    :returns: Two-argument guided velocity field.
    """
    return lambda x, t: (
        (1 - cfg_strength) * unconditional_field(x, t) + cfg_strength * conditional_field(x, t)
    )


@jaxtyped(typechecker=beartype)
def multi_cfg_velocity(
    unconditional_field: _TimeField,
    sketch_field: _TimeField,
    content_sketch_field: _TimeField,
    *,
    sketch_cfg_strength: float,
    content_cfg_strength: float,
) -> _TimeField:
    """Build a three-branch field with independent sketch and content guidance.

    :param unconditional_field: Field without content or sketch controls.
    :param sketch_field: Field conditioned only on sketch controls.
    :param content_sketch_field: Field conditioned on content and sketch controls.
    :param sketch_cfg_strength: Guidance scale for adding sketch controls.
    :param content_cfg_strength: Guidance scale for adding content conditioning.
    :returns: Two-argument guided velocity field.
    """

    @jaxtyped(typechecker=beartype)
    def guided(
        x: Shaped[torch.Tensor, "batch ..."],
        t: Shaped[torch.Tensor, "batch 1"],
    ) -> Shaped[torch.Tensor, "batch ..."]:
        """Evaluate each branch once and combine its guidance delta.

        :param x: Shared trajectory point evaluated by every branch.
        :param t: Shared flow time evaluated by every branch.
        :returns: Unconditional velocity plus separately scaled sketch and content deltas.
        """
        unconditional = unconditional_field(x, t)
        sketch = sketch_field(x, t)
        content_sketch = content_sketch_field(x, t)
        return (
            unconditional
            + sketch_cfg_strength * (sketch - unconditional)
            + content_cfg_strength * (content_sketch - sketch)
        )

    return guided


@jaxtyped(typechecker=beartype)
def build_guided_velocity(
    field: torch.nn.Module,
    conditioning: Shaped[torch.Tensor, "batch ..."] | None,
    cfg_strength: float,
    *,
    sketch_cfg_strength: float | None = None,
    control_tokens: ControlTokenBranches | None = None,
    output_transform: _FieldTransform | None = None,
) -> _TimeField:
    """Bind content and optional control tokens into classifier-free-guidance branches.

    :param field: Model velocity field.
    :param conditioning: Encoded content conditioning for the conditional branch.
    :param cfg_strength: Classifier-free-guidance scale for content conditioning.
    :param sketch_cfg_strength: Guidance scale for sketch controls; defaults to
        ``cfg_strength`` for joint-CFG compatibility.
    :param control_tokens: Complete full-sketch and PE-only control-token state.
    :param output_transform: Optional conversion applied after combining guidance branches.
    :returns: Two-argument guided velocity field.
    """
    if control_tokens is None:
        guided = joint_cfg_velocity(
            _bind_branch(field, conditioning, None),
            _bind_branch(field, None, None),
            cfg_strength,
        )
    else:
        sketch_strength = cfg_strength if sketch_cfg_strength is None else sketch_cfg_strength
        unconditional = _bind_branch(field, None, control_tokens.unconditional)
        sketch = _bind_branch(field, None, control_tokens.conditional)
        if conditioning is None:
            guided = joint_cfg_velocity(sketch, unconditional, sketch_strength)
        else:
            guided = multi_cfg_velocity(
                unconditional,
                sketch,
                _bind_branch(field, conditioning, control_tokens.conditional),
                sketch_cfg_strength=sketch_strength,
                content_cfg_strength=cfg_strength,
            )
    if output_transform is None:
        return guided
    return lambda x, t: output_transform(guided(x, t))


@jaxtyped(typechecker=beartype)
def _bind_branch(
    field: torch.nn.Module,
    conditioning: Shaped[torch.Tensor, "batch ..."] | None,
    control_tokens: Float[torch.Tensor, "batch tokens d_model"] | None,
) -> _TimeField:
    """Bind one CFG branch's conditioning into a two-argument time field.

    Content conditioning binds positionally: ``ConditionalResidualMLP`` names the
    argument ``c`` while the other backbones name it ``conditioning``.

    :param field: Model velocity field.
    :param conditioning: Encoded content conditioning, or ``None`` for the
        unconditional branch.
    :param control_tokens: This branch's control tokens, or ``None`` without sketch support.
    :returns: Two-argument velocity field over parameter state and time.
    """

    @jaxtyped(typechecker=beartype)
    def evaluate(
        x: Shaped[torch.Tensor, "batch ..."],
        t: Shaped[torch.Tensor, "batch 1"],
    ) -> Shaped[torch.Tensor, "batch ..."]:
        """Evaluate one bound guidance branch.

        :param x: Shared trajectory point.
        :param t: Shared flow time.
        :returns: Raw field output.
        """
        return (
            field(x, t, conditioning)
            if control_tokens is None
            else field(x, t, conditioning, control_tokens=control_tokens)
        )

    return evaluate


@jaxtyped(typechecker=beartype)
def rk4_step(
    f: _TimeField,
    x: Float[torch.Tensor, "batch params"],
    t: Float[torch.Tensor, "batch 1"],
    dt: float | Float[torch.Tensor, "batch 1"],
) -> Float[torch.Tensor, "batch params"]:
    """Advance a two-argument time field by one classical RK4 step.

    :param f: Time field accepting only parameter state and time.
    :param x: Current parameter state.
    :param t: Current flow time.
    :param dt: Integration step in warped time.
    :returns: Parameter state after one RK4 step.
    """
    k1 = f(x, t)
    k2 = f(x + dt * k1 / 2, t + dt / 2)
    k3 = f(x + dt * k2 / 2, t + dt / 2)
    k4 = f(x + dt * k3, t + dt)

    return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


@jaxtyped(typechecker=beartype)
def integrate_flow(
    velocity: _TimeField,
    noise: Float[torch.Tensor, "batch params"],
    steps: int,
    *,
    warp_time: Callable[[Float[torch.Tensor, "batch 1"]], Float[torch.Tensor, "batch 1"]],
    parameterization: Parameterization = "velocity",
) -> Float[torch.Tensor, "batch params"]:
    """Integrate a velocity field from noise at t=0 to a sample at t=1.

    :param velocity: Two-argument time field over parameter state and time.
    :param noise: Initial state shaped ``(batch, params)``.
    :param steps: Number of equal steps in unwarped time.
    :param warp_time: Monotone map applied to the time grid before each step.
    :param parameterization: Field representation used to derive the velocity.
    :returns: Terminal parameter state.
    """
    t = torch.zeros(noise.shape[0], 1, device=noise.device)
    dt = 1.0 / steps
    sample = noise

    for step in range(steps):
        warped_t = warp_time(t)
        warped_dt = warp_time(t + dt) - warped_t
        if parameterization == "endpoint" and step == steps - 1:
            # RK4's final stage reaches t=1, where endpoint-to-velocity conversion is 0 / 0.
            sample = sample + warped_dt * velocity(sample, warped_t)
        else:
            sample = rk4_step(velocity, sample, warped_t, warped_dt)
        t = t + dt

    return sample


class VSTFlowMatchingModule(LightningModule):
    """Flow-matching LightningModule for VST parameter prediction (CFG + RK4 sampling)."""

    def __init__(
        self,
        encoder: torch.nn.Module,
        vector_field: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        # Keyword-only: a stale positional caller would silently train at a bogus width.
        *,
        num_params: int,
        param_spec: str | None = None,
        conditioning: Conditioning = "mel",
        sketch_controls: SketchControls = None,
        sketch_dropout_rate: float = 0.1,
        all_conditioning_dropout_rate: float = 0.1,
        audio_loss: AudioFeedbackLoss | None = None,
        encoder_num_heads: int | None = None,
        encoder_output_dim: int | None = None,
        warmup_steps: int = 5000,
        cfg_dropout_rate: float = 0.1,
        rectified_sigma_min: float = 0.0,
        parameterization: Parameterization = "velocity",
        endpoint_loss: EndpointLoss = "mse",
        seeded_evaluation: bool = False,
        validation_sample_steps: int = 50,
        validation_cfg_strength: float = 4.0,
        validation_sketch_cfg_strength: float | None = None,
        test_sample_steps: int = 100,
        test_cfg_strength: float = 4.0,
        test_sketch_cfg_strength: float | None = None,
        compile: bool = False,
    ) -> None:
        """Wire the encoder/vector-field and persist the flow-matching hyperparameters.

        :param encoder: Encoder over legacy mel or a fixed-shape embedding.
        :param vector_field: Network predicting the flow velocity field.
        :param optimizer: ``functools.partial``-style optimizer factory (Hydra
            ``_partial_: true``); invoked in :meth:`configure_optimizers`.
        :param scheduler: ``functools.partial``-style scheduler factory or ``None``.
        :param num_params: Parameter-vector width the field operates on.
        :param param_spec: Registered parameter spec enabling structured swap metrics.
        :param conditioning: Legacy mel/m2l mode or a fixed-shape embedding spec.
        :param sketch_controls: Optional sketch-control spec enabling concat
            control-token injection into the vector field (#2612).
        :param sketch_dropout_rate: Independent per-sketch-group drop probability.
        :param all_conditioning_dropout_rate: Probability of dropping content and
            every sketch group in one global event.
        :param audio_loss: Optional audio-feedback term on the rendered one-step
            estimate; requires an uncompiled, single-device, drop-last run (#2585).
        :param encoder_num_heads: Model-owned attention head count for sequence encoders.
        :param encoder_output_dim: Configured encoder width consumed by the vector field.
        :param warmup_steps: If positive, wrap the scheduler with a linear warmup.
        :param cfg_dropout_rate: Independent content-conditioning drop probability
            during training (CFG).
        :param rectified_sigma_min: Minimum noise scale for the rectified probability path.
        :param parameterization: What the field predicts: the velocity ``x1 - x0`` or the
            clean endpoint ``x1``; the sampler converts an endpoint to a velocity.
        :param endpoint_loss: Flat endpoint MSE, or per-parameter MSE/CE for one-hot spans.
        :param seeded_evaluation: Whether validation and test use seed-derived local noise.
        :param validation_sample_steps: RK4 integration steps used at validation.
        :param validation_cfg_strength: Content guidance strength at validation.
        :param validation_sketch_cfg_strength: Sketch guidance strength at validation;
            defaults to ``validation_cfg_strength``.
        :param test_sample_steps: RK4 integration steps used at test.
        :param test_cfg_strength: Content guidance strength at test and prediction.
        :param test_sketch_cfg_strength: Sketch guidance strength at test and prediction;
            defaults to ``test_cfg_strength``.
        :param compile: Whether to compile the encoder and vector field during fit setup.
        :raises ValueError: The ParamSpec width differs from ``num_params``, an objective
            option is invalid, mixed loss lacks endpoint parameterization or a ParamSpec,
            or ``audio_loss`` is combined with a nonzero ``rectified_sigma_min`` or
            ``compile=True`` (#2585).
        """
        super().__init__()
        if parameterization not in _PARAMETERIZATIONS:
            # Hydra passes strings through unchecked; a typo would silently train velocity.
            raise ValueError(
                f"parameterization must be one of {sorted(_PARAMETERIZATIONS)}, "
                f"got {parameterization!r}"
            )
        if endpoint_loss not in _ENDPOINT_LOSSES:
            raise ValueError(
                f"endpoint_loss must be one of {sorted(_ENDPOINT_LOSSES)}, got {endpoint_loss!r}"
            )
        if endpoint_loss == "mixed" and parameterization != "endpoint":
            raise ValueError("endpoint_loss='mixed' requires parameterization='endpoint'")
        if endpoint_loss == "mixed" and param_spec is None:
            raise ValueError("endpoint_loss='mixed' requires param_spec")

        # Saving hyperparameters deep-copies them, which a weight-normalized frozen encoder
        # inside the audio term cannot survive; the term is training-time only, so it is not
        # reconstructed from hparams either.
        self.save_hyperparameters(ignore=["encoder", "audio_loss"], logger=False)
        if not isinstance(encoder, PretrainedConditioningEncoder):
            # Existing load_from_checkpoint consumers reconstruct legacy encoders from hparams.
            self.hparams["encoder"] = encoder

        self.encoder = encoder
        self.vector_field = vector_field
        self._sketch_controls = resolve_sketch_controls(sketch_controls)
        self.sketch_tokens = (
            SketchControlTokens(
                d_model=vector_field.d_model,
                num_control_tokens=self._sketch_controls.num_control_tokens,
                profile=self._sketch_controls.profile,
            )
            if self._sketch_controls is not None
            else None
        )
        self.audio_loss = audio_loss
        if audio_loss is not None and rectified_sigma_min != 0.0:
            # theta_hat = x_t + (1 - t) * prediction is the exact one-step estimate only
            # on the sigma-free path; any other sigma silently biases the rendered params.
            raise ValueError(
                f"audio feedback requires rectified_sigma_min=0, got {rectified_sigma_min}"
            )
        if audio_loss is not None and compile:
            from synth_setter.models.components.audio_feedback import (
                validate_audio_feedback_runtime,
            )

            # Only `compiled` is known here; world_size is re-checked against the real
            # trainer in on_train_start. Must fail before setup() compiles (#2585).
            validate_audio_feedback_runtime(compiled=True, world_size=1)
        self._conditioning_key = conditioning_batch_key(conditioning)
        self._evaluation_seed = torch.initial_seed()

        self.val_param_mse_best_swap = BestSwapParamMSE()
        self.test_param_mse_best_swap = BestSwapParamMSE()
        metric_spec = None
        if param_spec is not None:
            from synth_setter.data.vst import param_specs

            metric_spec = param_specs[param_spec]
            if metric_spec.encoded_width != num_params:
                raise ValueError(
                    f"ParamSpec {param_spec!r} encoded width {metric_spec.encoded_width} "
                    f"does not match num_params {num_params}"
                )
        self._metric_param_spec = metric_spec
        self._pitch_metric_spec = (
            metric_spec
            if metric_spec is not None and supports_midi_pitch_residuals(metric_spec)
            else None
        )
        self.val_param_mse_number_group_swap = (
            NumberGroupSwapParamMSE(metric_spec) if metric_spec is not None else None
        )
        self.test_param_mse_number_group_swap = (
            NumberGroupSwapParamMSE(metric_spec) if metric_spec is not None else None
        )

    def on_train_start(self) -> None:
        if self.audio_loss is None:
            return

        from synth_setter.models.components.audio_feedback import (
            validate_audio_feedback_runtime,
        )

        validate_audio_feedback_runtime(
            compiled=self.hparams.compile,
            world_size=self.trainer.world_size,
        )

    @jaxtyped(typechecker=beartype)
    def on_save_checkpoint(self, checkpoint: dict[str, object]) -> None:
        """Stamp output semantics and exclude re-resolvable frozen CLAP state.

        :param checkpoint: Mutable Lightning checkpoint payload.
        :raises TypeError: A pretrained-encoder checkpoint has malformed state metadata.
        """
        checkpoint[_ENDPOINT_LOSS_KEY] = self.hparams.endpoint_loss
        checkpoint[_PARAMETERIZATION_KEY] = self.hparams.parameterization
        if not isinstance(self.encoder, PretrainedConditioningEncoder):
            return
        state = checkpoint.get("state_dict")
        if not isinstance(state, MutableMapping):
            raise TypeError("Lightning checkpoint state_dict must be a mutable mapping")
        for key in tuple(state):
            if isinstance(key, str) and key.startswith(_FROZEN_BACKBONE_PREFIX):
                del state[key]

        hyperparameters = checkpoint.get("hyper_parameters")
        if isinstance(hyperparameters, MutableMapping):
            hyperparameters.pop("encoder", None)

    @jaxtyped(typechecker=beartype)
    def on_load_checkpoint(self, checkpoint: dict[str, object]) -> None:
        """Restore current frozen CLAP state so Lightning can load trainable state strictly.

        :param checkpoint: Mutable Lightning checkpoint payload.
        :raises TypeError: A pretrained-encoder checkpoint has a malformed state dictionary.
        :raises ValueError: The checkpoint trained another parameterization or endpoint loss; same-
            shaped weights would load but carry incompatible output semantics.
        """
        stored_parameterization = checkpoint.get(_PARAMETERIZATION_KEY, _LEGACY_PARAMETERIZATION)
        if stored_parameterization != self.hparams.parameterization:
            raise ValueError(
                f"checkpoint trained parameterization={stored_parameterization!r}, "
                f"module expects {self.hparams.parameterization!r}"
            )
        stored_endpoint_loss = checkpoint.get(_ENDPOINT_LOSS_KEY, _LEGACY_ENDPOINT_LOSS)
        if stored_endpoint_loss != self.hparams.endpoint_loss:
            raise ValueError(
                f"checkpoint trained endpoint_loss={stored_endpoint_loss!r}, "
                f"module expects {self.hparams.endpoint_loss!r}"
            )
        if not isinstance(self.encoder, PretrainedConditioningEncoder):
            return
        state = checkpoint.get("state_dict")
        if not isinstance(state, MutableMapping):
            raise TypeError("Lightning checkpoint state_dict must be a mutable mapping")
        for key, value in self.state_dict().items():
            if key.startswith(_FROZEN_BACKBONE_PREFIX):
                state[key] = value

    def _sample_time(self, n: int, device: torch.device) -> torch.Tensor:
        return torch.rand(n, 1, device=device)

    def _weight_time(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)

    def _basic_sample(self, params: torch.Tensor, oversample: float = 1.0):
        if oversample == 1.0:
            x0 = torch.randn_like(params)
        elif oversample < 1.0:
            raise ValueError(f"oversample must be >= 1.0, got {oversample}")
        else:
            n = int(oversample * params.shape[0])
            x0 = torch.randn(n, *params.shape[1:], device=params.device)
        x1 = params

        return x0, x1

    def _rectified_probability_path(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor):
        x_t = x0 * (1 - t) * (1 - self.hparams.rectified_sigma_min) + x1 * t

        return x_t

    def _sample_probability_path(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor):
        x_t = self._rectified_probability_path(x0, x1, t)
        return x_t

    def _rectified_vector_field(self, x0: torch.Tensor, x1: torch.Tensor):
        return x1 - x0

    def _evaluate_target_field(
        self, x0: torch.Tensor, x1: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ):
        if self.hparams.parameterization == "endpoint":
            return x1
        return self._rectified_vector_field(x0, x1)

    def _get_conditioning_from_batch(self, batch: _SampleBatch) -> torch.Tensor:
        conditioning = batch[self._conditioning_key]
        if conditioning is None:
            raise ValueError(
                f"batch conditioning field {self._conditioning_key!r} must contain a tensor"
            )
        if (
            self._conditioning_key == "audio"
            and conditioning.ndim == 3
            and conditioning.shape[1] == 1
        ):
            return conditioning.squeeze(1)
        return conditioning

    @jaxtyped(typechecker=beartype)
    def _sample_conditioning_keep_masks(
        self, batch_size: int, device: torch.device
    ) -> ConditioningKeepMasks:
        """Draw independent stream keeps and apply the global all-drop event.

        :param batch_size: Rows in the current batch.
        :param device: Device the masks are drawn on.
        :returns: Positive content and per-sketch-group keep masks.
        """
        content = torch.rand(batch_size, device=device) > self.hparams.cfg_dropout_rate
        num_sketch_groups = (
            len(CONTROL_GROUPS)
            if self.sketch_tokens is None
            else len(self.sketch_tokens.layout.group_names)
        )
        sketch_groups = (
            torch.rand(batch_size, num_sketch_groups, device=device)
            > self.hparams.sketch_dropout_rate
        )
        global_keep = (
            torch.rand(batch_size, device=device) > self.hparams.all_conditioning_dropout_rate
        )
        return ConditioningKeepMasks(
            content=content & global_keep,
            sketch_groups=sketch_groups & global_keep.unsqueeze(-1),
        )

    @jaxtyped(typechecker=beartype)
    def _prepare_conditioning(
        self, batch: dict[str, Shaped[torch.Tensor, ...] | None]
    ) -> tuple[
        Shaped[torch.Tensor, "batch ..."],
        Float[torch.Tensor, "batch tokens d_model"] | None,
        ConditioningKeepMasks,
    ]:
        """Encode the content stream and apply this step's sampled dropout policy.

        :param batch: Model batch carrying content conditioning, plus ``sketch_ctrl``
            whenever a sketch spec is configured.
        :returns: Post-dropout conditioning, sketch control tokens (``None`` without a
            configured spec), and the keep masks that produced both.
        """
        conditioning = self.encoder(self._get_conditioning_from_batch(batch))
        if (
            conditioning.ndim == 3
            and conditioning.shape[1] > 1
            and self._is_trainer_logging_step()
        ):
            self._log_slot_cosine(conditioning.detach())
        if self.sketch_tokens is None:
            # Legacy path: apply_dropout draws its own mask, keeping the no-sketch
            # RNG stream identical to runs from before sketch support.
            z, content_keep = self.vector_field.apply_dropout(
                conditioning, self.hparams.cfg_dropout_rate
            )
            return z, None, ConditioningKeepMasks.content_only(content_keep)

        keep = self._sample_conditioning_keep_masks(conditioning.shape[0], conditioning.device)
        z, _ = self.vector_field.apply_dropout(conditioning, keep_mask=keep.content)
        return z, self.sketch_tokens(batch["sketch_ctrl"], keep.sketch_groups), keep

    @jaxtyped(typechecker=beartype)
    def _control_token_branches_from_batch(
        self, batch: _SampleBatch
    ) -> ControlTokenBranches | None:
        """Build complete full-sketch and PE-only control branches for inference.

        :param batch: Model batch carrying sketch controls when configured.
        :returns: Both control-token branches, or ``None`` without sketch support.
        :raises ValueError: The active sketch-control field is ``None``.
        """
        if self.sketch_tokens is None:
            return None
        controls = batch["sketch_ctrl"]
        if controls is None:
            raise ValueError("batch sketch_ctrl field must contain a tensor")
        keep = torch.ones(
            controls.shape[0],
            len(self.sketch_tokens.layout.group_names),
            dtype=torch.bool,
            device=controls.device,
        )
        return ControlTokenBranches(
            conditional=self.sketch_tokens(controls, keep),
            unconditional=self.sketch_tokens.unconditional(controls.shape[0]),
        )

    @jaxtyped(typechecker=beartype)
    def _is_trainer_logging_step(self) -> bool:
        """Whether this step pays for the probe's extra backward through the renderer.

        :returns: True on Lightning's own logging cadence; False when detached from a trainer.
        """
        # self.trainer raises when detached, so the private attribute is the only probe
        # that works for direct _train_step calls outside a fit loop.
        if self._trainer is None:
            return False
        every = self.trainer.log_every_n_steps
        return every > 0 and self.trainer.global_step % every == 0

    @jaxtyped(typechecker=beartype)
    def _log_slot_cosine(self, conditioning: Float[torch.Tensor, "batch slots dim"]) -> None:
        """Log how far apart the per-layer conditioning slots sit before dropout.

        A value approaching one means nominally separate slots have converged to the same read, so
        the extra slots carry nothing the field's layers can distinguish.

        :param conditioning: Detached layerwise conditioning.
        """
        slots = torch.nn.functional.normalize(conditioning, dim=-1)
        gram = slots @ slots.transpose(-2, -1)
        count = gram.shape[-1]
        off_diagonal = gram.sum(dim=(-2, -1)) - gram.diagonal(dim1=-2, dim2=-1).sum(-1)
        mean = (off_diagonal / (count * (count - 1))).mean()
        self.log("train/slot_cosine", mean, on_step=True, on_epoch=False)

    @jaxtyped(typechecker=beartype)
    def _log_gradient_time_profile(
        self,
        audio_row_norms: Float[torch.Tensor, _BATCH_SHAPE],
        t: Float[torch.Tensor, _BATCH_TIME_SHAPE],
    ) -> None:
        """Log where along the flow time axis the audio term's gradient actually lands.

        :param audio_row_norms: Per-row audio gradient norm.
        :param t: Flow time shaped ``(batch, 1)``.
        """
        from synth_setter.models.components.audio_feedback import time_bucket_means

        for index, mean in enumerate(time_bucket_means(audio_row_norms, t)):
            if torch.isfinite(mean):
                self.log(
                    f"train/audio_grad_norm_t_bucket_{index}", mean, on_step=True, on_epoch=False
                )

    def _train_step(self, batch: dict[str, torch.Tensor]) -> TrainStepOutputs:
        """Run one training forward pass and assemble every term the logger consumes.

        :param batch: Online or stored batch carrying params, noise, and audio.
        :returns: Flow loss plus whichever optional terms this configuration produces.
        """
        params = batch["params"]
        noise = batch["noise"]

        z, control_tokens, conditioning_keep = self._prepare_conditioning(batch)

        with torch.no_grad():
            t = self._sample_time(params.shape[0], params.device)
            w = self._weight_time(t)

            x0 = noise
            x1 = params

            x_t = self._sample_probability_path(x0, x1, t)
            target = self._evaluate_target_field(x0, x1, x_t, t)

        if control_tokens is None:
            prediction = self.vector_field(x_t, t, z)
        else:
            prediction = self.vector_field(x_t, t, z, control_tokens=control_tokens)

        endpoint_prediction = self._endpoint_prediction_to_model(prediction)
        squared_flow_error = (endpoint_prediction - target).square()
        per_param_flow_mse = (squared_flow_error * w).mean(dim=0)
        if self.hparams.endpoint_loss == "mixed":
            assert self._metric_param_spec is not None
            row_loss = mixed_endpoint_row_loss(prediction, target, self._metric_param_spec)
        else:
            row_loss = squared_flow_error.mean(dim=-1, keepdim=True)
        loss = (row_loss * w).mean()

        audio_term = None
        grad_balance = None
        if self.audio_loss is not None:
            # One-step estimate of x1 from the current field; rendering it keeps
            # autograd connected so latent audio error reaches the field's weights.
            theta_hat = self._one_step_estimate(x_t, t, prediction)
            # Fully unconditional rows estimate the marginal, so their row-specific
            # target-audio residual is high-variance noise rather than identity signal.
            audio_term = self.audio_loss(
                theta_hat,
                t,
                batch["audio"],
                keep=conditioning_keep.identity_keep,
            )
            if self._is_trainer_logging_step():
                from synth_setter.models.components.audio_feedback import gradient_balance

                grad_balance = gradient_balance(
                    flow_loss=loss, audio_term=audio_term, shared=prediction
                )

        penalty = None
        if hasattr(self.vector_field, "penalty"):
            penalty = self.vector_field.penalty()

        return TrainStepOutputs(
            loss=loss,
            per_param_flow_mse=per_param_flow_mse,
            audio_term=audio_term,
            penalty=penalty,
            grad_balance=grad_balance,
            t=t,
            conditioning_keep=conditioning_keep,
        )

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self._train_step(batch)
        self.log("train/loss", outputs.loss, on_step=True, on_epoch=True, prog_bar=True)
        if self._metric_param_spec is not None:
            # Velocity and endpoint errors are not comparable; the endpoint run logs under
            # its own prefix so shared dashboards never overlay the two.
            prefix = (
                "train/per_param_endpoint_mse"
                if self.hparams.parameterization == "endpoint"
                else "train/per_param_flow_mse"
            )
            metrics = {
                f"{prefix}/{param.name}": outputs.per_param_flow_mse[span].mean()
                for param, span in self._metric_param_spec.encoded_slices()
            }
            self.log_dict(
                metrics,
                on_step=False,
                on_epoch=True,
                batch_size=batch["params"].shape[0],
                sync_dist=True,
            )

        total = outputs.loss
        if outputs.audio_term is not None:
            # Dominated by high-t rows, where the weight is maximal; the gradient peaks at
            # mid t, so this scalar does not track where the term is actually teaching.
            self.log(
                "train/audio_loss", outputs.audio_term, on_step=True, on_epoch=True, prog_bar=True
            )
            total = total + outputs.audio_term

        if outputs.grad_balance is not None:
            # Set lambda_audio from the ratio, not the loss value; watch the cosine turn
            # negative for the point the audio term starts fighting the flow objective.
            ratio, cosine = outputs.grad_balance.ratio, outputs.grad_balance.cosine
            self.log("train/audio_grad_ratio", ratio, on_step=True, on_epoch=False)
            self.log("train/audio_grad_cosine", cosine, on_step=True, on_epoch=False)
            self._log_gradient_time_profile(outputs.grad_balance.audio_row_norms, outputs.t)

        if outputs.penalty is not None:
            self.log("train/penalty", outputs.penalty, on_step=True, on_epoch=True, prog_bar=True)
            total = total + outputs.penalty

        return total

    def on_train_epoch_end(self) -> None:
        pass

    def _warp_time(self, t: torch.Tensor) -> torch.Tensor:
        return t

    @jaxtyped(typechecker=beartype)
    def _velocity_field(
        self,
        conditioning: Shaped[torch.Tensor, "batch ..."] | None,
        cfg_strength: float,
        control_tokens: ControlTokenBranches | None,
        *,
        sketch_cfg_strength: float | None = None,
    ) -> _TimeField:
        """Build the time field the sampler integrates.

        A seam, not a wrapper: a subclass whose sampling velocity differs from its training
        velocity overrides this and inherits the integration loop unchanged.

        :param conditioning: Encoded content conditioning for the conditional branch.
        :param cfg_strength: Classifier-free-guidance scale for content conditioning.
        :param control_tokens: Complete control-token state, or ``None`` without sketch support.
        :param sketch_cfg_strength: Guidance scale for sketch controls.
        :returns: Two-argument velocity field over parameter state and time.
        """
        output_transform = (
            self._endpoint_prediction_to_model if self.hparams.endpoint_loss == "mixed" else None
        )
        guided = build_guided_velocity(
            self.vector_field,
            conditioning,
            cfg_strength,
            sketch_cfg_strength=sketch_cfg_strength,
            control_tokens=control_tokens,
            output_transform=output_transform,
        )
        if self.hparams.parameterization != "endpoint":
            return guided
        return lambda x, t: (guided(x, t) - x) / (1 - t)

    @jaxtyped(typechecker=beartype)
    def _endpoint_prediction_to_model(
        self,
        prediction: Float[torch.Tensor, "batch params"],
    ) -> Float[torch.Tensor, "batch params"]:
        """Interpret raw field output as the configured endpoint representation.

        :param prediction: Raw field output.
        :returns: Model-space endpoint, converting mixed one-hot logits exactly once.
        """
        if self.hparams.endpoint_loss == "mse":
            return prediction
        assert self._metric_param_spec is not None
        return endpoint_prediction_to_model(prediction, self._metric_param_spec)

    @jaxtyped(typechecker=beartype)
    def _one_step_estimate(
        self,
        x_t: Float[torch.Tensor, "batch params"],
        t: Float[torch.Tensor, _BATCH_TIME_SHAPE],
        prediction: Float[torch.Tensor, "batch params"],
    ) -> Float[torch.Tensor, "batch params"]:
        """Return the endpoint prediction, or extrapolate it from a velocity prediction.

        :param x_t: Trajectory point.
        :param t: Flow time.
        :param prediction: Field output at ``(x_t, t)`` under the configured parameterization.
        :returns: Estimate of ``x1``.
        """
        if self.hparams.parameterization == "endpoint":
            return self._endpoint_prediction_to_model(prediction)
        return x_t + (1 - t) * prediction

    def _sample(
        self,
        conditioning: torch.Tensor | None,
        noise: torch.Tensor,
        steps: int,
        cfg_strength: float,
        *,
        sketch_cfg_strength: float | None = None,
        control_tokens: ControlTokenBranches | None = None,
    ) -> torch.Tensor:
        if conditioning is not None:
            conditioning = self.encoder(conditioning)

        guided_velocity = self._velocity_field(
            conditioning,
            cfg_strength,
            control_tokens,
            sketch_cfg_strength=sketch_cfg_strength,
        )
        return integrate_flow(
            guided_velocity,
            noise,
            steps,
            warp_time=self._warp_time,
            parameterization=self.hparams.parameterization,
        )

    @torch.inference_mode()
    @jaxtyped(typechecker=beartype)
    def sample_batch(
        self,
        batch: _SampleBatch,
        *,
        noise: Float[torch.Tensor, "batch params"],
        content_cfg_strength: float,
        sketch_cfg_strength: float,
        sample_steps: int | None = None,
    ) -> Float[torch.Tensor, "batch params"]:
        """Sample model-space parameters from explicit reusable noise.

        :param batch: Model batch carrying content and sketch conditioning.
        :param noise: Float32 initial state shaped ``(batch, num_params)``.
        :param content_cfg_strength: Non-negative content guidance scale.
        :param sketch_cfg_strength: Non-negative sketch guidance scale.
        :param sample_steps: Positive integration steps, or the checkpoint test default.
        :returns: Sampled model-space parameter rows.
        :raises ValueError: Noise, guidance, or integration steps violate the checkpoint contract.
        """
        conditioning = self._get_conditioning_from_batch(batch)
        expected_shape = (conditioning.shape[0], self.hparams.num_params)
        if tuple(noise.shape) != expected_shape:
            raise ValueError(f"noise shape must be {expected_shape}, got {tuple(noise.shape)}")
        if noise.dtype is not torch.float32:
            raise ValueError(f"noise dtype must be float32, got {noise.dtype}")
        if noise.device != conditioning.device:
            raise ValueError(f"noise device must be {conditioning.device}, got {noise.device}")
        if not torch.isfinite(noise).all():
            raise ValueError("noise must contain only finite values")
        for name, strength in (
            ("content_cfg_strength", content_cfg_strength),
            ("sketch_cfg_strength", sketch_cfg_strength),
        ):
            if not math.isfinite(strength) or strength < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        steps = self.hparams.test_sample_steps if sample_steps is None else sample_steps
        if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
            raise ValueError("sample_steps must be a positive integer")

        return self._sample(
            conditioning,
            noise,
            steps,
            content_cfg_strength,
            sketch_cfg_strength=sketch_cfg_strength,
            control_tokens=self._control_token_branches_from_batch(batch),
        )

    @jaxtyped(typechecker=beartype)
    def _log_validation_pitch_residuals(
        self,
        predicted: Float[torch.Tensor, "batch params"],
        target: Float[torch.Tensor, "batch params"],
    ) -> None:
        """Log row-weighted signed pitch residual means in semitones.

        :param predicted: Sampled model-space parameter vectors.
        :param target: Ground-truth model-space parameter vectors.
        """
        if self._pitch_metric_spec is None:
            return
        residuals = midi_pitch_residuals(predicted, target, self._pitch_metric_spec)
        for decode_policy, values in residuals.items():
            self.log(
                f"val/pitch_residual_{decode_policy}_mean_semitones",
                values.mean(),
                on_step=False,
                on_epoch=True,
                batch_size=predicted.shape[0],
                sync_dist=True,
            )

    @jaxtyped(typechecker=beartype)
    def _evaluation_noise(
        self,
        params: Float[torch.Tensor, "batch params"],
        batch_idx: int,
        stage: Literal["test", "val"],
    ) -> Float[torch.Tensor, "batch params"]:
        """Generate stage-local sampling noise without advancing global RNG state.

        :param params: Target rows defining output shape and device; noise is float32.
        :param batch_idx: Stable loader batch position within the current rank.
        :param stage: Evaluation split namespace.
        :returns: Deterministic noise for a fixed seed and loader topology.
        """
        rank = 0 if self._trainer is None else self.trainer.global_rank
        stage_offset = 0 if stage == "val" else _EVAL_TEST_SEED_OFFSET
        seed = (
            self._evaluation_seed + stage_offset + batch_idx * _EVAL_BATCH_SEED_STRIDE + rank
        ) % _EVAL_SEED_MODULUS
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(params.shape, dtype=torch.float32, generator=generator)
        return noise.to(params.device)

    @jaxtyped(typechecker=beartype)
    def _per_param_mse_outputs(
        self,
        predicted: Float[torch.Tensor, "batch params"],
        target: Float[torch.Tensor, "batch params"],
        number_group_metric: NumberGroupSwapParamMSE | None,
    ) -> dict[str, Shaped[torch.Tensor, ...]]:
        """Build the per-parameter metrics consumed by evaluation callbacks.

        :param predicted: Sampled model-space parameter vectors.
        :param target: Ground-truth model-space parameter vectors.
        :param number_group_metric: Structured metric defining eligible parameter swaps.
        :returns: Scalar, per-parameter, and prediction tensors for one batch.
        """
        per_param_mse = (predicted - target).square().mean(dim=0)
        outputs = {
            "param_mse": per_param_mse.mean(),
            "per_param_mse": per_param_mse,
            "per_param_mse_best_swap": best_swap_per_param_mse(predicted, target),
            "preds": predicted,
        }
        if number_group_metric is not None:
            outputs["per_param_mse_number_group_swap"] = number_group_swap_per_param_mse(
                predicted,
                target,
                number_group_metric.param_spec,
            )
        return outputs

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        if self.hparams.seeded_evaluation:
            noise = self._evaluation_noise(batch["params"], batch_idx, "val")
            pred_params = self.sample_batch(
                batch,
                noise=noise,
                content_cfg_strength=self.hparams.validation_cfg_strength,
                sketch_cfg_strength=(
                    self.hparams.validation_cfg_strength
                    if self.hparams.validation_sketch_cfg_strength is None
                    else self.hparams.validation_sketch_cfg_strength
                ),
                sample_steps=self.hparams.validation_sample_steps,
            )
        else:
            conditioning = self._get_conditioning_from_batch(batch)
            pred_params = self._sample(
                conditioning,
                torch.randn_like(batch["params"]),
                self.hparams.validation_sample_steps,
                self.hparams.validation_cfg_strength,
                sketch_cfg_strength=self.hparams.validation_sketch_cfg_strength,
                control_tokens=self._control_token_branches_from_batch(batch),
            )

        self._log_validation_pitch_residuals(pred_params, batch["params"])
        outputs = self._per_param_mse_outputs(
            pred_params,
            batch["params"],
            self.val_param_mse_number_group_swap,
        )
        self.log(
            "val/param_mse", outputs["param_mse"], on_step=False, on_epoch=True, prog_bar=True
        )

        self.val_param_mse_best_swap.update(pred_params, batch["params"])
        self.log(
            "val/param_mse_best_swap",
            self.val_param_mse_best_swap,
            on_step=False,
            on_epoch=True,
        )
        if self.val_param_mse_number_group_swap is not None:
            self.val_param_mse_number_group_swap.update(pred_params, batch["params"])
            self.log(
                "val/param_mse_number_group_swap",
                self.val_param_mse_number_group_swap,
                on_step=False,
                on_epoch=True,
            )
        return outputs

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        if self.hparams.seeded_evaluation:
            noise = self._evaluation_noise(batch["params"], batch_idx, "test")
            pred_params = self.sample_batch(
                batch,
                noise=noise,
                content_cfg_strength=self.hparams.test_cfg_strength,
                sketch_cfg_strength=(
                    self.hparams.test_cfg_strength
                    if self.hparams.test_sketch_cfg_strength is None
                    else self.hparams.test_sketch_cfg_strength
                ),
                sample_steps=self.hparams.test_sample_steps,
            )
        else:
            conditioning = self._get_conditioning_from_batch(batch)
            pred_params = self._sample(
                conditioning,
                torch.randn_like(batch["params"]),
                self.hparams.test_sample_steps,
                self.hparams.test_cfg_strength,
                sketch_cfg_strength=self.hparams.test_sketch_cfg_strength,
                control_tokens=self._control_token_branches_from_batch(batch),
            )

        outputs = self._per_param_mse_outputs(
            pred_params,
            batch["params"],
            self.test_param_mse_number_group_swap,
        )
        self.log(
            "test/param_mse", outputs["param_mse"], on_step=False, on_epoch=True, prog_bar=True
        )

        self.test_param_mse_best_swap.update(pred_params, batch["params"])
        self.log(
            "test/param_mse_best_swap",
            self.test_param_mse_best_swap,
            on_step=False,
            on_epoch=True,
        )
        if self.test_param_mse_number_group_swap is not None:
            self.test_param_mse_number_group_swap.update(pred_params, batch["params"])
            self.log(
                "test/param_mse_number_group_swap",
                self.test_param_mse_number_group_swap,
                on_step=False,
                on_epoch=True,
            )
        return outputs

    def on_test_epoch_end(self) -> None:
        pass

    def predict_step(
        self, batch: dict[str, Shaped[torch.Tensor, _BATCH_ANY_SHAPE]], batch_idx: int
    ):
        conditioning = self._get_conditioning_from_batch(batch)
        return (
            self._sample(
                conditioning,
                torch.randn(
                    conditioning.shape[0],
                    self.hparams.num_params,
                    device=conditioning.device,
                ),
                self.hparams.test_sample_steps,
                self.hparams.test_cfg_strength,
                sketch_cfg_strength=self.hparams.test_sketch_cfg_strength,
                control_tokens=self._control_token_branches_from_batch(batch),
            ),
            batch,
        )

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.vector_field.compile()
            self.encoder.compile()

    def on_before_optimizer_step(self, optimizer) -> None:
        vf_norms = grad_norm(self.vector_field, 2.0)
        encoder_norms = grad_norm(self.encoder, 2.0)

        vf_norms = {f"vector_field/{k}": v for k, v in vf_norms.items()}
        encoder_norms = {f"encoder/{k}": v for k, v in encoder_norms.items()}

        self.log_dict(vf_norms, on_step=True, on_epoch=False)
        self.log_dict(encoder_norms, on_step=True, on_epoch=False)

    @jaxtyped(typechecker=beartype)
    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: int | float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        """Reject a non-finite gradient before clipping rescales every parameter by NaN.

        ``clip_grad_norm_`` defaults to ``error_if_nonfinite=False``, so one overflowing
        row turns the total norm into NaN and poisons all weights; the failure then
        surfaces a step later as a diverged parameter estimate. Runs under 32-bit
        precision only — an AMP ``GradScaler`` produces transient infs by design.

        :param optimizer: Optimizer whose gradients are about to be clipped.
        :param gradient_clip_val: Clip threshold Lightning resolves from the trainer.
        :param gradient_clip_algorithm: Clip algorithm Lightning resolves from the trainer.
        :raises ValueError: Any parameter carries a non-finite gradient.
        """
        corrupted = [
            name
            for name, parameter in self.named_parameters()
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        ]
        if corrupted:
            raise ValueError(
                f"non-finite gradient in {len(corrupted)} parameter(s) {corrupted}; rejecting "
                "the step at its source rather than letting clipping scale every parameter by NaN"
            )
        super().configure_gradient_clipping(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )

    def configure_optimizers(self) -> dict[str, object]:
        trainable_parameters = (
            parameter for parameter in self.trainer.model.parameters() if parameter.requires_grad
        )
        optimizer = self.hparams.optimizer(params=trainable_parameters)

        if self.hparams.warmup_steps > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, 1e-10, 1.0, self.hparams.warmup_steps
            )
        else:
            warmup_scheduler = None

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
        else:
            scheduler = None

        if warmup_scheduler is not None and scheduler is None:
            scheduler = warmup_scheduler
        elif warmup_scheduler is not None and scheduler is not None:
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, scheduler],
                milestones=[self.hparams.warmup_steps],
            )

        if scheduler is not None:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        return {"optimizer": optimizer}


# Deprecated alias: archived W&B run configs and external job scripts resolve the
# old ``_target_`` path.
SurgeFlowMatchingModule = VSTFlowMatchingModule
