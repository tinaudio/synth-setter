"""Simulator-feedback finetuning of a frozen VST flow model (arXiv 2410.22573).

Spike prototype: the pretrained :class:`VSTFlowMatchingModule` stays frozen and
a small control encoder + control field learn a late-time (``t >= t_min``)
correction from online renders of the one-step parameter estimate, compared to
the target in music2latent embedding space.
"""

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lightning import LightningModule
from torch import nn

from synth_setter.conditioning import resolve_embedding_conditioning
from synth_setter.models.vst_flow_matching_module import (
    VSTFlowMatchingModule,
    rk4_with_cfg,
)

# Numpy audio (B, C, T) -> latent tensor (B, C*D, T_lat) on the module device.
M2LEncodeFn = Callable[[np.ndarray], torch.Tensor]
# Prediction rows (B, P) in [-1, 1] -> rendered audio (B, C, T) and failure count.
RenderBatchFn = Callable[[np.ndarray], tuple[np.ndarray, int]]


class ControlEncoder(nn.Module):
    """Summarize (rendered, target) m2l latent pairs into one control vector."""

    def __init__(self, embed_dim: int, hidden_dim: int, control_dim: int):
        """Build the per-frame MLP and pooling head.

        :param embed_dim: Latent channel width of one m2l embedding (C*D).
        :param hidden_dim: Per-frame MLP width.
        :param control_dim: Output control-vector width.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(3 * embed_dim),
            nn.Linear(3 * embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.head = nn.Linear(hidden_dim, control_dim)

    def forward(self, rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Encode the pair into a control vector.

        :param rendered: Rendered-audio latents ``(B, C*D, T_lat)``.
        :param target: Target-audio latents ``(B, C*D, T_lat)``.
        :returns: Control vectors ``(B, control_dim)``.
        """
        feats = torch.cat((target, rendered, target - rendered), dim=1).permute(0, 2, 1)
        pooled = self.net(feats).mean(dim=1)
        return self.head(pooled)


class ControlField(nn.Module):
    """Residual correction field ``v_C(x_t, t, v, c)`` with a zero-init output head."""

    def __init__(
        self,
        num_params: int,
        control_dim: int,
        hidden_dim: int,
        num_blocks: int,
    ):
        """Build the conditioned residual MLP.

        :param num_params: Parameter-vector width the field corrects.
        :param control_dim: Control-vector width from :class:`ControlEncoder`.
        :param hidden_dim: Backbone width.
        :param num_blocks: Number of conditioned residual blocks.
        """
        super().__init__()
        from synth_setter.models.components.vector_field import VectorFieldBlock

        self.input = nn.Linear(2 * num_params, hidden_dim)
        self.blocks = nn.ModuleList(
            [VectorFieldBlock(hidden_dim, control_dim + 1) for _ in range(num_blocks)]
        )
        # Zero-init: the finetuned field starts exactly at the frozen base field.
        self.head = nn.Linear(hidden_dim, num_params)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        v: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the correction to the base velocity.

        :param x_t: Flow state ``(B, P)``.
        :param t: Flow time ``(B, 1)``.
        :param v: Frozen-base velocity ``(B, P)``.
        :param c: Control vectors ``(B, control_dim)``.
        :returns: Velocity correction ``(B, P)``.
        """
        z = torch.cat((c, t), dim=-1)
        y = self.input(torch.cat((x_t, v), dim=-1))
        for block in self.blocks:
            y = block(y, z)
        return self.head(y)


class VSTFlowMatchingFinetuneModule(LightningModule):
    """Online simulator-feedback finetune of a frozen pretrained VST flow model."""

    def __init__(
        self,
        pretrained_ckpt: str,
        optimizer: Any,
        scheduler: Any,
        *,
        feedback_enabled: bool = True,
        t_min: float = 0.8,
        control_dim: int = 128,
        control_hidden_dim: int = 512,
        control_num_blocks: int = 2,
        control_encoder_hidden_dim: int = 256,
        synth_name: str = "surge_simple",
        renderer_backend: str = "surgepy",
        sample_rate: int = 44100,
        channels: int = 2,
        signal_duration_seconds: float = 4.0,
        velocity: int = 100,
        warmup_steps: int = 0,
        validation_sample_steps: int = 10,
    ):
        """Load and freeze the base flow, then build the trainable control nets.

        :param pretrained_ckpt: Checkpoint path for the frozen
            :class:`VSTFlowMatchingModule` base.
        :param optimizer: ``functools.partial``-style optimizer factory over the
            control parameters only.
        :param scheduler: ``functools.partial``-style scheduler factory or ``None``.
        :param feedback_enabled: Whether the control vector carries simulator
            feedback; ``False`` trains the same capacity with ``c = 0``.
        :param t_min: Flow time below which the base field is used unchanged.
        :param control_dim: Control-vector width.
        :param control_hidden_dim: Control-field backbone width.
        :param control_num_blocks: Control-field residual block count.
        :param control_encoder_hidden_dim: Control-encoder per-frame MLP width.
        :param synth_name: Registered synth identity used for online rendering.
        :param renderer_backend: ``surgepy`` (fast, fully headless) or
            ``pedalboard`` (VST3 host; display-free with ``warmup=False``).
        :param sample_rate: Render sample rate in Hz (must match the m2l contract).
        :param channels: Render channel count.
        :param signal_duration_seconds: Render duration per sample.
        :param velocity: MIDI velocity for every online render.
        :param warmup_steps: If positive, wrap the scheduler with a linear warmup.
        :param validation_sample_steps: ODE steps for validation sampling.
        :raises ValueError: If the loaded base is not embedding-conditioned.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        self._optimizer_factory = optimizer
        self._scheduler_factory = scheduler
        self._feedback_enabled = feedback_enabled
        self._t_min = t_min
        self._control_dim = control_dim
        self._synth_name = synth_name
        self._renderer_backend = renderer_backend
        self._sample_rate = sample_rate
        self._channels = channels
        self._signal_duration_seconds = signal_duration_seconds
        self._velocity = velocity
        self._warmup_steps = warmup_steps
        self._validation_sample_steps = validation_sample_steps

        # weights_only=False unpickles the module graph; checkpoint provenance is
        # deployment-controlled (same trust stance as predict_capture).
        self.base = VSTFlowMatchingModule.load_from_checkpoint(
            pretrained_ckpt, map_location="cpu", weights_only=False
        )
        self.base.freeze()

        num_params = int(self.base.hparams["num_params"])
        embedding = resolve_embedding_conditioning(self.base.hparams["conditioning"])
        if embedding is None:
            raise ValueError(
                "the frozen base must be embedding-conditioned; a mel-conditioned "
                "base has no cached target representation for feedback"
            )
        embed_dim = int(embedding.input_shape[0])
        self.control_encoder = ControlEncoder(embed_dim, control_encoder_hidden_dim, control_dim)
        self.control_field = ControlField(
            num_params, control_dim, control_hidden_dim, control_num_blocks
        )

        # Injectable simulator resources (tests replace these with fakes).
        self.render_batch_fn: RenderBatchFn | None = None
        self.m2l_encode_fn: M2LEncodeFn | None = None

    def train(self, mode: bool = True) -> "VSTFlowMatchingFinetuneModule":
        """Keep the frozen base in eval mode across Lightning's mode switches.

        :param mode: Requested training mode for the trainable submodules.
        :returns: This module.
        """
        super().train(mode)
        self.base.eval()
        return self

    def _make_render_batch_fn(self) -> RenderBatchFn:
        """Build the display-free in-process batch renderer.

        Rendering uses ``warmup=False``: the Xvfb+warmup path crashes
        in-process (BadWindow) and warmup-less audio matches the production
        renders closely (#2556).

        :returns: Batch renderer mapping prediction rows to audio.
        """
        from synth_setter.data.vst import param_specs
        from synth_setter.data.vst.param_spec import decode_model_output
        from synth_setter.pipeline.schemas.spec import RenderConfig
        from synth_setter.renderer_factory import make_audio_renderer
        from synth_setter.synth_spec import SynthName, SynthSpec, resolve_synth

        synth = resolve_synth(SynthName(self._synth_name))
        spec = param_specs[synth.param_spec_name]
        if self._renderer_backend == "surgepy":
            # In-process engine identity mirroring configs/render/*_surgepy.yaml.
            synth = SynthSpec(
                name=synth.name,
                param_spec_name=synth.param_spec_name,
                plugin_path="surgepy",
                plugin_state_path=str(Path(synth.plugin_state_path).with_suffix(".fxp")),
                synth_version="1.3.master.f7b97c68",
            )
        if self._renderer_backend == "surgepy":
            config = RenderConfig(
                synth=synth,
                renderer_backend="surgepy",
                sample_rate=self._sample_rate,
                channels=self._channels,
                velocity=self._velocity,
                signal_duration_seconds=self._signal_duration_seconds,
                min_loudness=-55.0,
                samples_per_shard=1,
                plugin_reload_cadence="render",
                gui_toggle_cadence="never",
            )
        else:
            config = RenderConfig(
                synth=synth,
                renderer_backend="pedalboard",
                sample_rate=self._sample_rate,
                channels=self._channels,
                velocity=self._velocity,
                signal_duration_seconds=self._signal_duration_seconds,
                min_loudness=-55.0,
                samples_per_shard=1,
            )
        renderer = make_audio_renderer(config)
        samples = int(self._sample_rate * self._signal_duration_seconds)
        duration = self._signal_duration_seconds
        velocity = self._velocity
        channels = self._channels

        def render_batch(rows: np.ndarray) -> tuple[np.ndarray, int]:
            out = np.zeros((rows.shape[0], channels, samples), np.float32)
            failures = 0
            for i, row in enumerate(rows):
                synth_params, note_params = decode_model_output(row, spec)
                # Predicted rows can decode to degenerate note windows; clamp
                # into the rendered duration before the host sees them.
                start, end = note_params["note_start_and_end"]
                start = float(np.clip(start, 0.0, duration - 0.1))
                end = float(np.clip(end, start + 0.05, duration))
                try:
                    out[i] = renderer.render(
                        synth_params,
                        int(np.clip(note_params["pitch"], 0, 127)),
                        velocity,
                        (start, end),
                    )
                except (ValueError, RuntimeError):
                    # A failed render feeds back silence rather than killing the
                    # step; the failure count is logged per step.
                    failures += 1
            return out, failures

        return render_batch

    def _make_m2l_encode_fn(self) -> M2LEncodeFn:
        """Build the frozen device-resident music2latent encoder.

        :returns: Encoder mapping numpy audio to latents on the module device.
        """
        from einops import rearrange
        from music2latent import EncoderDecoder

        encoder = EncoderDecoder(device=str(self.device))

        def encode(audio: np.ndarray) -> torch.Tensor:
            flat = np.ascontiguousarray(rearrange(audio, "b c t -> (b c) t"), dtype=np.float32)
            with torch.no_grad():
                latents = encoder.encode(flat, max_batch_size=64)
            latents = rearrange(
                latents, "(b c) d t -> b (c d) t", b=audio.shape[0], c=audio.shape[1]
            )
            return latents.to(self.device, dtype=torch.float32)

        return encode

    def _simulator(self) -> tuple[RenderBatchFn, M2LEncodeFn]:
        """Return the renderer and m2l encoder, building them on first use.

        :returns: The render and encode callables.
        """
        if self.render_batch_fn is None:
            self.render_batch_fn = self._make_render_batch_fn()
        if self.m2l_encode_fn is None:
            self.m2l_encode_fn = self._make_m2l_encode_fn()
        return self.render_batch_fn, self.m2l_encode_fn

    def _sample_late_time(self, n: int, device: torch.device) -> torch.Tensor:
        return self._t_min + (1 - self._t_min) * torch.rand(n, 1, device=device)

    def _control_vector(
        self, theta_hat: torch.Tensor, target_m2l: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Render the one-step estimate and encode the feedback control vector.

        :param theta_hat: One-step parameter estimates ``(B, P)`` in model scale.
        :param target_m2l: Target m2l latents ``(B, C*D, T_lat)``.
        :returns: Control vectors ``(B, control_dim)`` and stage timings/failures.
        """
        stats: dict[str, float] = {}
        if not self._feedback_enabled:
            stats["render_failures"] = 0.0
            zeros = torch.zeros(theta_hat.shape[0], self._control_dim, device=theta_hat.device)
            return zeros, stats

        render_batch, m2l_encode = self._simulator()
        t0 = time.perf_counter()
        rows = theta_hat.detach().cpu().numpy().astype(np.float32)
        rendered, failures = render_batch(rows)
        stats["time_render"] = time.perf_counter() - t0
        stats["render_failures"] = float(failures)

        t0 = time.perf_counter()
        rendered_m2l = m2l_encode(rendered)
        stats["time_m2l"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        control = self.control_encoder(rendered_m2l, target_m2l)
        stats["time_control_enc"] = time.perf_counter() - t0
        return control, stats

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        """Run one online-feedback CFM step over late flow times.

        :param batch: Model batch with ``conditioning``, ``params`` and ``noise``.
        :param batch_idx: Lightning batch index.
        :returns: Scalar CFM loss over the effective velocity.
        """
        conditioning = batch["conditioning"]
        params = batch["params"]
        noise = batch["noise"]

        t0 = time.perf_counter()
        with torch.no_grad():
            z = self.base.encoder(conditioning)
            t = self._sample_late_time(params.shape[0], params.device)
            x0, x1 = noise, params
            x_t = self.base._sample_probability_path(x0, x1, t)
            target = x1 - x0
            v = self.base.vector_field(x_t, t, z)
            theta_hat = x_t + (1 - t) * v
        flow_s = time.perf_counter() - t0

        control, stats = self._control_vector(theta_hat, conditioning)

        t0 = time.perf_counter()
        v_eff = v + self.control_field(x_t, t, v, control)
        loss = (v_eff - target).square().mean()
        stats["time_control_field"] = time.perf_counter() - t0
        stats["time_flow_fwd"] = flow_s

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        base_loss = (v - target).square().mean()
        self.log("train/base_loss", base_loss, on_step=True, on_epoch=True)
        for name, value in stats.items():
            self.log(f"train/{name}", value, on_step=True, on_epoch=False)
        return loss

    @torch.no_grad()
    def sample(
        self,
        conditioning: torch.Tensor,
        noise: torch.Tensor,
        steps: int,
    ) -> torch.Tensor:
        """Integrate the finetuned field without CFG.

        RK4 on the frozen base below ``t_min``; Euler with the control
        correction (feedback-rendered when enabled) above it.

        :param conditioning: Raw conditioning batch ``(B, C*D, T_lat)``.
        :param noise: Initial noise ``(B, P)``.
        :param steps: Total ODE step count.
        :returns: Sampled parameter rows ``(B, P)`` in model scale.
        """
        z = self.base.encoder(conditioning)
        t = torch.zeros(noise.shape[0], 1, device=noise.device)
        dt = 1.0 / steps
        x = noise

        for _ in range(steps):
            if float(t[0, 0]) < self._t_min - 1e-6:
                # cfg_strength=1 reduces to the plain conditional field.
                x = rk4_with_cfg(self.base.vector_field, x, t, dt, z, 1.0)
            else:
                v = self.base.vector_field(x, t, z)
                theta_hat = x + (1 - t) * v
                control, _ = self._control_vector(theta_hat, conditioning)
                x = x + dt * (v + self.control_field(x, t, v, control))
            t = t + dt

        return x

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        """Sample with the finetuned field and report parameter MSE.

        :param batch: Model batch with ``conditioning`` and ``params``.
        :param batch_idx: Lightning batch index.
        :returns: Scalar validation parameter MSE.
        """
        pred = self.sample(
            batch["conditioning"],
            torch.randn_like(batch["params"]),
            self._validation_sample_steps,
        )
        param_mse = (pred - batch["params"]).square().mean()
        self.log("val/param_mse", param_mse, on_step=False, on_epoch=True, prog_bar=True)
        return param_mse

    def configure_optimizers(self) -> dict[str, Any]:
        """Optimize the control encoder and control field only.

        :returns: Lightning optimizer (and scheduler) configuration.
        """
        trainable = list(self.control_encoder.parameters()) + list(self.control_field.parameters())
        optimizer = self._optimizer_factory(params=trainable)

        scheduler = None
        if self._scheduler_factory is not None:
            scheduler = self._scheduler_factory(optimizer=optimizer)
        if self._warmup_steps > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(optimizer, 1e-10, 1.0, self._warmup_steps)
            scheduler = (
                warmup
                if scheduler is None
                else torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup, scheduler],
                    milestones=[self._warmup_steps],
                )
            )

        if scheduler is not None:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
            }
        return {"optimizer": optimizer}
