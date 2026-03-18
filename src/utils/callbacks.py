import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from lightning.pytorch.callbacks import BasePredictionWriter, Callback

from src.data.vst import param_specs
from src.models.components.transformer import (
    ApproxEquivTransformer,
    LearntProjection,
)
from src.models.ksin_flow_matching_module import KSinFlowMatchingModule
from src.models.surge_flow_matching_module import SurgeFlowMatchingModule


class PlotLossPerTimestep(Callback):
    """Takes a batch from the validation dataloader, and runs it through the model at a number of
    different values for t.

    Plots the loss as a function of t.
    """

    def __init__(self, num_timesteps: int = 100):
        super().__init__()
        self.num_timesteps = num_timesteps

    def _get_val_batch(self, trainer):
        val_dl = trainer.val_dataloaders
        return next(iter(val_dl))

    def _compute_losses(self, trainer, pl_module):
        batch = self._get_val_batch(trainer)
        signal, params, _ = batch

        # Get conditioning vector
        conditioning = pl_module.encoder(signal)
        z = pl_module.vector_field.apply_dropout(conditioning, pl_module.hparams.cfg_dropout_rate)

        x0, x1, z = pl_module._sample_x0_and_x1(params, z)

        losses = []
        for n in range(self.num_timesteps):
            t = torch.full(
                (signal.shape[0], 1), n / (self.num_timesteps - 1), device=signal.device
            )
            x_t = pl_module._sample_probability_path(x0, x1, t)
            target = pl_module._evaluate_target_field(x0, x1, x_t, t)

            prediction = pl_module.vector_field(x_t, t, z)
            loss = (prediction - target).square().mean(dim=-1)
            losses.append(loss)

        return torch.stack(losses, dim=-1)

    def _aggregate_losses(self, losses):
        mean = losses.mean(dim=0)
        std = losses.std(dim=0)
        lower_ci = mean - 2 * std
        upper_ci = mean + 2 * std
        return mean, lower_ci, upper_ci

    def _plot_losses(self, losses):
        t = np.linspace(0, 1, self.num_timesteps)
        mean, lower_ci, upper_ci = self._aggregate_losses(losses)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(t, mean.cpu().numpy())
        ax.fill_between(t, lower_ci.cpu().numpy(), upper_ci.cpu().numpy(), alpha=0.2)
        ax.set_xlabel("t")
        ax.set_ylabel("Loss")
        ax.set_title("Loss per noise level / timestep")
        return fig

    def _log_plot(self, fig, trainer):
        plot = wandb.Image(fig)
        wandb.log({"plot": plot}, step=trainer.global_step)
        plt.close(fig)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        losses = self._compute_losses(trainer, pl_module)
        fig = self._plot_losses(losses)
        self._log_plot(fig, trainer)


def _self_similarity(x):
    y = x.permute(1, 0, 2)
    sim = torch.nn.functional.cosine_similarity(x, y, dim=-1)
    return sim


class PlotPositionalEncodingSimilarity(Callback):
    def _compute_similarity(self, pl_module):
        if pl_module.vector_field.pe_type == "initial":
            return _self_similarity(pl_module.vector_field.pe.pe)
        elif pl_module.vector_field.pe_type == "layerwise":
            return [_self_similarity(pe.pe) for pe in pl_module.vector_field.pe]

    def _plot_single_similarity(self, sim):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        ax.imshow(sim.cpu().numpy(), vmin=-1, vmax=1, aspect="equal")
        fig.tight_layout()
        fig.suptitle("Positional Encoding Similarity")

        return fig

    def _plot_multiple_similarities(self, sims):
        n_pe = len(sims)
        n_rows = int(np.sqrt(n_pe))
        n_cols = int(np.ceil(n_pe / n_rows))

        fig, ax = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))

        for i, sim in enumerate(sims):
            ax[i // n_cols, i % n_cols].imshow(sim.cpu().numpy(), vmin=-1, vmax=1, aspect="equal")
            ax[i // n_cols, i % n_cols].set_title(f"PE {i // n_cols}-{i % n_cols}", fontsize=8)

        for i in range(n_pe, n_rows * n_cols):
            ax[i // n_cols, i % n_cols].axis("off")

        fig.tight_layout()
        fig.suptitle("Positional Encoding Similarities")

        return fig

    def _plot_similarity(self, sim):
        if isinstance(sim, torch.Tensor):
            return self._plot_single_similarity(sim)
        else:
            return self._plot_multiple_similarities(sim)

    def _log_plot(self, fig, trainer):
        plot = wandb.Image(fig)
        wandb.log({"pos_enc_similarity": plot}, step=trainer.global_step)
        plt.close(fig)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not isinstance(pl_module, KSinFlowMatchingModule):
            return

        if not isinstance(pl_module.vector_field, ApproxEquivTransformer):
            return

        if pl_module.vector_field.pe_type == "none":
            return

        pe_sims = self._compute_similarity(pl_module)
        fig = self._plot_similarity(pe_sims)
        self._log_plot(fig, trainer)


class PlotLearntProjection(Callback):
    def __init__(
        self,
        after_val: bool = True,
        every_n_steps: int | None = None,
        sort_assignments: bool = True,
    ):
        super().__init__()
        self.after_val = after_val
        self.every_n_steps = every_n_steps
        self.sort_assignments = sort_assignments

    def _get_assignment(self, pl_module):
        return pl_module.vector_field.projection.assignment

    def _sort_assignments(self, assignment):
        assignment = assignment.abs()
        k = torch.arange(assignment.shape[-1], device=assignment.device)[None]
        positional_average = torch.sum(assignment * k, dim=-1) / torch.sum(assignment, dim=-1)
        sorted_idxs = torch.argsort(positional_average)
        assignment = assignment[sorted_idxs]
        return assignment

    def _plot_assignments(self, pl_module):
        assignment = self._get_assignment(pl_module)

        if self.sort_assignments:
            assignment = self._sort_assignments(assignment)

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        maxval = assignment.abs().max().item()
        img = ax.imshow(
            assignment.cpu().numpy(),
            aspect="equal",
            vmin=-maxval,
            vmax=maxval,
            cmap="RdBu",
        )
        fig.colorbar(img, ax=ax)

        ax.set_title("Assignment")

        ax.set_xlabel("params")
        ax.set_ylabel("tokens")
        fig.tight_layout()
        fig.suptitle("Learnt Assignment")

        return fig

    def _get_value_similarity(self, pl_module):
        proj = pl_module.vector_field.projection.in_projection  # num_params x d_embed x d_model

        sim_proj = torch.nn.functional.cosine_similarity(proj[None], proj[:, None], dim=-1)

        return sim_proj

    def _get_output_similarity(self, pl_module):
        proj = pl_module.vector_field.projection.out_projection.T  # num_params x d_embed x d_model

        sim_proj = torch.nn.functional.cosine_similarity(proj[None], proj[:, None], dim=-1)

        return sim_proj

    def _plot_projections(self, pl_module):
        fig, ax = plt.subplots(2, 1, figsize=(5, 10))

        val_sim = self._get_value_similarity(pl_module)
        out_sim = self._get_output_similarity(pl_module)

        val_max = val_sim.abs().max()
        out_max = out_sim.abs().max()

        val_im = ax[0].imshow(
            val_sim.cpu().numpy(),
            aspect="equal",
            vmin=-val_max,
            vmax=val_max,
            cmap="RdBu",
        )
        ax[0].set_title("Value Projection")
        ax[0].set_xlabel("params")
        ax[0].set_ylabel("params")

        out_im = ax[1].imshow(
            out_sim.cpu().numpy(),
            aspect="equal",
            vmin=-out_max,
            vmax=out_max,
            cmap="RdBu",
        )
        ax[1].set_title("Out Projection")
        ax[1].set_xlabel("params")
        ax[1].set_ylabel("params")

        # show colorbar
        fig.colorbar(val_im, ax=ax[0])
        fig.colorbar(out_im, ax=ax[1])

        fig.tight_layout()

        return fig

    def _log_plots(self, fig_ass, fig_value, trainer):
        plot_ass = wandb.Image(fig_ass)
        plot_value = wandb.Image(fig_value)
        wandb.log({"assignment": plot_ass, "value": plot_value}, step=trainer.global_step)

        plt.close(fig_ass)
        plt.close(fig_value)

    def _do_plotting(self, trainer, pl_module):
        if not (
            isinstance(pl_module, KSinFlowMatchingModule)
            or isinstance(pl_module, SurgeFlowMatchingModule)
        ):
            return

        if not hasattr(pl_module.vector_field, "projection"):
            return

        if not isinstance(pl_module.vector_field, LearntProjection):
            return

        # if not isinstance(pl_module.vector_field, ApproxEquivTransformer):
        #     print("wrong vector field")
        #     return
        #
        # if not isinstance(pl_module.vector_field.projection, LearntProjection):
        #     print("wrong projection")
        #     return

        print("plotting")
        fig_ass = self._plot_assignments(pl_module)
        fig_value = self._plot_projections(pl_module)
        self._log_plots(fig_ass, fig_value, trainer)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not self.after_val:
            return

        self._do_plotting(trainer, pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.every_n_steps is None:
            return

        if trainer.global_step % self.every_n_steps != 0:
            return

        with torch.no_grad():
            self._do_plotting(trainer, pl_module)


class PredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        prediction, batch = prediction
        torch.save(prediction, os.path.join(self.output_dir, f"pred-{batch_idx}.pt"))
        torch.save(
            batch["audio"],
            os.path.join(self.output_dir, f"target-audio-{batch_idx}.pt"),
        )

        if "params" in batch:
            torch.save(
                batch["params"],
                os.path.join(self.output_dir, f"target-params-{batch_idx}.pt"),
            )

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        predictions, batch = predictions
        torch.save(predictions, os.path.join(self.output_dir, "predictions.pt"))
        torch.save(batch["audio"], os.path.join(self.output_dir, "target-audio.pt"))

        if "params" in batch:
            torch.save(batch["params"], os.path.join(self.output_dir, "target-params.pt"))


class LogPerParamMSE(Callback):
    def __init__(self, param_spec: str = "surge_simple"):
        super().__init__()
        self.param_spec = param_specs[param_spec]

    def on_validation_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.per_param_mse = 0.0
        self.count = 0

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ) -> None:
        per_param_mse = outputs["per_param_mse"]
        self.per_param_mse += per_param_mse.detach().cpu().numpy()
        self.count += 1

    def on_validation_epoch_end(
        self,
        trainer,
        pl_module,
    ) -> None:
        per_param_mse = self.per_param_mse / self.count
        names = self.param_spec.names
        pl_module.log_dict(
            {f"per_param_mse/{name}": mse for name, mse in zip(names, per_param_mse)},
        )
