"""Learning-rate schedulers with project checkpoint semantics."""

import logging

import torch

log = logging.getLogger(__name__)


class ResumeAwareCosineAnnealingLR(torch.optim.lr_scheduler.CosineAnnealingLR):
    """Keep the configured cosine horizon when extending a checkpointed run."""

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Restore progress while retaining a longer constructor-configured horizon.

        :param state_dict: Scheduler state from a full training checkpoint.
        :raises TypeError: The checkpoint horizon is not an integer.
        :raises ValueError: The configured horizon is shorter than the checkpoint horizon.
        """
        configured_t_max = self.T_max
        checkpoint_t_max = state_dict["T_max"]
        if not isinstance(checkpoint_t_max, int):
            raise TypeError(
                f"checkpoint T_max must be an int, got {type(checkpoint_t_max).__name__}"
            )
        if configured_t_max < checkpoint_t_max:
            raise ValueError(
                f"configured T_max {configured_t_max} is shorter than checkpoint "
                f"T_max {checkpoint_t_max}; reducing the cosine horizon on resume is unsupported"
            )

        super().load_state_dict(state_dict)
        if configured_t_max == checkpoint_t_max:
            return

        self.T_max = configured_t_max
        if self.last_epoch >= 0:
            remapped_lrs = self._get_closed_form_lr()
            for param_group, learning_rate in zip(self.optimizer.param_groups, remapped_lrs):
                param_group["lr"] = learning_rate
            self._last_lr = remapped_lrs
        log.info(
            "Remapped resumed cosine schedule from T_max=%d to T_max=%d at step %d",
            checkpoint_t_max,
            configured_t_max,
            self.last_epoch,
        )
