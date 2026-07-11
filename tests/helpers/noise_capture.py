"""Lightning callback recording each training batch's noise tensor.

Hydra-instantiated into a smoke ``train(cfg)`` run (the test injects a
``_target_`` pointing here), so ``tests/test_train.py`` can compare the noise
stream two runs actually consumed and pin that ``cfg.seed`` governs the
production noise draw end-to-end, dataloader workers included.
"""

from typing import Any, ClassVar

import torch
from lightning import Callback, LightningModule, Trainer


# DOC601/DOC603: pydoclint can't read the docstring-body attribute docs, so the
# ClassVar annotation is documented in prose instead.
class NoiseCaptureCallback(Callback):  # noqa: DOC601, DOC603
    """Append every training batch's ``noise`` tensor to a class-level list.

    ``captured`` is class-level because Hydra instantiates the callback out of
    the test's reach; tests clear it before each run and read it after.
    """

    captured: ClassVar[list[torch.Tensor]] = []

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: dict[str, torch.Tensor | None],
        batch_idx: int,
    ) -> None:
        """Record the batch's noise tensor.

        :param trainer: Unused; Lightning hook signature.
        :param pl_module: Unused; Lightning hook signature.
        :param outputs: Unused; Lightning hook signature.
        :param batch: The training batch; its ``noise`` entry is captured.
        :param batch_idx: Unused; Lightning hook signature.
        """
        noise = batch["noise"]
        assert noise is not None
        # clone(): worker batches arrive in shared memory the loader may reuse;
        # cpu(): don't retain device tensors for the whole run under GPU tests.
        NoiseCaptureCallback.captured.append(noise.detach().clone().cpu())
