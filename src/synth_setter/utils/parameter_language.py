"""Initialize parameter-language artifacts after Lightning restores checkpoint state."""

from lightning import Callback, LightningModule, Trainer

from synth_setter.models.components.language_projection import LanguageParameterProjection


def initialize_parameter_language(model: LightningModule) -> None:
    """Supply finalized field vectors at a lifecycle boundary before numeric transforms run.

    :param model: Consumer containing online and possibly EMA parameter projections.
    """
    for module in model.modules():
        if isinstance(module, LanguageParameterProjection):
            module.initialize_embeddings()


class ParameterLanguageInitializer(Callback):
    """Initialize fresh models while leaving restored, self-contained checkpoints untouched."""

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Initialize before the first optimization step.

        :param trainer: Active Lightning trainer.
        :param pl_module: Restored or freshly constructed consumer.
        """
        initialize_parameter_language(pl_module)

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Initialize before sanity validation or standalone validation.

        :param trainer: Active Lightning trainer.
        :param pl_module: Restored or freshly constructed consumer.
        """
        initialize_parameter_language(pl_module)

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Initialize before held-out testing.

        :param trainer: Active Lightning trainer.
        :param pl_module: Restored or freshly constructed consumer.
        """
        initialize_parameter_language(pl_module)

    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Initialize before inference.

        :param trainer: Active Lightning trainer.
        :param pl_module: Restored or freshly constructed consumer.
        """
        initialize_parameter_language(pl_module)
