"""Contract-accurate ``WandbLogger`` fake for entrypoint tests."""

from typing import Any, cast

from lightning.pytorch.loggers.wandb import WandbLogger


class RecordingWandbConfig(dict[str, Any]):
    """``wandb.config`` stand-in honoring ``update(..., allow_val_change=...)``."""

    def __init__(self) -> None:
        """Initialize the config recorder."""
        super().__init__()
        self.allow_val_change_calls: list[bool] = []

    def update(self, params: dict[str, Any], allow_val_change: bool = False) -> None:  # type: ignore[override]
        """Mirror ``wandb.config.update(params, allow_val_change=...)``, not ``dict.update``.

        The fake intentionally keeps W&B's narrower signature because Lightning
        calls ``experiment.config.update(..., allow_val_change=True)`` directly.

        :param params: Hyperparameters Lightning forwards to ``wandb.config``.
        :param allow_val_change: Whether W&B should allow later updates.
        """
        self.allow_val_change_calls.append(allow_val_change)
        super().update(params)


class RecordingWandbExperiment:
    """Injected fake run satisfying the ``WandbLogger.experiment`` contract."""

    def __init__(self) -> None:
        """Initialize the fake W&B run state."""
        self.config = RecordingWandbConfig()
        self.logged_artifacts: list[tuple[Any, list[str] | None]] = []
        self.logged_metrics: list[dict[str, float]] = []
        self.used_artifacts: list[str] = []
        self.id = "fake-run-id"
        self.name = "fake-run"

    def log(self, metrics: dict[str, float]) -> None:
        """Record metrics Lightning routes through ``WandbLogger.log_metrics``.

        :param metrics: Metric payload emitted by Lightning.
        """
        self.logged_metrics.append(metrics)

    def use_artifact(self, name_alias: str) -> None:
        """Record the artifact reference the entrypoint consumes.

        :param name_alias: W&B artifact name with its alias.
        """
        self.used_artifacts.append(name_alias)

    def log_artifact(self, artifact: Any, aliases: list[str] | None = None) -> None:
        """Record the artifact payload Lightning routes to the W&B run.

        :param artifact: Artifact object W&B would upload.
        :param aliases: Optional aliases attached to that artifact.
        """
        self.logged_artifacts.append((artifact, aliases))


class RecordingWandbLogger(WandbLogger):
    """A ``WandbLogger`` with real base initialization and an injected fake run."""

    def __init__(self) -> None:
        """Initialize the real logger base with an injected fake experiment."""
        self._recording_experiment = RecordingWandbExperiment()
        super().__init__(project="pytest", experiment=cast(Any, self._recording_experiment))

    @property
    def used_artifacts(self) -> list[str]:
        """:returns: Artifact references recorded on the injected fake run."""
        return self._recording_experiment.used_artifacts
