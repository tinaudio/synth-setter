"""Validated SkyPilot task configuration for Hydra compute options."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CloudName = Literal["runpod", "vast", "oci", "kubernetes"]

# `sky local up` uses the kubernetes backend; cred bootstrap calls it "local".
_CLOUD_TO_PROVIDER: dict[str, str] = {
    "runpod": "runpod",
    "vast": "vast",
    "oci": "oci",
    "kubernetes": "local",
}


class ComputeResources(BaseModel):
    """Validate one mapping accepted by SkyPilot's ``resources`` YAML key.

    .. attribute :: model_config

        Pydantic model configuration.

    .. attribute :: cloud

        SkyPilot cloud backend.

    .. attribute :: accelerators

        Accelerator alternatives and counts.

    .. attribute :: instance_type

        Provider-specific instance type.

    .. attribute :: cpus

        SkyPilot CPU request.

    .. attribute :: memory

        SkyPilot memory request.

    .. attribute :: disk_size

        Disk size in GiB.

    .. attribute :: use_spot

        Whether interruptible capacity is allowed.

    .. attribute :: image_id

        Optional static image identifier.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    cloud: CloudName
    accelerators: dict[str, int] | None = None
    instance_type: str | None = None
    cpus: str | None = None
    memory: str | None = None
    disk_size: int | None = None
    use_spot: bool = False
    image_id: str | None = None


class ComputeConfig(BaseModel):
    """Validate one Hydra compute option before task-document construction.

    .. attribute :: model_config

        Pydantic model configuration.

    .. attribute :: name

        Task name.

    .. attribute :: resources

        Non-empty resource alternatives.

    .. attribute :: config

        Public SkyPilot task-level config overrides.

    .. attribute :: mount_network_volume

        Worker mount path for the selected network volume.

    .. attribute :: image_delivery

        Worker-image delivery strategy.

    .. attribute :: setup_scripts

        Packaged scripts concatenated into ``setup``.

    .. attribute :: run_wrapper

        Packaged script containing the worker-command sentinel.

    .. attribute :: run_script

        Packaged standalone run script.

    .. attribute :: file_mounts

        SkyPilot file mounts.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    name: str
    resources: list[ComputeResources] = Field(min_length=1)
    config: dict[str, object] | None = None
    mount_network_volume: str | None = None
    image_delivery: Literal["resources-image-id", "docker-in-run"] = "resources-image-id"
    setup_scripts: list[str] = Field(default_factory=lambda: ["worker-ready.sh"])
    run_wrapper: str | None = None
    run_script: str | None = None
    file_mounts: dict[str, str] | None = None

    @model_validator(mode="after")
    def _pod_config_requires_kubernetes(self) -> ComputeConfig:
        """Reject task-level Kubernetes overrides for non-Kubernetes resources.

        :returns: Validated compute option.
        :raises ValueError: A config override targets non-Kubernetes resources.
        """
        if self.config is not None and any(
            entry.cloud != "kubernetes" for entry in self.resources
        ):
            raise ValueError("config overrides are only valid with cloud=kubernetes")
        return self

    @model_validator(mode="after")
    def _run_fields_are_consistent(self) -> ComputeConfig:
        """Enforce run-block and image-delivery invariants.

        :returns: Validated compute option.
        :raises ValueError: Run sources conflict or image delivery is invalid.
        """
        if self.run_script is not None and self.run_wrapper is not None:
            raise ValueError("run_script and run_wrapper are mutually exclusive")
        if self.image_delivery == "docker-in-run" and self.run_wrapper is None:
            raise ValueError("image_delivery=docker-in-run requires a run_wrapper script")
        if self.image_delivery == "resources-image-id" and any(
            entry.cloud == "oci" for entry in self.resources
        ):
            raise ValueError(
                "OCI rejects image_id=docker:<image>; use image_delivery=docker-in-run"
            )
        return self

    def provider(self) -> str:
        """Return the credential-bootstrap provider for the first alternative.

        :returns: Provider name accepted by credential bootstrap.
        """
        return _CLOUD_TO_PROVIDER[self.resources[0].cloud]

    def requests_runpod(self) -> bool:
        """Report whether any resource alternative targets RunPod.

        :returns: Whether RunPod account preflight is required.
        """
        return any(entry.cloud == "runpod" for entry in self.resources)
