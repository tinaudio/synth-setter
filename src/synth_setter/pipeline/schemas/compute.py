"""SkyPilot compute-option schema for the ``skypilot_launch/compute`` Hydra group.

``ComputeConfig`` is the validated form of one ``skypilot_launch/compute/*``
option; ``synth_setter.pipeline.compute_task.build_sky_task`` turns it into a
``sky.Task`` programmatically (no raw SkyPilot YAML involved).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CloudName = Literal["runpod", "vast", "oci", "kubernetes"]

# `sky local up` uses the kubernetes backend; cred bootstrap calls it "local".
_CLOUD_TO_PROVIDER: dict[str, str] = {
    "runpod": "runpod",
    "vast": "vast",
    "oci": "oci",
    "kubernetes": "local",
}


class ComputeResources(BaseModel):
    """One SkyPilot resources alternative — maps 1:1 onto ``sky.Resources``.

    A multi-key ``accelerators`` mapping is an any-of pool (SkyPilot picks the
    cheapest available entry), mirroring SkyPilot YAML semantics.

    .. attribute :: model_config

        Pydantic model config sentinel.

    .. attribute :: cloud

        SkyPilot cloud backend this entry targets.

    .. attribute :: accelerators

        GPU any-of pool as ``{accelerator: count}``.

    .. attribute :: instance_type

        Cloud-specific instance type (OCI shapes).

    .. attribute :: cpus

        CPU request in SkyPilot's ``"1+"`` grammar.

    .. attribute :: memory

        Memory request in GiB, SkyPilot's ``"4+"`` grammar.

    .. attribute :: disk_size

        Disk size in GiB; ``None`` defers to the backend default (the
        kubernetes backend rejects an explicit value).

    .. attribute :: use_spot

        Whether spot/interruptible instances are acceptable.

    .. attribute :: image_id

        Static image pin (``docker:<image>``); the launcher's per-launch
        ``worker_image`` pin overrides it.

    .. attribute :: kubernetes_pod_config

        Task-scoped kubernetes ``pod_config`` override, forwarded as
        ``sky.Resources(_cluster_config_overrides=...)``.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    cloud: CloudName
    accelerators: dict[str, int] | None = None
    instance_type: str | None = None
    cpus: str | None = None
    memory: str | None = None
    disk_size: int | None = None
    use_spot: bool = False
    image_id: str | None = None
    kubernetes_pod_config: dict[str, object] | None = None

    @model_validator(mode="after")
    def _pod_config_requires_kubernetes(self) -> ComputeResources:
        """Reject a pod_config override on a non-kubernetes cloud.

        :returns: Validated resources entry.
        :raises ValueError: ``kubernetes_pod_config`` is set with a
            non-kubernetes ``cloud``.
        """
        if self.kubernetes_pod_config is not None and self.cloud != "kubernetes":
            raise ValueError(
                f"kubernetes_pod_config is only valid with cloud=kubernetes, got {self.cloud!r}"
            )
        return self


class ComputeConfig(BaseModel):
    """One validated ``skypilot_launch/compute`` option.

    .. attribute :: model_config

        Pydantic model config sentinel.

    .. attribute :: name

        Human-readable option label (used in error messages and job metadata).

    .. attribute :: resources

        Resources alternatives; more than one entry is a SkyPilot ``any_of``.

    .. attribute :: mount_network_volume

        Worker-side mount path for the launch config's ``network_volume``.

    .. attribute :: image_delivery

        How the worker image reaches the pod: pinned into each resources
        entry's ``image_id``, or pulled by a sub-``docker run`` inside the run
        block (OCI — its backend rejects ``image_id: docker:<image>``).

    .. attribute :: setup_scripts

        Script filenames under ``configs/skypilot_launch/compute/scripts/``
        concatenated into the task's ``setup`` block.

    .. attribute :: run_wrapper

        Script whose ``${WORKER_CMD}`` sentinel receives the launcher's cmd.

    .. attribute :: run_script

        Full run block for debug options; the launcher's cmd is forbidden.

    .. attribute :: file_mounts

        SkyPilot ``file_mounts`` (``{remote: local}``) for debug probes.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    name: str
    resources: tuple[ComputeResources, ...] = Field(min_length=1)
    mount_network_volume: str | None = None
    image_delivery: Literal["resources-image-id", "docker-in-run"] = "resources-image-id"
    setup_scripts: tuple[str, ...] = ("worker-ready.sh",)
    run_wrapper: str | None = None
    run_script: str | None = None
    file_mounts: dict[str, str] | None = None

    @field_validator("resources", "setup_scripts", mode="before")
    @classmethod
    def _sequence_as_tuple(cls, v: object) -> object:
        """Coerce Hydra-composed lists into the tuple the strict field expects.

        :param v: Candidate sequence value pre-validation.
        :returns: ``v`` as a tuple when it arrived as a list, else unchanged.
        """
        return tuple(v) if isinstance(v, list) else v

    @model_validator(mode="after")
    def _run_fields_are_consistent(self) -> ComputeConfig:
        """Enforce run-block and image-delivery cross-field invariants.

        :returns: Validated compute option.
        :raises ValueError: ``run_script`` and ``run_wrapper`` are both set,
            ``docker-in-run`` lacks a ``run_wrapper``, or an OCI entry is
            combined with ``resources-image-id`` delivery.
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
        """Return the cred-bootstrap provider for this option.

        :returns: ``runpod`` / ``vast`` / ``oci`` / ``local``, keyed off the
            first resources entry (all entries share a cloud in practice).
        """
        return _CLOUD_TO_PROVIDER[self.resources[0].cloud]

    def requests_runpod(self) -> bool:
        """Report whether any resources entry targets RunPod.

        SkyPilot may satisfy the request with any listed alternative, so the
        balance preflight must consider them all.

        :returns: ``True`` when at least one entry names the RunPod cloud.
        """
        return any(entry.cloud == "runpod" for entry in self.resources)
