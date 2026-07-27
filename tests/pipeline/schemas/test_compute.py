"""Tests for the strict compute-option schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from synth_setter.pipeline.schemas.compute import ComputeConfig, ComputeResources


def _runpod_resources(**overrides: object) -> ComputeResources:
    """Build a RunPod resource entry with test defaults.

    :param **overrides: Resource field overrides.
    :returns: Validated resource entry.
    """
    kwargs: dict[str, object] = {
        "cloud": "runpod",
        "accelerators": {"RTX3090": 1},
        "disk_size": 50,
    }
    kwargs.update(overrides)
    return ComputeResources(**kwargs)  # type: ignore[arg-type]


def _compute_config(**overrides: object) -> ComputeConfig:
    """Build a compute option with test defaults.

    :param **overrides: Compute-option field overrides.
    :returns: Validated compute option.
    """
    kwargs: dict[str, object] = {
        "name": "test-option",
        "resources": [_runpod_resources()],
    }
    kwargs.update(overrides)
    return ComputeConfig(**kwargs)  # type: ignore[arg-type]


class TestComputeResources:
    """Validate one SkyPilot resources mapping."""

    def test_minimal_runpod_entry_validates(self) -> None:
        """Accept a typed RunPod resource request."""
        entry = _runpod_resources()
        assert entry.accelerators == {"RTX3090": 1}
        assert entry.use_spot is False

    def test_unknown_cloud_rejected(self) -> None:
        """Reject clouds outside the supported provider set."""
        with pytest.raises(ValidationError, match="cloud"):
            _runpod_resources(cloud="aws")

    def test_extra_field_rejected(self) -> None:
        """Reject fields SkyPilot options do not support."""
        with pytest.raises(ValidationError, match="extra"):
            _runpod_resources(region="us-east")

    def test_string_disk_size_rejected_in_strict_mode(self) -> None:
        """Keep numeric resource fields strict at the Hydra boundary."""
        with pytest.raises(ValidationError, match="disk_size"):
            _runpod_resources(disk_size="50")


class TestComputeConfig:
    """Validate option-level and cross-field invariants."""

    def test_minimal_config_uses_list_defaults(self) -> None:
        """Use Hydra-compatible lists without coercion validators."""
        compute = _compute_config()
        assert compute.setup_scripts == ["worker-ready.sh"]
        assert isinstance(compute.resources, list)

    def test_tuple_resources_rejected_in_strict_mode(self) -> None:
        """Require the same list shape emitted by Hydra."""
        with pytest.raises(ValidationError, match="resources"):
            _compute_config(resources=(_runpod_resources(),))

    def test_empty_resources_rejected(self) -> None:
        """Require at least one resource alternative."""
        with pytest.raises(ValidationError, match="resources"):
            _compute_config(resources=[])

    def test_run_script_and_run_wrapper_together_rejected(self) -> None:
        """Reject ambiguous run-block sources."""
        with pytest.raises(ValidationError, match="run_script"):
            _compute_config(run_script="debug-noop.sh", run_wrapper="oci-docker-run.sh")

    def test_docker_in_run_without_run_wrapper_rejected(self) -> None:
        """Require the wrapper that starts the nested OCI container."""
        oci = ComputeResources(cloud="oci", instance_type="VM.Standard.E5.Flex$_2_8")
        with pytest.raises(ValidationError, match="run_wrapper"):
            _compute_config(resources=[oci], image_delivery="docker-in-run")

    def test_oci_with_resources_image_id_delivery_rejected(self) -> None:
        """Reject OCI's unsupported resources image pinning mode."""
        oci = ComputeResources(cloud="oci", instance_type="VM.Standard.E5.Flex$_2_8")
        with pytest.raises(ValidationError, match="docker-in-run"):
            _compute_config(resources=[oci])

    def test_oci_with_docker_in_run_and_wrapper_validates(self) -> None:
        """Accept OCI nested-Docker delivery."""
        oci = ComputeResources(cloud="oci", instance_type="VM.Standard.E5.Flex$_2_8")
        compute = _compute_config(
            resources=[oci],
            image_delivery="docker-in-run",
            run_wrapper="oci-docker-run.sh",
        )
        assert compute.provider() == "oci"

    def test_nested_dict_input_validates(self) -> None:
        """Validate Hydra's nested dictionaries into typed resources."""
        compute = ComputeConfig(
            name="from-dict",
            resources=[{"cloud": "vast", "accelerators": {"RTX3090": 1}}],  # type: ignore[list-item]
        )
        assert compute.resources[0].cloud == "vast"

    def test_kubernetes_config_on_non_kubernetes_cloud_rejected(self) -> None:
        """Keep task-scoped pod overrides on Kubernetes options."""
        with pytest.raises(ValidationError, match="cloud=kubernetes"):
            _compute_config(config={"kubernetes": {"pod_config": {"spec": {}}}})

    def test_kubernetes_config_on_kubernetes_cloud_validates(self) -> None:
        """Accept SkyPilot's public task-level config key for Kubernetes."""
        compute = _compute_config(
            resources=[ComputeResources(cloud="kubernetes", cpus="1+")],
            config={"kubernetes": {"pod_config": {"spec": {}}}},
        )
        assert compute.config is not None

    def test_extra_field_rejected(self) -> None:
        """Reject unknown task-option fields."""
        with pytest.raises(ValidationError, match="extra"):
            _compute_config(workdir="/somewhere")


class TestProviderHelpers:
    """Map typed resource alternatives to launcher preflights."""

    @pytest.mark.parametrize(
        ("cloud", "expected"),
        [("runpod", "runpod"), ("vast", "vast"), ("kubernetes", "local")],
    )
    def test_provider_maps_first_entry_cloud(self, cloud: str, expected: str) -> None:
        """Map the selected compute cloud to its credential provider.

        :param cloud: Supported SkyPilot cloud name.
        :param expected: Expected credential-bootstrap provider.
        """
        entry = ComputeResources(cloud=cloud, cpus="1+")  # type: ignore[arg-type]
        assert _compute_config(resources=[entry]).provider() == expected

    def test_requests_runpod_true_when_any_entry_targets_runpod(self) -> None:
        """Preflight RunPod when any alternative can select it."""
        vast = ComputeResources(cloud="vast", accelerators={"RTX3090": 1})
        assert _compute_config(resources=[vast, _runpod_resources()]).requests_runpod() is True

    def test_requests_runpod_false_without_runpod_entries(self) -> None:
        """Skip RunPod preflight when no alternative uses it."""
        vast = ComputeResources(cloud="vast", accelerators={"RTX3090": 1})
        assert _compute_config(resources=[vast]).requests_runpod() is False
