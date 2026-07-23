"""Tests for the ``ComputeConfig`` / ``ComputeResources`` schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from synth_setter.pipeline.schemas.compute import ComputeConfig, ComputeResources


def _runpod_resources(**overrides: object) -> ComputeResources:
    """Build a RunPod resources entry with test defaults.

    :param **overrides: Field overrides applied on top of the defaults.
    :returns: Validated resources entry.
    """
    kwargs: dict[str, object] = {
        "cloud": "runpod",
        "accelerators": {"RTX3090": 1},
        "disk_size": 50,
    }
    kwargs.update(overrides)
    return ComputeResources(**kwargs)  # type: ignore[arg-type]


def _compute_config(**overrides: object) -> ComputeConfig:
    """Build a ComputeConfig with test defaults.

    :param **overrides: Field overrides applied on top of the defaults.
    :returns: Validated compute option.
    """
    kwargs: dict[str, object] = {
        "name": "test-option",
        "resources": (_runpod_resources(),),
    }
    kwargs.update(overrides)
    return ComputeConfig(**kwargs)  # type: ignore[arg-type]


class TestComputeResources:
    """Field validation for a single resources entry."""

    def test_minimal_runpod_entry_validates(self) -> None:
        """Minimal runpod entry validates."""
        entry = _runpod_resources()
        assert entry.cloud == "runpod"
        assert entry.accelerators == {"RTX3090": 1}
        assert entry.use_spot is False

    def test_unknown_cloud_rejected(self) -> None:
        """Unknown cloud rejected."""
        with pytest.raises(ValidationError, match="cloud"):
            _runpod_resources(cloud="aws")

    def test_extra_field_rejected(self) -> None:
        """Extra field rejected."""
        with pytest.raises(ValidationError, match="extra"):
            _runpod_resources(region="us-east")

    def test_pod_config_on_non_kubernetes_cloud_rejected(self) -> None:
        """Pod config on non kubernetes cloud rejected."""
        with pytest.raises(ValidationError, match="kubernetes_pod_config"):
            _runpod_resources(kubernetes_pod_config={"spec": {}})

    def test_pod_config_on_kubernetes_cloud_validates(self) -> None:
        """Pod config on kubernetes cloud validates."""
        entry = ComputeResources(
            cloud="kubernetes",
            cpus="1+",
            memory="4+",
            kubernetes_pod_config={"spec": {"containers": [{"imagePullPolicy": "Never"}]}},
        )
        assert entry.kubernetes_pod_config is not None

    def test_string_disk_size_rejected_in_strict_mode(self) -> None:
        """String disk size rejected in strict mode."""
        with pytest.raises(ValidationError, match="disk_size"):
            _runpod_resources(disk_size="50")


class TestComputeConfig:
    """Cross-field validation for a full compute option."""

    def test_minimal_config_validates_with_default_setup_scripts(self) -> None:
        """Minimal config validates with default setup scripts."""
        compute = _compute_config()
        assert compute.setup_scripts == ("worker-ready.sh",)
        assert compute.image_delivery == "resources-image-id"

    def test_empty_resources_rejected(self) -> None:
        """Empty resources rejected."""
        with pytest.raises(ValidationError, match="resources"):
            _compute_config(resources=())

    def test_run_script_and_run_wrapper_together_rejected(self) -> None:
        """Run script and run wrapper together rejected."""
        with pytest.raises(ValidationError, match="run_script"):
            _compute_config(run_script="debug-noop.sh", run_wrapper="oci-docker-run.sh")

    def test_docker_in_run_without_run_wrapper_rejected(self) -> None:
        """Docker in run without run wrapper rejected."""
        oci = ComputeResources(cloud="oci", instance_type="VM.Standard.E5.Flex$_2_8", disk_size=50)
        with pytest.raises(ValidationError, match="run_wrapper"):
            _compute_config(resources=(oci,), image_delivery="docker-in-run")

    def test_oci_with_resources_image_id_delivery_rejected(self) -> None:
        """Oci with resources image id delivery rejected."""
        oci = ComputeResources(cloud="oci", instance_type="VM.Standard.E5.Flex$_2_8", disk_size=50)
        with pytest.raises(ValidationError, match="docker-in-run"):
            _compute_config(resources=(oci,))

    def test_oci_with_docker_in_run_and_wrapper_validates(self) -> None:
        """Oci with docker in run and wrapper validates."""
        oci = ComputeResources(cloud="oci", instance_type="VM.Standard.E5.Flex$_2_8", disk_size=50)
        compute = _compute_config(
            resources=(oci,),
            image_delivery="docker-in-run",
            run_wrapper="oci-docker-run.sh",
        )
        assert compute.provider() == "oci"

    def test_nested_dict_input_validates(self) -> None:
        """Nested dict input validates."""
        compute = ComputeConfig(
            name="from-dict",
            resources=({"cloud": "vast", "accelerators": {"RTX3090": 1}, "disk_size": 50},),  # type: ignore[arg-type]
        )
        assert compute.resources[0].cloud == "vast"

    def test_extra_field_rejected(self) -> None:
        """Extra field rejected."""
        with pytest.raises(ValidationError, match="extra"):
            _compute_config(workdir="/somewhere")


class TestProviderHelpers:
    """``provider()`` / ``requests_runpod()`` replace YAML-dict provider sniffing."""

    @pytest.mark.parametrize(
        ("cloud", "expected"),
        [("runpod", "runpod"), ("vast", "vast"), ("kubernetes", "local")],
    )
    def test_provider_maps_first_entry_cloud(self, cloud: str, expected: str) -> None:
        """Provider maps first entry cloud.

        :param cloud: Parametrized cloud name under test.
        :param expected: Parametrized expected provider.
        """
        entry = ComputeResources(cloud=cloud, accelerators=None, cpus="1+")  # type: ignore[arg-type]
        assert _compute_config(resources=(entry,)).provider() == expected

    def test_requests_runpod_true_when_any_entry_targets_runpod(self) -> None:
        """Requests runpod true when any entry targets runpod."""
        vast = ComputeResources(cloud="vast", accelerators={"RTX3090": 1})
        assert _compute_config(resources=(vast, _runpod_resources())).requests_runpod() is True

    def test_requests_runpod_false_without_runpod_entries(self) -> None:
        """Requests runpod false without runpod entries."""
        vast = ComputeResources(cloud="vast", accelerators={"RTX3090": 1})
        assert _compute_config(resources=(vast,)).requests_runpod() is False
