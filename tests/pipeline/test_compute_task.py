"""Tests for SkyPilot task documents and compute-option loading."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import sky

from synth_setter.pipeline.compute_task import (
    build_task_doc,
    compute_option_names,
    load_compute_option,
    load_compute_script,
)
from synth_setter.pipeline.schemas.compute import ComputeConfig, ComputeResources
from synth_setter.resources import configs_dir


def _build_task(
    compute: ComputeConfig,
    *,
    cmd: str | None,
    network_volume: str | None = None,
) -> sky.Task:
    """Parse a generated task document through SkyPilot's public loader.

    :param compute: Validated compute option.
    :param cmd: Worker command for injected-command options.
    :param network_volume: Optional network-volume name.
    :returns: Parsed SkyPilot task.
    """
    return sky.Task.from_yaml_config(
        build_task_doc(compute, cmd=cmd, network_volume=network_volume)
    )


def _runpod_compute(**overrides: object) -> ComputeConfig:
    """Build a RunPod ComputeConfig with test defaults.

    :param **overrides: Field overrides applied on top of the defaults.
    :returns: Validated compute option.
    """
    kwargs: dict[str, object] = {
        "name": "test-runpod",
        "resources": [ComputeResources(cloud="runpod", accelerators={"RTX3090": 1}, disk_size=50)],
    }
    kwargs.update(overrides)
    return ComputeConfig(**kwargs)  # type: ignore[arg-type]


class TestLoadComputeScript:
    """Scripts ship as package data; the shebang/header comment lines are stripped."""

    def test_worker_ready_script_matches_legacy_setup_block(self) -> None:
        """Worker ready script matches legacy setup block."""
        text = load_compute_script("worker-ready.sh")
        assert text == 'set -euo pipefail\necho "synth-setter worker ready (host: $(hostname))"\n'

    def test_header_comments_are_stripped(self) -> None:
        """Header comments are stripped."""
        text = load_compute_script("operator-ssh.sh")
        assert text.startswith("set -euo pipefail\n")
        assert "#!/usr/bin/env bash" not in text

    def test_unknown_script_raises_file_not_found(self) -> None:
        """Unknown script raises file not found."""
        with pytest.raises(FileNotFoundError, match="no-such-script.sh"):
            load_compute_script("no-such-script.sh")


class TestBuildTaskDocRunBlock:
    """Cmd / run_wrapper / run_script resolution preserves legacy failure semantics."""

    def test_cmd_becomes_run_block(self) -> None:
        """Cmd becomes run block."""
        task = _build_task(_runpod_compute(), cmd="echo hello")
        assert task.run == "echo hello"

    def test_run_script_with_cmd_raises(self) -> None:
        """Run script with cmd raises."""
        compute = _runpod_compute(run_script="debug-noop.sh")
        with pytest.raises(ValueError, match="cmd cannot be silently dropped"):
            _build_task(compute, cmd="echo hello")

    def test_run_script_without_cmd_ships_script_verbatim(self) -> None:
        """Run script without cmd ships script verbatim."""
        compute = _runpod_compute(run_script="debug-noop.sh")
        task = _build_task(compute, cmd=None)
        assert task.run is not None
        assert "skypilot-debug job done" in task.run

    def test_run_wrapper_substitutes_worker_cmd_sentinel(self) -> None:
        """Run wrapper substitutes worker cmd sentinel."""
        oci = ComputeResources(cloud="oci", instance_type="VM.Standard.E5.Flex$_2_8", disk_size=50)
        compute = ComputeConfig(
            name="oci-cpu",
            resources=[oci],
            image_delivery="docker-in-run",
            setup_scripts=["oci-install-docker.sh"],
            run_wrapper="oci-docker-run.sh",
        )
        task = _build_task(compute, cmd="echo hello && exec foo")
        assert task.run is not None
        assert "${WORKER_CMD}" not in task.run
        assert 'bash -c "echo hello && exec foo"' in task.run

    def test_run_wrapper_without_cmd_raises(self) -> None:
        """Run wrapper without cmd raises."""
        oci = ComputeResources(cloud="oci", instance_type="VM.Standard.E5.Flex$_2_8", disk_size=50)
        compute = ComputeConfig(
            name="oci-cpu",
            resources=[oci],
            image_delivery="docker-in-run",
            run_wrapper="oci-docker-run.sh",
        )
        with pytest.raises(ValueError, match="cmd"):
            _build_task(compute, cmd=None)

    def test_no_run_source_and_no_cmd_raises(self) -> None:
        """No run source and no cmd raises."""
        with pytest.raises(ValueError, match="cmd"):
            _build_task(_runpod_compute(), cmd=None)


class TestBuildTaskDocResources:
    """Resources construction matches SkyPilot's YAML any-of semantics."""

    def test_multi_key_accelerators_split_into_any_of_entries(self) -> None:
        """Multi key accelerators split into any of entries."""
        compute = _runpod_compute(
            resources=[
                ComputeResources(
                    cloud="runpod", accelerators={"RTX3090": 1, "A40": 1}, disk_size=50
                )
            ]
        )
        task = _build_task(compute, cmd="echo hi")
        accels = sorted(str(res.accelerators) for res in task.resources)
        assert accels == ["{'A40': 1}", "{'RTX3090': 1}"]

    def test_oci_docker_in_run_entries_get_no_image_id(self) -> None:
        """Oci docker in run entries get no image id."""
        oci = ComputeResources(cloud="oci", instance_type="VM.Standard.E5.Flex$_2_8", disk_size=50)
        compute = ComputeConfig(
            name="oci-cpu",
            resources=[oci],
            image_delivery="docker-in-run",
            run_wrapper="oci-docker-run.sh",
        )
        task = _build_task(compute, cmd="echo hi")
        assert [res.image_id for res in task.resources] == [None]

    def test_kubernetes_pod_config_round_trips_as_cluster_config_override(self) -> None:
        """Kubernetes pod config round trips as cluster config override."""
        pod_config: dict[str, object] = {"spec": {"containers": [{"imagePullPolicy": "Never"}]}}
        compute = ComputeConfig(
            name="local-kind",
            config={"kubernetes": {"pod_config": pod_config}},
            resources=[ComputeResources(cloud="kubernetes", cpus="1+", memory="4+")],
        )
        task = _build_task(compute, cmd="echo hi")
        rendered = task.to_yaml_config()["resources"]
        assert rendered["_cluster_config_overrides"] == {"kubernetes": {"pod_config": pod_config}}


class TestBuildTaskDocVolumes:
    """Network-volume validation and task-document fields."""

    def test_mount_with_volume_name_populates_task_volumes(self) -> None:
        """Mount with volume name populates task volumes."""
        compute = _runpod_compute(mount_network_volume="/workspace/network-volume")
        task = _build_task(
            compute,
            cmd="echo hi",
            network_volume="ss-datasets-us-ca-2",
        )
        assert task.volumes == {"/workspace/network-volume": "ss-datasets-us-ca-2"}

    def test_mount_without_volume_name_raises(self) -> None:
        """Mount without volume name raises."""
        compute = _runpod_compute(mount_network_volume="/workspace/network-volume")
        with pytest.raises(ValueError, match="network_volume"):
            _build_task(compute, cmd="echo hi")

    def test_volume_name_without_mount_raises(self) -> None:
        """Volume name without mount raises."""
        with pytest.raises(ValueError, match="mount_network_volume"):
            _build_task(_runpod_compute(), cmd="echo hi", network_volume="vol-x")

    def test_setup_concatenates_scripts_in_order(self) -> None:
        """Setup concatenates scripts in order."""
        compute = _runpod_compute(setup_scripts=["worker-ready.sh", "operator-ssh.sh"])
        task = _build_task(compute, cmd="echo hi")
        assert task.setup is not None
        assert task.setup.index("worker ready") < task.setup.index("OPERATOR_SSH_PUBKEYS_B64")

    def test_file_mounts_are_forwarded_to_the_task(self) -> None:
        """File mounts are forwarded to the task."""
        compute = _runpod_compute(file_mounts={"/remote/probe.txt": "src/some/local.yaml"})
        task = _build_task(compute, cmd="echo hi")
        assert task.file_mounts == {"/remote/probe.txt": "src/some/local.yaml"}


def _task_fields_digest(fields: dict[str, object]) -> str:
    """Hash task fields after canonicalizing unordered resource alternatives.

    :param fields: Observable task fields from the constructor or native loader.
    :returns: Stable SHA-256 digest.
    """
    resources = fields["resources"]
    if isinstance(resources, dict) and "any_of" in resources:
        resources["any_of"] = sorted(
            resources["any_of"], key=lambda item: json.dumps(item, sort_keys=True)
        )
    payload = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def test_all_compute_options_match_constructor_oracle() -> None:
    """Native parsing preserves every checked-in option's observable task fields."""
    fixture_path = Path(__file__).parent / "fixtures" / "build_sky_task_oracle.json"
    expected = json.loads(fixture_path.read_text(encoding="utf-8"))

    actual = {}
    for option in compute_option_names():
        compute = load_compute_option(option)
        cmd = None if compute.run_script is not None else "echo oracle"
        volume = "oracle-volume" if compute.mount_network_volume is not None else None
        task = _build_task(compute, cmd=cmd, network_volume=volume)
        actual[option] = _task_fields_digest(
            {
                "resources": task.to_yaml_config()["resources"],
                "run": task.run,
                "setup": task.setup,
                "file_mounts": task.file_mounts,
                "volumes": task.volumes,
            }
        )

    assert actual == expected


def _all_compute_option_names() -> list[str]:
    """Discover every checked-in compute option name from the config tree.

    :returns: Sorted option names relative to the compute group.
    """
    root = Path(str(configs_dir())) / "skypilot_launch" / "compute"
    names = [
        str(path.relative_to(root)).removesuffix(".yaml")
        for path in root.rglob("*.yaml")
        if path.parent.name != "scripts"
    ]
    assert len(names) == 17
    return sorted(names)


class TestComputeOptionCompose:
    """Every checked-in compute option composes and validates as ComputeConfig."""

    @pytest.mark.parametrize("option", _all_compute_option_names())
    def test_option_composes_to_valid_compute_config(self, option: str) -> None:
        """Option composes to valid compute config.

        :param option: Parametrized compute option name.
        """
        compute = load_compute_option(option)
        assert isinstance(compute, ComputeConfig)
        for script in compute.setup_scripts:
            assert load_compute_script(script)
        if compute.run_script is not None:
            assert load_compute_script(compute.run_script)
        if compute.run_wrapper is not None:
            assert "${WORKER_CMD}" in load_compute_script(compute.run_wrapper)

    def test_compute_option_names_lists_all_options(self) -> None:
        """Compute option names lists all options."""
        assert compute_option_names() == _all_compute_option_names()

    @pytest.mark.parametrize(
        "debug_option",
        [
            "runpod/debug/noop",
            "runpod/debug/image-pull",
            "runpod/debug/spec-mount",
            "runpod/debug/headless",
            "runpod/debug/rclone",
            "runpod/debug/headless-rclone",
            "runpod/debug/pedalboard",
            "runpod/debug/launcher-minimal",
        ],
    )
    def test_debug_pool_equals_smoke_pool_exactly(self, debug_option: str) -> None:
        """Debug pool equals smoke pool exactly.

        :param debug_option: Parametrized debug option name.
        """
        smoke = load_compute_option("runpod/smoke")
        debug = load_compute_option(debug_option)
        assert debug.resources == smoke.resources
        assert debug.run_script is not None

    def test_unknown_option_raises_value_error(self) -> None:
        """Unknown option raises value error."""
        with pytest.raises(ValueError, match="no-such-option"):
            load_compute_option("runpod/no-such-option")

    def test_production_network_volume_options_mount_the_volume(self) -> None:
        """Production network volume options mount the volume."""
        for option in (
            "runpod/network-volume/training",
            "runpod/network-volume/training-hclass",
            "runpod/network-volume/staging",
        ):
            assert load_compute_option(option).mount_network_volume == "/workspace/network-volume"

    def test_smoke_option_builds_a_seven_gpu_any_of_task(self) -> None:
        """Smoke option builds a seven gpu any of task."""
        compute = load_compute_option("runpod/smoke")
        task = _build_task(compute, cmd="echo hi")
        assert len(list(task.resources)) == 7
