"""Tests for ``build_sky_task`` and the ``skypilot_launch/compute`` option loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from synth_setter.pipeline.compute_task import (
    build_sky_task,
    compute_option_names,
    load_compute_option,
    load_compute_script,
)
from synth_setter.pipeline.schemas.compute import ComputeConfig, ComputeResources
from synth_setter.resources import configs_dir


def _runpod_compute(**overrides: object) -> ComputeConfig:
    """Build a RunPod ComputeConfig with test defaults.

    :param **overrides: Field overrides applied on top of the defaults.
    :returns: Validated compute option.
    """
    kwargs: dict[str, object] = {
        "name": "test-runpod",
        "resources": (
            ComputeResources(cloud="runpod", accelerators={"RTX3090": 1}, disk_size=50),
        ),
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


class TestBuildSkyTaskRunBlock:
    """Cmd / run_wrapper / run_script resolution preserves legacy failure semantics."""

    def test_cmd_becomes_run_block(self) -> None:
        """Cmd becomes run block."""
        task = build_sky_task(_runpod_compute(), cmd="echo hello", worker_image="repo:tag")
        assert task.run == "echo hello"

    def test_run_script_with_cmd_raises(self) -> None:
        """Run script with cmd raises."""
        compute = _runpod_compute(run_script="debug-noop.sh")
        with pytest.raises(ValueError, match="cmd cannot be silently dropped"):
            build_sky_task(compute, cmd="echo hello", worker_image="repo:tag")

    def test_run_script_without_cmd_ships_script_verbatim(self) -> None:
        """Run script without cmd ships script verbatim."""
        compute = _runpod_compute(run_script="debug-noop.sh")
        task = build_sky_task(compute, cmd=None, worker_image="repo:tag")
        assert task.run is not None
        assert "skypilot-debug job done" in task.run

    def test_run_wrapper_substitutes_worker_cmd_sentinel(self) -> None:
        """Run wrapper substitutes worker cmd sentinel."""
        oci = ComputeResources(cloud="oci", instance_type="VM.Standard.E5.Flex$_2_8", disk_size=50)
        compute = ComputeConfig(
            name="oci-cpu",
            resources=(oci,),
            image_delivery="docker-in-run",
            setup_scripts=("oci-install-docker.sh",),
            run_wrapper="oci-docker-run.sh",
        )
        task = build_sky_task(compute, cmd="echo hello && exec foo", worker_image="repo:tag")
        assert task.run is not None
        assert "${WORKER_CMD}" not in task.run
        assert 'bash -c "echo hello && exec foo"' in task.run

    def test_run_wrapper_without_cmd_raises(self) -> None:
        """Run wrapper without cmd raises."""
        oci = ComputeResources(cloud="oci", instance_type="VM.Standard.E5.Flex$_2_8", disk_size=50)
        compute = ComputeConfig(
            name="oci-cpu",
            resources=(oci,),
            image_delivery="docker-in-run",
            run_wrapper="oci-docker-run.sh",
        )
        with pytest.raises(ValueError, match="cmd"):
            build_sky_task(compute, cmd=None, worker_image="repo:tag")

    def test_no_run_source_and_no_cmd_raises(self) -> None:
        """No run source and no cmd raises."""
        with pytest.raises(ValueError, match="cmd"):
            build_sky_task(_runpod_compute(), cmd=None, worker_image="repo:tag")


class TestBuildSkyTaskResources:
    """Resources construction matches SkyPilot's YAML any-of semantics."""

    def test_multi_key_accelerators_split_into_any_of_entries(self) -> None:
        """Multi key accelerators split into any of entries."""
        compute = _runpod_compute(
            resources=(
                ComputeResources(
                    cloud="runpod", accelerators={"RTX3090": 1, "A40": 1}, disk_size=50
                ),
            )
        )
        task = build_sky_task(compute, cmd="echo hi", worker_image="repo:tag")
        accels = sorted(str(res.accelerators) for res in task.resources)
        assert accels == ["{'A40': 1}", "{'RTX3090': 1}"]

    def test_image_pinned_as_docker_reference_on_every_entry(self) -> None:
        """Image pinned as docker reference on every entry."""
        task = build_sky_task(
            _runpod_compute(), cmd="echo hi", worker_image="tinaudio/synth-setter:abc"
        )
        assert [res.image_id for res in task.resources] == [
            {None: "docker:tinaudio/synth-setter:abc"}
        ]

    def test_oci_docker_in_run_entries_get_no_image_id(self) -> None:
        """Oci docker in run entries get no image id."""
        oci = ComputeResources(cloud="oci", instance_type="VM.Standard.E5.Flex$_2_8", disk_size=50)
        compute = ComputeConfig(
            name="oci-cpu",
            resources=(oci,),
            image_delivery="docker-in-run",
            run_wrapper="oci-docker-run.sh",
        )
        task = build_sky_task(compute, cmd="echo hi", worker_image="repo:tag")
        assert [res.image_id for res in task.resources] == [None]

    def test_no_worker_image_keeps_static_image_pin(self) -> None:
        """No worker image keeps static image pin."""
        compute = _runpod_compute(
            resources=(
                ComputeResources(
                    cloud="runpod",
                    accelerators={"RTX3090": 1},
                    disk_size=50,
                    image_id="docker:tinaudio/synth-setter:dev-snapshot",
                ),
            )
        )
        task = build_sky_task(compute, cmd="echo hi", worker_image=None)
        assert [res.image_id for res in task.resources] == [
            {None: "docker:tinaudio/synth-setter:dev-snapshot"}
        ]

    def test_kubernetes_pod_config_round_trips_as_cluster_config_override(self) -> None:
        """Kubernetes pod config round trips as cluster config override."""
        pod_config: dict[str, object] = {"spec": {"containers": [{"imagePullPolicy": "Never"}]}}
        compute = ComputeConfig(
            name="local-kind",
            resources=(
                ComputeResources(
                    cloud="kubernetes",
                    cpus="1+",
                    memory="4+",
                    kubernetes_pod_config=pod_config,
                ),
            ),
        )
        task = build_sky_task(compute, cmd="echo hi", worker_image="repo:tag")
        rendered = task.to_yaml_config()["resources"]
        assert rendered["_cluster_config_overrides"] == {"kubernetes": {"pod_config": pod_config}}


class TestBuildSkyTaskVolumesAndEnvs:
    """Network-volume both-or-neither and env forwarding."""

    def test_mount_with_volume_name_populates_task_volumes(self) -> None:
        """Mount with volume name populates task volumes."""
        compute = _runpod_compute(mount_network_volume="/workspace/network-volume")
        task = build_sky_task(
            compute,
            cmd="echo hi",
            worker_image="repo:tag",
            network_volume="ss-datasets-us-ca-2",
        )
        assert task.volumes == {"/workspace/network-volume": "ss-datasets-us-ca-2"}

    def test_mount_without_volume_name_raises(self) -> None:
        """Mount without volume name raises."""
        compute = _runpod_compute(mount_network_volume="/workspace/network-volume")
        with pytest.raises(ValueError, match="network_volume"):
            build_sky_task(compute, cmd="echo hi", worker_image="repo:tag")

    def test_volume_name_without_mount_raises(self) -> None:
        """Volume name without mount raises."""
        with pytest.raises(ValueError, match="mount_network_volume"):
            build_sky_task(
                _runpod_compute(), cmd="echo hi", worker_image="repo:tag", network_volume="vol-x"
            )

    def test_envs_are_forwarded_to_the_task(self) -> None:
        """Envs are forwarded to the task."""
        task = build_sky_task(
            _runpod_compute(),
            cmd="echo hi",
            worker_image="repo:tag",
            envs={"WANDB_API_KEY": "k"},
        )
        assert task.envs == {"WANDB_API_KEY": "k"}

    def test_setup_concatenates_scripts_in_order(self) -> None:
        """Setup concatenates scripts in order."""
        compute = _runpod_compute(setup_scripts=("worker-ready.sh", "operator-ssh.sh"))
        task = build_sky_task(compute, cmd="echo hi", worker_image="repo:tag")
        assert task.setup is not None
        assert task.setup.index("worker ready") < task.setup.index("OPERATOR_SSH_PUBKEYS_B64")

    def test_file_mounts_are_forwarded_to_the_task(self) -> None:
        """File mounts are forwarded to the task."""
        compute = _runpod_compute(file_mounts={"/remote/probe.txt": "src/some/local.yaml"})
        task = build_sky_task(compute, cmd="echo hi", worker_image="repo:tag")
        assert task.file_mounts == {"/remote/probe.txt": "src/some/local.yaml"}


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
        task = build_sky_task(compute, cmd="echo hi", worker_image="repo:tag")
        assert len(list(task.resources)) == 7
