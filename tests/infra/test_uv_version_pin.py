"""Uv toolchain entry points stay aligned."""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parents[2]
UV_VERSION = "0.11.28"


def test_setup_uv_steps_pin_supported_version() -> None:
    """Every setup-uv step selects the supported exact release."""
    workflow_paths = [
        *sorted((REPO_ROOT / ".github" / "workflows").glob("*.y*ml")),
        *sorted((REPO_ROOT / ".github" / "actions").glob("*/action.y*ml")),
    ]

    setup_steps = []
    for workflow_path in workflow_paths:
        document = yaml.safe_load(workflow_path.read_text())
        jobs = document.get("jobs", {"composite": document})
        for job in jobs.values():
            for step in job.get("steps", []):
                if str(step.get("uses", "")).startswith("astral-sh/setup-uv@"):
                    setup_steps.append((workflow_path, step))

    assert setup_steps
    for workflow_path, step in setup_steps:
        assert step.get("with", {}).get("version") == UV_VERSION, workflow_path


def test_supported_bootstraps_pin_uv_version() -> None:
    """Local and image bootstrap paths select the supported exact release."""
    expected_pins = {
        "Makefile": f"https://astral.sh/uv/{UV_VERSION}/install.sh",
        "docker/ubuntu22_04/Dockerfile": f"ghcr.io/astral-sh/uv:{UV_VERSION}",
        "tart/macos.pkr.hcl": f'default     = "{UV_VERSION}"',
    }

    for relative_path, expected_pin in expected_pins.items():
        assert expected_pin in (REPO_ROOT / relative_path).read_text()
