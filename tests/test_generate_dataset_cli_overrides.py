"""Parametrized subprocess smoke for the ``synth-setter-generate-dataset`` CLI.

Each parametrize case is a single string of Hydra-style overrides; the test
body shells out to the installed entrypoint and lets ``check=True`` raise on
any non-zero exit. The list is intentionally tiny — extend ``OVERRIDE_ARGS``
with whatever override combinations you want to pin.
"""

from __future__ import annotations

import shlex
import subprocess

import pytest

from tests._vst import PLUGIN_PATH

_ENTRYPOINT = "synth-setter-generate-dataset"

# ``render/surge_xt.yaml`` hardcodes ``plugin_path``, so the path only reaches
# the CLI when baked into the override string below.
OVERRIDE_ARGS: list[str] = [
    # ``--help`` still walks the defaults list, so an ``experiment=`` override
    # is required even for the cheapest invocation.
    "experiment=generate_dataset/smoke-shard --help",
    f"experiment=generate_dataset/smoke-shard render.plugin_path={shlex.quote(PLUGIN_PATH)}",
]


# Bounded so a hung VST init or stalled R2 upload can't block CI indefinitely.
_CLI_TIMEOUT_S = 600


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.r2
@pytest.mark.requires_vst
@pytest.mark.parametrize("override_args", OVERRIDE_ARGS)
def test_generate_dataset_cli_accepts_overrides(override_args: str) -> None:
    """Assert the CLI exits 0 for each override string (shlex-split into argv tail).

    :param override_args: argv tail after ``shlex.split``; may mix Hydra overrides and flags.
    """
    argv = [_ENTRYPOINT, *shlex.split(override_args)]
    # capture_output so CalledProcessError surfaces the actual error on CI.
    subprocess.run(  # noqa: S603 — argv built from in-test literals
        argv, check=True, capture_output=True, text=True, timeout=_CLI_TIMEOUT_S
    )
