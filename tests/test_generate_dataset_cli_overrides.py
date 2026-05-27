"""Parametrized subprocess smoke for the ``synth-setter-generate-dataset`` CLI.

Each parametrize case is a single string of Hydra-style overrides; the test
body shells out to the installed entrypoint and lets ``check=True`` raise on
any non-zero exit. The list is intentionally tiny — extend ``OVERRIDE_ARGS``
with whatever override combinations you want to pin.
"""

from __future__ import annotations

import os
import shlex
import subprocess

import pytest

_ENTRYPOINT = "synth-setter-generate-dataset"

# ``render/surge_xt.yaml`` hardcodes ``plugin_path``, so the env var only
# reaches the CLI when baked into the override string below. Matches the
# convention in ``tests/data/vst/test_preset_params.py`` and friends.
_PLUGIN_PATH = os.environ.get("SYNTH_SETTER_PLUGIN_PATH") or "plugins/Surge XT.vst3"

OVERRIDE_ARGS: list[str] = [
    # ``--help`` still walks the defaults list, so an ``experiment=`` override
    # is required even for the cheapest invocation.
    "experiment=generate_dataset/smoke-shard --help",
    f"experiment=generate_dataset/smoke-shard render.plugin_path={shlex.quote(_PLUGIN_PATH)}",
]


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.r2
@pytest.mark.requires_vst
@pytest.mark.parametrize("override_args", OVERRIDE_ARGS, ids=lambda s: s or "<empty>")
def test_generate_dataset_cli_accepts_overrides(override_args: str) -> None:
    """Invoke the CLI with ``override_args`` and assert it exits 0.

    :param override_args: Whitespace-separated override string passed straight
        through to the CLI after ``shlex.split``.
    """
    argv = [_ENTRYPOINT, *shlex.split(override_args)]
    subprocess.run(argv, check=True)  # noqa: S603 — argv built from in-test literals
