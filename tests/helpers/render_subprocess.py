"""Shared ``subprocess.check_call`` side effects for render-orchestration tests.

The state-based generate-entrypoint tests and the parallel-render integration
test both patch the single ``subprocess.check_call`` symbol that the renderer
*and* the rclone shard upload go through. They need one dispatch contract so the
two lanes don't drift into incompatible subprocess-patching styles — extracting
it here is that single source of truth (see #1354).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from tests.helpers.subprocess_args import find_script_index

# Captured at import — before any test patches ``subprocess.check_call`` — so the
# rclone passthrough reaches the real binary instead of recursing through the
# patch the production code shares with the renderer.
REAL_CHECK_CALL = subprocess.check_call


def materialize_shard(args: list[str]) -> int:
    """``check_call`` side effect that writes the shard file the renderer promises.

    Mirrors the production contract: ``generate_vst_dataset.py`` exits 0 only
    after writing the HDF5 to its output path, so a test without this side
    effect would trip the ``shard_path.is_file()`` check in
    ``_render_and_upload_shard``.

    :param args: argv list passed to the patched ``subprocess.check_call``.
    :returns: 0 after creating the expected (empty) shard file.
    """
    script_idx = find_script_index(args)
    output_file = Path(args[script_idx + 1])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_bytes(b"")
    return 0


def materialize_or_passthrough_rclone(args: list[str]) -> int:
    """Simulate the renderer but let rclone uploads hit the real binary.

    The renderer and the rclone shard upload share the patched
    ``subprocess.check_call`` symbol; dispatching on ``args[0]`` distinguishes
    them so rclone copies actually land a file on the fake-local remote while
    renderer calls only materialize the shard.

    :param args: argv list passed to the patched ``subprocess.check_call``.
    :returns: 0 on renderer simulation; rclone's exit code on the real subprocess.
    """
    if args and args[0] == "rclone":
        return REAL_CHECK_CALL(args)  # noqa: S603 — test-only passthrough
    return materialize_shard(args)
