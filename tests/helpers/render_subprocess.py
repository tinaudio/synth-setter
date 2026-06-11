"""Shared ``_check_call_streamed`` side effects for render-orchestration tests.

The state-based generate-entrypoint tests in
``tests/pipeline/entrypoints/test_generate_dataset_unit.py`` patch the single
``synth_setter.cli.generate_dataset._check_call_streamed`` seam that the
renderer *and* the rclone shard upload both go through; this module is the one
dispatch contract those ~11 call sites share instead of each re-deriving it
(see #1354).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from tests.helpers.subprocess_args import find_script_index

# Real binary for the rclone passthrough; tests patch ``_check_call_streamed``,
# never this symbol.
REAL_CHECK_CALL = subprocess.check_call


def materialize_shard(args: list[str]) -> None:
    """``_check_call_streamed`` side effect that writes the shard the renderer promises.

    Mirrors the production contract: ``generate_vst_dataset.py`` exits 0 only
    after writing the HDF5 to its output path, so a test without this side
    effect would trip the ``shard_path.is_file()`` check in
    ``_render_and_upload_shard``.

    :param args: argv list passed to the patched ``_check_call_streamed``.
    """
    script_idx = find_script_index(args)
    output_file = Path(args[script_idx + 1])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_bytes(b"")


def materialize_or_passthrough_rclone(args: list[str]) -> None:
    """Simulate the renderer but let rclone uploads hit the real binary.

    The renderer and the rclone shard upload share the patched
    ``_check_call_streamed`` seam; dispatching on ``args[0]`` distinguishes them
    so rclone copies actually land a file on the fake-local remote while
    renderer calls only materialize the shard.

    :param args: argv list passed to the patched ``_check_call_streamed``.
    """
    if args and args[0] == "rclone":
        REAL_CHECK_CALL(args)  # noqa: S603 — test-only passthrough
        return
    materialize_shard(args)
