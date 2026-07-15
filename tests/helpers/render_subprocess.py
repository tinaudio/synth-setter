"""Shared ``_check_call_streamed`` side effects for render-orchestration tests.

The state-based generate-entrypoint tests in
``tests/pipeline/entrypoints/test_generate_dataset_unit.py`` patch the single
``synth_setter.cli.generate_dataset._check_call_streamed`` seam that the
renderer *and* the rclone shard upload both go through; this module is the one
dispatch contract those call sites share instead of each re-deriving it
(see #1354).
"""

from __future__ import annotations

from pathlib import Path

from synth_setter.pipeline.subprocess_stream import check_call_streamed
from tests.helpers.subprocess_args import find_script_index

# rclone passthrough: bound from the pipeline module, so it bypasses the patched
# cli seam (no recursion) yet runs the real streamed runner on the real binary.
REAL_CHECK_CALL = check_call_streamed


def materialize_shard(args: list[str]) -> None:
    """``_check_call_streamed`` side effect that writes the Lance shard the renderer promises.

    Mirrors the production contract: ``generate_vst_dataset.py`` exits 0 only
    after writing the ``.lance`` dataset directory to its output path, so a test
    without this side effect would trip the ``shard_path.is_dir()`` check in
    ``_render_and_upload_shard``. The ``_versions/`` subdir is populated because
    the resume skip-probe and the ordered directory upload both target it.

    :param args: argv list passed to the patched ``_check_call_streamed``.
    """
    script_idx = find_script_index(args)
    output_dir = Path(args[script_idx + 1])
    (output_dir / "data").mkdir(parents=True, exist_ok=True)
    (output_dir / "data" / "shard.bin").write_bytes(b"\x00")
    (output_dir / "_versions").mkdir(parents=True, exist_ok=True)
    (output_dir / "_versions" / "1.manifest").write_bytes(b"\x00")


def materialize_or_passthrough_rclone(args: list[str]) -> None:
    """Simulate the renderer but let rclone uploads hit the real binary.

    The renderer and the rclone shard upload share the patched
    ``_check_call_streamed`` seam; dispatching on ``args[0]`` distinguishes them
    so rclone copies actually land a file on the fake-local remote while
    renderer calls only materialize the shard.

    :param args: argv list passed to the patched ``_check_call_streamed``.
    """
    if args and args[0] == "rclone":
        REAL_CHECK_CALL(args)
        return
    materialize_shard(args)
