"""Production-path flow-sketch RIR evaluation against real Cloudflare R2."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from synth_setter.pipeline import r2_io

pytestmark = [pytest.mark.gpu, pytest.mark.integration_r2, pytest.mark.r2, pytest.mark.slow]

_R2_BUCKET = "experiments"
_CLI_TIMEOUT_SECONDS = 1_800


def _unique_test_prefix() -> str:
    """Return a purge-safe evaluation prefix owned by this invocation.

    :returns: Unique R2 key prefix.
    """
    run_id = os.environ.get("GITHUB_RUN_ID", "local")
    run_attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "0")
    return f"ci-flow-sketch-rir/{run_id}/{run_attempt}/{uuid.uuid4().hex[:8]}"


def test_flow_sketch_rir_eval_cli_real_checkpoint_publishes_five_rows(tmp_path: Path) -> None:
    """Evaluate five real MIT RIRs and consume the immutable published attempt.

    :param tmp_path: Isolated local output and downloaded publication roots.
    """
    if not r2_io.is_r2_reachable():
        pytest.skip("R2 not reachable (rclone missing, credentials missing, or rclone lsd failed)")

    prefix = _unique_test_prefix()
    publication_uri = f"r2://{_R2_BUCKET}/{prefix}"
    output_dir = tmp_path / "output"
    command = [
        str(Path(sys.executable).with_name("synth-setter-eval")),
        "experiment=pyfdn/eval_flow_sketch_rir_mit_ir_survey",
        "logger=csv",
        "hydra.job.chdir=false",
        f"paths.output_dir={output_dir}",
        f"hydra.run.dir={output_dir}",
        f"evaluation.upload_output_dir_uri={publication_uri}",
    ]

    try:
        result = subprocess.run(  # noqa: S603 — installed CLI and fixed trusted arguments
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=_CLI_TIMEOUT_SECONDS,
        )
        assert result.returncode == 0, result.stderr

        pointer_path = tmp_path / "latest.json"
        r2_io.download_to_path(f"{publication_uri}/latest.json", pointer_path)
        payload_uri = json.loads(pointer_path.read_text())["payload_uri"]
        published = tmp_path / "published"
        r2_io.download_dir_no_overwrite(payload_uri, published)

        with (published / "metrics" / "metrics.csv").open(newline="") as handle:
            assert len(list(csv.DictReader(handle))) == 5
        rendered_audio, _ = sf.read(published / "audio" / "sample_4" / "pred.wav")
        assert np.isfinite(rendered_audio).all()
        assert np.count_nonzero(rendered_audio) > 0
    finally:
        r2_io.purge_prefix(_R2_BUCKET, f"{prefix}/")
