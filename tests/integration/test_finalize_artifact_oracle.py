"""Verify finalize artifacts against the ``surge/fake_oracle`` invariants.

Operator ask on PR #1202: "Write a helper to verify the metrics. You should
use the fake oracle model. See test_train.py for reference."

The Surge XT fake oracle's ``predict_step`` returns ``batch["params"]``
verbatim (see :mod:`synth_setter.models.surge_fake_oracle_module`). The
strongest invariant the oracle pins is therefore ``pred == target_params``
exactly — ``test_train.py::test_train_eval_surge_xt`` pins this at the
"oracle pred != target-params" assertion. Loss is exactly zero by
construction (``loss = 0.0 * net(mel_spec).sum()``); per-param MSE is
exactly zero for the same reason.

This helper exercises those invariants against the *finalized* dataset
artifacts (the train split files finalize uploaded to R2) — not synthetic
batches — so a regression in finalize that silently corrupts the
``param_array`` dataset surfaces here as a non-zero ``pred - target``
delta. Audio-metric bounds from the test_train.py oracle leg
(``mss < 15``, ``wmfcc < 30``, ``sot < 0.5``, ``rms > 0.95``) require
Surge XT VST headless rendering and live behind ``@pytest.mark.requires_vst``;
this CI helper runs the param-space invariants only.

Auto-skips when R2 is unreachable via :func:`r2_io.is_r2_reachable` —
matches the convention in :mod:`tests.integration.test_finalize_dataset_r2`
so the test is safe to collect on PRs without R2 secrets (fork PRs, local
dev runs).
"""

from __future__ import annotations

import io
import os
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from synth_setter.pipeline import r2_io
from tests.evaluation._oracle_helpers import build_oracle_module

pytestmark = [pytest.mark.integration_r2, pytest.mark.r2, pytest.mark.slow]

_FINALIZE_PREFIX_ENV = "FINALIZE_RUN_PREFIX"


def _finalize_prefix_from_env() -> str | None:
    """Read ``$FINALIZE_RUN_PREFIX`` (e.g. ``r2:intermediate-data/data/<task>/<run>/``).

    The CI workflow exports this so the helper targets the same run prefix
    ``rclone ls`` just pinned. When unset (local dev with no upstream
    generate / finalize run) the helper skips rather than fabricating a
    prefix that isn't on R2.

    :returns: The rclone-form prefix string, or ``None`` when unset.
    """
    return os.environ.get(_FINALIZE_PREFIX_ENV)


def _prefix_to_r2_uri(prefix: str, leaf: str) -> str:
    """Rewrite ``r2:<bucket>/<key>/`` + ``leaf`` → ``r2://<bucket>/<key>/<leaf>``.

    ``r2_io.download_to_path`` consumes the ``r2://`` scheme; the workflow
    exports the rclone-form ``r2:`` prefix because that is what ``rclone ls``
    accepts. The two forms differ only in the ``://`` vs ``:`` separator.

    :param prefix: Rclone-form prefix; must start with ``r2:`` and end with ``/``.
    :param leaf: Object name under the prefix (no leading slash).
    :returns: Full ``r2://<bucket>/<key>/<leaf>`` URI.
    :raises ValueError: ``prefix`` does not start with ``r2:`` or end with ``/``.
    """
    if not prefix.startswith("r2:") or not prefix.endswith("/"):
        raise ValueError(
            f"FINALIZE_RUN_PREFIX must be of the form 'r2:<bucket>/<key>/' (got {prefix!r})"
        )
    body = prefix[len("r2:") :]
    return f"r2://{body}{leaf}"


def _maybe_skip_when_no_r2() -> None:
    """Skip the test when R2 is unreachable or no run prefix is provided.

    Two independent skip conditions — the messages tell the operator which
    credential or env var is missing rather than ``test skipped`` with no
    reason.
    """
    if not r2_io.is_r2_reachable():
        pytest.skip("R2 not reachable (rclone not on PATH or rclone lsd r2: failed)")
    if _finalize_prefix_from_env() is None:
        pytest.skip(
            f"${_FINALIZE_PREFIX_ENV} not set; CI exports it after generate + finalize succeed."
        )


def _load_param_array_from_hdf5(local_h5: Path) -> np.ndarray:
    """Read the ``param_array`` dataset out of a finalized HDF5 split file.

    :param local_h5: Local path to a downloaded ``train.h5`` / ``val.h5`` / ``test.h5``.
    :returns: ``param_array`` as a float32 numpy array of shape ``(N, P)``.
    """
    import h5py

    with h5py.File(local_h5, "r") as f:
        dataset = f["param_array"]
        return np.asarray(dataset[...], dtype=np.float32)  # type: ignore[index]


def _load_param_array_from_wds_tar(local_tar: Path) -> np.ndarray:
    """Concatenate every ``param_array.npy`` member of a wds shard into a single array.

    ``synth_setter.data.vst.writers.save_wds_samples`` keys each tar record by
    ``f"{start_idx:08d}"`` and stores the batch's ``param_array`` as a single
    ``(B, P)`` payload (one record per render batch, not one record per row).
    Members are concatenated along axis 0 in sorted name order so the returned
    array's row order matches ``start_idx`` ascending.

    :param local_tar: Local path to a downloaded shard tarball.
    :returns: ``param_array`` rows concatenated into a ``(N, P)`` float32 array,
        where ``N`` is the sum of per-record batch sizes; ``(0, 0)`` when the
        shard carries no ``param_array.npy`` member.
    """
    rows: list[np.ndarray] = []
    with tarfile.open(local_tar, mode="r") as tar:
        members = sorted(
            (m for m in tar.getmembers() if m.name.endswith(".param_array.npy")),
            key=lambda m: m.name,
        )
        for member in members:
            fh = tar.extractfile(member)
            if fh is None:
                continue
            rows.append(np.load(io.BytesIO(fh.read())).astype(np.float32))
    return np.concatenate(rows, axis=0) if rows else np.zeros((0, 0), dtype=np.float32)


def _download_first_train_artifact(prefix: str, work_dir: Path) -> tuple[Path, str]:
    """Probe the finalized prefix for ``train.h5`` (hdf5) or the lowest-shard tar (wds).

    Tries ``train.h5`` first — that's the hdf5 layout finalize writes. When
    absent, falls back to listing the prefix and grabbing the first
    ``shard-*.tar`` (wds layout). Either way returns the local download path
    plus the format tag so the caller can branch on how to read params.

    :param prefix: Rclone-form prefix; must end with ``/``.
    :param work_dir: Local scratch dir for the download.
    :returns: ``(local_path, format)`` where ``format`` is ``"hdf5"`` or ``"wds"``.
    :raises FileNotFoundError: No recognized finalize artifact under ``prefix``.
    """
    train_h5_uri = _prefix_to_r2_uri(prefix, "train.h5")
    if r2_io.object_size(train_h5_uri) is not None:
        local = work_dir / "train.h5"
        r2_io.download_to_path(train_h5_uri, local)
        return local, "hdf5"
    import subprocess

    listing = subprocess.run(  # noqa: S603
        ["rclone", "lsf", prefix, "--include", "shard-*.tar"],  # noqa: S607
        capture_output=True,
        check=True,
        text=True,
    )
    candidates = sorted(line.strip() for line in listing.stdout.splitlines() if line.strip())
    if not candidates:
        raise FileNotFoundError(
            f"no finalize artifact under {prefix}: looked for train.h5 and shard-*.tar"
        )
    leaf = candidates[0]
    local = work_dir / leaf
    r2_io.download_to_path(_prefix_to_r2_uri(prefix, leaf), local)
    return local, "wds"


def test_finalize_train_split_passes_fake_oracle_invariants() -> None:
    """``surge/fake_oracle`` predict_step returns finalized params verbatim.

    Downloads the first finalize-written train artifact at
    ``$FINALIZE_RUN_PREFIX`` (``train.h5`` for the hdf5 row, the lowest-index
    ``shard-*.tar`` for the wds row), runs the oracle's ``predict_step`` /
    eval step over the loaded ``param_array``, and pins three invariants the
    oracle leg of ``tests/test_train.py`` already requires:

      1. ``predict_step`` returns ``batch["params"]`` bit-identically.
      2. ``per_param_mse`` is exactly zero (no float drift).
      3. ``training_step`` loss is exactly zero with a grad-bearing tensor
         (the oracle's ``0.0 * net(mel_spec).sum()`` construction).

    Audio-metric bounds (``mss < 15`` etc.) live behind ``requires_vst`` in
    ``test_train.py::test_train_eval_surge_xt`` — they need Surge XT
    rendering and are out of scope for this VST-free CI helper. An assertion
    failure here signals real corruption in the finalize pipeline, not a
    model regression.
    """
    _maybe_skip_when_no_r2()
    prefix = _finalize_prefix_from_env()
    assert prefix is not None

    with tempfile.TemporaryDirectory() as raw_work_dir:
        work_dir = Path(raw_work_dir)
        local_artifact, fmt = _download_first_train_artifact(prefix, work_dir)

        if fmt == "hdf5":
            param_array = _load_param_array_from_hdf5(local_artifact)
        else:
            param_array = _load_param_array_from_wds_tar(local_artifact)

    assert param_array.size > 0, (
        f"finalized {fmt} artifact at {prefix} carries no param_array rows"
    )
    assert param_array.ndim == 2, (
        f"expected param_array of shape (N, P); got {param_array.shape!r}"
    )

    num_samples, num_params = param_array.shape
    params_tensor = torch.from_numpy(param_array)
    mel_spec = torch.zeros(num_samples, 2, 4, 5)
    batch = {
        "params": params_tensor,
        "mel_spec": mel_spec,
        "audio": torch.zeros(num_samples, 2, 16),
    }

    module = build_oracle_module(num_params=num_params)
    preds, returned_batch = module.predict_step(batch, batch_idx=0)
    assert torch.equal(preds, params_tensor), (
        "oracle predict_step did not return batch['params'] verbatim — "
        "finalize may have corrupted the param_array column."
    )
    assert returned_batch is batch

    eval_out = module.validation_step(batch, batch_idx=0)
    assert eval_out["param_mse"].item() == 0.0, (
        f"oracle param_mse on finalized data must be exactly 0; got {eval_out['param_mse'].item()!r}"
    )
    assert torch.all(eval_out["per_param_mse"] == 0), (
        f"oracle per_param_mse on finalized data must be exactly 0; got {eval_out['per_param_mse'].tolist()!r}"
    )

    loss = module.training_step(batch, batch_idx=0)
    assert loss.item() == 0.0, (
        f"oracle training_step loss on finalized data must be exactly 0; got {loss.item()!r}"
    )
    assert loss.requires_grad, "oracle loss must carry grad for loss.backward()"
