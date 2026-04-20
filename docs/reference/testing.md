# Testing Primer

> **Code version**: `1bfeff0` (2026-04-20, `main`)
> **Scope**: what you need to read a test in this repo and write a new one that won't eat eight review comments. Not a complete testing philosophy — see `ml-test` skill for that.

______________________________________________________________________

## 1. Layout + invocation

Tests mirror the source tree — `tests/` shadows `src/` and `pipeline/`. Locations roughly:

- `tests/test_eval.py`, `tests/test_train.py` — end-to-end train/eval round-trips
- `tests/test_configs.py` — Hydra config composition + env-var interpolation
- `tests/test_instantiators.py`, `tests/test_datamodules.py` — unit-level checks of setup helpers
- `tests/helpers/` — shared test utilities (`RunIf`, `package_available`, `run_sh_command`)

Invocation:

| Command                                        | What it runs                                                                       |
| ---------------------------------------------- | ---------------------------------------------------------------------------------- |
| `make test`                                    | Quick tests — excludes `slow` and skips anything gated behind missing dependencies |
| `make test-full`                               | Everything — includes `slow` and GPU-gated tests                                   |
| `pytest -m "slow and gpu"`                     | Only the GPU end-to-end tests (what GPU CI runs)                                   |
| `pytest -m "not slow"`                         | Same as `make test` under the hood                                                 |
| `pytest tests/path/to/test_x.py::test_name -v` | One test, verbose                                                                  |

Markers registered in `pyproject.toml` (must be applied via `@pytest.mark.<name>`):
`slow`, `gpu`, `hypothesis`, `pipeline`, `benchmark`, `requires_vst`, `r2`, `docker_smoke`.

Strict markers are on — unknown marker names fail collection, so new markers must be added to `pyproject.toml` first.

______________________________________________________________________

## 2. Fixtures (`tests/conftest.py`)

Two canonical fixtures, both defined at `tests/conftest.py`. Use these rather than composing Hydra yourself.

### `cfg_train`

Per-function fixture wrapping `cfg_train_global`. Returns a `DictConfig` pre-composed from `configs/train.yaml` with test-friendly overrides already applied:

- `trainer.max_epochs=1`, `trainer.max_steps=-1` (epoch-bounded, not step-bounded)
- `trainer.limit_train_batches=0.01`, `trainer.limit_val_batches=0.1`, `trainer.limit_test_batches=0.1`
- `trainer.accelerator="cpu"`, `trainer.devices=1`
- `data.num_workers=0`, `data.pin_memory=False`
- `logger=None` (no W&B/TB in tests)
- `paths.output_dir` / `paths.log_dir` point at pytest's `tmp_path`

If your test needs a different data module, model, or trainer config, override inside an `open_dict(cfg_train):` block before calling `train(cfg_train)`.

### `cfg_eval`

Per-function fixture wrapping `cfg_eval_global`. Same treatment, composed from `configs/eval.yaml`:

- Same trainer defaults as `cfg_train` **except** `limit_val_batches` is **not** preset (only `limit_test_batches=0.1` is). See gotcha #3.
- `ckpt_path="."` as a placeholder — set it to the real checkpoint inside your test
- Defaults for `data`, `model`, `callbacks` come from `configs/eval.yaml` and **differ from** `cfg_train`'s defaults — if you want a train→eval round-trip, you must align them (gotcha #2)

Both fixtures clear the global Hydra state on teardown via `GlobalHydra.instance().clear()`.

______________________________________________________________________

## 3. The train → eval e2e pattern

Canonical template, lightly annotated. Use `tests/test_eval.py::test_train_eval` as the reference implementation when you write a new one.

```python
import os
from pathlib import Path

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from src.eval import evaluate
from src.train import train
from tests.helpers.run_if import RunIf


@pytest.mark.gpu
@RunIf(min_gpus=1)
@pytest.mark.slow
def test_train_<what>(tmp_path: Path, cfg_train: DictConfig, cfg_eval: DictConfig) -> None:
    assert str(tmp_path) == cfg_train.paths.output_dir == cfg_eval.paths.output_dir

    # 1. Override cfg_train for this run
    with open_dict(cfg_train):
        cfg_train.trainer.accelerator = "gpu"
        cfg_train.test = False                 # skip the post-train test phase if you don't need it

    # 2. Train; assert the checkpoint landed
    HydraConfig().set_config(cfg_train)
    train_metric_dict, _ = train(cfg_train)
    assert "last.ckpt" in os.listdir(tmp_path / "checkpoints")

    # 3. Align cfg_eval with cfg_train so the checkpoint loads cleanly
    with open_dict(cfg_eval):
        cfg_eval.trainer.accelerator = "gpu"
        cfg_eval.trainer.limit_val_batches = cfg_train.trainer.limit_val_batches  # parity (gotcha #3)
        cfg_eval.data      = cfg_train.data                                        # ckpt's DataModule
        cfg_eval.model     = cfg_train.model                                       # ckpt's LightningModule
        cfg_eval.callbacks = cfg_train.callbacks                                   # matches train hooks
        cfg_eval.ckpt_path = str(tmp_path / "checkpoints" / "last.ckpt")
        cfg_eval.mode = "validate"             # or "test", or "predict"

    # 4. Evaluate; assert metrics
    HydraConfig().set_config(cfg_eval)
    eval_metric_dict, _ = evaluate(cfg_eval)
    assert eval_metric_dict["val/loss"] < float("inf")
    # Optional parity check: train's in-fit val vs. standalone eval's val
    assert abs(train_metric_dict["val/loss"].item() - eval_metric_dict["val/loss"].item()) < 0.001
```

The three-marker stack (`@pytest.mark.gpu`, `@RunIf(min_gpus=1)`, `@pytest.mark.slow`) is what makes GPU CI pick the test up and CPU-only `make test` skip it. Drop all three for CPU-only unit tests.

______________________________________________________________________

## 4. Gotchas

1. **DataModule `setup(stage)` must cover every stage you invoke.** Lightning passes one of `{"fit", "validate", "test", "predict"}` depending on which Trainer method you call. A `setup()` that only handles `"fit"` will silently build the wrong dataloader (or none at all) for other stages. See Lightning's [DataModule docs](https://lightning.ai/docs/pytorch/stable/data/datamodule.html) and `src/data/ksin_datamodule.py` for the canonical three-branch pattern (`fit` → train, `{fit, validate}` → val, `{test, predict}` → test).

2. **`cfg_eval.data/model/callbacks` default to different values than `cfg_train`'s.** `configs/eval.yaml` composes surge-oriented defaults; `configs/train.yaml` composes ksin defaults. If you want to round-trip a checkpoint produced by `train(cfg_train)`, align `cfg_eval.data`, `cfg_eval.model`, and `cfg_eval.callbacks` with `cfg_train`'s before loading the checkpoint — otherwise Lightning refuses to restore state into a differently-shaped LightningModule.

3. **Pin `cfg_eval.trainer.limit_val_batches = cfg_train.trainer.limit_val_batches` for train→val parity checks.** The fixture presets `limit_val_batches=0.1` on `cfg_train` but **not** on `cfg_eval`. Without the pin, `train()`'s in-fit val loop runs on 10% of batches while `evaluate()` runs on 100%, and any parity assertion becomes noise.

4. **Three-marker stack for GPU e2e tests.** `@pytest.mark.slow`, `@pytest.mark.gpu`, and `@RunIf(min_gpus=1)` are all needed:

   - `slow` + `gpu` markers tell `make test` to skip the test locally and tell GPU CI to include it via `-m "slow and gpu"`
   - `RunIf(min_gpus=1)` is a runtime guard that skips on hosts without a CUDA GPU (protects against mismatches between marker and environment)

5. **`weights_only=False` when loading a checkpoint for eval.** PyTorch 2.6 tightened the default for `torch.load` to `weights_only=True`, which refuses to unpickle the Lightning checkpoint metadata. `src/eval.py` passes `weights_only=False` explicitly to `trainer.test/validate/predict(ckpt_path=...)` — any new standalone loader code needs the same flag.

6. **If you add a new `mode=<x>` dispatch to `src/eval.py`, audit every affected DataModule's `setup()`.** `src/eval.py` dispatches on `cfg.mode` to `trainer.test/validate/predict`, each of which passes a different `stage` to `DataModule.setup()`. Adding a mode without verifying the DataModules handle its stage is how gotcha #1 bites.

______________________________________________________________________

## 5. Pointers

- **Conftest**: [`tests/conftest.py`](../../tests/conftest.py) — source of `cfg_train` / `cfg_eval` fixtures, all preset overrides visible in one file
- **Reference e2e test**: [`tests/test_eval.py::test_train_eval`](../../tests/test_eval.py) — cleanest example of the open_dict → train → ckpt → open_dict → evaluate pattern
- **Markers config**: [`pyproject.toml`](../../pyproject.toml) `[tool.pytest.ini_options]` — source of truth for registered marker names
- **GPU skipping helper**: [`tests/helpers/run_if.py`](../../tests/helpers/run_if.py) — ported from Lightning's own test harness
- **Lightning DataModule**: <https://lightning.ai/docs/pytorch/stable/data/datamodule.html> — stage semantics, setup/teardown contract
- **Lightning Trainer methods**: <https://lightning.ai/docs/pytorch/stable/common/trainer.html> — which method triggers which stage
- **W&B in tests**: tests set `logger=None` so nothing hits W&B. See [`docs/reference/wandb-integration.md`](wandb-integration.md) for runtime logger behavior.
