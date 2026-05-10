# Testing Primer

> **Scope**: enough to read a test in this repo and write a new one without eight review comments. Not a complete testing philosophy — see the `ml-test` skill for that.
>
> This primer deliberately **points at source files rather than echoing their contents**. Specifics (marker names, Makefile flags, fixture presets, CI selectors) drift fast; open the linked source to see today's truth.

______________________________________________________________________

## 1. What lives under `tests/`

Browse [`tests/`](../../tests) for the current layout. The tree has several **distinct categories**, not all of which use the same patterns:

| Category                                          | Where                                                                                                                              | Style                                                                                  | Typical markers                                               |
| ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| Hydra-config validation                           | [`tests/test_configs.py`](../../tests/test_configs.py)                                                                             | Uses `cfg_train` / `cfg_eval` fixtures; light assertions + `hydra.utils.instantiate()` | (none by default)                                             |
| Hydra+Lightning **E2E** train/eval                | [`tests/test_eval.py`](../../tests/test_eval.py), [`tests/test_train.py`](../../tests/test_train.py)                               | Uses fixtures; round-trips `train()` → checkpoint → `evaluate()`                       | `@pytest.mark.gpu`, `@pytest.mark.slow`, `@RunIf(min_gpus=1)` |
| Hydra sweeps                                      | [`tests/test_sweeps.py`](../../tests/test_sweeps.py)                                                                               | Invokes `src/train.py` as a subprocess via `run_sh_command` helper                     | `gpu`, `slow`                                                 |
| Pure-Python unit tests                            | [`tests/test_logging_utils.py`](../../tests/test_logging_utils.py), [`tests/test_datamodules.py`](../../tests/test_datamodules.py) | No Hydra fixtures; direct module/class tests                                           | (usually none)                                                |
| Property-based tests                              | [`tests/test_properties.py`](../../tests/test_properties.py)                                                                       | Hypothesis (`@given`, `@settings`); no fixtures                                        | `hypothesis`, `slow`                                          |
| Performance benchmarks                            | [`tests/test_benchmarks.py`](../../tests/test_benchmarks.py)                                                                       | `pytest-benchmark`'s `benchmark` fixture                                               | `benchmark`, `slow`                                           |
| Pipeline tests (schemas, CI scripts, entrypoints) | [`tests/pipeline/`](../../tests/pipeline) — has its own [`conftest.py`](../../tests/pipeline/conftest.py)                          | Pydantic-style unit tests; no Lightning involvement                                    | (none)                                                        |
| Script tests                                      | [`tests/scripts/`](../../tests/scripts)                                                                                            | Direct-import tests for utilities under `scripts/`                                     | (none)                                                        |
| Docker image smoke tests                          | [`tests/docker/test_smoke.py`](../../tests/docker/test_smoke.py)                                                                   | Runs inside the built container image                                                  | `docker_smoke`                                                |
| VST integration tests                             | [`tests/data/vst/test_preset_params.py`](../../tests/data/vst/test_preset_params.py)                                               | Requires a Surge XT VST3 binary on disk                                                | `requires_vst`                                                |
| Test helpers                                      | [`tests/helpers/`](../../tests/helpers) — `RunIf`, `run_sh_command`, `package_available`                                           | Not tests themselves — import from `tests.helpers.<name>`                              | —                                                             |

Two top-level conftests: [`tests/conftest.py`](../../tests/conftest.py) (Hydra `cfg_train` / `cfg_eval` fixtures) and [`tests/pipeline/conftest.py`](../../tests/pipeline/conftest.py) (pipeline-specific fixtures). Pytest resolves parent conftests, so the Hydra fixtures are reachable under `tests/pipeline/` too; pipeline tests typically don't use them and lean on their own fixtures.

______________________________________________________________________

## 2. Running tests

All common selectors are defined as Makefile targets — read [`Makefile`](../../Makefile) for the exact `pytest` flags each invokes (they evolve; don't memorize):

- `make test-fast` — quick CPU-only suite. Excludes slow, gpu, mps, requires_vst.
- `make test-full-cpu` — all CPU tests (slow + requires_vst included; gpu/mps excluded). Linux: bootstraps Xvfb.
- `make test-full-gpu` — GPU + CPU tests on a host with a CUDA GPU. Serial — exclusive device access. Linux: bootstraps Xvfb.
- `make test-full-mps` — MPS + CPU tests on a host with Apple silicon. Serial — exclusive device access.
- `make test-vst-cpu` — VST-only suite (requires_vst, slow included; gpu/mps excluded). Linux: bootstraps Xvfb.
- `pytest tests/path/to/test_x.py::test_name -v` — one test, for iteration.

CI selectors live in [`.github/workflows/`](../../.github/workflows):

- [`test.yml`](../../.github/workflows/test.yml) — fast CPU tests on every PR (`-m "not slow and not gpu and not mps"`).
- [`test-gpu.yml`](../../.github/workflows/test-gpu.yml) — GPU-marked tests on a GPU runner. Twice-weekly cron + manual dispatch.
- [`test-mps.yml`](../../.github/workflows/test-mps.yml) — MPS-marked tests on a `macos-latest` (Apple Silicon) runner. Triggered on push to `main` **and** on PRs that touch `src/` or `configs/` — the closest thing to pre-submit coverage for slow Surge tests, since the macOS runner is large enough to host the FFN forward pass without OOM.
- [`cpu-slow.yml`](../../.github/workflows/cpu-slow.yml) — slow CPU-only suite (`-m "slow and not gpu and not mps and not requires_vst"`), post-merge on `main`. **Post-merge by design**: PyTorch CPU forward passes OOM the standard 2-core PR runner. The lane is sized to avoid that (see the workflow's `runs-on:` for the current label) and runs after merge so PR feedback isn't gated on it. PR-time coverage of these tests comes from `test-mps.yml`. VST-marked tests live in [`test-vst-slow.yml`](../../.github/workflows/test-vst-slow.yml), which runs the Surge XT suite inside the project's Docker dev image. On post-merge failure, this workflow auto-opens a `ci-automation` Bug ticket to `@ktinubu` (deduped by title).
- [`test-conda.yml`](../../.github/workflows/test-conda.yml) — single conda-env run (micromamba from `environment.yaml`) on `ubuntu-latest`; covers the non-slow suite under the locked conda deps.
- [`nightly.yml`](../../.github/workflows/nightly.yml) — scheduled full `pytest` run on CPU (`ubuntu-latest`); no marker filter, so GPU-gated tests skip via `RunIf`.

**Coverage strategy in one sentence:** `[cpu]` parametrizations of slow tests run post-merge on the large runner (`cpu-slow.yml`); `[mps]` parametrizations run pre-submit on the macOS runner (`test-mps.yml`); `[gpu]` parametrizations run twice-weekly on the GPU runner (`test-gpu.yml`). A regression in the cpu path is caught after merge, not before — accepted because the cost of running CPU PyTorch on every PR is OOM failures, not just minutes.

CI and `make` selectors are **not identical** — CI may use different marker combinations to partition work across runners. Source of truth is always the file, not this doc.

**Markers** are registered in [`pyproject.toml`](../../pyproject.toml) under `[tool.pytest.ini_options].markers`. The file lists each marker's purpose. Strict markers are on — unknown marker names fail collection, so new markers must be added to `pyproject.toml` first.

______________________________________________________________________

## 3. Which shape fits your test?

| Goal                                            | Pattern to copy from                                     | Fixtures                            | Markers                            |
| ----------------------------------------------- | -------------------------------------------------------- | ----------------------------------- | ---------------------------------- |
| Assert a Hydra config composes and instantiates | `tests/test_configs.py::test_train_config`               | `cfg_train` / `cfg_eval`            | —                                  |
| E2E `train → ckpt → evaluate` round-trip        | `tests/test_eval.py::test_train_eval`                    | `cfg_train`, `cfg_eval`, `tmp_path` | `gpu`, `slow`, `RunIf(min_gpus=1)` |
| Unit-test a pure helper / utility               | `tests/test_logging_utils.py`                            | —                                   | —                                  |
| Hypothesis property test                        | `tests/test_properties.py`                               | —                                   | `hypothesis`, `slow`               |
| Benchmark a hot path                            | `tests/test_benchmarks.py::test_config_resolution_speed` | `benchmark` (from pytest-benchmark) | `benchmark`, `slow`                |
| Pydantic schema / pipeline logic                | `tests/pipeline/test_schemas/test_dataset_spec.py`       | pipeline conftest                   | —                                  |
| CLI script behavior                             | `tests/scripts/test_r2_shard_report.py`                  | —                                   | —                                  |
| VST-dependent integration                       | `tests/data/vst/test_preset_params.py`                   | —                                   | `requires_vst`                     |

Most categories don't need the `cfg_train` / `cfg_eval` fixtures. They're specifically for Hydra-composed code paths.

______________________________________________________________________

## 4. `cfg_train` / `cfg_eval` fixtures (`tests/conftest.py`)

Only needed for tests that exercise Hydra-composed configs (test_configs, test_train, test_eval, test_benchmarks, etc.).

Both defined in [`tests/conftest.py`](../../tests/conftest.py) as package-scoped `*_global` fixtures wrapped by function-scoped fixtures that inject `tmp_path`. Each `*_global` fixture composes the corresponding entry-point YAML with explicit `data=` / `model=` / `trainer=` overrides at compose time, then applies test-friendly tweaks via an `open_dict(cfg):` block. Read both blocks for today's presets — they change as the fixtures evolve.

`cfg_train_global` and `cfg_eval_global` compose with the **same** `data=ksin model=ffn trainer=cpu` overrides, and dataset shape is pinned via integer `train_val_test_sizes=[2,2,2]` rather than fractional `limit_*_batches`. A train→eval round-trip no longer needs to copy `data` / `model` / `callbacks` from one config to the other.

Both fixtures clear global Hydra state on teardown via `GlobalHydra.instance().clear()`.

A third fixture group — `cfg_surge_xt_global` / `cfg_surge_xt` / `cfg_surge_xt_eval` — follows the same pattern for Surge XT FFN tests (`tests/test_train.py::test_train_surge_xt_*`). It composes `train.yaml` with `experiment=surge/ffn_full`, points at the 5-sample dataset generated on demand by the `surge_xt_smoke_datasets` fixture, and bakes in the smoke-test trainer knobs. `cfg_surge_xt_global` is parametrized over both `accelerator` and `param_spec_name` (indirect fixtures); see [`tests/conftest.py`](../../tests/conftest.py) for the current parameter sets and the shared `_build_surge_xt_smoke_cfg` helper. `cfg_surge_xt_eval` is **not** built by composing `eval.yaml` — it's a copy of the train-side fixture (`cfg_surge_xt_global`) with `mode="predict"` set and `ckpt_path` pointed at the checkpoint the matching `cfg_surge_xt` run will write under the same `tmp_path`. This guarantees the evaluator sees the exact `data` / `model` shape that produced the checkpoint, with no `eval.yaml` / `train.yaml` reconciliation step. Read [`tests/conftest.py`](../../tests/conftest.py) for the current field-level overrides.

______________________________________________________________________

## 5. The train → eval E2E template

Reference implementation: [`tests/test_eval.py::test_train_eval`](../../tests/test_eval.py). The shape is three phases: override `cfg_train`, train + assert checkpoint, point `cfg_eval` at the checkpoint and evaluate. Because `cfg_train` and `cfg_eval` are composed with the same `data` / `model` / `trainer` overrides in `conftest.py`, no manual alignment is needed.

```python
import math
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
def test_train_e2e(tmp_path: Path, cfg_train: DictConfig, cfg_eval: DictConfig) -> None:  # rename `_e2e` to describe your case
    assert str(tmp_path) == cfg_train.paths.output_dir == cfg_eval.paths.output_dir

    # 1. Override cfg_train for this run
    with open_dict(cfg_train):
        cfg_train.trainer.accelerator = "gpu"
        cfg_train.test = False                 # skip post-train test phase if not needed

    # 2. Train; assert the checkpoint landed
    HydraConfig().set_config(cfg_train)
    train_metric_dict, _ = train(cfg_train)
    assert "last.ckpt" in os.listdir(tmp_path / "checkpoints")

    # 3. Point cfg_eval at the checkpoint and evaluate
    with open_dict(cfg_eval):
        cfg_eval.trainer.accelerator = "gpu"
        cfg_eval.ckpt_path = str(tmp_path / "checkpoints" / "last.ckpt")
        cfg_eval.mode = "validate"             # or "test", or "predict"

    HydraConfig().set_config(cfg_eval)
    eval_metric_dict, _ = evaluate(cfg_eval)
    # `math.isfinite` rejects +inf, -inf, and NaN. `< float("inf")` silently
    # accepts -inf — safe for non-negative losses like MSE but a copy-paste
    # hazard for losses that can go negative (log-likelihood, reward, ...).
    assert math.isfinite(eval_metric_dict["val/loss"].item())
    # Optional parity check: train's in-fit val vs. standalone eval's val
    assert abs(train_metric_dict["val/loss"].item() - eval_metric_dict["val/loss"].item()) < 0.001
```

The three-marker stack (`gpu` + `slow` + `RunIf`) exists because each does a different job — the `pytest.mark` stack lets the Makefile / CI workflow select tests, `RunIf` is a runtime guard that skips when no CUDA device is present. Drop all three for CPU-only unit tests.

Non-E2E categories (unit, property-based, benchmark, pipeline) use **simpler** shapes — don't over-apply this template to them. Start from a sibling test in the same file.

______________________________________________________________________

## 6. Gotchas

1. **DataModule `setup(stage)` must cover every stage you invoke.** Lightning passes one of `{"fit", "validate", "test", "predict"}` depending on which Trainer method runs. A `setup()` that only handles `"fit"` silently builds the wrong (or no) dataloader for the others. See Lightning's [DataModule docs](https://lightning.ai/docs/pytorch/stable/data/datamodule.html) for the contract, and [`src/data/ksin_datamodule.py`](../../src/data/ksin_datamodule.py) for the canonical three-branch pattern in this repo.

2. **`configs/train.yaml` and `configs/eval.yaml` require explicit `data=` / `model=`.** Both entry points use `???` for `data` and `model`, so Hydra fails fast if either is missing. Production runs pass them via an `experiment=` config; tests pass them at compose time inside `conftest.py`. There is no fallback to a researcher-local default.

3. **GPU tests use a three-marker stack.** `@pytest.mark.gpu`, `@pytest.mark.slow`, `@RunIf(min_gpus=1)` each do distinct things. The CI selector for GPU tests lives in [`.github/workflows/test-gpu.yml`](../../.github/workflows/test-gpu.yml); local `make test-fast` and `make test-full-gpu` filters live in the Makefile. If the CI filter changes, the docs don't need updating — the code does.

4. **`weights_only=False` when loading a checkpoint for eval.** PyTorch 2.6 tightened `torch.load`'s default to `weights_only=True`, which refuses Lightning checkpoint metadata. `src/eval.py` passes `weights_only=False` explicitly to `trainer.test/validate/predict(ckpt_path=...)`. New standalone loader code needs the same.

5. **Adding a new `cfg.mode=<x>` dispatch?** Audit every DataModule's `setup()` for that stage. `src/eval.py` routes `mode` to `trainer.test/validate/predict`, each passing a different `stage` string. Gotcha #1 bites if a DataModule doesn't handle the new stage.

______________________________________________________________________

## 7. Pointers

- [`tests/conftest.py`](../../tests/conftest.py) — the Hydra fixtures, including all preset overrides
- [`tests/pipeline/conftest.py`](../../tests/pipeline/conftest.py) — pipeline-test fixtures (separate tree, separate fixtures)
- [`tests/helpers/`](../../tests/helpers) — `RunIf`, `run_sh_command`, `package_available`
- [`tests/test_eval.py::test_train_eval`](../../tests/test_eval.py) — canonical E2E template
- [`Makefile`](../../Makefile) — `test`, `test-full`, and friends
- [`pyproject.toml`](../../pyproject.toml) — registered markers + pytest config
- [`.github/workflows/`](../../.github/workflows) — which CI job runs which markers
- Lightning [DataModule](https://lightning.ai/docs/pytorch/stable/data/datamodule.html) and [Trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html) docs — stage semantics and the `fit/validate/test/predict` contract
- [`docs/reference/wandb-integration.md`](wandb-integration.md) — tests disable the logger (`cfg.logger=None`); this covers the runtime behavior
