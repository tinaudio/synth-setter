# Testing Primer

> **Scope**: enough to read a test in this repo and write a new one without eight review comments. Not a complete testing philosophy — see the `ml-test` skill for that.
>
> This primer deliberately **points at source files rather than echoing their contents**. Specifics (marker names, Makefile flags, fixture presets, CI selectors) drift fast; open the linked source to see today's truth.

______________________________________________________________________

## 1. What lives under `tests/`

Browse [`tests/`](../../tests) for the current layout. The tree has several **distinct categories**, not all of which use the same patterns:

| Category                                          | Where                                                                                                                                          | Style                                                                                                                    | Typical markers                                               |
| ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------- |
| Hydra-config validation                           | [`tests/test_configs.py`](../../tests/test_configs.py), [`tests/test_generate_dataset_shards.py`](../../tests/test_generate_dataset_shards.py) | Uses `cfg_train` / `cfg_eval` / `cfg_dataset` fixtures; light assertions + `hydra.utils.instantiate()` / `spec_from_cfg` | (none by default)                                             |
| Hydra+Lightning **E2E** train/eval                | [`tests/test_eval.py`](../../tests/test_eval.py), [`tests/test_train.py`](../../tests/test_train.py)                                           | Uses fixtures; round-trips `train()` → checkpoint → `evaluate()`                                                         | `@pytest.mark.gpu`, `@pytest.mark.slow`, `@RunIf(min_gpus=1)` |
| Hydra sweeps                                      | [`tests/test_sweeps.py`](../../tests/test_sweeps.py)                                                                                           | Invokes `src/synth_setter/cli/train.py` as a subprocess via `run_sh_command` helper                                      | `gpu`, `slow`                                                 |
| Pure-Python unit tests                            | [`tests/test_logging_utils.py`](../../tests/test_logging_utils.py), [`tests/test_datamodules.py`](../../tests/test_datamodules.py)             | No Hydra fixtures; direct module/class tests                                                                             | (usually none)                                                |
| Property-based tests                              | [`tests/test_properties.py`](../../tests/test_properties.py)                                                                                   | Hypothesis (`@given`, `@settings`); no fixtures                                                                          | `hypothesis`, `slow`                                          |
| Performance benchmarks                            | [`tests/test_benchmarks.py`](../../tests/test_benchmarks.py)                                                                                   | `pytest-benchmark`'s `benchmark` fixture                                                                                 | `benchmark`, `slow`                                           |
| Pipeline tests (schemas, CI scripts, entrypoints) | [`tests/pipeline/`](../../tests/pipeline) — has its own [`conftest.py`](../../tests/pipeline/conftest.py)                                      | Pydantic-style unit tests; no Lightning involvement                                                                      | (none)                                                        |
| Script tests                                      | [`tests/scripts/`](../../tests/scripts)                                                                                                        | Direct-import tests for utilities under `scripts/`                                                                       | (none)                                                        |
| Docker image smoke tests                          | [`tests/docker/test_smoke.py`](../../tests/docker/test_smoke.py)                                                                               | Runs inside the built container image                                                                                    | `docker_smoke`                                                |
| VST integration tests                             | [`tests/data/vst/test_preset_params.py`](../../tests/data/vst/test_preset_params.py)                                                           | Requires a Surge XT VST3 binary on disk                                                                                  | `requires_vst`                                                |
| Devcontainer + GHA invariants                     | [`tests/infra/`](../../tests/infra) — has its own [`conftest.py`](../../tests/infra/conftest.py)                                               | Static checks on `.devcontainer/`, `Dockerfile`, workflow YAML; opt-in subprocess timing                                 | `infra`                                                       |
| End-to-end launcher integration                   | [`tests/integration/test_local_launcher_roundtrip.py`](../../tests/integration/test_local_launcher_roundtrip.py)                               | Real-R2 launcher round-trip with the VST renderer subprocess stubbed                                                     | `integration_r2`, `slow`                                      |
| Cross-platform shard dispatch                     | [`tests/integration/test_parallel_shard_dispatch.py`](../../tests/integration/test_parallel_shard_dispatch.py)                                 | Verifies the ThreadPoolExecutor dispatch path on every OS; renderer fully stubbed                                        | (none — runs in the default fast suite)                       |
| Test helpers                                      | [`tests/helpers/`](../../tests/helpers)                                                                                                        | Not tests themselves — import from `tests.helpers.<name>`                                                                | —                                                             |

Multiple conftests live under `tests/`; pytest resolves parent conftests, but each subtree carries its own fixtures: [`tests/conftest.py`](../../tests/conftest.py) (Hydra `cfg_train` / `cfg_eval` / `cfg_dataset` / `cfg_surge_xt`), [`tests/pipeline/conftest.py`](../../tests/pipeline/conftest.py) (pipeline-specific), [`tests/infra/conftest.py`](../../tests/infra/conftest.py) (intentionally minimal — only stdlib imports), and [`tests/integration/conftest.py`](../../tests/integration/conftest.py) (thin re-export of `fake_r2_remote` from `tests/pipeline/conftest.py`). The infra suite's own conftest pulls in no heavy deps, but pytest still walks up to the top-level `tests/conftest.py` (torch/h5py/Hydra) during normal collection. To skip the parent and keep the infra suite stdlib-only, run [`make test-infra`](../../Makefile), which invokes `pytest tests/infra/ --confcutdir=tests/infra`.

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
- [`test-mps.yml`](../../.github/workflows/test-mps.yml) — MPS-marked tests on a `macos-latest` (Apple Silicon) runner. Triggered on push to `main` **and** on PRs that touch `src/` or `src/synth_setter/configs/` — the closest thing to pre-submit coverage for slow Surge tests, since the macOS runner is large enough to host the Surge XT smoke tests (`test-mps-fake-oracle.yaml` / `test-mps-ffn.yaml`) without OOM.
- [`cpu-slow.yml`](../../.github/workflows/cpu-slow.yml) — slow CPU-only suite (`-m "slow and not gpu and not mps and not requires_vst"`), post-merge on `main`. **Post-merge by design**: PyTorch CPU forward passes OOM the standard 2-core PR runner. The lane is sized to avoid that (see the workflow's `runs-on:` for the current label) and runs after merge so PR feedback isn't gated on it. PR-time coverage of these tests comes from `test-mps.yml`. VST-marked tests live in [`test-vst-slow.yml`](../../.github/workflows/test-vst-slow.yml), which runs the Surge XT suite inside the project's Docker dev image. On post-merge failure, this workflow auto-opens a `ci-automation` Bug ticket to `@ktinubu` (deduped by title).
- [`test-local-launcher-roundtrip.yml`](../../.github/workflows/test-local-launcher-roundtrip.yml) — same-repo PR-only fast (`~8–10 min`) launcher round-trip lane (`-m integration_r2`). Hits real R2 with the VST renderer subprocess stubbed; no kind, no sky.launch, no dev-snapshot image pull. PR-tier alternative to `test-dataset-generation.yml`'s kind matrix.
- [`test-conda.yml`](../../.github/workflows/test-conda.yml) — single conda-env run (micromamba from `environment.yaml`) on `ubuntu-latest`; covers the non-slow suite under the locked conda deps.
- [`nightly.yml`](../../.github/workflows/nightly.yml) — scheduled full `pytest` run on CPU (`ubuntu-latest`); excludes `gpu`/`mps`/`requires_vst` markers (no CUDA/MPS device and no Surge XT VST3 on the runner — those suites run on dev-snapshot via sibling workflows, see [#1279](https://github.com/tinaudio/synth-setter/issues/1279)).

**Coverage strategy in one sentence:** `[cpu]` parametrizations of slow tests run post-merge on the large runner (`cpu-slow.yml`); `[mps]` parametrizations run pre-submit on the macOS runner (`test-mps.yml`); `[gpu]` parametrizations run twice-weekly on the GPU runner (`test-gpu.yml`). A regression in the cpu path is caught after merge, not before — accepted because the cost of running CPU PyTorch on every PR is OOM failures, not just minutes.

CI and `make` selectors are **not identical** — CI may use different marker combinations to partition work across runners. Source of truth is always the file, not this doc.

**Markers** are registered in [`pyproject.toml`](../../pyproject.toml) under `[tool.pytest.ini_options].markers`. The file lists each marker's purpose. Strict markers are on — unknown marker names fail collection, so new markers must be added to `pyproject.toml` first.

**Test order and seeding are randomized.** `pytest-randomly` is on by default; each run shuffles test order and reseeds `random` / `numpy.random` / `torch`. To reproduce a flake, copy the `Using --randomly-seed=NNN` line printed at session start and pass it back via `-p randomly --randomly-seed=NNN`; disable shuffling with `-p no:randomly`. The on-demand deflake harness ([`deflake-mps.yml`](../../.github/workflows/deflake-mps.yml) / `make deflake TEST=<node-id> [COUNT=50]`) reruns a single node N times and uploads failed iterations' `tmp_path` as artifacts — see [#1260](https://github.com/tinaudio/synth-setter/issues/1260).

______________________________________________________________________

## 3. Which shape fits your test?

| Goal                                                 | Pattern to copy from                                                                             | Fixtures                                                                               | Markers                                        |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------- | ---------------------------------------------- |
| Assert a Hydra config composes and instantiates      | `tests/test_configs.py::test_train_config`                                                       | `cfg_train` / `cfg_eval`                                                               | —                                              |
| Assert a dataset Hydra config composes and validates | `tests/test_generate_dataset_shards.py::test_cfg_dataset_composes_and_validates_as_dataset_spec` | `cfg_dataset`                                                                          | —                                              |
| E2E `from_hydra(cfg_dataset)` shard render → real R2 | `tests/test_generate_dataset_shards.py::test_generate_dataset_renders_shards_to_r2`              | `cfg_dataset`, `monkeypatch`                                                           | `integration_r2`, `r2`, `requires_vst`, `slow` |
| E2E `train → ckpt → evaluate` round-trip             | `tests/test_eval.py::test_train_eval`                                                            | `cfg_train`, `cfg_eval`, `tmp_path`                                                    | `gpu`, `slow`, `RunIf(min_gpus=1)`             |
| Unit-test a pure helper / utility                    | `tests/test_logging_utils.py`                                                                    | —                                                                                      | —                                              |
| Hypothesis property test                             | `tests/test_properties.py`                                                                       | —                                                                                      | `hypothesis`, `slow`                           |
| Benchmark a hot path                                 | `tests/test_benchmarks.py::test_config_resolution_speed`                                         | `benchmark` (from pytest-benchmark)                                                    | `benchmark`, `slow`                            |
| Pydantic schema / pipeline logic                     | `tests/pipeline/schemas/test_dataset_spec.py`                                                    | pipeline conftest                                                                      | —                                              |
| CLI script behavior                                  | `tests/scripts/test_r2_shard_report.py`                                                          | —                                                                                      | —                                              |
| VST-dependent integration                            | `tests/data/vst/test_preset_params.py`                                                           | —                                                                                      | `requires_vst`                                 |
| Invariant of devcontainer/workflow config            | `tests/infra/test_devcontainer_attached_mode.py`                                                 | `devcontainer_json_paths`, `post_create_script`, etc. (from `tests/infra/conftest.py`) | `infra`                                        |

Most categories don't need the `cfg_train` / `cfg_eval` / `cfg_dataset` / `cfg_surge_xt` fixtures. They're specifically for Hydra-composed code paths.

______________________________________________________________________

## 4. Hydra cfg fixtures (`tests/conftest.py`)

Only needed for tests that exercise Hydra-composed configs (test_configs, test_train, test_eval, test_benchmarks, test_generate_dataset_shards, etc.).

Both defined in [`tests/conftest.py`](../../tests/conftest.py) as package-scoped `*_global` fixtures wrapped by function-scoped fixtures that inject `tmp_path`. Each `*_global` fixture composes the corresponding entry-point YAML with explicit `datamodule=` / `model=` / `trainer=` overrides at compose time, then applies test-friendly tweaks via an `open_dict(cfg):` block. Read both blocks for today's presets — they change as the fixtures evolve.

`cfg_train_global` and `cfg_eval_global` compose with the **same** `datamodule=ksin model=ffn trainer=cpu` overrides, and dataset shape is pinned via integer `train_val_test_sizes=[2,2,2]` rather than fractional `limit_*_batches`. A train→eval round-trip shares the same `datamodule` / `model` / `callbacks` shape across both fixtures, so neither side has to copy fields from the other.

Both fixtures clear global Hydra state on teardown via `GlobalHydra.instance().clear()`.

A third fixture group — `cfg_surge_xt_global` / `cfg_surge_xt` / `cfg_surge_xt_eval` — follows the same pattern for the Surge XT smoke tests (`tests/test_train.py::test_train_surge_xt_*`). It composes `train.yaml` with `experiment=<experiment_name>` (defaulting to `surge/fake_oracle`, an oracle stand-in module whose `predict_step` returns the target params verbatim, so the default leg exercises the predict→render→audio-metrics pipeline without depending on a real trained model), points at the 5-sample dataset generated on demand by the `surge_xt_smoke_datasets` fixture, and bakes in the smoke-test trainer knobs. `cfg_surge_xt_global` is parametrized over `accelerator`, `param_spec_name`, and `experiment_name` (all indirect fixtures); `test_train_surge_xt` and `test_train_eval_surge_xt` cycle through both `surge/fake_oracle` (oracle-correctness invariants — exact-zero `train/loss`, bit-identical `pred-{i}.pt` vs `target-params-{i}.pt`, tight per-sample audio-metric bounds) and `surge/ffn_full` (real FFN training, real loss-progression coverage) via the module-level `_SURGE_SMOKE_EXPERIMENTS` constant. See [`tests/conftest.py`](../../tests/conftest.py) for the current parameter sets and the shared `_build_surge_xt_smoke_cfg` helper. `cfg_surge_xt_eval` is **not** built by composing `eval.yaml` — it's a copy of the train-side fixture (`cfg_surge_xt_global`) with `mode="predict"` set and `ckpt_path` pointed at the checkpoint the matching `cfg_surge_xt` run will write under the same `tmp_path`. This guarantees the evaluator sees the exact `data` / `model` shape that produced the checkpoint, with no `eval.yaml` / `train.yaml` reconciliation step. Read [`tests/conftest.py`](../../tests/conftest.py) for the current field-level overrides.

A fourth fixture group — `cfg_dataset_global` / `cfg_dataset` — covers the `synth-setter-generate-dataset` CLI (`tests/test_generate_dataset_shards.py`). It composes `dataset.yaml` with `experiment=generate_dataset/smoke-shard`, then redirects `paths.{output_dir,work_dir,log_dir}` into `tmp_path`. Unlike its siblings it **omits** `return_hydra_config=True` because the dataset cfg's `hydra.sweep.subdir` interpolates `${hydra.job.num}` (resolved only at runtime by `@hydra.main`); leaking the `hydra.*` sub-tree into the fixture would break `spec_from_cfg`'s `resolve=True` round-trip. The e2e test pairs `cfg_dataset` with a uuid-suffixed `r2.prefix` override so `from_hydra(cfg_dataset)` runs the real renderer subprocess and real `rclone copy` end-to-end against Cloudflare R2; it is gated by `[integration_r2, r2, requires_vst, slow]` so a default `pytest` run skips it, and auto-skips at runtime when `r2_io.is_r2_reachable()` returns false (no rclone on PATH or no R2 creds). The `requires_vst` marker reflects the renderer-subprocess dependency on the Surge XT VST3 plugin; [`test-generate-dataset-shards.yml`](../../.github/workflows/test-generate-dataset-shards.yml) runs it inside the `tinaudio/synth-setter:dev-snapshot` Docker image (where `/usr/lib/vst3/Surge XT.vst3` is always present) on every PR that touches the test or its surrounding dataset / cli surface.

______________________________________________________________________

## 5. The train → eval E2E template

Reference implementation: [`tests/test_eval.py::test_train_eval`](../../tests/test_eval.py). The shape is three phases: override `cfg_train`, train + assert checkpoint, point `cfg_eval` at the checkpoint and evaluate. Because `cfg_train` and `cfg_eval` are composed with the same `datamodule` / `model` / `trainer` overrides in `conftest.py`, no manual alignment is needed.

```python
import math
import os
from pathlib import Path

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from synth_setter.cli.eval import evaluate
from synth_setter.cli.train import train
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

1. **DataModule `setup(stage)` must cover every stage you invoke.** Lightning passes one of `{"fit", "validate", "test", "predict"}` depending on which Trainer method runs. A `setup()` that only handles `"fit"` silently builds the wrong (or no) dataloader for the others. See Lightning's [DataModule docs](https://lightning.ai/docs/pytorch/stable/data/datamodule.html) for the contract, and [`src/synth_setter/data/ksin_datamodule.py`](../../src/synth_setter/data/ksin_datamodule.py) for the canonical three-branch pattern in this repo.

2. **`src/synth_setter/configs/train.yaml` and `src/synth_setter/configs/eval.yaml` require explicit `datamodule=` / `model=`.** Both entry points use `???` for `datamodule` and `model`, so Hydra fails fast if either is missing. Production runs pass them via an `experiment=` config; tests pass them at compose time inside `conftest.py`. There is no fallback to a researcher-local default.

3. **GPU tests use a three-marker stack.** `@pytest.mark.gpu`, `@pytest.mark.slow`, `@RunIf(min_gpus=1)` each do distinct things. The CI selector for GPU tests lives in [`.github/workflows/test-gpu.yml`](../../.github/workflows/test-gpu.yml); local `make test-fast` and `make test-full-gpu` filters live in the Makefile. If the CI filter changes, the docs don't need updating — the code does.

4. **`weights_only=False` when loading a checkpoint for eval.** PyTorch 2.6 tightened `torch.load`'s default to `weights_only=True`, which refuses Lightning checkpoint metadata. `src/synth_setter/cli/eval.py` passes `weights_only=False` explicitly to `trainer.test/validate/predict(ckpt_path=...)`. New standalone loader code needs the same.

5. **Adding a new `cfg.mode=<x>` dispatch?** Audit every DataModule's `setup()` for that stage. `src/synth_setter/cli/eval.py` routes `mode` to `trainer.test/validate/predict`, each passing a different `stage` string. Gotcha #1 bites if a DataModule doesn't handle the new stage.

______________________________________________________________________

## 7. Pointers

- [`tests/conftest.py`](../../tests/conftest.py) — the Hydra fixtures, including all preset overrides
- [`tests/pipeline/conftest.py`](../../tests/pipeline/conftest.py) — pipeline-test fixtures (separate tree, separate fixtures)
- [`tests/helpers/`](../../tests/helpers)
- [`tests/test_eval.py::test_train_eval`](../../tests/test_eval.py) — canonical E2E template
- [`Makefile`](../../Makefile) — `test-fast`, the `test-full-*` targets, and friends
- [`pyproject.toml`](../../pyproject.toml) — registered markers + pytest config
- [`.github/workflows/`](../../.github/workflows) — which CI job runs which markers
- Lightning [DataModule](https://lightning.ai/docs/pytorch/stable/data/datamodule.html) and [Trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html) docs — stage semantics and the `fit/validate/test/predict` contract
- [`docs/reference/wandb-integration.md`](wandb-integration.md) — tests disable the logger (`cfg.logger=None`); this covers the runtime behavior
