<div align="center">
<h1>synth-setter</h1>
<p>Synthesizer parameter prediction, sound matching, and preset exploration tools.</p>
<p>
  <a href="https://github.com/tinaudio/synth-setter/actions/workflows/test.yml"><img src="https://github.com/tinaudio/synth-setter/actions/workflows/test.yml/badge.svg" alt="CI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-GPL--3.0-blue" alt="License: GPL-3.0"></a>
</p>
</div>

## Overview

synth-setter provides tools for automatic synthesizer parameter estimation
(synth inversion), sound matching, and preset exploration. Given an audio
recording of a synthesizer sound, models predict the parameters that reproduce
it. Built on PyTorch Lightning with Hydra configs.

## Status

**Early-stage research project — work in progress.** Many features are
partially implemented or not yet wired end-to-end, and the end-to-end MVP
pipeline is still being built out (see [Project Tracking](#project-tracking)
below). Current contributions are infrastructure and code quality
improvements on top of the original codebase; no novel modeling work yet.
Expect breaking changes to APIs, configs, and on-disk data formats. This
repository is published to share ongoing research and invite discussion.

## Acknowledgments

This project builds on [*Audio synthesizer inversion in symmetric parameter
spaces with approximately equivariant flow
matching*](https://benhayes.net/synth-perm/) by [Ben Hayes et
al.](https://benhayes.net). The original code is available
[here](https://github.com/ben-hayes/synth-permutations).

[Surge XT](https://surge-synthesizer.github.io/), developed by the Surge
Synth Team, is the synthesizer used for dataset generation and is integrated
under the GPL-3.0 license.

## Features

- **Flow matching and baseline models** for synthesizer parameter estimation
- **Distributed data pipeline** for VST audio dataset generation with cloud support
- **W&B integration** for experiment tracking and model checkpointing — fresh installs log to W&B + CSV + TensorBoard by default; pass `logger=csv` or edit `src/synth_setter/configs/logger/many_loggers.yaml` to drop W&B
- **Docker support** for reproducible training and generation environments
- **Hydra configs** for flexible experiment management

## Prerequisites

- **Supported platforms**: Linux (x86_64) and macOS only. Windows is not supported — the `sh` test dependency and the VST rendering tooling are POSIX-only, and CI covers Ubuntu and macOS only.
- **Git**, **curl**, **make** (for the canonical install path)

`make install` handles uv, Python 3.10, and all dependencies for you.
`make install-surge-xt` fetches the pinned Surge XT VST3 release — no need
to install Surge XT yourself.

## Installation

```bash
# 1. Clone
git clone https://github.com/tinaudio/synth-setter.git
cd synth-setter

# 2. Install uv, create .venv (Python 3.10), install deps, register pre-commit
#    (pre-commit install is skipped if core.hooksPath is set, e.g. in the dev
#    container)
make install

# 3. Activate the venv
source .venv/bin/activate

# 4. Download the Surge XT VST3 into plugins/
make install-surge-xt

# 5. Export environment variables (R2, W&B — see §4b in getting-started)
set -a && source .env && set +a
```

> **Experiment tracking:** the default training run logs to W&B + CSV +
> TensorBoard. Run `wandb login` (or set `WANDB_API_KEY`) before your first
> training run, or drop W&B from the default compose by commenting out (or
> removing) `- wandb` in `src/synth_setter/configs/logger/many_loggers.yaml`. See
> [getting-started §4c](docs/getting-started.md#4c-weights--biases-wb) for
> the full configuration workflow.

> **Already have Surge XT installed system-wide?** Skip `make install-surge-xt`
> and symlink it manually:
> `ln -s "/path/to/Surge XT.vst3" "plugins/Surge XT.vst3"`.
> See [docs/getting-started.md §2d](docs/getting-started.md#2d-install-the-surge-xt-vst3).

> **Prefer pip or conda?** If you'd rather manage the Python interpreter and
> venv yourself, see
> [docs/getting-started.md Appendix A](docs/getting-started.md#appendix-a-manual-environment-setup)
> for a walkthrough using `uv pip install --group dev -e .` inside your own
> environment.

### GPU vs CPU

The PyTorch backend is routed per platform from the committed `uv.lock`. Pick
the install command for your hardware:

| Target                       | Command                                                        |
| ---------------------------- | -------------------------------------------------------------- |
| macOS (Apple Silicon, MPS)   | `uv sync --frozen`                                             |
| Linux GPU box (CUDA 12.8)    | `uv sync --frozen --extra cu128`                               |
| Linux CPU-only (CI / laptop) | `uv sync --frozen --extra cpu --no-default-groups --group dev` |
| Lint / type-check only       | `uv sync --frozen --only-group dev`                            |

Two table caveats:

- Mac must use bare `uv sync --frozen` — backend extras are a silent no-op there.
- CUDA is the source of truth for reported numbers; MPS may diverge slightly.

Full rationale plus the relock cadence and the Linux-only resync flip live
in [docs/reference/dependency-management.md](docs/reference/dependency-management.md).

## Quick Start

```bash
# Run tests
make test-fast

# Run all pre-commit hooks (formatting + linting)
make format

# See all available targets
make help
```

See the project documentation for a full walkthrough.

## Project Structure

```
src/synth_setter/   ML code and data pipeline (PEP src-layout package)
  cli/                 Hydra entrypoints (also published as synth-setter-* console scripts):
    train.py             Training entrypoint
    eval.py              Evaluation entrypoint
    generate_dataset.py  Dataset-generation entrypoint
  data/                Datamodules and dataset construction
  models/              LightningModules and components
  utils/               Logging, callbacks, instantiators, math
  metrics.py           Audio + parameter-space metrics
  pipeline/            Distributed data pipeline:
    schemas/             Pydantic models (DatasetSpec, RenderConfig, prefix, image_config)
    ci/                  CI validation scripts (materialize_spec, validate_shard, validate_spec)
    data/                Dataset-shaping utilities (reshard, rewrite_to_latest, stats, r2_report)
    skypilot_launch.py   SkyPilot launcher CLI
  evaluation/          predict_vst_audio, compute_audio_metrics (called by cli/eval.py)
  tools/               python -m utilities (surge_xt_interactive, plot_param2tok, ...)
configs/        Hydra YAML configs (top-level: train.yaml / eval.yaml / dataset.yaml)
scripts/        SkyPilot / CI shell tooling (skypilot/, ci/)
tests/          Test suite (mirrors src/synth_setter/ structure)
docs/design/    Design documents
```

## Key Files

```
src/synth_setter/models/components/transformer.py       DiT and AST implementations
src/synth_setter/models/components/residual_mlp.py      Residual MLP implementations
src/synth_setter/models/components/cnn.py               CNN encoder implementations
src/synth_setter/models/components/vae.py               VAE+RealNVP baseline implementation
src/synth_setter/models/*_module.py                     LightningModule implementations
src/synth_setter/data/vst/*                             Dataset generation
src/synth_setter/data/vst/surge_xt_param_spec.py        Surge XT dataset sampling distributions
src/synth_setter/data/ot.py                             Optimal transport minibatch coupling
src/synth_setter/data/kosc_datamodule.py                k-osc task data module
configs/experiment/kosc                    k-osc experiment configs
configs/experiment/surge                   Surge XT experiment configs
```

## Project Tracking

Work is organized as epics → phases → tasks, tracked publicly on GitHub.
Since the project is in flux, the board is the best place to see what's
actually being built right now.

- **Project board:** [tinaudio/projects/1](https://github.com/orgs/tinaudio/projects/1)
- **MVP epic:** [#264 — end-to-end MVP pipeline](https://github.com/tinaudio/synth-setter/issues/264) (Docker entrypoint → candidate image creation → dataset generation → training → validation)
- **Active epics:**
  - [#74 — distributed data pipeline](https://github.com/tinaudio/synth-setter/issues/74)
  - [#98 — evaluation pipeline (predict, render, metrics)](https://github.com/tinaudio/synth-setter/issues/98)
  - [#99 — R2 integration for datasets and checkpoints](https://github.com/tinaudio/synth-setter/issues/99)
  - [#107 — training pipeline & ops](https://github.com/tinaudio/synth-setter/issues/107)
  - [#114 — codebase modernization](https://github.com/tinaudio/synth-setter/issues/114)
  - [#148 — CI & automation platform](https://github.com/tinaudio/synth-setter/issues/148)
  - [#149 — test infrastructure & coverage](https://github.com/tinaudio/synth-setter/issues/149)
- **Key milestones:** [data-pipeline v1.0.0](https://github.com/tinaudio/synth-setter/milestone/1), [evaluation v1.0.0](https://github.com/tinaudio/synth-setter/milestone/2), [training v1.0.0](https://github.com/tinaudio/synth-setter/milestone/3)

## Documentation

New to the project? These are the docs worth skimming first, in order:

1. **[Getting started](docs/getting-started.md)** — setup, running the test
   suite, training your first model, and configuring the external dependencies
   needed for the full data pipeline.
2. **[Architecture overview](docs/architecture.md)** — system diagram and how
   the `generate → finalize → train → evaluate` stages fit together.
3. **[Glossary](docs/glossary.md)** — domain terms (synth inversion, flow
   matching, `param_spec`, mel spectrogram, VST, …). Useful as a dictionary
   while reading the other docs.
4. **[Data pipeline design](docs/design/data-pipeline.md)** — the canonical
   design doc for the distributed data pipeline, referenced throughout the
   codebase.

Further reading (mostly for contributors and maintainers):

- [`docs/design/`](docs/design/) — training pipeline, evaluation pipeline,
  storage provenance spec, SkyPilot integration, implementation plans
- [`docs/reference/`](docs/reference/) — configuration reference, Docker,
  GitHub Actions, W&B integration
- [`docs/guides/surge-xt-interactive.md`](docs/guides/surge-xt-interactive.md) —
  human-in-the-loop tool for auditioning predicted Surge XT params and
  capturing patches into a labeled dataset

Run `make help` for available commands.

## Codespaces & Docker

GitHub Codespaces and local dev containers provide a pre-built environment with
Surge XT, rclone, and all Python dependencies already installed. See
[docs/getting-started.md Appendix B](docs/getting-started.md#appendix-b-container-based-setup)
for setup instructions covering both paths.

> **Devcontainer as root:** The default dev container runs as a non-root user.
> If your workflow requires root (e.g., installing system packages), set
> `"remoteUser": "root"` in the devcontainer config you use
> (`.devcontainer/cpu/devcontainer.json` or
> `.devcontainer/gpu/devcontainer.json`). See the
> [devcontainer docs](https://containers.dev/implementors/json_reference/)
> for details.

## License

Released under the [GNU General Public License v3.0](LICENSE). Note that
Surge XT, which this project integrates with, is also GPL-3.0.
