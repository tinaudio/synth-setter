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
- **W&B integration** (opt-in) for experiment tracking and model checkpointing — fresh installs log to CSV + TensorBoard; pass `logger=wandb` to enable W&B
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

> **Experiment tracking:** the default training run logs to CSV + TensorBoard
> (no external account needed). W&B is **opt-in** — `python src/train.py
> ... logger=wandb` after `wandb login` (or setting `WANDB_API_KEY`). See
> [getting-started §4c](docs/getting-started.md#4c-weights--biases-wb) for
> the full enable/disable workflow.

> **Already have Surge XT installed system-wide?** Skip `make install-surge-xt`
> and symlink it manually:
> `ln -s "/path/to/Surge XT.vst3" "plugins/Surge XT.vst3"`.
> See [docs/getting-started.md &sect;2d](docs/getting-started.md#2d-install-the-surge-xt-vst3).

> **Prefer pip or conda?** If you'd rather manage the Python interpreter and
> venv yourself, see
> [docs/getting-started.md Appendix A](docs/getting-started.md#appendix-a-manual-environment-setup)
> for a walkthrough using `pip install -r requirements.txt -e .` inside your
> own environment.

### GPU vs CPU

This project depends on PyTorch (`torch>=2.0.0`), but the requirements do not fix
whether you use a CPU-only build or a CUDA-enabled build. Choose and install the
appropriate PyTorch package for your system (CPU-only or a specific CUDA version)
using the [PyTorch install matrix](https://pytorch.org/get-started/locally/), then
install the remaining dependencies as described above.

## Quick Start

```bash
# Run tests
make test

# Run all pre-commit hooks (formatting + linting)
make format

# See all available targets
make help
```

See the project documentation for a full walkthrough.

## Project Structure

```
src/           ML code (models, data modules, training, evaluation)
pipeline/      Distributed data pipeline
  schemas/     Pydantic models (config, spec, prefix, image_config)
  entrypoints/ Pipeline entry points
  ci/          CI validation scripts
configs/       Hydra YAML configs and pipeline configs
scripts/       Standalone scripts
tests/         Test suite (mirrors src/ and pipeline/ structure)
docs/design/   Design documents
```

## Key Files

```
src/models/components/transformer.py       DiT and AST implementations
src/models/components/residual_mlp.py      Residual MLP implementations
src/models/components/cnn.py               CNN encoder implementations
src/models/components/vae.py               VAE+RealNVP baseline implementation
src/models/*_module.py                     LightningModule implementations
src/data/vst/*                             Dataset generation
src/data/vst/surge_xt_param_spec.py        Surge XT dataset sampling distributions
src/data/ot.py                             Optimal transport minibatch coupling
src/data/kosc_datamodule.py                k-osc task data module
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
