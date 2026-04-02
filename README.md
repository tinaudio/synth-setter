<div align="center">
<h1>synth-setter</h1>
<p>Synthesizer parameter prediction, sound matching, and preset exploration tools.</p>
<p>
  <a href="https://github.com/tinaudio/synth-setter/actions/workflows/test.yml"><img src="https://github.com/tinaudio/synth-setter/actions/workflows/test.yml/badge.svg" alt="CI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+"></a>
</p>
</div>

## Overview

synth-setter provides tools for automatic synthesizer parameter estimation
(synth inversion), sound matching, and preset exploration. Given an audio
recording of a synthesizer sound, models predict the parameters that reproduce
it. Built on PyTorch Lightning with Hydra configs.

## Features

- **Flow matching models** for synthesizer parameter estimation
- **Distributed data pipeline** for VST audio dataset generation (RunPod + Cloudflare R2)
- **W&B integration** for experiment tracking and model checkpointing
- **Docker support** for reproducible training and generation environments
- **Hydra configs** for flexible experiment management

## Prerequisites

- **Python 3.10+**
- **uv** (recommended) -- [install uv](https://docs.astral.sh/uv/getting-started/installation/)
- **Git** with submodule support (clone with `--recurse-submodules`)
- **System dependencies for VST rendering** -- see the project documentation for details

## Installation

Clone the repository:

```bash
git clone --recurse-submodules https://github.com/tinaudio/synth-setter.git
cd synth-setter
```

> **Note on submodules**
>
> This repository uses Git submodules that are configured with SSH URLs (`git@github.com:...`).
> If you clone via HTTPS as above and do not have SSH access set up, submodule fetching may fail.
> You can either configure an SSH key with GitHub, or ask Git to rewrite SSH URLs to HTTPS:
>
> ```bash
> git config --global url."https://github.com/".insteadOf git@github.com:
> ```

### Using uv (recommended)

```bash
uv pip install -r requirements.txt
```

### Using pip

```bash
pip install -r requirements.txt
```

### Using conda

```bash
conda env create -f environment.yaml
```

### Editable install

```bash
make install
```

### GPU vs CPU

The default installation installs CPU-only PyTorch. For GPU training, install
PyTorch with CUDA wheels for your system -- see the
[PyTorch install matrix](https://pytorch.org/get-started/locally/) for the
correct command.

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
  stages/      Generate and finalize stage logic
  backends/    Compute providers (local, RunPod)
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

## Publication

This repository accompanies a submission to ISMIR 2025. An online supplement with
audio examples is available at
[benhayes.net/synth-perm/](https://benhayes.net/synth-perm/).

## Documentation

- Getting started guide (coming soon)
- [Design documents](docs/design/)
- Contributing guidelines (coming soon)
- Run `make help` for available commands

## License

License information will be added soon. See the repository for updates.
