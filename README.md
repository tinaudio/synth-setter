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
conda env create -f environment.yaml  # creates the "myenv" environment by default
conda activate myenv
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

## Documentation

- Getting started guide (coming soon)
- [Design documents](docs/design/)
- Contributing guidelines (coming soon)
- Run `make help` for available commands

## Acknowledgments

This project builds on prior work by Ben Hayes (Queen Mary University of London),
whose research and generation infrastructure provided the foundation for the
synthesizer parameter estimation pipeline. The accompanying paper is available at
[benhayes.net/synth-perm](https://benhayes.net/synth-perm/).

[Surge XT](https://surge-synthesizer.github.io/) is developed by the
Surge Synth Team and is used in this project under the GPL-3.0 license.

## License

License information will be added soon. See the repository for updates.
