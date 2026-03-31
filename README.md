<div align="center">

# synth-setter

Synth inversion, sound matching, and preset exploration tools.

</div>

## Overview

synth-setter provides tools for automatic synthesizer parameter estimation
(synth inversion), sound matching, and preset exploration. Built on PyTorch
Lightning with Hydra configs.

## Key Features

- **Flow matching models** for synthesizer parameter estimation
- **Distributed data pipeline** for VST audio dataset generation (RunPod + Cloudflare R2)
- **W&B integration** for experiment tracking and model checkpointing
- **Docker support** for reproducible training and generation environments

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
make test

# Run all pre-commit hooks
make format

# See all available targets
make help
```

## Project Structure

```
src/           ML code (models, data modules, training, evaluation)
pipeline/      Distributed data pipeline (python -m pipeline)
scripts/       Standalone scripts
configs/       Hydra YAML configs and pipeline configs
tests/         Test suite
docs/design/   Design documents
```

## Publication

This repository accompanies a submission to ISMIR 2025. An online supplement with
audio examples is available at
[benhayes.net/synth-perm/](https://benhayes.net/synth-perm/).

## Key Files

```
src/models/components/transformer.py       <- DiT and AST implementations
src/models/components/residual_mlp.py      <- Residual MLP implementations
src/models/components/cnn.py               <- CNN encoder implementations
src/models/components/vae.py               <- VAE+RealNVP baseline implementation
src/models/*_module.py                     <- LightningModule implementations, containing training logic
src/data/vst/*                             <- Dataset generation
src/data/vst/surge_xt_param_spec.py        <- Specification of Surge XT dataset sampling distributions
src/data/ot.py                             <- Optimal transport minibatch coupling
src/data/kosc_datamodule.py                <- Implementation of k-osc task
configs/experiment/kosc                    <- k-osc experiment configs
configs/experiment/surge                   <- Surge XT experiment configs
```

## Documentation

- Design docs live in `docs/design/`
- See `make help` for available commands
