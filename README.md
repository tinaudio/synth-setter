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

## Documentation

- Design docs live in `docs/design/`
- See `make help` for available commands
