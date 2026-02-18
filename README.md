<div align="center">

# Audio synthesizer inversion in symmetric parameter spaces with approximately equivariant flow matching

This repository accompanies a submission to ISMIR 2025. A full README explaining how to use this code will be provided before the conference. In the meantime, audio examples are available at the [online supplement](https://benhayes.net/synth-perm/).

If you would like to explore the source code, you may find the below helpful:

</div>

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

...existing code...

## Setup

1. Install requirements:

   ```bash
   # [OPTIONAL] create conda environment
   conda update --name base conda
   conda env create -f environment.yaml
   conda activate myenv

   # install torch stack and app dependencies separately
   python -m pip install --upgrade pip
   python -m pip install -r requirements-torch.txt
   python -m pip install -r requirements-app.txt

   # backward-compatible one-liner
   python -m pip install -r requirements.txt
   ```

2. Configure Weights & Biases (optional but recommended):

   ```bash
   wandb login
   ```

   - Adjust project/team defaults in [configs/logger/wandb.yaml](configs/logger/wandb.yaml).
   - You can also set `WANDB_ENTITY`, `WANDB_PROJECT`, or run with `logger=wandb`.

## Tests

Run the fast test suite:

```bash
pytest -k "not slow"
```

Run the full suite:

```bash
pytest
```

(You can also use [Makefile](Makefile) targets like `make test` or `make test-full`.)
