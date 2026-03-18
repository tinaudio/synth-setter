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
