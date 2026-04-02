# Scripts

Standalone scripts for data generation, evaluation, and analysis.

| Script                         | Purpose                                                                                                   | Example                                                                      |
| ------------------------------ | --------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `add_music2latent.py`          | Add music2latent embeddings to HDF5 shards                                                                | `python scripts/add_music2latent.py --help`                                  |
| `aggregate_samples.sh`         | Collect audio samples from multiple model runs into a flat directory                                      | `bash scripts/aggregate_samples.sh`                                          |
| `compute_audio_metrics.py`     | Compute multi-scale spectrogram and MFCC audio metrics on predicted vs target audio                       | `python scripts/compute_audio_metrics.py --help`                             |
| `docker_entrypoint.sh`         | Docker container entrypoint dispatching on the `MODE` env var (`idle`, `passthrough`, `generate_dataset`) | `MODE=idle bash scripts/docker_entrypoint.sh`                                |
| `generate_surge_xt_data.py`    | Generate Surge XT synthesizer audio dataset (stub)                                                        | `python scripts/generate_surge_xt_data.py --surge-path "vsts/Surge XT.vst3"` |
| `get-ckpt-from-wandb.sh`       | Find the latest checkpoint file for a W&B run ID                                                          | `bash scripts/get-ckpt-from-wandb.sh <wandb-run-id>`                         |
| `get_dataset_stats.py`         | Compute per-feature mean and std of an HDF5 dataset using Dask                                            | `python scripts/get_dataset_stats.py`                                        |
| `make_sig_perf.py`             | Benchmark k-osc and k-sin signal generation performance                                                   | `python scripts/make_sig_perf.py`                                            |
| `model_from_wandb_repl.py`     | Load a trained model from a W&B run and drop into an IPython REPL                                         | `python scripts/model_from_wandb_repl.py`                                    |
| `paramspec_to_table.py`        | Convert VST parameter specs to a LaTeX longtable                                                          | `python scripts/paramspec_to_table.py`                                       |
| `plot_param2tok.py`            | Plot parameter-to-token mappings from a trained model checkpoint                                          | `python scripts/plot_param2tok.py`                                           |
| `predict_vst_audio.py`         | Render audio from predicted VST parameters and compare with targets                                       | `python scripts/predict_vst_audio.py --help`                                 |
| `r2_shard_report.py`           | Generate a report of shards stored in an R2 bucket via rclone                                             | `python scripts/r2_shard_report.py --help`                                   |
| `reshard_data.py`              | Reshard HDF5 dataset into train/val/test splits                                                           | `python scripts/reshard_data.py <dataset_root> -t 200 -v 4 -e 1`             |
| `rewrite_dataset_to_latest.py` | Rewrite HDF5 shards with `libver="latest"` for SWMR support                                               | `python scripts/rewrite_dataset_to_latest.py --help`                         |
| `run-linux-vst-headless.sh`    | Bootstrap headless X11/D-Bus/xsettingsd for running VST3 plugins in CI                                    | `bash scripts/run-linux-vst-headless.sh <command>`                           |
| `schedule.sh`                  | Run a sequence of training jobs with different configs                                                    | `bash scripts/schedule.sh`                                                   |
| `subdir_funtime.py`            | Compute pairwise mel-spectrogram distances between audio files in subdirectories                          | `python scripts/subdir_funtime.py --help`                                    |

### Data directories

| Directory       | Purpose                                                         |
| --------------- | --------------------------------------------------------------- |
| `audio_dirs/`   | Text files listing audio output directories per dataset variant |
| `sample_lists/` | Text files listing sample IDs to aggregate per dataset variant  |
