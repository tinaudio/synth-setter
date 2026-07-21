# RunPod dataset network volume

The RunPod network volume is a persistent dataset cache seeded from R2. Staging
copies a finalized dataset from R2 once; training mounts that cache and hydrates
the pod's local disk before loading Lance.

## Create the volume

The checked-in definition selects RunPod data center `US-KS-2` and requests
750 GB. Creating it starts persistent storage billing:

```bash
uv run sky volumes apply \
  src/synth_setter/configs/volumes/runpod-datasets-us-ks-2.yaml
uv run sky volumes ls --refresh --verbose
```

RunPod network volumes are data-center bound. The attached task therefore runs
in the volume's data center even though the compute template does not repeat the
zone.

## Stage the 440k Surge Simple dataset

Run the balance preflight, then launch the checksum-verified R2 copy:

```bash
uv run python -c \
  "from synth_setter.pipeline.skypilot_launch import _check_runpod_balance; _check_runpod_balance(); print('balance preflight passed')"
uv run synth-setter-skypilot-launch \
  src/synth_setter/configs/launch/stage-runpod-surge-simple-440k-volume.yaml
```

The staging script uses `rclone copy --immutable --checksum`, checks source
parity, and writes `.synth-setter-stage-complete` only after validation. It is
safe to rerun after an interrupted transfer.

The SkyPilot template explicitly mounts the volume at
`/workspace/network-volume`. This is independent of RunPod's default
`/workspace` mount convention.

## Train from pod-local storage

```bash
uv run synth-setter-skypilot-launch \
  src/synth_setter/configs/launch/train-runpod-flow-simple-440k-volume.yaml
```

The launch checks the staging marker, then configures:

```text
datamodule.download_dataset_root_uri=file:///workspace/network-volume/<dataset>
```

`prepare_data()` copies that mounted directory into the experiment's local
`datamodule.dataset_root` with immutable checksum semantics. Training therefore
reads from the pod's local 750 GB disk rather than from the network mount.

Delete the persistent volume only when its cached datasets are no longer
needed:

```bash
uv run sky volumes delete synth-setter-datasets-us-ks-2
```
