# RunPod dataset network volumes

A RunPod network volume is a persistent, data-center-bound dataset cache seeded
from R2. Staging copies a finalized dataset from R2 once; training mounts that
cache and hydrates the pod's local disk before loading Lance.

Because a volume is pinned to one data center, the mounted volume decides where
the attached task runs. The compute template mounts `${NETWORK_VOLUME}`, and
each launch config (or a `--network-volume` CLI override) names the volume —
so the volume name is effectively the region selector.

## Create a volume

One definition per data center lives in `src/synth_setter/configs/volumes/`
(currently `US-CA-2` and `AP-JP-1`, 750 GB each). Creating one starts
persistent storage billing:

```bash
uv run sky volumes apply \
  src/synth_setter/configs/volumes/runpod-datasets-us-ca-2.yaml
uv run sky volumes ls --refresh --verbose
```

To add a region, copy a definition, set `name:` and `infra:` to the new data
center (zone IDs and live GPU/CPU stock come from RunPod's `dataCenters`
GraphQL query), and apply it. No code changes are needed.

## Stage the 440k Surge Simple dataset

Run the balance preflight, then launch the checksum-verified R2 copy. The
checked-in launch config targets the `us-ca-2` volume; pass
`--network-volume` to stage any other region's volume:

```bash
uv run python -c \
  "from synth_setter.pipeline.skypilot_launch import _check_runpod_balance; _check_runpod_balance(); print('balance preflight passed')"
uv run synth-setter-skypilot-launch \
  src/synth_setter/configs/launch/stage-runpod-surge-simple-440k-volume.yaml
uv run synth-setter-skypilot-launch \
  src/synth_setter/configs/launch/stage-runpod-surge-simple-440k-volume.yaml \
  --network-volume synth-setter-datasets-ap-jp-1
```

The staging script uses `rclone copy --immutable --checksum`, checks source
parity, and writes `.synth-setter-stage-complete` only after validation. It is
safe to rerun after an interrupted transfer.

The SkyPilot templates explicitly mount the volume at
`/workspace/network-volume`. This is independent of RunPod's default
`/workspace` mount convention. Staging uses a small-disk template
(`runpod-network-volume-staging-template.yaml`): the copy writes straight to
the mounted volume, and hosts with small container disks are far easier to
schedule than the 750 GB-disk hosts training needs.

## Train from pod-local storage

```bash
uv run synth-setter-skypilot-launch \
  src/synth_setter/configs/launch/train-runpod-flow-simple-440k-volume.yaml
```

The same `--network-volume` override retargets training at another region's
staged volume. The launch checks the staging marker, then configures:

```text
datamodule.download_dataset_root_uri=file:///workspace/network-volume/<dataset>
```

`prepare_data()` copies that mounted directory into the experiment's local
`datamodule.dataset_root` with immutable checksum semantics. Training therefore
reads from the pod's local 750 GB disk rather than from the network mount.

Delete a persistent volume only when its cached datasets are no longer needed —
each staged region can be re-seeded from R2 at any time:

```bash
uv run sky volumes delete synth-setter-datasets-us-ca-2
```
