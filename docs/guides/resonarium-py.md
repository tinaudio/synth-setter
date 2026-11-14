# Native Resonarium rendering

Select `synth=resonarium_py render=resonarium` in dataset-generation configurations to render
[Resonarium](https://github.com/gabrielsoule/resonarium) through its own Python bindings, with
no VST host, plugin binary, display server, or preset file. Resonarium is GPL-3.0.

## Installing the bindings

The bindings are the unmodified upstream `Resonarium_Python` CMake target at revision
`d7f013f9` (release tag `v0.1.1`). [`packages/resonarium-native`](../../packages/resonarium-native/)
builds it with scikit-build-core from SHA-256-pinned source archives
([provenance](../../packages/resonarium-native/resonarium_native/PROVENANCE.md)) and installs the
module as `import resonarium`. The upstream build compiles JUCE and gin, so it is an **opt-in**
dependency group: the default `uv sync` and CI never build it, and tests marked
`requires_resonarium` skip when it is absent.

```bash
# Ubuntu 22.04+ x86_64; the build takes about three minutes on 16 vCPUs.
sudo apt-get install -y --no-install-recommends build-essential libcurl4-openssl-dev \
  libx11-dev libxext-dev libxinerama-dev libxrandr-dev libxcursor-dev libxi-dev \
  libgl1-mesa-dev libasound2-dev libfreetype6-dev libfontconfig1-dev
uv sync --frozen --extra cpu --group resonarium
.venv/bin/python -c "import resonarium; print(resonarium.Resonarium().get_sample_rate())"
```

## Parameters and audio

`resonarium_py` has 692 synth coordinates plus MIDI pitch (48–72) and two note-timing coordinates.
Each coordinate is named by its native parameter ID, for example `decayTime wb0r0` (bank 0,
resonator 0) or `impExciter0 enabled`. Its value is the host-normalized position in [0, 1].
The renderer maps it through the same JUCE `NormalisableRange` skew and snapping a plugin host's
automation uses, so skewed ranges such as 10 Hz–20 kHz cutoffs sample perceptually. Stepped
controls (switches, filter types, coupling modes) are categorical over their legal positions.
77 coordinates are internal controls (effect enables, keytrack switches, distortion and amp
types), which a host can only set through saved state.

The specification omits controls that cannot change one fresh, unmodulated note, and controls
only the editor reads. That covers modulators and macros, host-tempo sync and beat divisions,
external input, the sampler, voice count, solo, `useSemitones`, `stereoResonators`, and
`distPrePostFilter`. It also omits the master and bank input/output gains: they only rescale
level, and sampling them over ±100 dB makes clipping or silence dominate. Omitted controls keep
their upstream defaults.

Each render starts a fresh processor at 44.1 kHz with 512-frame blocks. It applies the patch,
processes one block and resets the processor, as a host already running the patch would. It
then sends sample-accurate note-on and note-off events. A note window inside one frame gates for one sample. A note-on past the last
frame renders silence. Mono downmixes the native stereo output. Native gain is preserved, with
no clipping or normalization; dataset generation rejects clipped, silent, and non-finite draws
per attempt. Upstream defaults disable every exciter and resonator, so about a third of uniform
draws are audible. The default budget of 100 attempts per row covers that.

## Verification against the VST3

The native engine and the Resonarium v0.1.1 Linux VST3 compile the same source tag.

- **Mapping:** all 615 host-automatable coordinates map to VST3 parameters by name. Writing
  normalized positions 0, 0.137, 0.5, 0.861 and 1 through host automation stores exactly the
  native value this mapping computes (3,075 checks).
- **Readback:** the native engine stores each mapped value unchanged at the same positions.
- **Coverage:** from a patch with every switch on, changing any one of the 692 coordinates changes
  the rendered audio.
- **Determinism (A-B-A):** renders are bit-identical across interleaved patches, including
  noise-excited ones. Exciter noise is a deterministic per-voice stream, so there is no per-render
  randomness.
- **Audio parity:** 24 random audible patches had every coordinate loaded into the VST3 through
  saved state. 21 meet all of the KR-106 parity thresholds (MSS < 3, RMS > 0.95, SOT < 0.01,
  wMFCC < 4), and the median MSS is 0.001.
  - One miss equals the VST3's own difference between two fresh instances (MSS 5.28 both ways).
  - Two phaser-and-chorus patches have MSS under 3 but miss RMS (0.91) or SOT (0.011). The VST3
    is not exactly repeatable across instances on such patches either.
  - Effects start from a host's steady state. Without the processed block before the reset,
    phaser patches diverged from the plugin at onset (median MSS 1.9).

## Production dataset

`experiment=generate_dataset/resonarium-py-lance-2m-40k-10k` renders a 2,000,000 / 40,000 /
10,000-row Lance dataset into the `experiments` bucket through the shard queue (2,500 rows per
shard; 800 / 16 / 4 shards).

### Join the shard queue from a fresh pod

The current run was launched on 2026-11-11 from commit `b5164474`. Main doesn't contain this
backend until its PR merges, so joiners use that commit (or a later commit of the PR). Every
worker must render the same code, because the spec pins the synth identity and render contract.

```bash
SHA=b5164474bde43559721cb235dcc2e0aa22930726
SPEC='r2://experiments/data/resonarium-py-lance-2m-40k-10k/resonarium-py-lance-2m-40k-10k-20261111T034305220Z/input_spec.json'
WT=$HOME/synth-setter-resonarium
INF=(infisical run --projectId=f257dbf6-868f-4661-b62a-610625555539 --env=dev --)

# 1. Build dependencies (as root on RunPod; prefix sudo elsewhere).
apt-get update -qq && apt-get install -y --no-install-recommends build-essential \
  libcurl4-openssl-dev libx11-dev libxext-dev libxinerama-dev libxrandr-dev libxcursor-dev \
  libxi-dev libgl1-mesa-dev libasound2-dev libfreetype6-dev libfontconfig1-dev

# 2. Check out the code and build the environment, including the bindings.
git clone --filter=blob:none https://github.com/tinaudio/synth-setter.git "$WT"
git -C "$WT" fetch origin "$SHA" && git -C "$WT" checkout --detach "$SHA"
cd "$WT" && uv sync --frozen --extra cpu --group resonarium
.venv/bin/python -c "import resonarium, synth_setter; print('ok', synth_setter.__file__)"

# 3. Secrets. A fresh pod has no Infisical session: a person with access runs the interactive
#    `infisical login` once on each new pod. Then confirm, without printing any values:
"${INF[@]}" true && echo "infisical ok"

# 4. Join, detached, with math-library thread caps (see #3755).
mkdir -p "$HOME/logs"
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 setsid -f nohup \
  "${INF[@]}" "$WT/.venv/bin/synth-setter-generate-dataset-from-spec-uri" "$SPEC" \
  > "$HOME/logs/resonarium-2m-worker.log" 2>&1 < /dev/null
```

Each worker process runs up to 16 shard renderers, sized from the CPUs it can see. RunPod
containers often report the host's core count, so on a pod with fewer than 32 vCPUs you can
launch under `taskset -c 0-$((2 * VCPUS - 1))` to size the pool to about one renderer per vCPU.
Watch `uptime` against the pod's vCPUs.

### Check progress

```bash
cd "$WT" && "${INF[@]}" .venv/bin/python - "$SPEC" <<'PY'
import sys
from synth_setter.pipeline.r2_io import lance_target
from synth_setter.pipeline.shard_claims import ShardClaims
from synth_setter.pipeline.spec_io import load_spec_from_uri

spec = load_spec_from_uri(sys.argv[1])
print(ShardClaims.for_run(*lance_target(spec.r2.shard_claims_uri())).status_counts())
PY
tail -n 3 "$HOME/logs/resonarium-2m-worker.log"; uptime
```

`done` counts shards that finished and staged. The run is complete at 820 `done`.

### Finalize

After every shard is `done`, run finalization once from any worker:

```bash
"${INF[@]}" "$WT/.venv/bin/synth-setter-finalize-dataset" \
  "dataset_root_uri=${SPEC%input_spec.json}"
```

## Direct render

```python
from synth_setter.data.vst.resonarium_renderer import ResonariumRenderer

renderer = ResonariumRenderer(
    plugin_path="resonarium", sample_rate=44100, channels=2, signal_duration_seconds=4.0
)
audio = renderer.render(
    {"impExciter0 enabled": 1.0, "enabled wb0r0": 1.0, "decayFilterCutoff wb0r0": 0.9},
    midi_note=60,
    velocity=100,
    note_start_and_end=(0.0, 1.0),
)
# audio.shape == (2, 176400)
```
