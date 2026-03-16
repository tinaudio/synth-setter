---
name: ml-data-pipeline
description: >-
  ML data pipeline coding standards synthesized from solid open-source repos:
  Netflix Metaflow, Kedro, ZenML, Dagster, Prefect, Google Magenta, and Mozilla/Coqui TTS.
  Use this skill whenever writing, reviewing, or refactoring ML data pipelines, training
  pipelines, audio/feature extraction pipelines, or any data processing workflow.
  Also trigger on: "pipeline step", "data loader", "feature extraction", "training pipeline",
  "artifact versioning", "dataset class", "AudioProcessor", "mel spectrogram pipeline",
  "data catalog", "flow/task/step/asset", or any request to design or critique pipeline
  structure. Use proactively when the user is working on synthesis research code, audio ML,
  or iterative model training scripts — even if they don't say "pipeline" explicitly.
---

# ML Data Pipeline — Coding Standards

Synthesized from: Netflix Metaflow · Kedro · ZenML · Dagster · Prefect · Google Magenta · Coqui TTS

______________________________________________________________________

## 1. Core abstraction model

Every world-class pipeline picks ONE fundamental unit and is consistent about it. Do not mix
paradigms within a project.

| Framework | Unit                                      | How dependencies are declared        |
| --------- | ----------------------------------------- | ------------------------------------ |
| Metaflow  | `@step` on a `FlowSpec` method            | `self.next(self.step_name)`          |
| Kedro     | `node(func, inputs, outputs)`             | string names resolved by DataCatalog |
| ZenML     | `@step` function, composed by `@pipeline` | function arguments (typed)           |
| Dagster   | `@asset` function                         | function arguments → implicit graph  |
| Prefect   | `@task` inside a `@flow`                  | return values passed between tasks   |

**Rule**: Pick one. A step/node/asset is the atomic resumable unit of computation. Keep it
small enough to be a meaningful checkpoint, not so small that I/O overhead dominates.

The `@step` / `@asset` as checkpoint boundary is the key insight from Metaflow: at the end
of each step, all `self.*` instance variables are automatically persisted. Stack variables
(`x = ...`) are NOT persisted. Use this distinction deliberately to control what gets checkpointed.

______________________________________________________________________

## 2. Data handling and versioning

### The Data Catalog pattern (Kedro)

Separate *what data is* from *how it's loaded*. Never hardcode file paths in pipeline logic.

```python
# Bad — path hardcoded in node function
def preprocess(raw_path: str = "data/raw/audio.csv") -> pd.DataFrame:
    return pd.read_csv(raw_path)

# Good — function is pure; catalog.yml owns the path
def preprocess(raw: pd.DataFrame) -> pd.DataFrame:
    return raw.dropna()
```

In `conf/base/catalog.yml`:

```yaml
raw_audio_features:
  type: pandas.CSVDataset
  filepath: data/01_raw/audio_features.csv

preprocessed_audio_features:
  type: pandas.ParquetDataset
  filepath: data/02_intermediate/audio_features.parquet
  versioned: true
```

Use Kedro's 8-layer data folder convention to communicate data maturity:
`01_raw → 02_intermediate → 03_primary → 04_feature → 05_model_input → 06_models → 07_model_output → 08_reporting`

### Automatic artifact versioning (Metaflow / ZenML)

In Metaflow: every `self.x = value` inside a `@step` is automatically versioned and
content-addressed in the artifact store. Retrieve any past run's artifacts via the Client API.

In ZenML: every typed step output is automatically serialized via a `Materializer` and
versioned. Always annotate return types — ZenML uses them to pick the right serializer:

```python
from zenml import step
import pandas as pd

@step
def load_audio_features(dataset_path: str) -> pd.DataFrame:
    """Load and return audio feature matrix."""
    return pd.read_parquet(dataset_path)
```

Without type annotations, ZenML falls back to `cloudpickle` which breaks across Python
versions. **Always annotate step inputs and outputs.**

### Dagster: assets over tasks

Prefer `@asset` (declarative, outcome-focused) over `@op`/`@job` (imperative, task-focused).
Assets know what they produce and what they depend on. Dependencies are implicit from
function arguments:

```python
from dagster import asset
import pandas as pd

@asset
def raw_spectrograms() -> pd.DataFrame:
    ...

@asset
def normalized_spectrograms(raw_spectrograms: pd.DataFrame) -> pd.DataFrame:
    ...
```

Use `key_prefix` for hierarchical namespacing: `@asset(key_prefix=["audio", "train"])`.

______________________________________________________________________

## 3. Configuration philosophy

**Core rule**: Config belongs in YAML, not in function signatures or module-level constants.

Kedro's `conf/base/` vs `conf/local/` split is the gold standard:

- `conf/base/` — shared project config committed to git
- `conf/local/` — user/environment overrides, git-ignored

Reference params in nodes via the catalog name `params:model_options`, never by importing
a config module directly.

Metaflow's three-tier config hierarchy (most recent addition, 2024):

- **Artifacts** — resolved and persisted at end of each task (experiment outputs)
- **Parameters** — resolved at start of a run (can be passed via CLI: `python flow.py run --lr 0.01`)
- **Configs** — resolved at deploy time; can configure decorators themselves

For audio/research pipelines, define all signal processing constants in a config dataclass,
not as magic numbers scattered through functions:

```python
from dataclasses import dataclass

@dataclass
class AudioConfig:
    sample_rate: int = 22050
    n_mels: int = 80
    hop_length: int = 256
    win_length: int = 1024
    mel_fmin: float = 0.0
    mel_fmax: float = 8000.0
    do_trim_silence: bool = True
    trim_db: float = 60.0
```

______________________________________________________________________

## 4. Step / node / asset design conventions

### From Metaflow

- A step is the **smallest resumable unit** — if it fails, you resume from there
- Instance variables (`self.x`) persist; local variables don't — use this deliberately
- Decorators are the extension mechanism: `@retry`, `@timeout`, `@resources(gpu=1)`,
  `@batch`, `@kubernetes` — all composable and stackable
- Steps should be small but not trivially small — balance checkpoint overhead vs granularity

### From Kedro

- Node functions are **pure Python functions** — no side effects, no I/O directly
- Function signature is the contract: `inputs` and `outputs` are string names in the catalog
- Always name your nodes: `node(func=preprocess, inputs="raw", outputs="clean", name="preprocess_node")`
- Use namespaced pipelines for reuse: the same pipeline instance can run under
  `active_modelling_pipeline` and `candidate_modelling_pipeline` namespaces

### From ZenML

- Steps separated by concern: one step per logical transformation
- Retry at the step level, not pipeline level: `@step(retry=StepRetryConfig(max_retries=3))`
- Fan-out / fan-in pattern for parallelism: split one step into N parallel steps, aggregate in final
- Enable caching carefully — `@step(enable_cache=False)` for steps that read external state

### From Dagster

- Assets are data products, not tasks — name them after what they produce, not what they do
  (`preprocessed_spectrograms`, not `run_preprocessing`)
- Use `@multi_asset` when one function naturally produces multiple outputs
- Auto-materialization policies replace cron-style scheduling:
  materialize when upstream changes, not on a fixed schedule

______________________________________________________________________

## 5. Type safety and contracts

ZenML enforces types at step boundaries — lean into this:

```python
from typing import Tuple, Annotated
import numpy as np
from zenml import step

@step
def extract_features(
    audio_path: str,
    config: AudioConfig,
) -> Tuple[
    Annotated[np.ndarray, "mel_spectrogram"],
    Annotated[np.ndarray, "f0_contour"],
]:
    ...
```

Named outputs (`Annotated[..., "name"]`) appear in the ZenML dashboard and are retrievable
by name from any past run.

Dagster uses `DagsterType` and integrates with Pandera for DataFrame validation:
define asset checks (`@asset_check`) for data quality gates at pipeline boundaries.

For audio arrays specifically, encode shape contracts in the type annotation or docstring:

```python
def compute_mel(waveform: np.ndarray) -> np.ndarray:
    """Compute mel spectrogram.

    :param waveform: Shape (T,), float32, normalized to [-1, 1].
    :returns: Shape (n_mels, T // hop_length), float32, log-scaled.
    """
```

______________________________________________________________________

## 6. Observability and artifact tracking

**Metaflow Cards**: attach rich HTML reports to any step with `@card` — no extra services needed.
Produces visual outputs (spectrograms, loss curves, sample audio) that deploy alongside the flow.

**ZenML**: `log_metadata()` and `add_tags()` can be called directly from inside a step.
Log anything JSON-serializable:

```python
from zenml import step, log_metadata

@step
def train_model(data: pd.DataFrame) -> nn.Module:
    model = ...
    log_metadata({"val_loss": 0.043, "n_params": 1_200_000})
    return model
```

**Prefect**: task results can be explicitly published as artifacts with `create_artifact()`.
Unlike ZenML/Metaflow, Prefect does NOT automatically version outputs — you must opt in.

**General rule**: log at minimum — run ID, git commit hash, data version, key hyperparams,
and evaluation metrics. These four together make any run reproducible.

______________________________________________________________________

## 7. Audio and research pipeline specifics

### Coqui TTS / Mozilla TTS — the AudioProcessor pattern

The `AudioProcessor` class is the canonical pattern for audio ML pipelines. It centralizes
all signal processing operations and is always instantiated from config, never with raw args:

```python
from TTS.utils.audio import AudioProcessor

# Always use init_from_config — never construct with raw kwargs in pipeline code
ap = AudioProcessor.init_from_config(config)

# All processing through the same object — no scattered librosa calls
waveform = ap.load_wav(path)             # load + resample + normalize
mel = ap.melspectrogram(waveform)        # consistent STFT params
ap.save_wav(waveform, output_path)       # save with same params
```

Key AudioProcessor responsibilities: loading/resampling wavs, silence trimming,
mel spectrogram computation, normalization (RMS or signal norm), Griffin-Lim inversion,
mean-variance statistics for normalization. All configured from `BaseAudioConfig`.

Separate dataset config from audio config:

```python
dataset_config = BaseDatasetConfig(
    formatter="ljspeech",         # how to parse metadata files
    meta_file_train="metadata.csv",
    path="/data/LJSpeech-1.1",
)
audio_config = BaseAudioConfig(
    sample_rate=22050,
    n_mels=80,
    hop_length=256,
)
```

### Google Magenta — data pipeline patterns

Magenta uses **Apache Beam** for large-scale dataset preprocessing (converts raw audio +
MIDI to TFRecord format). For research-scale work, use `tf.data.Dataset` with `.shuffle()`,
`.batch()`, `.prefetch(tf.data.AUTOTUNE)`:

```python
dataset = tfds.load("groove/2bar-16000hz", split="train", try_gcs=True)
dataset = dataset.shuffle(1024).batch(32).prefetch(tf.data.AUTOTUNE)
```

Magenta's `NoteSequence` protobuffer is the canonical intermediate representation for
symbolic music — prefer structured intermediates over raw arrays when the domain has
well-defined structure.

### For synthesizer / timbre research pipelines

Apply the Catalog pattern to audio datasets — never hardcode paths to audio files or
precomputed features. Version precomputed features separately from raw audio:

```yaml
# catalog.yml
raw_audio_files:
  type: partitions.PartitionedDataset
  path: data/01_raw/synth_recordings
  dataset: audio.AudioDataset

precomputed_timbre_features:
  type: pandas.ParquetDataset
  filepath: data/04_feature/timbre_features.parquet
  versioned: true
```

Keep feature extraction functions pure — `extract_timbre_features(waveform: np.ndarray, sr: int, config: FeatureConfig) -> pd.Series`
is testable without filesystem access.

______________________________________________________________________

## 8. Anti-patterns to avoid

**Notebook-style pipeline** — sequential script with hardcoded paths, no checkpointing.
If step 4 of 8 fails, you rerun from scratch.

**Config in code** — `HOP_LENGTH = 256` as a module-level constant. When you change it,
downstream cached features are silently stale.

**I/O inside transform functions** — a function named `preprocess_audio` should not open
files. It should accept an array and return an array. I/O belongs in a loader step.

**Untyped step outputs** — without type annotations, ZenML pickles everything with cloudpickle,
breaking portability and artifact inspection.

**Manual versioning** — naming files `features_v2_final_fixed.parquet`. Use the framework's
built-in versioning (`versioned: true` in Kedro catalog, automatic in Metaflow/ZenML).

**Giant steps** — a single step/node that loads data, extracts features, trains a model,
and evaluates. If evaluation fails, the whole training reruns. Split at natural checkpoints.

**Magic audio parameters** — `librosa.feature.melspectrogram(y, sr=22050, n_mels=80, hop_length=256)`
scattered across multiple files. One change requires hunting all call sites.

______________________________________________________________________

## 9. Code review checklist for ML pipelines

Use this when reviewing any pipeline PR:

| #   | Check                              | Pass criteria                                                    |
| --- | ---------------------------------- | ---------------------------------------------------------------- |
| 1   | **Unit selection**                 | One abstraction (@step / node / @asset) used consistently        |
| 2   | **Pure transforms**                | Node/step functions have no file I/O, only data in → data out    |
| 3   | **Config externalized**            | No hardcoded paths, sample rates, or hyperparams in functions    |
| 4   | **Type annotations**               | All step inputs and outputs typed                                |
| 5   | **Artifact versioning**            | Intermediate datasets are versioned, not overwritten             |
| 6   | **Resumability**                   | Pipeline can restart from any step without rerunning prior steps |
| 7   | **AudioProcessor / config object** | Audio params flow through one config object                      |
| 8   | **Retry/timeout**                  | Steps with external I/O have retry decorators                    |
| 9   | **Metadata logged**                | Key metrics + git hash logged per run                            |
| 10  | **Named nodes**                    | All nodes/steps/assets have explicit names (not auto-generated)  |
| 11  | **No YAGNI abstractions**          | No base classes or extension points for unbuilt features         |
| 12  | **Shape contracts**                | Array-accepting functions document expected shape in docstring   |

______________________________________________________________________

## Reference repos

- `github.com/Netflix/metaflow` — step checkpointing, decorator system, artifact store
- `github.com/kedro-org/kedro` — Data Catalog, project layout, pure node functions
- `github.com/zenml-io/zenml` — typed steps, stack abstraction, automatic lineage
- `github.com/dagster-io/dagster` — asset-based model, data quality checks, auto-materialization
- `github.com/PrefectHQ/prefect` — @flow/@task, retries, caching, dynamic mapping
- `github.com/magenta/magenta` — audio pipeline with Beam, NoteSequence, tf.data patterns
- `github.com/coqui-ai/TTS` — AudioProcessor pattern, BaseAudioConfig, TTSDataset
