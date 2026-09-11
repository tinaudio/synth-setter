# Flow-sketch RIR evaluations

Issue #3498 defines three directly runnable five-row evaluations of the pinned
`flow_sketch` checkpoint against filtered third-party room impulse responses. Each suite
uses the real pyFDN renderer, the `pyfdn_reverb` sketch, and the production audio metrics.
No explicit seed is required; evaluation retains the existing unseeded default.

## Checkpoint lineage

- Checkpoint: `s3://experiments/checkpoints/pyfdn-flow_sketch__sketch-pyfdn_reverb__conditioning-mel__ot-true/flow_sketch-20260908T170724945Z/model.ckpt`
- SHA-256: `9a17f57e32e318b6f057cd308d28cc7600d5f0fda62183f548947ac1e73e8bcf`
- Source run: `flow_sketch-20260908T170724945Z`
- Source git SHA: `487568b24b58063ba907c4406cf0a0083400c645`
- W&B artifact: `model-flow_sketch:v0`
- Immutable lineage fields: `consumed_train_config_id=flow_sketch` and
  `consumed_train_artifact_alias=v0`
- Legacy checkpoint compatibility: `model.encoder.use_fixed_ast_padding=false`

All publication roots share
`r2://experiments/evaluations/flow_sketch/flow_sketch-20260908T170724945Z/rir/`.
Each leaf publishes immutable attempts and its own `latest.json` pointer below that
namespace.

## Commands

Run MIT IR Survey (`audio_decodable = true`, first five filtered rows):

```bash
uv run synth-setter-eval experiment=pyfdn/eval_flow_sketch_rir_mit_ir_survey
```

Run EchoThief (`audio_decodable = true`, first five filtered rows):

```bash
uv run synth-setter-eval experiment=pyfdn/eval_flow_sketch_rir_echothief
```

Run ASHIR (`audio_decodable = true AND source_path LIKE 'BRIRs/%'`, first five
filtered rows):

```bash
uv run synth-setter-eval experiment=pyfdn/eval_flow_sketch_rir_ashir
```

These commands use local resources and existing R2/W&B credentials. They do not launch
SkyPilot or paid compute.

## Production-path verification

The marked integration test downloads the real checkpoint and MIT corpus, loads the real
model, generates and renders five predictions through pyFDN, computes metrics, publishes
to a unique test-owned R2 prefix, downloads the immutable attempt, and consumes its CSV
and rendered WAV:

```bash
uv run pytest tests/integration/test_flow_sketch_rir_evaluation_r2.py -m "slow and gpu and integration_r2 and r2" -v
```
