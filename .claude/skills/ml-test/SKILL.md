---
name: ml-test
description: >-
  ML-specific testing standards synthesized from Eugene Yan's ML testing guides and Karpathy's
  neural network training recipe. Covers pre-train tests, post-train behavioral tests, model
  evaluation, pipeline testing, data validation, training sanity checks, and debugging.
  Use when writing or reviewing tests for ML models, training pipelines, data pipelines,
  or audio/feature processing code.
---

# ML Testing Standards

Synthesized from:

- [Testing ML (Eugene Yan)](https://eugeneyan.com/writing/testing-ml/)
- [Testing Pipelines (Eugene Yan)](https://eugeneyan.com/writing/testing-pipelines/)
- [Unit Testing ML (Eugene Yan)](https://eugeneyan.com/writing/unit-testing-ml/)
- [A Recipe for Training Neural Networks (Karpathy)](https://karpathy.github.io/2019/04/25/recipe/)

______________________________________________________________________

## Core Principle

In software, we write code that *contains* logic. In ML, we write code that *learns* logic.
This means traditional tests verify written code, but ML also needs tests that verify
**learned behavior** — what the model picked up from data.

Testing ML systems splits into:

- **Testing** — model behavior checks (does it do what we expect?)
- **Evaluation** — performance metrics (does it do it well enough?)

______________________________________________________________________

## 1. Pre-Train Tests (Implementation Correctness)

Run without trained parameters. Verify written logic before any training.

### 1.1 Output Shape and Type Validation

Every model and transform function should be tested for correct output structure:

```python
def test_model_output_shape():
    model = MyModel(config)
    x = torch.randn(batch_size, channels, samples)
    y = model(x)
    assert y.shape == (batch_size, num_params)

def test_mel_spectrogram_shape():
    audio = np.random.randn(16000 * 4).astype(np.float32)
    mel = compute_mel(audio, config)
    assert mel.shape == (config.n_mels, expected_frames)
    assert mel.dtype == np.float32
```

### 1.2 Output Range Validation

- Probabilities must be in [0, 1]
- Audio output must be in [-1, 1]
- Softmax outputs must sum to ~1.0
- Regression outputs should be within expected bounds

```python
def test_output_range():
    preds = model(sample_input)
    assert (preds >= 0).all() and (preds <= 1).all()
```

### 1.3 Data Leakage Detection

Check that test data hasn't leaked into training data:

```python
def test_no_train_test_leakage():
    combined = pd.concat([train_df, test_df])
    deduped = combined.drop_duplicates()
    assert len(combined) == len(deduped), "Train/test overlap detected"
```

### 1.4 Loss at Initialization and Final Layer Bias

Two related checks that catch silent bugs early:

**A. Initialize final layer bias correctly.** The network should predict sensible values
before any training. Without correct bias initialization, the first few iterations waste
compute recovering from a bad starting point (hockey-stick loss curve):

- **Classification with N classes:** set final bias so output ≈ uniform → loss starts at `-log(1/N)`
- **Regression with mean target M:** set final bias to M, not zero
- **Imbalanced dataset (1:10 ratio):** set logit bias so network predicts 0.1 at initialization

**B. Verify loss at initialization matches expectation:**

```python
def test_loss_at_initialization():
    model = MyModel(config)  # with correctly initialized bias
    loss = criterion(model(sample_input), sample_target)
    expected = -math.log(1.0 / num_classes)
    assert abs(loss.item() - expected) < 0.1, f"Init loss {loss.item()} != expected {expected}"
```

If the initial loss is wrong, check weight initialization and final layer bias first.

### 1.5 Overfitting Capacity (Single Batch)

The model must be able to achieve near-zero loss on a single batch. If it can't, the
architecture or training loop is broken:

```python
@pytest.mark.slow
def test_overfit_single_batch():
    model = MyModel(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch_x, batch_y = next(iter(train_loader))

    for _ in range(200):
        loss = criterion(model(batch_x), batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert loss.item() < 0.01, f"Cannot overfit single batch: loss={loss.item()}"
```

### 1.6 Model Capacity Scaling

Increasing model capacity should monotonically improve training performance:

```python
def test_capacity_improves_training():
    losses = []
    for hidden_size in [32, 64, 128, 256]:
        model = MyModel(hidden_size=hidden_size)
        final_loss = train_for_n_steps(model, train_data, steps=100)
        losses.append(final_loss)
    # Each larger model should achieve lower or equal training loss
    for i in range(1, len(losses)):
        assert losses[i] <= losses[i-1] + 0.01  # small tolerance
```

### 1.7 Mathematical Function Tests

Test custom loss functions, metrics, and mathematical utilities with known inputs:

```python
def test_gini_impurity_pure_node():
    assert gini_impurity([1, 1, 1]) == 0.0

def test_gini_impurity_mixed():
    assert abs(gini_impurity([1, 0, 1, 0]) - 0.5) < 1e-6
```

Always use hardcoded expected values, never mirror the production logic in tests.

### 1.8 Backprop Dependency Verification

Use backpropagation to verify that data dependencies are correct — that example i's
loss only flows back to example i's input, not to other examples in the batch.
This catches batch dimension mixing bugs (common in autoregressive models and
attention mechanisms):

```python
def test_backprop_data_dependencies():
    """Verify example i's loss depends only on example i's input."""
    model = MyModel(config)
    x = torch.randn(batch_size, channels, samples, requires_grad=True)
    y = model(x)

    # Loss for example i only
    for i in range(batch_size):
        model.zero_grad()
        if x.grad is not None:
            x.grad.zero_()
        loss = y[i].sum()
        loss.backward(retain_graph=True)
        # Gradient should be non-zero ONLY for input i
        for j in range(batch_size):
            if j == i:
                assert x.grad[j].abs().sum() > 0, f"No gradient for input {j}"
            else:
                assert x.grad[j].abs().sum() == 0, f"Leaking gradient from {i} to {j}"
```

Also verify all parameters receive gradients (separate, simpler check):

```python
def test_all_parameters_receive_gradients():
    model = MyModel(config)
    loss = criterion(model(sample_input), sample_target)
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"
```

### 1.9 Input-Independent Baseline

Verify the model actually uses its inputs — zero out the input and confirm the model
performs worse than with real data:

```python
def test_model_uses_input():
    real_loss = train_for_n_steps(model, real_data, steps=50)
    zero_loss = train_for_n_steps(model, zeroed_data, steps=50)
    assert real_loss < zero_loss, "Model doesn't use input data"
```

______________________________________________________________________

## 2. Post-Train Tests (Learned Behavior)

Verify that the trained model exhibits expected patterns. Inspired by the
[CheckList](https://arxiv.org/abs/2005.04118) methodology.

### 2.1 Invariance Tests

Changing irrelevant features should NOT change the prediction:

```python
def test_prediction_invariant_to_irrelevant_features():
    # For audio: changing metadata shouldn't affect mel prediction
    pred_a = model(audio_a, metadata={"name": "Alice"})
    pred_b = model(audio_a, metadata={"name": "Bob"})
    assert torch.allclose(pred_a, pred_b, atol=1e-5)
```

For synthesizer parameter prediction: changing the sample filename or metadata
shouldn't affect predicted parameters.

### 2.2 Directional Expectation Tests

Known causal relationships should be reflected in predictions:

```python
def test_directional_expectations():
    # Higher resonance should produce higher spectral peak
    pred_low = model(audio_with_low_resonance)
    pred_high = model(audio_with_high_resonance)
    assert pred_high[resonance_param_idx] > pred_low[resonance_param_idx]
```

### 2.3 Model Comparison Tests

Better models should outperform simpler baselines:

```python
def test_ensemble_beats_single():
    single_score = evaluate(single_model, test_data)
    ensemble_score = evaluate(ensemble_model, test_data)
    assert ensemble_score >= single_score
```

### 2.4 Minimum Functionality Tests

Test specific known examples where the expected output is unambiguous:

```python
def test_known_preset_prediction():
    # A sine wave at 440Hz should predict oscillator frequency near 440
    audio = generate_sine(440, duration=4.0, sr=16000)
    params = model.predict(audio)
    assert abs(params["osc_frequency"] - 440) < 50
```

______________________________________________________________________

## 3. Model Evaluation (Performance Standards)

### 3.1 Metric Thresholds

Set minimum acceptable performance on held-out data:

```python
def test_minimum_accuracy():
    accuracy = evaluate(model, test_set)
    assert accuracy > 0.82, f"Accuracy {accuracy} below threshold 0.82"

def test_minimum_auc():
    auc = evaluate_auc(model, test_set)
    assert auc > 0.84
```

### 3.2 Latency Benchmarks

```python
def test_inference_latency():
    times = [time_inference(model, sample) for _ in range(100)]
    p99 = np.percentile(times, 99)
    assert p99 < 0.004, f"p99 latency {p99}s exceeds 4ms"
```

### 3.3 Performance Stability

Track that algorithm updates don't drastically change training/inference times.
RandomForest training/inference should be roughly ~5x DecisionTree (non-parallelized).
Capture timing variability across multiple runs.

______________________________________________________________________

## 4. Pipeline Testing

### 4.1 Test Granularity Hierarchy

Maximize tests at the lowest level. Higher-level tests are more brittle:

| Level                   | Volume | Brittleness | What to test                                      |
| ----------------------- | ------ | ----------- | ------------------------------------------------- |
| Row-level unit tests    | Many   | Low         | Individual transform functions with single inputs |
| Schema tests            | Many   | Low         | Column presence and dtypes at pipeline stages     |
| Column-level unit tests | Some   | Medium      | Operations on entire columns                      |
| Table-level unit tests  | Few    | High        | Aggregation/filtering on complete tables          |
| Integration tests       | Few    | High        | End-to-end pipeline segments                      |

**Rule:** Many row-level and schema tests, a handful of the rest.

### 4.2 Row-Level Unit Tests

Test individual functions with single inputs. Use `@pytest.mark.parametrize` for
multiple scenarios:

```python
@pytest.mark.parametrize("audio,expected_length", [
    (np.zeros(16000), 63),       # 1 second -> 63 mel frames
    (np.zeros(32000), 126),      # 2 seconds -> 126 mel frames
    (np.zeros(0), 0),            # empty -> 0 frames
])
def test_mel_frame_count(audio, expected_length):
    mel = compute_mel(audio, sr=16000, hop_length=256)
    assert mel.shape[1] == expected_length
```

### 4.3 Schema Tests

Validate column presence and dtypes at pipeline boundaries. Schema tests pass if
minimum required columns exist — pipelines can add columns without breaking:

```python
def test_shard_schema():
    with h5py.File(shard_path) as f:
        assert "audio" in f, "Missing 'audio' dataset"
        assert "mel_spec" in f, "Missing 'mel_spec' dataset"
        assert "param_array" in f, "Missing 'param_array' dataset"
        assert f["audio"].dtype == np.float32
```

### 4.4 Integration Tests

Test complete pipeline segments. Use **loose granularity** — count rows, check ranges,
verify uniqueness rather than exact values:

```python
def test_generate_pipeline_produces_valid_shards(tmp_path):
    run_generate(config, output=tmp_path, num_shards=2, shard_size=10)
    shards = list(tmp_path.glob("shard-*.h5"))
    assert len(shards) == 2
    for shard in shards:
        with h5py.File(shard) as f:
            assert f["audio"].shape[0] == 10  # row count, not exact values
            assert f["audio"].shape[1] > 0     # has samples
```

### 4.5 Avoid Exact-Value Assertions in Integration Tests

Exact values are fine for row-level unit tests (use `parametrize`). But for integration
and table-level tests, exact values couple tests to implementation — when logic changes
correctly, these tests break unnecessarily. Test properties instead:

```python
# Bad (integration test): breaks when processing logic changes
assert df["ctr"].iloc[0] == 0.2

# Good (integration test): tests the property, not the value
assert (df["ctr"] >= 0).all() and (df["ctr"] <= 1).all()
assert df["ctr"].notna().all()
assert len(df) == expected_row_count

# Fine (row-level unit test): exact values with parametrize
@pytest.mark.parametrize("input,expected", [(5, 0.5), (10, 1.0)])
def test_normalize(input, expected):
    assert normalize(input, max_val=10) == expected
```

### 4.6 Additive vs Retroactive Test Impact

A central insight for pipeline test design: understand which tests survive pipeline
evolution and which require constant updates.

**Additive** (unchanged when new data/logic is added):

- Row-level unit tests for new methods need new tests only
- Schema tests remain unchanged if minimum required columns are preserved

**Retroactive** (must be manually updated when logic changes):

- Column/table-level tests need expected-output revisions
- Integration tests require updated predictions reflecting new logic

If a test breaks and has to be updated frequently, ask: is it a valid test? Design for
additive resilience — maximize row-level and schema tests that survive pipeline evolution.

### 4.7 The Probabilistic Testing Tension

ML outputs are inherently probabilistic. This creates a fundamental tension:

- **Tighten assertions** → meaningless failing tests (random seed changes break them)
- **Loosen assertions** → tests don't actually assert anything useful

There is no perfect solution. Practical approaches:

- Use deterministic seeds for reproducibility in unit tests
- Test properties (monotonicity, bounds, shapes) rather than exact values
- For trained models, use wide tolerance bands based on observed variance
- Accept that some tests need actual trained models (`@pytest.mark.slow`)

______________________________________________________________________

## 5. Data Validation

### 5.1 Become One with the Data

This is Karpathy's Phase 1 and the most important step. Spend hours — not minutes —
examining your data before writing any model code:

- Scan for duplicates, corrupted samples, label errors, and class imbalances
- Write code to search, filter, sort, and visualize by label type, annotation count, feature distributions
- Visualize distributions and outliers on every axis
- Listen to audio samples — do they sound correct? Are labels accurate?
- Look at outliers — they often reveal data quality issues or labeling errors
- Understand your own classification process — can you do the task yourself?
- Ask: Are local features sufficient or is global context needed? How much variation
  exists? What variation is spurious? How noisy are the labels?
- Your model's mispredictions will make more sense if you've deeply understood the data

### 5.2 Establish a Human Baseline

Before training any model, measure human performance on your task:

- Annotate a sample of test data yourself
- Annotate the same data twice to measure your own consistency
- Use human accuracy as the floor — if your model can't beat random but you can,
  the problem is in your pipeline, not the task
- Monitor human-interpretable metrics alongside model metrics

### 5.3 Visualize What Enters the Network

Visualize data **immediately before** `y_hat = model(x)` — after all preprocessing,
augmentation, and batching. This is the only "source of truth":

```python
# Decode the actual tensor that enters forward()
# Don't trust the preprocessing pipeline — verify the output
save_audio(x[0].cpu().numpy(), "debug_input_sample.wav")
plot_spectrogram(x[0].cpu().numpy(), "debug_input_mel.png")
```

### 5.4 Start Without Augmentation

Disable all data augmentation initially. Introduce it only as a regularizer in phase 4
(regularize), after confirming the baseline model works. Augmentation bugs are a common
source of silent training failures.

______________________________________________________________________

## 6. Training Sanity Checks (Karpathy's Recipe)

### 6.1 Set Up End-to-End Skeleton First

Before any real training:

- Fix random seeds for reproducibility
- Verify loss at initialization
- Initialize final layer bias correctly for the target distribution
- Compare against a human baseline
- Overfit a single batch to verify the training loop works
- Visualize prediction dynamics during training

### 6.2 Loss Should Decrease

If loss doesn't decrease:

- Learning rate too high (loss explodes) or too low (loss doesn't move)
- Wrong loss function for the task
- Data pipeline bug (labels don't match inputs)
- Gradient flow issue (dead ReLUs, vanishing gradients)

### 6.3 Use Adam at 3e-4 as Default

Start with Adam optimizer at learning rate 3e-4. It's forgiving of hyperparameter
choices. Don't tune the learning rate until everything else works. Disable learning
rate decay entirely at first.

### 6.4 Add Complexity One Component at a Time

Never change multiple things at once. Add one signal, feature, or architectural
component, verify it helps, then add the next:

- Start with the simplest model that can learn
- Add components one by one
- Verify each addition improves the metric
- If it doesn't help, remove it

### 6.5 Don't Innovate on Architecture

Copy proven architectures from related papers. "Don't be a hero" with custom
architectures until the baseline is solid. The pipeline, data, and training setup
matter more than architectural novelty.

### 6.6 Regularization Order

When the model overfits (low train loss, high val loss), add regularization in this
order of effectiveness:

01. More real data (the only guaranteed way)
02. Aggressive data augmentation
03. Creative augmentation (domain randomization, simulation)
04. Pretrain on related tasks
05. Reduce input dimensionality
06. Decrease model size
07. Reduce batch size (stronger regularization via approximate batch norm)
08. Add dropout (use dropout2d for ConvNets; use carefully with batch normalization)
09. Increase weight decay
10. Early stopping on validation loss
11. Try a larger model with early stopping

### 6.7 Monitor First-Layer Weights and Activations

- First-layer weights should show meaningful patterns (edges for vision, frequency responses for audio)
- Activations should not be all zero (dead neurons) or all saturated
- Gradient magnitudes should be roughly uniform across layers

### 6.8 Training Failure Modes

| Symptom                            | Likely Cause                                    |
| ---------------------------------- | ----------------------------------------------- |
| Loss doesn't decrease              | LR too low, data bug, gradient flow issue       |
| Loss explodes                      | LR too high, numerical instability              |
| Loss decreases then plateaus early | Model too small, or underfitting                |
| Train loss low, val loss high      | Overfitting — add regularization                |
| Train loss high, val loss high     | Underfitting — increase capacity                |
| Loss oscillates wildly             | LR too high, batch size too small               |
| NaN loss                           | Numerical instability, log(0), division by zero |

### 6.9 Silent Failure Modes

Bugs that don't crash but silently degrade quality — the hardest to catch:

- Forgetting to flip labels when flipping images (network compensates internally)
- Off-by-one in autoregressive models (predicting from wrong position)
- Gradient clipping applied to loss instead of gradients
- Using pretrained weights without the original mean subtraction
- Dropout and batch normalization interacting badly
- Learning rate schedule copied from another codebase with different data size

### 6.10 Hyperparameter Tuning

- Use random search over grid search (networks are differentially sensitive to parameters)
- Consider Bayesian hyperparameter optimization tools
- Tune learning rate last, after everything else works
- Do not trust learning rate decay defaults from other codebases

### 6.11 Final Squeeze

- Model ensembles give ~2% accuracy gain ("pretty much guaranteed")
- Knowledge distillation if ensemble inference is too expensive
- Networks keep training for unintuitively long time — be patient
- "A fast and furious approach to training neural networks does not work" — patience
  and attention to detail correlate with success

______________________________________________________________________

## 7. ML-Specific Test Data Guidelines

### 7.1 Use Minimal, Self-Contained Test Data

Define sample data directly in test code, not external files:

```python
def test_preprocess():
    sample = {"audio": np.random.randn(16000), "params": np.array([0.5, 0.3])}
    result = preprocess(sample)
    assert result["mel"].shape[0] == 128
```

### 7.2 Initialize Models with Random Weights for Unit Tests

Don't download pretrained weights for structure tests:

```python
def test_model_architecture():
    config = ModelConfig(hidden_size=64)
    model = MyModel(config)  # random weights, fast
    assert model.output_head.out_features == num_params
```

Mark tests that need real weights with `@pytest.mark.slow`.

### 7.3 Don't Test External Libraries

Assume PyTorch, h5py, librosa, etc. work correctly. Test YOUR code, not theirs.

### 7.4 Test Preprocessing and Postprocessing Thoroughly

These are pure functions with known inputs/outputs — test them aggressively:

- Encoding/decoding
- Normalization/denormalization
- Augmentation transforms
- Filtering and ranking
- Error handling for malformed inputs

### 7.5 Property-Based Testing

Generate synthetic data via distributions and categorical specifications. Tests verify
output *properties* rather than exact values — more resilient to data changes.

**Libraries:**

- **Hypothesis** — fuzzy/property-based test generation
- **Pandera** — DataFrame schema validation with Hypothesis integration
- **Faker** — realistic fake data generation

**Pattern:** (1) generate data, (2) run pipeline, (3) check properties:

```python
from hypothesis import given
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp

@given(audio=hnp.arrays(np.float32, shape=(16000,), elements=st.floats(-1, 1)))
def test_mel_always_non_negative(audio):
    mel = compute_mel(audio, config)
    assert (mel >= 0).all()
```

**Important caveat:** Property-based testing works well for data validation and schema
enforcement, but cannot easily verify specific business logic (correct item ranking,
diversification rules). Use alongside row-level tests, not as a replacement.

### 7.6 When to Test Against Actual Models vs Random Weights

| Scenario                                   | Use random weights | Use actual trained model |
| ------------------------------------------ | ------------------ | ------------------------ |
| Output shape/type validation               | Yes                | No                       |
| Device compatibility                       | Yes                | No                       |
| Architecture structure checks              | Yes                | No                       |
| Loss decreases with training               | No                 | Yes                      |
| Overfitting single batch                   | No                 | Yes                      |
| Inference output class mapping             | No                 | Yes                      |
| Model server accepts batches               | Either             | Either                   |
| Behavioral tests (invariance, directional) | No                 | Yes                      |

______________________________________________________________________

## Review Checklist

| #    | Check                                                                                       | Severity |
| ---- | ------------------------------------------------------------------------------------------- | -------- |
| MT1  | Output shape tests for all models and transform functions                                   | BLOCK    |
| MT2  | Output range validation (probabilities in [0,1], audio in [-1,1])                           | BLOCK    |
| MT3  | No train/test data leakage                                                                  | BLOCK    |
| MT4  | Final layer bias initialized correctly; loss at init matches expected value                 | WARN     |
| MT5  | Model can overfit a single batch (slow test)                                                | BLOCK    |
| MT6  | Backprop dependency verified — example i's loss only affects example i's input              | WARN     |
| MT7  | All parameters receive non-zero gradients                                                   | WARN     |
| MT8  | Model uses input data (not learning from bias alone)                                        | WARN     |
| MT9  | Invariance tests for irrelevant features                                                    | WARN     |
| MT10 | Directional expectation tests for known causal relationships                                | WARN     |
| MT11 | Minimum metric thresholds on held-out data                                                  | BLOCK    |
| MT12 | Human baseline established and compared against                                             | WARN     |
| MT13 | Schema tests at pipeline boundaries (column presence, dtypes)                               | BLOCK    |
| MT14 | Row-level unit tests with parametrize for transforms                                        | WARN     |
| MT15 | Integration tests use loose granularity (counts, ranges, not exact values)                  | WARN     |
| MT16 | Tests designed for additive resilience (row-level + schema over table/integration)          | WARN     |
| MT17 | Test data is self-contained in test code, not loaded from files                             | WARN     |
| MT18 | Models initialized with random weights for unit tests, `@pytest.mark.slow` for real weights | WARN     |
| MT19 | Preprocessing/postprocessing tested as pure functions with known inputs                     | BLOCK    |
| MT20 | No testing of external libraries (PyTorch, h5py, librosa)                                   | WARN     |
| MT21 | Data deeply explored before modeling (Karpathy Phase 1)                                     | WARN     |
| MT22 | Data visualized immediately before model forward pass                                       | WARN     |
| MT23 | Training starts without augmentation; augmentation added as regularizer                     | WARN     |
| MT24 | Random seeds fixed for reproducibility in tests                                             | WARN     |
| MT25 | Probabilistic assertion tension addressed (properties over exact values for ML outputs)     | WARN     |
