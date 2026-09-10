# Browser pyFDN ports

`src/synth_setter/web/fdn/` holds JavaScript ports of the Python code a browser page needs to
prepare inputs for, and score outputs of, the pyFDN sketch flow without a Python host:

| Module        | Python reference                                                          |
| ------------- | ------------------------------------------------------------------------- |
| `sketch.mjs`  | `synth_setter.features.pyfdn_controls.extract_reverb_sketch`              |
| `decode.mjs`  | `decode_model_output` on the `pyfdn_n8_mono_householder` spec             |
| `metrics.mjs` | MSS, octave EDC RMSE, octave RT60 log-RMSE, T30 percentage error, C50 MAE |
| `dsp.mjs`     | scipy `sosfilt`, Schroeder integration, window helpers                    |
| `fft.mjs`     | Power spectrum for any frame length (radix-2 plus Bluestein)              |

The ports are pinned to the Python references through golden fixtures. Both languages rebuild the
same 4 s synthetic responses from a 32-bit LCG recipe, so the fixtures store only expected outputs
and the scipy-designed octave-band SOS tables at 44.1 kHz. Regenerate them after changing any
reference:

```bash
uv run python -m synth_setter.tools.browser_fdn_fixtures
npm test --prefix src/synth_setter/web
```

`tests/tools/test_browser_fdn_fixtures.py` fails when the committed JSON no longer matches what
the Python references produce, and the node tests fail when a port drifts from the JSON.
