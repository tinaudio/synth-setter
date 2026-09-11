# FaustWasm browser player

Export an artifact produced by `scripts/faustwasm/export-artifacts.mjs` as a self-contained browser player:

```bash
cd scripts/faustwasm/browser
npm ci
node export-browser.mjs --artifact /path/to/artifact --output /path/to/site
python -m http.server --directory /path/to/site 8000
```

Open `http://localhost:8000`. The Start audio button is the required browser gesture that creates and resumes the `AudioContext`. The player compiles no Faust source and makes no network request outside the exported directory.

Controls are normalized to `[0, 1]` and translated through the manifest's explicit canonical-to-Wasm address map. Polyphonic artifacts show note buttons; mono artifacts run freely. Startup and asset-loading failures are shown in the status region.

After startup, `window.faustPlayer` exposes the loaded `manifest`, `applyPatch(params)`, `recordFrames(frameCount)`, and `renderNote(request)`. Captures resolve with the artifact's native channel count. Only one capture may run at a time; another `recordFrames` or `renderNote` call rejects with `Audio capture is already in progress` before scheduling notes.

## Browser test

The real Playwright test consumes a bright-organ artifact already exported by the shared compiler:

```bash
npx playwright install chromium
FAUSTWASM_E2E_ARTIFACT=/path/to/bright-organ-artifact \
FAUSTWASM_MONO_E2E_ARTIFACT=/path/to/filter-osc-artifact \
npm run test:e2e
```

The tests serve each exported directory over localhost, start audio through a click, and capture samples after the Faust AudioWorklet. They check finite non-silent output, native mono/stereo shape, capture concurrency, note and parameter causality, and browser/offline RMS parity for the same artifact and timing.
