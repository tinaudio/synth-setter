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

After startup, `window.faustPlayer` exposes the loaded `manifest`, `applyPatch(params)`, `recordFrames(frameCount)`, and `renderNote(request)`. `renderNote` schedules note boundaries on the `AudioContext` sample clock and resolves with the recorded stereo channels.

## Browser test

The real Playwright test consumes a bright-organ artifact already exported by the shared compiler:

```bash
npx playwright install chromium
FAUSTWASM_E2E_ARTIFACT=/path/to/bright-organ-artifact npm run test:e2e
```

The test serves the exported directory over localhost, starts audio through a click, captures samples after the Faust AudioWorklet, and checks finite non-silent stereo output, note and parameter causality, and browser/offline RMS parity for the same artifact and timing.
