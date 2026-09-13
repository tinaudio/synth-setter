import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { extname, join } from 'node:path';
import { after, before, test } from 'node:test';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { promisify } from 'node:util';

import { renderSurge } from './runtime.mjs';

const execute = promisify(execFile);
const bundlePath = process.env.SURGE_WASM_BUNDLE;
if (!bundlePath) throw new Error('SURGE_WASM_BUNDLE must name a bundle produced by build.mjs');

const presetPath = new URL('../../../presets/surge-simple.fxp', import.meta.url);
const contentTypes = new Map([
    ['.js', 'text/javascript'],
    ['.json', 'application/json'],
    ['.mjs', 'text/javascript'],
    ['.wasm', 'application/wasm'],
]);

let root;
let server;
let loader;
let mappedParameters;
let preset;

before(async () => {
    loader = (await import(pathToFileURL(join(bundlePath, 'surge-host.mjs')).href)).default;
    preset = new Uint8Array(await readFile(presetPath));
    const repository = fileURLToPath(new URL('../../../', import.meta.url));
    const python = `
import json
from synth_setter.cli.sketch_render import load_render_config
from synth_setter.data.vst.param_spec_registry import param_specs
from synth_setter.renderer_factory import make_audio_renderer
render = load_render_config()
renderer = make_audio_renderer(render)
identities = renderer.parameter_map.surgepy_params()
parameters = [
    {"id": identities[param.name].synth_side_id, "name": identities[param.name].name, "value": 0.5}
    for param in param_specs[render.param_spec_name].synth_params
]
print(json.dumps(parameters))
`;
    const pythonExecutable = process.env.PYTHON ?? join(repository, '.venv', 'bin', 'python');
    const { stdout } = await execute(pythonExecutable, ['-c', python], { cwd: repository });
    mappedParameters = JSON.parse(stdout);
    server = createServer(async (request, response) => {
        try {
            const path = join(bundlePath, new URL(request.url, 'http://localhost').pathname);
            response.setHeader('Content-Type', contentTypes.get(extname(path)) ?? 'application/octet-stream');
            response.end(await readFile(path));
        } catch {
            response.writeHead(404).end();
        }
    });
    await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
    const { port } = server.address();
    root = new URL(`http://127.0.0.1:${port}/`);
});

after(async () => {
    if (server) {
        await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
    }
});

const render = (parameters = [], overrides = {}) => renderSurge({
    root,
    loader,
    preset,
    parameters,
    note: 60,
    velocity: 100,
    noteStart: 0,
    noteEnd: 0.5,
    sampleRate: 44100,
    frames: 44100,
    ...overrides,
});

const loaderInspectingValuesAtDestroy = (observedValues) => async (options) => {
    const module = await loader(options);
    const cwrap = module.cwrap.bind(module);
    module.cwrap = (name, returnType, argumentTypes) => {
        const wrapped = cwrap(name, returnType, argumentTypes);
        if (name !== 'ss_destroy') return wrapped;
        return () => {
            const paramValue = cwrap('sh_param_value', 'number', ['number']);
            for (const id of observedValues.keys()) observedValues.set(id, paramValue(id));
            wrapped();
        };
    };
    return module;
};

test('renderSurge_fixed_fxp_returns_finite_nonzero_stereo_audio', async () => {
    const result = await render();

    assert.equal(result.version, 'Surge XT 1.4.HEAD.dd68c743');
    assert.equal(result.engineCommit, 'dd68c74346c828ef25bd6504867936648161b7a7');
    assert.equal(result.left.length, 44100);
    assert.equal(result.right.length, 44100);
    assert.ok(result.left.every(Number.isFinite));
    assert.ok(result.right.every(Number.isFinite));
    assert.ok(result.left.some((sample) => sample !== 0));
    assert.ok(result.parameterCount > 700);
});

test('renderSurge_model_native_parameter_subset_matches_live_host', async () => {
    const result = await render(mappedParameters);

    assert.equal(mappedParameters.length, 89);
    assert.ok(result.left.some((sample) => sample !== 0));
});

test('renderSurge_saved_fxp_reloads_continuous_and_categorical_values', async () => {
    const changed = await render([
        { id: 308, name: 'A Filter 1 Cutoff', value: 0.2 },
        { id: 223, name: 'A Osc 1 Octave', value: 0.8305 },
    ]);
    const reloadedValues = new Map([[308, undefined], [223, undefined]]);

    await render([], {
        loader: loaderInspectingValuesAtDestroy(reloadedValues),
        preset: changed.fxp,
    });

    assert.ok(Math.abs(reloadedValues.get(308) - 0.2) < 1e-6);
    assert.ok(Math.abs(reloadedValues.get(223) - 0.83) < 1e-6);
});

test('renderSurge_saved_fxp_has_native_chunk_header', async () => {
    const { fxp } = await render();
    const view = new DataView(fxp.buffer, fxp.byteOffset, fxp.byteLength);
    const text = (start, length) => new TextDecoder().decode(fxp.subarray(start, start + length));

    assert.equal(text(0, 4), 'CcnK');
    assert.equal(view.getUint32(4), 0);
    assert.equal(text(8, 4), 'FPCh');
    assert.equal(view.getUint32(12), 1);
    assert.equal(text(16, 4), 'cjs3');
    assert.equal(view.getUint32(20), 1);
    assert.equal(view.getUint32(24), 1);
    assert.deepEqual(fxp.slice(28, 56), new Uint8Array(28));
    assert.equal(view.getUint32(56), fxp.byteLength - 60);
    assert.equal(text(60, 4), 'sub3');
});

test('renderSurge_closed_filter_attenuates_note_energy', async () => {
    const baseline = await render();
    const changed = await render([{ id: 308, name: 'A Filter 1 Cutoff', value: 0 }]);

    // Random oscillator phase can change samples even when parameter writes are ignored.
    const baselineEnergy = baseline.left.reduce((sum, sample) => sum + sample * sample, 0);
    const filteredEnergy = changed.left.reduce((sum, sample) => sum + sample * sample, 0);
    assert.ok(filteredEnergy < baselineEnergy * 0.1, `cutoff did not attenuate the note: ${filteredEnergy} / ${baselineEnergy}`);
});

test('renderSurge_equal_note_endpoints_preserve_zero_duration', async () => {
    const result = await renderSurge({
        root,
        loader,
        preset,
        parameters: [],
        note: 60,
        velocity: 100,
        noteStart: 0.5,
        noteEnd: 0.5,
        sampleRate: 44100,
        frames: 44100,
    });

    assert.ok(result.left.every(Number.isFinite));
});

test('renderSurge_malformed_fxp_rejects_before_rendering', async () => {
    await assert.rejects(
        renderSurge({
            root,
            loader,
            preset: new Uint8Array([0, 1, 2, 3]),
            parameters: [],
            note: 60,
            velocity: 100,
            noteStart: 0,
            noteEnd: 0.001,
            sampleRate: 44100,
            frames: 64,
        }),
        /invalid FXP/,
    );
});
