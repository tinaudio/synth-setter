import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { compileFaustArtifact } from './export-artifacts.mjs';
import {
    applyCanonicalPatch,
    createOfflineSynth,
    loadFaustArtifact,
    renderNote,
} from './runtime.mjs';
import { readFaustWasmPackageVersion } from './package-version.mjs';

const SOURCE = `declare name "contract";
import("stdfaust.lib");
gain = hslider("gain", 0.5, 0, 1, 0.01);
process = os.osc(440) * gain;
`;

test('offline boundaries reject non-positive and fractional loop sizes', async () => {
    const artifact = { manifest: { mode: 'mono' } };
    for (const blockSize of [0, -1, 1.5, true]) {
        await assert.rejects(
            createOfflineSynth(artifact, { sampleRate: 44_100, blockSize }),
            /blockSize must be a positive integer/,
        );
    }
    for (const frames of [-1, 1.5, true]) {
        assert.throws(
            () => renderNote(
                { processor: {}, blockSize: 128, manifest: { outputs: 1 } },
                { frames, note: 60, velocity: 100, startFrame: 0, endFrame: 1 },
            ),
            /frames must be a non-negative integer/,
        );
    }
});

test('canonical discrete domains reject fractions and accept listed values', () => {
    const written = [];
    const synth = { processor: { setParamValue: (address, value) => written.push([address, value]) } };
    const manifest = {
        parameters: [{
            canonicalAddress: '/canonical/gate',
            wasmAddress: '/native/gate',
            min: 0,
            max: 1,
            kind: 'discrete',
            values: [0, 1],
        }],
    };

    assert.throws(
        () => applyCanonicalPatch(synth, manifest, { '/canonical/gate': 0.5 }),
        /parameter outside discrete native domain/,
    );
    applyCanonicalPatch(synth, manifest, { '/canonical/gate': 0 });
    applyCanonicalPatch(synth, manifest, { '/canonical/gate': 1 });
    assert.deepEqual(written, [['/native/gate', 0], ['/native/gate', 1]]);
});

test('compiled artifact loads and canonical patch changes real audio', async () => {
    const directory = await mkdtemp(join(tmpdir(), 'faustwasm-contract-'));
    try {
        const packageVersion = await readFaustWasmPackageVersion();
        const request = {
            expectedFaustWasmVersion: packageVersion,
            identity: 'contract',
            source: SOURCE,
            mode: 'mono',
            voices: 0,
            outputs: 1,
            reservedWasmAddresses: [],
            parameters: [
                {
                    canonicalAddress: '/canonical/gain',
                    wasmAddress: '/contract/gain',
                    min: 0,
                    max: 1,
                    kind: 'continuous',
                    values: null,
                },
            ],
        };
        const manifest = await compileFaustArtifact(request, directory);
        assert.equal(manifest.libfaustVersion, '2.88.0');
        assert.equal(manifest.compileOptions, '-ftz 2');
        const artifact = await loadFaustArtifact(
            manifest,
            async (path) => new Uint8Array(await readFile(join(directory, path))),
            packageVersion,
        );
        const quietSynth = await createOfflineSynth(artifact, { sampleRate: 44_100, blockSize: 128 });
        applyCanonicalPatch(quietSynth, manifest, { '/canonical/gain': 0 });
        const quiet = renderNote(quietSynth, {
            frames: 512,
            note: 60,
            velocity: 100,
            startFrame: 17,
            endFrame: 300,
        });
        const loudSynth = await createOfflineSynth(artifact, { sampleRate: 44_100, blockSize: 128 });
        applyCanonicalPatch(loudSynth, manifest, { '/canonical/gain': 1 });
        const loud = renderNote(loudSynth, {
            frames: 512,
            note: 60,
            velocity: 100,
            startFrame: 17,
            endFrame: 300,
        });
        assert.equal(Math.max(...quiet[0].map(Math.abs)), 0);
        assert.ok(Math.max(...loud[0].map(Math.abs)) > 0.1);
    } finally {
        await rm(directory, { recursive: true, force: true });
    }
});
