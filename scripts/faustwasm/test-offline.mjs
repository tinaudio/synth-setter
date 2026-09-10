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

const SOURCE = `declare name "contract";
import("stdfaust.lib");
gain = hslider("gain", 0.5, 0, 1, 0.01);
process = os.osc(440) * gain;
`;

test('compiled artifact loads and canonical patch changes real audio', async () => {
    const directory = await mkdtemp(join(tmpdir(), 'faustwasm-contract-'));
    try {
        const request = {
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
                },
            ],
        };
        const manifest = await compileFaustArtifact(request, directory);
        assert.equal(manifest.libfaustVersion, '2.88.0');
        assert.equal(manifest.compileOptions, '-ftz 2');
        const artifact = await loadFaustArtifact(
            manifest,
            async (path) => new Uint8Array(await readFile(join(directory, path))),
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
