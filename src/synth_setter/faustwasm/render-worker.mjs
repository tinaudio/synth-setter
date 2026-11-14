import { readFile, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
    applyCanonicalPatch,
    createOfflineSynth,
    loadFaustArtifact,
    renderAudioInput,
    renderNote,
} from './runtime.mjs';
import { readFaustWasmPackageVersion } from './package-version.mjs';

export const main = async () => {
    const [, , manifestPath, requestPath, outputPath] = process.argv;
    if (!manifestPath || !requestPath || !outputPath) {
        throw new Error('usage: render-worker.mjs MANIFEST REQUEST OUTPUT');
    }
    const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
    const request = JSON.parse(await readFile(requestPath, 'utf8'));
    const packageVersion = await readFaustWasmPackageVersion();
    if (request.expectedFaustWasmVersion !== packageVersion) {
        throw new Error(
            `FaustWasm version mismatch: expected ${request.expectedFaustWasmVersion}, installed ${packageVersion}`,
        );
    }
    const baseDir = dirname(resolve(manifestPath));
    const artifact = await loadFaustArtifact(
        manifest,
        async (path) => new Uint8Array(await readFile(join(baseDir, path))),
        packageVersion,
    );
    const synth = await createOfflineSynth(artifact, request);
    applyCanonicalPatch(synth, manifest, request.params);
    let channels;
    if (request.inputFile) {
        const inputBytes = await readFile(join(dirname(resolve(requestPath)), request.inputFile));
        const expectedBytes = manifest.inputs * request.frames * Float32Array.BYTES_PER_ELEMENT;
        if (inputBytes.byteLength !== expectedBytes) {
            throw new Error(`input audio has ${inputBytes.byteLength} bytes; expected ${expectedBytes}`);
        }
        const values = new Float32Array(
            inputBytes.buffer,
            inputBytes.byteOffset,
            inputBytes.byteLength / Float32Array.BYTES_PER_ELEMENT,
        );
        const input = Array.from(
            { length: manifest.inputs },
            (_, channel) => values.subarray(channel * request.frames, (channel + 1) * request.frames),
        );
        channels = renderAudioInput(synth, input, request);
    } else {
        channels = renderNote(synth, request);
    }
    const bytes = Buffer.concat(
        channels.map((channel) => Buffer.from(channel.buffer, channel.byteOffset, channel.byteLength)),
    );
    await writeFile(outputPath, bytes);
};

if (resolve(process.argv[1] ?? '') === fileURLToPath(import.meta.url)) {
    await main();
}
