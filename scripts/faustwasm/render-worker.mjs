import { readFile, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
    applyCanonicalPatch,
    createOfflineSynth,
    loadFaustArtifact,
    renderNote,
} from './runtime.mjs';
import { readFaustWasmPackageVersion } from './package-version.mjs';

const main = async () => {
    const [, , manifestPath, requestPath, outputPath] = process.argv;
    if (!manifestPath || !requestPath || !outputPath) {
        throw new Error('usage: render-worker.mjs MANIFEST REQUEST OUTPUT');
    }
    const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
    const request = JSON.parse(await readFile(requestPath, 'utf8'));
    const packageVersion = await readFaustWasmPackageVersion();
    if (request.expectedFaustWasmVersion !== packageVersion) {
        throw new Error('FaustWasm version mismatch');
    }
    const baseDir = dirname(resolve(manifestPath));
    const artifact = await loadFaustArtifact(
        manifest,
        async (path) => new Uint8Array(await readFile(join(baseDir, path))),
        packageVersion,
    );
    const synth = await createOfflineSynth(artifact, request);
    applyCanonicalPatch(synth, manifest, request.params);
    const channels = renderNote(synth, request);
    const bytes = Buffer.concat(
        channels.map((channel) => Buffer.from(channel.buffer, channel.byteOffset, channel.byteLength)),
    );
    await writeFile(outputPath, bytes);
};

if (resolve(process.argv[1] ?? '') === fileURLToPath(import.meta.url)) {
    await main();
}
