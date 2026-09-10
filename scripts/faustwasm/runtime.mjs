import {
    FaustMonoDspGenerator,
    FaustPolyDspGenerator,
} from '../../node_modules/@grame/faustwasm/dist/esm/index.js';

const PACKAGE_VERSION = '0.18.3';

const verifyBytes = async (entry, loadBytes) => {
    const bytes = await loadBytes(entry.path);
    const digest = await crypto.subtle.digest('SHA-256', bytes);
    const actual = [...new Uint8Array(digest)]
        .map((value) => value.toString(16).padStart(2, '0'))
        .join('');
    if (actual !== entry.sha256) throw new Error(`artifact digest mismatch: ${entry.path}`);
    return bytes;
};

const makeFactory = async (bytes, meta, poly) => ({
    code: bytes,
    module: await WebAssembly.compile(bytes),
    json: JSON.stringify(meta),
    poly,
    shaKey: '',
    soundfiles: {},
});

export const loadFaustArtifact = async (manifest, loadBytes) => {
    if (manifest.schemaVersion !== 1) throw new Error('unsupported Faust artifact schema');
    if (manifest.faustwasmVersion !== PACKAGE_VERSION) throw new Error('FaustWasm version mismatch');
    const dspBytes = await verifyBytes(manifest.files.dsp, loadBytes);
    const dspFactory = await makeFactory(dspBytes, manifest.dspMeta, manifest.mode === 'poly');
    const artifact = { manifest, dspFactory };
    if (manifest.files.mixer) {
        artifact.mixerModule = await WebAssembly.compile(
            await verifyBytes(manifest.files.mixer, loadBytes),
        );
    }
    if (manifest.files.effect) {
        artifact.effectFactory = await makeFactory(
            await verifyBytes(manifest.files.effect, loadBytes),
            manifest.effectMeta,
            false,
        );
    }
    return artifact;
};

export const createOfflineSynth = async (artifact, { sampleRate, blockSize }) => {
    if (!Number.isFinite(sampleRate) || sampleRate <= 0) {
        throw new Error('sampleRate must be finite and positive');
    }
    if (!Number.isInteger(blockSize) || blockSize < 1) {
        throw new Error('blockSize must be a positive integer');
    }
    const { manifest } = artifact;
    if (manifest.mode === 'poly') {
        const generator = new FaustPolyDspGenerator();
        const processor = await generator.createOfflineProcessor(
            sampleRate,
            blockSize,
            manifest.voices,
            artifact.dspFactory,
            artifact.mixerModule,
            artifact.effectFactory ?? null,
        );
        return { processor, blockSize, manifest };
    }
    const generator = new FaustMonoDspGenerator();
    const processor = await generator.createOfflineProcessor(
        sampleRate,
        blockSize,
        artifact.dspFactory,
    );
    return { processor, blockSize, manifest };
};

export const applyCanonicalPatch = (synth, manifest, params) => {
    const expected = new Set(manifest.parameters.map((parameter) => parameter.canonicalAddress));
    const supplied = new Set(Object.keys(params));
    const unknown = [...supplied].filter((address) => !expected.has(address));
    const missing = [...expected].filter((address) => !supplied.has(address));
    if (unknown.length) throw new Error(`unknown canonical parameter(s): ${unknown.join(', ')}`);
    if (missing.length) throw new Error(`missing canonical parameter(s): ${missing.join(', ')}`);
    for (const parameter of manifest.parameters) {
        const value = params[parameter.canonicalAddress];
        if (!Number.isFinite(value) || value < parameter.min || value > parameter.max) {
            throw new Error(`parameter outside native domain: ${parameter.canonicalAddress}`);
        }
        if (
            parameter.kind === 'discrete'
            && (!Array.isArray(parameter.values) || !parameter.values.includes(value))
        ) {
            throw new Error(`parameter outside discrete native domain: ${parameter.canonicalAddress}`);
        }
        synth.processor.setParamValue(parameter.wasmAddress, value);
    }
};

export const renderNote = (synth, { frames, note, velocity, startFrame, endFrame }) => {
    if (!Number.isInteger(frames) || frames < 0) {
        throw new Error('frames must be a non-negative integer');
    }
    if (!Number.isInteger(startFrame) || !Number.isInteger(endFrame)) {
        throw new Error('note frames must be integers');
    }
    if (!(0 <= startFrame && startFrame < endFrame && endFrame <= frames)) {
        throw new Error('note frames must satisfy 0 <= start < end <= frames');
    }
    const { processor, blockSize, manifest } = synth;
    if (!Number.isInteger(blockSize) || blockSize < 1) {
        throw new Error('blockSize must be a positive integer');
    }
    if (!Number.isInteger(manifest.outputs) || manifest.outputs < 1) {
        throw new Error('manifest outputs must be a positive integer');
    }
    const output = Array.from({ length: manifest.outputs }, () => new Float32Array(frames));
    processor.start();
    for (let blockStart = 0; blockStart < frames; blockStart += blockSize) {
        const count = Math.min(blockSize, frames - blockStart);
        const blockOutput = Array.from(
            { length: manifest.outputs },
            () => new Float32Array(blockSize),
        );
        const events = [];
        if (manifest.mode === 'poly' && startFrame >= blockStart && startFrame < blockStart + blockSize) {
            events.push({
                frame: startFrame - blockStart,
                apply: () => processor.keyOn(0, note, velocity),
            });
        }
        if (manifest.mode === 'poly' && endFrame >= blockStart && endFrame < blockStart + blockSize) {
            events.push({
                frame: endFrame - blockStart,
                apply: () => processor.keyOff(0, note, 0),
            });
        }
        processor.fDSPCode.compute([], blockOutput, events);
        for (let channel = 0; channel < output.length; channel += 1) {
            output[channel].set(blockOutput[channel].subarray(0, count), blockStart);
        }
    }
    processor.stop();
    return output;
};
