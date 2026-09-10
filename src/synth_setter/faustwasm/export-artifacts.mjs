import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
    FaustCompiler,
    FaustMonoDspGenerator,
    FaustPolyDspGenerator,
    LibFaust,
    instantiateFaustModuleFromFile,
} from './vendor/faustwasm.mjs';

export const FAUSTWASM_VERSION = '0.18.3';
export const COMPILE_OPTIONS = '-ftz 2';

const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');

const flattenInputs = (groups, result = []) => {
    for (const item of groups) {
        if (item.type === 'vgroup' || item.type === 'hgroup' || item.type === 'tgroup') {
            flattenInputs(item.items, result);
        } else if (
            'address' in item
            && ['button', 'checkbox', 'hslider', 'vslider', 'nentry'].includes(item.type)
        ) {
            result.push(item);
        }
    }
    return result;
};

const writeArtifactFile = async (outputDir, name, bytes) => {
    const path = `${name}.wasm`;
    await writeFile(join(outputDir, path), bytes);
    return { path, sha256: sha256(bytes) };
};

export const compileFaustArtifact = async (request, outputDir) => {
    const moduleDirectory = dirname(fileURLToPath(import.meta.url));
    const packagedRoot = resolve(moduleDirectory, 'vendor');
    const packagedCompiler = join(packagedRoot, 'libfaust-wasm/libfaust-wasm.wasm');
    const packageRoot = existsSync(packagedCompiler)
        ? packagedRoot
        : resolve(moduleDirectory, '../../../node_modules/@grame/faustwasm');
    const modulePath = join(packageRoot, 'libfaust-wasm/libfaust-wasm.js');
    const faustModule = await instantiateFaustModuleFromFile(modulePath);
    const libFaust = new LibFaust(faustModule);
    const compiler = new FaustCompiler(libFaust);
    const poly = request.mode === 'poly';
    const generator = poly ? new FaustPolyDspGenerator() : new FaustMonoDspGenerator();
    const compiled = await generator.compile(compiler, request.identity, request.source, COMPILE_OPTIONS);
    if (!compiled) throw new Error(`Faust compilation failed: ${compiler.getErrorMessage()}`);

    const dspFactory = poly ? generator.voiceFactory : generator.factory;
    if (!dspFactory?.code) throw new Error('Faust compiler returned no DSP bytecode');
    const dspMeta = JSON.parse(dspFactory.json);
    const compiledMeta = generator.getMeta();
    const nativeOutputs = compiledMeta.outputs;
    if (!Number.isInteger(nativeOutputs) || nativeOutputs < 1) {
        throw new Error('compiled output count must be a positive integer');
    }
    if (request.expectedOutputs !== null && request.expectedOutputs !== undefined) {
        if (!Number.isInteger(request.expectedOutputs) || request.expectedOutputs < 1) {
            throw new Error('expectedOutputs must be a positive integer');
        }
        if (nativeOutputs !== request.expectedOutputs) {
            throw new Error('compiled output count differs');
        }
    }
    const descriptors = new Map(flattenInputs(compiledMeta.ui).map((item) => [item.address, item]));
    for (const parameter of request.parameters) {
        const descriptor = descriptors.get(parameter.wasmAddress);
        if (!descriptor) throw new Error(`compiled DSP is missing ${parameter.wasmAddress}`);
        const discrete = descriptor.type === 'button' || descriptor.type === 'checkbox';
        const minimum = discrete ? 0 : descriptor.min;
        const maximum = discrete ? 1 : descriptor.max;
        const kind = discrete ? 'discrete' : 'continuous';
        if (minimum !== parameter.min || maximum !== parameter.max || kind !== parameter.kind) {
            throw new Error(`compiled domain differs for ${parameter.wasmAddress}`);
        }
        if (discrete) {
            if (
                !Array.isArray(parameter.values)
                || parameter.values.length !== 2
                || parameter.values[0] !== 0
                || parameter.values[1] !== 1
            ) {
                throw new Error(`compiled discrete values differ for ${parameter.wasmAddress}`);
            }
        } else if (parameter.values !== null) {
            throw new Error(`continuous parameter has discrete values: ${parameter.wasmAddress}`);
        }
    }
    const expectedWasm = new Set([
        ...request.parameters.map((parameter) => parameter.wasmAddress),
        ...request.reservedWasmAddresses,
    ]);
    const unexpected = [...descriptors.keys()].filter((address) => !expectedWasm.has(address));
    const absent = [...expectedWasm].filter((address) => !descriptors.has(address));
    if (unexpected.length) throw new Error(`unmapped compiled parameter(s): ${unexpected.join(', ')}`);
    if (absent.length) throw new Error(`missing compiled parameter(s): ${absent.join(', ')}`);

    await mkdir(outputDir, { recursive: true });
    const files = { dsp: await writeArtifactFile(outputDir, 'dsp', dspFactory.code) };
    let effectMeta = null;
    if (poly) {
        files.mixer = await writeArtifactFile(outputDir, 'mixer', generator.mixerBuffer);
        if (generator.effectFactory?.code) {
            files.effect = await writeArtifactFile(outputDir, 'effect', generator.effectFactory.code);
            effectMeta = JSON.parse(generator.effectFactory.json);
        }
    }
    const manifest = {
        schemaVersion: 1,
        identity: request.identity,
        faustwasmVersion: FAUSTWASM_VERSION,
        libfaustVersion: libFaust.version(),
        compileOptions: COMPILE_OPTIONS,
        sourceSha256: sha256(request.source),
        mode: request.mode,
        voices: request.voices,
        outputs: nativeOutputs,
        parameters: request.parameters,
        files,
        dspMeta,
        ...(effectMeta ? { effectMeta } : {}),
    };
    await writeFile(join(outputDir, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
    return manifest;
};

export const main = async () => {
    const [, , requestPath, outputDir] = process.argv;
    if (!requestPath || !outputDir) throw new Error('usage: export-artifacts.mjs REQUEST_JSON OUTPUT_DIR');
    const request = JSON.parse(await readFile(requestPath, 'utf8'));
    await compileFaustArtifact(request, outputDir);
};

if (resolve(process.argv[1] ?? '') === fileURLToPath(import.meta.url)) {
    await main();
}
