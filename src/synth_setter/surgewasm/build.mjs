#!/usr/bin/env node

import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { access, copyFile, mkdir, mkdtemp, readFile, realpath, rename, rm, writeFile } from 'node:fs/promises';
import { basename, dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const SOURCE_COMMIT = 'dd68c74346c828ef25bd6504867936648161b7a7';
const SOURCE_URL = 'https://github.com/surge-synthesizer/surge.git';
const EMSDK_VERSION = '6.0.9';
const VERSION = 'Surge XT 1.4.HEAD.dd68c743';
const scriptDirectory = dirname(fileURLToPath(import.meta.url));

const fail = (message) => {
    throw new Error(message);
};

const run = (command, args, options = {}) => new Promise((resolveRun, reject) => {
    const child = spawn(command, args, { stdio: 'inherit', ...options });
    child.on('error', (error) => reject(new Error(`cannot run ${command}: ${error.message}`)));
    child.on('exit', (code) => code === 0
        ? resolveRun()
        : reject(new Error(`${command} exited with status ${code}`)));
});

const output = (command, args, options = {}) => new Promise((resolveOutput, reject) => {
    const child = spawn(command, args, { ...options, stdio: ['ignore', 'pipe', 'pipe'] });
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (chunk) => { stdout += chunk; });
    child.stderr.on('data', (chunk) => { stderr += chunk; });
    child.on('error', (error) => reject(new Error(`cannot run ${command}: ${error.message}`)));
    child.on('exit', (code) => code === 0
        ? resolveOutput(stdout.trimEnd())
        : reject(new Error(`${command} exited with status ${code}: ${stderr.trim()}`)));
});

const parseArguments = (args) => {
    const parsed = {};
    for (let index = 0; index < args.length; index += 2) {
        const option = args[index];
        const value = args[index + 1];
        if (!['--cache', '--output', '--source'].includes(option) || !value) {
            fail('usage: node build.mjs (--source PATH | --cache PATH) --output PATH');
        }
        parsed[option.slice(2)] = resolve(value);
    }
    if (!parsed.output || Boolean(parsed.source) === Boolean(parsed.cache)) {
        fail('usage: node build.mjs (--source PATH | --cache PATH) --output PATH');
    }
    return parsed;
};

const pathExists = async (path) => {
    try {
        await access(path);
        return true;
    } catch (error) {
        if (error.code === 'ENOENT') return false;
        throw error;
    }
};

const prepareCache = async (cache) => {
    if (!await pathExists(cache)) {
        await mkdir(dirname(cache), { recursive: true });
        await run('git', ['clone', '--filter=blob:none', '--no-checkout', SOURCE_URL, cache]);
    } else {
        const root = await output('git', ['rev-parse', '--show-toplevel'], { cwd: cache });
        if (await realpath(root) !== await realpath(cache)) fail('cache must name a Git checkout root');
        const status = await output('git', ['status', '--porcelain'], { cwd: cache });
        if (status) fail(`cached source is dirty: ${cache}`);
    }
    await run('git', ['fetch', 'origin', SOURCE_COMMIT], { cwd: cache });
    await run('git', ['checkout', '--detach', SOURCE_COMMIT], { cwd: cache });
    await run('git', ['submodule', 'sync', '--recursive'], { cwd: cache });
    await run('git', ['submodule', 'update', '--init', '--recursive'], { cwd: cache });
    return cache;
};

const verifySource = async (source) => {
    if (await output('git', ['rev-parse', 'HEAD'], { cwd: source }) !== SOURCE_COMMIT) {
        fail(`Surge source must be exactly ${SOURCE_COMMIT}`);
    }
    if (await output('git', ['status', '--porcelain'], { cwd: source })) {
        fail(`Surge source is dirty: ${source}`);
    }
    const submodules = await output('git', ['submodule', 'status', '--recursive'], { cwd: source });
    for (const line of submodules.split('\n').filter(Boolean)) {
        if (!line.startsWith(' ')) fail(`submodule is not at its pinned commit: ${line}`);
        const path = line.slice(42).split(' ')[0];
        if (await output('git', ['status', '--porcelain'], { cwd: join(source, path) })) {
            fail(`Surge submodule is dirty: ${path}`);
        }
    }
};

const resolveToolchain = async () => {
    const emscripten = process.env.EMSDK
        ? join(process.env.EMSDK, 'upstream', 'emscripten')
        : null;
    const tools = {
        cmake: process.env.CMAKE ?? 'cmake',
        emcmake: emscripten ? join(emscripten, 'emcmake') : 'emcmake',
        emcc: emscripten ? join(emscripten, 'emcc') : 'emcc',
        emxx: emscripten ? join(emscripten, 'em++') : 'em++',
    };
    const version = await output(tools.emcc, ['--version']);
    if (!version.includes(` ${EMSDK_VERSION} `)) {
        fail(`emcc ${EMSDK_VERSION} is required`);
    }
    await output(tools.cmake, ['--version']);
    return tools;
};

const sha256 = async (path) => createHash('sha256').update(await readFile(path)).digest('hex');

const compileHost = async ({ source, stage, tools }) => {
    const build = join(source, 'build-wasm');
    await run(tools.emcmake, [tools.cmake, '-S', source, '-B', build, '-DCMAKE_BUILD_TYPE=Release']);
    await run(tools.cmake, ['--build', build, '--target', 'surge-wasm-demo', '--parallel', '4']);

    const hostObject = join(stage, 'host.c.o');
    await run(tools.emcc, [
        '-O3', '-DNDEBUG', '-flto=thin', '-fPIE', '-fvisibility=hidden',
        '-Wno-multichar', '-fno-math-errno', '-fno-trapping-math', '-Werror',
        '-msimd128', '-msse4.1', '-fwasm-exceptions', '-Wno-deprecated-pragma',
        '-Wno-unused-command-line-argument', '-Wno-deprecated-declarations',
        '-Werror=inconsistent-missing-override', '-Werror=logical-op-parentheses',
        '-Werror=dynamic-class-memaccess', '-Werror=undefined-bool-conversion',
        '-Werror=bitwise-op-parentheses', '-Werror=pointer-bool-conversion',
        '-Dsh_start=ss_upstream_start',
        '-include', join(source, 'src', 'surge-wasm', 'demo', 'demo-host.c'),
        '-I', join(source, 'src', 'surge-wasm', 'demo'),
        '-I', join(source, 'libs', 'clap-juce-extensions', 'clap-libs', 'clap', 'include'),
        '-c', join(scriptDirectory, 'host.c'), '-o', hostObject,
    ]);
    await run(tools.emxx, [
        '-O3', '-DNDEBUG', '-flto=thin', '-fwasm-exceptions',
        '-sMAIN_MODULE=1', '-sMODULARIZE=1', '-sEXPORT_ES6=1',
        '-sEXPORT_NAME=createSurgeHost', '-sALLOW_MEMORY_GROWTH', '-sSTACK_SIZE=2097152',
        '-sINCOMING_MODULE_JS_API=locateFile,wasmBinary',
        '-sEXPORTED_RUNTIME_METHODS=cwrap,FS,HEAPF32,HEAPU8',
        '-sEXPORTED_FUNCTIONS=_malloc,_free', '--no-entry', hostObject,
        '-o', join(stage, 'surge-host.mjs'),
    ]);

    const engineSource = join(build, 'surge_xt_products', 'wasm-demo', 'surge-xt.clap.wasm');
    await copyFile(engineSource, join(stage, 'surge-xt.clap.wasm'));
};

const writePackageMetadata = async ({ source, stage }) => {
    await copyFile(join(source, 'LICENSE'), join(stage, 'LICENSE.Surge-XT'));
    const files = {};
    for (const name of ['surge-host.mjs', 'surge-host.wasm', 'surge-xt.clap.wasm']) {
        files[name] = { sha256: await sha256(join(stage, name)) };
    }
    const manifest = {
        schemaVersion: 1,
        version: VERSION,
        source: { commit: SOURCE_COMMIT, url: SOURCE_URL },
        emsdkVersion: EMSDK_VERSION,
        files,
    };
    await writeFile(join(stage, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
    await writeFile(join(stage, 'SOURCE.md'), `# Reproducible Surge XT WASM source\n\nThis bundle is GPL-3.0-or-later software built from unmodified Surge XT commit\n\`${SOURCE_COMMIT}\` with recursive pinned submodules and emsdk ${EMSDK_VERSION}.\n\nFrom the synth-setter source tree, install and activate emsdk ${EMSDK_VERSION}, then run:\n\n\`\`\`sh\nnode src/synth_setter/surgewasm/build.mjs --cache /tmp/surge-wasm-source --output /tmp/surge-wasm-bundle\n\`\`\`\n\nUpstream source: ${SOURCE_URL}\n`);
};

const main = async () => {
    const args = parseArguments(process.argv.slice(2));
    if (await pathExists(args.output)) fail(`output directory already exists: ${args.output}`);
    const source = args.source ?? await prepareCache(args.cache);
    await verifySource(source);
    const tools = await resolveToolchain();

    await mkdir(dirname(args.output), { recursive: true });
    const stage = await mkdtemp(join(dirname(args.output), `${basename(args.output)}.tmp-`));
    try {
        await compileHost({ source, stage, tools });
        await writePackageMetadata({ source, stage });
        await rm(join(stage, 'host.c.o'));
        await rename(stage, args.output);
    } catch (error) {
        await rm(stage, { force: true, recursive: true });
        throw error;
    }
};

main().catch((error) => {
    console.error(`surgewasm build failed: ${error.message}`);
    process.exitCode = 1;
});
