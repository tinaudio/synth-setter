const SOURCE_COMMIT = 'dd68c74346c828ef25bd6504867936648161b7a7';
const EMSDK_VERSION = '6.0.9';
const VERSION = 'Surge XT 1.4.HEAD.dd68c743';
const RENDER_QUANTUM = 32;
const FLOAT_BYTES = Float32Array.BYTES_PER_ELEMENT;

const requireInteger = (value, name, [minimum, maximum]) => {
    if (!Number.isInteger(value) || value < minimum || value > maximum) {
        throw new Error(`${name} must be an integer in [${minimum}, ${maximum}]`);
    }
};

const validateRequest = (request) => {
    const root = new URL(request.root);
    if (!root.pathname.endsWith('/')) root.pathname += '/';
    if (!(request.preset instanceof Uint8Array)) throw new Error('preset must be a Uint8Array');
    if (!Array.isArray(request.parameters)) throw new Error('parameters must be an array');
    if (request.loader !== undefined && typeof request.loader !== 'function') {
        throw new Error('loader must be an Emscripten module factory');
    }
    requireInteger(request.note, 'note', [0, 127]);
    requireInteger(request.velocity, 'velocity', [0, 127]);
    requireInteger(request.frames, 'frames', [1, Number.MAX_SAFE_INTEGER]);
    if (!Number.isFinite(request.sampleRate) || request.sampleRate <= 0) {
        throw new Error('sampleRate must be finite and positive');
    }
    const duration = request.frames / request.sampleRate;
    if (
        !Number.isFinite(request.noteStart)
        || !Number.isFinite(request.noteEnd)
        || request.noteStart < 0
        || request.noteStart > request.noteEnd
        || request.noteEnd > duration
    ) {
        throw new Error('note times must satisfy 0 <= noteStart <= noteEnd <= frames / sampleRate');
    }

    const ids = new Set();
    for (const parameter of request.parameters) {
        if (!parameter || typeof parameter !== 'object') throw new Error('parameter must be an object');
        requireInteger(parameter.id, 'parameter id', [0, 0xffffffff]);
        if (typeof parameter.name !== 'string' || parameter.name.length === 0) {
            throw new Error('parameter name must be a non-empty string');
        }
        if (!Number.isFinite(parameter.value) || parameter.value < 0 || parameter.value > 1) {
            throw new Error(`parameter ${parameter.id} value must be normalized to [0, 1]`);
        }
        if (ids.has(parameter.id)) throw new Error(`duplicate parameter id: ${parameter.id}`);
        ids.add(parameter.id);
    }
    return root;
};

const fetchBytes = async (url) => {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`failed to load ${url}: HTTP ${response.status}`);
    return new Uint8Array(await response.arrayBuffer());
};

const digest = async (bytes) => {
    const value = await crypto.subtle.digest('SHA-256', bytes);
    return [...new Uint8Array(value)].map((byte) => byte.toString(16).padStart(2, '0')).join('');
};

const loadManifest = async (root) => {
    const response = await fetch(new URL('manifest.json', root));
    if (!response.ok) throw new Error(`failed to load Surge manifest: HTTP ${response.status}`);
    const manifest = await response.json();
    if (
        manifest.schemaVersion !== 1
        || manifest.source?.commit !== SOURCE_COMMIT
        || manifest.emsdkVersion !== EMSDK_VERSION
        || manifest.version !== VERSION
    ) {
        throw new Error('unsupported Surge WASM bundle');
    }
    return manifest;
};

const loadFactory = async (root) => (await import(new URL('surge-host.mjs', root).href)).default;

const verifyArtifact = async (root, manifest, name) => {
    const bytes = await fetchBytes(new URL(name, root));
    if (await digest(bytes) !== manifest.files?.[name]?.sha256) {
        throw new Error(`Surge artifact digest mismatch: ${name}`);
    }
    return bytes;
};

const bindHost = (module) => ({
    destroy: module.cwrap('ss_destroy', null, []),
    load: module.cwrap('sh_load', 'number', ['string']),
    loadFxp: module.cwrap('ss_load_fxp', 'number', ['number', 'number']),
    note: module.cwrap('sh_note', null, ['number', 'number', 'number']),
    paramCount: module.cwrap('sh_param_count', 'number', []),
    paramId: module.cwrap('sh_param_id', 'number', ['number']),
    paramName: module.cwrap('sh_param_name', 'string', ['number']),
    pluginName: module.cwrap('sh_plugin_name', 'string', []),
    poll: module.cwrap('sh_poll', 'number', []),
    render: module.cwrap('sh_render', 'number', ['number', 'number', 'number']),
    setParam: module.cwrap('sh_set_param', null, ['number', 'number']),
    start: module.cwrap('sh_start', 'number', ['number', 'number']),
});

const verifyParameters = (host, parameters, parameterCount) => {
    const live = new Map();
    for (let index = 0; index < parameterCount; index += 1) {
        live.set(host.paramId(index) >>> 0, host.paramName(index));
    }
    for (const parameter of parameters) {
        const liveName = live.get(parameter.id);
        if (liveName === undefined) throw new Error(`unknown native Surge parameter id: ${parameter.id}`);
        if (liveName !== parameter.name) {
            throw new Error(`Surge parameter ${parameter.id} is named "${liveName}", not "${parameter.name}"`);
        }
    }
};

const floorToQuantum = (frame) => Math.floor(frame / RENDER_QUANTUM) * RENDER_QUANTUM;
const ceilToQuantum = (frame) => Math.ceil(frame / RENDER_QUANTUM) * RENDER_QUANTUM;

const renderAudio = ({ module, host, pointers }, request) => {
    const left = new Float32Array(request.frames);
    const right = new Float32Array(request.frames);
    const paddedFrames = ceilToQuantum(request.frames);
    const noteStartFrame = Math.floor(request.noteStart * request.sampleRate);
    const noteEndFrame = Math.min(Math.ceil(request.noteEnd * request.sampleRate), request.frames);
    const noteStart = floorToQuantum(noteStartFrame);
    const noteEnd = request.noteStart === request.noteEnd
        ? noteStart
        : ceilToQuantum(noteEndFrame);

    for (let blockStart = 0; blockStart < paddedFrames; blockStart += RENDER_QUANTUM) {
        if (blockStart === noteStart) host.note(request.note, request.velocity, 1);
        if (blockStart === noteEnd) host.note(request.note, 0, 0);
        if (!host.render(pointers.left, pointers.right, RENDER_QUANTUM)) {
            throw new Error(`Surge render failed at frame ${blockStart}`);
        }
        const count = Math.min(RENDER_QUANTUM, request.frames - blockStart);
        if (count > 0) {
            const leftBlock = module.HEAPF32.subarray(
                pointers.left / FLOAT_BYTES,
                pointers.left / FLOAT_BYTES + count,
            );
            const rightBlock = module.HEAPF32.subarray(
                pointers.right / FLOAT_BYTES,
                pointers.right / FLOAT_BYTES + count,
            );
            left.set(leftBlock, blockStart);
            right.set(rightBlock, blockStart);
        }
        host.poll();
    }
    return { left, right };
};

export const renderSurge = async (request) => {
    const root = validateRequest(request);
    const manifest = await loadManifest(root);
    const [, hostWasm, engineBytes] = await Promise.all([
        verifyArtifact(root, manifest, 'surge-host.mjs'),
        verifyArtifact(root, manifest, 'surge-host.wasm'),
        verifyArtifact(root, manifest, 'surge-xt.clap.wasm'),
    ]);
    const createSurgeHost = request.loader ?? await loadFactory(root);
    const module = await createSurgeHost({
        locateFile: (path) => path === 'surge-host.wasm'
            ? new URL('surge-host.wasm', root).href
            : new URL(path, root).href,
        wasmBinary: hostWasm,
    });
    module.FS.writeFile('/surge-xt.clap.wasm', engineBytes);
    const host = bindHost(module);
    const pointers = { left: 0, preset: 0, right: 0 };

    try {
        if (!host.load('/surge-xt.clap.wasm')) throw new Error('Surge CLAP load failed');
        pointers.preset = module._malloc(request.preset.byteLength);
        module.HEAPU8.set(request.preset, pointers.preset);
        if (!host.loadFxp(pointers.preset, request.preset.byteLength)) {
            throw new Error('invalid FXP or Surge state load failed');
        }

        const parameterCount = host.paramCount();
        verifyParameters(host, request.parameters, parameterCount);
        for (const parameter of request.parameters) host.setParam(parameter.id, parameter.value);
        if (!host.start(request.sampleRate, RENDER_QUANTUM)) throw new Error('Surge start failed');

        pointers.left = module._malloc(RENDER_QUANTUM * FLOAT_BYTES);
        pointers.right = module._malloc(RENDER_QUANTUM * FLOAT_BYTES);
        const audio = renderAudio({ module, host, pointers }, request);
        return {
            ...audio,
            engineCommit: manifest.source.commit,
            parameterCount,
            version: host.pluginName(),
        };
    } finally {
        host.destroy();
        for (const pointer of Object.values(pointers)) {
            if (pointer) module._free(pointer);
        }
    }
};
