import { readFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

export const FAUSTWASM_PACKAGE_ROOT = resolve(
    dirname(fileURLToPath(import.meta.url)),
    '../../node_modules/@grame/faustwasm',
);

export const readFaustWasmPackageVersion = async () => {
    const metadata = JSON.parse(
        await readFile(join(FAUSTWASM_PACKAGE_ROOT, 'package.json'), 'utf8'),
    );
    if (typeof metadata.version !== 'string' || metadata.version.length === 0) {
        throw new Error('FaustWasm package metadata has no valid version');
    }
    return metadata.version;
};
