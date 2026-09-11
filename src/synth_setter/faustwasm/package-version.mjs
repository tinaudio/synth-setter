import { readFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const MODULE_DIRECTORY = dirname(fileURLToPath(import.meta.url));

export const FAUSTWASM_PACKAGE_ROOT = resolve(MODULE_DIRECTORY, 'vendor');
export const REPOSITORY_FAUSTWASM_PACKAGE_ROOT = resolve(
    MODULE_DIRECTORY,
    '../../../node_modules/@grame/faustwasm',
);

export const readFaustWasmPackageVersion = async (packageRoot = FAUSTWASM_PACKAGE_ROOT) => {
    const metadata = JSON.parse(
        await readFile(join(packageRoot, 'package.json'), 'utf8'),
    );
    if (typeof metadata.version !== 'string' || metadata.version.length === 0) {
        throw new Error('FaustWasm package metadata has no valid version');
    }
    return metadata.version;
};
