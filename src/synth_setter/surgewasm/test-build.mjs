import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { mkdtemp, mkdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import test from 'node:test';

const execute = promisify(execFile);

test('build rejects a cache nested in another repository without using its Git state', async () => {
    const parent = await mkdtemp(join(tmpdir(), 'surge-build-root-'));
    try {
        await execute('git', ['init', parent]);
        const cache = join(parent, 'not-a-checkout');
        await mkdir(cache);
        await assert.rejects(
            execute(process.execPath, [fileURLToPath(new URL('./build.mjs', import.meta.url)), '--cache', cache, '--output', join(parent, 'output')]),
            (error) => {
                assert.match(error.stderr, /cache must name a Git checkout root/);
                return true;
            },
        );
        const { stdout } = await execute('git', ['-C', parent, 'status', '--porcelain']);
        assert.equal(stdout, '');
    } finally {
        await rm(parent, { recursive: true, force: true });
    }
});
