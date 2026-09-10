import { pathToFileURL } from 'node:url';

import { main } from '../../src/synth_setter/faustwasm/export-artifacts.mjs';

export * from '../../src/synth_setter/faustwasm/export-artifacts.mjs';

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) await main();
