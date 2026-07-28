#!/usr/bin/env node

import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';

const defaultHelper = 'node_modules/@open-audio-stack/core/build/helpers/file.js';
const defaultManager = 'node_modules/@open-audio-stack/core/build/classes/ManagerLocal.js';
const helperTarget = resolve(process.argv[2] ?? defaultHelper);
const managerTarget = resolve(process.argv[3] ?? defaultManager);
const helperSource = readFileSync(helperTarget, 'utf8');
const managerSource = readFileSync(managerTarget, 'utf8');

const bundleBefore = "if (fileExists(path.join(f, 'Contents', 'Info.plist'))) {";
const bundleMarker = "path.extname(f).toLowerCase() === '.vst3'";
const bundleAfter =
  "if (path.extname(f).toLowerCase() === '.vst3' || " +
  "fileExists(path.join(f, 'Contents', 'Info.plist'))) {";
const dmgBefore = "const pkgs = dirRead(path.join(mountPoint, '**', '*.pkg'));";
const dmgMarker = "dirRead(path.join(mountPoint, '*.pkg'))";
const dmgAfter =
  "const pkgs = [\n" +
  "            ...dirRead(path.join(mountPoint, '*.pkg')),\n" +
  "            ...dirRead(path.join(mountPoint, '**', '*.pkg')),\n" +
  '        ];';
const elevationBefore = 'if (!isAdmin() && !isTests()) {';
const elevationMarker = 'files.every(file => file.type === FileType.Archive)';
const elevationAfter =
  'if (!isAdmin() && !isTests() && ' +
  '!files.every(file => file.type === FileType.Archive)) {';

if (
  (!helperSource.includes(bundleMarker) && !helperSource.includes(bundleBefore)) ||
  (!helperSource.includes(dmgMarker) && !helperSource.includes(dmgBefore))
) {
  console.error(`expected Studiorack helper source was not found in ${helperTarget}`);
  process.exit(1);
}
if (!managerSource.includes(elevationMarker) && !managerSource.includes(elevationBefore)) {
  console.error(`expected Studiorack manager source was not found in ${managerTarget}`);
  process.exit(1);
}

// Linux VST3 bundles need extension detection until open-audio-stack-core#82 ships.
// Root-level DMG packages need an explicit glob until open-audio-stack-core#84 ships.
const bundlePatched = helperSource.includes(bundleMarker)
  ? helperSource
  : helperSource.replace(bundleBefore, bundleAfter);
const patchedHelper = bundlePatched.includes(dmgMarker)
  ? bundlePatched
  : bundlePatched.replace(dmgBefore, dmgAfter);
// User-owned archives bypass elevation until open-audio-stack-core#83 ships.
const patchedManager = managerSource.includes(elevationMarker)
  ? managerSource
  : managerSource.replace(elevationBefore, elevationAfter);
writeFileSync(helperTarget, patchedHelper);
writeFileSync(managerTarget, patchedManager);
