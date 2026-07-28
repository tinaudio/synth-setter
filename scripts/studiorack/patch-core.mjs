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
const tarBefore = 'else if (isTarFile) {';
const tarMarker = 'else if (isTarFile) {\n        dirCreate(dirPath);';
const tarAfter = 'else if (isTarFile) {\n        dirCreate(dirPath);';
const dmgBefore = "const pkgs = dirRead(path.join(mountPoint, '**', '*.pkg'));";
const dmgMarker = 'readdirSync(mountPoint)';
const dmgAfter =
  "const rootPkgs = readdirSync(mountPoint)\n" +
  "            .filter(entry => path.extname(entry).toLowerCase() === '.pkg')\n" +
  '            .map(entry => path.join(mountPoint, entry));\n' +
  "        const pkgs = [...rootPkgs, ...dirRead(path.join(mountPoint, '**', '*.pkg'))];";
const filesDeclarationBefore = 'const files = packageCompatibleFiles(';
const filesDeclarationMarker = 'let files = packageCompatibleFiles(';
const elevationBefore = 'if (!isAdmin() && !isTests()) {';
const elevationMarker = 'files.every(file => file.type === FileType.Archive)';
const elevationAfter =
  'if (!isAdmin() && files.some(file => file.type === FileType.Archive)) {\n' +
  '            files = files.filter(file => file.type === FileType.Archive);\n' +
  '        }\n' +
  '        if (!isAdmin() && !isTests() && ' +
  '!files.every(file => file.type === FileType.Archive)) {';

if (
  (!helperSource.includes(bundleMarker) && !helperSource.includes(bundleBefore)) ||
  (!helperSource.includes(tarMarker) && !helperSource.includes(tarBefore)) ||
  (!helperSource.includes(dmgMarker) && !helperSource.includes(dmgBefore))
) {
  console.error(`expected Studiorack helper source was not found in ${helperTarget}`);
  process.exit(1);
}
if (
  (!managerSource.includes(filesDeclarationMarker) &&
    !managerSource.includes(filesDeclarationBefore)) ||
  (!managerSource.includes(elevationMarker) && !managerSource.includes(elevationBefore))
) {
  console.error(`expected Studiorack manager source was not found in ${managerTarget}`);
  process.exit(1);
}

// Linux VST3 bundles need extension detection until open-audio-stack-core#82 ships.
// Tar extraction needs an existing target until open-audio-stack-core#85 ships.
// Root-level DMG packages need direct enumeration until open-audio-stack-core#84 ships.
const bundlePatched = helperSource.includes(bundleMarker)
  ? helperSource
  : helperSource.replace(bundleBefore, bundleAfter);
const tarPatched = bundlePatched.includes(tarMarker)
  ? bundlePatched
  : bundlePatched.replace(tarBefore, tarAfter);
const patchedHelper = tarPatched.includes(dmgMarker)
  ? tarPatched
  : tarPatched.replace(dmgBefore, dmgAfter);
// User-owned archives bypass elevation until open-audio-stack-core#83 ships.
const declarationPatched = managerSource.includes(filesDeclarationMarker)
  ? managerSource
  : managerSource.replace(filesDeclarationBefore, filesDeclarationMarker);
const patchedManager = declarationPatched.includes(elevationMarker)
  ? declarationPatched
  : declarationPatched.replace(elevationBefore, elevationAfter);
writeFileSync(helperTarget, patchedHelper);
writeFileSync(managerTarget, patchedManager);
