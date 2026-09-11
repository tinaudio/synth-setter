#!/usr/bin/env node

import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const defaultHelper =
  "node_modules/@open-audio-stack/core/build/helpers/file.js";
const defaultManager =
  "node_modules/@open-audio-stack/core/build/classes/ManagerLocal.js";
const defaultAdmin =
  "node_modules/@open-audio-stack/core/build/helpers/admin.js";

/**
 * @typedef {object} SourcePatch
 * @property {boolean} [allowPartialMarkers] Whether a known migration may have partial markers.
 * @property {string[]} markers Strings proving the patch is fully applied.
 * @property {(source: string, target: string) => string} mutate Apply the patch to validated source.
 * @property {string} name Human-readable patch name used in failures.
 * @property {string[]} preconditions Supported unpatched source strings.
 */

/**
 * Stop patching before any target files are written.
 * @param {string} message Actionable patch failure.
 * @returns {never}
 */
function fail(message) {
  throw new Error(message);
}

/**
 * Count exact, non-overlapping source occurrences.
 * @param {string} source Source text to inspect.
 * @param {string} marker Nonempty source marker.
 * @returns {number} Marker occurrence count.
 */
function countOccurrences(source, marker) {
  if (!marker) fail("source marker must be nonempty");
  let count = 0;
  let offset = 0;
  while ((offset = source.indexOf(marker, offset)) !== -1) {
    count += 1;
    offset += marker.length;
  }
  return count;
}

/**
 * @typedef {object} UniqueMatchOptions
 * @property {string[]} candidates Supported source strings.
 * @property {string} patchName Human-readable patch name.
 * @property {string} source Source text to inspect.
 * @property {string} target Source file path for diagnostics.
 */

/**
 * @typedef {object} UniqueReplacementOptions
 * @property {string[]} candidates Supported source strings.
 * @property {string} patchName Human-readable patch name.
 * @property {string} replacement Replacement source text.
 * @property {string} source Source text to patch.
 * @property {string} target Source file path for diagnostics.
 */

/**
 * Select the sole supported source precondition or fail closed.
 * @param {UniqueMatchOptions} options Match inputs and diagnostic context.
 * @returns {string} The unique matching source string.
 */
function findUniqueMatch({ candidates, patchName, source, target }) {
  const matches = candidates.flatMap((candidate) =>
    Array(countOccurrences(source, candidate)).fill(candidate),
  );
  if (matches.length === 0) {
    fail(`expected ${patchName} source was not found in ${target}`);
  }
  if (matches.length > 1) {
    fail(
      `ambiguous ${patchName} source in ${target}: found ${matches.length} matches`,
    );
  }
  return matches[0];
}

/**
 * Replace one validated source precondition.
 * @param {UniqueReplacementOptions} options Replacement inputs and diagnostic context.
 * @returns {string} Patched source text.
 */
function replaceUnique({ candidates, patchName, replacement, source, target }) {
  const matched = findUniqueMatch({ candidates, patchName, source, target });
  if (matched === undefined) {
    fail(`expected ${patchName} source was not found in ${target}`);
  }
  return source.replace(matched, replacement);
}

/**
 * Apply an ordered patch set without writing partially patched files.
 * @param {string} source Original source text.
 * @param {string} target Source file path for diagnostics.
 * @param {SourcePatch[]} patches Ordered source patches.
 * @returns {string} Fully patched source text.
 */
function applyPatches(source, target, patches) {
  let patched = source;
  for (const patch of patches) {
    const markerCounts = patch.markers.map((marker) =>
      countOccurrences(patched, marker),
    );
    const ambiguousMarker = markerCounts.find((count) => count > 1);
    if (ambiguousMarker !== undefined) {
      fail(
        `ambiguous ${patch.name} marker in ${target}: found ${ambiguousMarker} matches`,
      );
    }
    if (markerCounts.every((count) => count === 1)) continue;

    const hasPartialMarkers = markerCounts.some((count) => count === 1);
    const hasPrecondition = patch.preconditions.some(
      (text) => countOccurrences(patched, text) > 0,
    );
    if ((hasPartialMarkers && !patch.allowPartialMarkers) || !hasPrecondition) {
      fail(`expected ${patch.name} source was not found in ${target}`);
    }
    patched = patch.mutate(patched, target);
    const patchedMarkerCounts = patch.markers.map((marker) =>
      countOccurrences(patched, marker),
    );
    if (!patchedMarkerCounts.every((count) => count === 1)) {
      fail(`failed to apply ${patch.name} marker in ${target}`);
    }
  }
  return patched;
}

/**
 * Build a patch that replaces one exact upstream source fragment.
 * @param {object} options Patch definition.
 * @param {string[]} [options.alternatives] Additional supported source fragments.
 * @param {string} options.after Replacement source text.
 * @param {string} options.before Primary upstream source fragment.
 * @param {string} options.marker Marker proving the replacement is applied.
 * @param {string} options.name Human-readable patch name.
 * @returns {SourcePatch} Exact replacement patch.
 */
function replacementPatch({ name, before, after, marker, alternatives = [] }) {
  const candidates = [before, ...alternatives];
  return {
    name,
    markers: [marker],
    preconditions: candidates,
    mutate: (source, target) =>
      replaceUnique({
        candidates,
        patchName: name,
        replacement: after,
        source,
        target,
      }),
  };
}

/** @returns {SourcePatch[]} Filesystem helper compatibility patches. */
function helperPatches() {
  const bundleBefore =
    "if (fileExists(path.join(f, 'Contents', 'Info.plist'))) {";
  const bundleMarker = "path.extname(f).toLowerCase() === '.vst3'";
  const tarBefore = "else if (isTarFile) {";
  const tarMarker = "else if (isTarFile) {\n        dirCreate(dirPath);";
  const dmgBefore =
    "const pkgs = dirRead(path.join(mountPoint, '**', '*.pkg'));";
  const dmgMarker = "readdirSync(mountPoint)";
  return [
    replacementPatch({
      name: "Linux VST3 bundle detection",
      before: bundleBefore,
      marker: bundleMarker,
      after:
        "if (path.extname(f).toLowerCase() === '.vst3' || " +
        "fileExists(path.join(f, 'Contents', 'Info.plist'))) {",
    }),
    replacementPatch({
      name: "tar destination creation",
      before: tarBefore,
      marker: tarMarker,
      after: tarMarker,
    }),
    replacementPatch({
      name: "root DMG package discovery",
      before: dmgBefore,
      marker: dmgMarker,
      after:
        "const rootPkgs = readdirSync(mountPoint)\n" +
        "            .filter(entry => path.extname(entry).toLowerCase() === '.pkg')\n" +
        "            .map(entry => path.join(mountPoint, entry));\n" +
        "        const pkgs = [...rootPkgs, ...dirRead(path.join(mountPoint, '**', '*.pkg'))];",
    }),
  ];
}

/** @returns {string} Strict runtime validation inserted at the JSON trust boundary. */
function artifactLockValidationSource() {
  return [
    "        const artifactLockFailure = message => {",
    "            throw new Error(`invalid artifact lock ${artifactLockPath}: ${message}`);",
    "        };",
    "        const isRecord = value => value !== null && typeof value === 'object' && !Array.isArray(value);",
    "        const requireExactFields = (value, expectedFields, context) => {",
    "            if (!isRecord(value))",
    "                artifactLockFailure(`${context} must be an object`);",
    "            const actualFields = Object.keys(value).sort();",
    "            if (actualFields.length !== expectedFields.length ||",
    "                actualFields.some((field, index) => field !== expectedFields[index]))",
    "                artifactLockFailure(`${context} must have exactly fields: ${expectedFields.join(', ')}`);",
    "        };",
    "        const requireSupportedList = (value, supportedValues, context) => {",
    "            if (!Array.isArray(value) || value.length === 0)",
    "                artifactLockFailure(`${context} must be a nonempty array`);",
    "            const seen = new Set();",
    "            for (const item of value) {",
    "                if (typeof item !== 'string' || !supportedValues.has(item))",
    "                    artifactLockFailure(`${context} contains unsupported value ${JSON.stringify(item)}`);",
    "                if (seen.has(item))",
    "                    artifactLockFailure(`${context} contains duplicate value ${JSON.stringify(item)}`);",
    "                seen.add(item);",
    "            }",
    "        };",
    "        const supportedArchitectures = new Set(['arm32', 'arm64', 'arm64ec', 'x32', 'x64']);",
    "        const supportedSystems = new Set(['linux', 'mac', 'win']);",
    "        const supportedTypes = new Set(Object.values(FileType));",
    "        const validateArtifact = (artifact, context) => {",
    "            requireExactFields(artifact, ['architectures', 'sha256', 'systems', 'type', 'url'], context);",
    "            requireSupportedList(artifact.architectures, supportedArchitectures, `${context}.architectures`);",
    "            requireSupportedList(artifact.systems, supportedSystems, `${context}.systems`);",
    "            if (typeof artifact.type !== 'string' || !supportedTypes.has(artifact.type))",
    "                artifactLockFailure(`${context}.type contains unsupported value ${JSON.stringify(artifact.type)}`);",
    "            if (typeof artifact.sha256 !== 'string' || !/^[0-9a-f]{64}$/.test(artifact.sha256))",
    "                artifactLockFailure(`${context}.sha256 must be 64 lowercase hexadecimal characters`);",
    "            if (typeof artifact.url !== 'string')",
    "                artifactLockFailure(`${context}.url must be a valid HTTPS URL`);",
    "            let artifactUrl;",
    "            try {",
    "                artifactUrl = new URL(artifact.url);",
    "            } catch {",
    "                artifactLockFailure(`${context}.url must be a valid HTTPS URL`);",
    "            }",
    "            if (artifactUrl === undefined || artifactUrl.protocol !== 'https:' || !artifactUrl.hostname)",
    "                artifactLockFailure(`${context}.url must be a valid HTTPS URL`);",
    "        };",
    "        const normalizeArtifact = file => ({",
    "            architectures: [...file.architectures].sort(),",
    "            sha256: file.sha256,",
    "            systems: file.systems.map(value => typeof value === 'string' ? value : value.type).sort(),",
    "            type: file.type,",
    "            url: file.url,",
    "        });",
    "        const artifactIdentity = artifact => JSON.stringify(normalizeArtifact(artifact));",
    "        const validateArtifactLock = lock => {",
    "            if (!isRecord(lock))",
    "                artifactLockFailure('artifact lock root must be an object');",
    "            for (const [reference, entry] of Object.entries(lock)) {",
    "                const versionSeparator = reference.lastIndexOf('@');",
    "                const packageName = reference.slice(0, versionSeparator).trim();",
    "                const packageVersion = reference.slice(versionSeparator + 1).trim();",
    "                if (versionSeparator <= 0 || !packageName || !packageVersion)",
    "                    artifactLockFailure(`package reference ${JSON.stringify(reference)} must contain a nonempty package and version`);",
    "                requireExactFields(entry, ['artifacts'], reference);",
    "                if (!Array.isArray(entry.artifacts) || entry.artifacts.length === 0)",
    "                    artifactLockFailure(`${reference}.artifacts must be a nonempty array`);",
    "                const seenArtifacts = new Set();",
    "                entry.artifacts.forEach((artifact, index) => {",
    "                    validateArtifact(artifact, `${reference}.artifacts[${index}]`);",
    "                    const identity = artifactIdentity(artifact);",
    "                    if (seenArtifacts.has(identity))",
    "                        artifactLockFailure(`${reference}.artifacts contains duplicate artifact at index ${index}`);",
    "                    seenArtifacts.add(identity);",
    "                });",
    "            }",
    "            return lock;",
    "        };",
    "        let parsedArtifactLock;",
    "        try {",
    "            parsedArtifactLock = JSON.parse(readFileSync(artifactLockPath, 'utf8'));",
    "        } catch (error) {",
    "            const detail = error instanceof Error ? error.message : String(error);",
    "            throw new Error(`invalid artifact lock JSON at ${artifactLockPath}: ${detail}`);",
    "        }",
    "        const artifactLock = validateArtifactLock(parsedArtifactLock);",
  ].join("\n");
}

/** @returns {SourcePatch} Runtime artifact-lock enforcement patch. */
function artifactLockPatch() {
  const installedBlock =
    "        if (this.isPackageInstalled(slug, versionNum)) {\n" +
    "            this.log(`Package ${slug} version ${versionNum} already installed`);\n" +
    "            pkgVersion.installed = true;\n" +
    "            return pkgVersion;\n" +
    "        }\n";
  const preferenceBlock =
    "        if (files.some(file => file.type === FileType.Archive)) {\n" +
    "            files = files.filter(file => file.type === FileType.Archive);\n" +
    "        }\n";
  const artifactPathMarker =
    "const artifactLockPath = process.env.SYNTH_SETTER_STUDIORACK_ARTIFACT_LOCK || this.config.get('artifactLockPath');";
  const legacyArtifactPathMarker =
    "const artifactLockPath = this.config.get('artifactLockPath') || process.env.SYNTH_SETTER_STUDIORACK_ARTIFACT_LOCK;";
  const comparisonMarker = "artifact lock mismatch for ${packageReference}";
  const preferenceMarker = "const preferArchives = artifacts =>";
  const validationMarker = "const artifactIdentity = artifact =>";
  const legacyParseBlock =
    "        const artifactLock = JSON.parse(readFileSync(artifactLockPath, 'utf8'));\n";
  const legacySelectionBlock =
    "        const lockedSelection = canonicalArtifacts(lockedArtifacts.filter(file =>\n" +
    "            file.architectures.includes(architecture) && file.systems.includes(system)));\n" +
    "        const liveSelection = canonicalArtifacts(files);\n";
  const preferredSelectionBlock =
    `        ${preferenceMarker} artifacts.some(file => file.type === FileType.Archive)\n` +
    "            ? artifacts.filter(file => file.type === FileType.Archive)\n" +
    "            : artifacts;\n" +
    "        const lockedCompatibleArtifacts = lockedArtifacts.filter(file =>\n" +
    "            file.architectures.includes(architecture) && file.systems.includes(system));\n" +
    "        const lockedSelection = canonicalArtifacts(preferArchives(lockedCompatibleArtifacts));\n" +
    "        const liveSelection = canonicalArtifacts(preferArchives(files));\n";
  const artifactBlock =
    `        ${artifactPathMarker}\n` +
    "        if (!artifactLockPath)\n" +
    "            throw new Error('Studiorack artifact lock path is required');\n" +
    artifactLockValidationSource() +
    "\n        const packageReference = `${slug}@${versionNum}`;\n" +
    "        const lockedArtifacts = artifactLock[packageReference]?.artifacts;\n" +
    "        if (!Array.isArray(lockedArtifacts))\n" +
    "            throw new Error(`artifact lock missing ${packageReference}`);\n" +
    "        const canonicalArtifacts = artifacts => artifacts\n" +
    "            .map(normalizeArtifact)\n" +
    "            .sort((left, right) => JSON.stringify(left).localeCompare(JSON.stringify(right)));\n" +
    preferredSelectionBlock +
    "        if (JSON.stringify(liveSelection) !== JSON.stringify(lockedSelection))\n" +
    "            throw new Error(`artifact lock mismatch for ${packageReference} (${system}-${architecture})`);\n" +
    installedBlock;
  return {
    name: "artifact lock comparison",
    markers: [
      artifactPathMarker,
      comparisonMarker,
      preferenceMarker,
      validationMarker,
    ],
    preconditions: [
      installedBlock,
      legacyArtifactPathMarker,
      legacySelectionBlock,
    ],
    allowPartialMarkers: true,
    mutate: (source, target) => {
      const hasComparison = source.includes(comparisonMarker);
      const hasCurrentPath = source.includes(artifactPathMarker);
      const hasLegacyPath = source.includes(legacyArtifactPathMarker);
      if (hasComparison) {
        let currentPathSource = source;
        if (hasLegacyPath) {
          currentPathSource = replaceUnique({
            candidates: [legacyArtifactPathMarker],
            patchName: "artifact lock path precedence",
            replacement: artifactPathMarker,
            source,
            target,
          });
        } else if (!hasCurrentPath) {
          fail(`expected artifact lock path marker was not found in ${target}`);
        }
        let migratedSource = currentPathSource;
        if (!migratedSource.includes(preferenceMarker)) {
          migratedSource = replaceUnique({
            candidates: [legacySelectionBlock],
            patchName: "artifact lock platform selection",
            replacement: preferredSelectionBlock,
            source: migratedSource,
            target,
          });
        }
        if (!migratedSource.includes(validationMarker)) {
          migratedSource = replaceUnique({
            candidates: [legacyParseBlock],
            patchName: "artifact lock validation",
            replacement: `${artifactLockValidationSource()}\n`,
            source: migratedSource,
            target,
          });
        }
        return migratedSource;
      }
      if (hasCurrentPath || hasLegacyPath) {
        fail(`incomplete artifact lock comparison source in ${target}`);
      }
      const withoutInstalledBlock = replaceUnique({
        candidates: [installedBlock],
        patchName: "artifact lock installed-state placement",
        replacement: "",
        source,
        target,
      });
      return replaceUnique({
        candidates: [preferenceBlock],
        patchName: "artifact lock comparison insertion",
        replacement: preferenceBlock + artifactBlock,
        source: withoutInstalledBlock,
        target,
      });
    },
  };
}

/** @returns {SourcePatch[]} Local manager compatibility and lock patches. */
function managerPatches() {
  const elevationBefore =
    "if (!isAdmin() && !isTests()) {\n" +
    "            await runCliAsAdmin({\n" +
    "                appDir: this.config.get('appDir'),\n" +
    "                operation: 'install',";
  const elevationMarker = "files.every(file => file.type === FileType.Archive)";
  const fsImportBefore = "import path from 'path';";
  const fsImportMarker = "import { readFileSync } from 'fs';";
  const architectureOriginal =
    "const files = packageCompatibleFiles(pkgVersion, [getArchitecture()], [getSystem()], excludedFormats);";
  const architectureBefore =
    "let files = packageCompatibleFiles(pkgVersion, [getArchitecture()], [getSystem()], excludedFormats);";
  const architectureMarker = "const architecture = getArchitecture();";
  const payloadBefore =
    "                appDir: this.config.get('appDir'),\n                operation: 'install',";
  const payloadMarker = "                artifactLockPath,\n";
  return [
    replacementPatch({
      name: "archive preference before elevation",
      before: elevationBefore,
      marker: elevationMarker,
      after:
        "if (files.some(file => file.type === FileType.Archive)) {\n" +
        "            files = files.filter(file => file.type === FileType.Archive);\n" +
        "        }\n" +
        "        if (!isAdmin() && !isTests() && " +
        "!files.every(file => file.type === FileType.Archive)) {\n" +
        "            await runCliAsAdmin({\n" +
        "                appDir: this.config.get('appDir'),\n" +
        "                operation: 'install',",
    }),
    replacementPatch({
      name: "artifact lock file import",
      before: fsImportBefore,
      marker: fsImportMarker,
      after: `${fsImportMarker}\n${fsImportBefore}`,
    }),
    replacementPatch({
      name: "stable host identity",
      before: architectureBefore,
      alternatives: [architectureOriginal],
      marker: architectureMarker,
      after:
        `${architectureMarker}\n` +
        "        let files = packageCompatibleFiles(pkgVersion, [architecture], [system], excludedFormats);",
    }),
    artifactLockPatch(),
    replacementPatch({
      name: "elevated artifact lock forwarding",
      before: payloadBefore,
      marker: payloadMarker,
      after:
        "                appDir: this.config.get('appDir'),\n" +
        payloadMarker +
        "                operation: 'install',",
    }),
  ];
}

/** @returns {SourcePatch[]} Elevated-child lock forwarding patches. */
function adminPatches() {
  const before =
    "const manager = new ManagerLocal(args.type, { appDir: args.appDir });";
  const marker = "artifactLockPath: args.artifactLockPath";
  return [
    replacementPatch({
      name: "admin artifact lock forwarding",
      before,
      marker,
      after:
        "const manager = new ManagerLocal(args.type, {\n" +
        "            appDir: args.appDir,\n" +
        `            ${marker},\n` +
        "        });",
    }),
  ];
}

/**
 * Resolve explicit fixture targets or the installed Studiorack core files.
 * @returns {{adminTarget: string | null, helperTarget: string, managerTarget: string}} Patch targets.
 */
function resolveTargets() {
  const helperTarget = resolve(process.argv[2] ?? defaultHelper);
  const managerTarget = resolve(process.argv[3] ?? defaultManager);
  let adminTarget = null;
  if (process.argv[4]) adminTarget = resolve(process.argv[4]);
  else if (!process.argv[2]) adminTarget = resolve(defaultAdmin);
  return { helperTarget, managerTarget, adminTarget };
}

try {
  const { helperTarget, managerTarget, adminTarget } = resolveTargets();
  const patchedHelper = applyPatches(
    readFileSync(helperTarget, "utf8"),
    helperTarget,
    helperPatches(),
  );
  const patchedManager = applyPatches(
    readFileSync(managerTarget, "utf8"),
    managerTarget,
    managerPatches(),
  );
  const patchedAdmin = adminTarget
    ? applyPatches(
        readFileSync(adminTarget, "utf8"),
        adminTarget,
        adminPatches(),
      )
    : null;

  writeFileSync(helperTarget, patchedHelper);
  writeFileSync(managerTarget, patchedManager);
  if (adminTarget && patchedAdmin !== null)
    writeFileSync(adminTarget, patchedAdmin);
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
