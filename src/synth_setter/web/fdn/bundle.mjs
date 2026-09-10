// Load the exported model bundle and the Faust artifact, checking every digest in the manifests.
import { loadFaustArtifact } from "../faust/runtime.mjs";

async function fetchBytes(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load ${url}: HTTP ${response.status}`);
  return new Uint8Array(await response.arrayBuffer());
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load ${url}: HTTP ${response.status}`);
  return response.json();
}

async function sha256(bytes) {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((value) => value.toString(16).padStart(2, "0")).join("");
}

export async function loadModelBundle(root) {
  const manifest = await fetchJson(`${root}/manifest.json`);
  if (manifest.schemaVersion !== 1) throw new Error("unsupported model bundle schema");
  // Graph files are keyed by filename; sessions are keyed by the filename stem.
  const graphs = {};
  for (const [filename, entry] of Object.entries(manifest.files)) {
    const bytes = await fetchBytes(`${root}/${filename}`);
    if ((await sha256(bytes)) !== entry.sha256) throw new Error(`model bundle digest mismatch: ${filename}`);
    graphs[filename.replace(/\.onnx$/, "")] = bytes;
  }
  return { manifest, graphs };
}

export async function loadFaustBundle(root) {
  const manifest = await fetchJson(`${root}/manifest.json`);
  const packageVersion = (await fetchJson(`${root}/vendor/package.json`)).version;
  return loadFaustArtifact(manifest, (path) => fetchBytes(`${root}/${path}`), packageVersion);
}
