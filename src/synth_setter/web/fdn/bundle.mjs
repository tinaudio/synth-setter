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
  const graphs = {};
  for (const [name, entry] of Object.entries(manifest.files)) {
    const bytes = await fetchBytes(`${root}/${entry.path}`);
    if ((await sha256(bytes)) !== entry.sha256) throw new Error(`model bundle digest mismatch: ${entry.path}`);
    graphs[name] = bytes;
  }
  return { manifest, graphs };
}

export async function loadFaustBundle(root) {
  const manifest = await fetchJson(`${root}/manifest.json`);
  const packageVersion = (await fetchJson(`${root}/vendor/package.json`)).version;
  return loadFaustArtifact(manifest, (path) => fetchBytes(`${root}/${path}`), packageVersion);
}
