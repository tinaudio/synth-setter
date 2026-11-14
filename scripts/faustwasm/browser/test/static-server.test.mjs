import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";

import { createStaticServer } from "./static-server.mjs";

const RESPONSE_TIMEOUT_MS = 5_000;

async function withServer(files, run) {
  const root = await mkdtemp(path.join(tmpdir(), "faust-static-server-"));
  for (const [name, body] of Object.entries(files)) {
    await writeFile(path.join(root, name), body);
  }
  const server = createStaticServer(root);
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  try {
    await run(`http://127.0.0.1:${server.address().port}`);
  } finally {
    await new Promise((resolve) => server.close(resolve));
    await rm(root, { force: true, recursive: true });
  }
}

test("missing_file_answers_404_instead_of_stranding_the_request", async () => {
  await withServer({ "index.html": "<!doctype html>" }, async (baseUrl) => {
    // A 200 written before the read leaves this request unanswerable, so the
    // fetch times out rather than returning any status at all.
    const response = await fetch(`${baseUrl}/absent.wasm`, {
      signal: AbortSignal.timeout(RESPONSE_TIMEOUT_MS),
    });

    assert.equal(response.status, 404);
    assert.equal(await response.text(), "Not found");
  });
});

test("bundle_root_serves_index_with_its_media_type", async () => {
  await withServer({ "index.html": "<!doctype html>" }, async (baseUrl) => {
    const response = await fetch(baseUrl, { signal: AbortSignal.timeout(RESPONSE_TIMEOUT_MS) });

    assert.equal(response.status, 200);
    assert.equal(response.headers.get("content-type"), "text/html");
    assert.equal(await response.text(), "<!doctype html>");
  });
});

test("wasm_module_is_served_as_application_wasm", async () => {
  await withServer({ "dsp.wasm": Buffer.from([0x00, 0x61, 0x73, 0x6d]) }, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/dsp.wasm`, {
      signal: AbortSignal.timeout(RESPONSE_TIMEOUT_MS),
    });

    assert.equal(response.headers.get("content-type"), "application/wasm");
    assert.deepEqual(new Uint8Array(await response.arrayBuffer()), new Uint8Array([0, 97, 115, 109]));
  });
});
