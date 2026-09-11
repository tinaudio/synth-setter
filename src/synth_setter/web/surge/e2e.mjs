import assert from "node:assert/strict";
import { createServer } from "node:http";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { chromium } from "@playwright/test";

const [site, content, sketch, output] = process.argv.slice(2);
if (!site || !content || !sketch || !output)
  throw new Error("Usage: e2e.mjs SITE CONTENT_WAV SKETCH_WAV OUTPUT_JSON");
const root = path.resolve(site);
const types = {
  ".html": "text/html",
  ".mjs": "text/javascript",
  ".js": "text/javascript",
  ".wasm": "application/wasm",
  ".json": "application/json",
};
const server = createServer(async (request, response) => {
  try {
    const target = path.resolve(
      root,
      `.${decodeURIComponent(new URL(request.url, "http://127.0.0.1").pathname)}`,
    );
    if (!target.startsWith(root + path.sep))
      return response.writeHead(403).end();
    const body = await readFile(target);
    response
      .writeHead(200, {
        "Content-Type":
          types[path.extname(target)] ?? "application/octet-stream",
      })
      .end(body);
  } catch (error) {
    response.writeHead(error.code === "ENOENT" ? 404 : 500).end();
  }
});
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
let browser;
try {
  browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) =>
    console.log(`browser ${message.type()}: ${message.text()}`),
  );
  await page.goto(`http://127.0.0.1:${server.address().port}/surge/index.html`);
  await page.waitForFunction(
    () =>
      !document.getElementById("run").disabled ||
      window.surgeEval?.state === "error",
    null,
    { timeout: 240000 },
  );
  if (await page.evaluate(() => window.surgeEval?.state === "error"))
    throw new Error(await page.locator("#status").textContent());
  const manifest = JSON.parse(
    await readFile(path.join(root, "model", "manifest.json"), "utf8"),
  );
  assert.equal(
    await page.locator("#steps").inputValue(),
    String(manifest.sampling.steps),
  );
  assert.equal(
    await page.locator("#contentCfg").inputValue(),
    String(manifest.sampling.contentCfg),
  );
  assert.equal(
    await page.locator("#sketchCfg").inputValue(),
    String(manifest.sampling.sketchCfg),
  );
  const amplitudeContract = await page.evaluate(async () => {
    const { decodeStereo, encodeStereoWav } = await import("./audio.mjs");
    const decoded = await decodeStereo(
      encodeStereoWav(
        [
          [2, -2, 0.5],
          [-2, 2, -0.5],
        ],
        44100,
      ),
      44100,
      3,
    );
    return Array.from(decoded);
  });
  assert.deepEqual(amplitudeContract, [1, -1, 0.5, -1, 1, -0.5]);
  await page.locator("#contentAudio").setInputFiles(path.resolve(content));
  await page.locator("#sketchAudio").setInputFiles(path.resolve(sketch));
  await page.locator("#contentCfg").fill("2");
  await page.locator("#sketchCfg").fill("3");
  await page.locator("#steps").fill("8");
  await page.locator("#seed").fill("17");
  await page.getByRole("button", { name: "Run evaluation" }).click();
  await page.waitForFunction(
    () => ["complete", "error"].includes(window.surgeEval?.state),
    null,
    { timeout: 480000 },
  );
  await page.screenshot({ path: `${output}.png`, fullPage: true });
  const record = await page.evaluate(() => window.surgeEval);
  if (record.state !== "complete" || errors.length)
    throw new Error(`${record.message ?? ""}; ${errors.join("; ")}`);
  const downloadEvent = page.waitForEvent("download");
  await page.getByRole("link", { name: "Download pred.wav" }).click();
  await (await downloadEvent).saveAs(`${output}.wav`);
  await writeFile(output, JSON.stringify(record));
  console.log("BROWSER_SURGE_WASM_E2E_COMPLETE");
} finally {
  await browser?.close();
  await new Promise((resolve) => server.close(resolve));
}
