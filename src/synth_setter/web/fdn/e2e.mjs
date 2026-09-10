// Drive the exported site in headless Chromium: upload a WAV, run one evaluation, and dump the
// page's complete run record (noise, parameters, render, metrics) for Python parity checks.
import { createServer } from "node:http";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { chromium } from "@playwright/test";

const [siteDirectory, wavPath, outputPath, mode = "both", contentCfg = "2", sketchCfg = "2", steps = "8", seed = "17"] = process.argv.slice(2);
if (!siteDirectory || !wavPath || !outputPath) {
  console.error("Usage: node e2e.mjs SITE_DIR TARGET_WAV OUTPUT_JSON [MODE CONTENT_CFG SKETCH_CFG STEPS SEED]");
  process.exit(2);
}

const types = { ".html": "text/html", ".mjs": "text/javascript", ".js": "text/javascript", ".json": "application/json", ".wasm": "application/wasm", ".onnx": "application/octet-stream" };
const root = path.resolve(siteDirectory);
const server = createServer(async (request, response) => {
  const target = path.resolve(root, `.${decodeURIComponent(new URL(request.url, "http://127.0.0.1").pathname)}`);
  if (!target.startsWith(root)) return response.writeHead(403).end();
  try {
    const body = await readFile(target);
    response.writeHead(200, { "Content-Type": types[path.extname(target)] ?? "application/octet-stream" });
    response.end(body);
  } catch {
    response.writeHead(404).end();
  }
});
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const url = `http://127.0.0.1:${server.address().port}/fdn/index.html`;

const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage();
  const pageErrors = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.goto(url);
  await page.waitForFunction(() => document.querySelector("#run").disabled === false || document.querySelector("#status").textContent.startsWith("Error:"), null, { timeout: 120000 });
  await page.locator("#target").setInputFiles(path.resolve(wavPath));
  await page.locator("#mode").selectOption(mode);
  await page.locator("#content").fill(contentCfg);
  await page.locator("#sketch").fill(sketchCfg);
  await page.locator("#steps").fill(steps);
  await page.locator("#seed").fill(seed);
  await page.getByRole("button", { name: "Run evaluation" }).click();
  await page.waitForFunction(() => ["complete", "error"].includes(window.fdnEval?.state), null, { timeout: 600000 });
  const record = await page.evaluate(() => window.fdnEval);
  const statusText = await page.locator("#status").textContent();
  await page.screenshot({ path: `${outputPath}.png`, fullPage: true });
  if (record.state !== "complete" || pageErrors.length) {
    throw new Error(`${statusText}; ${record.message ?? ""} ${pageErrors.join("; ")}`);
  }
  await writeFile(outputPath, JSON.stringify(record));
  console.log(`BROWSER_FDN_E2E_COMPLETE ${statusText}`);
} finally {
  await browser.close();
  server.close();
}
