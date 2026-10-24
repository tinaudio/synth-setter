import assert from "node:assert/strict";
import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import path from "node:path";
import { chromium, webkit } from "@playwright/test";

const [
  site,
  content,
  output,
  browserName = "chromium",
  mode = "inference",
  template,
] = process.argv.slice(2);
if (!site || !content || !output)
  throw new Error(
    "Usage: poster.e2e.mjs SITE CONTENT_WAV OUTPUT_DIR [chromium|webkit] [inference|layout|style] [TEMPLATE_DIR]",
  );
const browsers = { chromium, webkit };
if (!browsers[browserName] || !["inference", "layout", "style"].includes(mode))
  throw new Error(
    "browser must be chromium or webkit; mode must be inference, layout, or style",
  );
if (mode === "style" && !template)
  throw new Error("style mode requires TEMPLATE_DIR");

const remoteSite = /^https?:\/\//.test(site);
const root = remoteSite ? null : path.resolve(site);
const types = {
  ".css": "text/css",
  ".html": "text/html",
  ".js": "text/javascript",
  ".json": "application/json",
  ".mjs": "text/javascript",
  ".onnx": "application/octet-stream",
  ".wasm": "application/wasm",
};
function createStaticServer(rootDirectory) {
  return createServer(async (request, response) => {
    try {
      const pathname = decodeURIComponent(
        new URL(request.url, "http://127.0.0.1").pathname,
      );
      const relative = pathname.endsWith("/")
        ? `${pathname.slice(1)}index.html`
        : pathname.slice(1);
      const target = path.resolve(rootDirectory, relative);
      if (!target.startsWith(`${rootDirectory}${path.sep}`))
        return response.writeHead(403).end();
      const body = await readFile(target);
      response
        .writeHead(200, {
          "Content-Type":
            types[path.extname(target)] ?? "application/octet-stream",
          "Cross-Origin-Embedder-Policy": "require-corp",
          "Cross-Origin-Opener-Policy": "same-origin",
        })
        .end(body);
    } catch (error) {
      response.writeHead(error.code === "ENOENT" ? 404 : 500).end();
    }
  });
}

const server = root ? createStaticServer(root) : null;
if (server)
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const targetUrl = remoteSite
  ? site
  : `http://127.0.0.1:${server.address().port}/`;
const templateServer = template
  ? createStaticServer(path.resolve(template))
  : null;
if (templateServer)
  await new Promise((resolve) =>
    templateServer.listen(0, "127.0.0.1", resolve),
  );

let browser;
try {
  browser = await browsers[browserName].launch({ headless: true });
  const page = await browser.newPage({
    viewport:
      browserName === "webkit"
        ? { width: 390, height: 844 }
        : { width: 1440, height: 1000 },
  });
  const failures = [];
  const modelRequests = [];
  page.on("pageerror", (error) => failures.push(error.message));
  page.on("requestfailed", (request) => {
    if (!request.url().startsWith("blob:"))
      failures.push(`${request.url()}: ${request.failure()?.errorText}`);
  });
  page.on("request", (request) => {
    if (/\.(onnx|wasm)(\?|$)/.test(request.url()))
      modelRequests.push(request.url());
  });

  await page.goto(targetUrl, { waitUntil: "networkidle", timeout: 120000 });
  await assert.doesNotReject(() =>
    page
      .getByRole("heading", { name: /Sketch Controls for Synth Presets/ })
      .waitFor(),
  );
  assert.equal(
    modelRequests.length,
    0,
    "model and engine assets must remain lazy before demo opt-in",
  );
  assert.equal(await page.locator("#demo-root iframe").count(), 0);
  assert.equal(await page.locator("#demo-actions").isVisible(), false);
  const localLinks = await page
    .locator("a[href], link[href], script[src]")
    .evaluateAll(async (elements) => {
      const urls = [
        ...new Set(
          elements
            .map(
              (element) => new URL(element.href || element.src, location.href),
            )
            .filter((url) => url.origin === location.origin && !url.hash)
            .map(String),
        ),
      ];
      return Promise.all(
        urls.map(async (url) => ({ url, status: (await fetch(url)).status })),
      );
    });
  assert.ok(localLinks.length > 0);
  assert.ok(
    localLinks.every(({ status }) => status >= 200 && status < 400),
    JSON.stringify(localLinks),
  );
  await page.screenshot({
    path: path.join(output, `${browserName}-poster.png`),
    fullPage: true,
  });

  if (mode === "style") {
    const templatePage = await browser.newPage({
      viewport: page.viewportSize(),
    });
    await templatePage.goto(
      `http://127.0.0.1:${templateServer.address().port}/`,
      { waitUntil: "networkidle", timeout: 120000 },
    );
    await templatePage.screenshot({
      path: path.join(output, `${browserName}-original.png`),
      fullPage: true,
    });
    const styleContract = {
      body: [
        "fontFamily",
        "fontSize",
        "fontWeight",
        "lineHeight",
        "color",
        "backgroundColor",
      ],
      h1: [
        "fontFamily",
        "fontSize",
        "fontWeight",
        "lineHeight",
        "color",
        "marginTop",
        "marginBottom",
      ],
      ".lead": [
        "fontFamily",
        "fontSize",
        "fontWeight",
        "lineHeight",
        "color",
        "marginBottom",
      ],
      ".btn": [
        "fontFamily",
        "fontSize",
        "fontWeight",
        "lineHeight",
        "color",
        "backgroundColor",
        "borderRadius",
        "paddingTop",
        "paddingRight",
        "paddingBottom",
        "paddingLeft",
      ],
      ".container": ["width", "maxWidth", "paddingLeft", "paddingRight"],
      ".section": [
        "fontFamily",
        "fontSize",
        "lineHeight",
        "color",
        "marginTop",
        "marginBottom",
      ],
      h2: [
        "fontFamily",
        "fontSize",
        "fontWeight",
        "lineHeight",
        "color",
        "marginTop",
        "marginBottom",
      ],
    };
    const capture = async (targetPage) =>
      targetPage.evaluate(
        (contract) =>
          Object.fromEntries(
            Object.entries(contract).map(([selector, properties]) => {
              const element = document.querySelector(selector);
              if (!element)
                throw new Error(`missing style parity selector: ${selector}`);
              const computed = getComputedStyle(element);
              return [
                selector,
                Object.fromEntries(
                  properties.map((property) => [property, computed[property]]),
                ),
              ];
            }),
          ),
        styleContract,
      );
    assert.deepEqual(await capture(page), await capture(templatePage));
    assert.equal(
      await page
        .locator(
          'link[href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css"]',
        )
        .count(),
      1,
    );
    assert.equal(
      await page
        .locator(
          'link[href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.8.0/font/bootstrap-icons.css"]',
        )
        .count(),
      1,
    );
    assert.equal(
      await page
        .locator(".btn")
        .first()
        .evaluate((element) => getComputedStyle(element).display),
      "inline-block",
    );
    await templatePage.close();
    assert.deepEqual(failures, []);
    console.log(`POSTER_STYLE_PARITY_COMPLETE ${browserName}`);
  } else {
    await page.getByRole("button", { name: "Load interactive demo" }).click();
    const demo = page.locator("#demo-root");
    await page.locator("#demo-actions").waitFor({ state: "visible" });
    assert.equal(await demo.locator("iframe").count(), 0);
    await demo
      .getByRole("button", { name: "Run selected models" })
      .waitFor({ timeout: 120000 });
    assert.equal(await demo.locator("#models input:checked").count(), 4);
    assert.equal(
      modelRequests.length,
      0,
      "mounting inline controls must not download model or engine assets",
    );
    await page.screenshot({
      path: path.join(output, `${browserName}-demo.png`),
      fullPage: true,
    });
    const demoBox = await demo.boundingBox();
    assert.ok(demoBox.width <= (browserName === "webkit" ? 390 : 1440));
    assert.ok(demoBox.height >= 640);

    if (mode === "inference") {
      await demo.locator("#contentAudio").setInputFiles(path.resolve(content));
      await demo.locator("#sketchAudio").setInputFiles(path.resolve(content));
      await demo.locator("#steps").fill("8");
      await demo.getByRole("button", { name: "Run selected models" }).click();
      await demo.evaluate(
        () =>
          new Promise((resolve, reject) => {
            const deadline = setTimeout(
              () => reject(new Error("comparison inference timed out")),
              600000,
            );
            const timer = setInterval(() => {
              if (
                ["complete", "error"].includes(window.surgeComparison?.state)
              ) {
                clearInterval(timer);
                clearTimeout(deadline);
                resolve();
              }
            }, 100);
          }),
      );
      assert.equal(
        await page.evaluate(() => window.surgeComparison.state),
        "complete",
        await demo.locator("#status").textContent(),
      );
      const result = await page.evaluate(() => {
        const records = Object.values(window.surgeComparison.records);
        return {
          count: records.length,
          nonSilent: records.every((record) =>
            record.audio.flat().some((sample) => Math.abs(sample) > 1e-5),
          ),
        };
      });
      assert.deepEqual(result, { count: 4, nonSilent: true });

      const comparison = page.waitForEvent("download");
      await demo
        .getByRole("link", { name: "Download comparison records" })
        .click();
      const recordPath = path.join(output, "poster-comparison.json");
      await (await comparison).saveAs(recordPath);
      assert.ok((await stat(recordPath)).size > 0);
      const savedRecord = JSON.parse(await readFile(recordPath, "utf8"));
      assert.equal(Object.keys(savedRecord.records).length, 4);
    }

    assert.deepEqual(failures, []);
    console.log(`POSTER_E2E_COMPLETE ${browserName} ${mode}`);
  }
} finally {
  await browser?.close();
  if (server) await new Promise((resolve) => server.close(resolve));
  if (templateServer)
    await new Promise((resolve) => templateServer.close(resolve));
}
