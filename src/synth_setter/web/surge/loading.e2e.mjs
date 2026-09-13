import assert from "node:assert/strict";
import test from "node:test";
import { createServer } from "node:http";
import { Readable } from "node:stream";
import { once } from "node:events";
import { chromium } from "@playwright/test";

test("model loading shows downloaded bytes and graph initialization before ready", { timeout: 120000 }, async () => {
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage();
    await page.addInitScript(() => {
      window.loadingMessages = [];
      new MutationObserver(() => {
        const message = document.getElementById("status")?.textContent;
        if (message) window.loadingMessages.push(message);
      }).observe(document, { subtree: true, childList: true, characterData: true });
    });
    const url = new URL(process.env.SURGE_BROWSER_URL);
    url.search = "?model=simple-baseline";
    await page.goto(url.href);
    await page.waitForFunction(() => !document.getElementById("run").disabled, null, { timeout: 90000 });
    const messages = await page.evaluate(() => window.loadingMessages);
    assert.ok(messages.some((message) => /Downloading velocity\.onnx: [\d.]+ \/ [\d.]+ MiB/.test(message)), messages.join("\n"));
    assert.ok(messages.some((message) => message.includes("Initializing velocity")), messages.join("\n"));
  } finally {
    await browser.close();
  }
});

test("a stalled real model transfer reports its filename instead of loading forever", { timeout: 120000 }, async () => {
  const proxy = createServer(async (request, response) => {
    try {
      const upstream = await fetch(new URL(request.url, process.env.SURGE_BROWSER_URL));
      response.writeHead(upstream.status, {
        "content-type": upstream.headers.get("content-type"),
        "content-length": upstream.headers.get("content-length"),
      });
      if (request.url.endsWith("frontend.onnx")) {
        for await (const chunk of upstream.body) {
          response.write(chunk);
          break;
        }
      } else {
        Readable.fromWeb(upstream.body).pipe(response);
      }
    } catch (error) {
      response.destroy(error);
    }
  });
  proxy.listen(0, "127.0.0.1");
  await once(proxy, "listening");
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage();
    await page.goto(`http://127.0.0.1:${proxy.address().port}/surge/index.html?model=simple-baseline`);
    await page.waitForFunction(() => window.surgeEval?.state === "error", null, { timeout: 90000 });
    assert.match(await page.locator("#status").textContent(), /Download stalled for 60 seconds: frontend\.onnx/);
  } finally {
    await browser.close();
    proxy.closeAllConnections();
    await new Promise((resolve) => proxy.close(resolve));
  }
});
