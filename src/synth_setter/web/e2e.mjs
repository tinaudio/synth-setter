import { spawn } from "node:child_process";
import { chromium } from "@playwright/test";

const [screenshot, executable, ...args] = process.argv.slice(2);
const child = spawn(executable, args, { stdio: ["ignore", "pipe", "pipe"] });
let output = "";
child.stderr.on("data", (chunk) => process.stderr.write(chunk));
const ended = new Promise((resolve, reject) => {
  child.on("error", reject);
  child.on("exit", (code) => code === 0 ? resolve() : reject(new Error(`CLI exited with ${code}`)));
});
ended.catch(() => {});
let timer;
const timeout = new Promise((_, reject) => {
  timer = setTimeout(() => reject(new Error("Browser E2E exceeded six minutes")), 360000);
});
let browser;
try {
  const ready = new Promise((resolve) => child.stdout.on("data", (chunk) => {
    process.stdout.write(chunk);
    output += chunk;
    const match = output.match(/Browser evaluation: (http:\/\/127\.0\.0\.1(?::\d+)?)\r?\n/);
    if (match) resolve(match[1]);
  }));
  const url = await Promise.race([ready, timeout, ended.then(() => {throw new Error("CLI produced no browser URL");})]);
  browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  const pageErrors = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.goto(url);
  const rejected = await page.evaluate(async () => {
    const payload = await (await fetch("input.json")).json();
    const response = await fetch("prediction", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token: payload.token, params: [] }),
    });
    const invalidToken = await fetch("prediction", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token: "tökén", params: payload.noise }),
    });
    return [response.status, invalidToken.status];
  });
  if (rejected.some((status) => status !== 400)) throw new Error("Malformed browser prediction was not rejected");
  const outside = await page.request.get(new URL("/pyproject.toml", url).href);
  if (outside.status() !== 404) throw new Error("Server exposed a non-bundle resource");
  await page.getByRole("button", { name: "Run flow evaluation" }).click();
  await page.waitForFunction(() => /Inference complete|Error:/.test(document.querySelector("#status").textContent), null, { timeout: 240000 });
  const status = await page.locator("#status").textContent();
  if (status.startsWith("Error:") || pageErrors.length) throw new Error(`${status}; ${pageErrors.join("; ")}`);
  await page.screenshot({ path: screenshot });
  await Promise.race([ended, timeout]);
  console.log("BROWSER_E2E_COMPLETE");
} finally {
  clearTimeout(timer);
  await browser?.close();
  if (child.exitCode === null) child.kill("SIGTERM");
}
