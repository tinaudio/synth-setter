import { readFile } from "node:fs/promises";
import { createServer } from "node:http";
import path from "node:path";

function contentType(filePath) {
  if (filePath.endsWith(".html")) return "text/html";
  if (filePath.endsWith(".js") || filePath.endsWith(".mjs")) return "text/javascript";
  if (filePath.endsWith(".json")) return "application/json";
  if (filePath.endsWith(".wasm")) return "application/wasm";
  if (filePath.endsWith(".css")) return "text/css";
  return "application/octet-stream";
}

export function createStaticServer(siteDirectory) {
  return createServer(async (request, response) => {
    try {
      const requestPath = new URL(request.url, "http://localhost").pathname;
      const relativePath = requestPath === "/" ? "index.html" : requestPath.slice(1);
      const filePath = path.resolve(siteDirectory, relativePath);
      if (!filePath.startsWith(`${siteDirectory}${path.sep}`)) throw new Error("Invalid path");
      // Read before the first writeHead: a read that rejects after a 200 is on
      // the wire leaves the catch unable to answer at all (ERR_HTTP_HEADERS_SENT).
      const body = await readFile(filePath);
      response.writeHead(200, { "Content-Type": contentType(filePath) });
      response.end(body);
    } catch {
      response.writeHead(404);
      response.end("Not found");
    }
  });
}
