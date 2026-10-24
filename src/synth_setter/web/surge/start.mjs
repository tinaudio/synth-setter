if (new URLSearchParams(location.search).has("compare"))
  await import("./comparison.mjs");
else
  await import("./app.mjs");
