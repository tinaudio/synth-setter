const root = document.getElementById("demo-root");
const load = document.getElementById("load-demo");
const actions = document.getElementById("demo-actions");
const fullscreen = document.getElementById("fullscreen-demo");

load.addEventListener("click", async () => {
  load.disabled = true;
  document.getElementById("demo-status").textContent =
    "Loading comparison controls…";
  try {
    const response = await fetch("/surge/index.html");
    if (!response.ok)
      throw new Error(`Evaluator interface HTTP ${response.status}`);
    const source = new DOMParser().parseFromString(
      await response.text(),
      "text/html",
    );
    const evaluator = document.createElement("div");
    evaluator.className = "poster-evaluator";
    for (const selector of ["#controls", "#status", "#results"]) {
      const node = source.querySelector(selector);
      if (!node) throw new Error(`Evaluator interface lacks ${selector}`);
      evaluator.append(document.adoptNode(node));
    }
    root.classList.remove("placeholder", "placeholder-fig");
    root.classList.add("demo-host-loaded");
    root.replaceChildren(evaluator);
    actions.hidden = false;
    await import("/surge/comparison.mjs");
  } catch (error) {
    root.classList.remove("placeholder-fig");
    root.textContent = `Unable to load the interactive demo: ${error.message}`;
  }
});

fullscreen.addEventListener("click", async () => {
  if (root.requestFullscreen) await root.requestFullscreen();
});
