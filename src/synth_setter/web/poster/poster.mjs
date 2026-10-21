const root = document.getElementById("demo-root");
const load = document.getElementById("load-demo");
const actions = document.getElementById("demo-actions");
const fullscreen = document.getElementById("fullscreen-demo");

load.addEventListener("click", () => {
  const frame = document.createElement("iframe");
  frame.className = "demo-frame";
  frame.src = "/surge/";
  frame.title = "Interactive Surge XT sound-matching evaluator";
  frame.allow = "autoplay; fullscreen";
  frame.setAttribute("allowfullscreen", "");
  root.classList.remove("placeholder", "placeholder-fig");
  root.classList.add("demo-host-loaded");
  root.replaceChildren(frame);
  actions.hidden = false;
});

fullscreen.addEventListener("click", async () => {
  const frame = root.querySelector("iframe");
  if (frame?.requestFullscreen) await frame.requestFullscreen();
});
