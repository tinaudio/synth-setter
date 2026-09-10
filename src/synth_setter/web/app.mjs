import { sampleFlow } from "./flow.mjs";

const button = document.querySelector("#run");
const status = document.querySelector("#status");
button.addEventListener("click", async () => {
  button.disabled = true;
  try {
    status.textContent = "Loading ONNX graphs";
    const response = await fetch("input.json");
    if (!response.ok) throw new Error(`Input request failed: ${response.status}`);
    const payload = await response.json();
    const result = await sampleFlow(payload, (step, steps) => {
      status.textContent = `Sampling ${step}/${steps}`;
    });
    const accepted = await fetch("prediction", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token: payload.token, params: Array.from(result) }),
    });
    if (!accepted.ok) throw new Error(`Prediction rejected: ${await accepted.text()}`);
    document.querySelector("#prediction").textContent = JSON.stringify(Array.from(result));
    status.textContent = "Inference complete — native rendering and metrics continue in the CLI";
  } catch (error) {
    status.textContent = `Error: ${error.message}`;
    button.disabled = false;
  }
});
