import { appliedGroups, sampleFlow } from "./flow.mjs";

const button = document.querySelector("#run");
const status = document.querySelector("#status");
const groupList = document.querySelector("#sketch-groups");

function renderGroups(names, enabled) {
  groupList.replaceChildren(
    ...names.map((name) => {
      const label = document.createElement("label");
      const input = document.createElement("input");
      input.type = "checkbox";
      input.name = "sketch-group";
      input.value = name;
      input.checked = enabled.includes(name);
      label.append(input, ` ${name}`);
      return label;
    }),
  );
}

function selectedGroups() {
  return Array.from(groupList.querySelectorAll("input:checked"), (input) => input.value);
}

const payloadReady = (async () => {
  const response = await fetch("input.json");
  if (!response.ok) throw new Error(`Input request failed: ${response.status}`);
  const payload = await response.json();
  renderGroups(payload.sketch_groups.names, payload.sketch_groups.enabled);
  return payload;
})();
payloadReady.catch((error) => {
  status.textContent = `Error: ${error.message}`;
});

button.addEventListener("click", async () => {
  button.disabled = true;
  try {
    status.textContent = "Loading ONNX graphs";
    const payload = await payloadReady;
    const groups = appliedGroups(payload.sketch_groups.names, selectedGroups());
    const result = await sampleFlow(payload, (step, steps) => {
      status.textContent = `Sampling ${step}/${steps}`;
    }, groups);
    const accepted = await fetch("prediction", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token: payload.token, params: Array.from(result), sketch_groups: groups }),
    });
    if (!accepted.ok) throw new Error(`Prediction rejected: ${await accepted.text()}`);
    document.querySelector("#prediction").textContent = JSON.stringify(Array.from(result));
    status.textContent = "Inference complete — native rendering and metrics continue in the CLI";
  } catch (error) {
    status.textContent = `Error: ${error.message}`;
    button.disabled = false;
  }
});
