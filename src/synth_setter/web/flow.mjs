import * as ort from "./ort/ort.wasm.min.mjs";
import { integrateRK4 } from "./rk4.mjs";

export { branchWeights } from "./guidance.mjs";

ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = new URL("./ort/", import.meta.url).href;

function tensor(value) {
  return new ort.Tensor("float32", Float32Array.from(value.data), value.shape);
}

export async function sampleFlow(payload, onStep) {
  const encoder = await ort.InferenceSession.create("conditioning.onnx", { executionProviders: ["wasm"] });
  let velocity;
  try {
    velocity = await ort.InferenceSession.create("velocity.onnx", { executionProviders: ["wasm"] });
    const encoded = await encoder.run({ mel: tensor(payload.mel), sketch_ctrl: tensor(payload.sketch_ctrl) });
    const weights = new ort.Tensor("float32", Float32Array.from(payload.branch_weights), [4]);
    return await integrateRK4({
      noise: Float32Array.from(payload.noise),
      steps: payload.steps,
      onStep,
      field: async (state, time) => {
        const output = await velocity.run({
          ...encoded,
          x: new ort.Tensor("float32", state, [1, state.length]),
          t: new ort.Tensor("float32", Float32Array.of(time), [1, 1]),
          branch_weights: weights,
        });
        return output.velocity.data;
      },
    });
  } finally {
    await velocity?.release();
    await encoder.release();
  }
}
