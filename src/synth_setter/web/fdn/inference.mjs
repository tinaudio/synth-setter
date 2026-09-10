// Flow sampling through the exported ONNX graphs: frontend (waveform → normalized mel),
// conditioning (mel + sketch → content and both control-token branches), and the guided
// velocity field integrated with RK4. Graph input/output names come from the sessions.
import { integrateRK4 } from "../rk4.mjs";

const tensor = (ort, data, shape) => new ort.Tensor("float32", data, shape);

export async function loadGraphs(ort, bytes) {
  const options = { executionProviders: ["wasm"] };
  const [frontend, conditioning, velocity] = await Promise.all(
    [bytes.frontend, bytes.conditioning, bytes.velocity].map((graph) => ort.InferenceSession.create(graph, options)),
  );
  return { frontend, conditioning, velocity };
}

export async function sampleParameters({ ort, graphs, waveform, sketch, weights, noise, steps, onStep }) {
  const frontendOut = await graphs.frontend.run({ [graphs.frontend.inputNames[0]]: tensor(ort, waveform, [1, waveform.length]) });
  const mel = frontendOut[graphs.frontend.outputNames[0]];
  const [melName, sketchName] = graphs.conditioning.inputNames;
  const encoded = await graphs.conditioning.run({
    [melName]: mel,
    [sketchName]: tensor(ort, sketch, [1, 10, 32]),
  });
  const [contentName, controlsName, nullControlsName] = graphs.conditioning.outputNames;
  const [xName, tName, conditioningName, controlsIn, nullControlsIn, weightsName] = graphs.velocity.inputNames;
  const weightTensor = tensor(ort, weights, [4]);
  const field = async (x, t) => {
    const out = await graphs.velocity.run({
      [xName]: tensor(ort, x, [1, x.length]),
      [tName]: tensor(ort, Float32Array.of(t), [1, 1]),
      [conditioningName]: encoded[contentName],
      [controlsIn]: encoded[controlsName],
      [nullControlsIn]: encoded[nullControlsName],
      [weightsName]: weightTensor,
    });
    return Float32Array.from(out[graphs.velocity.outputNames[0]].data);
  };
  const params = await integrateRK4({ field, noise, steps, onStep });
  return { params, mel: Float32Array.from(mel.data), melShape: mel.dims };
}
