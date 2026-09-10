class FaustOutputCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.request = null;
    this.port.onmessage = ({ data }) => {
      this.request = {
        channels: [new Float32Array(data.frames), new Float32Array(data.frames)],
        startFrame: data.startFrame,
      };
    };
  }

  process(inputs, outputs) {
    const input = inputs[0];
    const output = outputs[0];
    for (let channel = 0; channel < output.length; channel += 1) {
      const inputChannel = input[channel] ?? input[0];
      if (inputChannel) output[channel].set(inputChannel);
      else output[channel].fill(0);
    }
    if (!this.request) return true;

    const requestEnd = this.request.startFrame + this.request.channels[0].length;
    const blockEnd = currentFrame + output[0].length;
    const overlapStart = Math.max(currentFrame, this.request.startFrame);
    const overlapEnd = Math.min(blockEnd, requestEnd);
    if (overlapStart < overlapEnd) {
      const inputOffset = overlapStart - currentFrame;
      const outputOffset = overlapStart - this.request.startFrame;
      const frameCount = overlapEnd - overlapStart;
      for (let channel = 0; channel < 2; channel += 1) {
        const samples = input[channel] ?? input[0];
        if (samples) {
          this.request.channels[channel].set(
            samples.subarray(inputOffset, inputOffset + frameCount),
            outputOffset,
          );
        }
      }
    }
    if (blockEnd >= requestEnd) {
      this.port.postMessage(this.request.channels, this.request.channels.map((channel) => channel.buffer));
      this.request = null;
    }
    return true;
  }
}

registerProcessor("faust-output-capture", FaustOutputCaptureProcessor);
