"""Interactive Surge XT preview with real-time audio streaming via pedalboard."""

import threading

import numpy as np
from pedalboard import VST3Plugin
from pedalboard.io import AudioStream

CHANNELS = 2
SAMPLE_RATE = 44100
BUFFER_SIZE = 512


def play_audio(plugin: VST3Plugin) -> None:
    """Stream silence through Surge XT and write synthesized audio to the output device."""
    silence = np.zeros((CHANNELS, BUFFER_SIZE), dtype=np.float32)
    with AudioStream(
        output_device_name=AudioStream.default_output_device_name,
        sample_rate=SAMPLE_RATE,
        buffer_size=BUFFER_SIZE,
    ) as stream:
        while True:
            synth_output = plugin(silence, SAMPLE_RATE, reset=False)
            stream.write(synth_output, SAMPLE_RATE)


if __name__ == "__main__":
    plugin = VST3Plugin("plugins/Surge XT.vst3")

    t = threading.Thread(target=play_audio, args=(plugin,), daemon=True)
    t.start()

    plugin.show_editor()
