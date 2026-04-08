"""Interactive Surge XT preview with real-time audio streaming via pedalboard."""

import threading

import numpy as np
import pedalboard
from pedalboard import Pedalboard
from pedalboard.io import AudioStream

CHANNELS = 2
SAMPLE_RATE = 44100
BUFFER_SIZE = 512

plugin = pedalboard.load_plugin("plugins/Surge XT.vst3")


def play_audio():
    """Stream silence through Surge XT and write synthesized audio to the output device."""
    with AudioStream(
        output_device_name=AudioStream.default_output_device_name,
        sample_rate=SAMPLE_RATE,
        buffer_size=BUFFER_SIZE,
    ) as stream:
        board = Pedalboard([plugin])
        while True:
            silence = np.zeros((CHANNELS, BUFFER_SIZE), dtype=np.float32)
            synth_output = board(silence, SAMPLE_RATE, reset=False)
            stream.write(synth_output, SAMPLE_RATE)


t = threading.Thread(target=play_audio)
t.start()

plugin.show_editor()
