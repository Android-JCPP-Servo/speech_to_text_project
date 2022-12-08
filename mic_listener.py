"""
MIC_LISTENER.PY
This file is meant to handle all microphone events
"""

# Import modules
import pyaudio as pya
import numpy as np

# Initialize constants
FRAMES_PER_BUFFER = 3200
FORMAT = pya.paInt16
CHANNELS = 1
RATE = 16000
p = pya.PyAudio()

