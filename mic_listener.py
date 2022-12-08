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

# Helper function for listening to/recording audio
def record_audio():
    # Establish audio stream for listening and recording
    stream = p.open(
        format=FORMAT, 
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=FRAMES_PER_BUFFER
    )

    # Initialize empty frames and number of seconds, which will be used to establish number of samples within recording
    frames = []
    seconds = 1
    for i in range(0, int(RATE / FRAMES_PER_BUFFER * seconds)):
        # Establish the data for the model
        data = stream.read(FRAMES_PER_BUFFER)
        # Append the data to the frames array, so all samples can be tested
        frames.append(data)
    
    # Stop and close stream
    stream.stop_stream()
    stream.close()

    # Join frames and return buffer
    return np.frombuffer(b''.join(frames), dtype=np.int16)

# Terminate PyAudio I/O
def terminate():
    p.terminate()