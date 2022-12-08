"""
TF_HELPER.PY
This file is meant to handle all Tensorflow logic, methods, and functions
Referenced from https://www.tensorflow.org/tutorials/audio/simple_audio
"""

# Import modules
import numpy as np
import tensorflow as tf

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Method for getting or calling waveform object, then transforming it into a spectrogram
def get_spectrogram(waveform):
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

# Helper method for normalizing the waveform and getting the spectrogram
def preprocess_buffer(waveform):
    """
    np.array waveform size/shape: (16000,)
    
    output: spectrogram tensor with size/shape: (1, `height`, `width`, `channels`)
    """
    # Verify pipeline
    waveform = waveform / 32768
    waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)
    # waveform

    # Get spectrogram
    spec = get_spectrogram(waveform)

    # Add one dimension to the spectrogram
    spec = tf.expand_dims(spec, 0)

    # Return the spectrogram
    return spec