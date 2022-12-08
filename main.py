"""
MAIN.PY
This file is meant to handle all events called within __main__
"""

# Import modules
import numpy as np
from tensorflow.keras import models

# Import modules and functions from helper files
from mic_listener import record_audio, terminate
from tf_helper import preprocess_buffer

# Get commands from Colab
commands = ['stop', 'yes', 'right', 'left', 'go', 'up', 'down', 'no']

# Load model from saved_model
loaded_model = models.load_model("saved_model")