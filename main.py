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

# Helper function to predict audio
def predict_audio():
    audio = record_audio()
    spec = preprocess_buffer(audio)
    prediction = loaded_model(spec)
    label_pred = np.argmax(prediction, axis=1)
    command = commands[label_pred[0]]
    print("Predicted label:", command)
    return command

# Run full program
if __name__ == "__main__":
    # Inform the user the program is running
    print("I'm listening...")
    # While program is running...
    while True:
        # Get the predicted command
        command = predict_audio()
        # If the command is "stop"...
        if command == "stop":
            # Stop the program
            terminate()
            break