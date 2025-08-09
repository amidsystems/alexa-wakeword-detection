# This module handles wake word detection

import pyaudio
import numpy as np
import openwakeword
from openwakeword.model import Model
from playsound import playsound
import time
from VINI_voice import vini_voice 


openwakeword.utils.download_models()  # only runs for the first time to download the models. Gets ignored if models are already downloaded

model_path = r"models\alexa_v0.1.onnx" # the wake word model path to use (onnx model)

VINI = vini_voice() # used for text to speech, only works on windows. Comment it out if not on windows

def acknowledge_wakeup_call():
    """
    This function gets triggered when the wake word is detected.
    """
    playsound("chime.wav", False) # plays the chime sound
    # VINI.speak("Hello Sir!") # only works on windows currently. Comment it out if not on windows

    """ Write your custom logic here to route the flow to your ai assistant/Chatgpt etc."""

# some audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280 # buffer size to use for realtime predictions
audio = pyaudio.PyAudio()
mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK) # create mic stream

alexaModel = Model([model_path, ], 
                #    enable_speex_noise_suppression=True # to use this, do pip install speexdsp-ns on linux/mac only.
                   )

n_models = len(alexaModel.models.keys())

last_detection_time = time.time()


if __name__ == "__main__":
    print("Listening for wake word...\n\n")

    while True: 
        audio = np.frombuffer(mic_stream.read(CHUNK), dtype=np.int16) # read audio buffer from the mic stream, of size 'CHUNK'
        prediction = alexaModel.predict(audio)
        n_spaces = 16
        output_string_header = """
            Model Name         | Score | Wakeword Status
            --------------------------------------
            """
        for mdl in alexaModel.prediction_buffer.keys():
            scores = list(alexaModel.prediction_buffer[mdl])
            curr_score = format(scores[-1], '.20f').replace("-", "")
            output_string_header += f"""{mdl}{" "*(n_spaces - len(mdl))}   | {curr_score[0:5]} | {"--"+" "*20 if scores[-1] <= 0.5 else "Wakeword Detected!"}
            """
            if scores[-1] > 0.6: # your custom threshold for detecting if wake word prediction is actually correct
                # wake word was detected
                if time.time() - last_detection_time > 1.0:
                    acknowledge_wakeup_call()
                    alexaModel.prediction_buffer[mdl].clear()
                    last_detection_time = time.time()

        print("\033[F"*(4*n_models+1))
        print(output_string_header, "                             ", end='\r')



        