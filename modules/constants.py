import pyaudio

BUFFER_SIZE = 1
FORMAT = pyaudio.paInt16
SAMPLE_RATE = 16000
CHANNELS = 1
INPUT_DEVICE_INDEX = 2
CALLBACK_INTERVAL = 0.5
FRAMES_PER_BUFFER = int(SAMPLE_RATE * CALLBACK_INTERVAL)
IS_DEBUG = False

DEBUG_FOLDER = "./debug"

