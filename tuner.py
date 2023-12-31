import pyaudio as pa
import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import timeit as ti
import time
import sounddevice as sd

# This script helps you tune a guitar by telling you by identifying the current string
# being strummed (if its close) and telling you the current frequency vs the target for
# that string.  Its just an infinite loop in the command line but eventually this script
# will operate with a GUI. 

## Parameters

# 'Fs' is the sample frequency of your audio device.
Fs = 48000

# 'chunk_size' is the integer number of samples used for each recording.
# If you want each chunk to be 1/10 of a second, divide Fs by 10. For a
# 2 second sample, multiply by 2.  The smaller the sample, the less resolution
# you get in the frequency analysis but the system is now more responsive to
# note changes.
chunk_size = Fs//10

# Obtain a liast of sound devices.  While I would gladly set 'kind' to 'input',
# the input device that I actually want isn't coming through.  Instead, I am
# pulling all sound devices and letting the user pick the right one.
devices = sd.query_devices(device=None, kind=None)

if type(devices) is dict:
    # Single device
    print("Device: " + str(devices["index"]) + " - " + str(devices["name"]))
else:
    # Multiple devices in a DeviceList
    for device in devices:
        print("Device: " + str(device["index"]) + " - " + str(device["name"]))


input_device = int(input("Which device do you want to use?"))

# 'format' is dependent on your audio device, but paInt16 is by far the most
# common.
format = pa.paInt16

# Data structure for all of the notes we are looking for in tuning a guitar.
guitar_notes = {"E_82": 82,
                "A_110": 110,
                "D_147": 147,
                "G_196": 196,
                "B_247": 247,
                "E_330": 330}

# Having these as lists is useful later.    
key_list = list(guitar_notes.keys())
val_list = list(guitar_notes.values())

# Query user for input device
sd.query_devices(device=None,
                 kind="input")


# Open audio stream.
p = pa.PyAudio()
stream = p.open(format=format, 
                channels=2, 
                rate=Fs,
                frames_per_buffer=chunk_size,
                input=True,
                input_device_index=input_device)

print("Audiostream opened")

quitflag = False

# Delay program 1 chunk's worth of time here to allow the buffer to fill up.
time.sleep(chunk_size / Fs)

while quitflag == False:
    # Start timing program execution 
    start = ti.timeit()

    # Read 1 chunk's worth of data from the audiostream.
    chunk = stream.read(chunk_size)

    # It comes in as a text buffer for some reason, so we convert it to int16 and
    # throw away one of the two channels.
    chunk = np.frombuffer(chunk, np.int16)        
    chunk = np.reshape(chunk, (chunk_size, 2))
    chunk = chunk[ : , 1]

    # Take the absolute value of a real fft of the chunk
    fourier = abs(np.fft.rfft(chunk))

    # Construct a frequency domain array.
    n = chunk.size
    freq = np.fft.fftfreq(n, d=1/Fs)

    # Find the dominant frequency in the fft data.
    current_freq_value = freq[fourier.argmax()]

    # Find the closest note from the 6 guitar string values we care about.
    closest_freq = min(guitar_notes.values(), key=lambda x: abs(x - current_freq_value))

    # Use the frequency value to index the list of note names.
    position = val_list.index(closest_freq)

    # Now we have the name of the note that is closest to the one being played.
    closest_note = key_list[position]

    # How long did that take?
    end = ti.timeit()

    # Lets delay long enough for this loop iteration to equal 1 chunk's worth of time.
    time.sleep((chunk_size / Fs) - (end - start))

    # Print current frequency and the closest note.
    print("Current freq: " + str(current_freq_value) + " Hz which is closest to note: " + closest_note + " which is " + str(closest_freq) + " Hz")