from faster_whisper import WhisperModel
import os
import time
t1 = time.perf_counter()

model = WhisperModel("small", compute_type="int8")  # gut für CPU

for file in os.listdir("Audio/"):
    if file.endswith(".m4a"):
        segments, info = model.transcribe(f"Audio/{file}")
        for segment in segments:
            print(segment.text)

t2= time.perf_counter()

print("Time: ", t2-t1)