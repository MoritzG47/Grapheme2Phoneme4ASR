from faster_whisper import WhisperModel
import numpy as np
import sounddevice as sd
import queue
import threading
from datetime import datetime
import os

# Config
SAMPLE_RATE = 16000
CHUNK_DURATION = 4.0 
OVERLAP = 1.0     

SYS_PROMPT = ""
previous_text = ""
total_text = ""

model = WhisperModel("small", compute_type="int8")

audio_queue = queue.Queue()
buffer = np.array([], dtype=np.float32)

chunk_size = int(SAMPLE_RATE * CHUNK_DURATION)
overlap_size = int(SAMPLE_RATE * OVERLAP)


def audio_callback(indata, frames, time, status):
    audio_queue.put(indata.copy())


def asr_worker():
    global buffer, previous_text

    while True:
        data = audio_queue.get()
        audio = np.squeeze(data)
        buffer = np.concatenate((buffer, audio))

        # Wenn genug Audio da dann verarbeiten
        if len(buffer) >= chunk_size:
            chunk = buffer[:chunk_size]

            # Overlap behalten
            buffer = buffer[chunk_size - overlap_size:]

            energy = np.mean(np.abs(chunk))

            if energy < 0.01:
                continue

            init_prompt = SYS_PROMPT + " Kontext: " + previous_text
            segments, _ = model.transcribe(
                chunk,
                initial_prompt=init_prompt,
                language="de",
                vad_filter=True
            )

            text = " ".join([s.text.strip() for s in segments])
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{current_time}]: {text}")

            previous_text = (previous_text + " " + text)[-500:]
            total_text += text

def finished_callback():
    current_time = datetime.now().strftime("%Y-%m-%d_%Hh-%Mm-%Ss")

    file_name = f"Transkripts/{current_time}.txt"

    with open(file=file_name, mode="w", encoding="utf-8") as f:
        f.write(total_text)

if __name__ == "__main__":
    threading.Thread(target=asr_worker, daemon=True).start()

    with sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, finished_callback=finished_callback):
        input("Recording... Press Enter to stop.")