#!/usr/bin/env python3
"""Spielt eine echte Audio-Sample-Datei (WAV oder MP3) über dieselbe Route ab
wie der Live-Betrieb, damit man mit echten Kick-/Snare-Samples kalibrieren
kann statt mit synthetischen Tönen (die ihre eigenen Streuungs-Artefakte haben).

Nutzung:
    python tools/sample_test.py samples/kick1.wav
    python tools/sample_test.py samples/snare1.mp3 5 0.4   # 5x, 0.4s Abstand

MP3 wird per ffmpeg zu einer temporären WAV-Datei dekodiert (muss installiert
sein - `brew install ffmpeg`).
"""
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd


def load_audio(path: str):
    if Path(path).suffix.lower() == ".mp3":
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", path, tmp_path],
                check=True,
            )
            return load_wav(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    return load_wav(path)


def load_wav(path):
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[sampwidth]
    data = np.frombuffer(frames, dtype=dtype)

    if sampwidth == 2:
        data = data.astype(np.float32) / 32768.0
    elif sampwidth == 4:
        data = data.astype(np.float32) / 2147483648.0
    else:
        data = (data.astype(np.float32) - 128) / 128.0

    if n_channels > 1:
        data = data.reshape(-1, n_channels)
    else:
        data = np.column_stack([data, data])
    return data, sr


path = sys.argv[1]
repeat = int(sys.argv[2]) if len(sys.argv) > 2 else 4
interval = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6

data, sr = load_audio(path)
duration = len(data) / sr
print(f"Spiele '{path}' ({duration:.2f}s, {sr} Hz) {repeat}x ab, {interval:.2f}s Abstand...")
for i in range(repeat):
    sd.play(data, sr)
    sd.wait()
    print(f"  Schlag {i + 1}/{repeat}")
    time.sleep(max(0.0, interval - duration))
print("Fertig.")
