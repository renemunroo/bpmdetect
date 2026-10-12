#!/usr/bin/env python3
"""Spielt gepulste Sinustöne ab, um Kick/Snare/Hihat-Bandtrennung gegen eine
bekannte Frequenz zu kalibrieren, statt gegen echte (unvorhersehbare) Musik
zu raten.

Wichtig: der Detector misst spektralen FLUX (Veränderung), nicht Lautstärke -
ein Dauerton erzeugt nach dem Einsetzen keine weitere Veränderung mehr und
zeigt daher praktisch nichts an. Dieses Tool spielt deshalb kurze Bursts
("Schläge") mit Pausen dazwischen, wie ein echtes Schlagzeug.

Nutzung:
    python tools/tone_test.py 60          # 60 Hz, 6 Schläge, 500ms Abstand
    python tools/tone_test.py 300 8 0.4   # 300 Hz, 8 Schläge, 400ms Abstand
"""
import sys
import time
import numpy as np
import sounddevice as sd

freq = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
count = int(sys.argv[2]) if len(sys.argv) > 2 else 6
interval = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5

sr = 44100
burst_len = 0.06  # 60ms Burst - kurz genug um ein Schlagzeug-Attack nachzuahmen
amplitude = 0.4

t = np.linspace(0, burst_len, int(sr * burst_len), endpoint=False)
# Hüllkurve (schneller Attack, exponentieller Decay) statt hartem Rechteck -
# vermeidet Klick-Artefakte, die selbst breitbandiges Rauschen erzeugen würden.
envelope = np.exp(-t * 40)
tone = (amplitude * np.sin(2 * np.pi * freq * t) * envelope).astype(np.float32)
stereo = np.column_stack([tone, tone])

print(f"{count}x {freq:.0f} Hz-Schlag, alle {interval:.2f}s...")
for i in range(count):
    sd.play(stereo, sr)
    sd.wait()
    print(f"  Schlag {i + 1}/{count}")
    time.sleep(max(0.0, interval - burst_len))
print("Fertig.")
