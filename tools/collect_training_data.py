#!/usr/bin/env python3
"""Records isolated drum hits (kick, snare, hihat, or clap - one at a time)
and appends a feature vector per detected hit to training_data.csv, for
training the KNN classifier (see tools/train_drum_classifier.py).

Usage:
    python tools/collect_training_data.py kick   [config/club.yaml]
    python tools/collect_training_data.py snare  [config/club.yaml]
    python tools/collect_training_data.py hihat  [config/club.yaml]
    python tools/collect_training_data.py clap   [config/club.yaml]

Play ONLY that drum while it's running (same isolated-instrument methodology
already used tonight for the kick/clap verification tests) - mixing in other
sounds would mislabel the training data. Run it once per label, ideally with
a good number of hits (dozens+) and some natural velocity/timing variation.
"""

import csv
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from capture import AudioCapture, resolve_device  # noqa: E402
from drum_features import extract_features, FEATURE_NAMES  # noqa: E402

LABELS = {"kick", "snare", "hihat", "clap"}

ONSET_THRESHOLD_MULT = 2.5
WINDOW_SECONDS = 0.1


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in LABELS:
        print(f"Usage: python {sys.argv[0]} <{'|'.join(sorted(LABELS))}> [config.yaml]")
        sys.exit(1)
    label = sys.argv[1]
    config_path = sys.argv[2] if len(sys.argv) > 2 else "config/club.yaml"

    cfg = yaml.safe_load(open(config_path))["audio"]
    dev = resolve_device(cfg.get("device_name", ""), cfg.get("fallback_to_default", True))
    sample_rate = cfg.get("sample_rate", 44100)
    block_size = cfg.get("block_size", 512)
    window_samples = int(sample_rate * WINDOW_SECONDS)
    refractory_blocks = max(1, int(sample_rate * 0.15 / block_size))  # ~150ms min gap between hits

    out_path = Path(__file__).parent.parent / "training_data.csv"
    is_new = not out_path.exists()
    out_file = open(out_path, "a", newline="")
    writer = csv.writer(out_file)
    if is_new:
        writer.writerow(FEATURE_NAMES + ["label"])

    recent_flux = deque(maxlen=43)  # ~1s of history at 512/44100 blocks
    state = {"prev_spectrum": None, "capturing": False, "buffer": [], "blocks_since_onset": refractory_blocks, "hits": 0}

    def on_block(block):
        mono = (block[:, 0] + block[:, 1]) * 0.5
        spectrum = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
        state["blocks_since_onset"] += 1

        if state["capturing"]:
            state["buffer"].append(mono.copy())
            if sum(len(b) for b in state["buffer"]) >= window_samples:
                window = np.concatenate(state["buffer"])[:window_samples]
                feats = extract_features(window, sample_rate)
                writer.writerow(list(feats) + [label])
                out_file.flush()
                state["hits"] += 1
                print(f"\r{label}: {state['hits']} Treffer aufgenommen", end="", flush=True)
                state["capturing"] = False
                state["buffer"] = []
        elif state["prev_spectrum"] is not None and len(state["prev_spectrum"]) == len(spectrum):
            flux = float(np.sum(np.maximum(spectrum - state["prev_spectrum"], 0)))
            local_avg = np.mean(recent_flux) if recent_flux else 0.0
            if flux > ONSET_THRESHOLD_MULT * max(local_avg, 1e-6) and state["blocks_since_onset"] >= refractory_blocks:
                state["blocks_since_onset"] = 0
                state["capturing"] = True
                state["buffer"] = [mono.copy()]
            recent_flux.append(flux)
        state["prev_spectrum"] = spectrum

    capture = AudioCapture(dev["index"], sample_rate, cfg.get("channels", 2), cfg.get("channel_pair", 0), block_size, on_block)
    capture.start()
    print(f"Gerät: [{dev['index']}] {dev['name']}")
    print(f"Nimmt '{label}'-Treffer auf. Spiele NUR {label}, moeglichst viele Treffer mit etwas Varianz. Strg+C zum Beenden.\n")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        capture.stop()
        out_file.close()
        print(f"\n\nFertig. {state['hits']} '{label}'-Treffer in {out_path} gespeichert (insgesamt, ueber alle bisherigen Laeufe).")


if __name__ == "__main__":
    main()
