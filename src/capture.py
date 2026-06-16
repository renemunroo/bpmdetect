"""Audio-Capture mit Echtzeit-Pegelanzeige im Terminal."""

import sys
import math
import threading
import time
from typing import Callable, Optional
import numpy as np
import sounddevice as sd

from device_list import find_input_device, get_all_devices


class LevelMeter:
    """Berechnet RMS-Pegel und rendert eine ASCII-Pegelleiste."""

    def __init__(self, bar_width: int = 40, db_floor: float = -60.0) -> None:
        self.bar_width = bar_width
        self.db_floor = db_floor
        self._lock = threading.Lock()
        self._rms_l: float = 0.0
        self._rms_r: float = 0.0
        self._clip_l: bool = False
        self._clip_r: bool = False

    def update(self, block: np.ndarray) -> None:
        """block: shape (frames, 2) — linker und rechter Kanal."""
        left = block[:, 0].astype(np.float32)
        right = block[:, 1].astype(np.float32)
        with self._lock:
            self._rms_l = float(np.sqrt(np.mean(left ** 2)))
            self._rms_r = float(np.sqrt(np.mean(right ** 2)))
            self._clip_l = bool(np.any(np.abs(left) >= 1.0))
            self._clip_r = bool(np.any(np.abs(right) >= 1.0))

    def _db(self, rms: float) -> float:
        if rms < 1e-9:
            return self.db_floor
        return max(self.db_floor, 20.0 * math.log10(rms))

    def _bar(self, db: float, clip: bool) -> str:
        ratio = (db - self.db_floor) / (-self.db_floor)
        ratio = max(0.0, min(1.0, ratio))
        filled = round(ratio * self.bar_width)
        bar = "█" * filled + "░" * (self.bar_width - filled)
        clip_marker = " CLIP" if clip else "     "
        return f"[{bar}]{clip_marker} {db:+6.1f} dBFS"

    def render(self) -> str:
        with self._lock:
            db_l = self._db(self._rms_l)
            db_r = self._db(self._rms_r)
            clip_l = self._clip_l
            clip_r = self._clip_r
        return (
            f"  L {self._bar(db_l, clip_l)}\n"
            f"  R {self._bar(db_r, clip_r)}"
        )


class AudioCapture:
    def __init__(
        self,
        device_index: int,
        sample_rate: int,
        channels: int,
        channel_pair: int,
        block_size: int,
        on_block,
    ) -> None:
        self._device_index = device_index
        self._sample_rate = sample_rate
        self._channels = channels
        self._channel_pair = channel_pair
        self._block_size = block_size
        self._on_block = on_block
        self._stream: Optional[sd.InputStream] = None

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            print(f"\n[audio] {status}", file=sys.stderr)
        ch = self._channel_pair * 2
        # Stereo-Slice aus dem Eingangs-Block extrahieren
        block = indata[:, ch: ch + 2]
        if block.shape[1] < 2:
            # Fallback: Mono auf Stereo duplizieren
            block = np.column_stack([block[:, 0], block[:, 0]])
        self._on_block(block.copy())

    def start(self) -> None:
        self._stream = sd.InputStream(
            device=self._device_index,
            samplerate=self._sample_rate,
            channels=self._channels,
            blocksize=self._block_size,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None


def resolve_device(device_name: str, fallback: bool) -> dict:
    dev = find_input_device(device_name)
    if dev:
        return dev
    if fallback:
        all_devs = get_all_devices()
        default_idx = sd.default.device[0]
        for d in all_devs:
            if d["index"] == default_idx and d["max_input_channels"] > 0:
                print(
                    f'[warn] "{device_name}" nicht gefunden — '
                    f'verwende Standard-Eingabe: {d["name"]}',
                    file=sys.stderr,
                )
                return d
    raise RuntimeError(
        f'Eingabegerät "{device_name}" nicht gefunden und kein Fallback verfügbar.'
    )


def run_monitor(cfg: dict, monitor: dict) -> None:
    a = cfg
    device_name: str = a.get("device_name", "")
    fallback: bool = a.get("fallback_to_default", True)
    sample_rate: int = a.get("sample_rate", 44100)
    channels: int = a.get("channels", 2)
    channel_pair: int = a.get("channel_pair", 0)
    block_size: int = a.get("block_size", 512)

    bar_width: int = monitor.get("bar_width", 40)
    refresh_every: int = monitor.get("refresh_every", 4)
    db_floor: float = monitor.get("db_floor", -60)

    dev = resolve_device(device_name, fallback)
    print(f"\nGerät   : [{dev['index']}] {dev['name']}")
    print(f"API     : {dev['hostapi']}")
    print(f"Kanäle  : {channels} geöffnet, Paar {channel_pair} ({channel_pair*2+1}+{channel_pair*2+2})")
    print(f"Rate    : {sample_rate} Hz  |  Block {block_size} samples")
    print("\nStrg+C zum Beenden\n")

    meter = LevelMeter(bar_width=bar_width, db_floor=db_floor)
    counter = 0

    def on_block(block: np.ndarray) -> None:
        nonlocal counter
        meter.update(block)
        counter += 1
        if counter % refresh_every == 0:
            # Cursor zwei Zeilen hoch, dann überschreiben
            sys.stdout.write("\033[2A")
            sys.stdout.write(meter.render() + "\n")
            sys.stdout.flush()

    # Zwei Leerzeilen als Platzhalter für den ersten Render
    print("  L\n  R")

    capture = AudioCapture(
        device_index=dev["index"],
        sample_rate=sample_rate,
        channels=channels,
        channel_pair=channel_pair,
        block_size=block_size,
        on_block=on_block,
    )
    capture.start()
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        capture.stop()
        print("\n\nAufnahme beendet.")
