#!/usr/bin/env python3
"""Runde 2, Phase 5/6 - schickt eine oder mehrere Audiodateien blockweise
durch BeatDetector.process() und druckt für jeden erkannten ODER
verworfenen Kick-Kandidaten die vollständigen kick_debug-Werte, dazu
Snare-/Clap-Trigger und eine Zusammenfassung pro Datei. Rührt NICHT an der
Live-Audio-Architektur (soundlight_bridge.py, main.py) und NICHT an der
Erkennungslogik in beat_detector.py - reine Offline-Analyse/Ausgabe für die
Validierung mit echten House-/Rap-808-/Latino-Aufnahmen.

── Unterstützte Formate ──────────────────────────────────────────────────
WAV nativ (stdlib `wave`, PCM 8/16/32-bit). MP3/FLAC/M4A/... werden bei
Bedarf per `ffmpeg` nach WAV konvertiert (gleiches Prinzip wie
tools/sample_test.py) - dafür muss ffmpeg installiert sein
(`brew install ffmpeg`); ohne ffmpeg bricht das Skript mit einer klaren
Fehlermeldung ab, statt eine Datei falsch einzulesen.

── Mono/Stereo, Samplingrate, Pegel ──────────────────────────────────────
Stereo-Dateien werden mit BEIDEN Kanälen an BeatDetector.process() gereicht
(das mittelt intern L+R zu mono, siehe process()) - nie nur der linke Kanal.
Mono-Dateien werden auf beide Kanäle dupliziert. Die Samplingrate wird 1:1
aus der Datei übernommen, kein Resampling (ändert sonst die FFT-Bin-Breite
und würde die Kick-Band/HPSS-Kernel-Messung aus Runde 2 verfälschen) - bei
Vergleichen zwischen mehreren Aufnahmen mit UNTERSCHIEDLICHER Samplingrate
wird das im Kopf jeder Analyse explizit ausgegeben. Kein Pegel-Normalisieren
(würde Lautstärke-Vergleiche zwischen Aufnahmen verfälschen) - stattdessen
wird Peak/RMS in dBFS ausgegeben, damit sichtbar ist, ob zwei Dateien
überhaupt ähnlich laut sind, bevor ihre Debugwerte verglichen werden.

── Nutzung ────────────────────────────────────────────────────────────────
  # Variante 1: Defaults
  python tools/analyze_wav.py samples/house1.wav samples/house2.wav --label defaults --csv-out results.csv

  # Variante 2: kleinerer Kick-HPSS-Kernel (nicht dauerhaft übernehmen)
  python tools/analyze_wav.py samples/house1.wav --kick-hpss-kernel-bins 5 --label kernel5 --csv-out results.csv
  python tools/analyze_wav.py samples/house1.wav --kick-hpss-kernel-bins 7 --label kernel7 --csv-out results.csv

  # Variante 3: nur Bassline-Suppression, mehrere min_confidence-Werte
  python tools/analyze_wav.py samples/rap808_1.wav --bassline-suppression --bassline-min-confidence 0.3 --label bl_mc03 --csv-out results.csv
  python tools/analyze_wav.py samples/rap808_1.wav --bassline-suppression --bassline-min-confidence 0.5 --label bl_mc05 --csv-out results.csv

  # Variante 4: beide zusammen, nur wenn 2+3 einzeln verständliche Ergebnisse liefern
  python tools/analyze_wav.py samples/rap808_1.wav --kick-hpss-kernel-bins 7 --bassline-suppression --bassline-min-confidence 0.3 --label combo --csv-out results.csv

Wichtig für Vergleichbarkeit: dieselbe Aufnahme + dieselbe --block-size
zwischen Varianten nicht ändern, nur den einen zu testenden Parameter.
--csv-out sammelt eine Zeile pro Lauf, damit sich die Vergleichstabelle
(Datei × Konfigurationsvariante) über mehrere Aufrufe hinweg zusammenbaut,
ohne Ausgaben manuell abzutippen.

Die Zeit wird NICHT über die echte Wanduhr getaktet (Einlesen ist viel
schneller als Echtzeit) - stattdessen wird time.time() im beat_detector-
Modul auf eine synthetische, mit der Audio-Position mitlaufende Uhr
umgebogen, damit Cooldown/Holdoff-Fenster sich exakt wie live verhalten.

── Benötigte Testdateien (aktuell keine im Projekt vorhanden) ────────────
tools/../samples/ ist leer. Für eine aussagekräftige Auswertung werden
mindestens 3 Aufnahmen pro Richtung empfohlen (30-90s reichen, möglichst
nicht übersteuert/geclippt, keine bereits vorverarbeiteten Kick-Samples
sondern volle Mixe mit überlagerten Basslines/Vocals/Percussion):
  samples/house_1.wav / house_2.wav / house_3.wav
  samples/rap808_1.wav / rap808_2.wav / rap808_3.wav
  samples/latino_1.wav / latino_2.wav / latino_3.wav
Idealerweise alle mit 48000Hz (die tatsächliche Live-Rate von REV7/
BlackHole 2ch, siehe club.yaml, Stand 2026-09-22), damit die FFT-Bin-Breite
(siehe Runde-2-Kernel-Messung) zwischen Offline- und Live-Analyse
vergleichbar bleibt. Diese Datei lädt/verarbeitet keine externen Quellen -
die WAV/MP3-Dateien müssen manuell ins samples/-Verzeichnis gelegt werden.
"""
import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
import wave
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import beat_detector as bd  # noqa: E402


class _AudioClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> float:
        self.t += dt
        return self.t


def _load_wav(path: str):
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

    return data, sr, n_channels


def load_audio(path: str):
    """Lädt eine Audiodatei als (data, sample_rate, original_channels).
    `data` ist immer 2-kanalig (Stereo dupliziert bei Mono-Quelle, NIE nur
    ein Kanal verworfen) - siehe Moduldoku oben. WAV direkt, alles andere
    per ffmpeg nach WAV konvertiert (bricht mit klarer Meldung ab, wenn
    ffmpeg fehlt statt die Datei falsch/leer einzulesen)."""
    p = Path(path)
    if p.suffix.lower() != ".wav":
        if shutil.which("ffmpeg") is None:
            raise SystemExit(
                f"'{path}' ist keine WAV-Datei und ffmpeg ist nicht installiert "
                f"(`brew install ffmpeg`) - kann nicht konvertiert werden."
            )
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, tmp_path], check=True)
            data, sr, n_channels = _load_wav(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    else:
        data, sr, n_channels = _load_wav(path)

    if n_channels > 1:
        stereo = data.reshape(-1, n_channels)[:, :2].astype(np.float32)
        if stereo.shape[1] == 1:
            stereo = np.column_stack([stereo[:, 0], stereo[:, 0]])
    else:
        stereo = np.column_stack([data, data]).astype(np.float32)
    return stereo, sr, n_channels


def _level_dbfs(stereo: np.ndarray) -> tuple:
    mono = (stereo[:, 0] + stereo[:, 1]) * 0.5
    peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
    rms = float(np.sqrt(np.mean(mono ** 2))) if len(mono) else 0.0
    to_db = lambda x: 20 * np.log10(max(x, 1e-9))
    return to_db(peak), to_db(rms)


# Ein Block gilt als "Kick-Kandidat" (loggenswert), wenn seine Hüllkurve
# tatsächlich die ADAPTIVE Basis-Schwelle überschritten hat - reject_reason
# ist dann alles außer "threshold_not_reached"/"insufficient_rise" (reine
# Stille/Rauschen unterhalb der Schwelle, siehe kick_debug-Dokstring in
# beat_detector.py). Reine Ausgabe-Filterlogik, keine Änderung an
# _update_kalman_kick selbst.
_NOISE_REASONS = {"threshold_not_reached", "insufficient_rise"}

KICK_FIELDS = [
    "envelope", "threshold", "is_sharp", "hpss_energy", "hpss_confirms",
    "harmonic_energy", "duration_above_threshold_s", "decay_ratio",
    "centroid_drift_hz", "bassline_score", "confidence", "kalman_state",
    "kalman_accepted", "reject_reason",
]


def _fmt_kalman_state(ks) -> str:
    if not ks:
        return "uninit"
    return f"t_next={ks['t_next']:.3f} period={ks['period_s']:.3f}s"


def analyze_file(path: str, args) -> dict:
    stereo, sr, orig_channels = load_audio(path)
    duration = len(stereo) / sr
    peak_db, rms_db = _level_dbfs(stereo)
    print(f"\n{'=' * 70}\n{path}")
    print(f"  {duration:.1f}s @ {sr}Hz  |  Original-Kanäle: {orig_channels}  |  Peak {peak_db:.1f} dBFS  RMS {rms_db:.1f} dBFS")
    print(f"  Config: block_size={args.block_size}  kick_hpss_freq_kernel_bins={args.kick_hpss_kernel_bins}  "
          f"bassline_suppression={args.bassline_suppression}  bassline_min_confidence={args.bassline_min_confidence}"
          + (f"  Label={args.label}" if args.label else ""))

    detector_kwargs = dict(
        sample_rate=sr, block_size=args.block_size, debug_triggers=True,
        kick_hpss_freq_kernel_bins=args.kick_hpss_kernel_bins,
        kick_bassline_suppression_enabled=args.bassline_suppression,
        kick_bassline_min_confidence=args.bassline_min_confidence,
        kick_bassline_duration_threshold_s=args.bassline_duration_threshold,
        kick_bassline_decay_threshold=args.bassline_decay_threshold,
        kick_pitch_drift_threshold=args.bassline_pitch_drift_threshold,
        kick_bassline_penalty=args.bassline_penalty,
    )
    detector = bd.BeatDetector(**detector_kwargs)
    clock = _AudioClock()
    hop = args.block_size / sr

    # Letzten Teilblock mit Nullen auffüllen statt zu verwerfen - sonst
    # fehlen bei jeder Datei bis zu block_size-1 Samples am Ende.
    n_full = len(stereo) // args.block_size
    remainder = len(stereo) - n_full * args.block_size
    padded = stereo
    n_blocks = n_full
    if remainder:
        pad = np.zeros((args.block_size - remainder, 2), dtype=np.float32)
        padded = np.concatenate([stereo, pad], axis=0)
        n_blocks = n_full + 1

    kick_rows = []
    snare_triggers = 0
    clap_triggers = 0

    with patch.object(bd.time, "time", new=clock):
        for i in range(n_blocks):
            block = padded[i * args.block_size:(i + 1) * args.block_size]
            detector.process(block)
            t = i * hop

            kd = detector.kick_debug
            reason = kd.get("reject_reason")
            # detector.kick_trigger ist das tatsächliche DMX-Ereignis (feuert
            # auf dem VORHERGESAGTEN Kalman-Zeitplan, siehe _kalman_advance) -
            # das kann in einem Block feuern, dessen kick_debug trotzdem
            # einen reject_reason zeigt (der Front-End-Impuls DIESES Blocks
            # wurde z.B. wegen cooldown_active abgelehnt, während der
            # Lichteffekt trotzdem auf der Vorhersage auslöst). Beides wird
            # hier bewusst mitgeloggt, nicht nur "reason is None".
            if detector.kick_trigger or (reason is not None and reason not in _NOISE_REASONS):
                kick_rows.append((t, dict(kd), bool(detector.kick_trigger)))
            if detector.snare_trigger:
                snare_triggers += 1
            if detector.clap_trigger:
                clap_triggers += 1

            clock.advance(hop)

    if args.print_events:
        print(f"\n  {len(kick_rows)} Kick-Kandidaten (DMX-Trigger und/oder Front-End-Kandidat über Basis-Schwelle):\n")
        header = f"  {'Zeit':>7}  {'Trigger':<7}  " + "  ".join(f"{f:<26}" for f in KICK_FIELDS)
        print(header)
        for t, kd, trig in kick_rows:
            vals = []
            for f in KICK_FIELDS:
                v = kd.get(f)
                if f == "kalman_state":
                    v = _fmt_kalman_state(v)
                elif isinstance(v, float):
                    v = f"{v:.4f}"
                elif v is None:
                    v = "—"
                vals.append(f"{str(v):<26}")
            print(f"  {t:7.2f}s  {'JA' if trig else '—':<7}  " + "  ".join(vals))

    # "Kick-Trigger" = das tatsächliche DMX-Ereignis (detector.kick_trigger),
    # nicht dasselbe wie reject_reason is None (siehe Kommentar oben bei der
    # Filterlogik - kann auf dem vorhergesagten Kalman-Zeitplan auch in
    # einem Block feuern, dessen Front-End-Impuls abgelehnt wurde).
    fired = [kd for _, kd, trig in kick_rows if trig]
    rejected = [kd for _, kd, _ in kick_rows if kd.get("reject_reason") is not None]
    reject_counter = Counter(kd["reject_reason"] for kd in rejected)
    confidences = [kd.get("confidence", 0.0) for _, kd, _ in kick_rows]
    bassline_scores = [kd.get("bassline_score", 0.0) for _, kd, _ in kick_rows]
    bassline_suspected = reject_counter.get("bassline_suspected", 0)

    summary = {
        "file": path,
        "label": args.label,
        "sample_rate": sr,
        "duration_s": round(duration, 1),
        "peak_dbfs": round(peak_db, 1),
        "rms_dbfs": round(rms_db, 1),
        "block_size": args.block_size,
        "kick_hpss_freq_kernel_bins": args.kick_hpss_kernel_bins,
        "bassline_suppression_enabled": args.bassline_suppression,
        "bassline_min_confidence": args.bassline_min_confidence,
        "kick_triggers": len(fired),
        "kick_candidates_rejected": len(rejected),
        "top_reject_reasons": ", ".join(f"{r}×{n}" for r, n in reject_counter.most_common(3)),
        "confidence_avg": round(float(np.mean(confidences)), 3) if confidences else None,
        "confidence_max": round(float(np.max(confidences)), 3) if confidences else None,
        "bassline_score_avg": round(float(np.mean(bassline_scores)), 3) if bassline_scores else None,
        "bassline_suspected_count": bassline_suspected,
        "snare_triggers": snare_triggers,
        "clap_triggers": clap_triggers,
        "bpm_estimate": round(detector.bpm, 1),
        "bpm_confidence": round(detector.confidence, 2),
    }

    print(f"\n  Zusammenfassung {path}:")
    for k, v in summary.items():
        print(f"    {k:<28}: {v}")

    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav_paths", nargs="*", help="eine oder mehrere Audiodateien (WAV nativ, sonst per ffmpeg konvertiert)")
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--print-events", action="store_true", default=True, help="druckt jede Kick-Kandidaten-Zeile einzeln (Default: an)")
    ap.add_argument("--no-print-events", dest="print_events", action="store_false")
    ap.add_argument("--label", default="", help="Kennzeichnung der Konfigurationsvariante für Vergleichstabellen (z.B. 'kernel7')")
    ap.add_argument("--csv-out", default=None, help="hängt eine Zusammenfassungszeile pro Datei an diese CSV an")
    # Runde-2-Parameter, testweise pro Lauf überschreibbar - NICHT die Defaults in club.yaml ändern.
    ap.add_argument("--kick-hpss-kernel-bins", dest="kick_hpss_kernel_bins", type=int, default=None)
    ap.add_argument("--bassline-suppression", dest="bassline_suppression", action="store_true", default=False)
    ap.add_argument("--bassline-min-confidence", dest="bassline_min_confidence", type=float, default=0.3)
    ap.add_argument("--bassline-duration-threshold", dest="bassline_duration_threshold", type=float, default=0.35)
    ap.add_argument("--bassline-decay-threshold", dest="bassline_decay_threshold", type=float, default=0.6)
    ap.add_argument("--bassline-pitch-drift-threshold", dest="bassline_pitch_drift_threshold", type=float, default=15.0)
    ap.add_argument("--bassline-penalty", dest="bassline_penalty", type=float, default=0.5)
    args = ap.parse_args()

    if not args.wav_paths:
        print(__doc__)
        return

    summaries = [analyze_file(p, args) for p in args.wav_paths]

    if args.csv_out:
        write_header = not Path(args.csv_out).exists()
        with open(args.csv_out, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            if write_header:
                writer.writeheader()
            writer.writerows(summaries)
        print(f"\n{len(summaries)} Zusammenfassungszeile(n) an '{args.csv_out}' angehängt.")

    if len(summaries) > 1:
        print(f"\n{'=' * 70}\nÜbersicht ({len(summaries)} Dateien):")
        cols = ["file", "label", "kick_triggers", "kick_candidates_rejected", "bassline_suspected_count", "snare_triggers", "clap_triggers", "bpm_estimate"]
        print("  " + "  ".join(f"{c:<20}" for c in cols))
        for s in summaries:
            print("  " + "  ".join(f"{str(s[c]):<20}" for c in cols))


if __name__ == "__main__":
    main()
