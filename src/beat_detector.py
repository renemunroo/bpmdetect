"""Echtzeit-Beat-Detection und BPM-Schätzung.

Ansatz:
  1. Onset Strength Signal (OSS): gewichtete Summe aus drei Drum-Bändern
     (Kick, Snare, Hi-Hat), jeweils halbwellen-gleichgerichteter Spectral Flux.
  2. Tempo-Schätzung: Autokorrelation der OSS über ein gleitendes Fenster
     (~8 s). Liefert mehrere Kandidaten mit Scores.
  3. Octave-Korrektur: Verhältnis von Kandidaten-Scores für 1x/2x/0.5x-Tempo.
  4. Stabilisierung: exponentieller gleitender Mittelwert mit Hysterese.
  5. Beat-Phase: Flux-Peaks als Onset-Ereignisse für Phasen-Tracking.
"""

import sys
import time
import threading
from collections import deque

import numpy as np

from capture import AudioCapture, LevelMeter, resolve_device


# ── Hilfsfunktionen ───────────────────────────────────────────────────────────

def _find_peaks(arr: np.ndarray, min_dist: int = 2) -> np.ndarray:
    """Lokale Maxima in arr, sortiert nach Wert (absteigend)."""
    n = len(arr)
    peaks = []
    for i in range(1, n - 1):
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
            peaks.append(i)
    # Mindestabstand: benachbarte Peaks zusammenführen
    if min_dist > 1 and peaks:
        filtered, prev = [peaks[0]], peaks[0]
        for p in peaks[1:]:
            if p - prev >= min_dist:
                filtered.append(p)
                prev = p
        peaks = filtered
    peaks.sort(key=lambda i: -arr[i])
    return np.array(peaks, dtype=int)


def _phase_bar(phase: float, width: int = 16) -> str:
    pos = int(phase * width) % width
    bar = ["·"] * width
    bar[pos] = "●"
    return "".join(bar)


def _conf_bar(conf: float, width: int = 20) -> str:
    filled = round(conf * width)
    return "█" * filled + "░" * (width - filled)


# ── Beat-Detector ─────────────────────────────────────────────────────────────

class BeatDetector:
    """
    Autokorrelations-basierter Tempo-Schätzer mit gewichtetem Mehrband-OSS.

    Das OSS (Onset Strength Signal) ist eine gewichtete Summe aus drei
    Drum-Bändern (Kick, Snare, Hi-Hat). Alle weiteren Schritte
    (Autokorrelation, Oktav-Korrektur, EMA) bleiben unverändert.

    Parameters
    ----------
    sample_rate, block_size
        Audio-Parameter des Capture-Streams.
    bpm_min, bpm_max
        Zulässiger Tempo-Bereich.
    kick_lo/hi, snare_lo/hi, hihat_lo/hi
        Frequenzgrenzen (Hz) der drei Bänder.
    kick_weight, snare_weight, hihat_weight
        Mischgewichte (müssen nicht auf 1 normiert sein).
    oss_window_s
        Länge des OSS-Puffers für die Autokorrelation (Sekunden).
    onset_threshold, refractory
        Flux-Peak-Erkennung für Beat-Phase-Tracking.
    smoothing
        EMA-Glättungsfaktor α (0 = starr, 1 = sofort).
    hysteresis
        Mindeständerung in BPM, ab der der EMA reagiert.
    """

    def __init__(
        self,
        sample_rate: int,
        block_size: int,
        bpm_min: float = 70.0,
        bpm_max: float = 180.0,
        kick_lo: float = 30.0,
        kick_hi: float = 120.0,
        snare_lo: float = 180.0,
        snare_hi: float = 600.0,
        hihat_lo: float = 2000.0,
        hihat_hi: float = 8000.0,
        kick_weight: float = 0.6,
        snare_weight: float = 0.25,
        hihat_weight: float = 0.15,
        oss_window_s: float = 8.0,
        onset_threshold: float = 1.4,
        refractory: float = 0.2,
        smoothing: float = 0.2,
        hysteresis: float = 1.0,
        # ── Tempo-Hold / Lock ──────────────────────────────────────────
        lock_confidence_min: float = 0.3,
        relock_confidence_min: float = 0.5,
        max_jump_bpm: float = 10.0,
        hard_block_jump_bpm: float = 15.0,
        relock_windows: int = 3,
        hold_seconds: float = 8.0,
        large_jump_bpm: float = 20.0,
        large_jump_hold_s: float = 5.0,
    ) -> None:
        self._sr = sample_rate
        self._bs = block_size
        self._hop = block_size / sample_rate
        self._bpm_min = bpm_min
        self._bpm_max = bpm_max
        self._onset_threshold = onset_threshold
        self._refractory = refractory
        self._smoothing = smoothing
        self._hysteresis = hysteresis
        self._band_weights = (kick_weight, snare_weight, hihat_weight)

        # ── Tempo-Hold-Parameter ──────────────────────────────────────
        self._lock_conf_min = lock_confidence_min
        self._relock_conf_min = relock_confidence_min
        self._max_jump_bpm = max_jump_bpm
        self._hard_block_jump_bpm = hard_block_jump_bpm
        self._relock_windows = relock_windows
        self._hold_seconds = hold_seconds
        self._large_jump_bpm = large_jump_bpm
        self._large_jump_hold_s = large_jump_hold_s

        # OSS-Puffer (ein Wert pro Block)
        oss_len = max(64, int(oss_window_s / self._hop))
        self._oss: deque = deque(maxlen=oss_len)
        self._oss_short: deque = deque(maxlen=max(8, int(0.5 / self._hop)))
        self._prev_mag = None  # type: np.ndarray

        # Analyse-FFT über mehrere Blöcke akkumuliert:
        # fft_size >> block_size → bessere Frequenzauflösung im Kick-Band.
        # Bei 512/44100 wären es 86 Hz/Bin (Kick-Band = 1 Bin).
        # Bei 4096/44100 sind es 10.8 Hz/Bin (Kick-Band = ~8 Bins).
        self._fft_size = 4096
        self._sample_buf: deque = deque(maxlen=self._fft_size)
        freqs = np.fft.rfftfreq(self._fft_size, d=1.0 / sample_rate)
        self._mask_kick  = ((freqs >= kick_lo)  & (freqs <= kick_hi)).astype(np.float32)
        self._mask_snare = ((freqs >= snare_lo) & (freqs <= snare_hi)).astype(np.float32)
        self._mask_hihat = ((freqs >= hihat_lo) & (freqs <= hihat_hi)).astype(np.float32)
        self._hann = np.hanning(self._fft_size).astype(np.float32)

        # Onset-Tracking für Beat-Phase
        self._last_onset_t: float = 0.0
        self._last_beat_t: float = 0.0

        # Geschützter Zustand
        self._lock = threading.Lock()
        self._bpm_raw: float = 0.0
        self._bpm_corrected: float = 0.0
        self._bpm_smooth: float = 0.0      # interner EMA (Phase-Fallback)
        self._candidates: list = []
        self._confidence: float = 0.0
        self._beat_phase: float = 0.0
        # Band-Scores (normierte mittlere Flux-Energie pro Band, für Anzeige)
        self._kick_score: float = 0.0
        self._snare_score: float = 0.0
        self._hihat_score: float = 0.0
        # ── Tempo-Hold-State ──────────────────────────────────────────
        self._tempo_state: str = "searching"   # searching | locked | relocking
        self._locked_bpm: float = 0.0
        self._relock_candidate: float = 0.0
        self._relock_progress: int = 0
        self._relock_candidate_since: float = 0.0
        self._low_conf_since: float = 0.0

        # Tempo-Update nur alle ~1 s
        self._tempo_update_every = max(1, int(1.0 / self._hop))
        self._block_count: int = 0

    # ── public API ────────────────────────────────────────────────────────────

    def process(self, block: np.ndarray) -> None:
        """block: shape (frames, 2), float32."""
        now = time.time()
        self._block_count += 1

        mono = (block[:, 0] + block[:, 1]) * 0.5
        kick_f, snare_f, hihat_f, combined = self._spectral_flux_bands(mono)

        # Band-Scores für Anzeige: gleitender Mittelwert (EMA α=0.1)
        a = 0.1
        with self._lock:
            self._kick_score  = a * kick_f  + (1 - a) * self._kick_score
            self._snare_score = a * snare_f + (1 - a) * self._snare_score
            self._hihat_score = a * hihat_f + (1 - a) * self._hihat_score

        self._oss.append(combined)
        self._oss_short.append(combined)

        # Onset-Erkennung für Beat-Phase (kombinierter Flux über lokalem Mittel)
        if len(self._oss_short) >= 4:
            local_avg = float(np.mean(self._oss_short))
            if (
                combined > self._onset_threshold * local_avg
                and combined > 1e-9
                and (now - self._last_onset_t) >= self._refractory
            ):
                self._last_onset_t = now
                self._last_beat_t = now

        # Tempo-Schätzung periodisch
        if self._block_count % self._tempo_update_every == 0:
            self._update_tempo(now)

        # Beat-Phase: locked_bpm wenn verfügbar, sonst interner EMA
        with self._lock:
            bpm_for_phase = (
                self._locked_bpm if self._locked_bpm > 0 else self._bpm_smooth
            )
            if bpm_for_phase > 0 and self._last_beat_t > 0:
                period = 60.0 / bpm_for_phase
                elapsed = (now - self._last_beat_t) % period
                self._beat_phase = elapsed / period

    @property
    def bpm(self) -> float:
        with self._lock:
            return self._bpm_smooth

    @property
    def bpm_raw(self) -> float:
        with self._lock:
            return self._bpm_raw

    @property
    def bpm_corrected(self) -> float:
        with self._lock:
            return self._bpm_corrected

    @property
    def candidates(self) -> list:
        with self._lock:
            return list(self._candidates)

    @property
    def confidence(self) -> float:
        with self._lock:
            return self._confidence

    @property
    def beat_phase(self) -> float:
        with self._lock:
            return self._beat_phase

    @property
    def band_scores(self) -> tuple[float, float, float]:
        """(kick_score, snare_score, hihat_score) — EMA-geglättete Flux-Energie."""
        with self._lock:
            return (self._kick_score, self._snare_score, self._hihat_score)

    @property
    def locked_bpm(self) -> float:
        with self._lock:
            return self._locked_bpm

    @property
    def tempo_state(self) -> str:
        with self._lock:
            return self._tempo_state

    @property
    def relock_candidate(self) -> float:
        with self._lock:
            return self._relock_candidate

    @property
    def relock_progress(self) -> tuple[int, int]:
        """(current_progress, relock_windows_needed)"""
        with self._lock:
            return (self._relock_progress, self._relock_windows)

    def set_band_weights(self, kick: float, snare: float, hihat: float) -> None:
        """Setzt die Mischgewichte der drei Drum-Bänder zur Laufzeit."""
        self._band_weights = (kick, snare, hihat)  # tuple-Zuweisung ist atomar

    # ── interne Methoden ──────────────────────────────────────────────────────

    def _spectral_flux_bands(
        self, mono: np.ndarray
    ) -> tuple[float, float, float, float]:
        """
        Halbwellen-gleichgerichteter Spectral Flux pro Drum-Band.

        Akkumuliert Samples in _sample_buf auf fft_size=4096 (≈93 ms bei
        44100 Hz), was eine Frequenzauflösung von ~10.8 Hz/Bin ergibt und
        damit mehrere Bins im Kick-Band (30–120 Hz) sicherstellt.

        Gibt (kick_flux, snare_flux, hihat_flux, combined) zurück.
        """
        self._sample_buf.extend(mono.tolist())

        # Erst rechnen wenn Puffer voll ist
        if len(self._sample_buf) < self._fft_size:
            return 0.0, 0.0, 0.0, 0.0

        frame = np.array(self._sample_buf, dtype=np.float32)
        windowed = frame * self._hann
        mag = np.abs(np.fft.rfft(windowed)).astype(np.float32)

        if self._prev_mag is None:
            self._prev_mag = mag
            return 0.0, 0.0, 0.0, 0.0

        diff = np.maximum(0.0, mag - self._prev_mag)
        self._prev_mag = mag

        kick_f  = float(np.sum(diff * self._mask_kick))
        snare_f = float(np.sum(diff * self._mask_snare))
        hihat_f = float(np.sum(diff * self._mask_hihat))

        wk, ws, wh = self._band_weights
        combined = wk * kick_f + ws * snare_f + wh * hihat_f
        return kick_f, snare_f, hihat_f, combined

    def _acf(self, signal: np.ndarray) -> np.ndarray:
        """Normalisierte Autokorrelation via FFT."""
        n = len(signal)
        # Zero-padding auf nächste 2er-Potenz für Effizienz
        pad = int(2 ** np.ceil(np.log2(2 * n)))
        f = np.fft.rfft(signal - signal.mean(), n=pad)
        acf = np.fft.irfft(f * np.conj(f))[:n].real
        if acf[0] > 1e-12:
            acf /= acf[0]
        return acf

    def _octave_correct(self, bpm: float, acf: np.ndarray, lag_min: int) -> float:
        """
        Prüft ob 0.5x oder 2x des Kandidaten einen höheren ACF-Score hat
        und korrigiert entsprechend (Half-/Double-Time).
        """
        hop = self._hop
        lag = 60.0 / bpm / hop  # Lag in Blöcken

        def acf_at(b: float) -> float:
            i = int(round(b))
            if lag_min <= i < len(acf):
                return float(acf[i])
            return -1.0

        score_1x = acf_at(lag)
        score_half = acf_at(lag * 2)   # halbes Tempo (doppelter Lag)
        score_double = acf_at(lag / 2) # doppeltes Tempo (halber Lag)

        best_bpm = bpm
        best_score = score_1x

        # Half-time: nur wenn deutlich stärker UND im erlaubten Bereich
        half_bpm = bpm / 2
        if (
            score_half > best_score * 1.15
            and self._bpm_min <= half_bpm <= self._bpm_max
        ):
            best_bpm, best_score = half_bpm, score_half

        # Double-time
        double_bpm = bpm * 2
        if (
            score_double > best_score * 1.15
            and self._bpm_min <= double_bpm <= self._bpm_max
        ):
            best_bpm, best_score = double_bpm, score_double

        return best_bpm

    def _apply_lock(self, bpm_corrected: float, conf: float, now: float) -> None:
        """
        Tempo-Hold-State-Machine — wird innerhalb des Locks aufgerufen.

        Zustandsübergänge
        -----------------
        searching  → locked    : conf ≥ lock_conf_min, gültige BPM
        locked     → locked    : kleine Änderung (≤ max_jump) per EMA absorbiert
        locked     → relocking : jede größere Änderung mit conf ≥ relock_conf_min
        locked     → searching : conf < lock_conf_min für > hold_seconds
        relocking  → locked    : relock_progress ≥ relock_windows
        relocking  → locked    : Kandidat driftet weg → Relock abgebrochen
        relocking  → locked    : Conf fällt → Relock abgebrochen
        """
        state = self._tempo_state

        if state == "searching":
            if conf >= self._lock_conf_min and bpm_corrected > 0:
                self._locked_bpm = bpm_corrected
                self._tempo_state = "locked"
                self._low_conf_since = 0.0

        elif state == "locked":
            if conf < self._lock_conf_min:
                # Low-Confidence-Phase: Pegel halten
                if self._low_conf_since == 0.0:
                    self._low_conf_since = now
                if (now - self._low_conf_since) > self._hold_seconds:
                    self._tempo_state = "searching"
                # locked_bpm bleibt unverändert
            else:
                self._low_conf_since = 0.0
                jump = abs(bpm_corrected - self._locked_bpm)

                if jump <= self._max_jump_bpm:
                    # Kleine Änderung: per Smoothing absorbieren
                    if jump > self._hysteresis:
                        self._locked_bpm = (
                            self._smoothing * bpm_corrected
                            + (1.0 - self._smoothing) * self._locked_bpm
                        )
                elif conf >= self._relock_conf_min:
                    # Jede größere Änderung: Relock einleiten (kein hard_block mehr)
                    tol = bpm_corrected * 0.04
                    if (
                        self._relock_candidate == 0.0
                        or abs(bpm_corrected - self._relock_candidate) > tol
                    ):
                        # Neuer Kandidat — Zähler zurücksetzen
                        self._relock_candidate = bpm_corrected
                        self._relock_progress = 1
                        self._relock_candidate_since = now
                    else:
                        self._relock_progress += 1

                    # Große Sprünge (> large_jump_bpm) brauchen zusätzlich
                    # large_jump_hold_s Sekunden stabile Erkennung
                    is_large = jump > self._large_jump_bpm
                    hold_ok = (
                        (now - self._relock_candidate_since) >= self._large_jump_hold_s
                        if is_large else True
                    )

                    if self._relock_progress >= self._relock_windows and hold_ok:
                        self._locked_bpm = self._relock_candidate
                        self._relock_candidate = 0.0
                        self._relock_progress = 0
                        self._relock_candidate_since = 0.0
                        self._tempo_state = "locked"
                    else:
                        self._tempo_state = "relocking"

        elif state == "relocking":
            if conf < self._lock_conf_min:
                # Signal weg während Relock → abbrechen
                self._relock_candidate = 0.0
                self._relock_progress = 0
                self._tempo_state = "locked"
            else:
                tol = bpm_corrected * 0.04
                if abs(bpm_corrected - self._relock_candidate) <= tol:
                    self._relock_progress += 1
                    if self._relock_progress >= self._relock_windows:
                        self._locked_bpm = self._relock_candidate
                        self._relock_candidate = 0.0
                        self._relock_progress = 0
                        self._tempo_state = "locked"
                else:
                    # Kandidat hat sich verschoben → Relock abbrechen
                    self._relock_candidate = 0.0
                    self._relock_progress = 0
                    self._relock_candidate_since = 0.0
                    self._tempo_state = "locked"

    def _update_tempo(self, now: float) -> None:
        oss = np.array(self._oss, dtype=np.float32)
        if len(oss) < 32:
            return

        acf = self._acf(oss)
        hop = self._hop

        # Lag-Bereich: BPM-Range → Block-Lags
        lag_min = max(1, int(60.0 / self._bpm_max / hop))
        lag_max = min(len(acf) - 1, int(60.0 / self._bpm_min / hop))
        if lag_min >= lag_max:
            return

        region = acf[lag_min : lag_max + 1]

        # Mindestabstand zwischen Peaks: ~10 BPM in Block-Lags
        min_dist = max(2, int((60.0 / (self._bpm_max - 10) / hop) * 0.1))
        peak_indices = _find_peaks(region, min_dist=min_dist)

        if len(peak_indices) == 0:
            return

        # Top-Kandidaten: BPM + ACF-Score
        candidates_raw = []
        for pi in peak_indices[:8]:
            lag = lag_min + int(pi)
            bpm = 60.0 / (lag * hop)
            score = float(region[pi])
            if self._bpm_min <= bpm <= self._bpm_max and score > 0:
                candidates_raw.append((bpm, score))

        if not candidates_raw:
            return

        # Bester Kandidat (höchster ACF-Score)
        best_bpm, best_score = candidates_raw[0]
        bpm_raw = best_bpm

        # Oktav-Korrektur
        bpm_corrected = self._octave_correct(best_bpm, acf, lag_min)

        # Confidence: wie stark der beste Peak über dem Median der Lag-Region liegt.
        # (best_score - median) / (max - median) → 0 bei flachem ACF (Rauschen),
        # → 1 bei klar periodischem Signal.
        acf_median = float(np.median(region))
        acf_range  = float(np.max(region)) - acf_median
        if acf_range > 1e-9:
            conf = min(1.0, max(0.0, (best_score - acf_median) / acf_range))
        else:
            conf = 0.0

        # Energy-Gate: bei beatlosen Passagen/Stille fällt die aktuelle OSS-Energie
        # unter den Fensterdurchschnitt. Confidence wird proportional gedämpft,
        # damit der Lock-Zähler schnell auf Low-Confidence reagiert.
        if len(self._oss_short) >= 4:
            recent_energy = float(np.mean(list(self._oss_short)))
            window_energy = float(np.mean(oss)) + 1e-12
            if window_energy > 1e-9:
                energy_ratio = recent_energy / window_energy
                if energy_ratio < 0.15:
                    conf *= energy_ratio / 0.15

        # Top-3 für Anzeige (nach Score)
        top3 = [(bpm, sc) for bpm, sc in candidates_raw[:3]]

        with self._lock:
            self._bpm_raw = bpm_raw
            self._bpm_corrected = bpm_corrected
            self._candidates = top3
            self._confidence = conf

            # Interner EMA — nur für Phase-Fallback im searching-Zustand
            if conf > 0.05:
                if self._bpm_smooth == 0.0:
                    self._bpm_smooth = bpm_corrected
                    self._last_beat_t = now
                else:
                    d = abs(bpm_corrected - self._bpm_smooth)
                    if d > self._hysteresis:
                        self._bpm_smooth = (
                            self._smoothing * bpm_corrected
                            + (1.0 - self._smoothing) * self._bpm_smooth
                        )

            # Tempo-Hold-State-Machine
            self._apply_lock(bpm_corrected, conf, now)


# ── Terminal-Display ──────────────────────────────────────────────────────────

def _score_bar(val: float, peak: float, width: int = 12) -> str:
    """Mini-Balken für einen Band-Score relativ zum Peak-Wert."""
    ratio = (val / peak) if peak > 1e-12 else 0.0
    ratio = max(0.0, min(1.0, ratio))
    filled = round(ratio * width)
    return "▪" * filled + "·" * (width - filled)


class BpmDisplay:
    """Rendert BPM-Diagnose, Band-Scores und Pegelleiste ins Terminal."""

    def __init__(self, bar_width: int = 40, db_floor: float = -60) -> None:
        self._meter = LevelMeter(bar_width=bar_width, db_floor=db_floor)
        self._initialized = False
        # Laufendes Maximum pro Band — für relativen Score-Balken
        self._peak_kick: float = 1e-12
        self._peak_snare: float = 1e-12
        self._peak_hihat: float = 1e-12

    def update_meter(self, block: np.ndarray) -> None:
        self._meter.update(block)

    def render(self, det: BeatDetector, link=None) -> None:
        locked      = det.locked_bpm
        bpm_raw     = det.bpm_raw
        bpm_cor     = det.bpm_corrected
        conf        = det.confidence
        phase       = det.beat_phase
        cands       = det.candidates
        state       = det.tempo_state
        relock_cand = det.relock_candidate
        prog, total = det.relock_progress
        kick_s, snare_s, hihat_s = det.band_scores

        # Peak-Tracking mit langsamem Decay (×0.999 pro Render-Aufruf)
        self._peak_kick  = max(kick_s,  self._peak_kick  * 0.999)
        self._peak_snare = max(snare_s, self._peak_snare * 0.999)
        self._peak_hihat = max(hihat_s, self._peak_hihat * 0.999)

        # State-Label mit Farb-Marker (ANSI: grün=locked, gelb=relocking, rot=searching)
        _state_color = {"locked": "\033[32m", "relocking": "\033[33m", "searching": "\033[31m"}
        _reset = "\033[0m"
        state_str = f"{_state_color.get(state,'')}{state:<10}{_reset}"

        locked_str   = f"{locked:6.1f}"  if locked  > 0 else "  ---.-"
        raw_str      = f"{bpm_raw:6.1f}" if bpm_raw > 0 else "  ---.-"
        cor_str      = f"{bpm_cor:6.1f}" if bpm_cor > 0 else "  ---.-"
        cand_str     = "  ".join(f"{b:.1f}({s:.2f})" for b, s in cands) if cands else "—"
        relock_str   = (
            f"{relock_cand:.1f} BPM  [{prog}/{total} Fenster]"
            if relock_cand > 0 else "—"
        )

        # Link-Sektion (immer mit fixer Zeilenanzahl, egal ob enabled)
        if link is not None and link.available:
            _lc = "\033[32m"  # grün
            link_enabled_str = f"{_lc}yes{_reset}"
            sess_str  = f"{link.session_tempo:6.1f} BPM"
            exp_str   = f"{link.exported_bpm:6.1f} BPM" if link.exported_bpm > 0 else "  ---.- BPM"
            poff      = link.phase_offset
            poff_str  = f"{poff:+.3f}" if link.session_tempo > 0 else "  ---"
            peers_str = str(link.peers)
        else:
            _lc = "\033[31m"
            link_enabled_str = f"{_lc}no{_reset} " if link is not None else f"{_lc}disabled{_reset}"
            sess_str  = "  ---.- BPM"
            exp_str   = "  ---.- BPM"
            poff_str  = "  ---"
            peers_str = "—"

        lines = [
            f"  tempo_state     : {state_str}",
            f"  locked_bpm      : {locked_str}  [{_conf_bar(conf)}]  Conf {conf*100:5.1f}%",
            f"  raw_bpm         : {raw_str}",
            f"  corrected_bpm   : {cor_str}",
            f"  candidate_bpms  : {cand_str}",
            f"  relock_candidate: {relock_str}",
            f"  Beat-Phase      : [{_phase_bar(phase)}]",
            f"  {'─' * 55}",
            f"  kick_score      : [{_score_bar(kick_s,  self._peak_kick)}]  {kick_s:8.4f}",
            f"  snare_score     : [{_score_bar(snare_s, self._peak_snare)}]  {snare_s:8.4f}",
            f"  hihat_score     : [{_score_bar(hihat_s, self._peak_hihat)}]  {hihat_s:8.4f}",
            f"  {'─' * 55}",
            f"  link_enabled    : {link_enabled_str}  Peers {peers_str}",
            f"  session_tempo   : {sess_str}",
            f"  exported_bpm    : {exp_str}",
            f"  phase_offset    : {poff_str}",
            f"  {'─' * 55}",
        ]
        # Pegelzeilen anhängen (meter.render() → "  L ...\n  R ...")
        lines += self._meter.render().split("\n")

        if not self._initialized:
            print("\n" * len(lines), end="")
            self._initialized = True

        sys.stdout.write(f"\033[{len(lines)}A")
        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()


# ── Entry-Point ───────────────────────────────────────────────────────────────

def run_bpm(cfg: dict, monitor: dict, bpm_cfg: dict, link_cfg=None) -> None:
    device_name: str = cfg.get("device_name", "")
    fallback: bool = cfg.get("fallback_to_default", True)
    sample_rate: int = cfg.get("sample_rate", 44100)
    channels: int = cfg.get("channels", 2)
    channel_pair: int = cfg.get("channel_pair", 0)
    block_size: int = cfg.get("block_size", 512)

    bar_width: int = monitor.get("bar_width", 40)
    refresh_every: int = monitor.get("refresh_every", 4)
    db_floor: float = monitor.get("db_floor", -60)

    bpm_min: float        = bpm_cfg.get("bpm_min", 70.0)
    bpm_max: float        = bpm_cfg.get("bpm_max", 180.0)
    kick_lo: float        = bpm_cfg.get("kick_lo", 30.0)
    kick_hi: float        = bpm_cfg.get("kick_hi", 120.0)
    snare_lo: float       = bpm_cfg.get("snare_lo", 180.0)
    snare_hi: float       = bpm_cfg.get("snare_hi", 600.0)
    hihat_lo: float       = bpm_cfg.get("hihat_lo", 2000.0)
    hihat_hi: float       = bpm_cfg.get("hihat_hi", 8000.0)
    kick_weight: float    = bpm_cfg.get("kick_weight", 0.6)
    snare_weight: float   = bpm_cfg.get("snare_weight", 0.25)
    hihat_weight: float   = bpm_cfg.get("hihat_weight", 0.15)
    oss_window_s: float   = bpm_cfg.get("oss_window_s", 8.0)
    onset_threshold: float    = bpm_cfg.get("onset_threshold", 1.4)
    refractory: float         = bpm_cfg.get("refractory", 0.2)
    smoothing: float          = bpm_cfg.get("smoothing", 0.2)
    hysteresis: float         = bpm_cfg.get("hysteresis", 1.0)
    lock_conf_min: float      = bpm_cfg.get("lock_confidence_min", 0.3)
    relock_conf_min: float    = bpm_cfg.get("relock_confidence_min", 0.5)
    max_jump_bpm: float       = bpm_cfg.get("max_jump_bpm", 10.0)
    hard_block_jump: float    = bpm_cfg.get("hard_block_jump_bpm", 15.0)
    relock_windows: int       = bpm_cfg.get("relock_windows", 3)
    hold_seconds: float       = bpm_cfg.get("hold_seconds", 8.0)

    dev = resolve_device(device_name, fallback)
    print(f"\nGerät        : [{dev['index']}] {dev['name']}")
    print(f"Rate         : {sample_rate} Hz  |  Block {block_size}  |  Paar {channel_pair}")
    print(f"BPM-Bereich  : {bpm_min:.0f}–{bpm_max:.0f} BPM")
    print(f"Bänder       : Kick {kick_lo:.0f}–{kick_hi:.0f} Hz (×{kick_weight})"
          f"  Snare {snare_lo:.0f}–{snare_hi:.0f} Hz (×{snare_weight})"
          f"  HiHat {hihat_lo:.0f}–{hihat_hi:.0f} Hz (×{hihat_weight})")
    print(f"OSS-Fenster  : {oss_window_s:.1f} s  |  Smoothing α={smoothing}  Hysterese {hysteresis} BPM")
    print(f"Tempo-Hold   : lock≥{lock_conf_min}  relock≥{relock_conf_min}  "
          f"max_jump {max_jump_bpm} BPM  hard_block {hard_block_jump} BPM  "
          f"relock {relock_windows}× Fenster  hold {hold_seconds} s")

    # ── Ableton Link ──────────────────────────────────────────────────────────
    link_bridge = None
    if link_cfg and link_cfg.get("enabled", False):
        from link_bridge import LinkBridge
        quantum           = link_cfg.get("quantum", 4)
        upd_interval_ms   = link_cfg.get("update_interval_ms", 200.0)
        link_hysteresis   = link_cfg.get("tempo_hysteresis", 0.5)
        link_bridge = LinkBridge(
            quantum=quantum,
            update_interval_ms=upd_interval_ms,
            tempo_hysteresis=link_hysteresis,
        )
        if link_bridge.start():
            print(f"Link         : aktiv  quantum={quantum}  update={upd_interval_ms:.0f} ms  hysterese={link_hysteresis} BPM")
        else:
            print("[link] Nicht verfügbar — fahre ohne Link fort.")
            link_bridge = None

    print("\nStrg+C zum Beenden\n")

    detector = BeatDetector(
        sample_rate=sample_rate,
        block_size=block_size,
        bpm_min=bpm_min,
        bpm_max=bpm_max,
        kick_lo=kick_lo,
        kick_hi=kick_hi,
        snare_lo=snare_lo,
        snare_hi=snare_hi,
        hihat_lo=hihat_lo,
        hihat_hi=hihat_hi,
        kick_weight=kick_weight,
        snare_weight=snare_weight,
        hihat_weight=hihat_weight,
        oss_window_s=oss_window_s,
        onset_threshold=onset_threshold,
        refractory=refractory,
        smoothing=smoothing,
        hysteresis=hysteresis,
        lock_confidence_min=lock_conf_min,
        relock_confidence_min=relock_conf_min,
        max_jump_bpm=max_jump_bpm,
        hard_block_jump_bpm=hard_block_jump,
        relock_windows=relock_windows,
        hold_seconds=hold_seconds,
    )
    display = BpmDisplay(bar_width=bar_width, db_floor=db_floor)
    counter = 0

    def on_block(block: np.ndarray) -> None:
        nonlocal counter
        detector.process(block)
        display.update_meter(block)
        counter += 1
        if counter % refresh_every == 0:
            if link_bridge:
                link_bridge.update(detector.locked_bpm, detector.beat_phase)
            display.render(detector, link=link_bridge)

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
        if link_bridge:
            link_bridge.stop()
        print("\n\nBeat-Detection beendet.")
