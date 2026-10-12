"""TempoCounter — eigenständige BPM-Schätzung aus einem Onset-Strength-
Signal (OSS), per Autokorrelation + Lock/Relock-Zustandsmaschine.

Macht NUR das: ein OSS-Wert pro Analyse-Block rein (`push_oss`), eine
BPM-Schätzung raus. Kennt nichts von Kick/Snare/HPSS/Ableton Link/DMX/
Audio-Capture - komplett entkoppelt vom Rest der Pipeline. Die OSS-
Berechnung selbst (gewichtete Summe aus Kick-/Snare-/Hihat-Spectral-Flux)
bleibt bewusst in beat_detector.py, weil sie dieselben Spektraldaten nutzt
wie die Drum-Erkennung - aber ab dem fertigen OSS-Wert ist diese Klasse
komplett unabhängig und für sich alleine testbar.

Extrahiert 2026-09-23 aus BeatDetector: eine zu großzügig dimensionierte
Schwelle (max_jump_bpm=10.0) sorgte live im Club dafür, dass ein falscher
Tempo-Lock (104.2 BPM) trotz durchgehend korrekt gemessenem neuem Tempo
(97.0 BPM, Differenz 7.2 BPM) NIE relockte, weil ein Sprung "kleiner
gleich max_jump_bpm" bedingungslos als Messrauschen abgetan und ignoriert
wird (siehe _apply_lock: kleiner Sprung -> "pass", locked_bpm bleibt
stehen). Der Bug blieb unbemerkt, weil die Tempo-Logik tief in einer
~2000-Zeilen-Datei zwischen Kick-/Snare-/HPSS-Code vergraben lag und keine
eigenen Tests hatte. Diese Klasse ist bewusst klein, hat eine klare
Zustandsmaschine und einen dedizierten Testfall genau für dieses Szenario
(siehe tests/test_tempo_counter.py), damit sowas nicht wieder unbemerkt
durchrutscht.

Zustandsmaschine (siehe TempoCounter._apply_lock)
--------------------------------------------------
searching  -> locked    : confidence >= lock_confidence_min, gültige BPM
locked     -> locked    : kleiner Sprung (<= max_jump_bpm) wird ignoriert
                           (Messrauschen der Autokorrelation zwischen
                           Analyse-Fenstern), locked_bpm bleibt stehen
locked     -> relocking : größerer Sprung (> max_jump_bpm, <= hard_block_
                           jump_bpm) mit confidence >= relock_confidence_min
locked     -> locked    : Sprung > hard_block_jump_bpm - zu unplausibel,
                           um überhaupt als Relock-Kandidat verfolgt zu
                           werden (kein hartes Umschalten auf jeden Ausreißer)
locked     -> searching : confidence < lock_confidence_min für > hold_seconds
relocking  -> locked    : relock_progress erreicht relock_windows
                           konsistente Fenster (bei großen Sprüngen
                           zusätzlich large_jump_hold_s Sekunden Mindest-
                           dauer, damit ein kurzer Ausreißer nicht sofort
                           übernommen wird)
relocking  -> locked    : Kandidat driftet weg oder confidence fällt ->
                           Relock-Versuch abgebrochen, alter Lock bleibt
"""
import threading
from collections import deque

import numpy as np


def _find_peaks(arr: np.ndarray, min_dist: int = 2) -> np.ndarray:
    """Lokale Maxima in arr, sortiert nach Wert (absteigend)."""
    n = len(arr)
    peaks = []
    for i in range(1, n - 1):
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
            peaks.append(i)
    if min_dist > 1 and peaks:
        filtered, prev = [peaks[0]], peaks[0]
        for p in peaks[1:]:
            if p - prev >= min_dist:
                filtered.append(p)
                prev = p
        peaks = filtered
    peaks.sort(key=lambda i: -arr[i])
    return np.array(peaks, dtype=int)


class TempoCounter:
    def __init__(
        self,
        hop_s: float,
        bpm_min: float = 70.0,
        bpm_max: float = 180.0,
        oss_window_s: float = 8.0,
        oss_short_window_s: float = 0.5,
        smoothing: float = 0.2,
        hysteresis: float = 1.0,
        lock_confidence_min: float = 0.3,
        relock_confidence_min: float = 0.5,
        max_jump_bpm: float = 4.0,
        hard_block_jump_bpm: float = 15.0,
        relock_windows: int = 3,
        hold_seconds: float = 8.0,
        large_jump_bpm: float = 20.0,
        large_jump_hold_s: float = 5.0,
        update_every_s: float = 1.0,
    ) -> None:
        if hop_s <= 0:
            raise ValueError("hop_s muss > 0 sein")
        # push_oss() läuft im Audio-Callback-Thread, die Properties werden
        # aber auch aus SoundlightBridges eigenem Sende-Thread gelesen
        # (soundlight_bridge.py _build_payload liest .locked_bpm/.bpm) -
        # ohne Lock ein echtes Daten-Race, wie zuvor in BeatDetector.
        self._lock = threading.Lock()
        self._hop = hop_s
        self._bpm_min = bpm_min
        self._bpm_max = bpm_max
        self._smoothing = smoothing
        self._hysteresis = hysteresis

        self._lock_conf_min = lock_confidence_min
        self._relock_conf_min = relock_confidence_min
        self._max_jump_bpm = max_jump_bpm
        self._hard_block_jump_bpm = hard_block_jump_bpm
        self._relock_windows = relock_windows
        self._hold_seconds = hold_seconds
        self._large_jump_bpm = large_jump_bpm
        self._large_jump_hold_s = large_jump_hold_s

        oss_len = max(64, int(oss_window_s / hop_s))
        self._oss: deque = deque(maxlen=oss_len)
        self._oss_short: deque = deque(maxlen=max(8, int(oss_short_window_s / hop_s)))

        self._bpm_raw: float = 0.0
        self._bpm_corrected: float = 0.0
        self._bpm_smooth: float = 0.0
        self._candidates: list = []
        self._confidence: float = 0.0

        self._tempo_state: str = "searching"  # searching | locked | relocking
        self._locked_bpm: float = 0.0
        self._relock_candidate: float = 0.0
        self._relock_progress: int = 0
        self._relock_candidate_since: float = 0.0
        self._low_conf_since: float = 0.0

        self._update_every_blocks = max(1, int(update_every_s / hop_s))
        self._block_count = 0
        self._now: float = 0.0  # letzter push_oss-Zeitstempel, für Zustandsmaschine

    # ── public API ────────────────────────────────────────────────────────

    def push_oss(self, value: float, now: float) -> None:
        """Ein OSS-Sample (kombinierter Flux-Wert eines Analyse-Blocks)
        einspeisen - einmal pro Audio-Block aufrufen. `now` ist ein
        monotoner/wall-clock Zeitstempel (time.time()), für die Lock-
        Zustandsmaschine (hold_seconds/large_jump_hold_s). Die eigentliche
        Autokorrelation läuft intern gedrosselt (update_every_s) - das
        Pushen selbst ist billig (ein deque.append)."""
        with self._lock:
            self._now = now
            self._block_count += 1
            self._oss.append(value)
            self._oss_short.append(value)
            if self._block_count % self._update_every_blocks == 0:
                self._update(now)

    @property
    def bpm(self) -> float:
        """Interner EMA der letzten korrigierten Schätzung - reagiert
        schneller als locked_bpm, aber ohne dessen Stabilitätsgarantie.
        Für einen Phase-Fallback gedacht, solange noch kein Lock steht."""
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
    def locked_bpm(self) -> float:
        """Die bestätigte, stabile Schätzung - 0.0 bis genug Konfidenz für
        den ersten Lock da war. Das ist der Wert, den externe Konsumenten
        (Grand Vision, Anzeige) für "das aktuelle Tempo" nutzen sollten."""
        with self._lock:
            return self._locked_bpm

    @property
    def confidence(self) -> float:
        with self._lock:
            return self._confidence

    @property
    def candidates(self) -> list:
        with self._lock:
            return list(self._candidates)

    @property
    def state(self) -> str:
        with self._lock:
            return self._tempo_state

    @property
    def relock_candidate(self) -> float:
        with self._lock:
            return self._relock_candidate

    @property
    def relock_progress(self) -> tuple:
        """(aktueller Fortschritt, benötigte Fenster)"""
        with self._lock:
            return (self._relock_progress, self._relock_windows)

    # ── intern ────────────────────────────────────────────────────────────

    def _acf(self, signal: np.ndarray) -> np.ndarray:
        """Normalisierte Autokorrelation via FFT."""
        n = len(signal)
        pad = int(2 ** np.ceil(np.log2(2 * n)))
        f = np.fft.rfft(signal - signal.mean(), n=pad)
        acf = np.fft.irfft(f * np.conj(f))[:n].real
        if acf[0] > 1e-12:
            acf /= acf[0]
        return acf

    def _octave_correct(self, bpm: float, acf: np.ndarray, lag_min: int) -> float:
        """Prüft ob 0.5x oder 2x des Kandidaten einen höheren ACF-Score hat
        und korrigiert entsprechend (Half-/Double-Time)."""
        hop = self._hop
        lag = 60.0 / bpm / hop

        def acf_at(b: float) -> float:
            i = int(round(b))
            if lag_min <= i < len(acf):
                return float(acf[i])
            return -1.0

        best_bpm, best_score = bpm, acf_at(lag)

        half_bpm = bpm / 2
        score_half = acf_at(lag * 2)
        if score_half > best_score * 1.15 and self._bpm_min <= half_bpm <= self._bpm_max:
            best_bpm, best_score = half_bpm, score_half

        double_bpm = bpm * 2
        score_double = acf_at(lag / 2)
        if score_double > best_score * 1.15 and self._bpm_min <= double_bpm <= self._bpm_max:
            best_bpm, best_score = double_bpm, score_double

        return best_bpm

    def _update(self, now: float) -> None:
        oss = np.array(self._oss, dtype=np.float32)
        if len(oss) < 32:
            return

        acf = self._acf(oss)
        hop = self._hop

        lag_min = max(1, int(60.0 / self._bpm_max / hop))
        lag_max = min(len(acf) - 1, int(60.0 / self._bpm_min / hop))
        if lag_min >= lag_max:
            return

        region = acf[lag_min : lag_max + 1]
        min_dist = max(2, int((60.0 / (self._bpm_max - 10) / hop) * 0.1))
        peak_indices = _find_peaks(region, min_dist=min_dist)
        if len(peak_indices) == 0:
            return

        candidates_raw = []
        for pi in peak_indices[:8]:
            lag = lag_min + int(pi)
            bpm = 60.0 / (lag * hop)
            score = float(region[pi])
            if self._bpm_min <= bpm <= self._bpm_max and score > 0:
                candidates_raw.append((bpm, score))
        if not candidates_raw:
            return

        best_bpm, best_score = candidates_raw[0]
        bpm_raw = best_bpm
        bpm_corrected = self._octave_correct(best_bpm, acf, lag_min)

        acf_median = float(np.median(region))
        acf_range = float(np.max(region)) - acf_median
        conf = min(1.0, max(0.0, (best_score - acf_median) / acf_range)) if acf_range > 1e-9 else 0.0

        if len(self._oss_short) >= 4:
            recent_energy = float(np.mean(list(self._oss_short)))
            window_energy = float(np.mean(oss)) + 1e-12
            if window_energy > 1e-9:
                energy_ratio = recent_energy / window_energy
                if energy_ratio < 0.15:
                    conf *= energy_ratio / 0.15

        self._bpm_raw = bpm_raw
        self._bpm_corrected = bpm_corrected
        self._candidates = [(b, s) for b, s in candidates_raw[:3]]
        self._confidence = conf

        if conf > 0.05:
            if self._bpm_smooth == 0.0:
                self._bpm_smooth = bpm_corrected
            else:
                d = abs(bpm_corrected - self._bpm_smooth)
                if d > self._hysteresis:
                    self._bpm_smooth = self._smoothing * bpm_corrected + (1.0 - self._smoothing) * self._bpm_smooth

        self._apply_lock(bpm_corrected, conf, now)

    def _apply_lock(self, bpm_corrected: float, conf: float, now: float) -> None:
        state = self._tempo_state

        if state == "searching":
            if conf >= self._lock_conf_min and bpm_corrected > 0:
                self._locked_bpm = bpm_corrected
                self._tempo_state = "locked"
                self._low_conf_since = 0.0

        elif state == "locked":
            if conf < self._lock_conf_min:
                if self._low_conf_since == 0.0:
                    self._low_conf_since = now
                if (now - self._low_conf_since) > self._hold_seconds:
                    self._tempo_state = "searching"
                # locked_bpm bleibt unverändert
            else:
                self._low_conf_since = 0.0
                jump = abs(bpm_corrected - self._locked_bpm)

                if jump <= self._max_jump_bpm:
                    # Kleine Abweichung (Messrauschen zwischen Analyse-
                    # Fenstern) - locked_bpm bleibt stehen.
                    pass
                elif jump > self._hard_block_jump_bpm:
                    # Zu wilder Sprung, um plausibel zu sein (z.B.
                    # Autokorrelation komplett daneben, Stille, Genre-
                    # Wechsel-Transiente) - nicht mal als Relock-Kandidat
                    # verfolgen. Vorher (im alten BeatDetector) wurde dieser
                    # Parameter gelesen, aber nirgends ausgewertet - hier
                    # jetzt tatsächlich wirksam.
                    pass
                elif conf >= self._relock_conf_min:
                    tol = bpm_corrected * 0.04
                    if self._relock_candidate == 0.0 or abs(bpm_corrected - self._relock_candidate) > tol:
                        self._relock_candidate = bpm_corrected
                        self._relock_progress = 1
                        self._relock_candidate_since = now
                    else:
                        self._relock_progress += 1

                    is_large = jump > self._large_jump_bpm
                    hold_ok = (now - self._relock_candidate_since) >= self._large_jump_hold_s if is_large else True

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
                self._relock_candidate = 0.0
                self._relock_progress = 0
                self._tempo_state = "locked"
            else:
                tol = bpm_corrected * 0.04
                if abs(bpm_corrected - self._relock_candidate) <= tol:
                    self._relock_progress += 1
                    # large_jump_hold_s muss auch HIER gelten, nicht nur
                    # beim Einstieg aus "locked" (siehe oben) - Bugfix
                    # 2026-09-23: vorher wurde hold_ok nur beim allerersten
                    # relocking-Fenster geprüft, sobald der Zustand einmal
                    # "relocking" war, reichte danach jedes Mal allein
                    # relock_progress >= relock_windows, komplett ohne
                    # Rücksicht auf die verstrichene Zeit - ein großer
                    # Sprung konnte so trotz large_jump_hold_s=5s schon
                    # nach relock_windows aufeinanderfolgenden ~1s-Fenstern
                    # durchrutschen (< 5s), wenn relock_windows klein genug
                    # gewählt war. Die Distanz zum ALTEN locked_bpm ändert
                    # sich während des Relocks nicht (locked_bpm bleibt
                    # stehen bis zum Commit), daher hier einfach neu
                    # berechnet statt zusätzlich zwischengespeichert.
                    is_large = abs(self._relock_candidate - self._locked_bpm) > self._large_jump_bpm
                    hold_ok = (now - self._relock_candidate_since) >= self._large_jump_hold_s if is_large else True
                    if self._relock_progress >= self._relock_windows and hold_ok:
                        self._locked_bpm = self._relock_candidate
                        self._relock_candidate = 0.0
                        self._relock_progress = 0
                        self._relock_candidate_since = 0.0
                        self._tempo_state = "locked"
                else:
                    self._relock_candidate = 0.0
                    self._relock_progress = 0
                    self._relock_candidate_since = 0.0
                    self._tempo_state = "locked"
