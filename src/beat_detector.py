"""Echtzeit-Beat-Detection und BPM-Schätzung.

Ansatz:
  1. Onset Strength Signal (OSS): gewichtete Summe aus drei Drum-Bändern
     (Kick, Snare, Hi-Hat), jeweils halbwellen-gleichgerichteter Spectral Flux.
  2. Tempo-Schätzung: eigenständig in tempo_counter.py (Autokorrelation +
     Lock/Relock-Zustandsmaschine) - BeatDetector füttert sie nur noch mit
     dem fertigen OSS-Wert pro Block.
  3. Beat-Phase: Flux-Peaks als Onset-Ereignisse für Phasen-Tracking.

2026-09-23: die Tempo-Schätzung (Autokorrelation, Octave-Korrektur,
Lock/Relock) lebte vorher hier inline und wurde nach tempo_counter.py
extrahiert - ein zu großzügig dimensionierter Schwellwert (max_jump_bpm)
sorgte live im Club dafür, dass ein falscher Tempo-Lock nie relockte,
obwohl das korrekte Tempo die ganze Zeit korrekt gemessen wurde. Der Bug
blieb unbemerkt, weil die Logik tief in dieser Datei ohne eigene Tests
vergraben war - siehe tempo_counter.py und tests/test_tempo_counter.py.
"""

import re
import shutil
import sys
import time
import threading
from collections import deque

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi
from scipy.ndimage import median_filter

from capture import AudioCapture, LevelMeter, resolve_device
from drum_features import extract_features, DrumClassifier
from tempo_counter import TempoCounter


# ── Hilfsfunktionen ───────────────────────────────────────────────────────────
# _find_peaks lebt jetzt in tempo_counter.py (einziger Aufrufer war die
# dorthin extrahierte Autokorrelations-Tempo-Schätzung).


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
        clap_lo: float = 700.0,
        clap_hi: float = 3000.0,
        kick_weight: float = 0.6,
        snare_weight: float = 0.25,
        hihat_weight: float = 0.15,
        oss_window_s: float = 8.0,
        onset_threshold: float = 1.4,
        kick_onset_threshold: float = None,
        snare_onset_threshold: float = None,
        hihat_onset_threshold: float = None,
        clap_onset_threshold: float = None,
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
        classifier_model_path: str = None,
        # ── Boolescher Snare-Trigger-Detektor (für DMX-Events) ──────────
        trigger_snare_a_lo: float = 200.0,
        trigger_snare_a_hi: float = 300.0,
        trigger_snare_b_lo: float = 2500.0,
        trigger_snare_b_hi: float = 4500.0,
        trigger_window_s: float = 0.4,
        trigger_snare_std_factor: float = 1.5,
        trigger_snare_holdoff_s: float = 0.15,
        trigger_snare_coincidence_s: float = 0.06,
        # Gewichtete Snare-Bewertung statt starrer UND-Koinzidenz - siehe
        # _update_snare_trigger. body/snap_min sind relativ zur jeweils
        # EIGENEN adaptiven Schwelle (1.0 = genau an der Schwelle), nicht
        # absolut - "min 0.35" heisst "mindestens 35% des Wegs zur eigenen
        # Schwelle", nicht ein absoluter Energiewert.
        trigger_snare_body_weight: float = 0.5,
        trigger_snare_snap_weight: float = 0.5,
        trigger_snare_body_min: float = 0.35,
        trigger_snare_snap_min: float = 0.35,
        trigger_snare_allow_weak_body: bool = True,
        trigger_snare_cooldown_s: float = None,  # None -> nutzt trigger_snare_holdoff_s (Alias/Rueckwaertskompatibilitaet)
        # HPSS-Bestaetigung fuer Snare, analog zum Kick-Gate - 0 = aus.
        # Bewusst kein hartes Muss, da Latino-Claps/Rimshots sonst zu leicht
        # unterdrueckt werden koennten (siehe Analyse-Bericht Punkt C).
        trigger_snare_hpss_std_factor: float = 0.0,
        # ── Eigenstaendiger Clap-Trigger ─────────────────────────────────
        # Absichtlich getrennt vom bestehenden clap_lo/clap_hi (700-3000Hz,
        # nur fuer den kontinuierlichen Score/Onset) - der Trigger-Bereich
        # ist breiter/hoeher angesetzt, da ein Handclap-Transient typisch
        # bis in den 5-7kHz-Bereich hinein Energie hat (das eigentliche
        # "Klatschen"-Rauschen, nicht nur der tiefere Koerperanteil).
        # Ueberschneidet sich bewusst teilweise mit Snare-Snap/Hihat - die
        # Trennung kommt hier primaer ueber Anstiegsschaerfe + HPSS +
        # Cooldown, nicht ueber reine Bandisolierung (die kann Instrumente
        # strukturell nicht auseinanderhalten, siehe Analyse-Bericht).
        clap_trigger_lo: float = 1000.0,
        clap_trigger_hi: float = 6000.0,
        clap_trigger_std_factor: float = 2.0,
        clap_trigger_rise_factor: float = 3.0,
        clap_trigger_cooldown_s: float = 0.15,
        clap_trigger_min_confidence: float = 0.3,
        clap_trigger_hpss_std_factor: float = 0.5,
        # Debug-Diagnose fuer Snare/Clap (Rohenergie, Schwelle, Ablehnungs-
        # grund, ...) - siehe trigger_debug. Nur ein paar dict-Schreibvor-
        # gaenge pro Block, kein spuerbarer Overhead, aber trotzdem per
        # Flag abschaltbar wie gefordert.
        debug_triggers: bool = False,
        # ── Kalman-Filter-Beat-Tracker für den Kick-Trigger ─────────────
        # Ersetzt die vorherige FFT-Block-Phasen-Ankerung komplett - siehe
        # Klassenkommentar bei _update_kalman_kick.
        kalman_kick_cutoff_hz: float = 130.0,
        kalman_kick_std_factor: float = 2.5,
        kalman_kick_sharpness_ratio: float = 4.0,
        # Absolute Lautstärke-Untergrenze (dBFS) für einen Kick-Kandidaten -
        # unabhängig von allen relativen Schwellen oben. Default -100.0 lässt
        # praktisch jede Lautstärke durch (alte Verhalten unverändert für
        # jeden, der das nicht konfiguriert) - siehe Konstruktor-Kommentar
        # bei _update_kalman_kick für die Begründung.
        kalman_kick_min_level_dbfs: float = -100.0,
        kalman_kick_holdoff_s: float = 0.2,
        # 2026-10-14 (Live-Club-Test, Nutzer-Report "läuft keine Musik und
        # Kick triggert trotzdem"): _kalman_advance() (reine Vorhersage,
        # "Schwungrad durch Pausen/Drops") hatte KEINE Obergrenze, wie lange
        # es ohne einen einzigen neuen echten Kandidaten weiterfeuern darf -
        # einmal eingerastet (_kalman_initialized=True, schon nach dem
        # zweiten echten Treffer), lief die Vorhersage buchstäblich für
        # immer auf dem zuletzt gelockten Tempo weiter, auch nach dem Ende
        # des Songs/in Stille. Diese Obergrenze (Sekunden seit dem letzten
        # Kandidaten, der die Level-/Sharpness-/HPSS-/Holdoff-Gates
        # passiert hat - unabhängig davon, ob der Kalman-Filter ihn später
        # als Ausreißer verworfen hat, siehe _kalman_on_impulse) setzt
        # _kalman_initialized zurück, sobald sie überschritten ist -
        # danach bootstrapped der Tracker beim nächsten echten Doppel-
        # Treffer neu, exakt wie beim allerersten Start. Default 4.0s (~6-8
        # Beats im üblichen Clubtempo) lässt kurze Drops/Breaks im Song
        # weiterhin unbeeinflusst durchlaufen, beendet aber zuverlässig das
        # Nachfeuern zwischen zwei Songs oder bei Gesprächspausen.
        kalman_kick_silence_timeout_s: float = 4.0,
        kalman_kick_hpss_std_factor: float = 1.0,  # Bestätigungs-Gate: wie streng die HPSS-Perkussions-Prüfung ist (höher = strenger, mehr echte Kicks können durchfallen)
        kalman_q_time: float = 0.0004,     # Prozessrauschen: Zeit/Phase (s²) pro Beat
        kalman_q_period: float = 0.00002,  # Prozessrauschen: Tempo/Periode (s²) pro Beat
        kalman_r: float = 0.0009,          # Messrauschen: ein einzelner Kick-Zeitstempel (s²)
        kalman_outlier_gate: float = 3.0,  # Ablehnen, wenn |Fehler| > gate·σ (Bassline statt Kick)
        # ── Runde 2: optionaler kick-eigener HPSS-Kernel + Bassline-Analyse ──
        # Rein additiv - None/False lässt das gesamte Verhalten von Runde 1
        # unverändert (siehe _spectral_flux_bands / _update_kalman_kick).
        kick_hpss_freq_kernel_bins: int = None,  # None = weiter der geteilte 17-Bin-Kernel (self._hpss_freq_kernel); Zahl = eigener, kick-spezifischer Kernel nur fürs Kick-Band
        kick_bassline_suppression_enabled: bool = False,  # Opt-in; nie eine harte Ablehnung, nur ein Confidence-Abzug (siehe _update_kalman_kick)
        kick_bassline_duration_threshold_s: float = 0.35,  # ab wie langer ununterbrochener Schwellenüberschreitung ein Treffer "bassline-verdächtig" wird
        kick_bassline_decay_threshold: float = 0.6,  # Abklingverhältnis (env nach ~150ms / Peak) oberhalb dessen ein Treffer als "klingt nicht ab wie ein Kick" gilt
        kick_pitch_drift_threshold: float = 15.0,  # Hz Centroid-Drift zwischen zwei Blöcken, unterhalb dessen der Ton als tonhöhen-stabil (bassline-typisch) gilt
        kick_bassline_penalty: float = 0.5,  # wie stark ein voll "bassline"-bewerteter Treffer die Confidence maximal senken darf (0=kein Effekt, 1=kann auf 0 senken)
        kick_bassline_min_confidence: float = 0.3,  # nur wirksam wenn kick_bassline_suppression_enabled=True
        # ── Runde 5: Sharpness/HPSS-Timing-Diagnose (rein additiv) ──────
        # Default AUS. Protokolliert pro Kick-KANDIDAT (nicht pro Block)
        # eine detaillierte Zeilen-Momentaufnahme inkl. HPSS-Energie bei
        # +0/20/40/60/80ms - ändert NICHTS an der Trigger-Entscheidung,
        # siehe kick_gate_debug_rows.
        kick_gate_debug: bool = False,
    ) -> None:
        self._sr = sample_rate
        self._bs = block_size
        self._hop = block_size / sample_rate
        self._bpm_min = bpm_min
        self._bpm_max = bpm_max
        self._onset_threshold = onset_threshold
        self._band_onset_threshold = {
            "kick": kick_onset_threshold if kick_onset_threshold is not None else onset_threshold,
            "snare": snare_onset_threshold if snare_onset_threshold is not None else onset_threshold,
            "hihat": hihat_onset_threshold if hihat_onset_threshold is not None else onset_threshold,
            "clap": clap_onset_threshold if clap_onset_threshold is not None else onset_threshold,
        }
        self._refractory = refractory
        self._band_weights = (kick_weight, snare_weight, hihat_weight)

        # ── Tempo/BPM-Schätzung (2026-09-23 extrahiert, siehe tempo_counter.py) ──
        # Eigenständige, isoliert getestete Klasse für Autokorrelation +
        # Lock/Relock-Zustandsmaschine - BeatDetector füttert sie nur noch
        # mit dem fertigen OSS-Wert (push_oss) und liest ihre Properties
        # weiter unten durch (siehe locked_bpm/bpm/tempo_state/... Property-
        # Wrapper), damit sich an der öffentlichen API von BeatDetector
        # nichts ändert.
        self._tempo = TempoCounter(
            hop_s=self._hop,
            bpm_min=bpm_min,
            bpm_max=bpm_max,
            oss_window_s=oss_window_s,
            smoothing=smoothing,
            hysteresis=hysteresis,
            lock_confidence_min=lock_confidence_min,
            relock_confidence_min=relock_confidence_min,
            max_jump_bpm=max_jump_bpm,
            hard_block_jump_bpm=hard_block_jump_bpm,
            relock_windows=relock_windows,
            hold_seconds=hold_seconds,
            large_jump_bpm=large_jump_bpm,
            large_jump_hold_s=large_jump_hold_s,
        )
        # Separater, kleiner Kurzfenster-Puffer NUR für die Onset-/Beat-
        # Phase-Erkennung weiter unten in process() (lokaler Lautstärke-
        # Schnitt für den Onset-Schwellenvergleich) - bewusst nicht
        # dasselbe Objekt wie TempoCounters eigener interner Kurzfenster-
        # Puffer (der für dessen Energie-Gate gedacht ist), auch wenn
        # beide denselben `combined`-Wert bekommen: TempoCounter soll
        # nichts über Beat-Phase/Onsets wissen müssen, und BeatDetector
        # soll nicht in TempoCounters Interna greifen müssen.
        self._oss_short: deque = deque(maxlen=max(8, int(0.5 / self._hop)))
        self._prev_mag = None  # type: np.ndarray

        # Analyse-FFT über mehrere Blöcke akkumuliert:
        # fft_size >> block_size → bessere Frequenzauflösung im Kick-Band.
        # sample_rate-unabhängig fest codiert (4096) - die tatsächliche
        # Hz/Bin-Auflösung skaliert mit der jeweiligen sample_rate, z.B.
        # 10.77 Hz/Bin bei 44100 Hz oder 11.72 Hz/Bin bei 48000 Hz (aktuelle
        # Live-Konfiguration, siehe club.yaml - REV7/BlackHole laufen auf
        # 48000 Hz, Stand 2026-09-22). Bei nur 512 Samples wären es grobe
        # 86-94 Hz/Bin (Kick-Band nur ~1 Bin) - deshalb die separate,
        # größere Analyse-FFT statt direkt auf block_size zu rechnen.
        self._fft_size = 4096
        self._sample_buf: deque = deque(maxlen=self._fft_size)
        freqs = np.fft.rfftfreq(self._fft_size, d=1.0 / sample_rate)
        self._mask_kick  = ((freqs >= kick_lo)  & (freqs <= kick_hi)).astype(np.float32)
        self._mask_snare = ((freqs >= snare_lo) & (freqs <= snare_hi)).astype(np.float32)
        self._mask_hihat = ((freqs >= hihat_lo) & (freqs <= hihat_hi)).astype(np.float32)
        # Clap: breitbandiger "Knack" - füllt die Lücke zwischen Snare und
        # Hi-Hat. Fließt bewusst NICHT ins Tempo-Tracking-OSS ein (kein
        # clap_weight), nur eigener Score/Onset fürs Sound-to-Light.
        self._mask_clap  = ((freqs >= clap_lo)  & (freqs <= clap_hi)).astype(np.float32)
        self._hann = np.hanning(self._fft_size).astype(np.float32)
        # Für Kick-Band-Spectral-Centroid (Runde 2, Bassline-Analyse) -
        # einmal gecacht statt pro Block neu zu berechnen.
        self._kick_band_sel = self._mask_kick > 0.5
        self._kick_band_freqs = freqs[self._kick_band_sel]

        # ── HPSS (Harmonic-Percussive Source Separation) ────────────────
        # A sustained bassline note is narrowband-in-frequency but stable-
        # over-time ("harmonic"); a kick/snare hit is broadband-in-
        # frequency but brief-over-time ("percussive") - median-filtering
        # the same magnitude spectrum already computed for the flux
        # pipeline along each axis separates the two (Fitzgerald 2010 /
        # Driedger, Müller & Disch 2014). Causal variant: the time-axis
        # median only ever looks at past frames (a rolling buffer), never
        # future ones, so it adds no extra latency beyond what the flux
        # pipeline already has (~93ms, the accumulated FFT window).
        # Physically eliminates most bassline/pad/vocal bleed from the
        # kick/snare bands before they're even measured, instead of
        # trying to threshold it away afterwards.
        hpss_time_kernel = max(3, int(round(0.3 / self._hop)))
        if hpss_time_kernel % 2 == 0:
            hpss_time_kernel += 1  # median filter kernels must be odd-length
        self._hpss_mag_history: deque = deque(maxlen=hpss_time_kernel)
        self._hpss_freq_kernel = 17  # bins - same ballpark as librosa's HPSS default
        self._hpss_kick_energy: float = 0.0
        # Adaptive baseline for the Kalman kick-impulse HPSS confirmation
        # gate - see _update_kalman_kick.
        self._hpss_kick_baseline: deque = deque(maxlen=max(4, int(1.0 / self._hop)))

        # ── Boolescher Snare-Trigger-Detektor (für DMX-Events) ──────────
        # Snare = Koinzidenz zweier unabhängig schwellenwert-geprüfter Bänder
        # (Body 200-300Hz UND Snap 2.5-4.5kHz müssen beide innerhalb eines
        # kurzen Zeitfensters anschlagen - nicht zwingend im exakt selben
        # ~12ms-Analyseblock, echte Snares haben oft einen winzigen Versatz
        # zwischen tiefem Body und hohem Snap) statt eines gewichteten
        # Summen-Mixes - ein einzelnes Band, das durch Bleed aus
        # Bassline/Becken/Gesang anschlägt, kann so allein nicht mehr
        # triggern. Adaptive Schwelle ist Mean+Faktor*StdDev über ein
        # gleitendes ~400ms-Fenster (robuster gegen wechselnde Lautstärke
        # als der reine Mittelwert-Multiplikator der obigen _band_onset) -
        # plus ein absoluter Rausch-Boden, da Mean+StdDev bei echter Stille
        # gegen 0 geht und sonst jedes kleinste Rauschen träfe. Harter
        # Hold-off-Timer statt sanft abklingendem Puls → sauberes True/False
        # statt einer kontinuierlichen Stärke. (Kick hat einen eigenen,
        # Kalman-Filter-basierten Tracker - siehe weiter unten.)
        trig_win = max(4, int(trigger_window_s / self._hop))
        self._trig_snare_a_flux: deque = deque(maxlen=trig_win)
        self._trig_snare_b_flux: deque = deque(maxlen=trig_win)
        self._mask_trig_snare_a = ((freqs >= trigger_snare_a_lo) & (freqs <= trigger_snare_a_hi)).astype(np.float32)
        self._mask_trig_snare_b = ((freqs >= trigger_snare_b_lo) & (freqs <= trigger_snare_b_hi)).astype(np.float32)
        self._trig_snare_std_factor = trigger_snare_std_factor
        self._trig_noise_floor = 1e-4  # absoluter Boden gegen Mean+Std≈0 bei Stille
        self._trig_snare_holdoff_s = trigger_snare_holdoff_s if trigger_snare_cooldown_s is None else trigger_snare_cooldown_s
        self._trig_snare_coincidence_s = trigger_snare_coincidence_s
        self._trig_last_snare_t = 0.0
        self._trig_last_snare_a_hit_t = -999.0
        self._trig_last_snare_b_hit_t = -999.0
        self._kick_trigger = False
        self._snare_trigger = False

        # ── Gewichtete Snare-Bewertung (statt starrer UND-Koinzidenz) ────
        self._trig_snare_body_weight = trigger_snare_body_weight
        self._trig_snare_snap_weight = trigger_snare_snap_weight
        self._trig_snare_body_min = trigger_snare_body_min
        self._trig_snare_snap_min = trigger_snare_snap_min
        self._trig_snare_allow_weak_body = trigger_snare_allow_weak_body
        self._trig_snare_hpss_std_factor = trigger_snare_hpss_std_factor
        # Eigene Baseline fuer die HPSS-Bestaetigung (getrennt von der des
        # Kicks, da anderes Frequenzband -> andere typische Energie).
        self._trig_snare_hpss_baseline: deque = deque(maxlen=max(4, int(1.0 / self._hop)))
        # Nutzt die Snap-Band-Maske (self._mask_trig_snare_b) fuer die
        # HPSS-Bestaetigung - das Body-Band (200-300Hz) liegt zu nah an
        # Kick/Bassline, um dort verlaesslich "perkussiv" von "harmonisch"
        # zu unterscheiden; das breitbandig-rauschige Snap-Band ist dafuer
        # deutlich aussagekraeftiger.
        self._trig_snare_hpss_energy: float = 0.0

        # ── Eigenstaendiger Clap-Trigger ──────────────────────────────────
        self._mask_clap_trigger = ((freqs >= clap_trigger_lo) & (freqs <= clap_trigger_hi)).astype(np.float32)
        clap_trig_win = max(4, int(trigger_window_s / self._hop))
        self._clap_trig_flux_buf: deque = deque(maxlen=clap_trig_win)
        clap_attack_win = max(2, round(0.035 / self._hop))  # gleiche ~35ms-Anschlagsfenster-Logik wie beim Kick
        self._clap_trig_attack_buf: deque = deque(maxlen=clap_attack_win)
        self._clap_trig_hpss_buf: deque = deque(maxlen=max(4, int(1.0 / self._hop)))
        self._clap_trigger_std_factor = clap_trigger_std_factor
        self._clap_trigger_rise_factor = clap_trigger_rise_factor
        self._clap_trigger_cooldown_s = clap_trigger_cooldown_s
        self._clap_trigger_min_confidence = clap_trigger_min_confidence
        self._clap_trigger_hpss_std_factor = clap_trigger_hpss_std_factor
        self._clap_trig_last_t = 0.0
        self._clap_trig_freeze_s = max(clap_trigger_cooldown_s, 0.2)  # Nachklingzeit, Baseline waehrenddessen einfrieren
        self._clap_trigger = False
        self._clap_hpss_energy: float = 0.0

        # ── Debug-Diagnose (F) ────────────────────────────────────────────
        self._debug_triggers = debug_triggers
        self._snare_debug: dict = {}
        self._clap_debug: dict = {}
        self._kick_debug: dict = {}

        # ── Kalman-Filter-Beat-Tracker für den Kick-Trigger ─────────────
        # Front-End: kausaler IIR-Tiefpass (Butterworth, 4. Ordnung) statt
        # des FFT-Block-Ansatzes oben - Latenz im Mikrosekundenbereich statt
        # der ~93ms des akkumulierten FFT-Fensters (siehe _spectral_flux_bands).
        kick_cutoff = min(kalman_kick_cutoff_hz, sample_rate / 2 - 1)
        self._kalman_kick_sos = butter(4, kick_cutoff, btype="low", fs=sample_rate, output="sos")
        self._kalman_kick_zi = sosfilt_zi(self._kalman_kick_sos) * 0.0
        kalman_env_win = max(4, int(0.4 / self._hop))
        self._kalman_kick_env_recent: deque = deque(maxlen=kalman_env_win)
        # Anschlagsflanken-Prüfung: ein echter Kick hat eine viel schärfere
        # Energie-Anstiegsflanke als eine Bassline-Note im selben
        # Frequenzband, die meist sanfter einschwingt. Vergleicht die
        # aktuelle Hüllkurve mit der von ~35ms zuvor - nur ein SPRUNGHAFTER
        # Anstieg zählt als möglicher Kick, ein langsames Anschwellen nicht.
        attack_win = max(2, round(0.035 / self._hop))
        self._kalman_kick_attack_win: deque = deque(maxlen=attack_win)
        self._kalman_kick_sharpness_ratio = kalman_kick_sharpness_ratio
        self._kalman_kick_std_factor = kalman_kick_std_factor
        self._kalman_kick_noise_floor = 1e-5
        self._kalman_kick_min_level_dbfs = kalman_kick_min_level_dbfs
        self._kalman_kick_holdoff_s = kalman_kick_holdoff_s
        self._kalman_kick_silence_timeout_s = kalman_kick_silence_timeout_s
        self._kalman_kick_hpss_std_factor = kalman_kick_hpss_std_factor
        self._kalman_kick_freeze_s = max(kalman_kick_holdoff_s, 0.25)  # Filter-Ringing-Nachklingzeit
        self._kalman_kick_last_impulse_t = 0.0

        # ── Runde 2, Phase 2: optionaler kick-eigener HPSS-Kernel ───────
        # None (Default) = exakt das bisherige Verhalten, der geteilte
        # self._hpss_freq_kernel (17 Bins) wird weiter für Kick/Snare/Clap
        # gemeinsam genutzt. Nur wenn explizit gesetzt, wird zusätzlich
        # eine eigene, kleinere Perkussions-Maske NUR fürs Kick-Band
        # berechnet (_spectral_flux_bands) - ohne den globalen Kernel
        # anzufassen, der für Snare/Clap weiterläuft wie bisher.
        self._kick_hpss_freq_kernel_bins = kick_hpss_freq_kernel_bins
        if self._kick_hpss_freq_kernel_bins is not None and self._kick_hpss_freq_kernel_bins % 2 == 0:
            self._kick_hpss_freq_kernel_bins += 1  # Median-Kernel muss ungerade Länge haben

        # ── Runde 2, Phase 3: 808/Bassline-Analyse (nur Debug + optionaler
        # weicher Confidence-Abzug, siehe Konstruktor-Kommentar) ─────────
        self._kick_harmonic_energy: float = 0.0
        self._kick_prev_centroid: float = 0.0
        self._kick_centroid_drift: float = 0.0
        self._kick_above_threshold_blocks: int = 0
        kick_decay_win = max(4, int(round(0.15 / self._hop)))
        self._kick_decay_samples: deque = deque(maxlen=kick_decay_win)
        self._kick_decay_active = False
        self._kick_decay_peak = 0.0
        self._kick_decay_ratio = 1.0  # 1.0 = noch nicht gemessen / klingt nicht ab (bassline-verdächtig)
        self._kick_bassline_suppression_enabled = kick_bassline_suppression_enabled
        self._kick_bassline_duration_threshold_s = kick_bassline_duration_threshold_s
        self._kick_bassline_decay_threshold = kick_bassline_decay_threshold
        self._kick_pitch_drift_threshold = kick_pitch_drift_threshold
        self._kick_bassline_penalty = kick_bassline_penalty
        self._kick_bassline_min_confidence = kick_bassline_min_confidence
        self._kick_bassline_score: float = 0.0
        self._kick_confidence: float = 1.0

        # ── Runde 5: Sharpness/HPSS-Timing-Diagnose ─────────────────────
        # Ring-Puffer (t, hpss_energy, hpss_mean, hpss_std) pro Block -
        # NUR gefüllt wenn kick_gate_debug=True, sonst kein Overhead
        # (siehe _update_kalman_kick). ~340ms Reichweite deckt die
        # geforderten 0/20/40/60/80ms-Offsets nach dem Kandidaten ab.
        self._kick_gate_debug_enabled = kick_gate_debug
        self._hpss_gate_ring: deque = deque(maxlen=max(32, int(round(0.35 / self._hop))))
        self._kick_gate_pending: list = []
        self._kick_gate_finalized: deque = deque(maxlen=200)

        # Back-End: 2-Zustands-Kalman-Filter [nächster Beat-Zeitpunkt,
        # Periode] - Standard-State-Space-Modell fürs Beat-Tracking
        # (konstante "Geschwindigkeit" = Tempo). Miss-Update nur bei einem
        # echten, akzeptierten Kick-Impuls; sonst reine Prädiktion
        # ("Schwungrad" durch Pausen/Drops). Optimale, unsicherheits-
        # adaptive Verstärkung (Kalman-Gain) statt fest verdrahteter
        # Korrekturfaktoren wie beim vorherigen Ansatz.
        self._kalman_initialized = False
        self._kalman_t_next = 0.0     # geschätzter Zeitpunkt des nächsten Beats
        self._kalman_period = 0.0     # geschätzte Sekunden pro Beat
        # Drei-Punkt-Bootstrap-Bestätigung (siehe _kalman_on_impulse) -
        # None = noch keine Kandidat-Periode aus den ersten beiden
        # Impulsen abgeleitet; sonst wartet sie auf Bestätigung durch
        # einen dritten Impuls nahe self._kalman_bootstrap_confirm_t.
        self._kalman_bootstrap_candidate_period = None
        self._kalman_bootstrap_confirm_t = 0.0
        self._kalman_bootstrap_confirm_tol = 0.15  # ±15% der Kandidat-Periode
        self._kalman_P = np.array([[0.05 ** 2, 0.0], [0.0, (2.0 / 60.0) ** 2]])
        self._kalman_q_time = kalman_q_time
        self._kalman_q_period = kalman_q_period
        self._kalman_r = kalman_r
        self._kalman_gate = kalman_outlier_gate
        # Separater, geglätteter Flash-Zeitplan - feuert auf der
        # VORHERGESAGTEN Taktung, nicht direkt auf der (verrauschten)
        # Rohmessung ("vorausschauendes Blitzen" statt "Blitzen bei jedem
        # Treffer") - wird bei jeder akzeptierten Korrektur mit dem
        # aktuellen Schätzwert neu synchronisiert.
        self._kalman_flash_t = 0.0
        self._kalman_kick_trigger = False
        self._kalman_kick_is_impulse_now = False

        # Onset-Tracking für Beat-Phase
        self._last_onset_t: float = 0.0
        self._last_beat_t: float = 0.0

        # Per-Band-Onset (Transienten statt andauernder Energie) - fürs
        # Sound-to-Light gedacht: ein Sub-Bass/Bassline-Ton hält seine
        # Energie, ein Kick-Hit spitzt kurz auf und fällt sofort wieder ab.
        # Gleicher Schwellenwert-Vergleich wie beim kombinierten Onset oben,
        # nur pro Band statt global.
        onset_win = max(8, int(0.5 / self._hop))
        self._kick_recent: deque = deque(maxlen=onset_win)
        self._snare_recent: deque = deque(maxlen=onset_win)
        self._hihat_recent: deque = deque(maxlen=onset_win)
        self._clap_recent: deque = deque(maxlen=onset_win)
        self._kick_onset: float = 0.0
        self._snare_onset: float = 0.0
        self._hihat_onset: float = 0.0
        self._clap_onset: float = 0.0

        # Geschützter Zustand
        self._lock = threading.Lock()
        self._beat_phase: float = 0.0
        # Band-Scores (normierte mittlere Flux-Energie pro Band, für Anzeige)
        self._kick_score: float = 0.0
        self._snare_score: float = 0.0
        self._hihat_score: float = 0.0
        self._clap_score: float = 0.0
        # Rohe Broadband-Lautstärke (RMS des Mono-Signals, kein Frequenzband)
        # - für einen "echten" VU-Meter-Effekt, der sich wie Gesamtlautstärke
        # anfühlt statt wie ein einzelnes Drum-Band.
        self._overall_score: float = 0.0

        # ── Klassischer ML-Klassifikator (KNN) fuer Kick/Snare/Hihat ────
        # Alternative zur Frequenzband-Erkennung oben: statt vier
        # unabhaengige Bandschwellen zu vergleichen, klassifiziert er jeden
        # erkannten Treffer als GENAU eine Klasse - macht Bleed zwischen
        # Baendern (z.B. Bassline im Snare-Band) unwahrscheinlicher, kostet
        # aber ~100ms Latenz (muss das Analysefenster erst "hoeren", bevor
        # er entscheiden kann). Optional - nur aktiv wenn ein trainiertes
        # Modell (tools/train_drum_classifier.py) uebergeben wird.
        self._classifier = DrumClassifier(classifier_model_path) if classifier_model_path else None
        self._clf_window_samples = int(sample_rate * 0.1)
        self._clf_recent_flux: deque = deque(maxlen=90)
        self._clf_prev_spectrum = None
        self._clf_capturing = False
        self._clf_buffer: list = []
        self._clf_blocks_since_onset = 999
        self._classifier_hits = {"kick": 0.0, "snare": 0.0, "hihat": 0.0}  # one-shot pulses, reset every tick
        self._classifier_last_label = ""
        self._classifier_hit_counts = {"kick": 0, "snare": 0, "hihat": 0}

    # ── public API ────────────────────────────────────────────────────────────

    def process(self, block: np.ndarray) -> None:
        """block: shape (frames, 2), float32."""
        now = time.time()

        mono = (block[:, 0] + block[:, 1]) * 0.5
        if self._classifier is not None:
            self._update_classifier(mono)
        # Runde 6: HPSS-Analyse zuerst, damit _update_kalman_kick den
        # AKTUELLEN Block-HPSS-Wert liest (vorher: einen Block veraltet, da
        # _spectral_flux_bands erst danach lief - siehe Diagnose-Bericht).
        # _update_kalman_kick läuft weiterhin VOR der Flux-/Trigger-
        # Auswertung, damit Snare/Clap ihren kick_bleed-Ausschluss
        # weiterhin anhand des aktuellen (nicht vorherigen) Kickstatus
        # prüfen - reine Aufteilung von _spectral_flux_bands an ihrer
        # bestehenden internen Nahtstelle, keine Logikänderung.
        mag = self._update_hpss_kick_analysis(mono)
        self._update_kalman_kick(mono)
        kick_f, snare_f, hihat_f, clap_f, combined = self._flux_and_triggers(mag)

        # Band-Scores für Anzeige: gleitender Mittelwert (EMA α=0.1)
        a = 0.1
        with self._lock:
            self._kick_score  = a * kick_f  + (1 - a) * self._kick_score
            self._snare_score = a * snare_f + (1 - a) * self._snare_score
            self._hihat_score = a * hihat_f + (1 - a) * self._hihat_score
            self._clap_score  = a * clap_f  + (1 - a) * self._clap_score
            overall_f = float(np.sqrt(np.mean(mono ** 2)))
            self._overall_score = a * overall_f + (1 - a) * self._overall_score

            self._kick_onset  = self._band_onset(self._kick_recent,  kick_f,  self._kick_onset,  self._band_onset_threshold["kick"])
            snare_pulse = self._band_onset(self._snare_recent, snare_f, self._snare_onset, self._band_onset_threshold["snare"])
            # Ein Kick-Schlag hat einen breitbandigen Attack-Klick, der ins
            # Snare-Band mit reinstrahlt - wenn Kick gerade ausgelöst hat,
            # zählt Snare im selben Block nicht zusätzlich als eigener Treffer.
            if self._kick_onset > 0.3:
                snare_pulse = 0.0
            self._snare_onset = snare_pulse
            self._hihat_onset = self._band_onset(self._hihat_recent, hihat_f, self._hihat_onset, self._band_onset_threshold["hihat"])
            self._clap_onset  = self._band_onset(self._clap_recent,  clap_f,  self._clap_onset,  self._band_onset_threshold["clap"])

        self._tempo.push_oss(combined, now)
        self._oss_short.append(combined)

        # Onset-Erkennung für Beat-Phase (kombinierter Flux über lokalem Mittel).
        # Mindestabstand zwischen zwei Phasen-Korrekturen richtet sich nach
        # der tatsächlichen Beat-Periode (sobald bekannt), nicht nach dem
        # festen refractory von 0.2s - der erlaubt sonst mehr als eine
        # Korrektur pro Beat (z.B. durch ein Hihat kurz nach dem Kick),
        # was ständig kleine Sprünge in die exportierte Phase reinbringt,
        # statt nur einmal pro Beat sauber nachzujustieren.
        if len(self._oss_short) >= 4:
            local_avg = float(np.mean(self._oss_short))
            ref_bpm = self._tempo.locked_bpm if self._tempo.locked_bpm > 0 else self._tempo.bpm
            min_gap = max(self._refractory, 0.7 * 60.0 / ref_bpm) if ref_bpm > 0 else self._refractory
            if (
                combined > self._onset_threshold * local_avg
                and combined > 1e-9
                and (now - self._last_onset_t) >= min_gap
            ):
                self._last_onset_t = now
                self._last_beat_t = now

        # Beat-Phase: locked_bpm wenn verfügbar, sonst interner EMA.
        # Versuch, dies an den kick-spezifischen Trigger statt am
        # generischen kombinierten Onset zu hängen, hat es schlimmer
        # gemacht - der Kick-Trigger selbst ist noch nicht zuverlässig
        # genug, um als Phasen-Referenz zu taugen (vererbt seine eigene
        # Unzuverlässigkeit statt sie zu korrigieren). Zurück auf den
        # generischen Onset, bis die Kick-Erkennung selbst sauberer ist.
        with self._lock:
            bpm_for_phase = (
                self._tempo.locked_bpm if self._tempo.locked_bpm > 0 else self._tempo.bpm
            )
            if bpm_for_phase > 0 and self._last_beat_t > 0:
                period = 60.0 / bpm_for_phase
                elapsed = (now - self._last_beat_t) % period
                self._beat_phase = elapsed / period

    @property
    def bpm(self) -> float:
        return self._tempo.bpm

    @property
    def bpm_raw(self) -> float:
        return self._tempo.bpm_raw

    @property
    def bpm_corrected(self) -> float:
        return self._tempo.bpm_corrected

    @property
    def candidates(self) -> list:
        return self._tempo.candidates

    @property
    def confidence(self) -> float:
        return self._tempo.confidence

    @property
    def beat_phase(self) -> float:
        with self._lock:
            return self._beat_phase

    @property
    def band_scores(self) -> tuple[float, float, float, float]:
        """(kick_score, snare_score, hihat_score, clap_score) — EMA-geglättete
        Flux-Energie."""
        with self._lock:
            return (self._kick_score, self._snare_score, self._hihat_score, self._clap_score)

    @property
    def overall_score(self) -> float:
        """EMA-geglättete Broadband-RMS-Lautstärke (kein Frequenzband) -
        für einen VU-Meter-Effekt, der sich wie Gesamtlautstärke anfühlt."""
        with self._lock:
            return self._overall_score

    @property
    def onset_scores(self) -> tuple[float, float, float, float]:
        """(kick_onset, snare_onset, hihat_onset, clap_onset) — 0..1
        Transienten-Stärke pro Band, für Sound-to-Light gedacht: reagiert auf
        Treffer, nicht auf eine andauernde Basslinie/Gesang im selben
        Frequenzband."""
        with self._lock:
            return (self._kick_onset, self._snare_onset, self._hihat_onset, self._clap_onset)

    def _band_onset(self, recent: "deque[float]", flux: float, prev_onset: float, threshold: float) -> float:
        """0..1 Onset-Stärke: wie stark `flux` über dem lokalen Mittel dieses
        Bands liegt, relativ zu `threshold` - derselbe Vergleich wie beim
        kombinierten Onset (siehe oben), nur pro Band mit eigener Schwelle
        statt global. Snare/Gesang teilen sich z.B. dasselbe Frequenzband,
        daher oft eine höhere Schwelle für snare als für kick/hihat nötig.
        Klingt über ein paar Blöcke ab statt sofort auf 0 zu fallen, damit ein
        einzelner Treffer noch sichtbar ist, ohne wie die andauernde
        Band-Energie zu wirken."""
        recent.append(flux)
        if len(recent) < 4:
            return 0.0
        local_avg = float(np.mean(recent))
        if local_avg <= 1e-9:
            return 0.0
        strength = flux / local_avg - threshold
        pulse = max(0.0, min(1.0, strength))
        return max(pulse, prev_onset * 0.6)

    def _update_classifier(self, mono: np.ndarray) -> None:
        """Broadband onset gate + KNN classification, independent of the
        per-band threshold logic above. On a detected onset, waits for a
        ~100ms window of audio (the same feature window used for training,
        see tools/collect_training_data.py) then classifies it as exactly
        one of kick/snare/hihat - unlike the per-band detectors, which can
        fire on several bands "simultaneously" if a sound's energy spills
        across band boundaries."""
        for k in self._classifier_hits:
            self._classifier_hits[k] = 0.0

        spectrum = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
        self._clf_blocks_since_onset += 1

        if self._clf_capturing:
            self._clf_buffer.append(mono.copy())
            if sum(len(b) for b in self._clf_buffer) >= self._clf_window_samples:
                window = np.concatenate(self._clf_buffer)[: self._clf_window_samples]
                feats = extract_features(window, self._sr)
                label = self._classifier.predict(feats)
                with self._lock:
                    self._classifier_hits[label] = 1.0
                    self._classifier_last_label = label
                    self._classifier_hit_counts[label] += 1
                self._clf_capturing = False
                self._clf_buffer = []
        elif self._clf_prev_spectrum is not None and len(self._clf_prev_spectrum) == len(spectrum):
            flux = float(np.sum(np.maximum(spectrum - self._clf_prev_spectrum, 0)))
            local_avg = np.mean(self._clf_recent_flux) if self._clf_recent_flux else 0.0
            if flux > 2.5 * max(local_avg, 1e-6) and self._clf_blocks_since_onset >= 6:
                self._clf_blocks_since_onset = 0
                self._clf_capturing = True
                self._clf_buffer = [mono.copy()]
            self._clf_recent_flux.append(flux)
        self._clf_prev_spectrum = spectrum

    @property
    def classifier_hits(self) -> dict:
        """{"kick"/"snare"/"hihat": 0.0 or 1.0} - one-shot pulse, true only
        on the exact process() call where a hit just got classified."""
        with self._lock:
            return dict(self._classifier_hits)

    @property
    def has_classifier(self) -> bool:
        return self._classifier is not None

    @property
    def classifier_status(self) -> tuple[str, dict]:
        """(last_label, {"kick": count, "snare": count, "hihat": count}) -
        for the terminal display, so you can see it agrees with what you're
        actually hearing."""
        with self._lock:
            return self._classifier_last_label, dict(self._classifier_hit_counts)

    def band_scores_normalized(self) -> dict[str, float]:
        """Wie band_scores(), aber relativ zum langsam-zerfallenden Peak auf
        0..1 normiert (gleiche Methode wie BpmDisplay's Pegelleisten) - für
        die soundlight-Brücke (siehe soundlight_bridge.py), die einen
        0..1-Wertebereich erwartet, nicht die unskalierten Rohwerte."""
        with self._lock:
            scores = {"kick": self._kick_score, "snare": self._snare_score, "hihat": self._hihat_score}
            peaks = dict(self._band_peak)
        return {band: min(1.0, scores[band] / peaks[band]) for band in scores}

    def pop_hits(self) -> tuple[dict[str, float], dict[str, bool]]:
        """Liest die seit dem letzten Aufruf aufgelaufenen Pro-Band-Onset-
        Treffer (0..1, relativ zum Peak normiert) und löscht sie dabei
        atomar - verhindert, dass ein Treffer zwischen zwei (selteneren)
        Bridge-Pushes verloren geht, weil er nicht exakt auf einen
        Push-Block fiel. Gibt (hit_strength_by_band, triggered_by_band)."""
        with self._lock:
            peaks = dict(self._band_peak)
            hits = {band: min(1.0, v / peaks[band]) for band, v in self._hit_pending.items()}
            triggers = dict(self._trigger_pending)
            self._hit_pending = {band: 0.0 for band in self._hit_pending}
            self._trigger_pending = {band: False for band in self._trigger_pending}
        return hits, triggers

    @property
    def locked_bpm(self) -> float:
        return self._tempo.locked_bpm

    @property
    def tempo_state(self) -> str:
        return self._tempo.state

    @property
    def relock_candidate(self) -> float:
        return self._tempo.relock_candidate

    @property
    def relock_progress(self) -> tuple[int, int]:
        """(current_progress, relock_windows_needed)"""
        return self._tempo.relock_progress

    def set_band_weights(self, kick: float, snare: float, hihat: float) -> None:
        """Setzt die Mischgewichte der drei Drum-Bänder zur Laufzeit."""
        self._band_weights = (kick, snare, hihat)  # tuple-Zuweisung ist atomar

    # ── interne Methoden ──────────────────────────────────────────────────────

    def _spectral_flux_bands(
        self, mono: np.ndarray
    ) -> tuple[float, float, float, float, float]:
        """
        Halbwellen-gleichgerichteter Spectral Flux pro Drum-Band.

        Dünner Wrapper um _update_hpss_kick_analysis() + _flux_and_triggers()
        (Runde 6, Aufteilung wegen eines gefundenen Reihenfolge-Problems -
        siehe process()) - unverändertes Verhalten für bestehende isolierte
        Aufrufer (z.B. Tests, die diese Methode direkt statt process() nutzen).
        """
        mag = self._update_hpss_kick_analysis(mono)
        return self._flux_and_triggers(mag)

    def _update_hpss_kick_analysis(self, mono: np.ndarray):
        """
        Runde 6: erste Hälfte der bisherigen _spectral_flux_bands - FFT +
        HPSS-Trennung, setzt self._hpss_kick_energy (und die Snare-/Clap-
        HPSS-Energien sowie den Kick-Centroid) für den AKTUELLEN Block.
        Absichtlich von der zweiten Hälfte (_flux_and_triggers) getrennt,
        damit process() dazwischen _update_kalman_kick aufrufen kann - der
        Kick-HPSS-Gate braucht den Wert DIESES Blocks, nicht den des
        vorherigen (siehe process()-Kommentar zur Aufrufreihenfolge).

        Akkumuliert Samples in _sample_buf auf fft_size=4096 (≈93 ms bei
        44100 Hz bzw. ≈85 ms bei 48000 Hz - aktuelle Live-Rate, siehe
        club.yaml), was eine Frequenzauflösung von ~10.8 bzw. ~11.7 Hz/Bin
        ergibt und damit mehrere Bins im Kick-Band (30–120 Hz) sicherstellt.

        Gibt die HPSS-maskierte Magnitude zurück (für _flux_and_triggers),
        oder None während der Aufwärmphase (Puffer noch nicht voll).
        """
        self._sample_buf.extend(mono.tolist())

        # Erst rechnen wenn Puffer voll ist
        if len(self._sample_buf) < self._fft_size:
            return None

        frame = np.array(self._sample_buf, dtype=np.float32)
        windowed = frame * self._hann
        mag = np.abs(np.fft.rfft(windowed)).astype(np.float32)

        # HPSS percussive mask (see __init__ comment) - separates
        # transient/broadband content (kick, snare, hihat) from sustained/
        # narrowband content (basslines, pads, vocals) BEFORE any band
        # energy is measured, instead of trying to threshold the bleed
        # away afterwards.
        mag_pre_hpss = mag  # unmaskierte Rohmagnitude - für den optionalen kick-eigenen Kernel unten
        self._hpss_mag_history.append(mag)
        harmonic_est = None
        if len(self._hpss_mag_history) >= 4:
            harmonic_est = np.median(np.stack(self._hpss_mag_history), axis=0)
            percussive_est = median_filter(mag, size=self._hpss_freq_kernel, mode="nearest")
            mask = percussive_est / (harmonic_est + percussive_est + 1e-9)
            mag = mag * mask

        # Fresh per block (~11.6ms), used as a confirmation gate by the
        # low-latency Kalman kick tracker (_update_kalman_kick) - it needs
        # this ~93ms-windowed-but-content-aware signal to reject basslines
        # its own fast time-domain filter can't tell from a real kick,
        # without giving up its own microsecond timing.
        #
        # Runde 2, Phase 2: wenn ein eigener, kleinerer kick_hpss_freq_kernel_
        # bins gesetzt ist, wird NUR fürs Kick-Band eine eigene, schmalere
        # Perkussions-Maske berechnet (gemessen: der geteilte 17-Bin-Kernel
        # reicht ±86Hz - beim unteren Kickband-Rand deckt er praktisch das
        # gesamte 9-Bin-Kick-Band ab, beim oberen reicht er bis fast an die
        # Snare-Body-Untergrenze heran, siehe Analyse). Der geteilte Kernel
        # (`mag`/`mask` oben) bleibt für Snare/Clap unverändert bestehen.
        if self._kick_hpss_freq_kernel_bins and harmonic_est is not None:
            kick_percussive_est = median_filter(mag_pre_hpss, size=self._kick_hpss_freq_kernel_bins, mode="nearest")
            kick_mask = kick_percussive_est / (harmonic_est + kick_percussive_est + 1e-9)
            kick_mag = mag_pre_hpss * kick_mask
        else:
            kick_mag = mag

        self._hpss_kick_energy = float(np.sum(kick_mag * self._mask_kick))
        self._kick_harmonic_energy = float(np.sum(harmonic_est * self._mask_kick)) if harmonic_est is not None else 0.0
        # Gleiches Prinzip fuer die Snare-HPSS-Bestaetigung (Snap-Band) und
        # den eigenstaendigen Clap-Trigger - siehe _update_snare_trigger /
        # _update_clap_trigger.
        self._trig_snare_hpss_energy = float(np.sum(mag * self._mask_trig_snare_b))
        self._clap_hpss_energy = float(np.sum(mag * self._mask_clap_trigger))

        # Runde 2, Phase 3: Spectral Centroid im (HPSS-maskierten) Kick-Band -
        # nur Debug-/Score-Datum für die Bassline-Analyse (siehe
        # _update_kalman_kick), nie ein hartes Kriterium. Eine über mehrere
        # Blöcke stabile Tonhöhe (kleine Drift) ist bassline-typisch, ein
        # echter Kick hat oft einen kurzen Pitch-Sweep direkt im Attack.
        band_mag = kick_mag[self._kick_band_sel]
        total_band_mag = float(np.sum(band_mag))
        if total_band_mag > 1e-9:
            centroid = float(np.sum(self._kick_band_freqs * band_mag) / total_band_mag)
            self._kick_centroid_drift = (centroid - self._kick_prev_centroid) if self._kick_prev_centroid > 0 else 0.0
            self._kick_prev_centroid = centroid

        return mag

    def _flux_and_triggers(self, mag) -> tuple[float, float, float, float, float]:
        """Runde 6: zweite Hälfte der bisherigen _spectral_flux_bands -
        Diff/Flux-Berechnung + Snare-/Clap-Trigger-Auswertung
        (_update_triggers, das intern _update_clap_trigger aufruft).
        Läuft in process() NACH _update_kalman_kick, damit Snare/Clap ihren
        kick_bleed-Ausschluss anhand des AKTUELLEN Blocks prüfen (siehe
        process()) - unverändert gegenüber vorher, nur der Aufrufzeitpunkt
        relativ zu _update_kalman_kick hat sich verschoben.

        `mag` ist das Ergebnis von _update_hpss_kick_analysis() - None
        während der Aufwärmphase."""
        if mag is None:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        if self._prev_mag is None:
            self._prev_mag = mag
            return 0.0, 0.0, 0.0, 0.0, 0.0

        diff = np.maximum(0.0, mag - self._prev_mag)
        self._prev_mag = mag

        kick_f  = float(np.sum(diff * self._mask_kick))
        snare_f = float(np.sum(diff * self._mask_snare))
        hihat_f = float(np.sum(diff * self._mask_hihat))
        clap_f  = float(np.sum(diff * self._mask_clap))

        self._update_triggers(diff)

        wk, ws, wh = self._band_weights
        combined = wk * kick_f + ws * snare_f + wh * hihat_f
        return kick_f, snare_f, hihat_f, clap_f, combined

    def _adaptive_ratio(self, buf: "deque[float]", flux: float, std_factor: float, freeze: bool = False):
        """Gibt (ratio, threshold) zurück: `threshold` = Mean+Faktor*StdDev
        der Historie in `buf`, `ratio` = flux/threshold (>=1 heißt über der
        Schwelle). Verallgemeinert das alte boolesche _adaptive_hit zu einem
        kontinuierlichen Wert, den sowohl Snare (gewichtete Bewertung) als
        auch Clap (Confidence-Berechnung) brauchen.

        `freeze=True` hängt `flux` NICHT an `buf` an - während eines
        Treffers + seiner Nachklingzeit soll die Baseline eingefroren
        bleiben, sonst verschmutzt der Treffer selbst (oder sein Nachklingen)
        die Schwelle und der NÄCHSTE echte Treffer würde härter gemessen als
        er sollte (gleiches Prinzip wie beim Kalman-Kick-Tracker)."""
        if len(buf) < 4:
            if not freeze:
                buf.append(flux)
            return 0.0, float("inf")
        mean = float(np.mean(buf))
        std = float(np.std(buf))
        threshold = max(mean + std_factor * std, self._trig_noise_floor)
        ratio = flux / threshold
        if not freeze:
            buf.append(flux)
        return ratio, threshold

    def _update_triggers(self, diff: np.ndarray) -> None:
        """Gewichteter Snare-Trigger fürs DMX-Sound-to-Light (siehe
        Konstruktor-Kommentar bei trigger_snare_body_weight) - ersetzt die
        vorherige starre "Body UND Snap müssen beide ihre eigene Schwelle
        einzeln reißen"-Koinzidenz durch eine gewichtete Bewertung: entweder
        beide Bänder moderat aktiv, oder eines klar aktiv und das andere
        zumindest teilweise vorhanden. Ein einzelnes Band allein (das andere
        komplett still) reicht nie - das hält breitbandige Congas/Bongos/
        Shaker draußen, die oft nur eines der beiden Bänder treffen.
        Kick läuft separat über den Kalman-Filter-Tracker
        (_update_kalman_kick), da er die geglättete Vorhersage statt der
        Rohmessung braucht, nicht nur einen einfachen Schwellenwert-Vergleich."""
        now = time.time()
        # Baseline während Treffer+Nachklingzeit einfrieren (siehe
        # _adaptive_ratio) - referenziert den letzten BESTÄTIGTEN Snare-
        # Trigger, nicht die einzelnen Band-Schwellen, da die Bewertung
        # jetzt gemeinsam erfolgt statt pro Band unabhängig.
        freeze = (now - self._trig_last_snare_t) < max(self._trig_snare_holdoff_s, 0.2)

        snare_a_flux = float(np.sum(diff * self._mask_trig_snare_a))
        snare_b_flux = float(np.sum(diff * self._mask_trig_snare_b))

        body_ratio, _ = self._adaptive_ratio(self._trig_snare_a_flux, snare_a_flux, self._trig_snare_std_factor, freeze)
        snap_ratio, _ = self._adaptive_ratio(self._trig_snare_b_flux, snare_b_flux, self._trig_snare_std_factor, freeze)

        body_present = body_ratio >= self._trig_snare_body_min
        snap_present = snap_ratio >= self._trig_snare_snap_min
        body_strong = body_ratio >= 1.0
        snap_strong = snap_ratio >= 1.0

        if self._trig_snare_allow_weak_body:
            # Entweder ein Band voll über seiner Schwelle UND das andere
            # zumindest teilweise vorhanden (deckt schwachen Body bei
            # Latino/Reggaeton-Claps ab, siehe trigger_snare_allow_weak_body),
            # oder klassisch beide Bänder klar über ihrer Schwelle.
            coincidence_ok = (body_strong and snap_present) or (snap_strong and body_present) or (body_strong and snap_strong)
        else:
            coincidence_ok = body_strong and snap_strong  # altes strenges Verhalten, Opt-in

        score = (
            self._trig_snare_body_weight * min(body_ratio, 1.5)
            + self._trig_snare_snap_weight * min(snap_ratio, 1.5)
        )
        weight_sum = max(self._trig_snare_body_weight + self._trig_snare_snap_weight, 1e-9)
        confidence = max(0.0, min(1.0, score / weight_sum))

        hpss_ok = True
        if self._trig_snare_hpss_std_factor > 0:
            hbuf = self._trig_snare_hpss_baseline
            if len(hbuf) >= 4:
                h_mean = float(np.mean(hbuf))
                h_std = float(np.std(hbuf))
                hpss_ok = self._trig_snare_hpss_energy > (h_mean + self._trig_snare_hpss_std_factor * h_std)
            if not freeze:
                hbuf.append(self._trig_snare_hpss_energy)

        reason = None
        if not body_present and not snap_present:
            reason = "threshold_not_reached"
        elif not body_present:
            reason = "missing_body"
        elif not snap_present:
            reason = "missing_snap"
        elif not coincidence_ok:
            reason = "insufficient_rise"
        elif self._kalman_kick_is_impulse_now:
            reason = "kick_bleed"
        elif not hpss_ok:
            reason = "insufficient_percussive_energy"
        elif (now - self._trig_last_snare_t) < self._trig_snare_holdoff_s:
            reason = "cooldown_active"

        snare_trigger = reason is None

        if self._debug_triggers:
            self._snare_debug = {
                "body_flux": snare_a_flux, "snap_flux": snare_b_flux,
                "body_ratio": body_ratio, "snap_ratio": snap_ratio,
                "hpss_energy": self._trig_snare_hpss_energy, "hpss_ok": hpss_ok,
                "score": score, "confidence": confidence,
                "time_since_last": (now - self._trig_last_snare_t) if self._trig_last_snare_t else None,
                "reject_reason": reason,
            }

        with self._lock:
            if snare_trigger:
                self._trig_last_snare_t = now
            self._snare_trigger = snare_trigger

        self._update_clap_trigger(diff)

    def _update_clap_trigger(self, diff: np.ndarray) -> None:
        """Eigenständiger Clap-Trigger (siehe Konstruktor-Kommentar bei
        clap_trigger_lo) - mehrere unabhängige Kriterien statt reiner
        Bandenergie, damit kein einzelnes falsches Kriterium allein
        entscheidet: adaptive Schwelle (Energie im Clap-Band), Anschlags-
        schärfe (Energieanstieg ggü. ~35ms zuvor, gleiches Prinzip wie beim
        Kick), HPSS-perkussive Bestätigung, und Ausschluss bei
        gleichzeitigem Kick- oder Snare-Treffer (deren breitbandige
        Transienten sonst ins Clap-Band mit reinstrahlen)."""
        now = time.time()
        freeze = (now - self._clap_trig_last_t) < self._clap_trig_freeze_s

        flux = float(np.sum(diff * self._mask_clap_trigger))
        ratio, threshold = self._adaptive_ratio(self._clap_trig_flux_buf, flux, self._clap_trigger_std_factor, freeze)

        # Anschlagsfenster wird IMMER aktualisiert, auch während des
        # Freeze/Cooldown (anders als der Schwellenwert-Puffer oben) - bei
        # einem gehaltenen/dauerhaften Signal muss die Referenz für "vor
        # ~35ms" mit dem Signal mitwandern, sonst vergleicht der erste Block
        # nach Ablauf des Freeze das aktuelle (weiterhin laute) Signal noch
        # gegen einen alten, leisen Referenzwert von VOR dem ersten Treffer
        # und hält den Dauerton fälschlich für einen neuen scharfen Anstieg
        # (Doppeltrigger direkt an der Freeze-Grenze, empirisch beobachtet).
        attack = self._clap_trig_attack_buf
        is_sharp = len(attack) == attack.maxlen and flux > max(attack[0], self._trig_noise_floor) * self._clap_trigger_rise_factor
        attack.append(flux)

        hpss_ok = True
        if self._clap_trigger_hpss_std_factor > 0:
            hbuf = self._clap_trig_hpss_buf
            if len(hbuf) >= 4:
                h_mean = float(np.mean(hbuf))
                h_std = float(np.std(hbuf))
                hpss_ok = self._clap_hpss_energy > (h_mean + self._clap_trigger_hpss_std_factor * h_std)
            if not freeze:
                hbuf.append(self._clap_hpss_energy)

        # Snap-Band einer Snare reicht oft bis ins Clap-Band hinein - ohne
        # diesen Ausschluss würde praktisch jede Snare zusätzlich als Clap
        # zählen.
        recent_snare = (now - self._trig_last_snare_t) <= self._trig_snare_coincidence_s

        confidence = 0.0
        reason = None
        if ratio < 1.0:
            reason = "threshold_not_reached"
        elif not is_sharp:
            reason = "insufficient_rise"
        elif self._kalman_kick_is_impulse_now:
            reason = "kick_bleed"
        elif recent_snare:
            reason = "snare_bleed"
        elif not hpss_ok:
            reason = "insufficient_percussive_energy"
        elif (now - self._clap_trig_last_t) < self._clap_trigger_cooldown_s:
            reason = "cooldown_active"
        else:
            margin = min(ratio - 1.0, 2.0) / 2.0
            prior = attack[0] if attack else self._trig_noise_floor
            rise_component = min(flux / max(prior, self._trig_noise_floor) / self._clap_trigger_rise_factor, 1.0)
            confidence = max(0.0, min(1.0, 0.5 * margin + 0.5 * rise_component))
            if confidence < self._clap_trigger_min_confidence:
                reason = "low_confidence"

        clap_trigger = reason is None

        if self._debug_triggers:
            self._clap_debug = {
                "flux": flux, "threshold": threshold, "ratio": ratio,
                "hpss_energy": self._clap_hpss_energy, "hpss_ok": hpss_ok,
                "is_sharp": is_sharp, "confidence": confidence,
                "time_since_last": (now - self._clap_trig_last_t) if self._clap_trig_last_t else None,
                "reject_reason": reason,
            }

        with self._lock:
            if clap_trigger:
                self._clap_trig_last_t = now
            self._clap_trigger = clap_trigger

    def _update_kalman_kick(self, mono: np.ndarray) -> None:
        """Kick-Trigger über einen 2-Zustands-Kalman-Filter statt reinem
        Schwellenwert-Vergleich - siehe die ausführliche Diskussion, die zu
        diesem Ansatz geführt hat: reine FFT-Block-Heuristiken auf dem
        gemischten Signal sind strukturell zu ungenau/jitterig fürs
        DMX-Timing in einer Club/Bar-Umgebung.

        Zwei getrennte Stufen:

        1. Front-End (Signal-Isolation): ein KAUSALER IIR-Tiefpass
           (Butterworth, 4. Ordnung, ~130Hz) statt des FFT-Block-Ansatzes
           oben - Latenz im Mikrosekundenbereich statt der ~93ms des
           akkumulierten FFT-Fensters. Auf der gefilterten Energie läuft ein
           adaptiver Mean+StdDev-Schwellenwert (siehe _adaptive_hit), der
           einen rohen "Impuls"-Zeitstempel liefert - genau wie vorher,
           bloß viel schneller.

        2. Back-End (Zustandsschätzung): der rohe Impuls speist NUR eine
           Korrektur in ein Kalman-Filter mit State [nächster Beat-
           Zeitpunkt, Periode] - das Filter läuft dazwischen frei weiter
           (Prädiktion), auch durch Pausen/Drops hindurch ("Schwungrad").
           Der eigentliche Trigger feuert auf dieser VORHERGESAGTEN,
           geglätteten Taktung (_kalman_flash_t), nicht auf dem rohen
           Impuls direkt - das ist der Kern von "vorausschauendem Blitzen"
           statt "Blitzen bei jedem Treffer". Die Kalman-Verstärkung ist
           unsicherheits-adaptiv (klein wenn das Filter schon sicher ist,
           groß direkt nach einer längeren Pause) statt fest verdrahtet.
        """
        now = time.time()

        # Absolute Lautstärke-Untergrenze (Runde 9, 2026-10-13, Nutzer-
        # Feedback: Rap-Vocals/leise Sprech-Passagen ohne Beat triggern den
        # Kick, obwohl "das sind keine lauten Geräusche"). Die bisherigen
        # Gates unten sind alle RELATIV (wieviel Standardabweichungen über
        # dem eigenen, kurzen Hintergrund, wieviel schärfer als vor ~35ms) -
        # bei insgesamt leiser Musik/Sprache kann ein kleiner absoluter
        # Ausschlag trotzdem relativ "auffällig" wirken. Gleiche dBFS-Formel
        # wie capture.LevelMeter, auf dem vollen (ungefilterten) Mono-Block,
        # damit der konfigurierte Wert direkt mit dem dBFS-Pegel aus Monitor/
        # live_monitor.log vergleichbar ist - echte Club-Auswertung zeigte
        # 85-86% der (fälschlich akzeptierten) Kicks bei <-25dBFS.
        block_rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)))
        block_level_dbfs = 20.0 * np.log10(block_rms) if block_rms >= 1e-9 else -100.0

        filtered, self._kalman_kick_zi = sosfilt(self._kalman_kick_sos, mono, zi=self._kalman_kick_zi)
        envelope = float(np.sqrt(np.mean(filtered.astype(np.float64) ** 2)))

        attack = self._kalman_kick_attack_win
        is_sharp = len(attack) == attack.maxlen and envelope > max(attack[0], self._kalman_kick_noise_floor) * self._kalman_kick_sharpness_ratio
        # Nur fürs Debug (Runde 4, A/B-Test) - der tatsächliche Anstiegs-
        # Faktor als Zahl statt nur der booleschen is_sharp-Entscheidung.
        # Keine neue Logik, nur Sichtbarkeit auf einen bereits vorhandenen
        # Zwischenwert.
        sharpness_ratio = (envelope / max(attack[0], self._kalman_kick_noise_floor)) if len(attack) == attack.maxlen else None
        attack.append(envelope)

        buf = self._kalman_kick_env_recent
        # HPSS confirmation gate: the fast Butterworth envelope above can't
        # tell a real kick from a sharp-attacked bassline note in the same
        # band, since it only sees time-domain energy in a narrow filter.
        # The HPSS pipeline (_spectral_flux_bands, same block-latency this
        # already runs at) separates broadband/transient content from
        # narrowband/sustained content BEFORE band energy is measured, so
        # it's much better at telling "kick" from "bassline" - used here as
        # a second, independent vote rather than the timing source, so the
        # Kalman tracker's own low-latency prediction is unaffected either
        # way. Permissive (no veto) until its own baseline has enough
        # history, so it never blocks kicks during the first ~1s.
        #
        # Runde 2, Phase 2: folgt jetzt der gleichen "0=deaktiviert"-
        # Konvention wie bei Snare/Clap (kalman_kick_hpss_std_factor<=0
        # schaltet das Gate komplett ab, statt bei 0 effektiv am reinen
        # Mittelwert zu gaten) - vorher eine stille Inkonsistenz ggü. den
        # Trigger-Debug-Pfaden.
        hpss_buf = self._hpss_kick_baseline
        hpss_confirms = True
        hpss_mean = None
        hpss_std = None
        if self._kalman_kick_hpss_std_factor > 0 and len(hpss_buf) >= 4:
            hpss_mean = float(np.mean(hpss_buf))
            hpss_std = float(np.std(hpss_buf))
            hpss_confirms = self._hpss_kick_energy > (hpss_mean + self._kalman_kick_hpss_std_factor * hpss_std)
        hpss_buf.append(self._hpss_kick_energy)

        # Runde 5: Ring-Puffer für die Sharpness/HPSS-Timing-Diagnose - nur
        # befüllt wenn kick_gate_debug=True, sonst keine Kosten. Speichert
        # hpss_mean/std MIT (nicht nur die Energie), damit spätere Offset-
        # Auswertung denselben Vergleich nachbilden kann, den das Gate zu
        # diesem Zeitpunkt tatsächlich gemacht hätte.
        if self._kick_gate_debug_enabled:
            self._hpss_gate_ring.append((now, self._hpss_kick_energy, hpss_mean, hpss_std))

        # Runde 2, Phase 3: Dauer ununterbrochener Schwellenüberschreitung -
        # rein kausaler Zähler (kein Blick in die Zukunft nötig), direktes
        # Maß für "kurzer Kick-Blip" vs. "lang gehaltene Bassline/808-Note".
        # Nur Debug-/Score-Datum, siehe kick_bassline_suppression_enabled.
        threshold = float("inf")
        above_baseline = False
        baseline_mean = None
        baseline_std = None
        if len(buf) >= 4:
            baseline_mean = float(np.mean(buf))
            baseline_std = float(np.std(buf))
            threshold = baseline_mean + self._kalman_kick_std_factor * baseline_std
            above_baseline = envelope > threshold
        if above_baseline:
            self._kick_above_threshold_blocks += 1
        else:
            self._kick_above_threshold_blocks = 0
        duration_above_s = self._kick_above_threshold_blocks * self._hop

        # Runde 2, Phase 3: Abklingverhalten nach dem letzten Impuls - füllt
        # sich über die ~150ms NACH einem Treffer, daher zum Zeitpunkt des
        # Treffers selbst noch der Wert des VORHERIGEN Treffers (kausal,
        # kein Vorgriff). Ein echter Kick klingt in dieser Zeit stark ab
        # (kleines Verhältnis), eine gehaltene Bassline/808-Note bleibt nah
        # am Peak (Verhältnis nah 1).
        if self._kick_decay_active:
            self._kick_decay_samples.append(envelope)
            if len(self._kick_decay_samples) >= self._kick_decay_samples.maxlen:
                self._kick_decay_ratio = self._kick_decay_samples[-1] / max(self._kick_decay_peak, self._kalman_kick_noise_floor)
                self._kick_decay_active = False

        # Runde 2, Phase 3: optionale, rein additive Bassline-Confidence -
        # NIE ein einzelnes hartes Kriterium (kein starres Pitch-Stabilitäts-
        # Filter), sondern ein Mittel aus drei unabhängigen weichen Signalen
        # (Dauer, Abklingverhalten, Tonhöhen-Drift). Default deaktiviert ->
        # confidence bleibt immer 1.0, bassline_score immer 0.0, keinerlei
        # Verhaltensänderung ggü. Runde 1.
        bassline_score = 0.0
        confidence = 1.0
        if self._kick_bassline_suppression_enabled:
            duration_component = min(1.0, duration_above_s / max(self._kick_bassline_duration_threshold_s, 1e-6))
            decay_component = min(1.0, max(0.0, (self._kick_decay_ratio - self._kick_bassline_decay_threshold) / max(1.0 - self._kick_bassline_decay_threshold, 1e-6)))
            drift = abs(self._kick_centroid_drift)
            stability_component = 1.0 - min(1.0, drift / max(self._kick_pitch_drift_threshold, 1e-6))
            bassline_score = max(0.0, min(1.0, (duration_component + decay_component + stability_component) / 3.0))
            confidence = max(0.0, 1.0 - self._kick_bassline_penalty * bassline_score)
        self._kick_bassline_score = bassline_score
        self._kick_confidence = confidence

        # Ablehnungsgrund für kick_debug - schrittweise in derselben
        # Reihenfolge wie die eigentliche Entscheidung unten, damit beide
        # nie auseinanderlaufen können.
        reason = None
        if block_level_dbfs < self._kalman_kick_min_level_dbfs:
            reason = "level_too_quiet"
        elif len(buf) < 4 or envelope <= self._kalman_kick_noise_floor:
            reason = "threshold_not_reached"
        elif not above_baseline:
            reason = "insufficient_rise"
        elif not is_sharp:
            reason = "insufficient_sharpness"
        elif not hpss_confirms:
            reason = "insufficient_hpss"
        elif (now - self._kalman_kick_last_impulse_t) < self._kalman_kick_holdoff_s:
            reason = "cooldown_active"
        elif self._kick_bassline_suppression_enabled and confidence < self._kick_bassline_min_confidence:
            reason = "bassline_suspected"  # Zusatz zur Vorgabe-Liste, siehe Rundenbericht - nur wirksam wenn Suppression aktiv

        is_impulse = reason is None
        kalman_accepted = None

        self._kalman_kick_is_impulse_now = is_impulse
        if is_impulse:
            self._kalman_kick_last_impulse_t = now
            kalman_accepted = self._kalman_on_impulse(now)
            if kalman_accepted is False:
                reason = "kalman_rejected"  # nur informativ - ändert nichts an der bereits erfolgten Kalman-Prädiktion
            # Abklingverfolgung für den NÄCHSTEN Treffer neu starten.
            self._kick_decay_active = True
            self._kick_decay_peak = envelope
            self._kick_decay_samples.clear()

        if self._debug_triggers:
            self._kick_debug = {
                "envelope": envelope, "threshold": threshold if threshold != float("inf") else None,
                "level_dbfs": block_level_dbfs, "min_level_dbfs": self._kalman_kick_min_level_dbfs,
                "baseline_mean": baseline_mean, "baseline_std": baseline_std,
                "is_sharp": is_sharp, "sharpness_ratio": sharpness_ratio,
                "hpss_energy": self._hpss_kick_energy, "hpss_confirms": hpss_confirms,
                "harmonic_energy": self._kick_harmonic_energy,
                "duration_above_threshold_s": duration_above_s,
                "decay_ratio": self._kick_decay_ratio,
                "centroid_drift_hz": self._kick_centroid_drift,
                "bassline_score": bassline_score, "confidence": confidence,
                "kalman_state": {"t_next": self._kalman_t_next, "period_s": self._kalman_period, "initialized": self._kalman_initialized} if self._kalman_initialized else None,
                "kalman_accepted": kalman_accepted,
                "time_since_last": (now - self._kalman_kick_last_impulse_t) if self._kalman_kick_last_impulse_t and not is_impulse else (0.0 if is_impulse else None),
                "reject_reason": reason,
            }

        # Runde 5: Sharpness/HPSS-Timing-Diagnose - ein KANDIDAT (Envelope
        # über der Basis-Schwelle, unabhängig vom Endergebnis) wird zur
        # späteren Auswertung vorgemerkt, NICHT sofort geschrieben - siehe
        # _process_kick_gate_pending, das ein paar Blöcke später (sobald
        # genug "zukünftige" Ring-Daten da sind) die tatsächliche Zeile
        # fertigstellt. Rein additiv, keine Auswirkung auf is_impulse/reason.
        if self._kick_gate_debug_enabled and above_baseline:
            self._kick_gate_pending.append({
                "candidate_t": now,
                "envelope": envelope,
                "level_dbfs": block_level_dbfs,
                "sharpness_ratio": sharpness_ratio,
                "is_sharp": is_sharp,
                "hpss_energy_at_candidate": self._hpss_kick_energy,
                "hpss_mean_at_candidate": hpss_mean,
                "hpss_std_at_candidate": hpss_std,
                "hpss_threshold_at_candidate": (hpss_mean + self._kalman_kick_hpss_std_factor * hpss_std) if hpss_mean is not None else None,
                "hpss_confirms_at_candidate": hpss_confirms,
                "kalman_kick_hpss_std_factor": self._kalman_kick_hpss_std_factor,
                "reject_reason": reason,
            })
        if self._kick_gate_debug_enabled and self._kick_gate_pending:
            self._process_kick_gate_pending(now)

        # Das Tiefpassfilter klingt nach einem Treffer noch eine Weile nach
        # (Filter-Ringing) - würde diese erhöhte Nachklingphase (oder der
        # Treffer selbst) mit in die Hintergrund-Statistik einfließen,
        # bliebe die Schwelle künstlich hoch und der NÄCHSTE echte Kick
        # würde verpasst. Baseline während Treffer+Nachklingzeit einfrieren
        # (nicht in `buf` aufnehmen) statt sie zu verschmutzen. Nutzt den
        # frisch aktualisierten Zeitstempel, damit der Treffer selbst (bei
        # dem die Differenz sonst 0 wäre) sicher ausgeschlossen ist.
        if not is_impulse and (now - self._kalman_kick_last_impulse_t) >= self._kalman_kick_freeze_s:
            buf.append(envelope)

        fired = self._kalman_advance(now)
        with self._lock:
            self._kalman_kick_trigger = fired
            self._kick_trigger = fired

    _KICK_GATE_OFFSETS_MS = (0, 20, 40, 60, 80)

    def _process_kick_gate_pending(self, now: float) -> None:
        """Runde 5: schließt vorgemerkte Kick-Kandidaten ab, sobald genug
        "zukünftige" Ring-Daten vorliegen (letzter Offset 80ms + Marge),
        und liest für jeden Offset in _KICK_GATE_OFFSETS_MS den zeitlich
        nächstgelegenen Ring-Eintrag - keine Auswirkung auf die Live-
        Entscheidung, reiner Diagnose-Seitenkanal (siehe kick_gate_debug_rows).
        Läuft nur, wenn kick_gate_debug=True; sonst wird diese Methode nie
        aufgerufen (siehe _update_kalman_kick)."""
        ring = self._hpss_gate_ring
        if not ring:
            return
        max_offset_s = self._KICK_GATE_OFFSETS_MS[-1] / 1000.0
        still_pending = []
        for cand in self._kick_gate_pending:
            if now - cand["candidate_t"] < max_offset_s + 0.02:
                still_pending.append(cand)
                continue
            by_offset = {}
            for offset_ms in self._KICK_GATE_OFFSETS_MS:
                target_t = cand["candidate_t"] + offset_ms / 1000.0
                closest = min(ring, key=lambda e: abs(e[0] - target_t))
                _, energy, mean_, std_ = closest
                confirms = None
                if mean_ is not None and std_ is not None:
                    confirms = energy > (mean_ + cand["kalman_kick_hpss_std_factor"] * std_)
                by_offset[offset_ms] = {
                    "dt_actual_ms": round((closest[0] - cand["candidate_t"]) * 1000, 1),
                    "hpss_energy": energy, "hpss_mean": mean_, "hpss_std": std_,
                    "hpss_confirms": confirms,
                }
            row = dict(cand)
            row["hpss_confirms_by_offset"] = by_offset
            with self._lock:
                self._kick_gate_finalized.append(row)
        self._kick_gate_pending = still_pending

    @property
    def kick_gate_debug_rows(self) -> list:
        """Runde 5: liest UND LEERT die fertiggestellten Kick-Kandidaten-
        Diagnosezeilen (drain-on-read, wie eine Queue) - siehe
        _process_kick_gate_pending. Nur gefüllt wenn kick_gate_debug=True."""
        with self._lock:
            rows = list(self._kick_gate_finalized)
            self._kick_gate_finalized.clear()
        return rows

    def _kalman_on_impulse(self, t_impulse: float) -> bool:
        """Miss-Update-Schritt des Kalman-Filters bei einem akzeptierten
        Kick-Impuls. Rückgabewert ist rein informativ (für kick_debug,
        siehe _update_kalman_kick) - ändert nichts an der Filterlogik
        selbst: True = Messung wurde als Korrektur übernommen (oder
        Bootstrap-Phase), False = vom Ausreißer-Gate verworfen (die
        Prädiktion läuft trotzdem ungestört weiter, "Bassline statt Kick"
        relativ zum aktuellen Tempo)."""
        if not self._kalman_initialized:
            # Erststart: braucht eine grobe Periode als Startwert - direkt
            # aus dem Abstand echter Kick-Impulse gemessen, NICHT aus der
            # Autokorrelations-Schätzung (self._locked_bpm) - die kann
            # selbst oktav-falsch sein (halbes/doppeltes Tempo, bekanntes
            # Problem der Autokorrelation auf gemischtem Signal), und ein
            # falscher Bootstrap-Wert würde vom Ausreißer-Filter danach
            # sogar geschützt statt korrigiert, weil echte Treffer dann als
            # "Ausreißer" relativ zur falschen Periode erscheinen.
            #
            # Bugfix (2026-10-14, Live-Club-Test, Nutzer-Report "Kick-
            # Erkennung ist Kacke"): das reine Zwei-Punkt-Bootstrap unten
            # (Periode direkt aus dem Abstand der ERSTEN ZWEI akzeptierten
            # Impulse übernehmen, sofort eingerastet) hat sich im echten
            # Log als fragil erwiesen - live_monitor.log zeigte eine
            # eingerastete Periode von ~0.53s (≈114 BPM), obwohl der Song
            # längst stabil auf 93.8 BPM (0.639s) gelockt war
            # (lock_conf=1.00). EIN einzelner untypisch getimter/
            # fehlklassifizierter Kandidat (z.B. Snare-/Clap-Übersprechen,
            # das knapp durch die Sharpness-/HPSS-Gates rutscht) reicht
            # beim Zwei-Punkt-Verfahren, um dauerhaft auf der falschen
            # Periode einzurasten - und genau dieser Fall wird durch den
            # neuen kalman_kick_silence_timeout_s-Reset (siehe dort) sogar
            # HÄUFIGER erreicht, weil ein längerer Break mitten im Song
            # jetzt einen kompletten Neu-Bootstrap auslöst, also eine neue
            # Chance, erneut falsch einzurasten.
            # Jetzt: ein DRITTER Impuls muss die aus den ersten beiden
            # abgeleitete Kandidat-Periode erst bestätigen (muss nahe am
            # vorhergesagten übernächsten Takt liegen, Toleranz
            # _kalman_bootstrap_confirm_tol relativ zur Kandidat-Periode) -
            # erst dann rastet der Tracker wirklich ein. Bestätigt der
            # dritte Impuls NICHT, wird die Kandidat-Periode verworfen und
            # mit genau diesem dritten Impuls als neuem "ersten Punkt" neu
            # versucht (kein endloses Warten, nur ein strengerer Lock).
            if self._kalman_t_next == 0.0:
                self._kalman_t_next = t_impulse  # merkt sich den ersten Treffer
                self._kalman_bootstrap_candidate_period = None
            elif self._kalman_bootstrap_candidate_period is None:
                measured = t_impulse - self._kalman_t_next
                period_min = 60.0 / self._bpm_max
                period_max = 60.0 / self._bpm_min
                if period_min <= measured <= period_max:
                    self._kalman_bootstrap_candidate_period = measured
                    self._kalman_bootstrap_confirm_t = t_impulse + measured
                    self._kalman_t_next = t_impulse  # falls die Bestätigung scheitert, ist das hier der neue erste Punkt
                else:
                    self._kalman_t_next = t_impulse  # Ausreißer, neu ansetzen
            else:
                candidate_period = self._kalman_bootstrap_candidate_period
                predicted = self._kalman_bootstrap_confirm_t
                if abs(t_impulse - predicted) <= self._kalman_bootstrap_confirm_tol * candidate_period:
                    self._kalman_period = candidate_period
                    self._kalman_t_next = t_impulse + candidate_period
                    self._kalman_flash_t = self._kalman_t_next
                    self._kalman_initialized = True
                else:
                    self._kalman_t_next = t_impulse  # Kandidat verworfen, dieser Treffer wird der neue erste Punkt
                self._kalman_bootstrap_candidate_period = None
            return True  # Bootstrap-Phase, kein Ausreißer-Gate aktiv

        # Wie viele Beats sind seit dem letzten bestätigten Zeitpunkt
        # vergangen - erlaubt korrektes Vorwärtsspringen auch über
        # verpasste/ausgelassene Beats hinweg.
        t_last = self._kalman_t_next - self._kalman_period
        n = max(1, round((t_impulse - t_last) / self._kalman_period))

        t_pred = t_last + n * self._kalman_period
        F = np.array([[1.0, float(n)], [0.0, 1.0]])
        Q = n * np.array([[self._kalman_q_time, 0.0], [0.0, self._kalman_q_period]])
        P_pred = F @ self._kalman_P @ F.T + Q

        H = np.array([1.0, 0.0])
        y = t_impulse - t_pred  # Innovation (Phasenfehler)
        S = float(H @ P_pred @ H) + self._kalman_r

        if y * y > (self._kalman_gate ** 2) * S:
            # Ausreißer (z.B. Bassline statt echtem Kick) - Prädiktion
            # trotzdem übernehmen, aber ohne Messkorrektur.
            self._kalman_t_next = t_pred + self._kalman_period
            self._kalman_P = P_pred
            return False

        K = (P_pred @ H) / S  # Kalman-Verstärkung, Form (2,)
        t_new = t_pred + K[0] * y
        period_new = self._kalman_period + K[1] * y
        self._kalman_P = P_pred - np.outer(K, H) @ P_pred
        self._kalman_period = max(0.25, min(2.5, period_new))
        self._kalman_t_next = t_new + self._kalman_period
        self._kalman_flash_t = self._kalman_t_next  # Flash-Zeitplan mit der Korrektur synchronisieren
        return True

    def _kalman_advance(self, now: float) -> bool:
        """Reine Prädiktion, solange keine Messung ansteht - "Schwungrad"
        durch Pausen/Drops. Feuert auf dem vorhergesagten Zeitplan, nicht
        auf der Rohmessung - das eigentliche "vorausschauende Blitzen".

        Bugfix (siehe Konstruktor-Kommentar bei kalman_kick_silence_
        timeout_s): das Schwungrad lief früher unbegrenzt weiter, auch
        ohne jeden neuen echten Kandidaten - z.B. nach Songende. Jetzt:
        zu lange ohne neuen Kandidaten (_kalman_kick_last_impulse_t) ->
        deinitialisieren statt weiter vorherzusagen; der nächste echte
        Doppel-Treffer bootstrapped den Tracker dann neu."""
        if not self._kalman_initialized:
            return False
        if (now - self._kalman_kick_last_impulse_t) > self._kalman_kick_silence_timeout_s:
            self._kalman_initialized = False
            return False
        fired = False
        while now >= self._kalman_flash_t:
            fired = True
            self._kalman_flash_t += self._kalman_period
        return fired

    @property
    def kick_trigger(self) -> bool:
        """True nur exakt in dem process()-Aufruf, in dem ein Kick sauber
        ausgelöst hat (kein abklingender Puls wie onset_scores)."""
        with self._lock:
            return self._kick_trigger

    @property
    def snare_trigger(self) -> bool:
        """True nur exakt in dem process()-Aufruf, in dem eine Snare sauber
        ausgelöst hat (Koinzidenz zweier Bänder, kein abklingender Puls)."""
        with self._lock:
            return self._snare_trigger

    @property
    def clap_trigger(self) -> bool:
        """True nur exakt in dem process()-Aufruf, in dem ein Clap sauber
        ausgelöst hat (kein abklingender Puls wie onset_scores)."""
        with self._lock:
            return self._clap_trigger

    @property
    def trigger_debug(self) -> dict:
        """Diagnose-Informationen für Snare/Clap (Kick separat über
        kick_debug, da er einen eigenen Pfad/eigenes Vokabular hat). Nur
        gefüllt wenn debug_triggers=True beim Konstruktor gesetzt ist,
        sonst leere dicts.
        Struktur: {"snare": {...}, "clap": {...}} mit je Rohenergie,
        Schwelle, HPSS-Energie, Score/Confidence, Zeit seit letztem
        Trigger und - falls abgelehnt - reject_reason (z.B.
        "threshold_not_reached", "missing_body", "cooldown_active", ...)."""
        with self._lock:
            return {"snare": dict(self._snare_debug), "clap": dict(self._clap_debug)}

    @property
    def kick_debug(self) -> dict:
        """Diagnose-Informationen für den Kalman-Kick-Pfad (Runde 2) - siehe
        _update_kalman_kick. Nur gefüllt wenn debug_triggers=True, sonst
        leeres dict. Enthält u.a. envelope/threshold, is_sharp, HPSS-Energie
        + Bestätigung, Dauer ununterbrochener Schwellenüberschreitung,
        Abklingverhältnis, Centroid-Drift, Bassline-Score/Confidence
        (nur wirksam wenn kick_bassline_suppression_enabled=True), den
        Kalman-Zustand, ob die Kalman-Korrektur akzeptiert wurde
        (kalman_accepted) und - falls abgelehnt - reject_reason:
        "threshold_not_reached", "insufficient_rise", "insufficient_sharpness",
        "insufficient_hpss", "cooldown_active", "bassline_suspected"
        (Zusatz ggü. der Vorgabe-Liste, nur bei aktivierter Suppression) oder
        "kalman_rejected" (Impuls wurde akzeptiert, aber vom Kalman-
        Ausreißer-Gate verworfen - rein informativ, ändert den Trigger-
        Zeitpunkt selbst nicht)."""
        with self._lock:
            return dict(self._kick_debug)


# ── Terminal-Display ──────────────────────────────────────────────────────────

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _terminal_rows(lines: list, columns: int) -> int:
    """Zählt die tatsächlichen physischen Terminal-Zeilen, die `lines` beim
    Rendern belegen - NICHT einfach len(lines). Ursache eines echten Bugs
    (2026-09-23, live beim Club-Test aufgefallen): das Cursor-Redraw unten
    nutzte bisher len(lines) für '\\033[NA' (Cursor N Zeilen hoch), aber
    einzelne Log-Zeilen (z.B. kick_score/link-Zeilen mit Balken) sind oft
    breiter als ein schmales Terminal-Fenster und umbrechen dort in 2+
    physische Zeilen - dann korrigiert der Cursor-Sprung zu wenig, das
    Redraw setzt mitten im Vorframe an, alter Text bleibt stehen und das
    Terminal wächst pro Frame statt sich sauber zu überschreiben (bei
    schnellem refresh_every sichtbar als ständig wachsender Scrollback).
    Zählt sichtbare Zeichen (ANSI-Farbcodes rausgerechnet, die zählen nicht
    zur Zeilenbreite) und rundet pro Zeile auf volle Terminal-Zeilen auf."""
    if columns <= 0:
        return len(lines)
    total = 0
    for line in lines:
        visible = len(_ANSI_RE.sub("", line))
        total += max(1, -(-visible // columns))  # ceil, leere Zeile = 1 Zeile
    return total


def _score_bar(val: float, peak: float, width: int = 12) -> str:
    """Mini-Balken für einen Band-Score relativ zum Peak-Wert."""
    ratio = (val / peak) if peak > 1e-12 else 0.0
    ratio = max(0.0, min(1.0, ratio))
    filled = round(ratio * width)
    return "▪" * filled + "·" * (width - filled)


class BpmDisplay:
    """Rendert BPM-Diagnose, Band-Scores und Pegelleiste ins Terminal."""

    def __init__(self, bar_width: int = 40, db_floor: float = -60, debug_triggers: bool = False) -> None:
        self._meter = LevelMeter(bar_width=bar_width, db_floor=db_floor)
        self._initialized = False
        # Fix für die Laufzeit des Programms (Konstruktor-Flag, nicht der
        # jeweils aktuelle Debug-Dict-Inhalt) - hält die Zeilenanzahl pro
        # render()-Aufruf konstant, was das Cursor-Redraw unten braucht.
        self._debug_triggers = debug_triggers
        # Laufendes Maximum pro Band — für relativen Score-Balken
        self._peak_kick: float = 1e-12
        self._peak_snare: float = 1e-12
        self._peak_hihat: float = 1e-12
        self._peak_clap: float = 1e-12

    def update_meter(self, block: np.ndarray) -> None:
        self._meter.update(block)

    @property
    def meter(self) -> LevelMeter:
        """Für externe Konsumenten (live_monitor.py) - dieselbe Instanz
        statt einen zweiten, redundanten LevelMeter zu erzeugen."""
        return self._meter

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
        kick_s, snare_s, hihat_s, clap_s = det.band_scores
        clf_last_label, clf_counts = det.classifier_status
        if clf_last_label:
            clf_line = f"letzter Treffer: {clf_last_label:<6}  (kick:{clf_counts['kick']} snare:{clf_counts['snare']} hihat:{clf_counts['hihat']})"
        else:
            clf_line = "aus" if not det.has_classifier else "wartet auf ersten Treffer..."

        # Peak-Tracking mit langsamem Decay (×0.999 pro Render-Aufruf)
        self._peak_kick  = max(kick_s,  self._peak_kick  * 0.999)
        self._peak_snare = max(snare_s, self._peak_snare * 0.999)
        self._peak_hihat = max(hihat_s, self._peak_hihat * 0.999)
        self._peak_clap  = max(clap_s,  self._peak_clap  * 0.999)

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
            f"  clap_score      : [{_score_bar(clap_s,  self._peak_clap)}]  {clap_s:8.4f}",
            f"  {'─' * 55}",
        ]
        if self._debug_triggers:
            dbg = det.trigger_debug
            sd, cd, kd = dbg.get("snare", {}), dbg.get("clap", {}), det.kick_debug

            def _fmt_dbg(label: str, d: dict) -> str:
                if not d:
                    return f"  {label:<6}: warmup..."
                reason = d.get("reject_reason") or "TRIGGERED"
                age = d.get("time_since_last")
                age_str = f"{age*1000:5.0f}ms" if age is not None else "  —   "
                conf = d.get("confidence", d.get("score", 0.0))
                return f"  {label:<6}: {reason:<26} seit {age_str}  conf {conf:.2f}"

            lines += [
                _fmt_dbg("kick", kd),
                _fmt_dbg("snare", sd),
                _fmt_dbg("clap", cd),
                f"  {'─' * 55}",
            ]
        lines += [
            f"  Klassifikator   : {clf_line}",
            f"  {'─' * 55}",
            f"  link_enabled    : {link_enabled_str}  Peers {peers_str}",
            f"  session_tempo   : {sess_str}",
            f"  exported_bpm    : {exp_str}",
            f"  phase_offset    : {poff_str}",
            f"  {'─' * 55}",
        ]
        # Pegelzeilen anhängen (meter.render() → "  L ...\n  R ...")
        lines += self._meter.render().split("\n")

        # Tatsächliche physische Terminal-Zeilen statt len(lines) - siehe
        # _terminal_rows()-Doku: verhindert, dass umgebrochene (zu breite
        # für das Fenster) Zeilen das Cursor-Redraw aus dem Tritt bringen
        # und das Terminal pro Frame weiterwachsen lassen.
        columns = shutil.get_terminal_size(fallback=(80, 24)).columns
        rows = _terminal_rows(lines, columns)

        if not self._initialized:
            print("\n" * rows, end="")
            self._initialized = True

        sys.stdout.write(f"\033[{rows}A")
        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()


# ── Entry-Point ───────────────────────────────────────────────────────────────

def run_bpm(cfg: dict, monitor: dict, bpm_cfg: dict, link_cfg=None, soundlight_cfg=None, live_monitor_cfg=None) -> None:
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
    clap_lo: float        = bpm_cfg.get("clap_lo", 700.0)
    clap_hi: float        = bpm_cfg.get("clap_hi", 3000.0)
    kick_weight: float    = bpm_cfg.get("kick_weight", 0.6)
    snare_weight: float   = bpm_cfg.get("snare_weight", 0.25)
    hihat_weight: float   = bpm_cfg.get("hihat_weight", 0.15)
    oss_window_s: float   = bpm_cfg.get("oss_window_s", 8.0)
    onset_threshold: float    = bpm_cfg.get("onset_threshold", 1.4)
    kick_onset_threshold  = bpm_cfg.get("kick_onset_threshold")
    snare_onset_threshold = bpm_cfg.get("snare_onset_threshold")
    hihat_onset_threshold = bpm_cfg.get("hihat_onset_threshold")
    clap_onset_threshold  = bpm_cfg.get("clap_onset_threshold")
    refractory: float         = bpm_cfg.get("refractory", 0.2)
    smoothing: float          = bpm_cfg.get("smoothing", 0.2)
    hysteresis: float         = bpm_cfg.get("hysteresis", 1.0)
    lock_conf_min: float      = bpm_cfg.get("lock_confidence_min", 0.3)
    relock_conf_min: float    = bpm_cfg.get("relock_confidence_min", 0.5)
    max_jump_bpm: float       = bpm_cfg.get("max_jump_bpm", 10.0)
    hard_block_jump: float    = bpm_cfg.get("hard_block_jump_bpm", 15.0)
    relock_windows: int       = bpm_cfg.get("relock_windows", 3)
    hold_seconds: float       = bpm_cfg.get("hold_seconds", 8.0)
    trigger_snare_a_lo: float     = bpm_cfg.get("trigger_snare_a_lo", 200.0)
    trigger_snare_a_hi: float     = bpm_cfg.get("trigger_snare_a_hi", 300.0)
    trigger_snare_b_lo: float     = bpm_cfg.get("trigger_snare_b_lo", 2500.0)
    trigger_snare_b_hi: float     = bpm_cfg.get("trigger_snare_b_hi", 4500.0)
    trigger_window_s: float       = bpm_cfg.get("trigger_window_s", 0.4)
    trigger_snare_std_factor: float   = bpm_cfg.get("trigger_snare_std_factor", 1.5)
    trigger_snare_holdoff_s: float    = bpm_cfg.get("trigger_snare_holdoff_s", 0.15)
    trigger_snare_coincidence_s: float = bpm_cfg.get("trigger_snare_coincidence_s", 0.06)
    trigger_snare_body_weight: float  = bpm_cfg.get("trigger_snare_body_weight", 0.5)
    trigger_snare_snap_weight: float  = bpm_cfg.get("trigger_snare_snap_weight", 0.5)
    trigger_snare_body_min: float     = bpm_cfg.get("trigger_snare_body_min", 0.35)
    trigger_snare_snap_min: float     = bpm_cfg.get("trigger_snare_snap_min", 0.35)
    trigger_snare_allow_weak_body: bool = bpm_cfg.get("trigger_snare_allow_weak_body", True)
    trigger_snare_cooldown_s = bpm_cfg.get("trigger_snare_cooldown_s")
    trigger_snare_hpss_std_factor: float = bpm_cfg.get("trigger_snare_hpss_std_factor", 0.0)
    clap_trigger_lo: float           = bpm_cfg.get("clap_trigger_lo", 1000.0)
    clap_trigger_hi: float           = bpm_cfg.get("clap_trigger_hi", 6000.0)
    clap_trigger_std_factor: float   = bpm_cfg.get("clap_trigger_std_factor", 2.0)
    clap_trigger_rise_factor: float  = bpm_cfg.get("clap_trigger_rise_factor", 3.0)
    clap_trigger_cooldown_s: float   = bpm_cfg.get("clap_trigger_cooldown_s", 0.15)
    clap_trigger_min_confidence: float = bpm_cfg.get("clap_trigger_min_confidence", 0.3)
    clap_trigger_hpss_std_factor: float = bpm_cfg.get("clap_trigger_hpss_std_factor", 0.5)
    debug_triggers: bool = bpm_cfg.get("debug_triggers", False)
    kalman_kick_cutoff_hz: float  = bpm_cfg.get("kalman_kick_cutoff_hz", 130.0)
    kalman_kick_std_factor: float = bpm_cfg.get("kalman_kick_std_factor", 2.5)
    kalman_kick_sharpness_ratio: float = bpm_cfg.get("kalman_kick_sharpness_ratio", 4.0)
    kalman_kick_min_level_dbfs: float = bpm_cfg.get("kalman_kick_min_level_dbfs", -100.0)
    kalman_kick_holdoff_s: float  = bpm_cfg.get("kalman_kick_holdoff_s", 0.2)
    kalman_kick_silence_timeout_s: float = bpm_cfg.get("kalman_kick_silence_timeout_s", 4.0)
    kalman_kick_hpss_std_factor: float = bpm_cfg.get("kalman_kick_hpss_std_factor", 1.0)
    kalman_q_time: float          = bpm_cfg.get("kalman_q_time", 0.0004)
    kalman_q_period: float        = bpm_cfg.get("kalman_q_period", 0.00002)
    kalman_r: float                = bpm_cfg.get("kalman_r", 0.0009)
    kalman_outlier_gate: float     = bpm_cfg.get("kalman_outlier_gate", 3.0)
    kick_hpss_freq_kernel_bins = bpm_cfg.get("kick_hpss_freq_kernel_bins")
    kick_bassline_suppression_enabled: bool = bpm_cfg.get("kick_bassline_suppression_enabled", False)
    kick_bassline_duration_threshold_s: float = bpm_cfg.get("kick_bassline_duration_threshold_s", 0.35)
    kick_bassline_decay_threshold: float = bpm_cfg.get("kick_bassline_decay_threshold", 0.6)
    kick_pitch_drift_threshold: float = bpm_cfg.get("kick_pitch_drift_threshold", 15.0)
    kick_bassline_penalty: float = bpm_cfg.get("kick_bassline_penalty", 0.5)
    kick_bassline_min_confidence: float = bpm_cfg.get("kick_bassline_min_confidence", 0.3)
    kick_gate_debug: bool = bpm_cfg.get("kick_gate_debug", False)

    dev = resolve_device(device_name, fallback)
    dev_rate = dev.get("default_samplerate")
    rate_mismatch = "  ⚠ weicht von Device-Standardrate ab!" if dev_rate and int(dev_rate) != int(sample_rate) else ""
    print(f"\nGerät        : [{dev['index']}] {dev['name']}")
    print(f"Rate         : angefordert {sample_rate} Hz  |  Device-Standard {dev_rate} Hz{rate_mismatch}")
    print(f"Kanäle       : {channels} geöffnet, Block {block_size} Samples, Paar {channel_pair} ({channel_pair*2+1}+{channel_pair*2+2})")
    print(f"BPM-Bereich  : {bpm_min:.0f}–{bpm_max:.0f} BPM")
    print(f"Bänder       : Kick {kick_lo:.0f}–{kick_hi:.0f} Hz (×{kick_weight})"
          f"  Snare {snare_lo:.0f}–{snare_hi:.0f} Hz (×{snare_weight})"
          f"  HiHat {hihat_lo:.0f}–{hihat_hi:.0f} Hz (×{hihat_weight})"
          f"  Clap {clap_lo:.0f}–{clap_hi:.0f} Hz (kein Tempo-Gewicht)")
    print(f"OSS-Fenster  : {oss_window_s:.1f} s  |  Smoothing α={smoothing}  Hysterese {hysteresis} BPM")
    print(f"Tempo-Hold   : lock≥{lock_conf_min}  relock≥{relock_conf_min}  "
          f"max_jump {max_jump_bpm} BPM  hard_block {hard_block_jump} BPM  "
          f"relock {relock_windows}× Fenster  hold {hold_seconds} s")
    clf_path = bpm_cfg.get("drum_classifier_model")
    print(f"Klassifikator: {'aktiv -> ' + clf_path if clf_path else 'aus (nur Frequenzband-Erkennung)'}")

    # ── Ableton Link ──────────────────────────────────────────────────────────
    # mode: "output" (Default, bisheriges Verhalten - bpmdetects eigene Audio-
    # Schätzung wird in die Link-Session geschrieben, Link nur zur Anzeige) |
    # "input" (neu - Link ist externe Taktquelle, KEINE Rückschreibung, siehe
    # on_block unten) | "disabled" (wie enabled=false).
    link_mode = (link_cfg or {}).get("mode", "output")
    beat_grid_cfg = (link_cfg or {}).get("beat_grid", {}) or {}
    beat_grid_enabled = beat_grid_cfg.get("enabled", False)
    # Schlanke Debug-Zeile pro tatsächlichem beat_kick (siehe on_block) -
    # Default aus, rein additiv, keine Auswirkung auf Scheduler/Payload.
    debug_beats = (link_cfg or {}).get("debug_beats", False)

    link_bridge = None
    if link_cfg and link_cfg.get("enabled", False) and link_mode != "disabled":
        from link_bridge import LinkBridge
        quantum           = link_cfg.get("quantum", 4)
        upd_interval_ms   = link_cfg.get("update_interval_ms", 200.0)
        link_hysteresis   = link_cfg.get("tempo_hysteresis", 0.5)
        push_interval_ms  = link_cfg.get("push_interval_ms", 25.0)
        # Bugfix 2026-10-13 (Beat-Phase driftete, set_tempo korrigiert nur
        # die Taktrate) - siehe link_bridge.py update()/_phase_correction_target().
        phase_deadband    = link_cfg.get("phase_correction_deadband", 0.05)
        phase_interval_ms = link_cfg.get("phase_correction_interval_ms", 1000.0)
        link_bridge = LinkBridge(
            quantum=quantum,
            update_interval_ms=upd_interval_ms,
            tempo_hysteresis=link_hysteresis,
            push_interval_ms=push_interval_ms,
            phase_correction_deadband=phase_deadband,
            phase_correction_interval_ms=phase_interval_ms,
        )
        if link_bridge.start():
            print(f"Link         : aktiv  mode={link_mode}  quantum={quantum}  "
                  f"push={push_interval_ms:.0f}ms  update={upd_interval_ms:.0f} ms  hysterese={link_hysteresis} BPM  "
                  f"phasenkorrektur: Totzone={phase_deadband:.2f} Beat, min. Abstand={phase_interval_ms:.0f}ms")
        else:
            print("[link] Nicht verfügbar — fahre ohne Link fort.")
            link_bridge = None

    # ── Runde: Ableton-Link-Beat-Eingangsmodus (2026-09-22) ──────────────────
    # Rein additiv - None (Default, wenn beat_grid.enabled=false ODER kein
    # link_bridge) lässt das gesamte Verhalten unverändert. Ändert NICHTS an
    # der Audio-Kick-/HPSS-/Kalman-Erkennung in BeatDetector. Aktuell nur
    # kick_source="link" unterstützt.
    #
    # fallback_to_audio (2026-09-23 umgesetzt): wenn Link gerade keine
    # gültige externe Session hat (peers<=0 - z.B. weil nur Grand Visions
    # eigene, unabhängige Link-Verbindung als "Peer" gesehen wird, ohne
    # echte DJ-Software), treibt die bereits bestehende, live geprüfte
    # Audio-Kick-Erkennung (BeatDetector.kick_trigger) beat_kick stattdessen
    # an - siehe beat_scheduler.py. Reiner Durchreicher, keine neue
    # Erkennungslogik.
    beat_scheduler = None
    if link_bridge and beat_grid_enabled:
        from beat_scheduler import LinkBeatScheduler
        kick_source = beat_grid_cfg.get("kick_source", "link")
        if kick_source != "link":
            print(f"[beat_grid] kick_source='{kick_source}' wird noch nicht unterstützt (nur 'link') - beat_grid bleibt inaktiv.")
        else:
            fallback_to_audio = beat_grid_cfg.get("fallback_to_audio", False)
            beat_scheduler = LinkBeatScheduler(link_bridge, quantum=link_cfg.get("quantum", 4), fallback_to_audio=fallback_to_audio)
            print(f"Beat-Grid    : aktiv  kick_source=link  fallback_to_audio={fallback_to_audio}")

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
        clap_lo=clap_lo,
        clap_hi=clap_hi,
        kick_weight=kick_weight,
        snare_weight=snare_weight,
        hihat_weight=hihat_weight,
        oss_window_s=oss_window_s,
        onset_threshold=onset_threshold,
        kick_onset_threshold=kick_onset_threshold,
        snare_onset_threshold=snare_onset_threshold,
        hihat_onset_threshold=hihat_onset_threshold,
        clap_onset_threshold=clap_onset_threshold,
        refractory=refractory,
        smoothing=smoothing,
        hysteresis=hysteresis,
        lock_confidence_min=lock_conf_min,
        relock_confidence_min=relock_conf_min,
        max_jump_bpm=max_jump_bpm,
        hard_block_jump_bpm=hard_block_jump,
        relock_windows=relock_windows,
        hold_seconds=hold_seconds,
        classifier_model_path=bpm_cfg.get("drum_classifier_model"),
        trigger_snare_a_lo=trigger_snare_a_lo,
        trigger_snare_a_hi=trigger_snare_a_hi,
        trigger_snare_b_lo=trigger_snare_b_lo,
        trigger_snare_b_hi=trigger_snare_b_hi,
        trigger_window_s=trigger_window_s,
        trigger_snare_std_factor=trigger_snare_std_factor,
        trigger_snare_holdoff_s=trigger_snare_holdoff_s,
        trigger_snare_coincidence_s=trigger_snare_coincidence_s,
        trigger_snare_body_weight=trigger_snare_body_weight,
        trigger_snare_snap_weight=trigger_snare_snap_weight,
        trigger_snare_body_min=trigger_snare_body_min,
        trigger_snare_snap_min=trigger_snare_snap_min,
        trigger_snare_allow_weak_body=trigger_snare_allow_weak_body,
        trigger_snare_cooldown_s=trigger_snare_cooldown_s,
        trigger_snare_hpss_std_factor=trigger_snare_hpss_std_factor,
        clap_trigger_lo=clap_trigger_lo,
        clap_trigger_hi=clap_trigger_hi,
        clap_trigger_std_factor=clap_trigger_std_factor,
        clap_trigger_rise_factor=clap_trigger_rise_factor,
        clap_trigger_cooldown_s=clap_trigger_cooldown_s,
        clap_trigger_min_confidence=clap_trigger_min_confidence,
        clap_trigger_hpss_std_factor=clap_trigger_hpss_std_factor,
        debug_triggers=debug_triggers,
        kalman_kick_cutoff_hz=kalman_kick_cutoff_hz,
        kalman_kick_std_factor=kalman_kick_std_factor,
        kalman_kick_sharpness_ratio=kalman_kick_sharpness_ratio,
        kalman_kick_min_level_dbfs=kalman_kick_min_level_dbfs,
        kalman_kick_holdoff_s=kalman_kick_holdoff_s,
        kalman_kick_silence_timeout_s=kalman_kick_silence_timeout_s,
        kalman_kick_hpss_std_factor=kalman_kick_hpss_std_factor,
        kalman_q_time=kalman_q_time,
        kalman_q_period=kalman_q_period,
        kalman_r=kalman_r,
        kalman_outlier_gate=kalman_outlier_gate,
        kick_hpss_freq_kernel_bins=kick_hpss_freq_kernel_bins,
        kick_bassline_suppression_enabled=kick_bassline_suppression_enabled,
        kick_bassline_duration_threshold_s=kick_bassline_duration_threshold_s,
        kick_bassline_decay_threshold=kick_bassline_decay_threshold,
        kick_pitch_drift_threshold=kick_pitch_drift_threshold,
        kick_bassline_penalty=kick_bassline_penalty,
        kick_bassline_min_confidence=kick_bassline_min_confidence,
        kick_gate_debug=kick_gate_debug,
    )
    hz_per_bin = sample_rate / detector._fft_size
    print(f"FFT          : Größe {detector._fft_size} Samples  |  Auflösung {hz_per_bin:.2f} Hz/Bin  "
          f"(Kick-Band {kick_lo:.0f}-{kick_hi:.0f}Hz ≈ {(kick_hi - kick_lo) / hz_per_bin:.1f} Bins breit)")
    display = BpmDisplay(bar_width=bar_width, db_floor=db_floor, debug_triggers=debug_triggers)
    counter = 0
    last_beat_debug_t = 0.0  # für "Zeit seit dem letzten beat_kick" in der Debug-Zeile unten

    # ── Sound-to-Light Bridge (Grand Vision) ─────────────────────────────────
    soundlight_bridge = None
    if soundlight_cfg and soundlight_cfg.get("enabled", False):
        from soundlight_bridge import SoundlightBridge
        sl_url = soundlight_cfg.get("url", "http://localhost:3005/api/soundlight/levels")
        sl_interval_ms = soundlight_cfg.get("update_interval_ms", 33.0)
        # beat_scheduler NICHT mehr übergeben (2026-09-23, User-Entscheidung):
        # Grand Vision nutzt für Tempo/Takt nur noch seine eigene, direkte
        # Ableton-Link-Verbindung (linkSync.js/ctx.link) - der Umweg über
        # bpmdetects Link-Relais (beatKick/beatInBar/bpm im Payload) ist
        # raus. Code bleibt erhalten (nicht gelöscht), nur nicht mehr
        # verkabelt - beatKick/beatInBar/bpm fallen einfach auf ihre
        # Default-Werte (False/0) zurück, siehe soundlight_bridge.py.
        soundlight_bridge = SoundlightBridge(detector, url=sl_url, interval_ms=sl_interval_ms)
        soundlight_bridge.start()
        print(f"Sound-to-Light: aktiv  -> {sl_url}")

    # ── Live-Monitoring/Logging (Runde 3) - optional, Default aus ────────────
    # Rein additiv, liest nur bestehende Properties (kick_debug, trigger_debug,
    # bpm, ...), ändert nichts an der Erkennungslogik - siehe live_monitor.py.
    live_monitor = None
    if live_monitor_cfg and live_monitor_cfg.get("enabled", False):
        from live_monitor import LiveMonitor
        live_monitor = LiveMonitor(
            detector,
            level_meter=display.meter,
            refresh_hz=live_monitor_cfg.get("refresh_hz", 5.0),
            history_s=live_monitor_cfg.get("history_s", 60.0),
            log_path=live_monitor_cfg.get("log_path", "logs/live_monitor.log"),
            csv_path=live_monitor_cfg.get("csv_path"),
            keyboard_marker=live_monitor_cfg.get("keyboard_marker", True),
        )
        live_monitor.start()

    def on_block(block: np.ndarray) -> None:
        nonlocal counter, last_beat_debug_t
        detector.process(block)
        display.update_meter(block)
        if live_monitor:
            live_monitor.on_block()
        if beat_scheduler:
            # Jeden Block pollen (billig) statt nur alle refresh_every Blöcke
            # - beat_kick soll so zeitnah wie möglich zum Link-State-Push
            # erkannt werden, nicht zusätzlich an die Display-Rate gekoppelt.
            beat_scheduler.update(audio_kick_trigger=detector.kick_trigger)
            just_fired = beat_scheduler.beat_kick  # nicht-konsumierender Peek, siehe beat_scheduler.py
            if just_fired and soundlight_bridge:
                # Sofort wecken statt auf den nächsten ~33ms-Timer-Tick zu
                # warten (live beobachtet: Intervall-Loop + push_interval_ms
                # + Grand Visions eigener DMX-Tick summierten sich zu
                # spürbarem Nachlaufen). consume_beat_kick() in
                # _build_payload() bleibt die eigentliche, unabhängige
                # Quelle der Wahrheit - dieser Aufruf weckt nur den Thread.
                soundlight_bridge.notify_beat_kick()
            if debug_beats and just_fired:
                # Höchstens einmal pro echtem beat_kick (nicht pro Block) -
                # von Natur aus rate-begrenzt auf die Beat-Rate. Liest nur
                # bereits vorhandene Properties, verändert nichts an
                # LinkBridge/LinkBeatScheduler/Payload. Nach stderr statt
                # stdout, damit es nicht mit BpmDisplays ANSI-Redraw
                # kollidiert (gleiches Prinzip wie link_bridge._warn()).
                now_dbg = time.time()
                since_last = f"{now_dbg - last_beat_debug_t:.3f}s" if last_beat_debug_t else "—"
                ts_dbg = time.strftime("%H:%M:%S", time.localtime(now_dbg)) + f".{int(now_dbg % 1 * 1000):03d}"
                print(
                    f"[beat_grid] {ts_dbg}  source={beat_scheduler.source}  "
                    f"link_connected={link_bridge.available}  link_tempo={link_bridge.session_tempo:.1f}  "
                    f"link_phase={link_bridge.link_phase:.3f}  link_beat={link_bridge.link_beat:.3f}  "
                    f"beat_in_bar={beat_scheduler.beat_in_bar}  since_last={since_last}",
                    file=sys.stderr,
                )
                last_beat_debug_t = now_dbg
        counter += 1
        if counter % refresh_every == 0:
            if link_bridge and link_mode == "output":
                # Im input-Modus bewusst NICHT zurückschreiben - Link ist
                # dort die externe Taktquelle, nicht bpmdetects eigene
                # Audio-Schätzung (siehe Architektur-Bericht).
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
        if soundlight_bridge:
            soundlight_bridge.stop()
        if live_monitor:
            live_monitor.stop()
        print("\n\nBeat-Detection beendet.")
