"""Unit-Tests für die Snare-/Clap-Trigger-Logik (2026-09-22-Runde).

Reine unittest-Standardbibliothek (kein pytest installiert, siehe
requirements.txt) - läuft mit `python3 -m unittest discover tests` oder
`python3 tests/test_triggers.py` aus dem Projekt-Root.

Testet die Trigger-Methoden (_update_triggers/_update_clap_trigger) direkt
mit synthetischen, blockweise reproduzierbaren FFT-Differenz-Arrays statt
über echtes Audio - deterministisch und schnell. Die Uhr wird über
unittest.mock.patch auf einen kontrollierbaren FakeClock umgebogen, damit
Cooldown/Hold-off-Zeitfenster exakt getestet werden können.

Der Kick-Kalman-Pfad (_update_kalman_kick, siehe KickTriggerTests unten)
wird direkt mit synthetischen Zeitbereich-Audioblöcken getestet, nicht mit
FFT-Differenz-Arrays wie Snare/Clap oben - er arbeitet auf dem rohen
Butterworth-gefilterten Signal, nicht auf dem Spectral-Flux-Pfad. HPSS-
Bestätigung ist dabei bewusst deaktiviert (kalman_kick_hpss_std_factor=0.0),
da sie normalerweise parallel von _spectral_flux_bands befüllt wird, das
hier isoliert nicht mitläuft - gleiches Prinzip/gleiche Einschränkung wie
bei den ClapTriggerTests oben.
"""

import json
import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import beat_detector as bd
from soundlight_bridge import SoundlightBridge


class FakeClock:
    """Ersetzt time.time() im beat_detector-Modul durch eine kontrollierbare
    Uhr, die nur per .advance() vorwärts läuft - macht Cooldown/Freeze-
    Zeitfenster exakt und reproduzierbar testbar."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> float:
        self.t += dt
        return self.t


def make_detector(**overrides) -> bd.BeatDetector:
    kwargs = dict(sample_rate=44100, block_size=512, debug_triggers=True)
    kwargs.update(overrides)
    return bd.BeatDetector(**kwargs)


def band_diff(detector: bd.BeatDetector, lo_hz: float, hi_hz: float, magnitude: float) -> np.ndarray:
    """Synthetisches diff-Array (gleiche Länge wie die echten FFT-Magnitude-
    Arrays) mit `magnitude` gleichmäßig verteilt auf alle Bins im Bereich
    [lo_hz, hi_hz], sonst 0."""
    freqs = np.fft.rfftfreq(detector._fft_size, d=1.0 / detector._sr)
    diff = np.zeros_like(freqs, dtype=np.float32)
    mask = (freqs >= lo_hz) & (freqs <= hi_hz)
    n = int(np.sum(mask))
    if n > 0:
        diff[mask] = magnitude / n
    return diff


def noisy_band_diff(detector: bd.BeatDetector, lo_hz: float, hi_hz: float, magnitude: float, rng: np.random.Generator) -> np.ndarray:
    """Wie band_diff, aber mit zufälliger Streuung statt exakt konstanter
    Werte - für realistischere Baseline-/Warmup-Blöcke. Wichtig: die
    ERWARTETE Summe über alle Bins ist `magnitude` (gleiche Konvention wie
    band_diff), sonst hängt die tatsächliche Baseline-Energie von der
    Bandbreite (Anzahl Bins) ab statt vom übergebenen magnitude-Wert."""
    freqs = np.fft.rfftfreq(detector._fft_size, d=1.0 / detector._sr)
    diff = np.zeros_like(freqs, dtype=np.float32)
    mask = (freqs >= lo_hz) & (freqs <= hi_hz)
    n = int(np.sum(mask))
    if n > 0:
        diff[mask] = rng.uniform(0.0, 2.0 * magnitude / n, size=n)
    return diff


def make_kick_detector(**overrides) -> bd.BeatDetector:
    kwargs = dict(sample_rate=44100, block_size=512, debug_triggers=True, kalman_kick_hpss_std_factor=0.0)
    kwargs.update(overrides)
    return bd.BeatDetector(**kwargs)


def tone_blocks(detector: bd.BeatDetector, freq_hz: float, amplitude: float, n_blocks: int, start_sample: int = 0) -> list:
    """Phasenkontinuierliche Sinuston-Blöcke (gleiche Blockgröße wie der
    Detektor) - für _update_kalman_kick, das rohes Zeitbereich-Audio statt
    FFT-Differenz-Arrays erwartet."""
    bs, sr = detector._bs, detector._sr
    blocks = []
    for i in range(n_blocks):
        idx = start_sample + i * bs
        t = np.arange(idx, idx + bs) / sr
        blocks.append((amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32))
    return blocks


class ClapTriggerTests(unittest.TestCase):
    def setUp(self) -> None:
        # HPSS-Bestätigung für den Kern-Trigger-Test bewusst aus: die echte
        # HPSS-Energie wird normalerweise von _spectral_flux_bands parallel
        # zum diff-Array berechnet, das hier isoliert vorbei an
        # _update_clap_trigger gereicht wird - ein voller Spektrogramm-
        # Aufbau (harmonisch+perkussiv) für den HPSS-Pfad selbst ist für
        # diese leichte Testrunde bewusst nicht gebaut (siehe Abschlussbericht,
        # "bekannte Einschränkungen").
        self.det = make_detector(clap_trigger_hpss_std_factor=0.0)
        self.clock = FakeClock()
        self.rng = np.random.default_rng(42)

    def _warmup(self, n_blocks: int = 20, magnitude: float = 0.01) -> None:
        for _ in range(n_blocks):
            diff = noisy_band_diff(self.det, 1000, 6000, magnitude, self.rng)
            self.det._update_clap_trigger(diff)
            self.clock.advance(self.det._hop)

    def test_clap_trigger_on_valid_transient(self) -> None:
        """Ein einzelner, scharfer Transient im Clap-Band nach ruhiger
        Baseline muss auslösen."""
        with patch.object(bd.time, "time", new=self.clock):
            self._warmup()
            self.clock.advance(0.5)  # sicher außerhalb jeder Freeze-Zeit
            loud = band_diff(self.det, 1000, 6000, magnitude=5.0)
            self.det._update_clap_trigger(loud)
            self.assertTrue(self.det.clap_trigger, self.det.trigger_debug["clap"])
            self.assertIsNone(self.det.trigger_debug["clap"]["reject_reason"])

    def test_clap_suppressed_on_sustained_signal(self) -> None:
        """Ein dauerhaft lautes Signal (keine echten Transienten, z.B. ein
        gehaltener Ton/Rauschen im Clap-Band) darf nach dem ersten Anstieg
        nicht wiederholt auslösen."""
        with patch.object(bd.time, "time", new=self.clock):
            self._warmup()
            self.clock.advance(0.5)
            loud = band_diff(self.det, 1000, 6000, magnitude=5.0)

            # Erster Block: echter Anstieg, darf/soll auslösen.
            self.det._update_clap_trigger(loud)
            first_fired = self.det.clap_trigger

            # Viele weitere Blöcke auf GLEICHEM (nicht weiter ansteigendem)
            # Pegel - deckt sowohl die Cooldown-Zeit als auch die Zeit ab,
            # bis das Anschlagsfenster den Dauerpegel "eingeholt" hat.
            retriggers = 0
            for _ in range(40):
                self.clock.advance(self.det._hop)
                self.det._update_clap_trigger(loud)
                if self.det.clap_trigger:
                    retriggers += 1

            self.assertTrue(first_fired)
            self.assertEqual(retriggers, 0, "Ein gehaltenes Signal hat wiederholt getriggert")

    def test_cooldown_prevents_double_trigger(self) -> None:
        """Zwei sehr kurz aufeinanderfolgende, gleich starke Transienten
        dürfen nur einmal auslösen (Cooldown)."""
        with patch.object(bd.time, "time", new=self.clock):
            self._warmup()
            self.clock.advance(0.5)
            loud = band_diff(self.det, 1000, 6000, magnitude=5.0)

            self.det._update_clap_trigger(loud)
            self.assertTrue(self.det.clap_trigger)

            # Deutlich kürzer als clap_trigger_cooldown_s (Default 0.15s)
            self.clock.advance(0.02)
            self.det._update_clap_trigger(loud)
            self.assertFalse(self.det.clap_trigger)
            self.assertEqual(self.det.trigger_debug["clap"]["reject_reason"], "cooldown_active")


class SnareTriggerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.det = make_detector(trigger_snare_hpss_std_factor=0.0, clap_trigger_hpss_std_factor=0.0)
        self.clock = FakeClock()
        self.rng = np.random.default_rng(7)

    def _warmup(self, n_blocks: int = 20, magnitude: float = 0.01) -> None:
        for _ in range(n_blocks):
            diff = (
                noisy_band_diff(self.det, 200, 300, magnitude, self.rng)
                + noisy_band_diff(self.det, 2500, 4500, magnitude, self.rng)
            )
            self.det._update_triggers(diff)
            self.clock.advance(self.det._hop)

    def test_snare_strong_snap_weak_body(self) -> None:
        """Starker Snap (deutlich über Schwelle) mit nur schwachem, aber
        vorhandenem Body (z.B. Reggaeton-Clap/Rimshot) muss mit
        trigger_snare_allow_weak_body=True auslösen."""
        with patch.object(bd.time, "time", new=self.clock):
            self._warmup()
            self.clock.advance(0.5)
            # Body knapp über body_min (0.35 relativ zur eigenen Schwelle),
            # Snap weit über seiner eigenen Schwelle.
            diff = band_diff(self.det, 200, 300, magnitude=0.008) + band_diff(self.det, 2500, 4500, magnitude=5.0)
            self.det._update_triggers(diff)
            dbg = self.det.trigger_debug["snare"]
            self.assertTrue(self.det.snare_trigger, dbg)

    def test_snare_strict_mode_rejects_weak_body(self) -> None:
        """Gleiches Signal wie oben, aber mit trigger_snare_allow_weak_body
        =False (altes strenges Verhalten als Opt-out) darf NICHT auslösen,
        da der Body nie die volle eigene Schwelle erreicht."""
        det = make_detector(trigger_snare_allow_weak_body=False, trigger_snare_hpss_std_factor=0.0, clap_trigger_hpss_std_factor=0.0)
        clock = FakeClock()
        rng = np.random.default_rng(7)
        with patch.object(bd.time, "time", new=clock):
            for _ in range(20):
                diff = (
                    noisy_band_diff(det, 200, 300, 0.01, rng)
                    + noisy_band_diff(det, 2500, 4500, 0.01, rng)
                )
                det._update_triggers(diff)
                clock.advance(det._hop)
            clock.advance(0.5)
            diff = band_diff(det, 200, 300, magnitude=0.008) + band_diff(det, 2500, 4500, magnitude=5.0)
            det._update_triggers(diff)
            self.assertFalse(det.snare_trigger, det.trigger_debug["snare"])

    def test_snare_suppressed_single_band_snap_only(self) -> None:
        """Reine Snap-Energie ohne jede Body-Aktivität (z.B. eine Hihat/ein
        Becken) darf nicht als Snare zählen."""
        with patch.object(bd.time, "time", new=self.clock):
            self._warmup()
            self.clock.advance(0.5)
            diff = band_diff(self.det, 2500, 4500, magnitude=5.0)  # kein Body-Anteil
            self.det._update_triggers(diff)
            dbg = self.det.trigger_debug["snare"]
            self.assertFalse(self.det.snare_trigger, dbg)
            self.assertEqual(dbg["reject_reason"], "missing_body")

    def test_snare_suppressed_single_band_body_only(self) -> None:
        """Reine Body-Energie ohne jede Snap-Aktivität (z.B. eine Bassline/
        ein tiefes Tom) darf nicht als Snare zählen."""
        with patch.object(bd.time, "time", new=self.clock):
            self._warmup()
            self.clock.advance(0.5)
            diff = band_diff(self.det, 200, 300, magnitude=5.0)  # kein Snap-Anteil
            self.det._update_triggers(diff)
            dbg = self.det.trigger_debug["snare"]
            self.assertFalse(self.det.snare_trigger, dbg)
            self.assertEqual(dbg["reject_reason"], "missing_snap")

    def test_snare_cooldown_prevents_double_trigger(self) -> None:
        with patch.object(bd.time, "time", new=self.clock):
            self._warmup()
            self.clock.advance(0.5)
            diff = band_diff(self.det, 200, 300, magnitude=0.5) + band_diff(self.det, 2500, 4500, magnitude=5.0)
            self.det._update_triggers(diff)
            self.assertTrue(self.det.snare_trigger)

            self.clock.advance(0.02)  # < trigger_snare_holdoff_s (0.15s)
            self.det._update_triggers(diff)
            self.assertFalse(self.det.snare_trigger)
            self.assertEqual(self.det.trigger_debug["snare"]["reject_reason"], "cooldown_active")


class KickTriggerTests(unittest.TestCase):
    """Runde 2, Phase 4 - testet _update_kalman_kick direkt mit
    synthetischen Zeitbereich-Audioblöcken (Sinuston statt FFT-Diff-Array,
    siehe tone_blocks/make_kick_detector oben)."""

    def setUp(self) -> None:
        self.det = make_kick_detector()
        self.clock = FakeClock()

    def _silence(self, n_blocks: int) -> None:
        for _ in range(n_blocks):
            self.det._update_kalman_kick(np.zeros(self.det._bs, dtype=np.float32))
            self.clock.advance(self.det._hop)

    def test_short_deep_impulse_is_candidate(self) -> None:
        """Ein einzelner, kurzer, tiefer Impuls nach ruhiger Baseline muss
        als Kick-Kandidat akzeptiert werden (reject_reason is None)."""
        with patch.object(bd.time, "time", new=self.clock):
            self._silence(30)
            burst = tone_blocks(self.det, 60.0, 0.8, 1)[0]
            self.det._update_kalman_kick(burst)
            dbg = self.det.kick_debug
            self.assertIsNone(dbg["reject_reason"], dbg)
            self.assertTrue(dbg["is_sharp"])

    def test_kick_trigger_fires_directly_on_every_real_impulse_no_flywheel(self) -> None:
        """Bugfix 2026-10-14 (Live-Club-Test, klare Nutzer-Anforderung:
        "einen Trigger auf jede Kick und einen Trigger auf jede Snare.
        Punkt."): kick_trigger feuerte bisher auf der VORHERGESAGTEN
        Kalman-Taktung (_kalman_flash_t), nicht auf dem rohen akzeptierten
        Impuls - konnte dadurch auf eine falsche Periode einrasten und
        dauerhaft hörbar neben dem Takt blitzen (echtes live_monitor.log
        bestätigt: 0.51s statt der echten 0.64s/93.8 BPM). Jetzt muss
        kick_trigger exakt dem rohen, akzeptierten Impuls entsprechen -
        (1) feuert SOFORT beim allerersten jemals akzeptierten Impuls,
        ganz ohne vorherigen Bootstrap/Lock, und (2) bleibt in einer
        stillen Lücke zwischen zwei echten Treffern False, auch wenn ein
        bereits gelockter Flywheel an dieser Stelle vorhersagen würde."""
        with patch.object(bd.time, "time", new=self.clock):
            self._silence(30)
            burst = tone_blocks(self.det, 60.0, 0.8, 1)[0]

            # (1) Allererster Impuls überhaupt - kein Bootstrap nötig mehr.
            self.det._update_kalman_kick(burst)
            self.assertIsNone(self.det.kick_debug["reject_reason"])
            self.assertTrue(self.det.kick_trigger, "muss sofort beim ersten echten Impuls feuern, ganz ohne vorherigen Lock")

            # Direkt im nächsten Block (kein neuer Impuls) muss es wieder
            # False sein - kein abklingender/anhaltender Puls.
            self.det._update_kalman_kick(np.zeros(self.det._bs, dtype=np.float32))
            self.clock.advance(self.det._hop)
            self.assertFalse(self.det.kick_trigger)

            # (2) Lock etablieren (zweiter Treffer im gültigen Tempo-
            # Bereich), dann eine stille Lücke GENAU an der Stelle, wo ein
            # gelockter Flywheel vorhersagen würde (Periode ~0.46s bei
            # diesem 40-Block-Abstand) - kick_trigger darf dort trotzdem
            # nicht feuern, da kein echter Kandidat vorliegt.
            for _ in range(40):
                self.det._update_kalman_kick(np.zeros(self.det._bs, dtype=np.float32))
                self.clock.advance(self.det._hop)
            self.det._update_kalman_kick(burst)
            self.assertIsNone(self.det.kick_debug["reject_reason"])
            self.assertTrue(self.det._kalman_initialized, "Testvoraussetzung: Flywheel sollte jetzt gelockt sein")

            for _ in range(40):  # exakt die gelockte Periode, aber NUR Stille, kein echter Kandidat
                self.det._update_kalman_kick(np.zeros(self.det._bs, dtype=np.float32))
                self.assertFalse(self.det.kick_trigger, "darf trotz gelocktem Flywheel nicht auf Vorhersage feuern, nur auf echten Treffern")
                self.clock.advance(self.det._hop)

    def test_sustained_deep_signal_not_every_block(self) -> None:
        """Ein lang gehaltenes, tiefes Signal (kein echter Kick-Transient,
        z.B. eine Bassline/808-Sustain-Note) darf nicht bei jedem Block neu
        auslösen."""
        with patch.object(bd.time, "time", new=self.clock):
            self._silence(30)
            blocks = tone_blocks(self.det, 60.0, 0.8, 60)
            accepted = 0
            for b in blocks:
                self.det._update_kalman_kick(b)
                if self.det.kick_debug["reject_reason"] is None:
                    accepted += 1
                self.clock.advance(self.det._hop)
            self.assertGreaterEqual(accepted, 1, "der erste Anstieg sollte noch auslösen")
            self.assertLessEqual(accepted, 2, f"Dauerton hat {accepted}x ausgelöst statt nur beim ersten Anstieg")

    def test_cooldown_prevents_double_trigger(self) -> None:
        with patch.object(bd.time, "time", new=self.clock):
            self._silence(30)
            burst = tone_blocks(self.det, 60.0, 0.8, 1)[0]
            self.det._update_kalman_kick(burst)
            self.assertIsNone(self.det.kick_debug["reject_reason"])

            self.clock.advance(self.det._hop)  # deutlich < kalman_kick_holdoff_s (Default 0.2s)
            self.det._update_kalman_kick(burst)
            self.assertEqual(self.det.kick_debug["reject_reason"], "cooldown_active")

    def test_weak_kick_not_suppressed_by_stuck_baseline(self) -> None:
        """Ein schwacher echter Kick nach einer längeren lauten Phase darf
        nicht durch eine an der lauten Phase "hängengebliebene" Baseline
        unterdrückt werden - die Baseline friert während Treffer+Nachklingen
        ein (bestehende Logik), statt sich hochzuziehen."""
        with patch.object(bd.time, "time", new=self.clock):
            self._silence(30)
            burst = tone_blocks(self.det, 60.0, 0.8, 1)[0]
            self.det._update_kalman_kick(burst)
            self.assertIsNone(self.det.kick_debug["reject_reason"])

            self._silence(40)  # Holdoff + Freeze abgelaufen, Baseline erholt sich wieder auf niedrigem Pegel

            weak = tone_blocks(self.det, 60.0, 0.15, 1)[0]  # deutlich leiser als der erste Ausreißer
            self.det._update_kalman_kick(weak)
            dbg = self.det.kick_debug
            self.assertIsNone(dbg["reject_reason"], dbg)

    def test_kick_debug_contains_reason_and_fields(self) -> None:
        with patch.object(bd.time, "time", new=self.clock):
            self.det._update_kalman_kick(np.zeros(self.det._bs, dtype=np.float32))
            dbg = self.det.kick_debug
            self.assertEqual(dbg["reject_reason"], "threshold_not_reached")
            for key in (
                "envelope", "threshold", "is_sharp", "hpss_energy", "hpss_confirms",
                "duration_above_threshold_s", "decay_ratio", "centroid_drift_hz",
                "bassline_score", "confidence", "kalman_accepted", "reject_reason",
            ):
                self.assertIn(key, dbg)

    def test_kalman_flywheel_deinitializes_after_silence_timeout(self) -> None:
        """Bugfix 2026-10-14 (Live-Club-Test, Nutzer-Report "läuft keine
        Musik und Kick triggert trotzdem"): _kalman_advance() ("Schwungrad
        durch Pausen/Drops") feuerte früher unbegrenzt auf dem zuletzt
        gelockten Tempo weiter, auch ohne jeden neuen Kandidaten - z.B.
        nach Songende. Muss sich nach kalman_kick_silence_timeout_s ohne
        jeden neuen Kandidaten deinitialisieren statt weiterzuspielen."""
        det = make_kick_detector(kalman_kick_silence_timeout_s=0.2)
        clock = FakeClock()
        with patch.object(bd.time, "time", new=clock):
            # Bootstrap: zwei echte Treffer im Abstand von ~0.46s (innerhalb
            # bpm_min=70/bpm_max=180, Default) lassen den Tracker einrasten -
            # derselbe 40-Block-Silence-Abstand wie in
            # test_weak_kick_not_suppressed_by_stuck_baseline oben, dort
            # bereits als ausreichend für eine erneute Akzeptanz bestätigt.
            for _ in range(30):
                det._update_kalman_kick(np.zeros(det._bs, dtype=np.float32))
                clock.advance(det._hop)
            burst = tone_blocks(det, 60.0, 0.8, 1)[0]
            det._update_kalman_kick(burst)
            self.assertIsNone(det.kick_debug["reject_reason"])
            for _ in range(40):
                det._update_kalman_kick(np.zeros(det._bs, dtype=np.float32))
                clock.advance(det._hop)
            det._update_kalman_kick(burst)
            self.assertIsNone(det.kick_debug["reject_reason"])
            self.assertTrue(det._kalman_initialized, "nach zwei echten Treffern im Tempo-Bereich sollte der Tracker eingerastet sein")

            # Deutlich länger als der (für den Test verkürzte) Timeout
            # Stille, ohne jeden weiteren Kandidaten.
            for _ in range(60):
                det._update_kalman_kick(np.zeros(det._bs, dtype=np.float32))
                clock.advance(det._hop)

            self.assertFalse(det._kalman_initialized, "nach > silence_timeout ohne jeden Kandidaten muss sich der Tracker deinitialisieren")
            self.assertFalse(det.kick_trigger, "darf nach der Stille nicht mehr auf dem alten Tempo weiterfeuern")


class KickHpssKernelTests(unittest.TestCase):
    """Runde 2, Phase 2 - der optionale, kick-eigene HPSS-Kernel."""

    def test_default_is_none_unchanged_behavior(self) -> None:
        det = make_kick_detector()
        self.assertIsNone(det._kick_hpss_freq_kernel_bins)

    def test_even_value_gets_forced_odd(self) -> None:
        det = make_kick_detector(kick_hpss_freq_kernel_bins=8)
        self.assertEqual(det._kick_hpss_freq_kernel_bins, 9)

    def test_kernel_applies_without_error(self) -> None:
        det = make_kick_detector(kick_hpss_freq_kernel_bins=7)
        rng = np.random.default_rng(3)
        for _ in range(6):
            mono = (rng.standard_normal(det._bs) * 0.1).astype(np.float32)
            det._spectral_flux_bands(mono)
        self.assertIsInstance(det._hpss_kick_energy, float)
        self.assertGreaterEqual(det._hpss_kick_energy, 0.0)


class KickBasslineSuppressionTests(unittest.TestCase):
    """Runde 2, Phase 3 - Bassline-Suppression ist per Default deaktiviert
    und darf dann keinerlei Verhaltensänderung bewirken."""

    def test_disabled_by_default(self) -> None:
        det = make_kick_detector()
        self.assertFalse(det._kick_bassline_suppression_enabled)

    def test_disabled_never_suppresses_or_reports_bassline(self) -> None:
        det = make_kick_detector()  # Suppression bleibt Default False
        clock = FakeClock()
        with patch.object(bd.time, "time", new=clock):
            for _ in range(30):
                det._update_kalman_kick(np.zeros(det._bs, dtype=np.float32))
                clock.advance(det._hop)
            reasons = set()
            for b in tone_blocks(det, 60.0, 0.8, 50):
                det._update_kalman_kick(b)
                reasons.add(det.kick_debug["reject_reason"])
                clock.advance(det._hop)
            self.assertNotIn("bassline_suspected", reasons)
            self.assertEqual(det.kick_debug["confidence"], 1.0)
            self.assertEqual(det.kick_debug["bassline_score"], 0.0)


class KickGateTimingDiagnosticTests(unittest.TestCase):
    """Runde 5 - Offline-/Replay-Test für die Sharpness/HPSS-Timing-
    Diagnose (kick_gate_debug). Treibt den Detektor über den VOLLEN
    process()-Pfad (nicht nur _update_kalman_kick isoliert wie
    KickTriggerTests oben), damit HPSS tatsächlich befüllt wird - mit
    synthetischem Audio statt einer neuen Live-Aufnahme, reproduzierbar."""

    def setUp(self) -> None:
        self.sr = 48000
        self.bs = 512
        self.det = bd.BeatDetector(
            sample_rate=self.sr, block_size=self.bs,
            debug_triggers=True, kick_gate_debug=True,
        )
        self.clock = FakeClock()
        self.rng = np.random.default_rng(11)

    def _noise_block(self, amplitude: float = 0.02) -> np.ndarray:
        mono = (self.rng.standard_normal(self.bs) * amplitude).astype(np.float32)
        return np.column_stack([mono, mono])

    def _process(self, mono: np.ndarray) -> None:
        stereo = np.column_stack([mono, mono]).astype(np.float32)
        self.det.process(stereo)
        self.clock.advance(self.det._hop)

    def test_candidate_rows_recorded_with_all_offsets(self) -> None:
        """Ein scharfer, tiefer Ausreißer über einem verrauschten Boden
        muss (falls er die Basis-Schwelle überschreitet) als Kandidat mit
        allen 5 Offset-Werten protokolliert werden - unabhängig davon, ob
        er am Ende getriggert hat."""
        with patch.object(bd.time, "time", new=self.clock):
            # Aufwärmphase: genug Historie für Envelope- UND HPSS-Baseline
            # (1.0s-Fenster) - deutlich verrauschter Boden statt reiner
            # Stille, damit die HPSS-Trennung etwas Realistisches zum
            # Unterscheiden hat.
            for _ in range(120):
                self._process(self._noise_block())

            burst = (0.6 * np.sin(2 * np.pi * 60 * np.arange(self.bs) / self.sr)).astype(np.float32)
            burst += (self.rng.standard_normal(self.bs) * 0.02).astype(np.float32)
            self._process(burst)

            # Genug "Zukunft" für den 80ms-Offset + Marge (siehe
            # _process_kick_gate_pending) - weiter mit Boden-Rauschen.
            for _ in range(20):
                self._process(self._noise_block())

            rows = self.det.kick_gate_debug_rows

        self.assertGreater(len(rows), 0, "kein Kick-Kandidat wurde über die Basis-Schwelle erkannt - Burst-Amplitude ggf. zu niedrig")
        for row in rows:
            offsets = row["hpss_confirms_by_offset"]
            self.assertEqual(set(offsets.keys()), set(bd.BeatDetector._KICK_GATE_OFFSETS_MS))
            for offset_ms, data in offsets.items():
                self.assertIn("hpss_energy", data)
                self.assertIn("hpss_confirms", data)
                self.assertIn("dt_actual_ms", data)

    def test_gate_debug_disabled_by_default_produces_no_rows(self) -> None:
        """Default (kick_gate_debug=False) - keine Diagnosezeilen, kein
        zusätzlicher Zustand wird befüllt (Kosten-frei wenn ungenutzt)."""
        det = bd.BeatDetector(sample_rate=self.sr, block_size=self.bs, debug_triggers=True)
        clock = FakeClock()
        rng = np.random.default_rng(3)
        with patch.object(bd.time, "time", new=clock):
            for _ in range(60):
                mono = (rng.standard_normal(self.bs) * 0.02).astype(np.float32)
                stereo = np.column_stack([mono, mono]).astype(np.float32)
                det.process(stereo)
                clock.advance(det._hop)
            burst = (0.6 * np.sin(2 * np.pi * 60 * np.arange(self.bs) / self.sr)).astype(np.float32)
            stereo = np.column_stack([burst, burst]).astype(np.float32)
            det.process(stereo)
        self.assertEqual(det.kick_gate_debug_rows, [])
        self.assertEqual(len(det._hpss_gate_ring), 0)


class ProcessOrderingRegressionTests(unittest.TestCase):
    """Runde 6 - Regressionstests für die process()-Aufrufreihenfolge
    (_update_hpss_kick_analysis → _update_kalman_kick → _flux_and_triggers).
    Deckt genau die beiden Eigenschaften ab, die der Split garantieren muss:
    Kick-Gate sieht den AKTUELLEN Block-HPSS-Wert, Snare/Clap-kick_bleed
    sieht weiterhin den AKTUELLEN Block-Kickstatus - beides gleichzeitig,
    ohne dass eine Seite die andere veralten lässt."""

    def setUp(self) -> None:
        self.sr = 48000
        self.bs = 512
        self.clock = FakeClock()

    def test_kick_gate_uses_current_block_hpss_value_not_previous(self) -> None:
        """kick_debug["hpss_energy"] (während _update_kalman_kick erfasst)
        muss nach process() mit dem POST-Call-Zustand von
        self._hpss_kick_energy übereinstimmen - vor dem Fix (Aufruf-
        reihenfolge _update_kalman_kick VOR _spectral_flux_bands) hätte
        _update_kalman_kick noch den Wert des VORHERIGEN Blocks gesehen,
        während _spectral_flux_bands ihn direkt danach im selben process()-
        Aufruf mit dem aktuellen Wert überschrieben hätte - beide Werte
        wären dann unterschiedlich gewesen (siehe Diagnose-Bericht, dort
        empirisch mit einem Faktor >100 zwischen Block N und N+1
        beobachtet)."""
        det = bd.BeatDetector(sample_rate=self.sr, block_size=self.bs, debug_triggers=True)
        rng = np.random.default_rng(21)
        with patch.object(bd.time, "time", new=self.clock):
            for _ in range(120):
                mono = (rng.standard_normal(self.bs) * 0.02).astype(np.float32)
                stereo = np.column_stack([mono, mono]).astype(np.float32)
                det.process(stereo)
                self.clock.advance(det._hop)

            burst = (0.6 * np.sin(2 * np.pi * 60 * np.arange(self.bs) / self.sr)).astype(np.float32)
            burst += (rng.standard_normal(self.bs) * 0.02).astype(np.float32)
            stereo = np.column_stack([burst, burst]).astype(np.float32)
            det.process(stereo)

            self.assertEqual(
                det.kick_debug["hpss_energy"], det._hpss_kick_energy,
                "kick_debug wurde mit einem anderen hpss_energy-Wert befüllt als am Ende desselben "
                "process()-Aufrufs vorliegt - das Kick-Gate hat einen veralteten Block gesehen.",
            )

    def test_snare_kick_bleed_sees_current_block_impulse_flag(self) -> None:
        """_update_triggers (aufgerufen aus _flux_and_triggers) muss
        innerhalb DESSELBEN process()-Aufrufs den Wert sehen, den
        _update_kalman_kick soeben für _kalman_kick_is_impulse_now gesetzt
        hat - nicht den aus dem vorherigen Block. _update_kalman_kick wird
        hier gemockt (setzt das Flag deterministisch), um das gezielt zu
        prüfen, ohne auf zufällig realistische Kick-Schwellenwerte
        angewiesen zu sein (siehe A/B-Testreihe: reale Kicks triggern
        aktuell ohnehin praktisch nie, siehe Diagnose-Bericht)."""
        det = bd.BeatDetector(sample_rate=self.sr, block_size=self.bs, debug_triggers=True)
        rng = np.random.default_rng(5)
        with patch.object(bd.time, "time", new=self.clock):
            # Normale Aufwärmphase (echtes _update_kalman_kick), bis
            # _sample_buf voll ist und _prev_mag gesetzt ist - sonst würde
            # _flux_and_triggers noch früh zurückkehren und _update_triggers
            # nie aufrufen.
            for _ in range(10):
                mono = (rng.standard_normal(self.bs) * 0.02).astype(np.float32)
                stereo = np.column_stack([mono, mono]).astype(np.float32)
                det.process(stereo)
                self.clock.advance(det._hop)

            seen_flags = []
            original_update_triggers = det._update_triggers

            def spy_update_triggers(diff):
                seen_flags.append(det._kalman_kick_is_impulse_now)
                return original_update_triggers(diff)

            def fake_update_kalman_kick(mono):
                det._kalman_kick_is_impulse_now = True

            with patch.object(det, "_update_kalman_kick", side_effect=fake_update_kalman_kick), \
                 patch.object(det, "_update_triggers", side_effect=spy_update_triggers):
                mono = (rng.standard_normal(self.bs) * 0.02).astype(np.float32)
                stereo = np.column_stack([mono, mono]).astype(np.float32)
                det.process(stereo)

        self.assertEqual(seen_flags, [True], "_update_triggers hat nicht den im selben Block gesetzten Kickstatus gesehen")


class BridgePayloadTests(unittest.TestCase):
    """Prüft, dass clap_trigger tatsächlich bis in die HTTP-Payload der
    Sound-to-Light-Bridge durchgereicht wird (siehe soundlight_bridge.py
    _build_payload)."""

    class _FakeDetector:
        band_scores = (0.1, 0.2, 0.3, 0.4)
        overall_score = 0.5
        onset_scores = (0.6, 0.7, 0.8, 0.9)
        kick_trigger = False
        snare_trigger = False
        clap_trigger = True
        beat_phase = 0.25
        locked_bpm = 0.0
        bpm = 0.0

    def test_clap_trigger_reaches_payload(self) -> None:
        bridge = SoundlightBridge(self._FakeDetector(), url="http://example.invalid/levels")
        payload = json.loads(bridge._build_payload())
        self.assertIn("clapTrigger", payload)
        self.assertIs(payload["clapTrigger"], True)
        # Bestehende Felder müssen unverändert vorhanden bleiben (rückwärts-
        # kompatible Erweiterung, kein Ersatz-Eventformat).
        self.assertIn("kickTrigger", payload)
        self.assertIn("snareTrigger", payload)
        self.assertIn("kickHit", payload)


if __name__ == "__main__":
    unittest.main()
