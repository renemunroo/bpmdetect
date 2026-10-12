"""Tests für tempo_counter.TempoCounter - die eigenständige BPM-Lock/Relock-
Zustandsmaschine, 2026-09-23 aus BeatDetector extrahiert.

Die meisten Tests rufen _apply_lock() direkt auf (white-box) statt über
push_oss() mit einem synthetischen Audio-Signal - das macht die Szenarien
exakt und deterministisch reproduzierbar (u.a. der Live-Club-Bug: ein
Lock bei 104.2 BPM, der trotz durchgehend korrekt gemessenem neuem Tempo
97.0 BPM nie relockte). Ein zusätzlicher Ende-zu-Ende-Test über push_oss()
mit einem echten periodischen Signal prüft, dass die komplette Pipeline
(Autokorrelation -> Lock) tatsächlich verkabelt ist.
"""
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tempo_counter import TempoCounter


def _locked_counter(locked_bpm: float, **kwargs) -> TempoCounter:
    """Ein TempoCounter, der bereits im "locked"-Zustand auf locked_bpm
    steht - Ausgangspunkt für die meisten Zustandsmaschinen-Tests."""
    tc = TempoCounter(hop_s=0.02, **kwargs)
    tc._apply_lock(locked_bpm, conf=1.0, now=0.0)
    assert tc.state == "locked"
    assert tc.locked_bpm == locked_bpm
    return tc


class InitialLockTests(unittest.TestCase):
    def test_searching_to_locked_on_sufficient_confidence(self) -> None:
        tc = TempoCounter(hop_s=0.02, lock_confidence_min=0.3)
        self.assertEqual(tc.state, "searching")
        tc._apply_lock(120.0, conf=0.5, now=1.0)
        self.assertEqual(tc.state, "locked")
        self.assertEqual(tc.locked_bpm, 120.0)

    def test_searching_stays_searching_below_confidence_threshold(self) -> None:
        tc = TempoCounter(hop_s=0.02, lock_confidence_min=0.3)
        tc._apply_lock(120.0, conf=0.1, now=1.0)
        self.assertEqual(tc.state, "searching")
        self.assertEqual(tc.locked_bpm, 0.0)


class SmallJumpAbsorbedTests(unittest.TestCase):
    """Sprünge <= max_jump_bpm gelten als Messrauschen - locked_bpm bleibt
    stehen, keine Relock-Kandidatur wird begonnen."""

    def test_small_jump_within_max_jump_bpm_is_ignored(self) -> None:
        tc = _locked_counter(120.0, max_jump_bpm=4.0)
        tc._apply_lock(122.0, conf=1.0, now=1.0)  # Sprung 2.0 <= 4.0
        self.assertEqual(tc.state, "locked")
        self.assertEqual(tc.locked_bpm, 120.0, "kleine Sprünge dürfen locked_bpm nicht verändern")
        self.assertEqual(tc.relock_candidate, 0.0, "keine Relock-Kandidatur bei kleinem Sprung")


class RealBugRegressionTests(unittest.TestCase):
    """Reproduziert exakt den Live-Club-Bug vom 2026-09-23: ein Lock bei
    104.2 BPM, während corrected_bpm durchgehend 97.0 BPM zeigt (Differenz
    7.2 BPM). Mit der alten max_jump_bpm=10.0 wurde das für immer als
    Messrauschen abgetan (kleiner Sprung -> "pass", niemals ein Relock-
    Versuch). Mit dem neuen Default (4.0) und genug konsistenten Fenstern
    muss es jetzt tatsächlich relocken."""

    def test_sustained_different_tempo_eventually_relocks(self) -> None:
        tc = _locked_counter(104.2, max_jump_bpm=4.0, relock_confidence_min=0.5, relock_windows=3)
        now = 0.0
        for _ in range(3):
            now += 1.0
            tc._apply_lock(97.0, conf=0.8, now=now)
        self.assertEqual(tc.state, "locked")
        self.assertEqual(tc.locked_bpm, 97.0, "nach genug konsistenten Fenstern muss der echte neue Wert übernommen werden")

    def test_old_threshold_would_have_stayed_stuck_forever(self) -> None:
        """Dokumentiert den Bug selbst: mit der alten Schwelle (10.0) wird
        derselbe Sprung (7.2 BPM) als "klein" eingestuft und NIE verfolgt,
        egal wie viele Fenster vergehen."""
        tc = _locked_counter(104.2, max_jump_bpm=10.0, relock_confidence_min=0.5, relock_windows=3)
        now = 0.0
        for _ in range(50):
            now += 1.0
            tc._apply_lock(97.0, conf=0.8, now=now)
        self.assertEqual(tc.locked_bpm, 104.2, "mit der alten Schwelle bleibt der Bug reproduzierbar - Regressionsanker")
        self.assertEqual(tc.relock_candidate, 0.0)


class RelockConfidenceGateTests(unittest.TestCase):
    def test_relock_not_attempted_below_relock_confidence_min(self) -> None:
        tc = _locked_counter(120.0, max_jump_bpm=4.0, relock_confidence_min=0.5)
        # Sprung groß genug (>max_jump_bpm), aber confidence zu niedrig
        # für einen Relock-Versuch (nicht dasselbe wie lock_confidence_min!).
        tc._apply_lock(97.0, conf=0.4, now=1.0)
        self.assertEqual(tc.state, "locked")
        self.assertEqual(tc.locked_bpm, 120.0)
        self.assertEqual(tc.relock_candidate, 0.0)


class HardBlockJumpTests(unittest.TestCase):
    """hard_block_jump_bpm: zu wilde Sprünge werden nie verfolgt - selbst
    mit hoher confidence und beliebig vielen Fenstern. Im alten BeatDetector
    wurde dieser Parameter aus club.yaml gelesen, aber nie ausgewertet
    (totes Config-Feld) - hier jetzt tatsächlich wirksam."""

    def test_jump_beyond_hard_block_is_never_followed(self) -> None:
        tc = _locked_counter(100.0, max_jump_bpm=4.0, hard_block_jump_bpm=15.0, relock_confidence_min=0.5, relock_windows=3)
        now = 0.0
        for _ in range(10):
            now += 1.0
            tc._apply_lock(120.0, conf=1.0, now=now)  # Sprung 20.0 > hard_block 15.0
        self.assertEqual(tc.state, "locked")
        self.assertEqual(tc.locked_bpm, 100.0)
        self.assertEqual(tc.relock_candidate, 0.0)


class RelockDriftResetTests(unittest.TestCase):
    def test_drifting_candidate_aborts_then_a_fresh_candidate_starts_clean(self) -> None:
        tc = _locked_counter(100.0, max_jump_bpm=4.0, relock_confidence_min=0.5, relock_windows=3, hard_block_jump_bpm=50.0)
        now = 0.0
        now += 1.0
        tc._apply_lock(90.0, conf=0.8, now=now)  # Kandidat A, state -> relocking
        self.assertEqual(tc.state, "relocking")
        self.assertEqual(tc.relock_progress[0], 1)

        now += 1.0
        tc._apply_lock(80.0, conf=0.8, now=now)  # driftet zu weit von Kandidat A weg (relocking-Zweig) -> Abbruch zurück auf "locked"
        self.assertEqual(tc.state, "locked", "ein während relocking abweichender Wert muss den Versuch abbrechen, nicht einen neuen Kandidaten starten")
        self.assertEqual(tc.relock_candidate, 0.0)
        self.assertEqual(tc.locked_bpm, 100.0, "während des abgebrochenen Relocks darf sich locked_bpm nicht ändern")

        now += 1.0
        tc._apply_lock(80.0, conf=0.8, now=now)  # erneuter Versuch, jetzt wieder aus "locked" heraus -> sauberer neuer Kandidat
        self.assertEqual(tc.state, "relocking")
        self.assertEqual(tc.relock_candidate, 80.0)
        self.assertEqual(tc.relock_progress[0], 1, "der neue Versuch nach einem Abbruch muss bei 1 starten, keine Leichen vom vorigen Versuch")


class LargeJumpHoldTests(unittest.TestCase):
    """Sehr große Sprünge (> large_jump_bpm) brauchen zusätzlich
    large_jump_hold_s Sekunden Mindestdauer, bevor sie übernommen werden -
    selbst wenn relock_windows schon früher erreicht ist."""

    def test_large_jump_requires_minimum_hold_time(self) -> None:
        tc = _locked_counter(100.0, max_jump_bpm=4.0, relock_confidence_min=0.5,
                              relock_windows=2, large_jump_bpm=15.0, large_jump_hold_s=5.0,
                              hard_block_jump_bpm=50.0)  # sonst würde der 30-BPM-Sprung schon vom Hard-Block abgefangen, bevor large_jump_hold überhaupt geprüft wird
        now = 0.0
        now += 0.1
        tc._apply_lock(130.0, conf=0.9, now=now)  # Sprung 30 > large_jump_bpm(15)
        now += 0.1
        tc._apply_lock(130.0, conf=0.9, now=now)  # relock_progress erreicht 2, aber hold_s noch nicht um
        self.assertEqual(tc.state, "relocking", "trotz genug Fenstern darf ein großer Sprung ohne Mindestdauer nicht übernommen werden")
        self.assertEqual(tc.locked_bpm, 100.0)

        now += 5.0
        tc._apply_lock(130.0, conf=0.9, now=now)  # jetzt ist genug Zeit vergangen
        self.assertEqual(tc.state, "locked")
        self.assertEqual(tc.locked_bpm, 130.0)


class RelockAbortTests(unittest.TestCase):
    def test_relock_aborts_on_confidence_drop(self) -> None:
        tc = _locked_counter(100.0, max_jump_bpm=4.0, relock_confidence_min=0.5, relock_windows=5)
        now = 1.0
        tc._apply_lock(90.0, conf=0.8, now=now)
        self.assertEqual(tc.state, "relocking")
        now += 1.0
        tc._apply_lock(90.0, conf=0.1, now=now)  # Signal weg während Relock
        self.assertEqual(tc.state, "locked", "Confidence-Einbruch während Relock muss abbrechen, nicht crashen")
        self.assertEqual(tc.locked_bpm, 100.0, "abgebrochener Relock darf den alten Lock nicht verändern")
        self.assertEqual(tc.relock_candidate, 0.0)

    def test_relock_aborts_when_candidate_drifts_away(self) -> None:
        tc = _locked_counter(100.0, max_jump_bpm=4.0, relock_confidence_min=0.5, relock_windows=5)
        now = 1.0
        tc._apply_lock(90.0, conf=0.8, now=now)
        self.assertEqual(tc.state, "relocking")
        now += 1.0
        tc._apply_lock(70.0, conf=0.8, now=now)  # während relocking, driftet weit weg vom Kandidaten (90)
        self.assertEqual(tc.state, "locked", "ein wegdriftender Kandidat während relocking muss abbrechen, nicht hängen bleiben")
        self.assertEqual(tc.locked_bpm, 100.0)


class LowConfidenceToSearchingTests(unittest.TestCase):
    def test_locked_falls_back_to_searching_after_sustained_low_confidence(self) -> None:
        tc = _locked_counter(100.0, hold_seconds=2.0)
        tc._apply_lock(100.0, conf=0.05, now=1.0)  # unter lock_confidence_min(0.3)
        self.assertEqual(tc.state, "locked", "kurzer Low-Confidence-Ausschlag darf noch nicht auf searching fallen")
        tc._apply_lock(100.0, conf=0.05, now=4.0)  # 3s > hold_seconds(2.0)
        self.assertEqual(tc.state, "searching")

    def test_locked_bpm_survives_brief_low_confidence_dip(self) -> None:
        tc = _locked_counter(100.0, hold_seconds=8.0)
        tc._apply_lock(100.0, conf=0.05, now=1.0)
        tc._apply_lock(100.0, conf=0.8, now=1.5)  # erholt sich rechtzeitig
        self.assertEqual(tc.state, "locked")
        self.assertEqual(tc.locked_bpm, 100.0)


class EndToEndPushOssTests(unittest.TestCase):
    """Prüft, dass push_oss() tatsächlich bis zur Autokorrelation und dem
    Lock durchverkabelt ist - nicht nur die isolierten Zustandsübergänge."""

    @staticmethod
    def _synthetic_pulse_train(bpm: float, hop_s: float, n_blocks: int, seed: int = 0) -> list:
        rnd = random.Random(seed)
        period_blocks = 60.0 / bpm / hop_s
        vals, next_pulse = [], period_blocks
        for i in range(n_blocks):
            v = 0.02 + rnd.uniform(0.0, 0.02)
            if i >= next_pulse - 0.5:
                v = 1.0
                next_pulse += period_blocks
            vals.append(v)
        return vals

    def test_periodic_signal_locks_within_generous_bpm_range(self) -> None:
        hop_s = 2048 / 48000
        tc = TempoCounter(hop_s=hop_s, update_every_s=1.0, hold_seconds=8.0)
        now = 0.0
        for v in self._synthetic_pulse_train(120.0, hop_s, n_blocks=400):
            now += hop_s
            tc.push_oss(v, now)
        self.assertEqual(tc.state, "locked")
        # Block-Lag-Quantisierung erlaubt keine exakte Übereinstimmung -
        # großzügige Toleranz, es geht hier nur darum, dass die Pipeline
        # überhaupt etwas Plausibles liefert, nicht um Präzisions-Tuning.
        self.assertAlmostEqual(tc.locked_bpm, 120.0, delta=15.0)

    def test_insufficient_data_does_not_crash_or_lock(self) -> None:
        tc = TempoCounter(hop_s=0.02)
        tc.push_oss(1.0, now=0.02)
        self.assertEqual(tc.state, "searching")
        self.assertEqual(tc.locked_bpm, 0.0)


class ThreadSafetyTests(unittest.TestCase):
    """Properties müssen aus einem anderen Thread lesbar sein, während
    push_oss() aus dem Audio-Thread läuft (siehe soundlight_bridge.py, das
    locked_bpm/bpm aus seinem eigenen Sende-Thread liest) - kein Crash,
    keine Exception, egal wie die Aufrufe verschachtelt sind."""

    def test_concurrent_push_and_read_does_not_raise(self) -> None:
        import threading

        tc = TempoCounter(hop_s=0.02, update_every_s=0.1)
        errors = []

        def writer() -> None:
            try:
                now = 0.0
                for i in range(500):
                    now += 0.02
                    tc.push_oss(0.5 + (i % 7) * 0.1, now)
            except Exception as exc:  # pragma: no cover - Diagnose bei Fehlschlag
                errors.append(exc)

        def reader() -> None:
            try:
                for _ in range(500):
                    _ = (tc.bpm, tc.locked_bpm, tc.state, tc.confidence, tc.candidates, tc.relock_progress)
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
