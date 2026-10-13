"""Tests für link_bridge._phase_correction_target() - die reine Entscheidungs-
logik der Beat-Phasenkorrektur (Bugfix 2026-10-13: "BPM Flash" auf den
Kugeln lief mit der Zeit auseinander, weil set_tempo allein nur die
Taktrate korrigiert, nie die Beat-Phase selbst - Link lief seitdem rein
rechnerisch weiter, jede noch so kleine Resttempo-Ungenauigkeit (unter der
Tempo-Hysterese) sammelte sich unkorrigiert auf).

Testet NUR die reine Funktion, kein echter Subprocess/keine echte Link-
Session (gleiches Prinzip wie test_tempo_counter.py) - LinkBridge selbst
darf in einem automatisierten Test nicht konstruiert werden, siehe
Kommentar oben in link_bridge.py.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from link_bridge import _phase_correction_target


DEADBAND = 0.05
INTERVAL = 1.0  # Sekunden


class DeadbandTests(unittest.TestCase):
    def test_small_offset_within_deadband_is_not_corrected(self) -> None:
        # 0.03 Beat Versatz - normales Mess-Rauschen, soll NICHT korrigiert werden.
        result = _phase_correction_target(0.03, DEADBAND, now=10.0, last_correction_t=0.0, correction_interval=INTERVAL, current_beat=40.0)
        self.assertIsNone(result)

    def test_offset_exactly_at_deadband_is_not_corrected(self) -> None:
        result = _phase_correction_target(DEADBAND, DEADBAND, now=10.0, last_correction_t=0.0, correction_interval=INTERVAL, current_beat=40.0)
        self.assertIsNone(result)

    def test_offset_beyond_deadband_is_corrected(self) -> None:
        result = _phase_correction_target(0.12, DEADBAND, now=10.0, last_correction_t=0.0, correction_interval=INTERVAL, current_beat=40.0)
        self.assertIsNotNone(result)

    def test_negative_offset_beyond_deadband_is_corrected(self) -> None:
        # Link hinkt nach (negativer Offset) - muss genauso korrigiert werden wie "Link ist voraus".
        result = _phase_correction_target(-0.12, DEADBAND, now=10.0, last_correction_t=0.0, correction_interval=INTERVAL, current_beat=40.0)
        self.assertIsNotNone(result)


class TargetBeatTests(unittest.TestCase):
    def test_target_beat_pulls_link_back_toward_detected_phase(self) -> None:
        # Link ist 0.12 Beat VORAUS (positiver Offset) -> Ziel-Beat muss
        # entsprechend ZURÜCK korrigiert werden (current_beat - offset).
        target = _phase_correction_target(0.12, DEADBAND, now=10.0, last_correction_t=0.0, correction_interval=INTERVAL, current_beat=40.0)
        self.assertAlmostEqual(target, 40.0 - 0.12)

    def test_target_beat_pushes_link_forward_when_lagging(self) -> None:
        # Link hinkt 0.12 Beat NACH (negativer Offset) -> Ziel-Beat muss nach VORN korrigiert werden.
        target = _phase_correction_target(-0.12, DEADBAND, now=10.0, last_correction_t=0.0, correction_interval=INTERVAL, current_beat=40.0)
        self.assertAlmostEqual(target, 40.0 - (-0.12))
        self.assertGreater(target, 40.0)


class RateLimitTests(unittest.TestCase):
    def test_no_correction_before_interval_elapsed_since_last_one(self) -> None:
        # Letzte Korrektur vor 0.5s, Mindestabstand 1.0s -> noch nicht wieder korrigieren,
        # obwohl der Offset weit außerhalb der Totzone liegt.
        result = _phase_correction_target(0.3, DEADBAND, now=10.5, last_correction_t=10.0, correction_interval=INTERVAL, current_beat=40.0)
        self.assertIsNone(result)

    def test_correction_allowed_once_interval_elapsed(self) -> None:
        result = _phase_correction_target(0.3, DEADBAND, now=11.0, last_correction_t=10.0, correction_interval=INTERVAL, current_beat=40.0)
        self.assertIsNotNone(result)

    def test_first_ever_correction_not_blocked_by_zero_last_correction_t(self) -> None:
        # last_correction_t=0.0 ist der Initialwert vor der allerersten
        # Korrektur - `now` kommt in der echten Nutzung IMMER aus time.time()
        # (Unix-Epoch, Milliarden Sekunden), liegt also immer weit mehr als
        # `correction_interval` nach 0.0 - die erste Korrektur darf also nie
        # blockiert werden.
        result = _phase_correction_target(0.3, DEADBAND, now=1_760_000_000.0, last_correction_t=0.0, correction_interval=INTERVAL, current_beat=40.0)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
