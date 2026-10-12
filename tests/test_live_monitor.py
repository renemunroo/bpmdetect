"""Tests für das Runde-7-erweiterte CSV-Schema in live_monitor.py -
vollständige kick_debug-Felder + kick_event_source-Klassifikation
(accepted_candidate/kalman_prediction/none), fehlende Debugwerte ohne
Absturz, Rückwärtskompatibilität mit älteren CSV-Dateien.

Reine unittest-Standardbibliothek, siehe tests/test_triggers.py für die
gleiche Konvention. Testet live_monitor.py isoliert (kein echter Detector/
Audio nötig) mit einem kleinen Fake-Detector.
"""
import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from live_monitor import (  # noqa: E402
    LiveMonitor,
    _CSV_COLUMNS,
    _KICK_FIELD_COUNT,
    _classify_kick_event_source,
    _kick_csv_fields,
)


class KickCsvFieldsTests(unittest.TestCase):
    """1. CSV-Spalten und Feldzuordnung."""

    def test_full_kick_debug_maps_to_expected_columns(self) -> None:
        kd = {
            "envelope": 0.08, "threshold": 0.1,
            "baseline_mean": 0.02, "baseline_std": 0.03,
            "is_sharp": True, "sharpness_ratio": 6.5,
            "hpss_confirms": False, "hpss_energy": 71.9,
            "duration_above_threshold_s": 0.023, "decay_ratio": 0.4,
            "centroid_drift_hz": 12.3,
            "kalman_state": {"t_next": 1000.5, "period_s": 0.48, "initialized": True},
            "kalman_accepted": True,
            "reject_reason": "insufficient_hpss",
        }
        fields = _kick_csv_fields(kd, kick_trigger_active=False)
        self.assertEqual(len(fields), _KICK_FIELD_COUNT)
        # Reihenfolge muss exakt zu _CSV_COLUMNS passen (kick_envelope ist
        # die erste kick_*-Spalte, kick_event_source die letzte).
        kick_columns = [c for c in _CSV_COLUMNS if c.startswith("kick_")]
        self.assertEqual(len(kick_columns), _KICK_FIELD_COUNT)
        as_dict = dict(zip(kick_columns, fields))
        self.assertAlmostEqual(as_dict["kick_envelope"], 0.08)
        self.assertAlmostEqual(as_dict["kick_threshold"], 0.1)
        self.assertAlmostEqual(as_dict["kick_rise_ratio"], 0.8)
        self.assertEqual(as_dict["kick_is_sharp"], True)
        self.assertAlmostEqual(as_dict["kick_duration_above_threshold_s"], 0.023)
        self.assertAlmostEqual(as_dict["kick_decay_ratio"], 0.4)
        self.assertAlmostEqual(as_dict["kick_centroid_drift_hz"], 12.3)
        self.assertEqual(as_dict["kick_kalman_accepted"], True)
        self.assertEqual(as_dict["kick_debug_event_reason"], "insufficient_hpss")
        self.assertEqual(as_dict["kick_event_source"], "none")  # nicht akzeptiert, kein Trigger

        # kalman_state wird als JSON-String serialisiert, nicht als dict
        # (CSV-Zellen muessen flach sein) - muss rueckparsbar sein.
        import json
        parsed = json.loads(as_dict["kick_kalman_state"])
        self.assertEqual(parsed["period_s"], 0.48)


class KickEventSourceClassificationTests(unittest.TestCase):
    """2. Unterscheidung accepted_candidate / kalman_prediction / none."""

    def test_accepted_candidate_regardless_of_trigger(self) -> None:
        self.assertEqual(_classify_kick_event_source(None, kick_trigger_active=True), "accepted_candidate")
        self.assertEqual(_classify_kick_event_source(None, kick_trigger_active=False), "accepted_candidate")

    def test_kalman_prediction_when_rejected_but_triggering(self) -> None:
        """Exakt der reorder_fixed-Befund: alle 10 Live-Trigger hatten
        einen reject_reason, feuerten aber trotzdem (Kalman-Vorhersage)."""
        self.assertEqual(_classify_kick_event_source("insufficient_rise", kick_trigger_active=True), "kalman_prediction")
        self.assertEqual(_classify_kick_event_source("insufficient_hpss", kick_trigger_active=True), "kalman_prediction")

    def test_none_when_rejected_and_not_triggering(self) -> None:
        self.assertEqual(_classify_kick_event_source("insufficient_rise", kick_trigger_active=False), "none")
        self.assertEqual(_classify_kick_event_source("threshold_not_reached", kick_trigger_active=False), "none")

    def test_kick_csv_fields_uses_classification_consistently(self) -> None:
        """_kick_csv_fields muss dieselbe Klassifikation wie
        _classify_kick_event_source liefern - keine zweite, abweichende
        Implementierung."""
        kd_accepted = {"reject_reason": None}
        kd_predicted = {"reject_reason": "insufficient_sharpness"}
        kick_cols = [c for c in _CSV_COLUMNS if c.startswith("kick_")]
        idx = kick_cols.index("kick_event_source")
        self.assertEqual(_kick_csv_fields(kd_accepted, kick_trigger_active=False)[idx], "accepted_candidate")
        self.assertEqual(_kick_csv_fields(kd_predicted, kick_trigger_active=True)[idx], "kalman_prediction")
        self.assertEqual(_kick_csv_fields(kd_predicted, kick_trigger_active=False)[idx], "none")


class MissingDebugValueTests(unittest.TestCase):
    """3. Fehlende Debugwerte ohne Absturz (z.B. debug_triggers=False -> kick_debug={})."""

    def test_empty_kick_debug_dict_does_not_raise(self) -> None:
        """Leeres kd (z.B. debug_triggers=False) darf NICHT als
        accepted_candidate durchgehen, nur weil kd.get("reject_reason")
        genauso None liefert wie bei einem echten akzeptierten Kandidaten -
        siehe Docstring von _kick_csv_fields."""
        fields = _kick_csv_fields({}, kick_trigger_active=False)
        self.assertEqual(len(fields), _KICK_FIELD_COUNT)
        self.assertTrue(all(v is None for v in fields[:-1]))  # alles None außer event_source
        self.assertEqual(fields[-1], "none")

    def test_missing_kalman_state_serializes_as_none_not_crash(self) -> None:
        kd = {"envelope": 0.05, "reject_reason": "insufficient_rise"}  # kein kalman_state-Key
        fields = _kick_csv_fields(kd, kick_trigger_active=True)
        kick_cols = [c for c in _CSV_COLUMNS if c.startswith("kick_")]
        as_dict = dict(zip(kick_cols, fields))
        self.assertIsNone(as_dict["kick_kalman_state"])

    def test_write_status_and_event_survive_empty_kick_debug(self) -> None:
        """End-to-End: LiveMonitor darf mit kick_debug={} (debug_triggers=
        False) nicht abstürzen, weder bei _write_status noch bei
        _write_event."""

        class FakeDetector:
            kick_trigger = False
            snare_trigger = False
            clap_trigger = False
            bpm = 0.0
            locked_bpm = 0.0
            confidence = 0.0
            tempo_state = "searching"
            kick_debug = {}
            trigger_debug = {"snare": {}, "clap": {}}
            kick_gate_debug_rows = []  # sonst crasht der Hintergrund-Thread (siehe _loop) beim Drain

        with tempfile.TemporaryDirectory() as d:
            log_path = Path(d) / "test.log"
            csv_path = Path(d) / "test.csv"
            mon = LiveMonitor(FakeDetector(), log_path=str(log_path), csv_path=str(csv_path), keyboard_marker=False)
            mon.start()
            try:
                mon._write_status(1000.0)
                mon._write_event({"t": 1000.0, "kick": True, "snare": False, "clap": False,
                                   "kick_debug": {}, "trigger_debug": None})
            finally:
                mon.stop()
            self.assertTrue(csv_path.exists())


class CsvBackwardCompatibilityTests(unittest.TestCase):
    """4. Rückwärtskompatibilität, falls ältere CSV-Daten geladen werden."""

    def test_old_narrower_header_is_not_appended_to(self) -> None:
        """Eine bestehende CSV mit dem ALTEN (schmaleren) Schema darf
        nicht mit neuen, breiteren Zeilen verschmutzt werden - LiveMonitor
        muss auf einen anderen Dateinamen ausweichen statt die Spalten zu
        verschieben."""
        old_columns = [
            "t", "iso_time", "type",
            "kick_envelope", "kick_threshold", "kick_baseline_mean", "kick_baseline_std",
            "kick_rise_ratio", "kick_is_sharp", "kick_sharpness_ratio",
            "kick_hpss_confirms", "kick_hpss_energy",
            "confidence", "bassline_score", "reject_reason",
            "bpm", "locked_bpm", "lock_confidence", "level_dbfs",
        ]
        with tempfile.TemporaryDirectory() as d:
            old_csv = Path(d) / "old.csv"
            with open(old_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(old_columns)
                w.writerow([1000.0, "10:00:00", "status"] + [""] * (len(old_columns) - 3))

            resolved = LiveMonitor._resolve_compatible_csv_path(old_csv)
            self.assertNotEqual(resolved, old_csv, "hätte auf einen neuen Dateinamen ausweichen müssen")

            # Alte Datei bleibt unangetastet (noch die alte, schmalere Kopfzeile).
            with open(old_csv, newline="") as f:
                header = next(csv.reader(f))
            self.assertEqual(header, old_columns)

    def test_matching_header_is_reused(self) -> None:
        """Eine bereits im AKTUELLEN Schema geschriebene CSV wird normal
        weiter angehängt (kein unnötiges Ausweichen)."""
        with tempfile.TemporaryDirectory() as d:
            existing = Path(d) / "current.csv"
            with open(existing, "w", newline="") as f:
                csv.writer(f).writerow(_CSV_COLUMNS)
            resolved = LiveMonitor._resolve_compatible_csv_path(existing)
            self.assertEqual(resolved, existing)

    def test_no_existing_file_is_unaffected(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            fresh = Path(d) / "fresh.csv"
            self.assertEqual(LiveMonitor._resolve_compatible_csv_path(fresh), fresh)


if __name__ == "__main__":
    unittest.main()
