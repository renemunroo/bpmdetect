"""Optionales Live-Monitoring/Logging für Kick-/Snare-/Clap-Debugwerte im
Live-Betrieb (echtes Audio vom Pioneer REV7 via BlackHole/sounddevice,
siehe tools/analyze_wav.py für die Offline-Variante mit WAV-Dateien).

Ändert NICHTS an der Erkennungslogik in beat_detector.py - liest nur
bereits vorhandene, Lock-geschützte Properties (kick_trigger/snare_trigger/
clap_trigger, kick_debug, trigger_debug, bpm, locked_bpm, confidence,
tempo_state) und den Pegel aus capture.LevelMeter.

Architektur, damit der Echtzeit-Audio-Callback NIE blockiert:
  - on_block() wird vom Audio-Callback-Thread nach jedem
    detector.process(block) aufgerufen (in run_bpm's on_block-Closure,
    genau wie das bestehende display.update_meter()). Macht NUR ein paar
    billige Property-Reads + ein NICHT-blockierendes queue.put_nowait -
    kein Datei-/Konsolen-I/O auf dem Audio-Thread.
  - Ein eigener Hintergrund-Thread (_loop) leert die Queue und macht ALLE
    Datei-I/O dort, rate-begrenzt auf höchstens refresh_hz (Default 5/s).
  - Schreibt in eine Log-Datei statt stdout, damit es nicht mit dem
    ANSI-Cursor-Redraw von BpmDisplay im selben Terminal kollidiert -
    `tail -f logs/live_monitor.log` in einem zweiten Terminal/Tab
    mitlesen (siehe Live-Testplan).
"""
import csv as _csv
import json
import os
import queue
import signal
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

# Runde 4 (A/B-Test kalman_kick_std_factor), erweitert Runde 7 (vollständige
# kick_debug-Felder + Event-Herkunft): "type" ist "status" (periodischer
# Snapshot, siehe _write_status) oder "kick"/"snare"/"clap" (diskretes
# Trigger-Event, siehe _write_event). Die kick_*-Spalten sind bei Snare-/
# Clap-Events leer, bei "status"- und "kick"-Zeilen befüllt - reicht für die
# Envelope-/Schwellen-Statistik über einen ganzen Testlauf, auch wenn (wie
# im Ausgangsbefund) kein einziger Kick jemals triggert.
#
# Runde 7: kick_event_source unterscheidet, WARUM eine Zeile einen Kick
# zeigt - ursprünglich war das relevant, weil detector.kick_trigger auf dem
# gelernten Kalman-Beat-Zeitplan feuerte und so auch dann True sein konnte,
# wenn der Frontend-Kandidat DIESES Blocks abgelehnt wurde ("kalman_
# prediction"). Seit 2026-10-14 (Nutzer-Anforderung "Trigger auf jede Kick")
# feuert kick_trigger nur noch direkt auf reject_reason is None - der
# "kalman_prediction"-Fall unten kann dadurch faktisch nicht mehr auftreten,
# die Unterscheidung bleibt aber als reine, harmlose Klassifikation aus
# bereits vorhandenen Werten stehen (kein Aufräumen ohne Auftrag).
_CSV_COLUMNS = [
    "t", "iso_time", "type",
    "kick_envelope", "kick_threshold", "kick_baseline_mean", "kick_baseline_std",
    "kick_rise_ratio", "kick_is_sharp", "kick_sharpness_ratio",
    "kick_hpss_confirms", "kick_hpss_energy",
    "kick_duration_above_threshold_s", "kick_decay_ratio", "kick_centroid_drift_hz",
    "kick_kalman_state", "kick_kalman_accepted",
    "kick_debug_event_reason", "kick_event_source",
    "confidence", "bassline_score", "reject_reason",
    "bpm", "locked_bpm", "lock_confidence", "level_dbfs",
]
_KICK_FIELD_COUNT = 16  # Anzahl der kick_*-Spalten oben (envelope...kick_event_source) - für Snare/Clap-Platzhalter


def _classify_kick_event_source(reject_reason, kick_trigger_active: bool) -> str:
    """accepted_candidate = _update_kalman_kick hat DIESEN Block selbst als
    Impuls akzeptiert (reject_reason is None), unabhängig davon, ob der
    Kalman-Zeitplan im selben Block schon feuert. kalman_prediction = der
    Frontend-Kandidat wurde abgelehnt, aber detector.kick_trigger feuert
    trotzdem (Vorhersage auf Basis früher akzeptierter Impulse). none =
    weder noch (gewöhnlicher Hintergrund-Block)."""
    if reject_reason is None:
        return "accepted_candidate"
    if kick_trigger_active:
        return "kalman_prediction"
    return "none"


def _kick_csv_fields(kd: dict, kick_trigger_active: bool = False) -> list:
    """Extrahiert die kick_*-Spaltenwerte aus einem kick_debug-Dict in
    fester Reihenfolge (siehe _CSV_COLUMNS) - gemeinsam genutzt von
    Status- und Kick-Event-Zeilen, damit beide vergleichbar bleiben. Fehlt
    ein Feld (z.B. kd={} bei debug_triggers=False), liefert .get(...) None
    statt einer Ausnahme - siehe test_live_monitor.py.

    Leeres kd (keine Debugdaten vorhanden) wird explizit VOR der
    reject_reason-Auswertung abgefangen: kd.get("reject_reason") würde bei
    einem leeren dict genauso None liefern wie bei einem ECHTEN akzeptierten
    Kandidaten (reject_reason is None) - ohne diese Unterscheidung würde
    "keine Daten" fälschlich als "accepted_candidate" klassifiziert."""
    if not kd:
        return [None] * (_KICK_FIELD_COUNT - 1) + ["none"]
    envelope = kd.get("envelope")
    threshold = kd.get("threshold")
    rise_ratio = (envelope / threshold) if envelope is not None and threshold else None
    kalman_state = kd.get("kalman_state")
    reject_reason = kd.get("reject_reason")
    return [
        envelope, threshold, kd.get("baseline_mean"), kd.get("baseline_std"),
        rise_ratio, kd.get("is_sharp"), kd.get("sharpness_ratio"),
        kd.get("hpss_confirms"), kd.get("hpss_energy"),
        kd.get("duration_above_threshold_s"), kd.get("decay_ratio"), kd.get("centroid_drift_hz"),
        json.dumps(kalman_state) if kalman_state is not None else None,
        kd.get("kalman_accepted"),
        reject_reason, _classify_kick_event_source(reject_reason, kick_trigger_active),
    ]


class LiveMonitor:
    def __init__(
        self,
        detector,
        level_meter=None,
        refresh_hz: float = 5.0,
        history_s: float = 60.0,
        log_path: str = "logs/live_monitor.log",
        csv_path: Optional[str] = None,
        keyboard_marker: bool = True,
    ) -> None:
        self._detector = detector
        self._level_meter = level_meter
        self._interval = 1.0 / max(refresh_hz, 0.1)
        self._history_s = history_s
        self._queue: "queue.Queue" = queue.Queue(maxsize=1000)
        self._history: deque = deque()
        self._hist_lock = threading.Lock()  # nur für _history, unabhängig vom BeatDetector-Lock
        self._running = False
        self._thread = None
        self._kbd_thread = None
        self._keyboard_marker = keyboard_marker

        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = None

        self._csv_path = Path(csv_path) if csv_path else None
        self._csv_file = None
        self._csv_writer = None

    # ── vom Audio-Callback-Thread aufgerufen ─────────────────────────────
    def on_block(self) -> None:
        """Nach detector.process(block) aufrufen. Nur billige Property-
        Reads + ein nicht-blockierendes Queue-Put - kein I/O, damit die
        Echtzeitverarbeitung nie wartet."""
        det = self._detector
        kick, snare, clap = det.kick_trigger, det.snare_trigger, det.clap_trigger
        if not (kick or snare or clap):
            return
        evt = {
            "t": time.time(),
            "kick": kick, "snare": snare, "clap": clap,
            "kick_debug": dict(det.kick_debug) if kick else None,
            "trigger_debug": det.trigger_debug if (snare or clap) else None,
        }
        try:
            self._queue.put_nowait(evt)
        except queue.Full:
            pass  # Consumer hängt hinterher - Event verwerfen statt den Audio-Thread zu blockieren

    # ── Lifecycle ─────────────────────────────────────────────────────────
    def start(self) -> None:
        self._log_file = open(self._log_path, "a", buffering=1)
        if self._csv_path:
            self._csv_path = self._resolve_compatible_csv_path(self._csv_path)
            write_header = not self._csv_path.exists()
            self._csv_file = open(self._csv_path, "a", newline="")
            self._csv_writer = _csv.writer(self._csv_file)
            if write_header:
                self._csv_writer.writerow(_CSV_COLUMNS)

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="live-monitor")
        self._thread.start()

        try:
            signal.signal(signal.SIGUSR1, lambda sig, frame: self.mark("signal"))
        except (ValueError, AttributeError):
            pass  # z.B. nicht im Haupt-Thread, oder Plattform ohne SIGUSR1

        if self._keyboard_marker and sys.stdin.isatty():
            self._kbd_thread = threading.Thread(target=self._keyboard_loop, daemon=True, name="live-monitor-kbd")
            self._kbd_thread.start()

        print(f"[live-monitor] aktiv -> {self._log_path}" + (f"  |  CSV -> {self._csv_path}" if self._csv_path else ""))
        print(f"[live-monitor] Marker setzen: 'm' + Enter in diesem Terminal, oder `kill -USR1 {os.getpid()}` von woanders")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._log_file:
            self._log_file.close()
        if self._csv_file:
            self._csv_file.close()

    @staticmethod
    def _resolve_compatible_csv_path(path: Path) -> Path:
        """Runde 7: verhindert, dass an eine bestehende CSV mit ÄLTEREM
        (schmalerem) Spaltenschema neue, breitere Zeilen angehängt werden -
        das würde die Spalten ab dort verschieben. Prüft nur die
        Kopfzeile, weicht bei Mismatch auf einen neuen Dateinamen aus statt
        die bestehende Datei zu verändern oder abzustürzen."""
        if not path.exists():
            return path
        try:
            with open(path, newline="") as f:
                header = next(_csv.reader(f), None)
        except OSError:
            return path
        if header == _CSV_COLUMNS:
            return path
        alt = path.with_name(f"{path.stem}_v{len(_CSV_COLUMNS)}col{path.suffix}")
        print(
            f"[live-monitor] '{path}' hat ein anderes CSV-Spaltenschema "
            f"({len(header) if header else 0} statt {len(_CSV_COLUMNS)} Spalten) - "
            f"weiche auf '{alt}' aus, um die bestehende Datei nicht zu verändern."
        )
        return alt

    def mark(self, label: str = "") -> None:
        """Speichert die letzten history_s Sekunden Debugdaten als
        JSON-Snapshot - für einen kurzen markierten Testabschnitt
        (siehe Live-Testplan). Auslösbar per Tastatur ('m'+Enter) oder
        Signal (kill -USR1 <pid>, funktioniert auch headless/über SSH)."""
        with self._hist_lock:
            snapshot = list(self._history)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = self._log_path.parent / f"marker_{ts}.json"
        with open(out_path, "w") as f:
            json.dump({"label": label, "marked_at": time.time(), "events": snapshot}, f, indent=2, default=str)
        msg = f"[live-monitor] Marker '{label or ts}' gespeichert -> {out_path} ({len(snapshot)} Events der letzten {self._history_s:.0f}s)"
        print(f"\n{msg}")
        if self._log_file:
            self._log_file.write(msg + "\n")

    def _keyboard_loop(self) -> None:
        while self._running:
            try:
                line = input()
            except (EOFError, RuntimeError):
                return
            cmd = line.strip().lower()
            if cmd in ("m", "mark"):
                self.mark("keyboard")

    # ── Hintergrund-Thread: alle Datei-I/O passiert hier, nie im Audio-Callback ──
    def _loop(self) -> None:
        last_status = 0.0
        while self._running:
            drained = []
            while True:
                try:
                    drained.append(self._queue.get_nowait())
                except queue.Empty:
                    break

            now = time.time()
            with self._hist_lock:
                for evt in drained:
                    self._history.append(evt)
                cutoff = now - self._history_s
                while self._history and self._history[0]["t"] < cutoff:
                    self._history.popleft()

            for evt in drained:
                self._write_event(evt)

            if now - last_status >= self._interval:
                self._write_status(now)
                last_status = now

            # Runde 6: kick_gate_debug_rows (Sharpness/HPSS-Timing-Diagnose,
            # siehe beat_detector.py) leeren, falls aktiviert - drain-on-read,
            # läuft im selben Hintergrund-Thread wie alles andere hier, nie
            # im Audio-Callback. Eine JSON-Zeile pro Kandidat, damit die
            # 0/20/40/60/80ms-Offsets vollständig und maschinell auswertbar
            # im Log landen, ohne die CSV-Spaltenzahl zu verändern.
            gate_rows = self._detector.kick_gate_debug_rows
            for row in gate_rows:
                ts = time.strftime("%H:%M:%S", time.localtime(row["candidate_t"]))
                self._log_file.write(f"{ts}  GATE_DIAG  {json.dumps(row, default=str)}\n")

            time.sleep(0.02)

    def _current_level(self) -> Optional[float]:
        return self._level_meter.current_dbfs() if self._level_meter else None

    def _write_event(self, evt: dict) -> None:
        ts = time.strftime("%H:%M:%S", time.localtime(evt["t"])) + f".{int(evt['t'] % 1 * 1000):03d}"
        parts = [ts]
        if evt["kick"]:
            kd = evt["kick_debug"] or {}
            parts.append(
                f"KICK  conf={kd.get('confidence', 1.0):.2f}  bassline={kd.get('bassline_score', 0.0):.2f}  "
                f"reason={kd.get('reject_reason')}"
            )
        if evt["snare"]:
            sd_ = (evt["trigger_debug"] or {}).get("snare", {})
            parts.append(f"SNARE conf={sd_.get('confidence', 0.0):.2f}")
        if evt["clap"]:
            cd = (evt["trigger_debug"] or {}).get("clap", {})
            parts.append(f"CLAP  conf={cd.get('confidence', 0.0):.2f}")
        self._log_file.write("  ".join(parts) + "\n")

        # Runde 8 (2026-10-13, Nutzer-Feedback "Snare ist unruhiger als
        # Kick"): volle Snare-/Clap-Diagnose (body_ratio/snap_ratio/score,
        # bereits von _update_snare_trigger()/_update_clap_trigger() in
        # trigger_debug() befüllt, siehe beat_detector.py) als eigene
        # JSON-Zeile - analog zu GATE_DIAG beim Kick (siehe _loop() unten).
        # Die kompakte "SNARE conf=..."-Zeile oben reicht nicht, um zu
        # unterscheiden, ob ein Trigger an einer stark ausgeprägten
        # Koinzidenz lag oder nur knapp über der "schwachen" 35%-Schwelle
        # (trigger_snare_allow_weak_body) - rein additiv, ändert nichts an
        # der bestehenden Zeile/CSV-Spaltenzahl.
        if evt["snare"]:
            sd_ = (evt["trigger_debug"] or {}).get("snare", {})
            self._log_file.write(f"{ts}  SNARE_DIAG  {json.dumps(sd_, default=str)}\n")
        if evt["clap"]:
            cd = (evt["trigger_debug"] or {}).get("clap", {})
            self._log_file.write(f"{ts}  CLAP_DIAG  {json.dumps(cd, default=str)}\n")

        if not self._csv_writer:
            return
        det = self._detector
        level = self._current_level()
        tail = (det.bpm, det.locked_bpm, det.confidence, level)
        if evt["kick"]:
            kd = evt["kick_debug"] or {}
            self._csv_writer.writerow([
                evt["t"], ts, "kick", *_kick_csv_fields(kd, kick_trigger_active=True),
                kd.get("confidence"), kd.get("bassline_score"), kd.get("reject_reason"), *tail,
            ])
        if evt["snare"]:
            sd_ = (evt["trigger_debug"] or {}).get("snare", {})
            self._csv_writer.writerow([
                evt["t"], ts, "snare", *([None] * _KICK_FIELD_COUNT),
                sd_.get("confidence"), None, sd_.get("reject_reason"), *tail,
            ])
        if evt["clap"]:
            cd = (evt["trigger_debug"] or {}).get("clap", {})
            self._csv_writer.writerow([
                evt["t"], ts, "clap", *([None] * _KICK_FIELD_COUNT),
                cd.get("confidence"), None, cd.get("reject_reason"), *tail,
            ])

    def _write_status(self, now: float) -> None:
        det = self._detector
        kd = det.kick_debug
        dbg = det.trigger_debug
        sd_, cd = dbg.get("snare", {}), dbg.get("clap", {})
        level = self._current_level()
        level_str = f"{level:+.1f}dBFS" if level is not None else "n/a"

        kick_thr = kd.get("threshold")
        kick_str = (
            f"env={kd.get('envelope', 0.0):.3f} thr={kick_thr:.3f} reason={kd.get('reject_reason')} "
            f"conf={kd.get('confidence', 1.0):.2f} bassline={kd.get('bassline_score', 0.0):.2f}"
            if kd and kick_thr is not None else "warmup..."
        )
        snare_str = f"reason={sd_.get('reject_reason')} conf={sd_.get('confidence', 0.0):.2f}" if sd_ else "warmup..."
        clap_str = f"reason={cd.get('reject_reason')} conf={cd.get('confidence', 0.0):.2f}" if cd else "warmup..."

        line = (
            f"{time.strftime('%H:%M:%S', time.localtime(now))}  STATUS  "
            f"bpm={det.bpm:.1f} locked={det.locked_bpm:.1f} state={det.tempo_state} lock_conf={det.confidence:.2f}  "
            f"level={level_str}  |  kick[{kick_str}]  snare[{snare_str}]  clap[{clap_str}]"
        )
        self._log_file.write(line + "\n")

        if not self._csv_writer or not kd:
            return
        # Periodischer Kick-Snapshot in die CSV (Runde 4, A/B-Test) - im
        # Gegensatz zu _write_event() unabhängig davon, ob ein Trigger
        # feuert. Nötig, weil genau das der zu untersuchende Fall ist
        # (kalman_kick_std_factor evtl. zu streng -> nie ein Trigger, aber
        # envelope/threshold/reason sollen trotzdem über die Zeit sichtbar
        # sein). Rate bleibt an refresh_hz gebunden, keine Änderung an der
        # Architektur/Rate-Begrenzung selbst.
        ts = time.strftime("%H:%M:%S", time.localtime(now))
        self._csv_writer.writerow([
            now, ts, "status", *_kick_csv_fields(kd, kick_trigger_active=det.kick_trigger),
            kd.get("confidence"), kd.get("bassline_score"), kd.get("reject_reason"),
            det.bpm, det.locked_bpm, det.confidence, level,
        ])
