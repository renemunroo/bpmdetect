"""Ableton Link Bridge — verwaltet den Node.js-Subprocess und IPC.

Protokoll (JSON, newline-delimitiert):
  Python → Node  (stdin):  {"type":"set_tempo","bpm":128.4}
  Node   → Python (stdout): {"type":"state","tempo":128.0,"phase":0.75,...}
"""

import json
import os
import select
import shutil
import subprocess
from typing import Optional
import sys
import threading
import time
from pathlib import Path

if getattr(sys, "frozen", False):
    import os as _os
    _BRIDGE_DIR = Path(_os.environ.get("RESOURCEPATH",
                       str(Path(sys.executable).parent.parent / "Resources"))) / "link"
else:
    _BRIDGE_DIR = Path(__file__).parent.parent / "link"
_BRIDGE_SCRIPT = _BRIDGE_DIR / "bridge.js"
_NODE_READY_TIMEOUT = 5.0   # Sekunden bis "ready" erwartet wird

# Übliche Node.js-Installationspfade (Finder/launchd haben eingeschränktes PATH)
_NODE_CANDIDATES = [
    "/usr/local/bin/node",
    "/opt/homebrew/bin/node",
    "/opt/local/bin/node",
    os.path.expanduser("~/.nvm/versions/node"),   # NVM-Fallback: Verzeichnis
]


def _find_node() -> Optional[str]:
    """Sucht node-Binary in PATH und üblichen Installationsorten."""
    found = shutil.which("node")
    if found:
        return found
    for p in _NODE_CANDIDATES:
        path = Path(p)
        if path.is_file() and os.access(str(path), os.X_OK):
            return str(path)
        # NVM: nimm das neueste
        if path.is_dir():
            bins = sorted(path.glob("*/bin/node"), reverse=True)
            if bins:
                return str(bins[0])
    return None


def _phase_correction_target(
    offset: float,
    deadband: float,
    now: float,
    last_correction_t: float,
    correction_interval: float,
    current_beat: float,
) -> Optional[float]:
    """Reine Entscheidungslogik für die Beat-Phasenkorrektur (Bugfix
    2026-10-13, siehe LinkBridge.update()) - bewusst als eigenständige
    Funktion statt Methode, damit sie ohne echten Subprocess/echte Link-
    Session testbar ist (gleiches Prinzip wie clampTempo in aura-lc's
    src/linkSync.js).

    `offset` ist Link-Beat-Phase minus erkannte Audio-Phase (auf
    [-0.5, 0.5] normiert, siehe Aufrufer). Gibt den Ziel-Beat zurück, auf
    den request_beat gesetzt werden soll, oder None, wenn (noch) nicht
    korrigiert werden soll:
    - innerhalb der Totzone (`abs(offset) <= deadband`) - normales
      Mess-Rauschen der Audio-Phasenerkennung soll keine ständige
      sichtbare Mini-Korrektur auslösen.
    - seit der letzten Korrektur ist noch nicht genug Zeit vergangen
      (`correction_interval`) - gibt Link Zeit, die letzte Korrektur
      tatsächlich zu übernehmen, bevor die nächste gesendet wird.
    """
    if abs(offset) <= deadband:
        return None
    if (now - last_correction_t) < correction_interval:
        return None
    return current_beat - offset


class LinkBridge:
    """
    Verwaltet den Node.js-Link-Subprocess und exponiert den Link-State.

    Thread-Sicherheit: Alle Properties sind thread-safe (interner Lock).
    `update()` darf aus dem Audio-Callback aufgerufen werden.
    """

    def __init__(
        self,
        quantum: int = 4,
        update_interval_ms: float = 200.0,
        tempo_hysteresis: float = 0.5,
        push_interval_ms: float = 25.0,
        phase_correction_deadband: float = 0.05,
        phase_correction_interval_ms: float = 1000.0,
    ) -> None:
        self._quantum = quantum
        self._update_interval = update_interval_ms / 1000.0
        self._tempo_hysteresis = tempo_hysteresis
        # Wie oft bridge.js den State pusht (Node-seitiger setInterval) -
        # vorher fest 100ms, jetzt konfigurierbar (Default 25ms) für
        # beat_kick-Zeitgenauigkeit, siehe beat_scheduler.py. Wird beim
        # Subprocess-Start als CLI-Arg an bridge.js übergeben.
        self._push_interval_ms = push_interval_ms
        # Bugfix 2026-10-13 (Beat-Phase lief auseinander, set_tempo allein
        # korrigiert nur die Taktrate, nie die Phase selbst - siehe
        # request_beat-Kommentar in bridge.js): Totzone in Bruchteilen EINES
        # Beats (0.05 = 5% eines Beats, z.B. bei 120BPM ~25ms) - darunter
        # wird NICHT korrigiert, sonst würde jede normale Mess-Unschärfe der
        # Audio-Phasenerkennung eine ständige, sichtbare Mini-Korrektur
        # auslösen. Mindestabstand zwischen zwei Korrekturen (Default 1s) -
        # verhindert, dass eine Korrektur noch nicht "angekommen" ist (Link
        # braucht einen Moment, bis andere Peers/der eigene State sie
        # übernommen haben), bevor schon die nächste gesendet wird.
        self._phase_correction_deadband = phase_correction_deadband
        self._phase_correction_interval = phase_correction_interval_ms / 1000.0
        self._last_phase_correction_t: float = 0.0

        self._lock = threading.Lock()
        self._session_tempo: float = 0.0
        self._link_phase: float = 0.0       # 0..1, über das volle Quantum (Takt) - nur fürs Logging/GUI
        self._link_beat_phase: float = 0.0  # 0..1, über GENAU EINEN Beat - für den Phasenvergleich/-korrektur
        self._link_beat: float = 0.0
        self._peers: int = 0
        self._exported_bpm: float = 0.0
        self._phase_offset: float = 0.0
        self._last_export_t: float = 0.0

        self._available: bool = False
        self._proc = None  # type: subprocess.Popen
        self._running: bool = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Startet den Node.js-Subprocess. Gibt True zurück wenn erfolgreich."""
        node = _find_node()
        if not node:
            _warn("Node.js nicht gefunden (PATH + übliche Pfade durchsucht).")
            return False

        if not _BRIDGE_SCRIPT.exists():
            _warn(f"Bridge-Script nicht gefunden: {_BRIDGE_SCRIPT}")
            return False

        try:
            self._proc = subprocess.Popen(
                [node, str(_BRIDGE_SCRIPT), str(int(self._push_interval_ms))],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(_BRIDGE_DIR),
                text=True,
                bufsize=1,
            )
        except OSError as e:
            _warn(f"Subprocess-Start fehlgeschlagen: {e}")
            return False

        if not self._wait_for_ready():
            return False

        self._running = True
        threading.Thread(
            target=self._stdout_reader, daemon=True, name="link-stdout"
        ).start()
        threading.Thread(
            target=self._stderr_logger, daemon=True, name="link-stderr"
        ).start()

        self._send({"type": "set_quantum", "quantum": self._quantum})
        self._available = True
        return True

    def stop(self) -> None:
        self._running = False
        self._available = False
        if self._proc:
            try:
                self._send({"type": "quit"})
            except Exception:
                pass
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except Exception:
                pass

    # ── Hauptschnittstelle ───────────────────────────────────────────────────

    def update(self, detected_bpm: float, detected_phase: float) -> None:
        """
        Soll periodisch (z. B. aus on_block) aufgerufen werden.

        Übergibt `detected_bpm` rate-limitiert und hysterese-gefiltert an Link.
        Berechnet `phase_offset` zwischen Link-Phase und detektierter Phase.
        """
        if not self._available:
            return

        now = time.time()

        # Tempo-Export: nur wenn Änderung groß genug und Interval abgelaufen
        with self._lock:
            last_bpm = self._exported_bpm
            interval_ok = (now - self._last_export_t) >= self._update_interval

        if (
            detected_bpm > 0
            and interval_ok
            and abs(detected_bpm - last_bpm) >= self._tempo_hysteresis
        ):
            self._send({"type": "set_tempo", "bpm": round(detected_bpm, 2)})
            with self._lock:
                self._exported_bpm = detected_bpm
                self._last_export_t = now

        # Phase-Offset: Differenz Link-BEAT-Phase (0..1, EIN Beat - siehe
        # _link_beat_phase, bridge.js getBeatPhase()) zu detektierter Phase
        # (0..1, ebenfalls pro Beat, siehe beat_detector.py beat_phase) -
        # auf [-0.5, 0.5] normiert (negativ = Link hinkt nach). Bugfix
        # 2026-10-13: vorher gegen `_link_phase` (Quantum-/Takt-Phase,
        # Standard quantum=4) verglichen - eine andere Domäne als die
        # Pro-Beat-Phase aus der Audio-Erkennung, der errechnete Offset war
        # dadurch bis zu `quantum`-fach falsch (hier allerdings folgenlos,
        # weil er bis jetzt nie für eine echte Korrektur genutzt wurde,
        # siehe unten).
        if detected_phase >= 0:
            with self._lock:
                raw = self._link_beat_phase - detected_phase
            offset = (raw + 0.5) % 1.0 - 0.5
            with self._lock:
                self._phase_offset = offset
                last_correction_t = self._last_phase_correction_t
                current_beat = self._link_beat

            # Bugfix 2026-10-13 ("BPM Flash" lief auseinander, fing sich
            # dann wieder - siehe request_beat-Kommentar in bridge.js):
            # set_tempo oben korrigiert nur die Taktrate, nie die Phase -
            # das hier ist die eigentliche Phasenkorrektur, sanft (request,
            # nicht force) und gedämpft (Totzone + Mindestabstand), damit
            # normales Mess-Rauschen der Audio-Phasenerkennung nicht in
            # einer ständigen sichtbaren Mini-Korrektur endet.
            target_beat = _phase_correction_target(
                offset,
                self._phase_correction_deadband,
                now,
                last_correction_t,
                self._phase_correction_interval,
                current_beat,
            )
            if target_beat is not None:
                self._send({"type": "request_beat", "beat": round(target_beat, 4), "quantum": 1})
                with self._lock:
                    self._last_phase_correction_t = now
                _warn(f"phase_correction: offset={offset:+.3f} beat -> Ziel-Beat {target_beat:.3f}")

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return self._available

    @property
    def session_tempo(self) -> float:
        """Aktuelles Tempo der Link-Session (von Link gemeldet)."""
        with self._lock:
            return self._session_tempo

    @property
    def exported_bpm(self) -> float:
        """Zuletzt an Link übertragenes Tempo."""
        with self._lock:
            return self._exported_bpm

    @property
    def phase_offset(self) -> float:
        """Phasendifferenz Link − Detektor, normiert auf [-0.5, 0.5]."""
        with self._lock:
            return self._phase_offset

    @property
    def peers(self) -> int:
        with self._lock:
            return self._peers

    @property
    def link_phase(self) -> float:
        """Link-Phase 0..1, über das volle Quantum (Takt)."""
        with self._lock:
            return self._link_phase

    @property
    def link_beat_phase(self) -> float:
        """Link-Phase 0..1, über GENAU EINEN Beat - dieselbe Domäne wie
        beat_detector.py's beat_phase, für die Phasenkorrektur (siehe
        update())."""
        with self._lock:
            return self._link_beat_phase

    @property
    def link_beat(self) -> float:
        """Fortlaufender Ableton-Link-Beat-Zähler (steigt kontinuierlich,
        kein Bar-/Quantum-Bezug) - für beat_scheduler.py, das daraus
        diskrete beat_kick-Ereignisse ableitet."""
        with self._lock:
            return self._link_beat

    # ── Interne Hilfsmethoden ────────────────────────────────────────────────

    def _send(self, cmd: dict) -> None:
        if self._proc and self._proc.stdin:
            try:
                self._proc.stdin.write(json.dumps(cmd) + "\n")
                self._proc.stdin.flush()
            except BrokenPipeError:
                self._available = False

    def _wait_for_ready(self) -> bool:
        """Wartet auf das ready-Signal von stderr (Timeout: 5 s)."""
        deadline = time.time() + _NODE_READY_TIMEOUT
        while time.time() < deadline:
            if self._proc.poll() is not None:
                err = self._proc.stderr.read()
                _warn(f"Bridge vorzeitig beendet: {err.strip()}")
                return False
            # Non-blocking stderr-Lesen
            rlist, _, _ = select.select([self._proc.stderr], [], [], 0.1)
            if rlist:
                line = self._proc.stderr.readline().strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                    if msg.get("type") == "ready":
                        return True
                    if msg.get("type") == "error":
                        _warn(f"Bridge-Fehler: {msg.get('msg', line)}")
                        return False
                except json.JSONDecodeError:
                    pass  # Non-JSON stderr (z. B. Node-Startmeldungen)
        _warn("Timeout: kein ready-Signal von Bridge erhalten.")
        return False

    def _stdout_reader(self) -> None:
        """Hintergrund-Thread: verarbeitet JSON-State-Updates von Node."""
        while self._running and self._proc:
            try:
                line = self._proc.stdout.readline()
                if not line:
                    break
                msg = json.loads(line.strip())
                if msg.get("type") == "state":
                    with self._lock:
                        self._session_tempo = float(msg.get("tempo", 0))
                        self._link_phase    = float(msg.get("phase", 0))
                        # Fällt auf die Quantum-Phase zurück, falls eine
                        # ältere bridge.js (ohne beatPhase-Feld) läuft -
                        # dann bleibt die Phasenkorrektur oben einfach
                        # inaktiv (Totzone/Vergleich wird dadurch nur
                        # ungenauer, nie falsch-positiv aktiv), kein Absturz.
                        self._link_beat_phase = float(msg.get("beatPhase", self._link_phase))
                        self._link_beat     = float(msg.get("beat",  0))
                        self._peers         = int(msg.get("peers",   0))
            except (json.JSONDecodeError, ValueError):
                pass
            except Exception:
                break
        self._available = False

    def _stderr_logger(self) -> None:
        """Hintergrund-Thread: leitet Bridge-Warnungen nach stderr."""
        while self._running and self._proc:
            try:
                line = self._proc.stderr.readline()
                if not line:
                    break
                line = line.strip()
                if line:
                    try:
                        msg = json.loads(line)
                        if msg.get("type") == "error":
                            _warn(f"Bridge: {msg.get('msg', line)}")
                    except json.JSONDecodeError:
                        pass  # Ignore Node startup noise
            except Exception:
                break


def _warn(msg: str) -> None:
    print(f"[link] {msg}", file=sys.stderr)
