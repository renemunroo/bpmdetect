#!/usr/bin/env python3
"""
Ableton Link Monitor — PySide6 GUI (macOS-stabil)
Berührt keine Beat-Detection-Dateien.

Start:
    cd /Users/rene/bpmdetect
    .venv/bin/python link/link_monitor.py [--tempo 120] [--quantum 4]

Diagnose (kein GUI):
    .venv/bin/python link/link_monitor.py --diag
"""
import argparse
import json
import logging
import queue
import select
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

if getattr(sys, "frozen", False):
    import os as _os
    _link_res = Path(_os.environ.get("RESOURCEPATH",
                    str(Path(sys.executable).parent.parent / "Resources"))) / "link"
    BRIDGE = _link_res / "monitor_bridge.js"
else:
    BRIDGE = Path(__file__).resolve().parent / "monitor_bridge.js"
TICK_MS = 120

# Wie lange Link aktiviert sein muss (Sekunden) bevor eine Warnung erscheint,
# wenn noch immer 0 Peers gefunden wurden.
PEER_WARN_AFTER_S = 15.0

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
log = logging.getLogger("link_monitor")

# ── Farben ─────────────────────────────────────────────────────────────────────
BG     = "#1e1e2e"
PANEL  = "#313244"
FG     = "#cdd6f4"
DIM    = "#585b70"
ACCENT = "#89b4fa"
GREEN  = "#a6e3a1"
RED    = "#f38ba8"
YELLOW = "#f9e2af"


# ── Umgebungs-Diagnose (wird immer beim Start geloggt) ────────────────────────

def _log_environment() -> str:
    """Logt Startkontext und gibt den Node-Pfad zurück (oder '' wenn nicht gefunden)."""
    # Bundle-ID aus CFBundleIdentifier ermitteln (nur wenn als .app gestartet)
    bundle_id = _get_bundle_id()
    start_method = "app-bundle" if bundle_id else "terminal/command"

    log.info("=== Startkontext ===")
    log.info("App-Name     : Ableton Link Monitor")
    log.info("Bundle-ID    : %s", bundle_id or "(keins — kein .app-Bundle)")
    log.info("Startmethode : %s", start_method)
    log.info("CWD          : %s", Path.cwd())
    log.info("Script       : %s", Path(__file__).resolve())
    log.info("Bridge       : %s  (exists=%s)", BRIDGE, BRIDGE.exists())
    log.info("Python       : %s", sys.executable)

    node = shutil.which("node")
    if node:
        try:
            ver = subprocess.check_output([node, "--version"],
                                          stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            ver = "?"
        log.info("node         : %s  (%s)", node, ver)
    else:
        log.error("node         : NICHT IM PATH GEFUNDEN")
        log.error("  → node installieren oder PATH prüfen")

    if not bundle_id:
        log.warning("Kein .app-Bundle erkannt. Local Network Permission wird in")
        log.warning("Systemeinstellungen ggf. Terminal.app zugeordnet, nicht dieser App.")
        log.warning("Für korrekte Zuordnung: LinkMonitor.app per Doppelklick starten.")
    log.info("===================")
    return node or ""


def _get_bundle_id() -> str:
    """Gibt CFBundleIdentifier zurück wenn als .app gestartet, sonst ''."""
    try:
        # Info.plist liegt drei Ebenen über dem MacOS/-Executable
        executable = Path(sys.executable).resolve()
        # Prüfe ob wir innerhalb eines .app-Bundles laufen
        # Typischer Pfad: .../LinkMonitor.app/Contents/MacOS/link-monitor → Python
        # oder direkt: .../bpmdetect/.venv/bin/python  (kein Bundle)
        script = Path(__file__).resolve()
        # Suche Info.plist relativ zum Script-Verzeichnis
        info_plist = script.parent.parent / "LinkMonitor.app" / "Contents" / "Info.plist"
        # Alternativ: über BUNDLE_ID Env-Variable (kann im Bundle-Executable gesetzt werden)
        import os
        env_id = os.environ.get("LINK_MONITOR_BUNDLE_ID", "")
        if env_id:
            return env_id
        return ""
    except Exception:
        return ""


def _trigger_local_network_permission() -> bool:
    """
    Beitritt zur Ableton-Link-Multicast-Gruppe (224.76.78.75:20808).
    Das ist der erste lokale Netzwerkzugriff dieser App-Instanz.
    macOS zeigt dabei den 'Lokales Netzwerk'-Berechtigungs-Dialog an,
    sofern die App als .app-Bundle mit NSLocalNetworkUsageDescription läuft.

    Gibt True zurück wenn der Join erfolgreich war.
    """
    LINK_MULTICAST = "224.76.78.75"
    LINK_PORT      = 20808
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(0.5)
        # Multicast-Gruppe beitreten — das triggert den macOS Permission-Dialog
        mreq = struct.pack("4sL", socket.inet_aton(LINK_MULTICAST), socket.INADDR_ANY)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        sock.close()
        log.info("Local Network: Multicast-Join %s:%d — OK (Permission-Dialog ggf. erschienen)",
                 LINK_MULTICAST, LINK_PORT)
        return True
    except OSError as e:
        log.warning("Local Network: Multicast-Join fehlgeschlagen: %s", e)
        log.warning("  → macOS blockiert evtl. den Zugriff (Permission verweigert)")
        return False


# ── CLI-Diagnosemodus ─────────────────────────────────────────────────────────

def run_diag() -> None:
    """
    Startet monitor_bridge.js, aktiviert Link und prüft nach 5 Sekunden
    ob Peers gefunden wurden. Beendet sich danach.
    """
    print("\n── Ableton Link Diagnose ──────────────────────────────")
    node = _log_environment()
    if not node:
        print("\n✗  node nicht gefunden — bitte installieren.")
        sys.exit(1)

    if not BRIDGE.exists():
        print(f"\n✗  monitor_bridge.js nicht gefunden: {BRIDGE}")
        sys.exit(1)

    print(f"\n  node    : {node}")
    print(f"  bridge  : {BRIDGE}")
    print("\n  Starte Bridge …")

    try:
        proc = subprocess.Popen(
            [node, str(BRIDGE)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=str(BRIDGE.parent), text=True, bufsize=1,
        )
    except OSError as e:
        print(f"\n✗  Subprocess-Start fehlgeschlagen: {e}")
        sys.exit(1)

    # Ready-Signal abwarten
    ready = False
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if proc.poll() is not None:
            print(f"\n✗  Bridge sofort beendet: {proc.stderr.read()}")
            sys.exit(1)
        r, _, _ = select.select([proc.stderr], [], [], 0.1)
        if r:
            raw = proc.stderr.readline().strip()
            try:
                msg = json.loads(raw)
                if msg.get("type") == "ready":
                    ready = True
                    break
            except json.JSONDecodeError:
                pass

    if not ready:
        print("\n✗  Kein ready-Signal von Bridge erhalten (Timeout 5 s)")
        proc.terminate()
        sys.exit(1)

    print("  Bridge  : bereit ✓")

    # Link aktivieren
    proc.stdin.write(json.dumps({"type": "set_enabled", "enabled": True}) + "\n")
    proc.stdin.flush()
    print("  Link    : aktiviert, warte 5 s auf Peers …")

    # 5 Sekunden State lesen
    state = {}
    deadline = time.time() + 5.0
    while time.time() < deadline:
        r, _, _ = select.select([proc.stdout], [], [], 0.1)
        if r:
            line = proc.stdout.readline()
            try:
                msg = json.loads(line.strip())
                if msg.get("type") == "state":
                    state = msg
            except json.JSONDecodeError:
                pass

    proc.stdin.write(json.dumps({"type": "quit"}) + "\n")
    proc.stdin.flush()
    proc.terminate()

    peers  = state.get("peers", 0)
    tempo  = state.get("tempo", 0.0)
    enabled = state.get("enabled", False)

    print(f"\n  enabled : {enabled}")
    print(f"  peers   : {peers}")
    print(f"  tempo   : {tempo:.2f} BPM")

    if peers > 0:
        print(f"\n✓  {peers} Peer(s) gefunden — Ableton Link funktioniert.")
    else:
        print("\n⚠  Keine Peers gefunden (0). Mögliche Ursachen:")
        print("   1. Kein anderes Ableton-Link-Gerät im Netzwerk aktiv")
        print("   2. macOS Local Network Permission fehlt:")
        print("      Systemeinstellungen → Datenschutz & Sicherheit")
        print("      → Lokales Netzwerk → Terminal ✓ aktivieren")
        print("   3. Firewall blockiert UDP-Multicast (Port 20808)")

    print("──────────────────────────────────────────────────────\n")
    sys.exit(0 if peers > 0 else 2)


# ── Bridge ─────────────────────────────────────────────────────────────────────

class Bridge:
    """
    Verwaltet monitor_bridge.js als Subprocess.
    _reader() läuft in Daemon-Thread; kommuniziert mit Main-Thread via queue.Queue.
    drain() und send() sind die einzigen Main-Thread-Schnittstellen.
    """

    def __init__(self):
        self._proc      = None
        self._stop      = threading.Event()
        self._q         = queue.Queue(maxsize=20)
        self._send_lock = threading.Lock()
        self.available  = False
        self._state     = {
            "enabled": False, "tempo": 120.0, "beat": 0.0,
            "phase": 0.0, "peers": 0, "quantum": 4,
            "playing": False, "ss_sync": False,
        }
        # Zeitpunkt wann Link zuletzt aktiviert wurde (für Peer-Warnung)
        self.enabled_since: float = 0.0

    def start(self, init_tempo: float, init_quantum: int) -> bool:
        node = shutil.which("node")
        if not node:
            log.error("node nicht im PATH gefunden.")
            return False
        if not BRIDGE.exists():
            log.error("monitor_bridge.js nicht gefunden: %s", BRIDGE)
            return False

        # Multicast-Join VOR Node-Start: triggert macOS Local-Network-Permission-Dialog
        # für diesen Prozess / dieses Bundle, bevor Node.js ihn selbst auslöst.
        _trigger_local_network_permission()

        log.info("Starte Bridge: node=%s  bridge=%s", node, BRIDGE)
        try:
            self._proc = subprocess.Popen(
                [node, str(BRIDGE)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=str(BRIDGE.parent), text=True, bufsize=1,
            )
        except OSError as e:
            log.error("Subprocess-Start fehlgeschlagen: %s", e)
            return False

        if not self._wait_for_ready():
            return False

        self._stop.clear()
        threading.Thread(target=self._reader, daemon=True, name="bridge-reader").start()
        log.info("Bridge gestartet (PID %d)", self._proc.pid)
        self.send("set_tempo",   bpm=init_tempo)
        self.send("set_quantum", quantum=init_quantum)
        self.available = True
        return True

    def stop(self):
        log.info("Bridge wird gestoppt.")
        self.available = False
        self._stop.set()
        self.send("quit")
        p = self._proc
        if p:
            try:
                p.stdin.close()
            except Exception:
                pass
            try:
                p.terminate()
                p.wait(timeout=2.0)
            except Exception:
                pass
        log.info("Bridge-Prozess beendet.")

    def send(self, type_: str, **kw):
        p = self._proc
        if p is None or p.stdin is None:
            return
        with self._send_lock:
            try:
                p.stdin.write(json.dumps({"type": type_, **kw}) + "\n")
                p.stdin.flush()
            except (BrokenPipeError, OSError):
                self.available = False

    def drain(self) -> dict:
        """Alle Queue-Nachrichten verarbeiten. Nur vom Main-Thread aufrufen."""
        while True:
            try:
                msg = self._q.get_nowait()
                enabled_now = bool(msg.get("enabled", False))
                # Zeitstempel für Peer-Warnung nachführen
                was_enabled = self._state.get("enabled", False)
                if enabled_now and not was_enabled:
                    self.enabled_since = time.time()
                elif not enabled_now:
                    self.enabled_since = 0.0
                self._state.update({
                    "enabled": enabled_now,
                    "tempo":   float(msg.get("tempo",   120.0)),
                    "beat":    float(msg.get("beat",    0.0)),
                    "phase":   float(msg.get("phase",   0.0)),
                    "peers":   int(msg.get("peers",     0)),
                    "quantum": float(msg.get("quantum", 4)),
                    "playing": bool(msg.get("playing",  False)),
                    "ss_sync": bool(msg.get("ss_sync",  False)),
                })
            except queue.Empty:
                break
        return self._state

    def peer_warn_active(self) -> bool:
        """True wenn Link seit >PEER_WARN_AFTER_S aktiviert ist, aber 0 Peers."""
        s = self._state
        if not s.get("enabled") or s.get("peers", 0) > 0:
            return False
        since = self.enabled_since
        return since > 0 and (time.time() - since) > PEER_WARN_AFTER_S

    def _wait_for_ready(self) -> bool:
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if self._proc.poll() is not None:
                log.error("Bridge unerwartet beendet: %s", self._proc.stderr.read().strip())
                return False
            r, _, _ = select.select([self._proc.stderr], [], [], 0.1)
            if r:
                raw = self._proc.stderr.readline().strip()
                if not raw:
                    continue
                try:
                    msg = json.loads(raw)
                    if msg.get("type") == "ready":
                        log.info("Bridge ready.")
                        return True
                    if msg.get("type") == "error":
                        log.error("Bridge-Fehler: %s", msg.get("msg"))
                        return False
                    # Alle anderen stderr-Zeilen von Node durchloggen
                    log.debug("bridge stderr: %s", raw)
                except json.JSONDecodeError:
                    log.debug("bridge stderr: %s", raw)
        log.error("Timeout: kein ready-Signal von Bridge erhalten.")
        return False

    def _reader(self):
        log.debug("reader-Thread gestartet.")
        proc = self._proc
        while not self._stop.is_set():
            try:
                if proc.poll() is not None:
                    log.warning("Bridge-Prozess beendet, reader stoppt.")
                    break
                line = proc.stdout.readline()
                if not line:
                    log.warning("Bridge stdout EOF.")
                    break
                msg = json.loads(line.strip())
                if msg.get("type") == "state":
                    peers = msg.get("peers", 0)
                    if peers > 0:
                        log.info("Peers: %d  tempo: %.2f", peers, msg.get("tempo", 0))
                    try:
                        self._q.put_nowait(msg)
                    except queue.Full:
                        try:
                            self._q.get_nowait()
                        except queue.Empty:
                            pass
                        self._q.put_nowait(msg)
            except (json.JSONDecodeError, ValueError):
                pass
            except OSError:
                break
            except Exception as e:
                log.error("reader-Exception: %s", e)
                break
        self.available = False
        log.debug("reader-Thread beendet.")


# ── Phase-Balken ───────────────────────────────────────────────────────────────

try:
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QColor, QPainter
    from PySide6.QtWidgets import (
        QApplication, QButtonGroup, QFrame, QHBoxLayout,
        QLabel, QPushButton, QRadioButton, QVBoxLayout, QWidget,
    )
    _HAS_GUI = True
except ImportError:
    _HAS_GUI = False


class PhaseBar:  # wird nur instanziert wenn _HAS_GUI
    pass


def _make_phase_bar(parent=None):
    class _PhaseBar(QWidget):
        def __init__(self, p=None):
            super().__init__(p)
            self._phase = 0.0
            self.setFixedSize(180, 18)

        def set_phase(self, v: float):
            self._phase = max(0.0, min(1.0, v))
            self.update()

        def paintEvent(self, _event):
            p = QPainter(self)
            p.fillRect(self.rect(), QColor(PANEL))
            bw = int(self._phase * 180)
            if bw > 0:
                p.fillRect(0, 0, bw, 18, QColor(ACCENT))
            dx = max(0, min(166, int(self._phase * 166)))
            p.fillRect(dx, 2, 14, 14, QColor(GREEN))

    return _PhaseBar(parent)


# ── Haupt-Widget ───────────────────────────────────────────────────────────────

def run_gui(init_tempo: float, init_quantum: int) -> None:
    if not _HAS_GUI:
        print("PySide6 nicht installiert. Bitte: pip install PySide6", file=sys.stderr)
        sys.exit(1)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = LinkMonitor(init_tempo=init_tempo, init_quantum=init_quantum)
    win.show()
    sys.exit(app.exec())


class LinkMonitor(QWidget):

    def __init__(self, init_tempo: float, init_quantum: int):
        super().__init__()
        self.setWindowTitle("Ableton Link Monitor")
        self.setFixedWidth(340)
        self.setStyleSheet(f"QWidget {{ background-color: {BG}; color: {FG}; }}")

        self._bridge   = Bridge()
        self._ss_on    = False
        self._init_q   = init_quantum

        self._build()

        ok = self._bridge.start(init_tempo, init_quantum)
        if not ok:
            self._set_status("Bridge nicht verfügbar", RED)
            log.warning("Bridge nicht gestartet.")

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(TICK_MS)
        log.info("GUI bereit.")

    # ── Layout ─────────────────────────────────────────────────────────────────

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(6)

        t = QLabel("Ableton Link Monitor")
        t.setStyleSheet(f"color: {ACCENT}; font-size: 14pt; font-weight: bold;")
        t.setAlignment(Qt.AlignCenter)
        root.addWidget(t)

        self._status_lbl = QLabel("Initialisiere…")
        self._status_lbl.setStyleSheet(f"color: {DIM}; font-size: 9pt;")
        self._status_lbl.setAlignment(Qt.AlignCenter)
        root.addWidget(self._status_lbl)

        # Warnung für Local Network Permission (zunächst versteckt)
        bundle_id = _get_bundle_id()
        import os as _os
        if not bundle_id and not _os.environ.get("LINK_MONITOR_BUNDLE_ID"):
            warn_text = (
                "⚠  Kein App-Bundle erkannt — starte über LinkMonitor.app\n"
                "   (Doppelklick auf LinkMonitor.app im Projektordner),\n"
                "   damit macOS die Local-Network-Berechtigung\n"
                "   dieser App zuordnen kann."
            )
        else:
            warn_text = (
                "⚠  Keine Peers — Local Network Permission prüfen:\n"
                "   Systemeinstellungen → Datenschutz → Lokales Netzwerk\n"
                "   → Ableton Link Monitor ✓ aktivieren"
            )
        self._warn_lbl = QLabel(warn_text)
        self._warn_lbl.setStyleSheet(
            f"color: {YELLOW}; font-size: 8pt; padding: 4px; "
            f"background-color: {PANEL}; border-radius: 4px;"
        )
        self._warn_lbl.setAlignment(Qt.AlignLeft)
        self._warn_lbl.setWordWrap(True)
        self._warn_lbl.setVisible(False)
        root.addWidget(self._warn_lbl)

        root.addWidget(_sep())

        self._link_btn = QPushButton("Link: AUS")
        self._link_btn.setStyleSheet(_btn_css(RED, BG))
        self._link_btn.clicked.connect(self._toggle_link)
        root.addWidget(self._link_btn)

        row, self._peers_val = _kv_row("Peers", "—")
        root.addLayout(row)

        tempo_row = QHBoxLayout()
        tempo_row.addWidget(_lbl("Tempo"))
        b_minus = QPushButton("−")
        b_minus.setFixedWidth(28)
        b_minus.setStyleSheet(_btn_css(PANEL, FG))
        b_minus.clicked.connect(lambda: self._adj_tempo(-1))
        tempo_row.addWidget(b_minus)
        self._tempo_val = QLabel("120.00")
        self._tempo_val.setStyleSheet(
            f"color: {FG}; font-family: Courier; font-size: 13pt; font-weight: bold;")
        self._tempo_val.setMinimumWidth(74)
        tempo_row.addWidget(self._tempo_val)
        b_plus = QPushButton("+")
        b_plus.setFixedWidth(28)
        b_plus.setStyleSheet(_btn_css(PANEL, FG))
        b_plus.clicked.connect(lambda: self._adj_tempo(+1))
        tempo_row.addWidget(b_plus)
        tempo_row.addStretch()
        root.addLayout(tempo_row)

        phase_row = QHBoxLayout()
        phase_row.addWidget(_lbl("Phase"))
        self._phase_bar = _make_phase_bar(self)
        phase_row.addWidget(self._phase_bar)
        phase_row.addStretch()
        root.addLayout(phase_row)

        row, self._beat_val = _kv_row("Beat", "—")
        root.addLayout(row)

        q_row = QHBoxLayout()
        q_row.addWidget(_lbl("Quantum"))
        self._q_group = QButtonGroup(self)
        for q in (2, 4, 8):
            rb = QRadioButton(str(q))
            rb.setStyleSheet(f"color: {FG};")
            if q == self._init_q:
                rb.setChecked(True)
            self._q_group.addButton(rb, q)
            q_row.addWidget(rb)
        self._q_group.idClicked.connect(self._set_quantum)
        q_row.addStretch()
        root.addLayout(q_row)

        row, self._play_val = _kv_row("Playing", "—")
        root.addLayout(row)

        root.addWidget(_sep())

        self._ss_btn = QPushButton("Start/Stop Sync: AUS")
        self._ss_btn.setStyleSheet(_btn_css(PANEL, DIM))
        self._ss_btn.clicked.connect(self._toggle_ss)
        root.addWidget(self._ss_btn)

    # ── Refresh ────────────────────────────────────────────────────────────────

    def _tick(self):
        try:
            self._refresh()
        except Exception as e:
            log.error("_tick Fehler: %s", e)

    def _refresh(self):
        s = self._bridge.drain()

        if self._bridge.available:
            peers = s["peers"]
            self._set_status(
                f"Verbunden  •  {peers} Peer{'s' if peers != 1 else ''}",
                GREEN if peers else YELLOW,
            )
        else:
            self._set_status("Bridge nicht verfügbar", RED)

        if s["enabled"]:
            self._link_btn.setText("Link: AN")
            self._link_btn.setStyleSheet(_btn_css(GREEN, BG))
        else:
            self._link_btn.setText("Link: AUS")
            self._link_btn.setStyleSheet(_btn_css(RED, BG))

        p = s["peers"]
        self._peers_val.setText(str(p))
        self._peers_val.setStyleSheet(_mono(GREEN if p > 0 else YELLOW))

        self._tempo_val.setText(f"{s['tempo']:7.2f}")
        self._phase_bar.set_phase(s["phase"])
        self._beat_val.setText(f"{s['beat']:.3f}")

        playing = s.get("playing", False)
        self._play_val.setText("▶  Ja" if playing else "◼  Nein")
        self._play_val.setStyleSheet(_mono(GREEN if playing else DIM))

        # Local Network Permission Warnung
        show_warn = self._bridge.peer_warn_active()
        if show_warn != self._warn_lbl.isVisible():
            self._warn_lbl.setVisible(show_warn)
            if show_warn:
                log.warning(
                    "Link seit >%.0fs enabled, aber 0 Peers. "
                    "Local Network Permission prüfen: "
                    "Systemeinstellungen → Datenschutz & Sicherheit → "
                    "Lokales Netzwerk → Terminal aktivieren",
                    PEER_WARN_AFTER_S,
                )
            self.adjustSize()

    def _set_status(self, txt: str, color: str = DIM):
        self._status_lbl.setText(txt)
        self._status_lbl.setStyleSheet(f"color: {color}; font-size: 9pt;")

    # ── Aktionen ───────────────────────────────────────────────────────────────

    def _toggle_link(self):
        new = not self._bridge._state["enabled"]
        log.info("Link → %s", new)
        self._bridge.send("set_enabled", enabled=new)

    def _adj_tempo(self, delta: float):
        t = self._bridge._state["tempo"]
        new = max(20.0, min(300.0, t + delta))
        log.info("Tempo → %.2f", new)
        self._bridge.send("set_tempo", bpm=new)

    def _set_quantum(self, q: int):
        log.info("Quantum → %d", q)
        self._bridge.send("set_quantum", quantum=q)

    def _toggle_ss(self):
        self._ss_on = not self._ss_on
        self._ss_btn.setText(f"Start/Stop Sync: {'AN' if self._ss_on else 'AUS'}")
        self._ss_btn.setStyleSheet(
            _btn_css(ACCENT, BG) if self._ss_on else _btn_css(PANEL, DIM))
        log.info("Start/Stop-Sync → %s", self._ss_on)
        self._bridge.send("set_startstop", enabled=self._ss_on)

    def closeEvent(self, event):
        log.info("Beende GUI.")
        self._timer.stop()
        self._bridge.stop()
        event.accept()
        log.info("GUI beendet.")


# ── Hilfsfunktionen ────────────────────────────────────────────────────────────

def _sep() -> "QFrame":
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setStyleSheet(f"color: {DIM};")
    return f

def _lbl(text: str) -> "QLabel":
    l = QLabel(text)
    l.setStyleSheet(f"color: {DIM}; font-size: 10pt;")
    l.setFixedWidth(68)
    return l

def _mono(color: str) -> str:
    return f"color: {color}; font-family: Courier; font-size: 12pt; font-weight: bold;"

def _kv_row(label: str, default: str):
    row = QHBoxLayout()
    row.addWidget(_lbl(label))
    val = QLabel(default)
    val.setStyleSheet(_mono(FG))
    row.addWidget(val)
    row.addStretch()
    return row, val

def _btn_css(bg: str, fg: str) -> str:
    return (
        f"QPushButton {{ background-color: {bg}; color: {fg}; "
        f"border: none; padding: 6px 10px; font-weight: bold; border-radius: 4px; }}"
        f"QPushButton:hover {{ background-color: {ACCENT}; color: {BG}; }}"
    )


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ableton Link Monitor")
    ap.add_argument("--tempo",   type=float, default=120.0,
                    help="Initialtempo in BPM")
    ap.add_argument("--quantum", type=int,   default=4,
                    help="Quantum (Schläge pro Phase)")
    ap.add_argument("--diag",    action="store_true",
                    help="Diagnose-Modus: Link-Status prüfen und beenden")
    a = ap.parse_args()

    import os as _os
    # Bundle-ID über Env-Variable weitergeben (wird vom Bundle-Executable gesetzt)
    bundle_id = _os.environ.get("LINK_MONITOR_BUNDLE_ID", "")
    if bundle_id:
        log.info("Bundle-ID (aus Env): %s", bundle_id)

    if a.diag:
        run_diag()
    else:
        _log_environment()
        log.info("Starte Link Monitor GUI (PySide6)")
        run_gui(init_tempo=a.tempo, init_quantum=a.quantum)
