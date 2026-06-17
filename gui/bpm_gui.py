#!/usr/bin/env python3
"""
BPM Detector GUI — PySide6
Hülle über bestehendem BeatDetector-Core, keine Beat-Logik hier.

Start:
    cd /Users/rene/bpmdetect
    .venv/bin/python gui/bpm_gui.py
    .venv/bin/python gui/bpm_gui.py --config config/example.yaml
"""
import argparse
import math
import queue
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFrame,
    QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

def _resource(rel: str) -> Path:
    """Pfad zu gebündelten Ressourcen — funktioniert in Entwicklung und PyInstaller-Bundle."""
    import sys, os
    if getattr(sys, "frozen", False):
        base = Path(os.environ.get("RESOURCEPATH",
                    str(Path(sys.executable).parent.parent / "Resources")))
        return base / rel
    return Path(__file__).resolve().parent.parent / rel

import config as cfg_mod
from beat_detector import BeatDetector
from capture import AudioCapture
from device_list import get_all_devices
from link_bridge import LinkBridge

# ── Farben ─────────────────────────────────────────────────────────────────────
BG     = "#1e1e2e"
PANEL  = "#313244"
FG     = "#cdd6f4"
DIM    = "#585b70"
ACCENT = "#89b4fa"
GREEN  = "#a6e3a1"
RED    = "#f38ba8"
YELLOW = "#f9e2af"
CYAN   = "#89dceb"

TICK_MS  = 80     # ~12 Hz GUI-Refresh
DB_FLOOR = -60.0


# ── Snapshot: Audio-Thread → Main-Thread ───────────────────────────────────────

@dataclass
class Snapshot:
    locked_bpm:    float = 0.0
    raw_bpm:       float = 0.0
    corrected_bpm: float = 0.0
    confidence:    float = 0.0
    tempo_state:   str   = "searching"
    candidates:    list  = field(default_factory=list)
    beat_phase:    float = 0.0
    kick_score:    float = 0.0
    snare_score:   float = 0.0
    hihat_score:   float = 0.0
    rms_l:         float = 0.0
    rms_r:         float = 0.0
    clip_l:        bool  = False
    clip_r:        bool  = False
    # Ableton Link
    link_available:    bool  = False
    link_peers:        int   = 0
    link_session_bpm:  float = 0.0
    link_exported_bpm: float = 0.0
    link_phase_offset: float = 0.0


# ── Primitive Custom-Widgets ───────────────────────────────────────────────────

class BarWidget(QWidget):
    """Horizontaler Füllstand-Balken. set_value(0.0–1.0), optionale Farbe."""

    def __init__(self, width: int = 160, height: int = 12,
                 fg: str = ACCENT, parent=None):
        super().__init__(parent)
        self.setFixedSize(width, height)
        self._v  = 0.0
        self._fg = QColor(fg)
        self._bg = QColor(PANEL)

    def set_value(self, v: float, color: Optional[str] = None):
        self._v = max(0.0, min(1.0, v))
        if color:
            self._fg = QColor(color)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), self._bg)
        w = int(self._v * self.width())
        if w > 0:
            p.fillRect(0, 0, w, self.height(), self._fg)


class PhaseWidget(QWidget):
    """Laufender Punkt entlang einer horizontalen Linie."""

    def __init__(self, width: int = 180, height: int = 12, parent=None):
        super().__init__(parent)
        self.setFixedSize(width, height)
        self._phase = 0.0

    def set_phase(self, v: float):
        self._phase = max(0.0, min(1.0, v))
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(PANEL))
        dot = 12
        x = int(self._phase * (self.width() - dot))
        p.fillRect(x, 1, dot, self.height() - 2, QColor(GREEN))


# ── Haupt-Fenster ──────────────────────────────────────────────────────────────

class BpmGui(QWidget):

    def __init__(self, initial_config: Optional[Path] = None):
        super().__init__()
        self.setWindowTitle("BPM Detector")
        self.setFixedWidth(430)
        self.setStyleSheet(f"QWidget {{ background-color: {BG}; color: {FG}; }}")

        self._cfg_path  = initial_config or _resource("config/example.yaml")
        self._cfg       = cfg_mod.load(self._cfg_path)
        self._capture   = None   # AudioCapture
        self._detector  = None   # BeatDetector
        self._link      = None   # LinkBridge
        self._q         = queue.Queue(maxsize=20)
        self._blk_count = 0
        self._running   = False

        # Laufendes Peak-Maximum für relative Band-Score-Anzeige
        self._peak_kick = self._peak_snare = self._peak_hihat = 1e-12

        self._build()
        self._populate_devices()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(TICK_MS)

        # Auto-Start: nach einem Tick starten, damit das Fenster erst erscheint
        app_cfg = self._cfg.get("app", {})
        if app_cfg.get("auto_start", True):
            QTimer.singleShot(100, self._start)

    # ── Layout ─────────────────────────────────────────────────────────────────

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(5)

        # ── Steuerleiste ───────────────────────────────────────────────────────
        ctrl = QHBoxLayout()
        self._dev_combo = QComboBox()
        self._dev_combo.setStyleSheet(
            f"QComboBox {{ background-color: {PANEL}; color: {FG}; "
            f"border: 1px solid {DIM}; padding: 3px; border-radius: 3px; }}"
            f"QComboBox QAbstractItemView {{ background-color: {PANEL}; color: {FG}; }}"
        )
        ctrl.addWidget(self._dev_combo, stretch=2)

        self._cfg_btn = QPushButton("Config…")
        self._cfg_btn.setStyleSheet(_btn(PANEL, FG))
        self._cfg_btn.clicked.connect(self._load_config)
        ctrl.addWidget(self._cfg_btn)

        self._start_btn = QPushButton("↺  Restart")
        self._start_btn.setStyleSheet(_btn(GREEN, BG))
        self._start_btn.clicked.connect(self._start)
        ctrl.addWidget(self._start_btn)

        self._stop_btn = QPushButton("■  Stop")
        self._stop_btn.setStyleSheet(_btn(RED, BG))
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop)
        ctrl.addWidget(self._stop_btn)
        root.addLayout(ctrl)

        # Status
        self._status = QLabel(f"Config: {self._cfg_path.name}  —  gestoppt")
        self._status.setStyleSheet(f"color: {DIM}; font-size: 9pt;")
        root.addWidget(self._status)

        root.addWidget(_hsep())

        # ── Tempo ──────────────────────────────────────────────────────────────
        self._state_lbl  = _mono("—")
        self._locked_lbl = _mono("—", size=20, bold=True)
        self._conf_bar   = BarWidget(110, 11, ACCENT)
        self._conf_lbl   = _dim("0.0%")
        self._raw_lbl    = _mono("—")
        self._cor_lbl    = _mono("—")
        self._cand_lbl   = _mono("—", size=9)
        self._cand_lbl.setWordWrap(True)

        root.addLayout(_row("Zustand",    self._state_lbl))

        lr = QHBoxLayout()
        lr.addWidget(_dim("locked_bpm"))
        lr.addStretch()
        lr.addWidget(self._locked_lbl)
        lr.addSpacing(8)
        lr.addWidget(self._conf_bar)
        lr.addSpacing(4)
        lr.addWidget(self._conf_lbl)
        root.addLayout(lr)

        root.addLayout(_row("raw_bpm",       self._raw_lbl))
        root.addLayout(_row("corrected_bpm", self._cor_lbl))
        root.addLayout(_row("candidates",    self._cand_lbl))

        root.addWidget(_hsep())

        # ── Bänder + Phase ─────────────────────────────────────────────────────
        self._kick_bar   = BarWidget(110, 10, RED)
        self._snare_bar  = BarWidget(110, 10, YELLOW)
        self._hihat_bar  = BarWidget(110, 10, CYAN)
        self._kick_val   = _dim("0.0000")
        self._snare_val  = _dim("0.0000")
        self._hihat_val  = _dim("0.0000")
        self._phase_wgt  = PhaseWidget(200, 11)

        root.addLayout(_row_bar("Kick",      self._kick_bar,  self._kick_val))
        root.addLayout(_row_bar("Snare",     self._snare_bar, self._snare_val))
        root.addLayout(_row_bar("Hi-Hat",    self._hihat_bar, self._hihat_val))
        root.addLayout(_row("Beat-Phase",    self._phase_wgt))

        root.addWidget(_hsep())

        # ── Pegel ──────────────────────────────────────────────────────────────
        self._lbar = BarWidget(160, 10, ACCENT)
        self._rbar = BarWidget(160, 10, ACCENT)
        self._lval = _dim("—")
        self._rval = _dim("—")

        root.addLayout(_row_bar("L", self._lbar, self._lval))
        root.addLayout(_row_bar("R", self._rbar, self._rval))

        root.addWidget(_hsep())

        # ── Ableton Link ───────────────────────────────────────────────────────
        link_hdr = QHBoxLayout()
        link_title = QLabel("Ableton Link")
        link_title.setStyleSheet(f"color: {DIM}; font-size: 9pt;")
        link_hdr.addWidget(link_title)
        link_hdr.addStretch()
        self._link_btn = QPushButton("Link AN")
        self._link_btn.setStyleSheet(_btn(PANEL, FG))
        self._link_btn.setCheckable(True)
        self._link_btn.clicked.connect(self._toggle_link)
        link_hdr.addWidget(self._link_btn)
        root.addLayout(link_hdr)

        self._link_peers_lbl   = _mono("—")
        self._link_session_lbl = _mono("—")
        self._link_export_lbl  = _mono("—")
        self._link_offset_lbl  = _dim("—")

        root.addLayout(_row("Peers",        self._link_peers_lbl))
        root.addLayout(_row("Session BPM",  self._link_session_lbl))
        root.addLayout(_row("Export BPM",   self._link_export_lbl))
        root.addLayout(_row("Phase Offset", self._link_offset_lbl))
        root.addSpacing(4)

    # ── Device-Liste ───────────────────────────────────────────────────────────

    def _populate_devices(self):
        self._dev_combo.clear()
        devs = [d for d in get_all_devices() if d["max_input_channels"] > 0]
        cfg_name = cfg_mod.audio_cfg(self._cfg).get("device_name", "").lower()
        preferred = -1
        for i, d in enumerate(devs):
            self._dev_combo.addItem(f"[{d['index']}] {d['name']}", userData=d["index"])
            if cfg_name and cfg_name in d["name"].lower() and preferred < 0:
                preferred = i
        if preferred >= 0:
            self._dev_combo.setCurrentIndex(preferred)

    # ── Config laden ───────────────────────────────────────────────────────────

    def _load_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Config laden", str(self._cfg_path.parent), "YAML (*.yaml *.yml)"
        )
        if not path:
            return
        try:
            self._cfg      = cfg_mod.load(Path(path))
            self._cfg_path = Path(path)
            self._status.setText(f"Config: {self._cfg_path.name}  —  gestoppt")
            self._populate_devices()
        except Exception as e:
            self._status.setText(f"Config-Fehler: {e}")
            self._status.setStyleSheet(f"color: {RED}; font-size: 9pt;")

    # ── Start / Stop ───────────────────────────────────────────────────────────

    def _start(self):
        if self._running:
            return

        dev_idx = self._dev_combo.currentData()
        if dev_idx is None:
            self._status.setText("Kein Eingabegerät ausgewählt.")
            return

        acfg = cfg_mod.audio_cfg(self._cfg)
        bcfg = cfg_mod.bpm_cfg(self._cfg)

        sr       = acfg.get("sample_rate", 44100)
        channels = acfg.get("channels", 2)
        ch_pair  = acfg.get("channel_pair", 0)
        bs       = acfg.get("block_size", 512)

        self._detector = BeatDetector(
            sample_rate=sr,           block_size=bs,
            bpm_min=bcfg.get("bpm_min", 70.0),
            bpm_max=bcfg.get("bpm_max", 180.0),
            kick_lo=bcfg.get("kick_lo", 30.0),
            kick_hi=bcfg.get("kick_hi", 120.0),
            snare_lo=bcfg.get("snare_lo", 180.0),
            snare_hi=bcfg.get("snare_hi", 600.0),
            hihat_lo=bcfg.get("hihat_lo", 2000.0),
            hihat_hi=bcfg.get("hihat_hi", 8000.0),
            kick_weight=bcfg.get("kick_weight", 0.6),
            snare_weight=bcfg.get("snare_weight", 0.25),
            hihat_weight=bcfg.get("hihat_weight", 0.15),
            oss_window_s=bcfg.get("oss_window_s", 8.0),
            onset_threshold=bcfg.get("onset_threshold", 1.4),
            refractory=bcfg.get("refractory", 0.2),
            smoothing=bcfg.get("smoothing", 0.2),
            hysteresis=bcfg.get("hysteresis", 1.0),
            lock_confidence_min=bcfg.get("lock_confidence_min", 0.3),
            relock_confidence_min=bcfg.get("relock_confidence_min", 0.5),
            max_jump_bpm=bcfg.get("max_jump_bpm", 10.0),
            hard_block_jump_bpm=bcfg.get("hard_block_jump_bpm", 15.0),
            relock_windows=bcfg.get("relock_windows", 3),
            hold_seconds=bcfg.get("hold_seconds", 8.0),
        )

        # Peak-Tracking zurücksetzen
        self._peak_kick = self._peak_snare = self._peak_hihat = 1e-12
        self._blk_count = 0

        # Link starten falls konfiguriert
        lcfg = cfg_mod.link_cfg(self._cfg)
        if lcfg.get("enabled", False) and self._link is None:
            bridge = LinkBridge(
                quantum=lcfg.get("quantum", 4),
                update_interval_ms=lcfg.get("update_interval_ms", 200.0),
                tempo_hysteresis=lcfg.get("tempo_hysteresis", 0.5),
            )
            if bridge.start():
                self._link = bridge
                self._link_btn.setChecked(True)
                self._link_btn.setText("Link AN")
                self._link_btn.setStyleSheet(_btn(GREEN, BG))

        det  = self._detector
        q    = self._q

        def on_block(block: np.ndarray) -> None:
            det.process(block)
            self._blk_count += 1

            # Link updaten — self._link dynamisch lesen damit Watchdog-Reconnect wirkt
            lnk = self._link
            if lnk is not None and lnk.available:
                lnk.update(det.locked_bpm, det.beat_phase)

            if self._blk_count % 4 != 0:   # ~12 Snapshots/s bei 44100/512
                return

            l = block[:, 0]
            r = block[:, 1]
            k, s, h = det.band_scores

            snap = Snapshot(
                locked_bpm=det.locked_bpm,
                raw_bpm=det.bpm_raw,
                corrected_bpm=det.bpm_corrected,
                confidence=det.confidence,
                tempo_state=det.tempo_state,
                candidates=det.candidates,
                beat_phase=det.beat_phase,
                kick_score=k, snare_score=s, hihat_score=h,
                rms_l=float(np.sqrt(np.mean(l ** 2))),
                rms_r=float(np.sqrt(np.mean(r ** 2))),
                clip_l=bool(np.any(np.abs(l) >= 1.0)),
                clip_r=bool(np.any(np.abs(r) >= 1.0)),
                link_available=(lnk is not None and lnk.available),
                link_peers=(lnk.peers if lnk is not None else 0),
                link_session_bpm=(lnk.session_tempo if lnk is not None else 0.0),
                link_exported_bpm=(lnk.exported_bpm if lnk is not None else 0.0),
                link_phase_offset=(lnk.phase_offset if lnk is not None else 0.0),
            )
            try:
                q.put_nowait(snap)
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                q.put_nowait(snap)

        try:
            self._capture = AudioCapture(
                device_index=dev_idx,
                sample_rate=sr,
                channels=channels,
                channel_pair=ch_pair,
                block_size=bs,
                on_block=on_block,
            )
            self._capture.start()
        except Exception as e:
            self._status.setText(f"Audio-Fehler: {e}")
            self._status.setStyleSheet(f"color: {RED}; font-size: 9pt;")
            self._capture = None
            self._detector = None
            # Retry ermöglichen
            self._start_btn.setText("↺  Retry")
            self._start_btn.setEnabled(True)
            self._stop_btn.setEnabled(False)
            self._dev_combo.setEnabled(True)
            self._cfg_btn.setEnabled(True)
            return

        self._running = True
        dev_name = self._dev_combo.currentText()
        self._status.setText(f"Läuft: {dev_name}")
        self._status.setStyleSheet(f"color: {GREEN}; font-size: 9pt;")
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._dev_combo.setEnabled(False)
        self._cfg_btn.setEnabled(False)

    def _stop(self):
        if not self._running:
            return
        self._running = False
        if self._capture:
            self._capture.stop()
            self._capture = None
        self._detector = None
        self._stop_link()
        self._status.setText(f"Config: {self._cfg_path.name}  —  gestoppt")
        self._status.setStyleSheet(f"color: {DIM}; font-size: 9pt;")
        self._start_btn.setText("↺  Restart")
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._dev_combo.setEnabled(True)
        self._cfg_btn.setEnabled(True)
        self._reset_display()

    def _stop_link(self):
        if self._link is not None:
            self._link.stop()
            self._link = None
        self._link_btn.setChecked(False)
        self._link_btn.setText("Link AN")
        self._link_btn.setStyleSheet(_btn(PANEL, FG))

    def _toggle_link(self):
        if self._link is not None:
            self._stop_link()
        else:
            lcfg = cfg_mod.link_cfg(self._cfg)
            bridge = LinkBridge(
                quantum=lcfg.get("quantum", 4),
                update_interval_ms=lcfg.get("update_interval_ms", 200.0),
                tempo_hysteresis=lcfg.get("tempo_hysteresis", 0.5),
            )
            if bridge.start():
                self._link = bridge
                self._link_btn.setChecked(True)
                self._link_btn.setText("Link AN")
                self._link_btn.setStyleSheet(_btn(GREEN, BG))
            else:
                self._link_btn.setChecked(False)
                self._link_btn.setText("Link AN")
                self._link_btn.setStyleSheet(_btn(RED, BG))

    # ── Refresh (Main-Thread) ──────────────────────────────────────────────────

    def _tick(self):
        # Link-Watchdog: Bridge automatisch neu starten wenn gestorben
        if self._running and self._link is not None and not self._link.available:
            self._link.stop()
            self._link = None
            lcfg = cfg_mod.link_cfg(self._cfg)
            bridge = LinkBridge(
                quantum=lcfg.get("quantum", 4),
                update_interval_ms=lcfg.get("update_interval_ms", 200.0),
                tempo_hysteresis=lcfg.get("tempo_hysteresis", 0.5),
            )
            if bridge.start():
                self._link = bridge
                self._link_btn.setChecked(True)
                self._link_btn.setText("Link AN")
                self._link_btn.setStyleSheet(_btn(GREEN, BG))

        try:
            snap = None
            while True:
                try:
                    snap = self._q.get_nowait()
                except queue.Empty:
                    break
            if snap is not None:
                self._refresh(snap)
        except Exception:
            pass

    def _refresh(self, s: Snapshot):
        state_col = {"locked": GREEN, "relocking": YELLOW, "searching": RED}
        col = state_col.get(s.tempo_state, FG)

        # Zustand + locked_bpm (gleiche Farbe)
        self._state_lbl.setText(s.tempo_state)
        self._state_lbl.setStyleSheet(
            f"color: {col}; font-family: Courier; font-size: 12pt; font-weight: bold;")
        self._locked_lbl.setText(f"{s.locked_bpm:.1f}" if s.locked_bpm > 0 else "—")
        self._locked_lbl.setStyleSheet(
            f"color: {col}; font-family: Courier; font-size: 20pt; font-weight: bold;")

        # Confidence
        self._conf_bar.set_value(s.confidence)
        self._conf_lbl.setText(f"{s.confidence * 100:.1f}%")

        # raw / corrected
        self._raw_lbl.setText(f"{s.raw_bpm:.1f}" if s.raw_bpm > 0 else "—")
        self._cor_lbl.setText(f"{s.corrected_bpm:.1f}" if s.corrected_bpm > 0 else "—")

        # Kandidaten
        self._cand_lbl.setText(
            "  ".join(f"{b:.1f}({sc:.2f})" for b, sc in s.candidates)
            if s.candidates else "—"
        )

        # Band-Scores mit laufendem Peak-Maximum (langsam decay 0.999)
        self._peak_kick  = max(s.kick_score,  self._peak_kick  * 0.999)
        self._peak_snare = max(s.snare_score, self._peak_snare * 0.999)
        self._peak_hihat = max(s.hihat_score, self._peak_hihat * 0.999)
        self._kick_bar.set_value(s.kick_score  / self._peak_kick)
        self._snare_bar.set_value(s.snare_score / self._peak_snare)
        self._hihat_bar.set_value(s.hihat_score / self._peak_hihat)
        self._kick_val.setText(f"{s.kick_score:.4f}")
        self._snare_val.setText(f"{s.snare_score:.4f}")
        self._hihat_val.setText(f"{s.hihat_score:.4f}")

        # Beat-Phase
        self._phase_wgt.set_phase(s.beat_phase)

        # Pegel: RMS → dBFS → Norm [0,1]
        def db(rms):
            return max(DB_FLOOR, 20.0 * math.log10(rms + 1e-9))

        def norm(d):
            return max(0.0, (d - DB_FLOOR) / (-DB_FLOOR))

        dl, dr = db(s.rms_l), db(s.rms_r)
        self._lbar.set_value(norm(dl), color=RED if s.clip_l else ACCENT)
        self._rbar.set_value(norm(dr), color=RED if s.clip_r else ACCENT)
        clip_l = "  CLIP" if s.clip_l else ""
        clip_r = "  CLIP" if s.clip_r else ""
        self._lval.setText(f"{dl:+.1f} dBFS{clip_l}")
        self._rval.setText(f"{dr:+.1f} dBFS{clip_r}")

        # Ableton Link
        if s.link_available:
            peers_col = GREEN if s.link_peers > 0 else YELLOW
            self._link_peers_lbl.setText(str(s.link_peers))
            self._link_peers_lbl.setStyleSheet(
                f"color: {peers_col}; font-family: Courier; font-size: 12pt;")
            self._link_session_lbl.setText(
                f"{s.link_session_bpm:.1f}" if s.link_session_bpm > 0 else "—")
            self._link_export_lbl.setText(
                f"{s.link_exported_bpm:.1f}" if s.link_exported_bpm > 0 else "—")
            self._link_offset_lbl.setText(f"{s.link_phase_offset:+.3f}")
        else:
            for lbl in (self._link_peers_lbl, self._link_session_lbl,
                        self._link_export_lbl):
                lbl.setText("—")
                lbl.setStyleSheet(f"color: {FG}; font-family: Courier; font-size: 12pt;")
            self._link_offset_lbl.setText("—")

    def _reset_display(self):
        for w in (self._state_lbl, self._locked_lbl, self._raw_lbl,
                  self._cor_lbl, self._cand_lbl, self._lval, self._rval):
            w.setText("—")
        for b in (self._conf_bar, self._kick_bar, self._snare_bar,
                  self._hihat_bar, self._lbar, self._rbar):
            b.set_value(0.0)
        for v in (self._kick_val, self._snare_val, self._hihat_val):
            v.setText("0.0000")
        self._conf_lbl.setText("0.0%")
        self._phase_wgt.set_phase(0.0)
        for lbl in (self._link_peers_lbl, self._link_session_lbl,
                    self._link_export_lbl, self._link_offset_lbl):
            lbl.setText("—")

    def closeEvent(self, event):
        self._stop()
        event.accept()


# ── Widget-Helfer ──────────────────────────────────────────────────────────────

def _hsep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setStyleSheet(f"color: {DIM};")
    return f

def _dim(text: str) -> QLabel:
    l = QLabel(text)
    l.setStyleSheet(f"color: {DIM}; font-size: 9pt;")
    return l

def _mono(text: str, size: int = 12, bold: bool = False) -> QLabel:
    l = QLabel(text)
    w = "bold" if bold else "normal"
    l.setStyleSheet(f"color: {FG}; font-family: Courier; font-size: {size}pt; font-weight: {w};")
    return l

def _row(label: str, widget: QWidget) -> QHBoxLayout:
    h = QHBoxLayout()
    lbl = QLabel(label)
    lbl.setStyleSheet(f"color: {DIM}; font-size: 9pt;")
    lbl.setFixedWidth(92)
    h.addWidget(lbl)
    h.addWidget(widget)
    h.addStretch()
    return h

def _row_bar(label: str, bar: BarWidget, val: QLabel) -> QHBoxLayout:
    h = QHBoxLayout()
    lbl = QLabel(label)
    lbl.setStyleSheet(f"color: {DIM}; font-size: 9pt;")
    lbl.setFixedWidth(50)
    h.addWidget(lbl)
    h.addWidget(bar)
    h.addSpacing(6)
    h.addWidget(val)
    h.addStretch()
    return h

def _btn(bg: str, fg: str) -> str:
    return (
        f"QPushButton {{ background-color: {bg}; color: {fg}; border: none; "
        f"padding: 5px 10px; font-weight: bold; border-radius: 3px; }}"
        f"QPushButton:hover {{ background-color: {ACCENT}; color: {BG}; }}"
        f"QPushButton:disabled {{ background-color: {DIM}; color: {BG}; opacity: 0.5; }}"
    )


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="BPM Detector GUI")
    ap.add_argument("--config", "-c", type=Path, default=None,
                    help="YAML-Config (Standard: config/example.yaml)")
    a = ap.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = BpmGui(initial_config=a.config)
    win.show()
    sys.exit(app.exec())
