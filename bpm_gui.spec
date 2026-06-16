# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec für BPM Detector GUI."""

import sys
from pathlib import Path

ROOT = Path(SPECPATH)
SRC  = ROOT / "src"
LINK = ROOT / "link"
CFG  = ROOT / "config"

# portaudio liegt im venv, nicht im System
PORTAUDIO = ROOT / ".venv/lib/python3.9/site-packages/_sounddevice_data/portaudio-binaries/libportaudio.dylib"

a = Analysis(
    [str(ROOT / "gui" / "bpm_gui.py")],
    pathex=[str(SRC)],
    binaries=[
        (str(PORTAUDIO), "_sounddevice_data/portaudio-binaries"),
    ],
    datas=[
        (str(CFG), "config"),
        (str(LINK / "bridge.js"),        "link"),
        (str(LINK / "package.json"),     "link"),
        (str(LINK / "node_modules"),     "link/node_modules"),
    ],
    hiddenimports=[
        "sounddevice",
        "numpy",
        "yaml",
        "beat_detector",
        "capture",
        "config",
        "device_list",
        "link_bridge",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "PIL"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="BPM Detector",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="BPM Detector",
)

app = BUNDLE(
    coll,
    name="BPM Detector.app",
    icon=None,
    bundle_identifier="de.bpmdetect.bpmdetector",
    info_plist={
        "NSMicrophoneUsageDescription": "BPM Detector benötigt Audio-Zugriff.",
        "LSUIElement": False,
        "NSHighResolutionCapable": True,
    },
)
