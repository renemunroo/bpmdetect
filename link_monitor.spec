# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec für Link Monitor GUI."""

from pathlib import Path

ROOT = Path(SPECPATH)
LINK = ROOT / "link"

a = Analysis(
    [str(LINK / "link_monitor.py")],
    pathex=[],
    binaries=[],
    datas=[
        (str(LINK / "monitor_bridge.js"), "link"),
        (str(LINK / "package.json"),      "link"),
        (str(LINK / "node_modules"),      "link/node_modules"),
    ],
    hiddenimports=["yaml"],
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
    name="Link Monitor",
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
    name="Link Monitor",
)

app = BUNDLE(
    coll,
    name="Link Monitor.app",
    icon=None,
    bundle_identifier="de.bpmdetect.linkmonitor",
    info_plist={
        "NSHighResolutionCapable": True,
        "LSUIElement": False,
    },
)
