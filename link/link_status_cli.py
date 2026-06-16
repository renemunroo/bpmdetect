#!/usr/bin/env python3
"""
Ableton Link Status — CLI-Fallback ohne GUI.
Nutzt dieselbe Bridge-Klasse wie link_monitor.py.

Start:
    cd /Users/rene/bpmdetect
    .venv/bin/python link/link_status_cli.py [--tempo 120] [--quantum 4]

Tastenbefehle (nur im Terminal, wenn kein Pipe):
    q / Ctrl-C  → beenden
    e           → Link ein/aus
    +/-         → Tempo ±1 BPM
"""
import argparse
import sys
import time

# Bridge aus link_monitor importieren
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from link_monitor import Bridge


def _ansi(code: str, text: str) -> str:
    return f"\x1b[{code}m{text}\x1b[0m"


def render(s: dict, available: bool):
    enabled = s["enabled"]
    peers   = s["peers"]
    tempo   = s["tempo"]
    beat    = s["beat"]
    phase   = s["phase"]
    playing = s.get("playing", False)

    ph_bar_w = 32
    pos = max(0, min(ph_bar_w - 1, int(phase * ph_bar_w)))
    ph_bar = "·" * pos + "●" + "·" * (ph_bar_w - pos - 1)

    status_str = (
        _ansi("32", "verbunden") if available else _ansi("31", "offline")
    )
    enabled_str = (
        _ansi("32", "AN ") if enabled else _ansi("31", "AUS")
    )
    peers_str = (
        _ansi("32", str(peers)) if peers > 0 else _ansi("33", f"{peers} (solo)")
    )
    playing_str = _ansi("32", "▶ ja") if playing else _ansi("90", "◼ nein")

    lines = [
        f"  Bridge   : {status_str}",
        f"  Link     : {enabled_str}   Peers {peers_str}",
        f"  Tempo    : {tempo:7.2f} BPM",
        f"  Beat     : {beat:.3f}",
        f"  Phase    : {phase:.3f}  [{ph_bar}]",
        f"  Playing  : {playing_str}",
        f"  ──────────────────────────────────────────────",
        "  Ctrl-C = beenden  |  e = Link  |  +/- = Tempo",
    ]
    # Cursor nach oben springen und überschreiben
    sys.stdout.write(f"\x1b[{len(lines)}A" + "\n".join(lines) + "\n")
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser(description="Ableton Link Status CLI")
    ap.add_argument("--tempo",   type=float, default=120.0)
    ap.add_argument("--quantum", type=int,   default=4)
    a = ap.parse_args()

    bridge = Bridge()
    ok = bridge.start(a.tempo, a.quantum)
    if not ok:
        print("Bridge konnte nicht gestartet werden. node im PATH?", file=sys.stderr)
        sys.exit(1)

    # Platzhalter-Zeilen für das Überschreiben
    print("\n" * 8, end="")

    # Tastatur-Eingabe (nur im echten Terminal)
    use_keys = sys.stdin.isatty()
    if use_keys:
        import tty, termios, select as sel
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setraw(sys.stdin.fileno())
            while True:
                s = bridge.drain()
                render(s, bridge.available)
                r, _, _ = sel.select([sys.stdin], [], [], 0.12)
                if r:
                    ch = sys.stdin.read(1)
                    if ch in ("q", "\x03"):  # q oder Ctrl-C
                        break
                    elif ch == "e":
                        bridge.send("set_enabled", enabled=not s["enabled"])
                    elif ch == "+":
                        bridge.send("set_tempo", bpm=min(300.0, s["tempo"] + 1))
                    elif ch == "-":
                        bridge.send("set_tempo", bpm=max(20.0, s["tempo"] - 1))
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    else:
        # Pipe-Modus: nur Ausgabe, kein Tastatur-Input
        try:
            while True:
                s = bridge.drain()
                render(s, bridge.available)
                time.sleep(0.12)
        except KeyboardInterrupt:
            pass

    bridge.stop()
    print("\nLink Monitor beendet.")


if __name__ == "__main__":
    main()
