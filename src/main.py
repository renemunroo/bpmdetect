#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import config as cfg_mod
from device_list import get_all_devices, filter_devices


# ── list ──────────────────────────────────────────────────────────────────────

def print_device(dev: dict, verbose: bool) -> None:
    marker = ""
    if dev["is_default_input"] and dev["is_default_output"]:
        marker = " [default in+out]"
    elif dev["is_default_input"]:
        marker = " [default in]"
    elif dev["is_default_output"]:
        marker = " [default out]"

    channels = f"in={dev['max_input_channels']} out={dev['max_output_channels']}"
    print(f"  [{dev['index']:2d}] {dev['name']}{marker}")
    if verbose:
        print(f"       API: {dev['hostapi']}  |  {channels}  |  {dev['default_samplerate']} Hz")


def cmd_list(args: argparse.Namespace) -> None:
    devices = get_all_devices()
    if args.filter:
        devices = filter_devices(devices, args.filter)
        if not devices:
            print(f'Kein Gerät gefunden das "{args.filter}" enthält.', file=sys.stderr)
            sys.exit(1)

    inputs = [d for d in devices if d["max_input_channels"] > 0]
    outputs = [d for d in devices if d["max_output_channels"] > 0 and d["max_input_channels"] == 0]

    print(f"\n=== Eingabegeräte ({len(inputs)}) ===")
    for dev in inputs:
        print_device(dev, args.verbose)

    print(f"\n=== Nur-Ausgabegeräte ({len(outputs)}) ===")
    for dev in outputs:
        print_device(dev, args.verbose)
    print()


# ── monitor ───────────────────────────────────────────────────────────────────

def cmd_monitor(args: argparse.Namespace) -> None:
    from capture import run_monitor
    cfg = cfg_mod.load(Path(args.config) if args.config else None)
    run_monitor(cfg_mod.audio_cfg(cfg), cfg_mod.monitor_cfg(cfg))


# ── bpm ───────────────────────────────────────────────────────────────────────

def cmd_bpm(args: argparse.Namespace) -> None:
    from beat_detector import run_bpm
    cfg = cfg_mod.load(Path(args.config) if args.config else None)

    # Runde 4: temporärer Override für einen A/B-Testlauf, OHNE club.yaml
    # dauerhaft zu ändern - bpm_cfg()/live_monitor_cfg() liefern das direkt
    # aus der Datei geladene dict, das hier NICHT mutiert wird (sonst würde
    # ein erneutes cfg_mod.load() im selben Prozess denselben dict wieder
    # zurückgeben - hier aber ohnehin pro Lauf ein frischer Prozess, sicherheitshalber
    # trotzdem eine Kopie).
    bpm_cfg = dict(cfg_mod.bpm_cfg(cfg))
    if args.kalman_kick_std_factor is not None:
        bpm_cfg["kalman_kick_std_factor"] = args.kalman_kick_std_factor
        print(f"[override] kalman_kick_std_factor = {args.kalman_kick_std_factor} (club.yaml unverändert)")

    live_cfg = dict(cfg_mod.live_monitor_cfg(cfg))
    if args.label:
        live_cfg.setdefault("enabled", True)
        live_cfg["log_path"] = args.live_log_path or f"logs/ab_{args.label}.log"
        live_cfg["csv_path"] = args.live_csv_path or f"logs/ab_{args.label}.csv"
    else:
        if args.live_log_path:
            live_cfg["log_path"] = args.live_log_path
        if args.live_csv_path:
            live_cfg["csv_path"] = args.live_csv_path

    link_cfg = dict(cfg_mod.link_cfg(cfg))
    if args.link_mode:
        if args.link_mode == "off":
            link_cfg["enabled"] = False
        else:
            link_cfg["enabled"] = True
            link_cfg["mode"] = args.link_mode
        print(f"[override] link.mode = {args.link_mode} (club.yaml unverändert)")

    run_bpm(
        cfg_mod.audio_cfg(cfg),
        cfg_mod.monitor_cfg(cfg),
        bpm_cfg,
        link_cfg,
        cfg_mod.soundlight_cfg(cfg),
        live_cfg,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="bpmdetect – Audio-Device-Tool für macOS"
    )
    parser.add_argument(
        "--config", "-c", metavar="FILE",
        help="Pfad zur YAML-Konfigurationsdatei (Standard: config/example.yaml)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # list
    p_list = sub.add_parser("list", help="Alle Audio-Devices auflisten")
    p_list.add_argument("--filter", "-f", metavar="NAME",
                        help="Nur Geräte anzeigen die NAME enthalten")
    p_list.add_argument("--verbose", "-v", action="store_true",
                        help="Zusätzliche Details anzeigen")
    p_list.set_defaults(func=cmd_list)

    # monitor
    p_mon = sub.add_parser("monitor", help="Echtzeit-Pegelanzeige starten")
    p_mon.set_defaults(func=cmd_monitor)

    # bpm
    p_bpm = sub.add_parser("bpm", help="Beat-Detection und BPM-Schätzung starten")
    p_bpm.add_argument(
        "--kalman-kick-std-factor", type=float, default=None, metavar="FACTOR",
        help="Testweiser Override für kalman_kick_std_factor (Default aus club.yaml bleibt unverändert, "
             "nur dieser Prozess-Lauf nutzt den neuen Wert) - für A/B-Testrunden."
    )
    p_bpm.add_argument(
        "--label", default=None, metavar="NAME",
        help="Kennzeichnung dieses Testlaufs - schreibt automatisch nach logs/ab_<NAME>.log/.csv "
             "(aktiviert live_monitor, falls in club.yaml noch aus) statt der club.yaml-Pfade."
    )
    p_bpm.add_argument("--live-log-path", default=None, metavar="PATH", help="Override für live_monitor.log_path")
    p_bpm.add_argument("--live-csv-path", default=None, metavar="PATH", help="Override für live_monitor.csv_path")
    p_bpm.add_argument(
        "--link-mode", choices=["input", "output", "off"], default=None, metavar="MODE",
        help="Testweiser Override für link.mode ('off' setzt zusätzlich link.enabled=false) - "
             "club.yaml bleibt unverändert. Für schnelles Umschalten ohne Config-Edit, z.B. wenn "
             "keine echte externe Link-Quelle da ist und bpmdetects eigener Link-Peer nur mit "
             "einem anderen lokalen Prozess (Grand Vision) auf ein Fake-Tempo einrastet."
    )
    p_bpm.set_defaults(func=cmd_bpm)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
