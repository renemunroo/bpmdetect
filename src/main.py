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
    run_bpm(
        cfg_mod.audio_cfg(cfg),
        cfg_mod.monitor_cfg(cfg),
        cfg_mod.bpm_cfg(cfg),
        cfg_mod.link_cfg(cfg),
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
    p_bpm.set_defaults(func=cmd_bpm)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
