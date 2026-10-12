#!/usr/bin/env python3
"""Runde 4 - wertet eine oder mehrere live_monitor-CSV-Dateien aus einem
A/B-Testlauf (kalman_kick_std_factor) aus und druckt die Vergleichskennzahlen.

Reine Auswertung der von live_monitor.py geschriebenen CSV (siehe
_CSV_COLUMNS dort) - liest nur, ändert nichts an Erkennungslogik oder
Live-Betrieb.

Nutzung:
    python tools/analyze_ab_run.py logs/ab_std25.csv logs/ab_std20.csv logs/ab_std15.csv
"""
import csv
import statistics as st
import sys
from collections import Counter


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def analyze(path: str) -> dict:
    status_rows = []
    kick_events = 0
    snare_events = 0
    clap_events = 0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            t = row["type"]
            if t == "status" and row.get("kick_envelope"):
                status_rows.append(row)
            elif t == "kick":
                kick_events += 1
            elif t == "snare":
                snare_events += 1
            elif t == "clap":
                clap_events += 1

    envelopes = [_f(r["kick_envelope"]) for r in status_rows if _f(r["kick_envelope"]) is not None]
    thresholds = [_f(r["kick_threshold"]) for r in status_rows if _f(r["kick_threshold"]) is not None]
    ratios = [_f(r["kick_rise_ratio"]) for r in status_rows if _f(r["kick_rise_ratio"]) is not None]
    confidences = [_f(r["confidence"]) for r in status_rows if _f(r["confidence"]) is not None]
    bpms = [_f(r["bpm"]) for r in status_rows if _f(r["bpm"])]
    reasons = Counter(r["reject_reason"] for r in status_rows if r["reject_reason"])
    candidates = sum(1 for r in status_rows if r["reject_reason"] not in ("threshold_not_reached", "insufficient_rise", ""))
    sharpness_true = sum(1 for r in status_rows if r.get("kick_is_sharp") == "True")
    hpss_true = sum(1 for r in status_rows if r.get("kick_hpss_confirms") == "True")

    def stats(vals):
        if not vals:
            return (None, None, None)
        return (min(vals), max(vals), st.mean(vals))

    env_min, env_max, env_mean = stats(envelopes)
    thr_min, thr_max, thr_mean = stats(thresholds)
    ratio_min, ratio_max, ratio_mean = stats(ratios)

    return {
        "file": path,
        "status_rows": len(status_rows),
        "kick_candidates": candidates,
        "kick_triggers": kick_events,
        "snare_triggers": snare_events,
        "clap_triggers": clap_events,
        "envelope_min": env_min, "envelope_max": env_max, "envelope_mean": env_mean,
        "threshold_min": thr_min, "threshold_max": thr_max, "threshold_mean": thr_mean,
        "rise_ratio_min": ratio_min, "rise_ratio_max": ratio_max, "rise_ratio_mean": ratio_mean,
        "sharp_true_pct": (100 * sharpness_true / len(status_rows)) if status_rows else None,
        "hpss_true_pct": (100 * hpss_true / len(status_rows)) if status_rows else None,
        "confidence_mean": st.mean(confidences) if confidences else None,
        "bpm_last": bpms[-1] if bpms else None,
        "top_reject_reasons": reasons.most_common(4),
    }


def _fmt(v, nd=3):
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        return
    results = [analyze(p) for p in sys.argv[1:]]

    for r in results:
        print(f"\n{'=' * 70}\n{r['file']}")
        print(f"  Status-Snapshots       : {r['status_rows']}")
        print(f"  Kick-Kandidaten        : {r['kick_candidates']}  (über Basis-Schwelle, egal ob getriggert)")
        print(f"  Kick-Trigger           : {r['kick_triggers']}")
        print(f"  Snare/Clap-Trigger     : {r['snare_triggers']} / {r['clap_triggers']}")
        print(f"  Envelope   min/max/avg : {_fmt(r['envelope_min'])} / {_fmt(r['envelope_max'])} / {_fmt(r['envelope_mean'])}")
        print(f"  Threshold  min/max/avg : {_fmt(r['threshold_min'])} / {_fmt(r['threshold_max'])} / {_fmt(r['threshold_mean'])}")
        print(f"  env/thr    min/max/avg : {_fmt(r['rise_ratio_min'])} / {_fmt(r['rise_ratio_max'])} / {_fmt(r['rise_ratio_mean'])}")
        print(f"  is_sharp erfüllt       : {_fmt(r['sharp_true_pct'], 1)}%  |  hpss_confirms erfüllt: {_fmt(r['hpss_true_pct'], 1)}%")
        print(f"  Confidence (Mittel)    : {_fmt(r['confidence_mean'])}")
        print(f"  BPM (letzter Wert)     : {_fmt(r['bpm_last'], 1)}")
        print(f"  Häufigste Reject-Gründe: {r['top_reject_reasons']}")

    if len(results) > 1:
        print(f"\n{'=' * 70}\nVergleich:")
        cols = ["file", "kick_candidates", "kick_triggers", "envelope_mean", "threshold_mean", "rise_ratio_mean"]
        print("  " + "  ".join(f"{c:<22}" for c in cols))
        for r in results:
            print("  " + "  ".join(f"{_fmt(r[c]):<22}" for c in cols))


if __name__ == "__main__":
    main()
