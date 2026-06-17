# bpmdetect

Realtime BPM Detector & Ableton Link Bridge

Ein hochpräziser Echtzeit-BPM-Detektor, der Audiosignale direkt am Eingang analysiert. 
Durch die gezielte Erkennung von Kick, Snare und Hi-Hat extrahiert das Tool das Tempo und wandelt dieses unmittelbar 
in ein Ableton Link-Signal um.

## Funktionsweise

1. **Spectral Flux OSS** — gewichtete Summe aus drei Drum-Bändern (Kick / Snare / Hi-Hat)
2. **Autokorrelation** über ein gleitendes 8-Sekunden-Fenster → stabile BPM-Kandidaten
3. **Oktav-Korrektur** — verhindert Half-Time / Double-Time-Fehler
4. **Tempo-Hold-State-Machine** — `locked_bpm` springt nicht bei kurzen Pausen oder Rauschen
5. **Energy-Gate** — bei beatlosen Passagen fällt die Confidence sofort, sodass kein falsches BPM-Lock entsteht

---

## Voraussetzungen

- macOS (Apple Silicon oder Intel)
- Python 3.9+
- [BlackHole 16ch](https://github.com/ExistentialAudio/BlackHole) (für Loopback-Routing)
- Node.js ≥ 18 (nur für Ableton-Link-Integration)

## Installation

```bash
git clone <repo>
cd bpmdetect

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional: Ableton-Link-Bridge
cd link && npm install && cd ..
```

---

## Verwendung

Alle Befehle aus dem Projektverzeichnis mit aktiviertem venv:

### Audio-Geräte anzeigen

```bash
python src/main.py list
python src/main.py list --filter "BlackHole"
python src/main.py list --verbose
```

### Pegel-Monitor (kein BPM)

```bash
python src/main.py monitor
python src/main.py --config config/example.yaml monitor
```

### BPM-Detection starten

```bash
python src/main.py bpm
python src/main.py --config config/example.yaml bpm
```

Beenden mit **Strg+C**.

---

## Konfiguration

Die Konfigurationsdatei liegt unter `config/example.yaml`. Alle Parameter sind dort dokumentiert. Wichtigste Einstellungen:

| Sektion | Schlüssel | Beschreibung |
|---------|-----------|--------------|
| `audio` | `device_name` | Name oder Teilstring des Eingabegeräts |
| `audio` | `fallback_to_default` | Systemstandard verwenden falls Gerät fehlt |
| `bpm` | `bpm_min` / `bpm_max` | Zulässiger Tempo-Bereich |
| `bpm` | `lock_confidence_min` | Mindest-Confidence für ersten Lock (0–1) |
| `bpm` | `hold_seconds` | Sekunden bis Lock bei Low-Confidence aufgegeben wird |
| `bpm` | `max_jump_bpm` | Sprünge ≤ diesem Wert werden per EMA absorbiert |
| `bpm` | `hard_block_jump_bpm` | Sprünge > diesem Wert werden komplett ignoriert |
| `bpm` | `relock_windows` | Wie viele Analyse-Fenster ein neues BPM stabil sein muss |
| `link` | `enabled` | Ableton-Link-Integration aktivieren |

### Empfohlene Werte nach Musikstil

| Stil | `bpm_min` | `bpm_max` | Anmerkung |
|------|-----------|-----------|-----------|
| Techno / House | 120 | 160 | Standard |
| Hip-Hop / Trap | 70 | 110 | — |
| Drum & Bass | 160 | 180 | — |
| Allgemein | 70 | 180 | konservativster Bereich |

---

## Terminal-Ausgabe

```
  tempo_state     : locked              ← searching / locked / relocking
  locked_bpm      : 128.0  [████░░░░░]  Conf  82.5%
  raw_bpm         : 128.3              ← bester ACF-Peak, vor Oktav-Korrektur
  corrected_bpm   : 128.3              ← nach Half/Double-Time-Prüfung
  candidate_bpms  : 128.3(0.91)  64.1(0.44)  256.6(0.21)
  relock_candidate: —                  ← aktiv wenn Relock läuft
  Beat-Phase      : [·····●··········]
  ───────────────────────────────────────────────────────
  kick_score      : [▪▪▪▪▪▪▪·····]   12.4800
  snare_score     : [▪▪▪▪····]         5.2100
  hihat_score     : [▪▪▪·····]         2.8300
  ───────────────────────────────────────────────────────
  link_enabled    : no  Peers —
  ...
  L [████████████████████░░░░░░░░░░░░░░░░░░░░]      -12.3 dBFS
  R [████████████████████░░░░░░░░░░░░░░░░░░░░]      -12.4 dBFS
```

### Tempo-Zustände

| Zustand | Farbe | Bedeutung |
|---------|-------|-----------|
| `searching` | rot | Kein stabiles BPM erkannt, `locked_bpm` = 0 |
| `locked` | grün | Stabiles BPM gefunden, `locked_bpm` wird glatt nachgeführt |
| `relocking` | gelb | Kandidat für neues BPM wird geprüft (noch nicht übernommen) |

### Confidence

Gibt an, wie deutlich der ACF-Peak über dem Hintergrausch der Lag-Region liegt:
- `0.0` — kein Rhythmus erkennbar (Stille, Rauschen, Break)
- `0.3+` — ausreichend für ersten Lock (`lock_confidence_min`)
- `0.5+` — ausreichend für Relock (`relock_confidence_min`)
- `1.0` — sehr klares, stabiles Tempo

---

## BlackHole-Setup (Loopback)

1. BlackHole 16ch installieren
2. In macOS **Audio MIDI Setup**: Multi-Output-Gerät anlegen aus BlackHole + Lautsprecher
3. Dieses Multi-Output als Systemausgabe setzen
4. In `config/example.yaml`: `device_name: "BlackHole 16ch"`

Alternativ: DJ-Interface als Eingabegerät direkt verwenden (`device_name: "DJM-A9"` o.ä.).

---

## Ableton Link (optional)

Erfordert Node.js und `npm install` im `link/`-Ordner.

```yaml
# config/example.yaml
link:
  enabled: true
  quantum: 4
  update_interval_ms: 200
  tempo_hysteresis: 0.5
```

Das `locked_bpm` wird rate-limitiert und hysterese-gefiltert an die Link-Session exportiert.

### Link-Monitor GUI

Separates Fenster, das den Ableton-Link-Status anzeigt (unabhängig vom BPM-Detector):

```bash
python link/link_monitor.py --tempo 128 --quantum 4
```

Benötigt PySide6 (`pip install PySide6`) und Node.js.

---

## Projektstruktur

```
bpmdetect/
├── src/
│   ├── main.py           # CLI-Einstiegspunkt (list / monitor / bpm)
│   ├── beat_detector.py  # BeatDetector, BpmDisplay, run_bpm()
│   ├── capture.py        # AudioCapture, LevelMeter, resolve_device()
│   ├── config.py         # YAML-Loader
│   ├── device_list.py    # sounddevice-Geräteliste
│   └── link_bridge.py    # Ableton-Link-IPC (Python ↔ Node.js)
├── link/
│   ├── bridge.js         # Node.js-Bridge für beat_detector
│   ├── monitor_bridge.js # Node.js-Bridge für link_monitor GUI
│   ├── link_monitor.py   # PySide6-GUI für Link-Status
│   ├── link_status_cli.py# CLI-Fallback für Link-Status
│   ├── status.js         # Node.js-CLI für Link-Status
│   └── package.json
├── config/
│   └── example.yaml      # Vollständig dokumentierte Konfiguration
└── requirements.txt
```
