"""Sound-to-Light Bridge — schickt Kick/Snare/Hihat-Werte an Grand Vision.

Läuft in einem eigenen Hintergrund-Thread mit fixem Intervall, nie aus dem
Audio-Callback heraus: ein langsamer/fehlschlagender HTTP-Request darf nie
zu Audio-Dropouts führen (gleiches Prinzip wie link_bridge.py).

Schickt pro Band ZWEI Werte, für unterschiedliche Lichteffekte:
  - kick/snare/hihat: andauernde Energie, peak-normalisiert (0..1ish) - für
    VU-Meter-artige Effekte, die der Lautstärke kontinuierlich folgen sollen.
  - kickHit/snareHit/hihatHit: Transienten-Stärke (siehe onset_scores in
    BeatDetector), schon 0..1-normiert - für Effekte, die auf echte Treffer
    reagieren sollen statt auf eine durchlaufende Basslinie/Gesang im
    selben Frequenzband.
Zusätzlich `level`: peak-normalisierte Broadband-Lautstärke (RMS des ganzen
Mixes, kein Frequenzband) - für einen VU-Meter-Effekt, der sich wie
Gesamtlautstärke anfühlt statt wie ein einzelnes Drum-Band.

Außerdem `kickTrigger`/`snareTrigger`/`clapTrigger`: saubere boolesche
Ein-Block-Ereignisse (adaptive Schwelle, Hold-off/Cooldown) statt der
abklingenden Hit-Werte - für Effekte, die auf einen klaren, einmaligen
Trigger statt auf eine 0..1-Pulskurve reagieren sollen.

Optional (Ableton-Link-Beat-Eingangsmodus, siehe beat_scheduler.py):
`beatKick` - Ein-Block-Ereignis aus dem externen Ableton-Link-Beat-Zähler
statt der Audio-Kick-Erkennung, `beatInBar` - Position 1..quantum im Takt.
Nur befüllt wenn ein beat_scheduler übergeben wurde, sonst immer False/0 -
ändert nichts an kickTrigger/etc., die weiterhin aus der Audio-Erkennung
kommen.

Außerdem `bpm`: bpmdetects eigene Audio-BPM-Schätzung (BeatDetector.locked_bpm,
sonst .bpm als Näherung), unabhängig von Ableton Link - für Grand Vision als
Tempo-Referenz, wenn Links Session gerade keine echte externe Quelle hat
(siehe fallback_to_audio in beat_scheduler.py).
"""

import json
import threading
import urllib.error
import urllib.request


class SoundlightBridge:
    def __init__(self, detector, url: str, interval_ms: float = 33.0, beat_scheduler=None) -> None:
        self._detector = detector
        self._url = url
        self._interval = interval_ms / 1000.0
        self._running = False
        self._thread = None
        self._peak = {"kick": 1e-9, "snare": 1e-9, "hihat": 1e-9, "clap": 1e-9, "level": 1e-9}
        # Optional: LinkBeatScheduler (beat_scheduler.py) - liefert beatKick/
        # beatInBar aus Ableton Link statt der Audio-Kick-Erkennung. None
        # (Default) = rückwärtskompatibel, Felder bleiben False/0, exakt wie
        # vor dieser Ergänzung.
        self._beat_scheduler = beat_scheduler
        # Sofort-Trigger für beat_kick statt auf den nächsten Timer-Tick zu
        # warten (siehe notify_beat_kick) - live beobachtet: der normale
        # ~33ms-Intervall-Loop allein addiert sich mit push_interval_ms und
        # Grand Visions eigenem DMX-Tick zu spürbarem Nachlaufen.
        self._wake_event = threading.Event()

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="soundlight-bridge")
        self._thread.start()

    def notify_beat_kick(self) -> None:
        """Vom Audio-Callback-Thread aufzurufen (run_bpm's on_block, direkt
        nach beat_scheduler.update()), wenn ein Beat erkannt wurde - weckt
        den Sende-Thread sofort statt bis zu einem vollen update_interval_ms
        zu warten. Nur ein event.set(), kein I/O, blockiert nie."""
        self._wake_event.set()

    def stop(self) -> None:
        self._running = False

    def _build_payload(self) -> bytes:
        """Baut die JSON-Payload aus dem aktuellen Detector-Zustand - aus
        _loop() herausgezogen, damit sie ohne Thread/HTTP isoliert testbar
        ist (siehe tests/test_triggers.py)."""
        kick, snare, hihat, clap = self._detector.band_scores
        raw = {"kick": kick, "snare": snare, "hihat": hihat, "clap": clap, "level": self._detector.overall_score}
        for band, val in raw.items():
            self._peak[band] = max(val, self._peak[band] * 0.999)

        kick_hit, snare_hit, hihat_hit, clap_hit = self._detector.onset_scores

        return json.dumps({
            "kick": round(raw["kick"] / self._peak["kick"], 4),
            "snare": round(raw["snare"] / self._peak["snare"], 4),
            "hihat": round(raw["hihat"] / self._peak["hihat"], 4),
            "clap": round(raw["clap"] / self._peak["clap"], 4),
            "level": round(raw["level"] / self._peak["level"], 4),
            "kickHit": round(kick_hit, 4),
            "snareHit": round(snare_hit, 4),
            "hihatHit": round(hihat_hit, 4),
            "clapHit": round(clap_hit, 4),
            # Saubere boolesche Events (Mean+StdDev-Schwelle, Snare-
            # Koinzidenz, Hold-off) statt der obigen abklingenden
            # 0..1-Werte - siehe BeatDetector.kick_trigger/snare_trigger/
            # clap_trigger. Rueckwaertskompatibel ergaenzt: bestehende
            # Felder (kickTrigger/snareTrigger) unveraendert, clapTrigger
            # ist ein reiner Zusatz, kein Ersatz-Eventformat.
            "kickTrigger": self._detector.kick_trigger,
            "snareTrigger": self._detector.snare_trigger,
            "clapTrigger": self._detector.clap_trigger,
            "phase": round(self._detector.beat_phase, 4),
            # Ableton-Link-Beat-Eingangsmodus (beat_scheduler.py) - rein
            # additiv, rückwärtskompatibel: ohne beat_scheduler immer
            # False/0, ändert nichts an kickTrigger/etc. aus der
            # Audio-Erkennung, die unverändert weiterläuft.
            # consume_beat_kick() statt der beat_kick-Property: drain-on-read,
            # damit dieses ~33ms-Poll-Intervall keinen Beat verpasst, der nur
            # für einen einzelnen ~11ms-Audioblock aktiv war (siehe
            # beat_scheduler.py) - live beobachtet, ohne das war beatKick
            # praktisch nie True.
            "beatKick": self._beat_scheduler.consume_beat_kick() if self._beat_scheduler else False,
            "beatInBar": self._beat_scheduler.beat_in_bar if self._beat_scheduler else 0,
            # Eigene Audio-BPM-Schätzung (Kalman/Autokorrelation, siehe
            # BeatDetector.locked_bpm/.bpm) - unabhängig von Ableton Link,
            # damit Grand Vision ein echtes Tempo hat, auch wenn Links
            # Session gerade nur mit sich selbst spricht (siehe
            # fallback_to_audio in beat_scheduler.py). locked_bpm ist die
            # bestätigte/stabile Schätzung (0 bis genug Konfidenz da ist),
            # bis dahin die laufende geglättete Schätzung als Näherung.
            "bpm": round(self._detector.locked_bpm if self._detector.locked_bpm > 0 else self._detector.bpm, 2),
        }).encode("utf-8")

    def _loop(self) -> None:
        while self._running:
            payload = self._build_payload()

            req = urllib.request.Request(
                self._url, data=payload, method="POST",
                headers={"Content-Type": "application/json"},
            )
            try:
                urllib.request.urlopen(req, timeout=0.5)
            except (urllib.error.URLError, OSError):
                pass  # Grand Vision gerade nicht erreichbar - naechster Tick versucht es erneut

            # Wie time.sleep(self._interval), aber unterbrechbar: wenn
            # notify_beat_kick() währenddessen aufgerufen wird, endet das
            # Warten sofort statt erst nach dem vollen Intervall - der
            # nächste Schleifendurchlauf sendet dann direkt (inkl. dem
            # gerade erkannten Beat über consume_beat_kick() in
            # _build_payload). Kein Datenverlust, falls das Event knapp
            # nach dem Zurücksetzen erneut gesetzt wird: consume_beat_kick()
            # bleibt unabhängig davon korrekt (siehe beat_scheduler.py) -
            # in dem seltenen Fall wird der Beat nur nicht ganz so sofort
            # zugestellt, nie verloren.
            self._wake_event.wait(timeout=self._interval)
            self._wake_event.clear()
