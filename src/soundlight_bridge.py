"""Soundlight-Brücke — pusht BeatDetector-Werte per HTTP an aura-lc.

aura-lc (server.js, POST /api/soundlight/levels -> engine.setSoundLevels())
erwartet ein JSON-Objekt mit (siehe dort, src/dmxEngine.js):
  kick/snare/hihat/clap   sustained Energie 0..1
  level                   Breitband-RMS 0..1 (hier NICHT geliefert - siehe
                          unten, kein entsprechender Messwert vorhanden)
  kickHit/snareHit/hihatHit/clapHit   transiente Onset-Stärke 0..1
  kickTrigger/snareTrigger/clapTrigger  saubere Ein-Block-Ereignisse
  phase/beatKick/beatInBar  aus Link abgeleitet (hier NICHT geliefert -
                          bpmdetect hat kein beat_scheduler.py-Äquivalent)
  bpm                     BeatDetector.locked_bpm

bpmdetect kennt kein Clap-Band und keine Breitband-RMS (die Terminal-
Anzeige nutzt dBFS, kein 0..1-Peak-Pegel) - `clap`/`clapHit`/`clapTrigger`/
`level`/`phase`/`beatKick`/`beatInBar` werden deshalb bewusst NICHT
mitgeschickt statt erfunden zu werden; der Server setzt dafür seine
eigenen, dokumentierten Defaults (0/false, siehe setSoundLevels()).

Der HTTP-POST läuft NIE auf dem Audio-Callback-Thread (sounddevice ruft
on_block() dort auf, ein blockierender Netzwerk-Call dort riskiert Audio-
Dropouts) - update() reiht nur eine Queue-Operation ein (Queue-Größe 1:
ein wartender, noch nicht gesendeter Payload wird durch einen neueren
ersetzt statt sich aufzustauen), ein einzelner Hintergrund-Thread sendet.
"""

import json
import queue
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Optional

_SEND_TIMEOUT_S = 1.0


class SoundlightBridge:
    """Pusht BeatDetector-Werte periodisch an aura-lcs /api/soundlight/levels.

    Thread-Sicherheit: update() darf aus dem Audio-Callback aufgerufen
    werden (reiht nur ein, blockiert nie auf Netzwerk-I/O).
    """

    def __init__(self, url: str, update_interval_ms: float = 50.0) -> None:
        self._url = url
        self._update_interval = update_interval_ms / 1000.0
        self._last_send_t: float = 0.0

        self._queue: "queue.Queue[Optional[dict]]" = queue.Queue(maxsize=1)
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> bool:
        if not self._url:
            _warn("Keine URL konfiguriert (soundlight.url) — Brücke bleibt aus.")
            return False
        self._running = True
        self._thread = threading.Thread(target=self._sender_loop, daemon=True, name="soundlight-sender")
        self._thread.start()
        return True

    def stop(self) -> None:
        self._running = False
        try:
            self._queue.put_nowait(None)  # weckt den wartenden Thread zum Beenden
        except queue.Full:
            pass
        if self._thread:
            self._thread.join(timeout=2)

    def update(self, detector, now: Optional[float] = None) -> None:
        """Soll periodisch (z.B. aus on_block) aufgerufen werden - rate-
        limitiert selbst auf update_interval_ms, günstig genug für jeden
        Block-Aufruf."""
        t = now if now is not None else time.time()
        if (t - self._last_send_t) < self._update_interval:
            return
        self._last_send_t = t

        kick, snare, hihat = (
            detector.band_scores_normalized()[b] for b in ("kick", "snare", "hihat")
        )
        hits, triggers = detector.pop_hits()

        payload = {
            "kick": kick,
            "snare": snare,
            "hihat": hihat,
            "kickHit": hits["kick"],
            "snareHit": hits["snare"],
            "hihatHit": hits["hihat"],
            "kickTrigger": triggers["kick"],
            "snareTrigger": triggers["snare"],
            "bpm": detector.locked_bpm,
        }
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            # Ein noch nicht gesendeter, älterer Payload wird verworfen -
            # der neue ist aktueller und wichtiger als der alte.
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(payload)
            except queue.Full:
                pass

    def _sender_loop(self) -> None:
        while self._running:
            try:
                payload = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if payload is None:
                break
            try:
                req = urllib.request.Request(
                    self._url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                urllib.request.urlopen(req, timeout=_SEND_TIMEOUT_S).close()
            except (urllib.error.URLError, OSError, TimeoutError) as e:
                _warn(f"Push fehlgeschlagen: {e}")


def _warn(msg: str) -> None:
    print(f"[soundlight] {msg}", file=sys.stderr)
