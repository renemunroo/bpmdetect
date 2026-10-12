"""LinkBeatScheduler — leitet aus dem fortlaufenden Ableton-Link-Beat-
Zähler (LinkBridge.link_beat) diskrete beat_kick-Ereignisse ab.

Nicht zu verwechseln mit Pioneer Pro-DJ-Link/Beat-Link (siehe Architektur-
Bericht) - diese Klasse nutzt ausschließlich die bereits vorhandene
Ableton-Link-Bridge (link_bridge.py) als externe Taktquelle. Rein lesend
gegenüber LinkBridge, verändert nichts an dessen State oder an der
bestehenden Audio-Erkennung (beat_detector.py bleibt unverändert).

Ein Ereignis pro BEAT, nicht pro Takt/Quantum: Ableton Links `beat`-Wert
steigt kontinuierlich (nicht quantum-begrenzt), `phase` (schon in
LinkBridge quantum-normiert auf 0..1) wrapt dagegen nur einmal pro Takt -
für ein Kick-artiges Ereignis auf jedem Beat ist der Beat-Zähler die
richtige Quelle, die Phase dient nur der Beat-in-Bar-Berechnung.

Bekannte Einschränkung: Ableton Link (bzw. die genutzte Node-Bindung
@ktamas77/abletonlink) exponiert kein Start/Stop-Sync - beat_kick feuert
weiter, auch wenn die Musik pausiert/zwischen Tracks ist. Einzige Näherung
hier: peers==0 wird als "keine aktive externe Session" behandelt und
unterdrückt Events (siehe update()).
"""
import math
import threading
import time
from typing import Optional


class LinkBeatScheduler:
    def __init__(self, link_bridge, quantum: int = 4, fallback_to_audio: bool = False) -> None:
        self._link = link_bridge
        self._quantum = max(1, int(quantum))
        self._fallback_to_audio = fallback_to_audio
        self._lock = threading.Lock()

        self._last_beat_floor: Optional[int] = None
        self._beat_kick: bool = False
        self._pending_beat_kick: bool = False  # siehe consume_beat_kick()
        self._beat_in_bar: int = 0
        self._last_beat_t: float = 0.0
        self._session_tempo: float = 0.0
        self._source: str = "none"  # "link" | "audio_fallback" | "none", siehe source-Property

    def update(self, audio_kick_trigger: bool = False) -> None:
        """Pollt LinkBridge - soll pro Audio-Block aufgerufen werden (billig:
        ein paar Property-Reads + ein Vergleich), damit beat_kick so zeitnah
        wie möglich zum tatsächlichen Link-State-Push erkannt wird (siehe
        bridge.js PUSH_INTERVAL_MS für die Ober-/Untergrenze der
        Zeitgenauigkeit).

        `audio_kick_trigger` ist der aktuelle Wert von BeatDetector.kick_trigger
        (bereits bestehende, saubere Ein-Block-Audio-Kick-Erkennung) - nur
        relevant, wenn fallback_to_audio aktiv ist UND Link gerade keine
        gültige externe Session hat (siehe unten). Sonst wirkungslos."""
        link = self._link
        link_ok = link is not None and link.available and link.peers > 0
        if not link_ok:
            if not self._fallback_to_audio:
                # Kein gültiger externer Link, kein Fallback konfiguriert -
                # keine scheinbar gültigen Events, und beim nächsten Verbinden
                # nicht sofort feuern (siehe unten: _last_beat_floor=None
                # übernimmt den ersten Messwert stumm).
                with self._lock:
                    self._beat_kick = False
                    self._last_beat_floor = None
                    self._source = "none"
                return
            # Fallback: keine gültige Link-Session, aber die bereits
            # bestehende, unabhängig funktionierende Audio-Kick-Erkennung
            # (BeatDetector.kick_trigger - Mean+StdDev-Schwelle, Hold-off,
            # siehe beat_detector.py) treibt beat_kick stattdessen an. Kein
            # eigenes Timing/keine eigene Schwelle hier - reiner Durchreicher,
            # damit dieselbe, bereits live geprüfte Erkennung genutzt wird.
            # beat_in_bar hat hier keine echte Takt-Bedeutung (Audio allein
            # kennt keinen Downbeat) - zyklischer Zähler nur, damit z.B.
            # link_grid_test_flash weiter zwischen zwei Helligkeiten
            # wechselt statt konstant auf "Taktanfang" zu stehen.
            # _last_beat_floor bleibt None, damit ein späteres Link-Comeback
            # sauber (ohne Fehlzündung) resynct, siehe oben.
            with self._lock:
                fired = bool(audio_kick_trigger)
                self._beat_kick = fired
                self._last_beat_floor = None
                self._source = "audio_fallback"
                if fired:
                    self._beat_in_bar = (self._beat_in_bar % self._quantum) + 1
                    self._last_beat_t = time.time()
                    self._pending_beat_kick = True
            return

        beat = link.link_beat
        floor_beat = math.floor(beat)
        phase = link.link_phase
        beat_in_bar = int(phase * self._quantum) % self._quantum + 1
        tempo = link.session_tempo

        fired = False
        with self._lock:
            if self._last_beat_floor is not None and floor_beat > self._last_beat_floor:
                fired = True
            # floor_beat <= _last_beat_floor (gleich, oder Link-Zähler ist
            # zurückgesprungen/resettet) -> NICHT feuern, nur resynchen.
            self._last_beat_floor = floor_beat
            self._beat_kick = fired
            self._beat_in_bar = beat_in_bar
            self._session_tempo = tempo
            self._source = "link"
            if fired:
                self._last_beat_t = time.time()
                self._pending_beat_kick = True  # bleibt True bis consume_beat_kick(), siehe dort

    @property
    def beat_kick(self) -> bool:
        """True nur exakt in dem update()-Aufruf, in dem ein neuer Beat
        erkannt wurde (kein abklingender Puls, wie kick_trigger) - NICHT
        konsumierend, für Beobachter, die jeden Block lesen (z.B. das
        debug_beats-Log in beat_detector.py, das im selben Takt wie
        update() läuft). Für Konsumenten mit selteneren, unregelmäßigen
        Poll-Abständen (z.B. SoundlightBridge alle ~33ms) siehe
        consume_beat_kick() - diese Property allein würde dort die meisten
        Beats verpassen, weil das ~11ms-Fenster zwischen zwei Polls landet."""
        with self._lock:
            return self._beat_kick

    def consume_beat_kick(self) -> bool:
        """Drain-on-read, UNABHÄNGIG von der beat_kick-Property: True genau
        einmal pro Beat, egal wie lange zwischen zwei Aufrufen vergeht -
        damit ein selteneres, unregelmäßiges Poll-Intervall (z.B.
        SoundlightBridge) keinen Beat verpasst, nur weil es gerade nicht
        exakt in dem ~11ms-Block pollt, in dem update() ihn erkannt hat.
        Bewusst eine EIGENE Markierung statt die beat_kick-Property selbst
        konsumierend zu machen: sonst würden sich debug_beats (liest jeden
        Block) und SoundlightBridge (pollt seltener) gegenseitig den Beat
        wegkonsumieren, je nachdem wer zuerst liest."""
        with self._lock:
            val = self._pending_beat_kick
            self._pending_beat_kick = False
            return val

    @property
    def beat_in_bar(self) -> int:
        """1..quantum - Position im Takt, aus der quantum-normierten
        Link-Phase abgeleitet."""
        with self._lock:
            return self._beat_in_bar

    @property
    def beat_period_s(self) -> float:
        """Sekunden pro Beat aus dem aktuellen Link-Tempo - für eine
        künftige Snare-/Clap-Fensterung vorbereitet (noch nicht genutzt,
        siehe Moduldoku/Architektur-Bericht), 0.0 wenn kein Tempo bekannt."""
        with self._lock:
            tempo = self._session_tempo
        return 60.0 / tempo if tempo > 0 else 0.0

    @property
    def last_beat_t(self) -> float:
        with self._lock:
            return self._last_beat_t

    @property
    def source(self) -> str:
        """"link" (echte externe Link-Session treibt beat_kick), "audio_fallback"
        (fallback_to_audio aktiv, keine gültige Link-Session - Audio-Kick-
        Erkennung treibt beat_kick stattdessen) oder "none" (kein gültiger
        Link UND kein Fallback konfiguriert - beat_kick bleibt aus)."""
        with self._lock:
            return self._source
