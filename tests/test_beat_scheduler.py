"""Tests für LinkBeatScheduler (Ableton-Link-Beat-Eingangsmodus) und die
zugehörige SoundlightBridge-Payload-Erweiterung (beatKick/beatInBar).

Reine unittest-Standardbibliothek, siehe tests/test_triggers.py für die
gleiche Konvention. Kein echter Node-Prozess/Link-Session nötig - ein
kleiner Fake-LinkBridge reicht, da LinkBeatScheduler nur dessen Properties
liest (link_beat/link_phase/session_tempo/peers/available).
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from beat_scheduler import LinkBeatScheduler  # noqa: E402
from soundlight_bridge import SoundlightBridge  # noqa: E402


class FakeLinkBridge:
    def __init__(self, available=True, peers=1, beat=0.0, phase=0.0, tempo=120.0):
        self.available = available
        self.peers = peers
        self.link_beat = beat
        self.link_phase = phase
        self.session_tempo = tempo


class OneEventPerIntegerBeatTests(unittest.TestCase):
    """a) ein Event pro neuem ganzzahligen link_beat."""

    def test_fires_once_per_beat_crossing(self) -> None:
        link = FakeLinkBridge(beat=0.0)
        sched = LinkBeatScheduler(link, quantum=4)
        sched.update()
        self.assertFalse(sched.beat_kick)  # erster Poll: kein Vorwert, kein Feuern

        link.link_beat = 0.5
        sched.update()
        self.assertFalse(sched.beat_kick)

        link.link_beat = 1.02  # floor springt 0 -> 1
        sched.update()
        self.assertTrue(sched.beat_kick)

        link.link_beat = 1.6
        sched.update()
        self.assertFalse(sched.beat_kick)  # noch derselbe floor(1)

        link.link_beat = 2.01
        sched.update()
        self.assertTrue(sched.beat_kick)

    def test_multiple_beat_crossings_over_time(self) -> None:
        """Läuft mehrere ganze Beats durch und zählt die Events."""
        link = FakeLinkBridge(beat=0.0)
        sched = LinkBeatScheduler(link, quantum=4)
        fired_count = 0
        for beat in [0.0, 0.3, 0.9, 1.05, 1.5, 1.99, 2.02, 2.8, 3.1, 3.9, 4.05]:
            link.link_beat = beat
            sched.update()
            if sched.beat_kick:
                fired_count += 1
        # Ganzzahlige Übergänge in dieser Sequenz: 0->1, 1->2, 2->3, 3->4 = 4
        self.assertEqual(fired_count, 4)


class NoDuplicateEventsOnRepeatedPushTests(unittest.TestCase):
    """b) keine Events bei doppelten/wiederholten State-Updates (gleicher
    floor(link_beat), z.B. weil bridge.js schneller pusht als sich der Beat
    ändert - genau der Fall, den die schnellere Push-Rate erzeugt)."""

    def test_same_floor_beat_polled_repeatedly_fires_only_once(self) -> None:
        link = FakeLinkBridge(beat=1.0)
        sched = LinkBeatScheduler(link, quantum=4)
        sched.update()  # etabliert floor=1, kein Feuern (erster Poll)

        fired = 0
        for b in [1.01, 1.05, 1.10, 1.15, 1.20, 1.25]:  # mehrere Pushes, gleicher floor(1)
            link.link_beat = b
            sched.update()
            if sched.beat_kick:
                fired += 1
        self.assertEqual(fired, 0)

        link.link_beat = 2.0  # jetzt ein echter neuer Beat
        sched.update()
        self.assertTrue(sched.beat_kick)


class BeatInBarComputationTests(unittest.TestCase):
    """c) korrekte beatInBar-Berechnung."""

    def test_quantum_4_positions(self) -> None:
        link = FakeLinkBridge(beat=1.0)
        sched = LinkBeatScheduler(link, quantum=4)
        cases = [
            (0.0, 1), (0.1, 1), (0.24, 1),
            (0.25, 2), (0.49, 2),
            (0.5, 3), (0.74, 3),
            (0.75, 4), (0.99, 4),
        ]
        for phase, expected in cases:
            link.link_phase = phase
            sched.update()
            self.assertEqual(sched.beat_in_bar, expected, f"phase={phase}")

    def test_quantum_2_positions(self) -> None:
        link = FakeLinkBridge(beat=1.0)
        sched = LinkBeatScheduler(link, quantum=2)
        link.link_phase = 0.1
        sched.update()
        self.assertEqual(sched.beat_in_bar, 1)
        link.link_phase = 0.6
        sched.update()
        self.assertEqual(sched.beat_in_bar, 2)


class NoEventWithoutValidLinkTests(unittest.TestCase):
    """d) kein Event ohne gültigen Link."""

    def test_unavailable_link_never_fires(self) -> None:
        link = FakeLinkBridge(available=False, beat=1.0)
        sched = LinkBeatScheduler(link, quantum=4)
        sched.update()
        link.link_beat = 2.0
        sched.update()
        self.assertFalse(sched.beat_kick)

    def test_none_link_bridge_never_fires(self) -> None:
        sched = LinkBeatScheduler(None, quantum=4)
        sched.update()  # darf nicht crashen
        self.assertFalse(sched.beat_kick)


class BackwardOrResetBeatCounterTests(unittest.TestCase):
    """e) kein falsches Event bei rückwärts laufendem oder zurückgesetztem
    Beat-Zähler (z.B. Link-Session-Neustart)."""

    def test_backward_jump_does_not_fire(self) -> None:
        link = FakeLinkBridge(beat=5.0)
        sched = LinkBeatScheduler(link, quantum=4)
        sched.update()  # floor=5 etabliert

        link.link_beat = 0.2  # Reset/Neustart der Session
        sched.update()
        self.assertFalse(sched.beat_kick, "Rueckwaertssprung darf nicht als Beat-Event zaehlen")

        # Danach muss normales Vorwaertszaehlen wieder funktionieren.
        link.link_beat = 1.1
        sched.update()
        self.assertTrue(sched.beat_kick)

    def test_same_beat_value_twice_does_not_fire(self) -> None:
        link = FakeLinkBridge(beat=3.0)
        sched = LinkBeatScheduler(link, quantum=4)
        sched.update()
        link.link_beat = 3.0  # exakt gleicher Wert
        sched.update()
        self.assertFalse(sched.beat_kick)


class ModeInputNoWritebackTests(unittest.TestCase):
    """f) mode=input schreibt nicht zurück nach Ableton Link.

    LinkBeatScheduler selbst ruft niemals link_bridge.update(...) auf (das
    bleibt Sache von run_bpm()'s on_block, siehe beat_detector.py) - dieser
    Test bestätigt genau das: LinkBeatScheduler.update() ruft ausschließlich
    lesende Properties auf, keine Methode mit Seiteneffekt auf der Bridge."""

    def test_scheduler_never_calls_writeback_style_methods(self) -> None:
        class StrictFakeLinkBridge(FakeLinkBridge):
            def update(self, *a, **kw):
                raise AssertionError("LinkBeatScheduler darf link_bridge.update() nicht aufrufen - das ist die Rueckschreib-Methode")

        link = StrictFakeLinkBridge(beat=1.0)
        sched = LinkBeatScheduler(link, quantum=4)
        sched.update()
        link.link_beat = 2.0
        sched.update()  # würde AssertionError werfen, wenn intern link.update() aufgerufen würde


class ModeOutputCompatibilityTests(unittest.TestCase):
    """g) mode=output bleibt kompatibel - SoundlightBridge ohne
    beat_scheduler verhält sich exakt wie vor dieser Runde."""

    class _FakeDetector:
        band_scores = (0.1, 0.2, 0.3, 0.4)
        overall_score = 0.5
        onset_scores = (0.6, 0.7, 0.8, 0.9)
        kick_trigger = False
        snare_trigger = False
        clap_trigger = False
        beat_phase = 0.1
        locked_bpm = 0.0
        bpm = 0.0

    def test_no_beat_scheduler_matches_pre_round_payload_shape(self) -> None:
        bridge = SoundlightBridge(self._FakeDetector(), url="http://example.invalid/levels")
        payload = json.loads(bridge._build_payload())
        # Alle bisherigen Felder unveraendert vorhanden.
        for key in ("kick", "snare", "hihat", "clap", "level", "kickHit", "snareHit",
                    "hihatHit", "clapHit", "kickTrigger", "snareTrigger", "clapTrigger", "phase"):
            self.assertIn(key, payload)


class PayloadDefaultsWithoutSchedulerTests(unittest.TestCase):
    """h) Payload-Defaults ohne Scheduler."""

    class _FakeDetector:
        band_scores = (0.1, 0.2, 0.3, 0.4)
        overall_score = 0.5
        onset_scores = (0.6, 0.7, 0.8, 0.9)
        kick_trigger = True
        snare_trigger = False
        clap_trigger = False
        beat_phase = 0.5
        locked_bpm = 0.0
        bpm = 0.0

    def test_beat_fields_default_false_zero(self) -> None:
        bridge = SoundlightBridge(self._FakeDetector(), url="http://example.invalid/levels")
        payload = json.loads(bridge._build_payload())
        self.assertIs(payload["beatKick"], False)
        self.assertEqual(payload["beatInBar"], 0)

    def test_beat_fields_reflect_scheduler_when_present(self) -> None:
        class FakeScheduler:
            beat_in_bar = 3

            def consume_beat_kick(self):
                return True

        bridge = SoundlightBridge(self._FakeDetector(), url="http://example.invalid/levels", beat_scheduler=FakeScheduler())
        payload = json.loads(bridge._build_payload())
        self.assertIs(payload["beatKick"], True)
        self.assertEqual(payload["beatInBar"], 3)
        # kickTrigger aus der Audio-Erkennung bleibt unabhaengig davon erhalten.
        self.assertIs(payload["kickTrigger"], True)


class PushIntervalConfigurationTests(unittest.TestCase):
    """i) Push-Intervall/Bridge-Konfiguration."""

    def test_link_bridge_accepts_push_interval_param(self) -> None:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        import link_bridge as lb_module
        bridge = lb_module.LinkBridge(push_interval_ms=25.0)
        self.assertEqual(bridge._push_interval_ms, 25.0)

    def test_link_bridge_default_push_interval_is_25ms(self) -> None:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        import link_bridge as lb_module
        bridge = lb_module.LinkBridge()
        self.assertEqual(bridge._push_interval_ms, 25.0)


class MissingPeerNoFalseEventsTests(unittest.TestCase):
    """j) fehlender Link-Peer erzeugt keine scheinbar gültigen Beat-Events."""

    def test_zero_peers_suppresses_events_even_if_available(self) -> None:
        link = FakeLinkBridge(available=True, peers=0, beat=1.0)
        sched = LinkBeatScheduler(link, quantum=4)
        sched.update()
        link.link_beat = 2.0
        sched.update()
        self.assertFalse(sched.beat_kick, "peers=0 sollte Events unterdruecken, auch wenn available=True")

    def test_regaining_peers_does_not_fire_immediately(self) -> None:
        """Nach Wiederverbindung darf der erste Poll nicht sofort feuern
        (sonst könnte ein beliebiger Beat-Sprung beim Reconnect als
        Beat-Event fehlinterpretiert werden)."""
        link = FakeLinkBridge(available=True, peers=0, beat=1.0)
        sched = LinkBeatScheduler(link, quantum=4)
        sched.update()

        link.peers = 1
        link.link_beat = 7.3  # großer Sprung durch Reconnect
        sched.update()
        self.assertFalse(sched.beat_kick, "erster Poll nach Wiederverbindung darf nicht feuern")

        link.link_beat = 8.1
        sched.update()
        self.assertTrue(sched.beat_kick, "danach muss normales Beat-Tracking wieder funktionieren")


class ConsumeBeatKickLatchTests(unittest.TestCase):
    """Live beobachtet (Lichttest): SoundlightBridge pollt seltener (~33ms)
    als beat_kick aktiv ist (~11ms) und hat dadurch praktisch nie True
    gesehen. consume_beat_kick() löst das als eigene, drain-on-read
    Markierung, UNABHÄNGIG von der bestehenden beat_kick-Property."""

    def test_stays_true_across_many_polls_until_consumed(self) -> None:
        link = FakeLinkBridge(beat=1.0)
        sched = LinkBeatScheduler(link, quantum=4)
        sched.update()  # etabliert floor=1, kein Feuern

        link.link_beat = 2.05  # Beat-Übergang
        sched.update()
        # beat_kick (nicht-konsumierend) ist nur in DIESEM update()-Aufruf True.
        self.assertTrue(sched.beat_kick)

        # Viele weitere, nicht-konsumierende Polls (z.B. debug_beats-artig) -
        # beat_kick selbst wird beim naechsten update() wieder False (altes
        # Verhalten, unveraendert), aber consume_beat_kick() muss trotzdem
        # noch True liefern, weil es NIE gelesen wurde.
        for _ in range(5):
            link.link_beat += 0.001  # kein neuer Beat, nur Mikro-Fortschritt
            sched.update()
        self.assertFalse(sched.beat_kick, "beat_kick ist wie bisher nur einen update()-Aufruf lang True")

        self.assertTrue(sched.consume_beat_kick(), "der Beat muss trotz langer Verzoegerung noch abholbar sein")

    def test_consuming_clears_it_exactly_once(self) -> None:
        link = FakeLinkBridge(beat=1.0)
        sched = LinkBeatScheduler(link, quantum=4)
        sched.update()
        link.link_beat = 2.0
        sched.update()

        self.assertTrue(sched.consume_beat_kick())
        self.assertFalse(sched.consume_beat_kick(), "ein zweiter Konsum ohne neuen Beat muss False liefern")
        self.assertFalse(sched.consume_beat_kick(), "bleibt False, bis der naechste echte Beat kommt")

    def test_beat_kick_property_reads_do_not_affect_consume_beat_kick(self) -> None:
        """Kernanforderung: debug_beats (liest beat_kick) und SoundlightBridge
        (liest consume_beat_kick()) duerfen sich nicht gegenseitig den Beat
        wegkonsumieren."""
        link = FakeLinkBridge(beat=1.0)
        sched = LinkBeatScheduler(link, quantum=4)
        sched.update()
        link.link_beat = 2.0
        sched.update()

        # "debug_beats" liest mehrfach die alte, nicht-konsumierende Property.
        self.assertTrue(sched.beat_kick)
        _ = sched.beat_kick
        _ = sched.beat_kick

        # "SoundlightBridge" pollt (viel) spaeter zum ersten Mal - muss den
        # Beat trotzdem noch sehen.
        self.assertTrue(sched.consume_beat_kick())

    def test_multiple_beats_before_consumption_still_reports_true_once(self) -> None:
        """Kein Aufstauen mehrerer Beats zu einem Zaehler - nur ein
        einfaches "mindestens ein Beat seit dem letzten Konsum"-Flag,
        bewusst einfach gehalten (kein Beat-Count noetig fuer den
        Lichttest)."""
        link = FakeLinkBridge(beat=1.0)
        sched = LinkBeatScheduler(link, quantum=4)
        sched.update()

        for target in (2.0, 3.0, 4.0):
            link.link_beat = target
            sched.update()

        self.assertTrue(sched.consume_beat_kick())
        self.assertFalse(sched.consume_beat_kick())

    def test_soundlight_bridge_payload_uses_consume_not_peek(self) -> None:
        """Regressionstest für den live beobachteten Bug: SoundlightBridge
        muss consume_beat_kick() aufrufen, nicht die alte beat_kick-Property
        (die zwischen zwei ~33ms-Polls fast immer schon wieder False war)."""
        class RecordingScheduler:
            beat_in_bar = 1

            def __init__(self):
                self.consume_calls = 0

            def consume_beat_kick(self):
                self.consume_calls += 1
                return True

            @property
            def beat_kick(self):
                raise AssertionError("SoundlightBridge darf die nicht-konsumierende beat_kick-Property nicht lesen")

        sched = RecordingScheduler()
        bridge = SoundlightBridge(ModeOutputCompatibilityTests._FakeDetector(), url="http://example.invalid/levels", beat_scheduler=sched)
        payload = json.loads(bridge._build_payload())
        self.assertIs(payload["beatKick"], True)
        self.assertEqual(sched.consume_calls, 1)


class NotifyBeatKickWakeTests(unittest.TestCase):
    """Live beobachtet (Lichttest): der reine ~33ms-Intervall-Loop von
    SoundlightBridge summierte sich mit push_interval_ms und Grand Visions
    eigenem DMX-Tick zu spürbarem Nachlaufen. notify_beat_kick() weckt den
    Sende-Thread sofort statt auf den naechsten Timer-Tick zu warten."""

    def test_notify_sets_wake_event(self) -> None:
        bridge = SoundlightBridge(ModeOutputCompatibilityTests._FakeDetector(), url="http://example.invalid/levels")
        self.assertFalse(bridge._wake_event.is_set())
        bridge.notify_beat_kick()
        self.assertTrue(bridge._wake_event.is_set())

    def test_loop_wakes_immediately_instead_of_waiting_full_interval(self) -> None:
        """Startet den echten _loop()-Thread mit einem langen Intervall
        (2s) gegen eine ungueltige URL (schlaegt schnell fehl, kein echtes
        Netzwerk noetig) - misst, dass ein zweiter Send-Versuch innerhalb
        von ~100ms nach notify_beat_kick() passiert, nicht erst nach den
        vollen 2s."""
        import time as time_module

        bridge = SoundlightBridge(
            ModeOutputCompatibilityTests._FakeDetector(),
            url="http://127.0.0.1:1/levels",  # Port 1 - verbindet nie, schlaegt schnell (ECONNREFUSED) fehl
            interval_ms=2000.0,
        )
        send_times = []
        original_build = bridge._build_payload

        def spy_build():
            send_times.append(time_module.time())
            return original_build()

        bridge._build_payload = spy_build
        bridge.start()
        try:
            time_module.sleep(0.05)  # erster Send (Loop-Start)
            self.assertEqual(len(send_times), 1)

            bridge.notify_beat_kick()
            time_module.sleep(0.1)  # sollte laengst wach sein, weit vor den 2s
            self.assertGreaterEqual(len(send_times), 2, "notify_beat_kick() haette den Loop sofort wecken muessen")
        finally:
            bridge.stop()


if __name__ == "__main__":
    unittest.main()
