'use strict';
/**
 * Ableton Link Bridge for bpmdetect
 *
 * IPC-Protokoll (JSON, newline-delimitiert):
 *   Python → Node (stdin):
 *     {"type":"set_tempo","bpm":128.4}
 *     {"type":"set_quantum","quantum":4}
 *     {"type":"get_state"}
 *     {"type":"quit"}
 *
 *   Node → Python (stdout, periodisch alle PUSH_INTERVAL_MS, siehe unten):
 *     {"type":"state","tempo":128.0,"phase":0.75,"beat":3.0,"peers":1,"quantum":4}
 *
 *   Node → Python (stderr, einmalig):
 *     {"type":"ready","tempo":120.0}
 *     {"type":"error","msg":"..."}
 *
 * Push-Intervall (Runde: Link-Beat-Eingangsmodus, 2026-09-22): konfigurierbar
 * über den ersten CLI-Arg (Millisekunden), Default 25ms (vorher fest 100ms) -
 * siehe LinkBridge.__init__(push_interval_ms=...) auf der Python-Seite, das
 * diesen Wert beim Subprocess-Start übergibt. Bestimmt die maximale
 * Zeitgenauigkeit von beat_kick-Ereignissen (siehe beat_scheduler.py) - der
 * Python-Scheduler feuert nur auf einen NEUEN ganzzahligen "beat"-Wert,
 * nicht auf jeden Push, daher erzeugt ein schnelleres Push-Intervall keine
 * doppelten Events, nur eine feinere zeitliche Auflösung.
 */
const PUSH_INTERVAL_MS = (() => {
  const arg = parseInt(process.argv[2], 10);
  return Number.isFinite(arg) && arg > 0 ? arg : 25;
})();

// ── Laden ──────────────────────────────────────────────────────────────────

let AbletonLink;
try {
  ({ AbletonLink } = require('@ktamas77/abletonlink'));
} catch (e) {
  process.stderr.write(JSON.stringify({
    type: 'error',
    msg: '@ktamas77/abletonlink nicht gefunden. Bitte "npm install" im link/-Verzeichnis ausführen. ' + e.message
  }) + '\n');
  process.exit(2);
}

// ── Link-Session ───────────────────────────────────────────────────────────

const INIT_BPM    = 120.0;
const INIT_QUANTUM = 4;

let link;
try {
  // Verschiedene Konstruktor-Signaturen probieren
  try       { link = new AbletonLink(INIT_BPM, INIT_QUANTUM, true); }
  catch (_) { try { link = new AbletonLink(INIT_BPM); } catch (__) { link = new AbletonLink(); } }
} catch (e) {
  process.stderr.write(JSON.stringify({ type: 'error', msg: 'Kann AbletonLink nicht instanziieren: ' + e.message }) + '\n');
  process.exit(3);
}

// Aktivieren
link.enable(true);

// ── API-Adapter (@ktamas77/abletonlink) ───────────────────────────────────

let _quantum = INIT_QUANTUM;

function getTempo()        { return link.getTempo(); }
function setTempo(bpm)     { link.setTempo(bpm); }
function getPhase()        { return link.getPhase(_quantum) / _quantum; } // → 0..1
function getBeat()         { return link.getBeat(); }
function getPeers()        { return link.getNumPeers(); }
function getQuantum()      { return _quantum; }

function snapshot() {
  return {
    type:    'state',
    tempo:   getTempo(),
    phase:   getPhase(),
    beat:    getBeat(),
    peers:   getPeers(),
    quantum: getQuantum(),
  };
}

// ── Periodischer State-Push an Python (Default 40 Hz, siehe PUSH_INTERVAL_MS oben) ──

const pushInterval = setInterval(() => {
  process.stdout.write(JSON.stringify(snapshot()) + '\n');
}, PUSH_INTERVAL_MS);

// ── Befehle von Python (stdin) ────────────────────────────────────────────

const readline = require('readline');
const rl = readline.createInterface({ input: process.stdin, terminal: false });

rl.on('line', (raw) => {
  const line = raw.trim();
  if (!line) return;
  let cmd;
  try { cmd = JSON.parse(line); } catch (_) { return; }

  switch (cmd.type) {
    case 'set_tempo': {
      const bpm = parseFloat(cmd.bpm);
      if (bpm > 20 && bpm < 400) setTempo(bpm);
      break;
    }
    case 'set_quantum': {
      const q = parseFloat(cmd.quantum);
      if (q > 0) _quantum = q;
      break;
    }
    case 'get_state':
      process.stdout.write(JSON.stringify(snapshot()) + '\n');
      break;
    case 'quit':
      clearInterval(pushInterval);
      process.exit(0);
      break;
  }
});

rl.on('close', () => {
  clearInterval(pushInterval);
  process.exit(0);
});

// ── Ready-Signal ──────────────────────────────────────────────────────────

process.stderr.write(JSON.stringify({ type: 'ready', tempo: getTempo(), pushIntervalMs: PUSH_INTERVAL_MS }) + '\n');
