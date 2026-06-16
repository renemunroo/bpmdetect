'use strict';
/**
 * Node.js Link-Bridge für link_monitor.py
 * Getrennt von bridge.js — ändert Beat-Detection nicht.
 *
 * stdin  ← JSON-Befehle von Python
 * stdout → JSON-Status-Updates (10 Hz)
 * stderr → {"type":"ready"} beim Start
 */

let AbletonLink;
try {
  ({ AbletonLink } = require('@ktamas77/abletonlink'));
} catch (e) {
  process.stderr.write(JSON.stringify({
    type: 'error',
    msg: '@ktamas77/abletonlink fehlt — bitte im link/-Ordner: npm install'
  }) + '\n');
  process.exit(2);
}

const link = new AbletonLink(120.0);
let quantum = 4;
let enabled = false;
let ssEnabled = false;

function snapshot() {
  return {
    type:     'state',
    enabled,
    tempo:    link.getTempo(),
    beat:     link.getBeat(),
    phase:    link.getPhase(quantum) / quantum,   // normiert 0..1
    peers:    link.getNumPeers(),
    quantum,
    playing:  typeof link.isPlaying === 'function' ? link.isPlaying() : false,
    ss_sync:  ssEnabled,
  };
}

// State-Push 10 Hz
const push = setInterval(() => {
  process.stdout.write(JSON.stringify(snapshot()) + '\n');
}, 100);

// Befehle von Python
const readline = require('readline');
const rl = readline.createInterface({ input: process.stdin, terminal: false });
rl.on('line', (raw) => {
  let cmd;
  try { cmd = JSON.parse(raw.trim()); } catch (_) { return; }
  switch (cmd.type) {
    case 'set_enabled':
      enabled = !!cmd.enabled;
      link.enable(enabled);
      break;
    case 'set_tempo':
      if (cmd.bpm > 20 && cmd.bpm < 400) link.setTempo(parseFloat(cmd.bpm));
      break;
    case 'set_quantum':
      if (cmd.quantum > 0) quantum = parseFloat(cmd.quantum);
      break;
    case 'set_startstop':
      ssEnabled = !!cmd.enabled;
      if (typeof link.enableStartStopSync === 'function')
        link.enableStartStopSync(ssEnabled);
      break;
    case 'quit':
      clearInterval(push);
      process.exit(0);
  }
});
rl.on('close', () => { clearInterval(push); process.exit(0); });

process.stderr.write(JSON.stringify({ type: 'ready' }) + '\n');
