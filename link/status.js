#!/usr/bin/env node
'use strict';
/**
 * Ableton Link Status CLI
 *
 * Usage:
 *   node status.js [--tempo 128] [--quantum 4] [--no-link]
 *
 * Keys (während Laufzeit):
 *   q  → beenden
 *   +  → Tempo +1 BPM
 *   -  → Tempo -1 BPM
 *   e  → Link ein/aus
 */

const { AbletonLink } = require('@ktamas77/abletonlink');

// ── CLI-Argumente ─────────────────────────────────────────────────────────────
const args = process.argv.slice(2);
const flag = (name) => args.includes(name);
const opt   = (name, def) => {
  const i = args.indexOf(name);
  return i >= 0 && args[i + 1] !== undefined ? args[i + 1] : def;
};

const initTempo   = parseFloat(opt('--tempo', '120'));
const initQuantum = parseFloat(opt('--quantum', '4'));
const startEnabled = !flag('--no-link');

// ── Link-Session ──────────────────────────────────────────────────────────────
const link = new AbletonLink(initTempo);
let enabled = startEnabled;
link.enable(enabled);

let quantum = initQuantum;

// ── Display ───────────────────────────────────────────────────────────────────
const LINES = 9;
let initialized = false;

function bar(value, max, width) {
  const filled = Math.round((value / max) * width);
  return '█'.repeat(Math.max(0, filled)) + '░'.repeat(Math.max(0, width - filled));
}

function phaseBar(phase01, width) {
  const pos = Math.round(phase01 * width) % width;
  return '·'.repeat(pos) + '●' + '·'.repeat(width - pos - 1);
}

function render() {
  const tempo   = link.getTempo();
  const beat    = link.getBeat();
  const phase01 = link.getPhase(quantum) / quantum;   // → 0..1
  const peers   = link.getNumPeers();

  const enabledStr  = enabled
    ? '\x1b[32menabled \x1b[0m'
    : '\x1b[31mdisabled\x1b[0m';
  const peerStr = peers === 0
    ? '\x1b[33m0 (solo)\x1b[0m'
    : `\x1b[32m${peers}\x1b[0m`;

  const lines = [
    `  Link         : ${enabledStr}   Peers ${peerStr}`,
    `  Tempo        : ${tempo.toFixed(2)} BPM   [${bar(tempo, 200, 30)}]`,
    `  Quantum      : ${quantum}`,
    `  Beat         : ${beat.toFixed(3)}`,
    `  Phase        : ${phase01.toFixed(3)}  [${phaseBar(phase01, 32)}]`,
    `  ─────────────────────────────────────────────────`,
    `  Keys: q=quit  e=link on/off  +=+1 BPM  -=-1 BPM`,
  ];

  if (!initialized) {
    process.stdout.write('\n'.repeat(lines.length));
    initialized = true;
  }
  process.stdout.write(`\x1b[${lines.length}A${lines.join('\n')}\n`);
}

// ── Update-Loop ───────────────────────────────────────────────────────────────
const interval = setInterval(render, 1000);
render();

// ── Tastatureingabe ───────────────────────────────────────────────────────────
if (process.stdin.isTTY) {
  process.stdin.setRawMode(true);
}
process.stdin.resume();
process.stdin.setEncoding('utf8');

process.stdin.on('data', (key) => {
  if (key === 'q' || key === '') {   // q oder Ctrl+C
    clearInterval(interval);
    link.enable(false);
    process.stdout.write('\n\nLink beendet.\n');
    process.exit(0);
  }
  if (key === 'e') {
    enabled = !enabled;
    link.enable(enabled);
    render();
  }
  if (key === '+') {
    link.setTempo(link.getTempo() + 1);
    render();
  }
  if (key === '-') {
    link.setTempo(Math.max(20, link.getTempo() - 1));
    render();
  }
});
