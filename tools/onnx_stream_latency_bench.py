#!/usr/bin/env python3
"""
Isolierter Latenz-Benchmark fuer kausale Streaming-Stem-Separation via ONNX
Runtime - beantwortet NUR die Frage "wie schnell laeuft ein Modell dieser
Groessenordnung auf dieser Maschine", bevor irgendetwas in die laufende
bpmdetect-Pipeline integriert wird.

Wichtig: dieses Skript braucht eine EIGENE venv (torch/onnx sind schwere,
optionale Abhaengigkeiten, die bewusst nicht in bpmdetect/.venv installiert
wurden, weil torch 2.2.2 gegen numpy 1.x gebaut ist und mit dem dort
installierten numpy 2.0.2 kollidiert). Setup:

    python3.9 -m venv /pfad/zu/onnx-bench-venv
    source /pfad/zu/onnx-bench-venv/bin/activate
    pip install "numpy<2" torch onnx onnxruntime
    python tools/onnx_stream_latency_bench.py

Architektur-Hinweis
--------------------
Es gibt kein oeffentlich verifizierbares vortrainiertes Modell namens
"StemgenRT" mit fertigen Gewichten - daher hier ein Platzhalter-Netz mit der
STRUKTUR eines kausalen Conv-TasNet/TCN-Separators (Encoder -> gestapelte
kausale dilatierte Conv1d-Bloecke -> Masken-Kopf -> Decoder), in zwei
Groessen. Die Gewichte sind zufaellig (unbrauchbar fuer echte Trennung) -
es geht ausschliesslich um die Rechenkosten der Architektur, nicht um
Trennqualitaet.

Streaming-Methode: "naiver" Sliding-Window-Ansatz - bei jedem neuen Audio-
Chunk wird ueber das GESAMTE Kontextfenster (Rezeptionsfeld + aktueller
Chunk) neu gerechnet, nicht nur inkrementell mit gecachtem Zustand pro
Schicht. Das ist die einfachste korrekt-kausale Variante und daher die
realistische Obergrenze fuer "wie es aussieht, wenn wir es heute so bauen
wuerden" - eine echte inkrementelle Implementierung (State-Caching pro
Conv-Schicht, wie z.B. Streaming-ASR-Modelle es machen) waere schneller,
aber deutlich komplexer im Export/Code und daher bewusst nicht Teil dieses
ersten Tests.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime as ort

SR = 44100


class CausalTCNBlock(nn.Module):
    def __init__(self, b_channels: int, h_channels: int, dilation: int, kernel: int = 3):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv_in = nn.Conv1d(b_channels, h_channels, 1)
        self.prelu1 = nn.PReLU()
        self.norm1 = nn.GroupNorm(1, h_channels)
        self.dwconv = nn.Conv1d(h_channels, h_channels, kernel, dilation=dilation, groups=h_channels)
        self.prelu2 = nn.PReLU()
        self.norm2 = nn.GroupNorm(1, h_channels)
        self.conv_out = nn.Conv1d(h_channels, b_channels, 1)

    def forward(self, x):
        y = self.norm1(self.prelu1(self.conv_in(x)))
        y = F.pad(y, (self.pad, 0))  # kausal: nur nach links padden, kein Blick in die Zukunft
        y = self.norm2(self.prelu2(self.dwconv(y)))
        y = self.conv_out(y)
        return x + y


class CausalSeparator(nn.Module):
    """Encoder -> gestapelte kausale TCN-Bloecke -> Sigmoid-Maske -> Decoder.
    Ein Ausgabekanal = isolierter Drum-Stem."""

    def __init__(self, n_filters, b_channels, h_channels, encoder_kernel, repeats, dilations):
        super().__init__()
        stride = encoder_kernel // 2
        self.encoder = nn.Conv1d(1, n_filters, encoder_kernel, stride=stride, bias=False)
        self.bottleneck = nn.Conv1d(n_filters, b_channels, 1)
        self.blocks = nn.ModuleList([
            CausalTCNBlock(b_channels, h_channels, d)
            for _ in range(repeats) for d in dilations
        ])
        self.mask_head = nn.Conv1d(b_channels, n_filters, 1)
        self.decoder = nn.ConvTranspose1d(n_filters, 1, encoder_kernel, stride=stride, bias=False)

    def forward(self, x):
        enc = self.encoder(x)
        h = self.bottleneck(enc)
        for block in self.blocks:
            h = block(h)
        mask = torch.sigmoid(self.mask_head(h))
        out = self.decoder(enc * mask)
        return out


CONFIGS = {
    "light": dict(n_filters=128, b_channels=64, h_channels=128, encoder_kernel=16,
                  repeats=2, dilations=[1, 2, 4, 8, 16, 32], context_samples=4112),
    "standard": dict(n_filters=256, b_channels=128, h_channels=256, encoder_kernel=16,
                      repeats=3, dilations=[1, 2, 4, 8, 16, 32], context_samples=8208),
}

CHUNK_SIZES = [128, 256]
N_WARMUP = 20
N_ITERS = 200


def export_onnx(name, cfg, path):
    model_kwargs = {k: v for k, v in cfg.items() if k != "context_samples"}
    model = CausalSeparator(**model_kwargs).eval()
    dummy = torch.zeros(1, 1, cfg["context_samples"])
    with torch.no_grad():
        out = model(dummy)
    torch.onnx.export(
        model, dummy, path,
        input_names=["audio_context"], output_names=["drum_stem"],
        opset_version=17,
    )
    n_params = sum(p.numel() for p in model.parameters())
    return out.shape, n_params


def bench_provider(session, input_name, context_samples, chunk_size):
    x = np.random.randn(1, 1, context_samples).astype(np.float32)
    for _ in range(N_WARMUP):
        session.run(None, {input_name: x})
    times_ms = []
    for _ in range(N_ITERS):
        t0 = time.perf_counter()
        session.run(None, {input_name: x})
        times_ms.append((time.perf_counter() - t0) * 1000)
    times_ms = np.array(times_ms)
    chunk_deadline_ms = chunk_size / SR * 1000
    return {
        "p50": np.percentile(times_ms, 50),
        "p95": np.percentile(times_ms, 95),
        "max": times_ms.max(),
        "chunk_deadline_ms": chunk_deadline_ms,
    }


def main():
    print(f"CPU: siehe `sysctl -n machdep.cpu.brand_string` | ONNX Runtime {ort.__version__}")
    print(f"Verfuegbare Execution Provider: {ort.get_available_providers()}\n")

    for name, cfg in CONFIGS.items():
        path = f"/tmp/causal_separator_{name}.onnx"
        out_shape, n_params = export_onnx(name, cfg, path)
        rf_ms = cfg["context_samples"] / SR * 1000
        print(f"=== Konfiguration '{name}' ===")
        print(f"  Params: {n_params/1e6:.2f}M | Kontextfenster: {cfg['context_samples']} Samples (~{rf_ms:.0f} ms) | Output-Shape: {tuple(out_shape)}")

        for provider in ["CPUExecutionProvider", "CoreMLExecutionProvider"]:
            try:
                session = ort.InferenceSession(path, providers=[provider])
                actual = session.get_providers()[0]
                if actual != provider:
                    print(f"  [{provider}] nicht nutzbar, ONNX Runtime ist auf {actual} zurueckgefallen - ueberspringe")
                    continue
            except Exception as e:
                print(f"  [{provider}] Session-Erstellung fehlgeschlagen: {e}")
                continue

            input_name = session.get_inputs()[0].name
            for chunk_size in CHUNK_SIZES:
                r = bench_provider(session, input_name, cfg["context_samples"], chunk_size)
                verdict = "OK" if r["p95"] < r["chunk_deadline_ms"] else ("OK ggue. 15ms-Ziel" if r["p95"] < 15 else "ZU LANGSAM")
                print(f"  [{provider}] chunk={chunk_size:>3} samples ({r['chunk_deadline_ms']:.1f} ms Deadline): "
                      f"p50={r['p50']:.2f} ms  p95={r['p95']:.2f} ms  max={r['max']:.2f} ms  -> {verdict}")
        print()


if __name__ == "__main__":
    main()
