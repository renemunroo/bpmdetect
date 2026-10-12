"""Feature extraction for classical (KNN) drum-hit classification.

Follows the general approach of the SMC-LAB streaming drum transcription
system (ICASSP 2013): onset detection, then a short spectral feature vector
per hit, then a nearest-neighbour classifier - no neural network, runs
comfortably in real time on any hardware since it's just an FFT + a handful
of scalar features per hit, not a continuous per-block inference pass.
"""

import numpy as np


class DrumClassifier:
    """Thin wrapper around the trained KNN model (see
    tools/train_drum_classifier.py) - loads the scaler+classifier pair once
    and classifies a single hit's feature vector into kick/snare/hihat."""

    def __init__(self, model_path: str) -> None:
        import joblib
        data = joblib.load(model_path)
        self._scaler = data["scaler"]
        self._clf = data["clf"]

    def predict(self, features: np.ndarray) -> str:
        return self._clf.predict(self._scaler.transform([features]))[0]

N_MEL_BANDS = 12
MEL_LO_HZ, MEL_HI_HZ = 20.0, 10000.0

FEATURE_NAMES = [f"mel{i}" for i in range(N_MEL_BANDS)] + ["centroid_khz", "flatness", "zcr"]

_band_edges_cache = {}


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _mel_band_edges(n_bands, lo, hi):
    mel_lo, mel_hi = _hz_to_mel(lo), _hz_to_mel(hi)
    return _mel_to_hz(np.linspace(mel_lo, mel_hi, n_bands + 2))


def extract_features(window: np.ndarray, sample_rate: int) -> np.ndarray:
    """window: 1D mono float array (~100ms of audio starting right at a
    detected onset). Returns a fixed-length feature vector: log Mel-band
    energies (captures spectral shape - kick's low-frequency concentration
    vs. snare's broadband body vs. hihat's high-frequency noise) + spectral
    centroid + spectral flatness (tonal vs. noisy - distinguishes a snare's
    noisy rattle from a tonal bass note that shares its frequency range) +
    zero-crossing rate."""
    n = len(window)
    if n == 0:
        return np.zeros(len(FEATURE_NAMES))

    windowed = window * np.hanning(n)
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    cache_key = (n, sample_rate)
    if cache_key not in _band_edges_cache:
        _band_edges_cache[cache_key] = _mel_band_edges(N_MEL_BANDS, MEL_LO_HZ, min(MEL_HI_HZ, sample_rate / 2))
    edges = _band_edges_cache[cache_key]

    band_energies = np.zeros(N_MEL_BANDS)
    for i in range(N_MEL_BANDS):
        mask = (freqs >= edges[i]) & (freqs < edges[i + 2])
        band_energies[i] = np.log1p(np.sum(spectrum[mask] ** 2))

    centroid = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-12)
    geo_mean = np.exp(np.mean(np.log(spectrum + 1e-12)))
    arith_mean = np.mean(spectrum) + 1e-12
    flatness = geo_mean / arith_mean
    zcr = np.mean(np.abs(np.diff(np.sign(window)))) / 2.0

    return np.concatenate([band_energies, [centroid / 1000.0, flatness, zcr]])
