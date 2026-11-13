"""Beat tracking by dynamic programming.

Implementation of D. P. W. Ellis, "Beat Tracking by Dynamic Programming",
Journal of New Music Research, 2007.
https://www.ee.columbia.edu/~dpwe/pubs/Ellis07-beattrack.pdf

Pipeline (paper section in brackets):
    1. onset_strength_envelope()  -- O(t), the "local score"        [3.1]
    2. estimate_tempo()           -- target beat period tau_p       [3.2]
    3. dp_beat_track()            -- C*(t) / P*(t) recursion + backtrace [2, Fig. 1]

Usage:
    python beat_analysis.py audio/song.mp3
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass, field
from math import gcd

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import lfilter, resample_poly

# ----------------------------------------------------------------------------
# Parameters from the paper
# ----------------------------------------------------------------------------
TARGET_SR = 8000          # resample input to 8 kHz                      [3.1]
WIN_SEC = 0.032           # 32 ms STFT window                            [3.1]
HOP_SEC = 0.004           # 4 ms hop -> 250 Hz onset envelope            [2, 3.1]
N_MELS = 40               # 40 Mel bands                                 [3.1]
HPF_CUTOFF_HZ = 0.4       # high-pass cutoff "around 0.4 Hz"             [3.1]
SMOOTH_WIDTH_SEC = 0.020  # Gaussian smoothing "about 20 ms wide"        [3.1]
TAU0_SEC = 0.5            # tempo bias centre, 0.5 s = 120 BPM           [3.2]
SIGMA_OCT = 0.9           # weighting width for the TPS2/TPS3 variant    [3.2]
MAX_LAG_SEC = 4.0         # autocorrelation lags examined (Fig. 3)       [3.2]
ALPHA = 680.0             # transition-cost weight (best score, Fig. 4)  [4.2]

# Not specified in the paper: floor for the dB conversion so that digital
# silence (log of ~0) doesn't produce huge spurious onsets.
DB_FLOOR = 80.0


# ----------------------------------------------------------------------------
# Audio I/O
# ----------------------------------------------------------------------------
def load_audio(path: str, offset: float = 0.0, duration: float | None = None,
               sr: int | None = None) -> tuple[np.ndarray, int]:
    """Decode any ffmpeg-readable file to a mono float32 signal.

    Returns (signal, sample_rate). If `sr` is None the native rate is kept.
    """
    if sr is None:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=sample_rate", "-of", "csv=p=0", path],
            capture_output=True, text=True, check=True)
        sr = int(probe.stdout.strip())
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if offset:
        cmd += ["-ss", f"{offset:.3f}"]
    cmd += ["-i", path]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-f", "f32le", "-ac", "1", "-ar", str(sr), "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).astype(np.float64), sr


def audio_duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


# ----------------------------------------------------------------------------
# 3.1  Onset strength envelope
# ----------------------------------------------------------------------------
def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int = N_MELS) -> np.ndarray:
    """Triangular Mel weighting matrix, shape (n_mels, n_fft//2 + 1)."""
    fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    mel_pts = np.linspace(_hz_to_mel(0), _hz_to_mel(sr / 2), n_mels + 2)
    hz_pts = _mel_to_hz(mel_pts)
    weights = np.zeros((n_mels, fft_freqs.size))
    for m in range(n_mels):
        lo, ctr, hi = hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]
        rising = (fft_freqs - lo) / (ctr - lo)
        falling = (hi - fft_freqs) / (hi - ctr)
        weights[m] = np.maximum(0.0, np.minimum(rising, falling))
    return weights


def mel_spectrogram_db(y: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """First half of section 3.1: 8 kHz resample, STFT, 40 Mel bands in dB.

    Returns (mel_db, frame_rate_hz). mel_db has shape (N_MELS, T); column j is
    the STFT frame centred at j / frame_rate + WIN_SEC / 2 seconds.
    """
    # Resample to 8 kHz
    if sr != TARGET_SR:
        g = gcd(sr, TARGET_SR)
        y = resample_poly(y, TARGET_SR // g, sr // g)

    # STFT magnitude: 32 ms windows (256 samples), 4 ms hop (32 samples)
    n_win = int(round(WIN_SEC * TARGET_SR))
    n_hop = int(round(HOP_SEC * TARGET_SR))
    if y.size < n_win:
        y = np.pad(y, (0, n_win - y.size))
    n_frames = 1 + (y.size - n_win) // n_hop
    frames = np.lib.stride_tricks.as_strided(
        y, shape=(n_frames, n_win), strides=(y.strides[0] * n_hop, y.strides[0]))
    spec = np.abs(np.fft.rfft(frames * np.hanning(n_win), axis=1)).T  # (bins, T)

    # Map to 40 Mel bands by weighted summing, convert to dB
    mel = mel_filterbank(TARGET_SR, n_win) @ spec
    mel_db = 20.0 * np.log10(np.maximum(mel, 1e-10))
    mel_db = np.maximum(mel_db, mel_db.max() - DB_FLOOR)
    return mel_db, TARGET_SR / n_hop


def onset_strength_envelope(y: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """Compute O(t) as described in section 3.1.

    Returns (onset_envelope, frame_rate_hz). The envelope is normalized to unit
    standard deviation.
    """
    mel_db, frame_rate = mel_spectrogram_db(y, sr)

    # First-order difference in time, half-wave rectify, sum across bands
    onset = np.maximum(0.0, np.diff(mel_db, axis=1)).sum(axis=0)

    # High-pass filter (~0.4 Hz) to make it locally zero-mean.
    # One-pole/one-zero DC blocker: pole p gives cutoff ~ (1-p)*fs/(2*pi).
    pole = 1.0 - 2.0 * np.pi * HPF_CUTOFF_HZ / frame_rate
    onset = lfilter([1.0, -1.0], [1.0, -pole], onset)

    # Smooth with a Gaussian ~20 ms wide (taken as FWHM)
    sigma_frames = SMOOTH_WIDTH_SEC * frame_rate / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    onset = gaussian_filter1d(onset, sigma_frames, mode="constant")

    # Normalize by standard deviation
    std = onset.std()
    if std > 0:
        onset = onset / std
    return onset, frame_rate


# ----------------------------------------------------------------------------
# 3.2  Global tempo estimate
# ----------------------------------------------------------------------------
@dataclass
class TempoEstimate:
    period: float              # target beat period tau_p, in frames
    bpm: float
    secondary_period: float    # adjacent metrical level (2x or 3x slower)
    meter: str                 # "duple" or "triple"
    weights: tuple[float, float]  # relative TPS weight (primary, secondary)


def estimate_tempo(onset: np.ndarray, frame_rate: float,
                   tau0_sec: float = TAU0_SEC, sigma_oct: float = SIGMA_OCT,
                   max_lag_sec: float = MAX_LAG_SEC) -> TempoEstimate:
    """Windowed autocorrelation tempo estimate with the TPS2/TPS3 refinement."""
    n = onset.size
    max_lag = min(int(round(max_lag_sec * frame_rate)), n - 1)

    # Raw autocorrelation sum_t O(t) O(t - tau), via FFT (O(N log N))
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    spec = np.fft.rfft(onset, nfft)
    acf = np.fft.irfft(spec * np.conj(spec), nfft)[: max_lag + 1]

    # Perceptual weighting W(tau): Gaussian on a log2-time axis (eqn. 6)
    lags = np.arange(max_lag + 1, dtype=float)
    tau0 = tau0_sec * frame_rate
    weight = np.zeros_like(lags)
    weight[1:] = np.exp(-0.5 * (np.log2(lags[1:] / tau0) / sigma_oct) ** 2)
    tps = weight * acf  # eqn. 5

    # TPS2 / TPS3 (eqns. 7, 8): add TPS resampled to 1/2 and 1/3 length
    def multi_level(k: int, coef: tuple[float, float, float]):
        taus = np.arange(1, (max_lag - 1) // k + 1)
        return taus, (tps[taus] + coef[0] * tps[k * taus]
                      + coef[1] * tps[k * taus - 1] + coef[2] * tps[k * taus + 1])

    t2, tps2 = multi_level(2, (0.5, 0.25, 0.25))
    t3, tps3 = multi_level(3, (0.33, 0.33, 0.33))

    if tps2.max() >= tps3.max():
        period, k, meter = int(t2[np.argmax(tps2)]), 2, "duple"
    else:
        period, k, meter = int(t3[np.argmax(tps3)]), 3, "triple"

    p1, p2 = max(tps[period], 0.0), max(tps[k * period], 0.0)
    total = p1 + p2 if (p1 + p2) > 0 else 1.0
    return TempoEstimate(period=float(period), bpm=60.0 * frame_rate / period,
                         secondary_period=float(k * period), meter=meter,
                         weights=(p1 / total, p2 / total))


# ----------------------------------------------------------------------------
# 2  Dynamic programming beat search (direct translation of Fig. 1)
# ----------------------------------------------------------------------------
def dp_beat_track(localscore: np.ndarray, period: float,
                  alpha: float = ALPHA) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find the beat sequence maximizing eqn. 1.

    C*(t) = O(t) + max_{tau in [t-2p, t-p/2]} { alpha * F(t - tau, p) + C*(tau) }
    F(dt, p) = -(log(dt / p))^2

    Returns (beat_frames, cumscore, backlink).
    """
    n = localscore.size
    backlink = -np.ones(n, dtype=np.int64)       # P*(t); -1 = no predecessor
    cumscore = localscore.astype(float).copy()   # C*(t)

    # Search range for previous beat, and log-Gaussian transition cost over it
    # Create an array of all the offsets we need to search for the previous beat
    prange = np.arange(-round(2 * period), -round(period / 2) + 1)
    txcost = -alpha * np.abs(np.log(prange / -period)) ** 2

    # Forward pass: O(N * |prange|) = O(N * tau_p) -> linear in N
    for i in range(-prange[0], n):
        timerange = i + prange  # array of timestamps we have to search for prev beat
        scorecands = txcost + cumscore[timerange]
        xx = int(np.argmax(scorecands))
        cumscore[i] = scorecands[xx] + localscore[i]
        backlink[i] = timerange[xx]

    # Backtrace from the best cumulated score: O(number of beats)
    beats = [int(np.argmax(cumscore))]
    while backlink[beats[-1]] >= 0:
        beats.append(int(backlink[beats[-1]]))
    return np.array(beats[::-1]), cumscore, backlink


# ----------------------------------------------------------------------------
# Full system
# ----------------------------------------------------------------------------
@dataclass
class BeatResult:
    beat_times: np.ndarray
    tempo: TempoEstimate
    n_frames: int
    dp_window: int                     # |prange|, candidates examined per frame
    timings: dict = field(default_factory=dict)  # seconds per stage


def frames_to_time(frames: np.ndarray, frame_rate: float) -> np.ndarray:
    # Frame k of the (differenced) envelope sits between STFT frames k and k+1;
    # STFT frame j is centred at j*hop + win/2.
    return (np.asarray(frames) + 1) / frame_rate + WIN_SEC / 2


def beat_track(y: np.ndarray, sr: int, alpha: float = ALPHA) -> BeatResult:
    """Run the complete system, timing each stage."""
    t0 = time.perf_counter()
    onset, frame_rate = onset_strength_envelope(y, sr)
    t1 = time.perf_counter()
    tempo = estimate_tempo(onset, frame_rate)
    t2 = time.perf_counter()
    beats, _, _ = dp_beat_track(onset, tempo.period, alpha)
    t3 = time.perf_counter()

    window = round(2 * tempo.period) - round(tempo.period / 2) + 1
    return BeatResult(
        beat_times=frames_to_time(beats, frame_rate), tempo=tempo,
        n_frames=onset.size, dp_window=window,
        timings={"onset": t1 - t0, "tempo": t2 - t1, "dp": t3 - t2,
                 "total": t3 - t0})


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 1
    y, sr = load_audio(argv[1])
    res = beat_track(y, sr)
    print(f"{argv[1]}: {y.size / sr:.1f} s, {res.n_frames} frames")
    print(f"tempo: {res.tempo.bpm:.1f} BPM ({res.tempo.meter}), "
          f"{res.beat_times.size} beats")
    print("timings: " + ", ".join(f"{k}={v * 1e3:.1f} ms"
                                  for k, v in res.timings.items()))
    print("first beats (s):", np.round(res.beat_times[:10], 3))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
