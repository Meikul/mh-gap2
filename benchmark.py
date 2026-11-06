"""Runtime benchmark for the DP beat tracker (beat_analysis.py).

Builds N test cases of increasing length from the audio files in ./audio,
times each pipeline stage, and compares the measurements with the complexity
claimed in Ellis (2007):

  * Section 2: the DP search covers "the entire exponentially-sized set of all
    possible time sequences in a linear-time operation".  Concretely, each of
    the N frames examines |prange| = round(2*tau_p) - round(tau_p/2) + 1
    predecessors, so the forward pass costs O(N * tau_p).  tau_p is fixed per
    piece, so runtime should grow linearly with audio length.
  * The front end (resampling, STFT, Mel, filtering) is O(N log W) with a
    fixed window W, i.e. also linear in N; the tempo autocorrelation is
    O(N log N) via the FFT.

The paper gives no absolute timings (only that it beat all but one MIREX-06
competitor, in Matlab), so the comparison is against the predicted growth rate:
a linear fit T = a*N + b, the log-log slope (expected 1.0), and per-frame cost
(expected constant).

Usage:
    python benchmark.py [--audio-dir audio] [--cases 10] [--repeats 7]
"""

from __future__ import annotations

import os

# Pin BLAS to one thread *before* numpy loads. Otherwise the Mel-filterbank
# matmul wakes a pool of OpenBLAS threads that keep spinning afterwards and
# slow the single-threaded DP loop by ~1.7x on short inputs, which distorts the
# growth-rate measurement. The paper's algorithm is single-threaded anyway.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import csv
import platform
import sys
from pathlib import Path

import numpy as np

import beat_analysis as ba

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}
STAGES = ["onset", "tempo", "dp", "total"]
DECODE_SR = 44100


# ----------------------------------------------------------------------------
# Test cases
# ----------------------------------------------------------------------------
def build_test_cases(audio_dir: Path, n_cases: int, min_dur: float | None,
                     max_dur: float | None) -> list[dict]:
    """Pick n_cases durations evenly spaced between the shortest and longest
    available audio and assign each to a source file long enough to supply it,
    rotating through eligible files so the cases cover different music."""
    files = sorted(p for p in audio_dir.iterdir() if p.suffix.lower() in AUDIO_EXTS)
    if not files:
        sys.exit(f"No audio files found in {audio_dir}/")
    durations = {p: ba.audio_duration(str(p)) for p in files}

    lo = min_dur if min_dur is not None else max(5.0, min(durations.values()))
    hi = max_dur if max_dur is not None else max(durations.values())
    hi = min(hi, max(durations.values()))
    # Round down to 0.1 s so a file's own length always qualifies
    targets = np.floor(np.linspace(lo, hi, n_cases) * 10) / 10

    cases, uses = [], {p: 0 for p in files}
    for i, d in enumerate(targets, 1):
        eligible = [p for p in files if durations[p] >= d]
        src = min(eligible, key=lambda p: (uses[p], durations[p]))
        uses[src] += 1
        cases.append({"case": i, "file": src, "duration": float(d)})
    return cases


# ----------------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------------
def time_case(y: np.ndarray, sr: int, repeats: int) -> dict:
    ba.beat_track(y, sr)  # untimed warm-up: caches + CPU clock after decoding
    runs = [ba.beat_track(y, sr) for _ in range(repeats)]
    out = {"n_frames": runs[0].n_frames, "dp_window": runs[0].dp_window,
           "bpm": runs[0].tempo.bpm, "n_beats": runs[0].beat_times.size}
    for s in STAGES:
        t = np.array([r.timings[s] for r in runs])
        out[f"{s}_median"] = float(np.median(t))
        out[f"{s}_min"] = float(t.min())
        out[f"{s}_max"] = float(t.max())
    # Work the paper's analysis predicts for the DP forward pass:
    # (frames visited) x (candidates per frame)
    start = round(2 * runs[0].tempo.period)
    out["dp_ops"] = max(0, out["n_frames"] - start) * out["dp_window"]
    return out


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Least-squares y = a*x + b; returns (a, b, R^2)."""
    a, b = np.polyfit(x, y, 1)
    resid = y - (a * x + b)
    r2 = 1 - (resid @ resid) / np.sum((y - y.mean()) ** 2)
    return float(a), float(b), float(r2)


def loglog_slope(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
# Light-mode chart tokens (validated default palette, categorical slots 1-3;
# "total" is the aggregate, drawn in primary ink rather than a 4th hue).
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
COLORS = {"total": INK, "onset": "#2a78d6", "dp": "#1baf7a", "tempo": "#eb6834"}
LABELS = {"total": "Total", "onset": "Onset envelope (§3.1)",
          "dp": "DP search (§2)", "tempo": "Tempo estimate (§3.2)"}


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot_results(rows: list[dict], fits: dict, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dur = np.array([r["duration"] for r in rows])
    frames = np.array([r["n_frames"] for r in rows], dtype=float)
    ms = {s: np.array([r[f"{s}_median"] for r in rows]) * 1e3 for s in STAGES}
    err = {s: np.vstack([ms[s] - np.array([r[f"{s}_min"] for r in rows]) * 1e3,
                         np.array([r[f"{s}_max"] for r in rows]) * 1e3 - ms[s]])
           for s in STAGES}
    marker = dict(marker="o", markersize=6, markeredgecolor=SURFACE,
                  markeredgewidth=1.5)

    fig = plt.figure(figsize=(13, 10), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=(1.1, 1))
    ax_a, ax_b, ax_c = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), \
        fig.add_subplot(gs[1, 1])

    # (a) runtime vs length, per stage, with the expected linear model
    for s in STAGES:
        ax_a.errorbar(dur, ms[s], yerr=err[s], color=COLORS[s], label=LABELS[s],
                      linewidth=2, capsize=3, **marker)
    a, b, r2 = fits["total"]["linear"]
    xs = np.linspace(0, frames.max() * 1.05, 50)
    fr = frames[-1] / dur[-1]
    ax_a.plot(xs / fr, (a * xs + b) * 1e3, "--", color=MUTED, linewidth=1.5,
              label=f"Expected O(N): linear fit to total "
                    f"({a * 1e6:.2f} µs/frame + {b * 1e3:.1f} ms), R² = {r2:.4f}")
    for i, r in enumerate(rows):  # case numbers above the total line
        ax_a.annotate(str(r["case"]), (dur[i], ms["total"][i]),
                      textcoords="offset points", xytext=(0, 9), ha="center",
                      fontsize=7.5, color=MUTED)
    _style(ax_a, "(a) Runtime of the 10 test cases",
           "Audio length (s)", "Runtime (ms, median; bars = min–max)")
    ax_a.set_xlim(0, dur.max() * 1.04)
    ax_a.set_ylim(bottom=0)
    ax_a.legend(fontsize=8.5, frameon=False, labelcolor=INK_2, loc="upper left")

    # (b) log-log growth rate vs. the paper's linear claim
    for s in ("total", "onset", "dp"):
        ax_b.loglog(frames, ms[s], color=COLORS[s], linewidth=2,
                    label=f"{LABELS[s]}: slope {fits[s]['slope']:.2f}", **marker)
    x0, y0 = frames[0], ms["total"][0]
    ax_b.loglog(frames, y0 * (frames / x0), "--", color=MUTED, linewidth=1.5,
                label="Expected O(N), slope 1 (paper §2)")
    ax_b.loglog(frames, y0 * (frames / x0) ** 2, ":", color=MUTED, linewidth=1.5,
                label="O(N²) for comparison (unrestricted τ in eqn. 3)")
    _style(ax_b, "(b) Growth rate vs. expected",
           "Onset-envelope frames N (250 per second)", "Runtime (ms)")
    ax_b.legend(fontsize=8, frameon=False, labelcolor=INK_2, loc="upper left")

    # (c) cost per frame per stage: flat if linear
    for s in ("total", "onset", "dp", "tempo"):
        ax_c.plot(dur, ms[s] * 1e3 / frames, color=COLORS[s], linewidth=2,
                  label=LABELS[s], **marker)
    rtf = np.median(dur / (ms["total"] / 1e3))
    _style(ax_c, f"(c) Cost per frame (flat = linear)  ·  ~{rtf:.0f}× real time",
           "Audio length (s)", "Runtime per frame (µs)")
    ax_c.set_xlim(0, dur.max() * 1.04)
    ax_c.set_ylim(0, (ms["total"] * 1e3 / frames).max() * 1.35)
    ax_c.legend(fontsize=8, frameon=False, labelcolor=INK_2, ncol=2,
                loc="upper right")

    fig.suptitle("Ellis (2007) DP beat tracker: measured runtime vs. the paper's "
                 "linear-time claim", fontsize=13, color=INK, x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio-dir", type=Path, default=Path("audio"))
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument("--cases", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=7,
                    help="timed runs per case (median reported)")
    ap.add_argument("--min-duration", type=float, default=None,
                    help="shortest case in seconds (default: shortest file, >= 5 s)")
    ap.add_argument("--max-duration", type=float, default=None,
                    help="longest case in seconds (default: longest file)")
    args = ap.parse_args()
    args.out_dir.mkdir(exist_ok=True)

    cases = build_test_cases(args.audio_dir, args.cases, args.min_duration,
                             args.max_duration)

    # Warm-up so first-call costs (imports, FFT setup) don't skew case 1
    y, sr = ba.load_audio(str(cases[0]["file"]), sr=DECODE_SR)
    for _ in range(3):
        ba.beat_track(y, sr)

    print(f"{platform.processor() or platform.machine()}, Python "
          f"{platform.python_version()}, numpy {np.__version__}; "
          f"{args.repeats} runs per case\n")
    hdr = (f"{'#':>2}  {'source':<28} {'len(s)':>7} {'frames':>7} {'BPM':>6} "
           f"{'onset':>8} {'tempo':>7} {'dp':>8} {'total':>8} {'x RT':>6}")
    print(hdr + "\n" + "-" * len(hdr))

    rows = []
    for c in cases:
        # Decoding is I/O, not part of the algorithm, so it is excluded. All
        # inputs are decoded at one rate so the (timed) resample to 8 kHz
        # costs the same per second regardless of the source file's rate.
        y, sr = ba.load_audio(str(c["file"]), duration=c["duration"], sr=DECODE_SR)
        r = {"case": c["case"], "source": c["file"].name,
             "duration": y.size / sr, **time_case(y, sr, args.repeats)}
        rows.append(r)
        print(f"{r['case']:>2}  {r['source'][:28]:<28} {r['duration']:>7.1f} "
              f"{r['n_frames']:>7} {r['bpm']:>6.1f} "
              f"{r['onset_median'] * 1e3:>6.1f}ms {r['tempo_median'] * 1e3:>5.1f}ms "
              f"{r['dp_median'] * 1e3:>6.1f}ms {r['total_median'] * 1e3:>6.1f}ms "
              f"{r['duration'] / r['total_median']:>6.0f}")

    frames = np.array([r["n_frames"] for r in rows], dtype=float)
    fits = {}
    for s in STAGES:
        t = np.array([r[f"{s}_median"] for r in rows])
        fits[s] = {"linear": linear_fit(frames, t), "slope": loglog_slope(frames, t)}
    ops = np.array([r["dp_ops"] for r in rows], dtype=float)
    dp_t = np.array([r["dp_median"] for r in rows])
    per_op = float(ops @ dp_t / (ops @ ops))            # fit through origin
    resid = dp_t - per_op * ops
    fits["dp"]["per_op"] = per_op
    fits["dp"]["per_op_r2"] = float(1 - resid @ resid / np.sum((dp_t - dp_t.mean()) ** 2))

    print("\nComparison with the paper's expected O(N) runtime "
          "(log-log slope should be ~1.0):")
    for s in STAGES:
        a, b, r2 = fits[s]["linear"]
        print(f"  {s:<6} slope {fits[s]['slope']:.3f}   linear fit "
              f"{a * 1e6:.2f} µs/frame + {b * 1e3:.2f} ms   R² = {r2:.4f}")
    print(f"  DP vs. N·|window| work model: {per_op * 1e9:.1f} ns per candidate, "
          f"R² = {fits['dp']['per_op_r2']:.4f}")

    csv_path = args.out_dir / "benchmark.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    png_path = args.out_dir / "runtime.png"
    plot_results(rows, fits, png_path)
    print(f"\nWrote {csv_path} and {png_path}")


if __name__ == "__main__":
    main()
