"""Export data for the beat-tracking visualizer (viz/index.html).

Runs the same front end as beat_analysis.py and writes the arrays the viewer
needs to viz/data.js, which index.html loads with a plain <script> tag (so the
page works straight from disk, no web server needed).

Usage:
    python export_viz.py "audio/Party 20s.mp3"
    python export_viz.py "audio/Who Likes to Party.mp3" --start 30 --duration 20

With --start/--duration, the excerpt is analysed on its own (exactly as
beat_analysis would see it) and also written to viz/clip.wav so playback lines
up sample-for-sample with the analysis. Otherwise the page plays the original
file in place.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
from pathlib import Path
from urllib.parse import quote

import numpy as np

import beat_analysis as ba

VIZ_DIR = Path(__file__).resolve().parent / "viz"


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("audio", help="any ffmpeg-readable audio file")
    p.add_argument("--start", type=float, default=0.0, help="excerpt start (s)")
    p.add_argument("--duration", type=float, default=None, help="excerpt length (s)")
    args = p.parse_args()

    src = Path(args.audio).resolve()
    y, sr = ba.load_audio(str(src), args.start, args.duration)
    mel_db, frame_rate = ba.mel_spectrogram_db(y, sr)
    onset, _ = ba.onset_strength_envelope(y, sr)

    # Spectrogram -> uint8, frame-major (T, n_mels) so the browser can build
    # image columns directly. 255 steps over <= 80 dB is ~0.3 dB resolution.
    db_min, db_max = float(mel_db.min()), float(mel_db.max())
    span = max(db_max - db_min, 1e-9)
    mel_q = np.round((mel_db - db_min) / span * 255.0).astype(np.uint8).T

    is_clip = args.start > 0 or args.duration is not None
    VIZ_DIR.mkdir(exist_ok=True)
    if is_clip:
        # Same -ss / -i / -t ordering as load_audio, so the samples match.
        cmd = ["ffmpeg", "-v", "error", "-nostdin", "-y"]
        if args.start:
            cmd += ["-ss", f"{args.start:.3f}"]
        cmd += ["-i", str(src)]
        if args.duration is not None:
            cmd += ["-t", f"{args.duration:.3f}"]
        cmd += ["-c:a", "pcm_s16le", str(VIZ_DIR / "clip.wav")]
        subprocess.run(cmd, check=True)
        audio_url = "clip.wav"
        title = f"{src.stem}  [{args.start:g}s – {args.start + y.size / sr:g}s]"
    else:
        audio_url = quote(Path(os.path.relpath(src, VIZ_DIR)).as_posix())
        title = src.stem

    nyquist = ba.TARGET_SR / 2
    centers = ba._mel_to_hz(np.linspace(ba._hz_to_mel(0), ba._hz_to_mel(nyquist),
                                        ba.N_MELS + 2))[1:-1]

    data = {
        "title": title,
        "audio": audio_url,
        "duration": y.size / sr,
        "frameRate": frame_rate,              # onset / STFT frames per second
        "winSec": ba.WIN_SEC,
        "nMels": ba.N_MELS,
        "nFrames": int(mel_db.shape[1]),      # STFT frames; onset has nFrames-1
        "melMaxHz": nyquist,
        "melCentersHz": [round(float(f), 1) for f in centers],
        "melDb": {"min": db_min, "max": db_max, "data": b64(mel_q)},
        "onset": b64(onset.astype("<f4")),
    }
    out = VIZ_DIR / "data.js"
    out.write_text("window.VIZ_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print(f"{title}: {data['duration']:.1f} s, {data['nFrames']} frames, "
          f"{out.stat().st_size / 1e6:.1f} MB -> {out.relative_to(Path.cwd()) if out.is_relative_to(Path.cwd()) else out}")
    print(f"Open {VIZ_DIR / 'index.html'} in a browser.")


if __name__ == "__main__":
    main()
