"""Export a song for the beat-tracking visualizer (viz/index.html).

Runs the full beat_analysis pipeline and writes everything the viewer needs
(Mel spectrogram, onset envelope, tempo, C*, P*, beats) to viz/<name>/data.js,
then rebuilds viz/songs.js, the list of exported songs the viewer's picker
shows. The page loads these with plain <script> tags, so it works straight
from disk with no web server. Open viz/index.html (or viz/index.html?song=<name>).

Usage:
    python export_viz.py "audio/Party 20s.mp3" --name party20
    python export_viz.py "audio/Monkeys Spinning Monkeys.mp3"      # -> viz/monkeys-spinning-monkeys
    python export_viz.py "audio/Who Likes to Party.mp3" --start 30 --duration 20 --name wltp-clip

To remove a song, delete its folder and re-run any export (or --rebuild-list).

With --start/--duration, the excerpt is analysed on its own (exactly as
beat_analysis would see it) and also written to viz/<name>/clip.wav so
playback lines up sample-for-sample with the analysis. Otherwise the page
plays the original file in place.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
from pathlib import Path
from urllib.parse import quote

import numpy as np

import beat_analysis as ba

VIZ_DIR = Path(__file__).resolve().parent / "viz"


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


DATA_PREFIX = "window.VIZ_DATA = "


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-") or "song"


def rebuild_song_list() -> list[dict]:
    """Write viz/songs.js listing every viz/<name>/data.js, sorted by name."""
    songs = []
    for data_js in sorted(VIZ_DIR.glob("*/data.js")):
        text = data_js.read_text()
        try:
            title = json.loads(text[len(DATA_PREFIX):].rstrip().rstrip(";"))["title"]
        except (ValueError, KeyError):
            print(f"skipping {data_js}: not a viewer data file")
            continue
        songs.append({"id": data_js.parent.name, "title": title})
    (VIZ_DIR / "songs.js").write_text(
        "window.VIZ_SONGS = " + json.dumps(songs, indent=2, ensure_ascii=False) + ";\n")
    return songs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("audio", nargs="?", help="any ffmpeg-readable audio file")
    p.add_argument("--start", type=float, default=0.0, help="excerpt start (s)")
    p.add_argument("--duration", type=float, default=None, help="excerpt length (s)")
    p.add_argument("--name", help="folder name under viz/ (default: from the file name)")
    p.add_argument("--rebuild-list", action="store_true",
                   help="only rebuild viz/songs.js from the existing song folders")
    args = p.parse_args()

    if args.rebuild_list or args.audio is None:
        if args.audio is None and not args.rebuild_list:
            p.error("an audio file is required (or pass --rebuild-list)")
        songs = rebuild_song_list()
        print(f"viz/songs.js: {', '.join(s['id'] for s in songs) or '(no songs)'}")
        return

    src = Path(args.audio).resolve()
    name = slugify(args.name or src.stem + (
        f"-{args.start:g}s" if args.start > 0 or args.duration is not None else ""))
    out_dir = VIZ_DIR / name
    y, sr = ba.load_audio(str(src), args.start, args.duration)
    mel_db, frame_rate = ba.mel_spectrogram_db(y, sr)
    onset, _ = ba.onset_strength_envelope(y, sr)
    tempo = ba.estimate_tempo(onset, frame_rate)
    beats, cumscore, backlink = ba.dp_beat_track(onset, tempo.period)
    assert beats[-1] == int(np.argmax(cumscore))

    # Spectrogram -> uint8, frame-major (T, n_mels) so the browser can build
    # image columns directly. 255 steps over <= 80 dB is ~0.3 dB resolution.
    db_min, db_max = float(mel_db.min()), float(mel_db.max())
    span = max(db_max - db_min, 1e-9)
    mel_q = np.round((mel_db - db_min) / span * 255.0).astype(np.uint8).T

    is_clip = args.start > 0 or args.duration is not None
    out_dir.mkdir(parents=True, exist_ok=True)
    if is_clip:
        # Same -ss / -i / -t ordering as load_audio, so the samples match.
        cmd = ["ffmpeg", "-v", "error", "-nostdin", "-y"]
        if args.start:
            cmd += ["-ss", f"{args.start:.3f}"]
        cmd += ["-i", str(src)]
        if args.duration is not None:
            cmd += ["-t", f"{args.duration:.3f}"]
        cmd += ["-c:a", "pcm_s16le", str(out_dir / "clip.wav")]
        subprocess.run(cmd, check=True)
        audio_url = "clip.wav"
        title = f"{src.stem}  [{args.start:g}s – {args.start + y.size / sr:g}s]"
    else:
        audio_url = quote(Path(os.path.relpath(src, out_dir)).as_posix())
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
        "tempo": {"period": tempo.period, "bpm": tempo.bpm, "meter": tempo.meter},
        "alpha": ba.ALPHA,
        # Search window offsets exactly as dp_beat_track builds prange
        # (Python's round() is half-to-even, so don't recompute this in JS).
        "prange": [int(-round(2 * tempo.period)), int(-round(tempo.period / 2))],
        "cumscore": b64(cumscore.astype("<f4")),
        "backlink": b64(backlink.astype("<i4")),
        "beats": [int(b) for b in beats],    # ascending onset-frame indices
    }
    out = out_dir / "data.js"
    out.write_text(DATA_PREFIX + json.dumps(data, separators=(",", ":")) + ";\n")
    songs = rebuild_song_list()
    print(f"{title}: {data['duration']:.1f} s, {data['nFrames']} frames, "
          f"{tempo.bpm:.1f} BPM, {len(beats)} beats, {out.stat().st_size / 1e6:.1f} MB -> viz/{name}/data.js")
    print(f"Songs in viewer: {', '.join(s['id'] for s in songs)}")
    print(f"Open {VIZ_DIR / 'index.html'}?song={name}")


if __name__ == "__main__":
    main()
