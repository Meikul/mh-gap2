# GAP 2 - Beat Tracking Algorithm

## Setup

Requires Python 3.10+ and [ffmpeg](https://ffmpeg.org/download.html) (used to decode the mp3 files; `ffmpeg` and `ffprobe` must be on your `PATH`).

I recommend setting up with conda:

```sh
conda create -n mh-gap2 -c conda-forge python=3.12 numpy scipy matplotlib ffmpeg
```
then activate the env:
```sh
conda activate mh-gap2
```

## Usage

The beat tracking algorithm's code is in `beat_analysis.py`.

To run the test-cases and see their relative runtime, run `benchmark.py`.

This will create the 10 test cases (using the mp3 files in `audio/`) and report the runtime results in the terminal. It also saves the results to `results/benchmark.csv` and creates a plot displaying the results in `results/runtime.png`.