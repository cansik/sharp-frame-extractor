# Sharp Frame Extractor [![PyPI](https://img.shields.io/pypi/v/sharp-frame-extractor)](https://pypi.org/project/sharp-frame-extractor/)

Sharp Frame Extractor is a command line utility for sampling videos into still images using sharpness scoring. It processes the input in short time windows and writes the highest scoring frame from each window to disk, which is useful for photogrammetry, volumetric capture, and similar pipelines.

![demo-dark](https://github.com/user-attachments/assets/674699c1-1be5-4cbe-9a61-7d3e2c20af47)

Version 2 focuses on:
- A simpler command line interface
- Better sharpness scoring
- Two-pass architecture for memory-efficient analysis and faster processing

## Example

![frames-all](https://user-images.githubusercontent.com/5220162/117341573-9a348400-aea2-11eb-9567-370a605c4f62.jpg)
*One hundred sharp frames extracted from an iPhone video.*

![from-to-trees](https://user-images.githubusercontent.com/5220162/117341592-a02a6500-aea2-11eb-89e4-f4eb3d1eac07.jpg)
*Reconstruction from the extracted frames, compared with the original cherry tree.*

More context for this project is on [Behance](https://www.behance.net/gallery/118822685/Immersive-Memories).

## Quick start

The easiest way to run sharp-frame-extractor is using [uvx](https://docs.astral.sh/uv/guides/tools/), which is included with [uv](https://docs.astral.sh/uv/). uvx runs the tool in an isolated environment without requiring a manual virtual environment setup.

First, [install uv](https://docs.astral.sh/uv/getting-started/installation/) once if you do not have it yet.

Then run sharp-frame-extractor directly:

```bash
uvx sharp-frame-extractor input.mp4 --every 0.3
```

Outputs are written next to the input video by default, see [output](#output-directory) section below.

### Install as a package

If you prefer a traditional installation, you can install the package with pip:

```bash
pip install sharp-frame-extractor
```

Extract approximately 300 sharp frames from a video:

```bash
sharp-frame-extractor input.mp4 --count 300

# If the command is not found, run as a module:
python -m sharp_frame_extractor input.mp4 --count 300
```

## How it works

The extractor processes the video in consecutive blocks. You choose how blocks are defined:

* `--every SECONDS` defines blocks by time (one block per N seconds)
* `--count N` defines blocks by targeting a total of about N blocks across the video

For each block, it evaluates frames with a sharpness score and writes only the sharpest frame of that block to disk. Sharpness scoring is based on the Tenengrad focus measure (Sobel gradient energy), a common approach for focus and blur evaluation.

Notes:

* If an entire block is blurry (fast motion, heavy compression, defocus), the best frame in that block can still be blurry.
* With `--count`, the final number is approximate because it depends on video duration and block sizing.

## Usage

The command accepts one or more input videos. Choose exactly one sampling mode: `--count` or `--every`.

### Extract a target number of frames

```bash
sharp-frame-extractor input.mp4 --count 300
```

Use this when you want a roughly fixed number of frames per clip, regardless of duration.

### Extract one frame every N seconds

```bash
sharp-frame-extractor input.mp4 --every 0.25
```

Use this when you want a consistent sampling interval across clips.

### Multiple videos

```bash
sharp-frame-extractor a.mp4 b.mp4 c.mp4 --count 100
```

### Output directory

If you omit `--output`, results are written next to each input video:

* `./video.mp4` becomes `./video/frame-00000.png`, `./video/frame-00001.png`, ...

To write everything into a single base directory (still grouped per input video):

```bash
sharp-frame-extractor a.mp4 b.mp4 -o frames --every 2
```

Outputs:

* `frames/a/frame-00000.png`
* `frames/b/frame-00000.png`

### Performance tuning

By default, the extractor automatically chooses performance settings based on the workload and the available hardware. The options below let you override those defaults when you want more direct control.

There are three main tuning knobs, with two layers of parallelism:

* `-j/--jobs` (`max_video_jobs`) = how many videos are processed at the same time. Each job mainly acts as an orchestrator: it drives frame decoding and hands blocks to the analysis stage.
* `-w/--workers` (`max_workers`) = how many analysis workers run in parallel. Workers are separate processes that perform the CPU intensive sharpness scoring and are shared across all jobs.
* `-m/--memory-limit` (`memory_limit_mb`) = the total memory budget for frame buffers. This limit is split across active jobs, so increasing `--jobs` reduces the buffer size available per video. Default memory limit is chosen automatically based on available RAM.

How the pipeline behaves:

* A job processes a video block by block.
* Each block needs an available worker to be analyzed.
* If no worker is available, the job waits and does not keep decoding more blocks.
* Frame buffering is bounded by the global memory limit, preventing unbounded memory growth when many jobs are active.

Practical guidance:

* Processing a single video: keep `--jobs 1` and tune `--workers`. This usually controls total throughput.
* Processing many videos: pick a sensible `--workers` value first, often close to your CPU core count, then increase `--jobs` until the workers stay busy. If the CPU is already fully utilized, increasing `--jobs` will mostly add overhead without speeding things up.
* If you run many jobs at once and see increased waiting or reduced throughput, consider raising the memory limit so each job has enough buffering.

Example:

```bash
sharp-frame-extractor a.mp4 b.mp4 --count 200 -j 2 --workers 6
```

## Output

The extractor writes one image per processed block:

* `frame-00000.png`
* `frame-00001.png`
* ...

The index is the block index, not the original frame number.

### Help

```bash
sharp-frame-extractor --help
```

```text
Usage: sharp-frame-extractor [-h] [-o DIR] (--count N | --every SECONDS) [-j N] [-w N]
                             [-m MEMORY_MB]
                             VIDEO [VIDEO ...]

Extract sharp frames from a video by scoring frames within blocks. Choose exactly one
sampling mode: --count or --every.

Positional Arguments:
  VIDEO                 One or more input video files.

Options:
  -h, --help            show this help message and exit
  -o, --output DIR      Base output directory. If omitted, outputs are written to
                        "<video_parent>/<video_stem>/". If set, outputs are written to
                        "<DIR>/<video_stem>/". (default: None)
  --count N             Target number of frames to extract per input video. (default: None)
  --every SECONDS       Extract one sharp frame every N seconds. Supports decimals, for example 0.25. (default: None)
  -j, --jobs N          Max number of videos processed in parallel (video jobs). (default: auto)
  -w, --workers N       Total analysis worker processes shared across all video jobs. (default: auto)
  -m, --memory-limit MEMORY_MB
                        Global memory limit for frame buffers in MB (shared across jobs). (default: auto)
                        
Examples:
  Extract frames by target count:
    sharp-frame-extractor input.mp4 --count 300

  Extract one sharp frame every 0.25 seconds:
    sharp-frame-extractor input.mp4 --every 0.25

  Process multiple videos, outputs next to each input:
    sharp-frame-extractor a.mp4 b.mp4 --count 100

  Write outputs into a single base folder (per input subfolder):
    sharp-frame-extractor a.mp4 b.mp4 -o out --every 2
```

## Migrating from version 1 to version 2

Version 1 remains available on the [version-1 branch](https://github.com/cansik/sharp-frame-extractor/tree/version-1).

Changes in version 2:

* New command name and argument layout
* Updated sharpness scoring
* Better performance via improved parallelism and worker pool design

```bash
# v1
sfextract --window 300 test.mov

# v2
sharp-frame-extractor test.mov --every 0.3
```

```bash
# v1
sfextract --frame-count 30 test.mov

# v2
sharp-frame-extractor test.mov --count 30
```

If you relied on v1 options such as method selection, cropping, preview, or debug output, keep using version 1 for now.

## Development

This project uses `uv` for environment management and `ruff` for formatting and linting.

Setup:

```bash
uv sync --dev
```

Autoformat:

```bash
make autoformat
```

## About

Copyright (c) 2026 Florian Bruggisser  
Released under the MIT License. See [LICENSE](LICENSE) for details.
