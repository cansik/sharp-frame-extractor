import argparse
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import cv2
import ffmpegio
import numpy as np
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, MofNCompleteColumn

from sharp_frame_extractor.analyzer.frame_analyzer_base import FrameAnalyzerTask, FrameAnalyzerResult
from sharp_frame_extractor.analyzer.frame_analyzer_pool import FrameAnalyzerWorkerPool
from sharp_frame_extractor.args_utils import positive_int, positive_float, default_concurrency
from sharp_frame_extractor.worker.Future import Future

analyzer_pool: FrameAnalyzerWorkerPool | None = None


@dataclass
class ExtractionOptions:
    # either one of the two have ot be set
    frame_interval_seconds: float | None = None
    total_frame_count: int | None = None


@dataclass
class ExtractionTask:
    video_path: Path
    result_path: Path
    options: ExtractionOptions


def process_extraction_task(task: ExtractionTask, progress: Progress) -> None:
    task_id = progress.add_task(description=f"analyzing {task.video_path.name}", total=None)

    video_path = task.video_path
    result_path = task.result_path
    options = task.options

    # read stream info
    video_streams = ffmpegio.probe.video_streams_basic(str(video_path))
    video_info = video_streams[0]

    # extract video information
    # video_duration_seconds = float(video_info["duration"])
    video_fps = float(video_info["frame_rate"])
    total_video_frames = int(video_info["nb_frames"])
    # video_frame_length_ms = 1000 / video_fps
    # video_width = int(video_info["width"])
    # video_height = int(video_info["height"])

    # calculate stream block size
    if options.frame_interval_seconds is not None:
        stream_block_size = max(1, int(round(options.frame_interval_seconds * video_fps)))
    elif options.total_frame_count is not None:
        stream_block_size = max(1, int(math.ceil(total_video_frames / options.total_frame_count)))
    else:
        progress.print('Please provide either "--every" or "--count".', style="bold yellow")
        progress.stop_task(task_id)
        return

    # ensure output path exists
    result_path.mkdir(parents=True, exist_ok=True)

    # setup progress bar
    total_sub_tasks = int(math.ceil(total_video_frames / stream_block_size))
    progress.update(task_id, total=total_sub_tasks, description=f"processing {task.video_path.name}")

    submitted_tasks: list[Future] = []

    def on_task_finished(future: Future[FrameAnalyzerResult]):
        result = future.result()
        output_file_name = task.result_path / f"frame-{result.block_index:05d}.png"

        if output_file_name.exists():
            output_file_name.unlink(missing_ok=True)

        cv2.imwrite(str(output_file_name.absolute()), result.frame)
        progress.update(task_id, advance=1)

    # start reading video file
    block_index = 0
    with ffmpegio.open(str(video_path), "rv", blocksize=stream_block_size, pix_fmt="rgb24") as fin:
        for frames in fin:
            # convert rgb to bgr frames
            frames_bgr = np.empty_like(frames)
            for i in range(frames.shape[0]):
                frames_bgr[i] = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)

            # analyze video block
            worker_task = analyzer_pool.submit_task(FrameAnalyzerTask(block_index, frames_bgr))
            worker_task.add_done_callback(on_task_finished)
            submitted_tasks.append(worker_task)

            block_index += 1

    # wait for all tasks to be done
    for worker_task in submitted_tasks:
        worker_task.result()

    progress.update(task_id, completed=total_sub_tasks)
    progress.stop_task(task_id)


def cpu_count_fraction(factor: float, min_value: int = 1) -> int:
    return max(min_value, int(os.cpu_count() * factor))


def parse_args() -> argparse.Namespace:
    examples = """
Examples:
  Extract frames by target count:
    sharp-frame-extractor input.mp4 --count 300

  Extract one sharp frame every 0.25 seconds:
    sharp-frame-extractor input.mp4 --every 0.25

  Process multiple videos, outputs next to each input:
    sharp-frame-extractor a.mp4 b.mp4 --count 100

  Write outputs into a single base folder (per input subfolder):
    sharp-frame-extractor a.mp4 b.mp4 -o out --every 2
"""

    default_jobs, default_workers = default_concurrency()

    parser = argparse.ArgumentParser(
        prog="sharp-frame-extractor",
        description=(
            "Extract the sharpest frame from regular blocks of a video.\n"
            "Choose exactly one sampling mode: --count or --every."
        ),
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="VIDEO",
        help="One or more input video files.",
    )

    parser.add_argument(
        "-o",
        "--output",
        metavar="DIR",
        help=(
            "Base output directory.\n"
            'If omitted, outputs are written to "<video_parent>/<video_stem>/".\n'
            'If set, outputs are written to "<DIR>/<video_stem>/".'
        ),
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--count",
        type=positive_int,
        metavar="N",
        help="Target number of frames to extract per input video.",
    )
    mode.add_argument(
        "--every",
        type=positive_float,
        metavar="SECONDS",
        help="Extract one sharp frame every N seconds. Supports decimals, for example 0.25.",
    )

    parser.add_argument(
        "-j",
        "--jobs",
        type=positive_int,
        default=default_jobs,
        metavar="N",
        help=f"Max number of videos processed in parallel. Default: {default_jobs}.",
    )

    parser.add_argument(
        "--workers",
        "--analyzers",
        dest="workers",
        type=positive_int,
        default=default_workers,
        metavar="N",
        help=f"Max number of analyzer workers. Default: {default_workers}.",
    )

    return parser.parse_args()


def main():
    console = Console()
    args = parse_args()

    input_paths: list[Path] = [Path(a) for a in args.inputs]
    output_base_dir: Path | None = Path(args.output) if args.output else None

    count: int | None = args.count
    every_seconds: float | None = args.every

    max_video_threads: int = int(args.jobs)
    max_workers: int = int(args.workers)

    if output_base_dir is not None:
        output_paths: list[Path] = [output_base_dir / p.stem for p in input_paths]
    else:
        output_paths = [p.parent / p.stem for p in input_paths]

    if every_seconds is not None:
        default_options = ExtractionOptions(frame_interval_seconds=every_seconds, total_frame_count=None)
    else:
        default_options = ExtractionOptions(frame_interval_seconds=None, total_frame_count=count)

    # create tasks
    with console.status("creating tasks..."):
        tasks: list[ExtractionTask] = [
            ExtractionTask(i.absolute(), o.absolute(), default_options) for i, o in zip(input_paths, output_paths)
        ]

    task_count = len(tasks)

    # update video thread tasks
    max_video_threads = min(task_count, max_video_threads)

    # print processing info
    console.print(f"Running {task_count} tasks with {max_video_threads} jobs and {max_workers} workers.")

    # create pool
    global analyzer_pool
    analyzer_pool = FrameAnalyzerWorkerPool(max_workers)

    # run processing
    start_time = time.time()
    analyzer_pool.start()
    with Progress(
        TextColumn("[progress.description]{task.description}"), BarColumn(), TimeRemainingColumn(), MofNCompleteColumn()
    ) as progress:
        # Create an overall progress bar
        overall_task_id = progress.add_task(description="Sharp Frame Extractor", total=task_count)

        # Sequential execution for debugging or single worker
        if max_video_threads <= 1:
            for task in tasks:
                process_extraction_task(task, progress)
                progress.advance(overall_task_id)
        else:
            # Parallel threaded execution with ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max_video_threads) as executor:
                futures = {}
                for task in tasks:
                    # Submit tasks to executor and track their futures
                    future = executor.submit(process_extraction_task, task, progress)
                    futures[future] = task

                # Process tasks as workers become available
                for future in as_completed(futures):
                    # Wait for the future to complete
                    future.result()
                    progress.advance(overall_task_id)

    analyzer_pool.stop()
    end_time = time.time()
    console.print(f"It took {str(timedelta(seconds=end_time - start_time))} to process {task_count} tasks.")


if __name__ == "__main__":
    main()
