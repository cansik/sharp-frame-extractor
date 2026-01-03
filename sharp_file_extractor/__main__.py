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

from sharp_file_extractor.analyzer.frame_analyzer_base import FrameAnalyzerTask, FrameAnalyzerResult
from sharp_file_extractor.analyzer.frame_analyzer_pool import FrameAnalyzerWorkerPool
from sharp_file_extractor.worker.Future import Future

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
    video_duration_seconds = float(video_info["duration"])
    video_fps = float(video_info["frame_rate"])
    total_video_frames = int(video_info["nb_frames"])
    video_frame_length_ms = 1000 / video_fps
    video_width = int(video_info["width"])
    video_height = int(video_info["height"])

    # calculate stream block size
    if options.frame_interval_seconds is not None:
        stream_block_size = max(1, int(round(options.frame_interval_seconds * video_fps)))
    elif options.total_frame_count is not None:
        stream_block_size = max(1, int(math.ceil(total_video_frames / options.total_frame_count)))
    else:
        progress.print("Please provide either frame-interval-seconds or total-frame_count.", style="bold yellow")
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
    parser = argparse.ArgumentParser("sfextract", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="Video input files.")
    parser.add_argument("--outputs", nargs="*",
                        help="Output directories for each input file. "
                             "If empty, name of video file will be used as output directory. "
                             "If only one is provided, it will be used as output directory for all videos.")
    parser.add_argument("--total-frames", type=float, default=100, help="Total frames to extract.")
    parser.add_argument("--frame-window", type=float, default=None,
                        help="Frame analysis window in seconds (overwrites total-frames).")
    parser.add_argument("--max-video-threads", type=int, default=None,
                        help="Max parallel videos to process.")
    parser.add_argument("--max-analyzers", type=int, default=None, help="Max parallel analyzers.")
    return parser.parse_args()


def main():
    console = Console()

    # process arguments
    args = parse_args()
    input_paths = [Path(a) for a in args.inputs]
    output_paths = [Path(a) for a in args.outputs] if args.outputs else []

    total_frames = args.total_frames
    frame_window = args.frame_window

    max_video_threads: int = args.max_video_threads if args.max_video_threads else cpu_count_fraction(0.4)
    max_analyzers: int = args.max_analyzers if args.max_analyzers else cpu_count_fraction(0.4)

    # create pool
    global analyzer_pool
    analyzer_pool = FrameAnalyzerWorkerPool(max_analyzers)

    # check output paths correctness
    if len(input_paths) != len(output_paths):
        if len(output_paths) == 0:
            output_paths = [i.parent / i.stem for i in input_paths]
        elif len(output_paths) == 1:
            output_paths = [output_paths[0]] * len(input_paths)
        else:
            console.print("Please provide zero, one or as many output as inputs.", style="bold yellow")
            exit(1)

    # create options
    default_options = ExtractionOptions(
        frame_interval_seconds=frame_window,
        total_frame_count=total_frames
    )

    # create tasks
    with console.status("creating tasks..."):
        tasks: list[ExtractionTask] = [
            ExtractionTask(i.absolute(), o.absolute(), default_options)
            for i, o in zip(input_paths, output_paths)
        ]

    task_count = len(tasks)

    # update video thread tasks
    max_video_threads = min(task_count, max_video_threads)

    # print processing info
    console.print(f"Running {task_count} tasks with {max_video_threads} video threads and {max_analyzers} analyzers.")

    # run processing
    start_time = time.time()
    analyzer_pool.start()
    with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeRemainingColumn(),
            MofNCompleteColumn()
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
