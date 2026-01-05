import argparse
import time
from datetime import timedelta
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from sharp_frame_extractor.args_utils import default_concurrency, positive_float, positive_int
from sharp_frame_extractor.models import (
    BlockEvent,
    BlockProcessedEvent,
    ExtractionOptions,
    TaskAnalyzedEvent,
    TaskEvent,
    TaskFinishedEvent,
    TaskStartedEvent,
)
from sharp_frame_extractor.output.file_output_handler import FileOutputHandler
from sharp_frame_extractor.SharpFrameExtractor import ExtractionTask, SharpFrameExtractor


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
        dest="workers",
        type=positive_int,
        default=default_workers,
        metavar="N",
        help=f"Max number of frame analyzer workers. Default: {default_workers}.",
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
        default_options = ExtractionOptions.from_interval(every_seconds)
    else:
        default_options = ExtractionOptions.from_count(count)

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

    # create output handler
    output_handlers = [FileOutputHandler()]

    # run processing
    start_time = time.time()
    with SharpFrameExtractor(output_handlers, max_video_threads, max_workers) as sfe:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            MofNCompleteColumn(),
        ) as progress:
            # Create an overall progress bar
            main_task_id = progress.add_task(description="Sharp Frame Extractor", total=task_count)

            task_to_progress_lut: dict[int, TaskID] = {}

            # handle progress events
            @sfe.on_task_event.register
            def _on_task_event(event: TaskEvent):
                if isinstance(event, TaskStartedEvent):
                    task_to_progress_lut[event.task.task_id] = progress.add_task(
                        description=f"analyzing {event.task.video_path.name}", total=None
                    )
                elif isinstance(event, TaskAnalyzedEvent):
                    progress.update(
                        task_to_progress_lut[event.task.task_id],
                        total=event.total_blocks,
                        description=f"processing {event.task.video_path.name}",
                    )
                elif isinstance(event, TaskFinishedEvent):
                    progress.stop_task(task_to_progress_lut[event.task.task_id])
                    progress.advance(main_task_id)

            @sfe.on_block_event.register
            def _on_block_event(event: BlockEvent):
                if isinstance(event, BlockProcessedEvent):
                    progress.advance(task_to_progress_lut[event.task.task_id])

            # run process
            _ = sfe.process(tasks)

    end_time = time.time()
    console.print(f"It took {str(timedelta(seconds=end_time - start_time))} to process {task_count} tasks.")


if __name__ == "__main__":
    main()
