import argparse
import time
from datetime import timedelta
from pathlib import Path

from rich.console import Console
from rich.progress import (
    MofNCompleteColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich_argparse import ArgumentDefaultsRichHelpFormatter

from sharp_frame_extractor.args_utils import default_concurrency, default_memory_limit_mb, positive_float, positive_int
from sharp_frame_extractor.models import (
    BlockAnalyzedEvent,
    BlockEvent,
    BlockFrameExtracted,
    ExtractionOptions,
    TaskAnalyzedEvent,
    TaskEvent,
    TaskFinishedEvent,
    TaskPreparedEvent,
    TaskStartedEvent,
)
from sharp_frame_extractor.output.file_output_handler import FileOutputHandler
from sharp_frame_extractor.SharpFrameExtractor import ExtractionTask, SharpFrameExtractor
from sharp_frame_extractor.ui.progress_bar import StatefulBarColumn


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
    default_memory_limit = default_memory_limit_mb()

    parser = argparse.ArgumentParser(
        prog="sharp-frame-extractor",
        description=(
            "Extract the sharpest frame from regular blocks of a video.\n"
            "Choose exactly one sampling mode: --count or --every."
        ),
        epilog=examples,
        formatter_class=ArgumentDefaultsRichHelpFormatter,
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
        help="Max number of videos processed in parallel.",
    )

    parser.add_argument(
        "-w",
        "--workers",
        dest="workers",
        type=positive_int,
        default=default_workers,
        metavar="N",
        help="Max number of frame analyzer workers.",
    )

    parser.add_argument(
        "-m",
        "--memory-limit",
        dest="memory_limit",
        type=positive_int,
        default=default_memory_limit,
        metavar="N",
        help="Max memory limit in MB.",
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
    max_memory_limit_mb: int = int(args.memory_limit)

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
    console.print(
        f"Running {task_count} tasks "
        f"with {max_video_threads} jobs, "
        f"{max_workers} workers "
        f"and a memory limit of ~{max_memory_limit_mb / 1024:.1f} GB."
    )

    # create output handler
    output_handlers = [FileOutputHandler()]

    # run processing
    start_time = time.time()
    with SharpFrameExtractor(output_handlers, max_video_threads, max_workers, max_memory_limit_mb) as sfe:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            StatefulBarColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            MofNCompleteColumn(),
        ) as progress:
            # Create an overall progress bar
            main_task_id = progress.add_task(
                description="[bold]Sharp Frame Extractor[/bold]",
                total=task_count,
                bar_complete_style="bright_white",
                bar_finished_style="bright_white",
                bar_pulse_style="bright_white",
            )

            task_to_progress_lut: dict[int, TaskID] = {}

            def _set_state(t: ExtractionTask, label: str, color: str) -> None:
                progress_task_id = task_to_progress_lut[t.task_id]
                progress.update(
                    progress_task_id,
                    description=f"[{color}]{label}[/{color}] [bold]{t.video_path.name}[/bold]",
                    bar_complete_style=color,
                    bar_finished_style=color,
                    bar_pulse_style=color,
                )

            # handle progress events
            @sfe.on_task_event.register
            def _on_task_event(event: TaskEvent):
                if isinstance(event, TaskStartedEvent):
                    task_to_progress_lut[event.task.task_id] = progress.add_task(
                        description=f"{event.task.task_id}", total=None
                    )
                    _set_state(event.task, "preparing", "gold1")
                elif isinstance(event, TaskPreparedEvent):
                    progress.update(
                        task_to_progress_lut[event.task.task_id], total=event.total_blocks + event.total_frames
                    )
                    _set_state(event.task, "analyzing", "slate_blue1")
                elif isinstance(event, TaskAnalyzedEvent):
                    progress.update(
                        task_to_progress_lut[event.task.task_id], total=event.total_blocks + event.total_frames
                    )
                    _set_state(event.task, "extracting", "dodger_blue1")
                elif isinstance(event, TaskFinishedEvent):
                    _set_state(event.task, "done", "spring_green1")
                    progress.stop_task(task_to_progress_lut[event.task.task_id])
                    progress.advance(main_task_id)

            @sfe.on_block_event.register
            def _on_block_event(event: BlockEvent):
                if isinstance(event, BlockAnalyzedEvent) or isinstance(event, BlockFrameExtracted):
                    progress.advance(task_to_progress_lut[event.task.task_id])

            # run process
            _ = sfe.process(tasks)

    end_time = time.time()
    console.print(f"It took {str(timedelta(seconds=end_time - start_time))} to process {task_count} tasks.")


if __name__ == "__main__":
    main()
