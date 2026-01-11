from __future__ import annotations

import argparse
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, stdev
from time import perf_counter
from typing import Iterable

from rich.box import SIMPLE
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from sharp_frame_extractor.args_utils import default_concurrency, positive_float, positive_int
from sharp_frame_extractor.models import ExtractionOptions
from sharp_frame_extractor.output.file_output_handler import FileOutputHandler
from sharp_frame_extractor.SharpFrameExtractor import ExtractionTask, SharpFrameExtractor


@dataclass(frozen=True)
class Scenario:
    name: str
    options: ExtractionOptions


def _format_seconds(s: float) -> str:
    if s < 1:
        return f"{s * 1000:.1f} ms"
    return f"{s:.3f} s"


def _safe_stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return stdev(values)


def parse_args() -> argparse.Namespace:
    default_jobs, default_workers = default_concurrency()

    parser = argparse.ArgumentParser(
        prog="sharp-frame-extractor-bench",
        description="Benchmark sharp frame extraction over multiple scenarios and runs.",
    )

    parser.add_argument("video", metavar="VIDEO", help="Input video file to benchmark.")

    parser.add_argument(
        "--runs",
        type=positive_int,
        default=5,
        metavar="N",
        help="Number of benchmark runs per scenario. Default: 5.",
    )

    parser.add_argument(
        "--every-small",
        type=positive_float,
        default=0.5,
        metavar="SECONDS",
        help="Small interval scenario in seconds. Default: 2.",
    )

    parser.add_argument(
        "--every-large",
        type=positive_float,
        default=5.0,
        metavar="SECONDS",
        help="Large interval scenario in seconds. Default: 5.0.",
    )

    parser.add_argument(
        "-o",
        "--output",
        metavar="DIR",
        help=(
            "Where to write benchmark outputs.\nIf omitted, writes under a temp folder and deletes it automatically."
        ),
    )

    parser.add_argument(
        "--keep-outputs",
        action="store_true",
        help="Keep benchmark outputs (only relevant when --output is set).",
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
        type=positive_int,
        default=default_workers,
        metavar="N",
        help=f"Max number of frame analyzer workers. Default: {default_workers}.",
    )

    return parser.parse_args()


def build_scenarios(args: argparse.Namespace) -> list[Scenario]:
    scenarios: list[Scenario] = [
        Scenario(
            name=f"every_small_{args.every_small:g}s",
            options=ExtractionOptions.from_interval(args.every_small),
        ),
        Scenario(
            name=f"every_large_{args.every_large:g}s",
            options=ExtractionOptions.from_interval(args.every_large),
        ),
    ]

    # Add new scenarios here, for example:
    # scenarios.append(Scenario("count_300", ExtractionOptions.from_count(300)))

    return scenarios


def ensure_empty_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def run_bench(
    *,
    video_path: Path,
    scenarios: Iterable[Scenario],
    runs: int,
    jobs: int,
    workers: int,
    bench_root: Path,
    console: Console,
) -> dict[str, list[float]]:
    results: dict[str, list[float]] = {s.name: [] for s in scenarios}

    output_handlers = [FileOutputHandler()]
    max_video_threads = max(1, min(jobs, 1))

    total_steps = runs * len(list(scenarios))

    with SharpFrameExtractor(output_handlers, max_video_threads, workers) as sfe:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            MofNCompleteColumn(),
            console=console,
        ) as progress:
            overall_id = progress.add_task(description="benchmark runs", total=total_steps)
            scenario_id = progress.add_task(description="scenario", total=runs)

            for scenario in scenarios:
                progress.update(scenario_id, description=f"{scenario.name}", total=runs, completed=0)

                for run_idx in range(1, runs + 1):
                    out_dir = bench_root / scenario.name / f"run_{run_idx:03d}"
                    ensure_empty_dir(out_dir)

                    task = ExtractionTask(video_path, out_dir, scenario.options)

                    t0 = perf_counter()
                    _ = sfe.process([task])
                    t1 = perf_counter()

                    results[scenario.name].append(t1 - t0)

                    progress.advance(scenario_id, 1)
                    progress.advance(overall_id, 1)

    return results


def print_summary(console: Console, results: dict[str, list[float]]) -> None:
    table = Table(title="Benchmark summary", box=SIMPLE)
    table.add_column("Scenario", no_wrap=True)
    table.add_column("Runs", justify="right")
    table.add_column("Mean", justify="right")
    table.add_column("Median", justify="right")
    table.add_column("Std dev", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")

    for name, times in results.items():
        m = mean(times)
        med = median(times)
        sd = _safe_stdev(times)
        mn = min(times)
        mx = max(times)

        table.add_row(
            name,
            str(len(times)),
            _format_seconds(m),
            _format_seconds(med),
            _format_seconds(sd),
            _format_seconds(mn),
            _format_seconds(mx),
        )

    console.print(table)

    runs_table = Table(title="Per run timings", box=SIMPLE)
    runs_table.add_column("Scenario", no_wrap=True)
    runs_table.add_column("Run", justify="right")
    runs_table.add_column("Time", justify="right")

    for name, times in results.items():
        for i, t in enumerate(times, start=1):
            runs_table.add_row(name, str(i), _format_seconds(t))

    console.print(runs_table)


def main() -> None:
    console = Console()
    args = parse_args()

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        raise SystemExit(f'Video not found: "{video_path}"')

    scenarios = build_scenarios(args)

    bench_root: Path
    created_root: Path | None = None

    if args.output:
        base = Path(args.output).expanduser().resolve()
        base.mkdir(parents=True, exist_ok=True)
        bench_root = base / f"bench_{video_path.stem}_{int(time.time())}"
        created_root = bench_root
    else:
        base = Path.cwd() / ".sharp_frame_extractor_bench_tmp"
        bench_root = base / f"bench_{video_path.stem}_{int(time.time())}"
        created_root = bench_root

    ensure_empty_dir(bench_root)

    console.print(
        f'Video: "{video_path.name}"\n'
        f"Scenarios: {len(scenarios)}\n"
        f"Runs per scenario: {args.runs}\n"
        f"Workers: {args.workers}\n"
        f'Outputs: "{bench_root}"'
    )

    results = run_bench(
        video_path=video_path,
        scenarios=scenarios,
        runs=int(args.runs),
        jobs=int(args.jobs),
        workers=int(args.workers),
        bench_root=bench_root,
        console=console,
    )

    print_summary(console, results)

    if args.output and args.keep_outputs:
        console.print(f'Kept outputs in: "{bench_root}"')
        return

    if created_root is not None:
        shutil.rmtree(created_root)


if __name__ == "__main__":
    main()
