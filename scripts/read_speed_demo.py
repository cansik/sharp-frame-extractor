import argparse
import time
from dataclasses import dataclass
from statistics import mean, median
from typing import List, Type

import psutil
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from sharp_frame_extractor.reader.av_video_reader import AvVideoReader
from sharp_frame_extractor.reader.batched_video_reader import BatchedVideoReader
from scripts.reader.ffmpegio_video_reader import FfmpegIoVideoReader
from sharp_frame_extractor.reader.opencv_video_reader import OpencvVideoReader
from sharp_frame_extractor.reader.video_reader import PixelFormat, VideoReader


@dataclass
class Scenario:
    name: str
    reader_class: Type[VideoReader]
    chunk_size: int = 1


@dataclass
class ScenarioResult:
    name: str
    duration: float
    fps: float
    pixels_processed: int
    cpu_max: float
    cpu_avg: float
    cpu_med: float
    mem_max: float
    mem_avg: float
    mem_med: float


def _format_bytes(b: float) -> str:
    return f"{b / 1024 / 1024:.1f} MB"


def run_scenario(scenario: Scenario, video_path: str, progress: Progress) -> ScenarioResult:
    # Setup reader
    if scenario.chunk_size > 1:
        # For batched reading, we instantiate the base reader and wrap it
        base_reader = scenario.reader_class(video_path)
        reader = BatchedVideoReader(base_reader, scenario.chunk_size)
        # Probe using the base reader (BatchedVideoReader delegates probe, but good to be explicit)
        info = reader.probe()
    else:
        reader = scenario.reader_class(video_path)
        info = reader.probe()

    total_frames = info.total_frames
    task = progress.add_task(f"[cyan]{scenario.name}", total=total_frames)

    # Resource monitoring
    process = psutil.Process()
    cpu_usage = []
    mem_usage = []

    # Initial CPU call to start measurement
    process.cpu_percent(interval=None)

    start_time = time.time()
    pixels = 0

    # We'll sample resources every N frames to avoid overhead
    sample_interval = 10
    frame_count = 0

    for batch in reader.read_frames(pixel_format=PixelFormat.RGB24):
        # Handle both single frame (H,W,C) and batched (N,H,W,C)
        if batch.ndim == 4:
            n = batch.shape[0]
            pixels += batch.size
        else:
            n = 1
            pixels += batch.size

        progress.update(task, advance=n)

        frame_count += 1
        if frame_count % sample_interval == 0:
            cpu_usage.append(process.cpu_percent(interval=None))
            mem_usage.append(process.memory_info().rss)

    end_time = time.time()
    duration = end_time - start_time

    reader.release()

    # Final sample
    cpu_usage.append(process.cpu_percent(interval=None))
    mem_usage.append(process.memory_info().rss)

    # Calculate stats
    if not cpu_usage:
        cpu_usage = [0.0]
    if not mem_usage:
        mem_usage = [0.0]

    return ScenarioResult(
        name=scenario.name,
        duration=duration,
        fps=total_frames / duration if duration > 0 else 0,
        pixels_processed=pixels,
        cpu_max=max(cpu_usage),
        cpu_avg=mean(cpu_usage),
        cpu_med=median(cpu_usage),
        mem_max=max(mem_usage),
        mem_avg=mean(mem_usage),
        mem_med=median(mem_usage),
    )


def main():
    parser = argparse.ArgumentParser(description="Video reading speed benchmark")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    args = parser.parse_args()

    scenarios: List[Scenario] = [
        Scenario("Opencv", OpencvVideoReader),
        Scenario("Av", AvVideoReader),
        Scenario("FfmpegIo", FfmpegIoVideoReader),
        Scenario("FfmpegIo (Batch=32)", FfmpegIoVideoReader, chunk_size=32),
        Scenario("Av (Batch=32)", AvVideoReader, chunk_size=32),
    ]

    results = []

    with Progress() as progress:
        for scenario in scenarios:
            results.append(run_scenario(scenario, args.video_path, progress))

    console = Console()

    # Performance Table
    table = Table(title="Benchmark Results - Speed")
    table.add_column("Scenario", style="cyan")
    table.add_column("Duration (s)", justify="right")
    table.add_column("FPS", justify="right")

    for res in reversed(sorted(results, key=lambda e: e.fps)):
        table.add_row(res.name, f"{res.duration:.2f}", f"{res.fps:.2f}")

    console.print(table)

    # Resource Table
    res_table = Table(title="Benchmark Results - Resources")
    res_table.add_column("Scenario", style="cyan")
    res_table.add_column("CPU Max", justify="right")
    res_table.add_column("CPU Avg", justify="right")
    res_table.add_column("CPU Med", justify="right")
    res_table.add_column("Mem Max", justify="right")
    res_table.add_column("Mem Avg", justify="right")
    res_table.add_column("Mem Med", justify="right")

    for res in reversed(sorted(results, key=lambda e: e.fps)):
        res_table.add_row(
            res.name,
            f"{res.cpu_max:.1f}%",
            f"{res.cpu_avg:.1f}%",
            f"{res.cpu_med:.1f}%",
            _format_bytes(res.mem_max),
            _format_bytes(res.mem_avg),
            _format_bytes(res.mem_med),
        )

    console.print(res_table)


if __name__ == "__main__":
    main()
