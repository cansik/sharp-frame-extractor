import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Self

import cv2
import ffmpegio
import numpy as np

from sharp_frame_extractor.analyzer.frame_analyzer_base import FrameAnalyzerResult, FrameAnalyzerTask
from sharp_frame_extractor.analyzer.frame_analyzer_pool import FrameAnalyzerWorkerPool
from sharp_frame_extractor.args_utils import default_concurrency
from sharp_frame_extractor.event import Event
from sharp_frame_extractor.models import (
    BlockEvent,
    BlockProcessedEvent,
    ExtractionResult,
    ExtractionTask,
    TaskAnalyzedEvent,
    TaskEvent,
    TaskFinishedEvent,
    TaskStartedEvent,
)
from sharp_frame_extractor.worker.Future import Future


class SharpFrameExtractor:
    def __init__(self, max_video_jobs: int | None, max_workers: int | None):
        default_jobs, default_workers = default_concurrency()

        self._max_video_jobs = max_video_jobs or default_jobs
        self._max_workers = max_workers or default_workers

        self._analyzer_pool = FrameAnalyzerWorkerPool(self._max_workers)

        # callbacks
        self.on_task_event: Event[TaskEvent] = Event()
        self.on_block_event: Event[BlockEvent] = Event()

    def start(self):
        self._analyzer_pool.start()

    def process(self, tasks: list[ExtractionTask]) -> list[ExtractionResult]:
        results: list[ExtractionResult] = []

        # Sequential execution for debugging or single worker
        if self._max_video_jobs <= 1:
            for task in tasks:
                result = self._process_extraction_task(task)
                results.append(result)
            return results

        # Parallel threaded execution with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self._max_video_jobs) as executor:
            futures = {}
            for task in tasks:
                # Submit tasks to executor and track their futures
                future = executor.submit(self._process_extraction_task, task)
                futures[future] = task

            # Process tasks as workers become available
            for future in as_completed(futures):
                # Wait for the future to complete
                result = future.result()
                results.append(result)

        # order results by input id
        results.sort(key=lambda r: r.task_id)

        return results

    def stop(self):
        self._analyzer_pool.stop()

    def _process_extraction_task(self, task: ExtractionTask) -> ExtractionResult:
        self.on_task_event(TaskStartedEvent(task))

        video_path = task.video_path
        result_path = task.result_path
        options = task.options

        # read stream info
        video_streams = ffmpegio.probe.video_streams_basic(str(video_path))
        video_info = video_streams[0]

        # extract video information
        video_duration_seconds = float(video_info["duration"])
        video_fps = float(video_info["frame_rate"])

        if "nb_frames" in video_info:
            total_video_frames = int(video_info["nb_frames"])
        else:
            total_video_frames = math.ceil(video_duration_seconds * video_fps)

        # calculate stream block size
        if options.frame_interval_seconds is not None:
            stream_block_size = max(1, int(round(options.frame_interval_seconds * video_fps)))
        elif options.total_frame_count is not None:
            stream_block_size = max(1, int(math.ceil(total_video_frames / options.total_frame_count)))
        else:
            raise ValueError('Please provide either "--every" or "--count".')

        # ensure output path exists
        result_path.mkdir(parents=True, exist_ok=True)

        # setup progress bar
        total_sub_tasks = int(math.ceil(total_video_frames / stream_block_size))
        self.on_task_event(TaskAnalyzedEvent(task, total_blocks=total_sub_tasks))

        submitted_tasks: list[Future] = []

        def on_task_finished(future: Future[FrameAnalyzerResult]):
            result = future.result()

            # todo: handle the export in a registered output handler
            output_file_name = task.result_path / f"frame-{result.block_index:05d}.png"

            if output_file_name.exists():
                output_file_name.unlink(missing_ok=True)

            cv2.imwrite(str(output_file_name.absolute()), result.frame)

            self.on_block_event(BlockProcessedEvent(result.block_index, task, result))

        # start reading video file
        block_index = 0
        with ffmpegio.open(str(video_path), "rv", blocksize=stream_block_size, pix_fmt="rgb24") as fin:
            for frames in fin:
                # convert rgb to bgr frames
                frames_bgr = np.empty_like(frames)
                for i in range(frames.shape[0]):
                    frames_bgr[i] = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)

                # analyze video block
                worker_task = self._analyzer_pool.submit_task(FrameAnalyzerTask(block_index, frames_bgr))
                worker_task.add_done_callback(on_task_finished)
                submitted_tasks.append(worker_task)

                block_index += 1

        # wait for all tasks to be done
        for worker_task in submitted_tasks:
            worker_task.result()

        self.on_task_event(TaskFinishedEvent(task))
        return ExtractionResult(task.task_id)

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
