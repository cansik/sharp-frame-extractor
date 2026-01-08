import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from itertools import chain
from typing import Self, Sequence

import ffmpegio
import numpy as np

from sharp_frame_extractor.analyzer.frame_analyzer_base import FrameAnalyzerResult, FrameAnalyzerTask
from sharp_frame_extractor.analyzer.frame_analyzer_pool import FrameAnalyzerWorkerPool
from sharp_frame_extractor.args_utils import MIN_MEMORY_LIMIT, default_concurrency, default_memory_limit_mb
from sharp_frame_extractor.event import Event
from sharp_frame_extractor.memory.shared_ndarray import SharedNDArrayRef, SharedNDArrayStoreBase
from sharp_frame_extractor.memory.shared_ndarray_pool import PooledSharedNDArrayStore
from sharp_frame_extractor.models import (
    BlockAnalyzedEvent,
    BlockEvent,
    BlockFrameExtracted,
    ExtractionResult,
    ExtractionTask,
    TaskAnalyzedEvent,
    TaskEvent,
    TaskFinishedEvent,
    TaskPreparedEvent,
    TaskStartedEvent,
    VideoFrameInfo,
)
from sharp_frame_extractor.output.frame_output_handler_base import FrameOutputHandlerBase
from sharp_frame_extractor.worker.Future import Future


class SharpFrameExtractor:
    def __init__(
        self,
        output_handlers: Sequence[FrameOutputHandlerBase],
        max_video_jobs: int | None = None,
        max_workers: int | None = None,
        memory_limit_mb: int | None = None,
    ):
        default_jobs, default_workers = default_concurrency()
        default_memory_limit = default_memory_limit_mb()

        self._output_handlers = output_handlers

        self._max_video_jobs = max_video_jobs or default_jobs
        self._max_workers = max_workers or default_workers
        self._total_memory_limit_mb = memory_limit_mb or default_memory_limit
        self.memory_limit_per_job_mb = max(
            MIN_MEMORY_LIMIT, math.ceil(self._total_memory_limit_mb / self._max_video_jobs)
        )

        self._analyzer_pool = FrameAnalyzerWorkerPool(self._max_workers)

        # callbacks
        self.on_task_event: Event[TaskEvent] = Event()
        self.on_block_event: Event[BlockEvent] = Event()

        # internal defaults
        self._preferred_block_size = 32
        self._analysis_pixel_format = "gray"
        self._analysis_channels = 1
        self._extraction_pixel_format = "rgb24"
        self._extraction_channels = 3

    def start(self):
        self._analyzer_pool.start()

        for handler in self._output_handlers:
            handler.open()

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

        for handler in self._output_handlers:
            handler.close()

    def _process_extraction_task(self, task: ExtractionTask) -> ExtractionResult:
        self.on_task_event(TaskStartedEvent(task))

        video_path = task.video_path
        options = task.options

        # read stream info
        video_streams = ffmpegio.probe.video_streams_basic(str(video_path))
        video_info = video_streams[0]

        # extract video information
        video_duration_seconds = float(video_info["duration"])
        video_fps = float(video_info["frame_rate"])

        video_width = int(video_info["width"])
        video_height = int(video_info["height"])

        if "nb_frames" in video_info:
            total_video_frames = int(video_info["nb_frames"])
        else:
            total_video_frames = math.ceil(video_duration_seconds * video_fps)

        # calculate frame interval for selecting the amount of output frames
        if options.frame_interval_seconds is not None:
            frame_interval = max(1, int(round(options.frame_interval_seconds * video_fps)))
        elif options.total_frame_count is not None:
            frame_interval = max(1, int(math.ceil(total_video_frames / options.total_frame_count)))
        else:
            raise ValueError('Please provide either "--every" or "--count".')

        # total frames to extract
        total_frames = int(math.ceil(total_video_frames / frame_interval))

        # calculate stream block size
        possible_block_size = self._calculate_block_size(
            video_width, video_height, self._extraction_channels, self.memory_limit_per_job_mb
        )

        # Distribute memory among the worker buffers
        max_block_size_per_worker = max(1, possible_block_size // self._max_workers)
        stream_block_size = min(self._preferred_block_size, max_block_size_per_worker)

        # setup progress bar for analysis
        total_sub_tasks = int(math.ceil(total_video_frames / stream_block_size))
        self.on_task_event(TaskPreparedEvent(task, total_blocks=total_sub_tasks, total_frames=total_frames))

        # prepare shared memory store
        buffer_size = video_width * video_height * self._analysis_channels * stream_block_size

        # limit buffers to max workers to prevent over-allocation
        with PooledSharedNDArrayStore(item_size=buffer_size, n_buffers=self._max_workers) as store:
            # analyze video first
            interval_ids, frame_ids, scores = self._analyze_frames(task, stream_block_size, frame_interval, store)
            self.on_task_event(TaskAnalyzedEvent(task, total_blocks=total_sub_tasks, total_frames=total_frames))

        # extraction run
        self._extract_frames(task, stream_block_size, interval_ids, frame_ids, scores)

        self.on_task_event(TaskFinishedEvent(task))
        return ExtractionResult(task.task_id)

    def _analyze_frames(
        self, task: ExtractionTask, stream_block_size: int, frame_interval: int, store: SharedNDArrayStoreBase
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        submitted_tasks: list[Future] = []
        analysis_results: list[FrameAnalyzerResult] = []

        # analysis run
        block_index = 0
        with ffmpegio.open(
            str(task.video_path), "rv", blocksize=stream_block_size, pix_fmt=self._analysis_pixel_format
        ) as fin:
            for frames in fin:
                # create shared memory
                shared_memory_ref = store.put(frames, worker_writeable=False)

                # analyze video block
                worker_task = self._analyzer_pool.submit_task(FrameAnalyzerTask(block_index, shared_memory_ref))
                worker_task.add_done_callback(
                    partial(
                        self._on_block_finished,
                        results=analysis_results,
                        task=task,
                        shared_memory_ref=shared_memory_ref,
                        store=store,
                    )
                )
                submitted_tasks.append(worker_task)

                block_index += 1

        # wait for all tasks to be done
        for worker_task in submitted_tasks:
            worker_task.wait()
            worker_task.clear()

        # select best frames per interval
        analysis_results.sort(key=lambda e: e.block_index)
        raw_frame_scores = list(chain.from_iterable(r.scores for r in analysis_results))
        best_frames_per_interval = self._select_best_frames_per_interval(raw_frame_scores, frame_interval)

        return best_frames_per_interval

    def _on_block_finished(
        self,
        future: Future[FrameAnalyzerResult],
        results: list[FrameAnalyzerResult],
        task: ExtractionTask,
        shared_memory_ref: SharedNDArrayRef,
        store: SharedNDArrayStoreBase,
    ):
        # append result to results list
        result = future.result()

        # todo: do we have to be careful here (regarding thread-safety)?
        results.append(result)

        # release memory
        store.release(shared_memory_ref)
        self.on_block_event(BlockAnalyzedEvent(task, result.block_index, result))

    def _extract_frames(
        self,
        task: ExtractionTask,
        stream_block_size: int,
        interval_ids: np.ndarray,
        frame_ids: np.ndarray,
        scores: np.ndarray,
    ):
        # setup output handlers for this task
        for handler in self._output_handlers:
            handler.prepare_task(task)

        global_start = 0  # first global frame index in current chunk

        with ffmpegio.open(
            str(task.video_path),
            "rv",
            blocksize=stream_block_size,
            pix_fmt=self._extraction_pixel_format,
        ) as fin:
            for block_index, frames in enumerate(fin):
                block_len = len(frames)
                if block_len == 0:
                    continue

                block_end = global_start + block_len  # exclusive

                i0 = np.searchsorted(frame_ids, global_start, side="left")
                i1 = np.searchsorted(frame_ids, block_end, side="left")

                if i0 == i1:
                    global_start = block_end
                    continue

                local_idxs = frame_ids[i0:i1] - global_start

                for k, local_idx in zip(range(i0, i1), local_idxs):
                    frame_id = int(frame_ids[k])
                    interval_id = int(interval_ids[k])
                    score = float(scores[k])

                    frame = frames[int(local_idx)]

                    frame_info = VideoFrameInfo(
                        interval_index=interval_id, frame_index=frame_id, score=score, frame=frame
                    )
                    for handler in self._output_handlers:
                        handler.handle_block(task, frame_info)

                    self.on_block_event(BlockFrameExtracted(task=task, frame_info=frame_info))

                global_start = block_end

                if i1 >= frame_ids.size:
                    break

    @staticmethod
    def _calculate_block_size(
        width: int, height: int, channels: int, memory_limit_mb: int, safe_factor: float = 0.8
    ) -> int:
        # RGB24 = 3 bytes per pixel
        frame_size_bytes = width * height * channels
        memory_limit_bytes = memory_limit_mb * 1024 * 1024

        # Allow using up to n% of the limit for the buffer to be safe
        safe_memory_bytes = memory_limit_bytes * safe_factor

        count = int(safe_memory_bytes / frame_size_bytes)
        return max(1, count)

    @staticmethod
    def _select_best_frames_per_interval(
        raw_frame_scores: list[float],
        frame_interval: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if frame_interval <= 0:
            raise ValueError("frame_interval must be > 0")

        scores = np.asarray(raw_frame_scores, dtype=np.float32)
        n = int(scores.size)
        if n == 0:
            return (
                np.empty((0,), dtype=np.int64),  # interval_index
                np.empty((0,), dtype=np.int64),  # frame_index
                np.empty((0,), dtype=np.float32),  # score
            )

        interval_index = np.arange(n, dtype=np.int64) // frame_interval
        num_intervals = int(interval_index[-1]) + 1

        best_frame_index = np.zeros(num_intervals, dtype=np.int64)
        best_score = np.full(num_intervals, -np.inf, dtype=np.float32)

        for frame_index in range(n):
            ii = interval_index[frame_index]
            s = scores[frame_index]
            if s > best_score[ii]:
                best_score[ii] = s
                best_frame_index[ii] = frame_index

        out_interval_index = np.arange(num_intervals, dtype=np.int64)

        order = np.argsort(best_frame_index, kind="stable")
        return out_interval_index[order], best_frame_index[order], best_score[order]

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
