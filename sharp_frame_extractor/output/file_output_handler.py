import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from sharp_frame_extractor.models import ExtractionTask, VideoFrameInfo
from sharp_frame_extractor.output.frame_output_handler_base import FrameOutputHandlerBase


class FileOutputHandler(FrameOutputHandlerBase):
    def __init__(self, max_workers: int = 4, max_queue_size: int = 32, output_format: str = "png"):
        self._max_workers = max_workers
        self._writer_pool: ThreadPoolExecutor | None = None

        self._output_format = output_format
        self._output_extension = f".{self._output_format}"

        # Semaphore to prevent unbounded memory usage if writing is slower than extraction
        self._queue_semaphore = threading.Semaphore(max_queue_size)

    def open(self):
        self._writer_pool = ThreadPoolExecutor(max_workers=self._max_workers, thread_name_prefix="writer")

    def prepare_task(self, task: ExtractionTask):
        # make the output directory exists
        task.result_path.mkdir(parents=True, exist_ok=True)

    def handle_block(self, task: ExtractionTask, frame_info: VideoFrameInfo):
        output_file_name = task.result_path / f"frame-{frame_info.interval_index:05d}{self._output_extension}"

        if output_file_name.exists():
            output_file_name.unlink(missing_ok=True)

        # Create a copy of the frame to detach it from the larger memory block
        # This ensures the large buffer from ffmpegio can be GC'd even if writing is pending
        frame_copy = frame_info.frame.copy()

        # Block if queue is full (backpressure)
        self._queue_semaphore.acquire()

        future = self._writer_pool.submit(self._write_output, output_file_name, frame_copy)
        future.add_done_callback(self._on_task_done)

    def _on_task_done(self, future: Future):
        self._queue_semaphore.release()
        try:
            future.result()
        except Exception as e:
            print(f"Error writing frame: {e}")

    def _write_output(self, output_file_name: Path, frame: np.ndarray):
        # using imencode prevents issues with non-ascii paths on windows
        ok, buf = cv2.imencode(self._output_extension, frame)
        if not ok:
            raise RuntimeError("cv2.imencode failed")

        output_file_name.write_bytes(buf.tobytes())

    def close(self):
        self._writer_pool.shutdown(wait=True)
