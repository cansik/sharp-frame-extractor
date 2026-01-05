import logging

from sharp_frame_extractor.analyzer.frame_analyzer_base import FrameAnalyzerBase, FrameAnalyzerResult, FrameAnalyzerTask
from sharp_frame_extractor.analyzer.tenegrad_frame_analyzer import TenengradFrameAnalyzer
from sharp_frame_extractor.worker.BaseWorker import BaseWorker
from sharp_frame_extractor.worker.BaseWorkerPool import BaseWorkerPool
from sharp_frame_extractor.worker.Future import Future

logger = logging.getLogger(__name__)


class FrameAnalyzerWorker(BaseWorker[FrameAnalyzerTask, FrameAnalyzerResult]):
    def __init__(self, worker_id: int, task_queue_size: int = 0, analyzer: FrameAnalyzerBase | None = None):
        super().__init__(worker_id, task_queue_size)
        self.analyzer = analyzer

    def setup(self):
        self.analyzer = TenengradFrameAnalyzer()

    def handle_task(self, task: FrameAnalyzerTask) -> FrameAnalyzerResult:
        self.analyzer.reset_states()
        return self.analyzer.process(task)

    def cleanup(self):
        self.analyzer = None


class FrameAnalyzerWorkerPool(BaseWorkerPool[FrameAnalyzerWorker]):
    def __init__(self, num_workers: int):
        super().__init__(num_workers, worker_class=FrameAnalyzerWorker)

    def submit_task(self, task: FrameAnalyzerTask) -> Future[FrameAnalyzerResult]:
        """
        Acquires a free worker, submits a task, and automatically releases
        the worker once the task is finished.
        """
        worker = self.acquire()
        future = worker.submit_task(task)

        # Ensure the worker is released back to the pool when the future is done
        future.add_done_callback(lambda _: self.release(worker))

        return future
