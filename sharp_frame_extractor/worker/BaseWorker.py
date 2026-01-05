import logging
import threading
from abc import ABC, abstractmethod
from multiprocessing import Event, Process, Queue, current_process
from typing import Dict, Generic

from .Future import Future
from .types import TResult, TTask

logger = logging.getLogger(__name__)


class BaseWorker(Process, Generic[TTask, TResult], ABC):
    """
    A generic worker process that uses a task queue to receive work and a results queue
    to return results. Task submission returns a Future, which is resolved when the
    worker places the result on the results queue.

    Subclasses must implement the handle_task() method.
    """

    def __init__(self, worker_id: int, task_queue_size: int = 0):
        super().__init__(target=self._run_loop)
        self.worker_id = worker_id
        self.stop_requested = Event()
        # The tasks queue will carry tuples: (task_id, TTask)
        self.tasks: Queue = Queue(maxsize=task_queue_size)
        # The results queue will carry tuples: (task_id, TResult or Exception)
        self.results: Queue = Queue(maxsize=task_queue_size)

        # Only in the main process do we maintain a futures dictionary and background thread.
        if current_process().name == "MainProcess":
            self._futures: Dict[int, Future[TResult]] = {}
            self._task_counter = 0
            self._result_listener_thread = threading.Thread(target=self._result_listener, daemon=True)
            self._result_listener_thread.start()

    def _run_loop(self):
        """
        The worker process loop. It expects to receive (task_id, task) tuples.
        """
        self.setup()
        while not self.stop_requested.is_set():
            item = self.tasks.get()
            if item is None:
                logger.debug(f"Worker {self.worker_id}: Stop signal received.")
                self.stop_requested.set()
                continue

            task_id, task = item
            try:
                result = self.handle_task(task)
                self.results.put((task_id, result))
            except Exception as e:
                logger.error(f"Worker {self.worker_id}: Error handling task: {e}")
                self.results.put((task_id, e))
        self.cleanup()

    @abstractmethod
    def setup(self):
        """
        Set up the worker and it's resources.
        """
        pass

    @abstractmethod
    def handle_task(self, task: TTask) -> TResult:
        """
        Process a single task. Subclasses must override this method.
        """
        pass

    def _result_listener(self):
        """
        A background thread (running in the main process) that listens for results from the worker process
        and resolves the corresponding Future.
        """
        while True:
            item = self.results.get()
            if item is None:  # Sentinel value signals shutdown
                break

            task_id, result = item
            future = self._futures.pop(task_id, None)
            if future:
                if isinstance(result, Exception):
                    future.set_exception(result)
                else:
                    future.set_result(result)

    def submit_task(self, task: TTask) -> Future[TResult]:
        """
        Submits a task for processing and returns a Future that can later be used to obtain the result.
        """
        if not hasattr(self, "_futures"):
            raise RuntimeError("submit_task should only be called from the main process.")
        future = Future[TResult]()
        task_id = self._task_counter
        self._task_counter += 1
        self._futures[task_id] = future
        self.tasks.put((task_id, task))
        return future

    def cleanup(self):
        """
        Optional cleanup code once the worker stops.
        """
        pass

    def stop(self):
        """
        Signals the worker to stop processing tasks.
        """
        self.tasks.put(None)

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove non-pickleable attributes
        state.pop("_result_listener_thread", None)
        state.pop("_futures", None)
        return state
