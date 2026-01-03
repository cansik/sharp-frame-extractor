import logging
from abc import ABC
from multiprocessing import Queue
from typing import TypeVar, Generic, Callable, List

from .BaseWorker import BaseWorker

logger = logging.getLogger(__name__)

TWorker = TypeVar("TWorker", bound=BaseWorker)


class BaseWorkerPool(Generic[TWorker], ABC):
    """
    A generic worker pool that manages multiple workers.
    The pool is initialized with a worker_class and any additional keyword arguments to pass to each worker.
    """

    def __init__(self, num_workers: int, worker_class: Callable[..., TWorker], **worker_kwargs):
        self.num_workers = num_workers
        self.worker_class = worker_class
        self.worker_kwargs = worker_kwargs
        self.workers: List[TWorker] = []
        self.active_worker_ids: Queue[int] = Queue()

    def start(self):
        logger.debug(f"Pool: Starting {self.num_workers} workers.")
        for i in range(self.num_workers):
            worker = self.worker_class(i, **self.worker_kwargs)
            worker.start()
            self.workers.append(worker)
            self.active_worker_ids.put(i)
        logger.debug("Pool: All workers started.")

    def acquire(self) -> TWorker:
        """
        Acquire a worker from the pool (blocking until one is available).
        """
        logger.debug("Pool: Acquiring a worker.")
        worker_id = self.active_worker_ids.get()
        worker = self.workers[worker_id]
        logger.debug(f"Pool: Worker {worker.worker_id} acquired.")
        return worker

    def release(self, worker: TWorker):
        """
        Release a worker back to the pool.
        """
        logger.debug(f"Pool: Releasing worker {worker.worker_id}.")
        self.active_worker_ids.put(worker.worker_id)

    def stop(self):
        """
        Stop all workers in the pool.
        """
        logger.debug("Pool: Stopping all workers.")
        for worker in self.workers:
            worker.stop()
        logger.debug("Pool: All workers stopped.")
