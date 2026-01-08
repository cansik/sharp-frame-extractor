import threading
from itertools import chain
from multiprocessing.shared_memory import SharedMemory

import numpy as np

from sharp_frame_extractor.memory.shared_ndarray import SharedNDArrayRef, SharedNDArrayStoreBase


class PooledSharedNDArrayStore(SharedNDArrayStoreBase):
    """
    A shared memory store that reuses memory segments.

    It is initialized with a fixed item size and a buffer limit.
    """

    def __init__(self, item_size: int, n_buffers: int):
        self._item_size = item_size
        self._n_buffers = n_buffers
        self._pool: list[SharedMemory] = []
        self._active: dict[str, SharedMemory] = {}
        self._lock = threading.Lock()
        # Semaphore limits the number of active shared memory segments
        # This provides backpressure if the consumers (workers) are slower than the producer
        self._semaphore = threading.Semaphore(n_buffers)

    def put(self, arr: np.ndarray, *, order: str = "C", worker_writeable: bool = False) -> SharedNDArrayRef:
        if order != "C":
            raise ValueError("only C order is implemented in this helper")

        # Ensure we don't exceed the buffer size
        if arr.nbytes > self._item_size:
            raise ValueError(f"Array size {arr.nbytes} exceeds configured pool item size {self._item_size}")

        # Block until a buffer is available (backpressure)
        self._semaphore.acquire()

        shm = None
        with self._lock:
            # Try to find a free buffer in the pool
            if self._pool:
                shm = self._pool.pop()

            # If no buffer in pool (but semaphore acquired), create a new one
            if shm is None:
                shm = SharedMemory(create=True, size=self._item_size)

            self._active[shm.name] = shm

        # Copy data into shared memory
        shm_array = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        shm_array[:] = arr[:]

        return SharedNDArrayRef(shm.name, arr.shape, arr.dtype.str, order=order, writeable=worker_writeable)

    def release(self, ref: SharedNDArrayRef) -> None:
        with self._lock:
            if ref.name in self._active:
                shm = self._active.pop(ref.name)
                self._pool.append(shm)

                # Signal that a buffer is free
                self._semaphore.release()

    def release_all(self) -> None:
        with self._lock:
            # Combine active and pool, close and unlink everything
            for shm in chain(self._active.values(), self._pool):
                try:
                    shm.close()
                    shm.unlink()
                except Exception:
                    pass
            self._active.clear()
            self._pool.clear()
