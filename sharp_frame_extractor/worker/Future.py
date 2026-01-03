from __future__ import annotations

import logging
import threading
from typing import Optional, Generic, Callable, List

from .types import TResult

logger = logging.getLogger(__name__)


class Future(Generic[TResult]):
    """
    A very simple Future implementation.
    """

    def __init__(self):
        self._done = threading.Event()
        self._result: Optional[TResult] = None
        self._exception: Optional[Exception] = None
        self._callbacks: List[Callable[[Future[TResult]], None]] = []
        self._lock = threading.Lock()

    def _invoke_callbacks(self):
        for callback in self._callbacks:
            try:
                callback(self)
            except Exception as e:
                logger.error(f"Error in Future callback: {e}")

    def set_result(self, result: TResult):
        with self._lock:
            self._result = result
            self._done.set()
            self._invoke_callbacks()

    def set_exception(self, exception: Exception):
        with self._lock:
            self._exception = exception
            self._done.set()
            self._invoke_callbacks()

    def add_done_callback(self, fn: Callable[[Future[TResult]], None]):
        """
        Attaches a callable that will be executed when the future is finished.
        If the future is already finished, the callback is executed immediately.
        """
        with self._lock:
            if self._done.is_set():
                fn(self)
            else:
                self._callbacks.append(fn)

    def result(self, timeout: Optional[float] = None) -> TResult:
        if self._done.wait(timeout):
            if self._exception:
                raise self._exception
            return self._result  # type: ignore
        else:
            raise TimeoutError("Future result not available within timeout.")

    def done(self) -> bool:
        return self._done.is_set()
