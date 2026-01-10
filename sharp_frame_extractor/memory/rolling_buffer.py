import numpy as np
from typing import Optional, Tuple


class IndexingView:
    """
    Helper class to convert between absolute and relative indices,
    and to perform slicing in terms of absolute sample positions.
    """

    def __init__(self, owner: "RollingBuffer") -> None:
        self._owner = owner

    @property
    def absolute_start_index(self) -> int:
        """
        Get the absolute index of the oldest retained sample (inclusive).

        :returns: Absolute index of the oldest sample.
        """
        return self._owner.total_samples_written - len(self._owner)

    @property
    def absolute_end_index(self) -> int:
        """
        Get the absolute index just past the most recent sample (exclusive).

        :returns: Absolute end index.
        """
        return self._owner.total_samples_written

    def absolute_to_relative_index(self, absolute_index: int) -> int:
        """
        Convert an absolute index to a relative index.

        :param absolute_index: Index to convert.

        :returns: Relative index for the current buffer state.

        :raises IndexError: If the index is not currently retained.
        """
        start = self.absolute_start_index
        end = self.absolute_end_index
        if not (start <= absolute_index < end):
            raise IndexError("absolute index is not in the current buffer")
        return absolute_index - start

    def relative_to_absolute_index(self, relative_index: int) -> int:
        """
        Convert a relative index to an absolute index.

        :param relative_index: Relative index, supports negative values.

        :returns: Corresponding absolute index.

        :raises IndexError: If the index is out of bounds.
        """
        total = len(self._owner)
        idx = relative_index
        if idx < 0:
            idx += total
        if not (0 <= idx < total):
            raise IndexError("relative index out of range")
        return self.absolute_start_index + idx

    def slice_by_absolute_indices(self, start: Optional[int], stop: Optional[int], copy: bool = True) -> np.ndarray:
        """
        Slice the buffer using absolute indices.

        :param start: Start index (inclusive).
        :param stop: Stop index (exclusive).
        :param copy: Whether to return a copy.

        :returns: Sliced array of samples.
        """
        a0 = self.absolute_start_index
        a1 = self.absolute_end_index
        s = a0 if start is None else start
        e = a1 if stop is None else stop
        s = max(a0, s)
        e = min(a1, e)
        if e <= s:
            empty = self._owner.buffer[:0]
            return empty if not copy else empty.copy()
        return self._owner.slice(s - a0, e - a0, copy=copy)


class RollingBuffer:
    """
    A fixed-size circular buffer for efficiently storing and accessing recent samples.
    Overwrites the oldest data when capacity is exceeded.
    """

    def __init__(self, capacity: int, shape: Tuple[int, ...] = (), dtype: np.dtype = np.float32) -> None:
        """
        Initialize the rolling buffer with a fixed capacity and data type.

        :param capacity: Number of samples to retain in the buffer.
        :param shape: Shape of a single sample (excluding the time dimension).
        :param dtype: Numpy data type to use for buffer storage.
        """
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.buffer = np.zeros((self.capacity,) + shape, dtype=dtype)
        self.write_position = 0
        self.is_buffer_full = False
        self.total_samples_written = 0

        self._indexing_view = IndexingView(self)

    def append(self, chunk: np.ndarray) -> None:
        """
        Append new samples to the buffer, overwriting the oldest ones if full.

        :param chunk: Array of samples to append. If the size exceeds capacity,
                      only the most recent `capacity` samples are retained.
        """
        if chunk is None:
            return
        array = np.asarray(chunk, dtype=self.buffer.dtype)
        n = array.shape[0]
        if n == 0:
            return

        if n >= self.capacity:
            self.buffer[:] = array[-self.capacity :]
            self.write_position = 0
            self.is_buffer_full = True
            self.total_samples_written += n
            return

        end = self.write_position + n
        if end <= self.capacity:
            self.buffer[self.write_position : end] = array
        else:
            tail = self.capacity - self.write_position
            self.buffer[self.write_position :] = array[:tail]
            self.buffer[: end - self.capacity] = array[tail:]

        self.write_position = end % self.capacity
        if not self.is_buffer_full and end >= self.capacity:
            self.is_buffer_full = True
        self.total_samples_written += n

    def clear(self) -> None:
        """
        Clear the buffer, resetting counters and write position.
        The underlying memory is not zeroed.
        """
        self.write_position = 0
        self.is_buffer_full = False
        self.total_samples_written = 0

    def slice(self, start: Optional[int], stop: Optional[int], copy: bool = True) -> np.ndarray:
        """
        Return a slice of samples in chronological order from [start:stop).

        :param start: Starting index relative to the oldest sample.
        :param stop: Stopping index (exclusive) relative to the oldest sample.
        :param copy: If False and the slice does not wrap, returns a view.
                     Otherwise returns a contiguous copy.

        :returns: Array slice of the buffer.
        """
        total = len(self)
        if total == 0:
            return self.buffer[:0] if not copy else self.buffer[:0].copy()

        start_index = 0 if start is None else start
        stop_index = total if stop is None else stop
        if start_index < 0:
            start_index += total
        if stop_index < 0:
            stop_index += total

        start_index = max(0, min(start_index, total))
        stop_index = max(0, min(stop_index, total))
        if stop_index <= start_index:
            return self.buffer[:0] if not copy else self.buffer[:0].copy()

        count = stop_index - start_index

        if not self.is_buffer_full:
            out = self.buffer[start_index:stop_index]
            return out if not copy else out.copy()

        physical_start = (self.write_position + start_index) % self.capacity
        head = self.capacity - physical_start
        if count <= head:
            out = self.buffer[physical_start : physical_start + count]
            return out if not copy else out.copy()

        first = self.buffer[physical_start:]
        second = self.buffer[: count - first.size]
        if copy:
            return np.concatenate((first, second))
        return np.concatenate((first, second))

    def get_last_samples(self, n: Optional[int] = None, copy: bool = True) -> np.ndarray:
        """
        Return the last `n` samples in chronological order.

        :param n: Number of most recent samples to return. If None, return all.
        :param copy: Whether to return a copy or view (if possible).

        :returns: Numpy array of the last `n` samples.
        """
        if n is None:
            return self.slice(0, None, copy=copy)
        if n < 0:
            raise ValueError("n must be nonnegative or None")
        total = len(self)
        n = min(n, total)
        return self.slice(total - n, total, copy=copy)

    def get(self, n: Optional[int] = None, copy: bool = True) -> np.ndarray:
        """
        Deprecated alias for get_last_samples.

        :param n: Number of samples to retrieve.
        :param copy: Whether to return a copy.

        :returns: Numpy array of samples.
        """
        return self.get_last_samples(n=n, copy=copy)

    def view(self, n: Optional[int] = None) -> Tuple[np.ndarray, ...]:
        """
        Return up to two memory views covering the last `n` samples.

        :param n: Number of samples to retrieve. If None, return all.
        :returns: One or two numpy views in chronological order.
        """
        total = len(self)
        if total == 0:
            return (self.buffer[:0],)

        if n is None or n >= total:
            start_index = 0
            count = total
        else:
            if n < 0:
                raise ValueError("n must be nonnegative or None")
            start_index = total - n
            count = n

        if not self.is_buffer_full:
            return (self.buffer[start_index : self.write_position],)

        physical_start = (self.write_position + start_index) % self.capacity
        head = self.capacity - physical_start
        if count <= head:
            return (self.buffer[physical_start : physical_start + count],)
        first = self.buffer[physical_start:]
        second = self.buffer[: count - first.size]
        return first, second

    @property
    def indices(self) -> IndexingView:
        """
        Provides access to index conversion and absolute indexing helpers.

        :returns: An IndexingView instance.
        """
        return self._indexing_view

    @property
    def size(self) -> int:
        """
        Number of valid samples currently stored in the buffer.

        :returns: Current size of the buffer.
        """
        return len(self)

    def __len__(self) -> int:
        return self.capacity if self.is_buffer_full else self.write_position

    @property
    def is_full(self) -> bool:
        """
        Whether the buffer is full.

        :returns: True if full, False otherwise.
        """
        return self.is_buffer_full

    @property
    def is_empty(self) -> bool:
        """
        Whether the buffer is empty.

        :returns: True if empty, False otherwise.
        """
        return len(self) == 0

    @property
    def free(self) -> int:
        """
        Number of available slots before the buffer is full.

        :returns: Number of free slots.
        """
        return self.capacity - len(self)

    def __getitem__(self, key):
        """
        Access buffer elements by index or slice.

        :param key: Integer or slice. Supports negative indexing.

        :returns: Sample or array of samples.

        :raises IndexError: If the index is out of bounds.
        :raises TypeError: If the key type is unsupported.
        """
        if isinstance(key, slice):
            if key.step not in (None, 1):
                return self.get_last_samples(copy=True)[key]
            return self.slice(key.start, key.stop, copy=True)
        if isinstance(key, (int, np.integer)):
            total = len(self)
            idx = int(key)
            if idx < 0:
                idx += total
            if not (0 <= idx < total):
                raise IndexError("index out of range")
            if not self.is_buffer_full:
                return self.buffer[idx]
            physical = (self.write_position + idx) % self.capacity
            return self.buffer[physical]
        raise TypeError("invalid index type")

    def __repr__(self) -> str:
        return (
            f"RollingBuffer(capacity={self.capacity}, size={len(self)}, "
            f"write_position={self.write_position}, full={self.is_buffer_full}, "
            f"written={self.total_samples_written}, dtype={self.buffer.dtype})"
        )
