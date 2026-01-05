from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True, slots=True)
class SharedNDArrayRef:
    name: str
    shape: Tuple[int, ...]
    dtype: str
    order: str = "C"
    writeable: bool = False


class SharedNDArray:
    def __init__(self, shm: shared_memory.SharedMemory, ref: SharedNDArrayRef):
        self._shm = shm
        self.ref = ref
        self.ndarray = np.ndarray(
            ref.shape,
            dtype=np.dtype(ref.dtype),
            buffer=shm.buf,
            order=ref.order,
        )
        if not ref.writeable:
            self.ndarray.setflags(write=False)

    @staticmethod
    def _nbytes(shape: Tuple[int, ...], dtype: np.dtype) -> int:
        if len(shape) == 0:
            raise ValueError("shape must not be empty")
        count = int(np.prod(shape))
        if count <= 0:
            raise ValueError(f"invalid shape {shape}")
        return count * int(dtype.itemsize)

    @classmethod
    def create(
        cls,
        shape: Tuple[int, ...],
        dtype: np.dtype | str,
        *,
        order: str = "C",
        writeable: bool = True,
        name: Optional[str] = None,
    ) -> SharedNDArray:
        dtype = np.dtype(dtype)
        nbytes = cls._nbytes(shape, dtype)
        shm = shared_memory.SharedMemory(create=True, size=nbytes, name=name)
        ref = SharedNDArrayRef(
            name=shm.name,
            shape=tuple(shape),
            dtype=dtype.str,
            order=order,
            writeable=writeable,
        )
        return cls(shm, ref)

    @classmethod
    def attach(cls, ref: SharedNDArrayRef) -> SharedNDArray:
        shm = shared_memory.SharedMemory(name=ref.name, create=False)
        return cls(shm, ref)

    def close(self) -> None:
        self._shm.close()

    def unlink(self) -> None:
        self._shm.unlink()

    def __enter__(self) -> SharedNDArray:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class SharedNDArrayStore:
    """
    Owns shared memory blocks in the producer process.
    Call release(ref) when you are done to avoid leaked segments.
    """

    def __init__(self) -> None:
        self._owned: Dict[str, SharedNDArray] = {}

    def put(self, arr: np.ndarray, *, order: str = "C", worker_writeable: bool = False) -> SharedNDArrayRef:
        if order != "C":
            raise ValueError("only C order is implemented in this helper")
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)

        shared = SharedNDArray.create(arr.shape, arr.dtype, order="C", writeable=True)
        shared.ndarray[...] = arr
        self._owned[shared.ref.name] = shared

        return SharedNDArrayRef(
            name=shared.ref.name,
            shape=shared.ref.shape,
            dtype=shared.ref.dtype,
            order=shared.ref.order,
            writeable=worker_writeable,
        )

    def release(self, ref: SharedNDArrayRef) -> None:
        shared = self._owned.pop(ref.name, None)
        if shared is None:
            return

        shared.close()
        try:
            shared.unlink()
        except FileNotFoundError:
            pass

    def release_all(self) -> None:
        names = list(self._owned.keys())
        for name in names:
            shared = self._owned.pop(name, None)
            if shared is None:
                continue
            shared.close()
            try:
                shared.unlink()
            except FileNotFoundError:
                pass
