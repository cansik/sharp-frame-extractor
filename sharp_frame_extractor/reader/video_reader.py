from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Generic, Iterator, Self, Type, TypeVar

import numpy as np


class PixelFormat(Enum):
    GRAY = "gray"
    RGB24 = "rgb24"
    BGR24 = "bgr24"

    @property
    def channels(self) -> int:
        if self == PixelFormat.GRAY:
            return 1
        if self == PixelFormat.RGB24:
            return 3
        if self == PixelFormat.BGR24:
            return 3
        return 3


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    duration: float
    total_frames: int


TNativeFrame = TypeVar("TNativeFrame")


class FrameHandle(ABC, Generic[TNativeFrame]):
    def __init__(self, native: TNativeFrame, target_pixel_format: PixelFormat):
        self._native = native
        self._target_pixel_format = target_pixel_format

    @abstractmethod
    def to_ndarray(self) -> np.ndarray:
        raise NotImplementedError


class VideoReader(ABC):
    def __init__(self, video_path: str | Path):
        self._video_path: Path = Path(video_path)

    @abstractmethod
    def probe(self) -> VideoInfo:
        pass

    @abstractmethod
    def read_frames(self, pixel_format: PixelFormat) -> Iterator[FrameHandle]:
        """
        Yields frames one by one.
        """
        pass

    @abstractmethod
    def release(self):
        pass

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


VideoReaderFactory = Callable[[str | Path], VideoReader] | Type[VideoReader]
