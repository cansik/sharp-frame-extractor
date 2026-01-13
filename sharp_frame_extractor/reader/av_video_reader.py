import math
from pathlib import Path
from typing import Iterator

import av
import numpy as np

from sharp_frame_extractor.reader.video_reader import FrameHandle, PixelFormat, VideoInfo, VideoReader


class AvFrameHandle(FrameHandle):
    def to_ndarray(self) -> np.ndarray:
        return self._native.to_ndarray(format=self._target_pixel_format.value)


class AvVideoReader(VideoReader):
    def __init__(self, video_path: str | Path):
        super().__init__(video_path)
        self._container = av.open(str(self._video_path))
        self._stream = self._container.streams.video[0]
        self._stream.thread_type = "AUTO"

    def probe(self) -> VideoInfo:
        width = self._stream.width
        height = self._stream.height
        fps = float(self._stream.average_rate)
        total_frames = self._stream.frames
        duration = float(self._stream.duration * self._stream.time_base) if self._stream.duration else 0

        # Fallback if total_frames is not available in stream metadata
        if total_frames == 0 and fps > 0 and duration > 0:
            total_frames = int(math.ceil(duration * fps))

        return VideoInfo(
            width=width,
            height=height,
            fps=fps,
            duration=duration,
            total_frames=total_frames,
        )

    def read_frames(self, pixel_format: PixelFormat) -> Iterator[AvFrameHandle]:
        # Seek to start
        self._container.seek(0)

        for frame in self._container.decode(self._stream):
            yield AvFrameHandle(frame, pixel_format)

    def release(self):
        if self._container:
            self._container.close()

    def __del__(self):
        self.release()
