from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from sharp_frame_extractor.reader.video_reader import FrameHandle, PixelFormat, VideoInfo, VideoReader


class OpencvFrameHandle(FrameHandle):
    def to_ndarray(self) -> np.ndarray:
        if self._target_pixel_format == PixelFormat.GRAY:
            return cv2.cvtColor(self._native, cv2.COLOR_BGR2GRAY)
        elif self._target_pixel_format == PixelFormat.RGB24:
            return cv2.cvtColor(self._native, cv2.COLOR_BGR2RGB)

        return self._native


class OpencvVideoReader(VideoReader):
    def __init__(self, video_path: str | Path):
        super().__init__(video_path)
        self._cap = cv2.VideoCapture(str(self._video_path))
        if not self._cap.isOpened():
            raise ValueError(f"Could not open video file {self._video_path}")

    def probe(self) -> VideoInfo:
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        return VideoInfo(
            width=width,
            height=height,
            fps=fps,
            duration=duration,
            total_frames=total_frames,
        )

    def read_frames(self, pixel_format: PixelFormat) -> Iterator[OpencvFrameHandle]:
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        try:
            while True:
                ret, frame = self._cap.read()
                if not ret:
                    break

                yield OpencvFrameHandle(frame, pixel_format)
        finally:
            pass

    def release(self):
        if self._cap.isOpened():
            self._cap.release()

    def __del__(self):
        self.release()
