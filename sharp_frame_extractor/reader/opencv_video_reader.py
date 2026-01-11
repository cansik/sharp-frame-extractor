from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from sharp_frame_extractor.reader.video_reader import PixelFormat, VideoInfo, VideoReader


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

    def read_frames(self, pixel_format: PixelFormat) -> Iterator[np.ndarray]:
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        try:
            while True:
                ret, frame = self._cap.read()
                if not ret:
                    break

                if pixel_format == PixelFormat.GRAY:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                elif pixel_format == PixelFormat.RGB24:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                yield frame
        finally:
            pass

    def release(self):
        if self._cap.isOpened():
            self._cap.release()

    def __del__(self):
        self.release()
