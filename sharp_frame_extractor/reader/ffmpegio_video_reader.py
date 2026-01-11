import math
import re
from pathlib import Path
from typing import Iterator

import ffmpegio
import numpy as np

from sharp_frame_extractor.reader.video_reader import PixelFormat, VideoInfo, VideoReader


class FfmpegIoVideoReader(VideoReader):
    FFMPEG_MIN_VERSION = 6

    def __init__(self, video_path: str | Path):
        super().__init__(video_path)
        self._ffmpeg_version_compatibility_check()

    def probe(self) -> VideoInfo:
        video_streams = ffmpegio.probe.video_streams_basic(str(self._video_path))
        if not video_streams:
            raise ValueError(f"No video streams found in {self._video_path}")

        info = video_streams[0]
        duration = float(info.get("duration", 0))
        fps = float(info.get("frame_rate", 0))

        width = int(info["width"])
        height = int(info["height"])

        nb_frames = info.get("nb_frames")
        if nb_frames:
            total_frames = int(nb_frames)
        else:
            total_frames = math.ceil(duration * fps) if fps > 0 else 0

        return VideoInfo(
            width=width,
            height=height,
            fps=fps,
            duration=duration,
            total_frames=total_frames,
        )

    def read_frames(self, pixel_format: PixelFormat) -> Iterator[np.ndarray]:
        with ffmpegio.open(str(self._video_path), "rv", pix_fmt=pixel_format.value) as fin:
            for frames in fin:
                for frame in frames:
                    yield frame

    def release(self):
        pass

    @staticmethod
    def _ffmpeg_version_compatibility_check():
        try:
            ffmpeg_info = ffmpegio.ffmpeg_info()
        except Exception as e:
            raise RuntimeError(
                "Could not detect FFmpeg installation. Please ensure FFmpeg is installed and accessible in your PATH."
            ) from e

        ffmpeg_version = ffmpeg_info.get("version", "")

        # Extract major version using regex to handle various version string formats
        # Matches the first sequence of digits which is usually the major version
        match = re.search(r"(\d+)", ffmpeg_version)

        if match:
            major = int(match.group(1))
            if major < FfmpegIoVideoReader.FFMPEG_MIN_VERSION:
                raise RuntimeError(
                    f"Detected FFmpeg version '{ffmpeg_version}' is too old. "
                    f"SharpFrameExtractor requires FFmpeg major version >= {FfmpegIoVideoReader.FFMPEG_MIN_VERSION}. "
                    "Please upgrade your FFmpeg installation."
                )
