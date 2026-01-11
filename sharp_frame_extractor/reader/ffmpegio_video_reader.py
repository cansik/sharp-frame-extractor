import math
import platform
from pathlib import Path
from typing import Iterator

import ffmpegio
import numpy as np

from sharp_frame_extractor.memory.rolling_buffer import RollingBuffer
from sharp_frame_extractor.reader.video_reader import PixelFormat, VideoInfo, VideoReader


class FfmpegIoVideoReader(VideoReader):
    FFMPEG_MIN_VERSION = 6
    UNIX_OPTIMAL_CHUNK_SIZE = 32

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

    def read_frames(self, chunk_size: int, pixel_format: PixelFormat) -> Iterator[np.ndarray]:
        # Windows optimization: read 1 frame at a time to improve performance
        # Unix/Mac: read larger blocks for less IPC overhead
        system = platform.system()
        internal_block_size = 1 if system == "Windows" else min(self.UNIX_OPTIMAL_CHUNK_SIZE, chunk_size)

        rolling_buffer: RollingBuffer | None = None

        with ffmpegio.open(
            str(self._video_path), "rv", blocksize=internal_block_size, pix_fmt=pixel_format.value
        ) as fin:
            for frames in fin:
                n_frames = len(frames)
                if n_frames == 0:
                    continue

                # Initialize buffer on first valid read to ensure correct shape/dtype
                if rolling_buffer is None:
                    rolling_buffer = RollingBuffer(chunk_size, shape=frames.shape[1:], dtype=frames.dtype)

                # Feed frames into the rolling buffer
                # We loop to handle cases where input block > chunk_size (unlikely on Windows, possible on Unix)
                idx = 0
                while idx < n_frames:
                    # If buffer is full, yield and clear
                    if rolling_buffer.is_full:
                        yield rolling_buffer.get(copy=False)
                        rolling_buffer.clear()

                    # Append what fits
                    to_add = min(rolling_buffer.free, n_frames - idx)
                    rolling_buffer.append(frames[idx : idx + to_add])
                    idx += to_add

            # Yield any remaining frames
            if rolling_buffer is not None and not rolling_buffer.is_empty:
                yield rolling_buffer.get(copy=False)

    @staticmethod
    def _ffmpeg_version_compatibility_check():
        ffmpeg_info = ffmpegio.ffmpeg_info()
        ffmpeg_version = ffmpeg_info["version"]
        major = int(ffmpeg_version.split(".")[0])
        if major < FfmpegIoVideoReader.FFMPEG_MIN_VERSION:
            raise Exception(
                f"ffmpeg version {ffmpeg_version} is maybe too old, successfully tested version is >={FfmpegIoVideoReader.FFMPEG_MIN_VERSION}."
            )
