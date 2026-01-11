from typing import Iterator

import numpy as np

from sharp_frame_extractor.reader.video_reader import PixelFormat, VideoInfo, VideoReader


class BatchedVideoReader(VideoReader):
    def __init__(self, video_reader: VideoReader, batch_size: int):
        super().__init__(video_reader._video_path)
        self._video_reader = video_reader
        self._batch_size = batch_size

    def probe(self) -> VideoInfo:
        return self._video_reader.probe()

    def read_frames(self, pixel_format: PixelFormat, copy: bool = True) -> Iterator[np.ndarray]:
        """
        Yields frames in batches of size `batch_size`.

        :param copy: If True, yields a deep copy of the batch. If False, yields a reference to the internal buffer.
                     Warning: When copy=False, the buffer is reused. The data must be processed before the next iteration.
        """
        # Pre-allocate buffer for better performance
        # We need to know frame shape first, so we peek the first frame
        iterator = self._video_reader.read_frames(pixel_format)

        try:
            first_frame = next(iterator)
        except StopIteration:
            return

        # Initialize buffer
        # Shape: (chunk_size, H, W, C) or (chunk_size, H, W)
        frame_shape = first_frame.shape
        dtype = first_frame.dtype

        # Create a buffer that can hold 'chunk_size' frames
        batch_buffer = np.empty((self._batch_size, *frame_shape), dtype=dtype)

        batch_buffer[0] = first_frame
        count = 1

        for frame in iterator:
            batch_buffer[count] = frame
            count += 1

            if count == self._batch_size:
                yield batch_buffer.copy() if copy else batch_buffer
                count = 0

        # Yield remaining frames
        if count > 0:
            yield batch_buffer[:count].copy() if copy else batch_buffer[:count]

    def release(self):
        self._video_reader.release()
