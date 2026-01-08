from abc import ABC, abstractmethod

from sharp_frame_extractor.models import ExtractionTask, VideoFrameInfo


class FrameOutputHandlerBase(ABC):
    @abstractmethod
    def open(self):
        pass

    @abstractmethod
    def prepare_task(self, task: ExtractionTask):
        pass

    @abstractmethod
    def handle_block(self, task: ExtractionTask, frame_info: VideoFrameInfo):
        pass

    @abstractmethod
    def close(self):
        pass
