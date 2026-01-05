from abc import ABC, abstractmethod

from sharp_frame_extractor.analyzer.frame_analyzer_base import FrameAnalyzerResult
from sharp_frame_extractor.models import ExtractionTask


class FrameOutputHandlerBase(ABC):
    @abstractmethod
    def prepare_task(self, task: ExtractionTask):
        pass

    @abstractmethod
    def handle_block(self, task: ExtractionTask, result: FrameAnalyzerResult):
        pass
