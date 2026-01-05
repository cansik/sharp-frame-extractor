from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from sharp_frame_extractor.memory.shared_ndarray import SharedNDArrayRef


@dataclass
class FrameAnalyzerTask:
    block_index: int
    frames_ref: SharedNDArrayRef


@dataclass
class FrameAnalyzerResult:
    block_index: int
    frame_index: int
    frame: np.ndarray
    score: float


class FrameAnalyzerBase(ABC):
    @abstractmethod
    def reset_states(self):
        pass

    @abstractmethod
    def process(self, task: FrameAnalyzerTask, frames: np.ndarray) -> FrameAnalyzerResult:
        pass
