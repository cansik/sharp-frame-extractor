from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class FrameAnalyzerTask:
    block_index: int
    frames: np.ndarray


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
    def process(self, task: FrameAnalyzerTask) -> FrameAnalyzerResult:
        pass
