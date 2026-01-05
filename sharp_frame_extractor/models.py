from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from threading import Lock
from typing import ClassVar
from typing import Self

from sharp_frame_extractor.analyzer.frame_analyzer_base import FrameAnalyzerResult


@dataclass
class AutoTaskIdMixin:
    _task_id: int = field(init=False, repr=False)

    _next_id: ClassVar[int]
    _id_lock: ClassVar[Lock]

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls._next_id = 1
        cls._id_lock = Lock()

    def __post_init__(self) -> None:
        cls = type(self)
        with cls._id_lock:
            self._task_id = cls._next_id
            cls._next_id += 1

        post_init = getattr(super(), "__post_init__", None)
        if post_init is not None:
            post_init()

    @property
    def task_id(self) -> int:
        return self._task_id


@dataclass
class ExtractionOptions:
    # either one of the two have ot be set
    frame_interval_seconds: float | None = None
    total_frame_count: int | None = None

    @classmethod
    def from_interval(cls, frame_interval_seconds: float) -> Self:
        return ExtractionOptions(frame_interval_seconds=frame_interval_seconds)

    @classmethod
    def from_count(cls, total_frame_count: int) -> Self:
        return ExtractionOptions(total_frame_count=total_frame_count)


@dataclass
class ExtractionTask(AutoTaskIdMixin):
    video_path: Path
    result_path: Path
    options: ExtractionOptions


@dataclass
class ExtractionResult:
    task_id: int


# events models


@dataclass
class TaskEvent(ABC):
    task: ExtractionTask


@dataclass
class TaskStartedEvent(TaskEvent):
    pass


@dataclass
class TaskAnalyzedEvent(TaskEvent):
    total_blocks: int


@dataclass
class TaskFinishedEvent(TaskEvent):
    pass


@dataclass
class BlockEvent(ABC):
    block_id: int
    task: ExtractionTask


@dataclass
class BlockProcessedEvent(BlockEvent):
    result: FrameAnalyzerResult
