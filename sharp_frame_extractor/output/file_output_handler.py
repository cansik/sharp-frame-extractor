import cv2

from sharp_frame_extractor.analyzer.frame_analyzer_base import FrameAnalyzerResult
from sharp_frame_extractor.models import ExtractionTask
from sharp_frame_extractor.output.frame_output_handler_base import FrameOutputHandlerBase


class FileOutputHandler(FrameOutputHandlerBase):
    def __init__(self):
        pass

    def prepare_task(self, task: ExtractionTask):
        # make the output directory exists
        task.result_path.mkdir(parents=True, exist_ok=True)

    def handle_block(self, task: ExtractionTask, result: FrameAnalyzerResult):
        output_file_name = task.result_path / f"frame-{result.block_index:05d}.png"

        if output_file_name.exists():
            output_file_name.unlink(missing_ok=True)

        cv2.imwrite(str(output_file_name.absolute()), result.frame)
