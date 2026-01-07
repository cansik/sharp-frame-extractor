import cv2

from sharp_frame_extractor.models import ExtractionTask, VideoFrameInfo
from sharp_frame_extractor.output.frame_output_handler_base import FrameOutputHandlerBase


class FileOutputHandler(FrameOutputHandlerBase):
    def __init__(self):
        pass

    def prepare_task(self, task: ExtractionTask):
        # make the output directory exists
        task.result_path.mkdir(parents=True, exist_ok=True)

    def handle_block(self, task: ExtractionTask, frame_info: VideoFrameInfo):
        output_file_name = task.result_path / f"frame-{frame_info.interval_index:05d}.png"

        if output_file_name.exists():
            output_file_name.unlink(missing_ok=True)

        # convert frame to bgr
        bgr_frame = cv2.cvtColor(frame_info.frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_file_name.absolute()), bgr_frame)
