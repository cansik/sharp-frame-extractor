import math
import multiprocessing
import os
import time
from multiprocessing import Pool

import cv2

from SharpFrameExtractor.SFEWorker import init_worker, extract
from SharpFrameExtractor.estimator.BaseEstimator import BaseEstimator


class SharpFrameExtractor:
    def __init__(self, estimator: BaseEstimator,
                 min_sharpness=-1,
                 crop_factor=0.25,
                 output_format="jpg",
                 cpu_count=multiprocessing.cpu_count()):
        self.estimator = estimator
        self.min_sharpness = min_sharpness
        self.crop_factor = crop_factor
        self.output_format = output_format
        self.cpu_count = cpu_count

    def extract(self, video_file, output_path, window_size_ms, target_frame_count: int = -1):
        start_time = time.time()
        vidcap = cv2.VideoCapture(video_file)
        vidcap.read()

        # prepare paths
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # prepare vars
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_length_ms = frame_count / float(fps) * 1000

        # calculate window if frame_count is set
        if target_frame_count > 0:
            window_size_ms = video_length_ms / target_frame_count
            print("set window size to %.2fms to create %d frames!" % (window_size_ms, target_frame_count))

        step_count = math.floor(video_length_ms / window_size_ms)

        print("Video '%s' with %d FPS and %d frames (%.2fs) resulting in %d stills"
              % (os.path.basename(video_file), fps, frame_count, video_length_ms / 1000, step_count))

        # create windows
        windows = []
        for i in range(0, step_count):
            window_start_ms = i * window_size_ms
            window_end_ms = window_start_ms + window_size_ms

            # check if it is last window
            if i == step_count - 1:
                window_end_ms = video_length_ms

            windows.append((i, window_start_ms, window_end_ms))
        vidcap.release()

        # run multiprocessing
        pool = Pool(processes=self.cpu_count, initializer=init_worker,
                    initargs=((video_file,
                               output_path,
                               self.estimator,
                               self.crop_factor,
                               self.output_format,
                               self.min_sharpness),))
        results = pool.map(extract, windows)
        end_time = time.time()

        frames = [result[0] for result in results if result is not None]

        print("Took %.4f seconds to extract %d frames!" % (end_time - start_time, len(windows)))
        return frames
