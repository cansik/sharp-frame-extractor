import math
import os
import time

import cv2

from SharpFrameExtractor.estimator.BaseEstimator import BaseEstimator
from SharpFrameExtractor.utils.ExponentialMovingAverage import ExponentialMovingAverage


class SharpFrameExtractor:
    def __init__(self, estimator: BaseEstimator, min_sharpness=-1, crop_factor=0.25, output_format="jpg"):
        self.estimator = estimator
        self.min_sharpness = min_sharpness
        self.crop_factor = crop_factor
        self.output_format = output_format

    def extract_images(self, video_file, output_path, window_size_ms, target_frame_count: int = -1):
        count = 0
        ema = ExponentialMovingAverage(0.2)
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

        files = []

        print("Video '%s' with %d FPS and %d frames (%.2fs) resulting in %d stills"
              % (os.path.basename(video_file), fps, frame_count, video_length_ms / 1000, step_count))

        for i in range(0, step_count):
            start_time = time.time()
            window_start_ms = i * window_size_ms
            window_end_ms = window_start_ms + window_size_ms

            # check if it is last window
            if i == step_count - 1:
                window_end_ms = video_length_ms

            print("analyzing batch %d/%d (%.2fs to %.2fs)..."
                  % (i + 1, step_count, window_start_ms / 1000, window_end_ms / 1000))

            # extracting frames and getting the one with the best metric
            frames = self._extract_frame_batch(vidcap, window_start_ms, window_size_ms)

            if len(frames) == 0:
                print("ERROR: No frames extracted (maybe a video error!)")
                continue

            frames = sorted(frames, key=lambda e: e[1], reverse=True)
            index, sharpness = frames[0]

            prefix = ""
            if sharpness < self.min_sharpness:
                print("WARNING: Sharpness not high enough (%.2fs)" % sharpness)
                prefix = "WARNING_"

            # extract and store best frame
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, index)
            success, image = vidcap.read()

            frame_path = os.path.join(output_path, "%sframe%04d.%s" % (prefix, count, self.output_format))
            cv2.imwrite(frame_path, image)
            files.append(frame_path)

            # time measurements
            end_time = time.time()
            duration = end_time - start_time
            ema.add(duration)

            steps_left = step_count - i
            time_left = ema.value * steps_left

            if i != 0 and i % 5 == 0:
                total_seconds = round(time_left)
                minutes = total_seconds // 60
                seconds = total_seconds - (minutes * 60)

                time_text = "Time left: "

                if minutes > 0:
                    time_text += "%dm " % minutes
                time_text += "%ds " % seconds
                time_text += " Avg: %.2fs" % ema.value

                print(time_text)
            count += 1

        return files

    def _extract_frame_batch(self, vidcap, start_ms, window_ms):
        results = []

        # jump to video start
        vidcap.set(cv2.CAP_PROP_POS_MSEC, start_ms)
        vidcap.grab()

        end_ms = start_ms + window_ms

        frame_available = True
        while frame_available:
            frame_index = vidcap.get(cv2.CAP_PROP_POS_FRAMES)
            time_code = vidcap.get(cv2.CAP_PROP_POS_MSEC)

            # check if end of window
            if time_code >= end_ms:
                frame_available = False
                continue

            # read next frame
            success, image = vidcap.read()

            # check end of stream
            if not success:
                frame_available = False
                continue

            # crop roi if necessary
            if self.crop_factor != 1.0:
                height, width, channels = image.shape
                cw = round(width * self.crop_factor)
                ch = round(height * self.crop_factor)

                x = round((width * 0.5) - (cw * 0.5))
                y = round((height * 0.5) - (ch * 0.5))

                image = image[y:y + ch, x:x + cw]

            # extract metrics
            sharpness = self.estimator.estimate(image)
            results.append((frame_index, sharpness))

            frame_available = True

        return results
