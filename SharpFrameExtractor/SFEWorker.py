import os
import cv2

from SharpFrameExtractor.estimator.BaseEstimator import BaseEstimator

vidcap: cv2.VideoCapture = None
estimator: BaseEstimator = None
crop_factor: float = None
output_path: str = None
output_format: str = None
min_sharpness: float = None


def init_worker(params):
    global vidcap, estimator, crop_factor, output_path, output_format, min_sharpness
    video_file, output_path, estimator, crop_factor, output_format, min_sharpness = params
    vidcap = cv2.VideoCapture(video_file)
    estimator.setup()


def extract(window):
    i, window_start_ms, window_end_ms = window
    window_size_ms = window_end_ms - window_start_ms

    print("analyzing batch (%.2fs to %.2fs)..."
          % (window_start_ms / 1000, window_end_ms / 1000))

    # extracting frames and getting the one with the best metric
    frames = _analyze_frame_batch(vidcap, window_start_ms, window_size_ms)

    if len(frames) == 0:
        print("ERROR: No frames extracted (maybe a video error!)")
        return None

    frames = sorted(frames, key=lambda e: e[1], reverse=True)
    index, sharpness = frames[0]

    prefix = ""
    if sharpness < min_sharpness:
        print("WARNING: Sharpness not high enough (%.2fs)" % sharpness)
        prefix = "WARNING_"

    # extract and store best frame
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, index)
    success, image = vidcap.read()

    frame_path = os.path.join(output_path, "%sframe%04d.%s" % (prefix, i, output_format))
    cv2.imwrite(frame_path, image)

    return frame_path, sharpness


def _analyze_frame_batch(self, start_ms, window_ms):
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
        if crop_factor != 1.0:
            height, width, channels = image.shape
            cw = round(width * crop_factor)
            ch = round(height * crop_factor)

            x = round((width * 0.5) - (cw * 0.5))
            y = round((height * 0.5) - (ch * 0.5))

            image = image[y:y + ch, x:x + cw]

        # extract metrics
        sharpness = estimator.estimate(image)
        results.append((frame_index, sharpness))

        frame_available = True

    return results
