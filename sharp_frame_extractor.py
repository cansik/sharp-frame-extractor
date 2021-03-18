import argparse
import math
import os
import time

import cv2
import numpy as np

debug = False


class ExponentialMovingAverage(object):
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.value = None

    def add(self, value):
        if self.value is None:
            self.value = value
            return

        self.value = self.value + self.alpha * (value - self.value)


def extract_images(video_file, output_path, window_size_ms, min_sharpness, output_format, crop_factor,
                   extraction_method, target_frame_count: int = -1):
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
    video_length_ms = frame_count / fps * 1000

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
        frames = extract_frame_batch(vidcap, window_start_ms, window_size_ms, crop_factor, extraction_method)

        if len(frames) == 0:
            print("ERROR: No frames extracted (maybe a video error!)")
            continue

        frames = sorted(frames, key=lambda e: e[1], reverse=True)
        index, sharpness, mean, std = frames[0]

        prefix = ""
        if sharpness < min_sharpness:
            print("WARNING: Sharpness not high enough (%.2fs)" % sharpness)
            prefix = "WARNING_"

        # extract and store best frame
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, index)
        success, image = vidcap.read()

        if debug:
            frame_path = os.path.join(output_path,
                                      "%sframe%04d_%d.%s" % (prefix, count, round(sharpness), output_format))
        else:
            frame_path = os.path.join(output_path, "%sframe%04d.%s" % (prefix, count, output_format))
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


def extract_frame_batch(vidcap, start_ms, window_ms, crop_factor, extraction_method):
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
        sharpness, mean, std = extraction_method(frame_index, image)
        results.append((frame_index, sharpness, mean, std))

        frame_available = True

    return results


def extract_sharpness_sobel(frame_index, frame):
    Gx = cv2.Sobel(frame, cv2.CV_32F, 1, 0)
    Gy = cv2.Sobel(frame, cv2.CV_32F, 0, 1)

    normGx = cv2.norm(Gx)
    normGy = cv2.norm(Gy)

    height, width, channels = frame.shape

    sumSq = normGx * normGx + normGy * normGy
    sharpness = 1. / (sumSq / (height * width) + 1e-6)
    return (1.0 - sharpness) * 100, 0, 0


def extract_sharpness_canny(frame_index, frame):
    # extract best parameters for canny
    v = np.median(frame)

    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    # detect mean and standard deviation
    edges = cv2.Canny(frame, lower, upper)
    mean, std = cv2.meanStdDev(edges)

    # unpack values
    mean = mean[0][0]
    std = std[0][0]

    sharpness = mean * std

    if debug:
        text = "Frame #%d Sharpness=%.2fs Mean=%.2fs StdDev=%.2fs" % (frame_index, sharpness, mean, std)
        print("Canny: L=%d H=%d" % (lower, upper))
        print(text)
        colored_edges = cv2.merge((edges, edges, edges))
        cv2.putText(colored_edges, text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Preview", colored_edges)
        cv2.waitKey(0)

    return sharpness, mean, std


if __name__ == "__main__":
    a = argparse.ArgumentParser(
        description="Extracts sharp frames from a video by using a time window to detect the sharpest frame.")
    a.add_argument("video", help="Path to the video input file.")
    a.add_argument("--method", default="canny", choices=['canny', 'sobel'], help="Sharpness detection method.")
    a.add_argument("--window", default=750, type=int,
                   help="Window in ms to slide over the video and detect sharpest frame from.")
    a.add_argument("--frame-count", default=-1, type=int,
                   help="Amount of output frames. "
                        "If the value is >0 the extractor calculates the window size to match the output frames.")
    a.add_argument("--crop", default=0.25, type=float, help="Crop to center factor for ROI  sharpness detection.")
    a.add_argument("--min", default=0, type=float,
                   help="Minimum sharpness level which is dependent on the detection method used.")
    a.add_argument("--output", default='frames', help="Path where to store the frames.")
    a.add_argument("--format", default="jpg", choices=['jpg', 'png', 'bmp', 'gif', 'tif'], help="Frame output format.")
    a.add_argument("--debug", action='store_true', help="Shows debug frames and information.")
    args = a.parse_args()

    extraction_method = extract_sharpness_canny
    if args.method == "sobel":
        extraction_method = extract_sharpness_sobel

    debug = bool(args.debug)
    extract_images(args.video, args.output, int(args.window), float(args.min), args.format, float(args.crop),
                   extraction_method, target_frame_count=args.frame_count)
