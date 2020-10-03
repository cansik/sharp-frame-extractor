import argparse
import math
import os

import cv2


def extractImages(video_file, output_path, window_size_ms, output_format):
    count = 0
    vidcap = cv2.VideoCapture(video_file)
    vidcap.read()

    # prepare paths
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # prepare vars
    fps = vidcap.get(cv2.CAP_PROP_FPS)  # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_length_ms = frame_count / fps * 1000
    step_count = math.ceil(video_length_ms / window_size_ms)

    print("Video '%s' with %d FPS and %d Frames (%.2fs)"
          % (os.path.basename(video_file), fps, frame_count, video_length_ms / 1000))

    for i in range(0, step_count):
        window_start_ms = i * window_size_ms
        window_end_ms = window_start_ms + window_size_ms
        print("analyzing batch #%d (%.2fs to %.2fs)..." % (i, window_start_ms / 1000, window_end_ms / 1000))

        # extracting frames and getting the best metric
        frames = extract_frame_batch(vidcap, window_start_ms, window_size_ms)

        if len(frames) == 0:
            continue

        frames = sorted(frames, key=lambda e: e[1], reverse=True)
        position_ms, sharpness, mean, std = frames[0]

        # extract and store best frame
        vidcap.set(cv2.CAP_PROP_POS_MSEC, position_ms)
        success, image = vidcap.read()

        frame_path = os.path.join(output_path, "frame%04d_%d%s" % (count, round(sharpness), output_format))
        cv2.imwrite(frame_path, image)

        count += 1


def extract_sharpness(frame):
    # detect mean and standard deviation
    edges = cv2.Canny(frame, 100, 200)
    mean, std = cv2.meanStdDev(edges)

    # unpack values
    mean = mean[0][0]
    std = std[0][0]

    sharpness = mean * std
    return sharpness, mean, std


def extract_frame_batch(vidcap, start_ms, window_ms):
    results = []

    # jump to video start
    vidcap.set(cv2.CAP_PROP_POS_MSEC, start_ms)
    end_ms = start_ms + window_ms

    frame_available = True
    while frame_available:
        # read next frame
        success, image = vidcap.read()

        # check if end of window
        time_code = vidcap.get(cv2.CAP_PROP_POS_MSEC)
        if time_code >= end_ms:
            frame_available = False
            continue

        # check end of stream
        if not success:
            frame_available = False
            continue

        # extract metrics
        sharpness, mean, std = extract_sharpness(image)
        results.append((time_code, sharpness, mean, std))

        frame_available = True

    return results


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("video", help="Path to video file")
    a.add_argument("output", help="Path to output folder")
    a.add_argument("--window", default=500, help="Step size per evaluation")
    a.add_argument("--format", default=".jpg", help="Frame output format")
    args = a.parse_args()
    extractImages(args.video, args.output, int(args.window), args.format)
