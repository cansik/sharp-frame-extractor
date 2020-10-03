import argparse
import math
import os

import cv2
import numpy as np

debug = False


def extractImages(video_file, output_path, window_size_ms, min_sharpness, output_format, crop_factor):
    count = 0
    vidcap = cv2.VideoCapture(video_file)
    vidcap.read()

    # prepare paths
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # prepare vars
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_length_ms = frame_count / fps * 1000
    step_count = math.ceil(video_length_ms / window_size_ms)

    print("Video '%s' with %d FPS and %d frames (%.2fs) resulting in %d stills"
          % (os.path.basename(video_file), fps, frame_count, video_length_ms / 1000, step_count))

    for i in range(0, step_count + 1):
        window_start_ms = i * window_size_ms
        window_end_ms = window_start_ms + window_size_ms
        print(
            "analyzing batch %d/%d (%.2fs to %.2fs)..." % (i, step_count, window_start_ms / 1000, window_end_ms / 1000))

        # extracting frames and getting the one with the best metric
        frames = extract_frame_batch(vidcap, window_start_ms, window_size_ms, crop_factor)

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

        frame_path = os.path.join(output_path, "%sframe%04d_%d%s" % (prefix, count, round(sharpness), output_format))
        cv2.imwrite(frame_path, image)

        count += 1


def extract_sharpness_enhanced(frame_index, frame):
    Gx = cv2.Sobel(frame, cv2.CV_32F, 1, 0)
    Gy = cv2.Sobel(frame, cv2.CV_32F, 0, 1)

    normGx = cv2.norm(Gx)
    normGy = cv2.norm(Gy)

    height, width, channels = frame.shape

    sumSq = normGx * normGx + normGy * normGy
    sharpness = 1. / (sumSq / (height * width) + 1e-6)
    return (1.0 - sharpness) * 100, 0, 0


def extract_sharpness(frame_index, frame):
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


def extract_frame_batch(vidcap, start_ms, window_ms, crop_factor):
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
        sharpness, mean, std = extract_sharpness_enhanced(frame_index, image)
        results.append((frame_index, sharpness, mean, std))

        frame_available = True

    return results


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("video", help="Path to video file")
    a.add_argument("output", help="Path to output folder")
    a.add_argument("--window", default=250, help="Step size per evaluation")
    a.add_argument("--min", default=60, help="Minimum sharpness level")
    a.add_argument("--format", default=".jpg", help="Frame output format")
    a.add_argument("--crop", default=0.25, help="Crop to center ROI for sharpness detection")
    a.add_argument("--debug", default=False, help="Shows debug frames and information")
    args = a.parse_args()

    debug = bool(args.debug)
    extractImages(args.video, args.output, int(args.window), int(args.min), args.format, float(args.crop))
