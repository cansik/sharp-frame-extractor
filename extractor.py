import argparse

from SharpFrameExtractor import SharpFrameExtractor
from estimator import estimators

if __name__ == "__main__":
    estimator_names = sorted(list(estimators.keys()))

    a = argparse.ArgumentParser(
        description="Extracts sharp frames from a video by using a time window to detect the sharpest frame.")
    a.add_argument("video", help="Path to the video input file.")
    a.add_argument("--method", default=estimator_names[0], choices=estimator_names,
                   help="Sharpness detection method (Default %s)." % estimator_names[0])
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

    with estimators[args.method] as estimator:
        extractor = SharpFrameExtractor(estimator=estimator,
                                        min_sharpness=float(args.min),
                                        crop_factor=float(args.crop),
                                        output_format=args.format)
        extractor.extract_images(args.video, args.output,
                                 window_size_ms=int(args.window),
                                 target_frame_count=args.frame_count)
