import argparse
import multiprocessing

from SharpFrameExtractor.SharpFrameExtractor import SharpFrameExtractor
from SharpFrameExtractor import DefaultEstimators

if __name__ == "__main__":
    estimator_names = sorted(list(DefaultEstimators.keys()))

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
    a.add_argument("--cpu-count", default=multiprocessing.cpu_count(), type=int,
                   help="How many cpu's are used for the extraction (by default all).")
    a.add_argument("--preview", action='store_true', help="Only shows how many frames would be extracted.")
    a.add_argument("--debug", action='store_true', help="Shows debug frames and information.")
    args = a.parse_args()

    extractor = SharpFrameExtractor(estimator=DefaultEstimators[args.method],
                                    min_sharpness=float(args.min),
                                    crop_factor=float(args.crop),
                                    output_format=args.format,
                                    cpu_count=args.cpu_count,
                                    preview=args.preview)
    extractor.extract(args.video, args.output,
                      window_size_ms=int(args.window),
                      target_frame_count=args.frame_count)
