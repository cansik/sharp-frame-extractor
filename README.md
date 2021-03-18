# Sharp Frame Extractor
Extracts sharp frames from a video by using a time window to detect the sharpest frame.

### Prerequisites

```
pip install opencv-python
```
    
### Usage

```
usage: sharp_frame_extractor.py [-h] [--output OUTPUT] [--window WINDOW]
                                [--min MIN] [--format {jpg,png,bmp,gif,tif}]
                                [--crop CROP] [--method {canny,sobel}]
                                [--debug]
                                video

positional arguments:
  video                 Path to video file

optional arguments:
  -h, --help            show this help message and exit
  --output OUTPUT       Path to output folder
  --window WINDOW       Step size per evaluation in ms
  --min MIN             Minimum sharpness level
  --format {jpg,png,bmp,gif,tif}
                        Frame output format
  --crop CROP           Crop to center ROI for sharpness detection
  --method {canny,sobel}
                        Extraction algorithm
  --debug               Shows debug frames and information
```

### About
Written by cansik 2020.
