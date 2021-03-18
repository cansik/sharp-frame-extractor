# Sharp Frame Extractor
Extracts sharp frames from a video by using a time window to detect the sharpest frame.

### Idea
The idea of the extractor is to provide a simple tool to extract sharp frames from videos to be used in photogrammetry and volumetric capturing.
The algorithm is currently based on the idea that the `standard deviation` represents a valid metric for how sharp a frame is. The value is calculated on an edge detection result which is by default created by a canny edge detection. The parameters for the canny edge detector are extracted per frame. To further enhance this detection, only the center of the frame is used (usually the focus of the scene).

Further ideas can be implemented, for example a sobel based method is already available.

### Prerequisites
The python script only depends on opencv-python which is prebuilt available on all common OS through pip:

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
