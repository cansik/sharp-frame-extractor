# Sharp Frame Extractor
Extracts sharp frames from a video by using a time window to detect the sharpest frame.

### Idea
The idea of the extractor is to provide a simple tool to extract sharp frames from videos to be used in photogrammetry and volumetric capturing.
The algorithm is currently based on the idea that the `standard deviation` represents a valid metric for how sharp a frame is. The value is calculated on an edge detection result which is by default created by a canny edge detection. The parameters for the canny edge detector are extracted per frame. To further enhance this detection, only the center of the frame is used (usually the focus of the scene).

Further ideas can be implemented, for example a sobel based method is already available.

### Prerequisites
To install the package use the following pip command:

```
pip install git+https://github.com/cansik/sharp-frame-extractor.git@1.6.4
```

### Usage

Here you find an example command that extracts a frame every `300ms` into `./frames` folder:

```bash
python -m sharp_frame_extractor --window 300 test.mov
```

It is also possible to extract a fix number of frames out of the video file. This example extracts `30` frames.

```bash
python -m sharp_frame_extractor --frame-count 30 test.mov
```

#### Performance

The script tries to estimate the best CPU count to not overload the available RAM or the computer specifications. But it is possible to overwrite this estimation by the following flag:

```bash
# uses all available CPU
python -m sharp_frame_extractor --frame-count 30 --force-cpu-count test.mov
```

#### Help

```
usage: sharp_frame_extractor [-h] [--method {canny,sobel}]
                                [--window WINDOW] [--frame-count FRAME_COUNT]
                                [--crop CROP] [--min MIN] [--output OUTPUT]
                                [--format {jpg,png,bmp,gif,tif}]
                                [--cpu-count CPU_COUNT] [--force-cpu-count]
                                [--preview] [--debug]
                                video

Extracts sharp frames from a video by using a time window to detect the
sharpest frame.

positional arguments:
  video                 Path to the video input file.

optional arguments:
  -h, --help            show this help message and exit
  --method {canny,sobel}
                        Sharpness detection method (Default canny).
  --window WINDOW       Window in ms to slide over the video and detect
                        sharpest frame from.
  --frame-count FRAME_COUNT
                        Amount of output frames. If the value is >0 the
                        extractor calculates the window size to match the
                        output frames.
  --crop CROP           Crop to center factor for ROI sharpness detection.
  --min MIN             Minimum sharpness level which is dependent on the
                        detection method used.
  --output OUTPUT       Path where to store the frames.
  --format {jpg,png,bmp,gif,tif}
                        Frame output format.
  --cpu-count CPU_COUNT
                        How many CPU's are used for the extraction (by default
                        it is calculate by available RAM and frame count).
  --force-cpu-count     Forces to use exact CPU number.
  --preview             Only shows how many frames would be extracted.
  --debug               Shows debug frames and information.
```

### About
MIT License - Copyright (c) 2021 Florian Bruggisser
