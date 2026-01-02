# Sharp Frame Extractor
Extracts sharp frames from a video by using a time window to detect the sharpest frame.

The idea of the extractor is to provide a simple tool to extract sharp frames from videos to be used in photogrammetry and volumetric capturing.
The algorithm is currently based on the idea that the `standard deviation` represents a valid metric for how sharp a frame is. The value is calculated on an edge detection result which is by default created by a canny edge detection. The parameters for the canny edge detector are extracted per frame. To further enhance this detection, only the center of the frame is used (usually the focus of the scene).

Further ideas can be implemented, for example a sobel based method is already available.

### Example

![frames-all](https://user-images.githubusercontent.com/5220162/117341573-9a348400-aea2-11eb-9567-370a605c4f62.jpg)
*One hundred sharp frame extracted from an iPhone video.*

![from-to-trees](https://user-images.githubusercontent.com/5220162/117341592-a02a6500-aea2-11eb-89e4-f4eb3d1eac07.jpg)
*The reconstruction of the one hundred frames compared with the original cherry tree.*

You can read more about this example project over at [behance](https://www.behance.net/gallery/118822685/Immersive-Memories).

### About
MIT License - Copyright (c) 2026 Florian Bruggisser
