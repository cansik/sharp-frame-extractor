from sharp_frame_extractor.estimator.CannyEstimator import CannyEstimator
from sharp_frame_extractor.estimator.SobelEstimator import SobelEstimator

DefaultEstimators = {
    "canny": CannyEstimator(),
    "sobel": SobelEstimator()
}
