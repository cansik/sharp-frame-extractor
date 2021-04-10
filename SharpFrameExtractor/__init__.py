from SharpFrameExtractor.estimator.CannyEstimator import CannyEstimator
from SharpFrameExtractor.estimator.SobelEstimator import SobelEstimator

DefaultEstimators = {
    "canny": CannyEstimator(),
    "sobel": SobelEstimator()
}
