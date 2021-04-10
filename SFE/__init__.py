from SFE.estimator.CannyEstimator import CannyEstimator
from SFE.estimator.SobelEstimator import SobelEstimator

DefaultEstimators = {
    "canny": CannyEstimator(),
    "sobel": SobelEstimator()
}
