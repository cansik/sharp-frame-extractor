import cv2
import numpy as np

from estimator.BaseEstimator import BaseEstimator


class CannyEstimator(BaseEstimator):
    def setup(self):
        pass

    def release(self):
        pass

    def estimate(self, image: np.ndarray) -> float:
        # extract best parameters for canny
        v = np.median(image)

        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))

        # detect mean and standard deviation
        edges = cv2.Canny(image, lower, upper)
        mean, std = cv2.meanStdDev(edges)

        # unpack values
        mean = mean[0][0]
        std = std[0][0]

        return mean * std
