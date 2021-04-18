import cv2
import numpy as np

from sharp_frame_extractor.estimator.BaseEstimator import BaseEstimator


class SobelEstimator(BaseEstimator):
    def setup(self):
        pass

    def release(self):
        pass

    def estimate(self, image: np.ndarray) -> float:
        Gx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
        Gy = cv2.Sobel(image, cv2.CV_32F, 0, 1)

        normGx = cv2.norm(Gx)
        normGy = cv2.norm(Gy)

        height, width, channels = image.shape

        sumSq = normGx * normGx + normGy * normGy
        sharpness = 1. / (sumSq / (height * width) + 1e-6)
        return (1.0 - sharpness) * 100, 0, 0
