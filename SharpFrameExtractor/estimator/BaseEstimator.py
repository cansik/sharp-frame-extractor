from abc import ABC, abstractmethod

import numpy as np


class BaseEstimator(ABC):
    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def release(self):
        pass

    @abstractmethod
    def estimate(self, image: np.ndarray) -> float:
        pass

    def __enter__(self):
        self.setup()
        return self

    def __exit__(self, type, value, traceback):
        self.release()
