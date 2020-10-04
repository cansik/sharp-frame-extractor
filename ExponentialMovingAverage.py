class ExponentialMovingAverage(object):
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.value = None

    def add(self, value):
        if self.value is None:
            self.value = value
            return

        self.value = self.value + self.alpha * (value - self.value)
