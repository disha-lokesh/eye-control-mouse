class EMA:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.value = None

    def smooth(self, new):
        if self.value is None:
            self.value = new
        self.value = self.alpha * new + (1 - self.alpha) * self.value
        return self.value
