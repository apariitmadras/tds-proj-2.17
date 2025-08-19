import time

class Deadline:
    def __init__(self, seconds: float):
        self.start = time.time()
        self.budget = seconds
    @property
    def elapsed(self): return time.time() - self.start
    @property
    def remaining(self): return max(0.0, self.budget - self.elapsed)
    def near(self, margin: float = 5.0) -> bool: return self.remaining <= margin
    def exceeded(self) -> bool: return self.remaining <= 0
