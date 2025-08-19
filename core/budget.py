import time

class Deadline:
    def __init__(self, total_seconds: float):
        self.t0 = time.time()
        self.total = float(total_seconds)

    @property
    def elapsed(self) -> float:
        return time.time() - self.t0

    @property
    def remaining(self) -> float:
        rem = self.total - self.elapsed
        return max(rem, 0.0)

    def near(self, threshold: float = 10.0) -> bool:
        return self.remaining <= threshold

    def exceeded(self) -> bool:
        return self.elapsed >= self.total
