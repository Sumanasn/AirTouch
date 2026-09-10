"""One-Euro filter (Casiez et al. 2012) for jitter-free cursor smoothing."""
import math
import time


class LowPassFilter:
    def __init__(self):
        self._initialized = False
        self.x_prev = 0.0

    def filter(self, x, alpha):
        if not self._initialized:
            self.x_prev = x
            self._initialized = True
            return x
        result = alpha * x + (1.0 - alpha) * self.x_prev
        self.x_prev = result
        return result

    def reset(self):
        self._initialized = False
        self.x_prev = 0.0


def _smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.t_prev = None
        self.x_prev = None
        self.last_velocity = 0.0  # filtered derivative (edx), exposed for callers that need velocity

    def __call__(self, x, t=None):
        t = time.perf_counter() if t is None else t
        if self.t_prev is None:
            self.t_prev = t
            self.x_prev = x
            self.x_filter.filter(x, 1.0)
            self.dx_filter.filter(0.0, 1.0)
            self.last_velocity = 0.0
            return x

        t_e = max(t - self.t_prev, 1e-6)
        dx = (x - self.x_prev) / t_e
        edx = self.dx_filter.filter(dx, _smoothing_factor(t_e, self.d_cutoff))
        self.last_velocity = edx

        cutoff = self.min_cutoff + self.beta * abs(edx)
        x_hat = self.x_filter.filter(x, _smoothing_factor(t_e, cutoff))

        self.t_prev = t
        self.x_prev = x
        return x_hat

    def reset(self):
        self.x_filter.reset()
        self.dx_filter.reset()
        self.t_prev = None
        self.x_prev = None
        self.last_velocity = 0.0


class OneEuroFilter2D:
    """Convenience wrapper filtering (x, y) as independent 1D signals."""

    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.fx = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self.fy = OneEuroFilter(min_cutoff, beta, d_cutoff)

    def __call__(self, x, y, t=None):
        return self.fx(x, t), self.fy(y, t)

    def reset(self):
        self.fx.reset()
        self.fy.reset()
