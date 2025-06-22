import numpy as np

def interpolate(q0, q1, t, T):
    """Linear interpolation between q0 and q1 over duration T."""

    alpha = np.clip(t / T, 0.0, 1.0)
    return (1 - alpha) * q0 + alpha * q1