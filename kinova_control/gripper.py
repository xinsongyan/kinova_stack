import numpy as np

# finger proximal joint order: F1-prox, F2-prox, F3-prox
_OPEN   = np.deg2rad([ 0,  0,  0])
_CLOSED = np.deg2rad([45, 45, 45])

def finger_targets(open: bool):
    """Return a 6-D q_des for *all* finger joints."""
    prox = _OPEN if open else _CLOSED
    # mirror to distal joints (if tendon coupling is missing)
    return np.concatenate([prox, prox * 0.5])
