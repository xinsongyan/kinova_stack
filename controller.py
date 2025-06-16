import numpy as np

class PDController:
    def __init__(self, Kp, Kd):
        self.Kp = np.array(Kp)
        self.Kd = np.array(Kd)

    def compute(self, q, qd, q_desired, qd_desired=None):
        if qd_desired is None:
            qd_desired = np.zeros_like(q)
        return self.Kp * (q_desired - q) + self.Kd * (qd_desired - qd)
