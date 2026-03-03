"""Saturation-aware reference governor for Joint 1.

Design: The governor uses the RAW (pre-saturation) J1 torque to compute
utilisation, not the clipped value.  This gives a true picture of how much
torque the controller *wants* to apply, which is the right signal for
deciding whether the trajectory is too aggressive.

The update happens AFTER computing tau_raw for the current tick.  The
returned scale is stored as ``scale_next`` and applied to the trajectory
limits on the NEXT tick, breaking the feedback loop that causes
oscillation when scale changes and torque react in the same tick.
"""

from __future__ import annotations


class ReferenceGovernor:
    """Monitors J1 raw torque utilisation and reduces trajectory limits.

    Returns (scale_current, scale_next) from update():
    - scale_current: the scale that was committed LAST tick (use for logging)
    - scale_next:    the scale to apply to trajectory limits for the NEXT tick
    """

    def __init__(
        self,
        torque_limit: float = 40.0,
        threshold: float = 0.90,
        alpha: float = 0.98,
        min_scale: float = 0.30,
        recovery_rate: float = 0.002,
    ) -> None:
        self.limit = abs(torque_limit)
        self.threshold = threshold
        self.alpha = alpha
        self.min_scale = min_scale
        self.recovery_rate = recovery_rate

        self.scale_current = 1.0   # committed scale (used this tick)
        self.scale_next = 1.0      # computed scale (for next tick)
        self.util_raw = 0.0        # instantaneous raw utilisation
        self.util_ema = 0.0        # EMA of raw utilisation

    def update(self, tau_raw_j1: float) -> tuple[float, float]:
        """Update governor with RAW (pre-saturation) J1 torque.

        Args:
            tau_raw_j1: J1 torque BEFORE tanh/clip saturation.

        Returns:
            (scale_current, scale_next) — both in [min_scale, 1.0].
        """
        # --- Commit: what was "next" last tick is now "current" ---
        self.scale_current = self.scale_next

        # --- Compute raw utilisation and EMA ---
        self.util_raw = abs(tau_raw_j1) / self.limit if self.limit > 0 else 0.0
        self.util_ema = (self.alpha * self.util_ema
                         + (1.0 - self.alpha) * self.util_raw)

        # --- Compute new scale for next tick ---
        s = self.scale_current
        if self.util_ema > self.threshold:
            overshoot = self.util_ema - self.threshold
            s -= overshoot * 0.1        # proportional reduction
            s = max(s, self.min_scale)
        else:
            s += self.recovery_rate      # slow recovery
            s = min(s, 1.0)

        self.scale_next = s
        return self.scale_current, self.scale_next

    def reset(self) -> None:
        """Reset governor state (e.g. after a new goal command)."""
        self.scale_current = 1.0
        self.scale_next = 1.0
        self.util_raw = 0.0
        self.util_ema = 0.0
