"""Compute ENMO from gal-valued accelerometer input."""
from __future__ import annotations

import numpy as np

GAL_TO_G = 0.01 / 9.80665  # 1 gal = 0.01 m/s^2 ; 1 g = 9.80665 m/s^2


def compute_enmo(x_axis, y_axis, z_axis):
    magnitude = np.sqrt(x_axis ** 2 + y_axis ** 2 + z_axis ** 2)
    return np.maximum(0.0, magnitude * GAL_TO_G - 1.0)
