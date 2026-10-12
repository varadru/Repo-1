"""Smoothing and gap filling for per-person landmark tracks."""
import math

import numpy as np


class OneEuroFilter:
    """One Euro filter (Casiez et al. 2012) over numpy arrays of any shape."""

    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, dt):
        x = np.asarray(x, dtype=np.float64)
        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            return x.copy()
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = 1.0 / (1.0 + (1.0 / (2 * math.pi * cutoff)) / dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


def smooth_sequence(arr, valid, fps, min_cutoff, beta):
    """Run a One Euro filter forward over arr[T, ...] (only valid frames)."""
    out = arr.copy()
    f = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
    dt = 1.0 / fps
    for t in range(len(arr)):
        if valid[t]:
            out[t] = f(arr[t], dt)
    return out


def fill_gaps(arr, valid, max_gap):
    """Linearly interpolate invalid frames between valid ones.

    Gaps longer than max_gap frames hold the previous pose instead. Leading and
    trailing invalid frames copy the nearest valid frame. Returns (arr, filled
    mask) where the mask is True for every frame that now has data.
    """
    arr = arr.copy()
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        return arr, valid.copy()
    arr[: idx[0]] = arr[idx[0]]
    arr[idx[-1] + 1:] = arr[idx[-1]]
    for a, b in zip(idx[:-1], idx[1:]):
        gap = b - a - 1
        if gap <= 0:
            continue
        if gap <= max_gap:
            for k in range(1, gap + 1):
                w = k / (gap + 1)
                arr[a + k] = (1 - w) * arr[a] + w * arr[b]
        else:
            arr[a + 1:b] = arr[a]
    return arr, np.ones_like(valid)
