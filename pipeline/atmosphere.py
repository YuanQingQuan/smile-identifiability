# -*- coding: utf-8 -*-

import numpy as np
import torch


VNIR_LINES = [
    (688.0, 0.20, 2.0),
    (720.0, 0.15, 6.0),
    (760.0, 0.60, 1.8),
    (820.0, 0.30, 8.0),
    (940.0, 0.70, 15.0),
]

def build_T_torch(fine_grid_t, lines, depth_scale, tilt, device):

    T = torch.ones_like(fine_grid_t)
    for k, (c, d, w) in enumerate(lines):
        T = T - depth_scale[k] * d * torch.exp(-0.5 * ((fine_grid_t - c) / w) ** 2)
    T = T * (1.0 + tilt * (fine_grid_t - 700.0) / 300.0)
    return torch.clamp(T, 0.05, 1.2)

def build_T_numpy(fine_grid, lines, depth_scale=None, tilt=0.0):
    if depth_scale is None:
        depth_scale = np.ones(len(lines))
    T = np.ones_like(fine_grid)
    for k, (c, d, w) in enumerate(lines):
        T = T - depth_scale[k] * d * np.exp(-0.5 * ((fine_grid - c) / w) ** 2)
    return np.clip(T * (1.0 + tilt * (fine_grid - 700.0) / 300.0), 0.05, 1.2)
