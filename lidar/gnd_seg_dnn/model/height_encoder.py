"""Elevation-map front end: per-cell max z as a 1-channel BEV image (no learned VFE).

Heights are clipped to +-height_clip and scaled to [0, 1]; empty cells are -1.
"""

import torch
import torch.nn as nn

EMPTY_VALUE = -1.0


class HeightMapEncoder(nn.Module):
    def __init__(self, x_min, y_min, cell_size, grid_w, grid_h, height_clip):
        super().__init__()
        self.x_min, self.y_min = x_min, y_min
        self.cell = cell_size
        self.w, self.h = grid_w, grid_h
        self.clip = height_clip

    def forward(self, points):
        # points: [M, 4] = (batch_idx, x, y, z)
        batch = int(points[:, 0].max().item()) + 1 if points.shape[0] else 1
        ix = torch.floor((points[:, 1] - self.x_min) / self.cell).long()
        iy = torch.floor((points[:, 2] - self.y_min) / self.cell).long()
        m = (ix >= 0) & (ix < self.w) & (iy >= 0) & (iy < self.h)
        flat = points[m, 0].long() * (self.h * self.w) + iy[m] * self.w + ix[m]
        z = points[m, 3].clamp(-self.clip, self.clip) / (2 * self.clip) + 0.5

        out = torch.full((batch * self.h * self.w,), EMPTY_VALUE,
                         dtype=points.dtype, device=points.device)
        out.scatter_reduce_(0, flat, z, reduce="amax", include_self=False)
        return out.view(batch, 1, self.h, self.w)


class HeightMapModel(nn.Module):
    def __init__(self, encoder, backbone):
        super().__init__()
        self.encoder = encoder
        self.backbone = backbone

    def forward(self, points):
        return self.backbone(self.encoder(points)).squeeze(1)
