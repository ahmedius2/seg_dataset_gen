"""Compute per-pillar relative XYZ statistics as a 15-channel BEV image."""

import torch
import torch.nn as nn

from .pillar_ops import pillar_statistics


class StatPillarsEncoder(nn.Module):
    def __init__(self, x_min, y_min, cell_size, grid_w, grid_h, point_cloud_range,
                 max_points_per_voxel=32, num_threads=4):
        super().__init__()
        self.x_min, self.y_min = x_min, y_min
        self.cell = cell_size
        self.w, self.h = grid_w, grid_h
        self.z_center = (point_cloud_range[2] + point_cloud_range[5]) / 2
        self.point_cloud_range = list(point_cloud_range)
        self.voxel_size = [cell_size, cell_size, point_cloud_range[5] - point_cloud_range[2]]
        self.max_points_per_voxel = max_points_per_voxel
        self.num_threads = num_threads

    def forward(self, points, batch_size=None):
        # points: [M, 4] = (batch_idx, x, y, z)
        if points.device.type != "cpu":
            raise ValueError("StatPillars requires point tensors on CPU")
        if batch_size is None:
            batch_size = int(points[:, 0].max().item()) + 1 if points.shape[0] else 1
        stats, _ = pillar_statistics(
            points.detach().float().numpy(), self.point_cloud_range, self.voxel_size,
            [self.w, self.h], self.max_points_per_voxel, self.num_threads, batch_size)
        return torch.from_numpy(stats)


class StatPillarsModel(nn.Module):
    def __init__(self, encoder, backbone):
        super().__init__()
        self.encoder = encoder
        self.backbone = backbone

    def forward(self, points, batch_size=None):
        features = self.encoder(points, batch_size)
        return self.backbone(features.to(next(self.backbone.parameters()).device)).squeeze(1)
