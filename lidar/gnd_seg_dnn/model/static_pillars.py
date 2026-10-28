import torch
import torch.nn as nn
import torch.nn.functional as F
from .dynamic_pillars import VFETemplate

try:  # optional C++ voxelizer, build with `python setup.py build_ext --inplace` in lidar dir
    from .. import _voxelizer
except ImportError:
    _voxelizer = None

class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()

        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        # self.part = 50000

    def forward(self, inputs):
        # if inputs.shape[0] > self.part:
        #     # nn.Linear performs randomly when batch size is too large
        #     num_parts = inputs.shape[0] // self.part
        #     part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])
        #                        for num_part in range(num_parts+1)]
        #     x = torch.cat(part_linear_out, dim=0)
        # else:

        x = self.linear(inputs)
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range,
                 max_points_per_voxel=20, num_threads=4):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.max_points_per_voxel = max_points_per_voxel
        self.num_threads = num_threads
        self.scale_xy = grid_size[0] * grid_size[1]
        self.scale_y = grid_size[1]

        # Non-persistent so old checkpoints still load; follows .to(device).
        self.register_buffer("grid_size", torch.tensor(grid_size), persistent=False)
        self.register_buffer("voxel_size", torch.tensor(voxel_size), persistent=False)
        self.register_buffer("point_cloud_range", torch.tensor(point_cloud_range), persistent=False)

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def voxelize(self, points):
        """Uses the multi-threaded C++ voxelizer for CPU tensors in eval mode
        when it is built; training and GPU tensors use the torch version."""
        if (_voxelizer is not None and not self.training
                and points.device.type == "cpu"):
            return self._voxelize_cpp(points)
        return self._voxelize_torch(points)

    def _voxelize_cpp(self, points):
        vs, pr = self.voxel_size.tolist(), self.point_cloud_range.tolist()
        voxels, num_points, coords = _voxelizer.voxelize(
            points.detach().float().numpy(), pr[:2], vs[:2],
            self.grid_size[:2].tolist(), self.max_points_per_voxel,
            self.num_threads)
        return (torch.from_numpy(voxels).to(points.dtype),
                torch.from_numpy(num_points).long(),
                torch.from_numpy(coords))

    def _voxelize_torch(self, points):
        """
        Scatter points into pillars holding at most `max_points_per_voxel` points.

        Args:
            points: (M, 4) = (batch_idx, x, y, z)

        Returns:
            voxels:           (V, P, 3) xyz, zero padded
            voxel_num_points: (V,) valid points per voxel (<= P)
            voxel_coords:     (V, 4) = (batch_idx, 0, row, col), same layout as
                              DynamicPillarVFE so PointPillarScatter can be reused
        Points beyond the P-th in a pillar are dropped.
        """
        P = self.max_points_per_voxel
        # Drop points outside the grid
        points_coords = torch.floor(
            (points[:, [1, 2]] - self.point_cloud_range[[0, 1]]) / self.voxel_size[[0, 1]]
        ).int()
        mask = ((points_coords >= 0) & (points_coords < self.grid_size[[0, 1]])).all(dim=1)
        points = points[mask]
        points_coords = points_coords[mask]

        merge_coords = points[:, 0].int() * self.scale_xy + \
                       points_coords[:, 0] * self.scale_y + \
                       points_coords[:, 1]
        unq_coords, unq_inv, counts = torch.unique(
            merge_coords, return_inverse=True, return_counts=True)

        # Rank of every point inside its pillar: sorted position - pillar start
        order = torch.argsort(unq_inv, stable=True)
        starts = torch.cumsum(counts, dim=0) - counts
        rank = torch.empty_like(order)
        rank[order] = torch.arange(order.numel(), device=order.device) - starts[unq_inv[order]]

        keep = rank < P
        voxels = points.new_zeros((unq_coords.shape[0], P, 3))
        voxels[unq_inv[keep], rank[keep]] = points[keep, 1:4]
        voxel_num_points = counts.clamp(max=P)

        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // self.scale_xy,
                                    torch.zeros_like(unq_coords),
                                    unq_coords % self.scale_y,
                                    (unq_coords % self.scale_xy) // self.scale_y), dim=1)
        return voxels, voxel_num_points, voxel_coords

    def forward(self, points):
        """
        Args:
            points: (M, 4) = (batch_idx, x, y, z) in meters
        Returns:
            voxel_coords: (V, 4) = (batch_idx, 0, row, col)
            features:     (V, C) pillar features
        """
        voxel_features, voxel_num_points, voxel_coords = self.voxelize(points)
        coords = voxel_coords
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean

        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - self.z_offset

        features = [voxel_features, f_cluster, f_center] if self.use_absolute_xyz \
            else [f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        # Zero out padded slots
        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features = features * mask
        for pfn in self.pfn_layers:
            features = pfn(features)
        return voxel_coords, features.squeeze(1)
