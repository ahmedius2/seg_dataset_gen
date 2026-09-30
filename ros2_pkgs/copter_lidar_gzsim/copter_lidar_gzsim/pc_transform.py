#!/usr/bin/env python3
"""
Transform a PointCloud2 into world frame, re-center it at the ground
intersection of the lidar's +X axis, and yaw-align it using the drone's
yaw (frame 'iris_with_lidar') so the cloud's orientation is fixed in world.

Steps per cloud:
  1. Transform into world frame via TF (world <- cloud.header.frame_id).
  2. Find the origin on the ground:
       - 'centroid' mode (default): robust centroid of near-ground points,
         computed with MAD-based outlier rejection.
       - 'ray' mode: intersection of the drone's forward axis with the
         ground plane.
  3. Translate all points by -origin (origin at the footprint center).
  4. Look up world <- iris_with_lidar at the same stamp, extract yaw.
  5. Rotate points by Rz(-yaw) so the cloud's forward always points along
     world +X.
  6. Publish in world frame and save.

The output frame is world (points carry world-oriented axes and origin at
the footprint center), so downstream DNN crop boxes stay consistent.
"""

import math
import queue
import threading
import traceback
from pathlib import Path

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, HistoryPolicy, qos_profile_sensor_data

from sensor_msgs.msg import PointCloud2, PointField

from tf2_ros import Buffer, TransformListener
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud


# --------------------------------------------------------------------------- #
#  Quaternion helpers
# --------------------------------------------------------------------------- #

def quaternion_rotate_vector(qx, qy, qz, qw, vx, vy, vz):
    """Rotate vector v by quaternion q (Hamilton convention)."""
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    rx = vx + qw * tx + (qy * tz - qz * ty)
    ry = vy + qw * ty + (qz * tx - qx * tz)
    rz = vz + qw * tz + (qx * ty - qy * tx)
    return rx, ry, rz


def quaternion_to_yaw(qx, qy, qz, qw):
    """Yaw (rotation about world +Z) from a quaternion, R = Rz Ry Rx convention."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_to_roll(qx, qy, qz, qw):
    """Roll (rotation about +X) from a quaternion, R = Rz Ry Rx convention."""
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    return math.atan2(sinr_cosp, cosr_cosp)


# --------------------------------------------------------------------------- #
#  Robust centroid helpers
# --------------------------------------------------------------------------- #

def _robust_centroid_1d(values: np.ndarray,
                        mad_k: float,
                        max_iter: int = 5,
                        min_keep: int = 10):
    """
    Robust 1D centroid via iterative MAD-based trimming.

    Returns (centroid, mask) where mask selects the inlier subset that was
    used. If the initial set is too small or MAD collapses to zero, falls
    back to the plain median.

    `mad_k` is the number of MADs to keep (typically 2.5 - 3.5).
    """
    if values.size == 0:
        return 0.0, np.zeros(0, dtype=bool)

    mask = np.ones(values.size, dtype=bool)
    for _ in range(max_iter):
        kept = values[mask]
        if kept.size < min_keep:
            break
        med = np.median(kept)
        mad = np.median(np.abs(kept - med))
        # 1.4826 scales MAD to be consistent with std for Gaussian data.
        sigma = 1.4826 * mad
        if sigma < 1e-9:
            # Degenerate distribution — keep whatever we have.
            break
        new_mask = np.zeros_like(mask)
        new_mask[mask] = np.abs(kept - med) <= mad_k * sigma
        if new_mask.sum() == mask.sum():
            mask = new_mask
            break
        mask = new_mask

    if mask.sum() == 0:
        # Nothing survived — fall back to plain median.
        return float(np.median(values)), np.ones(values.size, dtype=bool)

    kept = values[mask]
    return float(np.mean(kept)), mask


def _robust_centroid_2d(x: np.ndarray,
                        y: np.ndarray,
                        mad_k: float):
    """
    Robust 2D centroid of (x, y) using MAD-based trimming.

    Trimming is applied jointly: a point is kept only if it survives the
    MAD test in BOTH x and y (computed iteratively on the surviving set).

    Returns (cx, cy, mask).
    """
    if x.size == 0:
        return 0.0, 0.0, np.zeros(0, dtype=bool)

    mask_x = np.ones(x.size, dtype=bool)
    mask_y = np.ones(y.size, dtype=bool)

    for _ in range(5):
        # X test on current joint mask
        mask = mask_x & mask_y
        if mask.sum() < 10:
            break
        mx = np.median(x[mask])
        madx = 1.4826 * np.median(np.abs(x[mask] - mx))
        if madx >= 1e-9:
            new_mask_x = np.zeros_like(mask_x)
            new_mask_x[mask] = np.abs(x[mask] - mx) <= mad_k * madx
        else:
            new_mask_x = mask_x.copy()

        # Y test on current joint mask
        mask = new_mask_x & mask_y
        if mask.sum() < 10:
            break
        my = np.median(y[mask])
        mady = 1.4826 * np.median(np.abs(y[mask] - my))
        if mady >= 1e-9:
            new_mask_y = np.zeros_like(mask_y)
            new_mask_y[mask] = np.abs(y[mask] - my) <= mad_k * mady
        else:
            new_mask_y = mask_y.copy()

        if new_mask_x.sum() == mask_x.sum() and new_mask_y.sum() == mask_y.sum():
            mask_x, mask_y = new_mask_x, new_mask_y
            break
        mask_x, mask_y = new_mask_x, new_mask_y

    mask = mask_x & mask_y
    if mask.sum() == 0:
        return float(np.median(x)), float(np.median(y)), np.ones(x.size, dtype=bool)

    return float(np.mean(x[mask])), float(np.mean(y[mask])), mask


# --------------------------------------------------------------------------- #
#  PointCloud2 field extraction
# --------------------------------------------------------------------------- #

_ROS_DTYPE = {
    1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
    5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64,
}


def _extract_field(cloud_msg: PointCloud2, field) -> np.ndarray:
    n = cloud_msg.width * cloud_msg.height
    step = cloud_msg.point_step
    dt = np.dtype(_ROS_DTYPE[field.datatype])
    field_bytes = dt.itemsize * field.count

    raw = np.frombuffer(cloud_msg.data, dtype=np.uint8)
    expected = n * step
    if raw.size < expected:
        raise ValueError(
            f"PointCloud2 data is shorter than expected: "
            f"{raw.size} < {expected} bytes (N={n}, point_step={step})")

    rows = raw[:expected].reshape(n, step)
    buf = np.ascontiguousarray(rows[:, field.offset: field.offset + field_bytes])
    if field.count == 1:
        return buf.view(dt).reshape(n).astype(dt, copy=True)
    return buf.view(dt).reshape(n, field.count).astype(dt, copy=True)


def _build_cloud_from_xyz(cloud_msg: PointCloud2,
                          x: np.ndarray, y: np.ndarray, z: np.ndarray,
                          intensity: np.ndarray,
                          frame_id: str) -> PointCloud2:
    n = x.size
    out = PointCloud2()
    out.header = cloud_msg.header
    out.header.frame_id = frame_id
    out.height = 1
    out.width = n
    out.is_bigendian = False
    out.is_dense = True

    out.fields = [
        PointField(name='x',         offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y',         offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z',         offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    out.point_step = 16
    out.row_step = 16 * n

    arr = np.empty((n, 4), dtype=np.float32)
    arr[:, 0] = x.astype(np.float32)
    arr[:, 1] = y.astype(np.float32)
    arr[:, 2] = z.astype(np.float32)
    arr[:, 3] = intensity.astype(np.float32)
    out.data = arr.tobytes(order='C')
    return out


def write_ply_binary(path,
                     x: np.ndarray, y: np.ndarray, z: np.ndarray,
                     intensity: np.ndarray) -> int:
    n = x.size
    if n == 0:
        return 0
    out = np.empty((n, 4), dtype=np.float32)
    out[:, 0] = x.astype(np.float32)
    out[:, 1] = y.astype(np.float32)
    out[:, 2] = z.astype(np.float32)
    out[:, 3] = intensity.astype(np.float32)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment generated by pc_align.py\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float intensity\n"
        "end_header\n"
    ).encode('ascii')

    with open(path, 'wb') as fh:
        fh.write(header)
        fh.write(out.tobytes(order='C'))
    return n


# --------------------------------------------------------------------------- #
#  Node
# --------------------------------------------------------------------------- #

class AlignCloudNode(Node):

    def __init__(self):
        super().__init__('align_cloud_node')

        # I/O
        self.declare_parameter('input_cloud_topic', '/sim_lidar/pointcloud/downsampled')
        self.declare_parameter('output_cloud_topic', '/sim_lidar/pointcloud/aligned')

        # Frames
        self.declare_parameter('world_frame', 'scene_0021_1')
        self.declare_parameter('lidar_frame',
                               'iris_with_lidar/lidar_link/robosense_emx192')
        self.declare_parameter('drone_frame', 'iris_with_lidar')

        # Ground / geometry
        self.declare_parameter('ground_z', 0.0)
        self.declare_parameter('forward_axis', [1.0, 0.0, 0.0])  # in lidar frame

        # Origin computation
        #   'centroid' -> robust centroid of near-ground points (recommended)
        #   'ray'      -> intersection of drone forward axis with ground
        self.declare_parameter('origin_mode', 'centroid')

        # Centroid-mode params
        self.declare_parameter('ground_band', 0.5)        # meters around ground_z
        self.declare_parameter('mad_k', 3.0)              # inlier threshold in MADs
        self.declare_parameter('min_ground_points', 20)   # below this, skip cloud

        # Ray-mode params (roll compensation, only used if origin_mode == 'ray')
        self.declare_parameter('tilt_axis', 'y')
        self.declare_parameter('roll_scale', 1.0 / 3.0)
        self.declare_parameter('roll_tilt_sign', 1.0)

        # Range / intensity filters for saving
        self.declare_parameter('min_range', 0.0)
        self.declare_parameter('max_range', 500.0)
        self.declare_parameter('min_intensity', -1.0e9)
        self.declare_parameter('max_intensity',  1.0e9)

        # Dataset saving
        self.declare_parameter('save_clouds', True)
        self.declare_parameter('save_poses', True)
        self.declare_parameter('output_dir', '/tmp/dnn_dataset')
        self.declare_parameter('filename_prefix', 'cloud_')

        # Queue tuning
        self.declare_parameter('subscription_depth', 1000)
        self.declare_parameter('writer_queue_size', 100000)
        self.declare_parameter('drop_when_full', False)

        # Read parameters
        self.world_frame = self.get_parameter('world_frame').value
        self.lidar_frame = self.get_parameter('lidar_frame').value
        self.drone_frame = self.get_parameter('drone_frame').value
        self.ground_z = float(self.get_parameter('ground_z').value)

        axis = [float(v) for v in self.get_parameter('forward_axis').value]
        n = math.sqrt(sum(v * v for v in axis))
        if n < 1e-9:
            raise ValueError('forward_axis must be non-zero')
        self.forward_axis = [v / n for v in axis]

        self.origin_mode = str(self.get_parameter('origin_mode').value).lower()
        if self.origin_mode not in ('centroid', 'ray'):
            raise ValueError("origin_mode must be 'centroid' or 'ray'")

        self.ground_band = float(self.get_parameter('ground_band').value)
        self.mad_k = float(self.get_parameter('mad_k').value)
        self.min_ground_points = int(self.get_parameter('min_ground_points').value)

        self.tilt_axis = str(self.get_parameter('tilt_axis').value).lower()
        if self.tilt_axis not in ('y', 'z'):
            raise ValueError("tilt_axis must be 'y' or 'z'")
        self.roll_scale = float(self.get_parameter('roll_scale').value)
        self.roll_tilt_sign = float(self.get_parameter('roll_tilt_sign').value)

        self.min_range = float(self.get_parameter('min_range').value)
        self.max_range = float(self.get_parameter('max_range').value)
        self.min_intensity = float(self.get_parameter('min_intensity').value)
        self.max_intensity = float(self.get_parameter('max_intensity').value)

        self.save_clouds = bool(self.get_parameter('save_clouds').value)
        self.save_poses = bool(self.get_parameter('save_poses').value)
        self.output_dir = Path(self.get_parameter('output_dir').value)
        self.filename_prefix = str(self.get_parameter('filename_prefix').value)

        self.subscription_depth = int(self.get_parameter('subscription_depth').value)
        self.writer_queue_size = int(self.get_parameter('writer_queue_size').value)
        self.drop_when_full = bool(self.get_parameter('drop_when_full').value)

        # State
        self._saved_count = 0
        self._pose_count = 0
        self._dropped_count = 0
        if self.save_clouds or self.save_poses:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publisher / subscriber
        self.cloud_pub = self.create_publisher(
            PointCloud2,
            self.get_parameter('output_cloud_topic').value,
            qos_profile_sensor_data)

        sub_qos = QoSProfile(
            depth=self.subscription_depth,
            history=HistoryPolicy.KEEP_LAST,
            reliability=qos_profile_sensor_data.reliability,
            durability=qos_profile_sensor_data.durability,
        )
        self.cloud_sub = self.create_subscription(
            PointCloud2,
            self.get_parameter('input_cloud_topic').value,
            self.cloud_callback,
            sub_qos)

        # Writer thread
        maxsize = self.writer_queue_size if self.writer_queue_size > 0 else 0
        self._writer_queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._writer_stop = threading.Event()
        self._writer_thread = threading.Thread(
            target=self._writer_loop, name='align_writer', daemon=True)
        self._writer_thread.start()

        self.get_logger().info(
            f"Aligning clouds from '{self.lidar_frame}' to world "
            f"'{self.world_frame}', yaw source '{self.drone_frame}', "
            f"ground_z={self.ground_z}, origin_mode='{self.origin_mode}', "
            f"ground_band={self.ground_band}, mad_k={self.mad_k}.")

    # ------------------------------------------------------------------ #

    def _compute_ground_intersection(self, stamp):
        """
        Ray-mode origin: intersection of the drone's forward axis (with a
        partial roll compensation) with z = ground_z. Returns (x, y, z) or
        None.
        """
        # world <- lidar (origin)
        try:
            tf_wl = self.tf_buffer.lookup_transform(
                self.world_frame, self.lidar_frame, stamp,
                timeout=Duration(seconds=0.2))
        except Exception as error:
            self.get_logger().warning(
                f'No TF {self.world_frame} <- {self.lidar_frame} at '
                f'{stamp.sec}.{stamp.nanosec:09d}: {error}',
                throttle_duration_sec=5.0)
            return None

        tr = tf_wl.transform
        sx, sy, sz = tr.translation.x, tr.translation.y, tr.translation.z

        # world <- drone (orientation)
        try:
            tf_wd = self.tf_buffer.lookup_transform(
                self.world_frame, self.drone_frame, stamp,
                timeout=Duration(seconds=0.2))
        except Exception as error:
            self.get_logger().warning(
                f'No TF {self.world_frame} <- {self.drone_frame} at '
                f'{stamp.sec}.{stamp.nanosec:09d}: {error}',
                throttle_duration_sec=5.0)
            return None

        dq = tf_wd.transform.rotation

        # drone forward (+X_drone) in world
        fx, fy, fz = quaternion_rotate_vector(dq.x, dq.y, dq.z, dq.w, 1.0, 0.0, 0.0)

        # roll compensation about the lateral axis
        roll = quaternion_to_roll(dq.x, dq.y, dq.z, dq.w)
        comp = self.roll_scale * self.roll_tilt_sign * roll

        lx, ly, lz = quaternion_rotate_vector(dq.x, dq.y, dq.z, dq.w, 0.0, 1.0, 0.0)

        kn = math.sqrt(lx * lx + ly * ly + lz * lz)
        if kn < 1e-9:
            return None
        kx, ky, kz = lx / kn, ly / kn, lz / kn

        vn = math.sqrt(fx * fx + fy * fy + fz * fz)
        if vn < 1e-9:
            return None
        vx, vy, vz = fx / vn, fy / vn, fz / vn

        ct, st = math.cos(comp), math.sin(comp)
        kxv = (ky * vz - kz * vy, kz * vx - kx * vz, kx * vy - ky * vx)
        kdv = kx * vx + ky * vy + kz * vz

        dx = vx * ct + kxv[0] * st + kx * kdv * (1.0 - ct)
        dy = vy * ct + kxv[1] * st + ky * kdv * (1.0 - ct)
        dz = vz * ct + kxv[2] * st + kz * kdv * (1.0 - ct)

        if abs(dz) < 1e-8:
            self.get_logger().warning(
                'Ray is parallel to ground — cannot find intersection.',
                throttle_duration_sec=5.0)
            return None

        t = (self.ground_z - sz) / dz
        if t < 0.0:
            self.get_logger().warning(
                'Ground intersection is behind the sensor.',
                throttle_duration_sec=5.0)
            return None

        return (sx + t * dx, sy + t * dy, self.ground_z)

    # ------------------------------------------------------------------ #

    def _compute_centroid_origin(self, x, y, z, stamp):
        """
        Centroid-mode origin: robust centroid of near-ground points.

        Returns (hx, hy, hz, n_used) or None if too few points are near
        the ground.
        """
        band = self.ground_band
        near = np.abs(z - self.ground_z) <= band
        n_near = int(near.sum())
        if n_near < self.min_ground_points:
            self.get_logger().warning(
                f'Only {n_near} points within ±{band} m of ground_z; '
                f'need >= {self.min_ground_points}. Skipping cloud.',
                throttle_duration_sec=5.0)
            return None

        cx, cy, mask = _robust_centroid_2d(
            x[near], y[near], self.mad_k)

        n_used = int(mask.sum())
        if n_used < self.min_ground_points:
            self.get_logger().warning(
                f'Robust centroid kept only {n_used} points (< '
                f'{self.min_ground_points}); falling back to median.',
                throttle_duration_sec=5.0)

        return float(cx), float(cy), float(self.ground_z), n_used

    # ------------------------------------------------------------------ #

    def _get_drone_yaw(self, stamp):
        """
        Return the yaw of 'drone_frame' in world at `stamp`, or None on
        lookup failure. Yaw is the rotation about world +Z.
        """
        try:
            tf_wd = self.tf_buffer.lookup_transform(
                self.world_frame,
                self.drone_frame,
                stamp,
                timeout=Duration(seconds=0.2))
        except Exception as error:
            self.get_logger().warning(
                f'No TF {self.world_frame} <- {self.drone_frame} at '
                f'{stamp.sec}.{stamp.nanosec:09d}: {error}',
                throttle_duration_sec=5.0)
            return None

        q = tf_wd.transform.rotation
        return quaternion_to_yaw(q.x, q.y, q.z, q.w)

    # ------------------------------------------------------------------ #

    def cloud_callback(self, cloud_msg: PointCloud2):
        item = (cloud_msg, cloud_msg.header.stamp)
        if self.drop_when_full:
            try:
                self._writer_queue.put_nowait(item)
            except queue.Full:
                self._dropped_count += 1
                if self._dropped_count % 50 == 1:
                    self.get_logger().warning(
                        f"Writer queue full — dropped {self._dropped_count} "
                        f"messages so far.",
                        throttle_duration_sec=5.0)
        else:
            self._writer_queue.put(item)

    # ------------------------------------------------------------------ #

    def _writer_loop(self):
        while not self._writer_stop.is_set():
            try:
                cloud_msg, stamp = self._writer_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                t = Time.from_msg(stamp)

                # 1. Transform the cloud into world frame via TF.
                try:
                    tf_cloud = self.tf_buffer.lookup_transform(
                        self.world_frame,
                        cloud_msg.header.frame_id,
                        t,
                        timeout=Duration(seconds=0.5))
                except Exception as error:
                    self.get_logger().warning(
                        f'[writer] Could not transform cloud into '
                        f'{self.world_frame} at '
                        f'{stamp.sec}.{stamp.nanosec:09d}: {error}',
                        throttle_duration_sec=5.0)
                    continue

                world_cloud = do_transform_cloud(cloud_msg, tf_cloud)

                # 2. Extract fields from the world-frame cloud.
                fields = {f.name: f for f in world_cloud.fields}
                if not all(k in fields for k in ('x', 'y', 'z')):
                    self.get_logger().warning(
                        'Cloud has no x/y/z fields — skipping.',
                        throttle_duration_sec=5.0)
                    continue
                x = _extract_field(world_cloud, fields['x']).astype(np.float64, copy=False)
                y = _extract_field(world_cloud, fields['y']).astype(np.float64, copy=False)
                z = _extract_field(world_cloud, fields['z']).astype(np.float64, copy=False)
                if 'intensity' in fields:
                    intensity = _extract_field(
                        world_cloud, fields['intensity']).astype(np.float64, copy=False)
                else:
                    intensity = np.zeros(x.size, dtype=np.float64)

                # Drop non-finite.
                finite = (np.isfinite(x) & np.isfinite(y) &
                          np.isfinite(z) & np.isfinite(intensity))
                if not finite.all():
                    x, y, z, intensity = (x[finite], y[finite],
                                          z[finite], intensity[finite])
                if x.size == 0:
                    continue

                # 3. Compute origin (robust centroid or ray intersection).
                if self.origin_mode == 'centroid':
                    hit = self._compute_centroid_origin(x, y, z, stamp)
                    if hit is None:
                        continue
                    hx, hy, hz, _n_used = hit
                else:
                    hit = self._compute_ground_intersection(t)
                    if hit is None:
                        continue
                    hx, hy, hz = hit

                # 4. Yaw of the drone body in world.
                yaw = self._get_drone_yaw(t)
                if yaw is None:
                    continue

                # 5. Translate so the origin is (0, 0, 0).
                x = x - hx
                y = y - hy
                z = z - hz

                # 6. Rotate by Rz(-yaw) to remove the drone's heading.
                cy, sy = math.cos(-yaw), math.sin(-yaw)
                xr = cy * x - sy * y
                yr = sy * x + cy * y
                zr = z

                # 7. Publish in world frame.
                out_msg = _build_cloud_from_xyz(
                    world_cloud, xr, yr, zr, intensity, self.world_frame)
                self.cloud_pub.publish(out_msg)

                # 8. Save.
                if self.save_poses:
                    self._save_pose(stamp, yaw, hx, hy, hz, x.size)
                if self.save_clouds:
                    self._save_cloud(stamp, xr, yr, zr, intensity)

            except Exception as e:
                self.get_logger().error(
                    f"[writer] Failed to process cloud: {e}\n"
                    f"{traceback.format_exc()}")
            finally:
                self._writer_queue.task_done()

    # ------------------------------------------------------------------ #

    def _save_cloud(self, stamp, x, y, z, intensity):
        sec, nsec = stamp.sec, stamp.nanosec
        fname = f"{self.filename_prefix}{sec:010d}_{nsec:09d}.ply"
        fpath = self.output_dir / fname

        mask = np.ones(x.size, dtype=bool)
        if self.max_range > 0.0:
            r = np.sqrt(x * x + y * y + z * z)
            mask &= r <= self.max_range
            if self.min_range > 0.0:
                mask &= r >= self.min_range
        if self.min_intensity > -1e29 or self.max_intensity < 1e29:
            mask &= (intensity >= self.min_intensity) & (intensity <= self.max_intensity)

        xf, yf, zf, inf = x[mask], y[mask], z[mask], intensity[mask]
        try:
            n_written = write_ply_binary(str(fpath), xf, yf, zf, inf)
            if n_written == 0:
                try:
                    fpath.unlink()
                except FileNotFoundError:
                    pass
                return
            self._saved_count += 1
            if self._saved_count % 50 == 0:
                self.get_logger().info(
                    f"Saved {self._saved_count} clouds (last: {fname}, "
                    f"{n_written} points).")
        except Exception as e:
            self.get_logger().error(
                f"Failed to write {fpath}: {e}\n{traceback.format_exc()}")

    # ------------------------------------------------------------------ #

    def _save_pose(self, stamp, yaw, hx, hy, hz, n_pts):
        sec, nsec = stamp.sec, stamp.nanosec
        base = f"{self.filename_prefix}{sec:010d}_{nsec:09d}"

        pose_path = self.output_dir / f"{base}.pose.txt"
        try:
            with open(pose_path, 'w') as fh:
                fh.write(f"# yaw_rad {yaw:.9e}\n")
                fh.write(f"# origin {hx:.9e} {hy:.9e} {hz:.9e}\n")
                cy, sy = math.cos(-yaw), math.sin(-yaw)
                fh.write(f"{cy:.9e} {-sy:.9e} 0.000000000e+00 0.000000000e+00\n")
                fh.write(f"{sy:.9e}  {cy:.9e} 0.000000000e+00 0.000000000e+00\n")
                fh.write(f"0.000000000e+00 0.000000000e+00 1.000000000e+00 0.000000000e+00\n")
        except Exception as e:
            self.get_logger().error(f"[pose] Failed to write {pose_path}: {e}")

        csv_path = self.output_dir / "poses.csv"
        need_header = not csv_path.exists()
        try:
            with open(csv_path, 'a') as fh:
                if need_header:
                    fh.write("sec,nsec,yaw_rad,hx,hy,hz,n_points\n")
                fh.write(
                    f"{sec},{nsec},{yaw:.9e},"
                    f"{hx:.9e},{hy:.9e},{hz:.9e},{n_pts}\n")
        except Exception as e:
            self.get_logger().error(f"[pose] Failed to append {csv_path}: {e}")

        self._pose_count += 1
        if self._pose_count % 50 == 0:
            self.get_logger().info(
                f"Saved {self._pose_count} poses (last: {base}, "
                f"yaw={math.degrees(yaw):.2f}°).")

    # ------------------------------------------------------------------ #

    def destroy_node(self):
        deadline = 10.0
        waited = 0.0
        while not self._writer_queue.empty() and waited < deadline:
            self.get_clock().sleep_for(Duration(seconds=0.1))
            waited += 0.1
        if not self._writer_queue.empty():
            self.get_logger().warning(
                f"Shutdown with {self._writer_queue.qsize()} clouds still "
                f"queued; they will be lost.")
        self._writer_stop.set()
        try:
            self._writer_thread.join(timeout=5.0)
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = AlignCloudNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
