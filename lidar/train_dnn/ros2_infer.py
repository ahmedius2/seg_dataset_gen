#!/usr/bin/env python3
"""
dnn_inference_node.py

Subscribes to AlignedCloudWithPose, runs the pillar-based BEV occupancy
model from train_dnn (on GPU, fp16), and projects the resulting local BEV
logits onto a 300x300 m^2 global occupancy grid.

Design:
  * Model runs on GPU under torch.autocast (fp16) — the real speedup.
  * Everything else (coverage rasterization, log-odds fusion, publishing)
    runs on CPU to avoid many small kernel launches and D2H syncs.
"""

import os
import sys
import time
import array

import numpy as np
import torch
import torch.nn.functional as F

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, Point, Quaternion
from pc_transform_cpp.msg import AlignedCloudWithPose


# ---------------------------------------------------------------------------
# Make the training package importable.
# ---------------------------------------------------------------------------
TRAIN_DNN_ROOT = os.environ.get(
    "TRAIN_DNN_ROOT",
    "/home/ubuntu/shared/seg_dataset_gen/lidar",
)
if TRAIN_DNN_ROOT not in sys.path:
    sys.path.insert(0, TRAIN_DNN_ROOT)

from train_dnn.config import Config                     # noqa: E402
from train_dnn.train import build_model                 # noqa: E402
from train_dnn.dataset import dilate_bev_mask           # noqa: E402

# ---------------------------------------------------------------------------
# Global map configuration
# ---------------------------------------------------------------------------
GLOBAL_MAP_SIZE_M      = 300.0
GLOBAL_MAP_RESOLUTION  = 0.3
GLOBAL_MAP_WIDTH       = int(GLOBAL_MAP_SIZE_M / GLOBAL_MAP_RESOLUTION)
GLOBAL_MAP_HEIGHT      = int(GLOBAL_MAP_SIZE_M / GLOBAL_MAP_RESOLUTION)

WORLD_FRAME = os.environ.get("SCENE_NAME", "scene_0000_0")

SCORE_THRESHOLD = 0.5

MODEL_WEIGHTS = os.environ.get(
    "DNN_WEIGHTS",
    "/home/ubuntu/shared/seg_dataset_gen/lidar/dnn_runs_001/best.pt",
)

# ---------------------------------------------------------------------------
# Global CUDA perf flags (safe on CPU-only machines too).
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


class DNNInferenceNode(Node):
    def __init__(self):
        super().__init__("dnn_inference_node")

        # ---- device + config + model ----
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = Config()
        self.cfg.__post_init__()
        self.model = self._load_model().to(self.device).eval()

        self.grid_h = self.cfg.grid_h
        self.grid_w = self.cfg.grid_w
        self.get_logger().info(
            f"Model loaded on {self.device}; BEV grid = "
            f"{self.grid_w} x {self.grid_h} cells @ {self.cfg.cell_size} m"
        )

        # ---- AMP settings for inference ----
        self.use_amp = (self.device.type == "cuda")
        self.amp_dtype = torch.float16 if self.use_amp else torch.float32

        # ---- global occupancy grid (CPU) ----
        self.global_map = np.full(
            (GLOBAL_MAP_HEIGHT, GLOBAL_MAP_WIDTH), -1, dtype=np.int8
        )
        self.log_odds = np.zeros(
            (GLOBAL_MAP_HEIGHT, GLOBAL_MAP_WIDTH), dtype=np.float32
        )
        self.global_origin_x = -GLOBAL_MAP_SIZE_M / 2.0
        self.global_origin_y = -GLOBAL_MAP_SIZE_M / 2.0

        # Log-odds update parameters
        self.L_OCC    =  0.85
        self.L_FREE   = -0.40
        self.L_MIN    = -2.00
        self.L_MAX    =  3.50
        self.L_THRESH =  0.00

        # ---- ROS interfaces ----
        self.subscription = self.create_subscription(
            AlignedCloudWithPose,
            "/sim_lidar/pointcloud/aligned_with_pose",
            self.inference_callback,
            qos_profile_sensor_data,
        )
        self.map_pub = self.create_publisher(
            OccupancyGrid, "/global_occupancy_grid", 10
        )
        self.local_pub = self.create_publisher(
            OccupancyGrid, "/local_occupancy_grid", 10
        )

        self.log_exec_time = False

        self.get_logger().info(
            f"Global map: {GLOBAL_MAP_WIDTH}x{GLOBAL_MAP_HEIGHT} cells "
            f"({GLOBAL_MAP_SIZE_M}x{GLOBAL_MAP_SIZE_M} m @ "
            f"{GLOBAL_MAP_RESOLUTION} m/cell), origin = "
            f"({self.global_origin_x:.1f}, {self.global_origin_y:.1f})"
        )

    # -----------------------------------------------------------------------
    # Model loading (unchanged)
    # -----------------------------------------------------------------------
    def _load_model(self):
        if not os.path.isfile(MODEL_WEIGHTS):
            raise FileNotFoundError(
                f"DNN weights not found at {MODEL_WEIGHTS}. "
                f"Set the DNN_WEIGHTS env var to a checkpoint."
            )

        ckpt = torch.load(MODEL_WEIGHTS, map_location="cpu", weights_only=False)

        if isinstance(ckpt, dict) and "model" in ckpt and "config" in ckpt:
            cfg_dict = ckpt["config"]
            for k in ("grid_h", "grid_w", "grid_size"):
                cfg_dict.pop(k, None)
            self.cfg = Config(**cfg_dict)
            self.cfg.__post_init__()
            state_dict = ckpt["model"]
            self.get_logger().info(
                f"Loaded checkpoint (epoch {ckpt.get('epoch')}, "
                f"best val IoU {ckpt.get('best_iou', float('nan')):.4f})"
            )
        else:
            state_dict = ckpt
            self.get_logger().info("Loaded raw state_dict")

        model = build_model(self.cfg)
        model.load_state_dict(state_dict, strict=True)
        return model

    # -----------------------------------------------------------------------
    # Inference — GPU, fp16; returns a numpy array on the CPU.
    # One D2H copy of a 512x128 fp32 tensor (~256 KB) is negligible.
    # -----------------------------------------------------------------------
    @torch.no_grad()
    def _run_inference(self, xyz_aligned):
        """
        xyz_aligned: (N, 3) float32 numpy array in the aligned frame.
        Returns: (H, W) float32 numpy array of sigmoid probabilities on CPU.
        """
        if xyz_aligned.shape[0] == 0:
            return np.zeros((self.grid_h, self.grid_w), dtype=np.float32)

        pts = torch.as_tensor(
            xyz_aligned, dtype=torch.float32, device=self.device
        )
        batch_idx = torch.zeros(
            (pts.shape[0], 1), dtype=torch.float32, device=self.device
        )
        model_input = torch.cat([batch_idx, pts], dim=1)

        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.use_amp,
        ):
            logits = self.model(model_input)          # (1, H, W)

        prob = torch.sigmoid(logits.float())[0]       # (H, W), fp32 on GPU
        # Single D2H copy, then hand the rest of the pipeline to CPU.
        return prob.cpu().numpy()

    # -----------------------------------------------------------------------
    # Coverage rasterize — CPU (same as original).
    # -----------------------------------------------------------------------
    def _rasterize_coverage(self, xyz_aligned):
        """
        Return a (grid_h, grid_w) bool mask that is True wherever a point
        landed, dilated by cfg.pred_coverage_kernel.
        """
        cfg = self.cfg
        observed = np.zeros((self.grid_h, self.grid_w), dtype=bool)
        if xyz_aligned.shape[0] == 0:
            return observed

        u = xyz_aligned[:, cfg.col_axis]
        v = xyz_aligned[:, cfg.row_axis]

        j = np.floor((u - cfg.x_min) / cfg.cell_size).astype(np.int64)
        i = np.floor((v - cfg.y_min) / cfg.cell_size).astype(np.int64)

        inside = (i >= 0) & (i < self.grid_h) & (j >= 0) & (j < self.grid_w)
        if inside.any():
            observed[i[inside], j[inside]] = True

        coverage = dilate_bev_mask(observed, cfg.pred_coverage_kernel)
        return np.asarray(coverage, dtype=bool)

    # -----------------------------------------------------------------------
    # Main callback
    # -----------------------------------------------------------------------
    def inference_callback(self, msg: AlignedCloudWithPose):
        # ---- 1. Unpack ----
        start = time.perf_counter()
        raw = np.asarray(msg.points, dtype=np.float32)
        if raw.size == 0 or raw.size % 3 != 0:
            return
        pts = raw.reshape(-1, 3)

        finite = np.isfinite(pts).all(axis=1)
        if not finite.all():
            pts = pts[finite]
        if pts.shape[0] == 0:
            return
        unpack_ms = (time.perf_counter() - start) * 1000.0

        # ---- 2. Inference (GPU, fp16) ----
        start = time.perf_counter()
        try:
            prob = self._run_inference(pts)         # numpy (H, W), CPU
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"Inference failed: {e}")
            return

        if prob.shape != (self.grid_h, self.grid_w):
            self.get_logger().error(
                f"Unexpected BEV shape {prob.shape}, expected "
                f"({self.grid_h}, {self.grid_w})"
            )
            return
        infer_ms = (time.perf_counter() - start) * 1000.0

        # ---- 3. Coverage (CPU) ----
        start = time.perf_counter()
        coverage = self._rasterize_coverage(pts)
        rasterize_ms = (time.perf_counter() - start) * 1000.0

        # ---- 4. Fusion (CPU) ----
        start = time.perf_counter()
        ox  = float(msg.origin_world.x)
        oy  = float(msg.origin_world.y)
        oz  = float(msg.origin_world.z)
        yaw = float(msg.yaw)
        stamp = msg.header.stamp

        if coverage.any():
            self._project_to_global(prob, coverage, ox, oy, yaw)

        project_ms = (time.perf_counter() - start) * 1000.0

        if self.log_exec_time:
            self.get_logger().info(
                "Callback execution times: "
                f"unpack {unpack_ms:.3f} ms, infer {infer_ms:.3f} ms, "
                f"rasterize {rasterize_ms:.3f} ms, project {project_ms:.3f} ms"
            )

        # ---- 5. Publish ----
        self._publish_global_map(stamp)
        self._publish_local_map(prob, coverage, ox, oy, oz, yaw, stamp)

    # -----------------------------------------------------------------------
    # Fusion — CPU, with np.add.at replaced by np.bincount.
    # -----------------------------------------------------------------------
    def _project_to_global(self, prob, coverage, origin_x, origin_y, yaw):
        occ_mask  = coverage & (prob >= SCORE_THRESHOLD)
        free_mask = coverage & (prob <  SCORE_THRESHOLD)

        self._scatter(occ_mask,  origin_x, origin_y, yaw, self.L_OCC)
        self._scatter(free_mask, origin_x, origin_y, yaw, self.L_FREE)

        self._refresh_global_map()

    def _scatter(self, mask, origin_x, origin_y, yaw, delta):
        """
        Add `delta` to log-odds at the global cells covered by `mask`.

        Uses np.bincount instead of np.add.at, which is 2-10x faster for
        large index arrays with duplicate indices.
        """
        rows, cols = np.where(mask)
        if rows.size == 0:
            return

        cfg = self.cfg

        # Cell centres in the aligned frame.
        pa_x = cfg.x_min + (cols.astype(np.float64) + 0.5) * cfg.cell_size
        pa_y = cfg.y_min + (rows.astype(np.float64) + 0.5) * cfg.cell_size

        # Inverse rotation: Rz(+yaw).
        c = np.cos(yaw)
        s = np.sin(yaw)
        pl_x = c * pa_x - s * pa_y
        pl_y = s * pa_x + c * pa_y

        # Inverse translation.
        pw_x = origin_x + pl_x
        pw_y = origin_y + pl_y

        # World coords -> global grid indices.
        gcol = np.floor(
            (pw_x - self.global_origin_x) / GLOBAL_MAP_RESOLUTION
        ).astype(np.int64)
        grow = np.floor(
            (pw_y - self.global_origin_y) / GLOBAL_MAP_RESOLUTION
        ).astype(np.int64)

        valid = (
            (grow >= 0) & (grow < GLOBAL_MAP_HEIGHT) &
            (gcol >= 0) & (gcol < GLOBAL_MAP_WIDTH)
        )
        if not np.any(valid):
            return
        grow = grow[valid]
        gcol = gcol[valid]

        flat = self.log_odds.reshape(-1)
        idx  = grow * GLOBAL_MAP_WIDTH + gcol

        # np.bincount(idx, minlength=flat.size) gives, for each cell, the
        # number of updates; multiply by delta. This is much faster than
        # np.add.at for many duplicate indices.
        counts = np.bincount(idx, minlength=flat.size).astype(np.float32)
        flat += counts * np.float32(delta)

        np.clip(flat, self.L_MIN, self.L_MAX, out=flat)

    def _refresh_global_map(self):
        lo = self.log_odds
        gm = self.global_map
        gm.fill(-1)
        gm[lo <  self.L_THRESH] = 0
        gm[lo >  self.L_THRESH] = 100

    # -----------------------------------------------------------------------
    # Publishers
    # -----------------------------------------------------------------------
    def _publish_global_map(self, stamp):
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = WORLD_FRAME
        msg.info.resolution = GLOBAL_MAP_RESOLUTION
        msg.info.width = GLOBAL_MAP_WIDTH
        msg.info.height = GLOBAL_MAP_HEIGHT
        msg.info.origin = Pose(
            position=Point(x=self.global_origin_x, y=self.global_origin_y, z=0.0),
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        )
        # Fast int8 -> bytes -> array.
        msg.data = array.array('b', self.global_map.flatten().tobytes())
        self.map_pub.publish(msg)

    def _publish_local_map(self, prob, coverage, ox, oy, oz, yaw, stamp):
        cfg = self.cfg

        corner_a = np.array([cfg.x_min, cfg.y_min, 0.0], dtype=np.float64)
        c = np.cos(yaw)
        s = np.sin(yaw)
        corner_l = np.array([
            c * corner_a[0] - s * corner_a[1],
            s * corner_a[0] + c * corner_a[1],
            0.0,
        ])
        origin_world = np.array([ox, oy, oz], dtype=np.float64) + corner_l

        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = WORLD_FRAME
        msg.info.resolution = cfg.cell_size
        msg.info.width = self.grid_w
        msg.info.height = self.grid_h
        msg.info.origin = Pose(
            position=Point(
                x=float(origin_world[0]),
                y=float(origin_world[1]),
                z=float(origin_world[2]),
            ),
            orientation=Quaternion(
                x=0.0,
                y=0.0,
                z=float(np.sin(yaw * 0.5)),
                w=float(np.cos(yaw * 0.5)),
            ),
        )

        # -1 unknown, 0 free, 100 occupied.
        data = np.full((self.grid_h, self.grid_w), -1, dtype=np.int8)
        data[coverage & (prob >= SCORE_THRESHOLD)] = 100
        data[coverage & (prob <  SCORE_THRESHOLD)] = 0
        msg.data = array.array('b', data.tobytes())
        self.local_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = DNNInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()