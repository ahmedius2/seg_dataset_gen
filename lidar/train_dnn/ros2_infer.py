#!/usr/bin/env python3
"""
dnn_inference_node.py

Subscribes to AlignedCloudWithPose, runs the pillar-based BEV occupancy
model from train_dnn (on GPU, fp16), and projects the resulting local BEV
logits onto a 300x300 m^2 global occupancy grid.

Key optimizations over the naive version:
  * Fusion is O(touched cells), not O(global grid). No more 9M-element
    bincount, no more 3-pass full-grid refresh.
  * Publishing (which IS inherently whole-grid because of OccupancyGrid)
    is decoupled from the inference path and rate-limited to 1 Hz per
    topic, on its own timer + callback group, under a MultiThreadedExecutor.
"""

import os
import sys
import time
import array
import threading

import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
)

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
GLOBAL_MAP_RESOLUTION  = 0.1
GLOBAL_MAP_WIDTH       = int(GLOBAL_MAP_SIZE_M / GLOBAL_MAP_RESOLUTION)
GLOBAL_MAP_HEIGHT      = int(GLOBAL_MAP_SIZE_M / GLOBAL_MAP_RESOLUTION)

WORLD_FRAME = os.environ.get("SCENE_NAME", "scene_0000_0")

SCORE_THRESHOLD = 0.5

PUBLISH_PERIOD_S = 1.0        # 1 Hz for both maps

MODEL_WEIGHTS = os.environ.get(
    "DNN_WEIGHTS",
    "/home/ubuntu/shared/seg_dataset_gen/lidar/dnn_runs_001/best.pt",
)

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

        self.use_amp = (self.device.type == "cuda")
        self.amp_dtype = torch.float16 if self.use_amp else torch.float32

        # ---- global occupancy grid ----
        self.global_map = np.full(
            (GLOBAL_MAP_HEIGHT, GLOBAL_MAP_WIDTH), -1, dtype=np.int8
        )
        self.log_odds = np.zeros(
            (GLOBAL_MAP_HEIGHT, GLOBAL_MAP_WIDTH), dtype=np.float32
        )
        # Flat views — reused everywhere to avoid reshape allocations.
        self._log_odds_flat = self.log_odds.reshape(-1)
        self._global_map_flat = self.global_map.reshape(-1)

        self.global_origin_x = -GLOBAL_MAP_SIZE_M / 2.0
        self.global_origin_y = -GLOBAL_MAP_SIZE_M / 2.0

        # Log-odds update parameters (unchanged).
        self.L_OCC   =  1.20
        self.L_FREE  = -0.60
        self.L_MIN   = -2.00
        self.L_MAX   =  3.50
        self.L_THRESH = 0.00

        # Cached constants for the fusion path.
        self._inv_global_res = 1.0 / GLOBAL_MAP_RESOLUTION

        # ---- latest local map snapshot for the 1 Hz local publisher ----
        self._local_lock = threading.Lock()
        self._local_snapshot = None   # dict or None

        # ---- counters ----
        self._infer_count = 0
        self._last_infer_count = 0

        self.log_exec_time = False

        # ------------------------------------------------------------------
        # ROS interfaces — split across callback groups so the 1 Hz
        # publishers never block incoming clouds.
        # ------------------------------------------------------------------
        self._sub_group = MutuallyExclusiveCallbackGroup()
        self._global_pub_group = MutuallyExclusiveCallbackGroup()
        # self._local_pub_group = MutuallyExclusiveCallbackGroup()

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
        )

        self.subscription = self.create_subscription(
            AlignedCloudWithPose,
            "/sim_lidar/pointcloud/aligned_with_pose",
            self.inference_callback,
            qos,
            callback_group=self._sub_group,
        )
        self.map_pub = self.create_publisher(
            OccupancyGrid, "/global_occupancy_grid", 10
        )
        self.local_pub = self.create_publisher(
            OccupancyGrid, "/local_occupancy_grid", 10
        )

        self._global_timer = self.create_timer(
            PUBLISH_PERIOD_S, self._publish_global_map,
            callback_group=self._global_pub_group,
        )
        #self._local_timer = self.create_timer(
        #    PUBLISH_PERIOD_S, self._publish_local_map,
        #    callback_group=self._local_pub_group,
        #)

        self.get_logger().info(
            f"Global map: {GLOBAL_MAP_WIDTH}x{GLOBAL_MAP_HEIGHT} cells "
            f"({GLOBAL_MAP_SIZE_M}x{GLOBAL_MAP_SIZE_M} m @ "
            f"{GLOBAL_MAP_RESOLUTION} m/cell), origin = "
            f"({self.global_origin_x:.1f}, {self.global_origin_y:.1f}); "
            f"publishing at {1.0/PUBLISH_PERIOD_S:.1f} Hz"
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
    # Inference — unchanged
    # -----------------------------------------------------------------------
    @torch.no_grad()
    def _run_inference(self, xyz_aligned):
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
            logits = self.model(model_input)

        prob = torch.sigmoid(logits.float())[0]
        return prob.cpu().numpy()

    # -----------------------------------------------------------------------
    # Coverage rasterization
    # -----------------------------------------------------------------------
    def _rasterize_coverage(self, xyz_aligned):
        cfg = self.cfg
        observed = np.zeros((self.grid_h, self.grid_w), dtype=bool)
        if xyz_aligned.shape[0] == 0:
            return observed

        u = xyz_aligned[:, cfg.col_axis]
        v = xyz_aligned[:, cfg.row_axis]

        j = np.floor((u - cfg.x_min) * (1.0 / cfg.cell_size)).astype(np.int64)
        i = np.floor((v - cfg.y_min) * (1.0 / cfg.cell_size)).astype(np.int64)

        inside = (i >= 0) & (i < self.grid_h) & (j >= 0) & (j < self.grid_w)
        if inside.any():
            observed[i[inside], j[inside]] = True

        coverage = dilate_bev_mask(observed, cfg.pred_coverage_kernel)
        return np.asarray(coverage, dtype=bool)

    # -----------------------------------------------------------------------
    # Main callback — inference + fusion only. No publishing.
    # -----------------------------------------------------------------------
    def inference_callback(self, msg: AlignedCloudWithPose):
        t_start = time.perf_counter()

        # ---- 1. Unpack ----
        t0 = time.perf_counter()
        raw = np.asarray(msg.points, dtype=np.float32)
        if raw.size == 0 or raw.size % 3 != 0:
            return
        pts = raw.reshape(-1, 3)

        finite = np.isfinite(pts).all(axis=1)
        if not finite.all():
            pts = pts[finite]
        if pts.shape[0] == 0:
            return
        unpack_ms = (time.perf_counter() - t0) * 1000.0

        # ---- 2. Inference ----
        t0 = time.perf_counter()
        try:
            prob = self._run_inference(pts)
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"Inference failed: {e}")
            return

        if prob.shape != (self.grid_h, self.grid_w):
            self.get_logger().error(
                f"Unexpected BEV shape {prob.shape}, expected "
                f"({self.grid_h}, {self.grid_w})"
            )
            return
        infer_ms = (time.perf_counter() - t0) * 1000.0

        # ---- 3. Coverage ----
        t0 = time.perf_counter()
        coverage = self._rasterize_coverage(pts)
        rasterize_ms = (time.perf_counter() - t0) * 1000.0

        # ---- 4. Fusion (now local) ----
        t0 = time.perf_counter()
        ox = float(msg.origin_world.x)
        oy = float(msg.origin_world.y)
        oz = float(msg.origin_world.z)
        yaw = float(msg.yaw)

        if coverage.any():
            self._project_to_global(prob, coverage, ox, oy, yaw)
        fuse_ms = (time.perf_counter() - t0) * 1000.0

        # ---- 5. Hand a snapshot to the 1 Hz local publisher ----
        # We only copy prob + coverage once per inference; the publisher
        # reads it under a lock. This is a few hundred KB at most.
        with self._local_lock:
            self._local_snapshot = {
                "prob": prob,
                "coverage": coverage,
                "ox": ox, "oy": oy, "oz": oz, "yaw": yaw,
                "stamp": msg.header.stamp,
            }

        self._infer_count += 1

        if self.log_exec_time:
            total_ms = (time.perf_counter() - t_start) * 1000.0
            self.get_logger().info(
                f"[callback] unpack {unpack_ms:.2f}  infer {infer_ms:.2f}  "
                f"raster {rasterize_ms:.2f}  fuse {fuse_ms:.2f}  "
                f"total {total_ms:.2f} ms"
            )

    # -----------------------------------------------------------------------
    # Fusion — local, O(touched cells)
    # -----------------------------------------------------------------------
    def _project_to_global(self, prob, coverage, origin_x, origin_y, yaw):
        occ_mask  = coverage & (prob >= SCORE_THRESHOLD)
        free_mask = coverage & (prob <  SCORE_THRESHOLD)

        occ_idx  = self._global_indices(occ_mask,  origin_x, origin_y, yaw)
        free_idx = self._global_indices(free_mask, origin_x, origin_y, yaw)

        # Occupied cells veto free votes in the same frame.
        if occ_idx.size and free_idx.size:
            free_idx = np.setdiff1d(free_idx, occ_idx, assume_unique=False)

        # Apply both updates locally and update the published int8 map for
        # only the cells that actually changed.
        self._apply_log_odds(occ_idx,  self.L_OCC)
        self._apply_log_odds(free_idx, self.L_FREE)

    def _global_indices(self, mask, origin_x, origin_y, yaw):
        """
        Rasterize a (grid_h, grid_w) bool mask from the aligned BEV frame
        to flat indices into the global grid. Out-of-bounds cells dropped.
        """
        rows, cols = np.nonzero(mask)
        if rows.size == 0:
            return np.empty(0, dtype=np.int64)

        cfg = self.cfg

        pa_x = cfg.x_min + (cols.astype(np.float32) + 0.5) * cfg.cell_size
        pa_y = cfg.y_min + (rows.astype(np.float32) + 0.5) * cfg.cell_size

        c = np.float32(np.cos(yaw))
        s = np.float32(np.sin(yaw))
        pl_x = c * pa_x - s * pa_y
        pl_y = s * pa_x + c * pa_y

        pw_x = origin_x + pl_x
        pw_y = origin_y + pl_y

        gcol = np.floor((pw_x - self.global_origin_x) * self._inv_global_res).astype(np.int64)
        grow = np.floor((pw_y - self.global_origin_y) * self._inv_global_res).astype(np.int64)

        valid = (
            (grow >= 0) & (grow < GLOBAL_MAP_HEIGHT) &
            (gcol >= 0) & (gcol < GLOBAL_MAP_WIDTH)
        )
        if not np.any(valid):
            return np.empty(0, dtype=np.int64)

        return grow[valid] * GLOBAL_MAP_WIDTH + gcol[valid]

    def _apply_log_odds(self, flat_idx, delta):
        """
        Apply `count * delta` to the touched cells, clamp, and update the
        published int8 map for just those cells. O(touched), not O(9M).
        """
        if flat_idx.size == 0:
            return

        uniq, counts = np.unique(flat_idx, return_counts=True)
        deltas = counts.astype(np.float32) * np.float32(delta)

        lo = self._log_odds_flat
        new_lo = np.clip(lo[uniq] + deltas, self.L_MIN, self.L_MAX)
        lo[uniq] = new_lo

        # Update only the touched cells of the published map.
        gm = self._global_map_flat
        gm[uniq] = np.where(
            new_lo > self.L_THRESH, np.int8(100),
            np.where(new_lo < self.L_THRESH, np.int8(0), np.int8(-1)),
        )

    # -----------------------------------------------------------------------
    # Publishers — run on their own timers at 1 Hz.
    # -----------------------------------------------------------------------
    def _publish_global_map(self):
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = WORLD_FRAME
        msg.info.resolution = GLOBAL_MAP_RESOLUTION
        msg.info.width = GLOBAL_MAP_WIDTH
        msg.info.height = GLOBAL_MAP_HEIGHT
        msg.info.origin = Pose(
            position=Point(x=self.global_origin_x, y=self.global_origin_y, z=0.0),
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        )
        msg.data = array.array('b', self.global_map.tobytes())
        self.map_pub.publish(msg)

        n = self._infer_count - self._last_infer_count
        self._last_infer_count = self._infer_count
        # self.get_logger().info(
        #    f"Published global map ({n} inferences since last publish)"
        # )

    def _publish_local_map(self):
        with self._local_lock:
            snap = self._local_snapshot
        if snap is None:
            return

        cfg = self.cfg
        yaw = snap["yaw"]

        corner_a = np.array([cfg.x_min, cfg.y_min, 0.0], dtype=np.float64)
        c = np.cos(yaw)
        s = np.sin(yaw)
        corner_l = np.array([
            c * corner_a[0] - s * corner_a[1],
            s * corner_a[0] + c * corner_a[1],
            0.0,
        ])
        origin_world = np.array(
            [snap["ox"], snap["oy"], snap["oz"]], dtype=np.float64
        ) + corner_l

        msg = OccupancyGrid()
        msg.header.stamp = snap["stamp"]
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
                x=0.0, y=0.0,
                z=float(np.sin(yaw * 0.5)),
                w=float(np.cos(yaw * 0.5)),
            ),
        )

        prob = snap["prob"]
        coverage = snap["coverage"]

        data = np.full((self.grid_h, self.grid_w), -1, dtype=np.int8)
        data[coverage & (prob >= SCORE_THRESHOLD)] = 100
        data[coverage & (prob <  SCORE_THRESHOLD)] = 0
        msg.data = array.array('b', data.tobytes())
        self.local_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = DNNInferenceNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()