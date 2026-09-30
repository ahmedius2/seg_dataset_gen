# Aligned point cloud + the pose metadata needed to place it in world frame.
#
# Produced by pc_transform_node after:
#   1. Transforming raw LiDAR points into world_frame.
#   2. Re-centering at origin_world (robust centroid of near-ground points,
#      or ray-ground intersection).
#   3. Rotating by -yaw so the cloud's forward axis aligns with +x.
#
# The DNN was trained on clouds in this re-centered, yaw-removed frame.
# To project a DNN output cell back to world:
#
#     xr = x_min + (j + 0.5) * cell_size
#     yr = y_min + (i + 0.5) * cell_size
#     x_world = hx + cos(yaw) * xr - sin(yaw) * yr
#     y_world = hy + sin(yaw) * xr + cos(yaw) * yr
#
# `points` is interleaved [x, y, z, intensity] in meters (SI), float32.
# Length is 4 * n_points.

std_msgs/Header header              # stamp + world_frame_id

uint32 n_points                     # number of XYZI tuples in `points`

# Origin in world frame that the cloud was re-centered to.
geometry_msgs/Point origin_world    # (hx, hy, hz)

# Drone body yaw in world frame (radians). The cloud was rotated by -yaw.
float64 yaw

# Optional: full pose of the drone in world frame, for debugging / future use.
geometry_msgs/Quaternion drone_orientation_world
geometry_msgs/Point      drone_position_world

# Aligned points: [x, y, z] * n_points, meters, float32.
float32[] points
