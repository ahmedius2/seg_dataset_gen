from matplotlib import pyplot as plt
import numpy as np
import os
import random
import math
import blenderproc as bproc
import bpy

from dataset_config import (
    AREA_SIZE_M,
    BARRICADE_HEIGHT,
    BARRICADE_MIN_BLOB_PX,
    BLACK_LEVEL,
    CLEARED_BARRICADE_FRACTION,
    GRID_N,
    MASK_DIR,
    MASK_MERGE_RANDOM_ORDER,
    MASK_PX,
    PIX_SIZE,
    RED_MIN,
    RED_OTHER_MAX,
    RUBBLE_EDGE_BARRICADE_MAX,
    RUBBLE_EDGE_BARRICADE_MIN,
    RUBBLE_EDGE_BARRICADE_OFFSET_M,
    RUBBLE_EDGE_BARRICADES_ENABLED,
    RUBBLE_MAX_HEIGHT_MAX,
    RUBBLE_MAX_HEIGHT_MIN,
    RUBBLE_MIN_BLOB_PX,
    RUBBLE_SURFACE_MIN_NOISE,
    RUBBLE_SURFACE_MAX_NOISE,
    RUBBLE_WALL_PROBABILITY,
    MIN_DIST_BTW_START_TARGET_PX,
)

from scatter_objects import scatter_background_objects

from utils import set_origin_to_center_bottom_safe

def save_occupancy_grid_preview(grid, output_path, start_px=None, target_px=None):
    """Save a quick visualization for the ground-truth occupancy grid."""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=180)
    image = ax.imshow(grid, cmap="gray_r", origin="lower", vmin=0, vmax=1)
    ax.set_title("Ground-truth occupancy grid")
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")

    if start_px is not None:
        ax.plot(start_px[1], start_px[0], "go", markersize=7, label="start")
    if target_px is not None:
        ax.plot(target_px[1], target_px[0], "ro", markersize=7, label="target")
    ax.legend(loc="upper right")

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="occupied")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def mask_px_to_world(px_col, px_row, mask_px=MASK_PX, area_size=AREA_SIZE_M):
    """Convert a mask pixel (col, row) to world XY (meters), centered on origin."""
    x = -area_size / 2 + (px_col + 0.5) * (area_size / mask_px)
    y = area_size / 2 - (px_row + 0.5) * (area_size / mask_px)
    return x, y


def load_mask(mask_path):
    """Load a MASK_PX x MASK_PX RGB mask as a float array in [0, 1]."""
    img = plt.imread(mask_path)
    img = np.asarray(img, dtype=np.float32)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.dtype == np.uint8 or img.max() > 1.0:
        img = img / 255.0
    return img[:, :, :3]


def classify_mask(mask):
    """Split the mask into boolean rubble (black) and barricade (red) maps."""
    r, g, b = mask[:, :, 0], mask[:, :, 1], mask[:, :, 2]
    maxc = np.maximum(np.maximum(r, g), b)
    black_mask = maxc < BLACK_LEVEL
    red_mask = (r >= RED_MIN) & (g <= RED_OTHER_MAX) & (b <= RED_OTHER_MAX)
    black_mask &= ~red_mask
    return black_mask, red_mask


def _label_blobs(binary):
    """Label 8-connected blobs in a boolean array without external deps."""
    H, W = binary.shape
    labels = np.zeros((H, W), dtype=np.int32)
    current = 0
    neigh = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
             (0, 1), (1, -1), (1, 0), (1, 1)]
    for i in range(H):
        for j in range(W):
            if binary[i, j] and labels[i, j] == 0:
                current += 1
                stack = [(i, j)]
                labels[i, j] = current
                while stack:
                    ci, cj = stack.pop()
                    for di, dj in neigh:
                        ni, nj = ci + di, cj + dj
                        if (0 <= ni < H and 0 <= nj < W
                                and binary[ni, nj] and labels[ni, nj] == 0):
                            labels[ni, nj] = current
                            stack.append((ni, nj))
    return labels, current


def gaussian_fade_blob(mask, black_mask):
    """Fade each black blob from dark center to white edge and return weights."""
    labels, num = _label_blobs(black_mask)
    faded = np.zeros(black_mask.shape, dtype=np.float32)
    blobs = []
    for lab in range(1, num + 1):
        ys, xs = np.where(labels == lab)
        if len(xs) < RUBBLE_MIN_BLOB_PX:
            continue
        cx, cy = xs.mean(), ys.mean()
        d = np.hypot(xs - cx, ys - cy)
        sigma = max(d.max() / 3.0, 0.75)
        w = np.exp(-(d ** 2) / (2.0 * sigma ** 2))
        faded[ys, xs] = w
        blobs.append(dict(label=lab, xs=xs, ys=ys, cx=cx, cy=cy,
                          sigma=sigma, weights=w))
    return faded, blobs


def build_rubble_mesh(blob, faded, idx, rng=random, barrier_objects=None):
    """Build a single rubble pile mesh from the faded blob data."""
    max_h = rng.uniform(RUBBLE_MAX_HEIGHT_MIN, RUBBLE_MAX_HEIGHT_MAX)
    # One surface-roughness std per rubble pile; per-vertex jitter is drawn from it below.
    surface_noise_std = rng.uniform(RUBBLE_SURFACE_MIN_NOISE, RUBBLE_SURFACE_MAX_NOISE)
    # A minority of piles become flat-topped walls (vertical sides, no dome curvature).
    is_wall = rng.random() < RUBBLE_WALL_PROBABILITY

    xs, ys = blob["xs"], blob["ys"]
    occ = {(int(c), int(r)) for c, r in zip(xs, ys)}

    verts, faces = [], []
    vindex = {}

    def _vert(col, row, corner):
        gx, gy = col + corner[0], row + corner[1]
        key = (gx, gy)
        if key in vindex:
            return vindex[key]

        wx = -AREA_SIZE_M / 2 + gx * PIX_SIZE
        wy = AREA_SIZE_M / 2 - gy * PIX_SIZE

        # A vertex sits on the blob's perimeter if any of its 4 surrounding
        # cells falls outside the blob; pin those to the ground so the mesh
        # boundary never floats above z=0.
        neighbor_cells = [(-1, -1), (0, -1), (-1, 0), (0, 0)]
        is_boundary = any((gx + dc, gy + dr) not in occ for dc, dr in neighbor_cells)

        h = 0.0
        if not is_boundary:
            if is_wall:
                h = max_h
            else:
                for dc, dr in neighbor_cells:
                    pc, pr = gx + dc, gy + dr
                    sel = (xs == pc) & (ys == pr)
                    if sel.any():
                        h = max(h, float(faded[pr, pc]) * max_h)
            h += rng.gauss(0.0, surface_noise_std)
            h = max(h, 0.0)
        vindex[key] = len(verts)
        verts.append([wx, wy, h])
        return vindex[key]

    for col, row in occ:
        v00 = _vert(col, row, (0, 0))
        v10 = _vert(col, row, (1, 0))
        v11 = _vert(col, row, (1, 1))
        v01 = _vert(col, row, (0, 1))
        faces.append([v00, v10, v11])
        faces.append([v00, v11, v01])

    mesh = bpy.data.meshes.new(f"rubble_mesh_{idx}")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(f"rubble_{idx}", mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    material_name = "Gray"
    # Get the material or create it if it doesn't exist
    mat = bpy.data.materials.get(material_name)
    if not mat:
        mat = bpy.data.materials.new(f"rubble_mat_{idx}")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf is not None:
            bsdf.inputs["Base Color"].default_value = (0.32, 0.30, 0.28, 1.0)
            bsdf.inputs["Roughness"].default_value = 0.95

    obj.data.materials.append(mat)

    # Add Solidify modifier
    solidify_mod = obj.modifiers.new(name="Solidify", type='SOLIDIFY')
    solidify_mod.thickness = 0.005  # Set thickness in meters (e.g., 5mm)
    bpy.ops.object.modifier_apply(modifier=solidify_mod.name)

    set_origin_to_center_bottom_safe(obj)

    cx_world, cy_world = mask_px_to_world(blob["cx"], blob["cy"])
    edge_barricades = build_rubble_edge_barricades(occ, verts, idx, barrier_objects, rng=rng)
    return dict(idx=idx, max_height=max_h, center_world=[cx_world, cy_world],
                num_pixels=int(len(xs)), edge_barricades=edge_barricades, is_wall=is_wall)


def rubble_boundary_cells(occ):
    """Return blob cells (col, row) that have at least one 4-connected neighbor outside the blob."""
    boundary = []
    for col, row in occ:
        neighbors = [(col - 1, row), (col + 1, row), (col, row - 1), (col, row + 1)]
        if any(n not in occ for n in neighbors):
            boundary.append((col, row))
    return boundary


def build_rubble_edge_barricades(occ, verts, idx, barrier_objects, rng=random):
    """Place a handful of barrier copies along a rubble pile's boundary, each
    facing the nearest vertex of the targeted rubble (rotated around Z only)."""
    if not barrier_objects:
        return []
    if not RUBBLE_EDGE_BARRICADES_ENABLED:
        return []
    boundary_cells = rubble_boundary_cells(occ)
    if not boundary_cells:
        return []

    verts_xy = np.array([[v[0], v[1]] for v in verts], dtype=np.float32)

    num_barricades = min(rng.randint(RUBBLE_EDGE_BARRICADE_MIN, RUBBLE_EDGE_BARRICADE_MAX),
                         len(boundary_cells))
    chosen_cells = rng.sample(boundary_cells, num_barricades)

    placed = []
    for i, (col, row) in enumerate(chosen_cells):
        wx, wy = mask_px_to_world(col, row)
        dists = (verts_xy[:, 0] - wx) ** 2 + (verts_xy[:, 1] - wy) ** 2
        tx, ty = verts_xy[int(np.argmin(dists))]

        # Push the barricade outward from the pile's edge, away from the rubble.
        dx, dy = wx - tx, wy - ty
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            dx, dy, dist = 1.0, 0.0, 1.0
        ox = wx + (dx / dist) * RUBBLE_EDGE_BARRICADE_OFFSET_M
        oy = wy + (dy / dist) * RUBBLE_EDGE_BARRICADE_OFFSET_M

        barrier = rng.choice(barrier_objects)
        obj_data = barrier.data.copy()
        obj = bpy.data.objects.new(f"rubble_barricade_{idx}_{i}", obj_data)
        bpy.context.scene.collection.objects.link(obj)
        obj.location = [ox, oy, -0.05]
        # Barrier's local Y axis is treated as its facing direction; rotate
        # (around Z only) so it points back at the rubble.
        angle_to_target = math.atan2(ty - oy, tx - ox)
        obj.rotation_euler = [0.0, 0.0, angle_to_target - math.pi / 2]
        obj.scale = [1.0, 1.0, 1.0]
        placed.append(dict(idx=i, barrier_name=barrier.name, location_world=[ox, oy]))
    return placed



def build_barricade(blob_pixels, idx, barrier=None):
    """Build one oriented barricade by copying a barrier object aligned to blob's major axis."""

    ys = np.array([p[0] for p in blob_pixels], dtype=np.float32)
    xs = np.array([p[1] for p in blob_pixels], dtype=np.float32)
    cx, cy = xs.mean(), ys.mean()

    pts = np.stack([xs - cx, ys - cy], axis=1)
    cov = np.cov(pts.T) if len(pts) > 1 else np.eye(2)
    evals, evecs = np.linalg.eigh(cov)
    major = evecs[:, np.argmax(evals)]
    minor = evecs[:, np.argmin(evals)]

    proj_major = pts @ major
    proj_minor = pts @ minor
    length = (proj_major.max() - proj_major.min() + 1.0) * PIX_SIZE
    width = (proj_minor.max() - proj_minor.min() + 1.0) * PIX_SIZE
    length = max(length, PIX_SIZE)
    width = max(width, PIX_SIZE)

    angle = math.atan2(-major[1], major[0])

    wx, wy = mask_px_to_world(cx, cy)

    # Copy a barrier object if available, otherwise create a primitive
    # ctrl + a , rotation and scale, do this in blender for each object in the source scene before running the script
    if barrier:
        # Check barrier y dimension and copy multiple times to match the required length
        barrier_length = barrier.dimensions.x
        num_copies = max(1, int(math.floor(length / (barrier_length + 0.2))))  # add small gap between copies
        # spread copies evenly along the barricade line (major axis direction)
        for i in range(num_copies):
            offset = (i - (num_copies - 1) / 2) * (barrier_length  + 0.2)
            obj_data = barrier.data.copy()
            obj = bpy.data.objects.new(f"barricade_{idx}_{i}", obj_data)
            bpy.context.scene.collection.objects.link(obj)
            # Place along the barricade line (perpendicular to blob's minor axis)
            # Use perpendicular direction so barriers align along the line
            # perp_angle = angle + math.pi / 2
            obj.location = [wx + offset * math.cos(angle), wy + offset * math.sin(angle), -0.05]
            obj.rotation_euler = [0.0, 0.0, angle]  # rotate to align with the barricade line
            obj.scale = [1.0, 1.0, 1.0]  # keep original scale
    else:
        obj = bproc.object.create_primitive(
            "CUBE", scale=[length / 2.0, width / 2.0, BARRICADE_HEIGHT / 2.0])
        obj.set_name(f"barricade_{idx}")
        mat = bproc.material.create(f"barricade_mat_{idx}")
        mat.set_principled_shader_value("Base Color", [0.85, 0.15, 0.15, 1.0]) # red
        mat.set_principled_shader_value("Roughness", 0.8)
        obj.replace_materials(mat)
        # bpy.context.scene.collection.objects.link(obj)
        obj.location = [wx, wy, BARRICADE_HEIGHT / 2.0]
        obj.rotation_euler = [0.0, 0.0, angle]

    return dict(idx=idx, center_world=[wx, wy], length=length, width=width,
                yaw_rad=angle, num_pixels=int(len(xs)))


def get_red_blobs(red_mask):
    """Return a list of red barricade blobs (each a list of (row, col) pixels)."""
    labels, num = _label_blobs(red_mask)
    blobs = []
    for lab in range(1, num + 1):
        ys, xs = np.where(labels == lab)
        if len(xs) < BARRICADE_MIN_BLOB_PX:
            continue
        blobs.append(list(zip(ys.tolist(), xs.tolist())))
    return blobs


def rasterize_mask_occupancy(black_mask, red_mask, grid_n=GRID_N,
                             cleared_red_pixels=None):
    """Occupancy grid (grid_n x grid_n) from rubble + (kept) barricade pixels."""
    cleared = cleared_red_pixels or set()
    src = black_mask.copy()
    H, W = red_mask.shape
    for i in range(H):
        for j in range(W):
            if red_mask[i, j] and (i, j) not in cleared:
                src[i, j] = True
    idx = (np.arange(grid_n) * (H / grid_n)).astype(int)
    idy = (np.arange(grid_n) * (W / grid_n)).astype(int)
    up = src[np.ix_(idx, idy)].astype(np.uint8)
    return up[::-1, :]


def sample_start_target_free_px(black_mask, red_mask, min_dist_px,
                                cleared_red_pixels=None, rng=random):
    """Pick start/target on white mask pixels far apart in world XY."""
    cleared = cleared_red_pixels or set()
    H, W = black_mask.shape
    free = []
    for i in range(H):
        for j in range(W):
            if black_mask[i, j]:
                continue
            if red_mask[i, j] and (i, j) not in cleared:
                continue
            free.append((i, j))
    if len(free) < 2:
        raise RuntimeError("Not enough free pixels to place start/target.")
    for _ in range(5000):
        a = free[rng.randrange(len(free))]
        b = free[rng.randrange(len(free))]
        if a == b:
            continue
        if math.hypot(a[0] - b[0], a[1] - b[1]) >= min_dist_px:
            return a, b
    raise RuntimeError("Unable to sample start/target far enough apart.")


def build_scene_from_mask(mask_path, source_scene_path, rng=random):
    """Reconstruct rubble piles + barricades for one mask."""
    # Load source blender scene which has barrier and rubble meshes, we will add them to the scene based on the mask
    print(f"Loading source scene: {source_scene_path}")
    if not os.path.isfile(source_scene_path):
        raise FileNotFoundError(f"Source scene file not found: {source_scene_path}")
    bpy.ops.wm.open_mainfile(filepath=source_scene_path)

    # Extract barrier objects from source scene
    barrier_objects = []
    if "Barriers" in bpy.data.collections:
        barrier_objects.extend(bpy.data.collections["Barriers"].objects)
    print(f"Found {len(barrier_objects)} barrier objects from source scene")

    # Scatter buildings/cars/trees/humans/animals/other outside the inner mask area
    scatter_info = scatter_background_objects(rng=rng)

    mask = load_mask(mask_path)
    black_mask, red_mask = classify_mask(mask)

    red_blobs = get_red_blobs(red_mask)
    cleared_red_pixels = set()
    cleared_barricade_indices = []
    if red_blobs:
        num_cleared = min(len(red_blobs),
                           max(1, math.ceil(len(red_blobs) * CLEARED_BARRICADE_FRACTION)))
        cleared_barricade_indices = rng.sample(range(len(red_blobs)), num_cleared)
        for cleared_idx in cleared_barricade_indices:
            cleared_red_pixels.update(red_blobs[cleared_idx])
    cleared_barricade_indices = set(cleared_barricade_indices)

    start_px, target_px = sample_start_target_free_px(
        black_mask, red_mask, MIN_DIST_BTW_START_TARGET_PX,
        cleared_red_pixels=cleared_red_pixels, rng=rng)

    faded, fade_blobs = gaussian_fade_blob(mask, black_mask)
    rubble_info = []
    for i, blob in enumerate(fade_blobs):
        rubble_info.append(build_rubble_mesh(blob, faded, i, rng=rng, barrier_objects=barrier_objects))

    barricade_info = []
    for i, blob in enumerate(red_blobs):
        if i in cleared_barricade_indices:
            continue
        #choose a random baricade index from the available barrier objects to use for this barricade
        barrier = rng.choice(barrier_objects) if barrier_objects else None
        barricade_info.append(build_barricade(blob, i, barrier=barrier))

    occupancy_grid = rasterize_mask_occupancy(
        black_mask, red_mask, cleared_red_pixels=cleared_red_pixels)

    start_world = mask_px_to_world(start_px[1], start_px[0])
    target_world = mask_px_to_world(target_px[1], target_px[0])

    return dict(
        start_px=start_px,
        target_px=target_px,
        start_world=start_world,
        target_world=target_world,
        rubble_info=rubble_info,
        barricade_info=barricade_info,
        cleared_barricade_index=sorted(cleared_barricade_indices),
        occupancy_grid=occupancy_grid,
        faded=faded,
        scatter_info=scatter_info,
    )


def list_mask_files(mask_dir=MASK_DIR):
    """Return sorted list of PNG mask paths in mask_dir (empty if none)."""
    if not os.path.isdir(mask_dir):
        return []
    files = [f for f in os.listdir(mask_dir)
             if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    return [os.path.join(mask_dir, f) for f in sorted(files)]


def build_merged_masks(mask_files, output_dir, random_order=MASK_MERGE_RANDOM_ORDER, rng=random):
    """Tile every 4 masks into a 2x2, (2*MASK_PX)x(2*MASK_PX) canvas, then
    nearest-neighbor downsample back to MASK_PX x MASK_PX so the result drops
    into the rest of the pipeline like any other mask file. Groups are
    consecutive by default, or drawn from a shuffled order if `random_order`.
    Saves each merged mask as a PNG in `output_dir` and returns their paths.
    """
    if not mask_files:
        return []
    order = list(mask_files)
    if random_order:
        rng.shuffle(order)

    os.makedirs(output_dir, exist_ok=True)
    tile_positions = [(0, 0), (0, MASK_PX), (MASK_PX, 0), (MASK_PX, MASK_PX)]
    num_groups = math.ceil(len(order) / 4)

    merged_paths = []
    for g in range(num_groups):
        group = [order[(g * 4 + k) % len(order)] for k in range(4)]
        canvas = np.zeros((MASK_PX * 2, MASK_PX * 2, 3), dtype=np.float32)
        for (row_off, col_off), path in zip(tile_positions, group):
            canvas[row_off:row_off + MASK_PX, col_off:col_off + MASK_PX, :] = load_mask(path)
        # Stride-2 (nearest-neighbor) downsample keeps colors pure instead of blending them.
        merged = canvas[0::2, 0::2, :]
        out_path = os.path.join(output_dir, f"merged_{g:04d}.png")
        plt.imsave(out_path, merged)
        merged_paths.append(out_path)

    print(f"[mask-merge] built {len(merged_paths)} merged mask(s) from {len(order)} "
          f"source mask(s) (random_order={random_order})")
    return merged_paths
