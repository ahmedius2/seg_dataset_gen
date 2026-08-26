import blenderproc as bproc
import bpy
import numpy as np
import os
import random
import math
import json

"""
generate_dataset.py
───────────────────
BlenderProc synthetic aerial dataset generator.

Simulates a fixed-wing drone flying over your 50×50 m obstacle course.
Per render it:
  • picks a random ground texture from your Poly Haven downloads
  • randomises UV tiling, sun direction, sun intensity
  • places the camera at a random altitude / slight off-nadir tilt
  • outputs one RGB PNG  →  output/images/XXXX.png
  • outputs one binary mask PNG  →  output/masks/XXXX.png
      (pixel = 0   → traversable ground)
      (pixel = 255 → obstacle)

Usage
-----
    blenderproc run generate_dataset.py

Dependencies (all come with BlenderProc):
    blenderproc, numpy, imageio

Edit every value in the CONFIG block before running.
"""


# ═══════════════════════════════════════════════════════════════════
#  CONFIG  ← edit these paths / values before running
# ═══════════════════════════════════════════════════════════════════
ROOT_PATH = "/home/dho/work/ileri_otonom/seg_dataset_gen"

# Absolute path to your master Blender scene file
BLEND_FILE = f"{ROOT_PATH}/scene_bl4/scene_bl4.blend"

# Root folder that contains the Poly Haven texture sub-folders
# (each sub-folder must have a 'textures/' child with the map files)
TEXTURES_ROOT = f"{ROOT_PATH}/scene_bl4/ground_textures"

# Where to write images/ and masks/
OUTPUT_DIR = f"{ROOT_PATH}/output"

# How many image–mask pairs to generate
NUM_IMAGES = 20

# Render resolution (W, H) — 640×640 is the YOLOv8 default
IMAGE_RES = (640, 640)

# ── Scene object names ──────────────────────────────────────────────
# Exact name of the ground plane mesh in your Blender scene.
# Open Blender → click the ground → look at the top of the Properties
# panel (the orange square) to find the exact name.
GROUND_OBJ_NAME = "Ground"

# Optional: list substrings that appear in OBSTACLE object names.
# The match is case-insensitive.  Leave as [] to treat every mesh
# that is NOT the ground plane as an obstacle automatically.
OBSTACLE_KEYWORDS = ["obs_"]   # e.g. ["Rock", "Barrier", "Plank", "Drone"]

# ── Drone / camera parameters ───────────────────────────────────────
DRONE_ALT_MIN  = 35.0   # minimum flight altitude above ground (metres)
DRONE_ALT_MAX  = 60.0   # maximum flight altitude

# Horizontal field of view in degrees.
# 70° ≈ a typical drone survey camera (DJI Zenmuse X7 style).
CAM_HFOV_DEG = 70.0

# Maximum off-nadir tilt in degrees.
# Fixed-wing gimbal stabilisation is not perfect → small tilt is realistic.
MAX_TILT_DEG = 8.0

# ── Inner obstacle zone (world-space XY, metres) ────────────────────
# Set these to match the actual centre of your 50×50 m obstacle area.
INNER_CX   = 0.0
INNER_CY   = 0.0
INNER_HALF = 25.0   # half of 50 m

# ── Segmentation class IDs ──────────────────────────────────────────
CLASS_TRAVERSABLE = 0   # ground   → mask pixel = 0   (black)
CLASS_OBSTACLE    = 1   # obstacle → mask pixel = 255 (white)

# ── Render quality ──────────────────────────────────────────────────
# Lower sample count = faster renders but noisier images.
# 64–128 is a good balance for training data.
RENDER_SAMPLES = 64

# ═══════════════════════════════════════════════════════════════════

# ────
#  Helper: hide child objects (collision meshes)
# ────

def hide_child_objects(loaded_objs: list) -> list:
    """
    Hide any mesh object that is a child of another object (these are
    collision meshes and should not be rendered).  Returns the list of
    objects that remain visible.

    BlenderProc renders every object in the scene, so we must set
    hide_render on the underlying Blender object — merely dropping the
    object from the returned list is not enough.
    """
    visible = []
    for obj in loaded_objs:
        blender_obj = obj.blender_obj
        if blender_obj is not None and blender_obj.parent is not None:
            blender_obj.hide_render = True
            blender_obj.hide_viewport = True
            print(f"[INFO] Hidden child object: '{obj.get_name()}'")
        else:
            visible.append(obj)
    return visible

# ───────────────────────────────────────────────────────────────────
#  Helper: discover Poly Haven texture sets
# ───────────────────────────────────────────────────────────────────

def discover_texture_sets(root: str) -> list:
    """
    Walk *root* looking for sub-folders that contain a 'textures/'
    directory with Poly Haven PBR map files.

    Returns a list of dicts, each with keys:
        name   folder name (for logging)
        diff   path to diffuse/albedo map   (required)
        rough  path to roughness map        (optional)
        nor    path to normal map (GL)      (optional)
        disp   path to displacement map     (optional)
    """
    sets = []
    if not os.path.isdir(root):
        raise FileNotFoundError(f"TEXTURES_ROOT not found: {root}")

    for entry in sorted(os.scandir(root), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        tex_dir = os.path.join(entry.path, "textures")
        if not os.path.isdir(tex_dir):
            continue

        s = {"name": entry.name}
        for fname in os.listdir(tex_dir):
            fl  = fname.lower()
            fp  = os.path.join(tex_dir, fname)
            if "_diff_" in fl:
                s["diff"] = fp
            elif "_disp_" in fl:
                s["disp"] = fp
            elif "_nor_gl_" in fl or ("_nor_" in fl and "_gl_" not in fl):
                s["nor"] = fp
            elif "_rough_" in fl:
                s["rough"] = fp

        if "diff" in s:          # need at least the albedo map
            sets.append(s)

    return sets


# ───────────────────────────────────────────────────────────────────
#  Helper: apply a PBR texture set to the ground plane
# ───────────────────────────────────────────────────────────────────

def apply_ground_texture(obj_name: str, tex_set: dict) -> None:
    """
    Rebuild the Principled BSDF material node tree on *obj_name*
    using the maps in *tex_set*.  UV scale is randomised each call
    to add domain-randomisation variety.
    """
    obj = bpy.data.objects.get(obj_name)
    if obj is None:
        print(f"[WARN] Object '{obj_name}' not found in scene — "
              "check GROUND_OBJ_NAME in the CONFIG block.")
        return

    # ── ensure a material slot exists ──────────────────────────────
    if obj.data.materials:
        mat = obj.data.materials[0]
    else:
        mat = bpy.data.materials.new(name="_gen_GroundMat")
        obj.data.materials.append(mat)

    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # ── node skeleton ───────────────────────────────────────────────
    out  = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    # UV coordinate + mapping (controls tiling scale)
    tc      = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    links.new(tc.outputs["UV"], mapping.inputs["Vector"])

    # Randomise UV tiling scale: larger scale → texture repeats more
    scale = random.uniform(2.0, 7.0)
    mapping.inputs["Scale"].default_value = (scale, scale, 1.0)

    # Slight random UV rotation for additional variety
    angle = random.uniform(0, math.pi * 2)
    mapping.inputs["Rotation"].default_value = (0.0, 0.0, angle)

    def _add_image_node(path: str, colorspace: str = "Non-Color"):
        """Load image (cached) and wire UV mapping to it."""
        n   = nodes.new("ShaderNodeTexImage")
        img = bpy.data.images.load(path, check_existing=True)
        img.colorspace_settings.name = colorspace
        n.image = img
        links.new(mapping.outputs["Vector"], n.inputs["Vector"])
        return n

    # ── PBR map connections ─────────────────────────────────────────
    if "diff" in tex_set:
        n = _add_image_node(tex_set["diff"], "sRGB")
        links.new(n.outputs["Color"], bsdf.inputs["Base Color"])

    if "rough" in tex_set:
        n = _add_image_node(tex_set["rough"])
        links.new(n.outputs["Color"], bsdf.inputs["Roughness"])

    if "nor" in tex_set:
        n  = _add_image_node(tex_set["nor"])
        nm = nodes.new("ShaderNodeNormalMap")
        links.new(n.outputs["Color"],  nm.inputs["Color"])
        links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])

    if "disp" in tex_set:
        n  = _add_image_node(tex_set["disp"])
        dn = nodes.new("ShaderNodeDisplacement")
        dn.inputs["Scale"].default_value = 0.008   # subtle bump
        links.new(n.outputs["Color"],       dn.inputs["Height"])
        links.new(dn.outputs["Displacement"], out.inputs["Displacement"])


# ───────────────────────────────────────────────────────────────────
#  Helper: randomise sun lighting
# ───────────────────────────────────────────────────────────────────

# We keep a single sun object and update its properties each iteration
# rather than deleting/recreating (faster).
_sun_obj = None

def setup_or_randomise_sun() -> None:
    """Create sun on first call; randomise properties on every call."""
    global _sun_obj

    if _sun_obj is None:
        # Remove any lights that came from the loaded blend file
        for o in list(bpy.data.objects):
            if o.type == "LIGHT":
                bpy.data.objects.remove(o, do_unlink=True)

        light = bproc.types.Light()
        light.set_type("SUN")
        light.set_location([0, 0, 100])
        _sun_obj = light

    # Randomise energy and direction each call
    _sun_obj.set_energy(random.uniform(0.8, 2.5))

    elevation = math.radians(random.uniform(25, 85))   # sun altitude
    azimuth   = math.radians(random.uniform(0, 360))   # sun compass

    # Blender sun rotation: X=elevation from horizon, Z=azimuth
    _sun_obj.set_rotation_euler([
        math.pi / 2.0 - elevation,
        0.0,
        azimuth
    ])


# ───────────────────────────────────────────────────────────────────
#  Helper: tag all mesh objects with category_id
# ───────────────────────────────────────────────────────────────────

def assign_category_ids(loaded_objs: list) -> None:
    """
    Set the 'category_id' custom property on every mesh object.
    BlenderProc's segmentation renderer reads this value per pixel.

    Tagging rules:
      • Object whose name == GROUND_OBJ_NAME  →  CLASS_TRAVERSABLE (0)
      • Everything else                        →  CLASS_OBSTACLE    (1)
        (or use OBSTACLE_KEYWORDS to be selective — see CONFIG)
    """
    ground_found = False
    for obj in loaded_objs:
        name = obj.get_name()
        print(name)

        if name == GROUND_OBJ_NAME:
            obj.set_cp("category_id", CLASS_TRAVERSABLE)
            ground_found = True
            print(f"[INFO] '{name}' → CLASS_TRAVERSABLE ({CLASS_TRAVERSABLE})")
        elif not OBSTACLE_KEYWORDS:
            # treat every non-ground mesh as an obstacle
            obj.set_cp("category_id", CLASS_OBSTACLE)
        else:
            is_obs = any(kw.lower() in name.lower() for kw in OBSTACLE_KEYWORDS)
            cat    = CLASS_OBSTACLE if is_obs else CLASS_TRAVERSABLE
            obj.set_cp("category_id", cat)

    if not ground_found:
        print(f"[WARN] No object named '{GROUND_OBJ_NAME}' was found. "
              "All meshes will be labelled as obstacles.  "
              "Fix GROUND_OBJ_NAME in the CONFIG block.")


# ───────────────────────────────────────────────────────────────────
#  Helper: sample a drone camera pose
# ───────────────────────────────────────────────────────────────────

def sample_drone_camera_pose() -> tuple:
    """
    Return (location_xyz, euler_rotation_xyz) that mimic a fixed-wing
    drone camera flying over the inner obstacle zone.

    • Altitude drawn uniformly from [DRONE_ALT_MIN, DRONE_ALT_MAX].
    • Camera position offset randomly from the inner-area centre.
    • Camera looks toward the inner area with at most MAX_TILT_DEG
      off nadir (simulates gimbal stabilisation imperfection).
    • A small random in-plane rotation models slight roll.
    """
    alt = random.uniform(DRONE_ALT_MIN, DRONE_ALT_MAX)

    # Camera XY position — drift up to 35 % of inner half-size
    drift_max = INNER_HALF * 0.35
    cam_x = INNER_CX + random.uniform(-drift_max, drift_max)
    cam_y = INNER_CY + random.uniform(-drift_max, drift_max)
    location = np.array([cam_x, cam_y, alt])

    # Target point (where the camera looks) near the inner-area centre
    tgt_drift = INNER_HALF * 0.20
    target = np.array([
        INNER_CX + random.uniform(-tgt_drift, tgt_drift),
        INNER_CY + random.uniform(-tgt_drift, tgt_drift),
        0.0
    ])

    # Enforce MAX_TILT_DEG from nadir
    forward     = target - location
    forward_len = np.linalg.norm(forward)
    nadir       = np.array([0.0, 0.0, -1.0])
    cos_angle   = np.clip(np.dot(forward / forward_len, nadir), -1.0, 1.0)
    off_nadir   = math.degrees(math.acos(cos_angle))

    if off_nadir > MAX_TILT_DEG:
        # Clamp: project target back to the allowed cone
        horiz_dist = alt * math.tan(math.radians(MAX_TILT_DEG))
        dx = target[0] - cam_x
        dy = target[1] - cam_y
        angle_to_target = math.atan2(dy, dx)
        target = np.array([
            cam_x + horiz_dist * math.cos(angle_to_target),
            cam_y + horiz_dist * math.sin(angle_to_target),
            0.0
        ])

    # Small in-plane roll (radians) — fixed-wing flight attitude
    inplane_roll = random.uniform(-6.0, 6.0) * math.pi / 180.0

    rotation = bproc.camera.rotation_from_forward_vec(
        target - location,
        inplane_rot=inplane_roll
    )

    return location, rotation


# ───────────────────────────────────────────────────────────────────
#  Helper: save one image + mask pair
# ───────────────────────────────────────────────────────────────────

def save_pair(idx: int,
              rgb: np.ndarray,
              segmap: np.ndarray,
              img_dir: str,
              mask_dir: str,
              tex_name: str) -> None:
    """Write RGB PNG and binary mask PNG for sample *idx*."""
    img_path  = os.path.join(img_dir,  f"{idx:04d}.png")
    mask_path = os.path.join(mask_dir, f"{idx:04d}.png")

    # ── RGB ─────────────────────────────────────────────────────────
    rgb_u8 = rgb.astype(np.uint8)

    # BlenderProc returns float [0,1] in some versions — handle both
    if rgb.dtype in (np.float32, np.float64) and rgb.max() <= 1.0:
        rgb_u8 = (rgb * 255).clip(0, 255).astype(np.uint8)

    # Use bpy to save the image (avoids imageio version issues)
    tmp_img = bpy.data.images.new(
        f"_tmp_rgb_{idx}", width=rgb_u8.shape[1], height=rgb_u8.shape[0]
    )
    # bpy stores pixels as RGBA float row-major from bottom-left;
    # flip vertically and add alpha channel
    rgb_flipped = rgb_u8[::-1, :, :]          # flip Y (bpy is bottom-up)
    rgba_flat   = np.ones(
        (rgb_flipped.shape[0], rgb_flipped.shape[1], 4), dtype=np.float32
    )
    rgba_flat[:, :, :3] = rgb_flipped.astype(np.float32) / 255.0
    tmp_img.pixels = rgba_flat.flatten().tolist()
    tmp_img.filepath_raw = img_path
    tmp_img.file_format  = "PNG"
    tmp_img.save()
    bpy.data.images.remove(tmp_img)

    # ── Binary mask ─────────────────────────────────────────────────
    # segmap dtype is int; obstacle pixels → 255, ground pixels → 0
    binary = (segmap == CLASS_OBSTACLE).astype(np.uint8) * 255

    mask_img = bpy.data.images.new(
        f"_tmp_mask_{idx}", width=binary.shape[1], height=binary.shape[0]
    )
    # Grayscale stored as RGBA in bpy
    binary_flipped = binary[::-1, :]
    rgba_mask = np.zeros(
        (binary_flipped.shape[0], binary_flipped.shape[1], 4), dtype=np.float32
    )
    v = binary_flipped.astype(np.float32) / 255.0
    rgba_mask[:, :, 0] = v
    rgba_mask[:, :, 1] = v
    rgba_mask[:, :, 2] = v
    rgba_mask[:, :, 3] = 1.0
    mask_img.pixels = rgba_mask.flatten().tolist()
    mask_img.filepath_raw = mask_path
    mask_img.file_format  = "PNG"
    mask_img.save()
    bpy.data.images.remove(mask_img)

    print(f"  [{idx+1:>4}/{NUM_IMAGES}] img={os.path.basename(img_path)}"
          f"  mask={os.path.basename(mask_path)}"
          f"  tex={tex_name}")


# ───────────────────────────────────────────────────────────────────
#  Main
# ───────────────────────────────────────────────────────────────────

def main():

    # 1. Discover texture sets ──────────────────────────────────────
    texture_sets = discover_texture_sets(TEXTURES_ROOT)
    if not texture_sets:
        raise RuntimeError(
            f"No valid texture sets found under '{TEXTURES_ROOT}'.\n"
            "Each sub-folder must have a 'textures/' child with "
            "Poly Haven map files (diff, rough, nor_gl, disp)."
        )
    print(f"\n[INFO] Found {len(texture_sets)} texture set(s):")
    for s in texture_sets:
        print(f"       • {s['name']}")

    # 2. Output directories ─────────────────────────────────────────
    img_dir  = os.path.join(OUTPUT_DIR, "images")
    mask_dir = os.path.join(OUTPUT_DIR, "masks")
    os.makedirs(img_dir,  exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    # 3. Initialise BlenderProc ─────────────────────────────────────
    bproc.init()

    # 4. Load scene ─────────────────────────────────────────────────
    print(f"\n[INFO] Loading blend file: {BLEND_FILE}")
    loaded_objs = bproc.loader.load_blend(BLEND_FILE)
    print(f"[INFO] Loaded {len(loaded_objs)} mesh object(s).")
    
    # 4b. Hide child objects (collision meshes) ────
    loaded_objs = hide_child_objects(loaded_objs)
    print(f"[INFO] {len(loaded_objs)} mesh object(s) remain after hiding children.")

    # 5. Tag objects with category IDs ──────────────────────────────
    assign_category_ids(loaded_objs)

    # 6. Camera intrinsics (constant across all renders) ────────────
    #bproc.camera.set_resolution(*IMAGE_RES)
    #bproc.camera.set_intrinsics_from_fov(math.radians(CAM_HFOV_DEG))
    
    # 6. Camera intrinsics (constant across all renders) ────
    bproc.camera.set_resolution(*IMAGE_RES)

    # Build the pinhole K matrix from the horizontal FOV.
    w, h = IMAGE_RES
    f = (w / 2.0) / math.tan(math.radians(CAM_HFOV_DEG) / 2.0)
    K = np.array([
        [f, 0.0, w / 2.0],
        [0.0, f, h / 2.0],
        [0.0, 0.0, 1.0],
    ])
    bproc.camera.set_intrinsics_from_K_matrix(K, w, h)

    # 7. Renderer settings ──────────────────────────────────────────
    bproc.renderer.set_max_amount_of_samples(RENDER_SAMPLES)
    bproc.renderer.enable_segmentation_output(map_by=["category_id"])

    # 8. Generation loop ────────────────────────────────────────────
    print(f"\n[INFO] Generating {NUM_IMAGES} image–mask pairs …\n")

    for i in range(NUM_IMAGES):
        # Clear keyframes left over from the previous iteration
        bproc.utility.reset_keyframes()

        tex_set = random.choice(texture_sets)

        # Randomise ground texture
        apply_ground_texture(GROUND_OBJ_NAME, tex_set)

        # Randomise sun
        setup_or_randomise_sun()

        # Sample drone camera pose (adds a single keyframe)
        location, rotation = sample_drone_camera_pose()
        cam2world = bproc.math.build_transformation_mat(location, rotation)
        bproc.camera.add_camera_pose(cam2world)

        data = bproc.renderer.render()

        # data["colors"][0]              → RGB   numpy array (H, W, 3)
        # data["category_id_segmaps"][0] → segmap numpy array (H, W) int
        save_pair(
            i,
            data["colors"][0],
            data["category_id_segmaps"][0],
            img_dir,
            mask_dir,
            tex_set["name"],
        )

    # 9. Write metadata JSON ─────────────────────────────────────────
    meta = {
        "num_images"    : NUM_IMAGES,
        "resolution"    : IMAGE_RES,
        "camera_hfov"   : CAM_HFOV_DEG,
        "altitude_range": [DRONE_ALT_MIN, DRONE_ALT_MAX],
        "max_tilt_deg"  : MAX_TILT_DEG,
        "classes"       : {
            str(CLASS_TRAVERSABLE): "traversable",
            str(CLASS_OBSTACLE)   : "obstacle",
        },
        "mask_encoding" : "grayscale — 0=traversable, 255=obstacle",
        "texture_sets"  : [s["name"] for s in texture_sets],
        "images_dir"    : img_dir,
        "masks_dir"     : mask_dir,
    }
    meta_path = os.path.join(OUTPUT_DIR, "dataset_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[DONE] {NUM_IMAGES} pairs written to: {OUTPUT_DIR}")
    print(f"       metadata: {meta_path}")
    print(
        "\nNext step: run  python convert_to_yolo.py  "
        "to produce YOLO-seg label files."
    )


if __name__ == "__main__":
    main()
