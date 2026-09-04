

# Not used anymore, run_blainder_scan already handles the rendering of camera frames, but this function is kept for reference.
def render_camera_frames(cam_obj, scene_dir, flight_poses, attitudes, image_dir_name="camera"):
    """Save RGB camera frames aligned with the flight path."""
    image_dir = os.path.join(scene_dir, image_dir_name)
    os.makedirs(image_dir, exist_ok=True)

    set_active_render_camera(cam_obj)

    scene = bpy.context.scene
    old_file_format = scene.render.image_settings.file_format
    old_color_mode = scene.render.image_settings.color_mode
    old_resolution_x = scene.render.resolution_x
    old_resolution_y = scene.render.resolution_y
    old_percentage = scene.render.resolution_percentage
    old_engine = scene.render.engine
    old_background = scene.world.color if scene.world else None

    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.resolution_x = RENDER_RES_X
    scene.render.resolution_y = RENDER_RES_Y
    scene.render.resolution_percentage = 100
    scene.render.engine = 'CYCLES'
    scene.render.film_transparent = False
    scene.cycles.samples = 64
    scene.cycles.use_adaptive_sampling = False

    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.color = (0.8, 0.8, 0.8)

    try:
        for frame_idx, ((x, y, z), (pdev, rdev, ydev)) in enumerate(zip(flight_poses, attitudes), start=1):
            scene.frame_set(frame_idx)
            set_pose(cam_obj, x, y, z, pdev, rdev, ydev)
            scene.render.filepath = os.path.join(image_dir, f"frame_{frame_idx:04d}.png")
            bpy.ops.render.render(write_still=True)
    finally:
        scene.render.image_settings.file_format = old_file_format
        scene.render.image_settings.color_mode = old_color_mode
        scene.render.resolution_x = old_resolution_x
        scene.render.resolution_y = old_resolution_y
        scene.render.resolution_percentage = old_percentage
        scene.render.engine = old_engine
        scene.render.film_transparent = False
        if old_background is not None and scene.world is not None:
            scene.world.color = old_background

    return image_dir

# This function creates a perfectly flat ground plane, and is not being used now
def create_ground_plane(start_cell=None, target_cell=None):
    """Create a single large ground plane and highlight just the start/target cells."""
    base_ground = bproc.object.create_primitive("PLANE", scale=[300.0 / 2, 300.0 / 2, 1])
    base_ground.set_name("ground_base")
    base_ground.set_location([0.0, 0.0, 0.0])
    base_mat = bproc.material.create("ground_base_mat")
    base_mat.set_principled_shader_value("Base Color", [0.3, 0.28, 0.25, 1.0])
    base_mat.set_principled_shader_value("Roughness", 0.9)
    base_ground.replace_materials(base_mat)

    highlight_objects = []
    start_cell = tuple(start_cell) if start_cell is not None else None
    target_cell = tuple(target_cell) if target_cell is not None else None

    for label, cell, color in [
        ("start_cell", start_cell, [0.15, 0.8, 0.25, 1.0]),
        ("target_cell", target_cell, [0.9, 0.18, 0.18, 1.0]),
    ]:
        if cell is None:
            continue
        x, y = cell_center_from_index(*cell)
        cell_obj = bproc.object.create_primitive("PLANE", scale=[CELL_SIZE_M / 2, CELL_SIZE_M / 2, 1])
        cell_obj.set_name(label)
        cell_obj.set_location([x, y, 0.01])
        mat = bproc.material.create(f"{label}_mat")
        mat.set_principled_shader_value("Base Color", color)
        mat.set_principled_shader_value("Roughness", 0.9)
        cell_obj.replace_materials(mat)
        highlight_objects.append(cell_obj)

    return [base_ground] + highlight_objects


# Doesn't really work, ignore for now
def settle_generated_obstacles():
    """Drop generated obstacles using the source barrier physics settings."""
    scene = bpy.context.scene
    obstacles = [obj for obj in bpy.data.objects if obj.get("generated_obstacle")]
    if not obstacles:
        return 0

    barrier_collection = bpy.data.collections.get("Barriers")
    barrier = next(
        (obj for obj in barrier_collection.objects if obj.rigid_body),
        None,
    ) if barrier_collection else None
    if barrier is None:
        raise RuntimeError(
            "No barrier with rigid-body settings found in the 'Barriers' collection."
        )

    for obj in obstacles:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.rigidbody.object_add()
        for prop in barrier.rigid_body.bl_rna.properties:
            if prop.is_readonly or prop.identifier in {"rna_type", "type"}:
                continue
            try:
                setattr(obj.rigid_body, prop.identifier,
                        getattr(barrier.rigid_body, prop.identifier))
            except (AttributeError, TypeError, ValueError):
                pass
        obj.rigid_body.type = barrier.rigid_body.type
        obj.select_set(False)

    scene.frame_start = 1
    scene.frame_end = PHYSICS_SETTLE_FRAMES+1
    scene.frame_current = 1
    scene.frame_set(1)
    scene.frame_set(PHYSICS_SETTLE_FRAMES)
    bpy.context.view_layer.update()

    # for obj in obstacles:
        # obj.rigid_body.type = 'PASSIVE'

    # bpy.context.view_layer.update()
    return len(obstacles)
