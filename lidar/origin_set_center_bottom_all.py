import bpy
import mathutils

def set_origin_to_center_bottom_safe():
    # Save current selection and active object
    original_selected = list(bpy.context.selected_objects)
    original_active = bpy.context.active_object
    saved_cursor_loc = list(bpy.context.scene.cursor.location)

    # Process all mesh objects
    for obj in list(bpy.context.scene.objects):
        if obj.type != 'MESH' or not obj.data.vertices:
            continue

        # Deselect all, select current object, set active
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        # Step 1: Safely apply rotation using Blender's built-in transform operator
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

        # Step 2: Compute bounding box corners in WORLD space
        world_matrix = obj.matrix_world
        world_bbox = [world_matrix @ mathutils.Vector(corner) for corner in obj.bound_box]

        min_x = min(v.x for v in world_bbox)
        max_x = max(v.x for v in world_bbox)
        min_y = min(v.y for v in world_bbox)
        max_y = max(v.y for v in world_bbox)
        min_z = min(v.z for v in world_bbox)

        # Center-Bottom coordinate in World Space
        center_bottom_world = mathutils.Vector((
            (min_x + max_x) / 2.0,
            (min_y + max_y) / 2.0,
            min_z
        ))

        # Step 3: Set 3D Cursor to target location and shift object origin to cursor
        bpy.context.scene.cursor.location = center_bottom_world
        bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')

    # Restore initial scene state
    bpy.context.scene.cursor.location = saved_cursor_loc
    bpy.ops.object.select_all(action='DESELECT')
    for o in original_selected:
        o.select_set(True)
    bpy.context.view_layer.objects.active = original_active

# Run script
set_origin_to_center_bottom_safe()
