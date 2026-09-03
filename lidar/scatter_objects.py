"""Scatter background decoration objects over the outer scene area.

Copies objects from the source scene's Buildings/Cars/Trees/Humans/Animals/Other/Debris
collections and spreads them across the PLANE_SIZE area using Poisson-disk-style
rejection sampling, keeping the inner AREA_SIZE_M mask-driven region clear and
avoiding collisions between all placed footprints.
"""

import math
import random

import bpy

from dataset_config import (
    AREA_SIZE_M,
    ANIMAL_DENSITY,
    BUILDING_DENSITY,
    CAR_DENSITY,
    DEBRIS_DENSITY,
    HUMAN_DENSITY,
    OTHER_DENSITY,
    PLANE_SIZE,
    SCATTER_MARGIN_M,
    SCATTER_MAX_ATTEMPTS,
    TREE_DENSITY,
)

# Fixed placement order: buildings, then cars, then trees, humans, animals, other, debris.
SCATTER_COLLECTIONS = [
    ("Buildings", BUILDING_DENSITY),
    ("Debris", DEBRIS_DENSITY),
    ("Cars", CAR_DENSITY),
    ("Trees", TREE_DENSITY),
    ("Humans", HUMAN_DENSITY),
    ("Animals", ANIMAL_DENSITY),
    ("Other", OTHER_DENSITY),
]


def build_copy_list(objects, density):
    """Duplicate `objects` per `density` (e.g. 1.5 -> full list + first half again)."""
    if not objects or density <= 0:
        return []
    full_copies = int(math.floor(density))
    frac = density - full_copies
    copy_list = list(objects) * full_copies
    extra_count = int(round(frac * len(objects)))
    copy_list += list(objects[:extra_count])
    return copy_list


def object_footprint_radius(obj):
    """Approximate placement radius from the object's XY bounding box."""
    dims = obj.dimensions
    radius = max(dims.x, dims.y) / 2.0
    return radius if radius > 1e-3 else 0.5


def sample_scatter_point(placements, radius, plane_half, exclude_half, margin, rng, max_attempts):
    """Rejection-sample a point outside the inner square and clear of existing placements."""
    for _ in range(max_attempts):
        x = rng.uniform(-plane_half, plane_half)
        y = rng.uniform(-plane_half, plane_half)
        if abs(x) <= exclude_half and abs(y) <= exclude_half:
            continue
        if all((x - px) ** 2 + (y - py) ** 2 >= (radius + pr + margin) ** 2
               for px, py, pr in placements):
            return x, y
    return None


def remove_source_objects(objects):
    """Delete the original (out-of-area) source objects once they've been copied."""
    for obj in objects:
        mesh = obj.data if obj.type == 'MESH' else None
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def scatter_collection(collection_name, density, placements, rng=random,
                        plane_size=PLANE_SIZE, exclude_size=AREA_SIZE_M,
                        margin=SCATTER_MARGIN_M, max_attempts=SCATTER_MAX_ATTEMPTS):
    """Copy and spread objects from one source collection over the outer area."""
    if collection_name not in bpy.data.collections:
        return []
    source_objects = list(bpy.data.collections[collection_name].objects)
    copy_list = build_copy_list(source_objects, density)

    plane_half = plane_size / 2.0
    exclude_half = exclude_size / 2.0
    placed = []
    for i, src_obj in enumerate(copy_list):
        radius = object_footprint_radius(src_obj)
        point = sample_scatter_point(placements, radius, plane_half, exclude_half, margin, rng, max_attempts)
        if point is None:
            print(f"[scatter] {collection_name}: could not place instance {i} after {max_attempts} attempts, skipping")
            continue
        x, y = point
        new_data = src_obj.data.copy()
        new_obj = bpy.data.objects.new(f"{collection_name.lower()}_{i}", new_data)
        bpy.context.scene.collection.objects.link(new_obj)
        new_obj.location = (x, y, src_obj.location.z)
        new_obj.rotation_euler = src_obj.rotation_euler.copy()
        new_obj.scale = src_obj.scale.copy()
        placements.append((x, y, radius))
        placed.append(dict(name=new_obj.name, location=[x, y, new_obj.location.z]))

    remove_source_objects(source_objects)

    print(f"[scatter] {collection_name}: placed {len(placed)}/{len(copy_list)} instance(s) (density={density})")
    return placed


def scatter_background_objects(rng=random):
    """Scatter all background collections, in order, over the outer area."""
    placements = []
    scatter_info = {}
    for collection_name, density in SCATTER_COLLECTIONS:
        scatter_info[collection_name.lower()] = scatter_collection(
            collection_name, density, placements, rng=rng, plane_size=PLANE_SIZE - 150.0)
    return scatter_info
