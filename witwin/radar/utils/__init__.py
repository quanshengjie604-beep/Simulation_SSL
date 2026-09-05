"""Internal helper utilities shared across the radar package."""

from .tensor import (
    real_dtype,
    resolve_scene_device,
    to_faces_array,
    to_tensor3,
    to_vertex_tensor,
)
from .vector import (
    normalize_rows,
    optional_vec3_tensor,
    scalar_tensor,
    vec3_tensor,
)
from .geometry import (
    apply_transform_to_points,
    apply_transform_to_vectors,
    axis_angle_rotation,
    geometry_local_to_world_points,
    geometry_local_to_world_vectors,
    identity_transform,
    rotation_about_origin_transform,
    translation_transform,
)
from .antenna import (
    DEFAULT_DIPOLE_ANGLES_DEG,
    DEFAULT_DIPOLE_VALUES,
    evaluate_antenna_pattern_vectors,
    evaluate_antenna_pattern_xy,
    half_wave_dipole_power_cut,
    interp1d_zero_outside,
    interp2d_zero_outside,
)

__all__ = [
    "real_dtype",
    "resolve_scene_device",
    "to_faces_array",
    "to_tensor3",
    "to_vertex_tensor",
    "normalize_rows",
    "optional_vec3_tensor",
    "scalar_tensor",
    "vec3_tensor",
    "apply_transform_to_points",
    "apply_transform_to_vectors",
    "axis_angle_rotation",
    "geometry_local_to_world_points",
    "geometry_local_to_world_vectors",
    "identity_transform",
    "rotation_about_origin_transform",
    "translation_transform",
    "DEFAULT_DIPOLE_ANGLES_DEG",
    "DEFAULT_DIPOLE_VALUES",
    "half_wave_dipole_power_cut",
    "interp1d_zero_outside",
    "interp2d_zero_outside",
    "evaluate_antenna_pattern_xy",
    "evaluate_antenna_pattern_vectors",
]
