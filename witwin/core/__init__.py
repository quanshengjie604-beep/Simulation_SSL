"""Witwin Core - Shared geometry, materials, and scene utilities."""

__version__ = "0.0.1"

from .material import (
    FrequencyMaterialSample,
    Material,
    MaterialCapabilities,
    MaterialSpec,
    StaticMaterialSample,
    Structure,
)
from .geometry import (
    Box,
    Cone,
    Cylinder,
    Ellipsoid,
    Geometry,
    GeometryBase,
    HollowBox,
    Mesh,
    Prism,
    Pyramid,
    SMPLBody,
    Sphere,
    Torus,
)
from .math import (
    quat_from_euler,
    quat_identity,
    quat_multiply,
    quat_to_rotation_matrix,
    quat_to_rotation_matrix_np,
)
from .scene import SceneBase
from .scene_to_mitsuba import (
    MitsubaRenderable,
    MitsubaSceneHandle,
    build_mitsuba_scene,
    create_mitsuba_mesh,
    update_mitsuba_scene_vertices,
)

__all__ = [
    "GeometryBase",
    "Box", "Sphere", "Cylinder", "Cone", "Ellipsoid",
    "Pyramid", "Prism", "Torus", "HollowBox",
    "Material",
    "MaterialCapabilities",
    "MaterialSpec",
    "Mesh",
    "SMPLBody",
    "FrequencyMaterialSample",
    "Geometry",
    "SceneBase",
    "StaticMaterialSample",
    "Structure",
    "MitsubaRenderable",
    "MitsubaSceneHandle",
    "quat_from_euler",
    "quat_identity",
    "quat_multiply",
    "quat_to_rotation_matrix",
    "quat_to_rotation_matrix_np",
    "create_mitsuba_mesh",
    "build_mitsuba_scene",
    "update_mitsuba_scene_vertices",
]
