"""Shared differentiable geometry package."""

from .base import GeometryBase
from .mesh import Mesh
from .primitives import Box, Cone, Cylinder, Ellipsoid, HollowBox, Prism, Pyramid, Sphere, Torus
from .smpl import SMPLBody

Geometry = Box | Sphere | Cylinder | Cone | Ellipsoid | Pyramid | Prism | Torus | HollowBox | Mesh | SMPLBody

__all__ = [
    "GeometryBase",
    "Box",
    "Sphere",
    "Cylinder",
    "Cone",
    "Ellipsoid",
    "Pyramid",
    "Prism",
    "Torus",
    "HollowBox",
    "Mesh",
    "SMPLBody",
    "Geometry",
]
