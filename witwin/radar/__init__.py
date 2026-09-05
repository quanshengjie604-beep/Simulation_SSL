"""Public radar API."""

from .radar import Radar, RadarConfig, quantize_complex_signal
from .solvers import Solver
from .trace import TraceResult, Tracer
from .material import fresnel
from .scene import Scene, SceneModule
from .timeline import Timeline, TransformMotion
from .types import DetectorType, MotionSampling, SamplingMode, SolverBackend
from witwin.core import (
    Box,
    Cone,
    Cylinder,
    Ellipsoid,
    Geometry,
    GeometryBase,
    HollowBox,
    Material,
    Mesh,
    Prism,
    Pyramid,
    SMPLBody,
    Sphere,
    Structure,
    Torus,
)

__all__ = [
    'Radar',
    'RadarConfig',
    'quantize_complex_signal',
    'Solver',
    'Tracer',
    'TraceResult',
    'SolverBackend',
    'DetectorType',
    'MotionSampling',
    'SamplingMode',
    'Scene',
    'SceneModule',
    'SMPLBody',
    'Timeline',
    'TransformMotion',
    'fresnel',
    'Material',
    'Structure',
    'GeometryBase',
    'Geometry',
    'Mesh',
    'Box',
    'Sphere',
    'Cylinder',
    'Cone',
    'Ellipsoid',
    'Pyramid',
    'Prism',
    'Torus',
    'HollowBox',
]
