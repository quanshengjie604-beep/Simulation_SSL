"""Public API enums and protocols for radar.

The enums subclass ``StrEnum`` so users can pass either a member or a raw
string; IDEs still complete members and invalid values raise immediately.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Callable, Protocol, TypeAlias, runtime_checkable

import torch


class SolverBackend(StrEnum):
    PYTORCH = "pytorch"
    SLANG = "slang"
    DIRICHLET = "dirichlet"


class DetectorType(StrEnum):
    CFAR = "cfar"
    TOPK = "topk"


class SamplingMode(StrEnum):
    PIXEL = "pixel"
    TRIANGLE = "triangle"


class MotionSampling(StrEnum):
    PER_FRAME = "per_frame"
    PER_CHIRP = "per_chirp"


@runtime_checkable
class TraceSample(Protocol):
    """Structural type for the payload an ``InterpolatorFn`` returns.

    ``TraceResult`` satisfies this protocol; user-defined closures
    may return any object with the same attributes.
    """

    points: torch.Tensor
    intensities: torch.Tensor
    entry_points: torch.Tensor
    fixed_path_lengths: torch.Tensor
    depths: torch.Tensor
    normals: torch.Tensor | None


InterpolatorFn: TypeAlias = Callable[[float], TraceSample]
"""``(time_seconds) -> TraceSample``. Used by ``Radar.frame`` / ``Radar.mimo``."""
