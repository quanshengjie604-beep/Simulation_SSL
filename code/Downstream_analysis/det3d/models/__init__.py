import importlib
spconv_spec = importlib.util.find_spec("spconv")
found = spconv_spec is not None
from .backbones import *  # noqa: F401,F403
if not found:
    print("No spconv, sparse convolution disabled!")
from .pose_heads import *  # noqa: F401,F403
from .builder import (
    build_backbone,
    build_detector,
    build_head,
    build_loss,
    build_neck,
    build_roi_head,
    build_feat_transform
)
from .detectors import *  # noqa: F401,F403
from .necks import *  # noqa: F401,F403
from .readers import *
from.feat_transforms import *
from .registry import (
    BACKBONES,
    DETECTORS,
    HEADS,
    LOSSES,
    NECKS,
    READERS,
    FEAT_TRANSFORMS
)
try:
    from .second_stage import *
except Exception as exc:
    print(f"Second-stage modules disabled: {exc}")
try:
    from .roi_heads import *
except Exception as exc:
    print(f"ROI head modules disabled: {exc}")

__all__ = [
    "READERS",
    "BACKBONES",
    "NECKS",
    "HEADS",
    "LOSSES",
    "DETECTORS",
    "FEAT_TRANSFORMS",
    "build_feat_transform",
    "build_backbone",
    "build_neck",
    "build_head",
    "build_loss",
    "build_detector",
]
