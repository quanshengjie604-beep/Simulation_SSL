"""Shared helpers for lowering mesh-based scenes into Mitsuba."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch


def _load_mitsuba(variant: str):
    import drjit as dr
    import mitsuba as mi

    mi.set_variant(variant)
    return mi, dr


def _as_vertices_array(vertices) -> np.ndarray:
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().cpu().numpy()
    array = np.asarray(vertices, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("Mitsuba vertices must have shape (V, 3).")
    return np.ascontiguousarray(array)


def _as_faces_array(faces) -> np.ndarray:
    if isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu().numpy()
    array = np.asarray(faces, dtype=np.uint32)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("Mitsuba faces must have shape (F, 3).")
    return np.ascontiguousarray(array)


def _materialize_plugin(mi, plugin):
    if isinstance(plugin, Mapping):
        return mi.load_dict(dict(plugin))
    return plugin


@dataclass(frozen=True)
class MitsubaRenderable:
    name: str
    vertices: torch.Tensor | np.ndarray
    faces: torch.Tensor | np.ndarray
    bsdf: dict[str, Any] | None = None


@dataclass(frozen=True)
class MitsubaSceneHandle:
    scene: Any
    params: Any


def create_mitsuba_mesh(
    renderable: MitsubaRenderable,
    *,
    default_bsdf: dict[str, Any] | None = None,
    variant: str = "cuda_ad_rgb",
    has_vertex_texcoords: bool = False,
):
    mi, dr = _load_mitsuba(variant)
    vertices = _as_vertices_array(renderable.vertices)
    faces = _as_faces_array(renderable.faces)
    bsdf = default_bsdf or {
        "type": "diffuse",
        "reflectance": {"type": "rgb", "value": (0.8, 0.8, 0.8)},
    }

    mesh = mi.Mesh(
        renderable.name,
        vertex_count=vertices.shape[0],
        face_count=faces.shape[0],
        has_vertex_texcoords=has_vertex_texcoords,
    )
    mesh_params = mi.traverse(mesh)
    mesh_params["vertex_positions"] = dr.ravel(mi.TensorXf(vertices))
    mesh_params["faces"] = dr.ravel(mi.TensorXu(faces))
    mesh_params.update()
    mesh.set_bsdf(mi.load_dict(renderable.bsdf or bsdf))
    return mesh


def build_mitsuba_scene(
    *,
    sensor,
    renderables: Mapping[str, MitsubaRenderable | Any],
    integrator: dict[str, Any] | Any | None = None,
    default_bsdf: dict[str, Any] | None = None,
    variant: str = "cuda_ad_rgb",
) -> MitsubaSceneHandle:
    mi, _ = _load_mitsuba(variant)
    scene_dict = {
        "type": "scene",
        "integrator": _materialize_plugin(mi, integrator or {"type": "direct"}),
        "sensor": _materialize_plugin(mi, sensor),
        "default_bsdf": default_bsdf
        or {
            "type": "diffuse",
            "reflectance": {"type": "rgb", "value": (0.8, 0.8, 0.8)},
        },
    }

    for name, renderable in renderables.items():
        if not isinstance(renderable, MitsubaRenderable):
            renderable = MitsubaRenderable(
                name=name,
                vertices=renderable.vertices,
                faces=renderable.faces,
                bsdf=getattr(renderable, "bsdf", None),
            )
        scene_dict[name] = create_mitsuba_mesh(
            renderable,
            default_bsdf=scene_dict["default_bsdf"],
            variant=variant,
        )

    scene = mi.load_dict(scene_dict)
    return MitsubaSceneHandle(scene=scene, params=mi.traverse(scene))


def update_mitsuba_scene_vertices(
    params,
    renderables: Mapping[str, MitsubaRenderable | Any],
    *,
    variant: str = "cuda_ad_rgb",
) -> None:
    mi, dr = _load_mitsuba(variant)
    for name, renderable in renderables.items():
        key = f"{name}.vertex_positions"
        if key not in params:
            continue
        vertices = renderable.vertices if not isinstance(renderable, MitsubaRenderable) else renderable.vertices
        params[key] = dr.ravel(mi.TensorXf(_as_vertices_array(vertices)))
    params.update()
