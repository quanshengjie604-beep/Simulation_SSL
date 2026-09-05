"""Declarative radar scene definitions built on shared core structures."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

from witwin.core import GeometryBase, Material, Mesh, SMPLBody, SceneBase, Structure
from .timeline import TransformMotion
from .utils.geometry import (
    apply_transform_to_points,
    apply_transform_to_vectors,
    geometry_local_to_world_points,
    geometry_local_to_world_vectors,
    identity_transform,
    rotation_about_origin_transform,
    translation_transform,
)
from .utils.tensor import (
    resolve_scene_device,
    to_faces_array,
    to_tensor3,
    to_vertex_tensor,
)

def _materialize_geometry(geometry: GeometryBase | Mesh, *, device: str) -> tuple[torch.Tensor, np.ndarray]:
    if isinstance(geometry, (Mesh, SMPLBody)):
        vertices, faces = geometry.to_mesh(device=device)
    else:
        vertices, faces = geometry.to_mesh()
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.to(device=device, dtype=torch.float32)
    else:
        vertices = torch.as_tensor(vertices, device=device, dtype=torch.float32)
    faces = to_faces_array(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("geometry.to_mesh() must return vertices with shape (V, 3).")
    return vertices.contiguous(), np.ascontiguousarray(faces)


def _radar_metadata(
    metadata: Mapping[str, Any] | None = None,
    *,
    bsdf: dict[str, Any] | None = None,
    dynamic: bool | None = None,
) -> dict[str, Any]:
    merged = dict(metadata or {})
    if bsdf is not None:
        merged["bsdf"] = dict(bsdf)
    if dynamic is not None:
        merged["dynamic"] = bool(dynamic)
    return merged


def _clone_geometry(geometry: GeometryBase, **changes):
    if isinstance(geometry, SMPLBody):
        return geometry.updated(**changes)
    if isinstance(geometry, Mesh):
        params = {
            "vertices": changes.pop("vertices", geometry._vertices_tensor),
            "faces": changes.pop("faces", geometry.faces),
            "position": changes.pop("position", geometry.position),
            "scale": changes.pop("scale", geometry.scale),
            "rotation": changes.pop("rotation", geometry.rotation),
            "recenter": changes.pop("recenter", geometry.recenter),
            "fill_mode": changes.pop("fill_mode", geometry.fill_mode),
            "surface_thickness": changes.pop("surface_thickness", geometry.surface_thickness),
            "source_path": changes.pop("source_path", geometry.source_path),
            "device": changes.pop("device", geometry.position.device),
        }
        if changes:
            unsupported = ", ".join(sorted(changes))
            raise TypeError(f"Unsupported Mesh updates: {unsupported}")
        return Mesh(**params)

    updated = copy.copy(geometry)
    if "position" in changes:
        updated.position = to_tensor3(changes.pop("position"), device=str(geometry.position.device))
    if "rotation" in changes:
        updated.rotation = torch.as_tensor(changes.pop("rotation"), device=geometry.position.device, dtype=torch.float32)
    if changes:
        unsupported = ", ".join(sorted(changes))
        raise TypeError(f"Unsupported geometry updates for {type(geometry).__name__}: {unsupported}")
    return updated


@dataclass
class CompiledMesh:
    name: str
    vertices: torch.Tensor
    faces: np.ndarray
    eps_r: float
    bsdf: dict[str, Any] | None = None
    dynamic: bool = False
    joints: torch.Tensor | None = None
    source_kind: str = "mesh"


class SceneModule(torch.nn.Module):
    """Module-style radar scenes that materialize to a standard Scene."""

    def to_scene(self) -> "Scene":
        raise NotImplementedError("SceneModule subclasses must implement to_scene().")


class Scene(SceneBase):
    """Radar scene built from shared core structures."""

    DIRTY_NONE = 0
    DIRTY_VERTICES = 1
    DIRTY_FULL = 2

    def __init__(
        self,
        *,
        structures=None,
        structure_motions: Mapping[str, TransformMotion] | None = None,
        metadata=None,
        device: str | None = "cuda",
        verbose: bool = False,
    ):
        resolved_device = resolve_scene_device(device)
        super().__init__(
            structures=structures,
            metadata=metadata,
            device=resolved_device,
            verbose=verbose,
        )
        self._structure_motions: dict[str, TransformMotion] = {}
        self._dirty_level = self.DIRTY_FULL
        self._last_compiled_joints: dict[str, torch.Tensor] = {}
        if structure_motions:
            for name, motion in structure_motions.items():
                if not isinstance(motion, TransformMotion):
                    raise TypeError(f"structure_motions['{name}'] must be a TransformMotion instance.")
                self._validate_structure_motion(str(name), motion)
                self._structure_motions[str(name)] = motion

    @property
    def dirty(self) -> bool:
        return self._dirty_level > self.DIRTY_NONE

    @property
    def dirty_level(self) -> int:
        return self._dirty_level

    @property
    def has_motion(self) -> bool:
        return bool(self._structure_motions)

    def mark_clean(self) -> None:
        self._dirty_level = self.DIRTY_NONE

    def _set_dirty(self, level: int) -> None:
        self._dirty_level = max(self._dirty_level, level)

    def add_structure(self, structure: Structure) -> "Scene":
        if not isinstance(structure, Structure):
            raise TypeError("Radar Scene structures must be witwin.core.Structure instances.")
        if structure.name is None:
            raise ValueError("Radar Scene structures must define a unique name.")
        if not isinstance(structure.geometry, GeometryBase):
            raise TypeError("Radar Scene structures must wrap a GeometryBase geometry.")
        if any(existing.name == structure.name for existing in self.structures):
            raise ValueError(f"Structure '{structure.name}' already exists.")
        self.structures.append(structure)
        self._set_dirty(self.DIRTY_FULL)
        return self

    def add_mesh(
        self,
        *,
        name: str,
        vertices=None,
        faces=None,
        geometry: GeometryBase | Mesh | None = None,
        material: Material | None = None,
        bsdf: dict[str, Any] | None = None,
        dynamic: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Scene":
        if geometry is None:
            if vertices is None or faces is None:
                raise ValueError("add_mesh requires either geometry or both vertices and faces.")
            geometry = Mesh(vertices, faces, position=(0.0, 0.0, 0.0), recenter=False, device=self.device)
        elif vertices is not None or faces is not None:
            raise ValueError("add_mesh accepts either geometry or vertices/faces, not both.")
        return self.add_structure(
            Structure(
                geometry=geometry,
                material=material if material is not None else Material(),
                name=name,
                metadata=_radar_metadata(metadata, bsdf=bsdf, dynamic=True if dynamic else None),
            )
        )

    def add_smpl(
        self,
        *,
        name: str,
        pose,
        shape,
        position=(0.0, 0.0, 0.0),
        gender: str = "male",
        model_root: str | None = None,
        rotation=None,
        material: Material | None = None,
        bsdf: dict[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Scene":
        return self.add_structure(
            Structure(
                geometry=SMPLBody(
                    pose=pose,
                    shape=shape,
                    position=position,
                    gender=gender,
                    model_root=model_root,
                    rotation=rotation,
                    device=self.device,
                ),
                material=material if material is not None else Material(),
                name=name,
                metadata=_radar_metadata(metadata, bsdf=bsdf, dynamic=True),
            )
        )

    def get_structure_motion(self, name: str) -> TransformMotion | None:
        return self._structure_motions.get(name)

    def add_structure_motion(self, name: str, motion: TransformMotion) -> "Scene":
        self._require_structure(name)
        if not isinstance(motion, TransformMotion):
            raise TypeError("motion must be a TransformMotion instance.")
        self._validate_structure_motion(name, motion)
        self._structure_motions[name] = motion
        self._set_dirty(self.DIRTY_VERTICES)
        return self

    def update_structure(self, name: str, **changes) -> "Scene":
        structure_keys = {"geometry", "material", "priority", "enabled", "tags", "metadata", "name"}
        metadata_keys = {"bsdf", "dynamic"}
        for index, structure in enumerate(self.structures):
            if structure.name != name:
                continue

            structure_changes = {key: changes[key] for key in structure_keys & changes.keys()}
            metadata = dict(structure.metadata)
            for key in metadata_keys & changes.keys():
                metadata[key] = changes[key]
            geometry_changes = {
                key: value
                for key, value in changes.items()
                if key not in structure_keys and key not in metadata_keys
            }

            topology_changed = "name" in structure_changes or "geometry" in structure_changes
            updated_geometry = structure_changes.pop("geometry", structure.geometry)
            if geometry_changes:
                updated_geometry = _clone_geometry(updated_geometry, **geometry_changes)
                topology_changed = topology_changed or any(
                    key in geometry_changes for key in ("faces", "gender", "model_root")
                )

            updated = Structure(
                geometry=updated_geometry,
                material=structure_changes.pop("material", structure.material),
                name=structure_changes.pop("name", structure.name),
                priority=structure_changes.pop("priority", structure.priority),
                enabled=structure_changes.pop("enabled", structure.enabled),
                tags=structure_changes.pop("tags", structure.tags),
                metadata=structure_changes.pop("metadata", metadata),
            )
            if structure_changes:
                unsupported = ", ".join(sorted(structure_changes))
                raise TypeError(f"Unsupported Structure updates: {unsupported}")
            self.structures[index] = updated
            if structure.name != updated.name:
                motion = self._structure_motions.pop(structure.name, None)
                if motion is not None:
                    self._structure_motions[updated.name] = motion
                for child_name, child_motion in list(self._structure_motions.items()):
                    if child_motion.parent == structure.name:
                        self._structure_motions[child_name] = TransformMotion(
                            offset=child_motion.offset,
                            velocity=child_motion.velocity,
                            axis=child_motion.axis,
                            angular_velocity=child_motion.angular_velocity,
                            angle=child_motion.angle,
                            origin=child_motion.origin,
                            space=child_motion.space,
                            t_ref=child_motion.t_ref,
                            parent=updated.name,
                        )
            self._set_dirty(self.DIRTY_FULL if topology_changed else self.DIRTY_VERTICES)
            return self
        raise KeyError(f"Structure '{name}' not found.")

    def remove(self, name: str) -> "Scene":
        for index, structure in enumerate(self.structures):
            if structure.name == name:
                self.structures.pop(index)
                self._set_dirty(self.DIRTY_FULL)
                self._last_compiled_joints.pop(name, None)
                self._structure_motions.pop(name, None)
                return self
        raise KeyError(f"Structure '{name}' not found.")

    def get_joints(self, name: str, *, time: float | None = None) -> np.ndarray:
        if time is not None or name not in self._last_compiled_joints:
            self.compile_renderables(time=time)
        if name not in self._last_compiled_joints:
            raise KeyError(f"SMPL body '{name}' not found.")
        return self._last_compiled_joints[name].detach().cpu().numpy()

    def _require_structure(self, name: str) -> Structure:
        for structure in self.structures:
            if structure.name == name:
                return structure
        raise KeyError(f"Structure '{name}' not found.")

    def _validate_structure_motion(self, name: str, motion: TransformMotion) -> None:
        self._require_structure(name)
        if motion.parent is not None:
            if motion.parent == name:
                raise ValueError("Structure motion parent cannot reference itself.")
            self._require_structure(motion.parent)
            current = motion.parent
            visited = {name}
            while current is not None:
                if current in visited:
                    raise ValueError("Structure motion graph must be acyclic.")
                visited.add(current)
                parent_motion = self._structure_motions.get(current)
                current = None if parent_motion is None else parent_motion.parent

    def _resolve_structure_transform(
        self,
        name: str,
        *,
        time: float,
        cache: dict[str, torch.Tensor],
        active: tuple[str, ...],
    ) -> torch.Tensor:
        if name in cache:
            return cache[name]
        if name in active:
            raise ValueError("Structure motion graph must be acyclic.")

        structure = self._require_structure(name)
        parent_transform = identity_transform(device=self.device, dtype=torch.float32)
        motion = self._structure_motions.get(name)
        if motion is not None and motion.parent is not None:
            parent_transform = self._resolve_structure_transform(
                motion.parent,
                time=time,
                cache=cache,
                active=active + (name,),
            )

        if motion is None:
            cache[name] = parent_transform
            return parent_transform

        self_transform = self._build_structure_motion_transform(
            structure.geometry,
            motion,
            time=time,
            parent_transform=parent_transform,
        )
        total_transform = self_transform @ parent_transform
        cache[name] = total_transform
        return total_transform

    def _build_structure_motion_transform(
        self,
        geometry: GeometryBase,
        motion: TransformMotion,
        *,
        time: float,
        parent_transform: torch.Tensor,
    ) -> torch.Tensor:
        dtype = torch.float32
        device = self.device
        transform = identity_transform(device=device, dtype=dtype)
        translation_delta = torch.zeros(3, device=device, dtype=dtype)

        translation_delta = self._resolve_translation_delta(
            geometry,
            motion,
            time=time,
            parent_transform=parent_transform,
        )
        if bool(translation_delta.abs().sum().item() > 0.0):
            transform = translation_transform(translation_delta, device=device, dtype=dtype)

        if motion.axis is None:
            return transform

        rotation_transform = self._resolve_rotation_transform(
            geometry,
            motion,
            translation_delta=translation_delta,
            parent_transform=parent_transform,
            device=device,
            dtype=dtype,
            time=time,
        )
        return rotation_transform @ transform

    def _resolve_translation_delta(
        self,
        geometry: GeometryBase,
        motion: TransformMotion,
        *,
        time: float,
        parent_transform: torch.Tensor,
    ) -> torch.Tensor:
        dtype = torch.float32
        offset = motion.offset.to(device=self.device, dtype=dtype)
        velocity = motion.velocity.to(device=self.device, dtype=dtype)
        t_ref = motion.t_ref.to(device=self.device, dtype=dtype)
        delta = offset + velocity * (torch.tensor(time, device=self.device, dtype=dtype) - t_ref)
        if motion.space == "world":
            return delta
        world_delta = geometry_local_to_world_vectors(geometry, delta.unsqueeze(0), device=self.device, dtype=dtype)[0]
        return apply_transform_to_vectors(world_delta.unsqueeze(0), parent_transform)[0]

    def _resolve_rotation_transform(
        self,
        geometry: GeometryBase,
        motion: TransformMotion,
        *,
        translation_delta: torch.Tensor,
        parent_transform: torch.Tensor,
        device: str,
        dtype: torch.dtype,
        time: float,
    ) -> torch.Tensor:
        axis = motion.axis.to(device=device, dtype=dtype)
        if motion.space == "local":
            axis_world = geometry_local_to_world_vectors(geometry, axis.unsqueeze(0), device=device, dtype=dtype)[0]
            origin_local = torch.zeros(1, 3, device=device, dtype=dtype)
            if motion.origin is not None:
                origin_local = motion.origin.to(device=device, dtype=dtype).view(1, 3)
            origin_world = geometry_local_to_world_points(geometry, origin_local, device=device, dtype=dtype)[0]
            axis_world = apply_transform_to_vectors(axis_world.unsqueeze(0), parent_transform)[0]
            origin_world = apply_transform_to_points(origin_world.unsqueeze(0), parent_transform)[0]
        else:
            axis_world = axis
            if motion.origin is None:
                origin_world = geometry.position.to(device=device, dtype=dtype)
            else:
                origin_world = motion.origin.to(device=device, dtype=dtype)

        origin_world = origin_world + translation_delta
        angle = motion.angle.to(device=device, dtype=dtype)
        angular_velocity = motion.angular_velocity.to(device=device, dtype=dtype)
        t_ref = motion.t_ref.to(device=device, dtype=dtype)
        theta = angle + angular_velocity * (torch.tensor(time, device=device, dtype=dtype) - t_ref)
        return rotation_about_origin_transform(origin_world, axis_world, theta, device=device, dtype=dtype)

    def compile_renderables(self, *, time: float | None = None) -> dict[str, CompiledMesh]:
        renderables: dict[str, CompiledMesh] = {}
        joints: dict[str, torch.Tensor] = {}
        transforms: dict[str, torch.Tensor] = {}
        time_value = 0.0 if time is None else float(time)
        for structure in self.structures:
            if not structure.enabled:
                continue
            motion_transform = self._resolve_structure_transform(
                structure.name,
                time=time_value,
                cache=transforms,
                active=(),
            )
            compiled = self._compile_structure(structure, motion_transform=motion_transform)
            renderables[compiled.name] = compiled
            if compiled.joints is not None:
                joints[compiled.name] = compiled.joints
        self._last_compiled_joints = joints
        return renderables

    def _compile_structure(self, structure: Structure, *, motion_transform: torch.Tensor) -> CompiledMesh:
        geometry = structure.geometry
        joints = None
        source_kind = "mesh"
        if isinstance(geometry, SMPLBody):
            vertices, faces, joints = geometry._evaluate(device=self.device)
            source_kind = "smpl"
        else:
            vertices, faces = _materialize_geometry(geometry, device=self.device)
        vertex_tensor = to_vertex_tensor(vertices, device=self.device).contiguous()
        vertex_tensor = apply_transform_to_points(vertex_tensor, motion_transform).contiguous()
        if joints is not None:
            joints = apply_transform_to_points(
                to_vertex_tensor(joints, device=self.device).contiguous(),
                motion_transform,
            ).contiguous()
        dynamic = bool(
            structure.metadata.get("dynamic", False)
            or vertex_tensor.requires_grad
            or joints is not None
            or structure.name in self._structure_motions
        )
        eps_r = float(structure.material.evaluate_static().eps_r)
        return CompiledMesh(
            name=structure.name,
            vertices=vertex_tensor,
            faces=to_faces_array(faces),
            eps_r=eps_r,
            bsdf=structure.metadata.get("bsdf"),
            dynamic=dynamic,
            joints=joints,
            source_kind=source_kind,
        )
