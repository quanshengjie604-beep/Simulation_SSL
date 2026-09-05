"""Triangle mesh representation with OBJ loading and topology analysis."""

from __future__ import annotations

from os import PathLike
from pathlib import Path

import numpy as np
import torch

from ..math import quat_to_rotation_matrix
from .base import GeometryBase, _as_vec3
from .mesh_sdf import (
    _build_triangle_bvh,
    _triangle_bvh_to_device,
    triangle_mesh_signed_distance_static_bvh,
    triangle_mesh_smooth_signed_distance,
    triangle_mesh_unsigned_distance_static_bvh,
    triangle_mesh_unsigned_distance,
)


def _vertices_array(vertices) -> np.ndarray:
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().cpu().numpy()
    array = np.asarray(vertices, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] == 0:
        raise ValueError("vertices must have shape (N, 3) with N > 0.")
    return np.ascontiguousarray(array)


def _vertices_tensor(vertices, *, device=None) -> torch.Tensor:
    if isinstance(vertices, torch.Tensor):
        tensor = vertices.to(device=device, dtype=torch.float32)
    else:
        tensor = torch.tensor(vertices, device=device, dtype=torch.float32)
    if tensor.ndim != 2 or tensor.shape[1] != 3 or tensor.shape[0] == 0:
        raise ValueError("vertices must have shape (N, 3) with N > 0.")
    return tensor.contiguous()


def _faces_array(faces, *, vertex_count: int) -> np.ndarray:
    if isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu().numpy()
    array = np.asarray(faces, dtype=np.int32)
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] == 0:
        raise ValueError("faces must have shape (M, 3) with M > 0.")
    if np.any(array < 0) or np.any(array >= vertex_count):
        raise ValueError("faces contains vertex indices outside the valid range.")
    return np.ascontiguousarray(array)


def _faces_tensor(faces, *, device=None) -> torch.Tensor:
    if isinstance(faces, torch.Tensor):
        tensor = faces.to(device=device, dtype=torch.int64)
    else:
        tensor = torch.as_tensor(faces, device=device, dtype=torch.int64)
    if tensor.ndim != 2 or tensor.shape[1] != 3 or tensor.shape[0] == 0:
        raise ValueError("faces must have shape (M, 3) with M > 0.")
    return tensor.contiguous()


def _triangulate_obj_face(indices: list[int]) -> list[tuple[int, int, int]]:
    if len(indices) < 3:
        return []
    triangles: list[tuple[int, int, int]] = []
    anchor = indices[0]
    for index in range(1, len(indices) - 1):
        tri = (anchor, indices[index], indices[index + 1])
        if len({tri[0], tri[1], tri[2]}) == 3:
            triangles.append(tri)
    return triangles


def _load_obj(path: str | PathLike[str]) -> tuple[np.ndarray, np.ndarray]:
    obj_path = Path(path).expanduser().resolve()
    if not obj_path.exists():
        raise FileNotFoundError(f"OBJ file not found: {obj_path}")

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    with obj_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                tokens = line.split()
                if len(tokens) < 4:
                    raise ValueError(f"Invalid vertex record at {obj_path}:{line_no}")
                vertices.append((float(tokens[1]), float(tokens[2]), float(tokens[3])))
                continue
            if not line.startswith("f "):
                continue
            tokens = line.split()[1:]
            if len(tokens) < 3:
                raise ValueError(f"Invalid face record at {obj_path}:{line_no}")
            face_indices: list[int] = []
            for token in tokens:
                vertex_token = token.split("/")[0]
                if not vertex_token:
                    raise ValueError(f"Unsupported OBJ face token at {obj_path}:{line_no}")
                raw_index = int(vertex_token)
                vertex_index = raw_index - 1 if raw_index > 0 else len(vertices) + raw_index
                if vertex_index < 0 or vertex_index >= len(vertices):
                    raise ValueError(f"OBJ face index out of range at {obj_path}:{line_no}")
                face_indices.append(vertex_index)
            faces.extend(_triangulate_obj_face(face_indices))

    if not vertices:
        raise ValueError(f"OBJ file does not contain any vertices: {obj_path}")
    if not faces:
        raise ValueError(f"OBJ file does not contain any triangular faces: {obj_path}")
    return _vertices_array(vertices), _faces_array(faces, vertex_count=len(vertices))


def _mesh_topology_stats(vertices: np.ndarray, faces: np.ndarray) -> tuple[int, int, int, int, float]:
    edge_counts: dict[tuple[int, int], int] = {}
    edge_orientation_balance: dict[tuple[int, int], int] = {}

    bbox_extent = vertices.max(axis=0) - vertices.min(axis=0)
    bbox_diagonal = float(np.linalg.norm(bbox_extent))
    area_epsilon = max((bbox_diagonal * bbox_diagonal) * 1.0e-12, np.finfo(np.float32).eps)

    degenerate_faces = 0
    for tri in faces:
        a_idx, b_idx, c_idx = (int(tri[0]), int(tri[1]), int(tri[2]))
        if len({a_idx, b_idx, c_idx}) < 3:
            degenerate_faces += 1
            continue

        a = vertices[a_idx]
        b = vertices[b_idx]
        c = vertices[c_idx]
        twice_area = np.linalg.norm(np.cross(b - a, c - a))
        if twice_area <= area_epsilon:
            degenerate_faces += 1

        edges = ((a_idx, b_idx), (b_idx, c_idx), (c_idx, a_idx))
        for start, end in edges:
            edge = (start, end) if start < end else (end, start)
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
            orientation = 1 if start == edge[0] else -1
            edge_orientation_balance[edge] = edge_orientation_balance.get(edge, 0) + orientation

    boundary_edges = sum(1 for count in edge_counts.values() if count == 1)
    non_manifold_edges = sum(1 for count in edge_counts.values() if count > 2)
    inconsistent_edge_orientations = sum(
        1
        for edge, count in edge_counts.items()
        if count == 2 and edge_orientation_balance.get(edge, 0) != 0
    )
    signed_volume = float(
        np.einsum(
            "ij,ij->i",
            vertices[faces[:, 0]],
            np.cross(vertices[faces[:, 1]], vertices[faces[:, 2]]),
        ).sum()
        / 6.0
    )
    return (
        boundary_edges,
        non_manifold_edges,
        degenerate_faces,
        inconsistent_edge_orientations,
        signed_volume,
    )


class Mesh(GeometryBase):
    """Explicit triangle mesh geometry."""

    kind = "mesh"

    def __init__(
        self,
        vertices,
        faces,
        *,
        position=(0.0, 0.0, 0.0),
        scale=1.0,
        rotation=None,
        recenter: bool = True,
        fill_mode: str = "auto",
        surface_thickness: float | None = None,
        source_path: str | PathLike[str] | None = None,
        device=None,
    ):
        super().__init__(position=position, rotation=rotation, device=device)

        normalized_fill_mode = str(fill_mode).lower()
        if normalized_fill_mode not in {"auto", "solid", "surface"}:
            raise ValueError("fill_mode must be 'auto', 'solid', or 'surface'.")
        if surface_thickness is not None and float(surface_thickness) <= 0.0:
            raise ValueError("surface_thickness must be positive when provided.")

        vertices_array = _vertices_array(vertices)
        faces_array = _faces_array(faces, vertex_count=len(vertices_array))
        vertices_tensor = _vertices_tensor(vertices, device=self.position.device)
        faces_tensor = _faces_tensor(faces, device=self.position.device)
        (
            boundary_edges,
            non_manifold_edges,
            degenerate_faces,
            inconsistent_edge_orientations,
            signed_volume,
        ) = _mesh_topology_stats(vertices_array, faces_array)
        bbox_extent = vertices_array.max(axis=0) - vertices_array.min(axis=0)
        bbox_diagonal = float(np.linalg.norm(bbox_extent))
        volume_epsilon = max((bbox_diagonal ** 3) * 1.0e-9, np.finfo(np.float32).eps)

        self.scale: torch.Tensor = _as_vec3(scale, device=device)
        self._vertices_tensor: torch.Tensor = vertices_tensor
        self._faces_tensor: torch.Tensor = faces_tensor
        self.recenter: bool = bool(recenter)
        self.fill_mode: str = normalized_fill_mode
        self.surface_thickness: float | None = None if surface_thickness is None else float(surface_thickness)
        self.source_path: str | None = None if source_path is None else str(Path(source_path).expanduser().resolve())
        self.boundary_edge_count: int = int(boundary_edges)
        self.non_manifold_edge_count: int = int(non_manifold_edges)
        self.degenerate_face_count: int = int(degenerate_faces)
        self.inconsistent_edge_orientation_count: int = int(inconsistent_edge_orientations)
        self.signed_volume: float = float(signed_volume)
        self.enclosed_volume: float = float(abs(signed_volume))
        self.is_watertight: bool = bool(
            boundary_edges == 0
            and non_manifold_edges == 0
            and degenerate_faces == 0
            and inconsistent_edge_orientations == 0
            and abs(signed_volume) > volume_epsilon
        )
        self._voxel_cache: dict = {}
        self._sdf_cache: dict = {}

    @property
    def vertices(self) -> np.ndarray:
        return self._vertices_tensor.detach().cpu().numpy()

    @property
    def faces(self) -> np.ndarray:
        return self._faces_tensor.detach().cpu().numpy()

    def _local_vertices_tensor(self, *, device=None, dtype=None) -> torch.Tensor:
        vertices = self._vertices_tensor
        if self.recenter:
            bounds_min = vertices.min(dim=0).values
            bounds_max = vertices.max(dim=0).values
            vertices = vertices - 0.5 * (bounds_min + bounds_max)
        if device is not None or dtype is not None:
            vertices = vertices.to(
                device=vertices.device if device is None else device,
                dtype=vertices.dtype if dtype is None else dtype,
            )
        return vertices

    def world_vertices_tensor(self, *, device=None, dtype=None) -> torch.Tensor:
        local_vertices = self._local_vertices_tensor(device=device, dtype=dtype)
        scale = self.scale.to(device=local_vertices.device, dtype=local_vertices.dtype)
        rotation = quat_to_rotation_matrix(self.rotation.to(device=local_vertices.device, dtype=local_vertices.dtype))
        position = self.position.to(device=local_vertices.device, dtype=local_vertices.dtype)
        return (local_vertices * scale) @ rotation.T + position

    @property
    def world_vertices(self) -> np.ndarray:
        return self.world_vertices_tensor().detach().cpu().numpy()

    @property
    def bounds_world(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        world_vertices = self.world_vertices_tensor().detach()
        world_min = world_vertices.min(dim=0).values
        world_max = world_vertices.max(dim=0).values
        return (
            (float(world_min[0]), float(world_max[0])),
            (float(world_min[1]), float(world_max[1])),
            (float(world_min[2]), float(world_max[2])),
        )

    @classmethod
    def from_obj(
        cls,
        path: str | PathLike[str],
        *,
        position=(0.0, 0.0, 0.0),
        scale=1.0,
        rotation=None,
        recenter: bool = True,
        fill_mode: str = "auto",
        surface_thickness: float | None = None,
        device=None,
    ) -> "Mesh":
        vertices, faces = _load_obj(path)
        return cls(
            vertices,
            faces,
            position=position,
            scale=scale,
            rotation=rotation,
            recenter=recenter,
            fill_mode=fill_mode,
            surface_thickness=surface_thickness,
            source_path=path,
            device=device,
        )

    @property
    def vertex_count(self) -> int:
        return int(self._vertices_tensor.shape[0])

    @property
    def face_count(self) -> int:
        return int(self._faces_tensor.shape[0])

    def has_trainable_geometry(self) -> bool:
        return any(
            tensor.requires_grad
            for tensor in (self.position, self.rotation, self.scale, self._vertices_tensor)
        )

    @staticmethod
    def _tensor_state_key(tensor: torch.Tensor) -> tuple:
        return (
            str(tensor.device),
            str(tensor.dtype),
            tuple(tensor.shape),
            tensor.data_ptr(),
            getattr(tensor, "_version", None),
            bool(tensor.requires_grad),
        )

    def geometry_state_key(self) -> tuple:
        return (
            self.recenter,
            self.fill_mode,
            self.surface_thickness,
            bool(self.is_watertight),
            self.degenerate_face_count,
            self.inconsistent_edge_orientation_count,
            self._tensor_state_key(self.position),
            self._tensor_state_key(self.rotation),
            self._tensor_state_key(self.scale),
            self._tensor_state_key(self._vertices_tensor),
            self._tensor_state_key(self._faces_tensor),
        )

    def clear_voxel_cache(self) -> None:
        self._voxel_cache.clear()

    def clear_sdf_cache(self) -> None:
        self._sdf_cache.clear()

    def _sdf_query_data(self, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict | None]:
        use_cache = not self.has_trainable_geometry()
        cache_key = (str(device), str(dtype))
        geometry_key = self.geometry_state_key()
        if use_cache:
            cached = self._sdf_cache.get(cache_key)
            if cached is not None and cached["geometry_key"] == geometry_key:
                return cached["vertices"], cached["faces"], cached["triangles"], cached.get("bvh")
        else:
            self._sdf_cache.clear()

        vertices = self.world_vertices_tensor(device=device, dtype=dtype)
        faces = self._faces_tensor.to(device=device)
        triangles = vertices[faces].contiguous()
        bvh = None
        if use_cache and device.type == "cuda" and dtype == torch.float32:
            bvh = _triangle_bvh_to_device(_build_triangle_bvh(triangles), device=device)

        if use_cache:
            self._sdf_cache[cache_key] = {
                "geometry_key": geometry_key,
                "vertices": vertices,
                "faces": faces,
                "triangles": triangles,
                "bvh": bvh,
            }
        return vertices, faces, triangles, bvh

    def _surface_band(self, xx: torch.Tensor, yy: torch.Tensor, zz: torch.Tensor) -> torch.Tensor:
        spacings = []
        for coord, dim in ((xx, 0), (yy, 1), (zz, 2)):
            if coord.ndim > dim and coord.shape[dim] > 1:
                delta = torch.diff(coord, dim=dim).abs()
                positive = delta[delta > 0]
                if positive.numel() > 0:
                    spacings.append(positive.min())
        if self.surface_thickness is not None:
            return torch.as_tensor(self.surface_thickness, device=xx.device, dtype=xx.dtype)
        if not spacings:
            return torch.as_tensor(1.0e-3, device=xx.device, dtype=xx.dtype)
        spacing_vec = torch.stack([spacing.to(device=xx.device, dtype=xx.dtype) for spacing in spacings])
        return 0.25 * torch.min(spacing_vec)

    def signed_distance(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
        if x.ndim == y.ndim == z.ndim == 1:
            xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
        else:
            xx, yy, zz = x, y, z

        points = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
        vertices, faces, triangles, bvh = self._sdf_query_data(device=points.device, dtype=points.dtype)
        use_static_bvh = (
            bvh is not None
            and not self.has_trainable_geometry()
            and points.device.type == "cuda"
            and points.dtype == torch.float32
            and not points.requires_grad
        )
        if self.fill_mode == "surface" or (self.fill_mode == "auto" and not self.is_watertight):
            if use_static_bvh:
                unsigned_distance = triangle_mesh_unsigned_distance_static_bvh(points, triangles, bvh)
            else:
                unsigned_distance = triangle_mesh_unsigned_distance(points, vertices, faces, _triangles=triangles)
            signed_distance = unsigned_distance - self._surface_band(xx, yy, zz)
        else:
            if use_static_bvh:
                signed_distance = triangle_mesh_signed_distance_static_bvh(points, triangles, bvh)
            else:
                signed_distance = triangle_mesh_smooth_signed_distance(points, vertices, faces, _triangles=triangles)
        return signed_distance.reshape(xx.shape)

    def to_mesh(self, segments=16, *, device=None):
        del segments
        target_device = self.position.device if device is None else torch.device(device)
        vertices = self.world_vertices_tensor(device=target_device, dtype=torch.float32)
        return vertices.contiguous(), _faces_tensor(self.faces, device=target_device)
