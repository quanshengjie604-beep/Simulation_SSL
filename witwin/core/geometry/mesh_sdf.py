"""Torch-native and Slang-backed triangle-mesh distance helpers."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


_MESH_SDF_MODULE_CACHE: dict[str, Any] = {}
_MESH_SDF_BVH_LEAF_SIZE = 8
_MESH_SDF_BVH_MIN_TRIANGLES = 1024


def _dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.sum(a * b, dim=-1)


def _safe_signed_denom(value: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.where(value >= 0.0, value.clamp_min(eps), value.clamp_max(-eps))


def _ensure_current_env_on_path() -> None:
    scripts_dir = os.path.join(os.path.dirname(sys.executable), "Scripts")
    if not os.path.isdir(scripts_dir):
        return
    current_path = os.environ.get("PATH", "")
    path_entries = current_path.split(os.pathsep) if current_path else []
    if scripts_dir not in path_entries:
        os.environ["PATH"] = scripts_dir + os.pathsep + current_path


def _get_mesh_sdf_module():
    try:
        import slangtorch
    except ImportError:
        return None

    _ensure_current_env_on_path()
    slang_path = str(Path(__file__).resolve().parents[1] / "mesh_sdf.slang")
    module = _MESH_SDF_MODULE_CACHE.get(slang_path)
    if module is None:
        module = slangtorch.loadModule(slang_path)
        _MESH_SDF_MODULE_CACHE[slang_path] = module
    return module


def _slang_mesh_sdf_available() -> bool:
    return _get_mesh_sdf_module() is not None


def _should_use_slang(points: torch.Tensor, vertices: torch.Tensor) -> bool:
    return (
        points.device.type == "cuda"
        and vertices.device.type == "cuda"
        and points.dtype == torch.float32
        and vertices.dtype == torch.float32
        and _slang_mesh_sdf_available()
    )


def _launch_shape_1d(length: int, block_size: int) -> tuple[int, int, int]:
    return ((length + block_size - 1) // block_size, 1, 1)


def _indexed_triangles(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    if faces.device != vertices.device:
        faces = faces.to(device=vertices.device)
    return vertices[faces].contiguous()


def _build_triangle_bvh(
    triangles: torch.Tensor,
    *,
    leaf_size: int = _MESH_SDF_BVH_LEAF_SIZE,
    min_triangles: int = _MESH_SDF_BVH_MIN_TRIANGLES,
) -> dict[str, np.ndarray] | None:
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError("triangles must have shape (T, 3, 3).")
    triangle_count = int(triangles.shape[0])
    if triangle_count < int(min_triangles):
        return None

    triangles_np = triangles.detach().to(device="cpu", dtype=torch.float32).numpy()
    tri_min = triangles_np.min(axis=1)
    tri_max = triangles_np.max(axis=1)
    centroids = 0.5 * (tri_min + tri_max)

    bbox_min_list: list[np.ndarray] = []
    bbox_max_list: list[np.ndarray] = []
    left_list: list[int] = []
    right_list: list[int] = []
    start_list: list[int] = []
    count_list: list[int] = []
    ordered_triangles: list[int] = []

    def build(indices: np.ndarray) -> int:
        node_index = len(bbox_min_list)
        bbox_min_list.append(tri_min[indices].min(axis=0).astype(np.float32))
        bbox_max_list.append(tri_max[indices].max(axis=0).astype(np.float32))
        left_list.append(-1)
        right_list.append(-1)
        start_list.append(-1)
        count_list.append(0)

        if len(indices) <= int(leaf_size):
            start = len(ordered_triangles)
            ordered_triangles.extend(int(index) for index in indices.tolist())
            start_list[node_index] = start
            count_list[node_index] = len(indices)
            return node_index

        centroid_extent = centroids[indices].max(axis=0) - centroids[indices].min(axis=0)
        split_axis = int(np.argmax(centroid_extent))
        order = np.argsort(centroids[indices, split_axis], kind="mergesort")
        ordered_indices = indices[order]
        split = len(ordered_indices) // 2

        if split <= 0 or split >= len(ordered_indices):
            start = len(ordered_triangles)
            ordered_triangles.extend(int(index) for index in ordered_indices.tolist())
            start_list[node_index] = start
            count_list[node_index] = len(ordered_indices)
            return node_index

        left_child = build(ordered_indices[:split])
        right_child = build(ordered_indices[split:])
        left_list[node_index] = left_child
        right_list[node_index] = right_child
        return node_index

    build(np.arange(triangle_count, dtype=np.int32))
    return {
        "bbox_min": np.asarray(bbox_min_list, dtype=np.float32),
        "bbox_max": np.asarray(bbox_max_list, dtype=np.float32),
        "left": np.asarray(left_list, dtype=np.int32),
        "right": np.asarray(right_list, dtype=np.int32),
        "start": np.asarray(start_list, dtype=np.int32),
        "count": np.asarray(count_list, dtype=np.int32),
        "triangle_indices": np.asarray(ordered_triangles, dtype=np.int32),
    }


def _triangle_bvh_to_device(bvh: dict[str, np.ndarray] | None, *, device: torch.device) -> dict[str, torch.Tensor] | None:
    if bvh is None:
        return None
    return {
        "bbox_min": torch.from_numpy(bvh["bbox_min"]).to(device=device, dtype=torch.float32).contiguous(),
        "bbox_max": torch.from_numpy(bvh["bbox_max"]).to(device=device, dtype=torch.float32).contiguous(),
        "left": torch.from_numpy(bvh["left"]).to(device=device, dtype=torch.int32).contiguous(),
        "right": torch.from_numpy(bvh["right"]).to(device=device, dtype=torch.int32).contiguous(),
        "start": torch.from_numpy(bvh["start"]).to(device=device, dtype=torch.int32).contiguous(),
        "count": torch.from_numpy(bvh["count"]).to(device=device, dtype=torch.int32).contiguous(),
        "triangle_indices": torch.from_numpy(bvh["triangle_indices"]).to(device=device, dtype=torch.int32).contiguous(),
    }


def _signed_distance_from_unsigned_and_winding(
    unsigned_distance: torch.Tensor,
    winding_angle: torch.Tensor,
    winding_beta: float,
) -> torch.Tensor:
    angle_threshold = unsigned_distance.new_tensor(2.0 * math.pi)
    beta = unsigned_distance.new_tensor(float(winding_beta))
    sign_factor = -torch.tanh((torch.abs(winding_angle) - angle_threshold) / beta)
    return unsigned_distance * sign_factor


def _point_triangle_squared_distance(points: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(points.dtype).eps
    ab = b - a
    ac = c - a
    bc = c - b

    ap = points - a
    bp = points - b
    cp = points - c

    d1 = _dot(ab, ap)
    d2 = _dot(ac, ap)
    d3 = _dot(ab, bp)
    d4 = _dot(ac, bp)
    d5 = _dot(ab, cp)
    d6 = _dot(ac, cp)

    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    denom = (va + vb + vc).clamp_min(eps)
    v = vb / denom
    w = vc / denom
    face_proj = a + ab * v.unsqueeze(-1) + ac * w.unsqueeze(-1)
    dist2 = _dot(points - face_proj, points - face_proj)

    vertex_a = (d1 <= 0.0) & (d2 <= 0.0)
    vertex_b = (d3 >= 0.0) & (d4 <= d3)
    vertex_c = (d6 >= 0.0) & (d5 <= d6)

    edge_ab = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    edge_ac = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    edge_bc = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)

    ab_t = (d1 / _safe_signed_denom(d1 - d3, eps)).clamp(0.0, 1.0)
    ac_t = (d2 / _safe_signed_denom(d2 - d6, eps)).clamp(0.0, 1.0)
    bc_t = ((d4 - d3) / ((d4 - d3) + (d5 - d6)).clamp_min(eps)).clamp(0.0, 1.0)

    proj_ab = a + ab * ab_t.unsqueeze(-1)
    proj_ac = a + ac * ac_t.unsqueeze(-1)
    proj_bc = b + bc * bc_t.unsqueeze(-1)

    dist2 = torch.where(vertex_a, _dot(ap, ap), dist2)
    dist2 = torch.where(vertex_b, _dot(bp, bp), dist2)
    dist2 = torch.where(vertex_c, _dot(cp, cp), dist2)
    dist2 = torch.where(edge_ab, _dot(points - proj_ab, points - proj_ab), dist2)
    dist2 = torch.where(edge_ac, _dot(points - proj_ac, points - proj_ac), dist2)
    dist2 = torch.where(edge_bc, _dot(points - proj_bc, points - proj_bc), dist2)
    return dist2


def _triangle_mesh_unsigned_distance_torch_from_triangles(
    points: torch.Tensor,
    triangles: torch.Tensor,
    *,
    point_chunk_size: int | None = None,
    triangle_chunk_size: int | None = None,
) -> torch.Tensor:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3).")

    point_chunk = 8192 if point_chunk_size is None else int(point_chunk_size)
    triangle_chunk = 256 if triangle_chunk_size is None else int(triangle_chunk_size)
    outputs: list[torch.Tensor] = []
    smooth_eps = 1.0e-12

    for point_start in range(0, points.shape[0], point_chunk):
        point_stop = min(points.shape[0], point_start + point_chunk)
        point_block = points[point_start:point_stop]
        point_block_expanded = point_block[:, None, :]
        min_dist2 = None

        for triangle_start in range(0, triangles.shape[0], triangle_chunk):
            triangle_stop = min(triangles.shape[0], triangle_start + triangle_chunk)
            triangle_block = triangles[triangle_start:triangle_stop]
            a = triangle_block[None, :, 0, :]
            b = triangle_block[None, :, 1, :]
            c = triangle_block[None, :, 2, :]
            dist2 = _point_triangle_squared_distance(point_block_expanded, a, b, c)
            block_min = dist2.min(dim=1).values
            min_dist2 = block_min if min_dist2 is None else torch.minimum(min_dist2, block_min)

        outputs.append(torch.sqrt(min_dist2.clamp_min(0.0) + smooth_eps) - math.sqrt(smooth_eps))

    return torch.cat(outputs, dim=0)


def _triangle_mesh_winding_angle_torch_from_triangles(
    points: torch.Tensor,
    triangles: torch.Tensor,
    *,
    point_chunk_size: int | None = None,
    triangle_chunk_size: int | None = None,
) -> torch.Tensor:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3).")

    point_chunk = 4096 if point_chunk_size is None else int(point_chunk_size)
    triangle_chunk = 256 if triangle_chunk_size is None else int(triangle_chunk_size)
    outputs: list[torch.Tensor] = []
    eps = torch.finfo(points.dtype).eps

    for point_start in range(0, points.shape[0], point_chunk):
        point_stop = min(points.shape[0], point_start + point_chunk)
        point_block = points[point_start:point_stop]
        total_angle = point_block.new_zeros((point_block.shape[0],))
        point_block_expanded = point_block[:, None, :]

        for triangle_start in range(0, triangles.shape[0], triangle_chunk):
            triangle_stop = min(triangles.shape[0], triangle_start + triangle_chunk)
            triangle_block = triangles[triangle_start:triangle_stop]
            a = triangle_block[None, :, 0, :] - point_block_expanded
            b = triangle_block[None, :, 1, :] - point_block_expanded
            c = triangle_block[None, :, 2, :] - point_block_expanded

            la = torch.linalg.norm(a, dim=-1).clamp_min(eps)
            lb = torch.linalg.norm(b, dim=-1).clamp_min(eps)
            lc = torch.linalg.norm(c, dim=-1).clamp_min(eps)
            numerator = _dot(a, torch.cross(b, c, dim=-1))
            denominator = (
                la * lb * lc
                + _dot(a, b) * lc
                + _dot(b, c) * la
                + _dot(c, a) * lb
            )
            total_angle = total_angle + 2.0 * torch.atan2(numerator, _safe_signed_denom(denominator, eps)).sum(dim=1)

        outputs.append(total_angle)

    return torch.cat(outputs, dim=0)


def _triangle_mesh_unsigned_distance_torch(
    points: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    point_chunk_size: int | None = None,
    triangle_chunk_size: int | None = None,
    _triangles: torch.Tensor | None = None,
) -> torch.Tensor:
    triangles = _indexed_triangles(vertices, faces) if _triangles is None else _triangles
    return _triangle_mesh_unsigned_distance_torch_from_triangles(
        points,
        triangles,
        point_chunk_size=point_chunk_size,
        triangle_chunk_size=triangle_chunk_size,
    )


def _triangle_mesh_winding_angle_torch(
    points: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    point_chunk_size: int | None = None,
    triangle_chunk_size: int | None = None,
    _triangles: torch.Tensor | None = None,
) -> torch.Tensor:
    triangles = _indexed_triangles(vertices, faces) if _triangles is None else _triangles
    return _triangle_mesh_winding_angle_torch_from_triangles(
        points,
        triangles,
        point_chunk_size=point_chunk_size,
        triangle_chunk_size=triangle_chunk_size,
    )


def _launch_slang_unsigned_distance(
    triangles: torch.Tensor,
    points: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    module = _get_mesh_sdf_module()
    if module is None:
        raise RuntimeError("Mesh SDF Slang module is not available.")
    if points.shape[0] == 0:
        empty_f = torch.empty((0,), device=points.device, dtype=torch.float32)
        empty_i = torch.empty((0,), device=points.device, dtype=torch.int32)
        return empty_f, empty_i

    query_points = points.detach().contiguous()
    query_triangles = triangles.detach().contiguous()
    distances = torch.empty((points.shape[0],), device=points.device, dtype=torch.float32)
    closest = torch.empty((points.shape[0],), device=points.device, dtype=torch.int32)
    block_size = (256, 1, 1)

    module.queryMeshUnsignedDistance(
        triangles=query_triangles,
        points=query_points,
        unsignedDistance=distances,
        closestTriangleIndex=closest,
    ).launchRaw(
        blockSize=block_size,
        gridSize=_launch_shape_1d(points.shape[0], block_size[0]),
    )

    return distances, closest


def _launch_slang_unsigned_distance_bvh(
    triangles: torch.Tensor,
    points: torch.Tensor,
    bvh: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    module = _get_mesh_sdf_module()
    if module is None:
        raise RuntimeError("Mesh SDF Slang module is not available.")
    if points.shape[0] == 0:
        empty_f = torch.empty((0,), device=points.device, dtype=torch.float32)
        empty_i = torch.empty((0,), device=points.device, dtype=torch.int32)
        return empty_f, empty_i

    query_points = points.detach().contiguous()
    query_triangles = triangles.detach().contiguous()
    distances = torch.empty((points.shape[0],), device=points.device, dtype=torch.float32)
    closest = torch.empty((points.shape[0],), device=points.device, dtype=torch.int32)
    block_size = (256, 1, 1)

    module.queryMeshUnsignedDistanceBVH(
        triangles=query_triangles,
        points=query_points,
        nodeBBoxMin=bvh["bbox_min"],
        nodeBBoxMax=bvh["bbox_max"],
        nodeLeft=bvh["left"],
        nodeRight=bvh["right"],
        nodeStart=bvh["start"],
        nodeCount=bvh["count"],
        triangleIndices=bvh["triangle_indices"],
        unsignedDistance=distances,
        closestTriangleIndex=closest,
    ).launchRaw(
        blockSize=block_size,
        gridSize=_launch_shape_1d(points.shape[0], block_size[0]),
    )

    return distances, closest


def _launch_slang_parity_sign_bvh(
    triangles: torch.Tensor,
    points: torch.Tensor,
    bvh: dict[str, torch.Tensor],
    *,
    jitter_scale: float = 1.0e-6,
) -> torch.Tensor:
    module = _get_mesh_sdf_module()
    if module is None:
        raise RuntimeError("Mesh SDF Slang module is not available.")
    if points.shape[0] == 0:
        return torch.empty((0,), device=points.device, dtype=torch.int32)

    inside = torch.empty((points.shape[0],), device=points.device, dtype=torch.int32)
    block_size = (256, 1, 1)
    module.queryMeshParitySignBVH(
        triangles=triangles.detach().contiguous(),
        points=points.detach().contiguous(),
        nodeBBoxMin=bvh["bbox_min"],
        nodeBBoxMax=bvh["bbox_max"],
        nodeLeft=bvh["left"],
        nodeRight=bvh["right"],
        nodeStart=bvh["start"],
        nodeCount=bvh["count"],
        triangleIndices=bvh["triangle_indices"],
        jitterScale=float(jitter_scale),
        inside=inside,
    ).launchRaw(
        blockSize=block_size,
        gridSize=_launch_shape_1d(points.shape[0], block_size[0]),
    )
    return inside


def _launch_slang_distance_and_winding(
    triangles: torch.Tensor,
    points: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    module = _get_mesh_sdf_module()
    if module is None:
        raise RuntimeError("Mesh SDF Slang module is not available.")
    if points.shape[0] == 0:
        empty_f = torch.empty((0,), device=points.device, dtype=torch.float32)
        empty_i = torch.empty((0,), device=points.device, dtype=torch.int32)
        return empty_f, empty_f.clone(), empty_i

    query_points = points.detach().contiguous()
    query_triangles = triangles.detach().contiguous()
    distances = torch.empty((points.shape[0],), device=points.device, dtype=torch.float32)
    winding_angles = torch.empty((points.shape[0],), device=points.device, dtype=torch.float32)
    closest = torch.empty((points.shape[0],), device=points.device, dtype=torch.int32)
    block_size = (256, 1, 1)

    module.queryMeshDistanceAndWinding(
        triangles=query_triangles,
        points=query_points,
        unsignedDistance=distances,
        windingAngle=winding_angles,
        closestTriangleIndex=closest,
    ).launchRaw(
        blockSize=block_size,
        gridSize=_launch_shape_1d(points.shape[0], block_size[0]),
    )

    return distances, winding_angles, closest


def triangle_mesh_unsigned_distance_static_bvh(
    points: torch.Tensor,
    triangles: torch.Tensor,
    bvh: dict[str, torch.Tensor] | None,
) -> torch.Tensor:
    if bvh is None:
        distances, _closest = _launch_slang_unsigned_distance(triangles, points)
    else:
        distances, _closest = _launch_slang_unsigned_distance_bvh(triangles, points, bvh)
    return distances.to(dtype=points.dtype)


def triangle_mesh_signed_distance_static_bvh(
    points: torch.Tensor,
    triangles: torch.Tensor,
    bvh: dict[str, torch.Tensor] | None,
) -> torch.Tensor:
    if bvh is None:
        distances, _winding_angles, _closest = _launch_slang_distance_and_winding(triangles, points)
        parity_inside = torch.signbit(_signed_distance_from_unsigned_and_winding(
            distances.to(dtype=points.dtype),
            _winding_angles.to(dtype=points.dtype),
            winding_beta=0.5,
        )).to(dtype=torch.int32)
    else:
        distances, _closest = _launch_slang_unsigned_distance_bvh(triangles, points, bvh)
        parity_inside = _launch_slang_parity_sign_bvh(triangles, points, bvh)

    sign = torch.where(
        parity_inside.to(dtype=torch.bool),
        -torch.ones_like(distances, dtype=points.dtype),
        torch.ones_like(distances, dtype=points.dtype),
    )
    return distances.to(dtype=points.dtype) * sign


def _launch_slang_unsigned_backward(
    triangles: torch.Tensor,
    points: torch.Tensor,
    closest_triangle_index: torch.Tensor,
    grad_unsigned_distance: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    module = _get_mesh_sdf_module()
    if module is None:
        raise RuntimeError("Mesh SDF Slang module is not available.")
    if points.shape[0] == 0:
        return torch.zeros_like(triangles), torch.zeros_like(points)

    grad_triangles = torch.zeros_like(triangles)
    grad_points = torch.zeros_like(points)
    block_size = (256, 1, 1)
    module.backwardMeshUnsignedDistance(
        triangles=triangles.detach().contiguous(),
        points=points.detach().contiguous(),
        closestTriangleIndex=closest_triangle_index.detach().contiguous(),
        gradUnsignedDistance=grad_unsigned_distance.detach().contiguous(),
        gradTriangles=grad_triangles,
        gradPoints=grad_points,
    ).launchRaw(
        blockSize=block_size,
        gridSize=_launch_shape_1d(points.shape[0], block_size[0]),
    )
    return grad_triangles, grad_points


def _launch_slang_distance_and_winding_backward(
    triangles: torch.Tensor,
    points: torch.Tensor,
    closest_triangle_index: torch.Tensor,
    grad_unsigned_distance: torch.Tensor,
    grad_winding_angle: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    module = _get_mesh_sdf_module()
    if module is None:
        raise RuntimeError("Mesh SDF Slang module is not available.")
    if points.shape[0] == 0:
        return torch.zeros_like(triangles), torch.zeros_like(points)

    grad_triangles = torch.zeros_like(triangles)
    grad_points = torch.zeros_like(points)
    block_size = (256, 1, 1)
    module.backwardMeshDistanceAndWinding(
        triangles=triangles.detach().contiguous(),
        points=points.detach().contiguous(),
        closestTriangleIndex=closest_triangle_index.detach().contiguous(),
        gradUnsignedDistance=grad_unsigned_distance.detach().contiguous(),
        gradWindingAngle=grad_winding_angle.detach().contiguous(),
        gradTriangles=grad_triangles,
        gradPoints=grad_points,
    ).launchRaw(
        blockSize=block_size,
        gridSize=_launch_shape_1d(points.shape[0], block_size[0]),
    )
    return grad_triangles, grad_points


class _TriangleMeshUnsignedDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, triangles: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        distances, closest_triangle_index = _launch_slang_unsigned_distance(triangles, points)
        ctx.save_for_backward(triangles.detach(), points.detach(), closest_triangle_index)
        return distances

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        triangles, points, closest_triangle_index = ctx.saved_tensors
        grad_triangles, grad_points = _launch_slang_unsigned_backward(
            triangles,
            points,
            closest_triangle_index,
            grad_output.to(device=triangles.device, dtype=torch.float32),
        )
        return grad_triangles, grad_points


class _TriangleMeshWindingAngleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, triangles: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        _unsigned_distance, winding_angle, closest_triangle_index = _launch_slang_distance_and_winding(triangles, points)
        ctx.save_for_backward(triangles.detach(), points.detach(), closest_triangle_index)
        return winding_angle

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        triangles, points, closest_triangle_index = ctx.saved_tensors
        zeros = torch.zeros_like(grad_output, device=triangles.device, dtype=torch.float32)
        grad_triangles, grad_points = _launch_slang_distance_and_winding_backward(
            triangles,
            points,
            closest_triangle_index,
            zeros,
            grad_output.to(device=triangles.device, dtype=torch.float32),
        )
        return grad_triangles, grad_points


class _TriangleMeshSignedDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, triangles: torch.Tensor, points: torch.Tensor, winding_beta: float) -> torch.Tensor:
        unsigned_distance, winding_angle, closest_triangle_index = _launch_slang_distance_and_winding(triangles, points)
        signed_distance = _signed_distance_from_unsigned_and_winding(unsigned_distance, winding_angle, winding_beta)
        ctx.winding_beta = float(winding_beta)
        ctx.save_for_backward(
            triangles.detach(),
            points.detach(),
            closest_triangle_index,
            unsigned_distance,
            winding_angle,
        )
        return signed_distance

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        triangles, points, closest_triangle_index, unsigned_distance, winding_angle = ctx.saved_tensors
        beta = grad_output.new_tensor(float(ctx.winding_beta))
        shifted = (torch.abs(winding_angle) - 2.0 * math.pi) / beta
        tanh_shifted = torch.tanh(shifted)
        sign_factor = -tanh_shifted
        grad_unsigned_distance = grad_output.to(device=triangles.device, dtype=torch.float32) * sign_factor
        grad_winding_angle = (
            grad_output.to(device=triangles.device, dtype=torch.float32)
            * unsigned_distance
            * (-(1.0 - tanh_shifted.square()) / beta)
            * torch.sign(winding_angle)
        )
        grad_triangles, grad_points = _launch_slang_distance_and_winding_backward(
            triangles,
            points,
            closest_triangle_index,
            grad_unsigned_distance,
            grad_winding_angle,
        )
        return grad_triangles, grad_points, None


def triangle_mesh_unsigned_distance(
    points: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    point_chunk_size: int | None = None,
    triangle_chunk_size: int | None = None,
    _triangles: torch.Tensor | None = None,
) -> torch.Tensor:
    triangles = _indexed_triangles(vertices, faces) if _triangles is None else _triangles
    if _should_use_slang(points, vertices):
        if torch.is_grad_enabled() and (triangles.requires_grad or points.requires_grad):
            return _TriangleMeshUnsignedDistanceFunction.apply(triangles, points.contiguous())
        primal, _closest_triangle_index = _launch_slang_unsigned_distance(triangles, points)
        return primal.to(dtype=points.dtype)

    return _triangle_mesh_unsigned_distance_torch_from_triangles(
        points,
        triangles,
        point_chunk_size=point_chunk_size,
        triangle_chunk_size=triangle_chunk_size,
    )


def triangle_mesh_winding_angle(
    points: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    point_chunk_size: int | None = None,
    triangle_chunk_size: int | None = None,
    _triangles: torch.Tensor | None = None,
) -> torch.Tensor:
    triangles = _indexed_triangles(vertices, faces) if _triangles is None else _triangles
    if _should_use_slang(points, vertices):
        if torch.is_grad_enabled() and (triangles.requires_grad or points.requires_grad):
            return _TriangleMeshWindingAngleFunction.apply(triangles, points.contiguous())
        _unsigned_distance, winding_angle, _closest_triangle_index = _launch_slang_distance_and_winding(triangles, points)
        return winding_angle.to(dtype=points.dtype)

    return _triangle_mesh_winding_angle_torch_from_triangles(
        points,
        triangles,
        point_chunk_size=point_chunk_size,
        triangle_chunk_size=triangle_chunk_size,
    )


def _triangle_mesh_smooth_signed_distance_torch(
    points: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    winding_beta: float = 0.5,
    point_chunk_size: int | None = None,
    triangle_chunk_size: int | None = None,
    _triangles: torch.Tensor | None = None,
) -> torch.Tensor:
    triangles = _indexed_triangles(vertices, faces) if _triangles is None else _triangles
    unsigned_distance = _triangle_mesh_unsigned_distance_torch_from_triangles(
        points,
        triangles,
        point_chunk_size=point_chunk_size,
        triangle_chunk_size=triangle_chunk_size,
    )
    winding_angle = _triangle_mesh_winding_angle_torch_from_triangles(
        points,
        triangles,
        point_chunk_size=point_chunk_size,
        triangle_chunk_size=triangle_chunk_size,
    )
    return _signed_distance_from_unsigned_and_winding(unsigned_distance, winding_angle, winding_beta)


def triangle_mesh_smooth_signed_distance(
    points: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    winding_beta: float = 0.5,
    point_chunk_size: int | None = None,
    triangle_chunk_size: int | None = None,
    _triangles: torch.Tensor | None = None,
) -> torch.Tensor:
    triangles = _indexed_triangles(vertices, faces) if _triangles is None else _triangles
    if _should_use_slang(points, vertices):
        if torch.is_grad_enabled() and (triangles.requires_grad or points.requires_grad):
            return _TriangleMeshSignedDistanceFunction.apply(triangles, points.contiguous(), float(winding_beta))
        primal_unsigned_distance, primal_winding_angle, _closest_triangle_index = _launch_slang_distance_and_winding(
            triangles,
            points,
        )
        return _signed_distance_from_unsigned_and_winding(
            primal_unsigned_distance.to(dtype=points.dtype),
            primal_winding_angle.to(dtype=points.dtype),
            winding_beta,
        )

    return _triangle_mesh_smooth_signed_distance_torch(
        points,
        vertices,
        faces,
        winding_beta=winding_beta,
        point_chunk_size=point_chunk_size,
        triangle_chunk_size=triangle_chunk_size,
        _triangles=triangles,
    )
