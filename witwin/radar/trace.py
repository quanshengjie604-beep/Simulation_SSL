"""Mitsuba-based ray tracing for radar scenes."""

from __future__ import annotations

import drjit as dr
import mitsuba as mi
from mitsuba.scalar_rgb import Transform4f as T
import numpy as np
import torch
from witwin.core import build_mitsuba_scene, update_mitsuba_scene_vertices

# drjit type aliases (avoid mitsuba wrappers for pure array types)
Float = dr.cuda.ad.Float
UInt32 = dr.cuda.ad.UInt
Bool = dr.cuda.ad.Bool
Point3f = dr.cuda.ad.Array3f
Vector2f = dr.cuda.ad.Array2f
TensorXf = dr.cuda.ad.TensorXf

from .material import fresnel
from .types import SamplingMode

mi.set_variant("cuda_ad_rgb")


class TraceResult:
    """Opaque trace result. Supports ``points, intensities = tracer.trace()``."""

    __slots__ = (
        "points",
        "intensities",
        "entry_points",
        "fixed_path_lengths",
        "depths",
        "normals",
        "_tri_indices",
    )

    def __init__(
        self,
        points,
        intensities,
        tri_indices=None,
        *,
        entry_points=None,
        fixed_path_lengths=None,
        depths=None,
        normals=None,
    ):
        self.points = points
        self.intensities = intensities
        self.entry_points = points if entry_points is None else entry_points
        if fixed_path_lengths is None:
            fixed_path_lengths = torch.zeros(points.shape[0], dtype=torch.float32, device=points.device)
        self.fixed_path_lengths = fixed_path_lengths
        if depths is None:
            depths = torch.zeros(points.shape[0], dtype=torch.int32, device=points.device)
        self.depths = depths
        self.normals = normals
        self._tri_indices = tri_indices

    def __iter__(self):
        yield self.points
        yield self.intensities

    def __repr__(self):
        return f"TraceResult({self.points.shape[0]} points)"


class Tracer:
    """Ray tracer for declarative radar scenes."""

    _RAY_EPSILON = 1e-4
    _VISIBILITY_TOLERANCE = 1e-3

    def __init__(
        self,
        scene,
        radar,
        resolution=128,
        epsilon_r=5.0,
        sampling: SamplingMode = "pixel",
        *,
        multipath: bool = False,
        max_reflections: int = 0,
        ray_batch_size: int = 65536,
    ):
        self.scene = scene
        self.radar = radar
        self.resolution = int(resolution)
        self.epsilon_r = float(epsilon_r)
        self.sampling = SamplingMode(sampling)
        self.multipath = bool(multipath)
        self.max_reflections = int(max_reflections)
        self.ray_batch_size = int(ray_batch_size)
        self.variant = "cuda_ad_rgb"

        if self.max_reflections < 0:
            raise ValueError("max_reflections must be >= 0.")
        if self.ray_batch_size <= 0:
            raise ValueError("ray_batch_size must be > 0.")
        if self.multipath and self.sampling != SamplingMode.PIXEL:
            raise ValueError("multipath=True requires sampling='pixel'.")

        self._mi_scene = None
        self._params = None
        self._shape_eps = []

    def _mitsuba_sensor_dict(self):
        return {
            "type": "perspective",
            "to_world": T.look_at(
                origin=self.radar.position.tolist(),
                target=self.radar.target.tolist(),
                up=self.radar.up.tolist(),
            ),
            "fov": self.radar.fov,
            "film": {
                "type": "hdrfilm",
                "width": self.resolution,
                "height": self.resolution,
                "rfilter": {"type": "gaussian"},
                "sample_border": True,
                "pixel_format": "luminance",
                "component_format": "float32",
            },
            "sampler": {
                "type": "independent",
                "sample_count": 1,
                "seed": 42,
            },
        }

    def _empty_trace(self, *, include_tri_indices: bool = False) -> TraceResult:
        device = self.radar.device
        tri_indices = None
        if include_tri_indices:
            tri_indices = torch.empty((0,), dtype=torch.int64, device=device)
        return TraceResult(
            torch.empty((0, 3), dtype=torch.float32, device=device),
            torch.empty((0,), dtype=torch.float32, device=device),
            tri_indices,
            entry_points=torch.empty((0, 3), dtype=torch.float32, device=device),
            fixed_path_lengths=torch.empty((0,), dtype=torch.float32, device=device),
            depths=torch.empty((0,), dtype=torch.int32, device=device),
            normals=torch.empty((0, 3), dtype=torch.float32, device=device),
        )

    def _build_scene(self, renderables):
        default_bsdf = {
            "type": "diffuse",
            "reflectance": {"type": "rgb", "value": (0.8, 0.8, 0.8)},
        }
        state = build_mitsuba_scene(
            sensor=self._mitsuba_sensor_dict(),
            renderables=renderables,
            integrator={"type": "direct"},
            default_bsdf=default_bsdf,
            variant=self.variant,
        )
        self._mi_scene = state.scene
        self._params = state.params
        eps_by_name = {name: mesh_data.eps_r for name, mesh_data in renderables.items()}
        self._shape_eps = [(shape, float(eps_by_name.get(shape.id(), self.epsilon_r))) for shape in self._mi_scene.shapes()]
        self.scene.mark_clean()

    def _update_vertices(self, renderables):
        update_mitsuba_scene_vertices(self._params, renderables, variant=self.variant)
        self.scene.mark_clean()

    def _gen_rays(self):
        sensor = self._mi_scene.sensors()[0]
        film = sensor.film()
        sampler = sensor.sampler()
        film_size = film.crop_size()
        spp = 1
        total_sample_count = dr.prod(film_size) * spp

        if sampler.wavefront_size() != total_sample_count:
            sampler.seed(0, total_sample_count)

        pos = dr.arange(UInt32, total_sample_count)
        pos //= spp
        scale = Vector2f(1.0 / film_size[0], 1.0 / film_size[1])
        pos = Vector2f(
            Float(pos % int(film_size[0])) + 0.5,
            Float(pos // int(film_size[0])) + 0.5,
        )
        rays, _ = sensor.sample_ray_differential(
            time=0,
            sample1=sampler.next_1d(),
            sample2=pos * scale,
            sample3=0,
        )
        return rays

    def _gen_rays_batch(self, start: int, count: int):
        sensor = self._mi_scene.sensors()[0]
        film = sensor.film()
        film_size = film.crop_size()
        pos = dr.arange(UInt32, count) + start
        scale = Vector2f(1.0 / film_size[0], 1.0 / film_size[1])
        pos = Vector2f(
            Float(pos % int(film_size[0])) + 0.5,
            Float(pos // int(film_size[0])) + 0.5,
        )
        zeros = dr.zeros(Float, count)
        rays, _ = sensor.sample_ray_differential(
            time=zeros,
            sample1=zeros,
            sample2=pos * scale,
            sample3=zeros,
        )
        return rays

    def _lookup_eps_r(self, shape):
        eps_r = Float(self.epsilon_r)
        for candidate_shape, candidate_eps in self._shape_eps:
            eps_r = dr.select(shape == candidate_shape, Float(candidate_eps), eps_r)
        return eps_r

    def trace(self, *, time: float | None = None):
        renderables = self.scene.compile_renderables(time=time)
        if self._mi_scene is None or self.scene.dirty_level >= self.scene.DIRTY_FULL:
            self._build_scene(renderables)
        else:
            self._update_vertices(renderables)

        if self.sampling == "triangle":
            return self._trace_triangles(renderables)
        if self.sampling == "pixel":
            if self.multipath:
                return self._trace_pixels_multipath()
            return self._trace_pixels(renderables)
        raise AssertionError(f"Unsupported sampling mode '{self.sampling}'.")

    def match(self, a, b):
        if a._tri_indices is not None and b._tri_indices is not None:
            _, idx_a, idx_b = np.intersect1d(
                a._tri_indices.detach().cpu().numpy(),
                b._tri_indices.detach().cpu().numpy(),
                return_indices=True,
            )
            return a.points[idx_a], b.points[idx_b], a.intensities[idx_a]
        n = min(a.points.shape[0], b.points.shape[0])
        return a.points[:n], b.points[:n], a.intensities[:n]

    def render_image(self, *, time: float | None = None):
        renderables = self.scene.compile_renderables(time=time)
        if self._mi_scene is None or self.scene.dirty_level >= self.scene.DIRTY_FULL:
            self._build_scene(renderables)
        else:
            self._update_vertices(renderables)

        rays = self._gen_rays()
        si = self._mi_scene.ray_intersect(rays)
        cos_i = dr.abs(dr.dot(-rays.d, si.n))
        eps_r = self._lookup_eps_r(si.shape)
        reflectance = fresnel(cos_i, eps_r)
        reflectance = dr.select(si.is_valid(), reflectance, 0.0)
        return TensorXf(reflectance).torch().reshape(self.resolution, self.resolution)

    def _get_dynamic_meshes(self, renderables):
        return [name for name, mesh_data in renderables.items() if mesh_data.dynamic]

    def _compute_visibility_mask(self, mesh_name):
        vp = self._params[f"{mesh_name}.vertex_positions"]
        fi = self._params[f"{mesh_name}.faces"]
        num_faces = dr.width(fi) // 3
        face_idx = dr.arange(UInt32, num_faces)

        i0 = dr.gather(UInt32, fi, face_idx * 3)
        i1 = dr.gather(UInt32, fi, face_idx * 3 + 1)
        i2 = dr.gather(UInt32, fi, face_idx * 3 + 2)

        def _vertex(index):
            return Point3f(
                dr.gather(Float, vp, index * 3),
                dr.gather(Float, vp, index * 3 + 1),
                dr.gather(Float, vp, index * 3 + 2),
            )

        v0, v1, v2 = _vertex(i0), _vertex(i1), _vertex(i2)
        centroid = (v0 + v1 + v2) / 3.0
        normal = dr.normalize(dr.cross(v1 - v0, v2 - v0))

        origin = Point3f(*self.radar.position.tolist())
        view_dir = dr.normalize(origin - centroid)
        front = dr.dot(view_dir, normal) > 0

        ray_d = dr.normalize(centroid - origin)
        expected_t = dr.norm(centroid - origin)
        si = self._mi_scene.ray_intersect(mi.Ray3f(origin, ray_d))
        not_occluded = si.is_valid() & (si.t >= expected_t - 0.01)

        valid = front & not_occluded
        return TensorXf(dr.select(valid, Float(1.0), Float(0.0))).torch().bool()

    def _trace_triangles(self, renderables):
        all_points = []
        all_intensities = []
        all_tri_indices = []
        all_normals = []
        tri_offset = 0

        for mesh_name in self._get_dynamic_meshes(renderables):
            pts, intensities, normals, tri_idx, num_faces = self._trace_mesh_triangles(
                mesh_name, renderables[mesh_name].vertices, renderables[mesh_name].eps_r
            )
            if pts.shape[0] > 0:
                all_points.append(pts)
                all_intensities.append(intensities)
                all_normals.append(normals)
                all_tri_indices.append(tri_idx + tri_offset)
            tri_offset += num_faces

        if not all_points:
            return self._empty_trace(include_tri_indices=True)
        return TraceResult(
            torch.cat(all_points),
            torch.cat(all_intensities),
            torch.cat(all_tri_indices),
            normals=torch.cat(all_normals),
        )

    def _trace_mesh_triangles(self, mesh_name, vertices_torch, eps_r):
        fi = self._params[f"{mesh_name}.faces"]
        num_faces = dr.width(fi) // 3
        visibility = self._compute_visibility_mask(mesh_name)
        visible_index = visibility.nonzero(as_tuple=True)[0]
        empty = (
            torch.empty((0, 3), dtype=torch.float32, device="cuda"),
            torch.empty((0,), dtype=torch.float32, device="cuda"),
            torch.empty((0, 3), dtype=torch.float32, device="cuda"),
            torch.empty((0,), dtype=torch.int64, device="cuda"),
        )
        if visible_index.numel() == 0:
            return (*empty, num_faces)

        sensor_origin = self.radar.position.tolist()
        num_faces_captured = num_faces
        fi_captured = fi

        @dr.wrap(source="torch", target="drjit")
        def _geometry(vertices):
            vp = dr.ravel(vertices)
            face_idx = dr.arange(UInt32, num_faces_captured)
            i0 = dr.gather(UInt32, fi_captured, face_idx * 3)
            i1 = dr.gather(UInt32, fi_captured, face_idx * 3 + 1)
            i2 = dr.gather(UInt32, fi_captured, face_idx * 3 + 2)

            def _vertex(index):
                return Point3f(
                    dr.gather(Float, vp, index * 3),
                    dr.gather(Float, vp, index * 3 + 1),
                    dr.gather(Float, vp, index * 3 + 2),
                )

            v0, v1, v2 = _vertex(i0), _vertex(i1), _vertex(i2)
            centroid = (v0 + v1 + v2) / 3.0
            cross = dr.cross(v1 - v0, v2 - v0)
            cross_len = dr.norm(cross)
            area = 0.5 * cross_len
            normal = cross / (cross_len + 1e-10)

            origin = Point3f(*sensor_origin)
            cos_i = dr.abs(dr.dot(dr.normalize(origin - centroid), normal))
            reflectance = fresnel(cos_i, eps_r)
            intensity = area * reflectance

            out = dr.zeros(Float, num_faces_captured * 7)
            idx = dr.arange(UInt32, num_faces_captured)
            dr.scatter(out, centroid.x, idx * 7)
            dr.scatter(out, centroid.y, idx * 7 + 1)
            dr.scatter(out, centroid.z, idx * 7 + 2)
            dr.scatter(out, intensity, idx * 7 + 3)
            dr.scatter(out, normal.x, idx * 7 + 4)
            dr.scatter(out, normal.y, idx * 7 + 5)
            dr.scatter(out, normal.z, idx * 7 + 6)
            return TensorXf(out, shape=(num_faces_captured, 7))

        result = _geometry(vertices_torch)
        return (
            result[:, :3][visible_index],
            result[:, 3][visible_index],
            result[:, 4:7][visible_index],
            visible_index,
            num_faces,
        )

    def _trace_pixels(self, renderables):
        first_name = None
        first_vertices = None
        first_eps_r = self.epsilon_r
        for name, mesh_data in renderables.items():
            key = f"{name}.vertex_positions"
            if key not in self._params:
                continue
            first_name = name
            first_vertices = mesh_data.vertices
            first_eps_r = mesh_data.eps_r
            break

        if first_name is None or first_vertices is None:
            return self._empty_trace()

        key = f"{first_name}.vertex_positions"
        params_ref = self._params
        scene_ref = self._mi_scene
        rays = self._gen_rays()

        @dr.wrap(source="torch", target="drjit")
        def _pixel(vertices):
            params_ref[key] = dr.ravel(vertices)
            params_ref.update()
            si = scene_ref.ray_intersect(rays)

            cos_i = dr.abs(dr.dot(-rays.d, si.n))
            reflectance = fresnel(cos_i, first_eps_r)
            valid_float = dr.select(si.is_valid(), Float(1.0), Float(0.0))
            reflectance = reflectance * valid_float

            count = dr.width(reflectance)
            out = dr.zeros(Float, count * 8)
            idx = dr.arange(UInt32, count)
            dr.scatter(out, si.p.x * valid_float, idx * 8)
            dr.scatter(out, si.p.y * valid_float, idx * 8 + 1)
            dr.scatter(out, si.p.z * valid_float, idx * 8 + 2)
            dr.scatter(out, reflectance, idx * 8 + 3)
            dr.scatter(out, valid_float, idx * 8 + 4)
            dr.scatter(out, si.n.x * valid_float, idx * 8 + 5)
            dr.scatter(out, si.n.y * valid_float, idx * 8 + 6)
            dr.scatter(out, si.n.z * valid_float, idx * 8 + 7)
            return TensorXf(out, shape=(count, 8))

        result = _pixel(first_vertices)
        valid_index = (result[:, 4] > 0.5).nonzero(as_tuple=True)[0]
        if valid_index.numel() == 0:
            return self._empty_trace()
        points = result[:, :3][valid_index]
        intensities = result[:, 3][valid_index]
        normals = result[:, 5:8][valid_index]
        return TraceResult(points, intensities, normals=normals)

    def _trace_pixels_multipath(self):
        all_points = []
        all_intensities = []
        all_entry_points = []
        all_fixed_lengths = []
        all_depths = []
        all_normals = []

        for start in range(0, self.resolution * self.resolution, self.ray_batch_size):
            count = min(self.ray_batch_size, self.resolution * self.resolution - start)
            rays = self._gen_rays_batch(start=start, count=count)
            self._trace_pixel_batch(
                rays,
                count=count,
                all_points=all_points,
                all_intensities=all_intensities,
                all_entry_points=all_entry_points,
                all_fixed_lengths=all_fixed_lengths,
                all_depths=all_depths,
                all_normals=all_normals,
            )

        if not all_points:
            return self._empty_trace()
        return TraceResult(
            torch.cat(all_points, dim=0),
            torch.cat(all_intensities, dim=0),
            entry_points=torch.cat(all_entry_points, dim=0),
            fixed_path_lengths=torch.cat(all_fixed_lengths, dim=0),
            depths=torch.cat(all_depths, dim=0),
            normals=torch.cat(all_normals, dim=0),
        )

    def _trace_pixel_batch(
        self,
        rays,
        *,
        count: int,
        all_points,
        all_intensities,
        all_entry_points,
        all_fixed_lengths,
        all_depths,
        all_normals,
    ) -> None:
        active = dr.full(Bool, True, count)
        entry_points = Point3f(0.0, 0.0, 0.0)
        prev_bounce_points = Point3f(0.0, 0.0, 0.0)
        fixed_lengths = dr.zeros(Float, count)
        cumulative_reflectance = dr.full(Float, 1.0, count)

        for depth in range(self.max_reflections + 1):
            if not dr.any(active):
                break

            si = self._mi_scene.ray_intersect(rays, active=active)
            valid = active & si.is_valid()
            if not dr.any(valid):
                break

            hit_points = si.p
            normals = si.n
            incoming = rays.d
            eps_r = self._lookup_eps_r(si.shape)
            cos_i = dr.abs(dr.dot(-incoming, normals))
            reflectance = fresnel(cos_i, eps_r)

            if depth == 0:
                emitted_entry_points = hit_points
                emitted_fixed_lengths = dr.zeros(Float, count)
                visible = valid
            else:
                emitted_entry_points = entry_points
                segment_lengths = dr.norm(hit_points - prev_bounce_points)
                emitted_fixed_lengths = fixed_lengths + segment_lengths
                visible = valid & self._visible_from_origin(hit_points, normals, valid)

            emitted_intensities = cumulative_reflectance * reflectance
            emitted_valid = visible & (emitted_intensities > 0)

            self._append_trace_batch(
                hit_points=hit_points,
                intensities=emitted_intensities,
                entry_points=emitted_entry_points,
                fixed_lengths=emitted_fixed_lengths,
                normals=normals,
                depth=depth,
                valid=emitted_valid,
                all_points=all_points,
                all_intensities=all_intensities,
                all_entry_points=all_entry_points,
                all_fixed_lengths=all_fixed_lengths,
                all_depths=all_depths,
                all_normals=all_normals,
            )

            if depth == self.max_reflections:
                break

            reflected_dir = dr.normalize(incoming - 2.0 * dr.dot(incoming, normals) * normals)
            offset_sign = dr.select(dr.dot(reflected_dir, normals) >= 0.0, 1.0, -1.0)
            next_origin = hit_points + normals * (offset_sign * self._RAY_EPSILON)

            rays = mi.Ray3f(next_origin, reflected_dir)
            entry_points = hit_points if depth == 0 else emitted_entry_points
            prev_bounce_points = hit_points
            fixed_lengths = dr.zeros(Float, count) if depth == 0 else emitted_fixed_lengths
            cumulative_reflectance = cumulative_reflectance * reflectance
            active = valid

    def _visible_from_origin(self, hit_points, normals, active):
        origin = Point3f(*self.radar.position.tolist())
        to_origin = origin - hit_points
        direction = dr.normalize(to_origin)
        offset_sign = dr.select(dr.dot(direction, normals) >= 0.0, 1.0, -1.0)
        shadow_origin = hit_points + normals * (offset_sign * self._RAY_EPSILON)
        shadow_rays = mi.Ray3f(shadow_origin, direction)
        shadow_si = self._mi_scene.ray_intersect(shadow_rays, active=active)
        expected_t = dr.norm(to_origin)
        return (~shadow_si.is_valid()) | (shadow_si.t >= expected_t - self._VISIBILITY_TOLERANCE)

    def _append_trace_batch(
        self,
        *,
        hit_points,
        intensities,
        entry_points,
        fixed_lengths,
        normals,
        depth: int,
        valid,
        all_points,
        all_intensities,
        all_entry_points,
        all_fixed_lengths,
        all_depths,
        all_normals,
    ) -> None:
        if not dr.any(valid):
            return

        count = dr.width(intensities)
        idx = dr.arange(UInt32, count)
        out = dr.zeros(Float, count * 12)
        valid_float = dr.select(valid, Float(1.0), Float(0.0))
        dr.scatter(out, hit_points.x, idx * 12)
        dr.scatter(out, hit_points.y, idx * 12 + 1)
        dr.scatter(out, hit_points.z, idx * 12 + 2)
        dr.scatter(out, intensities, idx * 12 + 3)
        dr.scatter(out, entry_points.x, idx * 12 + 4)
        dr.scatter(out, entry_points.y, idx * 12 + 5)
        dr.scatter(out, entry_points.z, idx * 12 + 6)
        dr.scatter(out, fixed_lengths, idx * 12 + 7)
        dr.scatter(out, Float(float(depth)), idx * 12 + 8)
        dr.scatter(out, normals.x, idx * 12 + 9)
        dr.scatter(out, normals.y, idx * 12 + 10)
        dr.scatter(out, normals.z, idx * 12 + 11)

        packed = TensorXf(out, shape=(count, 12)).torch()
        mask = TensorXf(valid_float).torch() > 0.5
        packed = packed[mask]
        if packed.numel() == 0:
            return

        all_points.append(packed[:, :3])
        all_intensities.append(packed[:, 3])
        all_entry_points.append(packed[:, 4:7])
        all_fixed_lengths.append(packed[:, 7])
        all_depths.append(packed[:, 8].to(torch.int32))
        all_normals.append(packed[:, 9:12])
