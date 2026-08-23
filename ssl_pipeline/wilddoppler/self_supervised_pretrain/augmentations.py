from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class ViewConfig:
    min_ratio: float = 1.0
    max_ratio: float = 1.0


def random_time_crop_resize(
    x: torch.Tensor,
    cfg: ViewConfig,
    forced_windows: Optional[list[Optional[dict[str, float]]]] = None,
) -> torch.Tensor:
    b, c, f, t = x.shape
    min_ratio = max(1e-3, min(float(cfg.min_ratio), 1.0))
    max_ratio = max(min_ratio, min(float(cfg.max_ratio), 1.0))
    samples = []
    for i in range(b):
        forced = None if forced_windows is None else forced_windows[i]
        if forced is None:
            ratio = random.uniform(min_ratio, max_ratio)
            length = max(1, min(t, int(round(t * ratio))))
            start = None
        else:
            length = max(1, min(t, int(round(forced.get("length", t)))))
            start = max(0, min(t - length, int(round(forced.get("start", 0)))))
        if length >= t:
            samples.append(x[i : i + 1].clone())
            continue
        if start is None:
            start = random.randint(0, t - length)
        crop = x[i : i + 1, :, :, start : start + length]
        samples.append(F.interpolate(crop, size=(f, t), mode="bilinear", align_corners=False))
    return torch.cat(samples, dim=0)


class RadarAugmentations:
    def __init__(
        self,
        num_global_views: int = 2,
        num_local_views: int = 2,
        global_time_crop: ViewConfig = ViewConfig(1.0, 1.0),
        local_time_crop: ViewConfig = ViewConfig(0.5, 1.0),
        gaussian_noise_std: float = 0.1,
        flip_time_prob: float = 0.1,
        flip_freq_prob: float = 0.5,
        time_shift_prob: float = 0.5,
        time_shift_max_ratio: float = 0.2,
        time_mask_prob: float = 0.2,
        time_mask_max_ratio: float = 0.05,
        freq_mask_prob: float = 0.2,
        freq_mask_max_ratio: float = 0.05,
        patch_mask_prob: float = 0.2,
        patch_mask_num_min: int = 0,
        patch_mask_num_max: int = 2,
        patch_mask_min_time_ratio: float = 0.02,
        patch_mask_max_time_ratio: float = 0.08,
        patch_mask_min_freq_ratio: float = 0.02,
        patch_mask_max_freq_ratio: float = 0.08,
        interference_mix_prob: float = 0.1,
        interference_alpha_min: float = 0.2,
        interference_alpha_max: float = 0.6,
        local_limb_aug_prob: float = 0.4,
        local_limb_gauss_sigma: float = 3.0,
        local_limb_gauss_alpha: float = 2.0,
        local_limb_mask_percentile: float = 60.0,
        local_limb_mix: float = 0.85,
        local_limb_velnorm_smooth: int = 5,
        enforce_disjoint_global_crops: bool = True,
    ) -> None:
        self.num_global_views = max(1, int(num_global_views))
        self.num_local_views = max(0, int(num_local_views))
        self.global_time_crop = global_time_crop
        self.local_time_crop = local_time_crop
        self.gaussian_noise_std = max(0.0, float(gaussian_noise_std))
        self.flip_time_prob = max(0.0, min(1.0, float(flip_time_prob)))
        self.flip_freq_prob = max(0.0, min(1.0, float(flip_freq_prob)))
        self.time_shift_prob = max(0.0, min(1.0, float(time_shift_prob)))
        self.time_shift_max_ratio = max(0.0, min(1.0, float(time_shift_max_ratio)))
        self.time_mask_prob = max(0.0, min(1.0, float(time_mask_prob)))
        self.time_mask_max_ratio = max(0.0, min(1.0, float(time_mask_max_ratio)))
        self.freq_mask_prob = max(0.0, min(1.0, float(freq_mask_prob)))
        self.freq_mask_max_ratio = max(0.0, min(1.0, float(freq_mask_max_ratio)))
        self.patch_mask_prob = max(0.0, min(1.0, float(patch_mask_prob)))
        self.patch_mask_num_min = max(0, int(patch_mask_num_min))
        self.patch_mask_num_max = max(self.patch_mask_num_min, int(patch_mask_num_max))
        self.patch_mask_min_time_ratio = max(0.0, min(1.0, float(patch_mask_min_time_ratio)))
        self.patch_mask_max_time_ratio = max(self.patch_mask_min_time_ratio, min(1.0, float(patch_mask_max_time_ratio)))
        self.patch_mask_min_freq_ratio = max(0.0, min(1.0, float(patch_mask_min_freq_ratio)))
        self.patch_mask_max_freq_ratio = max(self.patch_mask_min_freq_ratio, min(1.0, float(patch_mask_max_freq_ratio)))
        self.interference_mix_prob = max(0.0, min(1.0, float(interference_mix_prob)))
        self.interference_alpha_min = max(0.0, min(1.0, float(interference_alpha_min)))
        self.interference_alpha_max = max(self.interference_alpha_min, min(1.0, float(interference_alpha_max)))
        self.local_limb_aug_prob = max(0.0, min(1.0, float(local_limb_aug_prob)))
        self.local_limb_gauss_sigma = max(1e-3, float(local_limb_gauss_sigma))
        self.local_limb_gauss_alpha = float(local_limb_gauss_alpha)
        self.local_limb_mask_percentile = max(0.0, min(100.0, float(local_limb_mask_percentile)))
        self.local_limb_mix = max(0.0, min(1.0, float(local_limb_mix)))
        self.local_limb_velnorm_smooth = max(1, int(local_limb_velnorm_smooth))
        self.enforce_disjoint_global_crops = bool(enforce_disjoint_global_crops)

    def __call__(self, batch: torch.Tensor) -> list[torch.Tensor]:
        # Disjoint global crops prevent the same time segment from appearing in two positive pairs.
        forced = self._build_forced_global_windows(batch) if self.enforce_disjoint_global_crops else None
        views = [
            self._augment(
                batch,
                self.global_time_crop,
                forced_windows=None if forced is None else forced[i],
                apply_local_limb=False,
            )
            for i in range(self.num_global_views)
        ]
        views.extend(self._augment(batch, self.local_time_crop, apply_local_limb=True) for _ in range(self.num_local_views))
        return views

    def _augment(
        self,
        batch: torch.Tensor,
        crop_cfg: ViewConfig,
        forced_windows: Optional[list[Optional[dict[str, float]]]] = None,
        apply_local_limb: bool = False,
    ) -> torch.Tensor:
        x = random_time_crop_resize(batch, crop_cfg, forced_windows=forced_windows)
        if self.time_shift_prob > 0 and torch.rand(()) < self.time_shift_prob:
            max_shift = int(round(x.shape[-1] * self.time_shift_max_ratio))
            if max_shift > 0:
                x = torch.roll(x, shifts=random.randint(-max_shift, max_shift), dims=-1)
        if self.flip_time_prob > 0 and torch.rand(()) < self.flip_time_prob:
            x = torch.flip(x, dims=[-1])
        if self.flip_freq_prob > 0 and torch.rand(()) < self.flip_freq_prob:
            x = torch.flip(x, dims=[-2])
        if apply_local_limb and self.local_limb_aug_prob > 0:
            x = self._local_limb_batch(x)
        if self.interference_mix_prob > 0 and x.size(0) > 1 and torch.rand(()) < self.interference_mix_prob:
            x = self._interference_mix(x)
        x = self._mask_time(x)
        x = self._mask_freq(x)
        x = self._mask_patches(x)
        if self.gaussian_noise_std > 0:
            x = x + torch.randn_like(x) * self.gaussian_noise_std
        return x.contiguous()

    def _mask_time(self, x: torch.Tensor) -> torch.Tensor:
        if self.time_mask_prob <= 0 or self.time_mask_max_ratio <= 0 or torch.rand(()) >= self.time_mask_prob:
            return x
        t = x.shape[-1]
        length = max(1, min(t, int(round(t * random.random() * self.time_mask_max_ratio))))
        start = random.randint(0, max(0, t - length))
        x = x.clone()
        x[..., start : start + length] = x.mean()
        return x

    def _mask_freq(self, x: torch.Tensor) -> torch.Tensor:
        if self.freq_mask_prob <= 0 or self.freq_mask_max_ratio <= 0 or torch.rand(()) >= self.freq_mask_prob:
            return x
        f = x.shape[-2]
        length = max(1, min(f, int(round(f * random.random() * self.freq_mask_max_ratio))))
        start = random.randint(0, max(0, f - length))
        x = x.clone()
        x[..., start : start + length, :] = x.mean()
        return x

    def _mask_patches(self, x: torch.Tensor) -> torch.Tensor:
        if self.patch_mask_prob <= 0 or self.patch_mask_num_max <= 0 or torch.rand(()) >= self.patch_mask_prob:
            return x
        b, _, f, t = x.shape
        x = x.clone()
        fill = x.mean()
        for batch_idx in range(b):
            num = random.randint(self.patch_mask_num_min, self.patch_mask_num_max)
            for _ in range(num):
                tw = max(1, min(t, int(round(t * random.uniform(self.patch_mask_min_time_ratio, self.patch_mask_max_time_ratio)))))
                fw = max(1, min(f, int(round(f * random.uniform(self.patch_mask_min_freq_ratio, self.patch_mask_max_freq_ratio)))))
                ts = random.randint(0, max(0, t - tw))
                fs = random.randint(0, max(0, f - fw))
                x[batch_idx, :, fs : fs + fw, ts : ts + tw] = fill
        return x

    def _interference_mix(self, x: torch.Tensor) -> torch.Tensor:
        # Simulates receiving a faint radar return from a different person in the same scene.
        mixed = x.clone()
        batch_size = x.size(0)
        for sample_idx in range(batch_size):
            offset = random.randint(1, batch_size - 1)
            mix_idx = (sample_idx + offset) % batch_size
            alpha = float(
                torch.empty((), device=x.device, dtype=x.dtype)
                .uniform_(self.interference_alpha_min, self.interference_alpha_max)
                .item()
            )
            mixed[sample_idx] = x[sample_idx] * (1.0 - alpha) + x[mix_idx] * alpha
        return mixed

    @staticmethod
    def _normalize_like(x: torch.Tensor, ref: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        x_std = x.std(unbiased=False)
        if float(x_std.detach()) < eps:
            return torch.full_like(x, float(ref.mean().detach()))
        return (x - x.mean()) / (x_std + eps) * ref.std(unbiased=False) + ref.mean()

    def _local_limb_batch(self, x: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        for i in range(x.size(0)):
            if torch.rand((), device=x.device) < self.local_limb_aug_prob:
                out[i] = self._local_limb_sample(out[i])
        return out

    def _local_limb_sample(self, x: torch.Tensor) -> torch.Tensor:
        # Enhances high-frequency limb returns (above percentile threshold) via Gaussian-separated signal.
        ref = x
        low = self._gaussian_blur_velocity(x)
        high = x - low
        tau = torch.quantile(x, self.local_limb_mask_percentile / 100.0, dim=-2, keepdim=True)
        enhanced = x + self.local_limb_gauss_alpha * ((x > tau).to(x.dtype) * high)
        out = (1.0 - self.local_limb_mix) * x + self.local_limb_mix * self._normalize_like(enhanced, ref)
        return self._velocity_normalize(self._normalize_like(out, ref))

    def _gaussian_blur_velocity(self, x: torch.Tensor) -> torch.Tensor:
        c, f, t = x.shape
        sigma = self.local_limb_gauss_sigma
        radius = max(1, int(round(3.0 * sigma)))
        coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
        kernel = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
        kernel = (kernel / kernel.sum()).view(1, 1, -1)
        y = x.permute(0, 2, 1).reshape(c * t, 1, f)
        y = F.pad(y, (radius, radius), mode="reflect")
        y = F.conv1d(y, kernel)
        return y.reshape(c, t, f).permute(0, 2, 1)

    def _velocity_normalize(self, x: torch.Tensor) -> torch.Tensor:
        # Rolls each channel so its dominant velocity bin aligns to the frequency-axis center.
        c, f, t = x.shape
        window = self.local_limb_velnorm_smooth + (1 - self.local_limb_velnorm_smooth % 2)
        y = x.abs().reshape(c * f, 1, t)
        y = F.pad(y, (window // 2, window // 2), mode="reflect")
        kernel = torch.ones(1, 1, window, device=x.device, dtype=x.dtype) / float(window)
        profile = F.conv1d(y, kernel).reshape(c, f, t).mean(dim=-1)
        center = f // 2
        out = torch.empty_like(x)
        for chan in range(c):
            shift = center - int(torch.argmax(profile[chan]).item())
            out[chan] = torch.roll(x[chan], shifts=shift, dims=-2)
        return out

    def _build_forced_global_windows(self, batch: torch.Tensor) -> list[list[Optional[dict[str, float]]]]:
        per_view: list[list[Optional[dict[str, float]]]] = [[] for _ in range(self.num_global_views)]
        t = batch.shape[-1]
        for _ in range(batch.size(0)):
            available = [(0, t)]
            for view_idx in range(self.num_global_views):
                ratio = random.uniform(self.global_time_crop.min_ratio, self.global_time_crop.max_ratio)
                length = max(1, min(t, int(round(t * ratio))))
                candidates = [span for span in available if span[1] - span[0] >= length]
                if not candidates:
                    per_view[view_idx].append(None)
                    continue
                lo, hi = random.choice(candidates)
                start = random.randint(lo, hi - length)
                per_view[view_idx].append({"start": start, "length": length})
                available = self._remove_interval(available, start, start + length)
        return per_view

    @staticmethod
    def _remove_interval(intervals: list[tuple[int, int]], start: int, end: int) -> list[tuple[int, int]]:
        kept = []
        for lo, hi in intervals:
            if end <= lo or start >= hi:
                kept.append((lo, hi))
            else:
                if start > lo:
                    kept.append((lo, start))
                if end < hi:
                    kept.append((end, hi))
        return kept
