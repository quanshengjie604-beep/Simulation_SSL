from typing import Any, List, Optional
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import random
import torchvision.transforms as transforms
random.seed(1)


class ToTensor(object):
    """Convert numpy array to tensor format. For vhist.
    """
    def __init__(self):
        pass

    def __call__(self, vhist):
        return torch.from_numpy(vhist)

### Radar data Transforms 
class ToCHWTensor(object):
    """Convert numpy array to CHW tensor format.
    """
    def __init__(self):
        pass

    def __call__(self, radar_dat):
        radar_dat = radar_dat.transpose((2,0,1)) # shape: (radar channel, doppler, time)
        return torch.from_numpy(radar_dat)


class RandomizeStart(object):
    """Randomly select starting time index of snapshot and crop it to
    network input size.
    radar_dat: (channel, doppler, time)
    Args:
        output_len (int): desired output size of time, should be less
            than snapshot length of time
        time_win_start: snapshot time window starting index
    """

    def __init__(self, output_len, time_win_start, crop_idx=None, crop_num=None):
        self.output_len = output_len
        self.time_win_start = time_win_start
        self.crop_idx = crop_idx
        self.crop_num = crop_num

    def __call__(self, radar_dat):
        data_length = radar_dat.shape[2]
        start_idx_min = self.time_win_start
        assert self.output_len <= data_length, f"network output size {self.output_len} > radar_dat len"

        start_idx_max = data_length - self.output_len 
        assert start_idx_min <= start_idx_max, f"large start index {start_idx_min}"
        if start_idx_min == start_idx_max:
            start_idx = 0
        else:
            if self.crop_idx is not None:
                step = (data_length - self.output_len) / (self.crop_num - 1)
                # Determine each cropping point (rounding to the nearest integer)
                start_idx = int(round(step * (self.crop_idx)))
            else:
                start_idx =np.random.choice(np.arange(start_idx_min, start_idx_max))
        return radar_dat[..., start_idx:start_idx + self.output_len]

class Normalize(object):
    """
    Apply z-normalization
    """
    def __init__(self, mean1, std1, mean2, std2):
        self.mean = (mean1, mean2)
        self.std = (std1, std2)
    def __call__(self, radar_dat):
        radar_dat[0] = torchvision.transforms.functional.normalize(radar_dat[0], self.mean[0], self.std[0])
        radar_dat[1] = torchvision.transforms.functional.normalize(radar_dat[1], self.mean[1], self.std[1])
        return radar_dat

class CenterStart(object):
    """crop time range in the center to network input size.

    Args:
        output_len (int): desired output size of time, should be less
            than snapshot length of time
    """

    def __init__(self, output_len):
        self.output_len = output_len

    def __call__(self, radar_dat):
        start_idx = int((radar_dat.shape[2] - self.output_len)/2)
        return radar_dat[..., start_idx:start_idx + self.output_len]


class CropDoppler(object):
    """Crop micro-Doppler range in center into the network input shape
    """

    def __init__(self, output_len):
        self.output_len = output_len

    def __call__(self, radar_dat):
        assert self.output_len <= radar_dat.shape[1], f"network output size {self.output_len} > radar_dat len"

        pos_len = int(self.output_len/2)
        start = int(radar_dat.shape[1]/2) - pos_len
        return radar_dat[..., start:start+self.output_len, :]


class CropMicroDopplerTransform:
    def __init__(self, bins_per_second=90, crop_seconds=1.0, random_crop=True):
        """
        Args:
            bins_per_second: time resolution (your radar is 90 Hz)
            crop_seconds: output window size in seconds (1 or 2)
            random_crop: whether to randomly shift inside [t_start, t_end]
        """
        self.bins_per_second = bins_per_second
        self.crop_bins = int(crop_seconds * bins_per_second)
        self.random_crop = random_crop

    def __call__(self, md, t_start, t_end):
        """
        md: numpy array [T_total, F]
        t_start, t_end: floats in seconds from CSV
        """
        T_total = md.shape[0]

        # Convert times → bin indices
        start_idx = int(t_start * self.bins_per_second)
        end_idx   = int(t_end   * self.bins_per_second)

        # Clamp (dataset sometimes has floating error)
        start_idx = max(0, min(start_idx, T_total - 1))
        end_idx   = max(0, min(end_idx, T_total))

        # Interval length
        interval_len = end_idx - start_idx

        # If interval shorter than crop window → pad or skip
        if interval_len < self.crop_bins:
            # pad with zeros (simple)
            pad_len = self.crop_bins - interval_len
            crop = md[start_idx:end_idx]
            crop = np.pad(crop, ((0, pad_len), (0, 0)), mode='constant')
        else:
            # choose deterministic or random offset
            if self.random_crop:
                max_offset = interval_len - self.crop_bins
                offset = random.randint(0, max_offset)
            else:
                offset = 0

            crop = md[start_idx + offset : start_idx + offset + self.crop_bins]

        # shape [T_crop, F] → [1, F, T_crop]
        crop = crop.T[None, :, :]
        return crop.astype(np.float32)


def random_time_crop_resize_tensor(
    x: torch.Tensor,
    min_ratio: float,
    max_ratio: float,
    return_info: bool = False,
    forced_windows: Optional[List[Optional[dict]]] = None,
    resize_back: bool = True,
):
    """
    Randomly crop along the time axis (last dimension) for each sample.
    By default the crop is resized back to the original temporal length,
    but if `resize_back` is False the cropped region is simply zero-padded
    into the original canvas without interpolation. Optionally accepts a
    `forced_windows` list to specify deterministic (start, length)
    pairs per sample, and can return metadata describing each crop.
    """
    if x.dim() != 4:
        return (x.clone(), [None] * x.size(0)) if return_info else x.clone()

    B, C, freq_bins, time_bins = x.shape
    if forced_windows is not None and len(forced_windows) != B:
        raise ValueError("forced_windows must match the batch size.")
    augmented = []
    info = [] if return_info else None
    min_ratio = max(1e-3, min(min_ratio, 1.0))
    max_ratio = max(min_ratio, min(max_ratio, 1.0))

    for b in range(B):
        sample = x[b : b + 1]
        forced_window = None if forced_windows is None else forced_windows[b]
        if forced_window is not None:
            target_len = int(round(forced_window.get("length", time_bins)))
            target_len = max(1, min(time_bins, target_len))
            start = int(round(forced_window.get("start", 0)))
            start = max(0, min(start, time_bins - target_len))
            ratio = float(forced_window.get("ratio", target_len / float(time_bins)))
        else:
            ratio = random.uniform(min_ratio, max_ratio)
            target_len = int(round(time_bins * ratio))
            target_len = max(1, min(time_bins, target_len))
            start = None

        if target_len >= time_bins:
            augmented.append(sample.clone())
            if return_info:
                mask = torch.ones((1, 1, 1, time_bins), device=x.device)
                mask = F.interpolate(mask, size=(freq_bins, time_bins), mode="nearest")
                info.append(
                    {
                        "start": 0,
                        "end": time_bins,
                        "length": time_bins,
                        "ratio": 1.0,
                        "mask": mask,
                        "forced": forced_window is not None,
                    }
                )
            continue

        max_start = time_bins - target_len
        if start is None:
            start = random.randint(0, max_start)
        else:
            start = max(0, min(start, max_start))

        cropped = sample[..., start : start + target_len]
        if resize_back:
            resized = F.interpolate(
                cropped,
                size=(freq_bins, time_bins),
                mode="bilinear",
                align_corners=False,
            )
        else:
            fill_value = float(sample.mean().item()) if sample.numel() > 0 else 0.0
            resized = torch.full_like(sample, fill_value)
            noise_std = float(sample.std(unbiased=False).item()) if sample.numel() > 0 else 0.0
            if noise_std > 0:
                resized = resized + torch.randn_like(resized) * (noise_std * 0.05)
            resized[..., start : start + target_len] = cropped
        augmented.append(resized)

        if return_info:
            mask = torch.zeros((1, 1, 1, time_bins), device=x.device)
            mask[..., start : start + target_len] = 1.0
            mask = F.interpolate(mask, size=(freq_bins, time_bins), mode="nearest")
            info.append(
                {
                    "start": start,
                    "end": start + target_len,
                    "length": target_len,
                    "ratio": target_len / float(time_bins),
                    "mask": mask,
                    "forced": forced_window is not None,
                }
            )

    result = torch.cat(augmented, dim=0)
    if return_info:
        return result, info
    return result


def random_time_shift_tensor(
    x: torch.Tensor,
    max_ratio: float,
    return_info: bool = False,
):
    """
    Circularly shift the time axis (last dimension) by a random amount
    within [-max_ratio * T, max_ratio * T] bins for each sample.
    """
    if x.dim() != 4:
        return (x.clone(), [None] * x.size(0)) if return_info else x.clone()

    B, C, freq_bins, time_bins = x.shape
    max_ratio = max(0.0, min(max_ratio, 1.0))
    max_shift = int(round(time_bins * max_ratio))
    augmented = []
    info = [] if return_info else None

    for b in range(B):
        sample = x[b : b + 1]
        if max_shift == 0:
            shift = 0
        else:
            shift = random.randint(-max_shift, max_shift)
        shifted = torch.roll(sample, shifts=shift, dims=-1)
        augmented.append(shifted)

        if return_info:
            info.append(
                {
                    "shift": shift,
                    "shift_norm": shift / float(time_bins),
                }
            )

    result = torch.cat(augmented, dim=0)
    if return_info:
        return result, info
    return result
