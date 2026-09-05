"""
CFAR (Constant False Alarm Rate) detectors for Range-Doppler maps.

CA-CFAR and OS-CFAR operate on 2D RD magnitude maps. Unlike the top-K energy
detector, CFAR adapts the threshold locally so weaker reflections (e.g. from
limbs) can be detected as long as they stand out from their local noise floor.
"""

import torch
import torch.nn.functional as F


def _real_values(rd_map: torch.Tensor) -> torch.Tensor:
    real_dtype = torch.float64 if rd_map.dtype in {torch.float64, torch.complex128} else torch.float32
    values = torch.abs(rd_map) if torch.is_complex(rd_map) else rd_map
    return values.to(real_dtype)


def _replicate_pad(data: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
    return F.pad(data.unsqueeze(0).unsqueeze(0), (pad_w, pad_w, pad_h, pad_h), mode="replicate").squeeze(0).squeeze(0)


def _integral_rect_sum(
    integral: torch.Tensor,
    r0: torch.Tensor,
    c0: torch.Tensor,
    r1: torch.Tensor,
    c1: torch.Tensor,
) -> torch.Tensor:
    return integral[r1 + 1, c1 + 1] - integral[r0, c1 + 1] - integral[r1 + 1, c0] + integral[r0, c0]


def ca_cfar_2d(rd_map: torch.Tensor, guard_cells=(2, 3), training_cells=(4, 6), pfa: float = 1e-3):
    """Cell-Averaging CFAR on a 2D Range-Doppler magnitude map.

    For each cell under test (CUT), the threshold is computed from the average
    power in a rectangular ring of training cells surrounding the CUT, with a
    guard band in between to avoid target leakage.

    Args:
        rd_map: (Nd, Nr) magnitude tensor (linear or dB scale, real or complex).
        guard_cells: (gd, gr) guard band half-size in Doppler and Range.
        training_cells: (td, tr) training band half-size in Doppler and Range.
        pfa: probability of false alarm, controls the scaling factor alpha.

    Returns:
        detections: (Nd, Nr) boolean mask of detected cells.
        threshold_map: (Nd, Nr) adaptive threshold at each cell.
    """
    Nd, Nr = rd_map.shape
    gd, gr = guard_cells
    td, tr = training_cells

    outer_d, outer_r = gd + td, gr + tr
    n_train = (2 * outer_d + 1) * (2 * outer_r + 1) - (2 * gd + 1) * (2 * gr + 1)

    alpha = n_train * (pfa ** (-1.0 / n_train) - 1.0)

    values = _real_values(rd_map)
    device = values.device

    padded = _replicate_pad(values, outer_d, outer_r)
    integral = F.pad(padded, (1, 0, 1, 0), mode="constant", value=0).cumsum(dim=0).cumsum(dim=1)

    row = torch.arange(Nd, device=device, dtype=torch.int64).view(-1, 1)
    col = torch.arange(Nr, device=device, dtype=torch.int64).view(1, -1)
    pi = row + outer_d
    pj = col + outer_r

    outer_sum = _integral_rect_sum(integral, pi - outer_d, pj - outer_r, pi + outer_d, pj + outer_r)
    guard_sum = _integral_rect_sum(integral, pi - gd, pj - gr, pi + gd, pj + gr)
    noise_est = (outer_sum - guard_sum) / n_train
    threshold_map = alpha * noise_est
    detections = values > threshold_map
    return detections, threshold_map


def ca_cfar_2d_fast(rd_map: torch.Tensor, guard_cells=(2, 3), training_cells=(4, 6), pfa: float = 1e-3):
    """Vectorized CA-CFAR using pooled averages for speed.

    Equivalent to ca_cfar_2d but ~100x faster for typical RD map sizes.
    """
    gd, gr = guard_cells
    td, tr = training_cells

    outer_d, outer_r = gd + td, gr + tr
    n_outer = (2 * outer_d + 1) * (2 * outer_r + 1)
    n_guard = (2 * gd + 1) * (2 * gr + 1)
    n_train = n_outer - n_guard

    alpha = n_train * (pfa ** (-1.0 / n_train) - 1.0)

    values = _real_values(rd_map)

    def _pool_nearest(data: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        pad_h = size[0] // 2
        pad_w = size[1] // 2
        padded = F.pad(data.unsqueeze(0).unsqueeze(0), (pad_w, pad_w, pad_h, pad_h), mode="replicate")
        return F.avg_pool2d(padded, kernel_size=size, stride=1).squeeze(0).squeeze(0)

    outer_mean = _pool_nearest(values, (2 * outer_d + 1, 2 * outer_r + 1))
    guard_mean = _pool_nearest(values, (2 * gd + 1, 2 * gr + 1))

    noise_est = (outer_mean * n_outer - guard_mean * n_guard) / n_train
    threshold_map = alpha * noise_est
    detections = values > threshold_map
    return detections, threshold_map


def os_cfar_2d(
    rd_map: torch.Tensor,
    guard_cells=(2, 3),
    training_cells=(4, 6),
    rank_fraction: float = 0.75,
    pfa: float = 1e-3,
):
    """Ordered-Statistic CFAR on a 2D Range-Doppler map.

    More robust than CA-CFAR in multi-target environments. Uses the k-th
    ordered sample from the training cells as the noise estimate.
    Slower than CA-CFAR but better at separating closely spaced targets.
    """
    Nd, Nr = rd_map.shape
    gd, gr = guard_cells
    td, tr = training_cells
    outer_d, outer_r = gd + td, gr + tr

    n_guard = (2 * gd + 1) * (2 * gr + 1)
    n_outer = (2 * outer_d + 1) * (2 * outer_r + 1)
    n_train = n_outer - n_guard
    k = int(rank_fraction * n_train)

    alpha = n_train * (pfa ** (-1.0 / n_train) - 1.0)

    values = _real_values(rd_map)

    mask_2d = torch.ones((2 * outer_d + 1, 2 * outer_r + 1), dtype=torch.bool, device=values.device)
    mask_2d[td:td + 2 * gd + 1, tr:tr + 2 * gr + 1] = False
    mask_flat = mask_2d.reshape(-1)

    padded = _replicate_pad(values, outer_d, outer_r)
    patches = F.unfold(
        padded.unsqueeze(0).unsqueeze(0),
        kernel_size=(2 * outer_d + 1, 2 * outer_r + 1),
        stride=1,
    ).squeeze(0).transpose(0, 1)
    training = patches[:, mask_flat]
    training_sorted, _ = torch.sort(training, dim=1)

    k_idx = min(k, training_sorted.shape[1] - 1)
    noise_est = training_sorted[:, k_idx]
    threshold_map = (alpha * noise_est).reshape(Nd, Nr)
    detections = values > threshold_map
    return detections, threshold_map
