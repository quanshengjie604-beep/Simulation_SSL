"""
Fresnel reflection model for radar surface interaction.

Computes electromagnetic reflectance from incident angle and surface
normal using the Fresnel equations, all in drjit for GPU acceleration.
"""

import drjit as dr


def fresnel(cos_i, epsilon_r=5.0):
    """Compute Fresnel reflectance for unpolarized EM wave.

    Args:
        cos_i: mi.Float — cosine of incidence angle (|dot(-d, n)|)
        epsilon_r: relative permittivity of the reflecting material

    Returns:
        R: mi.Float — Fresnel reflectance [0, 1]
    """
    n2 = epsilon_r ** 0.5  # refractive index

    cos_i = dr.maximum(dr.minimum(cos_i, 1.0), 0.0)

    # Snell's law: cos(theta_t)
    sin_i_sq = 1.0 - cos_i * cos_i
    n2_sq = n2 * n2
    cos_t_sq = 1.0 - sin_i_sq / n2_sq
    cos_t = dr.sqrt(dr.maximum(cos_t_sq, 0.0))

    # s-polarization (TE) / p-polarization (TM)
    rs = (cos_i - n2 * cos_t) / (cos_i + n2 * cos_t)
    rp = (n2 * cos_i - cos_t) / (n2 * cos_i + cos_t)
    R = 0.5 * (rs * rs + rp * rp)

    # Total internal reflection
    R = dr.select(cos_t_sq < 0.0, 1.0, R)

    return R
