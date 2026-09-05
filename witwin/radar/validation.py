"""Centralized validation for radar configuration.

All config validation lives here. Dataclasses are pure data containers;
any parsing/validation of dict-shaped configuration goes through the
``validate_*`` functions exposed by this module.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Iterable

from .utils.antenna import DEFAULT_DIPOLE_ANGLES_DEG, DEFAULT_DIPOLE_VALUES

if TYPE_CHECKING:
    from .radar import RadarConfig


# ---------------------------------------------------------------------------
# Primitive validators
# ---------------------------------------------------------------------------

def _finite_float(name: str, value: Any, prefix: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{prefix} '{name}' must be a finite float.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{prefix} '{name}' must be a finite float.")
    return parsed


def _non_negative_float(name: str, value: Any, prefix: str) -> float:
    parsed = _finite_float(name, value, prefix)
    if parsed < 0.0:
        raise ValueError(f"{prefix} '{name}' must be non-negative.")
    return parsed


def _positive_float(name: str, value: Any, prefix: str) -> float:
    parsed = _finite_float(name, value, prefix)
    if parsed <= 0.0:
        raise ValueError(f"{prefix} '{name}' must be positive.")
    return parsed


def _positive_int(name: str, value: Any, prefix: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{prefix} '{name}' must be a positive int.")
    return value


def _optional_seed(value: Any, name: str, prefix: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{prefix} '{name}' must be a non-negative int.")
    return value


def _require_keys(config: dict[str, Any], keys: Iterable[str], label: str) -> None:
    missing = [key for key in keys if key not in config]
    if missing:
        raise ValueError(f"{label} is missing required keys: {', '.join(missing)}")


def _parse_vector3(
    name: str,
    value: Any,
    *,
    prefix: str,
    aliases: dict[str, tuple[float, float, float]] | None = None,
) -> tuple[float, float, float]:
    if isinstance(value, str):
        if aliases is None or value.lower() not in aliases:
            raise ValueError(
                f"{prefix} '{name}' must be a 3-element vector"
                + (" or an alias string." if aliases else ".")
            )
        return aliases[value.lower()]
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{prefix} '{name}' must be a 3-element vector.")
    vector = tuple(
        _finite_float(f"{name}[{i}]", component, prefix)
        for i, component in enumerate(value)
    )
    norm_sq = sum(c * c for c in vector)
    if norm_sq <= 1e-24:
        raise ValueError(f"{prefix} '{name}' must be non-zero.")
    return vector


# ---------------------------------------------------------------------------
# Antenna pattern
# ---------------------------------------------------------------------------

_ANTENNA_PREFIX = "Antenna pattern field"


def _validate_axis(name: str, value: Any) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{_ANTENNA_PREFIX} '{name}' must be a sequence of angles in degrees.")
    if len(value) < 2:
        raise ValueError(f"{_ANTENNA_PREFIX} '{name}' must contain at least 2 samples.")
    axis = tuple(_finite_float(f"{name}[{i}]", angle, _ANTENNA_PREFIX) for i, angle in enumerate(value))
    for i in range(1, len(axis)):
        if axis[i] <= axis[i - 1]:
            raise ValueError(f"{_ANTENNA_PREFIX} '{name}' must be strictly increasing.")
    return axis


def _validate_values_1d(name: str, value: Any, expected_count: int) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{_ANTENNA_PREFIX} '{name}' must be a sequence of gain values.")
    if len(value) != expected_count:
        raise ValueError(
            f"{_ANTENNA_PREFIX} '{name}' must contain exactly {expected_count} entries; got {len(value)}."
        )
    gains = []
    for i, item in enumerate(value):
        gain = _finite_float(f"{name}[{i}]", item, _ANTENNA_PREFIX)
        if gain < 0.0:
            raise ValueError(f"{_ANTENNA_PREFIX} '{name}[{i}]' must be non-negative.")
        gains.append(gain)
    return tuple(gains)


def _validate_values_2d(
    name: str, value: Any, expected_rows: int, expected_cols: int
) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{_ANTENNA_PREFIX} '{name}' must be a 2D sequence of gain values.")
    if len(value) != expected_rows:
        raise ValueError(
            f"{_ANTENNA_PREFIX} '{name}' must contain exactly {expected_rows} rows; got {len(value)}."
        )
    rows = []
    for row_index, row in enumerate(value):
        if not isinstance(row, (list, tuple)):
            raise ValueError(f"{_ANTENNA_PREFIX} '{name}[{row_index}]' must be a sequence of gain values.")
        if len(row) != expected_cols:
            raise ValueError(
                f"{_ANTENNA_PREFIX} '{name}[{row_index}]' must contain exactly {expected_cols} entries; got {len(row)}."
            )
        parsed_row = []
        for col_index, item in enumerate(row):
            gain = _finite_float(f"{name}[{row_index}][{col_index}]", item, _ANTENNA_PREFIX)
            if gain < 0.0:
                raise ValueError(
                    f"{_ANTENNA_PREFIX} '{name}[{row_index}][{col_index}]' must be non-negative."
                )
            parsed_row.append(gain)
        rows.append(tuple(parsed_row))
    return tuple(rows)


def _detect_antenna_kind(data: dict[str, Any]) -> str:
    raw_kind = data.get("kind")
    if raw_kind is None:
        if "values" in data:
            return "map"
        if "x_values" in data or "y_values" in data:
            return "separable"
        raise ValueError("Antenna pattern config must define 'kind' or provide fields for a known pattern type.")
    kind = str(raw_kind)
    if kind not in {"separable", "map"}:
        raise ValueError("Antenna pattern field 'kind' must be 'separable' or 'map'.")
    return kind


def validate_antenna_pattern_config(config: dict[str, Any]) -> dict[str, Any]:
    kind = _detect_antenna_kind(config)
    x_angles_deg = _validate_axis("x_angles_deg", config.get("x_angles_deg"))
    y_angles_deg = _validate_axis("y_angles_deg", config.get("y_angles_deg"))

    if kind == "separable":
        x_values = _validate_values_1d("x_values", config.get("x_values"), len(x_angles_deg))
        y_values = _validate_values_1d("y_values", config.get("y_values"), len(y_angles_deg))
        return {
            "kind": kind,
            "x_angles_deg": list(x_angles_deg),
            "y_angles_deg": list(y_angles_deg),
            "x_values": list(x_values),
            "y_values": list(y_values),
        }

    values = _validate_values_2d("values", config.get("values"), len(y_angles_deg), len(x_angles_deg))
    return {
        "kind": kind,
        "x_angles_deg": list(x_angles_deg),
        "y_angles_deg": list(y_angles_deg),
        "values": [list(row) for row in values],
    }


def default_dipole_antenna_pattern() -> dict[str, Any]:
    return {
        "kind": "separable",
        "x_angles_deg": list(DEFAULT_DIPOLE_ANGLES_DEG),
        "y_angles_deg": list(DEFAULT_DIPOLE_ANGLES_DEG),
        "x_values": list(DEFAULT_DIPOLE_VALUES),
        "y_values": list(DEFAULT_DIPOLE_VALUES),
    }


# ---------------------------------------------------------------------------
# Noise model
# ---------------------------------------------------------------------------

_NOISE_PREFIX = "Noise model field"


def _validate_thermal(config: dict[str, Any]) -> dict[str, Any]:
    return {"std": _non_negative_float("thermal.std", config.get("std"), _NOISE_PREFIX)}


def _validate_quantization(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "bits": _positive_int("quantization.bits", config.get("bits"), _NOISE_PREFIX),
        "full_scale": _positive_float(
            "quantization.full_scale", config.get("full_scale", 1.0), _NOISE_PREFIX
        ),
    }


def _validate_phase(config: dict[str, Any]) -> dict[str, Any]:
    return {"std": _non_negative_float("phase.std", config.get("std"), _NOISE_PREFIX)}


def validate_noise_model_config(config: dict[str, Any]) -> dict[str, Any]:
    thermal = _validate_thermal(config["thermal"]) if config.get("thermal") is not None else None
    quantization = (
        _validate_quantization(config["quantization"])
        if config.get("quantization") is not None
        else None
    )
    phase = _validate_phase(config["phase"]) if config.get("phase") is not None else None
    seed = _optional_seed(config.get("seed"), "seed", _NOISE_PREFIX)

    if thermal is None and quantization is None and phase is None:
        raise ValueError(
            "Noise model config must enable at least one of 'thermal', 'quantization', or 'phase'."
        )

    normalized: dict[str, Any] = {}
    if thermal is not None:
        normalized["thermal"] = thermal
    if quantization is not None:
        normalized["quantization"] = quantization
    if phase is not None:
        normalized["phase"] = phase
    if seed is not None:
        normalized["seed"] = seed
    return normalized


# ---------------------------------------------------------------------------
# Polarization
# ---------------------------------------------------------------------------

_POLARIZATION_PREFIX = "Polarization field"
_POLARIZATION_ALIASES = {
    "horizontal": (1.0, 0.0, 0.0),
    "h": (1.0, 0.0, 0.0),
    "vertical": (0.0, 1.0, 0.0),
    "v": (0.0, 1.0, 0.0),
}


def _is_single_vector_spec(value: Any) -> bool:
    if isinstance(value, str):
        return True
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return False
    return all(not isinstance(component, (list, tuple)) for component in value)


def _vector_bank(name: str, value: Any, expected_count: int) -> tuple[tuple[float, float, float], ...]:
    if _is_single_vector_spec(value):
        vector = _parse_vector3(name, value, prefix=_POLARIZATION_PREFIX, aliases=_POLARIZATION_ALIASES)
        return tuple(vector for _ in range(expected_count))
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"{_POLARIZATION_PREFIX} '{name}' must be a vector or a sequence of {expected_count} vectors."
        )
    if len(value) != expected_count:
        raise ValueError(
            f"{_POLARIZATION_PREFIX} '{name}' must contain exactly {expected_count} entries; got {len(value)}."
        )
    return tuple(
        _parse_vector3(f"{name}[{i}]", entry, prefix=_POLARIZATION_PREFIX, aliases=_POLARIZATION_ALIASES)
        for i, entry in enumerate(value)
    )


def validate_polarization_config(
    config: dict[str, Any], *, num_tx: int, num_rx: int
) -> dict[str, Any]:
    allowed = {"tx", "rx", "reflection_flip"}
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise TypeError(f"Unsupported polarization config keys: {', '.join(unknown)}")

    tx_value = config.get("tx")
    rx_value = config.get("rx")
    if tx_value is None and rx_value is None:
        raise ValueError("Polarization config must define at least one of 'tx' or 'rx'.")
    if tx_value is None:
        tx_value = rx_value
    if rx_value is None:
        rx_value = tx_value

    return {
        "tx": [list(vector) for vector in _vector_bank("tx", tx_value, num_tx)],
        "rx": [list(vector) for vector in _vector_bank("rx", rx_value, num_rx)],
        "reflection_flip": bool(config.get("reflection_flip", True)),
    }


# ---------------------------------------------------------------------------
# Receiver chain
# ---------------------------------------------------------------------------

_RECEIVER_PREFIX = "Receiver chain field"


def _validate_lna(config: dict[str, Any]) -> dict[str, Any]:
    return {"gain_db": _finite_float("lna.gain_db", config.get("gain_db", 0.0), _RECEIVER_PREFIX)}


def _validate_agc(config: dict[str, Any]) -> dict[str, Any]:
    mode = str(config.get("mode", "per_rx")).lower()
    if mode not in {"global", "per_rx"}:
        raise ValueError("Receiver chain field 'agc.mode' must be 'global' or 'per_rx'.")
    max_gain_db = _finite_float("agc.max_gain_db", config.get("max_gain_db", 60.0), _RECEIVER_PREFIX)
    min_gain_db = _finite_float("agc.min_gain_db", config.get("min_gain_db", -60.0), _RECEIVER_PREFIX)
    if min_gain_db > max_gain_db:
        raise ValueError("Receiver chain AGC requires min_gain_db <= max_gain_db.")
    return {
        "target_rms": _positive_float("agc.target_rms", config.get("target_rms"), _RECEIVER_PREFIX),
        "max_gain_db": max_gain_db,
        "min_gain_db": min_gain_db,
        "mode": mode,
    }


def validate_receiver_chain_config(config: dict[str, Any]) -> dict[str, Any]:
    lna = _validate_lna(config["lna"]) if config.get("lna") is not None else None
    agc = _validate_agc(config["agc"]) if config.get("agc") is not None else None
    adc = _validate_quantization(config["adc"]) if config.get("adc") is not None else None
    if lna is None and agc is None and adc is None:
        raise ValueError("Receiver chain config must enable at least one of 'lna', 'agc', or 'adc'.")
    normalized: dict[str, Any] = {
        "reference_impedance_ohm": _positive_float(
            "receiver_chain.reference_impedance_ohm",
            config.get("reference_impedance_ohm", 50.0),
            _RECEIVER_PREFIX,
        )
    }
    if lna is not None:
        normalized["lna"] = lna
    if agc is not None:
        normalized["agc"] = agc
    if adc is not None:
        normalized["adc"] = adc
    return normalized


# ---------------------------------------------------------------------------
# Radar config
# ---------------------------------------------------------------------------

_RADAR_PREFIX = "Radar config field"

_RADAR_REQUIRED_KEYS = (
    "num_tx",
    "num_rx",
    "fc",
    "slope",
    "adc_samples",
    "adc_start_time",
    "sample_rate",
    "idle_time",
    "ramp_end_time",
    "chirp_per_frame",
    "frame_per_second",
    "num_doppler_bins",
    "num_range_bins",
    "num_angle_bins",
    "power",
    "tx_loc",
    "rx_loc",
)


def _validate_antenna_locations(
    name: str, value: Any, expected_count: int
) -> tuple[tuple[float, float, float], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{_RADAR_PREFIX} '{name}' must be a sequence of 3D coordinates.")
    if len(value) != expected_count:
        raise ValueError(
            f"{_RADAR_PREFIX} '{name}' must contain exactly {expected_count} entries; got {len(value)}."
        )
    coords: list[tuple[float, float, float]] = []
    for index, coord in enumerate(value):
        if not isinstance(coord, (list, tuple)) or len(coord) != 3:
            raise ValueError(f"{_RADAR_PREFIX} '{name}[{index}]' must be a 3-element coordinate.")
        coords.append(
            (
                _finite_float(f"{name}[{index}][0]", coord[0], _RADAR_PREFIX),
                _finite_float(f"{name}[{index}][1]", coord[1], _RADAR_PREFIX),
                _finite_float(f"{name}[{index}][2]", coord[2], _RADAR_PREFIX),
            )
        )
    return tuple(coords)


def validate_radar_config(config: dict[str, Any]) -> RadarConfig:
    from .radar import RadarConfig

    _require_keys(config, _RADAR_REQUIRED_KEYS, "Radar config")

    num_tx = _positive_int("num_tx", config["num_tx"], _RADAR_PREFIX)
    num_rx = _positive_int("num_rx", config["num_rx"], _RADAR_PREFIX)

    antenna_pattern = (
        validate_antenna_pattern_config(config["antenna_pattern"])
        if config.get("antenna_pattern") is not None
        else None
    )
    noise_model = (
        validate_noise_model_config(config["noise_model"])
        if config.get("noise_model") is not None
        else None
    )
    polarization = (
        validate_polarization_config(config["polarization"], num_tx=num_tx, num_rx=num_rx)
        if config.get("polarization") is not None
        else None
    )
    receiver_chain = (
        validate_receiver_chain_config(config["receiver_chain"])
        if config.get("receiver_chain") is not None
        else None
    )

    return RadarConfig(
        num_tx=num_tx,
        num_rx=num_rx,
        fc=_finite_float("fc", config["fc"], _RADAR_PREFIX),
        slope=_finite_float("slope", config["slope"], _RADAR_PREFIX),
        adc_samples=_positive_int("adc_samples", config["adc_samples"], _RADAR_PREFIX),
        adc_start_time=_finite_float("adc_start_time", config["adc_start_time"], _RADAR_PREFIX),
        sample_rate=_finite_float("sample_rate", config["sample_rate"], _RADAR_PREFIX),
        idle_time=_finite_float("idle_time", config["idle_time"], _RADAR_PREFIX),
        ramp_end_time=_finite_float("ramp_end_time", config["ramp_end_time"], _RADAR_PREFIX),
        chirp_per_frame=_positive_int("chirp_per_frame", config["chirp_per_frame"], _RADAR_PREFIX),
        frame_per_second=_finite_float("frame_per_second", config["frame_per_second"], _RADAR_PREFIX),
        num_doppler_bins=_positive_int("num_doppler_bins", config["num_doppler_bins"], _RADAR_PREFIX),
        num_range_bins=_positive_int("num_range_bins", config["num_range_bins"], _RADAR_PREFIX),
        num_angle_bins=_positive_int("num_angle_bins", config["num_angle_bins"], _RADAR_PREFIX),
        power=_finite_float("power", config["power"], _RADAR_PREFIX),
        tx_loc=_validate_antenna_locations("tx_loc", config["tx_loc"], num_tx),
        rx_loc=_validate_antenna_locations("rx_loc", config["rx_loc"], num_rx),
        antenna_pattern=antenna_pattern,
        noise_model=noise_model,
        polarization=polarization,
        receiver_chain=receiver_chain,
    )
