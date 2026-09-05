from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np

VACUUM_PERMITTIVITY = 8.8541878128e-12


def _coerce_scalar(value: float, *, name: str) -> float:
    return float(value)


def _coerce_nonnegative(value: float, *, name: str) -> float:
    scalar = float(value)
    if scalar < 0.0:
        raise ValueError(f"{name} must be >= 0.")
    return scalar


@dataclass(frozen=True)
class MaterialCapabilities:
    conductive: bool = False
    magnetic: bool = False
    anisotropic: bool = False
    dispersive: bool = False


@dataclass(frozen=True)
class StaticMaterialSample:
    eps_r: Any
    mu_r: Any = 1.0
    sigma_e: Any = 0.0


@dataclass(frozen=True)
class FrequencyMaterialSample:
    eps_r: Any
    mu_r: Any = 1.0
    sigma_e: Any = 0.0


@runtime_checkable
class MaterialSpec(Protocol):
    name: str | None

    def capabilities(self) -> MaterialCapabilities:
        ...

    def evaluate_static(self) -> StaticMaterialSample:
        ...

    def evaluate_at_frequency(self, frequency: float) -> FrequencyMaterialSample:
        ...


@dataclass(frozen=True, init=False)
class Material(MaterialSpec):
    eps_r: float
    mu_r: float
    sigma_e: float
    name: str | None

    def __init__(
        self,
        eps_r: float = 1.0,
        mu_r: float = 1.0,
        sigma_e: float = 0.0,
        name: str | None = None,
    ):
        object.__setattr__(self, "eps_r", _coerce_scalar(eps_r, name="eps_r"))
        object.__setattr__(self, "mu_r", _coerce_scalar(mu_r, name="mu_r"))
        object.__setattr__(self, "sigma_e", _coerce_nonnegative(sigma_e, name="sigma_e"))
        object.__setattr__(self, "name", None if name is None else str(name))

    def capabilities(self) -> MaterialCapabilities:
        return MaterialCapabilities(
            conductive=not np.isclose(self.sigma_e, 0.0),
            magnetic=not np.isclose(self.mu_r, 1.0),
            anisotropic=False,
            dispersive=False,
        )

    def evaluate_static(self) -> StaticMaterialSample:
        return StaticMaterialSample(
            eps_r=self.eps_r,
            mu_r=self.mu_r,
            sigma_e=self.sigma_e,
        )

    def evaluate_at_frequency(self, frequency: float) -> FrequencyMaterialSample:
        resolved_frequency = float(frequency)
        if resolved_frequency < 0.0:
            raise ValueError("frequency must be >= 0.")
        if resolved_frequency == 0.0:
            if not np.isclose(self.sigma_e, 0.0):
                raise ValueError("Conductive materials require frequency > 0.")
            return FrequencyMaterialSample(
                eps_r=complex(self.eps_r, 0.0),
                mu_r=self.mu_r,
                sigma_e=self.sigma_e,
            )

        eps_r = complex(
            self.eps_r,
            -self.sigma_e / (2.0 * np.pi * resolved_frequency * VACUUM_PERMITTIVITY),
        )
        return FrequencyMaterialSample(
            eps_r=eps_r,
            mu_r=self.mu_r,
            sigma_e=self.sigma_e,
        )


@dataclass(frozen=True, init=False)
class Structure:
    geometry: Any
    material: MaterialSpec
    name: str | None
    priority: int
    enabled: bool
    tags: tuple[str, ...]
    metadata: Mapping[str, Any]

    def __init__(
        self,
        geometry: Any,
        material: MaterialSpec,
        name: str | None = None,
        priority: int = 0,
        enabled: bool = True,
        tags=(),
        metadata: Mapping[str, Any] | None = None,
    ):
        if material is None:
            raise ValueError("Structure requires material=.")
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "material", material)
        object.__setattr__(self, "name", None if name is None else str(name))
        object.__setattr__(self, "priority", int(priority))
        object.__setattr__(self, "enabled", bool(enabled))
        object.__setattr__(self, "tags", tuple(str(tag) for tag in tags))
        object.__setattr__(self, "metadata", MappingProxyType(dict(metadata or {})))
