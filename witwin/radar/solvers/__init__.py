"""
Solver base class and backend implementations for radar signal computation.

Available solvers:
- PytorchSolver: pure PyTorch, fully differentiable via autograd
- SlangSolver: Slang CUDA kernels for high-throughput time-domain computation
- DirichletSolver: direct frequency-domain spectrum via Dirichlet kernel
"""


class Solver:
    """Abstract base class for radar chirp/frame/MIMO computation.

    Subclasses implement the actual signal generation using different backends
    (pure PyTorch, Slang CUDA kernels, Dirichlet kernel).
    """

    def __init__(self, radar):
        self.radar = radar
        self.device = radar.device

    def _pop_bool_option(self, options, name: str, default: bool = False) -> bool:
        value = options.pop(name, default)
        if not isinstance(value, bool):
            raise TypeError(f"Solver option '{name}' must be a bool.")
        return value

    def _ensure_no_options(self, options) -> None:
        if options:
            unsupported = ", ".join(sorted(options))
            raise TypeError(f"Unsupported solver options: {unsupported}")

    def chirp(self, distances, amplitudes):
        """Compute one chirp sweep.

        Args:
            distances: (N,) one-way range to each target (meters)
            amplitudes: (N,) reflectance / intensity per target

        Returns:
            Beat signal or spectrum, shape depends on backend.
        """
        raise NotImplementedError

    def frame(self, interpolator, t0=0):
        """Compute a single TX-RX pair frame.

        Args:
            interpolator: callable(t) -> (intensities, positions)
            t0: frame start time (seconds)

        Returns:
            (chirps_per_frame, adc_samples) complex tensor
        """
        raise NotImplementedError

    def mimo(self, interpolator, t0=0, **options):
        """Compute a full MIMO data cube.

        Args:
            interpolator: callable(t) -> (intensities, positions)
            t0: frame start time (seconds)

        Returns:
            (TX, RX, chirps_per_frame, adc_samples) complex tensor
        """
        raise NotImplementedError


# Import after Solver is defined to avoid circular imports
from .solver_pytorch import PytorchSolver
from .solver_slang import SlangSolver
from .solver_dirichlet import DirichletSolver

__all__ = [
    'Solver',
    'PytorchSolver',
    'SlangSolver',
    'DirichletSolver',
]
