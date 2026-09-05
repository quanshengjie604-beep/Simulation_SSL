"""Base Scene class for all solver-specific scenes."""


class SceneBase:
    """Base scene for all solver-specific scenes.

    Solver-specific scenes (e.g. maxwell.Scene, radar.Scene) extend this
    with domain-specific fields and methods.
    """

    def __init__(
        self,
        *,
        structures=None,
        sources=None,
        monitors=None,
        metadata=None,
        device="cuda",
        verbose=False,
    ):
        self.structures = list(structures or [])
        self.sources = list(sources or [])
        self.monitors = list(monitors or [])
        self.metadata = dict(metadata or {})
        self.device = device
        self.verbose = verbose

    def add_structure(self, structure):
        self.structures.append(structure)
        return self

    def add_source(self, source):
        self.sources.append(source)
        return self

    def add_monitor(self, monitor):
        self.monitors.append(monitor)
        return self
