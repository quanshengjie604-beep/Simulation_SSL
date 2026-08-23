"""Top-level dispatcher.

Usage:
    python train.py [method=supervised|reconstruction|contrastive] [hydra overrides ...]

Default method is `supervised`. The dispatcher strips `method=` from sys.argv
before delegating to the per-method @hydra.main entry point, so Hydra never
sees an unknown override.
"""
from __future__ import annotations

import importlib
import sys

_METHODS = {
    "supervised":    "supervised.train_supervised",
    "reconstruction": "ssl_eval.mae_tune",
    "contrastive":   "ssl_eval.contrastive_tune",
}


def _pop_method(argv: list[str]) -> str:
    for i, tok in enumerate(argv[1:], start=1):
        if tok.startswith("method="):
            argv.pop(i)
            return tok.split("=", 1)[1]
    return "supervised"


def main() -> None:
    method = _pop_method(sys.argv)
    if method not in _METHODS:
        raise SystemExit(
            f"Unknown method={method!r}; choose from {sorted(_METHODS)}"
        )
    module = importlib.import_module(_METHODS[method])
    module.main()


if __name__ == "__main__":
    main()
