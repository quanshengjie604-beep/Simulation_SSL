#!/usr/bin/env python3
"""Convert legacy SMPL Python PKLs into smplx-compatible model files."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from witwin.core.geometry.smpl import _ChRecon, _Unpickler


MODEL_MAP = {
    "basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl": "SMPL_NEUTRAL.pkl",
    "basicmodel_m_lbs_10_207_0_v1.1.0.pkl": "SMPL_MALE.pkl",
    "basicmodel_f_lbs_10_207_0_v1.1.0.pkl": "SMPL_FEMALE.pkl",
}


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        default=str(REPO_ROOT / "smpl_models" / "smpl_source" / "SMPL_python_v.1.1.0" / "smpl" / "models"),
        help="Directory containing legacy basicmodel_* SMPL pkl files.",
    )
    parser.add_argument(
        "--out-root",
        default=str(REPO_ROOT / "smpl_models" / "smplx_compatible"),
        help="Output root. The script writes files under <out-root>/smpl/.",
    )
    return parser.parse_args()


def load_legacy_pkl(path: Path) -> dict:
    with path.open("rb") as handle:
        data = _Unpickler(handle, encoding="latin1").load()
    for key, value in list(data.items()):
        if isinstance(value, _ChRecon):
            data[key] = value._data
    return data


def main() -> int:
    args = parse_args()
    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_root) / "smpl"
    out_dir.mkdir(parents=True, exist_ok=True)

    for src_name, dst_name in MODEL_MAP.items():
        src = source_dir / src_name
        if not src.exists():
            raise FileNotFoundError(src)
        dst = out_dir / dst_name
        data = load_legacy_pkl(src)
        with dst.open("wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
