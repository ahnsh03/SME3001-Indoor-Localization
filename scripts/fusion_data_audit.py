"""원본 Train/ 검증 표 경로 및 행 수(정제 후, drop 없음)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "train"
VAL = ROOT / "data" / "validation"


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def infer_cols(df: pd.DataFrame):
    cols = df.columns.tolist()
    return cols[:2], cols[2:8], cols[8:14]


def main() -> int:
    from fusion_realtime_sanitize import (
        resolve_train_kgh_corrected_paths,
        resolve_validation_paths,
        standardize_loaded_median_variance,
    )

    tm, tv = resolve_train_kgh_corrected_paths(TRAIN)
    med, var, *_ = standardize_loaded_median_variance(tm, tv, coerce_numeric, infer_cols)
    print(f"Train: {tm.name} / {tv.name}")
    print(f"  → sanitize 후 행 수: {len(med)}")
    vm, vv = resolve_validation_paths(VAL)
    md, vr, *_ = standardize_loaded_median_variance(vm, vv, coerce_numeric, infer_cols)
    print(f"Validation: {vm.name} / {vv.name}")
    print(f"  → sanitize 후 행 수: {len(md)}")
    return 0


if __name__ == "__main__":
    from pathlib import Path

    from script_run_io import cli_entrypoint

    cli_entrypoint(Path(__file__), main)
