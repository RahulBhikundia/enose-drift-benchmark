"""Loader for the UCI Gas Sensor Array Drift dataset (UCI #224).

Each batchN.dat file is in libsvm format: "<label> 1:<v1> 2:<v2> ... 128:<v128>".
The 128 features are 16 sensors x 8 features per sensor (Vergara et al., 2012):
steady-state resistance change dR, normalised |dR|, and exponential-moving-average
rise and decay transients at three smoothing constants.

Row order inside each file is kept, because it may reflect acquisition order and we
want to be able to test for within-batch temporal clustering of classes.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UCI_DIR = ROOT / "data" / "raw" / "uci_gas_drift" / "Dataset"

GAS_NAMES = {1: "Ethanol", 2: "Ethylene", 3: "Ammonia",
             4: "Acetaldehyde", 5: "Acetone", 6: "Toluene"}
N_FEATURES = 128
N_BATCHES = 10


def _parse_libsvm_line(line: str) -> tuple[int, np.ndarray]:
    parts = line.split()
    label = int(float(parts[0].split(";")[0]))  # some mirrors use "label;conc"
    x = np.zeros(N_FEATURES, dtype=np.float64)
    for tok in parts[1:]:
        idx, val = tok.split(":")
        x[int(idx) - 1] = float(val)
    return label, x


def load_uci(data_dir: Path = UCI_DIR) -> pd.DataFrame:
    """Return a DataFrame with columns f0..f127, 'label' (0-5), 'gas', 'batch' (1-10),
    and 'row_in_batch' (original line position inside the batch file)."""
    rows = []
    for b in range(1, N_BATCHES + 1):
        with open(data_dir / f"batch{b}.dat") as fh:
            for i, line in enumerate(fh):
                if not line.strip():
                    continue
                label, x = _parse_libsvm_line(line)
                rows.append((b, i, label, x))
    X = np.stack([r[3] for r in rows])
    df = pd.DataFrame(X, columns=[f"f{j}" for j in range(N_FEATURES)])
    df["batch"] = [r[0] for r in rows]
    df["row_in_batch"] = [r[1] for r in rows]
    df["label"] = [r[2] - 1 for r in rows]          # 0-based for PyTorch
    df["gas"] = [GAS_NAMES[r[2]] for r in rows]
    return df


def feature_columns() -> list[str]:
    return [f"f{j}" for j in range(N_FEATURES)]


def sensor_feature_index(sensor: int, feat: int) -> int:
    """Column index for sensor in 0..15 and per-sensor feature in 0..7."""
    return sensor * 8 + feat


if __name__ == "__main__":
    df = load_uci()
    print(df.shape)
    print(df.groupby("batch").size())
