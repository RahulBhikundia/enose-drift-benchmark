"""Loader for SmellNet-Base (Feng et al., ICLR 2026), HF commit 45405fe.

Layout: base_data/{training,testing}/<substance>/<substance>_<session>.csv
Sessions 1-5 are in 'training', session 6 in 'testing'. Each CSV is one roughly
10-minute recording at about 1 Hz with 6 channels and no timestamp column.

ASSUMPTION (stated in data/README.md): session index = separate day, in chronological order.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SMELL_DIR = ROOT / "data" / "raw" / "smellnet" / "base_data"
CHANNELS = ["NO2", "C2H5OH", "VOC", "CO", "Alcohol", "LPG"]


def load_sessions(data_dir: Path = SMELL_DIR) -> list[dict]:
    """Return one dict per session: substance, session (1-6), split, data (T x 6 float array)."""
    out = []
    for split in ("training", "testing"):
        for f in sorted((data_dir / split).glob("*/*.csv")):
            substance = f.parent.name
            session = int(f.stem.rsplit("_", 1)[1])
            arr = pd.read_csv(f)[CHANNELS].to_numpy(dtype=np.float32)
            out.append(dict(substance=substance, session=session, split=split,
                            data=arr, path=str(f.relative_to(data_dir))))
    return out


def substance_index(sessions: list[dict]) -> dict[str, int]:
    names = sorted({s["substance"] for s in sessions})
    return {n: i for i, n in enumerate(names)}


def make_windows(arr: np.ndarray, win: int, stride: int, start: int = 0) -> np.ndarray:
    """Slice a (T, C) series into (N, win, C) windows."""
    idx = range(start, arr.shape[0] - win + 1, stride)
    return np.stack([arr[i:i + win] for i in idx]) if len(idx) else np.empty((0, win, arr.shape[1]))


def window_features(w: np.ndarray) -> np.ndarray:
    """Simple model-agnostic summary features per window (N, win, C) -> (N, 5C):
    mean, std, min, max, and least-squares slope per channel."""
    t = np.arange(w.shape[1], dtype=np.float32)
    t = (t - t.mean()) / (t.std() + 1e-8)
    slope = (w * t[None, :, None]).mean(axis=1)
    return np.concatenate([w.mean(1), w.std(1), w.min(1), w.max(1), slope], axis=1)


if __name__ == "__main__":
    s = load_sessions()
    print(len(s), "sessions;", len(substance_index(s)), "substances")
    lens = [x["data"].shape[0] for x in s]
    print("length min/median/max:", min(lens), int(np.median(lens)), max(lens))
