"""Loader for the UCI Twin Gas Sensor Arrays dataset (UCI #361; Fonollosa et al., Sens. Actuators B 2016).

Files data1/B{unit}_G{gas}_F{conc}_R{rep}.txt: column 0 = time (s, 100 Hz, ~600 s), columns 1-8 = resistance
(kOhm) of the 8 MOX sensors of that unit. 5 nominally identical units (B1-B5), 4 gases
(CO, Ea = ethanol, Ey = ethylene, Me = methane), 10 concentration levels (F010..F100), repetitions R1-R4
(units 1-3) or R1-R2 (units 4-5). The UCI page lists 4/4/4/2/2 measurement days for units 1-5, so we
ASSUME repetition r = the r-th measurement day of that unit (to verify; not stated in the files).

Features (mirroring Vergara et al. 2012, as used in UCI #224), 8 per sensor -> 64:
  dR = R_steady - R0, dR/R0, and the extreme of the exponential moving average of the signal's
  increments during exposure (rise) and recovery (decay) for alpha = 0.1, 0.01, 0.001.
R0 = mean over the 10 s before onset; R_steady = mean over the last 10 s of exposure. Onset/offset are
detected per file as the steepest fall/rise of the channel-averaged relative resistance (smoothed 1 s).
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "twin_gas" / "data1"
CACHE = ROOT / "data" / "processed" / "twin_raw.npz"
FEAT = ROOT / "data" / "processed" / "twin_features.csv"
GASES = {"CO": 0, "Ea": 1, "Ey": 2, "Me": 3}
GAS_NAMES = {0: "CO", 1: "Ethanol", 2: "Ethylene", 3: "Methane"}
FS = 100
PAT = re.compile(r"B(\d)_G(\w\w)_F(\d+)_R(\d)\.txt")


T_GRID = np.arange(0, 600, 1 / FS)   # uniform 100 Hz grid, 0-599.99 s


def load_raw():
    """All recordings resampled (linear interpolation on the time column) onto a uniform 100 Hz grid,
    cached as float32. 82/640 files are irregularly sampled (as few as 19,289 rows for 600 s), so
    row index != time. 'n' = number of raw rows (kept for the audit)."""
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=False)
        return z["X"], pd.DataFrame({k: z[k] for k in ("unit", "gas", "conc", "rep", "n")})
    files = sorted(RAW.glob("*.txt"))
    meta = []
    X = np.empty((len(files), len(T_GRID), 8), np.float32)
    for i, f in enumerate(files):
        u, g, c, r = PAT.fullmatch(f.name).groups()
        a = pd.read_csv(f, sep=r"\s+", header=None, engine="c").to_numpy(np.float64)
        t = a[:, 0]
        for ch in range(8):
            X[i, :, ch] = np.interp(T_GRID, t, a[:, 1 + ch])
        meta.append((int(u), GASES[g], int(c), int(r), len(a)))
    m = pd.DataFrame(meta, columns=["unit", "gas", "conc", "rep", "n"])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, X=X, **{k: m[k].to_numpy() for k in m.columns})
    return X, m


def onset_offset(x):
    """(onset, offset) sample indices on the 100 Hz grid, from the channel-mean relative resistance
    (1 s moving average). Onset = steepest fall within 20-250 s; offset = steepest rise within
    onset+60 s .. onset+130 s (typical exposure is ~106 s; an unconstrained search failed for weak,
    slowly recovering low-concentration responses, and one file has a ~65 s exposure starting at ~150 s)."""
    n = len(x)
    rel = (x / x[: 20 * FS].mean(0)).mean(1)
    sm = np.convolve(rel, np.ones(FS) / FS, mode="same")
    d = np.diff(sm[::10])                                  # 10 Hz slope
    lo, hi = 20 * 10, 250 * 10
    on10 = lo + int(np.argmin(d[lo:hi]))
    a, b = on10 + 60 * 10, min(len(d), on10 + 130 * 10)
    off10 = a + int(np.argmax(d[a:b]))
    return on10 * 10, min(n - 1, off10 * 10)


def ema_extreme(sig, alpha, start, stop, mode):
    """Extreme of y[k] = (1-a) y[k-1] + a (r[k] - r[k-1]), y starting at 0 (vectorised with lfilter)."""
    inc = np.diff(sig[start:stop])
    if len(inc) == 0:
        return 0.0
    y = lfilter([alpha], [1.0, -(1.0 - alpha)], inc)
    return float(min(0.0, y.min())) if mode == "min" else float(max(0.0, y.max()))


def features():
    if FEAT.exists():
        return pd.read_csv(FEAT)
    X, m = load_raw()
    rows = []
    for i in range(len(X)):
        on, off = onset_offset(X[i])
        n = X.shape[1]                                     # uniform 100 Hz grid (60,000 samples)
        x = X[i].astype(np.float64)
        f = {}
        for s in range(8):
            r = x[:, s]
            R0 = r[max(0, on - 10 * FS):on].mean(); Rss = r[max(on, off - 10 * FS):off].mean()
            f[f"s{s}_dR"] = Rss - R0; f[f"s{s}_dRn"] = (Rss - R0) / R0
            for a in (0.1, 0.01, 0.001):
                f[f"s{s}_rise{a}"] = ema_extreme(r, a, on, off, "min")
                f[f"s{s}_decay{a}"] = ema_extreme(r, a, off, n, "max")
        rows.append(dict(unit=int(m.unit[i]), gas=int(m.gas[i]), conc=int(m.conc[i]), rep=int(m.rep[i]),
                         onset_s=on / FS, offset_s=off / FS, **f))
    df = pd.DataFrame(rows)
    df.to_csv(FEAT, index=False)
    return df


def feature_columns(df):
    return [c for c in df.columns if c.startswith("s")]


if __name__ == "__main__":
    df = features()
    print(df.shape)
    print(df.groupby("unit").size())
    print(df[["onset_s", "offset_s"]].describe().round(2))
