"""Random-forest reference baseline on the same final splits (no tuning: 500 trees, sklearn defaults).

UCI: standardised 128 features (scaler fit on training batches). SmellNet: window summary features
(mean/std/min/max/slope per channel) on the same 100-step windows (stride 50) as the CNN.
Writes results/final/<ds>/<split>/rf_s<seed>.json in the same format as src/train.py.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data_smellnet import window_features  # noqa: E402
from src.train import load_data, metrics  # noqa: E402

OUT = ROOT / "results" / "final"
SPLITS = {"uci": ["main"] + [f"roll{k}" for k in range(2, 10)], "smell": [f"loso{h}" for h in range(1, 7)],
          "twin": [f"louo{h}" for h in range(1, 6)], "twinreg": [f"louo{h}" for h in range(1, 6)]}
N_SEEDS = {"uci": 5, "smell": 5, "twin": 10, "twinreg": 10}

for ds, splits in SPLITS.items():
    for split in splits:
        D = load_data(ds, split)
        Xtr, Xev = D["Xtr"], D["Xev"]
        if ds == "smell":
            Xtr, Xev = window_features(Xtr), window_features(Xev)
        for seed in range(N_SEEDS[ds]):
            out = OUT / ds / split / f"rf_s{seed}.json"
            if out.exists():
                continue
            RF = RandomForestRegressor if D["n_cls"] == 1 else RandomForestClassifier
            p = RF(500, n_jobs=-1, random_state=seed).fit(Xtr, D["ytr"]).predict(Xev)
            ysd = D.get("y_sd", 1.0)
            res = {"pooled": metrics(D["yev"], p, D["n_cls"], ysd)}
            for k in D["eval_domains"]:
                m = D["dev"] == k
                res[str(k)] = metrics(D["yev"][m], p[m], D["n_cls"], ysd)
            keys = ("r2", "rmse", "mae", "score") if D["n_cls"] == 1 else ("acc", "macro_f1_present", "bal_acc", "score")
            res["mean_over_domains"] = {key: float(np.mean([res[str(k)][key] for k in D["eval_domains"]])) for key in keys}
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(dict(dataset=ds, split=split, opt="rf", method="rf", seed=seed, role="test",
                                           final_eval=res, test_pred=p.tolist(), test_true=D["yev"].tolist(),
                                           test_dom=D["dev"].tolist())))
        print(ds, split, "done", flush=True)
