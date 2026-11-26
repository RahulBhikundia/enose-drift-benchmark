"""Re-train the frozen seed-0 runs of the four core optimizers and save their weights as checkpoints.

For every results/final/<dataset>/<split>/{sgd,adamw,sam,msam}_s0.json this script
  1. rebuilds the exact command-line arguments recorded in that JSON,
  2. trains the model again (src.train.run),
  3. saves the weights to checkpoints/<dataset>/<split>/<opt>_s0.pt,
  4. compares the new test score with the stored one and writes checkpoints/manifest.csv.
The comparison doubles as a re-run reproducibility check (see the report, Sec. "Reproducibility").

  python scripts/experiments/export_checkpoints.py            # export + verify (about 10 min on a GTX 1650)
  python scripts/experiments/export_checkpoints.py --load     # reload every saved .pt, re-evaluate, check against manifest
"""
import csv
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402

from src.models import build_model  # noqa: E402
from src.train import DEV, eval_all, load_data, parse, run  # noqa: E402

OPTS = ("sgd", "adamw", "sam", "msam")
CK = ROOT / "checkpoints"
MAN = CK / "manifest.csv"


def score(fe):
    m = fe["mean_over_domains"]
    return m["r2"] if "r2" in m else m["macro_f1_present"]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def export():
    rows = []
    jobs = sorted(f for f in (ROOT / "results" / "final").glob("*/*/*_s0.json") if f.stem[:-3] in OPTS)
    tmp = Path(tempfile.mkdtemp())
    for i, f in enumerate(jobs):
        j = json.loads(f.read_text())
        if j.get("diverged"):
            print("skip diverged", f); continue
        ds, split, opt = f.parent.parent.name, f.parent.name, j["opt"]
        argv = ["--dataset", ds, "--split", split, "--opt", opt, "--hp", json.dumps(j["hp"]), "--seed", "0",
                "--epochs", str(j["epochs"]), "--bs", str(j["bs"]), "--method", j["method"],
                "--mhp", json.dumps(j["mhp"]), "--sampler", j["sampler"], "--out", str(tmp / "x.json")]
        out, model, _ = run(parse(argv), return_model=True)
        dst = CK / ds / split / f"{opt}_s0.pt"; dst.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "args": argv[:-2], "source_json": str(f.relative_to(ROOT))}, dst)
        old, new = score(j["final_eval"]), score(out["final_eval"])
        rows.append(dict(dataset=ds, split=split, opt=opt, file=str(dst.relative_to(ROOT)).replace("\\", "/"),
                         bytes=dst.stat().st_size, sha256_16=sha(dst), reported_score=round(old, 6),
                         rerun_score=round(new, 6), abs_diff=round(abs(new - old), 6)))
        print(f"[{i + 1}/{len(jobs)}] {ds}/{split}/{opt}: reported {old:.4f} rerun {new:.4f}", flush=True)
    with MAN.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    d = [r["abs_diff"] for r in rows]
    print(f"DONE {len(rows)} checkpoints; max |diff| {max(d):.6f}; exact {sum(x == 0 for x in d)}/{len(d)}")


def load():
    rows = list(csv.DictReader(MAN.open()))
    worst = 0.0
    for r in rows:
        ck = torch.load(ROOT / r["file"], map_location=DEV)
        D = load_data(r["dataset"], r["split"])
        model = build_model(r["dataset"]).to(DEV); model.load_state_dict(ck["state_dict"])
        Xev = torch.tensor(D["Xev"], dtype=torch.float32, device=DEV)
        s = score(eval_all(model, D, Xev, D["n_cls"])[0])
        worst = max(worst, abs(s - float(r["rerun_score"])))
        print(f"{r['file']}: {s:.4f} (manifest {float(r['rerun_score']):.4f})")
    print(f"LOADED {len(rows)}; max |diff| vs manifest {worst:.2e}")


if __name__ == "__main__":
    load() if "--load" in sys.argv else export()
