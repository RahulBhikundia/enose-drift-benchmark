"""Write the two generated tables of the report's reproducibility section.

  reports/cvpr/commits.tex : commit hash, timestamp and message for every commit (from `git log`)
  reports/cvpr/rerun.tex   : re-training check of the shipped checkpoints (from checkpoints/manifest.csv)
                             against the seed-to-seed spread of the reported runs (results/final)

  python scripts/report/make_repro_tex.py
"""
import csv
import json
import statistics as st
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "cvpr"
NAMES = {"uci": "UCI (ageing)", "smell": "SmellNet", "twin": "Twin gas", "twinreg": "Twin conc."}


def tex(s):
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("$", r"\$")):
        s = s.replace(a, b)
    return s


def commits():
    log = subprocess.run(["git", "log", "--reverse", "--date=format:%d %b %H:%M", "--format=%h|%ad|%an|%s"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip().splitlines()
    rows = [l.split("|", 3) for l in log]
    who = lambda a: "Rahul" if "Rahul" in a else "Gourav" if "Gourav" in a else "Jharendra" if "Jharendra" in a.capitalize() else a
    body = "\n".join(rf"\texttt{{{h}}} & {d} & {who(a)} & {tex(s)} \\" for h, d, a, s in rows)
    (OUT / "commits.tex").write_text(
        "\\begin{tabular}{@{}l l l p{0.40\\linewidth}@{}}\n\\toprule\nCommit & Time (IST) & Author & Content \\\\\n"
        "\\midrule\n"
        + body + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
    print(f"commits.tex: {len(rows)} commits")


def score(fe):
    m = fe["mean_over_domains"]
    return m["r2"] if "r2" in m else m["macro_f1_present"]


def rerun():
    man = list(csv.DictReader((ROOT / "checkpoints" / "manifest.csv").open()))
    diff, spread, size = defaultdict(list), defaultdict(list), defaultdict(int)
    for r in man:
        ds = r["dataset"]; diff[ds].append(float(r["abs_diff"])); size[ds] += int(r["bytes"])
        seeds = [score(json.loads(f.read_text())["final_eval"]) for f in
                 sorted((ROOT / "results" / "final" / ds / r["split"]).glob(f"{r['opt']}_s*.json"))
                 if not json.loads(f.read_text()).get("diverged")]
        if len(seeds) > 1:
            spread[ds].append(st.stdev(seeds))
    lines = []
    for ds in ("uci", "smell", "twin", "twinreg"):
        d = diff[ds]
        lines.append(f"{NAMES[ds]} & {len(d)} & {st.median(d):.3f} & {max(d):.3f} & {st.median(spread[ds]):.3f} & "
                     f"{size[ds] / 1e6:.1f} \\\\")
    (OUT / "rerun.tex").write_text(
        "\\begin{tabular}{@{}lrrrrr@{}}\n\\toprule\n & & \\multicolumn{2}{c}{$|$re-run $-$ reported$|$} & seed SD & MB \\\\\n"
        "\\cmidrule(lr){3-4}\nDataset & ckpts & median & max & median & \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
    alld = [float(r["abs_diff"]) for r in man]
    print(f"rerun.tex: {len(man)} checkpoints, median diff {st.median(alld):.4f}, total {sum(size.values()) / 1e6:.1f} MB")


if __name__ == "__main__":
    rerun(); commits()
