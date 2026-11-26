import json, glob, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
rows=[]
for ds in ["uci","smell"]:
  for split in ["test","fold2"]:
    for f in glob.glob(f"results/selection/{ds}/{split}/*/*.json"):
        m=os.path.basename(os.path.dirname(f)); tag=os.path.basename(f)
        tf=f"results/tuning/{ds}/{m}/{tag}"
        if not os.path.exists(tf): continue
        r=json.load(open(f)); t=json.load(open(tf))
        if r["diverged"]: sc=0.0
        elif split=="test":
            mo=r["final_eval"]["mean_over_domains"]; sc=mo.get("score", mo.get("macro_f1_present"))
        else:
            sc=(r.get("best_val") or {}).get("score", (r.get("best_val") or {}).get("macro_f1", np.nan))
        bv=t.get("best_val") or {}
        v=0.0 if t["diverged"] else bv.get("score", bv.get("macro_f1", np.nan))
        rows.append(dict(ds=ds,split=split,m=m,cfg=tag.rsplit("_s",1)[0],val=v,sc=sc))
d=pd.DataFrame(rows); print(len(d)); print(d.isna().sum().to_dict())
g=d.groupby(["ds","split","m","cfg"])[["val","sc"]].mean().reset_index()
for (ds,sp),x in g.groupby(["ds","split"]):
    out=[]
    for m,y in x.groupby("m"):
        out.append((m, round(spearmanr(y.val,y.sc).correlation,2), round(y.sc.max()-y.loc[y.val.idxmax(),"sc"],3)))
    print(ds,sp,out)
