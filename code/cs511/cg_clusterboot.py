"""Cluster bootstrap for the dispersion-to-error calibration c_g
.

Replicates the e8_calibrate receiver-class headline (variant
tmin30_full, dev_class == receiver, sigma_gk > 0.05): ratio =
err_p95 / sigma_gk, c_g = median ratio. Reports the iid interval
(as published) alongside cluster-bootstrap intervals with clusters
defined by (seq, device) and by seq alone.
Output: cs511/cg_clusterboot.json
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

P = Path("/lustre/home/2406393544/sharefolder/proj3")
d = pd.read_csv(P / "E8_urbannav/e8_segments.csv")
d = d[(d.variant == "tmin30_full") & (d.dev_class == "receiver")
      & (d.sigma_gk > 0.05)].copy()
d["ratio"] = d.err_p95 / d.sigma_gk
d = d[np.isfinite(d.ratio)]
print(f"segments: {len(d)}  seqs: {d.seq.nunique()}  "
      f"devices: {d.device.nunique()}  "
      f"seq-device clusters: {d.groupby(['seq','device']).ngroups}",
      flush=True)

c_hat = float(d.ratio.median())
rng = np.random.default_rng(20260718)
B = 20000

def boot_iid():
    v = d.ratio.values
    return np.median(v[rng.integers(0, len(v), size=(B, len(v)))], axis=1)

def boot_cluster(keys):
    groups = [g.ratio.values for _, g in d.groupby(keys, sort=False)]
    n = len(groups)
    out = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, n, n)
        out[b] = np.median(np.concatenate([groups[k] for k in pick]))
    return out

res = {"n_segments": int(len(d)), "c_hat_median": c_hat}
for name, bs in [("iid", boot_iid()),
                 ("cluster_seq_device", boot_cluster(["seq", "device"])),
                 ("cluster_seq", boot_cluster(["seq"]))]:
    res[name] = {"lo95": float(np.quantile(bs, 0.025)),
                 "hi95": float(np.quantile(bs, 0.975)),
                 "mean": float(bs.mean())}
    print(name, res[name], flush=True)

json.dump(res, open(P / "cs511/cg_clusterboot.json", "w"), indent=2)
print("done", flush=True)
