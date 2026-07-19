"""r2_prep.py -- build the R2 headline field (denoised m_min>=5).

Recipe identical to bd_extra.field_m5krig (frozen): observed nodes with
fewer than 5 segments are replaced by the median of their 16 nearest
well-sampled observed nodes, clipped to [s_min, s_cap]; all other nodes
keep the kriged/observed value of the R-0 field.

Output: r2_layer/field_r2.npz
  sigma_full : (NV,) routing-grid field
  sig_j      : (17,899,) J-cell values (= sigma_full[J_nodes])
  j_denoised : bool mask of J cells whose value was replaced (n_seg < 5)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree

P = Path("/lustre/home/2406393544/sharefolder/proj3")
OUT = P / "r2_layer"
OUT.mkdir(exist_ok=True)

D = np.load(P / "c_layer/domain.npz")
F = np.load(P / "c_layer/sigma_fields.npz")
sigma0 = F["sigma_krig"].astype(float)
NV = len(sigma0)
node_grid = D["node_grid"]
obs_nodes = D["obs_nodes"].astype(np.int64)
obs_nseg = D["obs_nseg"]
J_nodes = D["J_nodes"].astype(np.int64)

s = sigma0.copy()
m_lt5 = obs_nseg < 5
rr, cc = np.where(node_grid >= 0)
idz = node_grid[rr, cc]
xy = np.zeros((NV, 2)); xy[idz, 0] = cc * 100.0; xy[idz, 1] = rr * 100.0
good = obs_nodes[~m_lt5]
goodv = sigma0[good]
tree = cKDTree(xy[good])
tgt = obs_nodes[m_lt5]
d_, nb = tree.query(xy[tgt], k=16, workers=8)
vals = np.median(goodv[nb], axis=1)
s[tgt] = np.clip(vals, float(F["s_min"]), float(F["s_cap"]))

sig_j = s[J_nodes]
lt5_set = set(tgt.tolist())
j_denoised = np.array([n in lt5_set for n in J_nodes], dtype=bool)

joined = pd.read_parquet(P / "gk_run/joined_wuhan_gk.parquet")
old_j = joined["sigma_gk_median"].values.astype(float)
print(f"nodes replaced: {m_lt5.sum():,}/{len(obs_nodes):,} observed "
      f"({100*m_lt5.mean():.1f}%)")
print(f"J cells replaced: {j_denoised.sum():,}/17,899 "
      f"({100*j_denoised.mean():.1f}%)")
print(f"J-cell field median: old(m1)={np.median(old_j):.3f}  "
      f"r2(m5)={np.median(sig_j):.3f}")
print(f"unchanged J cells identical: "
      f"{np.allclose(sig_j[~j_denoised], old_j[~j_denoised])}")

np.savez_compressed(OUT / "field_r2.npz", sigma_full=s, sig_j=sig_j,
                    j_denoised=j_denoised)
print("saved r2_layer/field_r2.npz")
