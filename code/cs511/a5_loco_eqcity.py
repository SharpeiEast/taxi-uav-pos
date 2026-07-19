"""A5 follow-up: LOCO transferability under balanced city sampling.

robustness batch: the pooled LOCO training set is dominated by the
large cities. Robustness check: downsample every training city to the
same cell count (the smallest city's n), repeat 100 times, and report
the distribution of held-out R^2 (raw) and Spearman rank correlation.

Features, target, and model grid identical to a5_loco.py (subset of
models for runtime: ridge / knn / rf). Output: a5_loco_eqcity.csv +
printed summary.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

P = Path("/lustre/home/2406393544/sharefolder/proj3")
CITIES = ["zhengzhou", "guangzhou", "wuhan", "shanghai", "xiamen", "beijing"]
N_REP = 100

BASE_FEATS = ["n_buildings", "building_density_per_km2", "height_mean_m",
              "height_max_m", "height_std_m", "height_p95_m",
              "footprint_ratio", "volume_density_m3_per_m2",
              "frac_func_residence", "frac_func_commercial",
              "frac_func_industry"]

frames = []
for c in CITIES:
    d = pd.read_parquet(P / f"output_regression_v2/joined_{c}_100m.parquet")
    d["city"] = c
    frames.append(d)
df = pd.concat(frames, ignore_index=True)

A_b = (df["total_footprint_m2"] / df["n_buildings"].clip(lower=1)).clip(lower=1.0)
L = np.sqrt(A_b)
dens = (df["building_density_per_km2"] / 1e6).clip(lower=1e-9)
S = 1.0 / np.sqrt(dens)
Wst = (S - L).clip(lower=2.0)
df["aspect_proxy"] = (df["height_mean_m"] / Wst).clip(upper=20)
df["svf_proxy"] = np.cos(np.arctan(2.0 * df["aspect_proxy"]))
FEATS = BASE_FEATS + ["aspect_proxy", "svf_proxy"]

df = df.dropna(subset=FEATS + ["sigma_pos_median"]).reset_index(drop=True)
counts = df.groupby("city").size()
n_eq = int(counts.min())
print("cells per city:", counts.to_dict(), "-> equal-n =", n_eq, flush=True)


def models():
    return {
        "ridge": Ridge(alpha=10.0),
        "knn": KNeighborsRegressor(n_neighbors=15),
        "rf": RandomForestRegressor(n_estimators=150, min_samples_leaf=5,
                                    random_state=0, n_jobs=16),
    }


y_all = df["sigma_pos_median"].values
rows = []
rng = np.random.default_rng(20260718)
for rep in range(N_REP):
    tr_idx = []
    for c in CITIES:
        idx_c = df.index[df.city == c].to_numpy()
        tr_idx.append(rng.choice(idx_c, size=n_eq, replace=False))
    tr_idx = {c: i for c, i in zip(CITIES, tr_idx)}
    for held in CITIES:
        tr = np.concatenate([tr_idx[c] for c in CITIES if c != held])
        te = df.index[df.city == held].to_numpy()
        sc = StandardScaler().fit(df.loc[tr, FEATS].values)
        Xtr = sc.transform(df.loc[tr, FEATS].values)
        Xte = sc.transform(df.loc[te, FEATS].values)
        ytr, yte = y_all[tr], y_all[te]
        for name, mk in models().items():
            pred = mk.fit(Xtr, ytr).predict(Xte)
            rows.append({"rep": rep, "held_city": held, "model": name,
                         "r2_raw": r2_score(yte, pred),
                         "spearman": float(spearmanr(yte, pred).statistic)})
    if (rep + 1) % 10 == 0:
        print(f"rep {rep + 1}/{N_REP}", flush=True)

res = pd.DataFrame(rows)
res.to_csv(P / "cs511/a5_loco_eqcity.csv", index=False)

print("\n=== equal-city LOCO distributions (100 reps) ===")
g = res.groupby(["held_city", "model"])
summ = g.agg(r2_med=("r2_raw", "median"),
             r2_p5=("r2_raw", lambda v: np.quantile(v, .05)),
             r2_p95=("r2_raw", lambda v: np.quantile(v, .95)),
             sp_med=("spearman", "median"),
             sp_p5=("spearman", lambda v: np.quantile(v, .05)),
             sp_p95=("spearman", lambda v: np.quantile(v, .95))).round(3)
print(summ.to_string())
summ.to_csv(P / "cs511/a5_loco_eqcity_summary.csv")
print(f"\nr2_raw negative share: {(res.r2_raw < 0).mean():.2%}")
print(f"spearman median overall: {res.spearman.median():.3f}")
print("done", flush=True)
