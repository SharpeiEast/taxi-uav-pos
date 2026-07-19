"""Day-scale stability of the sensed field .

1. Assign the 1.98M valid Wuhan segments to calendar days (7 main days
   2023-05-04..05-10; 0.12% stragglers dropped) using the v2 segment
   table, snapped to the gk grid via the affine lon/lat -> (gx, gy)
   mapping recovered from grid_100m.parquet cell centres (validated by
   reproducing the all-days per-cell median, rank corr vs frozen field).
2. Build 7 per-day cell fields (cells with >= 3 segments that day) +
   weekday/weekend and day/night fields; report pairwise Spearman.
3. Pick the two most dissimilar days (lowest pairwise Spearman) and the
   min/max-median days; rebuild the Medium kappa=0.3 chain with each
   day's field patched onto the headline field; K=30 MILPs at
   rho in {0, 0.5} on the 511 candidate set -> gap variation.

Outputs: cs511/daily_spearman.csv, cs511/daily_gaps_511.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import gurobipy as gp
from gurobipy import GRB
import sys

P = Path("/lustre/home/2406393544/sharefolder/proj3")
OUT = P / "cs511"
sys.path.insert(0, str(P / "r2_layer"))
import r2_config as C

seg = pd.read_parquet(
    P / "v2/data_intermediate/taxi_prior/poc_wuhan/intermediate/segments.parquet",
    columns=["median_lon", "median_lat", "sigma_gk_m", "start_time",
             "hour_start"])
seg["date"] = pd.to_datetime(seg.start_time).dt.date.astype(str)
main_days = [f"2023-05-{d:02d}" for d in range(4, 11)]
n0 = len(seg)
seg = seg[seg.date.isin(main_days)].copy()
print(f"segments: {n0:,} -> {len(seg):,} on 7 main days", flush=True)

# ---- recover the affine lon/lat -> grid mapping from cell centres ----
grid = pd.read_parquet(
    P / "opera/rev02_A1/gk_out/poc_wuhan/intermediate/grid_100m.parquet")
gx, gy = grid.grid_x.values, grid.grid_y.values
lon_c, lat_c = grid.center_lon.values, grid.center_lat.values
ax, bx = np.polyfit(lon_c, gx, 1)
ay, by = np.polyfit(lat_c, gy, 1)
gx_fit = ax * lon_c + bx; gy_fit = ay * lat_c + by
assert np.abs(gx_fit - gx).max() < 0.51 and np.abs(gy_fit - gy).max() < 0.51, \
    "grid mapping is not affine within half a cell"
seg["gx"] = np.floor(ax * seg.median_lon.values + bx + 0.5).astype(int)
seg["gy"] = np.floor(ay * seg.median_lat.values + by + 0.5).astype(int)

# validation: all-days aggregate must reproduce the frozen per-cell median
agg = seg.groupby(["gx", "gy"]).sigma_gk_m.median()
frozen = grid.set_index(["grid_x", "grid_y"]).sigma_gk_median
common = agg.index.intersection(frozen.index)
rs = spearmanr(agg.loc[common], frozen.loc[common]).statistic
print(f"mapping validation: {len(common):,}/{len(frozen):,} cells matched, "
      f"all-days vs frozen Spearman = {rs:.4f}", flush=True)
assert rs > 0.98, "grid mapping failed validation"

# ---- per-day fields ----
MMIN_DAY = 3
fields = {}
for d in main_days:
    g = seg[seg.date == d].groupby(["gx", "gy"])
    f = g.sigma_gk_m.agg(["median", "size"])
    fields[d] = f.loc[f["size"] >= MMIN_DAY, "median"]
    print(f"{d}: {len(fields[d]):,} cells (>= {MMIN_DAY} segs), "
          f"median {fields[d].median():.2f} m", flush=True)
seg["night"] = (seg.hour_start >= 22) | (seg.hour_start < 6)
for name, mask in [("weekday", pd.to_datetime(seg.date).dt.weekday < 5),
                   ("weekend", pd.to_datetime(seg.date).dt.weekday >= 5),
                   ("day", ~seg.night), ("night", seg.night)]:
    g = seg[mask.values].groupby(["gx", "gy"])
    f = g.sigma_gk_m.agg(["median", "size"])
    fields[name] = f.loc[f["size"] >= MMIN_DAY, "median"]

rows = []
keys = main_days + ["weekday", "weekend", "day", "night"]
for i, a in enumerate(keys):
    for b in keys[i + 1:]:
        cm = fields[a].index.intersection(fields[b].index)
        if len(cm) < 100:
            continue
        rows.append({"a": a, "b": b, "n_cells": len(cm),
                     "spearman": float(
                         spearmanr(fields[a].loc[cm],
                                   fields[b].loc[cm]).statistic)})
sp = pd.DataFrame(rows)
sp.to_csv(OUT / "daily_spearman.csv", index=False)
dd = sp[sp.a.isin(main_days) & sp.b.isin(main_days)]
print(f"\nday-pair Spearman: min {dd.spearman.min():.3f} "
      f"median {dd.spearman.median():.3f} max {dd.spearman.max():.3f}",
      flush=True)
worst = dd.loc[dd.spearman.idxmin()]
print(f"most dissimilar pair: {worst.a} vs {worst.b} "
      f"({worst.spearman:.3f})", flush=True)
med_by_day = {d: fields[d].median() for d in main_days}
d_lo = min(med_by_day, key=med_by_day.get)
d_hi = max(med_by_day, key=med_by_day.get)
print(f"min/max-median days: {d_lo} ({med_by_day[d_lo]:.2f}) / "
      f"{d_hi} ({med_by_day[d_hi]:.2f})", flush=True)

# ---- extreme-day K=30 rerun on the 511 set ----
D = np.load(P / "c_layer/domain.npz")
cand_nodes = D["cand_nodes"].astype(np.int64)
cs = pd.read_parquet(
    P / "v2/data_intermediate/demand_supply/candidate_screens.parquet"
).sort_values("cand_idx").reset_index(drop=True)
MASK = cs["base"].values.astype(bool)
joined = pd.read_parquet(P / "gk_run/joined_wuhan_gk.parquet")
dd_ = pd.read_parquet(P / "output_demand/wuhan_demand_pop_per_cell.parquet")
lut = dict(zip(zip(dd_.grid_x, dd_.grid_y), dd_.pop_density.astype(float)))
w = np.array([lut.get((int(x), int(y)), 0.0) for x, y in
              zip(joined.grid_x, joined.grid_y)])
TOT = float(dd_.pop_density.sum())
F = np.load(P / "r2_layer/field_r2.npz")
sigma_full = F["sigma_full"].astype(float)
sig_j = F["sig_j"].astype(float)
NV = len(sigma_full)
node_grid = D["node_grid"]; ii, jj = np.nonzero(node_grid >= 0)
ids = node_grid[ii, jj]
gx0, gy0 = int(D["gx0"]), int(D["gy0"])
gx_n = np.empty(ids.max() + 1, np.int64); gy_n = np.empty(ids.max() + 1, np.int64)
gx_n[ids] = gx0 + ii; gy_n[ids] = gy0 + jj
J_nodes = D["J_nodes"].astype(np.int64)
if not np.array_equal(gx_n[J_nodes], joined.grid_x.values):
    gx_n[ids] = gx0 + jj; gy_n[ids] = gy0 + ii
obs_nodes = D["obs_nodes"].astype(np.int64)
key_node = {(int(gx_n[n]), int(gy_n[n])): n for n in obs_nodes}
eu_ = D["edges_u"].astype(np.int64); ev_ = D["edges_v"].astype(np.int64)
el_ = D["edges_len"].astype(float)
jx = (joined.grid_x.values + 0.5) * 100.0
jy = (joined.grid_y.values + 0.5) * 100.0
DE = np.hypot(((gx_n[cand_nodes] + 0.5) * 100.0)[:, None] - jx[None, :],
              ((gy_n[cand_nodes] + 0.5) * 100.0)[:, None] - jy[None, :])
C_eucl_rt = DE * 11.1582 + DE * 9.2365
ET = np.load(P / "r2_layer/eterm.npz")
eterm0 = float(ET["eterm0_main"][0]); edep = float(ET["edep_main"][0])
jkey = {(int(x), int(y)): t for t, (x, y) in
        enumerate(zip(joined.grid_x.values, joined.grid_y.values))}
reg = C.REGIMES_BD["Medium"]
KP = 0.3


def P_inst(v, m_p):
    v = np.maximum(np.asarray(v, float), 1e-9)
    kap = 1.0 + m_p * C.G / C.W_FRAME
    v0sq = kap * C.V0_HOV ** 2
    return (C.P_B * (1 + 3 * v**2 / C.U_TIP**2)
            + (kap**1.5) * C.P_IND * np.sqrt(
                np.sqrt(1 + v**4/(4*v0sq**2)) - v**2/(2*v0sq))
            + 0.5 * C.D0 * C.RHO_AIR * C.S_SOL * C.A_DISC * v**3)


def route_cost(sf):
    removed = sf > reg.sigma_sat(KP)
    Crt = None
    for mp in [C.M_P_MAIN, 0.0]:
        vm = np.maximum(C.V_MIN, np.minimum(
            C.V_CRUISE, (reg.w_corr - reg.z_eps * sf) / (KP * reg.tau_react)))
        on = (sf > reg.sigma_sensor).astype(float)
        unit = P_inst(vm, mp) / vm + on * C.P_AUX / vm
        wgt = 0.5 * (unit[eu_] + unit[ev_]) * el_
        ok = ~(removed[eu_] | removed[ev_])
        g = csr_matrix((np.concatenate([wgt[ok], wgt[ok]]),
                        (np.concatenate([eu_[ok], ev_[ok]]),
                         np.concatenate([ev_[ok], eu_[ok]]))),
                       shape=(NV, NV))
        d = dijkstra(g, directed=False, indices=cand_nodes)[:, J_nodes]
        Crt = (d if Crt is None else Crt + d)
    return np.nan_to_num(Crt, nan=np.inf, posinf=np.inf)


def eterm_iid(s):
    ph_ld = C.P_B + C.P_IND * (1 + C.M_P_MAIN * C.G / C.W_FRAME)**1.5
    st_desc = (ph_ld - (C.W_FRAME + C.M_P_MAIN * C.G) * C.V_DESC / 2) \
        * (C.H_C / C.V_DESC)
    ph_e = C.P_B + C.P_IND
    st_climb_e = (ph_e + C.W_FRAME * C.V_CLIMB / 2) * (C.H_C / C.V_CLIMB)
    e_ga = 2 * ph_ld * C.T_GA
    q = 1 - np.exp(-C.R_PAD_MAIN**2 / (2 * np.maximum(s, 1e-12)**2))
    N = 1 / q
    return (st_desc + N * ph_ld * C.T_ALIGN + (N - 1) * e_ga
            + ph_ld * C.T_REL + st_climb_e)


def milp_cov(mask_feas, K=30):
    at = csr_matrix(mask_feas).T.tocsr()
    ns, nc = mask_feas.shape
    m = gp.Model(); m.Params.OutputFlag = 0; m.Params.MIPGap = 1e-6
    m.Params.Threads = 32
    x = m.addVars(ns, vtype=GRB.BINARY)
    y = m.addVars(nc, vtype=GRB.CONTINUOUS, ub=1.0)
    m.addConstr(gp.quicksum(x[i] for i in range(ns)) <= K)
    for j in range(nc):
        idx = at.indices[at.indptr[j]:at.indptr[j + 1]]
        m.addConstr(y[j] <= (gp.quicksum(x[i] for i in idx) if len(idx) else 0))
    m.setObjective(gp.quicksum(float(w[j]) * y[j] for j in range(nc)),
                   GRB.MAXIMIZE)
    m.optimize()
    assert m.Status == GRB.OPTIMAL
    return 100.0 * m.ObjVal / TOT


extreme = sorted({worst.a, worst.b, d_lo, d_hi})
rows = []
for d in extreme:
    f = fields[d]
    sf = sigma_full.copy(); sj = sig_j.copy()
    n_node = n_j = 0
    for (gxx, gyy), v in f.items():
        n = key_node.get((int(gxx), int(gyy)))
        if n is not None:
            sf[n] = v; n_node += 1
        t = jkey.get((int(gxx), int(gyy)))
        if t is not None:
            sj[t] = v; n_j += 1
    print(f"{d}: patched {n_node:,} nodes / {n_j:,} J cells", flush=True)
    Crt = route_cost(sf)
    et = eterm_iid(sj)
    for rho in [0.0, 0.5]:
        B = (1 - rho) * C.ETA_B
        fe = (C_eucl_rt + edep + eterm0 <= B) & MASK[:, None]
        fm = (Crt + edep + et[None, :] <= B) & MASK[:, None]
        ce = milp_cov(fe); cm = milp_cov(fm)
        rows.append({"day": d, "rho": rho, "cand_set": 511,
                     "cov_eucl": ce, "cov_M": cm, "gap_pp": ce - cm})
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(OUT / "daily_gaps_511.csv", index=False)
print("done", flush=True)
