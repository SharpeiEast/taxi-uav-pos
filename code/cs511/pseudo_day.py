"""Matched-sample day-stability control .

Builds 20 pseudo-day fields by sampling, without replacement, the mean
per-day segment count from the pooled 7-day segment table, using the
same per-cell rule as the actual-day fields (median of >= 3 segments,
patched onto the headline field), then re-runs the K=30 Medium chain at
rho in {0, 0.5}. Comparing the pseudo-day gap distribution with the
actual-day gaps separates the sample-size effect from any genuine
day-specific temporal effect.

Output: cs511/pseudo_day_gaps_511.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import gurobipy as gp
from gurobipy import GRB
import sys

P = Path("/lustre/home/2406393544/sharefolder/proj3")
OUT = P / "cs511"
sys.path.insert(0, str(P / "r2_layer"))
import r2_config as C

N_REP = 20
MMIN_DAY = 3

seg = pd.read_parquet(
    P / "v2/data_intermediate/taxi_prior/poc_wuhan/intermediate/segments.parquet",
    columns=["median_lon", "median_lat", "sigma_gk_m", "start_time"])
seg["date"] = pd.to_datetime(seg.start_time).dt.date.astype(str)
main_days = [f"2023-05-{d:02d}" for d in range(4, 11)]
seg = seg[seg.date.isin(main_days)].reset_index(drop=True)
n_day = int(round(len(seg) / 7))
print(f"pooled segments {len(seg):,}; pseudo-day size {n_day:,}", flush=True)

grid = pd.read_parquet(
    P / "opera/rev02_A1/gk_out/poc_wuhan/intermediate/grid_100m.parquet")
ax, bx = np.polyfit(grid.center_lon.values, grid.grid_x.values, 1)
ay, by = np.polyfit(grid.center_lat.values, grid.grid_y.values, 1)
seg["gx"] = np.floor(ax * seg.median_lon.values + bx + 0.5).astype(int)
seg["gy"] = np.floor(ay * seg.median_lat.values + by + 0.5).astype(int)

D = np.load(P / "c_layer/domain.npz")
cand_nodes = D["cand_nodes"].astype(np.int64)
cs = pd.read_parquet(
    P / "v2/data_intermediate/demand_supply/candidate_screens.parquet"
).sort_values("cand_idx").reset_index(drop=True)
MASK = cs["base"].values.astype(bool)
joined = pd.read_parquet(P / "gk_run/joined_wuhan_gk.parquet")
dd = pd.read_parquet(P / "output_demand/wuhan_demand_pop_per_cell.parquet")
lut = dict(zip(zip(dd.grid_x, dd.grid_y), dd.pop_density.astype(float)))
w = np.array([lut.get((int(x), int(y)), 0.0) for x, y in
              zip(joined.grid_x, joined.grid_y)])
TOT = float(dd.pop_density.sum())
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
jkey = {(int(x), int(y)): t for t, (x, y) in
        enumerate(zip(joined.grid_x.values, joined.grid_y.values))}
eu_ = D["edges_u"].astype(np.int64); ev_ = D["edges_v"].astype(np.int64)
el_ = D["edges_len"].astype(float)
jx = (joined.grid_x.values + 0.5) * 100.0
jy = (joined.grid_y.values + 0.5) * 100.0
DE = np.hypot(((gx_n[cand_nodes] + 0.5) * 100.0)[:, None] - jx[None, :],
              ((gy_n[cand_nodes] + 0.5) * 100.0)[:, None] - jy[None, :])
C_eucl_rt = DE * 11.1582 + DE * 9.2365
ET = np.load(P / "r2_layer/eterm.npz")
eterm0 = float(ET["eterm0_main"][0]); edep = float(ET["edep_main"][0])
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


rng = np.random.default_rng(20260719)
rows = []
for rep in range(N_REP):
    idx = rng.choice(len(seg), size=n_day, replace=False)
    sub = seg.iloc[idx]
    g = sub.groupby(["gx", "gy"]).sigma_gk_m.agg(["median", "size"])
    f = g.loc[g["size"] >= MMIN_DAY, "median"]
    sf = sigma_full.copy(); sj = sig_j.copy()
    n_node = 0
    for (gxx, gyy), v in f.items():
        n = key_node.get((int(gxx), int(gyy)))
        if n is not None:
            sf[n] = v; n_node += 1
        t = jkey.get((int(gxx), int(gyy)))
        if t is not None:
            sj[t] = v
    Crt = route_cost(sf)
    et = eterm_iid(sj)
    for rho in [0.0, 0.5]:
        B = (1 - rho) * C.ETA_B
        fe = (C_eucl_rt + edep + eterm0 <= B) & MASK[:, None]
        fm = (Crt + edep + et[None, :] <= B) & MASK[:, None]
        ce = milp_cov(fe); cm = milp_cov(fm)
        rows.append({"rep": rep, "rho": rho, "cand_set": 511,
                     "n_cells_patched": int(len(f)),
                     "cov_eucl": ce, "cov_M": cm, "gap_pp": ce - cm})
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(OUT / "pseudo_day_gaps_511.csv", index=False)
df = pd.DataFrame(rows)
for rho in [0.0, 0.5]:
    g = df[df.rho == rho].gap_pp
    print(f"rho={rho}: pseudo-day gap mean {g.mean():.2f} "
          f"range [{g.min():.2f}, {g.max():.2f}]", flush=True)
print("done", flush=True)
