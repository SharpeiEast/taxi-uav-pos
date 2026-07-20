#!/usr/bin/env python3
"""milp_recheck.py -- fresh MILP re-certification.
(1) Re-solve the headline Medium MCLP (pop, K=30, 4 reserve levels) from the
    frozen reachability matrices with a fresh Gurobi model; assert proven
    optimality and agreement with the frozen coverage table.
(2) Rebuild the A0 fixed-radius baseline FROM COORDINATES (independent
    Euclidean distance recomputation) and re-solve; assert agreement.
(3) Re-audit the imputed-node saturation counts reported in the supplement.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import load_npz
import gurobipy as gp
from gurobipy import GRB

P = Path("/lustre/home/2406393544/sharefolder/proj3")
ok = True
def chk(name, cond):
    global ok
    print(f"{name:68s} {'PASS' if cond else 'FAIL'}", flush=True)
    ok = ok and bool(cond)

# ---- shared data ----
D = np.load(P / "c_layer/domain.npz")
cand_nodes = D["cand_nodes"].astype(np.int64)
cs = pd.read_parquet(
    P / "v2/data_intermediate/demand_supply/candidate_screens.parquet"
).sort_values("cand_idx").reset_index(drop=True)
assert (cs["node"].values.astype(np.int64) == cand_nodes).all()
MASK = cs["base"].values.astype(bool)

joined = pd.read_parquet(P / "gk_run/joined_wuhan_gk.parquet")
dd = pd.read_parquet(P / "output_demand/wuhan_demand_pop_per_cell.parquet")
lut = dict(zip(zip(dd.grid_x, dd.grid_y), dd.pop_density.astype(float)))
w = np.array([lut.get((int(x), int(y)), 0.0)
              for x, y in zip(joined.grid_x, joined.grid_y)])
TOT = float(dd.pop_density.sum())

def solve_k30(a_bool, K=30):
    """Fresh equality-budget MCLP; returns (coverage_pct, mipgap, status)."""
    ns, nj = a_bool.shape
    m = gp.Model()
    m.Params.OutputFlag = 0
    m.Params.Threads = 16
    m.Params.MIPGap = 1e-9
    x = m.addVars(ns, vtype=GRB.BINARY)
    y = m.addVars(nj, vtype=GRB.BINARY)
    m.addConstr(gp.quicksum(x[i] for i in range(ns)) == K)
    rows, cols = np.nonzero(a_bool)
    covers = [[] for _ in range(nj)]
    for i, j in zip(rows, cols):
        covers[j].append(i)
    for j in range(nj):
        if covers[j]:
            m.addConstr(y[j] <= gp.quicksum(x[i] for i in covers[j]))
        else:
            m.addConstr(y[j] == 0)
    m.setObjective(gp.quicksum(w[j] * y[j] for j in range(nj)), GRB.MAXIMIZE)
    m.optimize()
    return 100.0 * m.ObjVal / TOT, m.MIPGap, m.Status

milp = pd.concat([pd.read_csv(P / f"cs511/milp_pop_rho{r}.csv")
                  for r in ("0.0", "0.2", "0.5", "0.8")])

# ---- (1) headline Medium re-solve ----
for rho in (0.0, 0.2, 0.5, 0.8):
    a = load_npz(P / f"r2_layer/reach/a_sigmaMedium_b1.0_rho{rho}.npz").toarray()
    a = a[MASK]
    cov, gap, st = solve_k30(a.astype(bool))
    frozen = milp[(milp.rho == rho) & (milp.ablation == "Medium")
                  & (milp.K == 30)].coverage_pct.iloc[0]
    chk(f"G1 Medium rho={rho}: fresh {cov:.4f} == frozen {frozen:.4f}, optimal",
        abs(cov - frozen) < 1e-4 and st == GRB.OPTIMAL and gap < 1e-8)

# ---- (2) A0 rebuilt from coordinates ----
node_grid = D["node_grid"]
ii, jj = np.nonzero(node_grid >= 0)
ids = node_grid[ii, jj]
gx0, gy0 = int(D["gx0"]), int(D["gy0"])
gx_n = np.empty(ids.max() + 1, np.int64)
gy_n = np.empty(ids.max() + 1, np.int64)
gx_n[ids] = gx0 + ii; gy_n[ids] = gy0 + jj
J_nodes = D["J_nodes"].astype(np.int64)
if not np.array_equal(gx_n[J_nodes], joined.grid_x.values):
    gx_n[ids] = gx0 + jj; gy_n[ids] = gy0 + ii
jx = (joined.grid_x.values + 0.5) * 100.0
jy = (joined.grid_y.values + 0.5) * 100.0
sx = (gx_n[cand_nodes] + 0.5) * 100.0
sy = (gy_n[cand_nodes] + 0.5) * 100.0
DE = np.hypot(sx[:, None] - jx[None, :], sy[:, None] - jy[None, :])
E_BASE_E, ETAB, PHI = 9.2365, 281_360.0, 0.7
for rho in (0.0, 0.2, 0.5, 0.8):
    a0 = (2.0 * DE * E_BASE_E <= PHI * (1 - rho) * ETAB)[MASK]
    cov, gap, st = solve_k30(a0)
    frozen = milp[(milp.rho == rho) & (milp.ablation == "A0")
                  & (milp.K == 30)].coverage_pct.iloc[0]
    chk(f"G2 A0 rho={rho}: rebuilt-from-coords {cov:.4f} == frozen {frozen:.4f}",
        abs(cov - frozen) < 1e-4 and st == GRB.OPTIMAL)

# ---- (3) imputed-node saturation audit ----
F = np.load(P / "r2_layer/field_r2.npz")
s = F["sigma_full"].astype(float)
obs = np.zeros(len(s), dtype=bool)
obs[D["obs_nodes"].astype(int)] = True
kmax = s[~obs].max()
n_low = int(((s > 7.05) & ~obs).sum())
n_med = int(((s > 9.7) & ~obs).sum())
n_high = int(((s > 19.7) & ~obs).sum())
chk(f"G3 imputed max {kmax:.2f}m<9.7; crossings L/M/H = {n_low}/{n_med}/{n_high}",
    kmax < 9.7 and n_low == 3 and n_med == 0 and n_high == 0
    and (~obs).sum() == 1_024_300)

print("=== verify_v14_gurobi.py",
      "ALL CHECKS PASS" if ok else "FAILURES PRESENT", "===", flush=True)
