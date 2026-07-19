"""bd_decomp.py -- R-3 data: five-step waterfall (Fig A) + infeasibility
three-class decomposition (Fig B).

Fig A (K=30, pop, Medium, rho in {0.5, 0.8}): coverage at five model stages
  1 Eucl                        : DE-based, E_term(0)
  2 + on-grid (Energy)          : grid routing at E_base, E_term(0)
  3 + speed envelope (zeroMedium): regime sigma=0 costs, E_term(0)
  4 + route sigma (M, E_term(0)): sigma-aware routing, terminal still blind
  5 + terminal sigma (full M)   : E_term(sigma_j)   [= headline model]

Fig B (K=605 budget-free, per regime x rho, pop-share of demand):
  blocked        : no path (no-fly removal / disconnection), min_i C = inf
  over-range     : min_i (C_out+C_ret) + E_dep + E_term(0) > budget
  terminal-driven: within budget at E_term(0) but over with E_term(sigma_j)
  (covered otherwise)

Output: bd_layer/waterfall.csv, bd_layer/threeclass.csv
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import gurobipy as gp
from gurobipy import GRB
from scipy.sparse import csr_matrix

sys.path.insert(0, "/lustre/home/2406393544/sharefolder/proj3/r2_layer")
import r2_config as C

P = Path("/lustre/home/2406393544/sharefolder/proj3")
CM = P / "r2_layer/costmat"
ET = np.load(P / "r2_layer/eterm.npz")
eterm = ET["eterm_main"].astype(np.float64)
eterm0 = float(ET["eterm0_main"][0]); edep = float(ET["edep_main"][0])

def load(n): return np.load(CM / f"{n}.npz")["C"].astype(np.float64)

joined = pd.read_parquet(P / "gk_run/joined_wuhan_gk.parquet")
dd = pd.read_parquet(P / "output_demand/wuhan_demand_pop_per_cell.parquet")
lut = dict(zip(zip(dd.grid_x, dd.grid_y), dd.pop_density.astype(float)))
w = np.array([lut.get((int(x), int(y)), 0.0) for x, y in
              zip(joined.grid_x, joined.grid_y)])
TOT = float(dd.pop_density.sum())

# snapped-centre Euclid (same as bd_reach)
D = np.load(P / "c_layer/domain.npz")
node_grid = D["node_grid"]; ii, jj = np.nonzero(node_grid >= 0)
ids = node_grid[ii, jj]
gx0, gy0 = int(D["gx0"]), int(D["gy0"])
gx_n = np.empty(ids.max() + 1, np.int64); gy_n = np.empty(ids.max() + 1, np.int64)
gx_n[ids] = gx0 + ii; gy_n[ids] = gy0 + jj
J_nodes = D["J_nodes"].astype(np.int64)
if not np.array_equal(gx_n[J_nodes], joined.grid_x.values):
    gx_n[ids] = gx0 + jj; gy_n[ids] = gy0 + ii
cand = D["cand_nodes"].astype(np.int64)
jx = (joined.grid_x.values + 0.5) * 100.0
jy = (joined.grid_y.values + 0.5) * 100.0
DE = np.hypot(((gx_n[cand] + 0.5) * 100.0)[:, None] - jx[None, :],
              ((gy_n[cand] + 0.5) * 100.0)[:, None] - jy[None, :])

def milp_cov(mask, K):
    at = csr_matrix(mask).T.tocsr()
    ns, nc = mask.shape
    m = gp.Model(); m.Params.OutputFlag = 0; m.Params.MIPGap = 1e-6
    m.Params.Threads = 16
    x = m.addVars(ns, vtype=GRB.BINARY)
    y = m.addVars(nc, vtype=GRB.CONTINUOUS, ub=1.0)
    m.addConstr(gp.quicksum(x[i] for i in range(ns)) <= K)
    for j in range(nc):
        idx = at.indices[at.indptr[j]:at.indptr[j + 1]]
        m.addConstr(y[j] <= (gp.quicksum(x[i] for i in idx) if len(idx) else 0))
    m.setObjective(gp.quicksum(float(w[j]) * y[j] for j in range(nc)), GRB.MAXIMIZE)
    m.optimize()
    return 100.0 * m.ObjVal / TOT

# route-cost matrices
C_eucl_rt = DE * 11.1582 + DE * 9.2365
C_energy_rt = load("C_ld_energy") + load("C_e_energy")
C_zeroM_rt = load("C_ld_zeroMedium") + load("C_e_zeroMedium")
C_M_rt = np.nan_to_num(load("C_ld_sigmaMedium_b1.0") + load("C_e_sigmaMedium_b1.0"),
                       nan=np.inf, posinf=np.inf)

# ---- Fig A: waterfall ----
rows = []
for rho in [0.5, 0.8]:
    B = (1 - rho) * C.ETA_B
    stages = [
        ("1_Eucl", C_eucl_rt + edep + eterm0),
        ("2_ongrid_Energy", C_energy_rt + edep + eterm0),
        ("3_speed_envelope", C_zeroM_rt + edep + eterm0),
        ("4_route_sigma", C_M_rt + edep + eterm0),
        ("5_terminal_sigma", C_M_rt + edep + eterm[None, :]),
    ]
    for name, Ct in stages:
        cov = milp_cov(Ct <= B, 30)
        rows.append({"rho": rho, "stage": name, "coverage_pct": cov})
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(P / "r2_layer/waterfall.csv", index=False)

# ---- Fig B: three-class (budget-free, per regime x rho) ----
rows = []
for rname in ["High", "Medium", "Low"]:
    Crt = np.nan_to_num(load(f"C_ld_sigma{rname}_b1.0") + load(f"C_e_sigma{rname}_b1.0"),
                        nan=np.inf, posinf=np.inf)
    best = Crt.min(axis=0)                      # (17,899,) best route cost
    for rho in C.RHOS:
        B = (1 - rho) * C.ETA_B
        blocked = ~np.isfinite(best)
        over = np.isfinite(best) & (best + edep + eterm0 > B)
        term = (np.isfinite(best) & (best + edep + eterm0 <= B)
                & (best + edep + eterm > B))
        covered = ~(blocked | over | term)
        rows.append({"regime": rname, "rho": rho,
                     "blocked_pct": 100 * w[blocked].sum() / TOT,
                     "overrange_pct": 100 * w[over].sum() / TOT,
                     "terminal_pct": 100 * w[term].sum() / TOT,
                     "covered_pct": 100 * w[covered].sum() / TOT})
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(P / "r2_layer/threeclass.csv", index=False)
print("done", flush=True)
