"""bd_milp.py -- R-1 step 4: full MILP grid under the new criterion.

Per weight (env WEIGHT in {pop, ev, uniform}):
  4 rho x 6 ablations (A0 phi=0.7, Eucl, energy, sigmaHigh/Medium/Low at
  beta=1) x 11 K = 264 max-coverage MILPs.
Coverage denominators: citywide totals (P3 convention).
Output: bd_layer/milp/results_{WEIGHT}.csv
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import load_npz
import gurobipy as gp
from gurobipy import GRB

sys.path.insert(0, "/lustre/home/2406393544/sharefolder/proj3/r2_layer")
import r2_config as C

P = Path("/lustre/home/2406393544/sharefolder/proj3")
RD = P / "r2_layer/reach"
OUT = P / "r2_layer/milp"
OUT.mkdir(exist_ok=True)
WEIGHT = os.environ.get("WEIGHT", "pop")

joined = pd.read_parquet(P / "gk_run/joined_wuhan_gk.parquet")
if WEIGHT == "uniform":
    w = np.ones(len(joined)); TOT = float(len(joined))
else:
    f, col = (("wuhan_demand_pop_per_cell.parquet", "pop_density") if WEIGHT == "pop"
              else ("wuhan_demand_ev_per_cell.parquet", "n_orders"))
    dd = pd.read_parquet(P / f"output_demand/{f}")
    lut = dict(zip(zip(dd.grid_x, dd.grid_y), dd[col].astype(float)))
    w = np.array([lut.get((int(x), int(y)), 0.0) for x, y in
                  zip(joined.grid_x, joined.grid_y)])
    TOT = float(dd[col].sum())
print(f"[{WEIGHT}] total={TOT:,.0f}", flush=True)

def milp_cov(a_csr, K):
    at = a_csr.T.tocsr()
    ns, nc = a_csr.shape
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

ABL = {"A0": "A0phi0.7", "Eucl": "Eucl", "energy": "energy",
       "sigma_High": "sigmaHigh_b1.0", "sigma_Medium": "sigmaMedium_b1.0",
       "sigma_Low": "sigmaLow_b1.0"}
rows = []
for rho in C.RHOS:
    for abl, tag in ABL.items():
        a = load_npz(RD / f"a_{tag}_rho{rho}.npz").tocsr()
        for K in C.K_GRID:
            cov = milp_cov(a, K)
            rows.append({"weight": WEIGHT, "rho": rho, "ablation": abl,
                         "K": K, "coverage_pct": cov})
            print(f"[{WEIGHT}] rho={rho} {abl} K={K}: {cov:.2f}", flush=True)
pd.DataFrame(rows).to_csv(OUT / f"results_{WEIGHT}.csv", index=False)
print("done", flush=True)
