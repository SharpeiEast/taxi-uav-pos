"""bd_regret.py -- R-2: Reported / Realized / Best decomposition (D2).

Reference ablation = sigma_Medium beta=1; weight = pop (headline).
Instances: K=30 x 4 rho  +  rho=0.5 x full K grid.
For each: solve Eucl MILP with solution pool (<=20 optima), Reported =
V*(Eucl); Realized = f_M(S_Eucl) for each pooled optimum (report spread);
Best = V*(M). Output: bd_layer/regret.csv
"""
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

joined = pd.read_parquet(P / "gk_run/joined_wuhan_gk.parquet")
dd = pd.read_parquet(P / "output_demand/wuhan_demand_pop_per_cell.parquet")
lut = dict(zip(zip(dd.grid_x, dd.grid_y), dd.pop_density.astype(float)))
w = np.array([lut.get((int(x), int(y)), 0.0) for x, y in
              zip(joined.grid_x, joined.grid_y)])
TOT = float(dd.pop_density.sum())

def solve_pool(a_csr, K, npool=20):
    """Max-coverage MILP; returns list of optimal station sets (<=npool)."""
    at = a_csr.T.tocsr()
    ns, nc = a_csr.shape
    m = gp.Model(); m.Params.OutputFlag = 0; m.Params.MIPGap = 1e-6
    m.Params.Threads = 16
    m.Params.PoolSearchMode = 2; m.Params.PoolSolutions = npool
    m.Params.PoolGap = 1e-9
    x = m.addVars(ns, vtype=GRB.BINARY)
    y = m.addVars(nc, vtype=GRB.CONTINUOUS, ub=1.0)
    m.addConstr(gp.quicksum(x[i] for i in range(ns)) <= K)
    for j in range(nc):
        idx = at.indices[at.indptr[j]:at.indptr[j + 1]]
        m.addConstr(y[j] <= (gp.quicksum(x[i] for i in idx) if len(idx) else 0))
    m.setObjective(gp.quicksum(float(w[j]) * y[j] for j in range(nc)), GRB.MAXIMIZE)
    m.optimize()
    best = m.ObjVal
    sets = []
    for s in range(m.SolCount):
        m.Params.SolutionNumber = s
        if m.PoolObjVal < best - 1e-6 * max(1.0, abs(best)):
            break
        sets.append([i for i in range(ns) if x[i].Xn > 0.5])
    return best, sets

def f_under(a_csr, S):
    cov = np.asarray(a_csr[S].max(axis=0).todense()).ravel().astype(bool)
    return float(w[cov].sum())

rows = []
instances = [(rho, 30) for rho in C.RHOS] + [(0.5, K) for K in C.K_GRID if K != 30]
for rho, K in instances:
    a_eu = load_npz(RD / f"a_Eucl_rho{rho}.npz").tocsr()
    a_m = load_npz(RD / f"a_sigmaMedium_b1.0_rho{rho}.npz").tocsr()
    reported, pool = solve_pool(a_eu, K)
    best_m, _ = solve_pool(a_m, K, npool=1)
    realized = [f_under(a_m, S) for S in pool]
    r0 = realized[0]
    rows.append({
        "rho": rho, "K": K,
        "Reported_pct": 100 * reported / TOT,
        "Realized_pct": 100 * r0 / TOT,
        "Best_pct": 100 * best_m / TOT,
        "ReportingGap_pp": 100 * (reported - r0) / TOT,
        "Regret_pp": 100 * (best_m - r0) / TOT,
        "n_pool": len(pool),
        "Realized_spread_pp": 100 * (max(realized) - min(realized)) / TOT,
    })
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(P / "r2_layer/regret.csv", index=False)
print("done", flush=True)
