#!/usr/bin/env python3
"""a0_eval_511.py -- evaluate the fixed-radius baseline's station sets under
the Medium-aware feasibility matrix (Proposition-3 framing for A0).

For each reserve level: solve the A0 model with the reproducible
redundancy-first secondary objective (max coverage, then, among optima,
max redundant coverage; both stages to proven optimality), then evaluate
the returned station set S_A0 under the Medium matrix:
  Reported = f_A0(S_A0),  Realized = f_M(S_A0),  Best = f_M(S_M),
  ReportingGap = Reported - Realized,  Regret = Best - Realized.
Output: cs511/a0_eval_511.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import load_npz
import gurobipy as gp
from gurobipy import GRB

P = Path("/lustre/home/2406393544/sharefolder/proj3")
K = 30

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

def two_stage(a):
    """Redundancy-first policy on matrix a; returns (station idx, reported%)."""
    ns, nj = a.shape
    m = gp.Model(); m.Params.OutputFlag = 0; m.Params.Threads = 16
    m.Params.MIPGap = 1e-9
    x = m.addVars(ns, vtype=GRB.BINARY)
    y = m.addVars(nj, vtype=GRB.BINARY)
    m.addConstr(gp.quicksum(x[i] for i in range(ns)) == K)
    covers = [np.nonzero(a[:, j])[0] for j in range(nj)]
    for j in range(nj):
        if len(covers[j]):
            m.addConstr(y[j] <= gp.quicksum(x[int(i)] for i in covers[j]))
        else:
            m.addConstr(y[j] == 0)
    m.setObjective(gp.quicksum(w[j] * y[j] for j in range(nj)), GRB.MAXIMIZE)
    m.optimize()
    assert m.Status == GRB.OPTIMAL
    v1 = m.ObjVal
    # stage 2: among coverage-optimal sets, maximise redundant coverage
    z = m.addVars(nj, vtype=GRB.BINARY)
    for j in range(nj):
        if len(covers[j]) >= 2:
            m.addConstr(2 * z[j] <= gp.quicksum(x[int(i)] for i in covers[j]))
        else:
            m.addConstr(z[j] == 0)
    m.addConstr(gp.quicksum(w[j] * y[j] for j in range(nj)) >= v1 - 1e-6)
    m.setObjective(gp.quicksum(w[j] * z[j] for j in range(nj)), GRB.MAXIMIZE)
    m.optimize()
    assert m.Status == GRB.OPTIMAL
    sel = np.array([i for i in range(ns) if x[i].X > 0.5])
    return sel, 100.0 * sum(w[j] * y[j].X for j in range(nj)) / TOT

def coverage_of(a, sel):
    cov = a[sel].any(axis=0)
    return 100.0 * w[cov].sum() / TOT

milp = pd.concat([pd.read_csv(P / f"cs511/milp_pop_rho{r}.csv")
                  for r in ("0.0", "0.2", "0.5", "0.8")])
rows = []
for rho in (0.0, 0.2, 0.5, 0.8):
    aA0 = load_npz(P / f"r2_layer/reach/a_A0phi0.7_rho{rho}.npz").toarray()[MASK]
    aM = load_npz(P / f"r2_layer/reach/a_sigmaMedium_b1.0_rho{rho}.npz"
                  ).toarray()[MASK]
    sel, reported = two_stage(aA0.astype(bool))
    realized = coverage_of(aM.astype(bool), sel)
    best = milp[(milp.rho == rho) & (milp.ablation == "Medium")
                & (milp.K == 30)].coverage_pct.iloc[0]
    rows.append({"rho": rho, "K": K, "phi": 0.7, "cand_set": int(MASK.sum()),
                 "policy": "max_coverage_then_max_redundant",
                 "reported_pct": reported, "realized_pct": realized,
                 "best_pct": best,
                 "reporting_gap_pp": reported - realized,
                 "regret_pp": best - realized})
    print(f"rho={rho}: Reported {reported:.2f}  Realized {realized:.2f}  "
          f"Best {best:.2f}  RepGap {reported-realized:.2f}  "
          f"Regret {best-realized:.2f}", flush=True)

pd.DataFrame(rows).to_csv(P / "cs511/a0_eval_511.csv", index=False)
print("saved cs511/a0_eval_511.csv", flush=True)
