"""bd_check02.py -- Check02 optional items:
(1) A0 vs Medium covered-set Jaccard at K=30 per rho (cells and pop-weighted
    overlap), backing the 'totals close, geography wrong' sentence.
(2) Exact min/max of Realized over the set of Euclidean optima (replaces the
    20-sample pool spread with a two-sided MILP bound). K=30, pop.
Output: bd_layer/check02_511.csv (+ stdout)
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
K = 30

def solve_cover(a_csr, sense_obj=None, K=30, fix_val=None, a_fix=None):
    """Max-coverage MILP under a_csr. If fix_val is given, constrain the
    a_fix-coverage to >= fix_val - eps and optimise sense_obj ('min'/'max')
    of the a_csr coverage instead."""
    ns, nc = a_csr.shape
    m = gp.Model(); m.Params.OutputFlag = 0; m.Params.MIPGap = 1e-6 if fix_val is None else 0.01
    m.Params.TimeLimit = 1800 if fix_val is not None else 1e9
    m.Params.Threads = 32
    x = m.addVars(ns, vtype=GRB.BINARY)
    m.addConstr(gp.quicksum(x[i] for i in range(ns)) <= K)
    for i in np.where(~CAND_MASK)[0]:
        m.addConstr(x[i] == 0)
    at = a_csr.T.tocsr()
    if fix_val is None:
        y = m.addVars(nc, vtype=GRB.CONTINUOUS, ub=1.0)
        for j in range(nc):
            idx = at.indices[at.indptr[j]:at.indptr[j + 1]]
            m.addConstr(y[j] <= (gp.quicksum(x[i] for i in idx) if len(idx) else 0))
        m.setObjective(gp.quicksum(float(w[j]) * y[j] for j in range(nc)),
                       GRB.MAXIMIZE)
        m.optimize()
        S = [i for i in range(ns) if x[i].X > 0.5]
        return m.ObjVal, S
    # two-sided: a_fix coverage held at optimum, optimise a_csr coverage
    atf = a_fix.T.tocsr()
    yf = m.addVars(nc, vtype=GRB.CONTINUOUS, ub=1.0)
    for j in range(nc):
        idx = atf.indices[atf.indptr[j]:atf.indptr[j + 1]]
        m.addConstr(yf[j] <= (gp.quicksum(x[i] for i in idx) if len(idx) else 0))
    m.addConstr(gp.quicksum(float(w[j]) * yf[j] for j in range(nc))
                >= fix_val - 1e-4 * max(1.0, abs(fix_val)))
    if sense_obj == "max":
        y = m.addVars(nc, vtype=GRB.CONTINUOUS, ub=1.0)
        for j in range(nc):
            idx = at.indices[at.indptr[j]:at.indptr[j + 1]]
            m.addConstr(y[j] <= (gp.quicksum(x[i] for i in idx) if len(idx) else 0))
        m.setObjective(gp.quicksum(float(w[j]) * y[j] for j in range(nc)),
                       GRB.MAXIMIZE)
    else:
        z = m.addVars(nc, vtype=GRB.BINARY)
        for j in range(nc):
            idx = at.indices[at.indptr[j]:at.indptr[j + 1]]
            for i in idx:
                m.addConstr(z[j] >= x[i])
        m.setObjective(gp.quicksum(float(w[j]) * z[j] for j in range(nc)),
                       GRB.MINIMIZE)
    m.optimize()
    return m.ObjVal, None

_csm = pd.read_parquet(P / "v2/data_intermediate/demand_supply/candidate_screens.parquet").sort_values("cand_idx")
CAND_MASK = _csm["base"].values.astype(bool)

rows = []
for rho in C.RHOS:
    a_m = load_npz(RD / f"a_sigmaMedium_b1.0_rho{rho}.npz").tocsr()
    a_a0 = load_npz(RD / f"a_A0phi0.7_rho{rho}.npz").tocsr()
    a_eu = load_npz(RD / f"a_Eucl_rho{rho}.npz").tocsr()
    # (1) Jaccard of covered-cell sets of the two plans (each under own model)
    vm, Sm = solve_cover(a_m)
    va, Sa = solve_cover(a_a0)
    cov_m = np.asarray(a_m[Sm].max(axis=0).todense()).ravel().astype(bool)
    cov_a = np.asarray(a_a0[Sa].max(axis=0).todense()).ravel().astype(bool)
    inter = cov_m & cov_a; union = cov_m | cov_a
    jac_cells = inter.sum() / max(1, union.sum())
    jac_pop = w[inter].sum() / max(1e-9, w[union].sum())
    st_jac = len(set(Sm) & set(Sa)) / max(1, len(set(Sm) | set(Sa)))
    # (2) exact Realized bounds over Euclidean optima
    ve, _ = solve_cover(a_eu)
    r_max, _ = solve_cover(a_m, "max", fix_val=ve, a_fix=a_eu)
    r_min, _ = solve_cover(a_m, "min", fix_val=ve, a_fix=a_eu)
    rows.append({"rho": rho,
                 "covM_pct": 100 * vm / TOT, "covA0_pct": 100 * va / TOT,
                 "jaccard_cells": jac_cells, "jaccard_pop": jac_pop,
                 "jaccard_stations": st_jac,
                 "realized_min_pct": 100 * r_min / TOT,
                 "realized_max_pct": 100 * r_max / TOT})
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(P / "cs511/check02_511.csv", index=False)
print("done", flush=True)
