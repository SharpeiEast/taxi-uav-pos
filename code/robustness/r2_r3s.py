"""trc_r3a_screens.py -- part (c) only: screened candidate sets, fixed mapping.

candidate_screens.parquet carries cand_idx + graph node id; verify the node
ids match c_layer/domain cand_nodes, then use the boolean screen columns
directly in candidate order.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix
import gurobipy as gp
from gurobipy import GRB
import sys

sys.path.insert(0, "/lustre/home/2406393544/sharefolder/proj3/bd_layer")
import bd_config as C

P = Path("/lustre/home/2406393544/sharefolder/proj3")
OUT = P / "trc_exp/r2base"

joined = pd.read_parquet(P / "gk_run/joined_wuhan_gk.parquet")
dd = pd.read_parquet(P / "output_demand/wuhan_demand_pop_per_cell.parquet")
lut = dict(zip(zip(dd.grid_x, dd.grid_y), dd.pop_density.astype(float)))
w = np.array([lut.get((int(x), int(y)), 0.0) for x, y in
              zip(joined.grid_x, joined.grid_y)])
TOT = float(dd.pop_density.sum())

D = np.load(P / "c_layer/domain.npz")
cand_nodes = D["cand_nodes"].astype(np.int64)

cs = pd.read_parquet(
    P / "v2/data_intermediate/demand_supply/candidate_screens.parquet"
).sort_values("cand_idx").reset_index(drop=True)
assert len(cs) == len(cand_nodes), (len(cs), len(cand_nodes))
match = (cs["node"].values.astype(np.int64) == cand_nodes)
print(f"node-id match: {match.mean():.4f}", flush=True)
assert match.all(), "cand_idx order does not match domain cand_nodes"

ET = np.load(P / "r2_layer/eterm.npz")
eterm = ET["eterm_main"].astype(np.float64)
edep = float(ET["edep_main"][0])
route_M = np.nan_to_num(
    np.load(P / "r2_layer/costmat/C_ld_sigmaMedium_b1.0.npz")["C"].astype(np.float64)
    + np.load(P / "r2_layer/costmat/C_e_sigmaMedium_b1.0.npz")["C"].astype(np.float64),
    nan=np.inf, posinf=np.inf)

def milp_cov(mask, K=30):
    at = csr_matrix(mask).T.tocsr()
    ns, nc = mask.shape
    m = gp.Model(); m.Params.OutputFlag = 0; m.Params.MIPGap = 1e-6
    m.Params.Threads = 8
    x = m.addVars(ns, vtype=GRB.BINARY)
    y = m.addVars(nc, vtype=GRB.CONTINUOUS, ub=1.0)
    m.addConstr(gp.quicksum(x[i] for i in range(ns)) <= K)
    for j in range(nc):
        idx = at.indices[at.indptr[j]:at.indptr[j + 1]]
        m.addConstr(y[j] <= (gp.quicksum(x[i] for i in idx) if len(idx) else 0))
    m.setObjective(gp.quicksum(float(w[j]) * y[j] for j in range(nc)), GRB.MAXIMIZE)
    m.optimize()
    assert m.Status == GRB.OPTIMAL
    return 100.0 * m.ObjVal / TOT

rows = []
for scol in ("permissive", "base", "strict"):
    cmask = cs[scol].values.astype(bool)
    for rho in (0.0, 0.2, 0.5):
        B = (1 - rho) * C.ETA_B
        feas = (route_M + edep + eterm[None, :] <= B) & cmask[:, None]
        covM = milp_cov(feas)
        # ceiling with screened set
        ceil = 100.0 * w[feas.any(axis=0)].sum() / TOT
        rows.append({"screen": scol, "n_cand": int(cmask.sum()), "rho": rho,
                     "cov_M_K30": covM, "ceiling_all_screened": ceil})
        print(rows[-1], flush=True)
# unscreened reference
for rho in (0.0, 0.2, 0.5):
    B = (1 - rho) * C.ETA_B
    feas = route_M + edep + eterm[None, :] <= B
    rows.append({"screen": "none_605", "n_cand": 605, "rho": rho,
                 "cov_M_K30": milp_cov(feas),
                 "ceiling_all_screened": 100.0 * w[feas.any(axis=0)].sum() / TOT})
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(OUT / "r3a_screens.csv", index=False)
print("done", flush=True)
