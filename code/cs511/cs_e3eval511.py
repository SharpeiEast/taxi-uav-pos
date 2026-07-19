"""cs_e3eval511.py -- keep-at-vmin vs removal at 511 (matrices reused)."""
import numpy as np, pandas as pd, sys
from pathlib import Path
from scipy.sparse import csr_matrix
import gurobipy as gp
from gurobipy import GRB
sys.path.insert(0, "/lustre/home/2406393544/sharefolder/proj3/r2_layer")
import r2_config as C
P = Path("/lustre/home/2406393544/sharefolder/proj3")
_cs = pd.read_parquet(P / "v2/data_intermediate/demand_supply/candidate_screens.parquet").sort_values("cand_idx")
MASK = _cs["base"].values.astype(bool)
joined = pd.read_parquet(P / "gk_run/joined_wuhan_gk.parquet")
dd = pd.read_parquet(P / "output_demand/wuhan_demand_pop_per_cell.parquet")
lut = dict(zip(zip(dd.grid_x, dd.grid_y), dd.pop_density.astype(float)))
w = np.array([lut.get((int(x), int(y)), 0.0) for x, y in zip(joined.grid_x, joined.grid_y)])
TOT = float(dd.pop_density.sum())
ET = np.load(P / "r2_layer/eterm.npz")
eterm = ET["eterm_main"].astype(float)
eterm0 = float(ET["eterm0_main"][0]); edep = float(ET["edep_main"][0])
D = np.load(P / "c_layer/domain.npz")
node_grid = D["node_grid"]; ii, jj = np.nonzero(node_grid >= 0)
ids = node_grid[ii, jj]; gx0, gy0 = int(D["gx0"]), int(D["gy0"])
gx_n = np.empty(ids.max()+1, np.int64); gy_n = np.empty(ids.max()+1, np.int64)
gx_n[ids] = gx0+ii; gy_n[ids] = gy0+jj
J = D["J_nodes"].astype(np.int64)
if not np.array_equal(gx_n[J], joined.grid_x.values):
    gx_n[ids] = gx0+jj; gy_n[ids] = gy0+ii
cand = D["cand_nodes"].astype(np.int64)
jx = (joined.grid_x.values+0.5)*100.0; jy = (joined.grid_y.values+0.5)*100.0
DE = np.hypot(((gx_n[cand]+0.5)*100.0)[:,None]-jx[None,:], ((gy_n[cand]+0.5)*100.0)[:,None]-jy[None,:])
C_eucl = DE*11.1582 + DE*9.2365

def milp(mask, K=30):
    at = csr_matrix(mask).T.tocsr(); ns, nc = mask.shape
    m = gp.Model(); m.Params.OutputFlag=0; m.Params.MIPGap=1e-6; m.Params.Threads=32
    x = m.addVars(ns, vtype=GRB.BINARY); y = m.addVars(nc, ub=1.0)
    m.addConstr(gp.quicksum(x[i] for i in range(ns)) <= K)
    for j in range(nc):
        idx = at.indices[at.indptr[j]:at.indptr[j+1]]
        m.addConstr(y[j] <= (gp.quicksum(x[i] for i in idx) if len(idx) else 0))
    m.setObjective(gp.quicksum(float(w[j])*y[j] for j in range(nc)), GRB.MAXIMIZE)
    m.optimize(); assert m.Status == GRB.OPTIMAL
    return 100.0*m.ObjVal/TOT

C_rm = np.nan_to_num(np.load(P/"r2_layer/costmat/C_ld_sigmaMedium_b1.0.npz")["C"].astype(float)
    + np.load(P/"r2_layer/costmat/C_e_sigmaMedium_b1.0.npz")["C"].astype(float), nan=np.inf, posinf=np.inf)
C_keep = (np.load(P/"trc_exp/r2base/costmat/C_ld_keepMedium.npz")["C"].astype(float)
    + np.load(P/"trc_exp/r2base/costmat/C_e_keepMedium.npz")["C"].astype(float))
rows = []
for tag, Crt in [("removal", C_rm), ("keep_at_vmin", C_keep)]:
    for rho in (0.0, 0.2, 0.5):
        B = (1-rho)*C.ETA_B
        fe = (C_eucl + edep + eterm0 <= B) & MASK[:,None]
        fm = (Crt + edep + eterm[None,:] <= B) & MASK[:,None]
        rows.append({"variant": tag, "rho": rho, "cand_set": 511,
                     "cov_eucl": milp(fe), "cov_M": milp(fm)})
        rows[-1]["gap_pp"] = rows[-1]["cov_eucl"] - rows[-1]["cov_M"]
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(P / "cs511/e3_keepvmin_511.csv", index=False)
print("done", flush=True)
