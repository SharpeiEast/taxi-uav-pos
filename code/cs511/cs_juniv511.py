"""r2_juniv.py -- planning-universe sensitivity (expert #4, item 八.1).

The headline demand universe (17,899 cells) requires a complete
building-feature vector, which is only needed for the transferability
test. This job re-evaluates coverage on the feature-agnostic universe of
ALL 26,032 observed-sigma cells (denoised R2 field values), Medium
regime, kappa=0.3, beta=1: K=30 MILP coverage + Eucl baseline + ceiling,
per rho. Output: trc_exp/r2base/juniv.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import gurobipy as gp
from gurobipy import GRB
import sys

sys.path.insert(0, "/lustre/home/2406393544/sharefolder/proj3/r2_layer")
import r2_config as C

KP = C.KAPPA_PERP_MAIN
P = Path("/lustre/home/2406393544/sharefolder/proj3")
OUT = P / "trc_exp/r2base"

D = np.load(P / "c_layer/domain.npz")
sigma = np.load(P / "r2_layer/field_r2.npz")["sigma_full"].astype(float)
NV = len(sigma)
eu = D["edges_u"].astype(np.int64); ev = D["edges_v"].astype(np.int64)
el = D["edges_len"].astype(float)
obs_nodes = D["obs_nodes"].astype(np.int64)
cand = D["cand_nodes"].astype(np.int64)
node_grid = D["node_grid"]

def P_inst(v, m_p):
    v = np.maximum(np.asarray(v, float), 1e-9)
    kap = 1.0 + m_p * C.G / C.W_FRAME
    v0sq = kap * C.V0_HOV ** 2
    return (C.P_B * (1 + 3 * v**2 / C.U_TIP**2)
            + (kap**1.5) * C.P_IND * np.sqrt(np.sqrt(1 + v**4/(4*v0sq**2)) - v**2/(2*v0sq))
            + 0.5 * C.D0 * C.RHO_AIR * C.S_SOL * C.A_DISC * v**3)

reg = C.REGIMES_BD["Medium"]

def E_phys(sig, m_p):
    vm = np.maximum(C.V_MIN, np.minimum(C.V_CRUISE,
        (reg.w_corr - reg.z_eps * sig) / (KP * reg.tau_react)))
    on = (sig > reg.sigma_sensor).astype(float)
    return P_inst(vm, m_p) / vm + on * C.P_AUX / vm

def dij(unit, removed):
    w = 0.5 * (unit[eu] + unit[ev]) * el
    ok = ~(removed[eu] | removed[ev])
    g = csr_matrix((np.concatenate([w[ok], w[ok]]),
                    (np.concatenate([eu[ok], ev[ok]]),
                     np.concatenate([ev[ok], eu[ok]]))), shape=(NV, NV))
    return dijkstra(g, directed=False, indices=cand)[:, obs_nodes]

removed = sigma > reg.sigma_sat(KP)
Crt = np.nan_to_num(dij(E_phys(sigma, C.M_P_MAIN), removed)
                    + dij(E_phys(sigma, 0.0), removed),
                    nan=np.inf, posinf=np.inf)

# grid coords of observed nodes -> pop weights
ii, jj = np.nonzero(node_grid >= 0)
ids = node_grid[ii, jj]
gx0, gy0 = int(D["gx0"]), int(D["gy0"])
gx_n = np.empty(ids.max() + 1, np.int64); gy_n = np.empty(ids.max() + 1, np.int64)
gx_n[ids] = gx0 + ii; gy_n[ids] = gy0 + jj
joined = pd.read_parquet(P / "gk_run/joined_wuhan_gk.parquet")
J_nodes = D["J_nodes"].astype(np.int64)
if not np.array_equal(gx_n[J_nodes], joined.grid_x.values):
    gx_n[ids] = gx0 + jj; gy_n[ids] = gy0 + ii
dd = pd.read_parquet(P / "output_demand/wuhan_demand_pop_per_cell.parquet")
lut = dict(zip(zip(dd.grid_x, dd.grid_y), dd.pop_density.astype(float)))
w = np.array([lut.get((int(gx_n[n]), int(gy_n[n])), 0.0) for n in obs_nodes])
TOT = float(w.sum())
print(f"universe 26,032: pop-carrying cells {(w>0).sum():,}, TOT share of "
      f"city pop {TOT/float(dd.pop_density.sum()):.4f}", flush=True)

# terminal energies on observed cells
sys.path.insert(0, str(P / "r2_layer"))
sig_o = sigma[obs_nodes]
ph_ld = C.P_B + C.P_IND * (1 + C.M_P_MAIN * C.G / C.W_FRAME) ** 1.5
ph_e = C.P_B + C.P_IND
t_v = C.H_C / C.V_CLIMB
DJ = (ph_ld - (C.W_FRAME + C.M_P_MAIN * C.G) * C.V_DESC / 2) * t_v
CJ = (ph_e + C.W_FRAME * C.V_CLIMB / 2) * t_v
CI = (ph_ld + (C.W_FRAME + C.M_P_MAIN * C.G) * C.V_CLIMB / 2) * t_v
DI = (ph_e - C.W_FRAME * C.V_DESC / 2) * t_v
E_GA = 2.0 * ph_ld * C.T_GA
EDEP = CI + DI
q = 1 - np.exp(-C.R_PAD_MAIN**2 / (2 * np.maximum(sig_o, 1e-12)**2))
N = 1 / q
eterm = DJ + N * ph_ld * C.T_ALIGN + (N - 1) * E_GA + ph_ld * C.T_REL + CJ
eterm0 = DJ + ph_ld * C.T_ALIGN + ph_ld * C.T_REL + CJ

# Euclid baseline distances (snapped centres)
jx = (gx_n[obs_nodes] + 0.5) * 100.0
jy = (gy_n[obs_nodes] + 0.5) * 100.0
DE = np.hypot(((gx_n[cand] + 0.5) * 100.0)[:, None] - jx[None, :],
              ((gy_n[cand] + 0.5) * 100.0)[:, None] - jy[None, :])
C_eucl = DE * 11.1582 + DE * 9.2365

_csm = pd.read_parquet(P / "v2/data_intermediate/demand_supply/candidate_screens.parquet").sort_values("cand_idx")
CAND_MASK = _csm["base"].values.astype(bool)

def milp_cov(mask, K=30):
    mask = mask & CAND_MASK[:, None]
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
    assert m.Status == GRB.OPTIMAL
    return 100.0 * m.ObjVal / TOT

rows = []
for rho in C.RHOS:
    B = (1 - rho) * C.ETA_B
    feas = (Crt + EDEP + eterm[None, :] <= B) & CAND_MASK[:, None]
    row = {"rho": rho, "universe": 26032,
           "ceiling_pct": 100.0 * w[feas.any(axis=0)].sum() / TOT}
    if rho in (0.0, 0.2, 0.5):
        cov_eu = milp_cov(C_eucl + EDEP + eterm0 <= B)
        covM = milp_cov(feas)
        row.update(cov_eucl=cov_eu, cov_M=covM, gap_pp=cov_eu - covM)
    rows.append(row); print(row, flush=True)
pd.DataFrame(rows).to_csv(P / "cs511/juniv_511.csv", index=False)
print("done", flush=True)
