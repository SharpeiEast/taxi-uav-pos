"""bd_kperp.py -- R-4: kappa_perp sensitivity {0.1, 0.3} (main 1.0 = R-1).

For each kappa_perp: rebuild v_max -> unit costs (loaded+empty, m_p=1.0,
beta=1) -> per-regime removal at sat(kp) -> Dijkstra -> reach under the
full criterion -> K=30 pop MILP at 4 rho. Eucl leg unchanged (no speed law).
Honest direction expectation: smaller kp -> higher v_max -> weaker route
sigma effect -> gap narrows; terminal term unaffected.
Output: bd_layer/kperp_k1.csv
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import gurobipy as gp
from gurobipy import GRB

sys.path.insert(0, "/lustre/home/2406393544/sharefolder/proj3/r2_layer")
import r2_config as C

P = Path("/lustre/home/2406393544/sharefolder/proj3")
ET = np.load(P / "r2_layer/eterm.npz")
eterm = ET["eterm_main"].astype(np.float64)
eterm0 = float(ET["eterm0_main"][0]); edep = float(ET["edep_main"][0])

D = np.load(P / "c_layer/domain.npz")
F = np.load(P / "c_layer/sigma_fields.npz")
sigma = np.load(P / "r2_layer/field_r2.npz")["sigma_full"].astype(float)  # R2 denoised field
NV = len(sigma)
eu = D["edges_u"].astype(np.int64); ev = D["edges_v"].astype(np.int64)
el = D["edges_len"].astype(float)
J_nodes = D["J_nodes"].astype(np.int64)
cand = D["cand_nodes"].astype(np.int64)

joined = pd.read_parquet(P / "gk_run/joined_wuhan_gk.parquet")
dd = pd.read_parquet(P / "output_demand/wuhan_demand_pop_per_cell.parquet")
lut = dict(zip(zip(dd.grid_x, dd.grid_y), dd.pop_density.astype(float)))
w = np.array([lut.get((int(x), int(y)), 0.0) for x, y in
              zip(joined.grid_x, joined.grid_y)])
TOT = float(dd.pop_density.sum())

node_grid = D["node_grid"]; ii, jj = np.nonzero(node_grid >= 0)
ids = node_grid[ii, jj]
gx0, gy0 = int(D["gx0"]), int(D["gy0"])
gx_n = np.empty(ids.max() + 1, np.int64); gy_n = np.empty(ids.max() + 1, np.int64)
gx_n[ids] = gx0 + ii; gy_n[ids] = gy0 + jj
if not np.array_equal(gx_n[J_nodes], joined.grid_x.values):
    gx_n[ids] = gx0 + jj; gy_n[ids] = gy0 + ii
jx = (joined.grid_x.values + 0.5) * 100.0
jy = (joined.grid_y.values + 0.5) * 100.0
DE = np.hypot(((gx_n[cand] + 0.5) * 100.0)[:, None] - jx[None, :],
              ((gy_n[cand] + 0.5) * 100.0)[:, None] - jy[None, :])
C_eucl = DE * 11.1582 + DE * 9.2365 + edep + eterm0

def P_inst(v, m_p):
    v = np.maximum(np.asarray(v, dtype=float), 1e-9)
    kap = 1.0 + m_p * C.G / C.W_FRAME
    v0sq = kap * C.V0_HOV**2
    return (C.P_B * (1.0 + 3.0 * v**2 / C.U_TIP**2)
            + (kap**1.5) * C.P_IND * np.sqrt(
                np.sqrt(1.0 + v**4 / (4.0 * v0sq**2)) - v**2 / (2.0 * v0sq))
            + 0.5 * C.D0 * C.RHO_AIR * C.S_SOL * C.A_DISC * v**3)

def E_phys(sig, reg, m_p, kp):
    vm = np.maximum(C.V_MIN, np.minimum(
        C.V_CRUISE, (reg.w_corr - reg.z_eps * sig) / (kp * reg.tau_react)))
    on = (sig > reg.sigma_sensor).astype(float)
    return P_inst(vm, m_p) / vm + on * C.P_AUX / vm

def run_dij(unit, removed):
    wgt = 0.5 * (unit[eu] + unit[ev]) * el
    ok = ~(removed[eu] | removed[ev])
    g = csr_matrix((np.concatenate([wgt[ok], wgt[ok]]),
                    (np.concatenate([eu[ok], ev[ok]]),
                     np.concatenate([ev[ok], eu[ok]]))), shape=(NV, NV))
    return dijkstra(g, directed=False, indices=cand)[:, J_nodes]

def milp_cov(mask, K=30):
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

rows = []
for kp in [1.0]:
    for rname, reg in C.REGIMES_BD.items():
        sat = reg.sigma_sat(kp)
        removed = sigma > sat
        Crt = (run_dij(E_phys(sigma, reg, C.M_P_MAIN, kp), removed)
               + run_dij(E_phys(sigma, reg, 0.0, kp), removed))
        Crt = np.nan_to_num(Crt, nan=np.inf, posinf=np.inf)
        Ctot = Crt + edep + eterm[None, :]
        for rho in C.RHOS:
            B = (1 - rho) * C.ETA_B
            cov = milp_cov(Ctot <= B)
            cov_e = milp_cov(C_eucl <= B)
            rows.append({"kperp": kp, "regime": rname, "rho": rho,
                         "cov_eucl": cov_e, "cov_sigma": cov,
                         "gap_pp": cov_e - cov, "sat_m": sat,
                         "removed": int(removed.sum())})
            print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(P / "r2_layer/kperp_k1.csv", index=False)
print("done", flush=True)
