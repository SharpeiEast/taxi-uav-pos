"""bd_scans.py -- R-7 (terminal params, no rerouting) + R-8 (payload sweep).

R-7: 18 eterm variants (r_pad x t_align x h_c) on cached Medium b1 route
     costs -> K=30 pop MILP x 4 rho.
R-8: m_p in {0, 0.5, 1.5} (1.0 = main): loaded outbound field re-routed per
     regime Medium (headline scope), empty return cached; matching eterm_mp.
Output: bd_layer/scan_terminal.csv, bd_layer/scan_payload.csv
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
CM = P / "r2_layer/costmat"
ET = np.load(P / "r2_layer/eterm.npz")

def load(n): return np.load(CM / f"{n}.npz")["C"].astype(np.float64)

joined = pd.read_parquet(P / "gk_run/joined_wuhan_gk.parquet")
dd = pd.read_parquet(P / "output_demand/wuhan_demand_pop_per_cell.parquet")
lut = dict(zip(zip(dd.grid_x, dd.grid_y), dd.pop_density.astype(float)))
w = np.array([lut.get((int(x), int(y)), 0.0) for x, y in
              zip(joined.grid_x, joined.grid_y)])
TOT = float(dd.pop_density.sum())

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

C_M_rt = np.nan_to_num(load("C_ld_sigmaMedium_b1.0") + load("C_e_sigmaMedium_b1.0"),
                       nan=np.inf, posinf=np.inf)

# ---- R-7 ----
rows = []
for rp in C.R_PAD_SWEEP:
    for ta in C.T_ALIGN_SWEEP:
        for hc in C.H_C_SWEEP:
            tag = f"rp{rp:g}_ta{ta:g}_hc{hc:g}"
            et = ET[f"eterm_{tag}"].astype(np.float64)
            ed = float(ET[f"edep_{tag}"][0])
            for rho in C.RHOS:
                cov = milp_cov(C_M_rt + ed + et[None, :] <= (1 - rho) * C.ETA_B)
                rows.append({"r_pad": rp, "t_align": ta, "h_c": hc,
                             "rho": rho, "cov_M": cov})
            print(f"R7 {tag} done", flush=True)
pd.DataFrame(rows).to_csv(P / "r2_layer/scan_terminal.csv", index=False)

# ---- R-8 ----
D = np.load(P / "c_layer/domain.npz")
F = np.load(P / "c_layer/sigma_fields.npz")
sigma = np.load(P / "r2_layer/field_r2.npz")["sigma_full"].astype(float)  # R2 denoised field
NV = len(sigma)
eu = D["edges_u"].astype(np.int64); ev = D["edges_v"].astype(np.int64)
el = D["edges_len"].astype(float)
J_nodes = D["J_nodes"].astype(np.int64)
cand = D["cand_nodes"].astype(np.int64)

def P_inst(v, m_p):
    v = np.maximum(np.asarray(v, dtype=float), 1e-9)
    kap = 1.0 + m_p * C.G / C.W_FRAME
    v0sq = kap * C.V0_HOV**2
    return (C.P_B * (1.0 + 3.0 * v**2 / C.U_TIP**2)
            + (kap**1.5) * C.P_IND * np.sqrt(
                np.sqrt(1.0 + v**4 / (4.0 * v0sq**2)) - v**2 / (2.0 * v0sq))
            + 0.5 * C.D0 * C.RHO_AIR * C.S_SOL * C.A_DISC * v**3)

reg = C.REGIMES_BD["Medium"]
vm = np.maximum(C.V_MIN, np.minimum(
    C.V_CRUISE, (reg.w_corr - reg.z_eps * sigma) / (C.KAPPA_PERP_MAIN * reg.tau_react)))
on = (sigma > reg.sigma_sensor).astype(float)
removed = sigma > reg.sigma_sat(C.KAPPA_PERP_MAIN)
C_e_ret = np.nan_to_num(load("C_e_sigmaMedium_b1.0"), nan=np.inf, posinf=np.inf)

rows = []
for mp in C.M_P_SWEEP:
    if mp == C.M_P_MAIN:
        C_out = np.nan_to_num(load("C_ld_sigmaMedium_b1.0"), nan=np.inf, posinf=np.inf)
    else:
        unit = P_inst(vm, mp) / vm + on * C.P_AUX / vm
        wgt = 0.5 * (unit[eu] + unit[ev]) * el
        ok = ~(removed[eu] | removed[ev])
        g = csr_matrix((np.concatenate([wgt[ok], wgt[ok]]),
                        (np.concatenate([eu[ok], ev[ok]]),
                         np.concatenate([ev[ok], eu[ok]]))), shape=(NV, NV))
        C_out = dijkstra(g, directed=False, indices=cand)[:, J_nodes]
        C_out = np.nan_to_num(C_out, nan=np.inf, posinf=np.inf)
    et = ET[f"eterm_mp{mp:g}"].astype(np.float64)
    ed = float(ET[f"edep_mp{mp:g}"][0])
    Ctot = C_out + C_e_ret + ed + et[None, :]
    for rho in C.RHOS:
        cov = milp_cov(Ctot <= (1 - rho) * C.ETA_B)
        rows.append({"m_p": mp, "rho": rho, "cov_M": cov})
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(P / "r2_layer/scan_payload.csv", index=False)
print("done", flush=True)
