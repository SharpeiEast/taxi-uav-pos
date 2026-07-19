"""trc_e3_fields.py -- E3 keep-at-vmin cost matrices (no removal).

Same physics as bd_fields.py, Medium regime, beta = 1.0, but saturated cells
(sigma > sat) are NOT removed from the routing graph: they are traversable at
the speed floor v_min, i.e. per-metre cost E_phys = e(v_min) + P_aux/v_min
(K_E ~ 182 J/m empty). Tests whether the removal-at-saturation modelling
choice drives the headline results.

Output: trc_exp/costmat/C_{ld,e}_keepMedium.npz
"""
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import sys

sys.path.insert(0, "/lustre/home/2406393544/sharefolder/proj3/bd_layer")
import bd_config as C

P = Path("/lustre/home/2406393544/sharefolder/proj3")
OUT = P / "trc_exp/r2base/costmat"
OUT.mkdir(parents=True, exist_ok=True)

D = np.load(P / "c_layer/domain.npz")
F = np.load(P / "c_layer/sigma_fields.npz")
sigma = np.load(P / "r2_layer/field_r2.npz")["sigma_full"].astype(float)
NV = len(sigma)
eu = D["edges_u"].astype(np.int64); ev = D["edges_v"].astype(np.int64)
el = D["edges_len"].astype(float)
J_nodes = D["J_nodes"].astype(np.int64)
cand_nodes = D["cand_nodes"].astype(np.int64)
print(f"|V|={NV:,} edges={len(eu):,}", flush=True)

def P_inst(v, m_p):
    v = np.maximum(np.asarray(v, dtype=float), 1e-9)
    kap = 1.0 + m_p * C.G / C.W_FRAME
    blade = C.P_B * (1.0 + 3.0 * v ** 2 / C.U_TIP ** 2)
    v0sq = kap * C.V0_HOV ** 2
    induced = (kap ** 1.5) * C.P_IND * np.sqrt(
        np.sqrt(1.0 + v ** 4 / (4.0 * v0sq ** 2)) - v ** 2 / (2.0 * v0sq))
    parasite = 0.5 * C.D0 * C.RHO_AIR * C.S_SOL * C.A_DISC * v ** 3
    return blade + induced + parasite

def e_of_v(v, m_p):
    return P_inst(v, m_p) / np.maximum(np.asarray(v, dtype=float), 1e-9)

reg = C.REGIMES_BD["Medium"]

def v_max(sig_arr):
    raw = (reg.w_corr - reg.z_eps * sig_arr) / (0.3 * reg.tau_react)
    return np.maximum(C.V_MIN, np.minimum(C.V_CRUISE, raw))

def E_phys(sig_arr, m_p):
    vm = v_max(sig_arr)
    on = (sig_arr > reg.sigma_sensor).astype(float)
    return e_of_v(vm, m_p) + on * C.P_AUX / vm

def run_dijkstra(unit):
    w = 0.5 * (unit[eu] + unit[ev]) * el
    g = csr_matrix((np.concatenate([w, w]),
                    (np.concatenate([eu, ev]), np.concatenate([ev, eu]))),
                   shape=(NV, NV))
    return dijkstra(g, directed=False, indices=cand_nodes)[:, J_nodes].astype(np.float32)

for leg, mp in [("ld", C.M_P_MAIN), ("e", 0.0)]:
    unit = E_phys(sigma, mp)
    sat_mask = sigma > reg.sigma_sat(0.3)
    print(f"{leg}: unit@sat cells median {np.median(unit[sat_mask]):.1f} J/m "
          f"(K_E-type), n_sat={sat_mask.sum():,}", flush=True)
    Cr = run_dijkstra(unit)
    assert np.isfinite(Cr).all(), "keep-variant must have no unreachable pairs"
    np.savez_compressed(OUT / f"C_{leg}_keepMedium.npz", C=Cr)
    print(f"saved C_{leg}_keepMedium ({Cr.shape})", flush=True)
print("done", flush=True)


# ================= analysis =================

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
OUT = P / "trc_exp"

joined = pd.read_parquet(P / "gk_run/joined_wuhan_gk.parquet")
sig = np.load(P / "r2_layer/field_r2.npz")["sig_j"].astype(float)
dd = pd.read_parquet(P / "output_demand/wuhan_demand_pop_per_cell.parquet")
lut = dict(zip(zip(dd.grid_x, dd.grid_y), dd.pop_density.astype(float)))
w = np.array([lut.get((int(x), int(y)), 0.0) for x, y in
              zip(joined.grid_x, joined.grid_y)])
TOT = float(dd.pop_density.sum())

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

ET = np.load(P / "r2_layer/eterm.npz")
eterm = ET["eterm_main"].astype(np.float64)
eterm0 = float(ET["eterm0_main"][0]); edep = float(ET["edep_main"][0])

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
    assert m.Status == GRB.OPTIMAL
    return 100.0 * m.ObjVal / TOT

C_eucl_rt = DE * 11.1582 + DE * 9.2365
C_rm = np.nan_to_num(
    np.load(P / "r2_layer/costmat/C_ld_sigmaMedium_b1.0.npz")["C"].astype(np.float64)
    + np.load(P / "r2_layer/costmat/C_e_sigmaMedium_b1.0.npz")["C"].astype(np.float64),
    nan=np.inf, posinf=np.inf)
C_keep = (np.load(OUT / "r2base/costmat/C_ld_keepMedium.npz")["C"].astype(np.float64)
          + np.load(OUT / "r2base/costmat/C_e_keepMedium.npz")["C"].astype(np.float64))

rows = []
for tag, Crt in [("removal_headline", C_rm), ("keep_at_vmin", C_keep)]:
    best = Crt.min(axis=0)
    for rho in C.RHOS:
        B = (1 - rho) * C.ETA_B
        blocked = ~np.isfinite(best)
        over = np.isfinite(best) & (best + edep + eterm0 > B)
        term = (np.isfinite(best) & (best + edep + eterm0 <= B)
                & (best + edep + eterm > B))
        covered = ~(blocked | over | term)
        row = {"variant": tag, "rho": rho,
               "blocked_pct": 100 * w[blocked].sum() / TOT,
               "overrange_pct": 100 * w[over].sum() / TOT,
               "terminal_pct": 100 * w[term].sum() / TOT,
               "ceiling_covered_pct": 100 * w[covered].sum() / TOT}
        if rho in (0.0, 0.2, 0.5):
            cov_eu = milp_cov(C_eucl_rt + edep + eterm0 <= B)
            covM = milp_cov(Crt + edep + eterm[None, :] <= B)
            row.update(cov_eucl=cov_eu, cov_M=covM, gap_pp=cov_eu - covM)
        rows.append(row)
        print(row, flush=True)
pd.DataFrame(rows).to_csv(OUT / "r2base/e3_keepvmin.csv", index=False)
print("done", flush=True)
