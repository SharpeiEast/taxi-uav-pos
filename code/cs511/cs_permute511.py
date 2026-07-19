"""bd_permute.py -- R-5: spatial permutation + constant-field counterfactuals.

Does the SPATIAL ARRANGEMENT of sigma (not just its marginal distribution)
drive the planning outcome? (backs the Sec-3.2 per-city-sensing claim, H13)

Permutation: sigma values permuted among the 17,899 J cells (marginals
preserved); imputed transit cells held fixed. Rebuild Medium beta=1 cost
fields (loaded+empty Dijkstra), E_term permuted likewise, K=30 pop MILP at
rho in {0, 0.5, 0.8}. Constant fields (worker 0): sigma == median 4.37 and
== p75 6.14 everywhere in J (transit fixed).

env WORKER 0..4; 10 permutations each (seeds 60000+draw).
Output: bd_layer/perm/gaps_w{W}.csv
"""
import os
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
OUT = P / "r2_layer/perm"
OUT.mkdir(exist_ok=True)
W_ID = int(os.environ.get("WORKER", "0"))
N_LOC = 10
RHOS = [0.0, 0.5, 0.8]

D = np.load(P / "c_layer/domain.npz")
F = np.load(P / "c_layer/sigma_fields.npz")
sigma0 = np.load(P / "r2_layer/field_r2.npz")["sigma_full"].astype(float)
NV = len(sigma0)
eu = D["edges_u"].astype(np.int64); ev = D["edges_v"].astype(np.int64)
el = D["edges_len"].astype(float)
J_nodes = D["J_nodes"].astype(np.int64)
cand = D["cand_nodes"].astype(np.int64)
_cs = pd.read_parquet(P / "v2/data_intermediate/demand_supply/candidate_screens.parquet").sort_values("cand_idx")
CAND_MASK = _cs["base"].values.astype(bool)
assert ( _cs["node"].values.astype(np.int64) == cand ).all()

joined = pd.read_parquet(P / "gk_run/joined_wuhan_gk.parquet")
sig_j0 = joined["sigma_gk_median"].values.astype(float)   # unscaled terminal field
dd = pd.read_parquet(P / "output_demand/wuhan_demand_pop_per_cell.parquet")
lut = dict(zip(zip(dd.grid_x, dd.grid_y), dd.pop_density.astype(float)))
w = np.array([lut.get((int(x), int(y)), 0.0) for x, y in
              zip(joined.grid_x, joined.grid_y)])
TOT = float(dd.pop_density.sum())
ET = np.load(P / "r2_layer/eterm.npz")
edep = float(ET["edep_main"][0])

def P_inst(v, m_p):
    v = np.maximum(np.asarray(v, dtype=float), 1e-9)
    kap = 1.0 + m_p * C.G / C.W_FRAME
    v0sq = kap * C.V0_HOV**2
    return (C.P_B * (1.0 + 3.0 * v**2 / C.U_TIP**2)
            + (kap**1.5) * C.P_IND * np.sqrt(
                np.sqrt(1.0 + v**4 / (4.0 * v0sq**2)) - v**2 / (2.0 * v0sq))
            + 0.5 * C.D0 * C.RHO_AIR * C.S_SOL * C.A_DISC * v**3)

reg = C.REGIMES_BD["Medium"]

def E_phys(sig, m_p):
    vmx = np.maximum(C.V_MIN, np.minimum(
        C.V_CRUISE, (reg.w_corr - reg.z_eps * sig) / (C.KAPPA_PERP_MAIN * reg.tau_react)))
    on = (sig > reg.sigma_sensor).astype(float)
    return P_inst(vmx, m_p) / vmx + on * C.P_AUX / vmx

def eterm_of(sig):
    ph_ld = C.P_B + C.P_IND * (1 + C.M_P_MAIN * C.G / C.W_FRAME)**1.5
    st_desc = (ph_ld - (C.W_FRAME + C.M_P_MAIN * C.G) * C.V_DESC / 2) * (C.H_C / C.V_DESC)
    ph_e = C.P_B + C.P_IND
    st_climb_e = (ph_e + C.W_FRAME * C.V_CLIMB / 2) * (C.H_C / C.V_CLIMB)
    e_ga = 2 * ph_ld * C.T_GA
    q = 1 - np.exp(-C.R_PAD_MAIN**2 / (2 * np.maximum(sig, 1e-12)**2))
    N = 1 / q
    return st_desc + N * ph_ld * C.T_ALIGN + (N - 1) * e_ga + ph_ld * C.T_REL + st_climb_e

def run_case(sig_full, sig_term):
    removed = sig_full > reg.sigma_sat(C.KAPPA_PERP_MAIN)
    et = eterm_of(sig_term)
    Crt = None
    for mp in [C.M_P_MAIN, 0.0]:
        unit = E_phys(sig_full, mp)
        wgt = 0.5 * (unit[eu] + unit[ev]) * el
        ok = ~(removed[eu] | removed[ev])
        g = csr_matrix((np.concatenate([wgt[ok], wgt[ok]]),
                        (np.concatenate([eu[ok], ev[ok]]),
                         np.concatenate([ev[ok], eu[ok]]))), shape=(NV, NV))
        d = dijkstra(g, directed=False, indices=cand)[:, J_nodes]
        d = np.nan_to_num(d, nan=np.inf, posinf=np.inf)
        Crt = d if Crt is None else Crt + d
    out = {}
    for rho in RHOS:
        mask = Crt + edep + et[None, :] <= (1 - rho) * C.ETA_B
        at = csr_matrix(mask).T.tocsr()
        ns, nc = mask.shape
        m = gp.Model(); m.Params.OutputFlag = 0; m.Params.MIPGap = 1e-6
        m.Params.Threads = 16
        x = m.addVars(ns, vtype=GRB.BINARY)
        y = m.addVars(nc, vtype=GRB.CONTINUOUS, ub=1.0)
        m.addConstr(gp.quicksum(x[i] for i in range(ns)) <= 30)
        for i in np.where(~CAND_MASK)[0]:
            m.addConstr(x[i] == 0)
        for j in range(nc):
            idx = at.indices[at.indptr[j]:at.indptr[j + 1]]
            m.addConstr(y[j] <= (gp.quicksum(x[i] for i in idx) if len(idx) else 0))
        m.setObjective(gp.quicksum(float(w[j]) * y[j] for j in range(nc)), GRB.MAXIMIZE)
        m.optimize()
        out[rho] = 100.0 * m.ObjVal / TOT
    return out

rows = []
if W_ID == 0:
    for tag, val in [("const_median", 4.37), ("const_p75", 6.14)]:
        sf = sigma0.copy(); sf[J_nodes] = val
        st = np.full(len(sig_j0), val)
        cov = run_case(sf, st)
        rows.append({"case": tag, **{f"cov_rho{r}": cov[r] for r in RHOS}})
        print(rows[-1], flush=True)
for k in range(N_LOC):
    draw = W_ID * N_LOC + k
    rng = np.random.default_rng(60_000 + draw)
    perm = rng.permutation(len(J_nodes))
    sf = sigma0.copy(); sf[J_nodes] = sig_j0[perm]
    st = sig_j0[perm]
    cov = run_case(sf, st)
    rows.append({"case": f"perm{draw}", **{f"cov_rho{r}": cov[r] for r in RHOS}})
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(OUT / f"gaps511_w{W_ID}.csv", index=False)
print("done", flush=True)
