"""Water-routing sensitivity : three scenarios
on the 511 headline chain (Medium, kappa=0.3, theta_term=1, pop).

  A open_water_headline : frozen headline matrices (water = 2.42 m corridor)
  B water_excluded      : water nodes impassable (edges touching water cut)
  C water_penalised     : water-touching edge costs x 3 (uniform risk factor)

Reported per scenario at K=30: gap at rho in {0, 0.5}, all-candidate
reachable share, station-set Jaccard vs scenario-A optimum, and the
cross-river service share (fraction of covered demand whose cheapest
serving open station lies in a different land component, components =
4-connected non-water land, which separates the Yangtze/Han banks).

Output: cs511/water_sens_511.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy import ndimage
import gurobipy as gp
from gurobipy import GRB
import sys

P = Path("/lustre/home/2406393544/sharefolder/proj3")
OUT = P / "cs511"
sys.path.insert(0, str(P / "r2_layer"))
import r2_config as C

D = np.load(P / "c_layer/domain.npz")
cand_nodes = D["cand_nodes"].astype(np.int64)
cs = pd.read_parquet(
    P / "v2/data_intermediate/demand_supply/candidate_screens.parquet"
).sort_values("cand_idx").reset_index(drop=True)
MASK = cs["base"].values.astype(bool)
joined = pd.read_parquet(P / "gk_run/joined_wuhan_gk.parquet")
dd = pd.read_parquet(P / "output_demand/wuhan_demand_pop_per_cell.parquet")
lut = dict(zip(zip(dd.grid_x, dd.grid_y), dd.pop_density.astype(float)))
w = np.array([lut.get((int(x), int(y)), 0.0) for x, y in
              zip(joined.grid_x, joined.grid_y)])
TOT = float(dd.pop_density.sum())

F = np.load(P / "r2_layer/field_r2.npz")
sigma_full = F["sigma_full"].astype(float)
NV = len(sigma_full)
node_grid = D["node_grid"]
water_grid = D["water"].astype(bool)
eu_ = D["edges_u"].astype(np.int64); ev_ = D["edges_v"].astype(np.int64)
el_ = D["edges_len"].astype(float)
J_nodes = D["J_nodes"].astype(np.int64)
ET = np.load(P / "r2_layer/eterm.npz")
eterm = ET["eterm_main"].astype(np.float64)
eterm0 = float(ET["eterm0_main"][0]); edep = float(ET["edep_main"][0])

water_node = np.zeros(NV, bool)
rr, cc = np.nonzero(node_grid >= 0)
water_node[node_grid[rr, cc]] = water_grid[rr, cc]
print(f"water nodes: {water_node.sum():,}/{NV:,}", flush=True)
print(f"water candidates: {water_node[cand_nodes].sum()}  "
      f"water J cells: {water_node[J_nodes].sum()}", flush=True)

# land components (4-connectivity; water = barrier)
land = (node_grid >= 0) & (~water_grid)
comp_grid, ncomp = ndimage.label(land, structure=np.array(
    [[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
comp_node = np.full(NV, -1, np.int64)
comp_node[node_grid[rr, cc]] = comp_grid[rr, cc]
sizes = np.bincount(comp_grid.ravel())[1:]
print(f"land components: {ncomp}, largest sizes {sorted(sizes)[-5:]}",
      flush=True)

gx0, gy0 = int(D["gx0"]), int(D["gy0"])
ii, jj = rr, cc
ids = node_grid[ii, jj]
gx_n = np.empty(ids.max() + 1, np.int64); gy_n = np.empty(ids.max() + 1, np.int64)
gx_n[ids] = gx0 + ii; gy_n[ids] = gy0 + jj
if not np.array_equal(gx_n[J_nodes], joined.grid_x.values):
    gx_n[ids] = gx0 + jj; gy_n[ids] = gy0 + ii
jx = (joined.grid_x.values + 0.5) * 100.0
jy = (joined.grid_y.values + 0.5) * 100.0
DE = np.hypot(((gx_n[cand_nodes] + 0.5) * 100.0)[:, None] - jx[None, :],
              ((gy_n[cand_nodes] + 0.5) * 100.0)[:, None] - jy[None, :])
C_eucl_rt = DE * 11.1582 + DE * 9.2365

reg = C.REGIMES_BD["Medium"]
KP = 0.3


def P_inst(v, m_p):
    v = np.maximum(np.asarray(v, float), 1e-9)
    kap = 1.0 + m_p * C.G / C.W_FRAME
    v0sq = kap * C.V0_HOV ** 2
    return (C.P_B * (1 + 3 * v**2 / C.U_TIP**2)
            + (kap**1.5) * C.P_IND * np.sqrt(
                np.sqrt(1 + v**4/(4*v0sq**2)) - v**2/(2*v0sq))
            + 0.5 * C.D0 * C.RHO_AIR * C.S_SOL * C.A_DISC * v**3)


def route_cost(scenario):
    removed = sigma_full > reg.sigma_sat(KP)
    Crt = None
    for mp in [C.M_P_MAIN, 0.0]:
        vm = np.maximum(C.V_MIN, np.minimum(
            C.V_CRUISE, (reg.w_corr - reg.z_eps * sigma_full)
            / (KP * reg.tau_react)))
        on = (sigma_full > reg.sigma_sensor).astype(float)
        unit = P_inst(vm, mp) / vm + on * C.P_AUX / vm
        wgt = 0.5 * (unit[eu_] + unit[ev_]) * el_
        ok = ~(removed[eu_] | removed[ev_])
        touches_water = water_node[eu_] | water_node[ev_]
        if scenario == "water_excluded":
            ok = ok & (~touches_water)
        elif scenario == "water_penalised":
            wgt = np.where(touches_water, 3.0 * wgt, wgt)
        g = csr_matrix((np.concatenate([wgt[ok], wgt[ok]]),
                        (np.concatenate([eu_[ok], ev_[ok]]),
                         np.concatenate([ev_[ok], eu_[ok]]))),
                       shape=(NV, NV))
        d = dijkstra(g, directed=False, indices=cand_nodes)[:, J_nodes]
        Crt = (d if Crt is None else Crt + d)
    return np.nan_to_num(Crt, nan=np.inf, posinf=np.inf)


def milp_set(mask_feas, K=30):
    at = csr_matrix(mask_feas).T.tocsr()
    ns, nc = mask_feas.shape
    m = gp.Model(); m.Params.OutputFlag = 0; m.Params.MIPGap = 1e-6
    m.Params.Threads = 32
    x = m.addVars(ns, vtype=GRB.BINARY)
    y = m.addVars(nc, vtype=GRB.CONTINUOUS, ub=1.0)
    m.addConstr(gp.quicksum(x[i] for i in range(ns)) <= K)
    for j in range(nc):
        idx = at.indices[at.indptr[j]:at.indptr[j + 1]]
        m.addConstr(y[j] <= (gp.quicksum(x[i] for i in idx) if len(idx) else 0))
    m.setObjective(gp.quicksum(float(w[j]) * y[j] for j in range(nc)),
                   GRB.MAXIMIZE)
    m.optimize()
    assert m.Status == GRB.OPTIMAL
    S = np.array([i for i in range(ns) if x[i].X > 0.5], np.int64)
    return 100.0 * m.ObjVal / TOT, S


def cross_river_share(Crt, feas, S):
    tot_cost = Crt + edep + eterm[None, :]
    sub = np.where(feas[S], tot_cost[S], np.inf)
    covered = np.isfinite(sub).any(axis=0)
    nearest = S[np.argmin(sub, axis=0)]
    scomp = comp_node[cand_nodes[nearest]]
    jcomp = comp_node[J_nodes]
    cross = covered & (scomp != jcomp) & (jcomp >= 0) & (scomp >= 0)
    return 100.0 * w[cross].sum() / max(w[covered].sum(), 1e-9)


rows = []
S_ref = None
for scen in ["open_water_headline", "water_excluded", "water_penalised"]:
    Crt = route_cost(scen)
    for rho in [0.0, 0.5]:
        B = (1 - rho) * C.ETA_B
        fe = (C_eucl_rt + edep + eterm0 <= B) & MASK[:, None]
        fm = (Crt + edep + eterm[None, :] <= B) & MASK[:, None]
        ce, _ = milp_set(fe)
        cm, S = milp_set(fm)
        reach = 100 * w[fm.any(axis=0)].sum() / TOT
        if scen == "open_water_headline" and rho == 0.5:
            S_ref = set(S.tolist())
        jac = (len(set(S.tolist()) & S_ref) / len(set(S.tolist()) | S_ref)
               if (S_ref is not None and rho == 0.5) else np.nan)
        rows.append({"scenario": scen, "rho": rho, "cand_set": 511,
                     "cov_eucl": ce, "cov_M": cm, "gap_pp": ce - cm,
                     "reachable_share_pct": reach,
                     "unreachable_share_pct": 100 - reach,
                     "jaccard_vs_headline_K30": jac,
                     "cross_river_service_pct":
                         cross_river_share(Crt, fm, S)})
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(OUT / "water_sens_511.csv", index=False)
print("done", flush=True)
