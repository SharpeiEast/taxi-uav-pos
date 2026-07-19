"""cs2_511.py -- second 511 batch: robustness batch:

All modes evaluate on the base-screen 511 candidate subset by row-masking
frozen cost matrices (and, for uncert, per-draw Dijkstra as in bd_uncert).

MODE (env):
  uncert_impband : bd-chain p25/p75 imputation band, dual-evaluated 605+511.
  uncert_boot0/1/2 : bd-chain segment bootstrap (10 draws per worker,
                   seeds 70000+draw), dual-evaluated 605+511.
  e1             : beta_term x beta_route separation on the R2 base,
                   511 rows (grid ceilings + K=30 MILP gaps).
  m1             : maximal-empirical (m_min=1) field sensitivities on 511:
                   (a) kappa=0.3 package (costmat_k03), (b) kappa=1
                   conservative envelope (bd costmat) removal + keep-at-vmin.
  lpgap          : WEIGHT in {pop,ev,uniform}; LP integrality gap on the
                   R2 base, 511 rows: 4 rho x 5 ablations x 11 K.

Output: proj3/cs511/<mode-specific>.csv
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix
import gurobipy as gp
from gurobipy import GRB
import sys

P = Path("/lustre/home/2406393544/sharefolder/proj3")
OUT = P / "cs511"
MODE = os.environ["MODE"]
THREADS = int(os.environ.get("GRB_THREADS", "16"))
EQ = os.environ.get("EQ", "0") == "1"   # budget as equality (sum x == K)

# ---------- candidate mask (base screen, verified aligned) ----------
D = np.load(P / "c_layer/domain.npz")
cand_nodes = D["cand_nodes"].astype(np.int64)
cs = pd.read_parquet(
    P / "v2/data_intermediate/demand_supply/candidate_screens.parquet"
).sort_values("cand_idx").reset_index(drop=True)
assert (cs["node"].values.astype(np.int64) == cand_nodes).all()
MASK = cs["base"].values.astype(bool)
print(f"base-screen candidates: {MASK.sum()}/605", flush=True)

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
J_nodes = D["J_nodes"].astype(np.int64)
if not np.array_equal(gx_n[J_nodes], joined.grid_x.values):
    gx_n[ids] = gx0 + jj; gy_n[ids] = gy0 + ii
jx = (joined.grid_x.values + 0.5) * 100.0
jy = (joined.grid_y.values + 0.5) * 100.0
DE = np.hypot(((gx_n[cand_nodes] + 0.5) * 100.0)[:, None] - jx[None, :],
              ((gy_n[cand_nodes] + 0.5) * 100.0)[:, None] - jy[None, :])
E_BASE_E, E_BASE_L = 9.2365, 11.1582
C_eucl_rt = DE * E_BASE_L + DE * E_BASE_E


def budget_constr(m, x, ns, K):
    if EQ:
        m.addConstr(gp.quicksum(x[i] for i in range(ns)) == K)
    else:
        m.addConstr(gp.quicksum(x[i] for i in range(ns)) <= K)


def milp_cov(mask_feas, wv, tot, K=30):
    at = csr_matrix(mask_feas).T.tocsr()
    ns, nc = mask_feas.shape
    m = gp.Model(); m.Params.OutputFlag = 0; m.Params.MIPGap = 1e-6
    m.Params.Threads = THREADS
    x = m.addVars(ns, vtype=GRB.BINARY)
    y = m.addVars(nc, vtype=GRB.CONTINUOUS, ub=1.0)
    budget_constr(m, x, ns, K)
    for j in range(nc):
        idx = at.indices[at.indptr[j]:at.indptr[j + 1]]
        m.addConstr(y[j] <= (gp.quicksum(x[i] for i in idx) if len(idx) else 0))
    m.setObjective(gp.quicksum(float(wv[j]) * y[j] for j in range(nc)),
                   GRB.MAXIMIZE)
    m.optimize()
    assert m.Status == GRB.OPTIMAL
    return 100.0 * m.ObjVal / tot


def threeclass_row(best_route, eterm_vec, eterm0_, edep_, B):
    blocked = ~np.isfinite(best_route)
    over = np.isfinite(best_route) & (best_route + edep_ + eterm0_ > B)
    term = (np.isfinite(best_route) & (best_route + edep_ + eterm0_ <= B)
            & (best_route + edep_ + eterm_vec > B))
    covered = ~(blocked | over | term)
    return {"blocked_pct": 100 * w[blocked].sum() / TOT,
            "overrange_pct": 100 * w[over].sum() / TOT,
            "terminal_pct": 100 * w[term].sum() / TOT,
            "covered_pct": 100 * w[covered].sum() / TOT}


# ==================================================================
if MODE.startswith("uncert"):
    from scipy.sparse.csgraph import dijkstra
    sys.path.insert(0, str(P / "bd_layer"))
    import bd_config as C
    F = np.load(P / "c_layer/sigma_fields.npz")
    NV = len(F["sigma_krig"])
    eu = D["edges_u"].astype(np.int64); ev = D["edges_v"].astype(np.int64)
    el = D["edges_len"].astype(float)
    sig_j0 = joined["sigma_gk_median"].values.astype(float)
    ET = np.load(P / "bd_layer/eterm.npz")
    edep = float(ET["edep_main"][0]); eterm0 = float(ET["eterm0_main"][0])
    C_eucl = C_eucl_rt + edep + eterm0
    reg = C.REGIMES_BD["Medium"]

    def P_inst(v, m_p):
        v = np.maximum(np.asarray(v, dtype=float), 1e-9)
        kap = 1.0 + m_p * C.G / C.W_FRAME
        v0sq = kap * C.V0_HOV**2
        return (C.P_B * (1.0 + 3.0 * v**2 / C.U_TIP**2)
                + (kap**1.5) * C.P_IND * np.sqrt(
                    np.sqrt(1.0 + v**4 / (4.0 * v0sq**2)) - v**2 / (2.0 * v0sq))
                + 0.5 * C.D0 * C.RHO_AIR * C.S_SOL * C.A_DISC * v**3)

    def E_phys(sig, m_p):
        vmx = np.maximum(C.V_MIN, np.minimum(
            C.V_CRUISE, (reg.w_corr - reg.z_eps * sig) / reg.tau_react))
        on = (sig > reg.sigma_sensor).astype(float)
        return P_inst(vmx, m_p) / vmx + on * C.P_AUX / vmx

    def eterm_of(sig):
        ph_ld = C.P_B + C.P_IND * (1 + C.M_P_MAIN * C.G / C.W_FRAME)**1.5
        st_desc = (ph_ld - (C.W_FRAME + C.M_P_MAIN * C.G) * C.V_DESC / 2) \
            * (C.H_C / C.V_DESC)
        ph_e = C.P_B + C.P_IND
        st_climb_e = (ph_e + C.W_FRAME * C.V_CLIMB / 2) * (C.H_C / C.V_CLIMB)
        e_ga = 2 * ph_ld * C.T_GA
        q = 1 - np.exp(-C.R_PAD_MAIN**2 / (2 * np.maximum(sig, 1e-12)**2))
        N = 1 / q
        return (st_desc + N * ph_ld * C.T_ALIGN + (N - 1) * e_ga
                + ph_ld * C.T_REL + st_climb_e)

    def route_cost(sig_full):
        removed = sig_full > reg.sigma_sat()
        Crt = None
        for mp in [C.M_P_MAIN, 0.0]:
            unit = E_phys(sig_full, mp)
            wgt = 0.5 * (unit[eu] + unit[ev]) * el
            ok = ~(removed[eu] | removed[ev])
            g = csr_matrix((np.concatenate([wgt[ok], wgt[ok]]),
                            (np.concatenate([eu[ok], ev[ok]]),
                             np.concatenate([ev[ok], eu[ok]]))),
                           shape=(NV, NV))
            d = dijkstra(g, directed=False, indices=cand_nodes)[:, J_nodes]
            d = np.nan_to_num(d, nan=np.inf, posinf=np.inf)
            Crt = d if Crt is None else Crt + d
        return Crt

    def both(feasM, feasE, rho, extra):
        out = []
        for nset, msk in (("605", np.ones(605, bool)), ("511", MASK)):
            cm = milp_cov(feasM & msk[:, None], w, TOT)
            ce = milp_cov(feasE & msk[:, None], w, TOT)
            row = {**extra, "rho": rho, "cand_set": nset,
                   "cov_eucl": ce, "cov_M": cm, "gap": ce - cm}
            out.append(row); print(row, flush=True)
        return out

    rows = []
    if MODE == "uncert_impband":
        for var in ["p25", "p75"]:
            Crt = route_cost(F[f"sigma_{var}"].astype(float))
            et = eterm_of(sig_j0)
            for rho in C.RHOS:
                B = (1 - rho) * C.ETA_B
                rows += both(Crt + edep + et[None, :] <= B, C_eucl <= B,
                             rho, {"variant": var})
        pd.DataFrame(rows).to_csv(OUT / "uncert511_impband.csv", index=False)
    else:
        W_ID = int(MODE.replace("uncert_boot", ""))
        seg = pd.read_parquet(P / "gk_run/seg_cell_table.parquet")
        groups = {k: g["sigma_gk_m"].values for k, g in
                  seg.groupby(["gx", "gy"], sort=False)}
        grid = pd.read_parquet(
            P / "opera/rev02_A1/gk_out/poc_wuhan/intermediate/grid_100m.parquet",
            columns=["grid_x", "grid_y"])
        obs_keys = list(zip(grid.grid_x.values, grid.grid_y.values))
        obs_nodes = D["obs_nodes"].astype(np.int64)
        jkey = {(int(x), int(y)): t for t, (x, y) in
                enumerate(zip(joined.grid_x.values, joined.grid_y.values))}
        sigma0 = F["sigma_krig"].astype(float)
        for k in range(10):
            draw = W_ID * 10 + k
            rng = np.random.default_rng(70_000 + draw)
            boot = np.array([float(np.median(v[rng.integers(0, len(v), len(v))]))
                             for v in (groups[key] for key in obs_keys)])
            sf = sigma0.copy(); sf[obs_nodes] = boot
            sj = sig_j0.copy()
            for t, key in enumerate(obs_keys):
                pos = jkey.get((int(key[0]), int(key[1])))
                if pos is not None:
                    sj[pos] = boot[t]
            Crt = route_cost(sf)
            et = eterm_of(sj)
            for rho in C.RHOS:
                B = (1 - rho) * C.ETA_B
                rows += both(Crt + edep + et[None, :] <= B, C_eucl <= B,
                             rho, {"draw": draw})
        pd.DataFrame(rows).to_csv(OUT / f"uncert511_boot_w{W_ID}.csv",
                                  index=False)

elif MODE == "e1":
    sys.path.insert(0, str(P / "r2_layer"))
    import r2_config as C
    CM = P / "r2_layer/costmat"
    sig = np.load(P / "r2_layer/field_r2.npz")["sig_j"].astype(float)
    ET = np.load(P / "r2_layer/eterm.npz")
    edep = float(ET["edep_main"][0]); eterm0 = float(ET["eterm0_main"][0])

    def hover_power(m_p):
        kap = 1.0 + m_p * C.G / C.W_FRAME
        return C.P_B + C.P_IND * kap ** 1.5

    ph_ld, ph_e = hover_power(C.M_P_MAIN), hover_power(0.0)
    w_ld, w_e = C.W_FRAME + C.M_P_MAIN * C.G, C.W_FRAME
    t_v = C.H_C / C.V_CLIMB
    ST = dict(dj=(ph_ld - w_ld * C.V_DESC / 2) * t_v,
              cj=(ph_e + w_e * C.V_CLIMB / 2) * t_v)
    E_GA = 2.0 * ph_ld * C.T_GA

    def eterm_iid(s):
        s = np.maximum(np.asarray(s, float), 1e-12)
        q = 1.0 - np.exp(-C.R_PAD_MAIN ** 2 / (2.0 * s ** 2))
        q = np.where(np.asarray(s) <= 1e-12, 1.0, q)
        N = 1.0 / q
        return (ST["dj"] + N * ph_ld * C.T_ALIGN + (N - 1.0) * E_GA
                + ph_ld * C.T_REL + ST["cj"])

    assert np.allclose(eterm_iid(sig), ET["eterm_main"], rtol=1e-9)
    print("frozen-eterm cross-check PASS", flush=True)

    def load(n):
        return np.load(CM / f"{n}.npz")["C"].astype(np.float64)

    route_M = {}
    for beta in C.BETAS:
        route_M[beta] = np.nan_to_num(
            load(f"C_ld_sigmaMedium_b{beta}") + load(f"C_e_sigmaMedium_b{beta}"),
            nan=np.inf, posinf=np.inf)
    best_M = {b: route_M[b][MASK].min(axis=0) for b in route_M}

    BT = [0.25, 0.5, 0.75, 1.0]
    eterm_bt = {bt: eterm_iid(bt * sig) for bt in BT}
    rows = []
    for br in C.BETAS:
        for bt in BT:
            for rho in C.RHOS:
                B = (1 - rho) * C.ETA_B
                tc = threeclass_row(best_M[br], eterm_bt[bt], eterm0, edep, B)
                rows.append({"beta_route": br, "beta_term": bt, "rho": rho,
                             "cand_set": 511, **tc})
    pd.DataFrame(rows).to_csv(OUT / "e1_grid_511.csv", index=False)

    combos = [(1.0, 1.0), (1.0, 0.5), (1.0, 0.25),
              (0.5, 1.0), (0.5, 0.5), (0.25, 1.0)]
    rows = []
    for rho in [0.0, 0.2, 0.5]:
        B = (1 - rho) * C.ETA_B
        cov_eu = milp_cov((C_eucl_rt + edep + eterm0 <= B) & MASK[:, None],
                          w, TOT)
        for br, bt in combos:
            covM = milp_cov((route_M[br] + edep + eterm_bt[bt][None, :] <= B)
                            & MASK[:, None], w, TOT)
            rows.append({"rho": rho, "beta_route": br, "beta_term": bt,
                         "cand_set": 511, "cov_eucl": cov_eu, "cov_M": covM,
                         "gap_pp": cov_eu - covM})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "e1_milp_511.csv", index=False)

elif MODE == "m1":
    sys.path.insert(0, str(P / "bd_layer"))
    import bd_config as C
    ET = np.load(P / "bd_layer/eterm.npz")
    eterm = ET["eterm_main"].astype(np.float64)
    eterm0 = float(ET["eterm0_main"][0]); edep = float(ET["edep_main"][0])

    def loadf(f):
        return np.load(f)["C"].astype(np.float64)

    # (a) kappa=0.3 package on the maximal-empirical chain
    rows = []
    for rname in ["High", "Medium", "Low"]:
        Crt = np.nan_to_num(
            loadf(P / f"trc_exp/costmat_k03/C_ld_sigma{rname}_k03.npz")
            + loadf(P / f"trc_exp/costmat_k03/C_e_sigma{rname}_k03.npz"),
            nan=np.inf, posinf=np.inf)
        best = Crt[MASK].min(axis=0)
        for rho in C.RHOS:
            B = (1 - rho) * C.ETA_B
            row = {"part": "k03", "regime": rname, "rho": rho, "kperp": 0.3,
                   "cand_set": 511,
                   **threeclass_row(best, eterm, eterm0, edep, B)}
            if rho in (0.0, 0.2, 0.5):
                row["cov_eucl"] = milp_cov(
                    (C_eucl_rt + edep + eterm0 <= B) & MASK[:, None], w, TOT)
                row["cov_sigma"] = milp_cov(
                    (Crt + edep + eterm[None, :] <= B) & MASK[:, None], w, TOT)
                row["gap_pp"] = row["cov_eucl"] - row["cov_sigma"]
            rows.append(row); print(row, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "m1k03_511.csv", index=False)

    # (b) kappa=1 conservative envelope: removal headline + keep-at-vmin
    C_rm = np.nan_to_num(
        loadf(P / "bd_layer/costmat/C_ld_sigmaMedium_b1.0.npz")
        + loadf(P / "bd_layer/costmat/C_e_sigmaMedium_b1.0.npz"),
        nan=np.inf, posinf=np.inf)
    C_keep = (loadf(P / "trc_exp/costmat/C_ld_keepMedium.npz")
              + loadf(P / "trc_exp/costmat/C_e_keepMedium.npz"))
    rows = []
    for tag, Crt in [("removal_headline", C_rm), ("keep_at_vmin", C_keep)]:
        best = Crt[MASK].min(axis=0)
        for rho in C.RHOS:
            B = (1 - rho) * C.ETA_B
            row = {"part": "k1env", "variant": tag, "rho": rho,
                   "cand_set": 511,
                   **threeclass_row(best, eterm, eterm0, edep, B)}
            if rho in (0.0, 0.2, 0.5):
                row["cov_eucl"] = milp_cov(
                    (C_eucl_rt + edep + eterm0 <= B) & MASK[:, None], w, TOT)
                row["cov_M"] = milp_cov(
                    (Crt + edep + eterm[None, :] <= B) & MASK[:, None], w, TOT)
                row["gap_pp"] = row["cov_eucl"] - row["cov_M"]
            rows.append(row); print(row, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "m1env_511.csv", index=False)

elif MODE == "exactrange":
    # Exact min/max of Realized coverage over ALL Euclidean optima:
    #   min/max f_M(S)  s.t.  f_Eucl(S) = V*_Eucl,  |S| <= K
    # on the 511 headline instance (two-stage MILP, robustness batch: 2).
    sys.path.insert(0, str(P / "r2_layer"))
    import r2_config as C
    CM = P / "r2_layer/costmat"
    ET = np.load(P / "r2_layer/eterm.npz")
    eterm = ET["eterm_main"].astype(np.float64)
    eterm0 = float(ET["eterm0_main"][0]); edep = float(ET["edep_main"][0])

    def load(n):
        return np.load(CM / f"{n}.npz")["C"].astype(np.float64)

    C_M = np.nan_to_num(load("C_ld_sigmaMedium_b1.0") + load("C_e_sigmaMedium_b1.0"),
                        nan=np.inf, posinf=np.inf)
    K = 30
    rows = []
    for rho in C.RHOS:
        B = (1 - rho) * C.ETA_B
        fe = (C_eucl_rt + edep + eterm0 <= B) & MASK[:, None]
        fm = (C_M + edep + eterm[None, :] <= B) & MASK[:, None]
        ate = csr_matrix(fe).T.tocsr()
        atm = csr_matrix(fm).T.tocsr()
        ns, nc = fe.shape
        vstar = milp_cov(fe, w, TOT, K) * TOT / 100.0

        def two_stage(sense):
            m = gp.Model(); m.Params.OutputFlag = 0
            m.Params.Threads = 64; m.Params.MIPGap = 1e-6
            m.Params.TimeLimit = 10800
            x = m.addVars(ns, vtype=GRB.BINARY)
            ye = m.addVars(nc, vtype=GRB.CONTINUOUS, ub=1.0)
            ym = m.addVars(nc, vtype=GRB.CONTINUOUS, ub=1.0)
            budget_constr(m, x, ns, K)
            for j in range(nc):
                idx = ate.indices[ate.indptr[j]:ate.indptr[j + 1]]
                m.addConstr(ye[j] <= (gp.quicksum(x[i] for i in idx)
                                      if len(idx) else 0))
            m.addConstr(gp.quicksum(float(w[j]) * ye[j] for j in range(nc))
                        >= vstar * (1 - 1e-9))
            if sense == "max":
                for j in range(nc):
                    idx = atm.indices[atm.indptr[j]:atm.indptr[j + 1]]
                    m.addConstr(ym[j] <= (gp.quicksum(x[i] for i in idx)
                                          if len(idx) else 0))
                m.setObjective(gp.quicksum(float(w[j]) * ym[j]
                                           for j in range(nc)), GRB.MAXIMIZE)
            else:
                for j in range(nc):
                    idx = atm.indices[atm.indptr[j]:atm.indptr[j + 1]]
                    for i in idx:
                        m.addConstr(ym[j] >= x[i])
                m.setObjective(gp.quicksum(float(w[j]) * ym[j]
                                           for j in range(nc)), GRB.MINIMIZE)
            m.optimize()
            gap = m.MIPGap if m.SolCount else float("nan")
            return 100.0 * m.ObjVal / TOT, gap, m.Status

        lo, lo_gap, lo_st = two_stage("min")
        hi, hi_gap, hi_st = two_stage("max")
        rows.append({"rho": rho, "K": K, "cand_set": 511,
                     "vstar_eucl_pct": 100.0 * vstar / TOT,
                     "realized_min": lo, "min_mipgap": lo_gap,
                     "min_status": lo_st,
                     "realized_max": hi, "max_mipgap": hi_gap,
                     "max_status": hi_st})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "exactrange_511.csv", index=False)

elif MODE == "e1env":
    # theta_term separation on the conservative-envelope chain, 511 rows
    # (replaces the 605-based 50.9 -> 40.2 -> 38.6).
    sys.path.insert(0, str(P / "bd_layer"))
    import bd_config as C
    ET = np.load(P / "bd_layer/eterm.npz")
    eterm0 = float(ET["eterm0_main"][0]); edep = float(ET["edep_main"][0])
    sig_j0 = joined["sigma_gk_median"].values.astype(float)

    def eterm_of(sig):
        ph_ld = C.P_B + C.P_IND * (1 + C.M_P_MAIN * C.G / C.W_FRAME)**1.5
        st_desc = (ph_ld - (C.W_FRAME + C.M_P_MAIN * C.G) * C.V_DESC / 2) \
            * (C.H_C / C.V_DESC)
        ph_e = C.P_B + C.P_IND
        st_climb_e = (ph_e + C.W_FRAME * C.V_CLIMB / 2) * (C.H_C / C.V_CLIMB)
        e_ga = 2 * ph_ld * C.T_GA
        q = 1 - np.exp(-C.R_PAD_MAIN**2 / (2 * np.maximum(sig, 1e-12)**2))
        N = 1 / q
        return (st_desc + N * ph_ld * C.T_ALIGN + (N - 1) * e_ga
                + ph_ld * C.T_REL + st_climb_e)

    assert np.allclose(eterm_of(sig_j0), ET["eterm_main"], rtol=1e-9)
    print("bd eterm cross-check PASS", flush=True)

    def loadf(f):
        return np.load(f)["C"].astype(np.float64)

    C_M = np.nan_to_num(
        loadf(P / "bd_layer/costmat/C_ld_sigmaMedium_b1.0.npz")
        + loadf(P / "bd_layer/costmat/C_e_sigmaMedium_b1.0.npz"),
        nan=np.inf, posinf=np.inf)
    rows = []
    for rho in [0.0, 0.2, 0.5]:
        B = (1 - rho) * C.ETA_B
        cov_eu = milp_cov((C_eucl_rt + edep + eterm0 <= B) & MASK[:, None],
                          w, TOT)
        for bt in [1.0, 0.5, 0.25]:
            et = eterm_of(bt * sig_j0)
            covM = milp_cov((C_M + edep + et[None, :] <= B) & MASK[:, None],
                            w, TOT)
            rows.append({"rho": rho, "theta_term": bt, "cand_set": 511,
                         "config": "envelope_m1_k1",
                         "cov_eucl": cov_eu, "cov_M": covM,
                         "gap_pp": cov_eu - covM})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "e1env_511.csv", index=False)

elif MODE == "btw":
    # between-segment combined field on the HEADLINE configuration
    # (noise-controlled field, kappa=0.3, 511 rows); replaces the
    # 605/envelope-based 9--51 -> 19--68 with headline values.
    from scipy.sparse.csgraph import dijkstra
    sys.path.insert(0, str(P / "r2_layer"))
    import r2_config as C
    CM = P / "r2_layer/costmat"
    F = np.load(P / "r2_layer/field_r2.npz")
    sigma_full = F["sigma_full"].astype(float)
    sig_j = F["sig_j"].astype(float)
    NV = len(sigma_full)
    eu_ = D["edges_u"].astype(np.int64); ev_ = D["edges_v"].astype(np.int64)
    el_ = D["edges_len"].astype(float)
    ET = np.load(P / "r2_layer/eterm.npz")
    eterm0 = float(ET["eterm0_main"][0]); edep = float(ET["edep_main"][0])

    tot = pd.read_parquet(P / "gk_run/joined_wuhan_tot.parquet")
    sb_lut = dict(zip(zip(tot.grid_x, tot.grid_y),
                      tot["sigma_b"].astype(float)))
    sb_j = np.array([sb_lut.get((int(x), int(y)), 0.0) for x, y in
                     zip(joined.grid_x, joined.grid_y)])
    sig_tot_j = np.sqrt(sig_j**2 + sb_j**2)
    obs_nodes = D["obs_nodes"].astype(np.int64)
    sb_node = np.zeros(NV)
    key_node = {(int(gx_n[n]), int(gy_n[n])): n for n in obs_nodes}
    for (gx, gy), sb in sb_lut.items():
        n = key_node.get((int(gx), int(gy)))
        if n is not None:
            sb_node[n] = sb
    sigma_tot_full = np.sqrt(sigma_full**2 + sb_node**2)
    print(f"sigma_tot: J median {np.median(sig_tot_j):.2f} vs within "
          f"{np.median(sig_j):.2f}; node nonzero sb {(sb_node>0).sum():,}",
          flush=True)

    reg = C.REGIMES_BD["Medium"]
    KP = C.KPERP_MAIN if hasattr(C, "KPERP_MAIN") else 0.3

    def P_inst(v, m_p):
        v = np.maximum(np.asarray(v, float), 1e-9)
        kap = 1.0 + m_p * C.G / C.W_FRAME
        v0sq = kap * C.V0_HOV ** 2
        return (C.P_B * (1 + 3 * v**2 / C.U_TIP**2)
                + (kap**1.5) * C.P_IND * np.sqrt(
                    np.sqrt(1 + v**4/(4*v0sq**2)) - v**2/(2*v0sq))
                + 0.5 * C.D0 * C.RHO_AIR * C.S_SOL * C.A_DISC * v**3)

    def route_cost(sf):
        removed = sf > reg.sigma_sat(KP)
        Crt = None
        for mp in [C.M_P_MAIN, 0.0]:
            vm = np.maximum(C.V_MIN, np.minimum(
                C.V_CRUISE, (reg.w_corr - reg.z_eps * sf) / (KP * reg.tau_react)))
            on = (sf > reg.sigma_sensor).astype(float)
            unit = P_inst(vm, mp) / vm + on * C.P_AUX / vm
            wgt = 0.5 * (unit[eu_] + unit[ev_]) * el_
            ok = ~(removed[eu_] | removed[ev_])
            g = csr_matrix((np.concatenate([wgt[ok], wgt[ok]]),
                            (np.concatenate([eu_[ok], ev_[ok]]),
                             np.concatenate([ev_[ok], eu_[ok]]))),
                           shape=(NV, NV))
            d = dijkstra(g, directed=False, indices=cand_nodes)[:, J_nodes]
            Crt = (d if Crt is None else Crt + d)
        return np.nan_to_num(Crt, nan=np.inf, posinf=np.inf)

    def eterm_iid_r2(s):
        ph_ld = C.P_B + C.P_IND * (1 + C.M_P_MAIN * C.G / C.W_FRAME)**1.5
        st_desc = (ph_ld - (C.W_FRAME + C.M_P_MAIN * C.G) * C.V_DESC / 2) \
            * (C.H_C / C.V_DESC)
        ph_e = C.P_B + C.P_IND
        st_climb_e = (ph_e + C.W_FRAME * C.V_CLIMB / 2) * (C.H_C / C.V_CLIMB)
        e_ga = 2 * ph_ld * C.T_GA
        q = 1 - np.exp(-C.R_PAD_MAIN**2 / (2 * np.maximum(s, 1e-12)**2))
        N = 1 / q
        return (st_desc + N * ph_ld * C.T_ALIGN + (N - 1) * e_ga
                + ph_ld * C.T_REL + st_climb_e)

    assert np.allclose(eterm_iid_r2(sig_j), ET["eterm_main"], rtol=1e-9)
    print("r2 eterm cross-check PASS", flush=True)

    Crt = route_cost(sigma_tot_full)
    et = eterm_iid_r2(sig_tot_j)
    rows = []
    for rho in C.RHOS:
        B = (1 - rho) * C.ETA_B
        fe = (C_eucl_rt + edep + eterm0 <= B) & MASK[:, None]
        fm = (Crt + edep + et[None, :] <= B) & MASK[:, None]
        row = {"rho": rho, "cand_set": 511, "config": "headline_k03_tot",
               "cov_eucl": milp_cov(fe, w, TOT),
               "cov_M_tot": milp_cov(fm, w, TOT)}
        row["gap_pp"] = row["cov_eucl"] - row["cov_M_tot"]
        rows.append(row); print(row, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "btw_headline_511.csv", index=False)

elif MODE == "blindpolicy":
    # Reproducible blind planning policy :
    # stage 1: max Euclidean coverage; stage 2: among Euclidean optima,
    # maximise Euclidean REDUNDANT coverage (population-weighted demand
    # covered by >= 2 open stations) -- a sigma-free, operationally
    # meaningful tie-breaking rule. Saves the chosen sets for
    # independent (MATLAB) re-evaluation.
    import scipy.io as sio
    sys.path.insert(0, str(P / "r2_layer"))
    import r2_config as C
    CM = P / "r2_layer/costmat"
    ET = np.load(P / "r2_layer/eterm.npz")
    eterm = ET["eterm_main"].astype(np.float64)
    eterm0 = float(ET["eterm0_main"][0]); edep = float(ET["edep_main"][0])

    def load(n):
        return np.load(CM / f"{n}.npz")["C"].astype(np.float64)

    C_M = np.nan_to_num(load("C_ld_sigmaMedium_b1.0") + load("C_e_sigmaMedium_b1.0"),
                        nan=np.inf, posinf=np.inf)
    K = 30
    rows = []
    matlab = {"w": w, "TOT": TOT}
    for rho in C.RHOS:
        B = (1 - rho) * C.ETA_B
        fe = (C_eucl_rt + edep + eterm0 <= B) & MASK[:, None]
        fm = (C_M + edep + eterm[None, :] <= B) & MASK[:, None]
        ate = csr_matrix(fe).T.tocsr()
        ns, nc = fe.shape
        vstar = milp_cov(fe, w, TOT, K) * TOT / 100.0

        m = gp.Model(); m.Params.OutputFlag = 0
        m.Params.Threads = 64; m.Params.MIPGap = 1e-6
        m.Params.TimeLimit = 7200
        x = m.addVars(ns, vtype=GRB.BINARY)
        y1 = m.addVars(nc, vtype=GRB.CONTINUOUS, ub=1.0)
        y2 = m.addVars(nc, vtype=GRB.CONTINUOUS, ub=1.0)
        budget_constr(m, x, ns, K)
        for j in range(nc):
            idx = ate.indices[ate.indptr[j]:ate.indptr[j + 1]]
            cov = (gp.quicksum(x[i] for i in idx) if len(idx) else 0)
            m.addConstr(y1[j] <= cov)
            m.addConstr(y2[j] + y1[j] <= cov)
        m.addConstr(gp.quicksum(float(w[j]) * y1[j] for j in range(nc))
                    >= vstar * (1 - 1e-9))
        m.setObjective(gp.quicksum(float(w[j]) * y2[j] for j in range(nc)),
                       GRB.MAXIMIZE)
        m.optimize()
        S = np.array([i for i in range(ns) if x[i].X > 0.5], dtype=np.int64)
        red_pct = 100.0 * m.ObjVal / TOT
        # direct evaluation (no solver): realized coverage of S
        cov_e = fe[S].any(axis=0); cov_m = fm[S].any(axis=0)
        reported = 100.0 * w[cov_e].sum() / TOT
        realized = 100.0 * w[cov_m].sum() / TOT
        best = milp_cov(fm, w, TOT, K)
        rows.append({"rho": rho, "K": K, "cand_set": 511,
                     "policy": "max_redundant_coverage",
                     "vstar_eucl_pct": 100.0 * vstar / TOT,
                     "reported_pct": reported,
                     "redundant_pct": red_pct,
                     "stage2_mipgap": m.MIPGap, "stage2_status": m.Status,
                     "realized_pct": realized, "best_pct": best,
                     "reporting_gap_pp": reported - realized,
                     "regret_pp": best - realized})
        print(rows[-1], flush=True)
        matlab[f"S_rho{int(rho*10):02d}"] = S + 1  # 1-based for MATLAB
        matlab[f"fe_rho{int(rho*10):02d}"] = csr_matrix(fe.astype(np.float64))
        matlab[f"fm_rho{int(rho*10):02d}"] = csr_matrix(fm.astype(np.float64))
        matlab[f"vals_rho{int(rho*10):02d}"] = np.array(
            [reported, realized, best])
    pd.DataFrame(rows).to_csv(OUT / "blindpolicy_511.csv", index=False)
    sio.savemat(OUT / "blindpolicy_check.mat", matlab, do_compression=True)

elif MODE == "bpolsweep":
    # blind-policy K sweep at rho = 0.5 (for the regret figure): same
    # two-stage max-coverage -> max-redundancy policy at each K.
    sys.path.insert(0, str(P / "r2_layer"))
    import r2_config as C
    CM = P / "r2_layer/costmat"
    ET = np.load(P / "r2_layer/eterm.npz")
    eterm = ET["eterm_main"].astype(np.float64)
    eterm0 = float(ET["eterm0_main"][0]); edep = float(ET["edep_main"][0])

    def load(n):
        return np.load(CM / f"{n}.npz")["C"].astype(np.float64)

    C_M = np.nan_to_num(load("C_ld_sigmaMedium_b1.0") + load("C_e_sigmaMedium_b1.0"),
                        nan=np.inf, posinf=np.inf)
    rho = 0.5
    B = (1 - rho) * C.ETA_B
    fe = (C_eucl_rt + edep + eterm0 <= B) & MASK[:, None]
    fm = (C_M + edep + eterm[None, :] <= B) & MASK[:, None]
    ate = csr_matrix(fe).T.tocsr()
    ns, nc = fe.shape
    rows = []
    for K0 in C.K_GRID:
        K = min(K0, int(MASK.sum()))
        vstar = milp_cov(fe, w, TOT, K) * TOT / 100.0
        m = gp.Model(); m.Params.OutputFlag = 0
        m.Params.Threads = 64; m.Params.MIPGap = 1e-6
        m.Params.TimeLimit = 3600
        x = m.addVars(ns, vtype=GRB.BINARY)
        y1 = m.addVars(nc, vtype=GRB.CONTINUOUS, ub=1.0)
        y2 = m.addVars(nc, vtype=GRB.CONTINUOUS, ub=1.0)
        budget_constr(m, x, ns, K)
        for j in range(nc):
            idx = ate.indices[ate.indptr[j]:ate.indptr[j + 1]]
            cov = (gp.quicksum(x[i] for i in idx) if len(idx) else 0)
            m.addConstr(y1[j] <= cov)
            m.addConstr(y2[j] + y1[j] <= cov)
        m.addConstr(gp.quicksum(float(w[j]) * y1[j] for j in range(nc))
                    >= vstar * (1 - 1e-9))
        m.setObjective(gp.quicksum(float(w[j]) * y2[j] for j in range(nc)),
                       GRB.MAXIMIZE)
        m.optimize()
        S = np.array([i for i in range(ns) if x[i].X > 0.5], dtype=np.int64)
        cov_e = fe[S].any(axis=0); cov_m = fm[S].any(axis=0)
        reported = 100.0 * w[cov_e].sum() / TOT
        realized = 100.0 * w[cov_m].sum() / TOT
        best = milp_cov(fm, w, TOT, K)
        rows.append({"rho": rho, "K": K, "cand_set": 511,
                     "policy": "max_redundant_coverage",
                     "reported_pct": reported, "realized_pct": realized,
                     "best_pct": best,
                     "ReportingGap_pp": reported - realized,
                     "Regret_pp": best - realized,
                     "stage2_status": m.Status})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "bpolsweep_511.csv", index=False)

elif MODE == "rawroute":
    # theta_route in {1.9, 2.5, 4.0}: raw / weakly aided route-side
    # stress test . Medium regime, kappa=0.3,
    # theta_term = 1, 511 candidates, population weighting.
    from scipy.sparse.csgraph import dijkstra
    sys.path.insert(0, str(P / "r2_layer"))
    import r2_config as C
    F = np.load(P / "r2_layer/field_r2.npz")
    sigma_full = F["sigma_full"].astype(float)
    sig_j = F["sig_j"].astype(float)
    NV = len(sigma_full)
    eu_ = D["edges_u"].astype(np.int64); ev_ = D["edges_v"].astype(np.int64)
    el_ = D["edges_len"].astype(float)
    ET = np.load(P / "r2_layer/eterm.npz")
    eterm = ET["eterm_main"].astype(np.float64)
    eterm0 = float(ET["eterm0_main"][0]); edep = float(ET["edep_main"][0])
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

    def route_cost(sf):
        removed = sf > reg.sigma_sat(KP)
        Crt = None
        for mp in [C.M_P_MAIN, 0.0]:
            vm = np.maximum(C.V_MIN, np.minimum(
                C.V_CRUISE, (reg.w_corr - reg.z_eps * sf) / (KP * reg.tau_react)))
            on = (sf > reg.sigma_sensor).astype(float)
            unit = P_inst(vm, mp) / vm + on * C.P_AUX / vm
            wgt = 0.5 * (unit[eu_] + unit[ev_]) * el_
            ok = ~(removed[eu_] | removed[ev_])
            g = csr_matrix((np.concatenate([wgt[ok], wgt[ok]]),
                            (np.concatenate([eu_[ok], ev_[ok]]),
                             np.concatenate([ev_[ok], eu_[ok]]))),
                           shape=(NV, NV))
            d = dijkstra(g, directed=False, indices=cand_nodes)[:, J_nodes]
            Crt = (d if Crt is None else Crt + d)
        return np.nan_to_num(Crt, nan=np.inf, posinf=np.inf)

    rows = []
    for th in [1.9, 2.5, 4.0]:
        Crt = route_cost(th * sigma_full)
        for rho in C.RHOS:
            B = (1 - rho) * C.ETA_B
            fe = (C_eucl_rt + edep + eterm0 <= B) & MASK[:, None]
            fm = (Crt + edep + eterm[None, :] <= B) & MASK[:, None]
            row = {"theta_route": th, "rho": rho, "cand_set": 511,
                   "config": "medium_k03_thetaterm1",
                   "ceiling_pct": 100 * w[fm.any(axis=0)].sum() / TOT}
            if rho in (0.0, 0.2, 0.5):
                row["cov_eucl"] = milp_cov(fe, w, TOT, 30)
                row["cov_M"] = milp_cov(fm, w, TOT, 30)
                row["gap_pp"] = row["cov_eucl"] - row["cov_M"]
            rows.append(row); print(row, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "rawroute_511.csv", index=False)

elif MODE == "imput":
    # Imputation-rule sensitivity : replace the
    # 16-NN denoising of 1 <= m < 5 observed cells by ordinary
    # log-kriging with the frozen variogram (k = 32), and compare the
    # headline coverage results.
    from scipy.sparse.csgraph import dijkstra
    from scipy.spatial import cKDTree
    sys.path.insert(0, str(P / "r2_layer"))
    import r2_config as C
    F0 = np.load(P / "c_layer/sigma_fields.npz")
    sigma0 = F0["sigma_krig"].astype(float)
    s_min = float(F0["s_min"]); s_cap = float(F0["s_cap"])
    F = np.load(P / "r2_layer/field_r2.npz")
    NV = len(sigma0)
    node_grid = D["node_grid"]
    obs_nodes = D["obs_nodes"].astype(np.int64)
    obs_nseg = D["obs_nseg"]
    eu_ = D["edges_u"].astype(np.int64); ev_ = D["edges_v"].astype(np.int64)
    el_ = D["edges_len"].astype(float)
    ET = np.load(P / "r2_layer/eterm.npz")
    eterm0 = float(ET["eterm0_main"][0]); edep = float(ET["edep_main"][0])

    rr, cc = np.nonzero(node_grid >= 0)
    idz = node_grid[rr, cc]
    xy = np.zeros((NV, 2)); xy[idz, 0] = cc * 100.0; xy[idz, 1] = rr * 100.0
    m_lt5 = obs_nseg < 5
    good = obs_nodes[~m_lt5]
    goodv = np.log(np.maximum(sigma0[good], 1e-6))
    NUG, PSILL, RNG = 0.179, 0.076, 3400.0

    def gamma(h):
        return NUG + PSILL * (1.0 - np.exp(-h / RNG))

    tree = cKDTree(xy[good])
    tgt = obs_nodes[m_lt5]
    kk = 32
    d_, nb = tree.query(xy[tgt], k=kk, workers=32)
    est = np.empty(len(tgt))
    for t in range(len(tgt)):
        pts = xy[good[nb[t]]]
        dmat = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
        A = np.empty((kk + 1, kk + 1))
        A[:kk, :kk] = gamma(dmat); np.fill_diagonal(A[:kk, :kk], 0.0)
        A[kk, :kk] = 1.0; A[:kk, kk] = 1.0; A[kk, kk] = 0.0
        b = np.empty(kk + 1); b[:kk] = gamma(d_[t]); b[kk] = 1.0
        try:
            lam = np.linalg.solve(A, b)[:kk]
        except np.linalg.LinAlgError:
            lam = np.full(kk, 1.0 / kk)
        est[t] = float(lam @ goodv[nb[t]])
        if (t + 1) % 2000 == 0:
            print(f"krig {t + 1}/{len(tgt)}", flush=True)
    s_alt = sigma0.copy()
    s_alt[tgt] = np.clip(np.exp(est), s_min, s_cap)
    sig_alt_full = s_alt
    J_map = {int(n): t for t, n in enumerate(J_nodes)}
    sig_alt_j = sig_alt_full[J_nodes]
    sig_r2_j = F["sig_j"].astype(float)
    diff = sig_alt_j - sig_r2_j
    print(f"krig-vs-16NN on J cells: median |diff| "
          f"{np.median(np.abs(diff)):.3f} m, p95 "
          f"{np.quantile(np.abs(diff), .95):.3f} m, corr "
          f"{np.corrcoef(sig_alt_j, sig_r2_j)[0,1]:.4f}", flush=True)

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

    def route_cost(sf):
        removed = sf > reg.sigma_sat(KP)
        Crt = None
        for mp in [C.M_P_MAIN, 0.0]:
            vm = np.maximum(C.V_MIN, np.minimum(
                C.V_CRUISE, (reg.w_corr - reg.z_eps * sf) / (KP * reg.tau_react)))
            on = (sf > reg.sigma_sensor).astype(float)
            unit = P_inst(vm, mp) / vm + on * C.P_AUX / vm
            wgt = 0.5 * (unit[eu_] + unit[ev_]) * el_
            ok = ~(removed[eu_] | removed[ev_])
            g = csr_matrix((np.concatenate([wgt[ok], wgt[ok]]),
                            (np.concatenate([eu_[ok], ev_[ok]]),
                             np.concatenate([ev_[ok], eu_[ok]]))),
                           shape=(NV, NV))
            d = dijkstra(g, directed=False, indices=cand_nodes)[:, J_nodes]
            Crt = (d if Crt is None else Crt + d)
        return np.nan_to_num(Crt, nan=np.inf, posinf=np.inf)

    def eterm_iid_r2(s):
        ph_ld = C.P_B + C.P_IND * (1 + C.M_P_MAIN * C.G / C.W_FRAME)**1.5
        st_desc = (ph_ld - (C.W_FRAME + C.M_P_MAIN * C.G) * C.V_DESC / 2) \
            * (C.H_C / C.V_DESC)
        ph_e = C.P_B + C.P_IND
        st_climb_e = (ph_e + C.W_FRAME * C.V_CLIMB / 2) * (C.H_C / C.V_CLIMB)
        e_ga = 2 * ph_ld * C.T_GA
        q = 1 - np.exp(-C.R_PAD_MAIN**2 / (2 * np.maximum(s, 1e-12)**2))
        N = 1 / q
        return (st_desc + N * ph_ld * C.T_ALIGN + (N - 1) * e_ga
                + ph_ld * C.T_REL + st_climb_e)

    Crt = route_cost(sig_alt_full)
    et = eterm_iid_r2(sig_alt_j)
    rows = []
    for rho in C.RHOS:
        B = (1 - rho) * C.ETA_B
        fe = (C_eucl_rt + edep + eterm0 <= B) & MASK[:, None]
        fm = (Crt + edep + et[None, :] <= B) & MASK[:, None]
        row = {"rho": rho, "cand_set": 511, "imputation": "kriging_k32",
               "cov_eucl": milp_cov(fe, w, TOT, 30) if rho < 0.8 else None,
               "cov_M": milp_cov(fm, w, TOT, 30) if rho < 0.8 else None,
               "ceiling_pct": 100 * w[fm.any(axis=0)].sum() / TOT}
        if row["cov_M"] is not None:
            row["gap_pp"] = row["cov_eucl"] - row["cov_M"]
        rows.append(row); print(row, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "imput_krig_511.csv", index=False)

elif MODE == "universe":
    sys.path.insert(0, str(P / "r2_layer"))
    import r2_config as C
    CM = P / "r2_layer/costmat"
    ET = np.load(P / "r2_layer/eterm.npz")
    eterm = ET["eterm_main"].astype(np.float64)
    eterm0 = float(ET["eterm0_main"][0]); edep = float(ET["edep_main"][0])

    def load(n):
        return np.load(CM / f"{n}.npz")["C"].astype(np.float64)

    rows = []
    for beta in C.BETAS:
        mats = {"Eucl": C_eucl_rt,
                "energy": load("C_ld_energy") + load("C_e_energy")}
        for r in ("High", "Medium", "Low"):
            mats[f"sigma{r}"] = np.nan_to_num(
                load(f"C_ld_sigma{r}_b{beta}") + load(f"C_e_sigma{r}_b{beta}"),
                nan=np.inf, posinf=np.inf)
        for rho in C.RHOS:
            B = (1 - rho) * C.ETA_B
            row = {"theta_route": beta, "rho": rho, "cand_set": 511}
            for abl, Crt in mats.items():
                et = eterm if abl.startswith("sigma") else \
                    np.full_like(eterm, eterm0)
                feas = (Crt + edep + et[None, :] <= B) & MASK[:, None]
                row[abl] = int(feas.any(axis=0).sum())
            rows.append(row); print(row, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "universe_511.csv", index=False)

elif MODE == "lpgap":
    sys.path.insert(0, str(P / "r2_layer"))
    import r2_config as C
    CM = P / "r2_layer/costmat"
    WEIGHT = os.environ["WEIGHT"]
    ET = np.load(P / "r2_layer/eterm.npz")
    eterm = ET["eterm_main"].astype(np.float64)
    eterm0 = float(ET["eterm0_main"][0]); edep = float(ET["edep_main"][0])

    if WEIGHT == "uniform":
        wv = np.ones(len(joined)); tot = float(len(joined))
    elif WEIGHT == "pop":
        wv, tot = w, TOT
    else:
        de = pd.read_parquet(P / "output_demand/wuhan_demand_ev_per_cell.parquet")
        lute = dict(zip(zip(de.grid_x, de.grid_y), de.n_orders.astype(float)))
        wv = np.array([lute.get((int(x), int(y)), 0.0) for x, y in
                       zip(joined.grid_x, joined.grid_y)])
        tot = float(de.n_orders.sum())

    def load(n):
        return np.load(CM / f"{n}.npz")["C"].astype(np.float64)

    def route(abl):
        if abl == "Eucl":
            return C_eucl_rt
        if abl == "energy":
            return load("C_ld_energy") + load("C_e_energy")
        return np.nan_to_num(
            load(f"C_ld_sigma{abl}_b1.0") + load(f"C_e_sigma{abl}_b1.0"),
            nan=np.inf, posinf=np.inf)

    def ip_lp(mask_feas, K):
        at = csr_matrix(mask_feas).T.tocsr()
        ns, nc = mask_feas.shape
        zs = {}
        for kind in ("ip", "lp"):
            m = gp.Model(); m.Params.OutputFlag = 0
            m.Params.Threads = THREADS
            if kind == "ip":
                m.Params.MIPGap = 1e-9
                x = m.addVars(ns, vtype=GRB.BINARY)
            else:
                x = m.addVars(ns, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0)
            y = m.addVars(nc, vtype=GRB.CONTINUOUS, ub=1.0)
            budget_constr(m, x, ns, K)
            for j in range(nc):
                idx = at.indices[at.indptr[j]:at.indptr[j + 1]]
                m.addConstr(y[j] <= (gp.quicksum(x[i] for i in idx)
                                     if len(idx) else 0))
            m.setObjective(gp.quicksum(float(wv[j]) * y[j]
                                       for j in range(nc)), GRB.MAXIMIZE)
            m.optimize()
            assert m.Status == GRB.OPTIMAL
            zs[kind] = m.ObjVal
        return zs["ip"], zs["lp"]

    rows = []
    for abl in ["Eucl", "energy", "High", "Medium", "Low"]:
        et = eterm if abl in ("High", "Medium", "Low") else \
            np.full_like(eterm, eterm0)
        Crt = route(abl)
        for rho in C.RHOS:
            B = (1 - rho) * C.ETA_B
            feas = (Crt + edep + et[None, :] <= B) & MASK[:, None]
            for K in C.K_GRID:
                zip_, zlp = ip_lp(feas, K)
                gam = 0.0 if zip_ <= 0 else (zlp - zip_) / zip_
                rows.append({"weight": WEIGHT, "ablation": abl, "rho": rho,
                             "K": K, "cand_set": 511, "z_ip": zip_,
                             "z_lp": zlp, "gamma": gam})
                print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / f"lpgap_511_{WEIGHT}.csv", index=False)

print("done", flush=True)
