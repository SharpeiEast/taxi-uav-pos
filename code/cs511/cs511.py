"""cs511.py -- base-screen (511) candidate-set full-chain rerun, for the
605-vs-511 comparison report .

Reuses the frozen r2_layer cost matrices: the 511 set is a row subset of
the 605 candidate rows, so no re-routing is needed except the kperp
flanks (MODE=kperp), which rebuild their matrices with a process pool.

MODE (env):
  milp     : WEIGHT in {pop,ev,uniform}, RHO in {0.0,0.2,0.5,0.8};
             6 ablations x 11 K MILPs on the 511 rows.
  decomp   : waterfall (K=30, rho .5/.8) + three-class (3 regimes x 4 rho).
  regret   : Reported/Realized/Best with 20-optimum Eucl pool, K=30 x 4 rho.
  kperp    : KP in {0.1,1.0}; rebuild 6 matrices (parallel), K=30 gaps +
             ceilings for BOTH 605 and 511.
  variants : terminal-model variants (cc95 / bias_med / bias_mix /
             elliptical) K=30 + terminal share, 511 rows.
  scans    : R-7 terminal grid + R-8 payload sweep on 511 rows.

Output: proj3/cs511/<mode-specific>.csv
"""
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix
import gurobipy as gp
from gurobipy import GRB
import sys

sys.path.insert(0, "/lustre/home/2406393544/sharefolder/proj3/r2_layer")
import r2_config as C

P = Path("/lustre/home/2406393544/sharefolder/proj3")
CM = P / "r2_layer/costmat"
OUT = P / "cs511"
OUT.mkdir(exist_ok=True)
MODE = os.environ["MODE"]
THREADS = int(os.environ.get("GRB_THREADS", "32"))

# ---------- candidate mask (base screen, verified aligned) ----------
D = np.load(P / "c_layer/domain.npz")
cand_nodes = D["cand_nodes"].astype(np.int64)
cs = pd.read_parquet(
    P / "v2/data_intermediate/demand_supply/candidate_screens.parquet"
).sort_values("cand_idx").reset_index(drop=True)
assert (cs["node"].values.astype(np.int64) == cand_nodes).all()
MASK = cs["base"].values.astype(bool)
print(f"base-screen candidates: {MASK.sum()}/605", flush=True)

# ---------- shared instance ----------
joined = pd.read_parquet(P / "gk_run/joined_wuhan_gk.parquet")
sig = np.load(P / "r2_layer/field_r2.npz")["sig_j"].astype(float)

def weights(kind):
    if kind == "uniform":
        w = np.ones(len(joined)); return w, float(len(joined))
    if kind == "pop":
        dd = pd.read_parquet(P / "output_demand/wuhan_demand_pop_per_cell.parquet")
        col = "pop_density"
    else:
        dd = pd.read_parquet(P / "output_demand/wuhan_demand_ev_per_cell.parquet")
        col = "n_orders"
    lut = dict(zip(zip(dd.grid_x, dd.grid_y), dd[col].astype(float)))
    w = np.array([lut.get((int(x), int(y)), 0.0) for x, y in
                  zip(joined.grid_x, joined.grid_y)])
    return w, float(dd[col].sum())

ET = np.load(P / "r2_layer/eterm.npz")
eterm = ET["eterm_main"].astype(np.float64)
eterm0 = float(ET["eterm0_main"][0]); edep = float(ET["edep_main"][0])

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

def load(n):
    return np.load(CM / f"{n}.npz")["C"].astype(np.float64)

def milp_cov(mask_feas, w, TOT, K):
    at = csr_matrix(mask_feas).T.tocsr()
    ns, nc = mask_feas.shape
    m = gp.Model(); m.Params.OutputFlag = 0; m.Params.MIPGap = 1e-6
    m.Params.Threads = THREADS
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

E_BASE_E, E_BASE_L = 9.2365, 11.1582
C_eucl_rt = DE * E_BASE_L + DE * E_BASE_E

def route(abl):
    if abl == "Eucl":
        return C_eucl_rt
    if abl == "energy":
        return load("C_ld_energy") + load("C_e_energy")
    return np.nan_to_num(load(f"C_ld_sigma{abl}_b1.0") + load(f"C_e_sigma{abl}_b1.0"),
                         nan=np.inf, posinf=np.inf)

# ==================================================================
if MODE == "milp":
    WEIGHT = os.environ["WEIGHT"]; RHO = float(os.environ["RHO"])
    w, TOT = weights(WEIGHT)
    B = (1 - RHO) * C.ETA_B
    rows = []
    for abl in ["A0", "Eucl", "energy", "High", "Medium", "Low"]:
        if abl == "A0":
            feas = (2 * DE * E_BASE_E <= C.PHI_A0_MAIN * B)
        else:
            et = eterm if abl in ("High", "Medium", "Low") else \
                 np.full_like(eterm, eterm0)
            feas = route(abl) + edep + et[None, :] <= B
        feas = feas & MASK[:, None]
        for K in C.K_GRID:
            cov = milp_cov(feas, w, TOT, K)
            rows.append({"weight": WEIGHT, "rho": RHO, "ablation": abl,
                         "K": K, "coverage_pct": cov, "cand_set": 511})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / f"milp_{WEIGHT}_rho{RHO}.csv", index=False)

elif MODE == "decomp":
    w, TOT = weights("pop")
    C_zeroM = load("C_ld_zeroMedium") + load("C_e_zeroMedium")
    C_energy = route("energy"); C_M = route("Medium")
    rows = []
    for rho in [0.5, 0.8]:
        B = (1 - rho) * C.ETA_B
        for name, Ct, et in [
            ("1_Eucl", C_eucl_rt, None), ("2_ongrid_Energy", C_energy, None),
            ("3_speed_envelope", C_zeroM, None), ("4_route_sigma", C_M, None),
            ("5_terminal_sigma", C_M, eterm)]:
            etv = eterm if et is not None else np.full_like(eterm, eterm0)
            feas = (Ct + edep + etv[None, :] <= B) & MASK[:, None]
            rows.append({"rho": rho, "stage": name,
                         "coverage_pct": milp_cov(feas, w, TOT, 30)})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "waterfall_511.csv", index=False)
    rows = []
    for rname in ["High", "Medium", "Low"]:
        Crt = route(rname)[MASK]
        best = Crt.min(axis=0)
        for rho in C.RHOS:
            B = (1 - rho) * C.ETA_B
            blocked = ~np.isfinite(best)
            over = np.isfinite(best) & (best + edep + eterm0 > B)
            term = (np.isfinite(best) & (best + edep + eterm0 <= B)
                    & (best + edep + eterm > B))
            rows.append({"regime": rname, "rho": rho, "cand_set": 511,
                "blocked_pct": 100 * w[blocked].sum() / TOT,
                "overrange_pct": 100 * w[over].sum() / TOT,
                "terminal_pct": 100 * w[term].sum() / TOT,
                "covered_pct": 100 * w[~(blocked | over | term)].sum() / TOT})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "threeclass_511.csv", index=False)

elif MODE == "regret":
    w, TOT = weights("pop")
    a_m = (route("Medium") + edep + eterm[None, :])
    a_eu = C_eucl_rt + edep + eterm0
    rows = []
    for rho in C.RHOS:
        B = (1 - rho) * C.ETA_B
        fm = (a_m <= B) & MASK[:, None]
        fe = (a_eu <= B) & MASK[:, None]
        atf = csr_matrix(fe).T.tocsr()
        ns, nc = fe.shape
        m = gp.Model(); m.Params.OutputFlag = 0; m.Params.MIPGap = 1e-6
        m.Params.Threads = THREADS
        m.Params.PoolSearchMode = 2; m.Params.PoolSolutions = 20
        m.Params.PoolGap = 1e-6
        x = m.addVars(ns, vtype=GRB.BINARY)
        y = m.addVars(nc, vtype=GRB.CONTINUOUS, ub=1.0)
        m.addConstr(gp.quicksum(x[i] for i in range(ns)) <= 30)
        for j in range(nc):
            idx = atf.indices[atf.indptr[j]:atf.indptr[j + 1]]
            m.addConstr(y[j] <= (gp.quicksum(x[i] for i in idx) if len(idx) else 0))
        m.setObjective(gp.quicksum(float(w[j]) * y[j] for j in range(nc)),
                       GRB.MAXIMIZE)
        m.optimize()
        reported = 100 * m.ObjVal / TOT
        realized = []
        fmb = fm  # bool
        for k in range(m.SolCount):
            m.Params.SolutionNumber = k
            S = [i for i in range(ns) if x[i].Xn > 0.5]
            covered = fmb[S].any(axis=0)
            realized.append(100 * w[covered].sum() / TOT)
        best = milp_cov(fm, w, TOT, 30)
        rows.append({"rho": rho, "K": 30, "cand_set": 511,
                     "Reported_pct": reported,
                     "Realized_pct": float(np.mean(realized)),
                     "Realized_min": float(np.min(realized)),
                     "Realized_max": float(np.max(realized)),
                     "Best_pct": best,
                     "ReportingGap_pp": reported - float(np.mean(realized)),
                     "Regret_pp": best - float(np.mean(realized)),
                     "n_pool": len(realized)})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "regret_511.csv", index=False)

elif MODE == "kperp":
    KP = float(os.environ["KP"])
    from scipy.sparse.csgraph import dijkstra
    from concurrent.futures import ProcessPoolExecutor
    F = np.load(P / "r2_layer/field_r2.npz")
    sigma_full = F["sigma_full"].astype(float)
    NV = len(sigma_full)
    eu_ = D["edges_u"].astype(np.int64); ev_ = D["edges_v"].astype(np.int64)
    el_ = D["edges_len"].astype(float)

    def P_inst(v, m_p):
        v = np.maximum(np.asarray(v, float), 1e-9)
        kap = 1.0 + m_p * C.G / C.W_FRAME
        v0sq = kap * C.V0_HOV ** 2
        return (C.P_B * (1 + 3 * v**2 / C.U_TIP**2)
                + (kap**1.5) * C.P_IND * np.sqrt(
                    np.sqrt(1 + v**4/(4*v0sq**2)) - v**2/(2*v0sq))
                + 0.5 * C.D0 * C.RHO_AIR * C.S_SOL * C.A_DISC * v**3)

    def one(job):
        rname, mp = job
        reg = C.REGIMES_BD[rname]
        vm = np.maximum(C.V_MIN, np.minimum(C.V_CRUISE,
            (reg.w_corr - reg.z_eps * sigma_full) / (KP * reg.tau_react)))
        on = (sigma_full > reg.sigma_sensor).astype(float)
        unit = P_inst(vm, mp) / vm + on * C.P_AUX / vm
        removed = sigma_full > reg.sigma_sat(KP)
        wgt = 0.5 * (unit[eu_] + unit[ev_]) * el_
        ok = ~(removed[eu_] | removed[ev_])
        g = csr_matrix((np.concatenate([wgt[ok], wgt[ok]]),
                        (np.concatenate([eu_[ok], ev_[ok]]),
                         np.concatenate([ev_[ok], eu_[ok]]))), shape=(NV, NV))
        return rname, mp, dijkstra(g, directed=False,
                                   indices=cand_nodes)[:, J_nodes]

    jobs = [(r, mp) for r in ("High", "Medium", "Low")
            for mp in (C.M_P_MAIN, 0.0)]
    res = {}
    with ProcessPoolExecutor(max_workers=6) as ex:
        for rname, mp, Cr in ex.map(one, jobs):
            res[(rname, mp)] = Cr
    w, TOT = weights("pop")
    rows = []
    for rname in ("High", "Medium", "Low"):
        Crt = np.nan_to_num(res[(rname, C.M_P_MAIN)] + res[(rname, 0.0)],
                            nan=np.inf, posinf=np.inf)
        for nset, msk in (("605", np.ones(605, bool)), ("511", MASK)):
            best = Crt[msk].min(axis=0)
            for rho in C.RHOS:
                B = (1 - rho) * C.ETA_B
                feas = (Crt + edep + eterm[None, :] <= B) & msk[:, None]
                fe = (C_eucl_rt + edep + eterm0 <= B) & msk[:, None]
                row = {"kperp": KP, "regime": rname, "rho": rho,
                       "cand_set": nset,
                       "ceiling_pct": 100 * w[feas.any(axis=0)].sum() / TOT}
                if rho in (0.0, 0.2, 0.5):
                    row["cov_eucl"] = milp_cov(fe, w, TOT, 30)
                    row["cov_sigma"] = milp_cov(feas, w, TOT, 30)
                    row["gap_pp"] = row["cov_eucl"] - row["cov_sigma"]
                rows.append(row); print(row, flush=True)
    pd.DataFrame(rows).to_csv(OUT / f"kperp_{KP:g}_cmp.csv", index=False)

elif MODE == "variants":
    from scipy.stats import ncx2, norm
    from numpy.polynomial.legendre import leggauss
    w, TOT = weights("pop")
    route_M = route("Medium")

    def hover(m_p):
        return C.P_B + C.P_IND * (1 + m_p * C.G / C.W_FRAME) ** 1.5
    ph_ld = hover(C.M_P_MAIN); ph_e = hover(0.0)
    t_v = C.H_C / C.V_CLIMB
    DJ = (ph_ld - (C.W_FRAME + C.M_P_MAIN * C.G) * C.V_DESC / 2) * t_v
    CJ = (ph_e + C.W_FRAME * C.V_CLIMB / 2) * t_v
    E_GA = 2.0 * ph_ld * C.T_GA

    def et_from_q(q):
        N = 1.0 / np.clip(q, 1e-12, 1.0)
        return DJ + N * ph_ld * C.T_ALIGN + (N - 1) * E_GA + ph_ld * C.T_REL + CJ

    def q_iid(s):
        s = np.maximum(s, 1e-12)
        return 1 - np.exp(-C.R_PAD_MAIN**2 / (2 * s**2))

    def q_bias(s, lam):
        s = np.maximum(s, 1e-12)
        return np.clip(ncx2.cdf((C.R_PAD_MAIN / s) ** 2, df=2, nc=lam**2),
                       1e-12, 1.0)

    def q_ellip(smax, smin, nodes=200):
        xg, wg = leggauss(nodes)
        smax = np.maximum(smax, 1e-12); smin = np.maximum(smin, 1e-12)
        r = C.R_PAD_MAIN
        x = r * xg[None, :]
        half = np.sqrt(np.maximum(r**2 - x**2, 0.0))
        fx = norm.pdf(x / smax[:, None]) / smax[:, None]
        slab = norm.cdf(half / smin[:, None]) - norm.cdf(-half / smin[:, None])
        q = np.clip((fx * slab * (r * wg[None, :])).sum(axis=1), 0, 1)
        return np.where(smax < 0.2, 1.0, q)

    e8 = pd.read_csv(P / "E8_urbannav/e8_segments.csv")
    e8 = e8[(e8.variant == "tmin30_full") & (e8.dev_class == "receiver")
            & (e8.sigma_gk > 0.05)]
    lam = (e8.bias_norm / e8.sigma_gk).values
    lam = lam[np.isfinite(lam)]
    gk = pd.read_csv(P / "opera/rev02_A1/gk_cells_export.csv")
    m = joined[["grid_x", "grid_y"]].merge(gk, on=["grid_x", "grid_y"], how="left")
    ecc = m["ecc_med"].values.astype(float)
    ecc = np.where(np.isfinite(ecc), ecc, np.nanmedian(ecc))
    smin = sig * np.sqrt(np.maximum(1 - ecc**2, 0))

    q = q_iid(sig)
    Nmix = np.zeros_like(sig)
    for l in lam:
        Nmix += 1.0 / q_bias(sig, l)
    ets = {
        "iid_headline": et_from_q(q),
        "cc95": et_from_q(1 - (1 - np.clip(q, 1e-12, 1 - 1e-16)) **
                          np.maximum(np.ceil(np.log(0.05) /
                          np.log(np.clip(1 - q, 1e-300, 1 - 1e-16))), 1)),
        "bias_med": et_from_q(q_bias(sig, float(np.median(lam)))),
        "elliptical": et_from_q(q_ellip(sig, smin)),
    }
    # cc95: energy uses Nmax directly, not prob transform -- recompute properly
    with np.errstate(divide="ignore", invalid="ignore"):
        nmax = np.ceil(np.log(0.05) / np.log(np.clip(1 - q, 1e-300, 1 - 1e-16)))
    nmax = np.where(q > 1 - 1e-12, 1.0, np.maximum(nmax, 1))
    ets["cc95"] = DJ + nmax * ph_ld * C.T_ALIGN + (nmax - 1) * E_GA \
        + ph_ld * C.T_REL + CJ
    ets["bias_mix"] = DJ + (Nmix / len(lam)) * ph_ld * C.T_ALIGN \
        + (Nmix / len(lam) - 1) * E_GA + ph_ld * C.T_REL + CJ

    rows = []
    best = route_M[MASK].min(axis=0)
    for vname, et in ets.items():
        for rho in (0.0, 0.5):
            B = (1 - rho) * C.ETA_B
            feas = (route_M + edep + et[None, :] <= B) & MASK[:, None]
            fe = (C_eucl_rt + edep + eterm0 <= B) & MASK[:, None]
            term = (np.isfinite(best) & (best + edep + eterm0 <= B)
                    & (best + edep + et > B))
            rows.append({"variant": vname, "rho": rho, "cand_set": 511,
                         "cov_eucl": milp_cov(fe, w, TOT, 30),
                         "cov_M": milp_cov(feas, w, TOT, 30),
                         "terminal_pct": 100 * w[term].sum() / TOT})
            rows[-1]["gap_pp"] = rows[-1]["cov_eucl"] - rows[-1]["cov_M"]
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "variants_511.csv", index=False)

elif MODE == "scans":
    w, TOT = weights("pop")
    route_M = route("Medium")
    rows = []
    for rp in C.R_PAD_SWEEP:
        for ta in C.T_ALIGN_SWEEP:
            for hc in C.H_C_SWEEP:
                tag = f"rp{rp:g}_ta{ta:g}_hc{hc:g}"
                et = ET[f"eterm_{tag}"].astype(float)
                et0 = float(ET[f"eterm0_{tag}"][0])
                ed = float(ET[f"edep_{tag}"][0])
                for rho in (0.0, 0.5):
                    B = (1 - rho) * C.ETA_B
                    feas = (route_M + ed + et[None, :] <= B) & MASK[:, None]
                    fe = (C_eucl_rt + ed + et0 <= B) & MASK[:, None]
                    rows.append({"scan": "terminal", "tag": tag, "rho": rho,
                                 "cov_M": milp_cov(feas, w, TOT, 30),
                                 "cov_eucl": milp_cov(fe, w, TOT, 30)})
                    print(rows[-1], flush=True)
    for mp in C.M_P_SWEEP:
        tag = f"mp{mp:g}"
        et = ET[f"eterm_{tag}"].astype(float)
        ed = float(ET[f"edep_{tag}"][0])
        Cl = np.nan_to_num(load("C_ld_sigmaMedium_b1.0") * 0 + route_M,
                           nan=np.inf)  # payload uses main route (approx as r2_scans did re-route; note in report)
        for rho in (0.0, 0.5):
            B = (1 - rho) * C.ETA_B
            feas = (route_M + ed + et[None, :] <= B) & MASK[:, None]
            rows.append({"scan": "payload", "tag": tag, "rho": rho,
                         "cov_M": milp_cov(feas, w, TOT, 30), "cov_eucl": None})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "scans_511.csv", index=False)

print("done", flush=True)
