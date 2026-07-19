"""trc_variants.py -- study revision experiments E1 / E2 / E4 / E5.

Frozen bd_layer is NOT touched; all outputs to proj3/trc_exp/.

E1  beta_term x beta_route separation grid (Medium reference regime):
    reach ceilings + three-class decomposition + K=30 MILP gaps.
E2  bias-aware correlated-retry terminal model, calibrated on UrbanNav E-8
    receiver-class segments (bias_norm / sigma_gk ratio lambda):
    per-attempt offset = b + eps, |b| = lambda*sigma (quasi-static NLOS,
    shared across retries), eps ~ N(0, sigma^2 I2) redrawn per attempt.
    q(b) = P(|b+eps| <= r_pad) = ncx2.cdf((r/sig)^2, df=2, nc=(|b|/sig)^2).
    E[N] variants: point lambda_med, point lambda_p95, empirical mixture.
E5  chance-constrained terminal: N_max = ceil(ln eps / ln(1-q)), eps=0.05.
E4  beta anchor table: UrbanNav ground receiver-class error quantiles vs
    AGZ (low-alt) / MARS ublox (high-alt) aerial quantiles.

Outputs: e1_beta_grid.csv, e1_milp_gaps.csv, e2_lambda_fit.json,
         e2_eterm_curves.csv, e2_e5_variants.csv, e4_beta_anchors.csv,
         trc_variants_summary.json, eterm_variants.npz
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.stats import ncx2
import gurobipy as gp
from gurobipy import GRB
import sys

sys.path.insert(0, "/lustre/home/2406393544/sharefolder/proj3/bd_layer")
import bd_config as C

P = Path("/lustre/home/2406393544/sharefolder/proj3")
CM = P / "r2_layer/costmat"
OUT = P / "trc_exp/r2base"
OUT.mkdir(exist_ok=True)

# ---------------- shared instance pieces (identical to bd_decomp) ----------------
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

def load(n):
    return np.load(CM / f"{n}.npz")["C"].astype(np.float64)

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

def threeclass(best_route, eterm_vec_, eterm0_, edep_, rho):
    B = (1 - rho) * C.ETA_B
    blocked = ~np.isfinite(best_route)
    over = np.isfinite(best_route) & (best_route + edep_ + eterm0_ > B)
    term = (np.isfinite(best_route) & (best_route + edep_ + eterm0_ <= B)
            & (best_route + edep_ + eterm_vec_ > B))
    covered = ~(blocked | over | term)
    return {"blocked_pct": 100 * w[blocked].sum() / TOT,
            "overrange_pct": 100 * w[over].sum() / TOT,
            "terminal_pct": 100 * w[term].sum() / TOT,
            "covered_pct": 100 * w[covered].sum() / TOT}

# ---------------- terminal-energy machinery (bd_eterm formulas) ----------------
def hover_power(m_p):
    kap = 1.0 + m_p * C.G / C.W_FRAME
    return C.P_B + C.P_IND * kap ** 1.5

def stages(m_p=C.M_P_MAIN, h_c=C.H_C):
    ph_ld, ph_e = hover_power(m_p), hover_power(0.0)
    w_ld, w_e = C.W_FRAME + m_p * C.G, C.W_FRAME
    t = h_c / C.V_CLIMB
    return dict(dj=(ph_ld - w_ld * C.V_DESC / 2) * t,
                cj=(ph_e + w_e * C.V_CLIMB / 2) * t,
                ci=(ph_ld + w_ld * C.V_CLIMB / 2) * t,
                di=(ph_e - w_e * C.V_DESC / 2) * t,
                ph=ph_ld)

ST = stages()
E_GA = 2.0 * ST["ph"] * C.T_GA
EDEP = ST["ci"] + ST["di"]

def eterm_from_N(N):
    return (ST["dj"] + N * ST["ph"] * C.T_ALIGN + (N - 1.0) * E_GA
            + ST["ph"] * C.T_REL + ST["cj"])

def q_iid(s, r_pad=C.R_PAD_MAIN):
    s = np.maximum(np.asarray(s, float), 1e-12)
    q = 1.0 - np.exp(-r_pad ** 2 / (2.0 * s ** 2))
    return np.where(np.asarray(s) <= 1e-12, 1.0, q)

def eterm_iid(s):
    return eterm_from_N(1.0 / q_iid(s))

def q_bias(s, lam, r_pad=C.R_PAD_MAIN):
    """capture prob with quasi-static bias |b| = lam * s, fast noise sd = s."""
    s = np.maximum(np.asarray(s, float), 1e-12)
    return np.clip(ncx2.cdf((r_pad / s) ** 2, df=2, nc=(lam) ** 2), 1e-12, 1.0)

def eterm_bias_point(s, lam):
    return eterm_from_N(1.0 / q_bias(s, lam))

def eterm_bias_mix(s, lams):
    N = np.zeros_like(np.asarray(s, float))
    for lam in lams:
        N += 1.0 / q_bias(s, lam)
    return eterm_from_N(N / len(lams))

def eterm_cc(s, eps=0.05):
    q = q_iid(s)
    with np.errstate(divide="ignore", invalid="ignore"):
        n = np.ceil(np.log(eps) / np.log(np.clip(1.0 - q, 1e-300, 1 - 1e-16)))
    n = np.where(q > 1 - 1e-12, 1.0, n)
    return eterm_from_N(np.maximum(n, 1.0))

ETERM0 = float(eterm_iid(np.array([0.0]))[0])
print(f"ETERM0={ETERM0/1e3:.2f} kJ EDEP={EDEP/1e3:.2f} kJ (bd: 29.76/16.97-ish)",
      flush=True)
# cross-check vs frozen npz
ET = np.load(P / "r2_layer/eterm.npz")
assert np.allclose(eterm_iid(sig), ET["eterm_main"], rtol=1e-9)
assert np.isclose(ETERM0, float(ET["eterm0_main"][0]))
assert np.isclose(EDEP, float(ET["edep_main"][0]))
print("frozen-eterm cross-check PASS", flush=True)

# ---------------- route matrices ----------------
C_eucl_rt = DE * 11.1582 + DE * 9.2365
route_M = {}
for beta in C.BETAS:
    route_M[beta] = np.nan_to_num(
        load(f"C_ld_sigmaMedium_b{beta}") + load(f"C_e_sigmaMedium_b{beta}"),
        nan=np.inf, posinf=np.inf)
best_M = {b: route_M[b].min(axis=0) for b in route_M}

summary = {}

# ================= E1: beta_term x beta_route grid =================
print("=== E1 grid ===", flush=True)
BT = [0.25, 0.5, 0.75, 1.0]
rows = []
eterm_bt = {bt: eterm_iid(bt * sig) for bt in BT}
for br in C.BETAS:
    for bt in BT:
        for rho in C.RHOS:
            tc = threeclass(best_M[br], eterm_bt[bt], ETERM0, EDEP, rho)
            rows.append({"beta_route": br, "beta_term": bt, "rho": rho, **tc})
pd.DataFrame(rows).to_csv(OUT / "e1_beta_grid.csv", index=False)

combos = [(1.0, 1.0), (1.0, 0.5), (1.0, 0.25), (0.5, 1.0), (0.5, 0.5), (0.25, 1.0)]
rows = []
for rho in [0.0, 0.2, 0.5]:
    B = (1 - rho) * C.ETA_B
    cov_eu = milp_cov(C_eucl_rt + EDEP + ETERM0 <= B)
    for br, bt in combos:
        covM = milp_cov(route_M[br] + EDEP + eterm_bt[bt][None, :] <= B)
        rows.append({"rho": rho, "beta_route": br, "beta_term": bt,
                     "cov_eucl": cov_eu, "cov_M": covM, "gap_pp": cov_eu - covM})
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(OUT / "e1_milp_gaps.csv", index=False)

# ================= E2: lambda fit from UrbanNav E-8 =================
print("=== E2 lambda fit ===", flush=True)
e8 = pd.read_csv(P / "E8_urbannav/e8_segments.csv")
e8 = e8[(e8.variant == "tmin30_full") & (e8.dev_class == "receiver")].copy()
e8 = e8[(e8.sigma_gk > 0.05)]                      # drop frozen-output degenerates
lam = (e8.bias_norm / e8.sigma_gk).values
lam = lam[np.isfinite(lam)]
lamfit = {"n_segments": int(len(lam)),
          "lambda_median": float(np.median(lam)),
          "lambda_p25": float(np.quantile(lam, .25)),
          "lambda_p75": float(np.quantile(lam, .75)),
          "lambda_p95": float(np.quantile(lam, .95)),
          "source": "UrbanNav E-8 receiver-class tmin30 segments"}
json.dump(lamfit, open(OUT / "e2_lambda_fit.json", "w"), indent=2)
print(lamfit, flush=True)
lam_med, lam_p95 = lamfit["lambda_median"], lamfit["lambda_p95"]

# eterm curves on a sigma grid (for the paper figure/table)
sgrid = np.concatenate([np.arange(0.5, 12.01, 0.25), np.arange(12.5, 30.1, 0.5)])
curves = pd.DataFrame({
    "sigma": sgrid,
    "N_iid": 1.0 / q_iid(sgrid),
    "N_cc95": np.maximum(np.ceil(np.log(0.05) /
              np.log(np.clip(1 - q_iid(sgrid), 1e-300, 1 - 1e-16))), 1),
    "N_bias_med": 1.0 / q_bias(sgrid, lam_med),
    "N_bias_mix": np.array([np.mean(1.0 / q_bias(np.array([s]), lam)) for s in sgrid]),
    "Eterm_iid_kJ": eterm_iid(sgrid) / 1e3,
    "Eterm_cc95_kJ": eterm_cc(sgrid) / 1e3,
    "Eterm_bias_med_kJ": eterm_bias_point(sgrid, lam_med) / 1e3,
})
curves.to_csv(OUT / "e2_eterm_curves.csv", index=False)

# ================= E2 + E5 variant reruns (Medium, beta_route=1) =================
print("=== E2/E5 variant reruns ===", flush=True)
variants = {
    "iid_headline": eterm_iid(sig),
    "cc95": eterm_cc(sig),
    "bias_med": eterm_bias_point(sig, lam_med),
    "bias_mix": eterm_bias_mix(sig, lam),
    "bias_p95": eterm_bias_point(sig, lam_p95),
}
np.savez_compressed(OUT / "eterm_variants.npz",
                    **{k: v for k, v in variants.items()},
                    sigma=sig, lam=lam)
rows = []
for vname, ev in variants.items():
    for rho in C.RHOS:
        tc = threeclass(best_M[1.0], ev, ETERM0, EDEP, rho)
        row = {"variant": vname, "rho": rho, **tc}
        if rho in (0.0, 0.2, 0.5):
            B = (1 - rho) * C.ETA_B
            cov_eu = milp_cov(C_eucl_rt + EDEP + ETERM0 <= B)
            covM = milp_cov(route_M[1.0] + EDEP + ev[None, :] <= B)
            row.update(cov_eucl=cov_eu, cov_M=covM, gap_pp=cov_eu - covM)
        rows.append(row)
        print(row, flush=True)
pd.DataFrame(rows).to_csv(OUT / "e2_e5_variants.csv", index=False)

# ================= E4: beta anchor table =================
print("=== E4 anchors ===", flush=True)
ff = json.load(open(P / "v2/data_intermediate/air_fair/fair_family.json"))
hk = e8[e8.seq.str.startswith("HK")]
g50 = float(np.median(hk.err_med)); g95 = float(np.median(hk.err_p95))
rows = [{"anchor": "ground vehicle-grade (UrbanNav HK canyons)",
         "band": "0 m (street)", "q50_m": g50, "q95_m": g95}]
lowmid = ff["anchors"]["low"]["mid"]["q"]
rows.append({"anchor": "aerial low (AGZ urban MAV)", "band": "5-15 m",
             "q50_m": lowmid["0.5"], "q95_m": lowmid["0.95"]})
for k, lab in [("open", "aerial high open (MARS airport ublox)"),
               ("mid", "aerial high built (MARS island ublox)")]:
    qq = ff["anchors"]["high"][k]["q"]
    rows.append({"anchor": lab, "band": "80-130 m",
                 "q50_m": qq["0.5"], "q95_m": qq["0.95"]})
an = pd.DataFrame(rows)
an["beta_vs_ground_q50"] = an.q50_m / g50
an["beta_vs_ground_q95"] = an.q95_m / g95
an.to_csv(OUT / "e4_beta_anchors.csv", index=False)
print(an.to_string(index=False), flush=True)

summary["lambda_fit"] = lamfit
summary["eterm0_kJ"] = ETERM0 / 1e3
summary["beta_anchor_q50_range"] = [float(an.beta_vs_ground_q50.min()),
                                    float(an.beta_vs_ground_q50.max())]
json.dump(summary, open(OUT / "trc_variants_summary.json", "w"), indent=2)
print("done", flush=True)
