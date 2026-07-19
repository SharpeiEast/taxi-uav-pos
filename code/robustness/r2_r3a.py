"""trc_r3a.py -- quick package (expert re-review items 4, 9, 10).

(a) Anisotropy-consistent terminal model: per-cell elliptical Gaussian
    capture probability (sigma_max from the field, sigma_min via measured
    eccentricity) vs the headline isotropic worst-direction Rayleigh.
    Verifies numerically that the isotropic form is a conservative
    envelope (q_ellip >= q_iso for every cell), then re-runs the Medium
    chain with the elliptical variant.
(b) Vehicle-day block bootstrap vs vehicle-cluster bootstrap for cell SEs.
(c) UAV-feasibility-screened candidate sets (v2 open-data screens):
    K=30 coverage with strict/base candidate subsets.

Outputs: trc_exp/r3a_ellip.csv, r3a_bootstrap.json, r3a_screens.csv
"""
import json
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
OUT = P / "trc_exp/r2base"

joined = pd.read_parquet(P / "gk_run/joined_wuhan_gk.parquet")
sig = np.load(P / "r2_layer/field_r2.npz")["sig_j"].astype(float)
dd = pd.read_parquet(P / "output_demand/wuhan_demand_pop_per_cell.parquet")
lut = dict(zip(zip(dd.grid_x, dd.grid_y), dd.pop_density.astype(float)))
w = np.array([lut.get((int(x), int(y)), 0.0) for x, y in
              zip(joined.grid_x, joined.grid_y)])
TOT = float(dd.pop_density.sum())

# ---------- terminal machinery (same frozen formulas) ----------
def hover_power(m_p):
    kap = 1.0 + m_p * C.G / C.W_FRAME
    return C.P_B + C.P_IND * kap ** 1.5

ph_ld = hover_power(C.M_P_MAIN)
ph_e = hover_power(0.0)
w_ld, w_e = C.W_FRAME + C.M_P_MAIN * C.G, C.W_FRAME
t_v = C.H_C / C.V_CLIMB
DJ = (ph_ld - w_ld * C.V_DESC / 2) * t_v
CJ = (ph_e + w_e * C.V_CLIMB / 2) * t_v
CI = (ph_ld + w_ld * C.V_CLIMB / 2) * t_v
DI = (ph_e - w_e * C.V_DESC / 2) * t_v
E_GA = 2.0 * ph_ld * C.T_GA
EDEP = CI + DI

def eterm_from_q(q):
    N = 1.0 / np.clip(q, 1e-12, 1.0)
    return DJ + N * ph_ld * C.T_ALIGN + (N - 1) * E_GA + ph_ld * C.T_REL + CJ

def q_iso(s, r=C.R_PAD_MAIN):
    s = np.maximum(np.asarray(s, float), 1e-12)
    return 1.0 - np.exp(-r ** 2 / (2.0 * s ** 2))

def q_ellip(smax, smin, r=C.R_PAD_MAIN, nodes=200):
    """P(X^2+Y^2 <= r^2), X~N(0,smax^2), Y~N(0,smin^2), by Gauss-Legendre
    over x with the y-slice in closed form."""
    from numpy.polynomial.legendre import leggauss
    from scipy.stats import norm
    xg, wg = leggauss(nodes)                     # on [-1,1]
    smax = np.maximum(np.asarray(smax, float), 1e-12)
    smin = np.maximum(np.asarray(smin, float), 1e-12)
    x = r * xg[None, :]                          # (ncell, nodes) via broadcast
    half = np.sqrt(np.maximum(r ** 2 - x ** 2, 0.0))
    fx = norm.pdf(x / smax[:, None]) / smax[:, None]
    slab = (norm.cdf(half / smin[:, None]) - norm.cdf(-half / smin[:, None]))
    q = np.clip((fx * slab * (r * wg[None, :])).sum(axis=1), 0.0, 1.0)
    # degenerate near-zero-dispersion cells: capture is certain
    return np.where(smax < 0.2, 1.0, q)

# per-cell eccentricity
gk = pd.read_csv(P / "opera/rev02_A1/gk_cells_export.csv")
print("gk_cells cols:", list(gk.columns), flush=True)
key = ["grid_x", "grid_y"]
m = joined[key].merge(gk, on=key, how="left")
ecc = m["ecc_med"].values.astype(float)
ecc = np.where(np.isfinite(ecc), ecc, np.nanmedian(ecc))
smin = sig * np.sqrt(np.maximum(1.0 - ecc ** 2, 0.0))

qi = q_iso(sig)
qe = q_ellip(sig, smin)
env_ok = bool((qe >= qi - 1e-9).all())
print(f"conservative envelope q_ellip>=q_iso: {env_ok} "
      f"(min diff {np.min(qe-qi):.2e})", flush=True)

et_iso = eterm_from_q(qi)
et_ell = eterm_from_q(qe)

# Medium chain rerun with elliptical terminal
def load(n):
    return np.load(P / f"r2_layer/costmat/{n}.npz")["C"].astype(np.float64)

route_M = np.nan_to_num(load("C_ld_sigmaMedium_b1.0") + load("C_e_sigmaMedium_b1.0"),
                        nan=np.inf, posinf=np.inf)
best = route_M.min(axis=0)
ET0 = float(eterm_from_q(q_iso(np.array([0.0])))[0])

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
C_eucl_rt = DE * 11.1582 + DE * 9.2365

def milp_cov(mask, K=30, cand_mask=None):
    if cand_mask is not None:
        mask = mask & cand_mask[:, None]
    at = csr_matrix(mask).T.tocsr()
    ns, nc = mask.shape
    mdl = gp.Model(); mdl.Params.OutputFlag = 0; mdl.Params.MIPGap = 1e-6
    mdl.Params.Threads = 16
    x = mdl.addVars(ns, vtype=GRB.BINARY)
    y = mdl.addVars(nc, vtype=GRB.CONTINUOUS, ub=1.0)
    mdl.addConstr(gp.quicksum(x[i] for i in range(ns)) <= K)
    for j in range(nc):
        idx = at.indices[at.indptr[j]:at.indptr[j + 1]]
        mdl.addConstr(y[j] <= (gp.quicksum(x[i] for i in idx) if len(idx) else 0))
    mdl.setObjective(gp.quicksum(float(w[j]) * y[j] for j in range(nc)), GRB.MAXIMIZE)
    mdl.optimize()
    assert mdl.Status == GRB.OPTIMAL
    return 100.0 * mdl.ObjVal / TOT

rows = []
for tag, et in [("iso_worstdir_headline", et_iso), ("elliptical_measured", et_ell)]:
    for rho in C.RHOS:
        B = (1 - rho) * C.ETA_B
        blocked = ~np.isfinite(best)
        over = np.isfinite(best) & (best + EDEP + ET0 > B)
        term = (np.isfinite(best) & (best + EDEP + ET0 <= B)
                & (best + EDEP + et > B))
        row = {"terminal": tag, "rho": rho,
               "terminal_pct": 100 * w[term].sum() / TOT,
               "ceiling_covered_pct": 100 * w[~(blocked | over | term)].sum() / TOT,
               "N_median": float(np.median(1 / np.clip(
                   qi if tag.startswith("iso") else qe, 1e-12, 1)))}
        if rho in (0.0, 0.5):
            cov_eu = milp_cov(C_eucl_rt + EDEP + ET0 <= B)
            covM = milp_cov(route_M + EDEP + et[None, :] <= B)
            row.update(cov_eucl=cov_eu, cov_M=covM, gap_pp=cov_eu - covM)
        rows.append(row); print(row, flush=True)
pd.DataFrame(rows).to_csv(OUT / "r3a_ellip.csv", index=False)

# ---------- (b) vehicle-day block bootstrap ----------
seg = pd.read_parquet(
    P / "v2/data_intermediate/taxi_prior/poc_wuhan/intermediate/segments.parquet")
print("segments cols:", list(seg.columns), flush=True)
# expected: vehicle_id, start_time, sigma_gk, and coordinates or cell ids
tcol = "start_time" if "start_time" in seg.columns else "hour_start"
seg["day"] = pd.to_datetime(seg[tcol]).dt.date.astype(str)
seg = seg.rename(columns={"sigma_gk_m": "sigma_gk"})
# cell keys from lon/lat (equirectangular, frozen origin)
LAT0, LON0 = 30.582140, 114.289100
mx = (seg.median_lon - LON0) * 111320.0 * np.cos(np.radians(LAT0))
my = (seg.median_lat - LAT0) * 110540.0
seg["cell"] = list(zip((mx // 100).astype(int), (my // 100).astype(int)))
rng = np.random.default_rng(20260716)
cells = [c for c, g in seg.groupby("cell") if len(g) >= 5]
samp = rng.choice(len(cells), size=min(2000, len(cells)), replace=False)
ratios = []
NB = 200
for k in samp:
    g = seg[seg.cell == cells[k]]
    se = {}
    for mode in ("vehicle", "vehicle_day"):
        blocks = (g.vehicle_id.astype(str) if mode == "vehicle"
                  else g.vehicle_id.astype(str) + "_" + g.day)
        uniq = blocks.unique()
        if len(uniq) < 2:
            se[mode] = np.nan
            continue
        by = {b: g.sigma_gk.values[blocks.values == b] for b in uniq}
        meds = []
        for _ in range(NB):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            meds.append(np.median(np.concatenate([by[b] for b in pick])))
        se[mode] = float(np.std(meds))
    if np.isfinite(se["vehicle"]) and np.isfinite(se["vehicle_day"]) and se["vehicle"] > 0:
        ratios.append(se["vehicle_day"] / se["vehicle"])
ratios = np.array(ratios)
boot = {"n_cells_sampled": int(len(ratios)),
        "se_ratio_day_over_vehicle": {q: round(float(np.quantile(ratios, qq)), 3)
            for q, qq in [("p10", .1), ("p50", .5), ("p90", .9)]},
        "note": "ratio ~1 => vehicle clustering already captures the dependence"}
json.dump(boot, open(OUT / "r3a_bootstrap.json", "w"), indent=2)
print(boot, flush=True)

# ---------- (c) screened candidate sets ----------
try:
    cs = pd.read_parquet(P / "v2/data_intermediate/demand_supply/candidate_screens.parquet")
    print("screens cols:", list(cs.columns), flush=True)
    st = pd.read_csv(P / "wuhan_stations_geocoded.csv")
    print("stations cols:", list(st.columns), flush=True)
    loncol = [c for c in cs.columns if "lon" in c.lower()][0]
    latcol = [c for c in cs.columns if "lat" in c.lower()][0]
    slon = [c for c in st.columns if "lon" in c.lower()][0]
    slat = [c for c in st.columns if "lat" in c.lower()][0]
    from scipy.spatial import cKDTree
    tree = cKDTree(np.c_[cs[loncol].values, cs[latcol].values])
    dist, idx = tree.query(np.c_[st[slon].values, st[slat].values])
    print(f"match: max coord dist {dist.max():.2e}", flush=True)
    passcols = [c for c in cs.columns if cs[c].dtype == bool or
                str(cs[c].dtype).startswith("bool")]
    print("bool cols:", passcols, flush=True)
    rows = []
    for scol in passcols:
        cmask = cs[scol].values[idx]
        for rho in (0.0, 0.2, 0.5):
            B = (1 - rho) * C.ETA_B
            covM = milp_cov(route_M + EDEP + et_iso[None, :] <= B, cand_mask=cmask)
            rows.append({"screen": scol, "n_cand": int(cmask.sum()), "rho": rho,
                         "cov_M": covM})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "r3a_screens.csv", index=False)
except Exception as e:
    print("SCREENS SKIPPED:", repr(e), flush=True)
print("done", flush=True)
