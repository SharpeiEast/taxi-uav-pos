"""bd_reach.py -- R-1 step 3: reachability under the NEW criterion (Decision I)

  a = 1[ C_out(ld) + C_ret(e) + E_dep + E_term(sigma~) <= (1-rho) etaB ]
  sigma~ = unscaled sigma_j for sigma-aware; 0 for blind (Eucl/energy/zero-r).
  A0 (B3): 2 D_ij E_base <= phi (1-rho) etaB, phi in {0.6, 0.7}; NO E_term
           (practitioner discount proxies hidden overheads; not in the chain).

Outputs:
  bd_layer/reach/a_{abl}[_b{beta}]_rho{rho}.npz   (sparse bool 605x17,899)
  bd_layer/table5_bd.csv   (universe sizes; Energy col must be beta-invariant)
  assertions: nesting a_L<=a_M<=a_H<=a_energy<=a_Eucl at all 16 (beta,rho).
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix, save_npz

sys.path.insert(0, "/lustre/home/2406393544/sharefolder/proj3/r2_layer")
import r2_config as C

P = Path("/lustre/home/2406393544/sharefolder/proj3")
CM = P / "r2_layer/costmat"
OUT = P / "r2_layer/reach"
OUT.mkdir(exist_ok=True)

ET = np.load(P / "r2_layer/eterm.npz")
eterm = ET["eterm_main"].astype(np.float64)          # (17,899,) sigma-aware
eterm0 = float(ET["eterm0_main"][0])
edep = float(ET["edep_main"][0])
print(f"E_term(0)={eterm0/1e3:.2f} kJ  E_dep={edep/1e3:.2f} kJ", flush=True)

def load(name):
    return np.load(CM / f"{name}.npz")["C"].astype(np.float64)

# Euclidean straight-line metres, measured from SNAPPED candidate cell
# centres (instance definition: stations live at their snapped cells,
# C_ii = 0) -- raw-coordinate endpoints would break energy <= Eucl nesting
# by up to the snap displacement (diagnosed 2026-07-07: 42 pairs, <=1.4 kJ).
D = np.load(P / "c_layer/domain.npz")
joined = pd.read_parquet(P / "gk_run/joined_wuhan_gk.parquet")
jx = (joined.grid_x.values + 0.5) * 100.0
jy = (joined.grid_y.values + 0.5) * 100.0
node_grid = D["node_grid"]
ii, jj = np.nonzero(node_grid >= 0)
ids = node_grid[ii, jj]
gx_n = np.empty(ids.max() + 1, dtype=np.int64)
gy_n = np.empty(ids.max() + 1, dtype=np.int64)
gx0, gy0 = int(D["gx0"]), int(D["gy0"])
# orientation A: axis0 = x
gx_n[ids] = gx0 + ii; gy_n[ids] = gy0 + jj
J_nodes = D["J_nodes"].astype(np.int64)
if not (np.array_equal(gx_n[J_nodes], joined.grid_x.values)
        and np.array_equal(gy_n[J_nodes], joined.grid_y.values)):
    gx_n[ids] = gx0 + jj; gy_n[ids] = gy0 + ii   # orientation B: axis0 = y
    assert (np.array_equal(gx_n[J_nodes], joined.grid_x.values)
            and np.array_equal(gy_n[J_nodes], joined.grid_y.values)), \
        "node_grid orientation mismatch"
cand_nodes = D["cand_nodes"].astype(np.int64)
sx = (gx_n[cand_nodes] + 0.5) * 100.0
sy = (gy_n[cand_nodes] + 0.5) * 100.0
DE = np.hypot(sx[:, None] - jx[None, :], sy[:, None] - jy[None, :])
print(f"DE from snapped centres: {DE.shape}", flush=True)

E_BASE_E = 9.2365
E_BASE_L = 11.1582

def reach_of(total_cost, rho):
    return total_cost <= (1.0 - rho) * C.ETA_B

def save_reach(a, tag):
    save_npz(OUT / f"a_{tag}.npz", csr_matrix(a))

rows = []
# ---- blind, beta-free ----
C_eucl = DE * E_BASE_L + DE * E_BASE_E + edep + eterm0
C_energy = load("C_ld_energy") + load("C_e_energy") + edep + eterm0
zero = {r: load(f"C_ld_zero{r}") + load(f"C_e_zero{r}") + edep + eterm0
        for r in C.REGIMES_BD}
for rho in C.RHOS:
    a_eu = reach_of(C_eucl, rho); save_reach(a_eu, f"Eucl_rho{rho}")
    a_en = reach_of(C_energy, rho); save_reach(a_en, f"energy_rho{rho}")
    for r, Cz in zero.items():
        save_reach(reach_of(Cz, rho), f"zero{r}_rho{rho}")
    for phi in C.PHI_A0_SWEEP:
        a0 = 2 * DE * E_BASE_E <= phi * (1 - rho) * C.ETA_B
        save_reach(a0, f"A0phi{phi:g}_rho{rho}")
    rows.append({"beta": "-", "rho": rho,
                 "Eucl": int(a_eu.any(axis=0).sum()),
                 "energy": int(a_en.any(axis=0).sum())})

# ---- sigma-aware ----
uni = {}
nest_fail = 0
for beta in C.BETAS:
    Ctot = {}
    for r in ["High", "Medium", "Low"]:
        Ctot[r] = (load(f"C_ld_sigma{r}_b{beta}") + load(f"C_e_sigma{r}_b{beta}")
                   + edep + eterm[None, :])
    for rho in C.RHOS:
        a = {r: reach_of(np.nan_to_num(Ctot[r], nan=np.inf, posinf=np.inf), rho)
             for r in Ctot}
        for r in Ctot:
            save_reach(a[r], f"sigma{r}_b{beta}_rho{rho}")
        a_en = reach_of(C_energy, rho)
        a_eu = reach_of(C_eucl, rho)
        ok = (bool((a["Low"] <= a["Medium"]).all())
              and bool((a["Medium"] <= a["High"]).all())
              and bool((a["High"] <= a_en).all())
              and bool((a_en <= a_eu).all()))
        if not ok:
            nest_fail += 1
            print(f"NEST FAIL beta={beta} rho={rho}", flush=True)
        uni[(beta, rho)] = {r: int(a[r].any(axis=0).sum()) for r in a}
        print(f"beta={beta} rho={rho}: universes "
              f"H={uni[(beta,rho)]['High']:,} M={uni[(beta,rho)]['Medium']:,} "
              f"L={uni[(beta,rho)]['Low']:,} nest_ok={ok}", flush=True)

# ---- Table 5 (BD version) ----
t5 = []
for beta in C.BETAS:
    for rho in C.RHOS:
        base = next(r for r in rows if r["rho"] == rho)
        t5.append({"beta": beta, "rho": rho, "Eucl": base["Eucl"],
                   "energy": base["energy"],
                   "sigma_High": uni[(beta, rho)]["High"],
                   "sigma_Medium": uni[(beta, rho)]["Medium"],
                   "sigma_Low": uni[(beta, rho)]["Low"]})
df = pd.DataFrame(t5)
df.to_csv(P / "r2_layer/table5_bd.csv", index=False)
print(df.to_string(index=False), flush=True)

# assertions
en_by_beta = df.groupby("rho")["energy"].nunique()
assert (en_by_beta == 1).all(), "Energy universe not beta-invariant!"
assert nest_fail == 0, f"{nest_fail} nesting failures"
print("ASSERT1 energy beta-invariant: PASS", flush=True)
print("ASSERT2 nesting 16/16: PASS", flush=True)
print("done", flush=True)
