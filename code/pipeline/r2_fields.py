"""bd_fields.py -- R-1 step 2: loaded/empty dual cost fields on the full grid.

Per R-0 frozen config (bd_config.py):
  - Table 3 v2: z unified 1.0; Low (ell, tau) = (7.5, 1.5) -> sat_L = 6.0 m.
  - Loaded power curve (Decision V): kappa = 1 + m_p g / W;
    P_ind -> kappa^1.5 P_ind, V0 -> sqrt(kappa) V0; blade/parasite unchanged.
    Speed law v_max(sigma) unchanged (safety constraint, not power).
  - sigma-aware regimes x betas x {loaded, empty}: B2(i) removal at NEW sats.
  - Constant-unit fields (energy loaded/empty, per-regime sigma=0 baselines)
    derived from ONE length-Dijkstra (no removal): C = dist_len * unit.

Output: bd_layer/costmat/C_{leg}_{abl}[_b{beta}].npz  (605 x 17,899 float32)
env: BETAS colon-separated (one job per beta recommended); SKIP_BETAFREE=1
     to skip the length-derived constants (run them in exactly one job).
"""
import os
import sys
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, "/lustre/home/2406393544/sharefolder/proj3/r2_layer")
import r2_config as C

P = Path("/lustre/home/2406393544/sharefolder/proj3")
OUT = P / "r2_layer/costmat"
OUT.mkdir(parents=True, exist_ok=True)
BETAS = [float(b) for b in os.environ.get("BETAS", "1.0").split(":") if b]
SKIP_BETAFREE = os.environ.get("SKIP_BETAFREE", "0") == "1"
MP = C.M_P_MAIN

D = np.load(P / "c_layer/domain.npz")
F = np.load(P / "c_layer/sigma_fields.npz")
sigma = np.load(P / "r2_layer/field_r2.npz")["sigma_full"].astype(float)  # R2 denoised field
NV = len(sigma)
eu = D["edges_u"].astype(np.int64); ev = D["edges_v"].astype(np.int64)
el = D["edges_len"].astype(float)
J_nodes = D["J_nodes"].astype(np.int64)
cand_nodes = D["cand_nodes"].astype(np.int64)
print(f"|V|={NV:,} edges={len(eu):,} betas={BETAS} m_p={MP}", flush=True)

def P_inst(v, m_p):
    """Zeng TWC power with payload correction (Decision V)."""
    v = np.maximum(np.asarray(v, dtype=float), 1e-9)
    kap = 1.0 + m_p * C.G / C.W_FRAME
    blade = C.P_B * (1.0 + 3.0 * v**2 / C.U_TIP**2)
    v0sq = kap * C.V0_HOV**2
    induced = (kap**1.5) * C.P_IND * np.sqrt(
        np.sqrt(1.0 + v**4 / (4.0 * v0sq**2)) - v**2 / (2.0 * v0sq))
    parasite = 0.5 * C.D0 * C.RHO_AIR * C.S_SOL * C.A_DISC * v**3
    return blade + induced + parasite

def e_of_v(v, m_p):
    return P_inst(v, m_p) / np.maximum(np.asarray(v, dtype=float), 1e-9)

def v_max(sig, reg, kperp=C.KAPPA_PERP_MAIN):
    raw = (reg.w_corr - reg.z_eps * sig) / (kperp * reg.tau_react)
    return np.maximum(C.V_MIN, np.minimum(C.V_CRUISE, raw))

def E_phys(sig, reg, m_p):
    vm = v_max(sig, reg)
    on = (sig > reg.sigma_sensor).astype(float)
    return e_of_v(vm, m_p) + on * C.P_AUX / vm

def run_dijkstra(unit, removed):
    w = 0.5 * (unit[eu] + unit[ev]) * el
    ok = ~(removed[eu] | removed[ev])
    g = csr_matrix((np.concatenate([w[ok], w[ok]]),
                    (np.concatenate([eu[ok], ev[ok]]),
                     np.concatenate([ev[ok], eu[ok]]))), shape=(NV, NV))
    return dijkstra(g, directed=False, indices=cand_nodes)[:, J_nodes].astype(np.float32)

E_BASE_E = float(e_of_v(np.array([C.V_CRUISE]), 0.0)[0])   # 9.2365
E_BASE_L = float(e_of_v(np.array([C.V_CRUISE]), MP)[0])    # ~11.04
print(f"E_base empty={E_BASE_E:.4f} loaded={E_BASE_L:.4f} J/m "
      f"(+{100*(E_BASE_L/E_BASE_E-1):.1f}%)", flush=True)

if not SKIP_BETAFREE:
    dist_len = run_dijkstra(np.ones(NV), np.zeros(NV, dtype=bool))  # metres
    np.savez_compressed(OUT / "dist_len.npz", C=dist_len)
    for leg, mp in [("ld", MP), ("e", 0.0)]:
        np.savez_compressed(OUT / f"C_{leg}_energy.npz",
                            C=(dist_len * e_of_v(np.array([C.V_CRUISE]), mp)[0]
                               ).astype(np.float32))
        for rname, reg in C.REGIMES_BD.items():
            u0 = float(E_phys(np.array([0.0]), reg, mp)[0])
            np.savez_compressed(OUT / f"C_{leg}_zero{rname}.npz",
                                C=(dist_len * u0).astype(np.float32))
            print(f"[betafree] {leg} zero-{rname}: unit={u0:.3f} J/m", flush=True)
    print("[betafree] energy + per-regime sigma=0 fields derived", flush=True)

for beta in BETAS:
    sb = beta * sigma
    for rname, reg in C.REGIMES_BD.items():
        sat = reg.sigma_sat(C.KAPPA_PERP_MAIN)
        removed = sb > sat
        for leg, mp in [("ld", MP), ("e", 0.0)]:
            Cr = run_dijkstra(E_phys(sb, reg, mp), removed)
            np.savez_compressed(OUT / f"C_{leg}_sigma{rname}_b{beta}.npz", C=Cr)
            print(f"sigma-{rname} b={beta} {leg}: sat={sat:.2f} "
                  f"removed={removed.sum():,} finite={np.isfinite(Cr).mean():.4f}",
                  flush=True)

print("=== no-fly nesting (new sats) ===")
for beta in BETAS:
    sb = beta * sigma
    NL = sb > C.REGIMES_BD["Low"].sigma_sat(C.KAPPA_PERP_MAIN)
    NM = sb > C.REGIMES_BD["Medium"].sigma_sat(C.KAPPA_PERP_MAIN)
    NH = sb > C.REGIMES_BD["High"].sigma_sat(C.KAPPA_PERP_MAIN)
    ok = bool((NH <= NM).all() and (NM <= NL).all())
    print(f"beta={beta}: N_H<=N_M<=N_L {ok} |N_L|={NL.sum():,} "
          f"|N_M|={NM.sum():,} |N_H|={NH.sum():,}")
    assert ok
print("done", flush=True)
