"""bd_eterm.py -- R-1 step 1: terminal/profile energy vectors (Decision I).

Builds, from the UNSCALED GK worst-direction field on the 17,899 J cells:
  E_term(sigma_j)  (main case + R-7 scan variants)
  E_term(0), E_dep (constants; blind ablations pay these)
plus the E-B accounting table (profile stage energies) printed to stdout.

Everything is closed-form; no routing. Output: bd_layer/eterm.npz
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, "/lustre/home/2406393544/sharefolder/proj3/r2_layer")
import r2_config as C

P = Path("/lustre/home/2406393544/sharefolder/proj3")
OUT = P / "r2_layer"
OUT.mkdir(exist_ok=True)

def hover_power(m_p):
    kap = 1.0 + m_p * C.G / C.W_FRAME
    return C.P_B + C.P_IND * kap**1.5

def stage_energies(m_p, h_c, v_c, v_d):
    """Deterministic profile stage energies (J) for payload m_p."""
    ph_ld = hover_power(m_p)
    ph_e = hover_power(0.0)
    w_ld = C.W_FRAME + m_p * C.G
    w_e = C.W_FRAME
    t_c, t_d = h_c / v_c, h_c / v_d
    return {
        "climb_i_ld": (ph_ld + w_ld * v_c / 2) * t_c,
        "desc_j_ld":  (ph_ld - w_ld * v_d / 2) * t_d,
        "climb_j_e":  (ph_e + w_e * v_c / 2) * t_c,
        "desc_i_e":   (ph_e - w_e * v_d / 2) * t_d,
        "ph_ld": ph_ld,
    }

def eterm_vec(sigma, m_p, r_pad, t_align, h_c):
    """E_term(sigma) per Decision I (loaded accounting, Confirm A)."""
    st = stage_energies(m_p, h_c, C.V_CLIMB, C.V_DESC)
    ph_ld = st["ph_ld"]
    e_ga = 2.0 * ph_ld * C.T_GA
    sigma = np.asarray(sigma, dtype=float)
    with np.errstate(divide="ignore"):
        q = 1.0 - np.exp(-r_pad**2 / (2.0 * np.maximum(sigma, 1e-12)**2))
    q = np.where(sigma <= 1e-12, 1.0, q)
    N = 1.0 / q
    return (st["desc_j_ld"] + N * ph_ld * t_align + (N - 1.0) * e_ga
            + ph_ld * C.T_REL + st["climb_j_e"])

def edep(m_p, h_c):
    st = stage_energies(m_p, h_c, C.V_CLIMB, C.V_DESC)
    return st["climb_i_ld"] + st["desc_i_e"]

# ---- J-cell unscaled field (GK worst-direction) ----
joined = pd.read_parquet(P / "gk_run/joined_wuhan_gk.parquet")
sig_j = np.load(P / "r2_layer/field_r2.npz")["sig_j"].astype(float)  # R2 denoised J-cell field
assert len(sig_j) == 17_899, len(sig_j)

out = {}
# main case
out["eterm_main"] = eterm_vec(sig_j, C.M_P_MAIN, C.R_PAD_MAIN, C.T_ALIGN, C.H_C)
out["eterm0_main"] = np.array([eterm_vec(np.array([0.0]), C.M_P_MAIN,
                                         C.R_PAD_MAIN, C.T_ALIGN, C.H_C)[0]])
out["edep_main"] = np.array([edep(C.M_P_MAIN, C.H_C)])
# R-7 scans (terminal params; payload fixed at main)
for rp in C.R_PAD_SWEEP:
    for ta in C.T_ALIGN_SWEEP:
        for hc in C.H_C_SWEEP:
            tag = f"rp{rp:g}_ta{ta:g}_hc{hc:g}"
            out[f"eterm_{tag}"] = eterm_vec(sig_j, C.M_P_MAIN, rp, ta, hc)
            out[f"eterm0_{tag}"] = np.array([eterm_vec(np.array([0.0]),
                                             C.M_P_MAIN, rp, ta, hc)[0]])
            out[f"edep_{tag}"] = np.array([edep(C.M_P_MAIN, hc)])
# R-8 payload variants (terminal params at main)
for mp in C.M_P_SWEEP:
    tag = f"mp{mp:g}"
    out[f"eterm_{tag}"] = eterm_vec(sig_j, mp, C.R_PAD_MAIN, C.T_ALIGN, C.H_C)
    out[f"eterm0_{tag}"] = np.array([eterm_vec(np.array([0.0]), mp,
                                     C.R_PAD_MAIN, C.T_ALIGN, C.H_C)[0]])
    out[f"edep_{tag}"] = np.array([edep(mp, C.H_C)])

np.savez_compressed(OUT / "eterm.npz", **out)

# ---- E-B accounting table + headline sanity ----
st = stage_energies(C.M_P_MAIN, C.H_C, C.V_CLIMB, C.V_DESC)
ph_ld = st["ph_ld"]
acct = {
    "climb@i loaded (kJ)": st["climb_i_ld"] / 1e3,
    "desc@j loaded (kJ)": st["desc_j_ld"] / 1e3,
    "align 1x loaded (kJ)": ph_ld * C.T_ALIGN / 1e3,
    "release loaded (kJ)": ph_ld * C.T_REL / 1e3,
    "climb@j empty (kJ)": st["climb_j_e"] / 1e3,
    "desc@i empty (kJ)": st["desc_i_e"] / 1e3,
}
acct["total sigma=0 (kJ)"] = sum(acct.values())
acct["share of etaB (%)"] = 100 * acct["total sigma=0 (kJ)"] * 1e3 / C.ETA_B
report = {
    "accounting_loaded_main": {k: round(v, 2) for k, v in acct.items()},
    "E_term(0)+E_dep (kJ)": round((out["eterm0_main"][0] + out["edep_main"][0]) / 1e3, 2),
    "P_hover_ld (W)": round(ph_ld, 1),
    "W_frame (N)": round(C.W_FRAME, 2),
    "field quantiles sigma (m)": {q: round(float(np.quantile(sig_j, q)), 2)
                                  for q in [0.25, 0.5, 0.75, 0.95]},
    "E_term quantiles (kJ)": {q: round(float(np.quantile(out["eterm_main"], q)) / 1e3, 1)
                              for q in [0.25, 0.5, 0.75, 0.95]},
    "cells where E_term alone > etaB": int((out["eterm_main"] > C.ETA_B).sum()),
    "cells where E_dep+E_term > 0.5 etaB": int(
        ((out["eterm_main"] + out["edep_main"][0]) > 0.5 * C.ETA_B).sum()),
}
print(json.dumps(report, indent=2), flush=True)
(OUT / "eterm_report.json").write_text(json.dumps(report, indent=2))
print("done", flush=True)
