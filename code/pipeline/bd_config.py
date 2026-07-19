"""bd_config.py -- R-0 FROZEN configuration for the B+D merged rerun (R-1..R-10).

Frozen 2026-07-07 per user sign-off on R0_decision_signoff.pdf (all items
approved). Any change to this file after R-1 starts violates the freeze.

Decision I   : E_term (Rayleigh hit-retry) in the headline criterion, all
               ablations; blind ablations pay E_term(0)+E_dep; E_term eats the
               UNSCALED ground field (beta- and regime-independent).
Decision II  : rho form kept; rho grid {0,.2,.5,.8}; 0.8 = stress wording.
Decision III : Medium = reference regime; Low = stress envelope.
Decision IV  : z unified 1.0; Low recalibrated (ell_L, tau_L) = (7.5, 1.5)
               -> sat_L = 6.0 m; kappa_perp sweep {0.1, 0.3, 1.0} in R-4.
Decision V   : loaded outbound (m_p = 1.0 kg) / empty return; payload sweep
               {0, 0.5, 1.0, 1.5} kg in R-8.
Decision VI  : no altitude-cap, no field campaign this round.
Decision VII : D5 as remark; C_plan one sensitivity run (R-6).
Confirm A    : LOADED profile accounting (47 kJ) everywhere; the empty 37 kJ
               figure appears only as the m_p = 0 sensitivity row.
Confirm B    : Table 5 caption must state that E_term is deliberately not
               scaled by beta (sigma-aware universes need not converge to the
               Energy universe as beta -> 0).
"""
from dataclasses import dataclass

# ---- power model (Zeng, Xu, Zhang 2019, IEEE TWC 18(4)) -- unchanged ----
P_B, P_IND = 79.86, 88.63
U_TIP, V0_HOV, D0 = 120.0, 4.03, 0.6
RHO_AIR, S_SOL, A_DISC = 1.225, 0.05, 0.503
V_CRUISE, V_MIN = 15.0, 1.0
G = 9.81
W_FRAME = 2.0 * RHO_AIR * A_DISC * V0_HOV**2          # ~20.0 N (self-consistency)

# ---- Decision V: payload ----
M_P_MAIN = 1.0                                          # kg, headline
M_P_SWEEP = [0.0, 0.5, 1.0, 1.5]                        # R-8

# ---- Decision I: mission profile / terminal (LOADED accounting, Confirm A) ----
H_C = 120.0            # cruise altitude (m)
V_CLIMB = 3.0          # m/s
V_DESC = 3.0           # m/s
T_ALIGN = 15.0         # s per alignment attempt
T_REL = 45.0           # s hover-and-release
T_GA = 10.0            # s go-around leg (E_ga = 2 * P_hover_ld * T_GA)
R_PAD_MAIN = 5.0       # m capture radius, headline
R_PAD_SWEEP = [3.0, 5.0, 8.0]           # R-7
T_ALIGN_SWEEP = [10.0, 15.0, 30.0]      # R-7
H_C_SWEEP = [60.0, 120.0]               # R-7

# ---- Decision IV: Table 3 v2 (z unified, Low recalibrated) ----
@dataclass(frozen=True)
class RegimeBD:
    name: str
    w_corr: float
    z_eps: float
    tau_react: float
    sigma_sensor: float
    sigma_detour: float
    def sigma_sat(self, kappa_perp: float = 1.0) -> float:
        return (self.w_corr - kappa_perp * self.tau_react * V_MIN) / self.z_eps

REGIMES_BD = {
    "Low":    RegimeBD("Low",    w_corr=7.5,  z_eps=1.0, tau_react=1.5,
                       sigma_sensor=5.0,  sigma_detour=5.0),   # sat = 6.0 m
    "Medium": RegimeBD("Medium", w_corr=10.0, z_eps=1.0, tau_react=1.0,
                       sigma_sensor=10.0, sigma_detour=8.0),   # sat = 9.0 m
    "High":   RegimeBD("High",   w_corr=20.0, z_eps=1.0, tau_react=1.0,
                       sigma_sensor=15.0, sigma_detour=12.0),  # sat = 19.0 m
}
KAPPA_PERP_MAIN = 1.0
KAPPA_PERP_SWEEP = [0.1, 0.3, 1.0]      # R-4

# ---- budget / reserve ----
ETA_B = 281_360.0
RHOS = [0.0, 0.2, 0.5, 0.8]
BETAS = [0.25, 0.5, 0.75, 1.0]
P_AUX = 15.0

# ---- A0 practitioner baseline (B3) ----
PHI_A0_MAIN = 0.7
PHI_A0_SWEEP = [0.6, 0.7]

# ---- MILP grid ----
K_GRID = [1, 5, 10, 20, 30, 50, 75, 100, 150, 300, 605]
WEIGHTS = ["pop", "ev", "uniform"]      # pop = headline
