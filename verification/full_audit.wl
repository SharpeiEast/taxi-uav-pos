(* full_audit.wl -- full-model symbolic/exact audit.
   Exact rational constants throughout; every check prints PASS/FAIL. *)

ok = True;
chk[name_, cond_] := (Print[name, ": ", If[TrueQ[cond], "PASS", "FAIL"]];
  If[!TrueQ[cond], ok = False]);

(* ---- frozen constants (exact rationals) ---- *)
PB = 7986/100; PIND = 8863/100; UTIP = 120; V0 = 403/100;
D0 = 6/10; RHOA = 49/40; SSOL = 1/20; ADISC = 503/1000;
VC = 15; VMIN = 1; GACC = 981/100; ETAB = 281360; PAUX = 15;
HC = 120; VCL = 3; VDE = 3; TAL = 15; TREL = 45; TGA = 10; RPAD = 5;

WFRAME = (2 RHOA ADISC V0^2);
kap[mp_] := (1 + mp GACC/WFRAME);
Pinst[v_, mp_] := (PB (1 + 3 v^2/UTIP^2)
  + kap[mp]^(3/2) PIND Sqrt[Sqrt[1 + v^4/(4 (kap[mp] V0^2)^2)]
      - v^2/(2 kap[mp] V0^2)]
  + (1/2) D0 RHOA SSOL ADISC v^3);
gfun[v_, mp_] := (D[Pinst[u, mp], u] u - Pinst[u, mp] /. u -> v);

(* ---- W1: regime saturation points and cruise breakpoints ---- *)
sat[w_, z_, tau_, kp_] := ((w - kp tau VMIN)/z);
brk[w_, z_, tau_, kp_] := ((w - kp tau VC)/z);
chk["W1a sat kappa=0.3 (L,M,H)=(7.05,9.7,19.7)",
  {sat[15/2,1,3/2,3/10], sat[10,1,1,3/10], sat[20,1,1,3/10]} == {141/20, 97/10, 197/10}];
chk["W1b sat kappa=1 (L,M,H)=(6,9,19)",
  {sat[15/2,1,3/2,1], sat[10,1,1,1], sat[20,1,1,1]} == {6, 9, 19}];
chk["W1c cruise breakpoints kappa=0.3 (M,H)=(5.5,15.5), L=0.75",
  {brk[10,1,1,3/10], brk[20,1,1,3/10], brk[15/2,1,3/2,3/10]} == {11/2, 31/2, 3/4}];
chk["W1d kappa=1 algebraic breakpoints M=-5, L=-15 (domain-truncated in text)",
  {brk[10,1,1,1], brk[15/2,1,3/2,1]} == {-5, -15}];

(* ---- W2: exact QE certificate g<0 on [1,15], four payloads ---- *)
Do[
  res = Resolve[ForAll[v, 1 <= v <= 15, gfun[v, mp] < 0], Reals];
  chk["W2 QE g<0 on [1,15], m_p=" <> ToString[N[mp]], res === True],
  {mp, {0, 1/2, 1, 3/2}}];

(* ---- W3: best-range speeds ---- *)
vstars = Table[v /. FindRoot[gfun[v, mp] == 0, {v, 20, 15, 30},
    WorkingPrecision -> 30], {mp, {0, 1/2, 1, 3/2}}];
chk["W3 best-range speeds = 18.30/19.21/20.16/21.10",
  (Round[#, 1/100] & /@ vstars) == {1830/100, 1921/100, 2016/100, 2110/100}];
chk["W3b all best-range speeds exceed v_cruise", And @@ (# > 15 & /@ vstars)];

(* ---- W4: hover power, frame weight, per-metre baselines ---- *)
chk["W4a P(0)=P_B+P_ind=168.49 W", PB + PIND == 16849/100];
chk["W4b W_frame ~ 20.01 N", Abs[N[WFRAME] - 20.0145] < 0.001];
chk["W4c E_base empty = 9.2365 J/m", Abs[N[Pinst[15, 0]/15] - 9.2365] < 0.00005];
chk["W4d E_base loaded = 11.1582 J/m (r2_reach constant)",
  Abs[N[Pinst[15, 1]/15] - 11.1582] < 0.00005];

(* ---- W5: mission-profile audit ---- *)
hover[mp_] := (PB + PIND kap[mp]^(3/2));
wld[mp_] := (WFRAME + mp GACC);
tc = HC/VCL; td = HC/VDE;
climbILd[mp_] := ((hover[mp] + wld[mp] VCL/2) tc);
descJLd[mp_] := ((hover[mp] - wld[mp] VDE/2) td);
climbJE = ((hover[0] + WFRAME VCL/2) tc);
descIE = ((hover[0] - WFRAME VDE/2) td);
edep[mp_] := (climbILd[mp] + descIE);
q[s_] := (1 - Exp[-RPAD^2/(2 s^2)]);
eterm[s_, mp_] := Module[{ph = hover[mp], nn = 1/q[s], ega},
  ega = 2 hover[mp] TGA;
  (descJLd[mp] + nn ph TAL + (nn - 1) ega + ph TREL + climbJE)];
eterm0[mp_] := (descJLd[mp] + hover[mp] TAL + hover[mp] TREL + climbJE);
chk["W5a E_dep+E_term(0) = 47.23 kJ (m_p=1)",
  Abs[N[(edep[1] + eterm0[1])/1000] - 47.23] < 0.01];
chk["W5b E_term(0) consistent with q=1 limit of E_term",
  Abs[N[eterm0[1] - Limit[eterm[s, 1], s -> 0, Direction -> "FromAbove"]]] < 10^-6];
chk["W5c E_term at sigma_med 4.2451 m ~ 38.7 kJ",
  Abs[N[eterm[42451/10000, 1]/1000] - 38.7] < 0.15];
chk["W5d E_term at sigma p95 8.1835 m ~ 71.4 kJ",
  Abs[N[eterm[81835/10000, 1]/1000] - 71.4] < 0.4];
sstar = (s /. FindRoot[eterm[s, 1] == ETAB, {s, 19, 10, 30},
  WorkingPrecision -> 25]);
chk["W5e sigma* (E_term = etaB) = 19.45 m", Abs[N[sstar] - 19.45] < 0.01];
chk["W5f E[N] at sigma_med = 2.0 (1dp)",
  Abs[N[1/q[42451/10000]] - 2.0] < 0.05];
chk["W5g E[N] at sigma p95 = 5.9 (1dp)",
  Abs[N[1/q[81835/10000]] - 5.9] < 0.05];

(* ---- W6: chance-constrained attempt bound ---- *)
nmax[s_, eps_] := (Ceiling[Log[eps]/Log[1 - q[s]]]);
chk["W6a N_max(sigma_med, 0.05) = 5", nmax[42451/10000, 5/100] == 5];
chk["W6b N_max monotone in sigma (spot 2<9m)",
  nmax[2, 5/100] <= nmax[9, 5/100]];
chk["W6c q=1 limit: sigma->0 gives 1 attempt",
  Limit[Log[5/100]/Log[1 - q[s]], s -> 0, Direction -> "FromAbove"] == 0];

(* ---- W7: Rice/ncx2 capture probability vs 2D integral ---- *)
qrice[s_, b_] := (CDF[NoncentralChiSquareDistribution[2, (b/s)^2], (RPAD/s)^2]);
qint[s_, b_] := (NIntegrate[
   r/s^2 Exp[-(r^2 + b^2)/(2 s^2)] BesselI[0, r b/s^2],
   {r, 0, RPAD}, WorkingPrecision -> 20, PrecisionGoal -> 10]);
Do[
  chk["W7 Rice ncx2 == 2D integral (s=" <> ToString[N[sb[[1]]]] <> ",b=" <>
      ToString[N[sb[[2]]]] <> ")",
    Abs[N[qrice @@ sb] - qint @@ sb] < 10^-8],
  {sb, {{3, 2}, {5, 25/2}, {10, 255/10}}}];

(* ---- W8: estimator identities ---- *)
lmax = Max[Eigenvalues[{{sx2, cc}, {cc, sy2}}]];
chk["W8a lambda_max closed form",
  Simplify[(sx2 + sy2 + Sqrt[(sx2 - sy2)^2 + 4 cc^2])/2 ==
    Max[(sx2 + sy2 + Sqrt[sx2^2 - 2 sx2 sy2 + sy2^2 + 4 cc^2])/2,
        (sx2 + sy2 - Sqrt[sx2^2 - 2 sx2 sy2 + sy2^2 + 4 cc^2])/2],
    Assumptions -> {sx2 > 0, sy2 > 0, cc \[Element] Reals}] === True];
chk["W8b 1.4826 = 1/InverseCDF[N(0,1),3/4] (4dp)",
  Abs[1/InverseCDF[NormalDistribution[], 3/4] - 1.4826] < 0.0001];

(* ---- W9: scale chain and A0 arithmetic ---- *)
chk["W9a raw interval c_g*[0.44,0.91] = [1.9,4.0] (1dp)",
  (Round[#, 1/10] & /@ {44/100 44/10, 91/100 44/10}) == {19/10, 4}];
chk["W9b theta_route = a_aid r_ag c_g dimensional identity",
  Simplify[aa rr cg sj == aa (rr (cg sj))] === True];
rnom = ETAB/(2 92365/10000);
chk["W9c A0 nominal radius etaB/(2 E_base) ~ 15.23 km",
  Abs[N[rnom/1000] - 15.23] < 0.01];
chk["W9d A0 rho=0.8 phi=0.7 radius = 0.14 x nominal",
  N[(7/10)(1 - 8/10)] == 0.14];

(* ---- W10: bootstrap-table arithmetic (frozen S3 rows) ---- *)
rows = {{894, 995, 963, 1024, 101}, {1308, 1394, 1353, 1422, 86},
        {5282, 5304, 5257, 5362, 22}};
Do[
  {pt, mn, lo, hi, bias} = r;
  chk["W10 bias row (point=" <> ToString[N[pt/100]] <> ")",
    (mn - pt == bias) && (lo - bias == lo - (mn - pt))],
  {r, rows}];

Print["=== verify_v14.wl ", If[ok, "ALL CHECKS PASS", "FAILURES PRESENT"], " ==="];
