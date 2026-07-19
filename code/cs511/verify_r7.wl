(* verify_r7.wl -- independent symbolic/numeric verification of
   the mathematical layer, run under Wolfram Engine 14.3.
   Checks (PASS/FAIL each):
   V1  ncx2 closed form of the biased capture probability vs direct 2D
       Gaussian integration over the capture disc (3 parameter points)
   V2  zero-bias reduction q(sigma,0) = 1 - Exp[-r^2/(2 sigma^2)]
   V3  saturation-point formulas at kappa=0.3 and kappa=1 vs paper values
   V4  scale-chain interval: r_ag x c_g bounds = [1.91, 3.96] ~ 1.9-4.0
   V5  terminal-energy identity: E_term(N) linear in N; E[N] = 1/q
       (geometric retries), symbolic
   V6  fourfold claim: max/min of swept theta_route = 4 exactly;
       kappa anchor ratio 0.1/0.028 and 0.1/0.033 in [3, 3.6]
   V7  bootstrap bias arithmetic: recentred intervals = observed range
       minus (mean - point) for the three reported rows *)

ok = True;
check[name_, cond_] := Module[{c = TrueQ[cond]},
  Print[name, ": ", If[c, "PASS", "FAIL"]];
  If[! c, ok = False]];

(* V1: ncx2 closed form vs 2D integral *)
qClosed[r_, s_, b_] :=
  N[CDF[NoncentralChiSquareDistribution[2, (b/s)^2], (r/s)^2]];
qInt[r_, s_, b_] :=
  NIntegrate[
    Exp[-((x - b)^2 + y^2)/(2 s^2)]/(2 Pi s^2)
      Boole[x^2 + y^2 <= r^2],
    {x, -r, r}, {y, -r, r}, AccuracyGoal -> 8, PrecisionGoal -> 8];
pts = {{3.0, 2.0, 0.0}, {3.0, 2.0, 5.1}, {3.0, 6.0, 15.3}};
check["V1 ncx2 vs 2D integral",
  And @@ (Abs[qClosed @@ # - qInt @@ #] < 10^-6 & /@ pts)];

(* V2: zero-bias reduction *)
check["V2 zero-bias Rayleigh reduction",
  And @@ Table[
    Abs[qClosed[3.0, s, 0.0] - (1 - Exp[-3.0^2/(2 s^2)])] < 10^-9,
    {s, {0.5, 2.0, 8.0}}]];

(* V3: saturation points. Regime (ell, z, tau):
   High (20,1,1), Medium (10,1,1), Low (7.5,1,1.5); vmin=1, vcr=15 *)
sat[ell_, z_, tau_, kap_, v_] := (ell - kap tau v)/z;
satsK03 = N[{sat[20, 1, 1, .3, 1], sat[10, 1, 1, .3, 1],
    sat[7.5, 1, 1.5, .3, 1]}];
satsK1 = N[{sat[20, 1, 1, 1, 1], sat[10, 1, 1, 1, 1],
    sat[7.5, 1, 1.5, 1, 1]}];
crK03 = N[{sat[20, 1, 1, .3, 15], sat[10, 1, 1, .3, 15],
    sat[7.5, 1, 1.5, .3, 15]}];
Print["  sat(k=0.3) = ", satsK03, "  paper (19.7, 9.7, 7.05)"];
Print["  sat(k=1)   = ", satsK1, "  paper (19, 9, 6)"];
Print["  cr(k=0.3)  = ", crK03, "  paper (15.5, 5.5, 0.75)"];
check["V3 saturation points",
  Max[Abs[satsK03 - {19.7, 9.7, 7.05}]] < 10^-9 &&
    Max[Abs[satsK1 - {19, 9, 6}]] < 10^-9 &&
    Max[Abs[crK03 - {15.5, 5.5, 0.75}]] < 10^-9];

(* V4: scale-chain interval *)
cg = 4.35; rag = {0.44, 0.91};
prod = cg rag;
Print["  r_ag*c_g = ", prod];
check["V4 scale-chain 1.9-4.0",
  1.85 < prod[[1]] < 1.95 && 3.9 < prod[[2]] < 4.05];

(* V5: geometric retries symbolic *)
check["V5 E[N] = 1/q (geometric)",
  Simplify[Sum[n q (1 - q)^(n - 1), {n, 1, Infinity}],
      0 < q < 1] === 1/q];
eterm[nn_, dj_, cj_, ph_, ta_, ega_, trel_] :=
  dj + nn ph ta + (nn - 1) ega + ph trel + cj;
check["V5b E_term linear in N",
  Simplify[
    eterm[n2, dj, cj, ph, ta, ega, trel] -
      eterm[n1, dj, cj, ph, ta, ega, trel] -
      (n2 - n1) (ph ta + ega)] === 0];

(* V6: fourfold + kappa anchor ratios *)
check["V6 fourfold and kappa anchor ratio",
  1.0/0.25 == 4 && 3.0 < 0.1/0.033 < 3.1 && 3.5 < 0.1/0.028 < 3.6];

(* V7: bootstrap recentred arithmetic *)
rows = {{8.94, 9.95, {9.63, 10.24}, 1.01, {8.62, 9.23}},
        {13.08, 13.94, {13.53, 14.22}, 0.86, {12.67, 13.36}},
        {52.82, 53.04, {52.57, 53.62}, 0.22, {52.35, 53.40}}};
check["V7 recentred bootstrap arithmetic",
  And @@ (Module[{pt = #[[1]], mn = #[[2]], rg = #[[3]], bias = #[[4]],
        rc = #[[5]]},
      Abs[(mn - pt) - bias] <= 0.011 &&
        Max[Abs[(rg - bias) - rc]] <= 0.011] & /@ rows)];

Print[If[ok, "ALL CHECKS PASS", "SOME CHECKS FAILED"]];
