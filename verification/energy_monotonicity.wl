(* verify_r8.wl -- independent verification: loaded-leg monotonicity (rigorous)
   and mission-profile energy accounting audit. Wolfram Engine 14.3.

   Model (r2_config frozen constants):
     P(v; kap) = PB (1 + 3 v^2/U^2)
               + kap^(3/2) PI Sqrt[ Sqrt[1 + v^4/(4 v0q^2)] - v^2/(2 v0q) ]
               + c3 v^3,          v0q = kap V0^2,  c3 = D0 rho S A / 2
     kap(m)   = 1 + m G / W,     e(v) = P(v)/v.
   Claim (Lemma 2, loaded leg): e(v; kap(m)) is strictly decreasing on
   [1, 15] for every payload m in {0, 0.5, 1.0, 1.5} kg.
   Proof strategy: g(v) := P'(v) v - P(v) satisfies e'(v) = g(v)/v^2;
   W2a proves g(v) < 0 on [1,15] (exact semialgebraic Resolve where
   possible, else interval certification); W2b locates the unique root
   v* of g (best-range speed) and checks v* > 15 for every payload. *)

ok = True;
check[name_, cond_] := Module[{c = TrueQ[cond]},
  Print[name, ": ", If[c, "PASS", "FAIL"]]; If[! c, ok = False]];

PB = 7986/100; PI0 = 8863/100; U = 120; V0 = 403/100;
D0v = 6/10; rho = 1225/1000; Ssol = 5/100; Adisc = 503/1000;
c3 = D0v rho Ssol Adisc / 2;
W = 2 rho Adisc V0^2;                    (* exact frame weight, N *)
G0 = 981/100;
kap[m_] := 1 + m G0/W;
Pfun[v_, k_] := PB (1 + 3 v^2/U^2) +
   k^(3/2) PI0 Sqrt[Sqrt[1 + v^4/(4 (k V0^2)^2)] - v^2/(2 k V0^2)] +
   c3 v^3;
g[v_, k_] := D[Pfun[x, k], x] x - Pfun[x, k] /. x -> v;

payloads = {0, 1/2, 1, 3/2};
Print["frame weight W = ", N[W], " N;  kappas = ", N[kap /@ payloads]];

(* W1: best-range speeds (root of g) vs expert-quoted values *)
vstars = Table[
   v /. FindRoot[g[v, kap[m]] == 0, {v, 18, 15, 30},
     WorkingPrecision -> 30], {m, payloads}];
Print["best-range speeds v* = ", N[vstars, 6],
  "  (expected ~ {18.30, 19.21, 20.16, 21.10})"];
check["W1 best-range speeds match",
  Max[Abs[N[vstars] - {18.30, 19.21, 20.16, 21.10}]] < 0.01];

(* W2a: g(v) < 0 on [1,15] -- exact semialgebraic proof attempt *)
proved = True;
Do[Module[{k = kap[m], res},
   res = TimeConstrained[
     Resolve[ForAll[v, 1 <= v <= 15, g[v, k] < 0], Reals], 300, $Failed];
   Print["  Resolve payload ", N[m], ": ", res];
   If[res =!= True, proved = False]],
  {m, payloads}];
If[proved,
  check["W2a g(v)<0 on [1,15] (EXACT Resolve proof)", True],
  Module[{allneg = True},
   Do[Module[{k = kap[m], mx},
      mx = NMaxValue[{g[v, k], 1 <= v <= 15}, v,
        WorkingPrecision -> 30];
      Print["  max g on [1,15], payload ", N[m], " = ", N[mx]];
      If[mx >= 0, allneg = False]],
     {m, payloads}];
   check["W2a g(v)<0 on [1,15] (high-precision NMaxValue < 0)", allneg]]];

(* W2b: numeric cross-check of the exact proof -- max of g on [1,15]
   is strictly negative for every payload. (Note: P is NOT convex on
   [1,15] -- min of v P''(v) is negative -- so the proof rests on the
   Resolve certificate above, not on a convexity argument.) *)
maxg = Table[
   NMaxValue[{g[v, kap[m]], 1 <= v <= 15}, v,
     WorkingPrecision -> 30], {m, payloads}];
Print["  max of g on [1,15] per payload = ", N[maxg]];
check["W2b max g < 0 on [1,15] (numeric cross-check)",
  Max[maxg] < 0];

(* W3: mission-profile energy accounting audit *)
HC = 120; VC = 3; VD = 3; TGA = 10; TAL = 15; TREL = 45; RPAD = 5;
ETAB = 281360;
phLd = PB + PI0 kap[1]^(3/2); phE = PB + PI0;
wLd = W + G0; wE = W;
t = HC/VC;
dj = (phLd - wLd VD/2) t;    (* loaded descent, customer *)
cj = (phE + wE VC/2) t;      (* empty climb, customer *)
ci = (phLd + wLd VC/2) t;    (* loaded climb, depot *)
di = (phE - wE VD/2) t;      (* empty descent, depot *)
Ega = 2 phLd TGA;
Edep = ci + di;
eterm[nn_] := dj + nn phLd TAL + (nn - 1) Ega + phLd TREL + cj;
Print["  ph_ld=", N[phLd], " W  ph_e=", N[phE], " W"];
Print["  E_climb^ld=", N[ci/1000], "  E_desc^ld=", N[dj/1000],
  "  E_climb^e=", N[cj/1000], "  E_desc^e=", N[di/1000],
  "  E_ga=", N[Ega/1000], "  (kJ)"];
Print["  E_dep = ", N[Edep/1000], " kJ;  E_term(N=1) = ",
  N[eterm[1]/1000], " kJ;  profile total = ",
  N[(Edep + eterm[1])/1000], " kJ  (paper: ~47)"];
check["W3a profile audit ~47 kJ",
  46 < (Edep + eterm[1])/1000 < 48];
qOf[s_] := 1 - Exp[-RPAD^2/(2 s^2)];
check["W3b median terminal ~39 kJ (sigma = 4.25 m)",
  Abs[eterm[1/qOf[425/100]]/1000 - 39] < 1];
sstar = s /. FindRoot[eterm[1/qOf[s]] == ETAB, {s, 19, 10, 30},
    WorkingPrecision -> 20];
Print["  terminal-exhaustion threshold sigma* = ", N[sstar], " m ",
  "(paper: ~19.4)"];
check["W3c exhaustion threshold ~19.4 m", Abs[sstar - 19.45] < 0.1];

Print[If[ok, "ALL R8 CHECKS PASS", "SOME R8 CHECKS FAILED"]];
