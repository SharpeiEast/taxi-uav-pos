# verify_v14.jl -- independent numeric audit (Julia; no numpy/scipy/Wolfram).
# Part 1: physics/terminal/probability layer recomputed from frozen constants.
# Part 2: structural properties checked directly on the frozen result CSVs.
using Distributions, QuadGK, Roots, SpecialFunctions, DelimitedFiles, Printf

const PASSFAIL = Ref(true)
function chk(name, cond)
    @printf("%-68s %s\n", name, cond ? "PASS" : "FAIL")
    cond || (PASSFAIL[] = false)
end

# ---- frozen constants ----
const PB, PIND = 79.86, 88.63
const UTIP, V0, D0 = 120.0, 4.03, 0.6
const RHOA, SSOL, ADISC = 1.225, 0.05, 0.503
const VC, VMIN, G = 15.0, 1.0, 9.81
const ETAB, PAUX = 281_360.0, 15.0
const HC, VCL, VDE = 120.0, 3.0, 3.0
const TAL, TREL, TGA, RPAD = 15.0, 45.0, 10.0, 5.0
const WFRAME = 2 * RHOA * ADISC * V0^2

kap(mp) = 1 + mp * G / WFRAME
function Pinst(v, mp)
    k = kap(mp); v0sq = k * V0^2
    (PB * (1 + 3v^2 / UTIP^2)
     + k^1.5 * PIND * sqrt(sqrt(1 + v^4 / (4v0sq^2)) - v^2 / (2v0sq))
     + 0.5 * D0 * RHOA * SSOL * ADISC * v^3)
end
gfun(v, mp) = let h = 1e-6
    (Pinst(v + h, mp) - Pinst(v - h, mp)) / (2h) * v - Pinst(v, mp)
end

# ---- J1: baselines ----
chk("J1a P(0) = 168.49 W", abs(Pinst(1e-12, 0.0) - 168.49) < 0.01)
chk("J1b E_base empty 9.2365 J/m", abs(Pinst(15.0, 0.0)/15 - 9.2365) < 5e-5)
chk("J1c E_base loaded 11.1582 J/m", abs(Pinst(15.0, 1.0)/15 - 11.1582) < 5e-5)
chk("J1d W_frame 20.01 N", abs(WFRAME - 20.0145) < 1e-3)

# ---- J2: monotonicity g<0 on [1,15] (fine grid) + best-range speeds ----
for (mp, vtgt) in [(0.0,18.30), (0.5,19.21), (1.0,20.16), (1.5,21.10)]
    gmax = maximum(gfun(v, mp) for v in range(1.0, 15.0, length=20001))
    chk(@sprintf("J2 g(v)<0 on [1,15], m_p=%.1f (max g=%.3f)", mp, gmax), gmax < 0)
    vstar = find_zero(v -> gfun(v, mp), (15.0, 30.0), Bisection())
    chk(@sprintf("J2 best-range v* m_p=%.1f = %.2f", mp, vtgt),
        abs(vstar - vtgt) < 0.005)
end

# ---- J3: regime saturation / breakpoints ----
sat(w, z, tau, kp) = (w - kp * tau * VMIN) / z
chk("J3a sat kappa=0.3 = (7.05, 9.7, 19.7)",
    isapprox.([sat(7.5,1,1.5,0.3), sat(10,1,1,0.3), sat(20,1,1,0.3)],
              [7.05, 9.7, 19.7], atol=1e-12) |> all)
chk("J3b sat kappa=1 = (6, 9, 19)",
    isapprox.([sat(7.5,1,1.5,1), sat(10,1,1,1), sat(20,1,1,1)],
              [6.0, 9.0, 19.0], atol=1e-12) |> all)

# ---- J4: mission profile / terminal audit ----
hover(mp) = PB + PIND * kap(mp)^1.5
wld(mp) = WFRAME + mp * G
tc, td = HC/VCL, HC/VDE
climb_i_ld(mp) = (hover(mp) + wld(mp)*VCL/2) * tc
desc_j_ld(mp)  = (hover(mp) - wld(mp)*VDE/2) * td
climb_j_e      = (hover(0.0) + WFRAME*VCL/2) * tc
desc_i_e       = (hover(0.0) - WFRAME*VDE/2) * td
edep(mp) = climb_i_ld(mp) + desc_i_e
qcap(s) = 1 - exp(-RPAD^2 / (2s^2))
function eterm(s, mp)
    ph = hover(mp); n = 1/qcap(s); ega = 2ph * TGA
    desc_j_ld(mp) + n*ph*TAL + (n-1)*ega + ph*TREL + climb_j_e
end
eterm0(mp) = desc_j_ld(mp) + hover(mp)*TAL + hover(mp)*TREL + climb_j_e
chk("J4a E_dep+E_term(0) = 47.23 kJ", abs((edep(1.0)+eterm0(1.0))/1e3 - 47.23) < 0.01)
chk("J4b E_term(sig_med 4.2451) ~ 38.7 kJ", abs(eterm(4.2451,1.0)/1e3 - 38.7) < 0.15)
chk("J4c E_term(sig_p95 8.1835) ~ 71.4 kJ", abs(eterm(8.1835,1.0)/1e3 - 71.4) < 0.4)
sstar = find_zero(s -> eterm(s,1.0) - ETAB, (10.0, 30.0), Bisection())
chk(@sprintf("J4d sigma* = 19.45 m (got %.3f)", sstar), abs(sstar - 19.45) < 0.01)
chk("J4e E[N](sig_med) = 2.0", abs(1/qcap(4.2451) - 2.0) < 0.05)
chk("J4f E[N](sig_p95) = 5.9", abs(1/qcap(8.1835) - 5.9) < 0.05)

# ---- J5: Rayleigh capture == polar integral; Rice ncx2 == Bessel integral ----
for s in (2.0, 5.0, 12.0)
    qi, _ = quadgk(r -> r/s^2 * exp(-r^2/(2s^2)), 0, RPAD, rtol=1e-12)
    chk(@sprintf("J5a Rayleigh closed form == integral (s=%g)", s),
        abs(qi - qcap(s)) < 1e-10)
end
for (s, b) in ((3.0,2.0), (5.0,12.5), (10.0,25.5))
    qr = cdf(NoncentralChisq(2, (b/s)^2), (RPAD/s)^2)
    qi, _ = quadgk(r -> r/s^2 * exp(-(r^2+b^2)/(2s^2)) * besseli(0, r*b/s^2),
                   0, RPAD, rtol=1e-12)
    chk(@sprintf("J5b Rice ncx2 == Bessel integral (s=%g,b=%g)", s, b),
        abs(qr - qi) < 1e-9)
end

# ---- J6: chance bound ----
nmax(s, eps) = ceil(Int, log(eps)/log(1 - qcap(s)))
chk("J6a N_max(sig_med, .05) = 5", nmax(4.2451, 0.05) == 5)
chk("J6b N_max nondecr in sigma", nmax(2.0,0.05) <= nmax(9.0,0.05) <= nmax(15.0,0.05))

# ---- Part 2: frozen result CSVs ----
const CS = "/lustre/home/2406393544/sharefolder/proj3/cs511"
function readcsv(f)
    d, h = readdlm(f, ',', header=true)
    d, vec(h)
end

# J7: Lemma-2 nesting + K/rho monotonicity on every frozen MILP row
d = vcat([readcsv(joinpath(CS, "milp_$(w)_rho$(r).csv"))[1]
          for w in ("pop","ev","uniform"), r in ("0.0","0.2","0.5","0.8")]...)
wt, rh, ab, KK, cov = d[:,1], float.(d[:,2]), d[:,3], Int.(d[:,4]), float.(d[:,5])
order = ["Eucl","energy","High","Medium","Low"]  # raw CSV ablation names
nest_viol = 0; kmono_viol = 0; rmono_viol = 0; nest_groups = 0
for w in unique(wt), r in unique(rh), k in unique(KK)
    v = [cov[(wt.==w) .& (rh.==r) .& (KK.==k) .& (ab.==a)] for a in order]
    all(length.(v) .== 1) || continue
    c = first.(v)
    global nest_groups += 1
    global nest_viol += count(diff(c) .> 1e-6)
end
for w in unique(wt), r in unique(rh), a in unique(ab)
    m = (wt.==w) .& (rh.==r) .& (ab.==a)
    c = cov[m][sortperm(KK[m])]
    global kmono_viol += count(diff(c) .< -1e-6)
end
for w in unique(wt), k in unique(KK), a in unique(ab)
    m = (wt.==w) .& (KK.==k) .& (ab.==a)
    c = cov[m][sortperm(rh[m])]
    global rmono_viol += count(diff(c) .> 1e-6)
end
chk("J7a Lemma-2 nesting on all frozen rows ($nest_groups groups)",
    nest_viol == 0 && nest_groups >= 100)
chk("J7b coverage nondecreasing in K (0 violations)", kmono_viol == 0)
chk("J7c coverage nonincreasing in rho (0 violations)", rmono_viol == 0)

# J8: Proposition-3 identities on the blind-policy table
bp, _ = readcsv(joinpath(CS, "blindpolicy_511.csv"))
okid = true
for i in 1:size(bp,1)
    rep, real, best = float(bp[i,6]), float(bp[i,10]), float(bp[i,11])
    rgap, regr = float(bp[i,12]), float(bp[i,13])
    global okid &= abs((rep-real) - rgap) < 1e-9
    global okid &= abs((best-real) - regr) < 1e-9
    global okid &= real <= best + 1e-9
end
chk("J8 ReportingGap/Regret identities + Realized<=Best (all rows)", okid)

# J9: exact-range table: min<=max<=V*_Eucl; rho=0.5 both certified, [81.30,81.37]
er, _ = readcsv(joinpath(CS, "exactrange_511.csv"))
ok9 = true
for i in 1:size(er,1)
    vs, mn, mx = float(er[i,4]), float(er[i,5]), float(er[i,8])
    global ok9 &= (mn <= mx + 1e-9) && (mx <= vs + 1e-9)
end
r05 = findfirst(i -> float(er[i,1]) == 0.5, 1:size(er,1))
ok9 &= Int(er[r05,7]) == 2 && Int(er[r05,10]) == 2
ok9 &= abs(float(er[r05,5]) - 81.30) < 0.01 && abs(float(er[r05,8]) - 81.37) < 0.01
chk("J9 exact-range ordering + rho=0.5 certified [81.30, 81.37]", ok9)

# J10: A0 cross-file consistency + tab:a0 values
c2, _ = readcsv(joinpath(CS, "check02_511.csv"))
a0_08 = cov[(wt.=="pop") .& (rh.==0.8) .& (ab.=="A0") .& (KK.==30)][1]
i08 = findfirst(i -> float(c2[i,1]) == 0.8, 1:size(c2,1))
chk("J10a A0(K=30,rho=.8) milp==check02==86.23",
    abs(a0_08 - float(c2[i08,3])) < 1e-6 && abs(a0_08 - 86.23) < 0.01)
i05 = findfirst(i -> float(c2[i,1]) == 0.5, 1:size(c2,1))
tab = [(1, 100.0, 97.2, 0.97), (2, 100.0, 97.2, 0.97),
       (Int(i05), 99.9, 93.8, 0.94), (Int(i08), 86.2, 1.8, 0.02)]
ok10 = true
for (i, a0c, mc, jw) in tab
    global ok10 &= abs(float(c2[i,3]) - a0c) < 0.05
    global ok10 &= abs(float(c2[i,2]) - mc) < 0.05
    global ok10 &= abs(float(c2[i,5]) - jw) < 0.005
end
chk("J10b tab:a0 rows match frozen check02 table", ok10)
jstat = maximum(float.(c2[:,6]))
chk("J10c station-set Jaccard <= 0.13 at every reserve", jstat <= 0.133)

println("=== verify_v14.jl ", PASSFAIL[] ? "ALL CHECKS PASS" : "FAILURES PRESENT", " ===")
