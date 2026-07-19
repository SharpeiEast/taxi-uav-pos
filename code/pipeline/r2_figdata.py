"""Export figure CSVs for the BD-layer results into Rcode_gk (pop headline).
Mirrors c_figdata.py formats; selections re-solved on bd reach matrices.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import load_npz
import gurobipy as gp
from gurobipy import GRB

P = Path("/lustre/home/2406393544/sharefolder/proj3")
RC = P / "Rcode_gk"
CORE = ["Eucl", "energy", "sigma_High", "sigma_Medium", "sigma_Low"]

allr = pd.concat([pd.read_csv(P / f"r2_layer/milp/results_{w}.csv")
                  for w in ["pop", "ev", "uniform"]])
allr = allr.rename(columns={"weight": "demand"})
k = allr[allr.ablation.isin(CORE)].copy()

k[(k.demand == "pop")][["ablation", "rho", "K", "coverage_pct"]] \
    .to_csv(RC / "fig1_data.csv", index=False)
k[(k.demand == "pop") & (k.K == 30)][["ablation", "rho", "coverage_pct"]] \
    .to_csv(RC / "fig2_data.csv", index=False)
k[["ablation", "rho", "K", "demand", "coverage_pct"]] \
    .to_csv(RC / "fig3_data.csv", index=False)
k[k.K == 30][["ablation", "rho", "demand", "coverage_pct"]] \
    .to_csv(RC / "fig5_data.csv", index=False)
f4 = k[k.K == 605].copy()
f4["infeasibility_pct"] = 100.0 - f4["coverage_pct"]
f4[f4.demand.isin(["pop", "uniform"])][["ablation", "rho", "demand", "infeasibility_pct"]] \
    .to_csv(RC / "fig4_data.csv", index=False)
print("[ok] fig1-5 data (BD, pop headline)", flush=True)

# ---- fig6: sigma_Medium beta=1 K=30 pop selections on the BD criterion ----
joined = pd.read_parquet(P / "gk_run/joined_wuhan_gk.parquet")
pop = pd.read_parquet(P / "output_demand/wuhan_demand_pop_per_cell.parquet")
lut = dict(zip(zip(pop.grid_x, pop.grid_y), pop.pop_density.astype(float)))
w_pop = np.array([lut.get((int(x), int(y)), 0.0) for x, y in
                  zip(joined.grid_x, joined.grid_y)])
TOT = float(pop.pop_density.sum())
cand = pd.read_csv(P / "wuhan_stations_geocoded.csv")

def solve_sel(a_csr, K=30):
    at = a_csr.T.tocsr()
    ns, nc = a_csr.shape
    m = gp.Model(); m.Params.OutputFlag = 0; m.Params.MIPGap = 1e-6
    m.Params.Threads = 16; m.Params.Seed = 7
    x = m.addVars(ns, vtype=GRB.BINARY)
    y = m.addVars(nc, vtype=GRB.CONTINUOUS, ub=1.0)
    m.addConstr(gp.quicksum(x[i] for i in range(ns)) <= K)
    for j in range(nc):
        idx = at.indices[at.indptr[j]:at.indptr[j + 1]]
        m.addConstr(y[j] <= (gp.quicksum(x[i] for i in idx) if len(idx) else 0))
    m.setObjective(gp.quicksum(float(w_pop[j]) * y[j] for j in range(nc)), GRB.MAXIMIZE)
    m.optimize()
    return [i for i in range(ns) if x[i].X > 0.5]

cell_rows, st_rows = [], []
pos_idx = np.where(w_pop > 0)[0]
for rho in [0.0, 0.2, 0.5, 0.8]:
    a = load_npz(P / f"r2_layer/reach/a_sigmaMedium_b1.0_rho{rho}.npz").tocsr()
    S = solve_sel(a)
    covered = np.asarray(a[S, :].sum(axis=0)).flatten() > 0
    for j in pos_idx:
        cell_rows.append({"lon": float(joined.center_lon.iloc[j]),
                          "lat": float(joined.center_lat.iloc[j]),
                          "n_orders": float(w_pop[j]), "rho": rho,
                          "covered": bool(covered[j])})
    for i in S:
        st_rows.append({"lon": float(cand["lng"].iloc[i]),
                        "lat": float(cand["lat"].iloc[i]), "rho": rho})
    print(f"  fig6 rho={rho}: pop coverage="
          f"{100 * w_pop[covered].sum() / TOT:.1f}%  K={len(S)}", flush=True)
pd.DataFrame(cell_rows).to_csv(RC / "fig6_cells.csv", index=False)
pd.DataFrame(st_rows).to_csv(RC / "fig6_stations.csv", index=False)
print("[ok] fig6 selections (BD)", flush=True)

# ---- new-figure data copies ----
pd.read_csv(P / "r2_layer/waterfall.csv").to_csv(RC / "waterfall_data.csv", index=False)
pd.read_csv(P / "r2_layer/threeclass.csv").to_csv(RC / "threeclass_data.csv", index=False)
print("done", flush=True)
