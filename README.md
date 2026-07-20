# taxi-uav-pos

Passive taxi-fleet sensing of urban positioning uncertainty for UAV
logistics: mission-feasible service coverage and charging-station
siting. A Wuhan case study: a grid-level empirical dispersion field is
estimated from large-scale taxi GPS traces, mapped to mission-feasible
UAV reachability through a physics-based energy model, and embedded in
a maximal covering charging-station location model.

## Contents
- `field/`     Aggregated cell-level positioning-dispersion field for
               Wuhan: `field_r2.npz` (headline noise-controlled field on
               the routing grid: sigma_full, sig_j, j_denoised) and
               `cells_observed.csv` (per-cell observed statistics:
               worst-direction dispersion, segment counts, anisotropy).
               No raw trajectories are included (data-provider
               agreement); these aggregates suffice to reproduce every
               optimisation result.
- `instance/`  Full 605-site candidate pool (coordinates only,
               anonymized), population demand weights per 100 m cell,
               open-data feasibility screens (the base screen defines
               the 511-site headline planning universe; the strict
               screen defines the 132-site variant), and the
               routing-grid domain (nodes, edges, candidate/demand
               node indices).
- `code/`      `extraction/` — stationary-segment GK extraction pipeline
               (consumes raw trajectories; included for transparency);
               `pipeline/` — physics layer, cost fields (Dijkstra),
               terminal model, reachability, MILP sweep, decompositions
               (entry order: r2_prep -> r2_fields -> r2_eterm ->
               r2_reach -> r2_milp -> r2_decomp/r2_regret/r2_kperp);
               `robustness/` — terminal-model variants, screens,
               keep-at-v_min, planning-universe sensitivity;
               `cs511/` — driver scripts that re-run the full chain on
               the screened 511-site headline universe (row-mask reuse
               of the cost matrices; MILP sweep, decompositions,
               permutation test, blind-benchmark comparison).
- `verification/` Independent verification scripts: exact
               quantifier-elimination certificate of the per-metre
               energy monotonicity (Wolfram), symbolic model-identity
               checks (Wolfram), coverage cross-checks with a
               greedy--IP--LP sandwich (MATLAB), a full-model audit in
               Wolfram (`full_audit.wl`) and, independently, in Julia
               (`full_audit.jl`: physics/terminal/probability layer plus
               structural checks on every frozen result row), and a
               fresh MILP re-certification (`milp_recheck.py`: headline
               re-solves, the fixed-radius baseline rebuilt from
               coordinates, and the imputed-node saturation audit).
               Julia scripts were run with Julia 1.11.7.
- `results/`   Frozen output tables behind the reported results.
               `cs511/` holds the headline (screened 511-site) tables:
               MILP sweep per demand model and rho, three-class
               decomposition, kappa_perp scenario comparison, regret,
               waterfall, terminal/payload scans, terminal-model
               variants, keep-at-v_min, planning-universe sensitivity,
               blind-benchmark check, and permutation-test gaps.
               Top-level tables are the unscreened 605-site run,
               reported in the paper as the upper-bound scenario.

## Environment
Tested with Python 3.12.7, numpy 2.4.4, scipy 1.16.3, pandas 2.3.3,
gurobipy 12.0.1 (Gurobi 12.0.1, academic license), and R 4.x with
ggplot2 for figures. Exact Python pins are in `requirements.txt`.
Verification scripts additionally use Wolfram Engine 14.3
(`wolframscript -file`) and MATLAB R2025a; both are optional and
independent of the Python pipeline. Each script is standalone; paths
are set at the top of each file.

## Entry point
To reproduce the headline tables from the shipped field and instance,
run the `code/pipeline/` stages in the order listed above
(r2_prep -> r2_fields -> r2_eterm -> r2_reach -> r2_milp), then the
`code/cs511/` drivers for the screened-511 headline results. Each
stage writes its outputs next to its inputs; `results/` contains the
frozen reference outputs to compare against.

## Runtime and hardware
The full chain was run on dual-socket Xeon 8358 nodes (64 cores,
256 GB). Approximate wall times: cost-field Dijkstra sweep (605
sources on a 1.05M-node grid) ~2 h on 64 cores; one MILP sweep
(4 reserve levels x 3 demand models) ~30 min; the bootstrap ensemble
(240 MILPs) ~6 h. A 16-core / 64 GB machine suffices for any single
stage at longer wall times.

## Configuration
`code/pipeline/r2_config.py` is the frozen headline configuration
(noise-controlled field, kappa_perp = 0.3 central scenario);
`bd_config.py` documents the conservative-envelope configuration.
