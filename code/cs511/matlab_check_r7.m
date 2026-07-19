% matlab_check_r7.m -- independent MATLAB re-verification .
% 1. Recompute the coverage percentages of the blind-policy station
%    sets directly from the exported feasibility matrices and compare
%    with the Python/Gurobi values (blindpolicy_check.mat).
% 2. Greedy sandwich: for each rho, run greedy maximal covering on the
%    Medium feasibility matrix and assert
%    greedy <= reported IP optimum <= LP bound (from lpgap_511_pop.csv).
P = '/lustre/home/2406393544/sharefolder/proj3/cs511/';
S = load([P 'blindpolicy_check.mat']);
w = double(S.w(:)); TOT = double(S.TOT);
lp = readtable([P 'lpgap_511_pop.csv']);
csvrows = readtable([P 'blindpolicy_511.csv']);
rhos = [0.0 0.2 0.5 0.8]; tags = {'00','02','05','08'};
fail = 0;
for t = 1:4
    tag = tags{t};
    Sset = double(S.(['S_rho' tag])(:));
    fe = S.(['fe_rho' tag]); fm = S.(['fm_rho' tag]);
    vals = double(S.(['vals_rho' tag])(:));   % [reported realized best]
    cov_e = full(any(fe(Sset, :), 1))';
    cov_m = full(any(fm(Sset, :), 1))';
    rep = 100 * sum(w(cov_e)) / TOT;
    rea = 100 * sum(w(cov_m)) / TOT;
    if abs(rep - vals(1)) > 1e-6 || abs(rea - vals(2)) > 1e-6
        fprintf('RHO %s coverage recompute FAIL: %.6f vs %.6f / %.6f vs %.6f\n', ...
            tag, rep, vals(1), rea, vals(2));
        fail = fail + 1;
    else
        fprintf('RHO %s coverage recompute PASS (reported %.4f, realized %.4f)\n', ...
            tag, rep, rea);
    end
    % greedy sandwich on the Medium matrix at K = 30
    K = 30; nS = size(fm, 1);
    covered = false(size(fm, 2), 1); pick = zeros(K, 1);
    wr = w;
    for k = 1:K
        gain = fm * (wr .* ~covered);   % marginal weighted gain per station
        [~, ibest] = max(gain);
        pick(k) = ibest;
        covered = covered | full(fm(ibest, :))';
    end
    greedy = 100 * sum(w(covered)) / TOT;
    ip = vals(3);   % best_pct from Gurobi
    row = lp(abs(lp.rho - rhos(t)) < 1e-9 & lp.K == 30 & ...
             strcmp(lp.ablation, 'Medium'), :);
    if ~isempty(row)
        lpb = 100 * row.z_lp(1) / TOT;
        okc = (greedy <= ip + 1e-6) && (ip <= lpb + 1e-6);
        fprintf('RHO %s sandwich %s: greedy %.4f <= IP %.4f <= LP %.4f\n', ...
            tag, string(okc), greedy, ip, lpb);
        if ~okc, fail = fail + 1; end
    else
        fprintf('RHO %s sandwich: greedy %.4f <= IP %.4f (no LP row)\n', ...
            tag, greedy, ip);
        if greedy > ip + 1e-6, fail = fail + 1; end
    end
end
if fail == 0
    fprintf('ALL MATLAB CHECKS PASS\n');
else
    fprintf('%d MATLAB CHECKS FAILED\n', fail);
end
exit(double(fail > 0));
