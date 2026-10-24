"""Summaries of probe_qspread.py's root records. usage: analyze.py 45 60 75"""
import pathlib
import sys
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
LNA = np.log(14)
FRAMES = np.array([4, 32, 4, 4, 4, 16, 4, 32, 4, 4, 16, 4, 32, 4])


def H(p):
    p = np.clip(p, 1e-12, None)
    return -(p * np.log(p)).sum(-1)


def stats(z):
    N, P, P0 = z["N"], z["prior"], z["raw_prior"]
    norm, comp, cinit, U, Q = z["norm"], z["comp"], z["comp_init"], z["U"], z["Q"]
    R, EV, D = z["R"], z["EV"], z["D"]
    span = z["maxv"] - z["minv"]
    vis = N > 0
    S = len(N)
    tgt = N / N.sum(1, keepdims=True)
    out = {}
    out["H_target/lnA"] = H(tgt) / LNA
    out["H_prior/lnA"] = H(P0) / LNA
    out["maxp_prior"] = P0.max(1)
    out["H_noised_prior/lnA"] = H(P) / LNA
    out["n_children_visited"] = vis.sum(1)
    out["visits_on_prior_top"] = N[np.arange(S), P0.argmax(1)]

    def masked(a, m, f):
        return np.array([f(a[i][m[i]]) if m[i].any() else np.nan for i in range(S)])
    out["normQ_range_vis"] = masked(norm, vis, np.ptp)
    out["normQ_std_vis"] = masked(norm, vis, np.std)
    out["rawcomp_range_vis"] = masked(comp, vis, np.ptp)
    out["rawcomp_std_vis"] = masked(comp, vis, np.std)
    out["rawQ_range_vis"] = masked(Q, vis, np.ptp)
    out["U_median_all"] = np.median(U, 1)
    out["U_median_vis"] = masked(U, vis, np.median)
    out["ratio_normQrange/Umed"] = out["normQ_range_vis"] / out["U_median_all"]
    out["ratio_normQstd/Umed"] = out["normQ_std_vis"] / out["U_median_all"]
    out["minmax_span"] = span
    # root children's share of the min-max range (all children: unvisited
    # ones at their init composite, which is what selection scores)
    out["rootchild_comp_range/span"] = np.ptp(comp, 1) / np.maximum(span, 0.01)
    out["rootchild_init_range/span"] = np.ptp(cinit, 1) / np.maximum(span, 0.01)
    out["range_above_rootmax/span"] = (z["maxv"] - comp.max(1)) / np.maximum(span, 0.01)
    out["range_below_rootmin/span"] = (comp.min(1) - z["minv"]) / np.maximum(span, 0.01)
    out["minmax_max_depth"] = z["max_depth"]
    out["minmax_min_depth"] = z["min_depth"]
    # argmax agreements (ties count as agreement when the reference action is
    # among the maxima)
    nmax = N.max(1, keepdims=True)
    pa = P0.argmax(1)
    cq = np.where(vis, comp, -np.inf)
    qa = cq.argmax(1)
    out["agree_visit_prior(top-in-max)"] = (N[np.arange(S), pa] == nmax[:, 0]).astype(float)
    out["agree_visit_Q(top-in-max)"] = (N[np.arange(S), qa] == nmax[:, 0]).astype(float)
    out["agree_visit_prior(strict)"] = ((N.argmax(1) == pa) & ((N == nmax).sum(1) == 1)).astype(float)
    out["visit_max_ties"] = (N == nmax).sum(1)
    out["agree_prior_Q"] = (pa == qa).astype(float)
    # predicted rewards / values
    out["|R| root edge mean"] = np.abs(R).mean(1)
    out["R root edge range"] = np.ptp(R, 1)
    out["|R| root edge max"] = np.abs(R).max(1)
    out["root_value_net"] = z["root_ev"]
    out["root_q"] = z["root_q"]
    out["child_EV_range"] = np.ptp(EV, 1)
    # Q drift with visits (children with >= 2 visits): backed-up composite
    # minus the child's own initial composite
    m2 = N >= 2
    drift = comp - cinit
    out["drift_raw(N>=2)"] = masked(drift, m2, np.mean)
    out["drift/span(N>=2)"] = out["drift_raw(N>=2)"] / np.maximum(span, 0.01)
    out["frac_drift_neg(N>=2)"] = masked(drift < 0, m2, np.mean)
    # correlation of init composite with log prior across children
    lp = np.log(np.clip(P0, 1e-9, None))
    out["corr(cinit,logprior)"] = np.array([np.corrcoef(cinit[i], lp[i])[0, 1] for i in range(S)])
    out["corr(final norm,logprior)"] = np.array([np.corrcoef(norm[i], lp[i])[0, 1] for i in range(S)])
    # selection-score composition for the prior's top child vs the rest
    sc = norm + U
    out["score_top - score_maxother"] = sc[np.arange(S), pa] - np.where(
        np.eye(14, dtype=bool)[pa], -np.inf, sc).max(1)
    out["normQ_top - mean normQ others(vis)"] = norm[np.arange(S), pa] - np.array(
        [norm[i][vis[i] & (np.arange(14) != pa[i])].mean() if (vis[i] & (np.arange(14) != pa[i])).any() else np.nan for i in range(S)])
    out["U_top"] = U[np.arange(S), pa]
    out["N_top_frac"] = N[np.arange(S), pa] / N.sum(1)
    # duration
    out["visit_share_32f"] = tgt[:, FRAMES == 32].sum(1)
    out["prior_share_32f"] = P0[:, FRAMES == 32].sum(1)
    return out


def depth_profile(z):
    rows = []
    for k in range(1, 9):
        n = z[f"prof{k}_n"]
        if n.sum() == 0:
            continue
        w = n / n.sum()
        rows.append((k, n.mean(), (w * z[f"prof{k}_ev"]).sum(), (w * z[f"prof{k}_rew"]).sum(),
                     (w * z[f"prof{k}_comp"]).sum()))
    return rows


def main(argv):
    its = [int(a) for a in argv if a.isdigit()] or [45, 60, 75]
    variants = [a for a in argv if not a.isdigit()] or ["noise", "nonoise"]
    S = {}
    for it in its:
        for v in variants:
            p = HERE / f"root-iter{it:03d}-{v}.npz"
            if p.exists():
                z = dict(np.load(p))
                S[(it, v)] = (stats(z), z)
    keys = list(next(iter(S.values()))[0].keys())
    cols = list(S.keys())
    print("metric".ljust(40) + "".join(f"{f'{it}-{v}':>22}" for it, v in cols))
    for k in keys:
        line = k.ljust(40)
        for c in cols:
            a = S[c][0][k]
            line += f"{np.nanmean(a):>11.3f}/{np.nanmedian(a):>9.3f} "
        print(line)
    print("(mean/median over states)")
    for c in cols:
        z = S[c][1]
        print(c, "ceiling", float(z["ceiling"]), "steps", int(z["training_steps"]))
        print("  min-max max-node depth hist", np.bincount(z["max_depth"].astype(int), minlength=9)[:9],
              " min-node depth hist", np.bincount(z["min_depth"].astype(int), minlength=9)[:9])
        for row in depth_profile(z):
            print("   depth %d: nodes/tree %.1f  mean EV %.3f  mean edge R %.3f  mean comp %.3f" % row)
    for it in its:
        p = HERE / f"acting-prior-iter{it:03d}.npy"
        if p.exists():
            a = np.load(p)
            print(f"acting-states prior @{it}: H/lnA {np.mean(H(a)) / LNA:.3f}, maxp {a.max(1).mean():.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
