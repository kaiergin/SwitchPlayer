"""Two follow-ups to probe_edge_anatomy on r12's live checkpoint:
(1) LOOKAHEAD: is slot 15's anti-behaviour target (beh mass 0.12 vs 0.5-0.7
    elsewhere, flag on or off) caused by its successor being UNSEARCHED in
    the window? Search each window with one extra root (the next window's
    first) but read out only the 16; compare slot 15 against the plain
    window. Flag on, sequential engine.
(2) PARITY CONTROL: vectorized (TRAJS=1) vs sequential on real GPU weights
    with the flag OFF, same trajectory and seeds as the flag-on run
    (96.5% of roots within 1e-12 there), to separate GPU nondeterminism
    from any flag-specific divergence."""
import sys, pathlib, glob, asyncio, time
ROOT = pathlib.Path(r"Y:\SwitchPlayer-communal"); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import numpy as np, torch, muzero, ablation
CK = glob.glob(r"Y:\SwitchPlayer\benchmarks\results\pong-r12-2026-09-14\resume-*.ckpt")[0]
arm = "effzero-repro-vtree-communal8-r12"; spec = ablation.ARMS[arm]
ablation.ENV_NAME, ablation.FRAMESKIP, ablation.EPISODE_CAP, ablation.ACTIONS = "Pong-v4", 4, 3000, 6


def H(p):
    p = np.asarray(p, float); s = p.sum()
    return float(-(p / s * np.log(p / s + 1e-12)).sum()) if s > 0 else float("nan")


def anatomy(run, trajs, lookahead):
    searcher = run.searcher
    S = muzero.ACT_SEARCH_SIMS; W = muzero.COMMUNAL_WINDOW_ROOTS
    rows = []
    for traj in trajs:
        indices = run._eligible_indices(traj)
        evaluated = searcher.eval_obs([(t["obs"], t["action"], t["reward"]) for t in traj])
        searcher.create_tree_from_game(evaluated, terminated=not traj[-1].get("truncated", False), seed_values=True)
        released = -1
        for w0 in range(0, len(indices), W):
            widx = indices[w0:w0 + W]
            extra = [indices[w0 + W]] if (lookahead and w0 + W < len(indices)) else []
            np.random.seed(1000 + w0)
            res = asyncio.run(searcher.run_communal_mcts(widx + extra, num_simulations=S, c_puct=muzero.C_PUCT, discount=muzero.DISCOUNT, chain_seeded=True))
            for p, idx in enumerate(widx):
                pol, val, has = res[idx]
                if not has:
                    continue
                beh = int(traj[idx]["action"])
                rows.append((p, pol[beh], int(np.argmax(pol) == beh), H(pol), val))
            searcher.release_span(released + 1, widx[-1]); released = widx[-1]
        searcher.release_span(released + 1, len(traj) - 1)
    return np.array(rows, dtype=float)


with ablation.arm_config(arm):
    run = ablation.Run(arm, 0, spec)
    ablation.load_run_checkpoint(run, CK, ablation.run_expect(arm, 0, 90, 2000, 1, 5, 0))
    run.set_train_mode(False)
    muzero.USE_COMMUNAL_REANALYSIS = True
    print(f"ckpt iter {run.trainer.training_steps // 2000}; S = {muzero.ACT_SEARCH_SIMS}, window = {muzero.COMMUNAL_WINDOW_ROOTS}", flush=True)
    nm = run.searcher.network_manager
    saved = (nm.representation, nm.prediction, nm.dynamics, nm.value_prefix)
    nm.representation, nm.prediction, nm.dynamics, nm.value_prefix = (run.trainer.target_representation, run.trainer.target_prediction, run.trainer.target_dynamics, run.trainer.target_value_prefix)
    trajs = run.replay[:2]

    # ---------------- (1) lookahead ----------------
    muzero.COMMUNAL_EDGE_VISITS = True
    plain = anatomy(run, trajs, lookahead=False)
    look = anatomy(run, trajs, lookahead=True)
    print("(1) flag on: plain 16-root windows vs 16 + one lookahead root (searched, not read out)")
    print(" pos     plain: beh_mass argmax=beh   H   |  lookahead: beh_mass argmax=beh   H   |  d(root value)")
    for p in list(range(0, 3)) + list(range(12, 16)):
        a = plain[plain[:, 0] == p]; b = look[look[:, 0] == p]
        print(f"{p:>4}          {a[:,1].mean():.3f}    {a[:,2].mean():.3f}   {a[:,3].mean():.3f} |            {b[:,1].mean():.3f}    {b[:,2].mean():.3f}   {b[:,3].mean():.3f} |  {b[:,4].mean() - a[:,4].mean():+.3f}")
    a, b = plain, look
    print(f" ALL          {a[:,1].mean():.3f}    {a[:,2].mean():.3f}   {a[:,3].mean():.3f} |            {b[:,1].mean():.3f}    {b[:,2].mean():.3f}   {b[:,3].mean():.3f} |  {b[:,4].mean() - a[:,4].mean():+.3f}", flush=True)

    # ---------------- (2) parity control, flag OFF ----------------
    muzero.COMMUNAL_EDGE_VISITS = False
    muzero.VTREE_COMMUNAL_TRAJS = 1
    engine = run._vtree_engine()
    t = trajs[0]; ix = run._eligible_indices(t)
    np.random.seed(321)
    vec = engine.search_trajectories([{"obs": [x["obs"] for x in t], "actions": [x["action"] for x in t], "rewards": [x["reward"] for x in t], "terminated": not t[-1].get("truncated", False)}], [ix])[0]
    np.random.seed(321)
    seq = run._search_traj_communal(t, ix)
    nm.representation, nm.prediction, nm.dynamics, nm.value_prefix = saved
    n = exact = 0; maxd = 0.0; dv = []
    for i in ix:
        if i not in vec or i not in seq:
            continue
        n += 1
        d = float(np.abs(np.asarray(vec[i][0]) - np.asarray(seq[i][0])).max())
        maxd = max(maxd, d); exact += int(d <= 1e-12); dv.append(abs(vec[i][1] - seq[i][1]))
    print(f"(2) flag OFF control, TRAJS=1 vectorized vs sequential, {n} roots: within 1e-12 on {exact}/{n} ({exact/n:.3f}), max |dpol| {maxd:.2e}, max |dvalue| {max(dv):.2e}")
    print("DONE", flush=True)
