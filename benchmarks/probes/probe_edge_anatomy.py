"""COMMUNAL_EDGE_VISITS on, r12's live checkpoint (the fix branch's code):
(A) per-slot anatomy of the communal readout with the flag -- who deposited
the counts (own / earlier roots of the window / previous windows / the
successor's root visits, which the readout now baselines out) and the
resulting behaviour-action mass and entropy -- against the legacy numbers
from probe_window_attrib; (B) vectorized (TRAJS=1) vs sequential parity on
real weights under the flag."""
import sys, pathlib, glob, asyncio, time
ROOT = pathlib.Path(r"Y:\SwitchPlayer-communal"); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import numpy as np, torch, muzero, ablation
CK = glob.glob(r"Y:\SwitchPlayer\benchmarks\results\pong-r12-2026-09-14\resume-*.ckpt")[0]
arm = "effzero-repro-vtree-communal8-r12"; spec = ablation.ARMS[arm]
ablation.ENV_NAME, ablation.FRAMESKIP, ablation.EPISODE_CAP, ablation.ACTIONS = "Pong-v4", 4, 3000, 6
CLS = ("own", "earlier", "prev", "succ")


def H(p):
    p = np.asarray(p, float); s = p.sum()
    return float(-(p / s * np.log(p / s + 1e-12)).sum()) if s > 0 else float("nan")


with ablation.arm_config(arm):
    run = ablation.Run(arm, 0, spec)
    ablation.load_run_checkpoint(run, CK, ablation.run_expect(arm, 0, 90, 2000, 1, 5, 0))
    run.set_train_mode(False)
    muzero.COMMUNAL_EDGE_VISITS = True          # the fix arm's single delta
    muzero.USE_COMMUNAL_REANALYSIS = True
    S = muzero.ACT_SEARCH_SIMS; W = max(1, muzero.COMMUNAL_WINDOW_ROOTS)
    print(f"ckpt iter {run.trainer.training_steps // 2000}; S = {S}, window = {W}, MAX_DEPTH = {muzero.MAX_DEPTH}, EDGE = {muzero.COMMUNAL_EDGE_VISITS}", flush=True)
    searcher = run.searcher; nm = searcher.network_manager
    saved = (nm.representation, nm.prediction, nm.dynamics, nm.value_prefix)
    nm.representation, nm.prediction, nm.dynamics, nm.value_prefix = (run.trainer.target_representation, run.trainer.target_prediction, run.trainer.target_dynamics, run.trainer.target_value_prefix)
    trajs = run.replay[:2]

    # ---------------- (A) anatomy under the flag ----------------
    orig_backup = searcher.backup_simulation
    def hooked_backup(path, leaf_value, discount=None, min_max_stats=None, count_root=True):
        rp = getattr(path[0], "pos", None) if (path and count_root) else None
        if rp is not None:
            for c in path[1:]:
                d = c.__dict__.setdefault("from_roots", {})
                d[rp] = d.get(rp, 0) + 1
        return orig_backup(path, leaf_value, discount=discount, min_max_stats=min_max_stats, count_root=count_root)
    searcher.backup_simulation = hooked_backup
    rows = []
    for traj in trajs:
        indices = run._eligible_indices(traj)
        evaluated = searcher.eval_obs([(t["obs"], t["action"], t["reward"]) for t in traj])
        searcher.create_tree_from_game(evaluated, terminated=not traj[-1].get("truncated", False), seed_values=True)
        for i, node in enumerate(searcher.current_trajectory):
            node.pos = i
        released = -1
        for w0 in range(0, len(indices), W):
            widx = indices[w0:w0 + W]; wstart = widx[0]
            np.random.seed(1000 + w0)
            res = asyncio.run(searcher.run_communal_mcts(widx, num_simulations=S, c_puct=muzero.C_PUCT, discount=muzero.DISCOUNT, chain_seeded=True))
            for p, idx in enumerate(widx):
                pol, val, has = res[idx]
                if not has:
                    continue
                root = searcher.current_trajectory[idx]
                beh = int(traj[idx]["action"])
                tot = {k: 0.0 for k in CLS}; onbeh = {k: 0.0 for k in CLS}
                edge = np.zeros(6)
                for a, c in root.children.items():
                    edge[a] = c.visit_count - ((1 + c.root_visits) if c.is_gameplay else 0)
                    for rp, n in getattr(c, "from_roots", {}).items():
                        k = "own" if rp == idx else ("earlier" if rp >= wstart else "prev")
                        tot[k] += n; onbeh[k] += n if a == beh else 0
                    if c.is_gameplay:
                        tot["succ"] += c.root_visits; onbeh["succ"] += c.root_visits
                assert np.allclose(pol, edge / edge.sum(), atol=1e-12), "readout must equal edge counts"
                rows.append([p, edge.sum(), pol[beh], int(np.argmax(pol) == beh), H(pol)] + [tot[k] for k in CLS] + [onbeh[k] for k in CLS])
            searcher.release_span(released + 1, widx[-1]); released = widx[-1]
        searcher.release_span(released + 1, len(traj) - 1)
    searcher.backup_simulation = orig_backup
    r = np.array(rows, dtype=float)
    print(f"(A) roots {len(r)} -- readout = edge counts (own + earlier + prev), successor root visits baselined out")
    print(" pos    n  readout beh_mass argmax=beh   H   |  own  (beh%) | earlier (beh%) |  prev  (beh%) | succ(removed)")
    def line(tag, s):
        def cls(j):
            t = s[:, 5 + j].sum(); b = s[:, 9 + j].sum()
            return f"{s[:,5+j].mean():5.1f} ({(b / t if t else float('nan')):.2f})"
        print(f"{tag:>4} {len(s):>4}  {s[:,1].mean():6.1f}  {s[:,2].mean():.3f}    {s[:,3].mean():.3f}   {s[:,4].mean():.3f} | {cls(0)} | {cls(1)} | {cls(2)} | {s[:,8].mean():5.1f}")
    for p in range(W):
        s = r[r[:, 0] == p]
        if len(s):
            line(str(p), s)
    line("ALL", r)

    # ---------------- (B) vectorized vs sequential under the flag ----------------
    muzero.VTREE_COMMUNAL_TRAJS = 1
    engine = run._vtree_engine()
    t = trajs[0]; ix = run._eligible_indices(t)
    np.random.seed(321); t0 = time.time()
    vec = engine.search_trajectories([{"obs": [x["obs"] for x in t], "actions": [x["action"] for x in t], "rewards": [x["reward"] for x in t], "terminated": not t[-1].get("truncated", False)}], [ix])[0]
    tv = time.time() - t0
    np.random.seed(321); t0 = time.time()
    seq = run._search_traj_communal(t, ix)
    ts = time.time() - t0
    nm.representation, nm.prediction, nm.dynamics, nm.value_prefix = saved
    n = exact = 0; maxd = 0.0; dv = []
    for i in ix:
        if i not in vec or i not in seq:
            continue
        n += 1
        d = float(np.abs(np.asarray(vec[i][0]) - np.asarray(seq[i][0])).max())
        maxd = max(maxd, d); exact += int(d <= 1e-12); dv.append(abs(vec[i][1] - seq[i][1]))
    print(f"(B) TRAJS=1 vectorized vs sequential, flag on, {n} roots: policies within 1e-12 on {exact}/{n} ({exact/n:.3f}), max |dpol| {maxd:.2e}, max |dvalue| {max(dv):.2e}; {len(ix)/tv:.0f} vs {len(ix)/ts:.0f} pos/s")
    print("DONE", flush=True)
