"""Why does the LAST root of a communal window get an anti-behaviour target
(beh mass ~0.12 vs 0.5-0.7 at every other slot; flag on or off; NOT cured
by searching its successor -- probe_edge_followups)? Variants on r12's
live checkpoint, sequential engine, flag on, one trajectory:
  base16   : window 16 (reference)
  rev16    : selections within a round in DESCENDING root order
  w8, w32  : window 8 / 32 -- does the anomaly sit at the last slot?
  nonoise  : no Dirichlet noise at the roots
  look+rec : window 16 + lookahead root, recording slot 16 as well
Prints beh mass / argmax==beh per slot."""
import sys, pathlib, glob, asyncio, builtins
ROOT = pathlib.Path(r"Y:\SwitchPlayer-communal"); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import numpy as np, torch, muzero, ablation
CK = glob.glob(r"Y:\SwitchPlayer\benchmarks\results\pong-r12-2026-09-14\resume-*.ckpt")[0]
arm = "effzero-repro-vtree-communal8-r12"; spec = ablation.ARMS[arm]
ablation.ENV_NAME, ablation.FRAMESKIP, ablation.EPISODE_CAP, ablation.ACTIONS = "Pong-v4", 4, 3000, 6


def run_variant(run, traj, W, lookahead=False, record_extra=False):
    searcher = run.searcher
    S = muzero.ACT_SEARCH_SIMS
    indices = run._eligible_indices(traj)
    evaluated = searcher.eval_obs([(t["obs"], t["action"], t["reward"]) for t in traj])
    searcher.create_tree_from_game(evaluated, terminated=not traj[-1].get("truncated", False), seed_values=True)
    rows = []
    released = -1
    for w0 in range(0, len(indices), W):
        widx = indices[w0:w0 + W]
        extra = [indices[w0 + W]] if (lookahead and w0 + W < len(indices)) else []
        np.random.seed(1000 + w0)
        res = asyncio.run(searcher.run_communal_mcts(widx + extra, num_simulations=S, c_puct=muzero.C_PUCT, discount=muzero.DISCOUNT, chain_seeded=True))
        rec = widx + (extra if record_extra else [])
        for p, idx in enumerate(rec):
            pol, val, has = res[idx]
            if has:
                rows.append((p, pol[int(traj[idx]["action"])], int(np.argmax(pol) == int(traj[idx]["action"]))))
        searcher.release_span(released + 1, widx[-1]); released = widx[-1]
    searcher.release_span(released + 1, len(traj) - 1)
    r = np.array(rows, dtype=float)
    slots = sorted(set(int(x) for x in r[:, 0]))
    mass = " ".join(f"{r[r[:,0]==p][:,1].mean():.2f}" for p in slots)
    arg = " ".join(f"{r[r[:,0]==p][:,2].mean():.2f}" for p in slots)
    return len(r), mass, arg


with ablation.arm_config(arm):
    run = ablation.Run(arm, 0, spec)
    ablation.load_run_checkpoint(run, CK, ablation.run_expect(arm, 0, 90, 2000, 1, 5, 0))
    run.set_train_mode(False)
    muzero.USE_COMMUNAL_REANALYSIS = True
    muzero.COMMUNAL_EDGE_VISITS = True
    nm = run.searcher.network_manager
    nm.representation, nm.prediction, nm.dynamics, nm.value_prefix = (run.trainer.target_representation, run.trainer.target_prediction, run.trainer.target_dynamics, run.trainer.target_value_prefix)
    traj = run.replay[0]
    print(f"ckpt iter {run.trainer.training_steps // 2000}; traj len {len(traj)}; S = {muzero.ACT_SEARCH_SIMS}", flush=True)

    def show(name, out):
        n, mass, arg = out
        print(f"{name:>9} n={n:>4}\n           beh_mass by slot: {mass}\n           argmax==beh     : {arg}", flush=True)

    show("base16", run_variant(run, traj, 16))
    muzero.sorted = lambda x, **kw: list(reversed(builtins.sorted(x, **kw)))   # descending selection order
    show("rev16", run_variant(run, traj, 16))
    del muzero.sorted
    show("w8", run_variant(run, traj, 8))
    show("w32", run_variant(run, traj, 32))
    real_noise = muzero.add_dirichlet_noise
    muzero.add_dirichlet_noise = lambda prior, epsilon=0.25, alpha=0.25: prior
    show("nonoise", run_variant(run, traj, 16))
    muzero.add_dirichlet_noise = real_noise
    show("look+rec", run_variant(run, traj, 16, lookahead=True, record_extra=True))
    print("DONE", flush=True)
