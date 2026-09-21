"""Localize the every-16th-position anti-behaviour cliff (pos % 16 == 15,
independent of window size, selection order, noise; present in the run's
own stored targets; stored actions/rewards/frame stacks normal there).
Re-search the SAME trajectory sliced at offsets 0..3: if the cliff moves
with the slice (position-in-chain % 16 == 15) it is chain arithmetic in
the search; if it stays on the absolute record (i % 16 == 15) it is in
the records. Sequential engine, flag on, window 16, one trajectory."""
import sys, pathlib, glob, asyncio
ROOT = pathlib.Path(r"Y:\SwitchPlayer-communal"); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import numpy as np, torch, muzero, ablation
CK = glob.glob(r"Y:\SwitchPlayer\benchmarks\results\pong-r12-2026-09-14\resume-*.ckpt")[0]
arm = "effzero-repro-vtree-communal8-r12"; spec = ablation.ARMS[arm]
ablation.ENV_NAME, ablation.FRAMESKIP, ablation.EPISODE_CAP, ablation.ACTIONS = "Pong-v4", 4, 3000, 6


def search(run, traj, W=16):
    searcher = run.searcher
    S = muzero.ACT_SEARCH_SIMS
    indices = run._eligible_indices(traj)
    evaluated = searcher.eval_obs([(t["obs"], t["action"], t["reward"]) for t in traj])
    searcher.create_tree_from_game(evaluated, terminated=not traj[-1].get("truncated", False), seed_values=True)
    out = {}
    released = -1
    for w0 in range(0, len(indices), W):
        widx = indices[w0:w0 + W]
        np.random.seed(1000 + w0)
        res = asyncio.run(searcher.run_communal_mcts(widx, num_simulations=S, c_puct=muzero.C_PUCT, discount=muzero.DISCOUNT, chain_seeded=True))
        for idx in widx:
            pol, val, has = res[idx]
            if has:
                out[idx] = pol[int(traj[idx]["action"])]
        searcher.release_span(released + 1, widx[-1]); released = widx[-1]
    searcher.release_span(released + 1, len(traj) - 1)
    return out


with ablation.arm_config(arm):
    run = ablation.Run(arm, 0, spec)
    ablation.load_run_checkpoint(run, CK, ablation.run_expect(arm, 0, 90, 2000, 1, 5, 0))
    run.set_train_mode(False)
    muzero.USE_COMMUNAL_REANALYSIS = True
    muzero.COMMUNAL_EDGE_VISITS = True
    nm = run.searcher.network_manager
    nm.representation, nm.prediction, nm.dynamics, nm.value_prefix = (run.trainer.target_representation, run.trainer.target_prediction, run.trainer.target_dynamics, run.trainer.target_value_prefix)
    traj = run.replay[0]
    print(f"ckpt iter {run.trainer.training_steps // 2000}; traj len {len(traj)}; S = {muzero.ACT_SEARCH_SIMS}; prefix-in-tree {muzero.USE_VALUE_PREFIX_IN_TREE}", flush=True)
    for k in (0, 1, 2, 3):
        out = search(run, traj[k:])
        by_chain = {s: [] for s in range(16)}; by_abs = {s: [] for s in range(16)}
        for j, m in out.items():
            by_chain[j % 16].append(m); by_abs[(j + k) % 16].append(m)
        print(f"offset {k}: beh mass by CHAIN pos%16: " + " ".join(f"{np.mean(by_chain[s]):.2f}" for s in range(16)), flush=True)
        print(f"          beh mass by ABS   pos%16: " + " ".join(f"{np.mean(by_abs[s]):.2f}" for s in range(16)), flush=True)
    print("DONE", flush=True)
