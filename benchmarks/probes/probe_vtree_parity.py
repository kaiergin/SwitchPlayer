"""Check B: on a REAL production checkpoint (keepnew-90 @ u 88: target nets +
replay), compare refresh targets from (i) the vectorized GPU engine exactly
as production calls it, (ii) the sequential fresh-root search (x2 -> repeat
noise floor), (iii) archfix's communal refresh (sequential engine), all
through the same lagged target nets on the same stored positions."""
import sys, time, pathlib
ROOT = pathlib.Path(r"Y:\SwitchPlayer")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import numpy as np, torch
import muzero, ablation

CK = ROOT / "benchmarks/results/pong-keepnew90-2026-09-08/resume-pong-skip4-effzero-repro-async-keepnew-seed1.ckpt"
arm = "effzero-repro-async-keepnew"; spec = ablation.ARMS[arm]
ablation.ENV_NAME, ablation.FRAMESKIP, ablation.EPISODE_CAP, ablation.ACTIONS = "Pong-v4", 4, 3000, 6
N_TRAJ, N_SUB = 2, 48

def kl(p, q):
    p = np.asarray(p, dtype=np.float64) + 1e-6; q = np.asarray(q, dtype=np.float64) + 1e-6
    p /= p.sum(); q /= q.sum(); return float((p * np.log(p / q)).sum())
def H(p):
    p = np.asarray(p, dtype=np.float64); return float(-(p * np.log(p + 1e-12)).sum())

with ablation.arm_config(arm):
    run = ablation.Run(arm, 1, spec)
    expect = ablation.run_expect(arm, 1, 90, 2000, 1, 5, 0)
    ablation.load_run_checkpoint(run, CK, expect)
    run.set_train_mode(False)
    print(f"loaded: {len(run.replay)} trajs, train {run.trainer.training_steps}, "
          f"sims {muzero.ACT_SEARCH_SIMS}, vtree batch {muzero.VTREE_REFRESH_BATCH}, "
          f"device {run.trainer.device}", flush=True)
    order = sorted(range(len(run.replay)), key=lambda i: len(run.replay[i]))
    picks = order[:N_TRAJ]  # shortest trajectories bound the communal full-pass cost
    engine = run._vtree_engine()
    nm = run.searcher.network_manager
    saved = (nm.representation, nm.prediction, nm.dynamics, nm.value_prefix)
    def with_target(fn):
        run.trainer._ensure_target_model()
        nm.representation, nm.prediction, nm.dynamics, nm.value_prefix = (
            run.trainer.target_representation, run.trainer.target_prediction,
            run.trainer.target_dynamics, run.trainer.target_value_prefix)
        try:
            return fn()
        finally:
            nm.representation, nm.prediction, nm.dynamics, nm.value_prefix = saved
    agg = {k: [] for k in ("kl_vt_seq", "kl_seq_seq2", "kl_vt_comm", "kl_seq_comm",
                           "H_vt", "H_seq", "H_comm", "H_stored",
                           "dv_vt_seq", "dv_seq_comm", "dv_vt_comm")}
    agree = {"vt_seq": 0, "seq_seq2": 0, "vt_comm": 0, "seq_comm": 0, "n": 0}
    for ti in picks:
        traj = run.replay[ti]; elig = run._eligible_indices(traj)
        sub = elig[::max(1, len(elig) // N_SUB)][:N_SUB]
        t0 = time.time(); vt = {}
        batch = max(1, int(muzero.VTREE_REFRESH_BATCH))
        for w0 in range(0, len(sub), batch):
            chunk = sub[w0:w0 + batch]
            obs = np.stack([traj[i]["obs"] for i in chunk])
            pols, vals, _ = engine.search(obs, num_simulations=muzero.ACT_SEARCH_SIMS,
                                          c_puct=muzero.C_PUCT, discount=muzero.DISCOUNT,
                                          root_noise=True)
            for i, p, v in zip(chunk, pols, vals):
                vt[i] = (np.asarray(p, dtype=np.float64), float(v))
        tv = time.time() - t0
        muzero.USE_COMMUNAL_REANALYSIS = False
        t0 = time.time(); seq = with_target(lambda: run._search_traj(traj, sub)); ts = time.time() - t0
        seq2 = with_target(lambda: run._search_traj(traj, sub))
        muzero.USE_COMMUNAL_REANALYSIS = True
        t0 = time.time(); comm = with_target(lambda: run._search_traj(traj, elig)); tc = time.time() - t0
        muzero.USE_COMMUNAL_REANALYSIS = False
        for i in sub:
            agg["kl_vt_seq"].append(kl(vt[i][0], seq[i][0]))
            agg["kl_seq_seq2"].append(kl(seq[i][0], seq2[i][0]))
            agg["kl_vt_comm"].append(kl(vt[i][0], comm[i][0]))
            agg["kl_seq_comm"].append(kl(seq[i][0], comm[i][0]))
            agg["H_vt"].append(H(vt[i][0])); agg["H_seq"].append(H(seq[i][0]))
            agg["H_comm"].append(H(comm[i][0])); agg["H_stored"].append(H(traj[i]["target_policy"]))
            agg["dv_vt_seq"].append(vt[i][1] - seq[i][1])
            agg["dv_seq_comm"].append(seq[i][1] - comm[i][1])
            agg["dv_vt_comm"].append(vt[i][1] - comm[i][1])
            agree["n"] += 1
            agree["vt_seq"] += int(np.argmax(vt[i][0]) == np.argmax(seq[i][0]))
            agree["seq_seq2"] += int(np.argmax(seq[i][0]) == np.argmax(seq2[i][0]))
            agree["vt_comm"] += int(np.argmax(vt[i][0]) == np.argmax(comm[i][0]))
            agree["seq_comm"] += int(np.argmax(seq[i][0]) == np.argmax(comm[i][0]))
        print(f"traj {ti}: len {len(traj)} sub {len(sub)} | vtree {tv:.0f}s  seq {ts:.0f}s  "
              f"communal(full {len(elig)}) {tc:.0f}s", flush=True)
    def s(x):
        x = np.asarray(x); return f"mean {x.mean():+.4f}  median {np.median(x):+.4f}  p90 {np.percentile(x, 90):+.4f}"
    print("\n=== RESULTS (n=%d positions) ===" % agree["n"], flush=True)
    for k, v in agg.items():
        print(f"{k:>12}: {s(v)}", flush=True)
    n = agree["n"]
    print("argmax agreement: vt-vs-seq %.2f | seq-vs-seq2 (noise) %.2f | vt-vs-comm %.2f | seq-vs-comm %.2f"
          % (agree["vt_seq"] / n, agree["seq_seq2"] / n, agree["vt_comm"] / n, agree["seq_comm"] / n), flush=True)
    print("DONE", flush=True)
