"""Action/observation alignment test for the every-16th-record cliff
(target anti-behaviour at absolute record index i % 16 == 15, in the
run's stored targets and in fresh sequential re-searches; follows the
record, not the chain position -- probe_edge_offset). For every record t
push latent(obs_t) through the dynamics model under EVERY action and
score each against latent(obs_{t+1}) by cosine similarity: if at
i % 16 == 15 the stored action explains the next observation worse than
elsewhere, or a neighbour's action explains it better, the records are
misaligned there. No search involved."""
import sys, pathlib, glob
ROOT = pathlib.Path(r"Y:\SwitchPlayer-communal"); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import numpy as np, torch, torch.nn.functional as F, muzero, ablation
CK = glob.glob(r"Y:\SwitchPlayer\benchmarks\results\pong-r12-2026-09-14\resume-*.ckpt")[0]
arm = "effzero-repro-vtree-communal8-r12"; spec = ablation.ARMS[arm]
ablation.ENV_NAME, ablation.FRAMESKIP, ablation.EPISODE_CAP, ablation.ACTIONS = "Pong-v4", 4, 3000, 6
A = 6

with ablation.arm_config(arm):
    run = ablation.Run(arm, 0, spec)
    ablation.load_run_checkpoint(run, CK, ablation.run_expect(arm, 0, 90, 2000, 1, 5, 0))
    run.set_train_mode(False)
    nm = run.searcher.network_manager
    rep, dyn = run.trainer.target_representation, run.trainer.target_dynamics
    dev = next(rep.parameters()).device
    rows = []
    with torch.no_grad():
        for traj in run.replay[:6]:
            L = len(traj)
            obs = np.stack([r["obs"] for r in traj])
            acts = np.array([int(r["action"]) for r in traj])
            lat = torch.cat([rep(torch.tensor(obs[lo:lo + 64], dtype=torch.float32, device=dev) / 255.0)
                             for lo in range(0, L, 64)])
            for lo in range(0, L - 1, 32):
                idx = np.arange(lo, min(lo + 32, L - 1))
                lb = lat[idx].repeat_interleave(A, 0)
                ab = torch.arange(A, device=dev).repeat(len(idx))
                nxt, _, _ = dyn(lb, ab)
                tgt = lat[idx + 1].repeat_interleave(A, 0)
                cos = F.cosine_similarity(nxt.flatten(1), tgt.flatten(1), dim=1).view(len(idx), A).cpu().numpy()
                best = cos.argmax(1)
                for j, t in enumerate(idx):
                    a = acts[t]
                    prev_a = acts[t - 1] if t > 0 else a
                    next_a = acts[t + 1]
                    rows.append((t % 16, int(best[j] == a), int(best[j] == prev_a), int(best[j] == next_a),
                                 cos[j, a] - cos[j].max(), cos[j].max() - np.sort(cos[j])[-2]))
    r = np.array(rows, dtype=float)
    print(f"records {len(r)} over {min(6, len(run.replay))} trajectories")
    print(" slot   best==stored  best==prev  best==next   cos(stored)-max   margin(best-2nd)")
    for k in range(16):
        s = r[r[:, 0] == k]
        print(f"  {k:>2}      {s[:,1].mean():.3f}       {s[:,2].mean():.3f}       {s[:,3].mean():.3f}      {s[:,4].mean():+.5f}        {s[:,5].mean():.5f}")
    print("DONE", flush=True)
