"""What is special about records at absolute index i % 16 == 15 (the
refresh's anti-behaviour cliff; the actor's own fresh-root targets there
are normal)? No search: per record, the target network's policy
probability of the stored action and its entropy, the network value, the
seeded Monte-Carlo return G_t (what the chain child carries), the
composite the chain child shows the root (r_{t+1} + g*G_{t+1}) against
the best imagined child's composite (r + g*v from the dynamics model),
and the frame-motion spike at slot 1 split by image region."""
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
    rep, dyn, pred = run.trainer.target_representation, run.trainer.target_dynamics, run.trainer.target_prediction
    dev = next(rep.parameters()).device
    g = muzero.DISCOUNT
    rows = []
    reg = {k: np.zeros(4) for k in range(16)}; regn = {k: 0 for k in range(16)}
    with torch.no_grad():
        for traj in run.replay[:8]:
            L = len(traj)
            obs = np.stack([r["obs"] for r in traj])
            acts = np.array([int(r["action"]) for r in traj])
            rews = np.array([float(r["reward"]) for r in traj])
            term = not traj[-1].get("truncated", False)
            lat = torch.cat([rep(torch.tensor(obs[lo:lo + 64], dtype=torch.float32, device=dev) / 255.0)
                             for lo in range(0, L, 64)])
            logits, vals, _ = pred(lat)
            pol = F.softmax(logits, dim=1).cpu().numpy(); v = vals.squeeze(-1).cpu().numpy()
            # seeded MC return: G_t = sum_{j>t} g^(j-t-1) r_j, leaf 0 at a terminal else v[-1]
            G = np.zeros(L); bv = 0.0 if term else float(v[-1])
            for k in range(L - 1, -1, -1):
                G[k] = bv; bv = rews[k] + g * bv
            # imagined children of every root: dynamics under every action
            best_im = np.zeros(L - 1)
            for lo in range(0, L - 1, 32):
                idx = np.arange(lo, min(lo + 32, L - 1))
                nxt, rw, _ = dyn(lat[idx].repeat_interleave(A, 0), torch.arange(A, device=dev).repeat(len(idx)))
                _, nv, _ = pred(nxt)
                comp = (rw.reshape(len(idx), A) if rw is not None else 0) + g * nv.reshape(len(idx), A)
                best_im[idx] = comp.max(1).values.cpu().numpy() if torch.is_tensor(comp) else comp.max(1)
            for t in range(L - 1):
                chain_comp = rews[t + 1] + g * G[t + 1]
                rows.append((t % 16, pol[t, acts[t]], float(-(pol[t] * np.log(pol[t] + 1e-12)).sum()), v[t], G[t],
                             chain_comp, best_im[t], chain_comp - best_im[t]))
            # frame motion by region: rows 0-20 (score), 21-84 split left/middle/right thirds
            for t in range(1, L):
                d = np.abs(obs[t, 9:12].astype(np.int16) - obs[t, 6:9].astype(np.int16)).mean(0)
                k = t % 16
                reg[k] += np.array([d[:20].mean(), d[20:, :28].mean(), d[20:, 28:56].mean(), d[20:, 56:].mean()]); regn[k] += 1
    r = np.array(rows, dtype=float)
    print(f"records {len(r)}")
    print(" slot  p(a_stored)  H(pi)   v(net)   G(MC)   chain_comp  best_imag  chain-best")
    for k in range(16):
        s = r[r[:, 0] == k]
        print(f"  {k:>2}    {s[:,1].mean():.3f}     {s[:,2].mean():.3f}  {s[:,3].mean():+.3f}  {s[:,4].mean():+.3f}   {s[:,5].mean():+.3f}     {s[:,6].mean():+.3f}    {s[:,7].mean():+.3f}")
    print(" frame motion |f3-f2| by slot and region (score-band, left, middle, right):")
    for k in range(16):
        m = reg[k] / max(regn[k], 1)
        print(f"  {k:>2}: " + " ".join(f"{x:.3f}" for x in m))
    print("DONE", flush=True)
