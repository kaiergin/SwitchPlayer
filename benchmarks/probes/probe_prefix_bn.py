"""PB read (1): the eval-mode step-1 value prefix against its train-mode-BN
value (CURRICULUM.md "PB: per-step prefix BatchNorm statistics on the token
arm"; Step 0 (iii)'s probe, benchmarks/probes/step0_prefix_bn/xrun_prefix.py,
without its search half). Read-only on the snapshots; CPU only.

usage:
  python probe_prefix_bn.py --states STATES.npz SNAPSHOT.pth [SNAPSHOT ...]
  python probe_prefix_bn.py --collect SNAPSHOT.pth --save STATES.npz
         [--games 8] [--keep 480] [--seed 4545]
  --env ENV_ID (default MsPacmanNoFrameskip-v4), --out FILE.jsonl (also
  append each record there).

Per snapshot, under the snapshot's OWN arm (its weights file's spec), on a
fixed state set (Step 0 (iii)'s 480 states: an .npz with `obs`):
  * p1_abs_eval: mean |step-1 prefix| from a fresh LSTM state after the
    snapshot's prior-argmax action, eval-mode BN -- the search's function:
    the arm's own step-1 statistics under PREFIX_BN_PER_STEP, the pooled
    running statistics otherwise;
  * p1_abs_trainBN: the same with train-mode BN (batch statistics of the
    same batch, on a deep copy -- what each training call sees);
    p1_ratio = p1_abs_eval / p1_abs_trainBN; fixed = p1_ratio <= 2 (PB's
    FIXED line; TK-90: 8-24x from row 55);
  * p1_allactions_abs_eval: the root expansion's step-1 prefix over every
    action, eval mode, on every 4th state;
  * normout_offset_sd_step{1,2}: the post-LSTM BatchNorm's input batch mean
    at step 1 / 2 against the statistics eval mode normalizes that step
    with (the arm's step set when on, pooled when off), in their running
    SDs, averaged over channels; *_pooled always against the pooled ones.
One JSON line per snapshot on stdout. --collect plays whole games with the
snapshot's DEPLOYED prior (its acting_states when present) at the actor's
temperature for its training step, as probe_qspread.py's collect did, and
saves `keep` acted states evenly spaced over every life."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # "" deletes the var on Windows
import argparse
import contextlib
import copy
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import numpy as np
import torch
import muzero
import ablation
torch.set_num_threads(int(os.environ.get("PROBE_THREADS", "2")))

SNAP_ARM = "__probe_prefix_bn_snapshot__"


def env_actions(env_id):
    import gymnasium as gym
    probe = gym.make(env_id)
    try:
        return int(probe.action_space.n)
    finally:
        probe.close()


def snapshot_arm(blob):
    """The snapshot's spec as an arm (TOKEN_VOCAB as tuples), minus keys
    this code's muzero lacks (returned, so a record shows them)."""
    spec = dict(blob["spec"])
    if spec.get("TOKEN_VOCAB") is not None:
        spec["TOKEN_VOCAB"] = tuple(tuple(t) for t in spec["TOKEN_VOCAB"])
    missing = [k for k in spec
               if not k.startswith("_") and not hasattr(muzero, k)]
    for k in missing:
        spec.pop(k)
    return spec, missing


@contextlib.contextmanager
def snapshot_run(blob, acting=False):
    """A Run under the snapshot's own arm with its weights loaded (eval
    mode); acting=True overlays its acting_states (the deployed prior)."""
    spec, missing = snapshot_arm(blob)
    ablation.ARMS[SNAP_ARM] = spec
    try:
        with ablation.arm_config(SNAP_ARM), \
                ablation._run_length_globals(90, 2000), \
                ablation.token_actions():
            run = ablation.Run(SNAP_ARM, 0, spec)
            try:
                for n in ablation.NET_NAMES:
                    getattr(run.trainer, n).load_state_dict(blob[n])
                renorm = spec.get("LATENT_RENORMALIZE", True)
                run.trainer.representation.renormalize_output = renorm
                run.trainer.dynamics.renormalize_output = renorm
                run.trainer.training_steps = int(blob.get("training_steps", 0))
                run.searcher.network_manager.training_steps = \
                    run.trainer.training_steps
                if acting and blob.get("acting_states"):
                    for n, sd in blob["acting_states"].items():
                        getattr(run.trainer, n).load_state_dict(sd)
                run.set_train_mode(False)
                yield run, missing
            finally:
                run.env.close()
    finally:
        del ablation.ARMS[SNAP_ARM]


def offset_sd(x, mean, var):
    """Mean over channels of |batch mean - mean| / running SD."""
    sd = var.sqrt().clamp(min=1e-5)
    return float(((x.mean(0) - mean).abs() / sd).mean())


def probe(path, obs):
    blob = torch.load(str(path), map_location="cpu", weights_only=False)
    with snapshot_run(blob) as (run, missing):
        rep, dyn, pred, vp = (run.trainer.representation, run.trainer.dynamics,
                              run.trainer.prediction, run.trainer.value_prefix)
        K = getattr(vp, "per_step", 0)
        rec = {"snapshot": str(path), "arm": blob.get("arm"),
               "iteration": blob.get("iteration"),
               "training_steps": blob.get("training_steps"),
               "A": ablation.ACTIONS, "n_states": int(len(obs)),
               "per_step": K, "missing_spec_keys": missing}
        if getattr(muzero, "REWARD_GROWTH", False):
            rec["ceiling_at_steps"] = float(muzero.reward_ceiling(
                int(blob.get("training_steps", 0))))
        lstm_out = {}
        hook = vp.lstm.register_forward_hook(
            lambda m, i, o: lstm_out.__setitem__("x", o[0].squeeze(0)))
        try:
            with torch.no_grad():
                x = torch.tensor(obs, dtype=torch.float32) / 255.0
                l0 = rep(x)
                lg0 = pred(l0)[0]
                l1, _, _ = dyn(l0, lg0.argmax(1))
                l2, _, _ = dyn(l1, pred(l1)[0].argmax(1))
                for mode in ("eval", "trainBN"):
                    net = vp if mode == "eval" else copy.deepcopy(vp).train()
                    s0 = net.initial_state(len(obs), "cpu")
                    p1, _, s1 = net(l1, s0, step=1)
                    p2, _, _ = net(l2, s1, step=2)
                    p1, p2 = p1.squeeze(-1).numpy(), p2.squeeze(-1).numpy()
                    rec[f"p1_mean_{mode}"] = float(p1.mean())
                    rec[f"p1_abs_{mode}"] = float(np.abs(p1).mean())
                    rec[f"p2_mean_{mode}"] = float(p2.mean())
                    rec[f"edge2_mean_{mode}"] = float((p2 - p1).mean())
                # The post-LSTM BN's input at steps 1 and 2 (eval path: the
                # input-side BN on the search's statistics too).
                s0 = vp.initial_state(len(obs), "cpu")
                _, _, s1 = vp(l1, s0, step=1)
                x1 = lstm_out["x"]
                vp(l2, s1, step=2)
                x2 = lstm_out["x"]
                # The root expansion: every action, step 1, eval mode.
                sub = l0[::4]
                A = ablation.ACTIONS
                n1, _, _ = dyn(sub.repeat_interleave(A, 0),
                               torch.arange(A).repeat(len(sub)))
                pall, _, _ = vp(n1, vp.initial_state(len(n1), "cpu"), step=1)
                rec["p1_allactions_abs_eval"] = float(pall.abs().mean())
        finally:
            hook.remove()
        bn = vp.norm_out
        for s, xs in ((1, x1), (2, x2)):
            rec[f"normout_offset_sd_step{s}_pooled"] = offset_sd(
                xs, bn.running_mean, bn.running_var)
            if K:
                rec[f"normout_offset_sd_step{s}"] = offset_sd(
                    xs, vp.norm_out_step_mean[s - 1],
                    vp.norm_out_step_var[s - 1])
            else:
                rec[f"normout_offset_sd_step{s}"] = \
                    rec[f"normout_offset_sd_step{s}_pooled"]
        rec["p1_ratio"] = rec["p1_abs_eval"] / max(rec["p1_abs_trainBN"],
                                                   1e-12)
        rec["fixed"] = bool(rec["p1_ratio"] <= 2.0)
    return rec


def collect(path, save, games, keep, seed):
    """probe_qspread.py's collect on any snapshot: whole games with the
    deployed prior at the actor's temperature for the snapshot's step."""
    blob = torch.load(str(path), map_location="cpu", weights_only=False)
    np.random.seed(seed)
    torch.manual_seed(seed)
    with snapshot_run(blob, acting=True) as (run, _):
        obs, meta = [], []
        for g in range(games):
            run.env.was_real_done = True
            run.env.reset(seed=seed + g)
            life = 0
            while True:
                traj, ret, steps, term = run.play_episode(greedy=False,
                                                          cap=3000)
                for i, r in enumerate(traj[:-1]):
                    nxt = traj[i + 1][2]
                    obs.append(r[0])
                    meta.append((g, life, i, int(r[1]),
                                 float(np.sum(nxt)) if np.size(nxt) else 0.0))
                life += 1
                if not term or getattr(run.env, "was_real_done", True):
                    break
    obs = np.stack(obs)
    idx = np.unique(np.linspace(0, len(obs) - 1, keep).round().astype(int))
    np.savez_compressed(save, obs=obs[idx],
                        meta=np.array(meta, dtype=np.float64)[idx],
                        n_total=len(obs))
    print(json.dumps({"collected": str(save), "kept": int(len(idx)),
                      "of": int(len(obs)), "snapshot": str(path)}),
          flush=True)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshots", nargs="*")
    ap.add_argument("--states")
    ap.add_argument("--collect")
    ap.add_argument("--save")
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--keep", type=int, default=480)
    ap.add_argument("--seed", type=int, default=4545)
    ap.add_argument("--env", default="MsPacmanNoFrameskip-v4")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    ablation.ENV_NAME, ablation.FRAMESKIP, ablation.EPISODE_CAP = \
        args.env, 4, 3000
    ablation.ACTIONS = env_actions(args.env)
    if args.collect:
        if not args.save:
            raise SystemExit("--collect needs --save")
        collect(args.collect, args.save, args.games, args.keep, args.seed)
        return
    if not args.states or not args.snapshots:
        raise SystemExit("need --states and at least one snapshot")
    obs = np.load(args.states)["obs"]
    for p in args.snapshots:
        t0 = time.time()
        rec = probe(pathlib.Path(p), obs)
        rec["secs"] = round(time.time() - t0)
        line = json.dumps(rec)
        print(line, flush=True)
        if args.out:
            with open(args.out, "a") as f:
                f.write(line + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
