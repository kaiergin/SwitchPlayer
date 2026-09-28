"""SVE stage-0 probe (2026-09-28; CURRICULUM.md "Future experiment --
search-based value estimation"). Kai decided SVE is the next run, on top of
the prioract stack, with the paper's T1 = 4e4 / T2 = 2e4, and to run this
probe first.

Material: a COPY of a live run's resume checkpoint (the buffer, online and
target nets, training step). Read on the CPU so the live run keeps the GPU.
Never point this at the live file: on Windows an open handle makes the
run's atomic os.replace fail.

STALE positions -- outside the newest T2 transitions, EZ-V2's buffer-index
rule (the positions SVE would relabel). Four value targets per position:
  TD      the trainer's target today: the age-shortened horizon of real
          rewards, then the lagged target nets' value at the tail
          (finish_batch_paper), or the pure return if the episode ends
          inside the horizon;
  stored  the stored root_value -- SVE stage 1 (birth search: online nets
          + root noise; overwritten by each refresh: target nets, communal);
  fresh50 / fresh16  a fresh-root search through the target nets, no
          noise, 50 / 16 sims -- SVE stage 2.
Questions (pre-registered in CURRICULUM):
  1. Is |SVE - TD| big enough to matter?
  2. What does staleness cost (stored vs fresh)?
  3. On the NEWEST terminated trajectories (nearly on-policy): which estimate
     is closest to the realized discounted return?

usage: python probe_sve_stage0.py CKPT_COPY [N_STALE=2048] [N_NEW=3000]
Appends one JSON record to probe_sve_stage0.jsonl next to itself."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # "" deletes the var on Windows
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import numpy as np
import torch
import muzero
import muzero_vtree
import ablation

torch.set_num_threads(int(os.environ.get("PROBE_THREADS", "4")))
ARM = "effzero-repro-faithful-prioract"
ablation.ENV_NAME, ablation.FRAMESKIP, ablation.EPISODE_CAP, ablation.ACTIONS = \
    "Pong-v4", 4, 3000, 6
# SVE fresh window, transitions (Kai 2026-09-24); PROBE_T2 only for
# dry runs on a not-yet-full buffer.
T2 = int(os.environ.get("PROBE_T2", "20000"))
OUT = pathlib.Path(__file__).with_name("probe_sve_stage0.jsonl")


def stats(est, ref):
    d = np.asarray(est, dtype=np.float64) - np.asarray(ref, dtype=np.float64)
    c = (float(np.corrcoef(est, ref)[0, 1])
         if np.std(est) > 0 and np.std(ref) > 0 else float("nan"))
    return {"bias": round(float(d.mean()), 4),
            "mae": round(float(np.abs(d).mean()), 4),
            "rmse": round(float(np.sqrt((d ** 2).mean())), 4),
            "corr": round(c, 4)}


def realized_return(traj, pos, discount):
    """sum_{k>=1} g^(k-1) r_{pos+k} to the end of a TERMINATED episode --
    the quantity v(s_pos) estimates (paper_value_parts' reward indexing)."""
    g, z = 1.0, 0.0
    for k in range(pos + 1, len(traj)):
        z += g * traj[k]["reward"]
        g *= discount
    return z


def main(ckpt, n_stale, n_new):
    t_start = time.time()
    state = ablation._read_checkpoint(pathlib.Path(ckpt))
    rec = {"ckpt": str(ckpt), "arm": state.get("arm"), "t2": T2,
           "training_steps": state["training_steps"],
           "env_steps": state["env_steps"]}
    spec = ablation.ARMS[ARM]
    iters = state.get("iterations", 90)
    tsteps = state.get("train_steps_per_iter", 2000)
    with ablation.arm_config(ARM), ablation._run_length_globals(iters, tsteps), \
            torch.no_grad():
        run = ablation.Run(ARM, 0, spec)
        tr = run.trainer
        for n in ablation.NET_NAMES:
            getattr(tr, n).load_state_dict(state["nets"][n])
        # Target model exactly as load_run_checkpoint restores it (nets,
        # double-buffer snapshot, sync stamp), so the trainer's own batch
        # builder below sees the state the live trainer had.
        tr._ensure_target_model()
        for n in ablation.NET_NAMES[:4]:
            getattr(tr, f"target_{n}").load_state_dict(state["target"]["nets"][n])
        tr._target_snapshot = state["target"]["snapshot"]
        tr._target_synced_at = state["target"]["synced_at"]
        tr.training_steps = ts = state["training_steps"]
        run.replay = replay = state["replay"]
        disc = muzero.DISCOUNT

        # Buffer order is FIFO by trajectory (oldest first); a position's
        # recency counts transitions from the buffer's end.
        flat = [(ti, p) for ti, traj in enumerate(replay) for p in range(len(traj))]
        n_total = len(flat)
        stale_all = [(ti, p) for ti, p in flat[:max(0, n_total - T2)]
                     if p < len(replay[ti]) - 1]          # acted positions
        rec.update(buffer_transitions=n_total, trajectories=len(replay),
                   stale_positions=len(stale_all),
                   stale_frac=round(len(stale_all) / max(1, n_total), 4))
        print(f"buffer {n_total} transitions in {len(replay)} trajectories; "
              f"stale acted positions {len(stale_all)} "
              f"({rec['stale_frac']:.1%}); training step {ts}", flush=True)

        rng = np.random.default_rng(0)
        pick = (rng.choice(len(stale_all), size=min(n_stale, len(stale_all)),
                           replace=False) if stale_all else [])
        stale = [stale_all[i] for i in sorted(pick)]

        # Newest TERMINATED trajectories, newest first, up to n_new positions.
        newest = []
        for ti in range(len(replay) - 1, -1, -1):
            traj = replay[ti]
            if traj[-1].get("truncated"):
                continue
            newest += [(ti, p) for p in range(len(traj) - 1)]
            if len(newest) >= n_new:
                break
        rec["newest_trajectories"] = len({ti for ti, _ in newest})

        def td_targets(positions):
            """The trainer's TD target for each position as a window ROOT
            (assemble_batch_paper + finish_batch_paper, USE_TARGET_MODEL)."""
            out, boot = [], []
            for ti, p in positions:
                traj = replay[ti]
                age = max(0, ts - traj[p].get("born_step", ts))
                rsum, g, tail = muzero.paper_value_parts(
                    traj, p, muzero.paper_horizon(age), disc)
                terminal = tail == len(traj) - 1 and not traj[tail].get("truncated")
                out.append(rsum)
                boot.append(None if terminal else (g, traj[tail]["obs"]))
            idx = [i for i, b in enumerate(boot) if b is not None]
            for lo in range(0, len(idx), 256):
                chunk = idx[lo:lo + 256]
                x = torch.tensor(np.stack([boot[i][1] for i in chunk]),
                                 dtype=torch.float32) / 255.0
                _, v, _ = tr.target_prediction(tr.target_representation(x))
                for i, vi in zip(chunk, v.squeeze(-1).numpy()):
                    out[i] += boot[i][0] * float(vi)
            return np.array(out), np.array([b is None for b in boot])

        # SELF-CHECK: the probe's TD target must equal the trainer's own
        # (assemble_batch_paper + finish_batch_paper) for the same roots.
        # Run FIRST: finish_batch_paper may perform the pending target
        # sync, and everything below must then see the synced target nets
        # -- the ones the trainer's next step would use.
        where = {id(t): (ti, p) for ti, traj in enumerate(replay)
                 for p, t in enumerate(traj)}
        np.random.seed(123)
        b = tr.finish_batch_paper(tr.assemble_batch_paper(
            replay, 64, muzero.UNROLL_STEPS, ablation.ACTIONS))
        roots = [where[id(t)] for t in b["sample_refs"]]
        mine, _ = td_targets(roots)
        rec["td_selfcheck_max_abs_diff"] = float(
            np.abs(mine - b["target_value"][:, 0]).max())
        print(f"TD self-check vs the trainer's batch builder (64 roots): "
              f"max |diff| {rec['td_selfcheck_max_abs_diff']:.2e}", flush=True)
        assert rec["td_selfcheck_max_abs_diff"] < 1e-4, "TD target mismatch"
        tnets = {n: getattr(tr, f"target_{n}") for n in ablation.NET_NAMES[:4]}
        for net in tnets.values():
            assert not net.training
        engine = muzero_vtree.VectorizedPUCT(tnets, ablation.ACTIONS, tr.device)

        def fresh(positions, sims):
            vals = []
            for lo in range(0, len(positions), 256):
                obs = np.stack([replay[ti][p]["obs"] for ti, p in positions[lo:lo + 256]])
                _, v, _ = engine.search(obs, num_simulations=sims,
                                        c_puct=muzero.C_PUCT, discount=disc,
                                        root_noise=False)
                vals += [float(x) for x in v]
            return np.array(vals)

        for name, positions in (("stale", stale), ("newest", newest)):
            if not positions:
                rec[name] = None
                continue
            t0 = time.time()
            td, pure = td_targets(positions)
            stored = np.array([replay[ti][p]["root_value"] for ti, p in positions])
            f50 = fresh(positions, 50)
            f16 = fresh(positions, 16)
            ages = np.array([max(0, ts - replay[ti][p].get("born_step", ts))
                             for ti, p in positions])
            r = {"n": len(positions), "pure_return_frac": round(float(pure.mean()), 4),
                 "age_steps_p50": int(np.median(ages)),
                 "age_steps_max": int(ages.max()),
                 "scale": {k: {"mean": round(float(v.mean()), 4),
                               "std": round(float(v.std()), 4)}
                           for k, v in (("td", td), ("stored", stored),
                                        ("fresh50", f50), ("fresh16", f16))},
                 "vs_td": {k: stats(v, td) for k, v in (("stored", stored),
                           ("fresh50", f50), ("fresh16", f16))},
                 "stored_vs_fresh50": stats(stored, f50),
                 "fresh16_vs_fresh50": stats(f16, f50)}
            boot_rows = ~pure
            if boot_rows.any():
                r["vs_td_bootstrapped_only"] = {
                    k: stats(v[boot_rows], td[boot_rows])
                    for k, v in (("stored", stored), ("fresh50", f50))}
            if name == "newest":
                g = np.array([realized_return(replay[ti], p, disc)
                              for ti, p in positions])
                r["scale"]["realized"] = {"mean": round(float(g.mean()), 4),
                                          "std": round(float(g.std()), 4)}
                r["vs_realized"] = {k: stats(v, g) for k, v in (
                    ("td", td), ("stored", stored), ("fresh50", f50),
                    ("fresh16", f16))}
            r["secs"] = round(time.time() - t0)
            rec[name] = r
            print(f"== {name}: n {r['n']}, pure-return TD {r['pure_return_frac']:.0%}, "
                  f"age p50 {r['age_steps_p50']} [{r['secs']}s]", flush=True)
            print("   scale " + json.dumps(r["scale"]), flush=True)
            print("   vs TD " + json.dumps(r["vs_td"]), flush=True)
            print("   stored vs fresh50 " + json.dumps(r["stored_vs_fresh50"])
                  + "  fresh16 vs fresh50 " + json.dumps(r["fresh16_vs_fresh50"]),
                  flush=True)
            if "vs_realized" in r:
                print("   vs REALIZED " + json.dumps(r["vs_realized"]), flush=True)
    rec["secs"] = round(time.time() - t_start)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"DONE [{rec['secs']}s]", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2048,
         int(sys.argv[3]) if len(sys.argv) > 3 else 3000)
