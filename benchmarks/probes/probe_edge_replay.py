"""Emulator replay of a stored r12 episode: rebuild the paper env stack
(raw Pong-v4 frameskip=1 -> n noops -> MaxAndSkip(4)), recover n by
matching the first stored frame, then step the RECORDED actions and
compare every produced frame (preprocessed) with the stored newest frame
and every clipped reward with the stored one. Exact match everywhere =
the records' actions/observations/rewards are aligned (no actor-side
staleness); a mismatch localizes a recording bug."""
import sys, pathlib, glob
ROOT = pathlib.Path(r"Y:\SwitchPlayer"); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import numpy as np, gymnasium as gym, ablation, muzero
from atari_wrappers import MaxAndSkipEnv
CK = glob.glob(r"Y:\SwitchPlayer\benchmarks\results\pong-r12-2026-09-14\resume-*.ckpt")[0]
replay = ablation._read_checkpoint(CK)["replay"]
pre = lambda o: muzero.preprocess_frame(o, muzero.IMAGE_RESHAPE)
for ti, traj in enumerate(replay[:3]):
    first = traj[0]["obs"][9:12]
    found = None
    for n in range(1, 31):
        raw = gym.make("Pong-v4", frameskip=1)
        obs, _ = raw.reset(seed=0)
        for _ in range(n):
            obs, *_ = raw.step(0)
        if np.array_equal(pre(obs), first):
            found = n; break
        raw.close()
    if found is None:
        print(f"traj {ti}: no noop count 1..30 reproduces record 0 (seed-dependent reset?) -- skipping"); continue
    env = MaxAndSkipEnv(raw, skip=4)
    env._buffer[0] = obs; env._buffer[1] = obs
    mism = []; rew_mism = []; slots = {k: [0, 0] for k in range(16)}
    for t in range(len(traj) - 1):
        a = int(traj[t]["action"])
        obs, r, term, trunc, _ = env.step(a)
        ok = np.array_equal(pre(obs), traj[t + 1]["obs"][9:12])
        slots[(t + 1) % 16][0] += int(not ok); slots[(t + 1) % 16][1] += 1
        if not ok: mism.append(t + 1)
        if float(np.sign(r)) != float(traj[t + 1]["reward"]): rew_mism.append(t + 1)
        if term or trunc: break
    print(f"traj {ti}: len {len(traj)}, noops {found}, first frame mismatch at {mism[:5] if mism else 'none'} ({len(mism)} total), reward mismatches {len(rew_mism)} {rew_mism[:5]}")
    print("   frame-mismatch rate by slot: " + " ".join(f"{m/max(n,1):.2f}" for m, n in (slots[k] for k in range(16))))
    raw.close()
print("DONE", flush=True)
