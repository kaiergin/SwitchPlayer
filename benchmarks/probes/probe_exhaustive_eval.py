"""EXACT prior-argmax eval of weight snapshots (2026-09-26, round-2 design
audit). Greedy policy-only play is a deterministic function of the NoopReset
count (verified: identical (return, length) across env seeds at a fixed
count), and the outcome depends mostly on noops mod 4 -- so an 8-game eval
row is really ~4-8 distinct games and single rows were off by up to +-6
points. Playing EVERY noop count 1..30 gives a snapshot's exact noop-start
expectation with zero sampling noise. This is the primary readout of the
prioract run and of its controls.

usage: python probe_exhaustive_eval.py SNAPSHOT.pth [SNAPSHOT.pth ...]
Writes probe_exhaustive_eval-<run>.jsonl (one line per snapshot, one file
per run so concurrent probes never share a file) next to itself; records
before 2026-09-27 are in probe_exhaustive_eval.jsonl.
CPU only; PROBE_THREADS (default 2) fixes the thread count so results are
reproducible. ~5.5 min per snapshot."""
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
import ablation
torch.set_num_threads(int(os.environ.get("PROBE_THREADS", "2")))
ARM = "effzero-repro-faithful-prioract"      # policy-only eval semantics
ablation.ENV_NAME, ablation.FRAMESKIP, ablation.EPISODE_CAP, ablation.ACTIONS = \
    "Pong-v4", 4, 3000, 6
HERE = pathlib.Path(__file__).parent


def load(run, path):
    d = torch.load(str(path), map_location="cpu", weights_only=False)
    for n in ablation.NET_NAMES:
        getattr(run.trainer, n).load_state_dict(d[n])
    renorm = d.get("spec", {}).get("LATENT_RENORMALIZE", True)
    run.trainer.representation.renormalize_output = renorm
    run.trainer.dynamics.renormalize_output = renorm
    run.trainer.training_steps = d.get("training_steps", 0)
    run.searcher.network_manager.training_steps = run.trainer.training_steps
    run.set_train_mode(False)
    return d


def _noop_env(env):
    e = env
    while not isinstance(e, __import__("atari_wrappers").NoopResetEnv):
        e = e.env
    return e


def play(run, noops):
    ne = _noop_env(run.env)

    def fixed(self, **kw):
        obs, info = self.env.reset(**kw)
        for _ in range(noops):
            obs, _, _, _, info = self.env.step(0)
        return obs, info

    ne.reset = fixed.__get__(ne)
    try:
        run.env.reset(seed=777)
        _, ret, steps, term = run.play_episode(greedy=True, cap=3000)
    finally:
        del ne.reset
    return float(ret), int(steps), bool(term)


with ablation.arm_config(ARM), ablation._run_length_globals(90, 2000):
    run = ablation.Run(ARM, 0, ablation.ARMS[ARM])
    assert muzero.USE_SEARCH_ACTING is False       # greedy = prior argmax
    for arg in sys.argv[1:]:
        path = pathlib.Path(arg)
        load(run, path)
        t0 = time.time()
        games = {n: play(run, n) for n in range(1, 31)}
        rets = np.array([games[n][0] for n in range(1, 31)])
        phase = {k: float(np.mean([games[n][0] for n in games if n % 4 == k]))
                 for k in range(4)}
        rec = {"snapshot": str(path), "run": path.parent.parent.name,
               "iteration": int(path.stem[-3:]) if path.stem[-3:].isdigit() else None,
               "exact_mean": round(float(rets.mean()), 3),
               "sd": round(float(rets.std()), 2),
               "min": float(rets.min()),
               "phase_means": phase,
               "phases_won": sum(1 for v in phase.values() if v > 0),
               "capped": sum(1 for g in games.values() if not g[2]),
               "games": {n: list(g) for n, g in games.items()},
               "secs": round(time.time() - t0)}
        print(f"{rec['run']} @{rec['iteration']}: exact {rec['exact_mean']:+.2f} "
              f"(sd {rec['sd']}, min {rec['min']:+.0f}) phases "
              f"{[round(v, 1) for v in phase.values()]} won {rec['phases_won']}/4 "
              f"capped {rec['capped']}/30 [{rec['secs']}s]", flush=True)
        with open(HERE / f"probe_exhaustive_eval-{rec['run']}.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
print("DONE", flush=True)
