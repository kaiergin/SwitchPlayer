"""BatchNorm recalibration probe (2026-09-29, Kai: option B after the
prioract readout). Both the prioract run (@70 -20.8, @90 -16.8) and the
faithful control (s1 @85 -14.9) collapse at single snapshots where the
prediction head's BN running variance is ~1.5x the ROOT batch variance
(probe_prior_bn_mismatch): the running statistics are averaged over the
training unroll depths 0..K, but prior-argmax acting only ever sees depth-0
root latents.

Test: re-estimate every BN running statistic of the REPRESENTATION and
PREDICTION nets (the only nets prior-argmax acting runs) from root-only
forward passes over a COMMON calibration set -- the frames of a reference
snapshot's greedy games at noops 1..4 (all four noop phases) -- then rerun
the EXACT eval (all 30 noop counts). If the collapsed snapshots recover,
depth-0 BN recalibration is a cheap stabilizer for acting.

usage: python probe_bn_recal_eval.py CALIB_SNAPSHOT.pth SNAPSHOT.pth [...]
Appends to probe_bn_recal_eval-<run>.jsonl next to itself. CPU only;
PROBE_THREADS (default 2)."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # "" deletes the var on Windows
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import numpy as np                       # noqa: E402
import torch                             # noqa: E402
import muzero                            # noqa: E402
import ablation                          # noqa: E402
torch.set_num_threads(int(os.environ.get("PROBE_THREADS", "2")))

HERE = pathlib.Path(__file__).parent
CALIB_NOOPS = (1, 2, 3, 4)


class pe:
    """probe_exhaustive_eval's helpers, copied: that module has no main
    guard, so importing it would run its eval loop on this argv."""
    ARM = "effzero-repro-faithful-prioract"          # policy-only eval

    @staticmethod
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

    @staticmethod
    def _noop_env(env):
        e = env
        while not isinstance(e, __import__("atari_wrappers").NoopResetEnv):
            e = e.env
        return e

    @staticmethod
    def play(run, noops):
        ne = pe._noop_env(run.env)

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


ablation.ENV_NAME, ablation.FRAMESKIP, ablation.EPISODE_CAP, ablation.ACTIONS =     "Pong-v4", 4, 3000, 6


def bn_layers(run):
    return [m for n in ("representation", "prediction")
            for m in getattr(run.trainer, n).modules()
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]


def collect_frames(run):
    """Observations of the loaded snapshot's greedy games at CALIB_NOOPS."""
    frames = []
    ne = pe._noop_env(run.env)
    for noops in CALIB_NOOPS:
        def fixed(self, _n=noops, **kw):
            obs, info = self.env.reset(**kw)
            for _ in range(_n):
                obs, _, _, _, info = self.env.step(0)
            return obs, info
        ne.reset = fixed.__get__(ne)
        try:
            run.env.reset(seed=777)
            traj, _, _, _ = run.play_episode(greedy=True, cap=3000)
        finally:
            del ne.reset
        frames += [rec[0] for rec in traj]
    return np.stack(frames)


@torch.no_grad()
def prior_entropy(run, frames):
    x = torch.tensor(frames[:2048], dtype=torch.float32) / 255.0
    logits, _, _ = run.trainer.prediction(run.trainer.representation(x))
    p = torch.softmax(logits.float(), dim=1)
    return float(-(p * torch.log(p + 1e-12)).sum(1).mean())


@torch.no_grad()
def recalibrate(run, frames, batch=256):
    """Cumulative (momentum=None) running statistics over root-only
    forwards in train mode; everything else in eval mode afterwards."""
    layers = bn_layers(run)
    saved = [m.momentum for m in layers]
    for m in layers:
        m.reset_running_stats()
        m.momentum = None
    rep, pred = run.trainer.representation, run.trainer.prediction
    rep.train(); pred.train()
    for lo in range(0, len(frames), batch):
        x = torch.tensor(frames[lo:lo + batch], dtype=torch.float32) / 255.0
        pred(rep(x))
    for m, mom in zip(layers, saved):
        m.momentum = mom
    run.set_train_mode(False)


def exact(run):
    games = {n: pe.play(run, n) for n in range(1, 31)}
    rets = np.array([games[n][0] for n in range(1, 31)])
    phase = {k: float(np.mean([games[n][0] for n in games if n % 4 == k]))
             for k in range(4)}
    return rets, phase, games


with ablation.arm_config(pe.ARM), ablation._run_length_globals(90, 2000):
    run = ablation.Run(pe.ARM, 0, ablation.ARMS[pe.ARM])
    assert muzero.USE_SEARCH_ACTING is False       # greedy = prior argmax
    calib_path = pathlib.Path(sys.argv[1])
    pe.load(run, calib_path)
    t0 = time.time()
    frames = collect_frames(run)
    print(f"calibration set: {len(frames)} frames from {calib_path.parent.parent.name}"
          f" @{calib_path.stem[-3:]} noops {CALIB_NOOPS} [{time.time() - t0:.0f}s]",
          flush=True)
    for arg in sys.argv[2:]:
        path = pathlib.Path(arg)
        pe.load(run, path)
        t0 = time.time()
        h_before = prior_entropy(run, frames)
        recalibrate(run, frames)
        h_after = prior_entropy(run, frames)
        rets, phase, games = exact(run)
        rec = {"snapshot": str(path), "run": path.parent.parent.name,
               "iteration": int(path.stem[-3:]) if path.stem[-3:].isdigit() else None,
               "calibration": str(calib_path), "calib_frames": len(frames),
               "H_eval_before": round(h_before, 4), "H_eval_after": round(h_after, 4),
               "exact_mean": round(float(rets.mean()), 3),
               "sd": round(float(rets.std()), 2), "min": float(rets.min()),
               "phase_means": phase,
               "phases_won": sum(1 for v in phase.values() if v > 0),
               "capped": sum(1 for g in games.values() if not g[2]),
               "games": {n: list(g) for n, g in games.items()},
               "secs": round(time.time() - t0)}
        print(f"{rec['run']} @{rec['iteration']}: RECAL exact {rec['exact_mean']:+.2f} "
              f"(sd {rec['sd']}, min {rec['min']:+.0f}) phases "
              f"{[round(v, 1) for v in phase.values()]} won {rec['phases_won']}/4 "
              f"capped {rec['capped']}/30 | prior H {h_before:.3f} -> {h_after:.3f} "
              f"[{rec['secs']}s]", flush=True)
        with open(HERE / f"probe_bn_recal_eval-{rec['run']}.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
print("DONE", flush=True)
