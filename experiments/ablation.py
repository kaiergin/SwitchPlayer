"""Long-running A/B over the changes made to this repo.

Runs the real loop -- self-play, MCTS re-analysis, training -- for each arm and
seed, and reports episode return alongside prediction errors that are always
scored against a trivial control.

    python experiments/ablation.py --arms per-step value-prefix --seeds 3
    python experiments/ablation.py --plot-only          # re-draw from results/

Design notes, mostly learned the hard way (see benchmarks/RESULTS.md):

* **Every prediction metric carries a predict-zero control.** Three earlier
  measurements in this repo looked conclusive and were not: a head that
  collapses to a constant beats a head that is merely noisy, and without the
  control that reads as a win.
* **Episode return is the primary metric.** It is the only one that cannot be
  gamed by a degenerate predictor.
* **Seeds, not runs.** Single-seed differences on Atari are noise. Arms share
  seeds so they see the same environment stream.
* **Single process, no Ray.** The actors share module-level constants, which is
  exactly what an ablation needs to vary. This drives the same classes directly
  so each arm is configured in isolation and the schedule is deterministic.
* Results append to JSONL after every evaluation, so a run that dies partway is
  still worth something and can be re-plotted.
"""
import argparse
import asyncio
import contextlib
import json
import pathlib
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import muzero
from muzero import (C_PUCT, IMAGE_RESHAPE, LATENT_CHANNELS, MCTS_BATCHES, MCTSHelper,
                    TOTAL_MCTS_SIMS, TrainingHelper, initial_frame_stack,
                    preprocess_frame, push_frame)
from networks import (DynamicsNetwork, LatentProjector, PredictionNetwork,
                      RepresentationNetwork, ValuePrefixNetwork)

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "benchmarks" / "results"
# Environment defaults. MsPacman keeps its historical 5-action restriction and
# 600-step (single-life) episode cap; other games take the env's full action
# space and their own cap via --env / --actions / --episode-cap. TAG suffixes
# every output file for a non-default env, so Pong results never mix with the
# MsPacman history in ablation.jsonl.
ENV_NAME = "MsPacman-v4"
ACTIONS = 5
EPISODE_CAP = 600
TAG = ""
# --frameskip N overrides the env's registered frameskip (e.g. Pong-v4 with
# frameskip=4 = the benchmark's fixed action-repeat; the registered default is
# the stochastic (2,5) skip, which the Atari-100k protocol does not use). This
# ale-py registers no *Deterministic-v4 ids, hence the kwarg route.
FRAMESKIP = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------
# Each arm is a set of module-level overrides applied for the duration of a run.
# Keeping them declarative means the diff between two arms is visible here
# rather than buried in the loop.
ARMS = {
    "per-step": {
        "USE_VALUE_PREFIX_IN_TREE": False,
        "_doc": "MuZero baseline: the tree reads the per-step reward head.",
    },
    "value-prefix": {
        "USE_VALUE_PREFIX_IN_TREE": True,
        "_doc": "EfficientZero 4.2: the tree reads the value-prefix LSTM.",
    },
    "no-transform": {
        "USE_VALUE_PREFIX_IN_TREE": False,
        "USE_H_TRANSFORM": False,
        "_linear_support": True,
        "_doc": "Pre-h() encoding: linear support, 51 bins, h() off on encode "
                "AND decode. (The archived first run of this arm was mislabeled: "
                "only the supports changed and h() still applied.)",
    },
    "pong-kit2": {
        "USE_VALUE_PREFIX_IN_TREE": True,
        "DISCOUNT": 0.997,
        "USE_GUMBEL_SEARCH": True,
        "ACTING_TEMPERATURE": 1.5,
        "LEARNING_RATE": 3e-4,
        "MAX_REPLAY": 60,
        "USE_AUGMENTATION": True,
        "TD_STEPS": 5,
        "REANALYSE_STRIDE": 2,
        "TRAINING_BATCHES": 128,
        "LR_DECAY_AT": 0.75,
        "_doc": "pong-kit plus the remaining published-config numbers: "
                "EfficientZero's td_steps=5, doubled policy-target density "
                "(stride 2), batch 128 (the 4070's honest step toward their "
                "256), and a single x0.1 LR decay at 75% of the run. Fires "
                "only if pong-kit floors -- completing the protocol, not "
                "guessing. Same run flags as pong-kit.",
    },
    "gumbel": {
        "USE_VALUE_PREFIX_IN_TREE": True,
        "DISCOUNT": 0.9,
        "USE_GUMBEL_SEARCH": True,
        "_doc": "horizon-0.9 configuration with Gumbel reanalysis targets "
                "(softmax(logits + sigma(completedQ)), sequential halving).",
    },
    "puct-search": {
        "USE_VALUE_PREFIX_IN_TREE": True,
        "DISCOUNT": 0.9,
        "USE_GUMBEL_SEARCH": False,
        "_doc": "Identical to the gumbel arm except the search: classic PUCT "
                "visit-count targets with Dirichlet noise. The controlled pair.",
    },
    "act-temp-1.5": {
        "USE_VALUE_PREFIX_IN_TREE": True,
        "DISCOUNT": 0.9,
        "USE_GUMBEL_SEARCH": True,
        "ACTING_TEMPERATURE": 1.5,
        "_doc": "gumbel arm plus a mild acting-entropy floor. Measured: sharp "
                "Gumbel targets collapsed behaviour entropy (H 1.61 -> 0.78) "
                "and returns sagged after an early peak -- the acting path IS "
                "the exploration here, and it narrowed with the policy.",
    },
    "act-temp-2.0": {
        "USE_VALUE_PREFIX_IN_TREE": True,
        "DISCOUNT": 0.9,
        "USE_GUMBEL_SEARCH": True,
        "ACTING_TEMPERATURE": 2.0,
        "_doc": "gumbel arm plus a stronger acting-entropy floor; brackets the "
                "knob with act-temp-1.5.",
    },
    "pong-100k": {
        "USE_VALUE_PREFIX_IN_TREE": True,
        "DISCOUNT": 0.997,
        "USE_GUMBEL_SEARCH": True,
        "ACTING_TEMPERATURE": 1.5,
        "_doc": "Equivalence check against the Atari-100k table (Pong: random "
                "-20.7, SimPLe 12.8, MuZero@100k -6.7, EfficientZero 20.6). "
                "Best current config, with gamma back at 0.997 -- a Pong rally "
                "spans seconds, and gamma=0.9's ~10-decision horizon is "
                "shorter than one ball crossing. Run with --env Pong-v4 "
                "--frameskip 4 --episode-cap 3000 --episodes-per-iter 1. "
                "The --frameskip 4 is mandatory: the benchmark protocol is "
                "fixed skip-4, while Pong-v4's registered stochastic (2,5) "
                "skip randomises action duration on a timing game -- the "
                "first Pong runs carried that confound.",
    },
    # Learning-rate sweep at the benchmark's replay ratio (~1 update/env-step,
    # i.e. --train-steps 1000): Adam 1e-3 was chosen to fix a bug, never tuned,
    # and it interacts directly with replay ratio -- too hot overfits the small
    # buffer, too cold under-trains, and either corrupts the equivalence test's
    # conclusion. Judge on the low-noise metrics (head MAE vs control, policy
    # entropy trajectory), not returns.
    "pong-lr-3e-4": {
        "USE_VALUE_PREFIX_IN_TREE": True,
        "DISCOUNT": 0.997,
        "USE_GUMBEL_SEARCH": True,
        "ACTING_TEMPERATURE": 1.5,
        "LEARNING_RATE": 3e-4,
        "USE_AUGMENTATION": False,
        "_doc": "pong-100k at Adam lr 3e-4, no augmentation -- the pre-kit "
                "baseline (its corrected-build run predates augmentation, so "
                "the flag is pinned off here to keep the arm meaning fixed).",
    },
    "pong-kit": {
        "USE_VALUE_PREFIX_IN_TREE": True,
        "DISCOUNT": 0.997,
        "USE_GUMBEL_SEARCH": True,
        "ACTING_TEMPERATURE": 1.5,
        "LEARNING_RATE": 3e-4,
        "MAX_REPLAY": 60,
        "USE_AUGMENTATION": True,
        "_doc": "pong-lr-3e-4 plus the data-efficiency kit's two biggest "
                "levers: 60-trajectory replay (roughly the whole 100k-step "
                "run stays in the buffer, ~9GB RAM of frame stacks) and DrQ "
                "shift+intensity augmentation. Run with --env Pong-v4 "
                "--frameskip 4 --episode-cap 3000 --episodes-per-iter 1.",
    },
    "pong-lr-1e-3": {
        "USE_VALUE_PREFIX_IN_TREE": True,
        "DISCOUNT": 0.997,
        "USE_GUMBEL_SEARCH": True,
        "ACTING_TEMPERATURE": 1.5,
        "LEARNING_RATE": 1e-3,
        "_doc": "pong-100k at Adam lr 1e-3 (the current default; control).",
    },
    "pong-lr-3e-3": {
        "USE_VALUE_PREFIX_IN_TREE": True,
        "DISCOUNT": 0.997,
        "USE_GUMBEL_SEARCH": True,
        "ACTING_TEMPERATURE": 1.5,
        "LEARNING_RATE": 3e-3,
        "_doc": "pong-100k at Adam lr 3e-3.",
    },
    "pong-act-search": {
        "USE_VALUE_PREFIX_IN_TREE": True,
        "DISCOUNT": 0.997,
        "USE_GUMBEL_SEARCH": True,
        "ACTING_TEMPERATURE": 1.5,
        "LEARNING_RATE": 3e-4,
        "USE_SEARCH_ACTING": True,
        "_doc": "pong-lr-3e-4 plus the paper's acting mode, TEST ONLY: 8-sim "
                "Gumbel search per self-play step, halving survivor acts (the "
                "Gumbel draw is the exploration; ACTING_TEMPERATURE unused on "
                "this branch). Greedy eval still never searches, so returns "
                "measure the deployable raw-policy agent. Tests the "
                "bootstrapping finding from the matched-budget run.",
    },
    "horizon-0.9": {
        "USE_VALUE_PREFIX_IN_TREE": True,
        "DISCOUNT": 0.9,
        "_doc": "Kai's fast-convergence arm. Horizon ~10 steps against MAX_DEPTH=8, "
                "so the tree is still meaningfully deep -- a depth-8 node keeps 43% "
                "of its reward, against 0.4% at gamma=0.5, where the search would be "
                "doing eight levels of work for an effectively depth-2 problem.",
    },
    "horizon-0.5": {
        "USE_VALUE_PREFIX_IN_TREE": True,
        "DISCOUNT": 0.5,
        "_doc": "Extreme control. Horizon ~2 steps, so the tree is effectively "
                "depth-2 and planning barely matters. Run only to bracket 0.9.",
    },
}


# Arms defined before augmentation existed keep their archived meaning: any
# arm that does not opt in explicitly runs WITHOUT augmentation, regardless of
# the muzero module default. Only pong-kit/pong-kit2 opt in. Without this, the
# USE_AUGMENTATION=True default retroactively changed every legacy arm --
# most sharply breaking the pong-act-search vs pong-lr-3e-4 controlled pair.
for _spec in ARMS.values():
    _spec.setdefault("USE_AUGMENTATION", False)


@contextlib.contextmanager
def arm_config(arm):
    """Apply an arm's overrides to the muzero module, then restore."""
    overrides = {k: v for k, v in ARMS[arm].items() if not k.startswith("_")}
    saved = {k: getattr(muzero, k) for k in overrides}
    for k, v in overrides.items():
        setattr(muzero, k, v)
    try:
        yield ARMS[arm]
    finally:
        for k, v in saved.items():
            setattr(muzero, k, v)


def build_networks(spec):
    """Fresh networks for one run. `_linear_support` reproduces the encoding
    that predated the h() transform: raw-ceiling supports AND transform=False
    on the decode side (the arm spec disables the encode side via
    USE_H_TRANSFORM). Changing only the supports, as the first run of this arm
    did, still applies h() and tests a mis-scaled encoding instead."""
    transform = not spec.get("_linear_support")
    if spec.get("_linear_support"):
        reward_sup, reward_bins = muzero.REWARD_CLIP, 51
        prefix_sup, prefix_bins = muzero.VALUE_PREFIX_CEILING, 51
        value_sup, value_bins = muzero.VALUE_CEILING, 51
    else:
        reward_sup, reward_bins = muzero.REWARD_SUPPORT, muzero.REWARD_BINS
        prefix_sup, prefix_bins = muzero.VALUE_PREFIX_SUPPORT, muzero.VALUE_PREFIX_BINS
        value_sup, value_bins = muzero.VALUE_SUPPORT, muzero.VALUE_BINS

    return dict(
        representation=RepresentationNetwork(input_channels=12,
                                             latent_channels=LATENT_CHANNELS).to(DEVICE),
        dynamics=DynamicsNetwork(latent_channels=LATENT_CHANNELS, action_space=ACTIONS,
                                 support_size=reward_sup, num_bins=reward_bins,
                                 transform=transform).to(DEVICE),
        prediction=PredictionNetwork(latent_channels=LATENT_CHANNELS, action_space=ACTIONS,
                                     support_size=value_sup, num_bins=value_bins,
                                     transform=transform).to(DEVICE),
        value_prefix=ValuePrefixNetwork(latent_channels=LATENT_CHANNELS,
                                        hidden_size=muzero.VALUE_PREFIX_HIDDEN,
                                        support_size=prefix_sup, num_bins=prefix_bins,
                                        transform=transform).to(DEVICE),
        projector=LatentProjector(LATENT_CHANNELS * muzero.LATENT_HW ** 2).to(DEVICE),
    )


class Run:
    """One arm at one seed: networks, trainer, searcher, environment."""

    def __init__(self, arm, seed, spec):
        self.arm, self.seed, self.spec = arm, seed, spec
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        nets = build_networks(spec)
        self.trainer = object.__new__(TrainingHelper)
        self.trainer.device = DEVICE
        self.trainer.action_space = ACTIONS
        self.trainer.training_steps = 0
        for name, net in nets.items():
            setattr(self.trainer, name, net)
        self.trainer.rep_optimizer = muzero.make_optimizer(nets["representation"].parameters())
        self.trainer.dyn_optimizer = muzero.make_optimizer(nets["dynamics"].parameters())
        self.trainer.pred_optimizer = muzero.make_optimizer(nets["prediction"].parameters())
        self.trainer.prefix_optimizer = muzero.make_optimizer(nets["value_prefix"].parameters())
        self.trainer.proj_optimizer = muzero.make_optimizer(nets["projector"].parameters())

        # The searcher shares the trainer's networks, so re-analysis always uses
        # current weights -- the equivalent of NetworkManager.update_if_needed.
        self.searcher = MCTSHelper(action_space=ACTIONS, device=DEVICE,
                                   network_ref=None, storage_ref=None,
                                   name=f"ablation-{arm}-{seed}")
        for name, net in nets.items():
            setattr(self.searcher.network_manager, name, net)

        # Eval is the resting state: because the searcher shares the trainer's
        # networks, self-play, search, and evaluation all run through the same
        # objects, and BatchNorm must read running statistics there -- exactly
        # as the pipeline's NetworkManager consumers do. run_one enters train
        # mode only around train_step, the one place batch statistics belong.
        self.set_train_mode(False)

        import gymnasium as gym
        from EpisodicLifeEnv import EpisodicLifeEnv
        env_kwargs = {"frameskip": FRAMESKIP} if FRAMESKIP else {}
        self.env = EpisodicLifeEnv(gym.make(ENV_NAME, **env_kwargs))
        # Seed the env stream too. Without this the module docstring's
        # paired-seed claim was false: gymnasium draws np_random from OS
        # entropy on an unseeded reset, so no run was reproducible even
        # against itself.
        self.env.reset(seed=seed)
        self.replay = []

    def set_train_mode(self, training):
        for net in (self.trainer.representation, self.trainer.dynamics,
                    self.trainer.prediction, self.trainer.value_prefix,
                    self.trainer.projector):
            net.train(training)

    # -- self-play ---------------------------------------------------------
    def play_episode(self, greedy=False, cap=None):
        cap = EPISODE_CAP if cap is None else cap
        obs, _ = self.env.reset()
        stack = initial_frame_stack(preprocess_frame(obs, IMAGE_RESHAPE))
        traj, prev_reward, total, done, steps = [], 0.0, 0.0, False, 0
        while not done and steps < cap:
            stack = push_frame(stack, preprocess_frame(obs, IMAGE_RESHAPE))
            if not greedy and muzero.USE_SEARCH_ACTING:
                # TEST feature: paper-style act-time search during self-play.
                # Greedy evaluation deliberately never takes this branch -- it
                # must always measure the deployable raw-policy agent.
                action = asyncio.run(self.searcher.act_search(stack))
            else:
                with torch.no_grad():
                    policy, _ = self.searcher.network_manager.initial_inference(stack)
                if greedy:
                    action = int(policy.argmax())
                else:
                    # Acting-side exploration floor, mirroring gameplay_process.
                    # Read per step so an arm override applies; identity at 1.0.
                    t = muzero.ACTING_TEMPERATURE
                    if t != 1.0:
                        policy = policy ** (1.0 / t)
                        policy = policy / policy.sum()
                    action = int(np.random.choice(ACTIONS, p=policy))
            obs, reward, term, trunc, _ = self.env.step(action)
            done = term or trunc
            traj.append((stack, action, prev_reward))
            prev_reward = muzero.clip_reward(reward)
            total += float(reward)
            steps += 1
        terminated = done
        if not terminated:
            # The cap truncated the episode mid-game. EpisodicLifeEnv only
            # truly resets on real game over, so without forcing it the next
            # reset() would RESUME the same match mid-rally -- every later
            # "episode" becoming an overlapping window of one continuous game,
            # eval windows starting wherever self-play left off.
            self.env.was_real_done = True
        traj.append((push_frame(stack, preprocess_frame(obs, IMAGE_RESHAPE)), 0, prev_reward))
        return traj, total, steps, terminated

    # -- re-analysis -------------------------------------------------------
    def reanalyse(self, traj, terminated=True):
        evaluated = self.searcher.eval_obs(traj)
        self.searcher.create_tree_from_game(evaluated, terminated=terminated)
        searched = self.searcher.select_reanalysis_indices(len(evaluated))
        # Two passes, mirroring MCTS_process: search first, then build the
        # paper's n-step targets with the root values as bootstraps -- a
        # position is never trained on its own search value.
        results = {}
        for idx in sorted(searched):
            # muzero.-qualified, not the from-imports: these must stay
            # arm-overridable at call time like every other knob.
            results[idx] = asyncio.run(self.searcher.run_full_mcts(
                idx, num_simulations=muzero.TOTAL_MCTS_SIMS,
                sim_batches=muzero.MCTS_BATCHES,
                c_puct=muzero.C_PUCT, discount=muzero.DISCOUNT))
        search_values = {i: r[1] for i, r in results.items()}
        out = []
        for idx, (obs, action, reward, *_ ) in enumerate(evaluated):
            if idx in results:
                policy, _, has_target = results[idx]
            else:
                policy = np.full(ACTIONS, 1.0 / ACTIONS)
                has_target = False
            value = self.searcher.nstep_value(idx, search_values=search_values)
            out.append({"obs": obs, "action": action, "reward": reward,
                        "target_policy": policy, "target_value": value,
                        "has_policy_target": has_target})
        if out:
            # prepare_batch reads this to pad unroll windows that cross a
            # capped tail with the tail's own value target instead of a
            # fabricated terminal V=0.
            out[-1]["truncated"] = not terminated
        return out

    # -- evaluation --------------------------------------------------------
    def evaluate(self, episodes=8, batches=8):
        """Greedy episode return, plus prediction errors against predict-zero.

        Runs entirely in eval mode -- the resting state. The greedy episodes
        used to run BEFORE the nets were switched, so evaluation acting went
        through train-mode BatchNorm: a different function from the one the
        prediction-error metrics measured a few lines later.
        """
        returns = [self.play_episode(greedy=True)[1] for _ in range(episodes)]

        K = muzero.UNROLL_STEPS
        errs = {"per_step": 0.0, "prefix": 0.0, "control": 0.0, "value": 0.0,
                "value_control": 0.0, "policy_entropy": 0.0, "policy_maxp": 0.0}
        if self.replay:
            with torch.no_grad():
                for _ in range(batches):
                    b = asyncio.run(self.trainer.prepare_batch(self.replay, 32, K, ACTIONS))
                    obs = torch.tensor(b["obs"], dtype=torch.float32, device=DEVICE) / 255.0
                    acts = torch.tensor(b["actions"], dtype=torch.long, device=DEVICE)
                    rews = torch.tensor(b["rewards"], dtype=torch.float32, device=DEVICE)
                    tval = torch.tensor(b["target_value"], dtype=torch.float32, device=DEVICE)
                    truth = muzero.discounted_prefix_targets(rews, discount=1.0)

                    latent = self.trainer.representation(obs[:, 0])
                    policy_logits, value, _ = self.trainer.prediction(latent)
                    errs["value"] += (value.squeeze(-1) - tval[:, 0]).abs().mean().item()
                    errs["value_control"] += tval[:, 0].abs().mean().item()
                    # Policy sharpening: the returns panel is too noisy to show
                    # whether the head is learning at all (uniform over 5
                    # actions: entropy 1.609, max-p 0.20).
                    p = torch.softmax(policy_logits, dim=1)
                    errs["policy_entropy"] += (-p * torch.log(p + 1e-12)).sum(dim=1).mean().item()
                    errs["policy_maxp"] += p.max(dim=1).values.mean().item()

                    state = self.trainer.value_prefix.initial_state(obs.size(0), DEVICE)
                    running = torch.zeros(obs.size(0), 1, device=DEVICE)
                    for i in range(1, K + 1):
                        latent, reward, _ = self.trainer.dynamics(latent, acts[:, i - 1])
                        running = running + reward
                        prefix, _, state = self.trainer.value_prefix(latent, state)
                    tgt = truth[:, K:K + 1]
                    errs["per_step"] += (running - tgt).abs().mean().item()
                    errs["prefix"] += (prefix - tgt).abs().mean().item()
                    errs["control"] += tgt.abs().mean().item()
            errs = {k: v / batches for k, v in errs.items()}

        return {"return_mean": float(np.mean(returns)),
                "return_max": float(np.max(returns)), **errs}


def run_one(arm, seed, iterations, episodes_per_iter, train_steps, eval_every, out):
    with arm_config(arm) as spec:
        run = Run(arm, seed, spec)
        started = time.time()
        for it in range(1, iterations + 1):
            for _ in range(episodes_per_iter):
                traj, _, _, terminated = run.play_episode()
                run.replay.append(run.reanalyse(traj, terminated=terminated))
                if len(run.replay) > muzero.MAX_REPLAY:
                    run.replay.pop(0)

            run.set_train_mode(True)
            for _ in range(train_steps):
                batch = asyncio.run(run.trainer.prepare_batch(
                    run.replay, muzero.TRAINING_BATCHES, muzero.UNROLL_STEPS, ACTIONS))
                asyncio.run(run.trainer.train_step(batch))
                run.trainer.training_steps += 1
            run.set_train_mode(False)

            if muzero.LR_DECAY_AT and it == max(1, int(iterations * muzero.LR_DECAY_AT)):
                muzero.decay_optimizer_lr(
                    (run.trainer.rep_optimizer, run.trainer.dyn_optimizer,
                     run.trainer.pred_optimizer, run.trainer.prefix_optimizer,
                     run.trainer.proj_optimizer), muzero.LR_DECAY_FACTOR)
                print(f"  [{arm} s{seed}] lr decayed x{muzero.LR_DECAY_FACTOR} "
                      f"at iter {it}", flush=True)

            if it == iterations:
                # Save the trained weights. The first full ablation discarded
                # 4.9 hours of training because nothing here wrote them out,
                # which made it impossible to watch an agent afterwards or to
                # resume from one.
                ckpt = RESULTS / f"weights{TAG}-{arm}-seed{seed}.pth"
                torch.save({
                    "arm": arm, "seed": seed, "spec": {k: v for k, v in spec.items()},
                    "training_steps": run.trainer.training_steps,
                    "representation": run.trainer.representation.state_dict(),
                    "dynamics": run.trainer.dynamics.state_dict(),
                    "prediction": run.trainer.prediction.state_dict(),
                    "value_prefix": run.trainer.value_prefix.state_dict(),
                }, ckpt)
                print(f"  saved {ckpt.name}", flush=True)

            if it % eval_every == 0 or it == iterations:
                metrics = run.evaluate()
                record = {"arm": arm, "seed": seed, "iteration": it,
                          "train_steps": run.trainer.training_steps,
                          "elapsed_s": round(time.time() - started, 1), **metrics}
                with open(out, "a") as fh:
                    fh.write(json.dumps(record) + "\n")
                print(f"  [{arm} s{seed}] iter {it:>3}  return {metrics['return_mean']:>7.1f}  "
                      f"prefix {metrics['prefix']:>6.2f} / control {metrics['control']:>6.2f}  "
                      f"per-step {metrics['per_step']:>6.2f}  "
                      f"H(pi) {metrics.get('policy_entropy', 0.0):>5.2f}", flush=True)
        run.env.close()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot(path, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [json.loads(l) for l in open(path) if l.strip()]
    if not rows:
        print("no results to plot")
        return
    arms = sorted({r["arm"] for r in rows})
    colours = dict(zip(arms, ["#0D6E75", "#B4531A", "#5A6472", "#7A3E8F", "#2E7D32"]))

    def series(arm, key):
        by_iter = {}
        for r in rows:
            if r["arm"] == arm:
                by_iter.setdefault(r["iteration"], []).append(r[key])
        xs = sorted(by_iter)
        return xs, [float(np.mean(by_iter[x])) for x in xs], [float(np.std(by_iter[x])) for x in xs]

    panels = [("return_mean", "Greedy episode return", "higher is better"),
              ("prefix", "Value-prefix MAE", "vs control (dashed)"),
              ("per_step", "Per-step summed MAE", "vs control (dashed)"),
              ("value", "Value MAE", "vs control (dashed)")]
    controls = {"prefix": "control", "per_step": "control", "value": "value_control"}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (key, title, note) in zip(axes.ravel(), panels):
        for arm in arms:
            xs, mean, sd = series(arm, key)
            if not xs:
                continue
            ax.plot(xs, mean, label=arm, color=colours[arm], linewidth=2)
            ax.fill_between(xs, np.array(mean) - np.array(sd), np.array(mean) + np.array(sd),
                            color=colours[arm], alpha=0.15)
            if key in controls:
                _, cmean, _ = series(arm, controls[key])
                ax.plot(xs, cmean, color=colours[arm], linewidth=1, linestyle="--", alpha=0.6)
        ax.set_title(f"{title}   ({note})", fontsize=11)
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.legend(fontsize=9)
    fig.suptitle("SwitchPlayer ablation — shaded band is +/- 1 sd across seeds.\n"
                 "A predictor that does not beat its dashed control has learned nothing.",
                 fontsize=12)
    fig.tight_layout()
    target = out_dir / f"ablation{TAG}.png"
    fig.savefig(target, dpi=140)
    print(f"wrote {target}")


def main():
    # FRAMESKIP was missing from this list: --frameskip set a main()-LOCAL,
    # the banner printed the local ("skip 4"), and Run.__init__ read the module
    # global -- still None -- so the env silently kept the stochastic skip the
    # flag exists to remove. Every name assigned below MUST be declared here.
    global ENV_NAME, ACTIONS, EPISODE_CAP, TAG, FRAMESKIP
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["per-step", "value-prefix"],
                    choices=sorted(ARMS))
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--iterations", type=int, default=40)
    ap.add_argument("--episodes-per-iter", type=int, default=2)
    ap.add_argument("--train-steps", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--env", default="MsPacman-v4",
                    help="ALE env id; non-default envs get suffixed output files")
    ap.add_argument("--actions", type=int, default=None,
                    help="restrict to the first N actions; default: env's full "
                         "space (MsPacman keeps its historical 5)")
    ap.add_argument("--episode-cap", type=int, default=None,
                    help="self-play/eval step cap; default 600 (MsPacman "
                         "single-life scale)")
    ap.add_argument("--frameskip", type=int, default=None,
                    help="fixed frameskip override (benchmark protocol: 4); "
                         "default keeps the env registration's skip")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()

    ENV_NAME = args.env
    if args.env != "MsPacman-v4":
        TAG = "-" + args.env.split("-")[0].lower()
    if args.episode_cap:
        EPISODE_CAP = args.episode_cap
    if args.frameskip:
        FRAMESKIP = args.frameskip
    if args.actions:
        ACTIONS = args.actions
    elif args.env != "MsPacman-v4":
        import gymnasium as gym
        probe = gym.make(args.env)
        ACTIONS = int(probe.action_space.n)
        probe.close()

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"ablation{TAG}.jsonl"

    if not args.plot_only:
        print(f"device {DEVICE} | env {ENV_NAME} ({ACTIONS} actions, cap {EPISODE_CAP}, "
              f"skip {FRAMESKIP or 'registered'}) | "
              f"arms {args.arms} | seeds {args.seeds} | "
              f"{args.iterations} iterations x ({args.episodes_per_iter} episodes + "
              f"{args.train_steps} train steps)")
        for arm in args.arms:
            print(f"\n=== {arm}: {ARMS[arm]['_doc']}")
            for seed in range(args.seeds):
                run_one(arm, seed, args.iterations, args.episodes_per_iter,
                        args.train_steps, args.eval_every, out)
    plot(out, RESULTS)


if __name__ == "__main__":
    main()
