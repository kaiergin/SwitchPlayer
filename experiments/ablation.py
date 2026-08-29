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
import os
import pathlib
import pickle
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import muzero
# Constants that arms may override are NEVER from-imported here -- they are
# read muzero.-qualified at call time so overrides genuinely apply.
from muzero import (IMAGE_RESHAPE, LATENT_CHANNELS, MCTSHelper, TrainingHelper,
                    initial_frame_stack, preprocess_frame, push_frame)
from networks import (DynamicsNetwork, LatentProjector, PredictionNetwork,
                      RepresentationNetwork, ValuePrefixNetwork)

# ABLATION_RESULTS overrides the output directory -- REQUIRED for smoke
# tests, which have twice now appended to (and once clobbered the weights of)
# live results by sharing the default directory.
RESULTS = pathlib.Path(os.environ.get(
    "ABLATION_RESULTS",
    pathlib.Path(__file__).resolve().parent.parent / "benchmarks" / "results"))
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
    "effzero-repro": {
        "USE_SEARCH_ACTING": True,
        "USE_GUMBEL_SEARCH": False,
        "PAPER_TARGETS": True,
        "PER_ENABLED": True,
        "USE_VALUE_PREFIX_IN_TREE": True,
        "REWARD_LOSS_WEIGHT": 0.0,
        # NOT a bug -- do not "fix" to 0.997. EffZero's released
        # config/atari/__init__.py passes discount=0.997 and then applies
        # self.discount **= self.frame_skip, so its Atari results were
        # produced at 0.997^4 = 0.98807 per decision. Challenged and
        # re-verified against the source twice (2026-08-22, 2026-08-26).
        "DISCOUNT": 0.997 ** 4,
        "TD_STEPS": 5,
        "LEARNING_RATE": 3e-4,
        "MAX_REPLAY": 60,
        "USE_AUGMENTATION": True,
        "REWARD_CLIP": 1.0,
        "ACT_SEARCH_SIMS": 50,
        # The papers' strictly sequential search: every simulation selects on
        # statistics the previous one updated. Added 2026-08-23 with the
        # virtual-loss sign fix; earlier archived runs of this arm searched in
        # stale-statistics waves of 8 (see the fidelity audit).
        "MCTS_BATCHES": 1,
        # The papers have no depth cap; 50 effectively uncaps a 50-sim
        # search (MAX_DEPTH 8 made capped leaves re-back their static
        # network value each visit). Requires the in-tree prefix LSTM
        # reset (PREFIX_LSTM_HORIZON), which keeps deep edges in the
        # head's trained regime. Added 2026-08-27.
        "MAX_DEPTH": 50,
        "DIRICHLET_ALPHA": 0.3,
        "PREFIX_HEAD_CHANNELS": 16,
        "PREFIX_FC_HIDDEN": 32,
        "USE_ATARI_WRAPPERS": True,
        "USE_TARGET_MODEL": True,
        "TRAINING_BATCHES": 256,
        "_doc": "The paper-reproduction configuration, constants source-"
                "verified against EfficientZero's released code: PUCT search "
                "acting at 50 sims (Dirichlet 0.25/0.3, noiseless at eval) "
                "with the staged visit temperature; per-step search policies "
                "and root values recorded; batch-time value targets with the "
                "4.3 age-shrunk horizon (td 5, decay 0.3*run); PER alpha 0.6 "
                "/ beta 0.4->1.0 / eps 1e-6 weighting the WHOLE per-sample "
                "loss; discount 0.997^4; sign-scale reward clip; EfficientZero "
                "prefix head dims; no per-step reward head. Part 3 closed "
                "the reanalyze gap: Atari wrappers (NoopReset + MaxAndSkip "
                "flicker max-pool), batch 256, value bootstraps recomputed "
                "through a DOUBLE-BUFFERED four-net lagged target model "
                "(staleness 200-400 steps), and the oldest trajectories' "
                "policies re-searched through it. Replay 60 trajectories (~9-15GB "
                "of Pong frame stacks; raise on a bigger box). Run with "
                "--env Pong-v4 --frameskip 4 --episode-cap 3000 "
                "--episodes-per-iter 1 --eval-every 10.",
    },
    "effzero-repro-refresh5x": {
        # effzero-repro with the refresh-throughput knobs raised -- the
        # mechanism test the extension run pointed at: at the paper's true
        # 100k-env-step position (measured via env_steps) with MORE gradient
        # steps than the paper spends, the value function kept improving
        # (value_control 1.33->0.54) while policy return plateaued at ~-16.5
        # -- the stale-policy-target signature. 5x refresh coverage
        # (POLICY_REFRESH_EVERY 5->2, COUNT 2->4; still ~0.5% of EffZero's
        # 0.99 batch revisit rate, so this tests direction, not parity) and
        # a full-history replay window (85 trajectories ~= the 100k budget;
        # 60 was silently DROPPING the oldest data past ~iter 60, where
        # EffZero's buffer keeps everything). A separate arm so
        # effzero-repro's archived meaning stays fixed.
        "USE_SEARCH_ACTING": True,
        "USE_GUMBEL_SEARCH": False,
        "PAPER_TARGETS": True,
        "PER_ENABLED": True,
        "USE_VALUE_PREFIX_IN_TREE": True,
        "REWARD_LOSS_WEIGHT": 0.0,
        # NOT a bug -- do not "fix" to 0.997. EffZero's released
        # config/atari/__init__.py passes discount=0.997 and then applies
        # self.discount **= self.frame_skip, so its Atari results were
        # produced at 0.997^4 = 0.98807 per decision. Challenged and
        # re-verified against the source twice (2026-08-22, 2026-08-26).
        "DISCOUNT": 0.997 ** 4,
        "TD_STEPS": 5,
        "LEARNING_RATE": 3e-4,
        "MAX_REPLAY": 85,
        "USE_AUGMENTATION": True,
        "REWARD_CLIP": 1.0,
        "ACT_SEARCH_SIMS": 50,
        # The papers' strictly sequential search: every simulation selects on
        # statistics the previous one updated. Added 2026-08-23 with the
        # virtual-loss sign fix; earlier archived runs of this arm searched in
        # stale-statistics waves of 8 (see the fidelity audit).
        "MCTS_BATCHES": 1,
        # The papers have no depth cap; 50 effectively uncaps a 50-sim
        # search (MAX_DEPTH 8 made capped leaves re-back their static
        # network value each visit). Requires the in-tree prefix LSTM
        # reset (PREFIX_LSTM_HORIZON), which keeps deep edges in the
        # head's trained regime. Added 2026-08-27.
        "MAX_DEPTH": 50,
        "DIRICHLET_ALPHA": 0.3,
        "PREFIX_HEAD_CHANNELS": 16,
        "PREFIX_FC_HIDDEN": 32,
        "USE_ATARI_WRAPPERS": True,
        "USE_TARGET_MODEL": True,
        "TRAINING_BATCHES": 256,
        "POLICY_REFRESH_EVERY": 2,
        "POLICY_REFRESH_COUNT": 4,
        "_doc": "effzero-repro + 5x reanalyze-refresh throughput (every 2 "
                "iters, 4 oldest trajectories) + full-history replay (85 "
                "trajectories, ~the 100k-step budget; ~12GB RAM). The "
                "extension run isolated stale policy targets as the binding "
                "gap; if 5x fresher targets steepen the curve, batch-time "
                "refresh is the next build. Run with --env Pong-v4 "
                "--frameskip 4 --episode-cap 3000 --episodes-per-iter 1 "
                "--eval-every 10 --checkpoint-every 10.",
    },
    "effzero-repro-refresh5x-communal": {
        # refresh5x with the COMMUNAL refresh topology (Kai's design,
        # 2026-08-24): the oldest trajectories are re-searched through the
        # SHARED chain tree -- COMMUNAL_WINDOW_ROOTS roots in lockstep, one
        # simulation per root per round, GPU batch formed across roots,
        # statistics accumulating across searches (only the chain backup's
        # bookkeeping visit subtracted) -- instead of one independent
        # act_search per position. Acting is identical to refresh5x
        # (sequential 50-sim PUCT), so a matched pair isolates the refresh
        # topology + statistics-sharing question. Paired control: the
        # post-search-fix refresh5x run (pong-searchfix-2026-08-23). Run
        # with the control's exact command, only --arms differing.
        "USE_SEARCH_ACTING": True,
        "USE_GUMBEL_SEARCH": False,
        "PAPER_TARGETS": True,
        "PER_ENABLED": True,
        "USE_VALUE_PREFIX_IN_TREE": True,
        "REWARD_LOSS_WEIGHT": 0.0,
        # NOT a bug -- do not "fix" to 0.997. EffZero's released
        # config/atari/__init__.py passes discount=0.997 and then applies
        # self.discount **= self.frame_skip, so its Atari results were
        # produced at 0.997^4 = 0.98807 per decision. Challenged and
        # re-verified against the source twice (2026-08-22, 2026-08-26).
        "DISCOUNT": 0.997 ** 4,
        "TD_STEPS": 5,
        "LEARNING_RATE": 3e-4,
        "MAX_REPLAY": 85,
        "USE_AUGMENTATION": True,
        "REWARD_CLIP": 1.0,
        "ACT_SEARCH_SIMS": 50,
        "MCTS_BATCHES": 1,
        # The papers have no depth cap; 50 effectively uncaps a 50-sim
        # search (MAX_DEPTH 8 made capped leaves re-back their static
        # network value each visit). Requires the in-tree prefix LSTM
        # reset (PREFIX_LSTM_HORIZON), which keeps deep edges in the
        # head's trained regime. Added 2026-08-27.
        "MAX_DEPTH": 50,
        "DIRICHLET_ALPHA": 0.3,
        "PREFIX_HEAD_CHANNELS": 16,
        "PREFIX_FC_HIDDEN": 32,
        "USE_ATARI_WRAPPERS": True,
        "USE_TARGET_MODEL": True,
        "TRAINING_BATCHES": 256,
        "POLICY_REFRESH_EVERY": 2,
        "POLICY_REFRESH_COUNT": 4,
        "USE_COMMUNAL_REANALYSIS": True,
        # Added 2026-08-26 (Kai): cap the buffer in TRANSITIONS -- the unit
        # EfficientZero's own config uses -- not just trajectories. Kai's
        # stepwise march toward the tiny-buffer/live-refresh design:
        # 125k -> 62.5k (2026-08-27, mid-lineage at the pong-communal
        # iter-70 reboot) -> 31.25k (2026-08-27, for the warm-started
        # +100k continuation): ~2.6 GB resident, ~10-13 fresh episodes at
        # late-stage lengths, checkpoints ~3 GB in seconds. MAX_REPLAY 85
        # stays as an (inert) secondary ceiling.
        "MAX_REPLAY_TRANSITIONS": 31_250,
        "_doc": "refresh5x + communal windowed refresh through the shared "
                "chain tree (statistics accumulate across roots; bookkeeping "
                "visit subtracted). Everything else matches "
                "effzero-repro-refresh5x, including sequential 50-sim "
                "acting. Run with --env Pong-v4 --frameskip 4 --iterations "
                "90 --train-steps 2000 --episode-cap 3000 "
                "--episodes-per-iter 1 --eval-every 5 --checkpoint-every 10, "
                "stopped at the paired horizon.",
    },
    "effzero-repro-smallnet": {
        # Audit finding 7: the communal configuration at EfficientZero's
        # OWN network dimensions -- 64 latent channels, single trunk/
        # dynamics blocks, 16-channel value/policy heads with a [32]
        # hidden, and their single-plane action encoding. The reference's
        # 100k-sample recipe treats the small network as part of the
        # method; this arm measures that claim against the rung-1
        # (repo-dims, skip+BN) run at matched budget. From scratch only:
        # every architecture knob breaks --init-weights compatibility.
        "USE_SEARCH_ACTING": True,
        "USE_GUMBEL_SEARCH": False,
        "PAPER_TARGETS": True,
        "PER_ENABLED": True,
        "USE_VALUE_PREFIX_IN_TREE": True,
        "REWARD_LOSS_WEIGHT": 0.0,
        "DISCOUNT": 0.997 ** 4,
        "TD_STEPS": 5,
        "LEARNING_RATE": 3e-4,
        "MAX_REPLAY": 85,
        "USE_AUGMENTATION": True,
        "REWARD_CLIP": 1.0,
        "ACT_SEARCH_SIMS": 50,
        "MCTS_BATCHES": 1,
        "MAX_DEPTH": 50,
        "DIRICHLET_ALPHA": 0.3,
        "PREFIX_HEAD_CHANNELS": 16,
        "PREFIX_FC_HIDDEN": 32,
        "USE_ATARI_WRAPPERS": True,
        "USE_TARGET_MODEL": True,
        "TRAINING_BATCHES": 256,
        "POLICY_REFRESH_EVERY": 2,
        "POLICY_REFRESH_COUNT": 4,
        "USE_COMMUNAL_REANALYSIS": True,
        "MAX_REPLAY_TRANSITIONS": 62_500,
        "LATENT_CHANNELS": 64,
        "REP_TRUNK_BLOCKS": 1,
        "DYN_BLOCKS": 1,
        "PRED_HEAD_CHANNELS": 16,
        "PRED_FC_HIDDEN": 32,
        "ACTION_ENCODING": "plane",
        "_doc": "effzero-repro-refresh5x-communal at EfficientZero's Atari "
                "network dimensions (64ch, 1/1 blocks, 16-ch heads + [32], "
                "single-plane actions). Buffer 62.5k. Run from scratch with "
                "the communal arm's command, only --arms differing.",
    },
    "effzero-repro-smallbuf20": {
        # refresh5x with the replay window cut to 20 trajectories --
        # "reanalyze by deletion". Old trajectories carry the search policy
        # recorded when they were played; instead of refreshing those targets,
        # evict them. Combined with the 5x refresh (4 oldest every 2 iters) a
        # 20-traj buffer gets a full re-search cycle every ~10 iterations.
        # NOTE: at 1 episode/iter a from-scratch 20-iteration run never
        # exceeds 20 trajectories, so the cap NEVER BINDS -- this arm is meant
        # to be WARM-STARTED (--init-weights; the 10-episode prefill makes the
        # cap bind from ~iter 11). Paired control: the refresh5x continuation
        # at matched iterations (same init weights, cap 85). A separate arm so
        # refresh5x's archived meaning stays fixed.
        "USE_SEARCH_ACTING": True,
        "USE_GUMBEL_SEARCH": False,
        "PAPER_TARGETS": True,
        "PER_ENABLED": True,
        "USE_VALUE_PREFIX_IN_TREE": True,
        "REWARD_LOSS_WEIGHT": 0.0,
        # NOT a bug -- do not "fix" to 0.997. EffZero's released
        # config/atari/__init__.py passes discount=0.997 and then applies
        # self.discount **= self.frame_skip, so its Atari results were
        # produced at 0.997^4 = 0.98807 per decision. Challenged and
        # re-verified against the source twice (2026-08-22, 2026-08-26).
        "DISCOUNT": 0.997 ** 4,
        "TD_STEPS": 5,
        "LEARNING_RATE": 3e-4,
        "MAX_REPLAY": 20,
        "USE_AUGMENTATION": True,
        "REWARD_CLIP": 1.0,
        "ACT_SEARCH_SIMS": 50,
        # The papers' strictly sequential search: every simulation selects on
        # statistics the previous one updated. Added 2026-08-23 with the
        # virtual-loss sign fix; earlier archived runs of this arm searched in
        # stale-statistics waves of 8 (see the fidelity audit).
        "MCTS_BATCHES": 1,
        # The papers have no depth cap; 50 effectively uncaps a 50-sim
        # search (MAX_DEPTH 8 made capped leaves re-back their static
        # network value each visit). Requires the in-tree prefix LSTM
        # reset (PREFIX_LSTM_HORIZON), which keeps deep edges in the
        # head's trained regime. Added 2026-08-27.
        "MAX_DEPTH": 50,
        "DIRICHLET_ALPHA": 0.3,
        "PREFIX_HEAD_CHANNELS": 16,
        "PREFIX_FC_HIDDEN": 32,
        "USE_ATARI_WRAPPERS": True,
        "USE_TARGET_MODEL": True,
        "TRAINING_BATCHES": 256,
        "POLICY_REFRESH_EVERY": 2,
        "POLICY_REFRESH_COUNT": 4,
        "_doc": "refresh5x with MAX_REPLAY 20: evict stale policy targets "
                "instead of refreshing them. Warm-start only (cap never "
                "binds from scratch in short runs). For a paired comparison "
                "run the CONTROL'S EXACT command (only --arms differs) -- "
                "the temperature/off-policy schedules key on --iterations "
                "via _run_length_globals, so a shorter --iterations anneals "
                "faster and confounds the pair; stop at the paired horizon "
                "instead. E.g. --env Pong-v4 --frameskip 4 --iterations 90 "
                "--episode-cap 3000 --episodes-per-iter 1 --eval-every 10 "
                "--checkpoint-every 2 --init-weights <parent>, kill after "
                "the iter-20 checkpoint.",
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
# the muzero module default. pong-kit/pong-kit2/effzero-repro opt in. Without this, the
# USE_AUGMENTATION=True default retroactively changed every legacy arm --
# most sharply breaking the pong-act-search vs pong-lr-3e-4 controlled pair.
for _spec in ARMS.values():
    _spec.setdefault("USE_AUGMENTATION", False)


@contextlib.contextmanager
def _run_length_globals(iterations, train_steps):
    """Run-length-dependent paper constants: the staged acting temperature's
    total and EfficientZero 4.3's horizon-decay denominator (their
    auto_td_steps = 0.3 * training_steps). Exception-safe like arm_config, so
    a mid-run failure cannot leak them into whatever runs next."""
    muzero.TEMPERATURE_TOTAL_STEPS = iterations * train_steps
    muzero.OFFPOLICY_DECAY_STEPS = max(1, int(0.3 * iterations * train_steps))
    try:
        yield
    finally:
        muzero.TEMPERATURE_TOTAL_STEPS = None
        muzero.OFFPOLICY_DECAY_STEPS = None


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

    # muzero.-qualified THROUGHOUT, including LATENT_CHANNELS: the from-import
    # binds at import time, which would silently ignore an arm's channel
    # override (the def-time-capture bug class) -- the smallnet arm is the
    # first to override it.
    return dict(
        representation=RepresentationNetwork(input_channels=12,
                                             latent_channels=muzero.LATENT_CHANNELS,
                                             trunk_blocks=muzero.REP_TRUNK_BLOCKS).to(DEVICE),
        dynamics=DynamicsNetwork(latent_channels=muzero.LATENT_CHANNELS, action_space=ACTIONS,
                                 support_size=reward_sup, num_bins=reward_bins,
                                 transform=transform,
                                 blocks=muzero.DYN_BLOCKS,
                                 action_encoding=muzero.ACTION_ENCODING).to(DEVICE),
        prediction=PredictionNetwork(latent_channels=muzero.LATENT_CHANNELS, action_space=ACTIONS,
                                     support_size=value_sup, num_bins=value_bins,
                                     transform=transform,
                                     head_channels=muzero.PRED_HEAD_CHANNELS,
                                     fc_hidden=muzero.PRED_FC_HIDDEN).to(DEVICE),
        value_prefix=ValuePrefixNetwork(latent_channels=muzero.LATENT_CHANNELS,
                                        hidden_size=muzero.VALUE_PREFIX_HIDDEN,
                                        support_size=prefix_sup, num_bins=prefix_bins,
                                        transform=transform,
                                        head_channels=muzero.PREFIX_HEAD_CHANNELS,
                                        fc_hidden=muzero.PREFIX_FC_HIDDEN).to(DEVICE),
        projector=LatentProjector(muzero.LATENT_CHANNELS * muzero.LATENT_HW ** 2).to(DEVICE),
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

        # The searcher shares the trainer's networks, so non-paper re-analysis
        # uses current weights -- the equivalent of
        # NetworkManager.update_if_needed. (refresh_targets, the paper arms'
        # reanalyze policy half, deliberately swaps to the lagged target nets
        # for its duration.)
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
        if muzero.USE_ATARI_WRAPPERS:
            # The lineage's env stack: raw no-skip env -> NoopReset(30) ->
            # MaxAndSkip flicker max-pool -> EpisodicLife.
            from atari_wrappers import paper_atari_env
            self.env = EpisodicLifeEnv(paper_atari_env(ENV_NAME,
                                                       skip=FRAMESKIP or 4))
        else:
            env_kwargs = {"frameskip": FRAMESKIP} if FRAMESKIP else {}
            self.env = EpisodicLifeEnv(gym.make(ENV_NAME, **env_kwargs))
        # Seed the env stream too. Without this the module docstring's
        # paired-seed claim was false: gymnasium draws np_random from OS
        # entropy on an unseeded reset, so no run was reproducible even
        # against itself.
        self.env.reset(seed=seed)
        self.replay = []
        # Cumulative TRAINING env interaction (the Atari-100k x-axis); greedy
        # eval episodes are protocol-exempt and not counted.
        self.env_steps = 0

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
            if muzero.USE_SEARCH_ACTING and (not greedy or muzero.PAPER_TARGETS):
                # Search acting: one search per step during self-play. Greedy
                # EVAL searches only in paper mode (the papers evaluate with
                # search); non-paper search-acting arms keep their documented
                # meaning -- eval measures the deployable RAW-POLICY agent.
                # Round-9 finding: the paper-mode commit had made eval search
                # unconditional, silently inverting pong-act-search's _doc
                # against its own archived rows.
                action, search_pol, search_val = asyncio.run(
                    self.searcher.act_search(stack, greedy=greedy))
            else:
                search_pol = search_val = None
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
            # 5-field records exist ONLY for the paper data path. Recording
            # them for every search-acting arm broke the pre-existing
            # pong-act-search arm, whose non-paper reanalysis unpacks 6-field
            # eval_obs tuples (round-5 finding, reproduced ValueError).
            if search_pol is None or not muzero.PAPER_TARGETS:
                traj.append((stack, action, prev_reward))
            else:
                traj.append((stack, action, prev_reward, search_pol, search_val))
            prev_reward = muzero.clip_reward(reward)
            total += float(reward)
            steps += 1
        terminated = done
        if not greedy:
            self.env_steps += steps
        if not terminated:
            # The cap truncated the episode mid-game. EpisodicLifeEnv only
            # truly resets on real game over, so without forcing it the next
            # reset() would RESUME the same match mid-rally -- every later
            # "episode" becoming an overlapping window of one continuous game,
            # eval windows starting wherever self-play left off.
            self.env.was_real_done = True
        final_stack = push_frame(stack, preprocess_frame(obs, IMAGE_RESHAPE))
        if muzero.USE_SEARCH_ACTING and muzero.PAPER_TARGETS:
            # The terminal record needs a stored root value for the paper
            # value targets: 0 at a true terminal; a truncated tail gets one
            # greedy search on the final observation.
            if terminated:
                tail_pol, tail_val = np.ones(ACTIONS) / ACTIONS, 0.0
            elif greedy:
                # Eval trajectories are discarded by evaluate(); a truncated
                # greedy episode must not burn a 50-sim tail search whose
                # result nothing reads (round-10 finding).
                tail_pol, tail_val = np.ones(ACTIONS) / ACTIONS, 0.0
            else:
                _, tail_pol, tail_val = asyncio.run(
                    self.searcher.act_search(final_stack, greedy=True))
            traj.append((final_stack, 0, prev_reward, tail_pol, tail_val))
        else:
            traj.append((final_stack, 0, prev_reward))
        return traj, total, steps, terminated

    # -- re-analysis -------------------------------------------------------
    def paper_targets(self, traj, terminated):
        """Paper data path: every step already carries its acting-search
        policy and root value. Store the target ingredients (values are
        computed at BATCH time under the age-adjusted horizon), the birth
        step for EfficientZero 4.3, and age-0 priorities |v - z|."""
        if len(traj[0]) != 5:
            raise ValueError(
                "PAPER_TARGETS requires USE_SEARCH_ACTING: trajectories must "
                "carry per-step search records (5 fields), got "
                f"{len(traj[0])}")
        born = self.trainer.training_steps
        out = []
        for i, (obs, action, reward, pol, val) in enumerate(traj):
            # The terminal record of a TERMINATED episode carried no search
            # (uniform placeholder -> masked); a TRUNCATED tail's record is a
            # genuine greedy search and keeps its policy target.
            has_policy = (i < len(traj) - 1) or (not terminated)
            out.append({"obs": obs, "action": action, "reward": reward,
                        "target_policy": np.asarray(pol, dtype=np.float64),
                        "root_value": float(val),
                        "target_value": float(val),  # for non-paper readers
                        "has_policy_target": has_policy,
                        "born_step": born})
        out[-1]["truncated"] = not terminated
        for idx, t in enumerate(out):
            z0 = muzero.paper_value_target(out, idx, muzero.TD_STEPS,
                                           muzero.DISCOUNT)
            t["priority"] = abs(t["root_value"] - z0) + muzero.PER_EPS
        return out

    def _refresh_traj_communal(self, traj):
        """Communal refresh (USE_COMMUNAL_REANALYSIS): re-search a stored
        trajectory's policies through the SHARED chain tree instead of one
        independent act_search per position. COMMUNAL_WINDOW_ROOTS roots run
        in lockstep -- one simulation per root per round, the papers'
        sequential schedule inside each tree, the GPU batch formed ACROSS
        roots -- and statistics accumulate communally across the searches.
        The chain is seeded NEUTRALLY (create_tree_from_game with
        seed_values=False): no Monte-Carlo value backup, so the behaviour
        child starts symmetric with its imagined siblings and there is no
        bookkeeping visit to subtract. Expansions along the chain are
        shared between neighbouring roots, which is the design's point.

        Runs under the same target-model swap as the act_search path it
        replaces (the caller swapped the searcher's networks already). One
        behavioural difference besides topology: a truncated tail's search
        takes root noise like every other position, where the act_search
        path searched it greedily.
        """
        searcher = self.searcher
        saved = getattr(searcher, "current_trajectory", None)
        try:
            evaluated = searcher.eval_obs(
                [(t["obs"], t["action"], t["reward"]) for t in traj])
            # NEUTRAL chain (2026-08-27 audit finding): no Monte-Carlo value
            # backup, so the behaviour child starts symmetric with its
            # imagined siblings -- a seeded chain gives it a realized return
            # against network estimates, pulling policy targets toward the
            # behaviour action early and away from it once the value head
            # calibrates. The chain still supplies structure, priors, and
            # real edge rewards (the reuse communal wants).
            searcher.create_tree_from_game(
                evaluated, terminated=not traj[-1].get("truncated", False),
                seed_values=False)
            indices = list(range(len(traj)))
            if not traj[-1].get("has_policy_target", True):
                # Terminated tail: carried no search then, none now.
                indices = indices[:-1]
            window = max(1, muzero.COMMUNAL_WINDOW_ROOTS)
            released = -1
            for w0 in range(0, len(indices), window):
                widx = indices[w0:w0 + window]
                results = asyncio.run(searcher.run_communal_mcts(
                    widx, num_simulations=muzero.ACT_SEARCH_SIMS,
                    c_puct=muzero.C_PUCT, discount=muzero.DISCOUNT,
                    chain_seeded=False))
                for idx, (pol, val, has_policy) in results.items():
                    if not has_policy:
                        continue
                    traj[idx]["target_policy"] = np.asarray(pol, dtype=np.float64)
                    traj[idx]["root_value"] = float(val)
                # Windows advance in ascending order, so everything up to
                # this window's last root is unreachable from any later
                # search and can be freed.
                searcher.release_span(released + 1, widx[-1])
                released = widx[-1]
            # Free the tail past the last root too: searches descend up to
            # MAX_DEPTH beyond their root and leave expansions there.
            searcher.release_span(released + 1, len(traj) - 1)
        finally:
            searcher.current_trajectory = saved

    def refresh_targets(self, count):
        """EfficientZero's reanalyze, policy half (approximated): re-search
        every position of the `count` OLDEST trajectories through the LAGGED
        TARGET MODEL -- noisy, as their reanalyze workers search -- replacing
        policy targets and root values (the terminated tail's record carried
        no search and is skipped), recomputing priorities, and stamping
        the refresh rotation (ages are preserved). Their
        revisit_policy_search_rate=0.99 re-searches sampled positions per
        batch on a CPU fleet; this is the compute-bounded analogue."""
        if not self.replay:
            return
        # Rotation stamp is refreshed_step, NOT born_step: fresh policy
        # targets do not make the stored ACTION SEQUENCE on-policy, so the
        # 4.3 age-shrunk value horizon (keyed on born_step) must keep
        # shrinking (their delta_td keys on buffer position and is never
        # restored by a re-search).
        def freshness(i):
            head = self.replay[i][0]
            return head.get("refreshed_step", head.get("born_step", 0))

        oldest = sorted(range(len(self.replay)), key=freshness)[:count]
        now = self.trainer.training_steps
        # Their reanalyze re-searches with the LAGGED TARGET MODEL, not the
        # current weights: swap the searcher's networks for the duration.
        self.trainer._ensure_target_model()
        nm = self.searcher.network_manager
        saved = (nm.representation, nm.prediction, nm.dynamics, nm.value_prefix)
        try:
            # Inside the try: a partial swap must still hit the restore.
            nm.representation = self.trainer.target_representation
            nm.prediction = self.trainer.target_prediction
            nm.dynamics = self.trainer.target_dynamics
            nm.value_prefix = self.trainer.target_value_prefix
            for ti in oldest:
                traj = self.replay[ti]
                if muzero.USE_COMMUNAL_REANALYSIS and not muzero.USE_GUMBEL_SEARCH:
                    # Kai's communal topology: shared chain tree, lockstep
                    # windows across roots, statistics accumulating.
                    self._refresh_traj_communal(traj)
                else:
                    for i, t in enumerate(traj):
                        terminal_record = i == len(traj) - 1
                        if terminal_record and not t.get("has_policy_target", True):
                            # Terminated tail: carried no search then, none now.
                            continue
                        _, pol, val = asyncio.run(self.searcher.act_search(
                            t["obs"], greedy=terminal_record))
                        t["target_policy"] = np.asarray(pol, dtype=np.float64)
                        t["root_value"] = float(val)
                for idx, t in enumerate(traj):
                    # Priority against the z the trainer will actually compute
                    # for this position: the 4.3 age-shrunk horizon, not the
                    # full TD_STEPS (refresh candidates are the oldest, where
                    # it shrinks most). The bootstrap is the just-refreshed
                    # stored search value -- MuZero's canonical priority --
                    # which tracks, without exactly equaling, the trainer's
                    # target-net value-head bootstrap.
                    age = max(0, now - t.get("born_step", now))
                    z0 = muzero.paper_value_target(
                        traj, idx, muzero.paper_horizon(age), muzero.DISCOUNT)
                    t["priority"] = abs(t["root_value"] - z0) + muzero.PER_EPS
                for t in traj:
                    # Stamped last, in its own pass: freshness() reads the head
                    # record, which must only carry a stamp once the whole
                    # trajectory -- searches AND priorities -- completed.
                    t["refreshed_step"] = now
        finally:
            (nm.representation, nm.prediction,
             nm.dynamics, nm.value_prefix) = saved

    def reanalyse(self, traj, terminated=True):
        if muzero.PAPER_TARGETS:
            return self.paper_targets(traj, terminated)
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


NET_NAMES = ("representation", "dynamics", "prediction", "value_prefix",
             "projector")
OPT_NAMES = ("rep_optimizer", "dyn_optimizer", "pred_optimizer",
             "prefix_optimizer", "proj_optimizer")


def resume_path(arm, seed):
    return RESULTS / f"resume{TAG}-{arm}-seed{seed}.ckpt"


CKPT_STREAM_MAGIC = b"SPCHKPT2\n"


def _cpu_state(obj):
    """Recursively detach+CPU every tensor: the streamed checkpoint pickles
    without torch.load's map_location, so tensors must be device-neutral on
    disk (load_state_dict re-places them on load)."""
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _cpu_state(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_cpu_state(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_cpu_state(v) for v in obj)
    return obj


def _read_checkpoint(path):
    """State dict from either checkpoint format: the SPCHKPT2 streamed
    container (head pickle + one frame per replay trajectory), or any
    torch.save file -- older checkpoints, completion markers, and
    --init-weights sources all load unchanged."""
    with open(path, "rb") as f:
        if f.read(len(CKPT_STREAM_MAGIC)) == CKPT_STREAM_MAGIC:
            state = pickle.load(f)
            state["replay"] = [pickle.load(f)
                               for _ in range(state.pop("replay_trajs", 0))]
            return state
    return torch.load(path, map_location="cpu", weights_only=False)


def save_run_checkpoint(run, path, meta):
    """Everything a resume needs, written atomically (tmp + replace) so a
    kill mid-write leaves the previous checkpoint intact. The replay buffer
    dominates the size (~150KB per Atari step); completed runs shrink the
    file to a marker via mark_run_completed.

    Mid-episode env state is NOT captured: checkpoints land at iteration
    boundaries, where the next play_episode resets. On life-count games the
    uninterrupted timeline may resume a game mid-life where a resumed run
    full-resets (EpisodicLifeEnv.was_real_done is not saved); inert on
    Pong, a documented divergence elsewhere -- like the env np_random
    stream, which a resume replays from its seed."""
    state = dict(meta)
    state["nets"] = _cpu_state({n: getattr(run.trainer, n).state_dict()
                                for n in NET_NAMES})
    state["optims"] = _cpu_state({n: getattr(run.trainer, n).state_dict()
                                  for n in OPT_NAMES})
    if getattr(run.trainer, "target_representation", None) is not None:
        state["target"] = _cpu_state({
            "nets": {n: getattr(run.trainer, f"target_{n}").state_dict()
                     for n in NET_NAMES[:4]},
            "snapshot": run.trainer._target_snapshot,
            "synced_at": run.trainer._target_synced_at,
        })
    state["env_steps"] = run.env_steps
    state["rng"] = {
        "np": np.random.get_state(),
        "py": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": (torch.cuda.get_rng_state_all()
                 if torch.cuda.is_available() else None),
    }
    # The replay buffer is NOT in the head pickle: it is streamed after it,
    # one pickle frame per trajectory. History of this function: a single
    # torch.save wedged a 32 GB box twice (2026-08-26 -- the default zipfile
    # format buffers the whole ~14 GB pickle in memory) and the legacy
    # format, though streaming, took 56 minutes: torch pickles at protocol
    # 2, and one monolithic dump grows a memo table with millions of
    # entries. Per-trajectory frames at protocol 5 give every frame a fresh
    # memo and modern framing; peak memory stays near the live state and
    # the write runs at disk speed. Single file, single atomic rename;
    # _read_checkpoint sniffs the magic and still torch.loads every older
    # checkpoint, completion marker, and --init-weights source.
    state["replay_trajs"] = len(run.replay)
    tmp = path.with_suffix(".tmp")
    import gc
    gc.collect()
    t0 = time.perf_counter()
    with open(tmp, "wb", buffering=16 * 1024 * 1024) as f:
        f.write(CKPT_STREAM_MAGIC)
        pickle.dump(state, f, protocol=5)
        for traj in run.replay:
            pickle.dump(traj, f, protocol=5)
    os.replace(tmp, path)
    print(f"  [checkpoint] wrote {path.stat().st_size / 1e9:.1f} GB "
          f"in {time.perf_counter() - t0:.0f}s", flush=True)


def load_run_checkpoint(run, path, expect):
    """Restore a run in place; returns the last COMPLETED iteration. Raises
    on a recipe mismatch: the temperature and off-policy schedules key on
    total run length, so resuming under a different recipe would silently
    train a different experiment."""
    # Tensors arrive CPU-side from either format (the streamed container
    # stores them CPU-neutral; torch.load gets map_location="cpu"):
    # load_state_dict copies across devices anyway, and the saved RNG
    # states are CPU ByteTensors that torch.set_rng_state rejects if a
    # cuda map_location relocates them.
    state = _read_checkpoint(path)
    # Recipe validation FIRST, completed-marker check second: a completed
    # 60-iter marker must not silently satisfy a NEW 90-iter --resume
    # invocation -- that is a different experiment, and the caller needs
    # the error, not a skip.
    for key, want in expect.items():
        if state.get(key) != want:
            have = (repr(state[key]) if key in state
                    else "absent (checkpoint predates this field)")
            raise ValueError(
                f"resume mismatch: checkpoint {key}={have} but this "
                f"invocation has {want!r}. If you meant to continue the old "
                f"experiment, rerun with its original recipe. To discard it "
                f"and run the experiment you just typed, rerun THIS command, "
                f"dropping --resume and adding: --arms {expect['arm']} "
                f"--seed {expect['seed']} --overwrite (this command's "
                f"--env/--frameskip already target these files -- the "
                f"checkpoint was found under them).")
    if state.get("completed"):
        return None
    for n in NET_NAMES:
        getattr(run.trainer, n).load_state_dict(state["nets"][n])
    for n in OPT_NAMES:
        getattr(run.trainer, n).load_state_dict(state["optims"][n])
    if "target" in state:
        run.trainer._ensure_target_model()
        for n in NET_NAMES[:4]:
            getattr(run.trainer, f"target_{n}").load_state_dict(
                state["target"]["nets"][n])
        run.trainer._target_snapshot = state["target"]["snapshot"]
        run.trainer._target_synced_at = state["target"]["synced_at"]
    run.replay = state["replay"]
    run.env_steps = state["env_steps"]
    run.trainer.training_steps = state["training_steps"]
    np.random.set_state(state["rng"]["np"])
    random.setstate(state["rng"]["py"])
    torch.set_rng_state(state["rng"]["torch"])
    if state["rng"]["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["rng"]["cuda"])
    return {"iteration": state["iteration"], "elapsed_s": state["elapsed_s"],
            "init_weights": state.get("init_weights")}


def mark_run_completed(path, meta):
    """Replace the bulky checkpoint with a small completion marker, so a
    multi-seed --resume sweep skips finished (arm, seed) pairs instead of
    re-running them from scratch."""
    tmp = path.with_suffix(".tmp")
    torch.save({**meta, "completed": True}, tmp)
    os.replace(tmp, path)


def read_rows_tolerant(path):
    """jsonl rows, tolerating ONE torn final line (a kill mid-append is not
    atomic; the torn tail is by definition stale for the dead writer) --
    returned as (rows, torn) with torn=True when a final partial line was
    dropped. Corruption anywhere else still raises: that is not a kill
    artifact, and silently skipping it would hide real damage.

    NOTE: one TAG's jsonl has ONE writer at a time. The harness is a
    single-process design; a second same-TAG run appending concurrently can
    race prune's read-rewrite and lose rows."""
    lines = [l for l in open(path) if l.strip()]
    rows = []
    for i, l in enumerate(lines):
        try:
            rows.append(json.loads(l))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                return rows, True
            raise
    return rows, False


def prune_stale_rows(out, arm, seed, start_it):
    """A kill between an eval row and the next checkpoint leaves jsonl rows
    from the abandoned timeline; the resumed run re-runs those iterations
    and appends its own rows, and plot() would average both timelines as if
    they were extra seeds. Drop this (arm, seed)'s rows beyond the
    checkpoint (start_it=0 drops them all), plus any torn final line the
    kill left. Returns the number of lines dropped."""
    out = pathlib.Path(out)
    if not out.exists():
        return 0
    rows, torn = read_rows_tolerant(out)
    keep = [r for r in rows
            if not (isinstance(r, dict)
                    and r.get("arm") == arm and r.get("seed") == seed
                    and r.get("iteration", 0) > start_it)]
    if len(keep) == len(rows) and not torn:
        return 0
    tmp = pathlib.Path(str(out) + ".tmp")
    with open(tmp, "w") as fh:
        for r in keep:
            fh.write(json.dumps(r) + "\n")
    os.replace(tmp, out)
    return len(rows) - len(keep) + (1 if torn else 0)


def rows_for_pair(out, arm, seed):
    """How many jsonl rows this (arm, seed) already has; undecodable and
    non-dict lines are ignored here (existence check, not a parse)."""
    out = pathlib.Path(out)
    if not out.exists():
        return 0
    n = 0
    for l in open(out):
        if not l.strip():
            continue
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue
        if isinstance(r, dict) and r.get("arm") == arm and r.get("seed") == seed:
            n += 1
    return n


def heal_torn_tail(out):
    """A torn final line -- from a kill mid-append, or any external damage
    -- is tolerated by the readers but MUST NOT be appended after: the next
    row would concatenate onto (or follow) the fragment, converting a
    benign torn tail into mid-file corruption that every reader raises on
    (round-3 HIGH; --overwrite, the advertised recovery, crashed on it
    too). Heal keys on the READERS' OWN verdict, not just a missing final
    newline: round 4 showed a newline-TERMINATED undecodable final line
    was tolerated by reads yet skipped by a byte-only heal, re-arming the
    same landmine. Rewrite without the fragment before anything appends.
    Returns True if healed."""
    out = pathlib.Path(out)
    if not out.exists() or out.stat().st_size == 0:
        return False
    try:
        rows, torn = read_rows_tolerant(out)
    except json.JSONDecodeError as e:
        # Mid-file corruption is NOT a kill artifact (heal runs before
        # every append era) and --overwrite must not "fix" it -- overwrite
        # is pair-scoped, and the damaged line may belong to another
        # pair's row. This needs a human, so fail with the path and say so.
        raise SystemExit(
            f"{out}: undecodable JSON mid-file ({e}). Not a kill artifact "
            f"and not repairable by --overwrite (which only prunes one "
            f"pair's rows) -- inspect the file by hand, or point "
            f"ABLATION_RESULTS elsewhere.")
    with open(out, "rb") as fh:
        fh.seek(-1, 2)
        ends_nl = fh.read(1) == b"\n"
    if not torn and ends_nl:
        return False
    tmp = pathlib.Path(str(out) + ".tmp")
    with open(tmp, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    os.replace(tmp, out)
    return True


def refuse_conflicting_state(ck, out, arm, seed, resume, overwrite):
    """The jsonl rows are the experiment record and a checkpoint is a killed
    run's resumable state; a fresh run over EITHER appends a second timeline
    that plot() would average as extra seeds -- the accident class that has
    twice contaminated live results. Refusal happens here, before the env
    or nets are built -- except the recipe MISMATCH, which surfaces from
    load_run_checkpoint after Run construction (it needs the checkpoint's
    contents, not read here). --overwrite is the mechanized opt-out: it
    drops this (arm, seed)'s rows and checkpoint and starts clean, so no
    advice ever funnels through hand-deleting files into a mixed
    timeline."""
    if heal_torn_tail(out):
        print(f"  [{arm} s{seed}] healed a torn jsonl tail (kill mid-append "
              f"or external damage)", flush=True)
    if overwrite:
        dropped = prune_stale_rows(out, arm, seed, 0)
        for stale in (ck, RESULTS / f"weights{TAG}-{arm}-seed{seed}.pth"):
            # The pair's weights go too: a redo dying mid-run must not leave
            # the OLD run's completed weights beside the NEW run's rows.
            if stale.exists():
                stale.unlink()
        if dropped:
            print(f"  [{arm} s{seed}] --overwrite dropped {dropped} old "
                  f"line(s)", flush=True)
        return
    if resume:
        if not ck.exists() and rows_for_pair(out, arm, seed):
            raise SystemExit(
                f"--resume for ({arm}, s{seed}) found no checkpoint at "
                f"{ck.name}, but {pathlib.Path(out).name} already has rows "
                f"for it (a run from before checkpointing, a deleted "
                f"checkpoint, or a FINISHED --checkpoint-every 0 run, which "
                f"writes no marker). There is nothing to resume, and starting "
                f"fresh would append a second timeline. To replace those "
                f"rows, rerun THIS command without --resume, adding: "
                f"--arms {arm} --seed {seed} --overwrite (this command's "
                f"--env/--frameskip already target these files). Or point "
                f"ABLATION_RESULTS somewhere fresh.")
        return
    if ck.exists():
        raise SystemExit(
            f"{ck} exists: a previous run of this (arm, seed) was killed "
            f"mid-flight or completed here. Pass --resume to continue/skip "
            f"it, or discard it by rerunning THIS command with: --arms "
            f"{arm} --seed {seed} --overwrite (this command's "
            f"--env/--frameskip already target these files).")
    if rows_for_pair(out, arm, seed):
        raise SystemExit(
            f"{pathlib.Path(out).name} already has rows for ({arm}, "
            f"s{seed}): a fresh run would append a second timeline over "
            f"them. Replace them by rerunning THIS command with: "
            f"--arms {arm} --seed {seed} --overwrite (this command's "
            f"--env/--frameskip already target these files), or point "
            f"ABLATION_RESULTS somewhere fresh.")


# Gentle warm start -- all three gated on init-weights lineage ONLY, so
# from-scratch arms keep their archived semantics. The first continuation
# attempt ran the loaded nets at full blast (fresh Adam, fresh projector at
# consistency weight 2.0, 2000 train steps per episode on a 10-episode
# buffer) and scrambled them: 19-32% relative parameter drift in 20k steps,
# iter-10 eval back at the cold-start floor.
WARMSTART_PREFILL_EPISODES = 10   # buffer diversity before the first gradient
WARMSTART_RAMP_ITERS = 10         # train steps/iter scale linearly over these
WARMSTART_LR_WARMUP_STEPS = 1000  # fresh-Adam lr ramps 0 -> LEARNING_RATE


def apply_warmstart_consistency(base, training_steps):
    """Warm-start-gated ramp of the SimSiam consistency weight 0 -> base
    over the lr-warmup window. The audit's point: Adam is loss-scale
    invariant for the projector (consistency is its ONLY loss), so the
    weight ramp attenuates the fresh projector's gradient pollution in the
    shared trunks WITHOUT slowing the projector's own convergence -- the
    lr warmup alone throttled cure and disease at the same rate. Parks
    exactly at base; at/past the boundary it re-asserts base, so the
    module global is always correct for whatever runs next."""
    if training_steps >= WARMSTART_LR_WARMUP_STEPS:
        muzero.CONSISTENCY_LOSS_WEIGHT = base
        return
    muzero.CONSISTENCY_LOSS_WEIGHT = (
        base * (training_steps + 1) / WARMSTART_LR_WARMUP_STEPS)


def apply_warmstart_lr(optimizers, training_steps):
    """Linear lr ramp for a warm start's FRESH optimizers: Adam's early
    bias-corrected steps are large exactly when the replay is tiny. A no-op
    at and past the boundary (the final applied value is exactly
    muzero.LEARNING_RATE, so the steady state matches a from-scratch run)."""
    if training_steps >= WARMSTART_LR_WARMUP_STEPS:
        return
    frac = (training_steps + 1) / WARMSTART_LR_WARMUP_STEPS
    for opt in optimizers:
        for g in opt.param_groups:
            g["lr"] = muzero.LEARNING_RATE * frac


def load_init_weights(trainer, path, arm=None):
    """Warm-start the four function networks from a run_one weights file
    (the completed-run artifact: representation/dynamics/prediction/
    value_prefix state_dicts -- no optimizers, no replay, no projector).
    A continuation run keeps the learned networks and rebuilds everything
    else fresh; the SimSiam projector restarts (auxiliary head, retrains
    quickly). Strict loads: an architecture mismatch must fail loudly, not
    silently skip tensors. Returns the file's recorded training_steps."""
    state = _read_checkpoint(pathlib.Path(path))
    for name in ("representation", "dynamics", "prediction", "value_prefix"):
        if name not in state:
            raise SystemExit(f"{path}: not a run_one weights file "
                             f"(missing '{name}')")
        getattr(trainer, name).load_state_dict(state[name])
    # Files saved since the gentle-warm-start era carry the projector too
    # (a fresh projector's consistency gradients were part of the
    # scrambling); older files simply lack the key.
    if "projector" in state:
        trainer.projector.load_state_dict(state["projector"])
    if arm and state.get("arm") and state["arm"] != arm:
        # A cross-arm warm start may be intentional research; warn, don't
        # refuse -- but never silently.
        print(f"  WARNING: init weights were trained by arm "
              f"'{state['arm']}', loading into '{arm}'", flush=True)
    return state.get("training_steps")


def handle_completed_marker(ck, out, arm, seed):
    """A completed marker with live rows is a genuinely finished pair:
    'skip'. With ZERO live rows it is an ARCHIVED era's leftover (a
    current-era completion always writes at least one row) -- drop it and
    run 'fresh'. The redo is announced by the caller, never silent."""
    if rows_for_pair(out, arm, seed):
        return "skip"
    ck.unlink()
    return "fresh"


def evict_replay(replay):
    """FIFO eviction under BOTH caps: muzero.MAX_REPLAY (trajectories) and
    muzero.MAX_REPLAY_TRANSITIONS (transitions; 0 = off). Whole trajectories
    only -- an unroll window or n-step target must never cross an eviction
    seam -- oldest first, always keeping the newest even if it alone
    exceeds the transition cap. PER sampling is untouched: priorities live
    on the surviving records; eviction just decides which records survive."""
    while len(replay) > muzero.MAX_REPLAY:
        replay.pop(0)
    cap = muzero.MAX_REPLAY_TRANSITIONS
    if cap:
        total = sum(len(t) for t in replay)
        while total > cap and len(replay) > 1:
            total -= len(replay.pop(0))


def seed_list(seed, seeds):
    """--seed N runs exactly one seed; otherwise 0..seeds-1."""
    return [seed] if seed is not None else list(range(seeds))


def env_tag_fragment(env):
    """Filename-safe TAG fragment from an env id. ALE-namespaced ids
    ('ALE/Pong-v5') carry a path separator: unsanitized, every output
    resolved into a nonexistent ablation-ale/ subdirectory and the run
    died at its FIRST write -- after training for eval_every iterations
    with nothing persisted (round-7 finding, reproduced)."""
    import re
    return re.sub(r"[^a-z0-9]+", "", env.split("-")[0].lower())


def run_expect(arm, seed, iterations, train_steps, episodes_per_iter, eval_every):
    """The validated recipe. EPISODE_CAP and eval_every are part of the
    experiment too: a resume that forgot --episode-cap would silently
    continue with truncated episodes stitched onto the first half's
    replay."""
    return {"arm": arm, "seed": seed, "env": ENV_NAME,
            "iterations": iterations, "train_steps_per_iter": train_steps,
            "episodes_per_iter": episodes_per_iter,
            "episode_cap": EPISODE_CAP, "eval_every": eval_every}


def run_one(arm, seed, iterations, episodes_per_iter, train_steps, eval_every, out,
            checkpoint_every=10, resume=False, overwrite=False,
            init_weights=None):
    with arm_config(arm) as spec, _run_length_globals(iterations, train_steps):
        ck = resume_path(arm, seed)
        expect = run_expect(arm, seed, iterations, train_steps,
                            episodes_per_iter, eval_every)
        if muzero.USE_ATARI_WRAPPERS and FRAMESKIP is None:
            # The wrapper stack forces fixed skip 4 regardless; without the
            # flag the TAG lacks "-skip4" and rows would land in the
            # UNTAGGED (stochastic-skip) history -- the banner would also
            # print "skip registered" while running fixed 4.
            raise SystemExit(
                f"arm '{arm}' uses the Atari wrapper stack, which forces "
                f"fixed skip 4 -- pass --frameskip 4 so the output files "
                f"carry the -skip4 tag instead of mixing into the "
                f"stochastic-skip history.")
        if init_weights and overwrite:
            # --overwrite unlinks the pair's own weights file BEFORE the
            # init load would read it: the natural continue-in-place command
            # would destroy the only copy of the trained nets (audit
            # finding). Refuse identity; copying the file out is the fix.
            pair_w = RESULTS / f"weights{TAG}-{arm}-seed{seed}.pth"
            if pathlib.Path(init_weights).resolve() == pair_w.resolve():
                raise SystemExit(
                    f"--init-weights points at the pair's own live weights "
                    f"file, which --overwrite is about to delete. Copy "
                    f"{pair_w.name} somewhere else (an archive folder) and "
                    f"point --init-weights there.")
        refuse_conflicting_state(ck, out, arm, seed, resume, overwrite)
        run = Run(arm, seed, spec)
        started = time.time()
        start_it = 0
        # Lineage survives a pause/resume via the checkpoint (audit finding:
        # the CLI arg alone stamped rows 1..pause and stranded the rest).
        init_name = pathlib.Path(init_weights).name if init_weights else None
        # Captured under arm_config so the consistency ramp parks at THIS
        # arm's weight and cannot leak across arms in a sweep.
        consistency_base = muzero.CONSISTENCY_LOSS_WEIGHT
        if resume and ck.exists():
            try:
                loaded = load_run_checkpoint(run, ck, expect)
                if loaded is not None:
                    start_it = loaded["iteration"]
                    started = time.time() - loaded["elapsed_s"]
                    init_name = loaded.get("init_weights") or init_name
                    pruned = prune_stale_rows(out, arm, seed, start_it)
            except Exception:
                run.env.close()
                raise
            if loaded is None:
                if handle_completed_marker(ck, out, arm, seed) == "fresh":
                    print(f"  [{arm} s{seed}] completion marker refers to an "
                          f"archived era; removed, starting fresh", flush=True)
                else:
                    print(f"  [{arm} s{seed}] already completed, skipping",
                          flush=True)
                    run.env.close()
                    return
            else:
                if pruned:
                    print(f"  [{arm} s{seed}] pruned {pruned} stale line(s) "
                          f"from the abandoned timeline", flush=True)
                print(f"  [{arm} s{seed}] resumed after iter {start_it} "
                      f"({run.env_steps} env steps, "
                      f"{run.trainer.training_steps} train steps)", flush=True)
        elif resume:
            # refuse_conflicting_state guarantees no rows exist for this
            # pair on this path, so a fresh start cannot mix timelines.
            print(f"  [{arm} s{seed}] --resume but no checkpoint at {ck.name}; "
                  f"starting fresh", flush=True)
        if init_weights and start_it == 0:
            try:
                prior = load_init_weights(run.trainer, init_weights, arm=arm)
            except BaseException:
                run.env.close()
                raise
            print(f"  [{arm} s{seed}] warm-started from "
                  f"{pathlib.Path(init_weights).name}"
                  f"{f' ({prior} prior train steps)' if prior else ''}; "
                  f"replay/optimizers/schedules start fresh", flush=True)
            for _ in range(WARMSTART_PREFILL_EPISODES):
                traj, _, _, terminated = run.play_episode()
                run.replay.append(run.reanalyse(traj, terminated=terminated))
            print(f"  [{arm} s{seed}] pre-filled replay with "
                  f"{WARMSTART_PREFILL_EPISODES} episodes "
                  f"({run.env_steps} env steps) before the first gradient",
                  flush=True)
        for it in range(start_it + 1, iterations + 1):
            # Paper acting mode reads training progress for its temperature
            # schedule from the searcher's manager; mirror the trainer's count.
            run.searcher.network_manager.training_steps = run.trainer.training_steps
            for _ in range(episodes_per_iter):
                traj, _, _, terminated = run.play_episode()
                run.replay.append(run.reanalyse(traj, terminated=terminated))
                evict_replay(run.replay)

            steps_this_iter = train_steps
            ramp_iters = min(WARMSTART_RAMP_ITERS, iterations)
            if init_name and it <= ramp_iters:
                steps_this_iter = max(1, train_steps * it // ramp_iters)
            run.set_train_mode(True)
            for _ in range(steps_this_iter):
                if init_name:
                    apply_warmstart_lr(
                        (run.trainer.rep_optimizer, run.trainer.dyn_optimizer,
                         run.trainer.pred_optimizer,
                         run.trainer.prefix_optimizer,
                         run.trainer.proj_optimizer),
                        run.trainer.training_steps)
                    apply_warmstart_consistency(
                        consistency_base, run.trainer.training_steps)
                batch = asyncio.run(run.trainer.prepare_batch(
                    run.replay, muzero.TRAINING_BATCHES, muzero.UNROLL_STEPS, ACTIONS))
                asyncio.run(run.trainer.train_step(batch))
                run.trainer.training_steps += 1
            run.set_train_mode(False)

            if (muzero.PAPER_TARGETS and muzero.POLICY_REFRESH_EVERY
                    and it % muzero.POLICY_REFRESH_EVERY == 0):
                run.refresh_targets(muzero.POLICY_REFRESH_COUNT)

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
                    # For future warm starts: a FRESH projector's consistency
                    # gradients helped scramble continuation attempt 1.
                    "projector": run.trainer.projector.state_dict(),
                }, ckpt)
                print(f"  saved {ckpt.name}", flush=True)

            if (eval_every and it % eval_every == 0) or it == iterations:
                metrics = run.evaluate()
                # The EFFECTIVE skip. The wrapper stack's skip-4 default is
                # unreachable now that wrapper arms REFUSE to run without
                # --frameskip, but this stays belt-and-braces: recording the
                # raw flag (None) would misdescribe the env if the refusal
                # ever regressed.
                eff_skip = (FRAMESKIP or 4) if muzero.USE_ATARI_WRAPPERS else FRAMESKIP
                record = {"arm": arm, "seed": seed,
                          "env": ENV_NAME, "frameskip": eff_skip,
                          "wrappers": bool(muzero.USE_ATARI_WRAPPERS),
                          # A warm-started row is a DIFFERENT lineage from a
                          # from-scratch row; the field keeps them separable
                          # (init_name flows through the checkpoint on
                          # resume, where the CLI arg is forbidden).
                          **({"init_weights": init_name} if init_name else {}),
                          "iteration": it,
                          "train_steps": run.trainer.training_steps,
                          "env_steps": run.env_steps,
                          "elapsed_s": round(time.time() - started, 1), **metrics}
                with open(out, "a") as fh:
                    fh.write(json.dumps(record) + "\n")
                print(f"  [{arm} s{seed}] iter {it:>3}  return {metrics['return_mean']:>7.1f}  "
                      f"prefix {metrics['prefix']:>6.2f} / control {metrics['control']:>6.2f}  "
                      f"per-step {metrics['per_step']:>6.2f}  "
                      f"H(pi) {metrics.get('policy_entropy', 0.0):>5.2f}", flush=True)

            # After the eval/record block, so a resumed run never repeats or
            # skips a jsonl row: it continues at it+1 with row `it` written.
            if checkpoint_every and it % checkpoint_every == 0 and it != iterations:
                save_run_checkpoint(run, ck, {
                    **expect, "iteration": it,
                    "training_steps": run.trainer.training_steps,
                    "elapsed_s": round(time.time() - started, 1),
                    "init_weights": init_name})
                print(f"  [{arm} s{seed}] checkpoint at iter {it} "
                      f"({ck.stat().st_size / 1e9:.1f} GB)", flush=True)
        # Also under --resume with checkpointing off: otherwise the loaded
        # checkpoint survives completion and every later --resume replays
        # the same tail.
        if checkpoint_every or resume:
            mark_run_completed(ck, expect)
        run.env.close()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot(path, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Tolerant of a torn final line: --plot-only runs against live files
    # whose writer may have been killed mid-append. Non-dict JSON lines are
    # external corruption; skip rather than crash.
    try:
        rows, _ = read_rows_tolerant(path)
    except FileNotFoundError:
        print("no results to plot")
        return
    except json.JSONDecodeError as e:
        raise SystemExit(f"{path}: undecodable JSON mid-file ({e}) -- "
                         f"inspect the file by hand.")
    rows = [r for r in rows if isinstance(r, dict)]
    if not rows:
        print("no results to plot")
        return
    # The per-row env/frameskip fields are the second line of defense
    # against mixed histories -- READ them: same-TAG recipe variants
    # (Pong-v4 vs Pong-v5 share "-pong") would otherwise average into one
    # curve silently.
    # No provenance guard here: pre-2026-08-17 rows have neither field and
    # appear as (None, None) -- a DISTINCT combo, so a legacy/modern
    # straddle warns too. Guarding them out blinded the warning to exactly
    # the rows it was retrofitted for (round-8 finding).
    mixed = {(r.get("env"), r.get("frameskip"), r.get("wrappers"),
              r.get("init_weights"))
             for r in rows}
    if len(mixed) > 1:
        print(f"WARNING: {pathlib.Path(path).name} mixes runs from "
              f"{sorted(mixed, key=str)} -- these are averaged together; "
              f"separate them before drawing conclusions.")
    arms = sorted({r["arm"] for r in rows if "arm" in r})
    palette = ["#0D6E75", "#B4531A", "#5A6472", "#7A3E8F", "#2E7D32"]
    # Cycle rather than zip: a sixth arm in one file used to KeyError here.
    colours = {arm: palette[i % len(palette)] for i, arm in enumerate(arms)}

    def series(arm, key):
        by_iter = {}
        for r in rows:
            # Rows from older eras may lack newer metric fields; skip
            # rather than crash the whole plot.
            if r.get("arm") == arm and "iteration" in r and key in r:
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
                cxs, cmean, _ = series(arm, controls[key])
                if cxs == xs:   # partial rows can leave the control sparser
                    ax.plot(xs, cmean, color=colours[arm], linewidth=1,
                            linestyle="--", alpha=0.6)
        ax.set_title(f"{title}   ({note})", fontsize=11)
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.25, linewidth=0.6)
        if ax.get_legend_handles_labels()[0]:
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
    ap.add_argument("--seed", type=int, default=None,
                    help="run ONLY this seed (overrides --seeds) -- the way "
                         "to point --overwrite or --resume at a single "
                         "(arm, seed) pair")
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
                         "default keeps the env registration's skip -- "
                         "except wrapper arms, which force skip 4 and "
                         "REQUIRE this flag for tagging")
    ap.add_argument("--checkpoint-every", type=int, default=10,
                    help="save a resumable full-state checkpoint every N "
                         "iterations (replay buffer included -- can be many "
                         "GB); 0 disables. Completed runs shrink theirs to a "
                         "marker.")
    ap.add_argument("--resume", action="store_true",
                    help="continue each (arm, seed) from its checkpoint if "
                         "one exists; the recipe must match the original "
                         "invocation")
    ap.add_argument("--overwrite", action="store_true",
                    help="discard each (arm, seed)'s previous jsonl rows and "
                         "checkpoint and start it fresh -- the mechanized "
                         "alternative to hand-deleting files")
    ap.add_argument("--init-weights", default=None, metavar="PATH",
                    help="warm-start the four function networks from a "
                         "run_one weights .pth (a continuation run: learned "
                         "nets carry over; replay, optimizers, and schedules "
                         "start fresh)")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    if args.resume and args.overwrite:
        # A hedging "--resume --overwrite" must not silently destroy a
        # resumable checkpoint (round-3 MEDIUM: overwrite won, wiping a
        # 45-iteration checkpoint with zero warning).
        ap.error("--resume and --overwrite are mutually exclusive: "
                 "resume continues saved state, overwrite destroys it")
    if args.resume and args.init_weights:
        ap.error("--resume and --init-weights are mutually exclusive: "
                 "a resumed run already has its weights; loading over them "
                 "would corrupt the continuation")
    if (any(ARMS[a].get("USE_ATARI_WRAPPERS", muzero.USE_ATARI_WRAPPERS)
            for a in args.arms)
            and not args.frameskip and not args.plot_only):
        # Pre-flight, not just the per-pair check inside run_one: in a
        # mixed sweep the non-wrapper arms would otherwise run for hours
        # (into the UNTAGGED history) before the first wrapper arm killed
        # the sweep mid-way (round-10 finding).
        ap.error(f"arms using the Atari wrapper stack "
                 f"({[a for a in args.arms if ARMS[a].get('USE_ATARI_WRAPPERS', muzero.USE_ATARI_WRAPPERS)]}) "
                 f"force fixed skip 4 -- pass --frameskip 4 so output files "
                 f"carry the -skip4 tag")

    ENV_NAME = args.env
    if args.env != "MsPacman-v4":
        TAG = "-" + env_tag_fragment(args.env)
    if args.episode_cap:
        EPISODE_CAP = args.episode_cap
    if args.frameskip:
        FRAMESKIP = args.frameskip
        # A frameskip override is a DIFFERENT MDP: suffix every output file so
        # fixed-skip rows can never mix into a stochastic-skip history (the
        # per-row env/frameskip fields in the jsonl are the second line of
        # defense).
        TAG += f"-skip{args.frameskip}"
    if args.actions:
        ACTIONS = args.actions
        TAG += f"-act{args.actions}"
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
              f"arms {args.arms} | seeds {seed_list(args.seed, args.seeds)} | "
              f"{args.iterations} iterations x ({args.episodes_per_iter} episodes + "
              f"{args.train_steps} train steps)")
        for arm in args.arms:
            print(f"\n=== {arm}: {ARMS[arm]['_doc']}")
            for seed in seed_list(args.seed, args.seeds):
                run_one(arm, seed, args.iterations, args.episodes_per_iter,
                        args.train_steps, args.eval_every, out,
                        checkpoint_every=args.checkpoint_every,
                        resume=args.resume, overwrite=args.overwrite,
                        init_weights=args.init_weights)
    plot(out, RESULTS)


if __name__ == "__main__":
    main()
