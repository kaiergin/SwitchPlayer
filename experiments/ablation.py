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
import copy
import json
import math
import os
import pathlib
import pickle
import random
import sys
import time
import uuid

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import muzero
# Constants that arms may override are NEVER from-imported here -- they are
# read muzero.-qualified at call time so overrides genuinely apply.
from muzero import (IMAGE_RESHAPE, MCTSHelper, TrainingHelper,
                    initial_frame_stack, preprocess_frame, push_frame)
# No direct networks imports: all fresh-net construction goes through
# muzero's call-time factories (build_representation & co.), the one place
# the architecture knobs are resolved. Direct construction here is the
# import-time-capture escape hatch the 2026-08-28 audit closed.

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
        # SEQUENTIAL simulations -- the papers' schedule, restored
        # 2026-08-30. Batch 16 (2026-08-29..30) was chosen on KL/signal-
        # to-noise metrics that are DIRECTION-BLIND: the paired re-probe
        # (seed-matched Dirichlet, wave width the only variable, arch1
        # weights) shows batched waves flatten the visit policy dose-
        # dependently -- dH vs sequential +0.025/+0.064/+0.135/+0.171
        # at widths 2/4/8/16 -- because in-wave statistics are frozen (a
        # 50-sim search refreshes stats 50 times sequentially, only 4 at
        # width 16). At width 16 targets sit near PRIOR entropy, so every
        # policy update flattens the prior: the arch1 run's climbing
        # H(pi) (1.10->1.34, iters 15-45). No width is free; buy acting
        # speed with --ray, not wave width.
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
        # SEQUENTIAL simulations -- the papers' schedule, restored
        # 2026-08-30. Batch 16 (2026-08-29..30) was chosen on KL/signal-
        # to-noise metrics that are DIRECTION-BLIND: the paired re-probe
        # (seed-matched Dirichlet, wave width the only variable, arch1
        # weights) shows batched waves flatten the visit policy dose-
        # dependently -- dH vs sequential +0.025/+0.064/+0.135/+0.171
        # at widths 2/4/8/16 -- because in-wave statistics are frozen (a
        # 50-sim search refreshes stats 50 times sequentially, only 4 at
        # width 16). At width 16 targets sit near PRIOR entropy, so every
        # policy update flattens the prior: the arch1 run's climbing
        # H(pi) (1.10->1.34, iters 15-45). No width is free; buy acting
        # speed with --ray, not wave width.
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
        "MCTS_BATCHES": 1,  # sequential, see effzero-repro's comment
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
        "MCTS_BATCHES": 1,  # sequential, see effzero-repro's comment
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
        # 31.25k matches the communal arm's CURRENT cap exactly -- the pair
        # must differ only in architecture (audit: 62.5k here would have
        # confounded network size with a 2x replay window).
        "MAX_REPLAY_TRANSITIONS": 31_250,
        "LATENT_CHANNELS": 64,
        "REP_TRUNK_BLOCKS": 1,
        "DYN_BLOCKS": 1,
        "PRED_HEAD_CHANNELS": 16,
        "PRED_FC_HIDDEN": 32,
        "REWARD_HEAD_CHANNELS": 16,
        "ACTION_ENCODING": "plane",
        "_doc": "effzero-repro-refresh5x-communal at EfficientZero's Atari "
                "network dimensions (64ch, 1/1 blocks, 16-ch value/policy/"
                "reward heads, [32] hiddens, single-plane actions). Run "
                "from scratch with the communal arm's command, only --arms "
                "differing; its control is a from-scratch run of the "
                "repo-dims architecture at the same 31.25k buffer.",
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
        # SEQUENTIAL simulations -- the papers' schedule, restored
        # 2026-08-30. Batch 16 (2026-08-29..30) was chosen on KL/signal-
        # to-noise metrics that are DIRECTION-BLIND: the paired re-probe
        # (seed-matched Dirichlet, wave width the only variable, arch1
        # weights) shows batched waves flatten the visit policy dose-
        # dependently -- dH vs sequential +0.025/+0.064/+0.135/+0.171
        # at widths 2/4/8/16 -- because in-wave statistics are frozen (a
        # 50-sim search refreshes stats 50 times sequentially, only 4 at
        # width 16). At width 16 targets sit near PRIOR entropy, so every
        # policy update flattens the prior: the arch1 run's climbing
        # H(pi) (1.10->1.34, iters 15-45). No width is free; buy acting
        # speed with --ray, not wave width.
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
                "the iter-20 checkpoint. ARCHITECTURE NOTE (2026-08-28): "
                "the residual-dynamics/BN commit renamed state_dict keys, "
                "so --init-weights strict-loads FAIL against any parent "
                "trained before it -- warm starts require a "
                "post-architecture parent run.",
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

# Phase 4 arm (2026-09-02, Kai's architecture): CPU latency-lane actor +
# GPU vectorized refresh. DERIVED from the communal arm; refresh topology
# changes to fresh-root vtree at 5x coverage with the entropy package.
ARMS["effzero-repro-vtree-refresh"] = {
    **ARMS["effzero-repro-refresh5x-communal"],
    "USE_COMMUNAL_REANALYSIS": False,
    "REFRESH_VTREE": True,
    "REFRESH_ENTROPY_PRIORITY": True,
    "REFRESH_NO_SHARPEN_GUARD": True,
    "POLICY_REFRESH_COUNT": 20,
    "_doc": "effzero-repro-refresh5x-communal with refresh moved to the "
            "phase-4 vectorized engine (2026-09-02): fresh-root GPU "
            "searches against the target model at VTREE_REFRESH_BATCH "
            "positions per wave (measured 307 searches/s at 256; early "
            "whole-buffer refreshes cost ~100s inline every 2 iters, "
            "shrinking as the guard retires steps -- no offload actor), "
            "flattest-first candidates, no-sharpen guard, COUNT 4->20. "
            "Acting is unchanged sequential PUCT; run with --ray "
            "--cpu-actor so the env actor holds local CPU nets and the "
            "GPU belongs entirely to training + refresh. Control: "
            "pong-archfix-2026-08-30 (same command minus the flags; "
            "deltas = refresh topology+coverage, actor device).",
}

# vtree v2 arm (2026-09-11): archfix's EXACT recipe with the refresh
# search moved onto the vectorized COMMUNAL engine -- one flag away.
# The 2026-09-11 in-situ probe showed fresh-root and communal targets
# disagree on the best action ~42% of the time (communal flatter, H
# 0.89 vs 0.74); no fresh-root arm reproduced archfix's endpoint, and
# that topology difference is the leading suspect. This arm keeps the
# communal topology (seeded chain, COMMUNAL_WINDOW_ROOTS lockstep,
# communal statistics, bookkeeping visit subtracted; parity-gated
# bitwise against run_communal_mcts in tests/test_vtree_communal.py)
# and changes NOTHING else: rotation candidates, COUNT 4 per 2 iters,
# no entropy priority, no guard, sequential 50-sim acting.
ARMS["effzero-repro-vtree-communal"] = {
    **ARMS["effzero-repro-refresh5x-communal"],
    "REFRESH_VTREE": True,
    "_doc": "effzero-repro-refresh5x-communal with ONLY REFRESH_VTREE "
            "on: the same communal refresh (shared seeded chain tree, "
            "16-root windows, communal statistics) computed by the v2 "
            "vectorized engine on the trainer's device against the "
            "lagged target model, parity-gated bitwise against the "
            "sequential engine. Everything else is archfix's recipe. "
            "Run with archfix's exact command (sequential mode; --ray "
            "--cpu-actor is the phase-4 speed variant, whose deltas then "
            "also include the actor device). Control: "
            "pong-archfix-2026-08-30.",
}

# Speed configuration of the arm above (2026-09-11, audit round 2): the
# same communal target semantics with 8 trajectories per forest.
ARMS["effzero-repro-vtree-communal8"] = {
    **ARMS["effzero-repro-vtree-communal"],
    "VTREE_COMMUNAL_TRAJS": 8,
    "_doc": "effzero-repro-vtree-communal with VTREE_COMMUNAL_TRAJS=8: "
            "archfix's communal target semantics at ~3.4x the sequential "
            "engine's refresh throughput (measured 109-111 vs 32 pos/s "
            "on the RTX 4070, production-shaped nets, 16-root windows, 50 "
            "sims). NOT bitwise vs the sequential engine: eight chain "
            "trees share each network batch, and the communal tree "
            "cascades the batch-shape wobble -- measured G=8 vs G=1 "
            "(noise off, tie-first, deterministic cuDNN): ~96% of roots "
            "keep identical visit counts, their root values shifting by a "
            "median 3e-3 (tail to 1.5 where a neighbour's path changed); "
            "~4% of roots flip counts, values shifting by a median 0.26, "
            "max 2.1. ~1.9 GB steady "
            "at production chain length (~4.3 GB if a forest growth "
            "triggers). Run with archfix's exact command; control: "
            "pong-archfix-2026-08-30 and the exact arm.",
}

# Higher-refresh-rate arm (2026-09-14, Kai): archfix's recipe with the
# communal vtree engine at TRAJS=8 paying for THREE TIMES archfix's
# refresh coverage at roughly archfix's refresh wall clock (3.4x engine
# throughput; COUNT 4 -> 12 per 2 iterations). Coverage by phase: the
# mid-run buffer (~20 trajs of 1500-2000 steps) turns over every ~3-4
# iterations instead of ~10; LATE, with episodes near the 3000 cap,
# the 31,250-transition buffer holds only ~10-13 trajectories, so
# COUNT 12 refreshes the ENTIRE buffer every cadence (rotation
# degenerates to whole-buffer, grouped e.g. 8+3) -- audit B2. The
# intended variable is the refresh RATE; the engine swap is its enabler
# (archfix-class targets, ~4% of roots perturbed below search noise).
# BOOTSTRAP_CACHE is OFF so the value-target builder is archfix's exact
# path: the cache was the one code delta between archfix (+17.1) and
# both non-crossing replications, and this run must not carry it.
# Question: does 3x refresh make the late consolidation MORE CONSISTENT
# (archfix crossed 1 run in 3 at 1x)? Run SEQUENTIAL (no --ray).
ARMS["effzero-repro-vtree-communal8-r12"] = {
    **ARMS["effzero-repro-vtree-communal8"],
    "POLICY_REFRESH_COUNT": 12,
    "BOOTSTRAP_CACHE": False,
    "_doc": "effzero-repro-vtree-communal8 with POLICY_REFRESH_COUNT 12 "
            "(3x archfix's 4 per 2 iterations, bought by the vectorized "
            "communal engine's ~3.4x throughput) and BOOTSTRAP_CACHE off "
            "(archfix's exact value-target builder). Deltas vs archfix's "
            "arm: REFRESH_VTREE, VTREE_COMMUNAL_TRAJS 8, POLICY_REFRESH_"
            "COUNT 12, BOOTSTRAP_CACHE False. Run sequential. Controls: "
            "pong-archfix-2026-08-30 (+17.1 @90), pong-archfix-ray-"
            "2026-09-11 (-11.4), pong-archfix-seq-s1-2026-09-13 (-14.0).",
}

# Edge-visit arm (2026-09-15, Kai): r12's recipe with COMMUNAL_EDGE_VISITS
# on -- PUCT and the readout count only selections that crossed the
# parent->child edge, so root n+1's own simulation budget no longer lands
# on root n's behaviour action (legacy: a floor of about half the target
# mass on the behaviour action at 15 of 16 window slots, measured on
# r12's checkpoint -- the largest single self-imitation term in
# archfix-class targets). Everything else identical to r12. Run
# SEQUENTIAL (no --ray).
ARMS["effzero-repro-vtree-communal8-r12-edge"] = {
    **ARMS["effzero-repro-vtree-communal8-r12"],
    "COMMUNAL_EDGE_VISITS": True,
    "_doc": "effzero-repro-vtree-communal8-r12 with COMMUNAL_EDGE_VISITS "
            "True: communal PUCT and readout use edge visits (visit_count "
            "- root_visits), removing the successor root's simulation "
            "budget from each root's behaviour-action count. Single delta "
            "vs r12. Controls: pong-r12-2026-09-14 (legacy readout), "
            "pong-archfix-2026-08-30 (+17.1 @90).",
}

# Edge visits at archfix's EXACT recipe (2026-09-21, Kai): the literal
# archfix arm -- sequential communal engine (no REFRESH_VTREE), coverage
# 4 per 2 iterations, bootstrap cache OFF (archfix predates the cache and
# the module default is on) -- with COMMUNAL_EDGE_VISITS the single
# training delta. Three archfix-recipe runs are its controls instead of
# r12's one. Run SEQUENTIAL (no --ray), with --snapshot-weights so the
# counterfactual-dynamics probes can time projector collapse.
ARMS["effzero-repro-refresh5x-communal-edge"] = {
    **ARMS["effzero-repro-refresh5x-communal"],
    "COMMUNAL_EDGE_VISITS": True,
    "BOOTSTRAP_CACHE": False,
    "_doc": "effzero-repro-refresh5x-communal (archfix) with "
            "COMMUNAL_EDGE_VISITS True and BOOTSTRAP_CACHE False (archfix's "
            "exact value-target builder): communal PUCT and readout count "
            "edge visits, so the successor root's simulation budget no "
            "longer lands on each root's behaviour action. Run sequential "
            "with archfix's command plus --snapshot-weights. Controls: "
            "pong-archfix-2026-08-30 (+17.1 @90, seed 0), pong-archfix-ray-"
            "2026-09-11 (-11.4, seed 0), pong-archfix-seq-s1-2026-09-13 "
            "(-14.0, seed 1).",
}

# Augmentation fidelity fix (2026-09-24, Kai; audit finding H1): the edge arm
# with the online root and the SimSiam targets augmented by INDEPENDENT draws,
# as EfficientZero's released code does. The single delta against the edge
# pair, whose per-eval snapshots time the collapse (batch-stat projector rank
# 34-55 at iter 10 -> 2.5-3.8 at iter 15, both seeds), so the readout is
# probe_counterfactual_edge / probe_shift_collapse on this arm's snapshots at
# iters 15-30, not the return at 90. Same seeds, same command.
ARMS["effzero-repro-refresh5x-communal-edge-augfix"] = {
    **ARMS["effzero-repro-refresh5x-communal-edge"],
    "AUG_INDEPENDENT_TARGETS": True,
    "_doc": "effzero-repro-refresh5x-communal-edge with "
            "AUG_INDEPENDENT_TARGETS True: the online root and the SimSiam "
            "targets get independent shift/intensity draws (EfficientZero "
            "core/train.py:73-75) instead of one shared draw, which let the "
            "consistency loss match on the shift (collapsed projectors "
            "encoded 83-85% shift, probe_shift_collapse). Run with the edge "
            "command. Controls: pong-edge-s0/s1-2026-09-21 (same seeds, "
            "per-eval snapshots).",
}

# EfficientZero-fidelity bundle (2026-09-24, Kai: apply every verified
# deviation from the 09-24 audit at once, audit, run). The edge arm plus:
# independent augmentation draws and no clamp (H1, L2), EfficientZero's
# optimizer -- SGD lr 0.2, momentum 0.9, coupled weight decay 1e-4, 1000-step
# linear warmup, constant after (H2) -- their gradient placement (M2, coupled
# to SGD: without it step 0 would take 5x their step), the consistency mask
# past trajectory ends (L1), and no latent min-max renormalisation (M1).
# Plus a safety net: skip non-finite gradient steps (their AMP GradScaler
# does; in fp32 one NaN step would poison weights and momentum for good).
# Deliberately NOT included: AMP, 64 channels, PER-off, affine=False
# (SimSiam's, not EfficientZero's). A reproduction question, not an
# attribution one: if it helps, bisect only if needed. Kill early on loss
# divergence/NaN, or on a batch-stat projector collapse by iters 15-20 like
# the edge pair's.
ARMS["effzero-repro-refresh5x-communal-edge-faithful"] = {
    **ARMS["effzero-repro-refresh5x-communal-edge"],
    "AUG_INDEPENDENT_TARGETS": True,
    "AUG_CLAMP": False,
    "OPTIMIZER": "sgd",
    "LEARNING_RATE": 0.2,
    "LR_WARMUP_STEPS": 1000,
    "EFFZERO_LOSS_SCALING": True,
    "CONSISTENCY_MASK_PADDING": True,
    "LATENT_RENORMALIZE": False,
    "SKIP_NONFINITE_GRAD_STEPS": True,
    "_doc": "effzero-repro-refresh5x-communal-edge with every verified "
            "deviation from EfficientZero's released code closed: "
            "independent root/target augmentation, no clamp, SGD 0.2 / "
            "momentum 0.9 / wd 1e-4 with a 1000-step warmup, whole-loss 1/K "
            "and pre-head 0.5 gradient hook, consistency masked past "
            "trajectory ends, no latent renormalisation; non-finite gradient "
            "steps skipped (their AMP scaler's behaviour). Run with the edge "
            "command. Controls: pong-edge-s0/s1-2026-09-21.",
}

# Policy-only acting on the parallel stack (2026-09-26, Kai): the faithful
# arm with (1) the vectorized COMMUNAL GPU re-search (the vtree-communal8
# pair of flags: same communal semantics as the faithful run at ~3.4x the
# throughput; ~4% of roots flip counts vs the sequential engine, below the
# search's own repeat noise), (2) POLICY-ONLY acting -- the actor samples
# the prior at the staged temperature, and every new game is searched once
# on arrival through the ONLINE nets (BIRTH_SEARCH: the targets the acting
# search would have recorded), and (3) ReDo every 1000 steps at tau 0.025.
# Eval is then the prior's argmax (policy-only), the number the Switch
# would see. Run with --ray --cpu-actor, one seed. A bundle by design (Kai):
# if it fails, the bisect starts with search acting on this stack.
ARMS["effzero-repro-faithful-prioract"] = {
    **ARMS["effzero-repro-refresh5x-communal-edge-faithful"],
    "REFRESH_VTREE": True,
    "VTREE_COMMUNAL_TRAJS": 8,
    "USE_SEARCH_ACTING": False,
    "BIRTH_SEARCH": True,
    "PRIOR_ACT_STAGED_TEMP": True,
    "REDO_EVERY": 1000,
    "REDO_TAU": 0.025,
    "REDO_NETS": ("representation", "dynamics", "prediction", "value_prefix"),
    "_doc": "effzero-repro-refresh5x-communal-edge-faithful with the "
            "vectorized communal re-search (REFRESH_VTREE, "
            "VTREE_COMMUNAL_TRAJS 8), policy-only acting (prior sampled at "
            "the staged temperature; every new game searched once on "
            "arrival through the online nets -- BIRTH_SEARCH), policy-only "
            "eval, and ReDo (every 1000 steps, tau 0.025, projector "
            "excluded). Run with "
            "--ray --cpu-actor. Control: pong-faithful-s0-2026-09-24.",
}

ARMS["effzero-repro-faithful-searchact"] = {
    **ARMS["effzero-repro-faithful-prioract"],
    "USE_SEARCH_ACTING": True,
    "BIRTH_SEARCH": False,
    "PRIOR_ACT_STAGED_TEMP": False,
    "EVAL_PRIOR_ONLY": True,
    "_doc": "The prioract bisect's first step (Kai 2026-09-29, after "
            "prioract read WORSE): the prioract stack -- vectorized "
            "communal re-search, --ray --cpu-actor, ReDo -- with SEARCH "
            "acting restored (the CPU actor's 50-sim search with root noise "
            "and staged visit temperature records the targets; no birth "
            "search). In-run eval is the prior's argmax (EVAL_PRIOR_ONLY), "
            "matching the exact-eval readout. Controls: "
            "pong-prioract-s0-2026-09-27 and pong-faithful-s0/s1-2026-09-24.",
}

# The factorization after the bisect read WORSE (Kai, 2026-09-30): two
# concurrent runs that split the parallel-stack suspects. Their flags are
# IDENTICAL (tests/test_factor.py pins it): the only difference between
# them is the launch mode -- --ray --cpu-actor (pipelined CPU actor) vs
# sequential. With searchact and the faithful control:
#   ReDo:                 searchact  vs  searchact-noredo
#   pipelined acting:     searchact-noredo  vs  seqvtree
#   vectorized re-search: seqvtree  vs  the faithful control
ARMS["effzero-repro-faithful-searchact-noredo"] = {
    **ARMS["effzero-repro-faithful-searchact"],
    "REDO_EVERY": 0,
    "STEP_DIAGNOSTICS": True,
    "_launch": {"ray": True, "cpu_actor": True},
    "_doc": "effzero-repro-faithful-searchact with ReDo OFF (bisect step "
            "2; step diagnostics kept on). Launch pinned: --ray "
            "--cpu-actor. Controls: pong-searchact-s0-2026-09-29 (ReDo on) "
            "and effzero-repro-faithful-seqvtree (same flags, sequential).",
}

ARMS["effzero-repro-faithful-seqvtree"] = {
    **ARMS["effzero-repro-refresh5x-communal-edge-faithful"],
    "REFRESH_VTREE": True,
    "VTREE_COMMUNAL_TRAJS": 8,
    "EVAL_PRIOR_ONLY": True,
    "STEP_DIAGNOSTICS": True,
    "_launch": {"ray": False},
    "_doc": "The faithful control with the vectorized communal re-search "
            "(REFRESH_VTREE; VTREE_COMMUNAL_TRAJS 8, i.e. every refreshed "
            "trajectory -- POLICY_REFRESH_COUNT = 4 -- in one forest) and "
            "prior-argmax in-run eval (EVAL_PRIOR_ONLY), no ReDo, step "
            "diagnostics on -- bisect step 3. Launch pinned: SEQUENTIAL "
            "(no --ray). Controls: pong-faithful-s0/s1-2026-09-24 and "
            "effzero-repro-faithful-searchact-noredo.",
}

# Policy-only acting again, WITHOUT ReDo, always parallel, with each game
# added to the buffer as soon as it finishes (Kai, 2026-10-01): after the
# factorization implicated ReDo (searchact -11.37 vs noredo +5.86).
ARMS["effzero-repro-faithful-prioract-noredo"] = {
    **ARMS["effzero-repro-faithful-prioract"],
    "REDO_EVERY": 0,
    "STEP_DIAGNOSTICS": True,
    "RAY_EAGER_COLLECT": True,
    "_launch": {"ray": True, "cpu_actor": True},
    "_doc": "effzero-repro-faithful-prioract with ReDo OFF (step "
            "diagnostics on) and RAY_EAGER_COLLECT (each game joins the "
            "buffer as soon as the CPU actor finishes it, mid training "
            "block -- sequential freshness): policy-only acting (prior at "
            "the staged temperature) + birth search, vectorized communal "
            "re-search. Launch pinned: --ray --cpu-actor. Twin: "
            "pong-seqvtree-s0-2026-09-30 (search acting, no lag); also "
            "pong-noredo-s0-2026-09-30 and pong-prioract-s0-2026-09-27.",
}

# Policy-only acting on the faithful control's refresh engine (Kai,
# 2026-10-05; pre-registered in CURRICULUM): prioract2's arm with the
# vectorized communal re-search OFF, so refresh runs the control's own
# sequential run.refresh_targets inline -- the effective delta from the
# faithful control is then only the acting/pipeline side. Two seeds.
ARMS["effzero-repro-faithful-prioract-noredo-seqref"] = {
    **ARMS["effzero-repro-faithful-prioract-noredo"],
    "REFRESH_VTREE": False,
    "VTREE_COMMUNAL_TRAJS": 1,
    "_doc": "effzero-repro-faithful-prioract-noredo with REFRESH_VTREE off "
            "(VTREE_COMMUNAL_TRAJS 1, the default): the faithful control's "
            "sequential refresh engine (run.refresh_targets, inline; "
            "REFRESH_OFFLOAD off) under policy-only acting + birth search, "
            "eager collect, --ray --cpu-actor, no ReDo. Twin: pong-"
            "prioract2-s0/s1 (vectorized communal re-search); bands from "
            "the faithful control (pong-faithful-s0/s1-2026-09-24).",
}

# Low-sim search acting with birth-search targets (Kai, 2026-10-09;
# pre-registered in CURRICULUM): prioract2's arm with the ACTIONS chosen by
# an 8-sim PUCT search on the CPU actor; the 50-sim birth search still makes
# every target. Seed 0.
ARMS["effzero-repro-faithful-searchact8-birth"] = {
    **ARMS["effzero-repro-faithful-prioract-noredo"],
    "USE_SEARCH_ACTING": True,
    "SEARCH_ACT_BIRTH_TARGETS": True,
    "ACTOR_SEARCH_SIMS": 8,
    "PRIOR_ACT_STAGED_TEMP": False,
    "EVAL_PRIOR_ONLY": True,
    "_doc": "effzero-repro-faithful-prioract-noredo with the actions chosen "
            "by an 8-sim PUCT search on the CPU actor (ACTOR_SEARCH_SIMS 8; "
            "visit sampling at the staged visit temperature) while the "
            "50-sim birth search makes every target "
            "(SEARCH_ACT_BIRTH_TARGETS); prior-argmax in-run eval. Twin: "
            "pong-prioract2-s0/s1 (prior sampling); also pong-noredo-s0 "
            "(50-sim search acting) and the faithful control's bands.",
}

# The BN stabilizer on the temperature config (Kai, 2026-10-08;
# pre-registered in CURRICULUM): the actor and eval actor play with
# representation / prediction BN statistics recalibrated on clean depth-0
# frames at every weight ship. Run as seed 1.
ARMS["effzero-repro-faithful-prioract-noredo-temp05-bnrecal"] = {
    **ARMS["effzero-repro-faithful-prioract-noredo"],
    "PRIOR_ACT_TEMP_SCALE": 0.5,
    "BN_RECAL_ACTING": True,
    "_doc": "effzero-repro-faithful-prioract-noredo-temp05 + BN_RECAL_ACTING "
            "(the BN stabilizer: recalibrated representation/prediction BN "
            "statistics in every actor / eval-actor weight ship; training, "
            "target nets, birth search and refresh unchanged). Run as seed 1. "
            "Twin: pong-tempscale-s0-2026-10-05; bands from the faithful "
            "control (pong-faithful-s0/s1-2026-09-24).",
}

# SVE on the temperature + stabilizer config (Kai, 2026-10-10; pre-
# registered in CURRICULUM): the BN stabilizer arm + EfficientZero V2's
# search-based value estimation exactly as pre-registered on 09-28 (stage 2:
# fresh 16-sim searches through the target nets). Seed 1.
ARMS["effzero-repro-faithful-prioract-noredo-temp05-bnrecal-sve"] = {
    **ARMS["effzero-repro-faithful-prioract-noredo-temp05-bnrecal"],
    "VALUE_TARGET_MODE": "mixed",
    "SVE_START_TRAIN_STEPS": 40000,
    "SVE_FRESH_TRANSITIONS": 20000,
    "SVE_SOURCE": "search",
    "SVE_SEARCH_SIMS": 16,
    "_doc": "effzero-repro-faithful-prioract-noredo-temp05-bnrecal + SVE "
            "(VALUE_TARGET_MODE mixed; from 40k training steps, every real "
            "position in the window of a root outside the newest 20k "
            "transitions takes a fresh 16-sim search through the target nets "
            "as its value target). Run as seed 1. "
            "Twin: pong-bnrecal-s1-2026-10-08; bands from the faithful "
            "control.",
}

# Gumbel search acting (Kai, 2026-10-12; pre-registered in CURRICULUM): the
# searchact8 arm with the actor's 8-sim search on the Gumbel engine instead
# of PUCT; the 50-sim PUCT birth search still makes every target. Seed 0.
ARMS["effzero-repro-faithful-gumbelact8-birth"] = {
    **ARMS["effzero-repro-faithful-searchact8-birth"],
    "ACTOR_GUMBEL_SEARCH": True,
    "ACTOR_GUMBEL_STAGED_SCALE": True,
    "_doc": "effzero-repro-faithful-searchact8-birth with the actor's 8-sim "
            "search on the GUMBEL engine (ACTOR_GUMBEL_SEARCH: sequential "
            "halving over all 6 actions, fresh Gumbel draws as the "
            "exploration, the halving survivor acts) instead of PUCT, its "
            "draws scaled by the twin's staged temperature 1 / 0.5 / 0.25 "
            "(ACTOR_GUMBEL_STAGED_SCALE); the 50-sim PUCT birth search makes "
            "every target; prior-argmax in-run eval. Seed 0. Twin: "
            "pong-searchact8-s0-2026-10-09 (PUCT-8 acting); also "
            "pong-prioract2-s0 (prior sampling).",
}

# The Ms. Pac-Man port (Kai, 2026-10-16: "test the current algorithm on ms
# pacman"; pre-registered in CURRICULUM): the real-time line's config with
# its two per-TRAJECTORY knobs converted from Pong's unit (a trajectory is
# a game) to Ms. Pac-Man's (a trajectory is one of the game's 3 lives --
# EpisodicLifeEnv ends each training episode at a life loss, as in
# EfficientZero), so both stay per GAME: MAX_REPLAY 85 -> 255 keeps the
# 31,250-transition cap the binding one, as it always was in Pong (85 lives
# of ~180 random-play steps would hold ~15k), and POLICY_REFRESH_COUNT
# 4 -> 12 re-searches 4 games' worth of trajectories per 2 iterations.
# Everything else, including the launch pin, is the base arm's. Gumbel seed
# 1's pre-registered decision picks which base runs.
ARMS["effzero-repro-faithful-gumbelact8-birth-mspacman"] = {
    **ARMS["effzero-repro-faithful-gumbelact8-birth"],
    "MAX_REPLAY": 255,
    "POLICY_REFRESH_COUNT": 12,
    "_doc": "effzero-repro-faithful-gumbelact8-birth for Ms. Pac-Man (one "
            "trajectory per LIFE, 3 per game): MAX_REPLAY 255 and "
            "POLICY_REFRESH_COUNT 12 (x3, the per-game values). With 9 "
            "actions the actor's 8-sim Gumbel search considers the top 8 "
            "by Gumbel + logit and halves once. Run with --env "
            "MsPacmanNoFrameskip-v4 --frameskip 4 --episode-cap 3000 "
            "--episodes-per-iter 3 --ray --cpu-actor.",
}

ARMS["effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman"] = {
    **ARMS["effzero-repro-faithful-prioract-noredo-temp05-bnrecal"],
    "MAX_REPLAY": 255,
    "POLICY_REFRESH_COUNT": 12,
    "_doc": "effzero-repro-faithful-prioract-noredo-temp05-bnrecal for Ms. "
            "Pac-Man (one trajectory per LIFE, 3 per game): MAX_REPLAY 255 "
            "and POLICY_REFRESH_COUNT 12 (x3, the per-game values). Run "
            "with --env MsPacmanNoFrameskip-v4 --frameskip 4 --episode-cap "
            "3000 --episodes-per-iter 3 --ray --cpu-actor.",
}

# Curriculum P2a, duration tokens (Kai, 2026-10-17; pre-registered in
# CURRICULUM): the real-time base with every decision one token of Pong's
# minimal NOOP / UP / DOWN held for 4, 8, 16 or 32 frames (slot order:
# action-major; slot 0 = NOOP x4 is the placeholder action's token), and the
# early long-token bias (b 0.5 -> 0 over the first 25% of the run). Seed 1
# (twin pong-bnrecal-s1-2026-10-08).
PONG_DURATION_TOKENS = tuple((a, f) for a in (0, 2, 3) for f in (4, 8, 16, 32))
ARMS["effzero-repro-faithful-prioract-noredo-temp05-bnrecal-tokens"] = {
    **ARMS["effzero-repro-faithful-prioract-noredo-temp05-bnrecal"],
    "TOKEN_VOCAB": PONG_DURATION_TOKENS,
    "TOKEN_LONG_BIAS": 0.5,
    "TOKEN_LONG_BIAS_END": 0.25,
    "_doc": "effzero-repro-faithful-prioract-noredo-temp05-bnrecal with "
            "TOKEN_VOCAB: 12 duration tokens -- Pong's NOOP / UP / DOWN x "
            "{4, 8, 16, 32} frames -- each one decision (inner wrapper "
            "steps, summed reward, edge discount DISCOUNT ** (frames / 4), "
            "frame-time value targets), and TOKEN_LONG_BIAS 0.5 falling to "
            "0 at 25% of the run. Run with --env Pong-v4 --frameskip 4 "
            "--episode-cap 3000 (base steps = 12000 frames) "
            "--episodes-per-iter 1 --ray --cpu-actor. Twin: "
            "pong-bnrecal-s1-2026-10-08 (same arm, no tokens, seed 1).",
}

# P2a on Ms. Pac-Man with reward-clip growth (Kai, 2026-10-19; pre-registered
# in CURRICULUM): the Ms. Pac-Man real-time arm with REWARD_GROWTH -- raw
# rewards in pellets (REWARD_UNIT 10), clipped at a ceiling growing from 1
# (the twin's sign clip) to 160 (a 1600-point ghost) by half the run -- alone
# (RG) and with P2a's duration tokens over Ms. Pac-Man's 9 actions (TK).
# Seed 0 (twin mspacman-s0-2026-10-16).
ARMS["effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-rgrow"] = {
    **ARMS["effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman"],
    "REWARD_GROWTH": True,
    "REWARD_UNIT": 10.0,
    "REWARD_GROWTH_START": 1.0,
    "REWARD_GROWTH_CEILING": 160.0,
    "REWARD_GROWTH_END": 0.5,
    "_doc": "effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman "
            "with REWARD_GROWTH: raw rewards / 10 (pellet = 1) clipped per "
            "base step at a ceiling growing log-linearly from 1 (= the sign "
            "clip) at step 0 to 160 at 50% of the run, the whole replay "
            "re-clipped each iteration. Run as its twin "
            "mspacman-s0-2026-10-16: --env MsPacmanNoFrameskip-v4 "
            "--frameskip 4 --episode-cap 3000 --episodes-per-iter 3 --ray "
            "--cpu-actor.",
}
MSPACMAN_DURATION_TOKENS = tuple((a, f) for a in range(9)
                                 for f in (4, 8, 16, 32))
ARMS["effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-tokens-rgrow"] = {
    **ARMS["effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-rgrow"],
    "TOKEN_VOCAB": MSPACMAN_DURATION_TOKENS,
    "TOKEN_LONG_BIAS": 0.5,
    "TOKEN_LONG_BIAS_END": 0.25,
    # Compute knobs only (audit r1 B-M2, measured on the RTX 4070): both
    # vectorized engines hold A latents per expansion, so 36-slot trees at
    # the base's shapes do not fit 12 GB -- birth search at 256 roots per
    # chunk peaked 11.4 GB, the communal refresh at 8 trajectories per
    # forest went out of memory (2 per forest: 8.9 GB reserved). At 64
    # roots per chunk / 1 trajectory per forest (the sequential engine's
    # exact batch shapes): 3.0 / 3.8 GB, the refresh ~0.21 s per position.
    "VTREE_REFRESH_BATCH": 64,
    "VTREE_COMMUNAL_TRAJS": 1,
    "_doc": "effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman"
            "-rgrow with TOKEN_VOCAB: 36 duration tokens -- Ms. Pac-Man's 9 "
            "actions x {4, 8, 16, 32} frames, P2a's mechanics (a token's "
            "reward is its inner base steps' clipped rewards summed) -- and "
            "TOKEN_LONG_BIAS 0.5 falling to 0 at 25% of the run; birth "
            "search in chunks of 64 roots and the communal refresh one "
            "trajectory per forest (memory: 36-slot trees). Run as the "
            "rgrow arm.",
}

# Curriculum P3's distillation step (Kai, 2026-10-21; pre-registered in
# CURRICULUM): the two students of one generation swap, over the STUDENT
# vocabulary experiments/distill.py derives from the master's recorded play
# (--token-vocab, required: the TOKEN_VOCAB inherited below is the master's,
# of which the file must be an ordered subset holding every 4-frame token).
# DS: the replay pre-filled with the master's re-tokenized games and their
# master-search targets (--distill-set), DISTILL_OFFLINE_STEPS gradient steps
# before iteration 1, the master's END-STATE schedules from the start
# (ceiling 160, prior-acting temperature 0.25 x 0.5 = 0.125, no long-token
# bias, PER beta 1). FS: the TK arm over the same vocabulary, from scratch.
# Seed 0.
ARMS["effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-tokens-rgrow-student-distill"] = {
    **ARMS["effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-tokens-rgrow"],
    "REWARD_GROWTH_START": 160.0,
    "PRIOR_ACT_STAGED_TEMP": False,
    "ACTING_TEMPERATURE": 0.25,
    "TOKEN_LONG_BIAS": 0.0,
    "PER_BETA": 1.0,
    "_student": "distill",
    "_distill_offline_steps": 20000,
    "_distill_snapshot_every": 4000,
    "_doc": "effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman"
            "-tokens-rgrow as the DISTILLED STUDENT of a generation swap "
            "(curriculum P3): the student vocabulary from --token-vocab; the "
            "replay pre-filled with the master's re-tokenized games and "
            "master-search policy targets (--distill-set), 20,000 offline "
            "gradient steps on it before iteration 1 (snapshots every 4,000; "
            "the last is row 0), then the TK loop; the master's end-state "
            "schedules throughout -- reward ceiling 160 (REWARD_GROWTH_START "
            "160), prior-acting temperature 0.125 (PRIOR_ACT_STAGED_TEMP off, "
            "ACTING_TEMPERATURE 0.25 x PRIOR_ACT_TEMP_SCALE 0.5), no long-"
            "token bias, PER beta 1.0. Run as TK. Twin: the -student-fresh "
            "arm (same vocabulary, from scratch).",
}
ARMS["effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-tokens-rgrow-student-fresh"] = {
    **ARMS["effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-tokens-rgrow"],
    "_student": "fresh",
    "_doc": "effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman"
            "-tokens-rgrow as the FRESH STUDENT of a generation swap "
            "(curriculum P3): the student vocabulary from --token-vocab, "
            "from scratch, TK's schedules. Run as TK. Twin: the "
            "-student-distill arm.",
}

# Test 1 of the per-generation clip (Kai, 2026-10-24: the clip changes ONLY
# between student / master swaps; pre-registered in CURRICULUM): the
# generation-0 token master -- TK with its reward ceiling held at 1 (the
# sign clip: every Ms. Pac-Man reward is at least one REWARD_UNIT) for the
# whole run instead of growing within it. Seed 0.
ARMS["effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-tokens-gen0"] = {
    **ARMS["effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-tokens-rgrow"],
    "REWARD_GROWTH_START": 1.0,
    "REWARD_GROWTH_CEILING": 1.0,
    "_doc": "effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman"
            "-tokens-rgrow with the reward ceiling CONSTANT at 1 "
            "(REWARD_GROWTH_START = REWARD_GROWTH_CEILING = 1: each base "
            "step's reward / 10 clipped at 1 -- the sign clip -- for the "
            "whole run; a token's reward the sum of its inner steps'): "
            "generation 0 of the per-generation clip, whose ceiling moves "
            "only at a student / master swap. Everything else TK's. Run as "
            "TK.",
}

# Test 2 (Kai, 2026-10-24; pre-registered in CURRICULUM): the first swap
# under the per-generation clip -- G0 the master, two generation-1 students
# over the prep's student vocabulary AT THE PREP'S CONSTANT CEILING (the
# halved-clip rule: --token-vocab's vocab.json "student_ceiling"), on TK's
# within-run schedules; the distilled one's offline phase does not advance
# them (SCHEDULE_OFFSET_STEPS), so both see the same settings at every
# online iteration. Seed 0. REBUILT WITH THE FIX (PB's decision, 2026-10-25;
# CURRICULUM Test 2 "REBUILD WITH THE FIX"): both students carry
# PREFIX_BN_PER_STEP; the master G0 stays an off arm (its prep search keeps
# its pooled statistics).
GEN0_ARM = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
            "tokens-gen0")
ARMS["effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-tokens-gen1-distill"] = {
    **ARMS[GEN0_ARM],
    "PREFIX_BN_PER_STEP": True,
    "SCHEDULE_OFFSET_STEPS": 20000,
    "_student": "distill",
    "_distill_offline_steps": 20000,
    "_distill_snapshot_every": 4000,
    "_ceiling_from_prep": True,
    "_master_arm": GEN0_ARM,
    "_master_iterations": 90,
    "_doc": "effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman"
            "-tokens-gen0 as the DISTILLED generation-1 STUDENT (Test 2): "
            "the student vocabulary and its constant ceiling from "
            "--token-vocab (the prep's vocab.json: the halved-clip rule), "
            "the replay pre-filled with G0's re-tokenized games and "
            "master-search targets (--distill-set), 20,000 offline gradient "
            "steps (snapshots every 4,000; the last is row 0), then TK's "
            "within-run schedules keyed to the ONLINE steps "
            "(SCHEDULE_OFFSET_STEPS 20,000); PREFIX_BN_PER_STEP (PB's fix, "
            "the base from 2026-10-25). Run as TK. Twin: the -gen1-fresh "
            "arm.",
}
ARMS["effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-tokens-gen1-fresh"] = {
    **ARMS[GEN0_ARM],
    "PREFIX_BN_PER_STEP": True,
    "_student": "fresh",
    "_ceiling_from_prep": True,
    "_master_arm": GEN0_ARM,
    "_master_iterations": 90,
    "_doc": "effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman"
            "-tokens-gen0 as the FRESH generation-1 STUDENT (Test 2): the "
            "student vocabulary and its constant ceiling from --token-vocab "
            "(the prep's vocab.json), from scratch, TK's schedules; "
            "PREFIX_BN_PER_STEP (PB's fix, the base from 2026-10-25). Run "
            "as TK. Twin: the -gen1-distill arm.",
}

# Generation 2 (Kai, 2026-10-27; pre-registered in CURRICULUM): the second
# swap -- DS2 (the generation-1 distilled student, its final snapshot) the
# master, one distilled student over the prep's generation-2 vocabulary (a
# student vocabulary of DS2's: the prep's master_vocab, `_master_vocab` once
# student_arm applies --token-vocab) at the prep's constant ceiling (the
# halved-clip rule from DS2's constant ceiling). Everything else DS2's.
# Seed 0, no fresh twin.
GEN1_DISTILL_ARM = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-"
                    "mspacman-tokens-gen1-distill")
ARMS["effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-tokens-gen2-distill"] = {
    **ARMS[GEN1_DISTILL_ARM],
    "_master_arm": GEN1_DISTILL_ARM,
    "_master_iterations": 90,
    "_doc": "effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman"
            "-tokens-gen1-distill as the DISTILLED generation-2 STUDENT "
            "(Generation 2): its master DS2 (the -gen1-distill arm's final "
            "snapshot); the generation-2 vocabulary (a student vocabulary of "
            "DS2's) and its constant ceiling from --token-vocab (the prep's "
            "vocab.json: the halved-clip rule from DS2's ceiling), the "
            "replay pre-filled with DS2's re-tokenized games and "
            "master-search targets (--distill-set), 20,000 offline gradient "
            "steps (snapshots every 4,000; the last is row 0), then TK's "
            "within-run schedules keyed to the ONLINE steps "
            "(SCHEDULE_OFFSET_STEPS 20,000); PREFIX_BN_PER_STEP. Run as TK. "
            "No twin.",
}

# Test B (Kai, 2026-10-27; pre-registered in CURRICULUM): a fast lineage
# from scratch -- five generations F0-F4, each ending at the iteration in
# which it passes its env-step stop (--stop-env-steps, the supervisor's
# ~20,000), its within-run schedules following its env-step progress
# (SCHEDULE_ENV_TARGET; planned length _nominal_iterations), its own play
# logged at raw-step resolution (OWN_PLAY_LOG) for the next generation's
# distillation set. F0 is TK at the sign clip with PREFIX_BN_PER_STEP and
# DISCOUNT 0.97 per base step (BBF's start; the token arms' 0.997 per frame
# is 0.988 per base step); each later generation is the distilled student
# of the previous one at the prep's ceiling and discount (the gap 1 - g
# halved, capped at 0.997).
FAST_ARM = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
            "tokens-fast")
ARMS[FAST_ARM + "-f0"] = {
    **ARMS[GEN0_ARM],
    "PREFIX_BN_PER_STEP": True,
    "DISCOUNT": 0.97,
    "OWN_PLAY_LOG": True,
    "_env_stop": True,
    "_nominal_iterations": 30,
    "_doc": "effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman"
            "-tokens-gen0 as Test B's generation F0: from scratch at the "
            "sign clip, PREFIX_BN_PER_STEP, DISCOUNT 0.97 per base step; "
            "ends at its env-step stop (--stop-env-steps), the within-run "
            "schedules following its env-step progress (planned length 30 "
            "iterations); its own play logged at raw-step resolution for "
            "F1's distillation set. Run as TK.",
}
for _k in range(1, 5):
    ARMS[f"{FAST_ARM}-f{_k}"] = {
        **ARMS[FAST_ARM + "-f0"],
        "SCHEDULE_OFFSET_STEPS": 20000,
        "_student": "distill",
        "_distill_offline_steps": 20000,
        "_distill_snapshot_every": 4000,
        "_ceiling_from_prep": True,
        "_discount_from_prep": True,
        "_master_arm": f"{FAST_ARM}-f{_k - 1}",
        # the master ends at a run-dependent iteration (its env-step stop);
        # the prep's master_checks pins it against the snapshot
        "_master_iterations": None,
        "_doc": f"Test B's generation F{_k}: the distilled student of "
                f"F{_k - 1}'s final snapshot -- its vocabulary, constant "
                f"ceiling and discount from --token-vocab (the prep of "
                f"F{_k - 1}'s own-play log: the 1% rule, the halved-clip "
                f"rule, the gap 1 - g halved and capped at 0.997), the "
                f"replay pre-filled with that log's re-tokenized lives and "
                f"master-search targets (--distill-set), 20,000 offline "
                f"steps (snapshots every 4,000; the last is row 0), then "
                f"online play to its env-step stop, the within-run "
                f"schedules following its env-step progress; its own play "
                f"logged for the next generation. Run as TK.",
    }
del _k

# Test B2 (Kai, 2026-10-29; pre-registered in CURRICULUM): Test B's lineage
# with two swaps -- three generations T0-T2 of ~33,333 env steps each
# (planned length 50 iterations) -- at the token arms' original constant
# DISCOUNT, 0.997 per frame (0.997^4 per base step), in every generation:
# the prep keeps the master's discount for the student (_discount_rule
# "keep"; Test B's arms carry none, so distill.discount_rule's default
# "halve").
TWO_ARM = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
           "tokens-two")
ARMS[TWO_ARM + "-t0"] = {
    **ARMS[FAST_ARM + "-f0"],
    "DISCOUNT": 0.997 ** 4,
    "_nominal_iterations": 50,
    "_discount_rule": "keep",
    "_doc": "effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman"
            "-tokens-fast-f0 as Test B2's generation T0: from scratch at the "
            "sign clip, PREFIX_BN_PER_STEP, the token arms' DISCOUNT 0.997^4 "
            "per base step (0.997 per frame); ends at its env-step stop "
            "(--stop-env-steps), the within-run schedules following its "
            "env-step progress (planned length 50 iterations); its own play "
            "logged at raw-step resolution for T1's distillation set. Run as "
            "TK.",
}
for _k in range(1, 3):
    ARMS[f"{TWO_ARM}-t{_k}"] = {
        **ARMS[FAST_ARM + "-f1"],
        "DISCOUNT": 0.997 ** 4,
        "_nominal_iterations": 50,
        "_discount_rule": "keep",
        "_master_arm": f"{TWO_ARM}-t{_k - 1}",
        "_doc": f"Test B2's generation T{_k}: the distilled student of "
                f"T{_k - 1}'s final snapshot through Test B's swap -- its "
                f"vocabulary and constant ceiling from --token-vocab (the "
                f"prep of T{_k - 1}'s own-play log: the 1% rule, the "
                f"halved-clip rule), its discount the master's (0.997^4 per "
                f"base step: the prep's _discount_rule \"keep\"), the "
                f"replay pre-filled with that log's re-tokenized lives and "
                f"master-search targets (--distill-set), 20,000 offline "
                f"steps (snapshots every 4,000; the last is row 0), then "
                f"online play to its env-step stop, the within-run "
                f"schedules following its env-step progress (planned length "
                f"50 iterations); its own play logged for the next "
                f"generation. Run as TK.",
    }
del _k

# E3 (Kai, 2026-11-05; CURRICULUM "E3", amendments E3-A1..A12): Test B2's
# lineage with tokens MINED from the master's own play at both swaps. Its
# generation 0 IS Test B2's T0 (E3-A1: the frozen run, reused), so only M1
# and M2 are arms: T1 and T2 exactly, but for `_mine` (the prep runs with
# distill.py --mine: the student's vocabulary appends mined tokens, E3-A3 ..
# A6) and M2's master M1. The chain supervisor's lineage "m".
MINE_ARM = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
            "tokens-mined")
ARMS[MINE_ARM + "-m1"] = {
    **ARMS[TWO_ARM + "-t1"],
    "_master_arm": TWO_ARM + "-t0",
    "_mine": True,
    "_doc": "E3's generation M1: Test B2's T1 (...-tokens-two-t1) with "
            "mining at the swap -- the distilled student of T0's final "
            "snapshot (Test B2's own T0, reused: E3-A1) through Test B2's "
            "swap, its prep (distill.py --mine) appending up to 8 tokens "
            "mined from T0's own-play log after the 1% rule (E3-A3, A4), "
            "relabeling T0's stretches that play them (A6), every target "
            "under the path-product rule (A5); then T1's offline steps and "
            "online play to its env-step stop. Run as TK.",
}
ARMS[MINE_ARM + "-m2"] = {
    **ARMS[TWO_ARM + "-t2"],
    "_master_arm": MINE_ARM + "-m1",
    "_mine": True,
    "_doc": "E3's generation M2: Test B2's T2 (...-tokens-two-t2) with "
            "mining at the swap -- the distilled student of M1's final "
            "snapshot, its prep (distill.py --mine) treating M1's mined "
            "tokens as tokens (E3-A8) and appending tokens mined from M1's "
            "own play. Run as TK.",
}

# Test Q (Kai, 2026-11-07; pre-registered in CURRICULUM "Test Q"): Test B2's
# recipe on Q*bert, nine generations of ~33,333 env steps. Q0 (q0) is T0 on
# Q*bert's 24 duration tokens at its reward unit (25: one new cube); the
# control QT1-QT8 (q1-q8) are T1's recipe; QM1-QM8 (p1-p8) are the same plus
# mining at every swap under Q-A1 and Q-A2 (`_mine_q`: the prep's --mine-q),
# QM1's master Q0 itself. Every arm holds `_ceiling_cap` 25 units (Q-A5).
# `_vocab_parent_arm`: the arm whose static vocabulary a first student's
# master played (Ms. Pac-Man's TK otherwise). The two per-trajectory knobs
# stay per GAME, the Ms. Pac-Man port's rule (Q-A7): a Q*bert game is 4 lives,
# so Pong's MAX_REPLAY 85 and POLICY_REFRESH_COUNT 4 become 340 and 16.
QBERT_DURATION_TOKENS = tuple((a, f) for a in range(6)
                              for f in (4, 8, 16, 32))
QT_ARM = "effzero-repro-faithful-prioract-noredo-temp05-bnrecal-qbert-tokens-two"
QM_ARM = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-qbert-tokens-"
          "mined")
QBERT_GENERATIONS = 9
QBERT_KNOBS = {"TOKEN_VOCAB": QBERT_DURATION_TOKENS, "REWARD_UNIT": 25.0,
               "_ceiling_cap": 25.0, "MAX_REPLAY": 340,
               "POLICY_REFRESH_COUNT": 16}
ARMS[QT_ARM + "-q0"] = {
    **ARMS[TWO_ARM + "-t0"],
    **QBERT_KNOBS,
    "_doc": "Test Q's Q0: Test B2's T0 (...-mspacman-tokens-two-t0) on "
            "Q*bert -- from scratch at the sign clip, 24 duration tokens, "
            "REWARD_UNIT 25, to its env-step stop, its own play logged for "
            "the first students' distillation sets; shared by the control "
            "and the mined lineage. Run as TK on QbertNoFrameskip-v4, at "
            "5 lives per iteration (Q-A6).",
}
for _k in range(1, QBERT_GENERATIONS):
    ARMS[f"{QT_ARM}-q{_k}"] = {
        **ARMS[TWO_ARM + "-t1"],
        **QBERT_KNOBS,
        "_vocab_parent_arm": QT_ARM + "-q0",
        "_master_arm": f"{QT_ARM}-q{_k - 1}",
        "_doc": f"Test Q's QT{_k}: T1's recipe on Q*bert, the distilled "
                f"student of {'Q0' if _k == 1 else f'QT{_k - 1}'}'s final "
                f"snapshot, no mining; its ceiling held at 25 units (Q-A5). "
                f"Run as TK on QbertNoFrameskip-v4, at 5 lives per iteration "
                f"(Q-A6).",
    }
    ARMS[f"{QM_ARM}-p{_k}"] = {
        **ARMS[TWO_ARM + "-t1"],
        **QBERT_KNOBS,
        "_vocab_parent_arm": QT_ARM + "-q0",
        "_master_arm": (QT_ARM + "-q0" if _k == 1 else f"{QM_ARM}-p{_k - 1}"),
        "_mine": True,
        "_mine_q": True,
        "_doc": f"Test Q's QM{_k}: QT{_k} plus mining at the swap (distill.py "
                f"--mine --mine-q), the distilled student of "
                f"{'Q0' if _k == 1 else f'QM{_k - 1}'}'s final snapshot, "
                f"holds out of the mined slots (Q-A1), played paths certain "
                f"in the targets (Q-A2). Run as TK on QbertNoFrameskip-v4, at "
                f"5 lives per iteration (Q-A6).",
    }
del _k

# Test B2W (Kai, 2026-11-04; pre-registered in CURRICULUM "Test B2W"): Test
# B2's lineage with one change, the optimizer -- BBF's AdamW (configs/BBF.gin,
# agents/spr_agent.py create_scaling_optimizer: lr 0.0001, eps 0.00015,
# decoupled weight decay 0.1, rank-1 parameters undecayed, no warmup) in
# place of EfficientZero's SGD (lr 0.2, momentum 0.9, coupled decay 1e-4,
# 1,000-step warmup). W0-W2 are T0-T2 otherwise, the students' offline steps
# included; the chain supervisor's lineage "w".
ADAM_ARM = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
            "tokens-adam")
BBF_ADAMW = {"OPTIMIZER": "adamw", "LEARNING_RATE": 1e-4,
             "WEIGHT_DECAY": 0.1, "ADAM_EPS": 0.00015, "DECAY_RANK1": False,
             "LR_WARMUP_STEPS": 0}
for _k in range(3):
    ARMS[f"{ADAM_ARM}-w{_k}"] = {
        **ARMS[f"{TWO_ARM}-t{_k}"],
        **BBF_ADAMW,
        **({"_master_arm": f"{ADAM_ARM}-w{_k - 1}"} if _k else {}),
        "_doc": f"{TWO_ARM}-t{_k} (Test B2's T{_k}) as Test B2W's "
                f"generation W{_k}, with BBF's AdamW in place of "
                f"EfficientZero's SGD: lr 1e-4, eps 1.5e-4, decoupled weight "
                f"decay 0.1 on every parameter but the rank-1 ones (biases, "
                f"BN affines), no warmup"
                + (f"; the distilled student of W{_k - 1}'s final snapshot "
                   f"through Test B2's swap, its offline steps under the "
                   f"same optimizer" if _k else "; from scratch")
                + ". Run as TK.",
    }
del _k

# Test C (Kai, 2026-10-29; pre-registered in CURRICULUM "Test C: BBF-style
# resets in one 100k run, against a no-reset twin"): T0's recipe run from
# scratch to 100,000 env steps (planned length 150 iterations: Test B's 30
# per 20k), without resets (C0, "-long-n0") and with BBF's periodic resets
# (C, "-reset-r0"); both log the plasticity diagnostics
# (DORMANT_DIAGNOSTIC). No swap follows either, so the sign clip and the 36
# tokens hold throughout. The variant suffixes are the chain supervisor's
# (supervise_testb.py lineages "n" and "r").
LONG_ARM = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
            "tokens-long")
RESET_ARM = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
             "tokens-reset")
ARMS[LONG_ARM + "-n0"] = {
    **ARMS[TWO_ARM + "-t0"],
    "_nominal_iterations": 150,
    "DORMANT_DIAGNOSTIC": True,
    "_doc": "effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman"
            "-tokens-two-t0 as Test C's control C0: one run from scratch at "
            "the sign clip, PREFIX_BN_PER_STEP, DISCOUNT 0.997^4 per base "
            "step, to its env-step stop (--stop-env-steps 100000), the "
            "within-run schedules following its env-step progress (planned "
            "length 150 iterations); no reset, no swap; the dormant-unit and "
            "first-layer filter diagnostics in every row. Run as TK.",
}
ARMS[RESET_ARM + "-r0"] = {
    **ARMS[LONG_ARM + "-n0"],
    "RESET_EVERY_STEPS": 40000,
    "RESET_UNTIL_ENV_STEPS": 80000,
    "RESET_SHRINK_NETS": ("representation", "dynamics"),
    "RESET_REINIT_NETS": ("prediction", "value_prefix", "projector"),
    "RESET_SHRINK_FACTOR": 0.5,
    "RESET_PERTURB_FACTOR": 0.5,
    "RESET_HOLD_STEPS": 10000,
    "_doc": "Test C's C: the control C0 plus BBF's periodic resets -- after "
            "every 40,000 gradient steps (20 iterations) while the run has "
            "played at most 80,000 env steps, the representation and "
            "dynamics move halfway to a fresh init (shrink 0.5, perturb "
            "0.5), the prediction, value prefix and projector are "
            "re-initialized, the target model is reset with them, their SGD "
            "momentum is zeroed and the lr warmup restarts; for the next "
            "10,000 gradient steps the actor keeps the pre-reset weights. "
            "Run as TK.",
}

# Test C2 (Kai, 2026-11-02; pre-registered in CURRICULUM "Test C2: BBF's
# n-step and discount ramps on top of the resets"): C ("-reset-r0", every
# reset knob unchanged) plus BBF's two ramps, restarted at every reset and
# keyed on the gradient steps since it (since the start before the first),
# over 10,000 of them (5 iterations of 2,000, the hold's window): n from 10
# to C's TD_STEPS 5, gamma from 0.97 to C's DISCOUNT 0.997^4 per base step,
# BBF's shapes. The end values are C's (only the start values are BBF's).
# The control is C itself, the run of 10-30 to 11-01. The variant suffix is
# the chain supervisor's (lineage "a").
RAMP_ARM = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
            "tokens-ramp")
ARMS[RAMP_ARM + "-a0"] = {
    **ARMS[RESET_ARM + "-r0"],
    "RAMP_STEPS": 10000,
    "RAMP_N_START": 10,
    "RAMP_GAMMA_START": 0.97,
    "RAMP_N_END": 5,
    "RAMP_GAMMA_END": 0.997 ** 4,
    "_doc": "Test C2: Test C's C (...-tokens-reset-r0, every reset knob "
            "unchanged) plus BBF's n-step and discount ramps, keyed on the "
            "gradient steps since the last reset (since the start before the "
            "first), over 10,000 of them (5 iterations): n = round(10 x "
            "0.5^f) (half to even), 10 -> 5 base steps, and 1 - gamma = "
            "0.03^(1-f) x (1 - 0.997^4)^f, 0.97 -> 0.98805 per base step, "
            "f = min(steps since the reset / 10,000, 1); the end values, C's "
            "TD_STEPS and DISCOUNT, hold to the next reset and after the "
            "last. Every reader of the horizon and the discount takes the "
            "current values (the batch-time value targets with "
            "EfficientZero's age shrink on top, the birth search, the "
            "refresh, the PER priorities, the row diagnostics); the actors "
            "read neither. Run as TK.",
}

# Test O (Kai, 2026-11-07; pre-registered in CURRICULUM "Test O: OptionZero's
# option head in place of the token vocabulary"): O0, the control for the
# option edge -- C0's recipe with no vocabulary, the 9 moves at one base step,
# the plain search, from scratch to 100,000 env steps (planned length 150
# iterations, C0's; the schedules follow env-step progress). No next
# generation reads its play, so its own-play log is off; the long-token bias
# is off too, inert without a vocabulary, as the registration leaves it for
# every arm. Its rows carry the O0 PRE-CHECK (OPTION_PRECHECK). The variant
# suffix is the chain supervisor's (lineage "o").
PLAIN_ARM = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
             "plain")
ARMS[PLAIN_ARM + "-o0"] = {
    **ARMS[LONG_ARM + "-n0"],
    "TOKEN_VOCAB": None,
    "TOKEN_LONG_BIAS": 0.0,
    "OWN_PLAY_LOG": False,
    "OPTION_PRECHECK": True,
    # Compute knobs only: TK's 64 / 1 was a 36-slot memory workaround; at 9
    # moves the base's shapes fit, as RG ran them, and 64 / 1 would about
    # double the iteration. Float numerics only (CURRICULUM TK's note).
    "VTREE_REFRESH_BATCH": 256,
    "VTREE_COMMUNAL_TRAJS": 8,
    "_doc": "Test O's O0: Test C's control C0 (...-tokens-long-n0) with no "
            "vocabulary -- Ms. Pac-Man's 9 moves at one base step (4 frames), "
            "the plain search, no option edge -- from scratch at the sign "
            "clip, PREFIX_BN_PER_STEP, DISCOUNT 0.997^4 per base step, to its "
            "env-step stop (--stop-env-steps 100000), the within-run "
            "schedules following its env-step progress (planned length 150 "
            "iterations); no reset, no swap, no own-play log; the dormant-"
            "unit and first-layer filter diagnostics in every row, and the "
            "O0 pre-check's shares (OPTION_PRECHECK). Run as TK.",
}
# Test O stage 4a (CURRICULUM "Test O", ARMS): OF, O0 plus OptionZero's option
# head at L = 3 base steps, first move only -- play unchanged (the prior's
# first-move sample, a fresh decision every base step); the option edge in
# the birth search; the heads and the jump rows train the shared net
# (DEPARTURE 1). OH, OF with the option edge OFF in the birth search: the same
# heads, the same suggestions and the same jump training. Everything else is
# O0's, OPTION_PRECHECK included: it reads plain play, which OF and OH keep,
# so the three arms' pre-check shares compare. The variant suffixes follow
# O0's (lineage "o").
ARMS[PLAIN_ARM + "-of"] = {
    **ARMS[PLAIN_ARM + "-o0"],
    "OPTION_LENGTH": 3,
    "_doc": "Test O's OF: Test O's O0 (...-mspacman-plain-o0) plus "
            "OptionZero's option head at OPTION_LENGTH 3 (12 frames), first "
            "move only: play unchanged (the prior's first-move sample, a "
            "fresh decision every base step); the option edge in the birth "
            "search (OPTION_EDGE True, muzero_vtree_options); the records "
            "carry the root's suggestion and PMAX; each unroll edge jumps "
            "the option's moves where the play matched it; the heads train "
            "at scale 1 in the shared loss; the option diagnostics and the "
            "on/off root probe in every row beside O0's pre-check shares. "
            "Run as TK.",
}
ARMS[PLAIN_ARM + "-oh"] = {
    **ARMS[PLAIN_ARM + "-of"],
    "OPTION_EDGE": False,
    "_doc": "Test O's OH: Test O's OF (...-mspacman-plain-of) with the "
            "option edge OFF in the birth search (OPTION_EDGE False): the "
            "same heads, the same per-node and root suggestions, the same "
            "records, jump training and option loss; no jump row and no "
            "option edge in the tree. OF minus OH is the edge alone. Run as "
            "TK.",
}

# PB (Kai, 2026-10-24; pre-registered in CURRICULUM.md "PB: per-step prefix
# BatchNorm statistics on the token arm", after Step 0 (iii)): TK with the
# value-prefix head's BatchNorms keeping one set of running statistics per
# LSTM step (PREFIX_BN_PER_STEP) -- the one change. Seed 0, 90 iterations,
# the TK recipe; reference TK-90 itself (2026-10-19).
ARMS["effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-tokens-rgrow-pbn"] = {
    **ARMS["effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-tokens-rgrow"],
    "PREFIX_BN_PER_STEP": True,
    "_doc": "effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman"
            "-tokens-rgrow with PREFIX_BN_PER_STEP: every BatchNorm in the "
            "value-prefix head keeps a separate set of running statistics "
            "for each LSTM step since the last reset (1..PREFIX_LSTM_HORIZON"
            "), each training step's batch statistics updating its own set "
            "(training otherwise unchanged), and every eval-mode call -- the "
            "search engines, birth search, refresh, the diagnostics -- "
            "normalizing each row with its own step's set (1 after a fresh "
            "or reset state). Everything else TK's. Run as TK. Reference: "
            "TK-90 (mspacman-tokens-rgrow-s0-2026-10-19).",
}

# The sharper prior-acting temperature (Kai, 2026-10-04; pre-registered in
# CURRICULUM before any seed-1 exact eval -- seed 1 was at in-run row 10):
# prioract-noredo with the staged prior-acting schedule halved. Runs only if
# the trigger fires and seed 1 does not read WORSE (supervise_tempscale.py).
ARMS["effzero-repro-faithful-prioract-noredo-temp05"] = {
    **ARMS["effzero-repro-faithful-prioract-noredo"],
    "PRIOR_ACT_TEMP_SCALE": 0.5,
    "_doc": "effzero-repro-faithful-prioract-noredo with PRIOR_ACT_TEMP_SCALE "
            "0.5: prior sampled at 0.5 / 0.25 / 0.125 (the staged schedule "
            "halved, same 50% / 75% breakpoints). Everything else identical "
            "(--ray --cpu-actor, eager collect, no ReDo). Twin: "
            "pong-prioract2-s0/s1 (same arm at scale 1.0); reads (1)-(3) "
            "use the faithful control's bands (pong-faithful-s0/s1-"
            "2026-09-24).",
}

ARMS["effzero-repro-faithful-prioract-sve"] = {
    **ARMS["effzero-repro-faithful-prioract"],
    "VALUE_TARGET_MODE": "mixed",
    "SVE_START_TRAIN_STEPS": 40000,
    "SVE_FRESH_TRANSITIONS": 20000,
    "SVE_SOURCE": "search",
    "SVE_SEARCH_SIMS": 16,
    "_doc": "effzero-repro-faithful-prioract with EfficientZero V2's "
            "search-based value estimation: from 40k training steps on, "
            "window roots outside the newest 20k transitions take a fresh "
            "16-sim search through the target nets as their value target "
            "(T1 / T2 = the paper's values, Kai 2026-09-28; batch-time "
            "fresh search, Kai 2026-09-29). Run with --ray --cpu-actor. "
            "Control: pong-prioract-s0-2026-09-27.",
}

# Staged-attribution arm (2026-09-02, Kai): saturation coverage with
# archfix's exact refresh SELECTION semantics -- staleness rotation,
# always-replace, no guard. Deltas vs archfix: parallel stack, fresh-
# root topology, coverage volume. Entropy+guard test layers on after.
ARMS["effzero-repro-vtree-sat-stale"] = {
    **ARMS["effzero-repro-vtree-refresh"],
    "REFRESH_SATURATE": True,
    "REFRESH_ENTROPY_PRIORITY": False,
    "REFRESH_NO_SHARPEN_GUARD": False,
    # The saturation FLOOR: base-schedule coverage (archfix's 4 per 2
    # iters) guaranteed even when training outlasts the episode and no
    # slack exists; saturation is strictly additive on top.
    "POLICY_REFRESH_COUNT": 4,
    "_doc": "Saturation refresh with STALENESS ordering and NO guard: "
            "while the CPU env actor plays, the GPU re-searches oldest-"
            "rotation trajectories one at a time (one pass over the "
            "buffer max per iteration; every position, always replaced) "
            "-- archfix's selection semantics at saturation volume. Run "
            "with --ray --cpu-actor. Control: pong-archfix-2026-08-30 "
            "(deltas: pipeline+cpu actor, fresh-root topology, coverage "
            "volume). The entropy+guard package tests separately in "
            "effzero-repro-vtree-saturate.",
}


# Phase 5 arm (2026-09-06, Kai): the fully asynchronous loop with the
# freshness contract -- every trained-on target refreshed within one
# sweep period, newcomers refreshed at collect. Refresh SEMANTICS are
# sat-stale's (rotation, always-replace, no guard: the only variant
# that accelerated, 2/2 seeds); what changes is the DRIVER (chunked
# async loop, continuous acting, train-ratio cap) and the coverage
# guarantee (contract, not slack).
ARMS["effzero-repro-async-fresh"] = {
    **ARMS["effzero-repro-vtree-sat-stale"],
    "REFRESH_SATURATE": False,
    "ASYNC_LOOP": True,
    "_doc": "Phase-5 chunked asynchronous driver: actor plays "
            "continuously (new episode requested at every collect), "
            "gradient steps run in ASYNC_TRAIN_CHUNK chunks capped at "
            "MAX_TRAIN_PER_ENV_STEP x env_steps, and the refresh lane "
            "keeps a standing contract -- full-buffer rotation sweep "
            "at least once per ASYNC_SWEEP_TRAIN_STEPS gradient steps "
            "(always-replace, no guard) plus newcomer refresh at "
            "collect. Rows re-key to gradient steps (u = "
            "training_steps/train-steps) so the family's per-row "
            "comparison axis survives; env_steps and the realized "
            "train ratio are logged per row. Run with --ray "
            "--cpu-actor. Controls: pong-satstale-2026-09-02 and "
            "pong-satstale-s1-2026-09-04 (same refresh semantics, "
            "iteration-bound driver).",
}

# Test 6 arm (2026-09-07, from the pong-async-s1 u-25 verdict): the
# async driver lost the satstale acceleration while running at
# train/env ~1.06-1.31 vs legacy's ~2.0 -- the continuous actor
# doubled buffer inflow and halved per-position reuse. One variable
# vs async-fresh: the admission floor. The actor is never blocked or
# slowed; excess episodes count env_steps but never enter the buffer
# (dropped BEFORE eviction can churn stored data). Newcomer-refresh
# timing deliberately unchanged -- if the throttle fails to restore
# the corridor, ASYNC_REFRESH_ON_COLLECT=False is the isolated test 7.
ARMS["effzero-repro-async-throttle"] = {
    **ARMS["effzero-repro-async-fresh"],
    "ASYNC_MIN_TRAIN_PER_ENV_STEP": 2.0,
    "_doc": "effzero-repro-async-fresh plus admission control: a "
            "collected episode enters the buffer only if the ADMITTED "
            "train/env ratio stays >= 2.0 (legacy's implicit data "
            "diet, made explicit); dropped episodes still count "
            "env_steps and still drive the sync/request cycle. Rows "
            "add episodes_dropped and train_per_admitted_env. Run "
            "with --ray --cpu-actor. Controls: pong-async-s1-"
            "2026-09-06 (no floor) and pong-satstale-s1-2026-09-04 "
            "(legacy driver, same seed and refresh semantics).",
}

# Test 7 arm (2026-09-07, from the throttle verdict): the admission
# floor did NOT restore the corridor (-16.0 @ u 25 vs the pair's
# -12.9/-9.9), so reuse alone is not the lever. One variable vs the
# throttle arm: newcomers KEEP their actor-generated on-policy search
# targets at collect (as in every accelerating legacy run) instead of
# being immediately overwritten by the lagged target model's re-search;
# the rotation sweep still reaches them within one sweep period. The
# premature-sharpness fingerprint (policy sharper than family norm at
# matched returns, present in every flop and both async runs from u 5)
# is the motivating observation.
ARMS["effzero-repro-async-keepnew"] = {
    **ARMS["effzero-repro-async-throttle"],
    "ASYNC_REFRESH_ON_COLLECT": False,
    "_doc": "effzero-repro-async-throttle with newcomer refresh OFF: "
            "collected episodes enter the buffer with the actor's own "
            "search targets and are first re-searched only when the "
            "freshness sweep rotates to them. Deltas vs the legacy "
            "accelerators shrink to the driver + the full-contract "
            "sweep cadence. Run with --ray --cpu-actor. Controls: "
            "pong-throttle-s1-2026-09-07 (newcomer refresh on) and "
            "pong-satstale-s1-2026-09-04 (legacy driver).",
}

# Attribution arm (2026-09-05, Kai): the replicating winner's allocation
# (staleness rotation, always-replace) with ONLY guard v2 added -- one
# variable vs sat-stale. Motivated by the buffer-scan correction: late-
# stage coverage is whole-buffer in every saturate arm, but during iters
# ~5-15 coverage is partial and ordering allocates differently; sat-stale
# accelerated 2/2 seeds while the entropy-ordered saturate arm tracked
# archfix. If THIS arm accelerates, entropy ordering caused test 3's
# flop and the guard is compatible with the speedup (best case: speed +
# collapse protection). If it flops, the guard is implicated.
ARMS["effzero-repro-vtree-sat-guard"] = {
    **ARMS["effzero-repro-vtree-sat-stale"],
    "REFRESH_NO_SHARPEN_GUARD": True,
    "_doc": "effzero-repro-vtree-sat-stale plus guard v2 retirement "
            "(stable-sharp only: H <= REFRESH_GUARD_SHARP_H and "
            "KL <= REFRESH_GUARD_KL_TOL, REFRESH_GUARD_STRIKES strikes "
            "in different calls). Rotation may pick a fully-guarded "
            "trajectory: it costs a stamp, not a search (the audited "
            "one-pass seen-set bound). Controls: pong-satstale-2026-09-02 "
            "and pong-satstale-s1-2026-09-04 (same allocation, no guard) "
            "vs pong-vsat2-2026-09-03 (guard, entropy ordering). Run "
            "with --ray --cpu-actor.",
}

# Saturation arm (2026-09-02, Kai): drop the fixed refresh quota; fill
# the actor-play slack with as much entropy-prioritized refresh as fits.
ARMS["effzero-repro-vtree-saturate"] = {
    **ARMS["effzero-repro-vtree-refresh"],
    "REFRESH_SATURATE": True,
    "_doc": "effzero-repro-vtree-refresh with SATURATION refresh: no "
            "fixed quota -- while the CPU environment actor plays each "
            "episode, the GPU refreshes one flattest candidate "
            "trajectory at a time (one pass over the buffer max per "
            "iteration; a refreshed trajectory re-enters next call, so "
            "guard strikes land across training windows), "
            "stopping when the episode arrives; floor of one candidate "
            "per old cadence point. The no-sharpen guard is mandatory "
            "(self-limiting). POLICY_REFRESH_COUNT is inert here. Run "
            "with --ray --cpu-actor. Control: pong-vtree-2026-09-02 "
            "(same command, quota arm) and pong-archfix-2026-08-30.",
}

# Phase 3a arm, DERIVED (not transcribed) from the communal arm so the base
# recipe can never drift from its control. Same everything, plus the
# entropy-refresh triangle: 5x coverage, flattest-first candidates, the
# no-sharpen guard, and --ray refresh offload to make the volume free.
ARMS["effzero-repro-communal-entropyrefresh"] = {
    **ARMS["effzero-repro-refresh5x-communal"],
    "POLICY_REFRESH_COUNT": 20,
    "REFRESH_ENTROPY_PRIORITY": True,
    "REFRESH_NO_SHARPEN_GUARD": True,
    "REFRESH_OFFLOAD": True,
    "_doc": "effzero-repro-refresh5x-communal plus the phase-3a "
            "entropy-refresh package (2026-08-31): 5x refresh coverage "
            "(COUNT 4->20; at the 31.25k transition cap that re-searches "
            "roughly the whole buffer every 2 iterations), candidates "
            "ordered flattest stored-target entropy first, the no-sharpen "
            "guard (v2: a step retires once its re-search reproduces a "
            "stable sharp answer; flat targets stay eligible), and refresh "
            "offloaded to a dedicated Ray actor under --ray (without "
            "--ray the same coverage runs inline and costs iteration "
            "time). Control: the communal arm at the same command; only "
            "--arms differs.",
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


@contextlib.contextmanager
def token_actions():
    """Under TOKEN_VOCAB (curriculum P2a) the action space IS the vocabulary:
    ACTIONS = len(TOKEN_VOCAB) for the run -- nets, searches, the Ray actors
    -- restored after, so a sweep's next arm sees the env's count again.
    Entered inside arm_config (it reads the arm's TOKEN_VOCAB)."""
    global ACTIONS
    saved = ACTIONS
    if muzero.TOKEN_VOCAB is not None:
        ACTIONS = len(muzero.TOKEN_VOCAB)
    try:
        yield
    finally:
        ACTIONS = saved


@contextlib.contextmanager
def env_schedule(stop_env_steps):
    """Test B: muzero.SCHEDULE_ENV_TARGET = the run's env-step stop (the
    env-keyed schedules' target) and SCHEDULE_ENV_STEPS = 0, restored after;
    no stop leaves both untouched (every other run, bit for bit)."""
    if not stop_env_steps:
        yield
        return
    saved = (muzero.SCHEDULE_ENV_TARGET, muzero.SCHEDULE_ENV_STEPS)
    muzero.SCHEDULE_ENV_TARGET = int(stop_env_steps)
    muzero.SCHEDULE_ENV_STEPS = 0
    try:
        yield
    finally:
        muzero.SCHEDULE_ENV_TARGET, muzero.SCHEDULE_ENV_STEPS = saved


@contextlib.contextmanager
def warmup_origin():
    """Test C: a run's resets move muzero.LR_WARMUP_ORIGIN (bbf_reset, a
    resume's load_run_checkpoint); it starts at 0 and is restored after, so
    a sweep's next arm warms up from its own start."""
    saved = muzero.LR_WARMUP_ORIGIN
    muzero.LR_WARMUP_ORIGIN = 0
    try:
        yield
    finally:
        muzero.LR_WARMUP_ORIGIN = saved


@contextlib.contextmanager
def ramp_globals():
    """Test C2: a ramp run's muzero.apply_ramps moves muzero.TD_STEPS and
    muzero.DISCOUNT; both are saved on entry (inside arm_config: the arm's
    static values) and restored after, an exception included, so no ramped
    value outlives its run (every other run: restored to what it never
    changed)."""
    saved = (muzero.TD_STEPS, muzero.DISCOUNT)
    try:
        yield
    finally:
        muzero.TD_STEPS, muzero.DISCOUNT = saved


def env_stop_gate(spec, stop_env_steps, use_ray, cpu_actor, iterations,
                  snapshot_weights):
    """Test B's arms, checked at startup (before any --overwrite deletion):
    an `_env_stop` arm runs until --stop-env-steps of its own env steps (a
    positive count), its planned length `_nominal_iterations` within the
    --iterations cap, on the pipelined --ray --cpu-actor loop (its
    own-play log comes from the actor), with --snapshot-weights (its final
    snapshot, acting states included, is the next generation's master;
    audit r1 A-L5); every other arm refuses --stop-env-steps; OWN_PLAY_LOG
    needs the actor."""
    if spec.get("_env_stop"):
        if (isinstance(stop_env_steps, bool)
                or not isinstance(stop_env_steps, int)
                or stop_env_steps <= 0):
            raise SystemExit("an env-stopped generation (Test B) needs "
                             "--stop-env-steps, a positive env-step count")
        if not snapshot_weights:
            raise SystemExit("an env-stopped generation (Test B) needs "
                             "--snapshot-weights (its final snapshot is the "
                             "next generation's master)")
        nominal = spec.get("_nominal_iterations")
        if not (isinstance(nominal, int) and 0 < nominal <= iterations):
            raise SystemExit(f"_nominal_iterations {nominal!r} must lie "
                             f"within the --iterations cap {iterations}")
        if muzero.ASYNC_LOOP:
            raise SystemExit("an env-stopped generation runs the pipelined "
                             "iteration loop, not ASYNC_LOOP")
    elif stop_env_steps:
        raise SystemExit("--stop-env-steps applies to Test B's env-stopped "
                         "arms only")
    if muzero.OWN_PLAY_LOG and not (use_ray and cpu_actor):
        raise SystemExit("OWN_PLAY_LOG logs the --ray --cpu-actor actor's "
                         "play")


def reset_gate(spec, use_ray, cpu_actor, ray_workers, train_steps,
               init_weights):
    """Test C's resets, checked at startup (before any --overwrite
    deletion): RESET_EVERY_STEPS runs on the pipelined --ray --cpu-actor
    iteration loop (the hold acts on its one actor; no worker pool, no
    ASYNC_LOOP, no offloaded or saturating refresh), from scratch (no
    student, no --init-weights), never beside ReDo, with the lagged target
    model (the reset resets it); RESET_SHRINK_NETS and RESET_REINIT_NETS
    split the five nets between them; the reset period and the hold are
    whole iterations (so a reset lands at an iteration's end and a hold
    ends at one -- at a row too under the registered recipe's
    --eval-every 5); the factors are positive. Without RESET_EVERY_STEPS,
    no other RESET_* knob may be set."""
    every = muzero.RESET_EVERY_STEPS
    if not every:
        if (muzero.RESET_UNTIL_ENV_STEPS or muzero.RESET_SHRINK_NETS
                or muzero.RESET_REINIT_NETS or muzero.RESET_HOLD_STEPS):
            raise SystemExit("the RESET_* knobs need RESET_EVERY_STEPS")
        return
    if not (use_ray and cpu_actor) or ray_workers:
        raise SystemExit("RESET_EVERY_STEPS runs on the pipelined --ray "
                         "--cpu-actor loop (the hold acts on its one actor)")
    if (muzero.ASYNC_LOOP or muzero.REFRESH_OFFLOAD
            or muzero.REFRESH_SATURATE):
        raise SystemExit("RESET_EVERY_STEPS runs the iteration loop with the "
                         "inline refresh: no ASYNC_LOOP, REFRESH_OFFLOAD or "
                         "REFRESH_SATURATE")
    if spec.get("_student") or init_weights:
        raise SystemExit("a reset run starts from scratch (no student, no "
                         "--init-weights)")
    if muzero.REDO_EVERY:
        raise SystemExit("RESET_EVERY_STEPS and ReDo are exclusive (one "
                         "plasticity mechanism at a time)")
    if not muzero.USE_TARGET_MODEL:
        raise SystemExit("RESET_EVERY_STEPS resets the lagged target model: "
                         "it needs USE_TARGET_MODEL")
    # (audit r1a LOW-2) the warmup a reset restarts sets the lr absolutely
    if muzero.LR_DECAY_AT:
        raise SystemExit("RESET_EVERY_STEPS restarts the lr warmup to "
                         "LEARNING_RATE, which would undo LR_DECAY_AT")
    if not muzero.LR_WARMUP_STEPS:
        raise SystemExit("RESET_EVERY_STEPS restarts the lr warmup: it needs "
                         "LR_WARMUP_STEPS")
    # ... and every NEW target comes from the learner's birth search: an
    # actor's own search during the hold would store the pre-reset
    # network's targets
    if not muzero.BIRTH_SEARCH or (muzero.USE_SEARCH_ACTING
                                   and not muzero.SEARCH_ACT_BIRTH_TARGETS):
        raise SystemExit("RESET_EVERY_STEPS needs BIRTH_SEARCH to make every "
                         "new target (search acting only under "
                         "SEARCH_ACT_BIRTH_TARGETS)")
    shrink = tuple(muzero.RESET_SHRINK_NETS)
    reinit = tuple(muzero.RESET_REINIT_NETS)
    if (set(shrink) & set(reinit) or len(set(shrink)) != len(shrink)
            or len(set(reinit)) != len(reinit)
            or set(shrink) | set(reinit) != set(NET_NAMES)):
        raise SystemExit(f"RESET_SHRINK_NETS {shrink} and RESET_REINIT_NETS "
                         f"{reinit} must split the nets {NET_NAMES}")
    hold = muzero.RESET_HOLD_STEPS
    if (isinstance(every, bool) or not isinstance(every, int)
            or every % train_steps or isinstance(hold, bool)
            or not isinstance(hold, int) or hold < 0 or hold % train_steps
            or hold >= every):
        raise SystemExit(f"RESET_EVERY_STEPS {every!r} and RESET_HOLD_STEPS "
                         f"{hold!r} must be whole iterations of "
                         f"--train-steps {train_steps}, the hold shorter")
    until = muzero.RESET_UNTIL_ENV_STEPS
    if isinstance(until, bool) or not isinstance(until, int) or until <= 0:
        raise SystemExit(f"RESET_UNTIL_ENV_STEPS {until!r} must be a "
                         f"positive env-step count")
    for name in ("RESET_SHRINK_FACTOR", "RESET_PERTURB_FACTOR"):
        f = getattr(muzero, name)
        if (isinstance(f, bool) or not isinstance(f, (int, float))
                or not 0 < f <= 1):
            raise SystemExit(f"{name} {f!r} must lie in (0, 1]")


# Test C2: muzero's RAMP_* defaults (RAMP_STEPS 0: off)
RAMP_OFF = {"RAMP_STEPS": 0, "RAMP_N_START": 10, "RAMP_GAMMA_START": 0.97,
            "RAMP_N_END": None, "RAMP_GAMMA_END": None}


def ramp_gate(spec):
    """Test C2's ramps, checked at startup (after reset_gate, before any
    --overwrite deletion):
      - RAMP_STEPS, a positive step count no longer than a reset period,
        runs on a reset run (RESET_EVERY_STEPS: C's shape, which reset_gate
        restricts to the pipelined --ray --cpu-actor loop from scratch);
      - every gradient step goes through train_block in run_one's iteration
        loop (after_train_step applies the ramp): no ASYNC_LOOP (its own
        loop), no student (its offline phase), no TRAIN_PREFETCH (a batch
        assembled ahead of its step would take an earlier step's values);
      - policy-only acting: no USE_SEARCH_ACTING (audit r1 LOW-1: an
        actor's own search runs in its process at the static spec
        DISCOUNT, which the ramp never reaches);
      - the ramped targets are the paper builder's TD value targets
        (PAPER_TARGETS, VALUE_TARGET_MODE "td");
      - n runs between ints >= 1, falling; gamma between floats in (0, 1),
        rising;
      - the end values ARE the arm's TD_STEPS and DISCOUNT, exactly: the
        spec carries both (so arm_config restores them too) and the module
        holds them (the searches assert their discount is DISCOUNT, and
        past the window the run is C's).
    Without RAMP_STEPS no other RAMP_* knob may leave its default."""
    every = muzero.RAMP_STEPS
    if not every:
        if any(getattr(muzero, k) != v for k, v in RAMP_OFF.items()):
            raise SystemExit("the RAMP_* knobs need RAMP_STEPS")
        return
    resets = muzero.RESET_EVERY_STEPS
    if (isinstance(every, bool) or not isinstance(every, int) or every <= 0
            or (resets and every > resets)):
        raise SystemExit(f"RAMP_STEPS {every!r} must be a positive "
                         f"gradient-step count, at most RESET_EVERY_STEPS")
    if not resets:
        raise SystemExit("RAMP_STEPS restarts at every reset: it needs "
                         "RESET_EVERY_STEPS (Test C's run shape)")
    if muzero.ASYNC_LOOP or spec.get("_student"):
        raise SystemExit("RAMP_STEPS needs every gradient step in run_one's "
                         "iteration loop: no ASYNC_LOOP, no student's "
                         "offline phase")
    if muzero.TRAIN_PREFETCH:
        raise SystemExit("RAMP_STEPS moves n and gamma every step: no "
                         "TRAIN_PREFETCH (it assembles batches ahead of "
                         "their step)")
    if muzero.USE_SEARCH_ACTING:
        raise SystemExit("RAMP_STEPS needs policy-only acting: an actor's "
                         "own search (USE_SEARCH_ACTING) runs at its static "
                         "spec DISCOUNT, which the ramp never reaches")
    if not muzero.PAPER_TARGETS or muzero.VALUE_TARGET_MODE != "td":
        raise SystemExit("RAMP_STEPS ramps the paper builder's TD value "
                         "targets: it needs PAPER_TARGETS and "
                         "VALUE_TARGET_MODE 'td'")
    n0, n1 = muzero.RAMP_N_START, muzero.RAMP_N_END
    for name, n in (("RAMP_N_START", n0), ("RAMP_N_END", n1)):
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise SystemExit(f"{name} {n!r} must be an int >= 1")
    g0, g1 = muzero.RAMP_GAMMA_START, muzero.RAMP_GAMMA_END
    for name, g in (("RAMP_GAMMA_START", g0), ("RAMP_GAMMA_END", g1)):
        if not isinstance(g, float) or not 0 < g < 1:
            raise SystemExit(f"{name} {g!r} must be a float in (0, 1)")
    if n0 < n1 or g0 > g1:
        raise SystemExit(f"the ramps run from the long horizon at the low "
                         f"discount to the end values: RAMP_N_START {n0} >= "
                         f"RAMP_N_END {n1}, RAMP_GAMMA_START {g0} <= "
                         f"RAMP_GAMMA_END {g1}")
    if ("TD_STEPS" not in spec or "DISCOUNT" not in spec
            or n1 != spec["TD_STEPS"] or g1 != spec["DISCOUNT"]
            or n1 != muzero.TD_STEPS or g1 != muzero.DISCOUNT):
        raise SystemExit(f"the ramps end on the arm's own TD_STEPS and "
                         f"DISCOUNT: RAMP_N_END {n1!r} / RAMP_GAMMA_END "
                         f"{g1!r} against {spec.get('TD_STEPS')!r} / "
                         f"{spec.get('DISCOUNT')!r} (the module: "
                         f"{muzero.TD_STEPS!r} / {muzero.DISCOUNT!r})")


def own_play_dir(arm, seed):
    """Test B: the run's OwnPlayLog folder, beside its other files."""
    return RESULTS / f"own-play{TAG}-{arm}-seed{seed}"


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

    # Architecture geometry comes from muzero's call-time factories, so an
    # arm's knob overrides (the smallnet arm's 64 channels included) always
    # apply; only the support/transform choices this harness owns are
    # passed explicitly. LATENT_CHANNELS is deliberately NOT from-imported
    # anywhere in this file: the import-time binding silently ignored
    # overrides (audit 2026-08-28).
    return dict(
        representation=muzero.build_representation().to(DEVICE),
        dynamics=muzero.build_dynamics(ACTIONS, support_size=reward_sup,
                                       num_bins=reward_bins,
                                       transform=transform).to(DEVICE),
        prediction=muzero.build_prediction(ACTIONS, support_size=value_sup,
                                           num_bins=value_bins,
                                           transform=transform).to(DEVICE),
        value_prefix=muzero.build_value_prefix(support_size=prefix_sup,
                                               num_bins=prefix_bins,
                                               transform=transform).to(DEVICE),
        projector=muzero.build_projector().to(DEVICE),
    )


@contextlib.contextmanager
def train_batches(run, n_steps):
    """Yield a next_batch() for a contiguous block of n_steps gradient
    steps: a muzero.BatchPrefetcher when TRAIN_PREFETCH is on (CPU batch
    assembly overlapped with GPU training), else the serial builder. The
    caller must not mutate run.replay inside the block."""
    # The buffer may have changed since the last block (collect/evict/
    # refresh): drop the bootstrap value cache so no id()-keyed entry can
    # outlive its trajectory.
    if hasattr(run.trainer, "invalidate_bootstrap_cache"):
        run.trainer.invalidate_bootstrap_cache()
    if (muzero.TRAIN_PREFETCH and n_steps > 0
            and hasattr(run.trainer, "assemble_batch")):
        with muzero.BatchPrefetcher(run.trainer, run.replay,
                                    muzero.TRAINING_BATCHES,
                                    muzero.UNROLL_STEPS, ACTIONS,
                                    n_steps) as pf:
            yield pf.next
    else:
        def serial():
            return asyncio.run(run.trainer.prepare_batch(
                run.replay, muzero.TRAINING_BATCHES, muzero.UNROLL_STEPS,
                ACTIONS))
        yield serial


def train_block(run, n_steps, pipeline=None, init_name=None,
                consistency_base=None, it=None):
    """One iteration's gradient steps. Default: a single contiguous
    train_batches block (the unchanged loop). RAY_EAGER_COLLECT with an
    episode in flight: poll the actor every EAGER_POLL_STEPS steps; when
    the episode is done, close the block, collect it into the buffer (birth
    search on the request-time weights, append, evict) and continue the
    remaining steps in a fresh block -- train_batches reopens, invalidating
    the id()-keyed caches, so the mid-iteration append never meets a stale
    cache or an open prefetcher. A game that finished during the previous
    boundary's work (refresh / eval / checkpoint) is collected before the
    first step. Each eager iteration records [it, step offset at which its
    game joined] (offset None: the iteration-end collect took it) in
    run._eager_window, emitted per row as eager_collect_steps."""
    eager = (pipeline is not None and muzero.RAY_EAGER_COLLECT
             and pipeline.pending is not None)
    if eager:
        window = getattr(run, "_eager_window", None)
        if window is None:
            window = run._eager_window = []
        window.append([it, None])
        if pipeline.acting_ready():
            pipeline.collect()
            window[-1][1] = 0
            print(f"  [eager collect] episode joined the buffer 0 of "
                  f"{n_steps} steps into the iteration", flush=True)
            eager = False
    remaining = n_steps
    while True:
        done, ready = 0, False
        with train_batches(run, remaining) as next_batch:
            for _ in range(remaining):
                if init_name:
                    apply_warmstart_lr(
                        (run.trainer.rep_optimizer, run.trainer.dyn_optimizer,
                         run.trainer.pred_optimizer,
                         run.trainer.prefix_optimizer,
                         run.trainer.proj_optimizer),
                        run.trainer.training_steps)
                    apply_warmstart_consistency(
                        consistency_base, run.trainer.training_steps)
                # Test C2: the ramp's values at this step, before its batch
                # is built (and the iteration's first step logged)
                ramp_step(run, it, first=(done == 0 and remaining == n_steps))
                asyncio.run(run.trainer.train_step(next_batch()))
                run.trainer.training_steps += 1
                after_train_step(run)
                done += 1
                if (eager and done < remaining
                        and done % muzero.EAGER_POLL_STEPS == 0
                        and pipeline.acting_ready()):
                    ready = True
                    break
        remaining -= done
        if not ready:
            return
        pipeline.collect()
        run._eager_window[-1][1] = n_steps - remaining
        print(f"  [eager collect] episode joined the buffer {n_steps - remaining}"
              f" of {n_steps} steps into the iteration", flush=True)
        eager = False


def eager_collect_gate(use_ray):
    """RAY_EAGER_COLLECT's supported regime, checked at startup: the
    pipelined iteration loop only (not the async driver), no saturation
    refresh (it fills the gap before collect), no batch prefetcher (an
    early block close would discard prefetched, RNG-drawn batches)."""
    if not muzero.RAY_EAGER_COLLECT:
        return
    if not use_ray:
        raise SystemExit("RAY_EAGER_COLLECT needs --ray (there is no "
                         "in-flight episode in the sequential harness)")
    if muzero.ASYNC_LOOP or muzero.REFRESH_SATURATE or muzero.TRAIN_PREFETCH:
        raise SystemExit("RAY_EAGER_COLLECT supports the pipelined iteration "
                         "loop only: not ASYNC_LOOP, REFRESH_SATURATE or "
                         "TRAIN_PREFETCH")
    poll = muzero.EAGER_POLL_STEPS
    if not (isinstance(poll, int) and not isinstance(poll, bool) and poll >= 1):
        raise SystemExit(f"EAGER_POLL_STEPS must be an int >= 1, got {poll!r}")


def bn_calibration_frames(replay, n):
    """The BN stabilizer's calibration set: up to `n` clean stacked
    observations from the NEWEST trajectories in `replay` (walking back from
    the end), thinned by a deterministic stride -- no RNG is drawn, so the
    stabilizer never reorders any training stream. A LIST of arrays (no
    full-pool copy; bn_recalibrated_states stacks per batch). None if the
    buffer is empty.

    Under TOKEN_VOCAB the window counts BASE steps (replay_units), like the
    replay cap: the newest trajectories totalling >= n base steps, and up
    to n of their decision states (strided down as above only when they
    number more than n) -- the twin's recency in game time rather than n
    decisions spanning ~4x the game time (audit r2 B-M1)."""
    pool, units = [], 0
    tokens = muzero.TOKEN_VOCAB is not None
    for traj in reversed(replay):
        pool.extend(rec["obs"] if isinstance(rec, dict) else rec[0]
                    for rec in traj)
        if tokens:
            units += replay_units(traj)
        if (units if tokens else len(pool)) >= n:
            break
    if not pool:
        return None
    if len(pool) > n:
        stride = len(pool) / n
        pool = [pool[int(i * stride)] for i in range(n)]
    return pool


@torch.no_grad()
def bn_recalibrated_states(trainer, frames, batch=256):
    """CPU state dicts of COPIES of the trainer's representation and
    prediction nets whose BN running statistics are re-estimated from
    root-only forwards over `frames` (uint8 stacks, /255 as in
    initial_inference): running stats reset, momentum=None (a cumulative
    average over the batches), BN in train mode, no grad. Batches are
    INTERLEAVED (batch b = frames[b::nb]), not contiguous: contiguous 256-
    frame batches are temporally correlated stretches of one game, whose
    within-batch variance misses the between-batch variance (bnrecal r1
    L1: var x0.96 at @90). Every weight is the trainer's; the trainer's own
    nets are untouched."""
    nets = {n: copy.deepcopy(getattr(trainer, n))
            for n in ("representation", "prediction")}
    layers = [m for net in nets.values() for m in net.modules()
              if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    for m in layers:
        m.reset_running_stats()
        m.momentum = None
    for net in nets.values():
        net.train()
    rep, pred = nets["representation"], nets["prediction"]
    dev = next(rep.parameters()).device
    nb = max(1, -(-len(frames) // batch))
    for b in range(nb):
        x = torch.tensor(np.stack(frames[b::nb]),
                         dtype=torch.float32, device=dev) / 255.0
        pred(rep(x))
    for net in nets.values():
        net.eval()
    return {n: {k: v.detach().cpu().clone() if torch.is_tensor(v) else v
                for k, v in net.state_dict().items()}
            for n, net in nets.items()}


def bn_recal_gate(use_ray):
    """BN_RECAL_ACTING's supported regime, checked at startup: the
    --ray pipeline (it rides the actor weight ships), policy-only acting
    (root-only statistics would distort an acting SEARCH, which runs the
    prediction head at depths > 0), not the async driver, and a sane
    calibration size."""
    if not muzero.BN_RECAL_ACTING:
        return
    if not use_ray:
        raise SystemExit("BN_RECAL_ACTING needs --ray (it rides the actor "
                         "weight ships)")
    if muzero.USE_SEARCH_ACTING or muzero.ASYNC_LOOP:
        raise SystemExit("BN_RECAL_ACTING supports policy-only acting on the "
                         "pipelined loop only: not USE_SEARCH_ACTING or "
                         "ASYNC_LOOP")
    n = muzero.BN_RECAL_FRAMES
    if isinstance(n, bool) or not isinstance(n, int) or n < 256:
        raise SystemExit(f"BN_RECAL_FRAMES must be an int >= 256, got {n!r}")


def search_act_birth_gate():
    """SEARCH_ACT_BIRTH_TARGETS / ACTOR_SEARCH_SIMS / ACTOR_GUMBEL_SEARCH,
    checked at startup: the mode needs search acting, birth search and the
    paper data path; ACTOR_SEARCH_SIMS must be None or an int >= 1, and
    ACTOR_GUMBEL_SEARCH a bool; both are accepted only in that mode, and
    ACTOR_GUMBEL_SEARCH not with USE_GUMBEL_SEARCH (every search would
    already be Gumbel -- and birth search refuses it)."""
    sims = muzero.ACTOR_SEARCH_SIMS
    if sims is not None and (isinstance(sims, bool) or not isinstance(sims, int)
                             or sims < 1):
        raise SystemExit(f"ACTOR_SEARCH_SIMS must be None or an int >= 1, "
                         f"got {sims!r}")
    if sims is not None and not muzero.SEARCH_ACT_BIRTH_TARGETS:
        # Without birth targets the acted steps would store targets from the
        # small search while the tail search and refresh run ACT_SEARCH_SIMS
        # -- an untested mixed-budget mode (searchact8 audit r1 LOW-3).
        raise SystemExit("ACTOR_SEARCH_SIMS is supported only with "
                         "SEARCH_ACT_BIRTH_TARGETS (search picks the action, "
                         "birth search makes every target)")
    if muzero.SEARCH_ACT_BIRTH_TARGETS and not (
            muzero.USE_SEARCH_ACTING and muzero.BIRTH_SEARCH
            and muzero.PAPER_TARGETS):
        raise SystemExit("SEARCH_ACT_BIRTH_TARGETS needs USE_SEARCH_ACTING, "
                         "BIRTH_SEARCH and PAPER_TARGETS")
    g = muzero.ACTOR_GUMBEL_SEARCH
    if not isinstance(g, bool):
        raise SystemExit(f"ACTOR_GUMBEL_SEARCH must be a bool, got {g!r}")
    if g and not muzero.SEARCH_ACT_BIRTH_TARGETS:
        raise SystemExit("ACTOR_GUMBEL_SEARCH is supported only with "
                         "SEARCH_ACT_BIRTH_TARGETS (the Gumbel search picks "
                         "the action, birth search makes every target)")
    if g and muzero.USE_GUMBEL_SEARCH:
        raise SystemExit("ACTOR_GUMBEL_SEARCH switches only the actor's "
                         "searches to Gumbel; USE_GUMBEL_SEARCH would switch "
                         "every search")
    st = muzero.ACTOR_GUMBEL_STAGED_SCALE
    if not isinstance(st, bool):
        raise SystemExit(f"ACTOR_GUMBEL_STAGED_SCALE must be a bool, got {st!r}")
    if st and not g:
        raise SystemExit("ACTOR_GUMBEL_STAGED_SCALE scales the actor's Gumbel "
                         "draws; it is inert without ACTOR_GUMBEL_SEARCH")


def prior_temp_gate():
    """PRIOR_ACT_TEMP_SCALE must be a positive real number, and it only
    acts under policy-only acting -- under search acting it would be a
    silently inert knob, so that mix is refused (checked at startup)."""
    scale = muzero.PRIOR_ACT_TEMP_SCALE
    if (isinstance(scale, bool) or not isinstance(scale, (int, float))
            or not scale > 0 or scale != scale or scale == float("inf")):
        raise SystemExit(f"PRIOR_ACT_TEMP_SCALE must be a finite number > 0, "
                         f"got {scale!r}")
    # The sampler raises the float32 prior to 1/T. Its largest entry is
    # >= 1/ACTIONS, and (1/A)**(1/T) drops below float32's smallest subnormal
    # (2**-149 = e**-103.28) for T < ln(A)/103.28 -- 0.0173 at Pong's 6
    # actions, 0.028 at 18 -- giving an all-NaN p and a crash loop (round-1
    # L3, round-2 L2). Refuse any schedule whose lowest temperature comes
    # within 20% of that.
    t_min = scale * (0.25 if muzero.PRIOR_ACT_STAGED_TEMP
                     else muzero.ACTING_TEMPERATURE)
    floor = 1.2 * math.log(max(2, ACTIONS)) / 103.28
    if t_min < floor:
        raise SystemExit(f"PRIOR_ACT_TEMP_SCALE {scale!r} takes the prior-"
                         f"acting temperature to {t_min:.4g} (< {floor:.4g} for "
                         f"{ACTIONS} actions): the float32 prior**(1/T) "
                         f"underflows")
    if scale != 1.0 and muzero.USE_SEARCH_ACTING:
        raise SystemExit("PRIOR_ACT_TEMP_SCALE scales the PRIOR-acting "
                         "temperature; it is inert under USE_SEARCH_ACTING")


def play_token(env, token, stack, inner=None):
    """One TOKEN_VOCAB decision: the token's ALE action for frames /
    TOKEN_BASE_FRAMES consecutive steps of the wrapper stack. Returns (obs,
    reward, terminated, truncated, stack, base_steps): reward is the SUM of
    the inner rewards (play_episode clips it as one step's -- or, under
    REWARD_GROWTH, records `inner` and the learner clips per base step),
    and every inner
    frame but the last is pushed into the stack here -- the last is pushed
    at the next decision, as a plain step's -- so the stack always holds the
    last four base steps, whatever tokens were played. Stops early at a
    terminal or truncation. `inner`, a list, receives each inner step's raw
    reward (REWARD_GROWTH clips per base step). A module function, not a
    Run method: the Ray actor's Run shell provides only env / searcher /
    env_steps (tests/test_ray_harness.py pins play_episode's self.*
    surface). E3 (E3-A2): a MINED token plays each base step with that
    step's own action (muzero.token_inputs; a duration token's are its
    action repeated, the same env.step calls as ever)."""
    inputs = muzero.token_inputs(muzero.TOKEN_VOCAB[int(token)])
    n = len(inputs)
    total, done = 0.0, 0
    pushed = None if INNER_FRAMES is None else []
    for i in range(n):
        obs, reward, term, trunc, _ = env.step(int(inputs[i]))
        total += float(reward)
        if inner is not None:
            inner.append(float(reward))
        done = i + 1
        if term or trunc:
            break
        if i < n - 1:
            frame = preprocess_frame(obs, IMAGE_RESHAPE)
            stack = push_frame(stack, frame)
            if pushed is not None:
                pushed.append(frame)
    if pushed is not None:
        INNER_FRAMES.append(pushed)
    return obs, total, term, trunc, stack, done


# Curriculum P3's distillation step (2026-10-21): None, or a list that
# receives, per play_token call, the list of inner base frames it pushed
# into the stack (preprocessed, in order: done - 1 of them) -- the raw-step
# resolution the replay never stores. Set only by experiments/distill.py's
# master recorder around a play_episode; None everywhere else (the Ray
# actors' processes never set it).
INNER_FRAMES = None


class OwnPlayLog:
    """Test B (Kai, 2026-10-27; CURRICULUM "Test B"): the trainer's own play
    at raw-step resolution -- the next generation's distillation set, at no
    extra env steps. One pickle file per iteration (`iter-NNN.pkl`, the
    iteration whose request the lives answered), one pickle frame per life:
    {"traj": play_episode's decision records, "inner": play_token's inner
    frames per decision, "terminated", "real_done" (the game ended with
    this life), "temperature" (the acting temperature it was sampled at),
    "steps", "env_steps" (the run's count after it), "iteration"}. A file is
    TRUNCATED at this process's first write to it: a resume replays the
    iterations after its checkpoint, so a dead timeline's lives for them
    are replaced, never mixed in (the reader takes iterations 1..the
    master's final one only)."""

    def __init__(self, folder):
        self.folder = pathlib.Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.written = set()

    def path(self, it):
        return self.folder / f"iter-{int(it):03d}.pkl"

    def write(self, it, traj, extra, terminated, steps, env_steps):
        life = {"traj": [tuple(r) for r in traj], "inner": extra["inner"],
                "terminated": bool(terminated),
                "real_done": bool(extra["real_done"]),
                "temperature": float(extra["temperature"]),
                "steps": int(steps), "env_steps": int(env_steps),
                "iteration": int(it)}
        mode = "ab" if it in self.written else "wb"
        with open(self.path(it), mode) as f:
            pickle.dump(life, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.written.add(it)


def read_own_play(folder, last_iteration):
    """The lives of an OwnPlayLog, iterations 1..last_iteration in order
    (each file's lives in their write order); a missing iteration file
    raises (the log must be whole for the generation it describes)."""
    lives = []
    for it in range(1, int(last_iteration) + 1):
        p = pathlib.Path(folder) / f"iter-{it:03d}.pkl"
        if not p.exists():
            raise SystemExit(f"own-play log {folder}: iteration {it} is "
                             f"missing")
        with open(p, "rb") as f:
            while True:
                try:
                    lives.append(pickle.load(f))
                except EOFError:
                    break
    return lives


def token_bias_weights(b):
    """TOKEN_LONG_BIAS sampling weights (frames / TOKEN_BASE_FRAMES) ** b per
    TOKEN_VOCAB slot. E3-A7 (E3-A13: as written): a MINED token's frames
    are capped at the constant muzero.TOKEN_BIAS_CAP_FRAMES (32), so length
    alone does not advantage it; a duration token's weight is unchanged."""
    vocab = muzero.TOKEN_VOCAB
    cap = muzero.TOKEN_BIAS_CAP_FRAMES

    def frames(t):
        f = muzero.token_total_frames(t)
        if muzero.token_is_mined(t):
            return min(f, cap)
        return f

    return np.array([(frames(t) / muzero.TOKEN_BASE_FRAMES) ** b
                     for t in vocab], dtype=np.float64)


def mined_overlap(vocab):
    """E3-A10 read (2), the redundancy E3a predicts: None for a vocabulary
    without MINED tokens; else an [A, A] 0/1 matrix whose row a marks the
    mined tokens m != a whose per-base-step inputs OVERLAP token a's -- the
    two agree on every base step they share (one's inputs a prefix of the
    other's): the search children that begin by playing what a plays."""
    if vocab is None or not any(muzero.token_is_mined(t) for t in vocab):
        return None
    inputs = [muzero.token_inputs(t) for t in vocab]
    A = len(vocab)
    out = np.zeros((A, A), dtype=np.float64)
    for m, tok in enumerate(vocab):
        if not muzero.token_is_mined(tok):
            continue
        for a in range(A):
            k = min(len(inputs[a]), len(inputs[m]))
            if a != m and inputs[a][:k] == inputs[m][:k]:
                out[a, m] = 1.0
    return out


def token_gate(spec, use_ray, cpu_actor, ray_workers):
    """TOKEN_VOCAB (curriculum P2a) runs only on the path it was built and
    tested on: policy-only acting with birth search, paper targets through
    the target model, the vectorized communal refresh, TD value targets,
    the paper wrapper stack at the base skip, --ray --cpu-actor without a
    worker pool. Refuses at startup -- before any --overwrite deletion --
    rather than training on a path that ignores the tokens' frames."""
    vocab = muzero.TOKEN_VOCAB
    if vocab is None:
        return
    base = int(muzero.TOKEN_BASE_FRAMES)
    need = {"PAPER_TARGETS": True, "BIRTH_SEARCH": True,
            "USE_TARGET_MODEL": True, "REFRESH_VTREE": True,
            "USE_ATARI_WRAPPERS": True, "USE_SEARCH_ACTING": False,
            "USE_GUMBEL_SEARCH": False, "ACTOR_GUMBEL_SEARCH": False,
            "REFRESH_OFFLOAD": False, "REFRESH_SATURATE": False,
            "ASYNC_LOOP": False, "VALUE_TARGET_MODE": "td"}
    bad = [f"{k}={getattr(muzero, k)!r}" for k, v in need.items()
           if getattr(muzero, k) != v]
    if bad:
        raise SystemExit("TOKEN_VOCAB runs only on the tested path "
                         f"(policy-only acting + birth search, paper "
                         f"targets, vectorized refresh); got {', '.join(bad)}")
    if not (use_ray and cpu_actor) or ray_workers:
        raise SystemExit("TOKEN_VOCAB runs with --ray --cpu-actor (no "
                         "--ray-workers pool)")
    if FRAMESKIP != base:
        raise SystemExit(f"TOKEN_VOCAB plays tokens as steps of the wrapper "
                         f"stack at --frameskip {base}; got {FRAMESKIP}")
    n_env = env_actions(ENV_NAME)
    for i, tok in enumerate(vocab):
        # E3 (E3-A2): a mined token's every step is checked as a duration
        # token is; it holds >= 2 steps and at most TOKEN_MINED_MAX_FRAMES
        mined = muzero.token_is_mined(tok)
        if mined and len(tok) < 2:
            raise SystemExit(f"TOKEN_VOCAB[{i}] = {tok!r}: a mined token "
                             f"holds at least two steps")
        for a, f in (tok if mined else (tok,)):
            if not (isinstance(f, int) and f >= base and f % base == 0):
                raise SystemExit(f"TOKEN_VOCAB[{i}] = {tok!r}: frames must be "
                                 f"a positive multiple of {base}")
            if not (isinstance(a, int) and 0 <= a < n_env):
                raise SystemExit(f"TOKEN_VOCAB[{i}] = {tok!r}: ALE action "
                                 f"outside {ENV_NAME}'s {n_env}")
        if mined and (muzero.token_total_frames(tok)
                      > muzero.TOKEN_MINED_MAX_FRAMES):
            raise SystemExit(f"TOKEN_VOCAB[{i}] = {tok!r}: a mined token of "
                             f"{muzero.token_total_frames(tok)} frames (at "
                             f"most {muzero.TOKEN_MINED_MAX_FRAMES})")
    # E3-A2: two tokens with the same per-base-step inputs are one token (for
    # duration tokens of whole base steps: the same as equal tokens)
    if len({muzero.token_inputs(t) for t in vocab}) != len(vocab):
        raise SystemExit("TOKEN_VOCAB has duplicate tokens")


def option_gate(spec, use_ray, cpu_actor, ray_workers, distill_set=None,
                init_weights=None):
    """Test O (CURRICULUM: "token_gate becomes option_gate"): an arm under
    OPTION_PRECHECK has no vocabulary, so token_gate passes it unchecked; this
    holds it to the same path instead -- policy-only acting with birth
    search, paper targets through the target model, the vectorized communal
    refresh, TD value targets, the paper wrapper stack at the base skip,
    --ray --cpu-actor without a worker pool -- with the env's whole action
    space, one move per base step. Refuses at startup -- before any
    --overwrite deletion. Off (OPTION_LENGTH 1, OPTION_EDGE True, no
    OPTION_PRECHECK): returns at once.

    Every arm, first, the knobs themselves: OPTION_LENGTH an int >= 1 and
    OPTION_EDGE a bool; OPTION_EDGE False needs option heads (OPTION_LENGTH
    >= 2: arm OH), so OPTION_LENGTH 1 with the edge off is refused.

    An OPTION ARM (OPTION_LENGTH >= 2, Test O stage 4a: OF, OH) is held to
    the pre-check's tested path above and also refused unless TRAIN_PREFETCH
    is off (a prefetched batch would read PMAX and the head targets one to
    TRAIN_PREFETCH_DEPTH + 1 steps late; muzero_option_targets' THE ORDER),
    ACTION_ENCODING is "embed" (the per-position jump rows are embeddings)
    and no TOKEN_VOCAB is set (options replace the vocabulary). The
    registered arms set OPTION_PRECHECK too; the option checks cover it.
    Audit r1 of stage 4a: an option arm also refuses what the registration
    leaves off for every arm ("Left off for all arms: long-token bias,
    swaps, distillation, mining, ceiling growth, own-play logs"; ARMS: "from
    scratch, one generation, no swaps") and Test C / C2's knobs:
    RESET_EVERY_STEPS, RAMP_STEPS, ceiling GROWTH (REWARD_GROWTH with
    REWARD_GROWTH_START != REWARD_GROWTH_CEILING -- REWARD_GROWTH itself is
    on in O0's recipe, holding the sign clip at a constant ceiling 1),
    OWN_PLAY_LOG, TOKEN_LONG_BIAS, SCHEDULE_OFFSET_STEPS (a distilled
    student's offline phase), the spec's _student / _master_arm /
    _distill_offline_steps / _mine / _mine_q, --distill-set and
    --init-weights."""
    L, edge = muzero.OPTION_LENGTH, muzero.OPTION_EDGE
    if (isinstance(L, bool) or not isinstance(L, (int, np.integer))
            or L < 1):
        raise SystemExit(f"OPTION_LENGTH must be an int >= 1 (1 = no "
                         f"options); got {L!r}")
    if not isinstance(edge, bool):
        raise SystemExit(f"OPTION_EDGE must be True or False; got {edge!r}")
    option = int(L) > 1
    if not option and edge is not True:
        raise SystemExit("OPTION_EDGE False turns off the option edge of an "
                         "option arm (OH): it needs option heads, "
                         "OPTION_LENGTH >= 2; got OPTION_LENGTH 1")
    if not (option or muzero.OPTION_PRECHECK):
        return
    who = f"OPTION_LENGTH {int(L)}" if option else "OPTION_PRECHECK"
    if muzero.TOKEN_VOCAB is not None:
        raise SystemExit(f"{who} reads plain moves, one per base step: it "
                         "refuses a TOKEN_VOCAB")
    if option:
        off = [f"{k}={getattr(muzero, k)!r}"
               for k in ("RESET_EVERY_STEPS", "RAMP_STEPS", "OWN_PLAY_LOG",
                         "TOKEN_LONG_BIAS", "SCHEDULE_OFFSET_STEPS")
               if getattr(muzero, k)]
        if (muzero.REWARD_GROWTH and muzero.REWARD_GROWTH_START
                != muzero.REWARD_GROWTH_CEILING):
            off.append(f"REWARD_GROWTH from {muzero.REWARD_GROWTH_START!r} "
                       f"to {muzero.REWARD_GROWTH_CEILING!r} (ceiling "
                       f"growth)")
        off += [f"{k}={spec[k]!r}" for k in
                ("_student", "_master_arm", "_distill_offline_steps",
                 "_mine", "_mine_q") if spec.get(k)]
        if distill_set:
            off.append("--distill-set")
        if init_weights:
            off.append("--init-weights")
        if off:
            raise SystemExit(f"{who} runs from scratch, one generation, "
                             "with what the registration leaves off for "
                             "every arm left off (resets, ramps, ceiling "
                             "growth, own-play logs, the long-token bias, "
                             "swaps, distillation, mining); got "
                             f"{', '.join(off)}")
    base = int(muzero.TOKEN_BASE_FRAMES)
    need = {"PAPER_TARGETS": True, "BIRTH_SEARCH": True,
            "USE_TARGET_MODEL": True, "REFRESH_VTREE": True,
            "USE_ATARI_WRAPPERS": True, "USE_SEARCH_ACTING": False,
            "USE_GUMBEL_SEARCH": False, "ACTOR_GUMBEL_SEARCH": False,
            "REFRESH_OFFLOAD": False, "REFRESH_SATURATE": False,
            "ASYNC_LOOP": False, "VALUE_TARGET_MODE": "td"}
    if option:
        need.update({"TRAIN_PREFETCH": False, "ACTION_ENCODING": "embed"})
    bad = [f"{k}={getattr(muzero, k)!r}" for k, v in need.items()
           if getattr(muzero, k) != v]
    if bad:
        raise SystemExit(f"{who} runs only on the tested path "
                         f"(policy-only acting + birth search, paper "
                         f"targets, vectorized refresh"
                         + (", no batch prefetch, embedded actions"
                            if option else "")
                         + f"); got {', '.join(bad)}")
    if not (use_ray and cpu_actor) or ray_workers:
        raise SystemExit(f"{who} runs with --ray --cpu-actor (no "
                         "--ray-workers pool)")
    if FRAMESKIP != base:
        raise SystemExit(f"{who} plays one move per step of the "
                         f"wrapper stack at --frameskip {base}; got "
                         f"{FRAMESKIP}")
    n_env = env_actions(ENV_NAME)
    if ACTIONS != n_env:
        raise SystemExit(f"{who} plays {ENV_NAME}'s whole action "
                         f"space ({n_env} moves); got {ACTIONS}")


def option_precheck_counts(pi, played, m):
    """Test O's pre-check (CURRICULUM "O0 PRE-CHECK") over one life's acted
    positions: pi [n, A] the birth targets, played [n] the moves played, m
    [n] the argmax of the learner's nets. Returns the sums a row adds up:
      n          every acted position: the denominator, as registered;
      hold2      positions t with played[t] == m[t] and pi[t+1][m[t+1]] >
                 0.5: able to hold a length-2 option;
      hold3      those with also played[t+1] == m[t+1] and pi[t+1][m[t+1]] x
                 pi[t+2][m[t+2]] > 0.5: able to hold a length-3 option;
      pi_m, pi_played, hit   the sums of pi[t][m[t]], pi[t][played[t]] and
                 played[t] == m[t].
    A LIFE'S END: t+1 and t+2 must be acted positions of this life. The
    closing record counts as terminal, as in the registered head targets
    (getOptionPolicy: past the last action every head is stop), whether the
    life ended or the cap cut it, though a cut life's closing record carries
    a search. So the last acted position counts in neither hold2 nor hold3,
    the one before it can count in hold2 but not in hold3, and both stay in
    n."""
    at = np.arange(len(played))
    pm = pi[at, m]
    hit = played == m
    two = hit[:-1] & (pm[1:] > 0.5)
    three = two[:-1] & hit[1:-1] & (pm[1:-1] * pm[2:] > 0.5)
    return {"n": len(played), "hold2": int(two.sum()),
            "hold3": int(three.sum()), "pi_m": float(pm.sum()),
            "pi_played": float(pi[at, played].sum()), "hit": int(hit.sum())}


# Test O stage 3: the option search's per-root counters a birth search sums
# (muzero_vtree_options' info dict; the diagnostics stage reads the sums)
OPTION_COUNT_SUMS = ("expanded_nodes", "expanded_with_option",
                     "option_len_hist", "simulations", "sims_crossing_option",
                     "crossed_len_hist", "leaf_depth_sum")


def option_search_counts(acc, info):
    """Add one option-search call's info (muzero_vtree_options) to `acc`
    in place: the per-root counters summed over its roots (the length
    histograms per length), leaf_depth_max their max, "roots" the roots
    searched. Read (3)'s rates are ratios of these sums."""
    for k in OPTION_COUNT_SUMS:
        v = np.asarray(info[k]).sum(axis=0)
        acc[k] = acc[k] + v if k in acc else v
    m = int(np.asarray(info["leaf_depth_max"]).max(initial=0))
    acc["leaf_depth_max"] = max(acc.get("leaf_depth_max", 0), m)
    acc["roots"] = acc.get("roots", 0) + len(info["root_pmax"])
    return acc


# Test O stage 4a: the option arms' per-row diagnostics (CURRICULUM "Test O",
# READS (3), (4), (6), DEPARTURES 3 and 7, DESIGN "Prefix BN"). Accumulated in
# Run._opt_window (plain ints, floats and lists: checkpointed, so a resume
# keeps a partial row), emitted by Run._merge_option_stats in the row that
# carries O0's pre-check shares, OPTION_LENGTH > 1 only. The on/off root
# probe's roots: OPTION_PROBE_ROOTS of the row's acted positions.
OPTION_PROBE_ROOTS = 256


def option_window_empty(L, A, K):
    """One row's option window, every count at zero. L the option length, A
    the moves, K the unroll."""
    return {
        # read (3): the LIVE birth searches' counters (option_search_counts,
        # every life's _birth_option_counts; the refresh never adds here)
        "roots": 0, "expanded_nodes": 0, "expanded_with_option": 0,
        "option_len_hist": [0] * (L + 1), "simulations": 0,
        "sims_crossing_option": 0, "crossed_len_hist": [0] * (L + 1),
        "leaf_depth_sum": 0.0, "leaf_depth_max": 0,
        # DEPARTURE 7 and read (4), over the acted positions at birth
        "acted": 0, "hit": 0, "pi_m": 0.0, "frames": 0, "play": [0] * A,
        "s4_n": [0] * L, "s4_opt": [0] * L, "s4_rep": [0] * L,
        "s4_prev": [0] * L, "s4_first": [0] * L,
        "s4_hist": [[0] * A for _ in range(L)],
        "with_option": 0, "nonrepeat": 0, "root_len_hist": [0] * (L + 1),
        # the row's train steps (TrainingHelper._option_last, note_option_step)
        "steps": 0, "mask_share": 0.0, "jump_share": [0.0] * K,
        "span": 0.0, "loss_steps": 0, "nonfinite": 0, "loss": 0.0,
        "cons_jump": 0.0, "cons_jump_n": 0, "cons_single": 0.0,
        "cons_single_n": 0, "birth_pos": 0, "birth_pos_n": 0,
        "birth_root": 0, "birth_root_n": 0,
    }


def option_window_add_search(w, counts):
    """Add one life's birth-search counters (_birth_option_counts, the
    option_search_counts sums; {} when nothing was searched) to window w in
    place."""
    for k in ("roots", "expanded_nodes", "expanded_with_option",
              "simulations", "sims_crossing_option"):
        w[k] += int(np.sum(counts.get(k, 0)))
    w["leaf_depth_sum"] += float(np.sum(counts.get("leaf_depth_sum", 0.0)))
    w["leaf_depth_max"] = max(w["leaf_depth_max"],
                              int(counts.get("leaf_depth_max", 0)))
    for k in ("option_len_hist", "crossed_len_hist"):
        if k in counts:
            v = np.asarray(counts[k]).reshape(-1)
            if len(v) != len(w[k]):
                raise ValueError(f"{k} of {len(v)} lengths in a window of "
                                 f"{len(w[k])}")
            w[k] = [int(a) + int(b) for a, b in zip(w[k], v)]
    return w


def option_state_counts(played, options, L, A):
    """Read (4), STATE DEPENDENCE, over one life's acted positions 0..n-1:
    played [n] the moves played, options [n] each position's root suggestion
    (the birth search's dominant option, its moves; () when none). Step j =
    1..L compares the option's j-th move with the move played at t + j - 1
    (step 1 = its first move against the move played at t), unconditionally
    (an earlier miss does not end the comparison).
    THE DENOMINATOR at step j (s4_n[j-1]): acted positions t >= 1 whose
    suggestion has at least j moves and where t + j - 1 is an acted position
    of this life (a life's end leaves the later steps nothing to compare
    with). t = 0 is left out of every step: the repeat baseline needs the
    move played before t, which a life's first position lacks; so the three
    predictors share their positions.
      s4_opt[j-1]   the option's j-th move == played[t + j - 1];
      s4_rep[j-1]   REPEAT THE LAST MOVE: played[t - 1] == played[t + j - 1]
                    (persistence from what is known at t, every step);
      s4_prev[j-1]  PER-STEP PERSISTENCE: played[t + j - 2] == played[t + j
                    - 1], the move played just before step j's position
                    (s4_rep at step 1; audit r1 of stage 4a);
      s4_first[j-1] THE OPTION'S FIRST MOVE REPEATED: o_1 == played[t + j -
                    1] at every step (s4_opt at step 1). Heads that only
                    copy head 0, (o_1, o_1, o_1), score exactly this, so
                    the later heads show state dependence of their own only
                    where s4_opt beats it (audit r1: they beat s4_rep);
      s4_hist[j-1]  the played moves at those positions, per move: the
                    MARGINAL baseline's hits are s4_hist[j-1][mode], mode the
                    row's most frequent played move (option_row_fields);
      with_option   positions holding an option (t = 0 included), of which
      nonrepeat     those whose moves are not one move repeated;
      root_len_hist positions by suggestion length (index 0 = none)."""
    n = len(played)
    out = {"s4_n": [0] * L, "s4_opt": [0] * L, "s4_rep": [0] * L,
           "s4_prev": [0] * L, "s4_first": [0] * L,
           "s4_hist": [[0] * A for _ in range(L)], "with_option": 0,
           "nonrepeat": 0, "root_len_hist": [0] * (L + 1)}
    for t, o in enumerate(options):
        k = len(o)
        out["root_len_hist"][k] += 1
        if k >= 2:
            out["with_option"] += 1
            out["nonrepeat"] += int(len(set(o)) > 1)
        if t == 0:
            continue
        prev = int(played[t - 1])
        for j in range(1, k + 1):
            q = t + j - 1
            if q >= n:
                break
            m = int(played[q])
            out["s4_n"][j - 1] += 1
            out["s4_opt"][j - 1] += int(int(o[j - 1]) == m)
            out["s4_rep"][j - 1] += int(prev == m)
            out["s4_prev"][j - 1] += int(int(played[q - 1]) == m)
            out["s4_first"][j - 1] += int(int(o[0]) == m)
            out["s4_hist"][j - 1][m] += 1
    return out


def option_probe_stats(pol_on, pol_off, root_option, root_len):
    """Read (3)'s ON/OFF ROOT PROBE over T roots searched twice, the option
    edge on (pol_on) and off (pol_off), [T, A] visit targets; root_option
    [T, L] the root's dominant option (-1 padded), root_len [T] its length
    (0 = none). Sums and counts a row turns into means:
      js_sum      the Jensen-Shannon divergence between the two targets, in
                  nats (0 log 0 = 0; at most ln 2), summed over roots;
      flips       roots whose top move FLIPS: no move is top in both (a tie
                  at the top counts every tied move as top, so a tie never
                  makes a flip by itself: "ties count as no flip");
      shift_roots roots holding an option (the others are left out), and
      shift_sum   their pi_on[m1] - pi_off[m1], m1 the option's first move.
    """
    P = np.asarray(pol_on, dtype=np.float64)
    Q = np.asarray(pol_off, dtype=np.float64)
    T = len(P)
    M = 0.5 * (P + Q)

    def kl(X):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(X > 0, X * np.log(X / np.where(M > 0, M, 1.0)), 0.0)
        return t.sum(axis=1)

    js = 0.5 * kl(P) + 0.5 * kl(Q)
    top_on = P == P.max(axis=1, keepdims=True)
    top_off = Q == Q.max(axis=1, keepdims=True)
    flip = ~(top_on & top_off).any(axis=1)
    has = np.asarray(root_len) >= 2
    first = np.asarray(root_option)[:, 0].clip(min=0)
    rows = np.arange(T)
    shift = P[rows, first] - Q[rows, first]
    return {"roots": int(T), "js_sum": float(js.sum()),
            "flips": int(flip.sum()), "shift_roots": int(has.sum()),
            "shift_sum": float(shift[has].sum())}


def _ratio(a, b, nd=6):
    return round(float(a) / float(b), nd) if b else None


def option_prefix_panel(acc, spans, err_k, ctl_k, err_1, ctl_1, abs1_eval,
                        abs1_train):
    """DESIGN "Prefix BN": the row's prefix panel (evaluate_diagnostics) on
    jump rows, one option batch's sums into `acc` in place. spans [B, K] the
    rows' edge spans; per row (all [B]): err_k / ctl_k the step-K prefix
    error |prefix - truth| and its control |truth| (the panel's "prefix" /
    "control", truth the undiscounted sum of the edges' rewards: a jump
    edge's reward is its positions' sum, DEPARTURE 4); err_1 / ctl_1 the
    step-1 error and control ("prefix_step1"); abs1_eval / abs1_train
    |step-1 prefix| through eval-mode and train-mode value-prefix BN (PB's
    probe, Step 0 (iii): the FIXED line was eval / train <= 2). Rows are
    "jump" when any edge of the window jumps, else "single"; for step 1,
    "jump1" when the FIRST edge jumps, else "single1"."""
    spans = np.asarray(spans)
    for kind, m in (("jump", spans.max(axis=1) > 1),
                    ("single", spans.max(axis=1) <= 1)):
        acc[f"{kind}_n"] = acc.get(f"{kind}_n", 0) + int(m.sum())
        acc[f"{kind}_err"] = acc.get(f"{kind}_err", 0.0) + float(err_k[m].sum())
        acc[f"{kind}_ctl"] = acc.get(f"{kind}_ctl", 0.0) + float(ctl_k[m].sum())
    for kind, m in (("jump1", spans[:, 0] > 1), ("single1", spans[:, 0] <= 1)):
        for k, v in (("n", m.sum()), ("err", err_1[m].sum()),
                     ("ctl", ctl_1[m].sum()), ("eval", abs1_eval[m].sum()),
                     ("train", abs1_train[m].sum())):
            acc[f"{kind}_{k}"] = acc.get(f"{kind}_{k}", 0) + (
                int(v) if k == "n" else float(v))
    return acc


def option_prefix_fields(acc):
    """option_prefix_panel's sums as row fields: means per row kind, the
    rows behind each -- step K by any-jump / no-jump windows, step 1 by
    first-edge jump / one-step first edge (error, control, rows for both:
    audit r1 of stage 4a) -- and the step-1 BN ratio (eval / train-mode
    |prefix|) for both first-edge kinds."""
    g = acc.get
    return {
        "opt_prefix_jump": _ratio(g("jump_err", 0.0), g("jump_n", 0)),
        "opt_prefix_jump_control": _ratio(g("jump_ctl", 0.0), g("jump_n", 0)),
        "opt_prefix_jump_rows": g("jump_n", 0),
        "opt_prefix_single": _ratio(g("single_err", 0.0), g("single_n", 0)),
        "opt_prefix_single_control": _ratio(g("single_ctl", 0.0),
                                            g("single_n", 0)),
        "opt_prefix_single_rows": g("single_n", 0),
        "opt_prefix_step1_jump": _ratio(g("jump1_err", 0.0), g("jump1_n", 0)),
        "opt_prefix_step1_jump_control": _ratio(g("jump1_ctl", 0.0),
                                                g("jump1_n", 0)),
        "opt_prefix_step1_jump_rows": g("jump1_n", 0),
        "opt_prefix_step1_single": _ratio(g("single1_err", 0.0),
                                          g("single1_n", 0)),
        "opt_prefix_step1_single_control": _ratio(g("single1_ctl", 0.0),
                                                  g("single1_n", 0)),
        "opt_prefix_step1_single_rows": g("single1_n", 0),
        "opt_prefix_step1_jump_bn_ratio": _ratio(g("jump1_eval", 0.0),
                                                 g("jump1_train", 0.0), 4),
        "opt_prefix_step1_single_bn_ratio": _ratio(g("single1_eval", 0.0),
                                                   g("single1_train", 0.0),
                                                   4),
    }


def option_row_fields(w, probe, base_frames):
    """The row fields of option window w (option_window_empty) and the
    row's on/off probe (option_probe_stats plus "seconds"; None when no root
    was probed). Counts are logged beside every rate, so a read pooled over
    rows (read (3)'s [90k, end]) is computed from counts. base_frames: the
    frames of one base step (TOKEN_BASE_FRAMES).
    opt_acted counts the row's acted positions from a CHECKPOINTED window,
    so it equals O0's pre-check pc_positions only in a row no resume cut:
    the pre-check window is not checkpointed (_note_precheck), and after a
    resume pc_positions holds the post-resume lives alone."""
    L = len(w["s4_n"])
    play = w["play"]
    mode = int(np.argmax(play)) if sum(play) else None
    marg = ([int(h[mode]) for h in w["s4_hist"]] if mode is not None
            else [0] * L)
    steps, ls = w["steps"], w["loss_steps"]
    out = {
        # read (3) ENGAGEMENT, the live birth searches
        "opt_roots": w["roots"], "opt_nodes": w["expanded_nodes"],
        "opt_nodes_with_option": w["expanded_with_option"],
        "opt_len_hist": list(w["option_len_hist"]),
        "opt_sims": w["simulations"],
        "opt_sims_crossing": w["sims_crossing_option"],
        "opt_crossed_len_hist": list(w["crossed_len_hist"]),
        "opt_leaf_depth_sum": round(w["leaf_depth_sum"], 3),
        "opt_leaf_depth_max": w["leaf_depth_max"],
        "opt_node_option_rate": _ratio(w["expanded_with_option"],
                                       w["expanded_nodes"]),
        "opt_crossing_rate": _ratio(w["sims_crossing_option"],
                                    w["simulations"]),
        "opt_leaf_depth_mean": _ratio(w["leaf_depth_sum"], w["simulations"],
                                      4),
        # DEPARTURE 7: the hit rate beside mean pi[m], acted positions
        "opt_acted": w["acted"], "opt_hits": w["hit"],
        "opt_hit_rate": _ratio(w["hit"], w["acted"], 4),
        "opt_pi_m": _ratio(w["pi_m"], w["acted"], 4),
        # read (4) STATE DEPENDENCE (option_state_counts)
        "opt_s4_positions": list(w["s4_n"]),
        "opt_s4_option_hits": list(w["s4_opt"]),
        "opt_s4_repeat_hits": list(w["s4_rep"]),
        "opt_s4_marginal_hits": marg, "opt_s4_mode": mode,
        "opt_s4_played_hist": [list(h) for h in w["s4_hist"]],
        "opt_s4_accuracy": [_ratio(a, n, 4)
                            for a, n in zip(w["s4_opt"], w["s4_n"])],
        "opt_s4_repeat_accuracy": [_ratio(a, n, 4)
                                   for a, n in zip(w["s4_rep"], w["s4_n"])],
        "opt_s4_persist_hits": list(w["s4_prev"]),
        "opt_s4_persist_accuracy": [_ratio(a, n, 4) for a, n in
                                    zip(w["s4_prev"], w["s4_n"])],
        "opt_s4_first_move_hits": list(w["s4_first"]),
        "opt_s4_first_move_accuracy": [_ratio(a, n, 4) for a, n in
                                       zip(w["s4_first"], w["s4_n"])],
        "opt_s4_marginal_accuracy": [_ratio(a, n, 4)
                                     for a, n in zip(marg, w["s4_n"])],
        "opt_play_counts": list(play),
        "opt_root_options": w["with_option"],
        "opt_root_option_share": _ratio(w["with_option"], w["acted"], 4),
        "opt_root_len_hist": list(w["root_len_hist"]),
        "opt_nonrepeat": w["nonrepeat"],
        "opt_nonrepeat_share": _ratio(w["nonrepeat"], w["with_option"], 4),
        # the row's train steps: unroll edges, the option loss, the batch
        # mask share (not DEPARTURE 7's hit rate), jump rows' consistency
        "opt_steps": steps, "opt_loss_steps": ls,
        "opt_steps_nonfinite": w["nonfinite"],
        "opt_loss": _ratio(w["loss"], ls, 6),
        "opt_mask_share": _ratio(w["mask_share"], steps, 4),
        "opt_jump_share": [_ratio(s, steps, 4) for s in w["jump_share"]],
        "opt_span_mean": _ratio(w["span"], steps, 4),
        "opt_span_frames": (round(w["span"] / steps * base_frames, 2)
                            if steps else None),
        "opt_cons_jump": _ratio(w["cons_jump"], w["cons_jump_n"], 6),
        "opt_cons_jump_edges": w["cons_jump_n"],
        "opt_cons_single": _ratio(w["cons_single"], w["cons_single_n"], 6),
        "opt_cons_single_edges": w["cons_single_n"],
        # DEPARTURE 3: the sampled birth-target share
        "opt_birth_target_share": _ratio(w["birth_pos"], w["birth_pos_n"],
                                         4),
        "opt_birth_target_positions": w["birth_pos_n"],
        "opt_birth_target_share_roots": _ratio(w["birth_root"],
                                               w["birth_root_n"], 4),
        "opt_birth_target_roots": w["birth_root_n"],
        # read (6): decisions per second of play (60 frames a second)
        "opt_decisions_per_s": _ratio(60 * w["acted"], w["frames"], 3),
    }
    p = probe or {"roots": 0}
    T = p["roots"]
    out.update({
        "opt_probe_roots": T,
        "opt_probe_js": _ratio(p.get("js_sum", 0.0), T, 6),
        "opt_probe_flips": p.get("flips", 0),
        "opt_probe_flip_share": _ratio(p.get("flips", 0), T, 4),
        "opt_probe_shift_roots": p.get("shift_roots", 0),
        "opt_probe_shift": _ratio(p.get("shift_sum", 0.0),
                                  p.get("shift_roots", 0), 6),
        "opt_probe_s": (round(p["seconds"], 2) if "seconds" in p else None),
    })
    return out


# REWARD_GROWTH's REWARD_UNIT per game (env_tag_fragment of the env id): the
# smallest positive raw reward, so that ceiling 1 is the sign clip. A game
# missing here is refused (audit r1 A-NIT2: Pong's +-1 at unit 10 would
# silently store +-0.1).
REWARD_UNITS = {"mspacman": 10.0, "mspacmannoframeskip": 10.0,
                "qbert": 25.0, "qbertnoframeskip": 25.0}


def reward_growth_gate(use_ray, cpu_actor, ray_workers):
    """REWARD_GROWTH (P2a on Ms. Pac-Man) runs only on the path it was built
    and tested on: policy-only acting with birth search into paper targets
    (where raw reward tuples become clipped records), the inline vectorized
    refresh (reads the re-clipped records), TD value targets, the
    synchronous driver (re-clips at each iteration's top), --ray
    --cpu-actor without a worker pool; on a game whose reward unit is
    REWARD_UNIT; with a schedule whose ceiling stays inside the supports.
    Refuses at startup -- before any --overwrite deletion."""
    if not muzero.REWARD_GROWTH:
        return
    need = {"PAPER_TARGETS": True, "BIRTH_SEARCH": True,
            "USE_SEARCH_ACTING": False, "REFRESH_VTREE": True,
            "REFRESH_OFFLOAD": False, "REFRESH_SATURATE": False,
            "ASYNC_LOOP": False, "VALUE_TARGET_MODE": "td"}
    bad = [f"{k}={getattr(muzero, k)!r}" for k, v in need.items()
           if getattr(muzero, k) != v]
    if bad:
        raise SystemExit("REWARD_GROWTH runs only on the tested path "
                         "(policy-only acting + birth search, paper targets, "
                         f"inline vectorized refresh); got {', '.join(bad)}")
    if not (use_ray and cpu_actor) or ray_workers:
        raise SystemExit("REWARD_GROWTH runs with --ray --cpu-actor (no "
                         "--ray-workers pool)")
    lo, hi = muzero.REWARD_GROWTH_START, muzero.REWARD_GROWTH_CEILING
    if not (muzero.REWARD_UNIT > 0 and 0 < lo <= hi):
        raise SystemExit(f"REWARD_GROWTH needs REWARD_UNIT > 0 and 0 < "
                         f"START <= CEILING; got unit {muzero.REWARD_UNIT!r}, "
                         f"{lo!r} -> {hi!r}")
    if not 0 < muzero.REWARD_GROWTH_END <= 1:
        raise SystemExit(f"REWARD_GROWTH_END {muzero.REWARD_GROWTH_END!r} is "
                         f"a fraction of the run in (0, 1]")
    unit = REWARD_UNITS.get(env_tag_fragment(ENV_NAME))
    if unit != muzero.REWARD_UNIT:
        raise SystemExit(f"REWARD_GROWTH: REWARD_UNIT {muzero.REWARD_UNIT!r} "
                         f"is not {ENV_NAME}'s reward unit ({unit!r}; "
                         f"REWARD_UNITS)")
    steps = support_steps(muzero.TOKEN_VOCAB)
    if not ceiling_fits_supports(hi, steps):
        raise SystemExit(f"REWARD_GROWTH_CEILING {hi!r} lies past the reward "
                         f"support (half-widths {muzero.REWARD_SUPPORT:.4g} "
                         f"per step, {muzero.VALUE_PREFIX_SUPPORT:.4g} per "
                         f"{muzero.UNROLL_STEPS}-step prefix, h-space)"
                         + (f" for a record of {steps} base steps (the "
                            f"vocabulary's longest token)" if steps > 1
                            else ""))


def ceiling_fits_supports(c, steps=1):
    """One base step's reward clipped at `c` fits the per-step reward
    support, and UNROLL_STEPS of them the value-prefix support (the head the
    paper arms train; a token's record sums several base steps -- a prefix
    past the support saturates, logged by reward_max per record). Also the
    distillation prep's check on a student's ceiling (Test 2). E3-A9:
    `steps`, a record's base steps -- support_steps: the vocabulary's
    longest token (16 base steps) once it holds a mined token, 1 (the check
    as it always was) otherwise."""
    sup = muzero.scalar_categorical.support_for
    c = c * steps if steps != 1 else c
    return not (sup(c) > muzero.REWARD_SUPPORT
                or sup(c * muzero.UNROLL_STEPS) > muzero.VALUE_PREFIX_SUPPORT)


def support_steps(vocab):
    """E3-A9: the base steps one record of `vocab` can sum -- its longest
    token's, once the vocabulary holds a MINED token; 1 for every vocabulary
    of duration tokens and for the plain action space (the support check
    they always ran: their records were never bounded by token length)."""
    if vocab is None or not any(muzero.token_is_mined(t) for t in vocab):
        return 1
    return max(muzero.token_total_frames(t) // muzero.TOKEN_BASE_FRAMES
               for t in vocab)


# -- curriculum P3: the distillation step's students (2026-10-21) -----------
def file_sha256(path):
    """Hex sha256 of a file's bytes (streamed)."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _canon_or_exit(vocab, what="student vocabulary"):
    try:
        return muzero.canon_vocab(vocab)
    except (TypeError, ValueError):
        raise SystemExit(f"{what}: not a list of [action, frames] pairs: "
                         f"{vocab!r}")


def validate_student_vocab(vocab, master_vocab, mined=None, max_mined=None):
    """A student vocabulary (curriculum P3, the 10-17 rule): the master's
    tokens minus the dropped ones, in the master's slot order, every
    TOKEN_BASE_FRAMES token kept (a dropped token is replayed as raw base
    steps), slot 0 unchanged (the placeholder action's token). Returns the
    vocabulary as a tuple of canonical tokens (muzero.canon_token: (int, int)
    tuples for duration tokens); SystemExit otherwise.

    E3 (E3-A4, E3-A8): `mined`, the MINED tail vocab.json declares (its
    "mined" tokens, in order; empty / None: the rule above, exactly). The
    vocabulary is then the rule's kept tokens FOLLOWED by exactly that tail:
    each a mined token (>= 2 steps of whole base steps, at most
    TOKEN_MINED_MAX_FRAMES frames), at most `max_mined` of them (None: no
    cap -- a vocabulary across several swaps), the slot count never past
    the master's. A master's own mined tokens (the second swap) are tokens
    of the master like any other. Whenever the vocabulary holds a mined
    token, no two tokens may share their per-base-step inputs (E3-A2)."""
    vocab = _canon_or_exit(vocab)
    mined = _canon_or_exit(mined or (), "the declared mined tokens")
    n_tail = len(mined)
    if n_tail and (len(vocab) < n_tail
                   or vocab[len(vocab) - n_tail:] != mined):
        raise SystemExit(f"student vocabulary: its last {n_tail} tokens are "
                         f"not the declared mined tokens {list(mined)}")
    kept = vocab[:len(vocab) - n_tail]
    master = _canon_or_exit(master_vocab, "master vocabulary")
    pos = {t: i for i, t in enumerate(master)}
    if any(t not in pos for t in kept):
        raise SystemExit(f"student vocabulary: tokens outside the master's: "
                         f"{[t for t in kept if t not in pos]}")
    order = [pos[t] for t in kept]
    if order != sorted(set(order)):
        raise SystemExit("student vocabulary: not in the master's slot order "
                         "(or duplicated)")
    base = [t for t in master if not muzero.token_is_mined(t)
            and t[1] == muzero.TOKEN_BASE_FRAMES]
    if any(t not in kept for t in base):
        raise SystemExit(f"student vocabulary drops a {muzero.TOKEN_BASE_FRAMES}"
                         f"-frame token: {[t for t in base if t not in kept]}")
    if kept[0] != master[0]:
        raise SystemExit(f"student vocabulary: slot 0 is {kept[0]!r}, not "
                         f"the master's {master[0]!r}")
    if n_tail:
        b = muzero.TOKEN_BASE_FRAMES
        for t in mined:
            if not muzero.token_is_mined(t):
                raise SystemExit(f"student vocabulary: declared mined token "
                                 f"{t!r} is a duration token")
            if any(f < b or f % b for _, f in t):
                raise SystemExit(f"student vocabulary: mined token {t!r}: "
                                 f"steps must be whole {b}-frame base steps")
            if muzero.token_total_frames(t) > muzero.TOKEN_MINED_MAX_FRAMES:
                raise SystemExit(f"student vocabulary: mined token {t!r} "
                                 f"lasts more than "
                                 f"{muzero.TOKEN_MINED_MAX_FRAMES} frames")
        if max_mined is not None and n_tail > max_mined:
            raise SystemExit(f"student vocabulary: {n_tail} mined tokens (at "
                             f"most {max_mined} per swap)")
        if len(vocab) > len(master):
            raise SystemExit(f"student vocabulary: {len(vocab)} slots, past "
                             f"the master's {len(master)}")
    if any(muzero.token_is_mined(t) for t in vocab):
        inputs = [muzero.token_inputs(t) for t in vocab]
        if len(set(inputs)) != len(inputs):
            raise SystemExit("student vocabulary: two tokens with the same "
                             "per-base-step inputs")
    return vocab


def declared_mined(doc, key="mined", path="vocab.json"):
    """E3 (E3-A4): the mined tokens a prep's vocab.json declares -- its
    "mined" entries' tokens (the prep's own mined tail) or, with key
    "master_mined", the master's mined tokens (the second swap) -- as
    canonical tokens, in order; () when the key is absent."""
    entries = doc.get(key)
    if entries is None:
        return ()
    if not isinstance(entries, list):
        raise SystemExit(f"{path}: '{key}' is not a list")
    toks = [e.get("token") if isinstance(e, dict) else e for e in entries]
    return _canon_or_exit(toks, f"{path}: '{key}'")


def load_student_doc(path, master_vocab):
    """--token-vocab: the vocab.json experiments/distill.py wrote (its
    "vocab" list), validated against the arm's master vocabulary. Returns
    (vocab, sha256 of the file, the parsed document) -- one read.

    E3: the declared mined tail ("mined") is validated with it (at most
    TOKEN_MINED_PER_SWAP). Against the arm's STATIC vocabulary, a student of
    a master holding mined tokens ("master_mined", E3-A8) may also keep some
    of the master's: its tail is then every mined token it holds, each
    declared in "mined" or "master_mined" -- student_arm checks it exactly
    against the master's own vocabulary."""
    import hashlib
    with open(path, "rb") as f:
        data = f.read()
    try:
        doc = json.loads(data.decode("utf-8"))
    except ValueError:
        doc = None
    if not isinstance(doc, dict) or "vocab" not in doc:
        raise SystemExit(f"{path}: no 'vocab' list (not a distill.py "
                         f"vocab.json)")
    mined = declared_mined(doc, "mined", path)
    master_mined = declared_mined(doc, "master_mined", path)
    if master_mined:
        vocab = _canon_or_exit(doc["vocab"])
        tail = tuple(t for t in vocab if muzero.token_is_mined(t))
        stray = [t for t in tail if t not in mined and t not in master_mined]
        if stray:
            raise SystemExit(f"{path}: mined tokens declared neither in "
                             f"'mined' nor in 'master_mined': {stray}")
        vocab = validate_student_vocab(vocab, master_vocab, mined=tail)
    else:
        vocab = validate_student_vocab(
            doc["vocab"], master_vocab, mined=mined,
            max_mined=muzero.TOKEN_MINED_PER_SWAP)
    return vocab, hashlib.sha256(data).hexdigest(), doc


def student_ceiling(doc, path, steps=1):
    """Test 2: the student's constant reward ceiling the prep chose
    (vocab.json's "student_ceiling": the halved-clip rule) -- a finite
    positive number that fits the reward supports (E3-A9: for a record of
    `steps` base steps, support_steps of the student's vocabulary)."""
    c = doc.get("student_ceiling")
    if (isinstance(c, bool) or not isinstance(c, (int, float))
            or not math.isfinite(c) or c <= 0):
        raise SystemExit(f"{path}: no valid 'student_ceiling' ({c!r}) -- "
                         f"not a Test 2 prep's vocab.json")
    if not ceiling_fits_supports(float(c), steps):
        raise SystemExit(f"{path}: student_ceiling {c!r} lies past the "
                         f"reward supports"
                         + (f" for a record of {steps} base steps (the "
                            f"vocabulary's longest token)" if steps > 1
                            else ""))
    return float(c)


def student_master_vocab(spec, doc, path):
    """The vocabulary a student's master played, when that master is itself
    a P3 student (generation 2, Kai 2026-10-27: DS2 ran a student
    vocabulary of TK's 36): the prep's recorded master_vocab, which must be
    a student vocabulary of the arm's static TOKEN_VOCAB -- with, E3-A8, the
    master's own mined tokens as its declared tail (vocab.json's
    "master_mined"; no per-swap cap: they may come from several swaps).
    None when the master is not a student (G0 played the static
    vocabulary)."""
    master_arm = spec.get("_master_arm")
    if master_arm is None or ARMS.get(master_arm, {}).get("_student") is None:
        return None
    if "master_vocab" not in doc:
        raise SystemExit(f"{path}: no 'master_vocab' (a prep of a student "
                         f"master records the vocabulary it played)")
    return validate_student_vocab(doc["master_vocab"], spec["TOKEN_VOCAB"],
                                  mined=declared_mined(doc, "master_mined",
                                                       path))


def student_arm(arm, path):
    """Apply --token-vocab to a student arm: ARMS[arm] becomes a copy with
    the student TOKEN_VOCAB and the file's sha256 (`_token_vocab_sha256`),
    so arm_config, the Ray actors' overrides and every weights file's spec
    carry the student's vocabulary -- and, for an arm with
    `_ceiling_from_prep` (Test 2's generation-1 students), the prep's
    student_ceiling as its constant ceiling (REWARD_GROWTH_START =
    REWARD_GROWTH_CEILING; `_student_ceiling`), from a prep of the arm's
    `_master_arm` at its `_master_iterations` (audit r1 A-L3: FS2 checks
    its master here, as DS2's set header does). When that master is itself
    a student (generation 2), the vocabulary must be a student vocabulary
    of the master's (the prep's master_vocab, kept as `_master_vocab`).
    An arm with `_discount_from_prep` (Test B) takes the prep's
    student_discount, whose rule must be its master arm's (Test B2's
    "keep": the master's own discount). Refuses a non-student arm.

    E3 (E3-A4, E3-A8): a vocab.json declaring MINED tokens ("mined": the
    prep's appended tail) gives the spec `_mined_tokens` (the tail, checked
    exactly against the master's vocabulary, at most TOKEN_MINED_PER_SWAP)
    and, when its master held mined tokens ("master_mined"),
    `_master_mined` beside `_master_vocab`; the ceiling's support check
    reads the vocabulary's longest token (E3-A9). A vocab.json without them
    applies exactly as before. An arm with `_mine` (E3's M1 / M2) takes
    only a mined prep's vocab.json (its "mine": distill.py --mine)."""
    spec = ARMS[arm]
    if spec.get("_student") not in ("distill", "fresh"):
        raise SystemExit(f"--token-vocab applies to the curriculum P3 student "
                         f"arms only; '{arm}' is not one")
    vocab, sha, doc = load_student_doc(path, spec["TOKEN_VOCAB"])
    if spec.get("_mine") and doc.get("mine") is not True:
        raise SystemExit(f"{path}: not a mined prep (distill.py --mine), "
                         f"but '{arm}' mines")
    if spec.get("_mine_q") and doc.get("mine_rules") != {
            "holds_out": True, "played_paths": True}:
        raise SystemExit(f"{path}: not a Test Q prep (distill.py --mine-q), "
                         f"but '{arm}' mines under Q-A1 and Q-A2")
    cap = spec.get("_ceiling_cap")
    if cap is not None and doc.get("ceiling_cap") != cap:
        raise SystemExit(f"{path}: its ceiling_cap "
                         f"{doc.get('ceiling_cap')!r} is not '{arm}''s "
                         f"{cap!r} (Q-A5)")
    mined = declared_mined(doc, "mined", path)
    master_mined = declared_mined(doc, "master_mined", path)
    master_vocab = student_master_vocab(spec, doc, path)
    if master_vocab is not None:
        vocab = validate_student_vocab(
            vocab, master_vocab, mined=mined,
            max_mined=muzero.TOKEN_MINED_PER_SWAP if mined else None)
    elif master_mined:
        raise SystemExit(f"{path}: 'master_mined', though this student's "
                         f"master {spec.get('_master_arm')!r} is no student")
    new = {**spec, "TOKEN_VOCAB": vocab, "_token_vocab_sha256": sha,
           "_vocab_set_sha256": doc.get("set_sha256")}
    if master_vocab is not None:
        new["_master_vocab"] = master_vocab
    if mined:
        new["_mined_tokens"] = mined
    if master_mined:
        new["_master_mined"] = master_mined
    if spec.get("_ceiling_from_prep"):
        for key in ("master_arm", "master_iterations"):
            want = spec.get(f"_{key}")
            if (key == "master_iterations" and want is None
                    and spec.get("_env_stop")):
                # Test B: an env-stopped master ends at a run-dependent
                # iteration (the prep's master_checks pins it against the
                # snapshot, and set_sha256 binds this file to that prep)
                continue
            if doc.get(key) != want:
                raise SystemExit(f"{path}: a prep of {key} "
                                 f"{doc.get(key)!r}, not this student's "
                                 f"{want!r}")
        c = student_ceiling(doc, path, support_steps(vocab))
        if cap is not None and c > cap:
            raise SystemExit(f"{path}: student_ceiling {c!r} is past "
                             f"'{arm}''s cap {cap!r} (Q-A5)")
        new.update({"REWARD_GROWTH_START": c, "REWARD_GROWTH_CEILING": c,
                    "_student_ceiling": c})
    if spec.get("_discount_from_prep"):
        # Test B: the prep's per-generation discount (the gap 1 - g
        # halved at each swap, capped at 0.997; Test B2: the master's)
        d = student_discount(doc, path)
        # Test B2 (audit r1a LOW-3): the prep ran under this student's
        # master arm's rule, and "keep" kept that master's discount (the
        # master arm's static DISCOUNT: under "keep", every generation's)
        import distill
        if spec.get("_master_arm") not in ARMS:
            raise SystemExit(f"{path}: this student's master arm "
                             f"{spec.get('_master_arm')!r} is unknown")
        master = ARMS[spec["_master_arm"]]
        rule = distill.discount_rule(master)
        if doc.get("discount_rule", "halve") != rule:
            raise SystemExit(f"{path}: a prep under the discount rule "
                             f"{doc.get('discount_rule', 'halve')!r}, not "
                             f"this student's master's {rule!r}")
        if rule == "keep" and not (
                d == doc.get("master_discount") == master.get("DISCOUNT")):
            raise SystemExit(f"{path}: student_discount {d!r} is not the "
                             f"kept master discount (vocab.json's "
                             f"{doc.get('master_discount')!r}, the master "
                             f"arm's {master.get('DISCOUNT')!r})")
        new.update({"DISCOUNT": d, "_student_discount": d})
    ARMS[arm] = new
    return ARMS[arm]


def student_discount(doc, path):
    """Test B: the student's DISCOUNT the prep chose (vocab.json's
    "student_discount": the master's gap 1 - g halved, capped at
    STUDENT_DISCOUNT_CAP; Test B2: the master's own) -- a float strictly
    between 0 and 1, at most the cap."""
    d = doc.get("student_discount")
    if (isinstance(d, bool) or not isinstance(d, (int, float))
            or not math.isfinite(d) or not 0 < d < 1
            or d > STUDENT_DISCOUNT_CAP):
        raise SystemExit(f"{path}: no valid 'student_discount' ({d!r}) -- "
                         f"not a Test B prep's vocab.json")
    return float(d)


# Test B: BBF's final discount (per base step), the per-swap rule's cap
STUDENT_DISCOUNT_CAP = 0.997


def student_gate(spec, distill_set, snapshot_weights):
    """The P3 student arms' inputs, checked at startup (before any
    --overwrite deletion): a student arm needs its --token-vocab applied
    (student_arm); DS needs --distill-set, whose vocabulary is this run's,
    whose master vocabulary is the arm's parent's and whose bytes are the
    ones vocab.json was written with (its set_sha256: the same prep; audit
    r1 A-NIT3), the master's end-state ceiling (START == CEILING) and
    --snapshot-weights (row 0 and the offline curve are snapshots; r1
    A-L2); FS and every other arm refuse a distillation set."""
    kind = spec.get("_student")
    # Test 2: the schedule offset is the distilled student's offline phase,
    # nothing else's
    offset = muzero.SCHEDULE_OFFSET_STEPS
    if offset and (kind != "distill"
                   or offset != spec.get("_distill_offline_steps")):
        raise SystemExit(f"SCHEDULE_OFFSET_STEPS {offset!r} belongs to a "
                         f"distilled student, equal to its "
                         f"_distill_offline_steps")
    if kind is None:
        if distill_set:
            raise SystemExit("--distill-set applies to the -student-distill "
                             "arm only")
        return
    if "_token_vocab_sha256" not in spec:
        raise SystemExit("a curriculum P3 student arm needs --token-vocab "
                         "(the student vocabulary distill.py wrote)")
    if spec.get("_ceiling_from_prep"):
        # Test 2: the prep's student_ceiling, held for the whole run
        c = spec.get("_student_ceiling")
        if (c is None or muzero.REWARD_GROWTH_START != c
                or muzero.REWARD_GROWTH_CEILING != c):
            raise SystemExit(f"a generation-1 student runs at the prep's "
                             f"constant student_ceiling ({c!r}); got "
                             f"{muzero.REWARD_GROWTH_START!r} -> "
                             f"{muzero.REWARD_GROWTH_CEILING!r}")
    if spec.get("_discount_from_prep"):
        # Test B: the prep's discount, held for the whole run
        d = spec.get("_student_discount")
        if d is None or muzero.DISCOUNT != d:
            raise SystemExit(f"a Test B / B2 student runs at the prep's "
                             f"student_discount ({d!r}); got "
                             f"{muzero.DISCOUNT!r}")
    if kind == "fresh":
        if distill_set:
            raise SystemExit("the -student-fresh arm takes no --distill-set")
        return
    if not distill_set:
        raise SystemExit("the -student-distill arm needs --distill-set")
    if not snapshot_weights:
        raise SystemExit("the -student-distill arm needs --snapshot-weights "
                         "(row 0 and the offline curve are its snapshots)")
    import distill
    header = distill.read_header(distill_set)
    # (E3: canonical tokens on both sides -- a mined token's JSON lists)
    if _canon_or_exit(header["vocab"], f"{distill_set}: its vocab") != \
            muzero.canon_vocab(muzero.TOKEN_VOCAB):
        raise SystemExit(f"{distill_set}: its student vocabulary differs "
                         f"from --token-vocab's")
    parent = ARMS[spec.get("_vocab_parent_arm",
                           "effzero-repro-faithful-prioract-noredo-temp05-"
                           "bnrecal-mspacman-tokens-rgrow")]["TOKEN_VOCAB"]
    # generation 2: a student master played its own student vocabulary
    # (student_arm's `_master_vocab`, from the same prep's vocab.json)
    want = spec.get("_master_vocab")
    if _canon_or_exit(header["master_vocab"],
                      f"{distill_set}: its master_vocab") != \
            muzero.canon_vocab(want if want is not None else parent):
        raise SystemExit(f"{distill_set}: recorded from a master vocabulary "
                         f"other than " + ("the student master's (its prep's "
                                           "master_vocab)" if want is not None
                                           else "the "
                                           + spec.get("_vocab_parent_arm",
                                                      "TK") + " arm's"))
    want = spec.get("_vocab_set_sha256")
    if not want or file_sha256(distill_set) != want:
        raise SystemExit(f"{distill_set}: not the set --token-vocab's "
                         f"vocab.json was written with (set_sha256)")
    master_arm = spec.get("_master_arm")
    if master_arm is not None and header.get("master_arm") != master_arm:
        raise SystemExit(f"{distill_set}: recorded from {header.get('master_arm')!r}"
                         f", not this student's master arm {master_arm!r}")
    if (spec.get("_ceiling_from_prep")
            and header.get("student_ceiling") != spec["_student_ceiling"]):
        raise SystemExit(f"{distill_set}: its student_ceiling "
                         f"{header.get('student_ceiling')!r} is not "
                         f"--token-vocab's {spec['_student_ceiling']!r}")
    if (spec.get("_discount_from_prep")
            and header.get("student_discount") != spec["_student_discount"]):
        raise SystemExit(f"{distill_set}: its student_discount "
                         f"{header.get('student_discount')!r} is not "
                         f"--token-vocab's {spec['_student_discount']!r}")
    if spec.get("_env_stop") and header.get("source") != "own-play log":
        # Test B: the swap costs no env steps -- never a fresh recording
        # (audit r1 A-L4)
        raise SystemExit(f"{distill_set}: a Test B student distils its "
                         f"master's own-play log, not a "
                         f"{header.get('source')!r} set")
    if muzero.REWARD_GROWTH_START != muzero.REWARD_GROWTH_CEILING:
        raise SystemExit("the distilled student runs at the master's end-state "
                         "ceiling (REWARD_GROWTH_START == CEILING)")
    steps = spec.get("_distill_offline_steps")
    every = spec.get("_distill_snapshot_every")
    if not (isinstance(steps, int) and isinstance(every, int)
            and 0 < every <= steps and steps % every == 0):
        raise SystemExit(f"_distill_offline_steps {steps!r} must be a positive "
                         f"multiple of _distill_snapshot_every {every!r}")


def student_expect(spec, distill_set):
    """The recipe keys a P3 student adds to run_expect (a resume under
    another vocabulary or distillation set is a different experiment):
    None for every other arm."""
    if spec.get("_student") is None:
        return None
    # (token_json: [a, f] as ever; a mined token as nested lists)
    out = {"token_vocab": [muzero.token_json(t) for t in muzero.TOKEN_VOCAB]}
    if spec.get("_ceiling_from_prep"):
        out["reward_ceiling"] = spec.get("_student_ceiling")
    if spec.get("_discount_from_prep"):
        # Test B: the prep's discount is the recipe too
        out["discount"] = spec.get("_student_discount")
    if distill_set:
        out["distill_set_sha256"] = file_sha256(distill_set)
    return out


def distill_prefill(run, lives):
    """DS's first replay: every re-tokenized master life (5-field records
    carrying the master-search policy and root value) through paper_targets
    -- the arm's own record builder (raw rewards clipped at the ceiling,
    age-0 priorities in frame time under the student's vocabulary) -- born
    at step SCHEDULE_OFFSET_STEPS (0; Test 2's DS2: its first online step),
    then the usual FIFO eviction to the caps (the recording
    overshoots the transition cap by up to a game, so its oldest lives go;
    audit r1 A-L1). The token / target diagnostics paper_targets
    accumulates describe the MASTER's play and are cleared, so row 5
    describes the student's. Returns (lives kept, lives evicted, base steps
    evicted)."""
    # Born at SCHEDULE_OFFSET_STEPS (0 but for Test 2's DS2): the 4.3
    # age-shrunk horizon is a within-run schedule keyed to the online steps
    # too, so the prefill ages from the first online step (audit r1 A-M1);
    # its age is max(0, ...), 0 through the offline phase.
    for life in lives:
        run.replay.append(run.paper_targets(
            life["traj"], life["terminated"],
            born_step=muzero.SCHEDULE_OFFSET_STEPS))
    before = (len(run.replay), sum(replay_units(t) for t in run.replay))
    evict_replay(run.replay)
    run._tok_window = None
    run._tok_totals = None
    run._fresh_target_H = None
    after = sum(replay_units(t) for t in run.replay)
    return len(run.replay), before[0] - len(run.replay), before[1] - after


def distill_acting_states(run):
    """The representation / prediction acting states a snapshot carries
    under BN_RECAL_ACTING -- what RayPipeline.snapshot_acting_states would
    ship (BN statistics recalibrated on the newest replay frames); None
    without the stabilizer."""
    if not muzero.BN_RECAL_ACTING:
        return None
    frames = bn_calibration_frames(run.replay, muzero.BN_RECAL_FRAMES)
    if frames is None:
        return None
    return bn_recalibrated_states(run.trainer, frames)


def distill_offline(run, arm, seed, spec, snapshot_weights):
    """DS's offline phase, before iteration 1 (no collection, no refresh):
    _distill_offline_steps gradient steps through train_block on the pre-
    filled replay, in chunks of _distill_snapshot_every; under
    --snapshot-weights a snapshot after each chunk -- the last is ROW 0
    (weights-iter000), the others the offline curve under
    RESULTS/<run>-offline (iteration = the chunk's index: weights-iter001 =
    after the first chunk), each with its acting states. The step
    diagnostics window restarts after it."""
    steps = spec["_distill_offline_steps"]
    every = spec["_distill_snapshot_every"]
    run.searcher.network_manager.training_steps = run.trainer.training_steps
    done = 0
    while done < steps:
        run.set_train_mode(True)
        train_block(run, every, it=0)
        run.set_train_mode(False)
        done += every
        if snapshot_weights:
            acting = distill_acting_states(run)
            if done == steps:
                save_weight_snapshot(run, arm, seed, spec, 0,
                                     acting_states=acting)
            else:
                save_weight_snapshot(
                    run, arm, seed, spec, done // every, acting_states=acting,
                    root=RESULTS / f"{RESULTS.name}-offline")
        print(f"  [{arm} s{seed}] offline step {done} / {steps}", flush=True)
    run.searcher.network_manager.training_steps = run.trainer.training_steps
    run._step_window = None


def env_actions(env_id):
    """The env's ALE action count."""
    import gymnasium as gym
    probe = gym.make(env_id)
    try:
        return int(probe.action_space.n)
    finally:
        probe.close()


def eval_summary(episodes):
    """Row fields from greedy eval games given as (return, decisions,
    terminated) triples (Run.play_game). eval_len_mean / eval_capped exist
    because greedy play in deterministic Pong can lock into a perpetual
    rally that only EPISODE_CAP ends -- such an episode's return is a
    partial score, and a row where eval_capped > 0 must be read with that
    in mind (2026-09-10; the fields are additive, older rows simply lack
    them). Mechanism: eval_capped counts games whose last life returned
    terminated=False, which under the paper stack means the per-life cap
    fired -- a gym truncation and a game over both read as terminated=True
    there. On Pong-v4 (no lives; ALE's 108000-frame limit is ~27000
    decisions, never before a 3000 cap) it is exactly the cap-truncation
    count. A multi-life game (Ms. Pac-Man) is played to its real game over
    by play_game, so its return is the GAME score, its length the game's
    decisions, and the cap applies per LIFE; an ALE 108000-frame
    truncation (~27000 decisions) would read as terminated, not capped."""
    returns = [float(r) for r, _, _ in episodes]
    return {"return_mean": float(np.mean(returns)),
            "return_max": float(np.max(returns)),
            "eval_len_mean": float(np.mean([s for _, s, _ in episodes])),
            "eval_capped": int(sum(1 for _, _, t in episodes if not t))}


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

        self.env = build_env(ENV_NAME, FRAMESKIP)
        # Seed the env stream too. Without this the module docstring's
        # paired-seed claim was false: gymnasium draws np_random from OS
        # entropy on an unseeded reset, so no run was reproducible even
        # against itself.
        self.env.reset(seed=seed)
        self.replay = []
        # Cumulative TRAINING env interaction (the Atari-100k x-axis); greedy
        # eval episodes are protocol-exempt and not counted.
        self.env_steps = 0
        # Test B: an OwnPlayLog under OWN_PLAY_LOG (run_one sets it)
        self.own_play = None
        # Test C (RESET_EVERY_STEPS): the resets done, one dict each
        # (index, iteration, training_steps, env_steps), the actor's hold
        # after the latest ({"until": step, "from": the reset's step,
        # "held": the acting states it keeps}) and the steps after a reset
        # at which the held states were shipped (held_ships, distinct); all
        # checkpointed
        self.reset_log = []
        self.hold = None
        self.held_ships = []
        # Test C2 (RAMP_STEPS): [iteration, steps since the last reset, n,
        # gamma] at the first gradient step of each iteration since the last
        # row (ramp_step); emitted and restarted by evaluate_diagnostics,
        # checkpointed. ramp_windows: the steps since the last row in and
        # out of the ramp, with their value loss and target sums
        # (note_ramp_window; None until the first), likewise
        self.ramp_trace = []
        self.ramp_windows = None

    def set_train_mode(self, training):
        for net in (self.trainer.representation, self.trainer.dynamics,
                    self.trainer.prediction, self.trainer.value_prefix,
                    self.trainer.projector):
            net.train(training)

    def uniform_windows(self, rng, n):
        """(obs, actions) of `n` unroll windows drawn uniformly from the
        replay by `rng` (a numpy Generator): each a root observation (float
        in [0, 1]) and the next UNROLL_STEPS stored actions, on the
        trainer's device; None when no trajectory holds a whole window.
        ReDo's sample, and Test C's dormant-unit read's."""
        K = muzero.UNROLL_STEPS
        # Valid window starts: K real actions after the root (the final
        # record's action is a placeholder, never a real one).
        starts = [(ti, len(t) - 1 - K) for ti, t in enumerate(self.replay)
                  if len(t) - 1 - K >= 0]
        if not starts:
            return None
        sizes = np.array([m + 1 for _, m in starts], dtype=np.int64)
        picks = rng.choice(int(sizes.sum()), size=n)
        bounds = np.cumsum(sizes)
        obs, acts = [], []
        for p in picks:
            k = int(np.searchsorted(bounds, p, side="right"))
            ti = starts[k][0]
            i = int(p - (bounds[k] - sizes[k]))
            traj = self.replay[ti]
            obs.append(traj[i]["obs"])
            acts.append([traj[i + j]["action"] for j in range(K)])
        dev = self.trainer.device
        obs = torch.tensor(np.stack(obs), dtype=torch.float32, device=dev) / 255.0
        acts = torch.tensor(np.asarray(acts), dtype=torch.long, device=dev)
        return obs, acts

    def dormant_stats(self):
        """Test C's plasticity read (DORMANT_DIAGNOSTIC), for a row: the
        fraction of ReDo's units (redo.recyclable_units over all five nets)
        whose normalized activation score (redo.dormancy_stats, on train-mode
        copies: the live nets and their statistics untouched) is <= 0.025
        and <= 0.1, per net and overall, on a uniform replay sample of
        REDO_BATCH windows from a generator keyed on (seed, training step)
        -- the global numpy / torch streams are never drawn; and the
        fraction of the representation's first-layer filters whose norm is
        below 10% of the median filter norm of a fresh draw of that layer
        (a dedicated torch generator). Nothing is recycled."""
        import redo
        steps = self.trainer.training_steps
        rng = np.random.default_rng((self.seed, steps, 0xD0A7))
        out = {}
        windows = self.uniform_windows(rng, int(muzero.REDO_BATCH))
        if windows is not None:
            obs, acts = windows
            nets = {n: getattr(self.trainer, n) for n in redo.ALL_NETS}
            units = redo.recyclable_units(nets, include=redo.ALL_NETS)
            scores = redo.dormancy_scores(nets, obs, acts, units)
            for tau, key in ((0.025, "dormant_025"), (0.1, "dormant_10")):
                per, all_n, all_d = {}, 0, 0
                for net in redo.ALL_NETS:
                    s = [scores[u] for u in units
                         if u.net == net and u in scores]
                    n = sum(x.numel() for x in s)
                    d = sum(int((x <= tau).sum()) for x in s)
                    per[net] = round(d / n, 4) if n else None
                    all_n, all_d = all_n + n, all_d + d
                out[key] = per
                out[key + "_all"] = round(all_d / all_n, 4) if all_n else None
        conv = self.trainer.representation.conv1.weight.detach()
        ref = getattr(self, "_filter_ref", None)
        if ref is None:
            fresh = torch.empty_like(conv, device="cpu")
            gen = torch.Generator().manual_seed(
                int(np.random.default_rng((self.seed, 0xF11)).integers(2**62)))
            # Conv2d's default init (kaiming_uniform_ at a = sqrt(5)):
            # U(-b, b), b = 1 / sqrt(fan_in)
            bound = 1.0 / math.sqrt(conv[0].numel())
            fresh.uniform_(-bound, bound, generator=gen)
            ref = float(fresh.flatten(1).norm(dim=1).median())
            self._filter_ref = ref
        norms = conv.flatten(1).norm(dim=1)
        out["conv1_weak_frac"] = round(float((norms < 0.1 * ref).float()
                                             .mean()), 4)
        return out

    def redo_step(self):
        """One ReDo pass (redo.py): score every recyclable unit on a fresh
        UNIFORM replay sample of REDO_BATCH unroll windows (root obs + the
        next UNROLL_STEPS stored actions; no augmentation, no PER) and
        recycle the tau-dormant ones in the live nets. The sample and the
        reinit draws come from generators keyed on (seed, training step), so
        a pass is reproducible across a resume and never touches the global
        numpy/torch streams. Returns units recycled."""
        import redo
        steps = self.trainer.training_steps
        rng = np.random.default_rng((self.seed, steps, 0x7ED0))
        windows = self.uniform_windows(rng, int(muzero.REDO_BATCH))
        if windows is None:
            return 0
        obs, acts = windows
        nets = {n: getattr(self.trainer, n) for n in redo.ALL_NETS}
        units = redo.recyclable_units(nets, include=tuple(muzero.REDO_NETS))
        scores, raw, inputs = redo.dormancy_stats(nets, obs, acts, units)
        # Re-dormancy of LAST pass's recycled units (the official code's
        # intersected_dead_neurons_with_last_reset): high = recycled units
        # die again, i.e. ReDo is churning rather than restoring.
        last = getattr(self, "_redo_last_plan", None) or {}
        tau = float(muzero.REDO_TAU)
        again = sum(int((scores[u][idx] <= tau).sum())
                    for u, idx in last.items() if u in scores)
        was = sum(len(idx) for u, idx in last.items() if u in scores)
        gen = torch.Generator().manual_seed(
            int(np.random.default_rng((self.seed, steps, 0x7ED1)).integers(2**62)))
        opts = [getattr(self.trainer, n) for n in OPT_NAMES]
        plan = redo.recycle(nets, opts, scores, tau, gen, raw=raw,
                            inputs=inputs)
        self._redo_last_plan = plan
        recycled = sum(len(v) for v in plan.values())
        total = sum(s.numel() for s in scores.values())
        by_net = {n: sum(len(v) for u, v in plan.items() if u.net == n)
                  for n in muzero.REDO_NETS}
        prev = getattr(self, "_redo_stats", None) or {}
        self._redo_stats = {
            "redo_recycled_total": prev.get("redo_recycled_total", 0) + recycled,
            # Since the previous eval row (zeroed by evaluate_diagnostics):
            # a row spans ~10 passes at eval-every 5, so _last alone would
            # show one pass in ten (round-1 audit LOW-1).
            "redo_recycled_window": prev.get("redo_recycled_window", 0) + recycled,
            "redo_recycled_last": recycled,
            "redo_last_by_net": by_net,
            "redo_by_net_window": {
                n: prev.get("redo_by_net_window", {}).get(n, 0) + by_net[n]
                for n in by_net},
            "redo_redormant_frac_last": (round(again / was, 4) if was
                                         else None),
            "redo_units": total,
            "redo_dormant_frac_last": round(recycled / max(1, total), 4)}
        return recycled

    # -- self-play ---------------------------------------------------------
    def play_episode(self, greedy=False, cap=None):
        cap = EPISODE_CAP if cap is None else cap
        obs, _ = self.env.reset()
        stack = initial_frame_stack(preprocess_frame(obs, IMAGE_RESHAPE))
        traj, prev_reward, total, done, steps = [], 0.0, 0.0, False, 0
        # EVAL_PRIOR_ONLY: a greedy episode never searches (per step or
        # tail) and records plain 3-field steps, as a non-search arm's.
        prior_eval = greedy and muzero.EVAL_PRIOR_ONLY
        # TOKEN_VOCAB (curriculum P2a): each decision plays a token; `steps`
        # (the cap, env_steps, eval lengths) then counts BASE steps --
        # frames / TOKEN_BASE_FRAMES, the plain arms' decisions -- and the
        # trajectory's length counts decisions.
        tokens = muzero.TOKEN_VOCAB is not None
        # REWARD_GROWTH: records carry the RAW per-base-step rewards as a
        # tuple (none before the first decision); paper_targets clips them
        # on the learner at the replay's current ceiling.
        growth = muzero.REWARD_GROWTH
        if growth:
            prev_reward = ()
        while not done and steps < cap:
            stack = push_frame(stack, preprocess_frame(obs, IMAGE_RESHAPE))
            if muzero.USE_SEARCH_ACTING and not prior_eval and (
                    not greedy or muzero.PAPER_TARGETS):
                # Search acting: one search per step during self-play. Greedy
                # EVAL searches only in paper mode (the papers evaluate with
                # search); non-paper search-acting arms keep their documented
                # meaning -- eval measures the deployable RAW-POLICY agent.
                # Round-9 finding: the paper-mode commit had made eval search
                # unconditional, silently inverting pong-act-search's _doc
                # against its own archived rows.
                # ACTOR_SEARCH_SIMS: the self-play search's budget (None =
                # ACT_SEARCH_SIMS); greedy searches keep the default.
                # ACTOR_GUMBEL_SEARCH: these searches run the Gumbel engine.
                action, search_pol, search_val = asyncio.run(
                    self.searcher.act_search(
                        stack, greedy=greedy,
                        num_simulations=(None if greedy
                                         else muzero.ACTOR_SEARCH_SIMS),
                        force_gumbel=muzero.ACTOR_GUMBEL_SEARCH))
            else:
                search_pol = search_val = None
                with torch.no_grad():
                    policy, _ = self.searcher.network_manager.initial_inference(stack)
                if greedy:
                    action = int(policy.argmax())
                else:
                    # Acting-side exploration floor, mirroring gameplay_process.
                    # Read per step so an arm override applies; identity at 1.0.
                    # PRIOR_ACT_STAGED_TEMP: the search's staged schedule,
                    # keyed on the same counter act_search reads.
                    steps_at = getattr(self.searcher.network_manager,
                                       "training_steps", 0)
                    t = muzero.prior_acting_temperature(steps_at)
                    if t != 1.0:
                        policy = policy ** (1.0 / t)
                        policy = policy / policy.sum()
                    b = muzero.token_long_bias(steps_at)
                    if b:
                        # The early long-token bias (TOKEN_LONG_BIAS):
                        # weights x (frames / base) ** b.
                        policy = policy * token_bias_weights(b)
                        policy = policy / policy.sum()
                    action = int(np.random.choice(ACTIONS, p=policy))
            after = stack
            inner = []
            if tokens:
                # play_token pushes the token's inner frames into its own
                # copy: the record below keeps the stack the decision was
                # made on (audit r1 A-1), the next decision continues from
                # `after`.
                obs, reward, term, trunc, after, n_base = play_token(
                    self.env, action, stack, inner)
            else:
                obs, reward, term, trunc, _ = self.env.step(action)
                n_base = 1
                inner.append(float(reward))
            done = term or trunc
            # 5-field records exist ONLY for the paper data path. Recording
            # them for every search-acting arm broke the pre-existing
            # pong-act-search arm, whose non-paper reanalysis unpacks 6-field
            # eval_obs tuples (round-5 finding, reproduced ValueError).
            # SEARCH_ACT_BIRTH_TARGETS: the search only chose the action;
            # BIRTH_SEARCH makes the targets from the plain record.
            if (search_pol is None or not muzero.PAPER_TARGETS
                    or muzero.SEARCH_ACT_BIRTH_TARGETS):
                traj.append((stack, action, prev_reward))
            else:
                traj.append((stack, action, prev_reward, search_pol, search_val))
            stack = after
            prev_reward = (tuple(inner) if growth
                           else muzero.clip_reward(reward))
            total += float(reward)
            steps += n_base
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
        if (muzero.USE_SEARCH_ACTING and muzero.PAPER_TARGETS
                and not prior_eval and not muzero.SEARCH_ACT_BIRTH_TARGETS):
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

    def play_game(self, cap=None):
        """One greedy evaluation GAME: every life, until the real game over
        (Ms. Pac-Man, 2026-10-16). EpisodicLifeEnv ends play_episode at
        each life loss -- the TRAINING unit, as in EfficientZero -- so an
        eval made of single play_episode calls would score one life and
        start the next eval "game" mid-match. Here a game starts from a real
        reset and plays on through every life; returns (game return,
        decisions, terminated) for eval_summary. `cap` bounds each LIFE
        (play_episode's cap): a capped life ends the game (play_episode
        forces the next reset to be real) and reads terminated=False. A
        single-life game (Pong) is exactly one play_episode, unchanged.
        Callers that seed the game reset the env with that seed first
        (setting was_real_done themselves, so the reseed is a real reset);
        the game then starts from play_episode's own real reset -- the
        double reset every eval had before. The sequential harness shares
        its env with training, so there an eval game ends the training
        match in progress (the --ray eval actor has its own env)."""
        self.env.was_real_done = True
        total, steps = 0.0, 0
        while True:
            # [1:]: the life's trajectory is dropped at once, never held
            # while the next life plays.
            ret, n, terminated = self.play_episode(greedy=True, cap=cap)[1:]
            total += ret
            steps += n
            if not terminated or getattr(self.env, "was_real_done", True):
                return total, steps, terminated

    # -- reward-clip growth --------------------------------------------------
    def apply_reward_ceiling(self):
        """REWARD_GROWTH: re-clip every stored record at the ceiling for the
        trainer's current step (muzero.reward_ceiling), when it changed --
        called at the top of each iteration, before anything is collected,
        so value / value-prefix targets, insertion priorities, the
        refresh's chains and the diagnostics all read one ceiling at a
        time. Stored priorities are not recomputed (training writebacks
        and the refresh rotation update them). Not checkpointed: the first
        call after a start or resume re-clips the loaded buffer."""
        if not muzero.REWARD_GROWTH:
            return
        c = muzero.reward_ceiling(self.trainer.training_steps)
        if c == getattr(self, "_reward_ceiling", None):
            return
        self._reward_ceiling = c
        for traj in self.replay:
            for t in traj:
                t["reward"] = muzero.grown_reward(t["raw_reward"], c)

    def reward_ceiling_now(self):
        """The ceiling the replay is clipped at (applied on first use), for
        records entering it."""
        if getattr(self, "_reward_ceiling", None) is None:
            self.apply_reward_ceiling()
        return self._reward_ceiling

    # -- re-analysis -------------------------------------------------------
    def paper_targets(self, traj, terminated, born_step=None,
                      birth_nets=None, birth_prior_nets=None):
        """Paper data path: every step already carries its acting-search
        policy and root value. Store the target ingredients (values are
        computed at BATCH time under the age-adjusted horizon), the birth
        step for EfficientZero 4.3, and age-0 priorities |v - z|.

        born_step: explicit birth stamp for callers that collect a
        trajectory AFTER further training happened (--ray's pipeline
        passes the request-time count so the 4.3 age matches the
        sequential harness exactly); default is the live counter.

        BIRTH_SEARCH: policy-only acting records -- and, under
        SEARCH_ACT_BIRTH_TARGETS, search-acting records -- carry no search
        (3 fields); _birth_search supplies the per-step policy and root
        value the acting search would have recorded, before anything below
        reads them. birth_nets: the nets the ACTOR played with, when they are
        not the live ones (--ray's pipelined collect runs after the next
        training block; see RayPipeline.request). birth_prior_nets: the
        representation / prediction the actor SAMPLED from, when they differ
        from birth_nets (BN_RECAL_ACTING) -- for the behaviour diagnostics."""
        born = (self.trainer.training_steps if born_step is None
                else born_step)
        birth = muzero.BIRTH_SEARCH and len(traj[0]) == 3
        if birth:
            traj = self._birth_search(traj, terminated, nets=birth_nets,
                                      prior_nets=birth_prior_nets)
        if len(traj[0]) != 5:
            raise ValueError(
                "PAPER_TARGETS requires USE_SEARCH_ACTING (or BIRTH_SEARCH): "
                "trajectories must carry per-step search records (5 fields), "
                f"got {len(traj[0])}")
        out = []
        growth = muzero.REWARD_GROWTH
        ceiling = self.reward_ceiling_now() if growth else None
        for i, (obs, action, reward, pol, val) in enumerate(traj):
            # The terminal record of a TERMINATED episode carried no search
            # (uniform placeholder -> masked); a TRUNCATED tail's record is a
            # genuine greedy search and keeps its policy target.
            has_policy = (i < len(traj) - 1) or (not terminated)
            raw = None
            if growth:
                # REWARD_GROWTH: the actor recorded raw per-base-step
                # rewards; keep them and store the clip at the replay's
                # current ceiling. A float here is a record clipped at
                # record time -- refused rather than double-clipped.
                if not isinstance(reward, tuple):
                    raise ValueError(
                        "REWARD_GROWTH records carry raw reward tuples; got "
                        f"{type(reward).__name__} (an actor without the arm?)")
                raw, reward = reward, muzero.grown_reward(reward, ceiling)
            out.append({"obs": obs, "action": action, "reward": reward,
                        "target_policy": np.asarray(pol, dtype=np.float64),
                        "root_value": float(val),
                        "target_value": float(val),  # for non-paper readers
                        "has_policy_target": has_policy,
                        "born_step": born})
            if growth:
                out[-1]["raw_reward"] = raw
            if muzero.VALUE_TARGET_MODE == "mixed":
                # When root_value was computed (the actor's weights' step),
                # for SVE's sve_value_age diagnostic.
                out[-1]["root_value_step"] = born
        out[-1]["truncated"] = not terminated
        if birth and int(muzero.OPTION_LENGTH) > 1:
            # Test O stage 3: each ACTED record's root suggestion and PMAX
            # from this life's birth search (_birth_search); the closing
            # record carries neither (terminal: nothing reads them there).
            # The refresh never writes them (DEPARTURE 3).
            got = self._birth_options
            self._birth_options = None
            for i in range(len(out) - 1):
                out[i]["root_option"] = got["root_option"][i]
                out[i]["pmax"] = got["pmax"][i]
        z0s = []
        for idx, t in enumerate(out):
            z0 = muzero.paper_value_target(out, idx, muzero.TD_STEPS,
                                           muzero.DISCOUNT)
            t["priority"] = abs(t["root_value"] - z0) + muzero.PER_EPS
            z0s.append(z0)
        if muzero.TOKEN_VOCAB is not None:
            self._note_tokens(out, z0s)
        if birth and not muzero.USE_SEARCH_ACTING:
            # The behaviour diagnostics describe PRIOR sampling; a
            # search-acting actor (SEARCH_ACT_BIRTH_TARGETS) did not sample
            # the prior, so they are not recorded.
            self._note_behaviour(out[:-1], born)
            if muzero.OPTION_PRECHECK:
                self._note_precheck(out[:-1])
            if int(muzero.OPTION_LENGTH) > 1:
                # Test O stage 4a: the option row window (this life's birth
                # search counters, its acted positions, the probe's roots)
                self._note_options(out[:-1])
        pols = [t["target_policy"] for t in out if t["has_policy_target"]]
        # Unreachable-empty from every current caller (play_episode always
        # yields >= 1 policy-carrying record; cap 0 keeps its greedy tail
        # search), but a future zero-target caller must clear the
        # diagnostic, not silently repeat the previous episode's entropy.
        if pols:
            pols = np.stack(pols)
            self._fresh_target_H = float(
                (-pols * np.log(pols + 1e-12)).sum(axis=1).mean())
        else:
            self._fresh_target_H = None
        return out

    def _birth_search(self, traj, terminated, nets=None, prior_nets=None):
        """BIRTH_SEARCH: give a 3-field trajectory (policy-only acting, or a
        search-acting actor under SEARCH_ACT_BIRTH_TARGETS) the per-step
        (policy, root value) a full acting search would have recorded,
        returning 5-field records for paper_targets.

        Same search as act_search on the acting path: fresh-root PUCT,
        ACT_SEARCH_SIMS, root noise on every acted step, a GREEDY search on a
        truncated tail, uniform/0 on a terminated tail -- vectorized through
        muzero_vtree (parity-gated against the sequential engine) over the
        ONLINE nets in eval mode, the nets the actor would have searched
        with. Policies are raw visit fractions and values root q-values, the
        act_search contract. Accounting in self._birth_stats.

        nets: default the live trainer nets (sequential mode, where the
        actor IS the live nets). --ray passes the request-time snapshot the
        actor was synced with, so the targets never come from weights the
        actor did not play with (round-1 audit HIGH-1).

        prior_nets: the representation / prediction the actor SAMPLED from,
        when they differ from `nets` (BN_RECAL_ACTING: recalibrated BN
        statistics; the search keeps `nets`, as it runs the prediction head
        at depths > 0) -- only the logged behaviour prior uses them.

        OPTION_PRECHECK: also self._birth_pmax, each acted position's argmax
        through `nets` -- the learner's, the ones this search runs -- never
        prior_nets (Test O's m, as OF's PMAX).

        OPTION_LENGTH > 1 (Test O stage 3): the same search on the option
        engine (muzero_vtree_options.VectorizedOptionPUCT, a v1 subclass:
        the same arguments and returns) with option_edge = OPTION_EDGE (OF
        True; OH False: the same heads and root suggestions, no edge), on
        every acted position and a cut life's tail alike. Play is
        untouched (DEPARTURE 1). Also self._birth_options, per ACTED
        position, for paper_targets' records: "root_option" (the root's
        dominant option, an interned L-tuple, -1 padded; OptionZero's OP1)
        and "pmax" (the root's head-0 argmax through `nets`, the learner's:
        zero_actor.cpp 306's p_max_id_, the PMAX tag 164); and
        self._birth_option_counts, the life's search counters summed
        (option_search_counts; for the diagnostics stage, not logged). The
        nets' heads must be of OPTION_LENGTH, either way (nets without
        option heads under OPTION_LENGTH 1). OPTION_LENGTH 1: v1, the call
        above, unchanged."""
        import muzero_vtree
        t0 = time.time()
        if nets is None:
            nets = {"representation": self.trainer.representation,
                    "dynamics": self.trainer.dynamics,
                    "prediction": self.trainer.prediction,
                    "value_prefix": self.trainer.value_prefix}
        opt_L = int(muzero.OPTION_LENGTH)
        # Both ways (audit r1 of stage 3): the trainer keys on the nets'
        # option_length, so nets with heads under OPTION_LENGTH 1 (or none
        # under L > 1) must not run the other engine silently. An attribute
        # read only: no RNG drawn, nothing else touched.
        have = int(getattr(nets["prediction"], "option_length", 1))
        if have != opt_L:
            raise ValueError(
                f"OPTION_LENGTH {opt_L}: the birth search's nets carry "
                f"option heads of length {have}")
        if opt_L > 1:
            import muzero_vtree_options
            opt_kw = {"option_edge": bool(muzero.OPTION_EDGE),
                      "return_info": True}
        modes = {k: n.training for k, n in nets.items()}
        for n in nets.values():
            n.eval()
        try:
            if opt_L > 1:
                engine = muzero_vtree_options.VectorizedOptionPUCT(
                    nets, ACTIONS, self.trainer.device)
            else:
                engine = muzero_vtree.VectorizedPUCT(nets, ACTIONS,
                                                     self.trainer.device)
            n_acted = len(traj) - 1
            pols = [None] * len(traj)
            vals = [0.0] * len(traj)
            batch = max(1, int(muzero.VTREE_REFRESH_BATCH))
            if opt_L > 1:
                r_opt = np.full((n_acted, opt_L), -1, dtype=np.int64)
                r_len = np.zeros(n_acted, dtype=np.int64)
                r_pmax = np.zeros(n_acted, dtype=np.int64)
                counts = {}
            for w0 in range(0, n_acted, batch):
                idx = range(w0, min(w0 + batch, n_acted))
                obs = np.stack([traj[i][0] for i in idx])
                if opt_L > 1:
                    p, v, _, info = engine.search(
                        obs, num_simulations=muzero.ACT_SEARCH_SIMS,
                        c_puct=muzero.C_PUCT, discount=muzero.DISCOUNT,
                        root_noise=True, **opt_kw)
                    sl = slice(w0, w0 + len(idx))
                    r_opt[sl] = info["root_option"]
                    r_len[sl] = info["root_option_len"]
                    r_pmax[sl] = info["root_pmax"]
                    option_search_counts(counts, info)
                else:
                    p, v, _ = engine.search(
                        obs, num_simulations=muzero.ACT_SEARCH_SIMS,
                        c_puct=muzero.C_PUCT, discount=muzero.DISCOUNT,
                        root_noise=True)
                for i, pi, vi in zip(idx, p, v):
                    pols[i], vals[i] = np.asarray(pi, dtype=np.float64), float(vi)
            tail = len(traj) - 1
            if terminated:
                pols[tail], vals[tail] = np.ones(ACTIONS) / ACTIONS, 0.0
            else:
                if opt_L > 1:
                    p, v, _, info = engine.search(
                        np.stack([traj[tail][0]]),
                        num_simulations=muzero.ACT_SEARCH_SIMS,
                        c_puct=muzero.C_PUCT, discount=muzero.DISCOUNT,
                        root_noise=False, **opt_kw)
                    option_search_counts(counts, info)
                else:
                    p, v, _ = engine.search(
                        np.stack([traj[tail][0]]),
                        num_simulations=muzero.ACT_SEARCH_SIMS,
                        c_puct=muzero.C_PUCT, discount=muzero.DISCOUNT,
                        root_noise=False)
                pols[tail] = np.asarray(p[0], dtype=np.float64)
                vals[tail] = float(v[0])
            if opt_L > 1:
                import muzero_option_targets as mot
                self._birth_options = {
                    "root_option": [mot.intern_option(r_opt[i], r_len[i])
                                    for i in range(n_acted)],
                    "pmax": [int(m) for m in r_pmax]}
                self._birth_option_counts = counts
            # The prior the actor sampled from (same nets -- or prior_nets
            # under BN_RECAL_ACTING -- same /255 preprocessing as
            # initial_inference), for the behaviour diagnostics
            # paper_targets logs (round-2 design audit; bnrecal r1 M1).
            pn = prior_nets or nets
            priors, pmax = [], []
            with torch.no_grad():
                for w0 in range(0, n_acted, batch):
                    x = torch.tensor(
                        np.stack([traj[i][0] for i in
                                  range(w0, min(w0 + batch, n_acted))]),
                        dtype=torch.float32, device=self.trainer.device) / 255.0
                    logits, _, _ = pn["prediction"](pn["representation"](x))
                    priors.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
                    if muzero.OPTION_PRECHECK:
                        own, _, _ = nets["prediction"](nets["representation"](x))
                        pmax.append(own.float().argmax(dim=1).cpu().numpy())
            self._birth_priors = (np.concatenate(priors) if priors
                                  else np.zeros((0, ACTIONS)))
            if muzero.OPTION_PRECHECK:
                self._birth_pmax = (np.concatenate(pmax) if pmax
                                    else np.zeros(0, dtype=np.int64))
        finally:
            for k, n in nets.items():
                n.train(modes[k])
        stats = getattr(self, "_birth_stats", None) or {}
        searched = n_acted + (0 if terminated else 1)
        self._birth_stats = {
            "birth_positions_total": stats.get("birth_positions_total", 0)
            + searched,
            "birth_positions_window": stats.get("birth_positions_window", 0)
            + searched,
            "birth_positions_last": searched,
            "birth_search_s_last": round(time.time() - t0, 2)}
        return [(o, a, r, pols[i], vals[i])
                for i, (o, a, r) in enumerate(traj)]

    def _note_behaviour(self, acted, born):
        """Behaviour diagnostics for policy-only acting, accumulated per row
        (round-1 fidelity + round-2 design audits): how the prior-sampled
        actions relate to what search would have done, and the exploration
        actually applied. `acted` are the episode's acted records (their
        target_policy is the birth search's), `born` the actor's step
        count. Emitted and reset by _merge_row_stats."""
        priors = getattr(self, "_birth_priors", None)
        if priors is None or len(priors) != len(acted) or not acted:
            return
        T = float(muzero.prior_acting_temperature(born))
        tgt = np.stack([t["target_policy"] for t in acted])
        act = np.array([int(t["action"]) for t in acted])
        q = np.clip(priors.astype(np.float64), 1e-12, None)
        qT = q ** (1.0 / T)
        qT /= qT.sum(axis=1, keepdims=True)
        b = muzero.token_long_bias(born)
        if b:
            # The distribution actually sampled includes the long-token bias.
            qT = qT * token_bias_weights(b)
            qT /= qT.sum(axis=1, keepdims=True)
        p_act = tgt[np.arange(len(act)), act]
        w = getattr(self, "_beh_window", None) or {"n": 0, "sums": {},
                                                    "counts": [0] * ACTIONS}
        add = {
            "act_agree": float((tgt.argmax(1) == act).sum()),
            "act_prob": float(p_act.sum()),
            "prior_agree": float((q.argmax(1) == tgt.argmax(1)).sum()),
            "kl_target_prior": float(np.sum(
                tgt * (np.log(np.clip(tgt, 1e-12, None)) - np.log(q)))),
            "H_act": float(-(qT * np.log(np.clip(qT, 1e-12, None))).sum()),
            "prior_maxp": float(q.max(1).sum()),
            "rare_rate": float((p_act < 0.05).sum()),
            "T": T * len(act),               # step-weighted mean per row
        }
        for k, v in add.items():
            w["sums"][k] = w["sums"].get(k, 0.0) + v
        w["n"] += len(act)
        w["counts"] = [c + int(n) for c, n in zip(
            w["counts"], np.bincount(act, minlength=ACTIONS))]
        w["Ts"] = sorted(set(w.get("Ts", [])) | {T})
        self._beh_window = w

    def _note_precheck(self, acted):
        """Test O's O0 pre-check, accumulated per row like _beh_window: a
        life's option_precheck_counts, from its acted records (the birth
        targets, _note_behaviour's tgt) and the learner's argmax
        (self._birth_pmax, this life's _birth_search). Emitted and reset by
        _merge_row_stats. Not checkpointed: a resume loses the window's
        positions so far, which pc_positions shows."""
        if not acted:
            return
        tgt = np.stack([t["target_policy"] for t in acted])
        act = np.array([int(t["action"]) for t in acted])
        got = option_precheck_counts(tgt, act, self._birth_pmax)
        w = getattr(self, "_pc_window", None) or {}
        self._pc_window = {k: w.get(k, 0) + v for k, v in got.items()}

    def _option_window(self):
        """The current row's option window (option_window_empty at the run's
        L, moves and unroll when none is open)."""
        w = getattr(self, "_opt_window", None)
        if w is None:
            w = self._opt_window = option_window_empty(
                int(muzero.OPTION_LENGTH), ACTIONS, int(muzero.UNROLL_STEPS))
        return w

    def _option_probe_pool(self):
        """The current row's probe roots: {"row": rows emitted before this
        one, "keys" [k] float64, "obs" [k] stacks, "rng"}. THE RULE: every
        acted position of the row draws a key from a generator pinned to
        (seed, row, 0x0B5E), in arrival order, and the probe takes the
        OPTION_PROBE_ROOTS positions with the smallest keys -- a uniform
        sample without replacement, the same however the row's lives are
        chunked. The training RNG streams are never drawn."""
        pool = getattr(self, "_opt_probe", None)
        if pool is None:
            pool = self._opt_probe = {
                "row": 0, "keys": np.zeros(0), "obs": [],
                "rng": np.random.default_rng([int(self.seed), 0, 0x0B5E])}
        return pool

    def _note_options(self, acted):
        """Test O stage 4a, OPTION_LENGTH > 1: one life into the row's
        option window (Run._opt_window, emitted and restarted by
        _merge_option_stats, checkpointed): read (3)'s engagement counters
        from this life's LIVE birth search (_birth_option_counts: every
        acted root and a cut life's greedy tail; never the refresh's), and
        over its acted records (the birth targets, the move played, the
        root suggestion and PMAX as written at birth): DEPARTURE 7's hit
        rate (played == PMAX) and pi[PMAX], read (4)'s state-dependence
        counts (option_state_counts), the moves played and their frames
        (read (6)'s decisions per second), and the on/off probe's root keys
        (_option_probe_pool)."""
        import muzero_option_targets as mot
        w = self._option_window()
        option_window_add_search(
            w, getattr(self, "_birth_option_counts", None) or {})
        if not acted:
            return
        L = int(muzero.OPTION_LENGTH)
        act = [int(t["action"]) for t in acted]
        pm = [mot.record_pmax(t) for t in acted]
        w["acted"] += len(act)
        w["hit"] += int(sum(a == m for a, m in zip(act, pm)))
        w["pi_m"] += float(sum(float(t["target_policy"][m])
                               for t, m in zip(acted, pm)))
        w["frames"] += int(sum(muzero.token_frames(a) for a in act))
        w["play"] = [int(c) + int(n) for c, n in zip(
            w["play"], np.bincount(act, minlength=len(w["play"])))]
        got = option_state_counts(act, [mot.option_moves(t) for t in acted],
                                  L, len(w["play"]))
        for k in ("with_option", "nonrepeat"):
            w[k] += got[k]
        for k in ("s4_n", "s4_opt", "s4_rep", "s4_prev", "s4_first",
                  "root_len_hist"):
            w[k] = [a + b for a, b in zip(w[k], got[k])]
        w["s4_hist"] = [[a + b for a, b in zip(x, y)]
                        for x, y in zip(w["s4_hist"], got["s4_hist"])]
        pool = self._option_probe_pool()
        keys = np.concatenate([pool["keys"], pool["rng"].random(len(acted))])
        obs = pool["obs"] + [t["obs"] for t in acted]
        keep = np.argsort(keys, kind="stable")[:OPTION_PROBE_ROOTS]
        pool["keys"] = keys[keep]
        pool["obs"] = [obs[i] for i in keep]

    def _option_probe(self, obs, row):
        """Read (3)'s ON/OFF ROOT PROBE (descriptive), at row time: the
        probe roots `obs` searched twice by the option engine
        (muzero_vtree_options.VectorizedOptionPUCT, the birth search's
        subclass), the option edge on and off, on ONE pinned net -- the
        learner's nets, in eval mode, as the birth search runs them -- with
        the birth search's simulations, c_puct and discount, root noise
        from ONE draw shared by both searches (noise_override; Dirichlet at
        DIRICHLET_ALPHA from a generator pinned to (seed, row, 0x015E): the
        engine's own draw would take torch's global RNG) and tie_break
        "first" (no tie draw). So no training RNG stream is drawn, and every
        net's train / eval mode is restored. Returns option_probe_stats plus
        "seconds" (the probe's wall clock), or None without roots."""
        if not obs:
            return None
        import muzero_vtree_options
        t0 = time.perf_counter()
        nets = {k: getattr(self.trainer, k) for k in
                ("representation", "dynamics", "prediction", "value_prefix")}
        modes = {k: n.training for k, n in nets.items()}
        for n in nets.values():
            n.eval()
        try:
            x = np.stack(obs)
            T = len(x)
            noise = np.random.default_rng([int(self.seed), int(row), 0x015E]
                                          ).dirichlet(np.full(
                                              ACTIONS,
                                              float(muzero.DIRICHLET_ALPHA)),
                                              size=T)
            engine = muzero_vtree_options.VectorizedOptionPUCT(
                nets, ACTIONS, self.trainer.device)
            batch = max(1, int(muzero.VTREE_REFRESH_BATCH))
            got = {}
            for edge in (True, False):
                pols, ropt, rlen = [], [], []
                for w0 in range(0, T, batch):
                    sl = slice(w0, w0 + batch)
                    p, _, _, info = engine.search(
                        x[sl], num_simulations=muzero.ACT_SEARCH_SIMS,
                        c_puct=muzero.C_PUCT, discount=muzero.DISCOUNT,
                        root_noise=True, noise_override=noise[sl],
                        tie_break="first", option_edge=edge,
                        return_info=True)
                    pols.append(np.asarray(p, dtype=np.float64))
                    ropt.append(info["root_option"])
                    rlen.append(info["root_option_len"])
                got[edge] = (np.concatenate(pols), np.concatenate(ropt),
                             np.concatenate(rlen))
        finally:
            for k, n in nets.items():
                n.train(modes[k])
        (pon, ropt, rlen), (poff, _, _) = got[True], got[False]
        out = option_probe_stats(pon, poff, ropt, rlen)
        out["seconds"] = time.perf_counter() - t0
        return out

    def _merge_option_stats(self, errs):
        """Test O stage 4a (OPTION_LENGTH > 1): the row's option fields
        (option_row_fields: read (3)'s engagement counts and rates and its
        on/off root probe, run here on the row's probe roots; DEPARTURES 3
        and 7; read (4); the train-step means) and read (6)'s cheap missing
        items into `errs`; then the window and the probe roots restart (the
        next row's generator pinned to its own index). Read (6)'s items:
          opt_grad_uses_per_record = training steps x TRAINING_BATCHES x
              (UNROLL_STEPS + 1) / env steps, cumulative: the window
              positions the gradient steps read per acted record so far.
              An option arm's records ARE base steps (env steps), so the
              denominator is its records; C0's records are tokens (several
              base steps each), so read (6) recomputes this per arm from
              each arm's own rows (train_steps, its records), never from
              this field across arms;
          opt_replay_obs_bytes = the replay's observation stacks, in bytes
              (the records' other fields apart)."""
        w = self._option_window()
        pool = self._option_probe_pool()
        probe = self._option_probe(pool["obs"], pool["row"])
        errs.update(option_row_fields(w, probe,
                                      int(muzero.TOKEN_BASE_FRAMES)))
        env = int(getattr(self, "env_steps", 0) or 0)
        errs["opt_grad_uses_per_record"] = (
            round(self.trainer.training_steps * int(muzero.TRAINING_BATCHES)
                  * (int(muzero.UNROLL_STEPS) + 1) / env, 3) if env else None)
        errs["opt_replay_obs_bytes"] = int(sum(
            np.asarray(r["obs"]).nbytes
            for t in (getattr(self, "replay", None) or []) for r in t))
        self._opt_window = None
        nxt = pool["row"] + 1
        self._opt_probe = {
            "row": nxt, "keys": np.zeros(0), "obs": [],
            "rng": np.random.default_rng([int(self.seed), nxt, 0x0B5E])}

    def _note_tokens(self, out, z0s):
        """TOKEN_VOCAB row diagnostics (P2a read 4), accumulated per row
        window like _beh_window: per token, how often it was played, its
        payoff (mean z - V over the positions where it was played: the
        age-0 n-step target against the birth search's root value), and
        the birth search's visit mass on it; plus frames per decision and
        decisions per game. Emitted and reset by _merge_row_stats."""
        acted = out[:-1]
        if not acted:
            return
        A = len(muzero.TOKEN_VOCAB)
        act = np.array([int(t["action"]) for t in acted])
        pay = np.array([z0s[i] - acted[i]["root_value"]
                        for i in range(len(acted))], dtype=np.float64)
        mass = np.stack([t["target_policy"] for t in acted]).sum(axis=0)
        frames = sum(muzero.token_frames(a) for a in act)
        w = getattr(self, "_tok_window", None) or {
            "counts": np.zeros(A, dtype=np.int64),
            "pay": np.zeros(A, dtype=np.float64),
            "mass": np.zeros(A, dtype=np.float64),
            "frames": 0, "decisions": 0, "games": 0}
        w["counts"] += np.bincount(act, minlength=A)
        w["pay"] += np.bincount(act, weights=pay, minlength=A)
        w["mass"] += mass
        w["frames"] += int(frames)
        w["decisions"] += len(acted)
        w["games"] += 1
        over = mined_overlap(muzero.TOKEN_VOCAB)
        if over is not None:
            # E3-A10 read (2), the REDUNDANCY: each record's search mass on
            # the mined tokens whose inputs overlap the played token's
            # (mined_overlap), summed over the window's decisions
            tgt = np.stack([t["target_policy"] for t in acted])
            w["overlap"] = w.get("overlap", 0.0) + float(
                (tgt * over[act]).sum())
        self._tok_window = w
        # Cumulative decisions beside env_steps (base steps), checkpointed
        # with the other diag counters (audit r1 B-L2: the per-row window
        # alone undercounts across a resume).
        tot = getattr(self, "_tok_totals", None) or {"tok_decisions_total": 0}
        tot["tok_decisions_total"] += len(acted)
        self._tok_totals = tot

    def _search_traj_communal(self, traj, indices):
        """Communal refresh search (USE_COMMUNAL_REANALYSIS): re-search a
        stored trajectory's policies through the SHARED chain tree instead
        of one independent act_search per position. COMMUNAL_WINDOW_ROOTS
        roots run in lockstep -- one simulation per root per round, the
        papers' sequential schedule inside each tree, the GPU batch formed
        ACROSS roots -- and statistics accumulate communally across the
        searches. The chain is SEEDED (create_tree_from_game with
        seed_values=True, restored 2026-08-30): the Monte-Carlo value
        backup anchors each behaviour child with its realized return, and
        readout subtracts the one bookkeeping visit (plus, under
        COMMUNAL_EDGE_VISITS, the behaviour child's own root visits --
        the successor's search budget). Neutral chains
        (2026-08-27..30) let the early value head alone steer every
        re-search; at tiny-buffer refresh coverage that rewrote most of
        the buffer's targets flatter each cycle. Expansions along the
        chain are shared between neighbouring roots, the design's point.

        SEARCH ONLY -- returns {index: (policy, root_value)} and applies
        nothing; _apply_refresh is the single application path for both
        the inline caller (which swapped the searcher's networks to the
        target model already) and the --ray refresh actor (whose own nets
        ARE the target copies). `indices` is the caller's eligible set
        (ascending; the terminated tail and guard-excluded steps are
        already filtered out). One behavioural difference besides
        topology: a truncated tail's search takes root noise like every
        other position, where the act_search path searched it greedily.
        """
        if not indices:
            return {}
        searcher = self.searcher
        saved = getattr(searcher, "current_trajectory", None)
        out = {}
        try:
            evaluated = searcher.eval_obs(
                [(t["obs"], t["action"], t["reward"]) for t in traj])
            searcher.create_tree_from_game(
                evaluated, terminated=not traj[-1].get("truncated", False),
                seed_values=True)
            window = max(1, muzero.COMMUNAL_WINDOW_ROOTS)
            released = -1
            for w0 in range(0, len(indices), window):
                widx = indices[w0:w0 + window]
                results = asyncio.run(searcher.run_communal_mcts(
                    widx, num_simulations=muzero.ACT_SEARCH_SIMS,
                    c_puct=muzero.C_PUCT, discount=muzero.DISCOUNT,
                    chain_seeded=True))
                for idx, (pol, val, has_policy) in results.items():
                    if not has_policy:
                        continue
                    out[idx] = (np.asarray(pol, dtype=np.float64), float(val))
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
        return out

    def _eligible_indices(self, traj):
        """Steps a refresh may re-search: policy-carrying (the terminated
        tail carried no search then, none now) and, under the no-sharpen
        guard, not yet retired as stable-sharp/converged (guard v2)."""
        return [i for i, t in enumerate(traj)
                if t.get("has_policy_target", True)
                and not (muzero.REFRESH_NO_SHARPEN_GUARD
                         and t.get("refresh_guarded"))]

    def _search_traj(self, traj, indices):
        """Search one stored trajectory's eligible steps with whatever
        networks the searcher currently holds; returns
        {index: (policy, root_value)}. Topology per the arm's flags."""
        if muzero.USE_COMMUNAL_REANALYSIS and not muzero.USE_GUMBEL_SEARCH:
            return self._search_traj_communal(traj, indices)
        out = {}
        for i in indices:
            terminal_record = i == len(traj) - 1
            _, pol, val = asyncio.run(self.searcher.act_search(
                traj[i]["obs"], greedy=terminal_record))
            out[i] = (np.asarray(pol, dtype=np.float64), float(val))
        return out

    def _refresh_candidates(self, count):
        """Which stored trajectories to re-search. Default: oldest by
        refresh rotation (EffZero's cadence). REFRESH_ENTROPY_PRIORITY:
        flattest mean stored target entropy first over eligible steps --
        the tiny buffer's sample reuse amplifies every flat target, so
        repairing the flattest first repairs the most-reused weakest
        signals; rotation age breaks ties so equal-entropy trajectories
        still rotate. A trajectory with no eligible steps is skipped
        outright (nothing left to repair)."""
        def freshness(i):
            head = self.replay[i][0]
            return head.get("refreshed_step", head.get("born_step", 0))

        if not muzero.REFRESH_ENTROPY_PRIORITY:
            return sorted(range(len(self.replay)), key=freshness)[:count]
        scored = []
        for i, traj in enumerate(self.replay):
            hs = [float(-(t["target_policy"]
                          * np.log(t["target_policy"] + 1e-12)).sum())
                  for t in (traj[j] for j in self._eligible_indices(traj))]
            if hs:
                scored.append((-float(np.mean(hs)), freshness(i), i))
        scored.sort()
        return [i for _, _, i in scored[:count]]

    def _apply_refresh(self, traj, results, now):
        """Apply fresh search results to a stored trajectory: guard
        bookkeeping, target/root-value replacement, priority recompute
        under the 4.3 age-shrunk horizon, and the rotation stamp. The
        SINGLE application path -- the inline refresh and the --ray
        refresh actor's swap-on-ready both land here, so offloaded
        semantics cannot drift from sequential. The fresh result is
        applied even when the guard trips: a newer target model gives a
        better value/priority regardless; the guard only stops FUTURE
        re-searches of a step whose target is STABLE-SHARP (v2: sharp
        and reproduced -- see the muzero constants block)."""
        for idx, (pol, val) in results.items():
            t = traj[idx]
            pol = np.asarray(pol, dtype=np.float64)
            if muzero.REFRESH_NO_SHARPEN_GUARD and not t.get("refresh_guarded"):
                # Guard v2 (2026-09-03): strike ONLY on stable-sharp --
                # target already sharp AND reproduced by the re-search.
                # Flat targets never strike (their flatness indicts the
                # model; test 2 froze weak-model flat targets and lost
                # test 1's acceleration), and a redirect (sharp toward a
                # DIFFERENT action: large KL) resets -- a correctable
                # collapse is never frozen into the buffer.
                old = np.asarray(t["target_policy"], dtype=np.float64)
                new_h = float(-(pol * np.log(pol + 1e-12)).sum())
                o = old + 1e-6
                q = pol + 1e-6
                o, q = o / o.sum(), q / q.sum()
                kl = float((q * np.log(q / o)).sum())
                if (new_h <= muzero.REFRESH_GUARD_SHARP_H
                        and kl <= muzero.REFRESH_GUARD_KL_TOL):
                    strikes = t.get("refresh_no_drop_strikes", 0) + 1
                    if strikes >= muzero.REFRESH_GUARD_STRIKES:
                        t["refresh_guarded"] = True
                        t.pop("refresh_no_drop_strikes", None)
                    else:
                        t["refresh_no_drop_strikes"] = strikes
                else:
                    t.pop("refresh_no_drop_strikes", None)
            t["target_policy"] = pol
            t["root_value"] = float(val)
            if muzero.VALUE_TARGET_MODE == "mixed":
                t["root_value_step"] = now
        # Priorities stay TD-based even under SVE: "the z the trainer will
        # compute" for a stale SVE position IS its root value, which would
        # zero its priority and starve it under PER. The trainer's
        # writeback (|v - target_value[:, 0]|) gives SVE roots |v - SVE|,
        # as EfficientZero V2's batch priorities do.
        for idx, t in enumerate(traj):
            # Priority against the TD z of the 4.3 age-shrunk horizon, not
            # the full TD_STEPS (refresh candidates are the oldest, where
            # it shrinks most) -- the z the trainer computes in "td" mode;
            # under "mixed" the trainer uses TD_STEPS or, for stale roots,
            # the stored value, and this stays a TD-based proxy (see above). The bootstrap is the just-refreshed
            # stored search value -- MuZero's canonical priority --
            # which tracks, without exactly equaling, the trainer's
            # target-net value-head bootstrap. Recomputed for EVERY
            # record: refreshed neighbours shift unsearched steps' z.
            age = max(0, now - t.get("born_step", now))
            z0 = muzero.paper_value_target(
                traj, idx, muzero.paper_horizon(age), muzero.DISCOUNT)
            t["priority"] = abs(t["root_value"] - z0) + muzero.PER_EPS
        for t in traj:
            # Stamped last, in its own pass: freshness() reads the head
            # record, which must only carry a stamp once the whole
            # trajectory -- searches AND priorities -- completed.
            t["refreshed_step"] = now

    def refresh_targets(self, count):
        """EfficientZero's reanalyze, policy half (approximated): re-search
        every eligible position of `count` candidate trajectories through
        the LAGGED TARGET MODEL -- noisy, as their reanalyze workers search
        -- replacing policy targets and root values, recomputing
        priorities, and stamping the refresh rotation (ages are
        preserved). Candidates are oldest-first by default; flattest-first
        under REFRESH_ENTROPY_PRIORITY (see _refresh_candidates). Their
        revisit_policy_search_rate=0.99 re-searches sampled positions per
        batch on a CPU fleet; this is the compute-bounded analogue.

        Rotation stamp is refreshed_step, NOT born_step: fresh policy
        targets do not make the stored ACTION SEQUENCE on-policy, so the
        4.3 age-shrunk value horizon (keyed on born_step) must keep
        shrinking (their delta_td keys on buffer position and is never
        restored by a re-search). Under --ray with REFRESH_OFFLOAD the
        search half of this runs on a dedicated actor instead
        (RayPipeline.request_refresh/poll_refresh); this inline path is
        the sequential-mode implementation and the offload's semantic
        reference."""
        if not self.replay:
            return
        candidates = self._refresh_candidates(count)
        if not candidates:
            return
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
            got = []
            for ti in candidates:
                traj = self.replay[ti]
                idxs = self._eligible_indices(traj)
                results = self._search_traj(traj, idxs)
                self._apply_refresh(traj, results, now)
                got.append(len(idxs))
        finally:
            (nm.representation, nm.prediction,
             nm.dynamics, nm.value_prefix) = saved
        self._note_quota_stats(got)

    def refresh_targets_vtree(self, count):
        """Phase-4 refresh (REFRESH_VTREE): fresh-root VECTORIZED
        searches on the trainer's device against the lagged target
        model -- hundreds of stored positions per wave through
        muzero_vtree, so 5x coverage costs tens of seconds inline and
        needs no offload actor. Topology is the pre-communal refresh5x
        shape (independent root per position; root noise at every
        position, which the communal path also applies, truncated tails
        included). Candidate selection (_refresh_candidates: rotation or
        entropy priority), eligibility (guard), and application
        (_apply_refresh: guard strikes, 4.3-horizon priorities, rotation
        stamps) are the SAME code as every other refresh path."""
        if not self.replay:
            return
        candidates = self._refresh_candidates(count)
        if not candidates:
            return
        now = self.trainer.training_steps
        engine = self._vtree_engine()
        got = self._refresh_trajs_vtree(
            [self.replay[ti] for ti in candidates], engine, now)
        # Accounting (2026-09-14): the quota path never wrote the refresh
        # counters the saturate/sweep paths do, so quota rows carried no
        # refresh volume -- and volume is the variable the r12 arm tests.
        # Same keys and the same per-call *_last_iter semantics as
        # refresh_saturate.
        self._note_quota_stats(got)

    def _vtree_engine(self):
        """The vectorized engine over the CURRENT lagged target nets;
        cheap to construct (wraps existing modules, allocates nothing).
        Under USE_COMMUNAL_REANALYSIS the v2 communal engine (a
        VectorizedPUCT subclass whose fresh-root search() is the v1
        method itself); otherwise v1, so the fresh-root path is
        untouched."""
        self.trainer._ensure_target_model()
        import muzero_vtree
        cls = (muzero_vtree.VectorizedCommunalPUCT
               if muzero.USE_COMMUNAL_REANALYSIS
               else muzero_vtree.VectorizedPUCT)
        return cls(
            {"representation": self.trainer.target_representation,
             "dynamics": self.trainer.target_dynamics,
             "prediction": self.trainer.target_prediction,
             "value_prefix": self.trainer.target_value_prefix},
            ACTIONS, self.trainer.device)

    def _refresh_traj_vtree(self, traj, engine, now):
        """Search-and-apply ONE trajectory's eligible steps in batched
        waves. The unit of work for both the scheduled quota and the
        saturation loop; application is the shared _apply_refresh (a
        no-eligible trajectory still rotates).

        Topology per the arm: USE_COMMUNAL_REANALYSIS routes through the
        v2 communal engine -- ONE shared seeded chain tree for the
        trajectory, the eligible positions processed window by window
        (COMMUNAL_WINDOW_ROOTS, ascending, release after each), exactly
        _search_traj_communal's drive of the sequential engine (parity
        gate: tests/test_vtree_communal.py). Otherwise the v1 fresh-root
        waves, byte-identical to before."""
        if muzero.USE_COMMUNAL_REANALYSIS:
            return self._refresh_trajs_communal([traj], engine, now)[0]
        idxs = self._eligible_indices(traj)
        results = {}
        batch = max(1, int(muzero.VTREE_REFRESH_BATCH))
        for w0 in range(0, len(idxs), batch):
            chunk = idxs[w0:w0 + batch]
            obs = np.stack([traj[i]["obs"] for i in chunk])
            pols, vals, _ = engine.search(
                obs, num_simulations=muzero.ACT_SEARCH_SIMS,
                c_puct=muzero.C_PUCT, discount=muzero.DISCOUNT,
                root_noise=True)
            for i, pol, val in zip(chunk, pols, vals):
                results[i] = (np.asarray(pol, dtype=np.float64),
                              float(val))
        self._apply_refresh(traj, results, now)
        return len(idxs)  # positions searched (accounting)

    def _refresh_trajs_vtree(self, trajs, engine, now):
        """Search-and-apply a LIST of trajectories; returns positions
        searched per trajectory. Fresh-root: _refresh_traj_vtree one at
        a time (unchanged). Communal: groups of VTREE_COMMUNAL_TRAJS
        trajectories share one forest (see muzero's constant)."""
        if muzero.USE_COMMUNAL_REANALYSIS:
            return self._refresh_trajs_communal(trajs, engine, now)
        return [self._refresh_traj_vtree(t, engine, now) for t in trajs]

    def _refresh_trajs_communal(self, trajs, engine, now):
        """The communal vectorized search for a list of trajectories,
        applied per trajectory through _apply_refresh. Each trajectory
        is ONE shared seeded chain tree searched window by window
        (COMMUNAL_WINDOW_ROOTS, ascending, release after each) --
        _search_traj_communal's drive of the sequential engine, parity-
        gated in tests/test_vtree_communal.py. VTREE_COMMUNAL_TRAJS
        trajectories per forest advance their windows in lockstep. (The
        sequential communal path falls back to per-root searches under
        Gumbel; the engine asserts instead -- loud.)"""
        out = []
        group = max(1, int(muzero.VTREE_COMMUNAL_TRAJS))
        for g0 in range(0, len(trajs), group):
            chunk = trajs[g0:g0 + group]
            idx_lists = [self._eligible_indices(t) for t in chunk]
            searchable = [(t, ix) for t, ix in zip(chunk, idx_lists) if ix]
            got = []
            if searchable:
                got = engine.search_trajectories(
                    [{"obs": [r["obs"] for r in t],
                      "actions": [r["action"] for r in t],
                      "rewards": [r["reward"] for r in t],
                      "terminated": not t[-1].get("truncated", False)}
                     for t, _ in searchable],
                    [ix for _, ix in searchable],
                    num_simulations=muzero.ACT_SEARCH_SIMS,
                    c_puct=muzero.C_PUCT, discount=muzero.DISCOUNT,
                    tie_break="random")
            found = {id(t): g for (t, _), g in zip(searchable, got)}
            for t, ix in zip(chunk, idx_lists):
                results = {i: (np.asarray(pol, dtype=np.float64), float(val))
                           for i, (pol, val) in found.get(id(t), {}).items()}
                self._apply_refresh(t, results, now)
                out.append(len(ix))
        return out

    def refresh_saturate(self, acting_ready, floor=1):
        """Kai's saturation refresh (REFRESH_SATURATE): refresh one
        flattest candidate at a time until the environment actor's
        episode is ready, with a minimum of `floor` candidates
        regardless of slack. Re-selection reorders the REMAINING unseen
        candidates each round; a refreshed trajectory re-enters the
        queue next call, never within one -- the seen-set bounds a call
        at one pass over the buffer, and the guard limits work ACROSS
        calls (round-2 F-C). Returns the number refreshed."""
        if not self.replay:
            return 0
        engine = self._vtree_engine()
        done = 0
        steps = 0
        seen = set()
        while True:
            if done >= floor and acting_ready():
                break
            # One pass over the buffer per call (audit F3): a trajectory
            # is never re-picked within a call, so consecutive guard
            # strikes land in DIFFERENT calls -- across real training
            # windows, the regime the (SHARP_H, KL_TOL, STRIKES) triple targets
            # for -- and the loop is bounded by the buffer size even if
            # the actor hangs.
            pick = None
            for ci in self._refresh_candidates(len(seen) + 1):
                if id(self.replay[ci]) not in seen:
                    pick = ci
                    break
            if pick is None:
                break  # everything guarded, ineligible, or already done
            seen.add(id(self.replay[pick]))
            now = self.trainer.training_steps
            got = self._refresh_traj_vtree(self.replay[pick], engine, now)
            steps += got
            # Only REAL work counts toward the floor and the applied
            # counters (sat-guard audit F1): under rotation ordering a
            # fully-guarded trajectory is a legal pick that searches
            # zero positions -- it still burns its seen-set slot (and
            # gets stamped in _apply_refresh, so it rotates to the
            # back), but letting it satisfy the floor would quietly
            # shrink cadence coverage below the base schedule as
            # retirement grows. Zero-work picks also kept
            # refresh_applied like-for-like across arms (audit F2).
            if got > 0:
                done += 1
        stats = getattr(self, "_refresh_offload_stats", None) or {}
        self._refresh_offload_stats = {
            "refresh_applied": stats.get("refresh_applied", 0) + done,
            "refresh_last_iter": done,
            # POSITIONS re-searched -- the "# of steps refreshed" Kai's
            # cross-run comparison needs (trajectory counts hide length
            # variance; archfix's volume reconstructs as ~2 trajs/iter
            # of every position, so this is the like-for-like number).
            "refresh_steps_total": stats.get("refresh_steps_total", 0)
            + steps,
            "refresh_steps_last_iter": steps}
        return done

    def refresh_trajs(self, trajs):
        """Phase 5: refresh EXACTLY these trajectory objects (newcomer
        refresh at collect), sharing refresh_saturate's accounting keys
        so eval rows stay comparable. Returns positions searched."""
        if not trajs:
            return 0
        engine = self._vtree_engine()
        done = steps = 0
        now = self.trainer.training_steps
        for got in self._refresh_trajs_vtree(list(trajs), engine, now):
            steps += got
            if got > 0:
                done += 1
        self._note_sweep_stats(done, steps)
        return steps

    def refresh_sweep(self, count):
        """Phase 5: unconditionally refresh the `count` least-recently-
        refreshed trajectories (rotation order via _refresh_candidates
        with entropy priority off), always-replace. No guard, no floor,
        no acting poll -- the async driver paces calls so the whole
        buffer turns over at least once per ASYNC_SWEEP_TRAIN_STEPS
        gradient steps (the freshness contract). Returns trajectories
        refreshed."""
        if not self.replay or count <= 0:
            return 0
        engine = self._vtree_engine()
        done = steps = 0
        # training_steps cannot move within this call (the loop is the
        # mutex), so one stamp serves every candidate.
        now = self.trainer.training_steps
        trajs = [self.replay[ci] for ci in self._refresh_candidates(count)]
        for got in self._refresh_trajs_vtree(trajs, engine, now):
            steps += got
            if got > 0:
                done += 1
        self._note_sweep_stats(done, steps)
        return done

    def _note_quota_stats(self, positions_per_traj):
        """Refresh accounting for the scheduled-quota paths (sequential
        refresh_targets and refresh_targets_vtree): per-call *_last_iter
        values and cumulative totals, the keys refresh_saturate writes,
        so quota rows are comparable to saturate/sweep rows."""
        steps = int(sum(positions_per_traj))
        done = int(sum(1 for g in positions_per_traj if g > 0))
        stats = getattr(self, "_refresh_offload_stats", None) or {}
        self._refresh_offload_stats = {
            "refresh_applied": stats.get("refresh_applied", 0) + done,
            "refresh_last_iter": done,
            "refresh_steps_total": stats.get("refresh_steps_total", 0)
            + steps,
            "refresh_steps_last_iter": steps}

    def _note_sweep_stats(self, done, steps):
        """Accumulate into the SAME keys refresh_saturate writes;
        *_last_iter accumulates between eval rows and is zeroed by the
        async driver when a row is requested (there is no iteration to
        scope it otherwise)."""
        stats = getattr(self, "_refresh_offload_stats", None) or {}
        self._refresh_offload_stats = {
            "refresh_applied": stats.get("refresh_applied", 0) + done,
            "refresh_last_iter": stats.get("refresh_last_iter", 0) + done,
            "refresh_steps_total": stats.get("refresh_steps_total", 0)
            + steps,
            "refresh_steps_last_iter":
                stats.get("refresh_steps_last_iter", 0) + steps}

    def zero_sweep_window(self):
        """Reset the per-row accumulators at an eval boundary."""
        stats = getattr(self, "_refresh_offload_stats", None) or {}
        if stats:
            stats["refresh_last_iter"] = 0
            stats["refresh_steps_last_iter"] = 0

    def reanalyse(self, traj, terminated=True, born_step=None,
                  birth_nets=None, birth_prior_nets=None):
        if muzero.PAPER_TARGETS:
            return self.paper_targets(traj, terminated, born_step=born_step,
                                      birth_nets=birth_nets,
                                      birth_prior_nets=birth_prior_nets)
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
        # Whole GAMES (play_game: every life of a multi-life game), which
        # keeps only (return, decisions, terminated) triples, so no eval
        # trajectory outlives its life (8 cap-3000 trajectories would be
        # ~2 GB transient; verification F1).
        played = [self.play_game() for _ in range(episodes)]
        return {**eval_summary(played),
                **self.evaluate_diagnostics(batches=batches)}

    def evaluate_diagnostics(self, batches=8):
        """The driver-bound half of evaluate(): the prediction-error panel
        on the TRAINER's nets. The greedy episodes are the offloadable half
        -- --ray plays them on the eval actor with the same synced weights."""
        # Runs after collect/evict with no training block in between: drop
        # the bootstrap cache so an id()-keyed entry from an evicted
        # trajectory can never serve a new one (prefetch audit F4).
        if hasattr(self.trainer, "invalidate_bootstrap_cache"):
            self.trainer.invalidate_bootstrap_cache()
        K = muzero.UNROLL_STEPS
        errs = {"per_step": 0.0, "prefix": 0.0, "control": 0.0, "value": 0.0,
                "value_control": 0.0, "policy_entropy": 0.0, "policy_maxp": 0.0,
                "prefix_step1": 0.0, "prefix_step1_control": 0.0}
        projs = []
        # Test O stage 4a (OPTION_LENGTH > 1, option batches): the prefix
        # panel split by row kind (option_prefix_panel's sums)
        opt_pf = {} if int(muzero.OPTION_LENGTH) > 1 else None
        if self.replay:
            with torch.no_grad():
                for _ in range(batches):
                    with td_value_targets():
                        b = asyncio.run(self.trainer.prepare_batch(
                            self.replay, 32, K, ACTIONS))
                    obs = torch.tensor(b["obs"], dtype=torch.float32, device=DEVICE) / 255.0
                    acts = torch.tensor(b["actions"], dtype=torch.long, device=DEVICE)
                    rews = torch.tensor(b["rewards"], dtype=torch.float32, device=DEVICE)
                    tval = torch.tensor(b["target_value"], dtype=torch.float32, device=DEVICE)
                    truth = muzero.discounted_prefix_targets(rews, discount=1.0)

                    latent = self.trainer.representation(obs[:, 0])
                    # Target-branch projection of the SAME root latents:
                    # no extra sampling, no RNG draw, eval-mode BN.
                    projs.append(self.trainer.projector(
                        latent, with_prediction=False).float().cpu())
                    policy_logits, value, _ = self.trainer.prediction(latent)
                    errs["value"] += (value.squeeze(-1) - tval[:, 0]).abs().mean().item()
                    errs["value_control"] += tval[:, 0].abs().mean().item()
                    # Policy sharpening: the returns panel is too noisy to show
                    # whether the head is learning at all (uniform over 5
                    # actions: entropy 1.609, max-p 0.20).
                    p = torch.softmax(policy_logits, dim=1)
                    errs["policy_entropy"] += (-p * torch.log(p + 1e-12)).sum(dim=1).mean().item()
                    errs["policy_maxp"] += p.max(dim=1).values.mean().item()

                    vp = self.trainer.value_prefix
                    state = vp.initial_state(obs.size(0), DEVICE)
                    running = torch.zeros(obs.size(0), 1, device=DEVICE)
                    # The prefix head in EVAL mode -- the search's function
                    # (the resting state already; expand_prefixes' toggle,
                    # so the step-1 number is the search-side one in any
                    # mode). The state is fresh here and never resets, so
                    # call i is LSTM step i.
                    was_training = vp.training
                    vp.eval()
                    jumps = (opt_pf is not None and "option_span" in b)
                    try:
                        for i in range(1, K + 1):
                            latent, reward, _ = self.trainer.dynamics(latent, acts[:, i - 1])
                            running = running + reward
                            prefix, _, state = vp(latent, state, step=i)
                            if i == 1:
                                # The step-1 prefix -- every root expansion
                                # and horizon reset in the search -- which
                                # the step-K error below hid (Step 0 (iii)).
                                t1 = truth[:, 1:2]
                                errs["prefix_step1"] += (prefix - t1).abs().mean().item()
                                errs["prefix_step1_control"] += t1.abs().mean().item()
                                if jumps:
                                    # Test O: the same step-1 prefix through
                                    # train-mode BN (batch statistics, on a
                                    # copy: PB's probe, Step 0 (iii))
                                    p1, l1 = prefix, latent
                                    vpt = copy.deepcopy(vp).train()
                                    p1t, _, _ = vpt(
                                        l1, vpt.initial_state(obs.size(0),
                                                              DEVICE), step=1)
                    finally:
                        vp.train(was_training)
                    tgt = truth[:, K:K + 1]
                    errs["per_step"] += (running - tgt).abs().mean().item()
                    errs["prefix"] += (prefix - tgt).abs().mean().item()
                    errs["control"] += tgt.abs().mean().item()
                    if jumps:
                        option_prefix_panel(
                            opt_pf, np.asarray(b["option_span"]),
                            (prefix - tgt).abs().squeeze(1).cpu().numpy(),
                            tgt.abs().squeeze(1).cpu().numpy(),
                            (p1 - t1).abs().squeeze(1).cpu().numpy(),
                            t1.abs().squeeze(1).cpu().numpy(),
                            p1.abs().squeeze(1).cpu().numpy(),
                            p1t.abs().squeeze(1).cpu().numpy())
            errs = {k: v / batches for k, v in errs.items()}
            errs.update(projector_collapse_stats(self.trainer.projector,
                                                 torch.cat(projs)))

        # Search-target flatness, straight from the last recorded episode
        # (2026-08-30 re-audit): policy_entropy above is the HEAD's entropy
        # on buffer states -- it lags its targets. This is the entropy of
        # the acting searches' own visit distributions, the quantity the
        # batch-width and chain-seeding regressions actually move, logged
        # so search-side flattening is visible without a probe.
        fresh = getattr(self, "_fresh_target_H", None)
        if fresh is not None:
            errs["fresh_target_entropy"] = fresh

        # Offload accounting (3a audit): applied/skipped/lag land in every
        # row so a slow refresh actor can never silently degrade coverage.
        stats = getattr(self, "_refresh_offload_stats", None)
        if stats:
            errs.update(stats)
        # BIRTH_SEARCH and ReDo accounting (2026-09-26), same pattern.
        self._merge_row_stats(errs)
        if opt_pf is not None:
            # Test O stage 4a: the option arms' row (read (3) with its on/off
            # root probe, read (4), DEPARTURES 3 and 7, the train steps, read
            # (6)) and the prefix panel on jump rows
            self._merge_option_stats(errs)
            errs.update(option_prefix_fields(opt_pf))
        if muzero.RESET_EVERY_STEPS:
            # Test C: the resets so far (a row written before its
            # iteration's reset counts it out) and the weight ships that
            # carried the held pre-reset states so far (audit r1a LOW-3: a
            # row-time "held now" flag is always False, the hold ending on
            # a row)
            errs["resets"] = len(self.reset_log)
            errs["held_ships"] = len(getattr(self, "held_ships", None)
                                     or [])
        if muzero.RAMP_STEPS:
            # Test C2: the ramp at the row -- TD_STEPS and DISCOUNT as every
            # reader sees them, and the gradient steps since the last reset
            # (a row written before its iteration's reset counts from the
            # previous one) -- and ramp_trace, [iteration, steps since the
            # last reset, n, gamma] at the first gradient step of each
            # iteration since the previous row: rows fall every 5
            # iterations and a ramp spans 5, so the row alone would only
            # ever show the end values. The trace is checkpointed (a resume
            # keeps the row's window) and restarts here.
            trace = getattr(self, "ramp_trace", None) or []
            errs["ramp"] = {
                "n": int(muzero.TD_STEPS),
                "gamma": float(muzero.DISCOUNT),
                "cycle_steps": int(self.trainer.training_steps
                                   - muzero.LR_WARMUP_ORIGIN)}
            errs["ramp_trace"] = [list(e) for e in trace]
            self.ramp_trace = []
            # ... and ramp_windows (audit r1 LOW-2: the row's own value
            # diagnostics fall where the ramp has ended): over the gradient
            # steps since the previous row, "in" the ramp (steps since the
            # reset < RAMP_STEPS) and "out", the steps, the steps scored and
            # their mean value loss (nats) and mean position-0 value target
            # (the batch's scalar z); None where none was scored.
            # Checkpointed like the trace; restarts here.
            wins = getattr(self, "ramp_windows", None) or ramp_windows_empty()
            errs["ramp_windows"] = {
                k: {"steps": w["steps"], "scored": w["scored"],
                    "value_loss": (round(w["value_loss"] / w["scored"], 6)
                                   if w["scored"] else None),
                    "target_mean": (round(w["target_mean"] / w["scored"], 6)
                                    if w["scored"] else None)}
                for k, w in wins.items()}
            self.ramp_windows = ramp_windows_empty()
        if muzero.DORMANT_DIAGNOSTIC:
            # Test C's plasticity read (both runs)
            errs.update(self.dormant_stats())

        return errs

    def _merge_row_stats(self, errs):
        """Copy the BIRTH_SEARCH / ReDo counters into a row and restart
        their per-row windows; emit the per-row behaviour means
        (beh_*) and training-step means (step_*, ReDo pre/post grad
        norms)."""
        for name, window in (("_birth_stats", "birth_positions_window"),
                             ("_redo_stats", "redo_recycled_window")):
            stats = getattr(self, name, None)
            if stats:
                errs.update(stats)
                stats[window] = 0          # the window restarts per row
                if name == "_redo_stats":
                    stats["redo_by_net_window"] = {}
        w = getattr(self, "_beh_window", None)
        if w and w["n"]:
            for k, v in w["sums"].items():
                errs[f"beh_{k}"] = round(v / w["n"], 4)
            errs["beh_action_counts"] = list(w["counts"])
            errs["beh_T_values"] = list(w.get("Ts", []))
            self._beh_window = None
        tw = getattr(self, "_tok_window", None)
        if tw and tw["decisions"]:
            # TOKEN_VOCAB (P2a read 4): per-token play counts, payoff and
            # birth-search share; per-duration shares of play and of search;
            # frames per decision and decisions per game. Per-row window,
            # not checkpointed (like _beh_window).
            vocab = muzero.TOKEN_VOCAB
            n = tw["counts"]
            errs["tok_counts"] = [int(c) for c in n]
            errs["tok_payoff"] = [round(float(p / c), 4) if c else None
                                  for p, c in zip(tw["pay"], n)]
            share = tw["mass"] / tw["mass"].sum()
            errs["tok_search_share"] = [round(float(s), 4) for s in share]
            # E3: duration tokens by their frames; MINED tokens in a "mined"
            # bucket of their own (absent without mined tokens), and each
            # mined token's play / search share (E3-A10 read (2))
            mined = [muzero.token_is_mined(t) for t in vocab]
            frames = [muzero.token_total_frames(t) for t in vocab]
            durs = sorted({f for f, m in zip(frames, mined) if not m})
            errs["tok_duration_play_share"] = {
                str(d): round(float(sum(c for f, m, c in zip(frames, mined, n)
                                        if not m and f == d) / n.sum()), 4)
                for d in durs}
            errs["tok_duration_search_share"] = {
                str(d): round(float(sum(s for f, m, s in zip(frames, mined,
                                                             share)
                                        if not m and f == d)), 4)
                for d in durs}
            if any(mined):
                errs["tok_duration_play_share"]["mined"] = round(float(sum(
                    c for m, c in zip(mined, n) if m) / n.sum()), 4)
                errs["tok_duration_search_share"]["mined"] = round(float(sum(
                    s for m, s in zip(mined, share) if m)), 4)
                errs["tok_mined_play_share"] = {
                    muzero.token_key(t): round(float(c / n.sum()), 4)
                    for t, m, c in zip(vocab, mined, n) if m}
                errs["tok_mined_search_share"] = {
                    muzero.token_key(t): round(float(s), 4)
                    for t, m, s in zip(vocab, mined, share) if m}
                errs["tok_mined_overlap_search_share"] = round(
                    tw.get("overlap", 0.0) / tw["decisions"], 4)
            errs["tok_frames_per_decision"] = round(
                tw["frames"] / tw["decisions"], 3)
            # Per TRAJECTORY: a game on Pong, one LIFE on Ms. Pac-Man
            # (EpisodicLifeEnv) -- the probe counts decisions per game.
            errs["tok_decisions_per_game"] = round(
                tw["decisions"] / tw["games"], 1)
            self._tok_window = None
        pw = getattr(self, "_pc_window", None)
        if pw and pw["n"]:
            # Test O's O0 pre-check: the shares of the row's acted positions
            # able to hold a length-2 and a length-3 option, their counts (a
            # read over several rows pools them) and the means beside them.
            # Per-row window, not checkpointed (like _beh_window).
            n = pw["n"]
            errs["pc_positions"] = n
            errs["pc_hold2"] = pw["hold2"]
            errs["pc_hold3"] = pw["hold3"]
            errs["pc_hold2_share"] = round(pw["hold2"] / n, 4)
            errs["pc_hold3_share"] = round(pw["hold3"] / n, 4)
            errs["pc_pi_m"] = round(pw["pi_m"] / n, 4)
            errs["pc_pi_played"] = round(pw["pi_played"] / n, 4)
            errs["pc_hit_rate"] = round(pw["hit"] / n, 4)
            self._pc_window = None
        if muzero.TOKEN_VOCAB is not None:
            tot = getattr(self, "_tok_totals", None)
            if tot:
                errs.update(tot)
        if muzero.TOKEN_VOCAB is not None or muzero.OPTION_PRECHECK:
            # The buffer's size in trajectories, records and base steps
            # (audit r1 B-M1: the transition cap counts base steps under
            # tokens; these show what it holds). Test O: on O0 too, for read
            # (6) against C0; its records are base steps.
            replay = getattr(self, "replay", None) or []
            errs["replay_trajs"] = len(replay)
            errs["replay_records"] = int(sum(len(t) for t in replay))
            errs["replay_base_steps"] = int(sum(replay_units(t)
                                                for t in replay))
        if muzero.REWARD_GROWTH:
            # Reward-clip growth (P2a on Ms. Pac-Man, read 3): the ceiling
            # the buffer is clipped at, the share of its raw reward mass (in
            # REWARD_UNITs) the clip keeps, and its largest stored reward
            # (above the reward support's raw ceiling it saturates).
            replay = getattr(self, "replay", None) or []
            unit = float(muzero.REWARD_UNIT)
            c = getattr(self, "_reward_ceiling", None)
            mags = [abs(float(r)) / unit for t in replay for rec in t
                    for r in rec["raw_reward"]]
            raw = sum(mags)
            errs["reward_ceiling"] = None if c is None else round(float(c), 4)
            errs["reward_kept"] = (round(sum(min(m, c) for m in mags) / raw, 4)
                                   if raw and c is not None else None)
            errs["reward_max"] = round(max(
                (abs(rec["reward"]) for t in replay for rec in t),
                default=0.0), 4)
        sv = getattr(getattr(self, "trainer", None), "_sve_window", None)
        if sv and sv["n_real"]:
            # SVE (VALUE_TARGET_MODE "mixed"): share of real value targets
            # taken from search, and |SVE - TD| / (SVE - TD) / value age
            # where it was used (CURRICULUM's pre-registered diagnostics).
            errs["sve_frac"] = round(sv["n_sve"] / sv["n_real"], 4)
            errs["sve_buffer_frac"] = round(sv["buf"] / sv["batches"], 4)
            if sv.get("searches"):
                # Stage 2 cost: fresh searches run (cache misses) and their
                # wall-clock seconds in this row's window.
                errs["sve_searches"] = sv["searches"]
                errs["sve_search_s"] = round(sv["search_s"], 1)
            if sv["n_sve"]:
                errs["sve_td_gap"] = round(sv["gap"] / sv["n_sve"], 4)
                errs["sve_td_bias"] = round(sv["bias"] / sv["n_sve"], 4)
                errs["sve_value_age"] = round(sv["age"] / sv["n_sve"], 1)
            self.trainer._sve_window = None
        ew = getattr(self, "_eager_window", None)
        if ew:
            # RAY_EAGER_COLLECT: [iteration, step offset at which its game
            # joined the buffer] per eager iteration in this row (offset
            # None: it missed the block and the iteration-end collect took
            # it). Primed iterations (1, and the first after a resume:
            # nothing in flight) have no entry. Per-row window, NOT
            # checkpointed (like _beh_window / _step_window): after a
            # resume the row lists only the post-resume iterations.
            errs["eager_collect_steps"] = [list(e) for e in ew]
            self._eager_window = None
        st = getattr(self, "_step_window", None)
        if st and st["n"]:
            errs["step_loss_mean"] = round(st["loss"] / st["n"], 4)
            errs["step_nonfinite"] = st.get("nonfinite", 0)
            errs["step_grad_norm_mean"] = round(st["gn"] / st["n"], 4)
            if st["passes"]:
                errs["redo_gn_pre20"] = round(st["pre"] / st["passes"], 4)
                errs["redo_gn_post20"] = round(st["post"] / st["passes"], 4)
            self._step_window = None


NET_NAMES = ("representation", "dynamics", "prediction", "value_prefix",
             "projector")
OPT_NAMES = ("rep_optimizer", "dyn_optimizer", "pred_optimizer",
             "prefix_optimizer", "proj_optimizer")


def projector_collapse_stats(projector, proj):
    """Dimensional-collapse panel for the SimSiam projector (2026-09-21).

    The counterfactual-dynamics probes found the projection space shrunk to
    ~5-8 effective dimensions in most plateau runs, with about a third of
    the head's final BatchNorm gammas dead -- the consistency loss, the
    densest signal teaching the dynamics, then carries little. archfix
    (+17.1) kept ~20; an iter-10 model ~45. Those numbers were over a
    438-state probe set; here the rank is over the diagnostics' buffer
    samples, so read the TREND within a run, not the absolute level.

    proj_erank: exp(entropy) of the normalized squared singular values of
    the centered projections (1 = everything on one line).
    proj_dead_frac: fraction of the final BatchNorm's |gamma| < 0.01.

    Non-finite projections log NaN rather than raise: svdvals throws on
    NaN/inf, and a logging panel must never kill (and, on resume,
    crash-loop) a run whose rows used to record the same state."""
    bns = [m for m in projector.projection
           if isinstance(m, torch.nn.BatchNorm1d)]
    dead = round(float((bns[-1].weight.detach().abs() < 0.01).float().mean()), 3)
    if not bool(torch.isfinite(proj).all()):
        return {"proj_erank": float("nan"), "proj_dead_frac": dead}
    x = proj.double()
    x = x - x.mean(0, keepdim=True)
    s = torch.linalg.svdvals(x) ** 2
    if float(s.sum()) > 0:
        p = s / s.sum()
        p = p[p > 0]
        erank = float(torch.exp(-(p * p.log()).sum()))
    else:
        erank = 0.0   # every sample identical: no spread at all
    return {"proj_erank": round(erank, 2), "proj_dead_frac": dead}


def weights_blob(run, arm, seed, spec):
    """A run_one weights file's contents. The final weights and the
    --snapshot-weights snapshots share it, so every probe loads either."""
    return {"arm": arm, "seed": seed,
            "spec": {k: v for k, v in spec.items()},
            "training_steps": run.trainer.training_steps,
            **{n: getattr(run.trainer, n).state_dict() for n in NET_NAMES}}


def save_weight_snapshot(run, arm, seed, spec, it, acting_states=None,
                         root=None):
    """--snapshot-weights: the nets at eval row `it`, in the weights-file
    format plus the row's iteration. Completed runs otherwise keep only
    their final weights, and the collapse-timing question needs mid-run
    models (2026-09-21). Atomic: a kill mid-write leaves the previous file
    or none, never a torn one. A resumed run replays the rows after its
    checkpoint, so their snapshots are overwritten by the surviving
    timeline, matching the jsonl. `root`: the folder holding the snapshots
    folder (default RESULTS; the distillation step's offline curve passes
    its own, so the probe files it under another run name)."""
    d = (RESULTS if root is None else root) / f"snapshots{TAG}-{arm}-seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"weights-iter{it:03d}.pth"
    tmp = d / f"weights-iter{it:03d}.pth.tmp"
    blob = {**weights_blob(run, arm, seed, spec), "iteration": it}
    if acting_states:
        # BN_RECAL_ACTING: the representation / prediction states the actor
        # and eval actor were shipped at this boundary (recalibrated BN
        # statistics, the trainer's weights) -- the DEPLOYED policy, which
        # probe_exhaustive_eval.py evaluates under PROBE_ACTING_STATES=1.
        blob["acting_states"] = acting_states
    torch.save(blob, tmp)
    os.replace(tmp, path)
    return path


def async_admit(floor, training_steps, admitted_env, ep_steps,
                buffer_empty):
    """Test-6 admission rule: admit unless the floor is set and taking
    this episode would push the ADMITTED train/env ratio below it. An
    empty buffer always admits (bootstrap: nothing can train until the
    first episode exists, and legacy's first iteration ran ratio ~2 on
    exactly one episode)."""
    if not floor or buffer_empty:
        return True
    return (admitted_env + ep_steps) * floor <= training_steps


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
    # BIRTH_SEARCH / ReDo row counters (round-1 audit LOW-1): cumulative
    # totals must survive a resume. Written only when present, so every
    # other arm's checkpoint head is unchanged.
    diag = {k: dict(getattr(run, k)) for k in ("_birth_stats", "_redo_stats",
                                               "_tok_totals")
            if getattr(run, k, None)}
    plan = getattr(run, "_redo_last_plan", None)
    if plan:
        # The re-dormancy baseline (round-2 audit LOW-2), as plain tuples.
        diag["_redo_last_plan"] = [
            ((u.net, u.producer, u.norm, u.consumer, u.hw, u.consumer_attr),
             list(idx)) for u, idx in plan.items()]
    if diag:
        state["diag_stats"] = diag
    if muzero.RESET_EVERY_STEPS:
        # Test C: the resets done and the actor's hold (its held acting
        # states), so a resume neither repeats nor skips a reset and keeps
        # a hold; only then, so every other arm's head is unchanged
        state["resets"] = {"log": [dict(r) for r in run.reset_log],
                           "hold": _cpu_state(run.hold),
                           "held_ships": list(getattr(run, "held_ships",
                                                      []))}
    if muzero.RAMP_STEPS:
        # Test C2: the next row's ramp trace and ramp windows so far
        # (checkpoints fall between rows: a resume neither drops nor
        # double-counts a step); the values themselves follow from the step
        # count and the reset log. Only then, so every other arm's head is
        # unchanged
        state["ramp"] = {
            "trace": [list(e) for e in getattr(run, "ramp_trace", None)
                      or []],
            "windows": copy.deepcopy(getattr(run, "ramp_windows", None)
                                     or ramp_windows_empty())}
    if int(muzero.OPTION_LENGTH) > 1:
        # Test O stage 4a: the open row's option window and its probe roots
        # (their keys, stacks -- at most OPTION_PROBE_ROOTS, about 22 MB --
        # and the pinned generator's state), so a resume keeps a partial
        # row's counts and its probe's sample; only then, so every other
        # arm's head is unchanged
        pool = getattr(run, "_opt_probe", None)
        state["option_diag"] = {
            "window": copy.deepcopy(getattr(run, "_opt_window", None)),
            "probe": None if pool is None else {
                "row": int(pool["row"]),
                "keys": [float(k) for k in pool["keys"]],
                "obs": list(pool["obs"]),
                "rng": copy.deepcopy(pool["rng"].bit_generator.state)}}
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
    # Pre-3b checkpoints predate the ray_workers field; every one of them
    # was a pool-less run, so absent means 0 -- normalized here rather
    # than special-cased in the strict comparison below.
    state.setdefault("ray_workers", 0)
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
    # REWARD_GROWTH: the loaded records were clipped at the saved run's
    # ceiling; the next apply_reward_ceiling re-clips them all.
    run._reward_ceiling = None
    run.env_steps = state["env_steps"]
    run.trainer.training_steps = state["training_steps"]
    for k, v in (state.get("diag_stats") or {}).items():
        if k == "_redo_last_plan":
            import redo
            run._redo_last_plan = {redo.Unit(*f): list(idx) for f, idx in v}
        else:
            setattr(run, k, dict(v))
    if "resets" in state:
        # Test C: the warmup counts from the latest reset's step
        run.reset_log = [dict(r) for r in state["resets"]["log"]]
        run.hold = state["resets"]["hold"]
        run.held_ships = list(state["resets"].get("held_ships", []))
        muzero.LR_WARMUP_ORIGIN = (run.reset_log[-1]["training_steps"]
                                   if run.reset_log else 0)
    if "ramp" in state:
        run.ramp_trace = [list(e) for e in state["ramp"]["trace"]]
        run.ramp_windows = copy.deepcopy(state["ramp"].get("windows")
                                         or ramp_windows_empty())
    if "option_diag" in state:
        # Test O stage 4a: the partial row's option window and probe roots
        od = state["option_diag"]
        run._opt_window = copy.deepcopy(od["window"])
        p = od["probe"]
        if p is None:
            run._opt_probe = None
        else:
            rng = np.random.Generator(np.random.PCG64())
            rng.bit_generator.state = p["rng"]
            run._opt_probe = {"row": int(p["row"]),
                              "keys": np.asarray(p["keys"], dtype=np.float64),
                              "obs": list(p["obs"]), "rng": rng}
    # Test C2: the ramp's values at the restored step and reset origin (a
    # no-op without RAMP_STEPS)
    muzero.apply_ramps(run.trainer.training_steps)
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
    # Latent renormalisation is a plain attribute, invisible to a strict
    # state_dict load: a file trained with the other setting would load
    # cleanly and then run under the wrong latent scaling. Files predating
    # the flag were all trained with it on.
    file_renorm = (state.get("spec") or {}).get("LATENT_RENORMALIZE", True)
    live_renorm = getattr(trainer.representation, "renormalize_output", True)
    if file_renorm != live_renorm:
        raise SystemExit(
            f"{path}: trained with LATENT_RENORMALIZE={file_renorm}, but this "
            f"arm runs {live_renorm}")
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
    on the surviving records; eviction just decides which records survive.
    Under TOKEN_VOCAB the transition cap counts BASE steps (replay_units)."""
    while len(replay) > muzero.MAX_REPLAY:
        replay.pop(0)
    cap = muzero.MAX_REPLAY_TRANSITIONS
    if cap:
        total = sum(replay_units(t) for t in replay)
        while total > cap and len(replay) > 1:
            total -= replay_units(replay.pop(0))


def replay_units(traj):
    """A stored trajectory's size against MAX_REPLAY_TRANSITIONS: its record
    count -- or, under TOKEN_VOCAB, its BASE steps (each acted token's
    frames / TOKEN_BASE_FRAMES, plus the final record), so the buffer spans
    the same game time as the plain arm's whatever tokens were played
    (audit r1 B-M1: counted in records, ~200-decision token games made
    MAX_REPLAY the binding cap and the buffer ~3-7x the twin's)."""
    if muzero.TOKEN_VOCAB is None:
        return len(traj)
    return 1 + sum(muzero.token_frames(t["action"]) // muzero.TOKEN_BASE_FRAMES
                   for t in traj[:-1])


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


def run_expect(arm, seed, iterations, train_steps, episodes_per_iter, eval_every,
               ray_workers=0, student=None):
    """The validated recipe. EPISODE_CAP and eval_every are part of the
    experiment too: a resume that forgot --episode-cap would silently
    continue with truncated episodes stitched onto the first half's
    replay. ray_workers likewise (3b audit): pool mode IS the recipe --
    N episodes/iteration -- so a resume with a different worker count is
    a different experiment and must be refused, not continued."""
    out = {"arm": arm, "seed": seed, "env": ENV_NAME,
           "iterations": iterations, "train_steps_per_iter": train_steps,
           "episodes_per_iter": episodes_per_iter,
           "episode_cap": EPISODE_CAP, "eval_every": eval_every,
           "ray_workers": ray_workers}
    if student:
        # Curriculum P3 students (student_expect): the vocabulary and the
        # distillation set are the recipe too. Absent for every other arm.
        out.update(student)
    return out


def ray_scope_gate(spec):
    """--ray's supported-regime check, called FIRST in run_one -- before
    refuse_conflicting_state's --overwrite branch can destroy the pair's
    rows/checkpoint/weights and before any prefill compute. Refused loudly
    rather than run subtly wrong: non-paper arms reanalyse through the
    searcher, which shares the trainer's LIVE nets (after-training
    collection would build targets with W_it instead of W_{it-1});
    linear-support arms build non-default net geometry the actor's
    factories don't know."""
    if not muzero.PAPER_TARGETS:
        raise SystemExit("--ray currently requires a PAPER_TARGETS arm: "
                         "non-paper reanalysis reads the trainer's live "
                         "nets and would compute targets with post-update "
                         "weights under the pipeline.")
    if spec.get("_linear_support"):
        raise SystemExit("--ray does not support _linear_support arms: the "
                         "actor builds default-geometry networks and cannot "
                         "load linear-support weights.")


def redo_due(steps):
    """ReDo's pass schedule: every REDO_EVERY gradient steps, MID-period
    (REDO_EVERY = 1000: steps 1500, 2500, ..., 179500). Aligned passes
    (multiples of 1000) landed on every eval row (every 10k steps: each
    row's eval and snapshot saw a pass from that very step), every
    checkpoint (a resume then dropped that pass's pre/post grad-norm pair)
    and the final step (a pass whose only effect was on the saved final
    weights) -- round-3 audit. The first pass waits a full period, as in
    the paper."""
    every = muzero.REDO_EVERY
    return bool(every) and steps > every and steps % every == every // 2


def reset_due(run, last):
    """Test C: is a reset due at the end of this iteration? The next one
    (len(run.reset_log) + 1) is due once the training steps reach its
    multiple of RESET_EVERY_STEPS, while the run's env steps are at most
    RESET_UNTIL_ENV_STEPS -- never at the run's last iteration (nothing
    trains after it). Env steps only grow, so once past the cutoff no reset
    follows."""
    every = muzero.RESET_EVERY_STEPS
    if not every or last:
        return False
    return (run.trainer.training_steps >= (len(run.reset_log) + 1) * every
            and run.env_steps <= muzero.RESET_UNTIL_ENV_STEPS)


def fresh_networks(spec, key):
    """A fresh draw of the five nets (build_networks: each layer's own
    initializer) under a dedicated torch RNG seeded from `key`; the global
    torch streams are untouched (the nets initialize on the CPU, then
    move)."""
    seed = int(np.random.default_rng(key).integers(2**62))
    with torch.random.fork_rng(devices=[]):
        # the CPU generator only: torch.manual_seed would also reseed every
        # CUDA generator, which fork_rng(devices=[]) does not restore
        # (audit r1a LOW-1)
        torch.default_generator.manual_seed(seed)
        return build_networks(spec)


def bbf_reset(run, pipeline, it):
    """Test C's reset (BBF's, CURRICULUM "Test C"), at the end of iteration
    `it` -- after its row's snapshot and diagnostics, before its
    checkpoint:
      - every parameter of RESET_SHRINK_NETS (representation, dynamics)
        becomes RESET_SHRINK_FACTOR x itself + RESET_PERTURB_FACTOR x a
        fresh draw (BN gamma / beta included); their BN running statistics
        stay;
      - RESET_REINIT_NETS (prediction, value prefix, projector) become a
        fresh draw, running statistics included, and their optimizers'
        state (SGD momentum) is cleared; the shrunk nets' is kept (BBF
        keeps the copied layers');
      - the lagged target model, both buffers, becomes the reset online
        nets (our version of BBF's reset_target: A7), synced at this step;
      - the lr warmup restarts here (muzero.LR_WARMUP_ORIGIN), and with it
        Test C2's ramps (muzero.apply_ramps);
      - the actor keeps the pre-reset acting states (the pipeline's last
        ship) until RESET_HOLD_STEPS gradient steps later (run.hold), while
        the in-flight game's birth search and every later target use the
        reset nets.
    The fresh draw is keyed on (seed, reset index), so a resume that
    repeats a reset repeats it exactly."""
    tr = run.trainer
    index = len(run.reset_log) + 1
    steps = tr.training_steps
    # the actor's pre-reset weights: what the last sync shipped (cached at
    # this step)
    held = pipeline._acting_states() if pipeline is not None else None
    fresh = fresh_networks(run.spec, (run.seed, index, 0xB8F))
    with torch.no_grad():
        for n in muzero.RESET_SHRINK_NETS:
            new = dict(fresh[n].named_parameters())
            for name, p in getattr(tr, n).named_parameters():
                p.mul_(float(muzero.RESET_SHRINK_FACTOR)).add_(
                    new[name], alpha=float(muzero.RESET_PERTURB_FACTOR))
        for n in muzero.RESET_REINIT_NETS:
            getattr(tr, n).load_state_dict(fresh[n].state_dict())
    for n in muzero.RESET_REINIT_NETS:
        getattr(tr, dict(zip(NET_NAMES, OPT_NAMES))[n]).state.clear()
    del fresh
    tr._ensure_target_model()
    online = ((tr.representation, tr.target_representation),
              (tr.prediction, tr.target_prediction),
              (tr.dynamics, tr.target_dynamics),
              (tr.value_prefix, tr.target_value_prefix))
    snap = [{k: v.detach().clone() for k, v in net.state_dict().items()}
            for net, _ in online]
    for (_, target), s in zip(online, snap):
        target.load_state_dict(s)
        target.eval()
    tr._target_snapshot = snap
    tr._target_synced_at = steps
    if hasattr(tr, "invalidate_bootstrap_cache"):
        tr.invalidate_bootstrap_cache()
    muzero.LR_WARMUP_ORIGIN = steps
    # Test C2: the ramps restart here too (n RAMP_N_START, gamma
    # RAMP_GAMMA_START for the next step's batch, the in-flight game's birth
    # search, ...); a no-op without RAMP_STEPS
    muzero.apply_ramps(steps)
    if pipeline is not None:
        # every cache keyed on the step count now holds pre-reset weights
        pipeline._states_cache = (None, None)
        pipeline._acting_cache = (None, None)
        pipeline._target_states_cache = (None, None)
        pipeline._refresh_shipped_stamp = None
        if pipeline.pending is not None and muzero.BIRTH_SEARCH:
            # the in-flight game (played on the pre-reset weights) is
            # birth-searched by the reset nets
            pipeline._birth_states = pipeline._states()
    until = steps + muzero.RESET_HOLD_STEPS
    # "from": the reset's step -- a ship AT it (a resume's pipeline
    # construction from this iteration's checkpoint) re-sends what the
    # uninterrupted run shipped before the reset, not a held ship (audit
    # r2b MEDIUM-1)
    run.hold = ({"until": until, "from": steps, "held": held}
                if held is not None and muzero.RESET_HOLD_STEPS else None)
    run.reset_log.append({"index": index, "iteration": it,
                          "training_steps": steps,
                          "env_steps": run.env_steps})
    print(f"  [{run.arm} s{run.seed}] RESET {index} after iter {it} (step "
          f"{steps}, {run.env_steps} env steps): "
          f"{'+'.join(muzero.RESET_SHRINK_NETS)} shrunk x"
          f"{muzero.RESET_SHRINK_FACTOR} + {muzero.RESET_PERTURB_FACTOR} x "
          f"fresh; {'+'.join(muzero.RESET_REINIT_NETS)} re-initialized; "
          f"target reset; warmup restarts"
          + (f"; the actor holds the pre-reset weights until step {until}"
             if run.hold else ""), flush=True)


def ramp_step(run, it, first):
    """Test C2, at the top of every gradient step of train_block, before its
    batch is built: TD_STEPS and DISCOUNT must be the ramp's values at this
    step (training_steps - LR_WARMUP_ORIGIN) -- a site that moved the step
    count or the reset origin without muzero.apply_ramps fails here, loudly,
    instead of training on stale values -- and the first step of iteration
    `it` (`first`) is logged for its row: [it, the steps since the last
    reset, n, gamma] in run.ramp_trace. Nothing while RAMP_STEPS is 0
    (every other arm)."""
    if not muzero.RAMP_STEPS:
        return
    steps = run.trainer.training_steps
    cycle = steps - muzero.LR_WARMUP_ORIGIN
    want = muzero.ramp_values(cycle)
    have = (muzero.TD_STEPS, muzero.DISCOUNT)
    if have != want:
        raise RuntimeError(
            f"the ramp's values at step {steps} (reset origin "
            f"{muzero.LR_WARMUP_ORIGIN}) are {want}, but TD_STEPS / DISCOUNT "
            f"are {have}: something moved the step count or the origin "
            f"without muzero.apply_ramps")
    if first and it is not None:
        trace = getattr(run, "ramp_trace", None)
        if trace is None:
            trace = run.ramp_trace = []
        trace.append([int(it), int(cycle), int(want[0]), float(want[1])])


def ramp_windows_empty():
    """Test C2's per-row ramp windows (read (4), audit r1 LOW-2), as sums:
    "in" the gradient steps whose steps since the last reset are below
    RAMP_STEPS, "out" the rest; per window the steps, the steps scored (a
    finite value loss and target mean), and their sums."""
    return {k: {"steps": 0, "scored": 0, "value_loss": 0.0,
                "target_mean": 0.0} for k in ("in", "out")}


def note_ramp_window(run):
    """Test C2: the gradient step just taken (training_steps - 1, before
    after_train_step re-applies the ramp; the origin cannot move between
    the two) joins run.ramp_windows' "in" while its steps since the last
    reset are below RAMP_STEPS, else "out", scored with the trainer's
    (value loss, position-0 value target mean) for it
    (TrainingHelper._note_ramp_step), which is consumed here so no step is
    scored twice; a step without one, or with a non-finite one, is counted
    but not scored. Emitted and restarted per row (evaluate_diagnostics'
    "ramp_windows"), checkpointed."""
    cycle = run.trainer.training_steps - 1 - muzero.LR_WARMUP_ORIGIN
    wins = getattr(run, "ramp_windows", None)
    if wins is None:
        wins = run.ramp_windows = ramp_windows_empty()
    w = wins["in" if cycle < muzero.RAMP_STEPS else "out"]
    w["steps"] += 1
    last = getattr(run.trainer, "_ramp_last", None)
    run.trainer._ramp_last = None
    if (last is not None and np.isfinite(last[0])
            and np.isfinite(last[1])):
        w["scored"] += 1
        w["value_loss"] += float(last[0])
        w["target_mean"] += float(last[1])


def note_option_step(run):
    """Test O stage 4a: the gradient step just taken into the row's option
    window (Run._opt_window), from the trainer's _option_last, which is
    consumed here so no step counts twice (none: nothing). Every step adds
    its batch's host shares -- the mask share (NOT DEPARTURE 7's hit rate),
    the per-edge jump share, the span, DEPARTURE 3's birth-target counts --
    which hold whether or not the step was finite; the option loss and the
    jump / one-step rows' consistency sums join only where all are finite
    AND the step itself is (the step window's rule: the trainer's total
    loss and pre-clip grad norm, _last_step), loss_steps; a non-finite step
    is counted apart. Read after the step's own loss sync, so its device
    scalars are ready: the float() copies wait on nothing."""
    last = getattr(run.trainer, "_option_last", None)
    if last is None:
        return
    run.trainer._option_last = None
    w = run._option_window()
    w["steps"] += 1
    w["mask_share"] += float(last["option_mask_share"])
    js = list(last["option_jump_share"])
    if len(js) != len(w["jump_share"]):
        raise ValueError(f"a batch of {len(js)} unroll edges in an option "
                         f"window of {len(w['jump_share'])}")
    w["jump_share"] = [a + float(b) for a, b in zip(w["jump_share"], js)]
    w["span"] += float(last["option_span_mean"])
    for k in ("birth_pos", "birth_pos_n", "birth_root", "birth_root_n"):
        w[k] += int(last[k])
    vals = (float(last["option_loss"]), float(last["cons_jump_sum"]),
            float(last["cons_single_sum"]))
    # The step window's rule too (after_train_step: the trainer's _last_step
    # (total loss, pre-clip grad norm), either non-finite = a non-finite
    # step, which SKIP_NONFINITE_GRAD_STEPS skips): such a step's option
    # loss is not averaged in even where it is finite itself (audit r1 of
    # stage 4a). A trainer that has taken no step yet has no _last_step.
    step = getattr(run.trainer, "_last_step", None)
    if step is not None:
        vals = vals + (float(step[0]), float(step[1]))
    if not all(np.isfinite(v) for v in vals):
        w["nonfinite"] += 1
        return
    w["loss_steps"] += 1
    w["loss"] += vals[0]
    w["cons_jump"] += vals[1]
    w["cons_jump_n"] += int(last["cons_jump_n"])
    w["cons_single"] += vals[2]
    w["cons_single_n"] += int(last["cons_single_n"])


def after_train_step(run):
    """Per-gradient-step hook, called by every training loop right after the
    step counter advances. Today: Test C2's ramp (the step just taken into
    its ramp window, then muzero.apply_ramps at the new step: the next
    step's batch takes its values; nothing while RAMP_STEPS is 0), then
    ReDo on its redo_due() schedule, plus the
    step diagnostics that make ReDo's effect attributable (row-mean loss
    and pre-clip grad norm; mean grad norm over the 20 steps before vs after
    each pass -- round-2 design audit). Touches nothing on `run` while ReDo
    and the ramp are off, so every other arm is unchanged and test stubs
    need no new surface. Test O stage 4a (OPTION_LENGTH > 1): first the
    step's option numbers into the row's option window (note_option_step)."""
    if int(muzero.OPTION_LENGTH) > 1:
        note_option_step(run)
    if muzero.RAMP_STEPS:
        note_ramp_window(run)
        muzero.apply_ramps(run.trainer.training_steps)
    if not (muzero.REDO_EVERY or muzero.STEP_DIAGNOSTICS):
        return
    last = getattr(run.trainer, "_last_step", None)
    if last is not None:
        loss, gn = last[0], float(last[1])
        w = getattr(run, "_step_window", None) or {
            "n": 0, "loss": 0.0, "gn": 0.0, "passes": 0, "pre": 0.0,
            "post": 0.0, "nonfinite": 0}
        if not (np.isfinite(gn) and np.isfinite(loss)):
            # A skipped non-finite step (SKIP_NONFINITE_GRAD_STEPS): counted,
            # never averaged into the row means or the ReDo windows.
            w["nonfinite"] = w.get("nonfinite", 0) + 1
            run._step_window = w
            last = None
    if last is not None:
        w["n"] += 1
        w["loss"] += float(loss)
        w["gn"] += gn
        recent = getattr(run, "_gn_recent", [])
        recent = (recent + [gn])[-20:]
        run._gn_recent = recent
        post = getattr(run, "_gn_post", None)
        if post is not None:
            post.append(gn)
            if len(post) == 20:
                w["passes"] += 1
                w["pre"] += run._gn_pre
                w["post"] += float(np.mean(post))
                run._gn_post = None
        run._step_window = w
    if run.replay and redo_due(run.trainer.training_steps):
        run._gn_pre = float(np.mean(getattr(run, "_gn_recent", []) or [0.0]))
        run.redo_step()
        run._gn_post = []


@contextlib.contextmanager
def td_value_targets():
    """Batches built inside use the plain TD value target even in an SVE
    arm. evaluate_diagnostics' prediction-error panel runs under it, so its
    `value` / `value_control` stay comparable with every td arm and its
    batches never enter the per-row SVE window (SVE audit r1, M1). The
    training loss (step_loss_mean) is against the training target."""
    saved = muzero.VALUE_TARGET_MODE
    muzero.VALUE_TARGET_MODE = "td"
    try:
        yield
    finally:
        muzero.VALUE_TARGET_MODE = saved


def launch_gate(spec, use_ray, cpu_actor):
    """An arm may pin its launch mode (`_launch`): the factorization arms'
    ONLY difference is the launch mode, so a hand relaunch in the wrong mode
    must fail loudly instead of resuming a different experiment (factor
    audit r1 L3). Arms without `_launch` are unaffected."""
    want = spec.get("_launch")
    if not want:
        return
    have = {"ray": bool(use_ray), "cpu_actor": bool(cpu_actor)}
    bad = {k: (v, have.get(k)) for k, v in want.items() if have.get(k) != v}
    if bad:
        raise SystemExit(f"this arm pins its launch mode {want}; got "
                         f"{have} -- relaunch with the arm's own flags")


def sve_gate():
    """VALUE_TARGET_MODE's supported regime, checked at startup (SVE audit
    r1 L5): any other string would silently train on TD, and "mixed" only
    exists in the paper batch builder."""
    if muzero.VALUE_TARGET_MODE not in ("td", "mixed"):
        raise SystemExit("VALUE_TARGET_MODE must be 'td' or 'mixed', got "
                         f"{muzero.VALUE_TARGET_MODE!r}")
    if muzero.VALUE_TARGET_MODE == "mixed" and not muzero.PAPER_TARGETS:
        raise SystemExit("VALUE_TARGET_MODE 'mixed' requires PAPER_TARGETS "
                         "(the paper batch builder)")
    if muzero.VALUE_TARGET_MODE == "mixed" and not (
            int(muzero.SVE_START_TRAIN_STEPS) >= 0
            and int(muzero.SVE_FRESH_TRANSITIONS) >= 1):
        raise SystemExit("SVE needs SVE_START_TRAIN_STEPS >= 0 and "
                         "SVE_FRESH_TRANSITIONS >= 1")
    if muzero.VALUE_TARGET_MODE == "mixed":
        if muzero.SVE_SOURCE not in ("stored", "search"):
            raise SystemExit("SVE_SOURCE must be 'stored' or 'search', got "
                             f"{muzero.SVE_SOURCE!r}")
        if muzero.SVE_SOURCE == "search" and (
                int(muzero.SVE_SEARCH_SIMS) < 1 or muzero.USE_GUMBEL_SEARCH):
            raise SystemExit("SVE_SOURCE 'search' needs SVE_SEARCH_SIMS >= 1 "
                             "and PUCT (the vectorized engine is PUCT-only)")
        if muzero.SVE_SOURCE == "search" and not muzero.USE_TARGET_MODEL:
            # Without it the target nets would be built lazily at the first
            # stale batch (a mid-run torch RNG draw) and first "lag" nothing.
            raise SystemExit("SVE_SOURCE 'search' requires USE_TARGET_MODEL "
                             "(it searches through the lagged target nets)")


def birth_search_gate():
    """BIRTH_SEARCH's supported regime, checked at startup (before any
    --overwrite destruction) instead of at the first collect: it makes the
    targets the acting search would have recorded, so it needs the paper
    data path and either policy-only acting or SEARCH_ACT_BIRTH_TARGETS
    (a search picks the action, birth search still makes the targets); its
    engine is PUCT-only."""
    if not muzero.BIRTH_SEARCH:
        return
    if not muzero.PAPER_TARGETS:
        raise SystemExit("BIRTH_SEARCH requires PAPER_TARGETS: it builds the "
                         "paper records the acting search would have made")
    if muzero.USE_SEARCH_ACTING and not muzero.SEARCH_ACT_BIRTH_TARGETS:
        raise SystemExit("BIRTH_SEARCH replaces the acting search's targets: "
                         "set USE_SEARCH_ACTING False (policy-only acting) or "
                         "SEARCH_ACT_BIRTH_TARGETS (search picks the action, "
                         "birth search makes the targets)")
    if muzero.USE_GUMBEL_SEARCH:
        raise SystemExit("BIRTH_SEARCH's vectorized engine is PUCT-only; a "
                         "Gumbel arm would silently change the search")


def build_env(env_name, frameskip):
    """The lineage's env stack, ONE implementation: raw no-skip env ->
    NoopReset -> MaxAndSkip -> EpisodicLife when the wrapper flag is on,
    ALE-internal skip otherwise. Run.__init__ and the Ray self-play actor
    both build through here -- a wrapper change cannot silently give --ray
    a different environment."""
    import gymnasium as gym
    from EpisodicLifeEnv import EpisodicLifeEnv
    if muzero.USE_ATARI_WRAPPERS:
        from atari_wrappers import paper_atari_env
        return EpisodicLifeEnv(paper_atari_env(env_name, skip=frameskip or 4))
    kwargs = {"frameskip": frameskip} if frameskip else {}
    return EpisodicLifeEnv(gym.make(env_name, **kwargs))


class RayPipeline:
    """One-deep concurrent acting for run_one (--ray).

    Episode k is requested immediately after the weight sync that closes
    iteration k-1 and collected after iteration k's gradient steps, so it
    plays on a Ray actor WHILE the driver trains. Acting-policy staleness
    is IDENTICAL to the sequential harness (episode k plays on
    end-of-(k-1) weights); the one semantic delta is data incorporation,
    which lags one iteration (episode k first trains in k+1) -- unless
    RAY_EAGER_COLLECT, where train_block collects it mid-block as soon as
    the actor is done (it then trains from ~step 10-40 of k). Refresh,
    eval, LR events, and checkpoints all run AFTER collect, so they see
    exactly the buffer state the sequential harness would at that point.

    Saturate-arm exception to the after-collect rule above: its refresh
    runs PRE-collect (between training and collect, filling the acting
    window), deliberately one episode behind -- see run_one (round-2 F-D).

    Cancellation semantics preserved by construction: the checkpoint blob
    is unchanged and never contains the in-flight episode, so a kill at
    any moment resumes from the last checkpoint and merely replays the
    lost episode -- the sequential harness's exact cost. The first
    iteration after any resume (or from scratch) runs sequential-ordered,
    since nothing is in flight. Paired-seed parity with sequential runs
    is NOT preserved (actor RNG is process-local); ray-mode runs compare
    to ray-mode runs.
    """

    def __init__(self, run, spec, seed, iterations, train_steps,
                 episodes_per_iter, ray_workers=0, cpu_actor=False):
        # Scope is validated by ray_scope_gate at the TOP of run_one --
        # before --overwrite destroys the pair's data and before the
        # warm-start prefill spends compute (verification finding).
        import ray
        self.ray = ray
        if not ray.is_initialized():
            here = pathlib.Path(__file__).resolve().parent
            ray.init(include_dashboard=False, ignore_reinit_error=True,
                     logging_level="ERROR", log_to_driver=False,
                     runtime_env={"env_vars": {"PYTHONPATH": os.pathsep.join(
                         (str(here.parent), str(here)))}})
        self.run = run
        self.seed = seed
        self.episodes_per_iter = episodes_per_iter
        self.overrides = {k: v for k, v in spec.items()
                          if not k.startswith("_")}
        self.pending = None
        self.eval_actor = None
        self.pending_eval = None
        self._states_cache = (None, None)   # (training_steps, state dicts)
        self._acting_cache = (None, None)   # BN_RECAL_ACTING, same keying
        # BIRTH_SEARCH (round-1 audit HIGH-1): the weights the actor was
        # synced with, captured at request time, and a driver-side net set
        # (built once) that the collect-time birth search loads them into.
        self._birth_states = None
        self._birth_nets = None
        self._prior_nets = None       # BN_RECAL_ACTING: the actor's rep/pred
        self._birth_acting = None
        self.refresh_actor = None           # phase 3a, lazy like eval_actor
        self.pending_refresh = None         # ([(uid, future)], now_stamp)
        self._target_states_cache = (None, None)  # (target sync stamp, dicts)
        self._records_refs = {}             # uid -> ray.put ref (immutable data)
        self._refresh_shipped_stamp = None  # last target stamp sent to actor
        self.refresh_workers = []           # 3b.5: CPU workers via server
        self.refresh_applied = 0            # trajectories applied (this process)
        self.refresh_skipped = 0            # dispatch points passed while cooking
        if muzero.REFRESH_OFFLOAD:
            # Zeroed stats from the first row on: a wedged actor that
            # never applies must read as applied=0 in the jsonl, not as
            # a run with no offload at all (verification-round F1).
            self._note_refresh_stats()
        self.server = None                  # phase 3b inference server
        self.workers = []
        if ray_workers:
            import ray_actors
            # Logical-CPU backstop (3b.5 audit F3; the primary refusal
            # sits in run_one BEFORE --overwrite/prefill): every actor
            # here reserves one Ray CPU, and past the cluster's count the
            # lazily-created eval/refresh actors would PEND forever and
            # the driver's first ray.get on them would block silently.
            need = ray_workers + 2 + (2 if muzero.REFRESH_OFFLOAD else 0)
            avail = int(self.ray.cluster_resources().get("CPU", 0))
            if need > avail:
                raise SystemExit(
                    f"--ray-workers {ray_workers} needs {need} Ray CPUs "
                    f"but the cluster has {avail}; lower --ray-workers.")
            # GPU: the server's 0.5 share is the pool mode's ONLY Ray-GPU
            # reservation -- eval and refresh are CPU workers through it
            # (3b.5); the driver trains outside Ray's accounting.
            gpu = 0.5 if torch.cuda.is_available() else 0
            srv_cls = self.ray.remote(num_gpus=gpu, max_concurrency=256)(
                ray_actors.InferenceServer)
            self.server = srv_cls.remote(
                self.overrides, ACTIONS,
                {"TEMPERATURE_TOTAL_STEPS": muzero.TEMPERATURE_TOTAL_STEPS,
                 "OFFPOLICY_DECAY_STEPS": muzero.OFFPOLICY_DECAY_STEPS},
                pool_size=ray_workers)
            self.workers = [self._make_actor(seed_extra=100 + w,
                                             server=self.server)
                            for w in range(ray_workers)]
            self.actor = None
        else:
            self.actor = self._make_actor(seed_extra=0,
                                          force_cpu=cpu_actor)
        self.cpu_actor = cpu_actor
        self.sync()

    def _make_actor(self, seed_extra, server=None, server_set="live",
                    force_cpu=False):
        """ONE construction path for every acting actor. Small FIXED gpu
        share: the driver trains outside Ray's accounting, and actors at
        mcts_gpu_share-style fractions could sum past 1.0 -- the excess
        would never schedule and its first ray.get would block forever.
        Phase 3b workers get a server handle and NO gpu (their forwards
        run on the InferenceServer)."""
        import ray_actors
        gpu = 0 if (server is not None or force_cpu) else (
            0.2 if torch.cuda.is_available() else 0)
        actor_cls = self.ray.remote(num_gpus=gpu)(ray_actors.SelfPlayActor)
        # The run-length-keyed schedule values are COMPUTED once by the
        # driver's _run_length_globals and shipped as values -- the actor
        # never re-derives them (no duplicated 0.3 formula).
        schedule = {"TEMPERATURE_TOTAL_STEPS": muzero.TEMPERATURE_TOTAL_STEPS,
                    "OFFPOLICY_DECAY_STEPS": muzero.OFFPOLICY_DECAY_STEPS,
                    # Test B: the env-keyed schedules' target (None: off);
                    # the count itself arrives with every sync
                    "SCHEDULE_ENV_TARGET": muzero.SCHEDULE_ENV_TARGET}
        # seed*8191: plain seed+extra made seed s, worker w+1 share RNG
        # streams with seed s+1, worker w (3b audit F7).
        return actor_cls.remote(self.overrides, ACTIONS,
                                self.seed * 8191 + seed_extra, schedule,
                                ENV_NAME, FRAMESKIP, EPISODE_CAP, server,
                                server_set, force_cpu)

    def _states(self):
        """CPU state dicts of the four synced nets, cached per
        training_steps so the eval request at a boundary reuses the sync's
        serialization instead of rebuilding identical dicts."""
        stamp = self.run.trainer.training_steps
        if self._states_cache[0] != stamp:
            # clone(): on CPU-only builds, detach().cpu() ALIASES the live
            # parameters, so a cached dict would silently track future
            # weight mutations; cloning turns the stamp check from an
            # invariant into a guarantee.
            self._states_cache = (stamp, {
                n: {k: v.clone() if torch.is_tensor(v) else v
                    for k, v in _cpu_state(
                        getattr(self.run.trainer, n).state_dict()).items()}
                for n in ("representation", "dynamics", "prediction",
                          "value_prefix")})
        return self._states_cache[1]

    def _acting_states(self):
        """What the actor and the eval actor play with: _states(), except
        that under BN_RECAL_ACTING the representation and prediction BN
        running statistics are re-estimated on the newest replay frames
        (the BN stabilizer). Cached per training_steps like _states();
        birth search keeps _states() (raw statistics: it searches)."""
        states = self._states()
        if not muzero.BN_RECAL_ACTING:
            return states
        stamp = self.run.trainer.training_steps
        if self._acting_cache[0] != stamp:
            frames = bn_calibration_frames(self.run.replay,
                                           muzero.BN_RECAL_FRAMES)
            recal = ({} if frames is None else
                     bn_recalibrated_states(self.run.trainer, frames))
            self._acting_cache = (stamp, {**states, **recal})
        return self._acting_cache[1]

    def _actor_states(self):
        """What the self-play ACTOR plays with: _acting_states(), except
        during a Test C hold (run.hold, after a reset: until its step) --
        then the pre-reset acting states it keeps. The eval actor and the
        snapshots always take _acting_states()."""
        hold = getattr(self.run, "hold", None)
        if hold is not None and self.run.trainer.training_steps < hold["until"]:
            return hold["held"]
        return self._acting_states()

    def snapshot_acting_states(self):
        """For save_weight_snapshot: the shipped representation /
        prediction acting states under BN_RECAL_ACTING, else None."""
        if not muzero.BN_RECAL_ACTING:
            return None
        acting = self._acting_states()
        return {n: acting[n] for n in ("representation", "prediction")}

    def sync(self):
        steps = self.run.trainer.training_steps
        # Test B: the env-keyed schedules' count rides along (only then:
        # every other arm's actors see the unchanged two-argument call)
        env = ((self.run.env_steps,) if muzero.SCHEDULE_ENV_TARGET
               else ())
        # Test C: past its hold's end the held states are dropped; within
        # it, the ship's step is recorded (held_ships: once per step, so a
        # resume's re-ship at its checkpoint's step counts once; and only
        # AFTER the reset's own step -- a resume from the reset iteration's
        # checkpoint re-ships at that step what the run shipped before the
        # reset: audit r2b MEDIUM-1)
        hold = getattr(self.run, "hold", None)
        if hold is not None and steps >= hold["until"]:
            self.run.hold = None
        elif hold is not None and steps > hold.get("from", -1):
            ships = getattr(self.run, "held_ships", None)
            if ships is None:
                ships = self.run.held_ships = []
            if steps not in ships:
                ships.append(steps)
        if self.server is not None:
            # One weight ship for the whole pool; workers only mirror the
            # temperature counter (their set_weights skips empty dicts).
            self.ray.get(self.server.set_weights.remote(
                self._actor_states(), steps))
            self.ray.get([w.set_weights.remote({}, steps, *env)
                          for w in self.workers])
        else:
            self.ray.get(self.actor.set_weights.remote(
                self._actor_states(), steps, *env))

    def request(self, it):
        # training_steps captured at REQUEST time: sequential reanalysis of
        # this episode would run at the TOP of the next iteration with
        # exactly this count, and paper_targets stamps born_step from it --
        # collecting after the gradient steps must not age the stamp.
        if self.server is not None:
            futures = [w.play.remote(self.seed * 100003 + it * 1009 + e)
                       for e, w in enumerate(self.workers)]
        else:
            futures = [self.actor.play.remote(
                self.seed * 100003 + it * 1009 + e)
                for e in range(self.episodes_per_iter)]
        self.pending = (futures, self.run.trainer.training_steps)
        # Test B: the iteration these lives belong to (the own-play log's
        # file)
        self._request_it = it
        # Under BIRTH_SEARCH the episode's targets must come from the
        # weights the actor plays with -- the ones just synced -- not the
        # post-training weights collect() would otherwise see (the pattern
        # ray_scope_gate refuses for non-paper arms). _states() is keyed on
        # the step count and returns clones, so this is the synced set.
        self._birth_states = self._states() if muzero.BIRTH_SEARCH else None
        # BN_RECAL_ACTING: the rep/pred the actor SAMPLES from this game
        # (the synced acting set; during a Test C hold the held one) -- for
        # the behaviour prior only.
        self._birth_acting = (self._actor_states()
                              if muzero.BIRTH_SEARCH and muzero.BN_RECAL_ACTING
                              else None)

    def _birth_net_set(self, states):
        """Load `states` into the persistent driver-side birth-search nets
        (deep copies of the trainer's four nets, so device, geometry and
        the LATENT_RENORMALIZE attribute match) and return them, eval."""
        if self._birth_nets is None:
            self._birth_nets = {
                n: copy.deepcopy(getattr(self.run.trainer, n))
                for n in ("representation", "dynamics", "prediction",
                          "value_prefix")}
        for n, net in self._birth_nets.items():
            net.load_state_dict(states[n])
            net.eval()
        return self._birth_nets

    def _prior_net_set(self, states):
        """Persistent driver-side copies of the trainer's representation /
        prediction loaded with the actor's acting states (BN_RECAL_ACTING),
        eval mode: the prior the actor sampled from, for the behaviour
        diagnostics."""
        if getattr(self, "_prior_nets", None) is None:
            self._prior_nets = {n: copy.deepcopy(getattr(self.run.trainer, n))
                                for n in ("representation", "prediction")}
        for n, net in self._prior_nets.items():
            net.load_state_dict(states[n])
            net.eval()
        return self._prior_nets

    @staticmethod
    def _own_arrays(record):
        """Copy plasma-backed arrays out of Ray's object store: stored by
        reference they pin store memory for the buffer's lifetime and
        arrive read-only."""
        return tuple(np.array(f) if isinstance(f, np.ndarray) else f
                     for f in record)

    def acting_ready(self):
        """Non-blocking: are ALL in-flight episode futures done? The
        saturation loop polls this between refresh candidates, and
        train_block (RAY_EAGER_COLLECT) between gradient steps."""
        if self.pending is None:
            return True
        futures, _ = self.pending
        ready, _ = self.ray.wait(futures, num_returns=len(futures),
                                 timeout=0)
        return len(ready) == len(futures)

    def collect(self, admit=None):
        futures, steps_at_request = self.pending
        results = self.ray.get(futures)
        self.pending = None
        birth_nets = (self._birth_net_set(self._birth_states)
                      if self._birth_states is not None else None)
        self._birth_states = None
        acting = getattr(self, "_birth_acting", None)
        prior_nets = (self._prior_net_set(acting) if acting is not None
                      else None)
        self._birth_acting = None
        # The request-time count flows as an EXPLICIT stamp -- no rewinding
        # of the live trainer counter, so no window in which the two
        # "current step" sources (trainer vs searcher mirror) disagree.
        for res in results:
            traj, terminated, steps = res[:3]
            self.run.env_steps += steps
            if len(res) > 3:
                # Test B (OWN_PLAY_LOG): the life at raw-step resolution,
                # logged as played -- before anything could drop it
                self.run.own_play.write(self._request_it, traj, res[3],
                                        terminated, steps,
                                        self.run.env_steps)
            # Test-6 admission control (async driver only; None
            # everywhere else): the decision runs BEFORE the append so
            # a dropped episode never evicts stored data to make room
            # for frames that are then discarded. env_steps always
            # counts -- the interaction happened.
            if admit is not None and not admit(steps):
                continue
            traj = [self._own_arrays(r) for r in traj]
            self.run.replay.append(self.run.reanalyse(
                traj, terminated=terminated, born_step=steps_at_request,
                birth_nets=birth_nets, birth_prior_nets=prior_nets))
            evict_replay(self.run.replay)
        # Test B: the env-keyed schedules read the generation's count
        # (inert unless muzero.SCHEDULE_ENV_TARGET is set) -- advanced
        # AFTER the lives' reanalysis, whose behaviour diagnostics must see
        # the count the actor played at (the last sync's; audit r1 A-L1)
        muzero.SCHEDULE_ENV_STEPS = self.run.env_steps

    def request_eval(self, it, episodes, diagnostics, write_row, elapsed_at):
        """Phase 2: greedy eval episodes run on a SECOND actor, concurrent
        with training. Diagnostics were computed on the driver at the
        boundary (they need the trainer's nets at W_it); the row is written
        when the episodes come back. Every counter INCLUDING elapsed_s is
        captured at request time. The never-skip-a-row guarantee is upheld
        by PREVENTION: run_one drains any pending row before every
        checkpoint, so no checkpoint can ever precede a row it implies."""
        if self.eval_actor is None:
            # 3b.5: in pool mode eval is one more CPU worker through the
            # server's dedicated "eval" set -- no private nets, no GPU
            # share, and FREEZE semantics preserved: the eval set is
            # written only here, never by sync(), so in-flight eval
            # episodes cannot see the next iteration's weights (3b.5
            # audit F2), and the final row ships W_N explicitly even
            # though the last sync() is skipped (F1).
            self.eval_actor = self._make_actor(
                seed_extra=5000, server=self.server, server_set="eval",
                force_cpu=getattr(self, "cpu_actor", False))
        steps = self.run.trainer.training_steps
        if self.server is not None:
            self.ray.get(self.server.set_weights.remote(
                self._acting_states(), steps, which="eval"))
            self.ray.get(self.eval_actor.set_weights.remote({}, steps))
        else:
            self.ray.get(self.eval_actor.set_weights.remote(
                self._acting_states(), steps))
        seeds = [self.seed * 100003 + it * 613 + 500000 + e
                 for e in range(episodes)]
        self.pending_eval = (
            it, self.eval_actor.eval_episodes.remote(seeds), diagnostics,
            write_row, self.run.trainer.training_steps, self.run.env_steps,
            elapsed_at)

    def drain_eval(self):
        if self.pending_eval is None:
            return
        (it, future, diagnostics, write_row,
         tsteps, esteps, elapsed_at) = self.pending_eval
        self.pending_eval = None
        write_row(it, {**eval_summary(self.ray.get(future)), **diagnostics},
                  tsteps, esteps, elapsed_at)

    # ---- phase 3a: refresh offload (swap-on-ready) ----

    def _target_states(self):
        """CPU state dicts of the four LAGGED TARGET nets, cached per
        target-sync stamp (the target model is what refresh searches
        with; _states above is the LIVE nets for acting/eval)."""
        self.run.trainer._ensure_target_model()
        stamp = self.run.trainer._target_synced_at
        if self._target_states_cache[0] != stamp:
            self._target_states_cache = (stamp, {
                n: {k: v.clone() if torch.is_tensor(v) else v
                    for k, v in _cpu_state(getattr(
                        self.run.trainer,
                        "target_" + n).state_dict()).items()}
                for n in ("representation", "dynamics", "prediction",
                          "value_prefix")})
        return self._target_states_cache[1]

    def request_refresh(self):
        """Dispatch one refresh batch (POLICY_REFRESH_COUNT candidate
        trajectories) to the refresh actor; lazy actor creation, at most
        one batch in flight (a still-cooking batch means this dispatch
        point passes and is COUNTED -- see refresh_skipped). Target nets
        ship only when their sync stamp changed. The driver's
        training_steps at dispatch is the rotation stamp `now`, matching
        the inline path, which reads the counter at refresh time.

        Trajectory data ships as ObjectRefs put ONCE per trajectory (3a
        audit F1: rebuilding record dicts per dispatch serialized the
        whole buffer -- gigabytes -- inside the driver every cadence; the
        shipped fields obs/action/reward/flags are immutable for a
        trajectory's lifetime, so one put is exact)."""
        if not self.run.replay:
            return
        if self.pending_refresh is not None:
            self.refresh_skipped += 1
            self._note_refresh_stats()
            return
        candidates = self.run._refresh_candidates(muzero.POLICY_REFRESH_COUNT)
        if not candidates:
            return
        if self.refresh_actor is None and not self.refresh_workers:
            import ray_actors
            schedule = {
                "TEMPERATURE_TOTAL_STEPS": muzero.TEMPERATURE_TOTAL_STEPS,
                "OFFPOLICY_DECAY_STEPS": muzero.OFFPOLICY_DECAY_STEPS}
            if self.server is not None:
                # 3b.5: refresh as CPU workers through the server's
                # "target" set -- refresh draws on the whole pooled GPU
                # instead of one actor's slice, and parallelizes across
                # trajectories. Two workers saturate the current
                # coverage; the count is deliberately modest so acting
                # workers keep CPU headroom.
                cls = self.ray.remote(num_gpus=0)(ray_actors.RefreshActor)
                self.refresh_workers = [
                    cls.remote(self.overrides, ACTIONS,
                               self.seed * 8191 + 9000 + w, schedule,
                               self.server)
                    for w in range(2)]
            else:
                gpu = 0.2 if torch.cuda.is_available() else 0
                cls = self.ray.remote(num_gpus=gpu)(ray_actors.RefreshActor)
                self.refresh_actor = cls.remote(
                    self.overrides, ACTIONS, self.seed * 8191 + 9000,
                    schedule)
        self.run.trainer._ensure_target_model()
        stamp = self.run.trainer._target_synced_at
        if stamp != self._refresh_shipped_stamp:
            if self.server is not None:
                # One target-set ship to the server covers every refresh
                # worker (3b.5 weight-sync collapse). The workers' counter
                # mirrors ride along -- unread by refresh today, but a
                # search that ever keys on training progress must not see
                # a frozen 0 (3b.5 audit F4).
                self.ray.get(self.server.set_weights.remote(
                    self._target_states(), self.run.trainer.training_steps,
                    which="target"))
                self.ray.get([w.set_weights.remote(
                    {}, self.run.trainer.training_steps)
                    for w in self.refresh_workers])
            else:
                self.ray.get(self.refresh_actor.set_weights.remote(
                    self._target_states(), self.run.trainer.training_steps))
            self._refresh_shipped_stamp = stamp
        now = self.run.trainer.training_steps
        # Prune record refs for evicted trajectories so the object store
        # holds exactly the live buffer's worth.
        live = {t[0].get("traj_uid") for t in self.run.replay if t}
        self._records_refs = {u: r for u, r in self._records_refs.items()
                              if u in live}
        jobs = []
        for ti in candidates:
            traj = self.run.replay[ti]
            idxs = self.run._eligible_indices(traj)
            if not idxs:
                # Inline parity (3a audit F3): a no-eligible candidate
                # still rotates -- priorities recomputed, stamp advanced
                # -- or it would occupy a candidate slot forever.
                self.run._apply_refresh(traj, {}, now)
                continue
            # uid, not id(): CPython reuses addresses, so a raw id could
            # match a NEW trajectory allocated where an evicted one lived
            # and apply results to the wrong data. The uid rides the head
            # record (harmlessly into checkpoints).
            uid = traj[0].setdefault("traj_uid", uuid.uuid4().hex)
            if uid not in self._records_refs:
                self._records_refs[uid] = self.ray.put(
                    [{"obs": t["obs"], "action": t["action"],
                      "reward": t["reward"],
                      "truncated": t.get("truncated", False),
                      "has_policy_target": t.get("has_policy_target", True)}
                     for t in traj])
            sink = (self.refresh_workers[len(jobs) % len(self.refresh_workers)]
                    if self.refresh_workers else self.refresh_actor)
            jobs.append((uid, sink.refresh.remote(
                self._records_refs[uid], idxs)))
        if jobs:
            self.pending_refresh = (jobs, now)

    def _note_refresh_stats(self, now=None):
        """Accounting the 3a audit demanded: without it, a slow actor
        silently degrades '5x coverage' to ~1x at high staleness and the
        A/B attributes the wrong thing. Refreshed at EVERY touch point --
        pipeline construction, skipped dispatch, and apply -- so rows
        carry live counters even when a batch cooks past the eval
        cadence or the actor wedges before its first apply
        (verification-round F1). Surfaced by evaluate_diagnostics."""
        stats = {
            "refresh_applied": self.refresh_applied,
            "refresh_skipped_dispatches": self.refresh_skipped,
        }
        if now is not None:
            # How stale the just-applied batch's search weights are at
            # application time (inline is bounded by the target sync
            # interval; the offload adds cook time on top).
            stats["refresh_lag_steps"] = (
                self.run.trainer.training_steps - now)
        else:
            prev = getattr(self.run, "_refresh_offload_stats", {})
            if "refresh_lag_steps" in prev:
                stats["refresh_lag_steps"] = prev["refresh_lag_steps"]
        self.run._refresh_offload_stats = stats

    def poll_refresh(self):
        """Swap-on-ready: apply any completed refresh results whose
        trajectory is still in the buffer (uid check -- an evicted
        trajectory's results are dropped: wasted work, never corruption).
        Non-blocking; called at ONE fixed point in the driver loop, whose
        single-threadedness is the mutex. Application goes through the
        same _apply_refresh as the inline path."""
        if self.pending_refresh is None:
            return
        jobs, now = self.pending_refresh
        alive = {t[0].get("traj_uid"): t for t in self.run.replay
                 if t and t[0].get("traj_uid")}
        remaining = []
        applied = 0
        for uid, fut in jobs:
            ready, _ = self.ray.wait([fut], timeout=0)
            if not ready:
                remaining.append((uid, fut))
                continue
            results = self.ray.get(fut)
            traj = alive.get(uid)
            if traj is None:
                continue
            self.run._apply_refresh(
                traj, {int(i): (np.array(p), float(v))
                       for i, (p, v) in results.items()}, now)
            applied += 1
        if applied:
            self.refresh_applied += applied
            self._note_refresh_stats(now)
        self.pending_refresh = (remaining, now) if remaining else None

    def shutdown(self):
        self.drain_eval()
        for actor in (self.actor, self.eval_actor, self.refresh_actor,
                      self.server, *self.workers, *self.refresh_workers):
            if actor is not None:
                try:
                    self.ray.kill(actor)
                except Exception:
                    pass


def run_one(arm, seed, iterations, episodes_per_iter, train_steps, eval_every, out,
            checkpoint_every=10, resume=False, overwrite=False,
            init_weights=None, use_ray=False, ray_workers=0,
            cpu_actor=False, snapshot_weights=False, distill_set=None,
            stop_env_steps=None):
    # Test B: an env-stopped generation's planned run (the schedules' scale
    # and the 4.3 decay) is its nominal length, not the --iterations cap
    planned = ARMS[arm].get("_nominal_iterations") or iterations
    with arm_config(arm) as spec, ramp_globals(), \
            _run_length_globals(planned, train_steps), \
            token_actions(), env_schedule(stop_env_steps), warmup_origin():
        if use_ray:
            ray_scope_gate(spec)
        launch_gate(spec, use_ray, cpu_actor)
        eager_collect_gate(use_ray)
        prior_temp_gate()
        bn_recal_gate(use_ray)
        search_act_birth_gate()
        birth_search_gate()
        sve_gate()
        token_gate(spec, use_ray, cpu_actor, ray_workers)
        option_gate(spec, use_ray, cpu_actor, ray_workers,
                    distill_set=distill_set, init_weights=init_weights)
        reward_growth_gate(use_ray, cpu_actor, ray_workers)
        student_gate(spec, distill_set, snapshot_weights)
        env_stop_gate(spec, stop_env_steps, use_ray, cpu_actor, iterations,
                      snapshot_weights)
        reset_gate(spec, use_ray, cpu_actor, ray_workers, train_steps,
                   init_weights)
        ramp_gate(spec)
        if muzero.RAMP_STEPS:
            print(f"  [{arm} s{seed}] RAMP: n {muzero.RAMP_N_START} -> "
                  f"{muzero.RAMP_N_END}, gamma {muzero.RAMP_GAMMA_START} -> "
                  f"{muzero.RAMP_GAMMA_END!r} (BBF's shapes) over the "
                  f"{muzero.RAMP_STEPS} gradient steps after each reset (and "
                  f"the start); the end values after", flush=True)
        if spec.get("_student") and init_weights:
            raise SystemExit("a curriculum P3 student starts from scratch or "
                             "from its distillation set, not --init-weights")
        if muzero.REWARD_GROWTH and (muzero.REWARD_GROWTH_START
                                     == muzero.REWARD_GROWTH_CEILING):
            print(f"  [{arm} s{seed}] REWARD_GROWTH: raw / "
                  f"{muzero.REWARD_UNIT} clipped at a CONSTANT ceiling "
                  f"{muzero.REWARD_GROWTH_CEILING}", flush=True)
        elif muzero.REWARD_GROWTH:
            print(f"  [{arm} s{seed}] REWARD_GROWTH: raw / "
                  f"{muzero.REWARD_UNIT} clipped at a ceiling "
                  f"{muzero.REWARD_GROWTH_START} -> "
                  f"{muzero.REWARD_GROWTH_CEILING} (log-linear to "
                  f"{muzero.REWARD_GROWTH_END:.0%} of the run)", flush=True)
        if muzero.SCHEDULE_ENV_TARGET:
            # Test B (audit r1 A-NIT9): the env-keyed schedules, the offset
            # then only the prefill's birth step
            print(f"  [{arm} s{seed}] SCHEDULE_ENV_TARGET "
                  f"{muzero.SCHEDULE_ENV_TARGET}: the within-run schedules "
                  f"follow this run's env-step progress (planned run "
                  f"{muzero.TEMPERATURE_TOTAL_STEPS} steps); the run stops "
                  f"at the iteration that passes it"
                  + (f"; SCHEDULE_OFFSET_STEPS "
                     f"{muzero.SCHEDULE_OFFSET_STEPS}: the prefill's 4.3 age "
                     f"starts at the first online step"
                     if muzero.SCHEDULE_OFFSET_STEPS else ""), flush=True)
        elif muzero.SCHEDULE_OFFSET_STEPS:
            print(f"  [{arm} s{seed}] SCHEDULE_OFFSET_STEPS "
                  f"{muzero.SCHEDULE_OFFSET_STEPS}: the within-run schedules "
                  f"(and the prefill's 4.3 age) read the online steps",
                  flush=True)
        if muzero.TOKEN_VOCAB is not None:
            # main()'s banner printed the env's action count; this arm's
            # action space is its vocabulary (audit r1 B-NIT).
            print(f"  [{arm} s{seed}] TOKEN_VOCAB: {ACTIONS} tokens "
                  f"(ALE action, frames) {list(muzero.TOKEN_VOCAB)}",
                  flush=True)
        if muzero.REDO_EVERY:
            # Fail at startup -- before any --overwrite deletion (round-2
            # audit LOW-3) and not at step REDO_EVERY: import redo and
            # validate its layer map against nets built exactly as this arm
            # builds them. (Consumes RNG, but Run() reseeds and a resume
            # restores the checkpointed streams.)
            import redo
            probe = build_networks(spec)
            redo.recyclable_units({n: probe[n] for n in redo.ALL_NETS},
                                  include=tuple(muzero.REDO_NETS))
            del probe
        if ray_workers and not use_ray:
            raise SystemExit("--ray-workers requires --ray")
        if cpu_actor and not use_ray:
            raise SystemExit("--cpu-actor requires --ray")
        if spec.get("REFRESH_SATURATE"):
            if not use_ray:
                raise SystemExit(
                    "REFRESH_SATURATE arms need --ray: saturation fills "
                    "the slack while the actor plays, and sequential "
                    "mode has no such window (use the scheduled-quota "
                    "vtree arm instead)")
            if not cpu_actor:
                raise SystemExit(
                    "REFRESH_SATURATE requires --cpu-actor: with a GPU "
                    "acting actor, saturation's searches contend with "
                    "acting inference and lengthen the very episode "
                    "window they fill")
            if not spec.get("REFRESH_VTREE"):
                raise SystemExit("REFRESH_SATURATE requires REFRESH_VTREE")
            # Guard and entropy priority are DELIBERATELY not required
            # (2026-09-02, Kai's staged attribution): the one-pass
            # seen-set bounds every saturate call regardless -- the
            # termination round proved one pass with the guard neutered
            # -- and an ineligible rotation pick costs a stamp, not a
            # search. Staleness-ordered guard-free saturation is the
            # minimal-delta first test; entropy+guard layer on next.
            if spec.get("REFRESH_OFFLOAD"):
                raise SystemExit(
                    "REFRESH_SATURATE and REFRESH_OFFLOAD are exclusive: "
                    "saturation's loop skips the cadence block, leaving "
                    "the offload silently inert (audit F4)")
        if spec.get("ASYNC_LOOP"):
            if not (use_ray and cpu_actor):
                raise SystemExit(
                    "ASYNC_LOOP arms need --ray --cpu-actor: the phase-5 "
                    "driver's whole point is a continuously-playing CPU "
                    "actor beside a GPU train+refresh lane")
            if ray_workers:
                raise SystemExit(
                    "ASYNC_LOOP is the single-latency-lane driver; it "
                    "excludes the --ray-workers pool")
            if not spec.get("REFRESH_VTREE"):
                raise SystemExit("ASYNC_LOOP requires REFRESH_VTREE (the "
                                 "sweep is vectorized GPU search)")
            if spec.get("REFRESH_SATURATE") or spec.get("REFRESH_OFFLOAD"):
                raise SystemExit(
                    "ASYNC_LOOP replaces the saturate/offload refresh "
                    "drivers; the arm must set both off")
            if init_weights:
                raise SystemExit(
                    "ASYNC_LOOP does not support --init-weights yet: the "
                    "warm-start ramp is keyed to the iteration loop")
        if cpu_actor and spec.get("REFRESH_OFFLOAD"):
            raise SystemExit(
                "--cpu-actor promises the whole GPU to the driver, but "
                "this arm's REFRESH_OFFLOAD would create a 0.2-GPU "
                "refresh actor; use a REFRESH_VTREE arm instead")
        if cpu_actor and ray_workers:
            raise SystemExit("--cpu-actor is the latency-lane mode "
                             "(single CPU env actor); it excludes the "
                             "--ray-workers pool")
        if ray_workers:
            # Budget refusal BEFORE refuse_conflicting_state's --overwrite
            # branch can destroy the pair's data and before prefill compute
            # (verification round F-1 -- the same ordering invariant
            # ray_scope_gate documents). os.cpu_count() equals a fresh
            # local Ray cluster's CPU total; the in-pipeline check remains
            # as the post-init backstop.
            need = ray_workers + 2 + (2 if muzero.REFRESH_OFFLOAD else 0)
            avail = os.cpu_count() or 0
            if need > avail:
                raise SystemExit(
                    f"--ray-workers {ray_workers} needs {need} CPUs "
                    f"(workers + server + eval"
                    f"{' + 2 refresh workers' if muzero.REFRESH_OFFLOAD else ''}) "
                    f"but this machine has {avail}; lower --ray-workers.")
            # Phase 3b RECIPE CHANGE, stated loudly: with a worker pool
            # each iteration collects one episode PER WORKER through the
            # shared inference server -- episodes-per-iter is superseded.
            print(f"  [{arm} s{seed}] worker pool: {ray_workers} env/tree "
                  f"workers, {ray_workers} episodes/iteration "
                  f"(--episodes-per-iter superseded)", flush=True)
        ck = resume_path(arm, seed)
        expect = run_expect(arm, seed, iterations, train_steps,
                            episodes_per_iter, eval_every,
                            ray_workers=ray_workers,
                            student=student_expect(spec, distill_set))
        if stop_env_steps:
            # Test B: the env-step stop is the recipe too (absent for every
            # other run, whose recipe is unchanged)
            expect["stop_env_steps"] = int(stop_env_steps)
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
        if muzero.OWN_PLAY_LOG:
            run.own_play = OwnPlayLog(own_play_dir(arm, seed))
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
        if spec.get("_student") == "distill" and start_it == 0:
            # Curriculum P3's distillation step: the master's re-tokenized
            # games become the first replay, then the offline phase -- once
            # per timeline (a resume restores its checkpoint's replay and
            # nets instead; a restart without a checkpoint runs it again).
            # Before the pipeline exists: its first sync ships the trained
            # weights and BN states calibrated on these frames.
            if run.replay or run.trainer.training_steps:
                # (no checkpoint lands at iteration 0 today; one that did
                # would make this a double prefill -- audit r2 A-NIT3)
                run.env.close()
                raise SystemExit("the distillation prefill needs a fresh "
                                 "run (empty replay, step 0)")
            import distill
            try:
                header, lives = distill.read_set(distill_set)
                kept, gone, gone_units = distill_prefill(run, lives)
                del lives
                print(f"  [{arm} s{seed}] distillation set: {kept} master "
                      f"lives in the replay "
                      f"({sum(replay_units(t) for t in run.replay)} base "
                      f"steps; {gone} oldest lives / {gone_units} base steps "
                      f"evicted to the caps; master "
                      f"{header['master_snapshot']})", flush=True)
                distill_offline(run, arm, seed, spec, snapshot_weights)
            except BaseException:
                run.env.close()
                raise
        # Test C2: the ramp's values at the run's step and reset origin --
        # step 0 at origin 0 from scratch (warmup_origin), a resume's
        # restored pair (load_run_checkpoint applied them too) -- before
        # anything reads TD_STEPS or DISCOUNT (the first collect's birth
        # search); after_train_step and bbf_reset keep them current from
        # here. Every other arm: nothing (not even a read of the run).
        if muzero.RAMP_STEPS:
            muzero.apply_ramps(run.trainer.training_steps)
        def write_row(row_it, metrics, train_steps_at, env_steps_at,
                      elapsed_at=None):
            eff_skip = (FRAMESKIP or 4) if muzero.USE_ATARI_WRAPPERS else FRAMESKIP
            record = {"arm": arm, "seed": seed,
                      "env": ENV_NAME, "frameskip": eff_skip,
                      "wrappers": bool(muzero.USE_ATARI_WRAPPERS),
                      **({"init_weights": init_name} if init_name else {}),
                      "iteration": row_it,
                      "train_steps": train_steps_at,
                      "env_steps": env_steps_at,
                      "elapsed_s": (elapsed_at if elapsed_at is not None
                                    else round(time.time() - started, 1)),
                      **metrics}
            with open(out, "a") as fh:
                fh.write(json.dumps(record) + "\n")
            print(f"  [{arm} s{seed}] iter {row_it:>3}  return {metrics['return_mean']:>7.1f}  "
                  f"prefix {metrics['prefix']:>6.2f} / control {metrics['control']:>6.2f}  "
                  f"per-step {metrics['per_step']:>6.2f}  "
                  f"H(pi) {metrics.get('policy_entropy', 0.0):>5.2f}", flush=True)

        pipeline = (RayPipeline(run, spec, seed, iterations, train_steps,
                                episodes_per_iter, ray_workers=ray_workers,
                                cpu_actor=cpu_actor)
                    if use_ray else None)

        if pipeline and muzero.ASYNC_LOOP:
            # ---- Phase 5: the chunked asynchronous driver ------------
            # "Iteration" dissolves. The loop interleaves chunks: poll
            # the in-flight episode (collect + immediately request the
            # next -- the actor NEVER idles waiting on the driver),
            # ~ASYNC_TRAIN_CHUNK gradient steps under the
            # MAX_TRAIN_PER_ENV_STEP cap, then the refresh sweep's next
            # trajectories (freshness contract: the whole buffer turns
            # over at least once per ASYNC_SWEEP_TRAIN_STEPS gradient
            # steps; newcomers are refreshed at collect before they can
            # be sampled). Rows/checkpoints re-key to the gradient-step
            # unit u = training_steps // train_steps, so the family's
            # per-row axis survives; the chunk is capped at the next
            # boundary, so rows land on EXACT multiples. Loop-as-mutex:
            # no threads, same as every driver in this file. "The actor
            # never blocks" means precisely: a new episode is requested
            # at every collect, and collection latency is bounded by one
            # loop iteration (a boundary's eval drain included) -- a
            # finished episode can sit briefly before collection, and at
            # shutdown at most one in-flight episode is discarded
            # (legacy's exact cancellation class; audit F3).
            total_train = iterations * train_steps
            u_done = start_it
            # Episode-seed key: only uniqueness matters, but the gap
            # must outrun any dead timeline's episode count past its
            # last checkpoint (audit F5) -- acting is unthrottled, so
            # 1000 was crossable in principle; 10^6 is not.
            ep_counter = start_it * 1_000_000
            episodes_proc = 0
            episodes_dropped = 0
            owed = 0.0                    # sweep debt, in trajectories
            floor = muzero.ASYNC_MIN_TRAIN_PER_ENV_STEP
            # Admitted-env is not checkpointed; on resume, initialize to
            # the steady-state value the invariant itself enforces
            # (train/floor), so post-resume admission continues at the
            # same rate instead of feasting or starving.
            admitted_env = (int(run.trainer.training_steps / floor)
                            if floor else 0)

            def collect_and_relaunch():
                nonlocal ep_counter, episodes_proc, episodes_dropped, \
                    admitted_env
                admitted_now = 0

                def admit(ep_steps):
                    nonlocal admitted_env, episodes_dropped, admitted_now
                    if async_admit(floor, run.trainer.training_steps,
                                   admitted_env, ep_steps,
                                   not run.replay):
                        admitted_env += ep_steps
                        admitted_now += 1
                        return True
                    episodes_dropped += 1
                    return False

                pipeline.collect(admit=admit if floor else None)
                if muzero.ASYNC_REFRESH_ON_COLLECT:
                    # The newcomers are the last append(s); eviction
                    # only ever removes from the front. Under admission
                    # control only ADMITTED episodes exist to refresh.
                    tail = admitted_now if floor else episodes_per_iter
                    if tail:
                        run.refresh_trajs(run.replay[-tail:])
                # Dropped or admitted, the actor never waits: sync and
                # request unconditionally.
                pipeline.sync()
                ep_counter += 1
                episodes_proc += 1
                pipeline.request(ep_counter)

            def async_boundary(v):
                if (muzero.LR_DECAY_AT
                        and v == max(1, int(iterations * muzero.LR_DECAY_AT))):
                    muzero.decay_optimizer_lr(
                        (run.trainer.rep_optimizer,
                         run.trainer.dyn_optimizer,
                         run.trainer.pred_optimizer,
                         run.trainer.prefix_optimizer,
                         run.trainer.proj_optimizer),
                        muzero.LR_DECAY_FACTOR)
                    print(f"  [{arm} s{seed}] lr decayed "
                          f"x{muzero.LR_DECAY_FACTOR} at u {v}", flush=True)
                if v == iterations:
                    ckpt = RESULTS / f"weights{TAG}-{arm}-seed{seed}.pth"
                    torch.save(weights_blob(run, arm, seed, spec), ckpt)
                    print(f"  saved {ckpt.name}", flush=True)
                if (eval_every and v % eval_every == 0) or v == iterations:
                    if snapshot_weights:
                        save_weight_snapshot(run, arm, seed, spec, v)
                    pipeline.drain_eval()
                    diag = run.evaluate_diagnostics()
                    # The phase-5 spec's required per-row realized ratio.
                    diag["train_per_env_step"] = round(
                        run.trainer.training_steps
                        / max(1, run.env_steps), 3)
                    diag["episodes_since_resume"] = episodes_proc
                    diag["episodes_dropped"] = episodes_dropped
                    if floor and admitted_env:
                        diag["train_per_admitted_env"] = round(
                            run.trainer.training_steps / admitted_env, 3)
                    run.zero_sweep_window()
                    pipeline.request_eval(v, 8, diag, write_row,
                                          round(time.time() - started, 1))
                    if v == iterations:
                        pipeline.drain_eval()
                if (checkpoint_every and v % checkpoint_every == 0
                        and v != iterations):
                    pipeline.drain_eval()
                    save_run_checkpoint(run, ck, {
                        **expect, "iteration": v,
                        "training_steps": run.trainer.training_steps,
                        "elapsed_s": round(time.time() - started, 1),
                        "init_weights": init_name})
                    print(f"  [{arm} s{seed}] checkpoint at u {v} "
                          f"({ck.stat().st_size / 1e9:.1f} GB)", flush=True)

            # Prime: play the first episode sequentially (nothing is in
            # flight from scratch or after a resume), then acting is
            # continuous -- every later collect immediately requests.
            pipeline.request(ep_counter)
            collect_and_relaunch()
            while run.trainer.training_steps < total_train:
                run.searcher.network_manager.training_steps = \
                    run.trainer.training_steps
                if pipeline.acting_ready():
                    collect_and_relaunch()
                allowed = (int(muzero.MAX_TRAIN_PER_ENV_STEP
                               * run.env_steps)
                           - run.trainer.training_steps)
                chunk = min(muzero.ASYNC_TRAIN_CHUNK,
                            total_train - run.trainer.training_steps,
                            allowed,
                            # Never cross a u boundary mid-chunk: rows
                            # then land on EXACT train-step multiples
                            # (the toy smoke showed two boundaries
                            # crossed by one chunk sharing counters).
                            train_steps - (run.trainer.training_steps
                                           % train_steps))
                if chunk > 0:
                    run.set_train_mode(True)
                    # Prefetcher scoped to the chunk: collect/refresh only
                    # ever run between chunks, after it is closed.
                    with train_batches(run, chunk) as next_batch:
                        for _ in range(chunk):
                            asyncio.run(run.trainer.train_step(next_batch()))
                            run.trainer.training_steps += 1
                            after_train_step(run)
                    run.set_train_mode(False)
                    owed += (len(run.replay) * chunk
                             / muzero.ASYNC_SWEEP_TRAIN_STEPS)
                due = int(owed)
                if due > 0:
                    run.refresh_sweep(due)
                    # due = int(owed) is consumed every accrual, so the
                    # residue is always < 1 trajectory (audit F4: no
                    # buffer-size clamp is needed -- debt can never
                    # exceed one trajectory, let alone one sweep).
                    owed -= due
                if chunk <= 0 and due <= 0 and not pipeline.acting_ready():
                    # Train lane capped, contract satisfied: the only
                    # remaining work is the in-flight episode. Block on
                    # it (the actor itself never blocks -- collect
                    # relaunches immediately).
                    collect_and_relaunch()
                u_now = run.trainer.training_steps // train_steps
                while u_done < u_now:
                    u_done += 1
                    async_boundary(u_done)
            if checkpoint_every or resume:
                mark_run_completed(ck, expect)
            pipeline.shutdown()
            run.env.close()
            return

        for it in range(start_it + 1, iterations + 1):
            # Paper acting mode reads training progress for its temperature
            # schedule from the searcher's manager; mirror the trainer's count.
            run.searcher.network_manager.training_steps = run.trainer.training_steps
            # Test B: the env-keyed schedules' count (a resume restored
            # run.env_steps; collect keeps it current)
            muzero.SCHEDULE_ENV_STEPS = run.env_steps
            # REWARD_GROWTH: this iteration's ceiling, over the whole buffer,
            # before anything is collected or trained.
            if muzero.REWARD_GROWTH:
                run.apply_reward_ceiling()
            if pipeline:
                primed_sequential = pipeline.pending is None
                if primed_sequential:
                    # From scratch or first iteration after a resume:
                    # nothing in flight, so play NOW (sequential ordering
                    # for this one iteration), then the pipeline primes.
                    pipeline.request(it)
                    pipeline.collect()
            else:
                for _ in range(episodes_per_iter):
                    traj, _, _, terminated = run.play_episode()
                    run.replay.append(run.reanalyse(traj, terminated=terminated))
                    evict_replay(run.replay)

            steps_this_iter = train_steps
            ramp_iters = min(WARMSTART_RAMP_ITERS, iterations)
            if init_name and it <= ramp_iters:
                steps_this_iter = max(1, train_steps * it // ramp_iters)
            run.set_train_mode(True)
            # TRAIN_PREFETCH: batch i+1 assembles on a worker thread while
            # the GPU trains on batch i (bit-identical batch sequence to the
            # serial loop; see muzero.BatchPrefetcher). The prefetcher is
            # scoped to each contiguous block -- nothing mutates the replay
            # buffer while one is open (RAY_EAGER_COLLECT closes the block
            # before its mid-iteration collect; see train_block).
            train_block(run, steps_this_iter, pipeline=pipeline,
                        init_name=init_name, consistency_base=consistency_base,
                        it=it)
            run.set_train_mode(False)

            if (pipeline and muzero.REFRESH_SATURATE
                    and muzero.PAPER_TARGETS):
                # Saturation (audit F1: this block must sit BETWEEN the
                # training block and collect, where episode `it` is
                # genuinely still in flight -- placed after request(it+1)
                # it would poll a just-launched episode and serialize
                # the pipeline for its whole duration): refresh fills
                # the GPU's idle window until the actor's episode is
                # ready. Refresh sees the PRE-collect buffer (episode
                # `it` joins afterwards) -- one episode behind, which
                # entropy priority makes immaterial (the freshest
                # trajectory is rarely a candidate). On the primed
                # first iteration pending is None -> floor-only.
                # Floor = the BASE schedule (POLICY_REFRESH_COUNT per
                # cadence), not a token 1: early iterations have NO
                # slack (short episodes finish during training), and a
                # 1-per-cadence floor under-refreshed the first rows
                # BELOW archfix's 2-trajs/iter base -- inverting the
                # coverage delta the saturate arms exist to test
                # (measured, pong-satstale iter 5: refresh_last_iter 0).
                # Saturation adds on top once episodes outgrow training.
                floor = (muzero.POLICY_REFRESH_COUNT
                         if (muzero.POLICY_REFRESH_EVERY and
                             it % muzero.POLICY_REFRESH_EVERY == 0)
                         else 0)
                run.refresh_saturate(pipeline.acting_ready, floor=floor)

            if pipeline:
                if not primed_sequential and pipeline.pending is not None:
                    # Episode `it`, played on the actor during the gradient
                    # steps above, joins the buffer BEFORE refresh/eval/
                    # checkpoint -- those see the sequential harness's state.
                    # (RAY_EAGER_COLLECT may already have collected it mid-
                    # block; pending is then None.)
                    pipeline.collect()
            # Test B: an env-stopped generation's LAST iteration is the one
            # in which its own env steps pass the stop (its episodes all
            # collected above); every other run's is `iterations`.
            last = it == iterations or bool(
                stop_env_steps and run.env_steps >= stop_env_steps)
            if pipeline:
                if not last:
                    # No final-iteration sync: nothing plays after the last
                    # collect, and request_eval ships fresh weights itself.
                    pipeline.sync()
                    pipeline.request(it + 1)

            if pipeline and muzero.REFRESH_OFFLOAD and muzero.PAPER_TARGETS:
                # Phase 3a swap-on-ready: THE fixed apply point in the
                # driver's loop (single-threaded, so the loop is the
                # mutex). Runs every iteration so finished batches land
                # promptly; dispatch stays on the refresh cadence below.
                pipeline.poll_refresh()
            if (muzero.PAPER_TARGETS and muzero.POLICY_REFRESH_EVERY
                    and it % muzero.POLICY_REFRESH_EVERY == 0
                    and not (pipeline and muzero.REFRESH_SATURATE)):
                if muzero.REFRESH_VTREE:
                    # Phase 4: fast enough inline (tens of seconds at 5x
                    # coverage) that no offload is needed; explicit
                    # precedence over REFRESH_OFFLOAD.
                    run.refresh_targets_vtree(muzero.POLICY_REFRESH_COUNT)
                elif pipeline and muzero.REFRESH_OFFLOAD:
                    pipeline.request_refresh()
                else:
                    run.refresh_targets(muzero.POLICY_REFRESH_COUNT)

            if muzero.LR_DECAY_AT and it == max(1, int(iterations * muzero.LR_DECAY_AT)):
                muzero.decay_optimizer_lr(
                    (run.trainer.rep_optimizer, run.trainer.dyn_optimizer,
                     run.trainer.pred_optimizer, run.trainer.prefix_optimizer,
                     run.trainer.proj_optimizer), muzero.LR_DECAY_FACTOR)
                print(f"  [{arm} s{seed}] lr decayed x{muzero.LR_DECAY_FACTOR} "
                      f"at iter {it}", flush=True)

            if last:
                # Save the trained weights. The first full ablation discarded
                # 4.9 hours of training because nothing here wrote them out,
                # which made it impossible to watch an agent afterwards or to
                # resume from one.
                # The projector rides along (weights_blob) for future warm
                # starts: a FRESH projector's consistency gradients helped
                # scramble continuation attempt 1.
                ckpt = RESULTS / f"weights{TAG}-{arm}-seed{seed}.pth"
                torch.save(weights_blob(run, arm, seed, spec), ckpt)
                print(f"  saved {ckpt.name}", flush=True)

            if (eval_every and it % eval_every == 0) or last:
                if snapshot_weights:
                    # W_it: the weights this row's diagnostics and (in
                    # --ray, the eval actor's synced copy) episodes see.
                    save_weight_snapshot(
                        run, arm, seed, spec, it,
                        acting_states=(pipeline.snapshot_acting_states()
                                       if pipeline else None))
                if pipeline:
                    # Any previous late row first, so rows stay file-ordered.
                    pipeline.drain_eval()
                    # Diagnostics on the driver at W_it (seconds); episodes
                    # on the eval actor, concurrent with the next
                    # iteration's training. EVERY row -- the final one
                    # included -- goes through the same actor regime
                    # (seeded resets, dedicated env), so the series never
                    # mixes evaluation distributions; the final row drains
                    # immediately so completion cannot race it.
                    pipeline.request_eval(it, 8, run.evaluate_diagnostics(),
                                          write_row,
                                          round(time.time() - started, 1))
                    if last:
                        pipeline.drain_eval()
                else:
                    write_row(it, run.evaluate(),
                              run.trainer.training_steps, run.env_steps)

            # Test C: a reset at this iteration's end -- after its row's
            # snapshot and diagnostics (row `it` is the pre-reset network,
            # its eval actor already synced), before its checkpoint (which
            # then holds the reset and the hold)
            if reset_due(run, last):
                bbf_reset(run, pipeline, it)

            # After the eval/record block, so a resumed run never repeats or
            # skips a jsonl row: it continues at it+1 with row `it` written.
            if checkpoint_every and it % checkpoint_every == 0 and not last:
                if pipeline:
                    # PREVENTION of skipped rows: no checkpoint may be
                    # written while an eval row it implies is in flight --
                    # this holds for ANY checkpoint/eval cadence and any
                    # later resume mode (the audit's misaligned-cadence and
                    # cross-mode-resume holes both close here, which is why
                    # there is no resume-heal to maintain). Known cost: on
                    # cadences where checkpoints land ON eval iterations,
                    # this drain blocks on the row requested moments ago
                    # and the eval overlap serializes for those boundaries
                    # -- correctness bought with throughput, deliberately.
                    pipeline.drain_eval()
                save_run_checkpoint(run, ck, {
                    **expect, "iteration": it,
                    "training_steps": run.trainer.training_steps,
                    "elapsed_s": round(time.time() - started, 1),
                    "init_weights": init_name})
                print(f"  [{arm} s{seed}] checkpoint at iter {it} "
                      f"({ck.stat().st_size / 1e9:.1f} GB)", flush=True)
            if last:
                # Test B: an env-stopped generation ends here (its final
                # weights, snapshot and row written above)
                break
        # Also under --resume with checkpointing off: otherwise the loaded
        # checkpoint survives completion and every later --resume replays
        # the same tail.
        if checkpoint_every or resume:
            mark_run_completed(ck, expect)
        if pipeline:
            pipeline.shutdown()
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
    ap.add_argument("--ray", action="store_true",
                    help="concurrent acting: a Ray actor plays each "
                         "iteration's episode(s) while the driver trains "
                         "(one-deep pipeline; see RayPipeline). Checkpoints, "
                         "resume, refresh, and eval semantics are unchanged; "
                         "paired-seed parity with sequential runs is not.")
    ap.add_argument("--cpu-actor", action="store_true",
                    help="phase 4 latency lane: the environment actor "
                         "(and eval actor) run LOCAL CPU nets with the "
                         "sequential engine -- no GPU share, no server "
                         "-- leaving the whole GPU to training and "
                         "vectorized refresh. Requires --ray; excludes "
                         "--ray-workers. Recipe-neutral (device only).")
    ap.add_argument("--ray-workers", type=int, default=0, metavar="N",
                    help="phase 3b: N CPU env/tree workers share ONE "
                         "GPU inference server that batches their network "
                         "calls across trees (per-tree search stays "
                         "strictly sequential). RECIPE CHANGE: each "
                         "iteration collects N episodes (one per worker); "
                         "--episodes-per-iter is superseded. Requires "
                         "--ray. 0 = the validated single-actor pipeline.")
    ap.add_argument("--snapshot-weights", action="store_true",
                    help="also save the nets at every eval row, to "
                         "RESULTS/snapshots<TAG>-<arm>-seed<N>/"
                         "weights-iterNNN.pth (weights-file format, ~50 MB "
                         "each) -- mid-run models for offline probes. "
                         "Read-only: training is unaffected.")
    ap.add_argument("--token-vocab", default=None, metavar="PATH",
                    help="curriculum P3 student arms only: the student "
                         "vocabulary file experiments/distill.py wrote "
                         "(vocab.json); required by them")
    ap.add_argument("--distill-set", default=None, metavar="PATH",
                    help="the -student-distill arm only: the master's "
                         "re-tokenized games and their targets "
                         "(distill.py's set.pkl), pre-filled into the "
                         "replay before the offline phase; ignored on a "
                         "resume past it (the checkpoint holds the replay), "
                         "but its sha256 must match the checkpoint's")
    ap.add_argument("--stop-env-steps", type=int, default=None,
                    metavar="N",
                    help="Test B's env-stopped arms only (required by "
                         "them): the run ends at the iteration in which "
                         "its own env steps pass N; the within-run "
                         "schedules follow its env-step progress")
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

    if args.token_vocab and not args.plot_only:
        for arm in args.arms:
            student_arm(arm, args.token_vocab)

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
                        init_weights=args.init_weights, use_ray=args.ray,
                        ray_workers=args.ray_workers,
                        cpu_actor=args.cpu_actor,
                        snapshot_weights=args.snapshot_weights,
                        distill_set=args.distill_set,
                        stop_env_steps=args.stop_env_steps)
    plot(out, RESULTS)


if __name__ == "__main__":
    main()
