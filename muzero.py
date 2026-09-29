import itertools

import torch
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import numpy as np
import cv2
import random
import ray
from collections import deque
import scalar_categorical
from memory import print_memory_diagnostics
from networks import (DynamicsNetwork, LatentProjector, PredictionNetwork,
                      RepresentationNetwork, ValuePrefixNetwork)
import logging
import os
import uuid
import asyncio
import sys
import time
from EpisodicLifeEnv import EpisodicLifeEnv

# Atari envs ship in the ale-py plugin. gymnasium >= 1.0 requires explicit
# registration; earlier versions register them on import of ale_py.
try:
    import ale_py
    gym.register_envs(ale_py)
except (ImportError, AttributeError):
    pass

TRAINING_BATCHES  = 64
MCTS_BATCHES = 8
TOTAL_MCTS_SIMS = 32
MAX_DEPTH = 8
# Reanalysis topology. False: one search per root, in ascending order, each
# spending TOTAL_MCTS_SIMS in MCTS_BATCHES-wide waves WITHIN its tree (the
# shipped shape). True: windowed COMMUNAL search -- COMMUNAL_WINDOW_ROOTS
# roots run concurrently in lockstep, ONE simulation per root per round (the
# papers' sequential schedule inside each tree), with the GPU batch formed
# ACROSS roots, so throughput matches the batched shape without seven of
# eight selections running on stale statistics. Statistics on the shared
# chain tree are communal: traffic that earlier or concurrent searches sent
# through a node stays in its counts and is read out rather than
# baseline-subtracted (AlphaZero's self-play tree reuse reads out inherited
# visits the same way). The chain backup's bookkeeping visit is still
# subtracted -- it is a value-warming device, not search traffic, and
# unsubtracted it nudges every policy target toward the behaviour action.
# (COMMUNAL_EDGE_VISITS below also removes the behaviour child's own ROOT
# visits, which this legacy readout counts as if they were traffic.)
# PUCT only; with USE_GUMBEL_SEARCH on, reanalysis falls back to sequential.
USE_COMMUNAL_REANALYSIS = False
# Edge-visit accounting for the communal window (2026-09-15, from Kai's
# "does root n+1's search affect root n's policy?" question). The chain
# node n+1 IS node n's child for the behaviour action, and
# backup_simulation increments EVERY path node's visit_count, the search
# root included -- so while root n+1 spends its S simulations, node n's
# behaviour child gains S visits that no selection at node n ever made.
# Legacy (False, archfix's semantics) reads them out and lets them shrink
# the behaviour child's exploration bonus during root n's own search.
# Measured on r12's checkpoint (window 16, S=50): slots 0-14 of every
# window read ~50 successor root visits on the behaviour action, a floor
# of about half the target's mass; slot 15, whose successor waits in the
# next window, none. True: every node also counts its ROOT visits
# (root_visits), and both PUCT and the readout use EDGE visits,
# visit_count - root_visits -- selections that actually crossed the
# parent->child edge, MCTS's N(s,a). Node VALUES stay communal (Q over
# every backup through the node, root n+1's own included). The
# vectorized engine mirrors the flag; the parity gate holds either way.
# PUCT only, like the communal window itself: the Gumbel paths (per-root,
# ascending, never concurrent) keep raw visit counts.
COMMUNAL_EDGE_VISITS = False
# 16, raised from 8 on 2026-08-30 (Kai): probe on arch1@96k weights, 150
# positions, 2 noise seeds -- width-16 targets differ from width-8 by
# LESS than width-8's own noise floor (cross-KL 0.172 vs 0.222) at
# identical entropy, 12% faster even on CPU. Width 32 drifted just past
# the floor (0.259) for 5 ms/pos more; not taken. Each root still gets
# exactly ONE simulation per round -- widening adds trees per GPU batch,
# never concurrent sims within a tree.
COMMUNAL_WINDOW_ROOTS = 16
MAX_GAMEPLAY = 20
MAX_REPLAY = 20
# Optional TRANSITION-count cap on the replay buffer (0 = off). MAX_REPLAY
# caps trajectories, which makes the resident window GROW as the agent
# improves: 85 early-training Pong episodes ~= 100k transitions, but
# episodes lengthen with skill -- same cap, double the memory (the 13.7 GB
# checkpoint wedge of 2026-08-26). EfficientZero's own buffer capacity is
# measured in transitions (core/config.py transition_num), so this is the
# faithful unit. Eviction drops oldest WHOLE trajectories while the total
# exceeds the cap -- never partial ones, so unroll windows and n-step
# targets never cross an eviction seam -- and always keeps the newest.
# Both caps apply; whichever binds first evicts.
MAX_REPLAY_TRANSITIONS = 0
IMAGE_RESHAPE = (84,84)
STEPS_UNTIL_MCTS = 500
# 128 channels at 6x6 = 4,608 values (18 KB/node), against 64 at 21x21 = 28,224
# (110 KB). Channels widen as resolution falls, so the residual blocks run at
# small spatial sizes: roughly 3x less convolution work for 6x less memory.
LATENT_CHANNELS = 128
# Spatial size the representation network emits: 84 -> 42 -> 21 -> 11 -> 6
# through four stride-2 convolutions. Kept here so consumers stop hardcoding a
# divisor of the observation size; test_networks asserts it matches reality.
LATENT_HW = 6
# Architecture knobs (audit findings 5-7), arm-overridable and read at
# construction time. Defaults are the repo's own dimensions; the
# effzero-repro-smallnet arm sets the reference's (trunk 1 / dyn 1 /
# 16-ch heads with a [32] hidden / "plane" actions). The residual dynamics
# skip and the strided-conv BatchNorms (findings 5-6) are NOT knobs -- they
# are unconditional fixes in networks.py.
REP_TRUNK_BLOCKS = 2
DYN_BLOCKS = 2
PRED_HEAD_CHANNELS = 32
PRED_FC_HIDDEN = None
REWARD_HEAD_CHANNELS = 32
# How the dynamics conv sees the action: "embed" (latent-width learned
# embedding, the repo's original), "plane" (EffZero's single a/|A| plane),
# "onehot" (|A| binary planes; the curriculum branch's encoding).
ACTION_ENCODING = "embed"
UNROLL_STEPS = 5
# gamma. Effective planning horizon is roughly 1/(1-DISCOUNT), so ~333 steps
# here. Lowering it shortens credit assignment and converges faster to a worse
# policy -- useful as a fast A/B arm, but note it also shrinks the horizon the
# value prefix exists to fix, so a low value will understate that change.
DISCOUNT = 0.997
REWARD_CLIP = 400
# Raw-unit ceilings. The value prefix is a discounted SUM over UNROLL_STEPS
# rewards, so it needs a ceiling that covers the sum rather than one reward.
VALUE_PREFIX_CEILING = REWARD_CLIP * UNROLL_STEPS
VALUE_CEILING = 5000
# Supports are half-widths in h-TRANSFORMED space, derived from those ceilings.
# h() compresses hard -- a raw 2000 becomes ~46 -- which is what lets a fixed
# bin count cover the worst case AND still resolve values near zero. With a
# linear support the bins were 40 wide while realistic prefixes were 0-5, so
# every target landed in bin 0 and the head could not train (measured in
# benchmarks/RESULTS.md).
REWARD_SUPPORT = scalar_categorical.support_for(REWARD_CLIP)
VALUE_PREFIX_SUPPORT = scalar_categorical.support_for(VALUE_PREFIX_CEILING)
VALUE_SUPPORT = scalar_categorical.support_for(VALUE_CEILING)

# Target bin width in transformed space. h() shrinks the range but not the bin
# COUNT, so a fixed 51 bins still left 1.8-wide bins over the prefix support --
# raw 0 to 7 in one bin, against realistic prefixes of 0-5. Bins have to scale
# with the transformed range. MuZero uses 601 bins over +/-300, i.e. width 1.0;
# 0.5 buys finer resolution near zero, which is where the rewards actually are.
H_BIN_WIDTH = 0.5

def bins_for(support):
    """Odd bin count giving ~H_BIN_WIDTH resolution across +/- support.

    Odd so the centre bin sits exactly on zero -- a value of zero should encode
    as one bin, not split across two.
    """
    n = int(2 * support / H_BIN_WIDTH) + 1
    return n if n % 2 == 1 else n + 1

REWARD_BINS = bins_for(REWARD_SUPPORT)
VALUE_PREFIX_BINS = bins_for(VALUE_PREFIX_SUPPORT)
VALUE_BINS = bins_for(VALUE_SUPPORT)
# h() on BOTH sides of the categorical codec: encode (train_step reads this at
# call time, so the ablation can override it per arm) and decode (the networks
# take it at construction via transform=...). The first "no-transform" ablation
# arm only changed support sizes -- h() was still applied, so it measured a
# mis-scaled h-encoding rather than the linear one; this flag exists so that
# arm can test what its label says. Change both sides together or the codec
# stops being an inverse pair.
USE_H_TRANSFORM = True
VALUE_PREFIX_HIDDEN = 512
# Whether the tree takes its edge rewards from the value-prefix LSTM or from
# the per-step reward head. Both heads are always trained; this selects which
# one the search actually consumes, so the two can be compared end to end.
#
# Default False on the 2026-08-13 ablation (5 arms x 2 seeds, ~15k steps each,
# benchmarks/RESULTS.md), where the prefix head sat at exactly 1.000x a
# predict-zero control -- collapsed -- while the per-step head reached 0.24x.
#
# THAT VERDICT IS STALE (fidelity audit, 2026-08-23). It predates the SimSiam
# projector, appendix-G latent renormalisation, AdamW + grad clipping, and the
# zero-init heads. On every archived effzero-repro run since, the prefix head
# reads 0.003x-0.09x control across all eval rows -- the best-learned head in
# the system -- and the repro arms drive the tree with it (flag True). The
# default stays False only until a controlled prefix-vs-per-step pair on the
# post-search-fix build settles it; do not cite the old collapse as the reason.
USE_VALUE_PREFIX_IN_TREE = False
# How many LSTM steps the in-tree prefix accumulates before its state is
# reset to zeros -- EfficientZero's lstm_horizon_len (5 on Atari), matching
# the horizon the head is TRAINED over (one UNROLL_STEPS window, state reset
# per unroll in train_step). Without the reset, tree depths past the horizon
# queried the LSTM outside its trained regime and every edge reward there
# was extrapolation. At a reset boundary the edge reward is the fresh
# prefix itself, not a difference (EffZero's is_reset semantics).
PREFIX_LSTM_HORIZON = UNROLL_STEPS

# ---------------------------------------------------------------------------
# Loss weighting and gradient scaling (MuZero appendix G)
#
# Every loss term used to be summed raw and every unroll step contributed at
# full weight, so a K-step unroll multiplied the gradient reaching the
# representation by roughly K, and the value head -- whose target has by far the
# widest range -- dominated the others.
# ---------------------------------------------------------------------------
VALUE_LOSS_WEIGHT = 0.25        # MuZero's Atari setting
POLICY_LOSS_WEIGHT = 1.0
REWARD_LOSS_WEIGHT = 1.0
PREFIX_LOSS_WEIGHT = 1.0
CONSISTENCY_LOSS_WEIGHT = 2.0   # EfficientZero's setting for its SimSiam loss
# DrQ-style augmentation on TRAINING observations only (EfficientZero ships it
# on, and its ablation shows it matters; it synergizes with the SimSiam loss).
# Acting, search, and evaluation never see augmented frames.
USE_AUGMENTATION = True
AUG_SHIFT = 4          # replicate-pad then random crop, DrQ's +/-4 pixels
AUG_INTENSITY = 0.05   # multiplicative intensity jitter, clamped at 2 sigma
# EfficientZero's released code augments the online root and the SimSiam
# targets with INDEPENDENT draws (core/train.py:73-75: obs_batch and
# obs_target_batch go through config.transform separately; the targets share
# one draw among themselves). Off = this repo's original shared draw, kept as
# the default so every archived arm keeps its meaning. With a shared draw the
# shift is perfectly predictable from online to target, and the collapsed
# projectors of the edge runs mostly encode it (2026-09-24,
# benchmarks/probes/probe_shift_collapse.log: 83-85% of s0's projection
# variance is the shift, 2-5% the game state).
AUG_INDEPENDENT_TARGETS = False
# Clamp augmented observations to [0, 1]. EfficientZero's Intensity does not
# (core/dataset.py); True keeps this repo's original behaviour.
AUG_CLAMP = True
# EfficientZero's gradient placement (core/train.py:139,167,264): the WHOLE
# weighted loss, step 0 included, is scaled by 1/K, and each dynamics output's
# gradient is halved via a hook -- which halves this step's heads too, so a
# step-k loss reaches the representation at 0.5^k / K. False = this repo's
# MuZero-pseudocode placement: step 0 unscaled, the halving applied after the
# step's heads (0.5^(k-1) / K). Loss VALUES are identical either way; only
# gradients differ, which matters most under SGD (Adam ignores overall scale).
EFFZERO_LOSS_SCALING = False
# Multiply the consistency loss by a real-position mask, zeroing unroll steps
# past a trajectory's end (EfficientZero multiplies it by mask_batch,
# core/train.py:158). Needs the paper batch's "real_mask".
CONSISTENCY_MASK_PADDING = False
# Min-max renormalise every emitted latent (MuZero appendix G). EfficientZero's
# Atari config leaves it off (state_norm=False, core/config.py:57).
LATENT_RENORMALIZE = True
# Gradient flowing back through each recurrent dynamics step is halved, so it
# does not compound across the unroll.
LATENT_GRADIENT_SCALE = 0.5

# ---------------------------------------------------------------------------
# Real-time pacing
#
# The console runs at 60 frames/s and on hardware the agent would decide once
# per frame. In ALE one env.step covers `frameskip` raw frames, so real time is
# GAME_FPS / frameskip decisions per second -- 20 for MsPacman-v4's stochastic
# (2,5) skip (ALE samples {2,3,4}, mean 3.0), NOT 60. Pacing at 60 would run
# the game ~3x faster than real time and defeat the point of the limiter.
#
# Unpaced, ALE runs ~80x real time, so producer/consumer ratios measured against
# it answer a question the real system never asks (see benchmarks/RESULTS.md).
# ---------------------------------------------------------------------------
GAME_FPS = 60.0
SELF_PLAY_REAL_TIME = True      # False lets self-play run as fast as the hardware allows
# Acting samples the policy head at this temperature. With no search on the
# acting path, this IS the exploration mechanism, so it stays at 1.0. (Policy
# TARGETS are never temperature-sharpened -- the paper's targets are raw visit
# fractions; the schedule that once sharpened them here was removed. Search
# acting has its own staged schedule, visit_softmax_temperature.)
ACTING_TEMPERATURE = 1.0
# Reanalysis searches every REANALYSE_STRIDE-th timestep instead of all of
# them. Measured on an RTX 4070, searching every step makes reanalysis ~2.5x
# slower than self-play generates data even with two MCTS actors, so the
# gameplay buffer overflows and take_random_gameplay discards the excess at random
# (see benchmarks/RESULTS.md). Unsearched timesteps still contribute
# observation, action, reward and n-step value targets (nstep_value); only
# their policy loss is masked. Set to 1 to search every step.
REANALYSE_STRIDE = 4
# n-step horizon for EVERY value target, as in MuZero-Reanalyze: TD_STEPS real
# rewards, then a bootstrap. The bootstrap prefers the nearest search root
# value at or past the horizon (stretching the horizon by up to
# REANALYSE_STRIDE-1 steps to land on a searched index -- the discount
# accounts for the stretch), and falls back to the current network's value
# where no search lies ahead. MuZero's Atari setting (td_steps=10).
TD_STEPS = 10
# ---------------------------------------------------------------------------
# Gumbel search (Danihelka et al. 2022) for the reanalysis policy targets.
#
# Measured motivation (2026-08-14, saved weights at 30k steps): with PUCT at
# 32 sims over 5 actions and 25% Dirichlet noise, one search added ~0.12 nats
# of information over the prior, and two independent searches of the SAME
# position disagreed with each other almost as much (~0.08) -- the visit-count
# target is mostly noise at this budget, and the policy CE fixed point under
# it sits near uniform. Gumbel's target is softmax(logits + sigma(completedQ))
# instead of visit fractions, which stays informative at small budgets, and
# fresh Gumbel draws per search replace Dirichlet noise entirely.
#
# USE_GUMBEL_SEARCH gates the whole path; False restores the PUCT search
# unchanged. Read at call time, so it can be flipped per ablation arm.
# ---------------------------------------------------------------------------
USE_GUMBEL_SEARCH = True
# m = min(GUMBEL_MAX_CANDIDATES, actions, sims); 16 is the paper's Atari cap,
# so with 5-6 actions every action is admitted -- the earlier value of 4
# permanently hid one action per search from visitation.
GUMBEL_MAX_CANDIDATES = 16
GUMBEL_C_VISIT = 50.0          # sigma(q) = (c_visit + max_visits) * c_scale * q
# 0.1 is the paper's setting FOR MIN-MAX NORMALIZED Q (its Atari appendix, and
# mctx's qtransform_completed_by_mix_value default) -- normalized advantages
# scale to roughly [-5, 5]. This sat at 1.0 ("paper defaults" -- the board-game
# regime, wrong for normalized Q): sigma ~50-60 amplified 32-sim Q noise into
# near-one-hot policy targets, disabled the Gumbel/logit exploration terms,
# and collapsed behaviour entropy -- the paper's own cscale ablation warns of
# exactly this failure.
GUMBEL_C_SCALE = 0.1
# ---------------------------------------------------------------------------
# Act-time search -- a TEST feature, not the product. The end goal of this
# project is acting WITHOUT search (the real-time thesis), so this defaults
# False and exists only to measure what the papers' search-based acting
# contributes: the Pong equivalence runs isolated the missing piece as
# bootstrapping -- a raw-prior actor never collects the successes the value
# function needs to differentiate actions. When True, harness SELF-PLAY acts
# via a fresh ACT_SEARCH_SIMS-simulation search per step (Gumbel or PUCT by
# USE_GUMBEL_SEARCH; for Gumbel, the sequential-halving survivor, and the
# Gumbel draw is the exploration). Greedy EVALUATION searches ONLY under
# PAPER_TARGETS -- the papers' evaluate-with-search protocol, which the
# effzero-repro numbers use; every non-paper eval measures the deployable
# raw-policy agent. The real-time pipeline agent cannot search at all (it
# does not load the dynamics network).
# ---------------------------------------------------------------------------
USE_SEARCH_ACTING = False
ACT_SEARCH_SIMS = 8            # measured ~40% of a 60fps frame at this budget
# Policy-only acting in PAPER mode (2026-09-26, Kai: the real-time design --
# no search on the acting path). With BIRTH_SEARCH on (and USE_SEARCH_ACTING
# off) the actor samples the prior, and every newly collected trajectory is
# searched ONCE on arrival -- fresh-root, vectorized (muzero_vtree), through
# the ONLINE nets, root noise on every step and a greedy search on a
# truncated tail -- exactly the targets the acting search would have
# recorded. So the only change from search acting is who picked the actions.
# Off = paper mode requires search acting, as before.
BIRTH_SEARCH = False
# Prior-acting exploration: sample prior^(1/T) with T from the papers' staged
# schedule (visit_softmax_temperature: 1.0 / 0.5 / 0.25 at 50% / 75% of the
# run) instead of the fixed ACTING_TEMPERATURE. Root noise is a search-only
# mechanism, so under policy-only acting this is the exploration knob.
PRIOR_ACT_STAGED_TEMP = False
# ---------------------------------------------------------------------------
# Paper-reproduction mode (EfficientZero/MuZero data path), flag-gated like
# everything else; the shipped product keeps all of these OFF. When the
# effzero-repro arm enables them: self-play acts through PUCT search
# (USE_SEARCH_ACTING + USE_GUMBEL_SEARCH=False) recording each step's search
# policy and root value; PAPER_TARGETS computes value targets at BATCH time
# with EfficientZero 4.3's age-shrunk horizon -- bootstrapping from stored
# root values by default (MuZero's make_target), or through the
# double-buffered lagged target model when USE_TARGET_MODEL is on (below;
# the repro arm enables it -- part 3 closed that gap); PER samples
# positions by |v - z| priority with importance-sampling correction.
# ---------------------------------------------------------------------------
PAPER_TARGETS = False
DIRICHLET_ALPHA = 0.25         # search-root noise; EfficientZero uses 0.3
# EfficientZero 4.3 (their code: auto_td_steps = 0.3 * training_steps): the
# horizon loses one real-reward step per 0.3-of-the-run of data age, clipped
# to [1, TD_STEPS]. Run-length dependent, so the harness sets it per run;
# None disables. (A fixed 300 shipped first -- ~8x too aggressive on typical
# runs; caught by the round-5 source verification.)
OFFPOLICY_DECAY_STEPS = None
# Search-based value estimation (EfficientZero V2, arXiv 2403.00564, sec. 4.3
# + released ez/agents/base.py; Kai 2026-09-21/24/28). "td" = today's target
# everywhere. "mixed": from SVE_START_TRAIN_STEPS on, a sampled window root
# that is NOT among the newest SVE_FRESH_TRANSITIONS transitions of the FIFO
# buffer (EZ-V2's buffer-index rule) takes the stored search root value as
# the value target at every real unroll position of its window, instead of
# the n-step TD target. Stage 1 of CURRICULUM's design: the stored value
# (birth search / refresh), no extra search. Kai 2026-09-28: T1 = 4e4 and
# T2 = 2e4, both the PAPER's values (the released code uses 3e4 / 5e3).
VALUE_TARGET_MODE = "td"
SVE_START_TRAIN_STEPS = 40000
SVE_FRESH_TRANSITIONS = 20000
# SVE value SOURCE. "stored": the stored root_value (stage 1). "search"
# (stage 2; Kai 2026-09-29, after the stage-0 probe found stored values
# ~40k steps old and +0.21 optimistic against a fresh search): a fresh-root
# PUCT search of the position through the lagged TARGET nets at
# SVE_SEARCH_SIMS -- EZ-V2's batch-time reanalysis value (their reanalyze
# model is the target model; 16 sims matched 50 at corr 0.97 in the probe),
# reported as EZ-V2's root mean over N+1 estimates (the root's own network
# value + the N simulation backups). Deliberate deviations (SVE stage-2
# audit): NO root noise -- EZ-V2's reanalysis search is noisy; noise-free
# is deterministic and cacheable, measured 0.004 above the noisy mean -- and
# a truncated tail is searched (EZ-V2 treats every chunk end as terminal:
# 0). Cached per (trajectory, position) until the next target sync.
SVE_SOURCE = "stored"
SVE_SEARCH_SIMS = 16
# Greedy (eval) episodes act on the prior's argmax even in a search-acting
# PAPER arm, whose eval otherwise searches (the papers evaluate with
# search). For the search-acting bisect on the --cpu-actor stack (Kai
# 2026-09-29): 8 CPU-searched eval games of up to 3000 steps take over an
# hour per row, and the pre-registered readout is the exact PRIOR-ARGMAX
# eval anyway (probe_exhaustive_eval) -- so the in-run rows match it.
EVAL_PRIOR_ONLY = False
# Staged acting temperature on visit counts (papers: 1.0 / 0.5 / 0.25 at
# 50% / 75% of the run). Requires a planned run length; None (the unbounded
# pipeline) stays at 1.0. The harness sets it per run.
TEMPERATURE_TOTAL_STEPS = None
PER_ENABLED = False
# EfficientZero's PER settings, from their released config: alpha 0.6, beta
# annealed linearly 0.4 -> 1.0 over the run, epsilon 1e-6. (MuZero's Atari
# values are alpha=beta=1.)
PER_ALPHA = 0.6
PER_BETA = 0.4
PER_BETA_FINAL = 1.0
PER_EPS = 1e-6


def per_beta(training_steps):
    """EfficientZero anneals the IS-correction exponent 0.4 -> 1.0 linearly
    over the planned run; without a planned length, the base value holds."""
    total = TEMPERATURE_TOTAL_STEPS
    if not total:
        return PER_BETA
    frac = min(1.0, training_steps / total)
    return PER_BETA + (PER_BETA_FINAL - PER_BETA) * frac
# EfficientZero A.1 value-prefix head: 16 conv planes and a 32-unit FC
# bottleneck before the categorical output. The historical head here used 32
# planes and no bottleneck; the repro arm overrides both.
PREFIX_HEAD_CHANNELS = 32
PREFIX_FC_HIDDEN = None
# The lineage's env stack (NoopReset(30) + MaxAndSkip flicker max-pool over a
# raw no-skip env) instead of ALE's internal last-frame skip. Harness-applied.
USE_ATARI_WRAPPERS = False
# EfficientZero's reanalyze, value half: recompute the bootstrap value at
# BATCH time with a lagged target model (their target_model_interval=200)
# instead of the stored play-time root value (MuZero pseudocode behavior,
# which USE_TARGET_MODEL=False preserves).
USE_TARGET_MODEL = False
TARGET_MODEL_INTERVAL = 200
# EfficientZero's reanalyze, policy half -- approximated: their workers
# re-search ~99% of sampled positions per batch (revisit_policy_search_rate),
# which needs a CPU fleet; here the harness re-searches every position of the
# POLICY_REFRESH_COUNT oldest trajectories every POLICY_REFRESH_EVERY
# iterations THROUGH THE LAGGED TARGET MODEL (as their reanalyze workers
# search; the terminal record of a terminated trajectory carried no search
# and is skipped), stamping a refresh rotation while PRESERVING born ages
# -- fresh policy targets never restore the 4.3 age-shrunk value horizon.
POLICY_REFRESH_EVERY = 5
POLICY_REFRESH_COUNT = 2
# Phase 3a (2026-08-31, spec in CURRICULUM.md). All default OFF so every
# archived arm keeps its recorded meaning; the entropy-refresh arm turns
# them on together with a raised POLICY_REFRESH_COUNT.
# REFRESH_ENTROPY_PRIORITY: refresh candidates are ordered flattest-mean-
# stored-target-entropy first (the tiny buffer's ~50x reuse amplifies
# every flat target, so repairing the flattest first repairs the most-
# reused weakest signals) instead of oldest-first rotation.
# REFRESH_NO_SHARPEN_GUARD: guard v2 (2026-09-03) -- a step is marked
# and excluded from FUTURE re-searches only once STABLE-SHARP: its
# re-search returns a sharp target (H <= REFRESH_GUARD_SHARP_H) that
# reproduces the stored one (KL <= REFRESH_GUARD_KL_TOL), STRIKES
# times running. Flat targets never retire (flatness indicts the
# model); redirects reset. The fresh result is always applied. (The
# flag/key names date from the v1 delta-H criterion and are kept for
# arm-dict and checkpoint compatibility.)
# REFRESH_OFFLOAD: under --ray, refresh runs on a dedicated actor
# against the lagged target nets and the driver applies results at a
# fixed point in its own loop (swap-on-ready; the single-threaded loop
# is the mutex). Inline refresh remains the sequential-mode path.
REFRESH_ENTROPY_PRIORITY = False
REFRESH_NO_SHARPEN_GUARD = False
# Guard criterion v2 (2026-09-03, from tests 1+2): retire a step ONLY
# when its target is STABLE-SHARP -- entropy below REFRESH_GUARD_SHARP_H
# AND the re-search reproduces it (KL(new||old) <= REFRESH_GUARD_KL_TOL)
# for REFRESH_GUARD_STRIKES consecutive visits; anything else resets.
# The v1 delta-H criterion punished flat->flat (dH ~ 0), which indicts
# the MODEL, not the state: test 2 froze weak-model flat targets at
# iter ~5 and trained on them at ~50x reuse until eviction -- killing
# test 1's 2x acceleration despite HIGHER refresh volume. v2 never
# retires a flat target (the model may yet sharpen it) and never
# freezes a redirect (sharp->different-sharp has large KL and resets),
# fixing the collapsed-policy freeze recorded in CURRICULUM.md. Repeat
# searches of a CONVERGED sharp target measure KL ~ 0.02-0.05; the
# all-states repeat-noise floor is ~0.19-0.22 -- tol 0.1 separates them.
REFRESH_GUARD_SHARP_H = 0.9
REFRESH_GUARD_KL_TOL = 0.1
REFRESH_GUARD_STRIKES = 2
REFRESH_OFFLOAD = False
# Phase 4 (2026-09-02, Kai's latency/throughput split): refresh runs as
# fresh-root VECTORIZED searches on the GPU against the lagged target
# model (muzero_vtree; measured 307 searches/s at batch 256 on this
# box, 77x one sequential CPU search) while the environment actor stays
# on CPU with local nets -- the latency lane never touches the GPU or
# any server. Fresh-root topology (the pre-communal refresh5x shape,
# noise at every root as the communal path also did); candidate
# selection, entropy priority, the no-sharpen guard, and application
# are the SAME code as every other refresh path.
REFRESH_VTREE = False
VTREE_REFRESH_BATCH = 256
# vtree v2 (2026-09-11): with USE_COMMUNAL_REANALYSIS the vectorized
# refresh runs archfix's communal chain-tree search (muzero_vtree.
# VectorizedCommunalPUCT, parity-gated bitwise against
# run_communal_mcts). This many trajectories share one forest per
# call -- their chain trees advance windows in lockstep and the network
# batch is formed across all their roots. 1 = one trajectory per
# forest: the sequential engine's exact batch shapes (the parity
# condition), but no throughput gain over it (the wave loop costs as
# much as its Python at 16 roots). Larger groups amortize the wave
# loop across trajectories (~x group throughput; 3.4x the sequential
# engine at 8, 5.3x at 16 on the RTX 4070) at the cost of batch-shape
# numerics the communal tree cascades (measured G=8 vs 1: ~96% of
# roots keep identical visit counts, values shifting by a median 3e-3,
# tail 1.5; ~4% flip counts, values shifting by a median 0.26, max
# 2.1; DESIGN.md) and forest memory (~22.7 KB per node at production
# geometry; growth transient 2.25x -- 8 fits beside a live run).
VTREE_COMMUNAL_TRAJS = 1
# Saturation refresh (2026-09-02, Kai): instead of a fixed quota every
# POLICY_REFRESH_EVERY iterations, the driver refreshes AS MUCH AS FITS
# in the slack while the CPU environment actor plays -- one candidate
# trajectory at a time, flattest first, polling the actor between
# trajectories, with a base-schedule floor so refresh never runs below
# the control's rate. The one-pass seen-set bounds every call; the v2
# guard additionally retires STABLE-SHARP (converged) targets so
# late-phase churn is damped where re-searching only replays noise --
# flat targets stay eligible (their flatness indicts the model).
# Requires
# REFRESH_VTREE and the --ray pipeline (there is no slack window in
# sequential mode; the scheduled quota remains its fallback).
REFRESH_SATURATE = False
# Phase 5 (2026-09-06, Kai): the fully asynchronous chunked loop --
# "iteration" dissolves. One driver loop interleaves CHUNKS (poll the
# episode future / ~ASYNC_TRAIN_CHUNK gradient steps / the refresh
# sweep's next trajectories); the actor plays CONTINUOUSLY (a new
# episode is requested the moment one is collected -- THE invariant of
# the real-time framing), and the loop stays the mutex (no threads).
# Freshness contract (Kai's requirement, in place of every guard/
# quota/priority mechanism): the WHOLE buffer is re-searched at least
# once per ASYNC_SWEEP_TRAIN_STEPS gradient steps (rotation order,
# always-replace, no guard -- the twice-replicated sat-stale
# semantics), and a newly collected trajectory is refreshed against
# the lagged target model BEFORE it can be sampled
# (ASYNC_REFRESH_ON_COLLECT) -- so no target trained on is ever older
# than one sweep period, INCLUDING during the early iterations where
# the slack-bounded saturate arms under-covered. MAX_TRAIN_PER_ENV_STEP
# is the REQUIRED train-ratio cap (CURRICULUM.md phase-5 spec, at
# today's ~2): without it the emergent hardware-dependent ratio could
# explode into tiny-buffer overtraining early; the realized ratio is
# logged in every row. Eval/checkpoint re-key to gradient steps
# (u = training_steps // train_steps keeps the family's row axis).
ASYNC_LOOP = False
ASYNC_TRAIN_CHUNK = 100
ASYNC_SWEEP_TRAIN_STEPS = 2000
ASYNC_REFRESH_ON_COLLECT = True
MAX_TRAIN_PER_ENV_STEP = 2.0
# Test 6 (2026-09-07, from the pong-async-s1 verdict): 0 = off. When
# set, ADMISSION CONTROL on the buffer -- a collected episode enters
# the replay buffer only if the realized train/env ratio stays >= this
# floor; excess episodes still count env_steps (they happened) and
# still drive the weight-sync/request cycle (the actor is NEVER
# blocked or slowed), but their data is dropped; rows carry the
# cumulative-since-resume episodes_dropped counter. Rationale: the unthrottled async driver ran at
# ratio ~1.06-1.31 vs the legacy driver's ~2.0 -- doubling buffer
# inflow, halving per-position sample reuse -- and lost the satstale
# acceleration (-17.5 @ u 25 vs the pair's -12.9/-9.9). Legacy's
# 1-episode-per-iteration was an implicit data throttle; this makes it
# explicit and tunable while keeping every other async property
# (continuous acting, freshness contract, newcomer refresh timing).
ASYNC_MIN_TRAIN_PER_ENV_STEP = 0.0
# Training-lane speedup (2026-09-11, Kai): a measured gradient step was
# ~255 ms of CPU batch assembly + ~285 ms of GPU compute, SERIALIZED, so
# neither device exceeded ~50% utilization. BatchPrefetcher overlaps
# them: a worker thread assembles batch i+1 (pure CPU: sampling, unroll
# windows, stored-value parts) while the GPU trains on batch i; the
# target-model bootstrap forward stays on the MAIN thread right before
# the train step, so the worker never touches CUDA or the target nets
# (no torn-read race with the target sync) and the sequence of batches
# is BIT-IDENTICAL to the serial loop for the same seed with PER off
# (the worker is the only consumer of the numpy RNG during a training
# block; batch i is assembled with training_steps = base + i, exactly
# the value the serial loop would have used). With PER on, priorities
# updated by step i reach the prefetcher late
# (a lag of TRAIN_PREFETCH_DEPTH + 1 batches: the queue holds DEPTH and
# the worker holds one more -- audit F3). MEASURED on the real
# checkpoint: no gain -- the assembly half is only ~40 ms and the
# worker's Python loop contends for the GIL with kernel launches; the
# ~210 ms that looked like CPU work was the target-model bootstrap
# forward, which the bootstrap CACHE in finish_batch_paper removes.
# Default OFF: opt-in, correct, but not worth a threading surface and
# the PER lag for a ~0-10% return.
TRAIN_PREFETCH = False
TRAIN_PREFETCH_DEPTH = 2
# Bootstrap value cache toggle (2026-09-14). Off = every tail is
# forwarded every step, exactly the pre-53161e22 builder (archfix's
# path). The cache is identity within a target-sync window up to
# chunk-composition numerics, but it is the ONE code delta between
# archfix (+17.1, cache off) and both non-crossing replications
# (cache on), so experiments meant to be comparable to archfix set
# it off via the arm until that confound is closed.
BOOTSTRAP_CACHE = True


def visit_softmax_temperature(training_steps):
    """The papers' staged acting temperature over visit counts."""
    total = TEMPERATURE_TOTAL_STEPS
    if not total:
        return 1.0
    frac = training_steps / total
    if frac < 0.5:
        return 1.0
    if frac < 0.75:
        return 0.5
    return 0.25



def prior_acting_temperature(training_steps):
    """Temperature for policy-only (prior-sampling) acting: the papers'
    staged schedule under PRIOR_ACT_STAGED_TEMP, else ACTING_TEMPERATURE."""
    if PRIOR_ACT_STAGED_TEMP:
        return visit_softmax_temperature(training_steps)
    return ACTING_TEMPERATURE


def paper_horizon(age_steps):
    """EfficientZero 4.3: older data gets a shorter real-reward horizon."""
    if not OFFPOLICY_DECAY_STEPS:
        return TD_STEPS
    return int(np.clip(TD_STEPS - age_steps // OFFPOLICY_DECAY_STEPS,
                       1, TD_STEPS))


def paper_value_target(traj, idx, horizon, discount):
    """MuZero's make_target on stored dicts:

        z_t = sum_{k=1..l} g^{k-1} r_{t+k} + g^l * root_value_{t+l}

    The bootstrap is the STORED (self-play-time) search root value -- exactly
    MuZero's pseudocode make_target, which bootstraps game.root_values.
    (EfficientZero goes further: it RECOMPUTES the bootstrap at batch time
    through a lagged target model, so its value never ages with the
    trajectory. That is the USE_TARGET_MODEL=False behavior gap ONLY:
    prepare_batch_paper recomputes exactly so when the flag is on, and the
    effzero-repro arm enables it -- part 3 closed the gap.) A terminal
    record stores
    root_value 0, so the terminal case reduces to pure Monte Carlo without a
    special branch; a truncated tail bootstraps its stored value.
    """
    reward_sum, g, tail = paper_value_parts(traj, idx, horizon, discount)
    return reward_sum + g * traj[tail]["root_value"]


def paper_value_parts(traj, idx, horizon, discount):
    """The decomposition (reward_sum, g, tail): z = reward_sum + g * v(tail).

    Separated so the target-model refresh can supply v(tail) from a lagged
    network at batch time instead of the stored root value."""
    tail = min(idx + horizon, len(traj) - 1)
    value, g = 0.0, 1.0
    for k in range(1, tail - idx + 1):
        value += g * traj[idx + k]["reward"]
        g *= discount
    return value, g, tail
# Q is normalised to [0,1] by MinMaxStats before the exploration term is added,
# so this is on the usual AlphaZero/MuZero scale. It is NOT comparable to the
# values used before normalisation existed (61.8, then 32).
C_PUCT = 1.25
# EfficientZero's epsilon for the min-max normalizer's range (their 0.01).
MINMAX_RANGE_FLOOR = 0.01

# ---------------------------------------------------------------------------
# Ray resource budget
#
# Fractional num_gpus is bookkeeping, not enforcement: it decides how many
# actors may share a device and does not cap memory. The requests must sum to
# no more than the cluster total, because Ray does NOT raise when an actor
# cannot be scheduled -- it queues it indefinitely, so ray.get(setup.remote())
# just never returns. The previous fractions summed to exactly 1.0 GPU, which
# is why a second MCTS actor could only ever hang.
#
# Resources are applied with .options() at construction rather than baked into
# the @ray.remote decorators, so the MCTS share can be split across actors and
# so GPU requests can be dropped entirely on a machine without CUDA.
# ---------------------------------------------------------------------------
NUM_MCTS_ACTORS = 2

CPU_STORAGE  = 1   # async data shuffling only
CPU_NETWORK  = 1   # async; the work is on the GPU
CPU_TRAINING = 2   # prepare_batch does real numpy work
CPU_AGENT    = 2   # event loop + the thread asyncio.to_thread runs env.step on
CPU_MCTS     = 2   # selection is single-threaded asyncio; eval_obs is numpy

GPU_NETWORK  = 0.05
GPU_TRAINING = 0.30
GPU_AGENT    = 0.05

def mcts_gpu_share(num_mcts_actors=NUM_MCTS_ACTORS):
    """The MCTS actors do by far the most inference, so they split whatever the
    other three actors leave behind. Derived from the actor count rather than
    hardcoded, so changing NUM_MCTS_ACTORS cannot silently oversubscribe."""
    return round((1.0 - GPU_NETWORK - GPU_TRAINING - GPU_AGENT) / num_mcts_actors, 4)

GPU_MCTS = mcts_gpu_share()

if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

def print_tree(node, depth=0):
    if depth > 6:
        return
    indent = "    " * depth  # 4 spaces per depth level.
    print(f"{indent}Node(prior={node.prior}, reward={node.reward}, visits={node.visit_count}, value={node.expanded_value}, depth={depth})")
    for action, child in node.children.items():
        print(f"{indent}  Action {action}:")
        print_tree(child, depth + 1)

def add_dirichlet_noise(prior, epsilon=0.25, alpha=0.25):
    """
    Adds Dirichlet noise to a given prior policy.
    
    Args:
      prior (np.array): Array of shape (num_actions,) representing the original prior probabilities.
      epsilon (float): Fraction of noise to mix in.
      alpha (float): Concentration parameter for the Dirichlet distribution.
    
    Returns:
      np.array: The noisy prior policy.
    """
    noise = np.random.dirichlet([alpha] * len(prior))
    noisy_prior = (1 - epsilon) * prior + epsilon * noise
    return noisy_prior

class MinMaxStats:
    """Tracks the value range seen during one search so Q can be normalised.

    Without this the PUCT exploration term is inert. Q here is a raw discounted
    return, which for MsPacman runs into the hundreds or thousands, while
    c_puct * prior * sqrt(N) / (1 + n) is order 10 -- so selection is
    effectively greedy no matter how c_puct is tuned. Normalising Q into [0,1]
    puts the two terms on the same scale and makes c_puct mean what it does in
    the AlphaZero/MuZero papers.
    """

    def __init__(self):
        self.maximum = -float('inf')
        self.minimum = float('inf')

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            # EfficientZero Eq 10 (and mctx): floor the range, "when the data
            # is limited, the small difference between the minimum and maximum
            # values will result in overconfidence" -- without the floor, a
            # hair-thin early-search Q spread is stretched across the full
            # [0,1], amplifying noise into large normalized advantages that
            # feed straight into the Gumbel sigma policy targets. Same failure
            # family as the fixed c_scale bug, at smaller magnitude.
            return (value - self.minimum) / max(
                self.maximum - self.minimum, MINMAX_RANGE_FLOOR)
        return value

# -----------------------------
# MCTS Node Definition
# -----------------------------
class MCTSNode:
    def __init__(self, latent, prior, reward=0.0, policy = None, expanded_value = None,
                 is_gameplay = False, prefix = 0.0, prefix_state = None,
                 prefix_len = 0):
        self.latent = latent            # torch.Tensor (latent state)
        self.prior = prior              # float value
        self.reward = reward            # immediate reward
        # Cumulative UNDISCOUNTED reward since the last LSTM reset (the search
        # root, a gameplay-chain restart, or a PREFIX_LSTM_HORIZON boundary),
        # and the LSTM state that produced it. Undiscounted so that an edge
        # reward is exactly the difference between a child's prefix and its
        # parent's within a segment (and the fresh prefix itself at a reset) --
        # discounting stays in backup_simulation, where it always was, rather
        # than being baked into the model's own prediction and then divided back
        # out by depth. The paper writes the prefix discounted; this keeps the
        # arithmetic in one place.
        self.prefix = prefix
        self.prefix_state = prefix_state
        # LSTM steps accumulated in prefix_state since the last reset; at
        # PREFIX_LSTM_HORIZON the next expansion feeds a fresh state.
        self.prefix_len = prefix_len
        self.children = {}              # dict: action -> child node
        self.visit_count = 0
        # Backups that STARTED here (this node as a search root). A chain
        # node is also its predecessor's behaviour child, so its parent's
        # edge count is visit_count - root_visits (COMMUNAL_EDGE_VISITS).
        self.root_visits = 0
        self.value_sum = 0.0
        self.id = uuid.uuid4()
        self.done = False
        
        self.is_expanded = False
        self.policy = policy              # cached policy (numpy array)
        self.lock = asyncio.Lock()
        self.expanded_value = expanded_value
        self.virtual_loss = 0.0
        self.is_gameplay = is_gameplay

    def q_value(self):
        # Mean over COMPLETED backups only. Virtual loss must NOT sit in this
        # average: dividing by (visits + virtual_loss) pulls Q toward zero,
        # which repels only where zero is pessimistic -- true for non-negative
        # returns, inverted on a losing sign-clipped game like Pong, where the
        # in-flight "zero" outscored every honestly-evaluated sibling by the
        # full width of the normalisation range and entire simulation batches
        # piled onto one action. The in-flight penalty lives in NORMALIZED
        # space at selection instead, where the worst value seen this search
        # is 0 by construction regardless of reward sign.
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

def apply_temperature(visit_counts, temperature):
    """
    Adjusts a vector of visit counts using a temperature parameter.
    
    Args:
        visit_counts (list or np.array): MCTS visit counts for each action.
        temperature (float): Temperature parameter. Higher values lead to softer distributions.
        
    Returns:
        np.array: Probability distribution over actions.
    """
    visit_counts = np.array(visit_counts, dtype=np.float32)
    
    # When temperature is close to 0, choose the action with the highest count deterministically.
    if temperature < 1e-3:
        policy = np.zeros_like(visit_counts)
        policy[np.argmax(visit_counts)] = 1.0
        return policy
    
    # Adjust counts by temperature.
    adjusted_counts = visit_counts ** (1.0 / temperature)
    policy = adjusted_counts / np.sum(adjusted_counts)
    return policy

def scale_gradient(tensor, scale):
    """Identity on the forward pass; multiplies the gradient by `scale`.

    The standard MuZero trick: `x*s + x.detach()*(1-s)` is exactly x going
    forward, and has derivative s going back.
    """
    return tensor * scale + tensor.detach() * (1.0 - scale)


def real_time_step_period(env, game_fps=GAME_FPS):
    """Seconds per agent decision corresponding to real-time play.

    Reads the env's frameskip rather than assuming one: MsPacman-v4 samples it
    from ALE's integers(2, 5) -- {2,3,4}, mean 3.0 -- so a decision covers 3
    raw frames on average and real time is 20 decisions/second, not 60.
    """
    skip = getattr(env.unwrapped, "_frameskip", None) or 1
    if isinstance(skip, (tuple, list)):
        # ALE samples integers(low, high) with an EXCLUSIVE high -- (2,5) draws
        # from {2,3,4}, mean 3.0. Averaging the tuple bounds gave 3.5 and paced
        # the "real-time" agent ~14% slower than real time.
        skip = (skip[0] + skip[1] - 1) / 2
    return float(skip) / game_fps


class RateLimiter:
    """Paces a loop to a target period without drift or catch-up bursts."""

    def __init__(self, period):
        self.period = period
        self.next_at = None
        self.behind = 0

    async def wait(self):
        if not self.period:
            return
        now = time.perf_counter()
        if self.next_at is None:
            self.next_at = now
        self.next_at += self.period
        delay = self.next_at - now
        if delay > 0:
            await asyncio.sleep(delay)
        else:
            # Running behind. Reset the clock rather than sprinting to catch up,
            # which would produce exactly the faster-than-real-time burst the
            # limiter exists to prevent.
            self.behind += 1
            self.next_at = now


def discounted_prefix_targets(rewards, discount=None):
    """Cumulative discounted reward sums along an unroll.

    The default resolves DISCOUNT at call time, not in the signature -- the
    same default-argument binding that silently pinned backup_simulation's
    Monte Carlo targets to 0.997 and would make any runtime DISCOUNT override
    (the ablation harness) measure nothing here.

    `rewards` is (B, unroll+1) where rewards[:, i] is the reward for *entering*
    step i. rewards[:, 0] therefore belongs to the state the unroll starts from
    and is part of no prefix. Column i of the result is

        prefix_i = sum_{j=1..i} discount^(j-1) * rewards[:, j]

    with column 0 zero: nothing is accumulated before the first step.
    """
    discount = DISCOUNT if discount is None else discount
    steps = rewards.shape[1] - 1
    weights = discount ** torch.arange(steps, device=rewards.device, dtype=rewards.dtype)
    prefix = torch.cumsum(rewards[:, 1:] * weights, dim=1)
    return torch.cat([torch.zeros_like(rewards[:, :1]), prefix], dim=1)


def augment_observations(obs, shift=None, intensity=None, independent_targets=None):
    """DrQ-style random shift + intensity jitter for training batches.

    obs: (B, T, C, H, W) float in [0, 1]; step 0 is the online root, steps
    1..T-1 the SimSiam targets. Independent draws across samples.

    independent_targets False (the original behaviour): one draw per sample,
    shared by every step. Its rationale was that the target must live in the
    same shifted frame as the online branch, but that makes the shift the one
    feature the consistency loss can always match, and collapsed projectors
    encode little else (see AUG_INDEPENDENT_TARGETS).
    independent_targets True (EfficientZero's released code): the root gets
    its own draw and the targets share a second one, so the consistency loss
    has to be invariant to the augmentation.
    """
    independent_targets = (AUG_INDEPENDENT_TARGETS if independent_targets is None
                           else independent_targets)
    if independent_targets and obs.shape[1] > 1:
        return torch.cat([_augment_shared(obs[:, :1], shift, intensity),
                          _augment_shared(obs[:, 1:], shift, intensity)], dim=1)
    return _augment_shared(obs, shift, intensity)


def _augment_shared(obs, shift=None, intensity=None):
    """One (shift, intensity) draw per sample, shared by all T steps."""
    shift = AUG_SHIFT if shift is None else shift
    intensity = AUG_INTENSITY if intensity is None else intensity
    B, T, C, H, W = obs.shape
    flat = obs.reshape(B * T, C, H, W)
    padded = F.pad(flat, (shift, shift, shift, shift), mode="replicate")
    out = torch.empty_like(flat)
    dx = torch.randint(0, 2 * shift + 1, (B,))
    dy = torch.randint(0, 2 * shift + 1, (B,))
    for b in range(B):
        rows = slice(int(dy[b]), int(dy[b]) + H)
        cols = slice(int(dx[b]), int(dx[b]) + W)
        out[b * T:(b + 1) * T] = padded[b * T:(b + 1) * T, :, rows, cols]
    out = out.view(B, T, C, H, W)
    if intensity:
        scale = 1.0 + intensity * torch.randn(
            B, 1, 1, 1, 1, device=obs.device).clamp_(-2.0, 2.0)
        out = out * scale
    return out.clamp_(0.0, 1.0) if AUG_CLAMP else out


def consistency_loss(online, target, weights=None):
    """The SimSiam objective: negative cosine similarity, -1 at a perfect match.

    target is detached here as well as at the call site. The stop-gradient is
    load-bearing -- it is half of what prevents the representation collapse
    this loss exists to avoid -- so it must not depend on the caller
    remembering to no_grad the target branch. `weights` are the per-sample
    PER importance weights (EfficientZero scales the ENTIRE per-sample loss).
    """
    per_sample = -F.cosine_similarity(online, target.detach(), dim=1)
    if weights is not None:
        per_sample = per_sample * weights
    return per_sample.mean()


def clip_reward(reward, ceiling=None):
    """Symmetric clip to the reward support's raw ceiling.

    One home for the clip, resolved against REWARD_CLIP at call time: the
    acting path hardcoded `min(reward, 400)`, so changing the constant would
    have resized the categorical supports while the actual data kept the old
    bound -- a silent codec mismatch. Symmetric (min() alone ignores the
    negative side) because MsPacman never goes negative but the Switch targets
    will.
    """
    ceiling = REWARD_CLIP if ceiling is None else ceiling
    return float(np.clip(reward, -ceiling, ceiling))


def masked_policy_loss(policy_logits, target_policy, mask, weights=None):
    """Cross-entropy averaged over the positions that carry a real target.

    Reanalysis searches a strided subset of timesteps, so most positions hold a
    uniform placeholder. Averaging across all of them would train the policy
    toward uniform in proportion to the stride -- the sharper the search got,
    the harder the placeholders would pull against it.

    Returns 0 when nothing in the batch is masked in; clamping the denominator
    keeps that branch-free so it does not force a device sync. `weights` are
    the per-sample PER importance weights (the papers scale the whole
    per-sample loss).
    """
    per_position = -torch.sum(target_policy * F.log_softmax(policy_logits, dim=1), dim=1)
    per_position = per_position * mask
    if weights is not None:
        per_position = per_position * weights
    return per_position.sum() / mask.sum().clamp(min=1.0)


def to_cpu(obj):
    """Recursively move every tensor in a nested structure to CPU.

    Optimiser state_dicts nest their tensors two levels down, under
    state[param_id][key], so a flat isinstance check over the top level finds
    none of them. That was harmless only because SGD without momentum keeps no
    state at all; with Adam it would ship exp_avg and exp_avg_sq across Ray
    still resident on the GPU.
    """
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu(v) for v in obj)
    return obj


def get_state_dict(network):
    return to_cpu(network.state_dict())


LEARNING_RATE = 1e-3
# The papers' only explicit regularizer (MuZero: L2 c=1e-4; EfficientZero:
# weight decay 1e-4) and their gradient-stability lever (EfficientZero clips
# grad norm at 5). Both were absent entirely.
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 5.0
# Staged LR decay, the papers' schedule shape (they decay x0.1 in steps; ours
# is a single stage). None disables. The HARNESS applies it once at this
# fraction of a run's iterations -- the Ray pipeline has no bounded run length
# to anchor a fraction to, so it ignores these.
LR_DECAY_AT = None
LR_DECAY_FACTOR = 0.1
# "adamw" (this repo's default) or "sgd" -- EfficientZero's released optimizer:
# optim.SGD(lr=0.2, momentum=0.9, weight_decay=1e-4) over every parameter,
# BN affines and biases included (core/train.py:348). The repo's earlier SGD
# trial ran at lr 1e-4, 2000x below theirs.
OPTIMIZER = "adamw"
SGD_MOMENTUM = 0.9
# Linear lr warmup from 0 over this many training steps, applied inside
# train_step (EfficientZero: lr_warm_up 0.01 x 100k = 1000 steps,
# core/train.py:30-41). 0 disables. Only the warmup is ported: their x0.1 decay
# lands at step 101k, inside their final 20k no-collection steps, so their
# runs trained at a constant lr while collecting data.
LR_WARMUP_STEPS = 0
# Skip any update whose clipped gradient norm is NaN/Inf instead of applying
# it (the check costs one device sync per step, hence opt-in). EfficientZero
# trains under AMP, whose GradScaler skips such steps; in fp32 one bad step
# would otherwise corrupt every weight and momentum buffer permanently.
SKIP_NONFINITE_GRAD_STEPS = False
# ReDo (redo.py; Sokar et al. 2023): every REDO_EVERY gradient steps, score
# each single-consumer unit's post-ReLU activation on a fresh uniform replay
# sample of REDO_BATCH windows and recycle those scoring <= REDO_TAU. 0 = off.
# The paper recycles every 1000 updates with tau 0.025 in its default
# setting. Motivation: under SGD + coupled weight decay the faithful runs lost
# ~8% of BN channels and up to half of the first-layer filters by iter 85.
# Passes land MID-period (experiments/ablation.py redo_due: 1.5k, 2.5k, ...),
# never on an eval row, a checkpoint or the final step.
REDO_EVERY = 0
REDO_TAU = 0.025
REDO_BATCH = 256
# Which nets ReDo recycles (redo.ALL_NETS by default). The prioract arm drops
# the projector: ~94% of dormant units sit in its hidden layers, so including
# it would re-randomize most of the projector every pass (Kai, 2026-09-26).
REDO_NETS = ("representation", "dynamics", "prediction", "value_prefix",
             "projector")


def decay_optimizer_lr(optimizers, factor):
    """Multiply every param group's learning rate by `factor`."""
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] *= factor

def make_optimizer(params):
    """AdamW. The repo moved Adam -> SGD without a recorded reason, and the one
    plausible cause -- get_state_dict shipping Adam's GPU-resident state through
    Ray -- is fixed above. SGD at 1e-4 was slow enough that the value prefix head
    could not converge inside a benchmark run (benchmarks/RESULTS.md). AdamW so
    the papers' 1e-4 weight decay is decoupled rather than folded into the
    adaptive gradient. OPTIMIZER "sgd" builds EfficientZero's instead.
    """
    if OPTIMIZER == "sgd":
        return optim.SGD(params, lr=LEARNING_RATE, momentum=SGD_MOMENTUM,
                         weight_decay=WEIGHT_DECAY)
    if OPTIMIZER != "adamw":
        raise ValueError(f"unknown OPTIMIZER {OPTIMIZER!r}")
    return optim.AdamW(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


def apply_lr_warmup(optimizers, training_steps):
    """EfficientZero's adjust_lr warmup: lr = LEARNING_RATE * step / warm for
    step < warm, then LEARNING_RATE. Touches the optimizers only up to and
    including the boundary step, so any later lr change (LR_DECAY_AT) is left
    alone."""
    if not LR_WARMUP_STEPS or training_steps > LR_WARMUP_STEPS:
        return
    lr = LEARNING_RATE * min(1.0, training_steps / LR_WARMUP_STEPS)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = lr

LOG_DIR = "logs"

def configure_actor_logger(name, enable_logging=False, log_dir=LOG_DIR):
    """Build a logger for a Ray actor, writing to logs/<name>.log.

    Each actor runs in its own process with its own stdout, and the console
    handler here used to be opt-in, so with default settings a failure inside
    an actor left no trace anywhere at all. The file handler is unconditional
    so there is always somewhere to look afterwards.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if enable_logging:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger

class StorageHelper:
    def __init__(self, max_replay=MAX_REPLAY, max_gameplay=MAX_GAMEPLAY):
        self.gameplay_buffer = deque(maxlen=max_gameplay)
        self.replay_buffer = deque(maxlen=max_replay)
        self.replay_buffer_version = uuid.uuid4()
        self.gameplay_lock = asyncio.Lock()
        self.replay_lock = asyncio.Lock()
        self.replay_versions = deque(maxlen=max_replay)
    
    async def append_replay_buffer(self, new_item):
        async with self.replay_lock:
            self.replay_buffer.append(new_item)
            self.replay_buffer_version = uuid.uuid4()
            self.replay_versions.append(self.replay_buffer_version)

    async def get_replay_buffer_updates(self, version=None):
        async with self.replay_lock:
            try:
                # Find the index of the passed-in version.
                index = self.replay_versions.index(version)
                # Return all replays that were added after the given version.
                missing_updates = list(self.replay_buffer)[index + 1:]
            except ValueError:
                # If the version is not found, return the whole replay buffer.
                missing_updates = list(self.replay_buffer)
            return self.replay_buffer_version, missing_updates

    async def is_replay_up_to_date(self, version):
        async with self.replay_lock:
            return self.replay_buffer_version == version

    async def append_gameplay_buffer(self, new_item):
        async with self.gameplay_lock:
            self.gameplay_buffer.append(new_item)
    
    async def take_random_gameplay(self):
        """Pop a random trajectory for reanalysis, or None if the buffer is empty.

        Was get_max_gameplay: priority ("max") selection was the intent but
        never implemented, so the name now matches the behaviour. Removal is
        what makes several MCTS actors safe -- no two may reanalyse the same
        trajectory.
        """
        async with self.gameplay_lock:
            if len(self.gameplay_buffer) == 0:
                return None
            idx, item = random.choice(list(enumerate(self.gameplay_buffer)))
            del self.gameplay_buffer[idx]
            return item
    

class StorageManager:
    def __init__(self, storage_helper, update_replay=True):
        self.storage_helper = storage_helper
        self.update_replay = update_replay

    async def setup(self):
        self.replay_buffer = deque(maxlen=MAX_REPLAY)
        self.replay_version, updates = await self.storage_helper.get_replay_buffer_updates.remote()
        self.replay_buffer.extend(updates)
    
    async def update_if_needed(self):
        if self.update_replay and not await self.storage_helper.is_replay_up_to_date.remote(self.replay_version):
            # The version has to be passed through. Without it the helper falls
            # into its ValueError branch and ships the entire replay buffer --
            # potentially gigabytes of frame stacks -- on every single call,
            # and duplicates entries locally until the deque fills.
            self.replay_version, updates = await self.storage_helper.get_replay_buffer_updates.remote(self.replay_version)
            self.replay_buffer.extend(updates)
    
    async def append_replay_buffer(self, new_item):
        self.replay_buffer.append(new_item)
        await self.storage_helper.append_replay_buffer.remote(new_item)

    async def append_gameplay_buffer(self, new_item):
        await self.storage_helper.append_gameplay_buffer.remote(new_item)

    async def take_random_gameplay(self):
        return await self.storage_helper.take_random_gameplay.remote()

# Call-time factories: every architecture knob resolved from the module
# constants AT CONSTRUCTION, so arm overrides always apply. The def-time/
# import-time capture bug class lives wherever these kwargs get threaded by
# hand -- six sites had grown six slightly different lists (audit,
# 2026-08-28) -- so all fresh-network construction funnels through here.
# The lagged target model deliberately does NOT: it mirrors the LIVE nets'
# geometry so its strict state_dict sync can never meet a shape mismatch.
def build_representation(input_channels=12):
    return RepresentationNetwork(input_channels=input_channels,
                                 latent_channels=LATENT_CHANNELS,
                                 trunk_blocks=REP_TRUNK_BLOCKS,
                                 renormalize=LATENT_RENORMALIZE)


def build_dynamics(action_space, support_size=None, num_bins=None, transform=True):
    return DynamicsNetwork(latent_channels=LATENT_CHANNELS,
                           action_space=action_space,
                           support_size=REWARD_SUPPORT if support_size is None else support_size,
                           num_bins=REWARD_BINS if num_bins is None else num_bins,
                           transform=transform,
                           blocks=DYN_BLOCKS,
                           action_encoding=ACTION_ENCODING,
                           reward_channels=REWARD_HEAD_CHANNELS,
                           renormalize=LATENT_RENORMALIZE)


def build_prediction(action_space, support_size=None, num_bins=None, transform=True):
    return PredictionNetwork(latent_channels=LATENT_CHANNELS,
                             action_space=action_space,
                             support_size=VALUE_SUPPORT if support_size is None else support_size,
                             num_bins=VALUE_BINS if num_bins is None else num_bins,
                             transform=transform,
                             head_channels=PRED_HEAD_CHANNELS,
                             fc_hidden=PRED_FC_HIDDEN)


def build_value_prefix(support_size=None, num_bins=None, transform=True):
    return ValuePrefixNetwork(latent_channels=LATENT_CHANNELS,
                              hidden_size=VALUE_PREFIX_HIDDEN,
                              support_size=VALUE_PREFIX_SUPPORT if support_size is None else support_size,
                              num_bins=VALUE_PREFIX_BINS if num_bins is None else num_bins,
                              transform=transform,
                              head_channels=PREFIX_HEAD_CHANNELS,
                              fc_hidden=PREFIX_FC_HIDDEN)


def build_projector():
    return LatentProjector(LATENT_CHANNELS * LATENT_HW * LATENT_HW)


#create versioning like storage
class NetworkHelper:
    def __init__(self, action_space, device, checkpoint_path = 'checkpoint.pth', enable_logging=False):
        self.device = device
        self.action_space = action_space
        self.representation = build_representation().to(device)
        self.dynamics = build_dynamics(action_space).to(device)
        self.prediction = build_prediction(action_space).to(device)
        self.value_prefix = build_value_prefix().to(device)
        self.projector = build_projector().to(device)
        self.rep_optimizer = make_optimizer(self.representation.parameters())
        self.dyn_optimizer = make_optimizer(self.dynamics.parameters())
        self.pred_optimizer = make_optimizer(self.prediction.parameters())
        self.prefix_optimizer = make_optimizer(self.value_prefix.parameters())
        self.proj_optimizer = make_optimizer(self.projector.parameters())
        self.training_steps = 0
        self.logger = configure_actor_logger("NetworkHelper", enable_logging)
        self.load_checkpoint(checkpoint_path)
        # The helper stores and serves weights -- TrainingHelper trains its own
        # copies and pushes them back. Its only forward passes are inference
        # (the viewer path), so BatchNorm runs in eval mode here too.
        for net in (self.representation, self.dynamics, self.prediction,
                    self.value_prefix, self.projector):
            net.eval()
        self.version = uuid.uuid4()

    def save_checkpoint(self, training_steps, checkpoint_path = 'checkpoint.pth'):
        self.training_steps = training_steps
        checkpoint = {
            'representation_state_dict': self.representation.state_dict(),
            'dynamics_state_dict': self.dynamics.state_dict(),
            'prediction_state_dict': self.prediction.state_dict(),
            'rep_optimizer_state_dict': self.rep_optimizer.state_dict(),
            'dyn_optimizer_state_dict': self.dyn_optimizer.state_dict(),
            'pred_optimizer_state_dict': self.pred_optimizer.state_dict(),
            'value_prefix_state_dict': self.value_prefix.state_dict(),
            'prefix_optimizer_state_dict': self.prefix_optimizer.state_dict(),
            'projector_state_dict': self.projector.state_dict(),
            'proj_optimizer_state_dict': self.proj_optimizer.state_dict(),
            'training_steps': training_steps
        }
        torch.save(checkpoint, checkpoint_path)
        self.version = uuid.uuid4()
        self.logger.info("Checkpoint saved at %s", checkpoint_path)
        return self.version

    def load_checkpoint(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.representation.load_state_dict(checkpoint['representation_state_dict'])
            self.dynamics.load_state_dict(checkpoint['dynamics_state_dict'])
            self.prediction.load_state_dict(checkpoint['prediction_state_dict'])
            self.rep_optimizer.load_state_dict(checkpoint['rep_optimizer_state_dict'])
            self.dyn_optimizer.load_state_dict(checkpoint['dyn_optimizer_state_dict'])
            self.pred_optimizer.load_state_dict(checkpoint['pred_optimizer_state_dict'])
            # The prefix head was SAVED but never LOADED: every restart resumed
            # with a fresh value-prefix head and optimizer while training_steps
            # claimed otherwise. Guarded like the projector for old files.
            if 'value_prefix_state_dict' in checkpoint:
                self.value_prefix.load_state_dict(checkpoint['value_prefix_state_dict'])
                self.prefix_optimizer.load_state_dict(checkpoint['prefix_optimizer_state_dict'])
            else:
                self.logger.info("Checkpoint predates the prefix head; keeping fresh init")
            # Checkpoints written before the SimSiam projector existed have no
            # entry for it; a fresh projector is fine (it is training-only and
            # carries no policy), so old checkpoints must keep loading.
            if 'projector_state_dict' in checkpoint:
                self.projector.load_state_dict(checkpoint['projector_state_dict'])
                self.proj_optimizer.load_state_dict(checkpoint['proj_optimizer_state_dict'])
            else:
                self.logger.info("Checkpoint predates the projector; keeping fresh init")
            self.training_steps =  checkpoint['training_steps']
            self.logger.info("Checkpoint loaded successfully from %s", checkpoint_path)
        else:
            self.logger.info("No checkpoint found at %s", checkpoint_path)
    
    def initial_inference(self, obs):
        # no_grad must be applied here, not at the call site: Ray actors run in
        # their own process, so a `with torch.no_grad()` in the driver has no
        # effect on the graph built inside this method.
        with torch.no_grad():
            converted_obs = torch.tensor(obs, dtype=torch.float32, device=self.device) / 255.0
            converted_obs = converted_obs.unsqueeze(0)
            latent = self.representation(converted_obs)
            policy_logits, value, _ = self.prediction(latent)
            policy = F.softmax(policy_logits, dim=1).cpu().numpy()[0]
        return policy, value.item()

    def update_state(self, state, path='checkpoint.pth'):
        self.representation.load_state_dict(state["representation"])
        self.dynamics.load_state_dict(state["dynamics"])
        self.prediction.load_state_dict(state["prediction"])
        self.value_prefix.load_state_dict(state["value_prefix"])
        self.projector.load_state_dict(state["projector"])
        self.rep_optimizer.load_state_dict(state["rep_optimizer"])
        self.dyn_optimizer.load_state_dict(state["dyn_optimizer"])
        self.pred_optimizer.load_state_dict(state["pred_optimizer"])
        self.prefix_optimizer.load_state_dict(state["prefix_optimizer"])
        self.proj_optimizer.load_state_dict(state["proj_optimizer"])
        self.save_checkpoint(state["training_steps"], checkpoint_path=path)

    def get_state(self):
        """Weights and optimiser state as a dict.

        This was an 8-tuple unpacked positionally in three places. Adding a
        fourth network to that arrangement is how misalignment bugs are made.
        """
        return {
            "representation":   get_state_dict(self.representation),
            "dynamics":         get_state_dict(self.dynamics),
            "prediction":       get_state_dict(self.prediction),
            "value_prefix":     get_state_dict(self.value_prefix),
            "projector":        get_state_dict(self.projector),
            "rep_optimizer":    get_state_dict(self.rep_optimizer),
            "dyn_optimizer":    get_state_dict(self.dyn_optimizer),
            "pred_optimizer":   get_state_dict(self.pred_optimizer),
            "prefix_optimizer": get_state_dict(self.prefix_optimizer),
            "proj_optimizer":   get_state_dict(self.proj_optimizer),
            "training_steps":   self.training_steps,
            "version":          self.version,
        }

    def is_up_to_date(self, version_cmp):
        return self.version == version_cmp

class NetworkManager:
    def __init__(self, network_helper, action_space, device, load_dynamics = True):
        self.network_helper = network_helper
        self.load_dynamics = load_dynamics
        self.device = device
        self.representation = build_representation().to(device)
        self.prediction = build_prediction(action_space).to(device)
        if self.load_dynamics:
            self.dynamics = build_dynamics(action_space).to(device)
            self.value_prefix = build_value_prefix().to(device)
        # Inference-only consumer: BatchNorm must read the running statistics
        # training accumulated, not the statistics of whatever batch -- or
        # single acting frame -- happens to pass through (which also polluted
        # the running stats themselves). load_state_dict refreshes weights and
        # running stats but never touches this flag, so once here holds across
        # every weight pull.
        self.representation.eval()
        self.prediction.eval()
        if self.load_dynamics:
            self.dynamics.eval()
            self.value_prefix.eval()

    async def setup(self):
        await self._pull_state()

    async def _pull_state(self):
        state = await self.network_helper.get_state.remote()
        self.training_steps = state["training_steps"]
        self.version = state["version"]
        self.representation.load_state_dict(state["representation"])
        self.prediction.load_state_dict(state["prediction"])
        if self.load_dynamics:
            self.dynamics.load_state_dict(state["dynamics"])
            self.value_prefix.load_state_dict(state["value_prefix"])

    def initial_inference(self, obs):
        # no_grad must be applied here, not at the call site: Ray actors run in
        # their own process, so a `with torch.no_grad()` in the driver has no
        # effect on the graph built inside this method.
        with torch.no_grad():
            converted_obs = torch.tensor(obs, dtype=torch.float32, device=self.device) / 255.0
            converted_obs = converted_obs.unsqueeze(0)
            latent = self.representation(converted_obs)
            policy_logits, value, _ = self.prediction(latent)
            policy = F.softmax(policy_logits, dim=1).cpu().numpy()[0]
        return policy, value.item()
    
    def batch_inference(self, batch_obs):
        # Without no_grad the returned latents carry grad_fn, and because every
        # gameplay node stores one for the lifetime of the episode, the entire
        # forward graph (all ResNet activations for the batch) is pinned in GPU
        # memory. That was several GB per episode.
        with torch.no_grad():
            converted = torch.tensor(batch_obs, dtype=torch.float32, device=self.device) / 255.0
            batch_latents = self.representation(converted)
            batch_policy_logits, batch_value, _ = self.prediction(batch_latents)
            batch_policy = F.softmax(batch_policy_logits, dim=1).cpu().numpy()
            # (B, 1) -> (B,) so node.expanded_value is a scalar, matching the
            # floats used for nodes created in batch_expand_nodes.
            batch_value = batch_value.squeeze(-1).cpu().numpy()
        return batch_latents, batch_policy, batch_value
    
    async def update_if_needed(self):
        if not await self.network_helper.is_up_to_date.remote(self.version):
            await self._pull_state()

class MCTSHelper:
    def __init__(self, action_space, device, network_ref, storage_ref, enable_logging=False, name="MCTSHelper"):
        self.network_manager = NetworkManager(network_ref, action_space=action_space, device=device)
        self.storage_manager = StorageManager(storage_ref, update_replay=False)
        self.device = device
        self.action_space = action_space
        self.selection_results = asyncio.Queue()
        self.min_max_stats = MinMaxStats()  # replaced per search in run_full_mcts
        self.last_search_action = None      # set by run_gumbel_mcts; read by act_search
        self.logger = configure_actor_logger(name, enable_logging)
        self.logger.info("Initialized MCTSHelper logger.")

    async def setup(self):
        await self.network_manager.setup()
        await self.storage_manager.setup()
    
    def eval_obs(self, game_trajectory, batch_size=32):
        new_traj = []
        for i in range(0, len(game_trajectory), batch_size):
            batch_transitions = game_trajectory[i:i+batch_size]
            # Pre-allocate a numpy array for observations.
            # We assume each observation has the same shape, so we use the first element.
            sample_obs = batch_transitions[0][0]
            batch_obs = np.empty((len(batch_transitions), *sample_obs.shape), dtype=sample_obs.dtype)
            for j, transition in enumerate(batch_transitions):
                batch_obs[j] = transition[0]
            # Now batch_obs is a single numpy array instead of a list of arrays.
            latents, policies, values = self.network_manager.batch_inference(batch_obs)
            for j, transition in enumerate(batch_transitions):
                new_traj.append([*transition, latents[j:j+1], policies[j], values[j]])
        return new_traj

    async def MCTS_process(self):
        try:
            while True:
                await self.storage_manager.update_if_needed()
                await self.network_manager.update_if_needed()
                game_trajectory_tuple = await self.storage_manager.take_random_gameplay()
                if game_trajectory_tuple == None:
                    self.logger.debug("MCTS_process sleeping")
                    await asyncio.sleep(1)
                    continue
                else:
                    game_trajectory, _, terminated = game_trajectory_tuple
                if not game_trajectory:
                    self.logger.warning("Skipping empty game trajectory")
                    continue
                self.logger.debug("Running MCTS_process")
                trajectory = []
                game_trajectory = self.eval_obs(game_trajectory=game_trajectory)
                self.create_tree_from_game(game_trajectory=game_trajectory,
                                           terminated=terminated)
                if self.network_manager.training_steps < STEPS_UNTIL_MCTS:
                    self.logger.info("%s steps until MCTS", STEPS_UNTIL_MCTS - self.network_manager.training_steps)
                    for idx, (obs_sample, action_taken, reward, _, _, _) in enumerate(game_trajectory):
                        transition = {
                            "obs": obs_sample,
                            "action": action_taken,
                            "reward": reward,
                            "target_policy": np.ones(self.action_space) / self.action_space,
                            # Pure Monte Carlo during warmup: the network is
                            # untrained, so an n-step target would bootstrap
                            # noise.
                            "target_value": self.get_value(idx),
                            # The uniform policy is a placeholder, not a
                            # target. Training toward it at full weight spent
                            # the whole warmup confidently pulling the logits
                            # flat, which the first real search targets then
                            # had to fight back out of. An untrained head is
                            # already high-entropy, so masking loses nothing.
                            "has_policy_target": False,
                        }
                        trajectory.append(transition)
                else:
                    searched = self.select_reanalysis_indices(len(game_trajectory))
                    self.logger.info("MCTS on %s of %s timesteps (stride %s)",
                                     len(searched), len(game_trajectory), REANALYSE_STRIDE)
                    # Pass 1: search. Root values are collected as BOOTSTRAPS
                    # for pass 2, not as the searched positions' own value
                    # targets. A root value is ~97% network-value backups (32
                    # sims bootstrapping the value head at the leaves against
                    # one real-trajectory visit), so training a position on its
                    # own root value feeds the head search-filtered versions of
                    # its own predictions -- a systematic error inherits itself
                    # through the target. Ascending order preserves the shared
                    # chain-tree's forward-only reuse.
                    search_results = {}
                    order = sorted(searched)
                    communal = USE_COMMUNAL_REANALYSIS and not USE_GUMBEL_SEARCH
                    if USE_COMMUNAL_REANALYSIS and USE_GUMBEL_SEARCH:
                        self.logger.warning(
                            "USE_COMMUNAL_REANALYSIS supports only the PUCT "
                            "search; USE_GUMBEL_SEARCH is on, falling back "
                            "to sequential per-root searches.")
                    if communal:
                        # Windows advance in ascending order, so release_span
                        # can free everything up to each window's last root:
                        # no later search descends backward along the chain.
                        released_up_to = -1
                        for w0 in range(0, len(order), COMMUNAL_WINDOW_ROOTS):
                            window = order[w0:w0 + COMMUNAL_WINDOW_ROOTS]
                            self.logger.debug("Communal window %s..%s", window[0], window[-1])
                            search_results.update(await self.run_communal_mcts(
                                window, num_simulations=TOTAL_MCTS_SIMS,
                                c_puct=C_PUCT, discount=DISCOUNT))
                            self.release_span(released_up_to + 1, window[-1])
                            released_up_to = window[-1]
                    else:
                        for idx in order:
                            self.logger.debug("Running MCTS index %s out of %s", idx, len(game_trajectory))
                            search_results[idx] = await self.run_full_mcts(idx,
                                                                           sim_batches=MCTS_BATCHES,
                                                                           num_simulations=TOTAL_MCTS_SIMS,
                                                                           c_puct=C_PUCT,
                                                                           discount=DISCOUNT)
                    search_values = {i: r[1] for i, r in search_results.items()}
                    # Pass 2: every position gets the paper's n-step target --
                    # TD_STEPS real rewards, then the nearest search root value
                    # at or past the horizon. Facts first; the network's
                    # opinion only speaks about what lies beyond them.
                    for idx, (obs_sample, action_taken, reward, _, _, _) in enumerate(game_trajectory):
                        if idx in search_results:
                            improved_policy, _, has_policy_target = search_results[idx]
                        else:
                            # No improved policy here: masked out rather than
                            # trained toward a placeholder that would drag the
                            # head uniform.
                            improved_policy = np.ones(self.action_space) / self.action_space
                            has_policy_target = False
                        trajectory.append({
                            "obs": obs_sample,
                            "action": action_taken,
                            "reward": reward,
                            "target_policy": improved_policy,
                            "target_value": self.nstep_value(idx, search_values=search_values),
                            "has_policy_target": has_policy_target,
                        })
                
                if trajectory:
                    trajectory[-1]["truncated"] = not terminated
                await self.storage_manager.append_replay_buffer(trajectory)
        except Exception:
            # exception() rather than error(): this is the only record of why
            # the actor stopped, and "%s" on the exception drops the traceback.
            self.logger.exception("MCTS_process failed")
        self.logger.info("MCTS process exiting")
    
    # -----------------------------
    # Batch expansion helper function
    # -----------------------------
    def batch_expand_nodes(self, nodes, batch_size=40):
        """
        Given a list of MCTSNode objects that need expansion, expand them in batches.
        
        Args:
        nodes: list of MCTSNode objects to expand.
        batch_size: number of nodes to process in a single batch. (actual batch size is batch_size*self.action_space)
        """
        # Terminal nodes have no successors, and re-expanding an expanded node
        # would discard its accumulated statistics. Nodes whose latent has been
        # released cannot be expanded at all. Filter all three out before the
        # network call rather than after, so we don't pay for them.
        nodes = [n for n in nodes
                 if not n.done and not n.is_expanded and n.latent is not None]
        if not nodes:
            return

        A = self.action_space  # number of actions
        N = len(nodes)
        
        # Process nodes in chunks of batch_size.
        for i in range(0, N, batch_size):
            chunk = nodes[i:i + batch_size]
            num_chunk = len(chunk)
            
            # Gather latent states from each node in the chunk.
            latent_list = [node.latent for node in chunk]  # each shape: (1, C, H, W)
            # Stack into a tensor of shape (num_chunk, C, H, W)
            stacked_latents = torch.cat(latent_list, dim=0)
            # For each node, we want to evaluate dynamics for each action.
            # Repeat each latent A times: resulting shape (num_chunk * A, C, H, W)
            latent_batch = stacked_latents.repeat_interleave(A, dim=0)
            latent_batch = latent_batch.to(self.device)
            
            # Create a tensor of action indices. For each node in the chunk, we need actions 0,...,A-1.
            action_tensor = torch.arange(A, dtype=torch.long, device=self.device).unsqueeze(0).repeat(num_chunk, 1)
            action_tensor = action_tensor.view(-1)  # shape: (num_chunk * A,)
        
            with torch.no_grad():
                # next_latents: (num_chunk*A, C, H, W)
                # rewards: (num_chunk*A, 1) or (num_chunk*A,) depending on implementation
                next_latents, rewards, _ = self.network_manager.dynamics(latent_batch, action_tensor)
                batch_policy_logits, estimated_values, _ = self.network_manager.prediction(next_latents)
                next_policy = F.softmax(batch_policy_logits, dim=1).cpu().numpy()

                child_prefixes, child_states, prefix_resets = self.expand_prefixes(
                    chunk, next_latents, A)
            
            # Update each node in the chunk.
            for j, node in enumerate(chunk):
                start = j * A
                end = start + A
                # Assume node.policy is already a numpy array of shape (A,)
                policy = node.policy
                for a in range(A):
                    if a in node.children and node.children[a].is_gameplay:
                        continue
                    idx = start + a
                    if child_prefixes is None:
                        edge_reward = (rewards[idx].item() if rewards[idx].ndim == 0
                                       else rewards[idx].squeeze().item())
                        child_prefix, child_state, child_len = 0.0, None, 0
                    else:
                        # Within a segment the edge reward is the increment
                        # over the parent's accumulation; at a reset boundary
                        # (fresh LSTM state) the child's prefix IS the edge
                        # reward -- EfficientZero's is_reset backup, where
                        # true_reward = value_prefix rather than the
                        # difference. Undiscounted either way;
                        # backup_simulation applies the discount.
                        child_prefix = float(child_prefixes[idx])
                        edge_reward = (child_prefix if prefix_resets[j]
                                       else child_prefix - node.prefix)
                        child_state = (child_states[0][:, idx:idx + 1].clone(),
                                       child_states[1][:, idx:idx + 1].clone())
                        child_len = (1 if prefix_resets[j]
                                     else node.prefix_len + 1)
                    # Create the child node with latent state
                    child = MCTSNode(
                        latent=next_latents[idx].unsqueeze(0),
                        prior=policy[a],
                        reward=edge_reward,
                        policy=next_policy[idx],
                        expanded_value=estimated_values[idx].item(),
                        prefix=child_prefix,
                        prefix_state=child_state,
                        prefix_len=child_len,
                    )
                    node.children[a] = child
                node.is_expanded = True
        
    def expand_prefixes(self, chunk, next_latents, action_space):
        """Step the value-prefix LSTM from each parent's state, once per child.

        Returns (prefixes, state, resets) aligned with next_latents, where
        resets[j] says parent j's children start a FRESH accumulation --
        either the parent had no state (search root, gameplay-chain restart)
        or it had run PREFIX_LSTM_HORIZON steps, EfficientZero's
        lstm_horizon_len reset "every horizon steps in search". Returns
        (None, None, None) when the tree uses the per-step reward head.
        """
        if not USE_VALUE_PREFIX_IN_TREE or not hasattr(self.network_manager, "value_prefix"):
            return None, None, None

        prefix_net = self.network_manager.value_prefix
        parent_states = []
        resets = []
        for node in chunk:
            state = node.prefix_state
            reset = state is None or node.prefix_len >= PREFIX_LSTM_HORIZON
            if reset:
                state = prefix_net.initial_state(1, self.device)
            parent_states.append(state)
            resets.append(reset)

        # (1, num_chunk, H) -> one copy per action -> (1, num_chunk*A, H)
        h = torch.cat([s[0] for s in parent_states], dim=1).repeat_interleave(action_space, dim=1)
        c = torch.cat([s[1] for s in parent_states], dim=1).repeat_interleave(action_space, dim=1)

        was_training = prefix_net.training
        prefix_net.eval()
        try:
            prefixes, _, state = prefix_net(next_latents, (h.contiguous(), c.contiguous()))
        finally:
            prefix_net.train(was_training)
        return prefixes.squeeze(-1).cpu().numpy(), state, resets

    # -----------------------------
    # Backup Phase: Propagate the leaf value up the simulation path.
    # -----------------------------
    def backup_simulation(self, path, leaf_value, discount=None, min_max_stats=None,
                          count_root=True):
        # Resolved at call time, not baked into the signature: create_tree_from_game
        # calls this without a discount, so a hardcoded default silently pinned the
        # Monte Carlo value targets to 0.997 no matter what DISCOUNT was set to.
        # That made any experiment varying gamma measure nothing.
        discount = DISCOUNT if discount is None else discount
        backup_value = leaf_value
        last = len(path) - 1
        if count_root and path:
            # The root's own visit crossed no edge into it (see
            # COMMUNAL_EDGE_VISITS). The chain seeding passes False: its
            # visits are bookkeeping, not a search.
            path[0].root_visits += 1
        for i, n in enumerate(reversed(path)):
            n.visit_count += 1
            n.value_sum += backup_value
            n.virtual_loss = 0.0
            backup_value = n.reward + discount * backup_value
            if min_max_stats is not None and i != last:
                # Track the same composite selection normalises: r + discount*V,
                # not V alone, or the recorded range is on the wrong scale.
                # The search ROOT (path[0], iterated last) is excluded: its
                # reward is the edge INTO the root, outside the searched
                # subtree, and folding it in inflates the range -- compressing
                # every normalized advantage precisely at positions that follow
                # big reward events.
                min_max_stats.update(n.reward + discount * n.q_value())

    def create_tree_from_game(self, game_trajectory, terminated=True,
                              seed_values=True):
        """Link a played trajectory into the chain tree.

        seed_values=False builds the chain NEUTRALLY: structure, priors,
        real edge rewards and the terminal flag, but no Monte-Carlo value
        backup -- every chain node starts at zero visits like an imagined
        sibling. The MC backup seeds the behaviour child with a realized
        return while its siblings carry network estimates, which biases
        selection toward (early training) or away from (calibrated value
        head) the action already taken; neutral seeding removes that
        asymmetry. Both the product pipeline and the communal refresh run
        SEEDED (the refresh went neutral 2026-08-27..30, but at tiny-buffer
        coverage the unanchored re-searches flattened most stored targets
        each cycle -- see _search_traj_communal); neutral remains available
        for callers that want a structure-only chain.
        """
        prev_node = None
        prev_action_taken = None
        path = []
        for (obs_sample, action_taken, reward, latent, policy, expanded_value) in game_trajectory:
            new_node = MCTSNode(latent=latent, prior=1.0, reward=reward, is_gameplay=True)
            new_node.policy = policy
            new_node.expanded_value = expanded_value
            if prev_node != None:
                new_node.prior = prev_node.policy[prev_action_taken]
                prev_node.children[prev_action_taken] = new_node
            path.append(new_node)
            prev_node = new_node
            prev_action_taken = action_taken
        path[-1].done = terminated

        # Nodes are expanded lazily, the first time a simulation selects them.
        # Expanding the whole chain up front allocated action_space child
        # latents for every timestep in the episode (~5x the episode length)
        # before the search had even started.
        #
        # Chain backup leaf: 0 at a true terminal. A TRUNCATED tail (the
        # harness episode cap) is not terminal -- backing up 0 there taught
        # V=0 at a mid-game state on every capped trajectory -- so it
        # bootstraps the network's value instead.
        if seed_values:
            leaf_value = 0.0 if terminated else (path[-1].expanded_value or 0.0)
            self.backup_simulation(path, leaf_value, count_root=False)
        self.current_trajectory = path
    
    # DIRICHLET_ALPHA defaults 0.25, MuZero's Atari setting (EfficientZero
    # uses 0.3; the repro arm overrides). The old hardcoded 0.03 was AlphaGo's
    # value, tuned for 361 moves: over 5 actions it made the Dirichlet sample
    # effectively one-hot, ~92% of the noise mass on a single random action.
    def add_noise_to_node(self, idx, epsilon = 0.25, alpha = None):
        alpha = DIRICHLET_ALPHA if alpha is None else alpha
        node = self.current_trajectory[idx]
        node.policy = add_dirichlet_noise(node.policy, epsilon=epsilon, alpha=alpha)
        for action, child in node.children.items():
            child.prior = node.policy[action]

    async def selection_task_no_expand(self, index, c_puct, discount, max_depth=None,
                                       min_max_stats=None):
        """
        Traverse the tree from the given root until a leaf is reached.

        min_max_stats defaults to the helper's per-search instance; the
        communal window driver passes each root's own stats explicitly, since
        several searches with different normalisation ranges run concurrently
        over the shared chain tree.
        """
        # None -> module global at CALL time: a def-time default froze the
        # import-time MAX_DEPTH and silently ignored arm overrides (the
        # backup_simulation-discount bug class).
        max_depth = MAX_DEPTH if max_depth is None else max_depth
        stats = self.min_max_stats if min_max_stats is None else min_max_stats
        node = self.current_trajectory[index]
        # Edge visits: selections that crossed the parent->child edge.
        # Legacy also counts a chain child's own ROOT visits (see
        # COMMUNAL_EDGE_VISITS); values (q_value) stay communal.
        edge = ((lambda c: c.visit_count - c.root_visits) if COMMUNAL_EDGE_VISITS
                else (lambda c: c.visit_count))
        path = []
        while len(path) < max_depth:
            if not node.is_expanded or node.done:
                break  # leaf: either not yet expanded, or terminal
            path.append(node)
            total_visits = sum(edge(child) + child.virtual_loss for child in node.children.values())
            best_score = -float('inf')
            best_candidates = []  # List to store (action, child) pairs with highest score.
            for a, child in node.children.items():
                u = c_puct * child.prior * np.sqrt(total_visits) / (1 + edge(child) + child.virtual_loss)
                # Score the edge as r + discount * V(child) -- the paper's
                # Q(s,a). q_value() alone is the value from the child ONWARD:
                # backup folds the edge reward into the parent's statistics, so
                # scoring on it left selection blind to immediate reward
                # differences between actions. An unvisited child -- in-flight
                # or not -- reads its network value estimate rather than 0,
                # which on this tree's raw-return scale would rank every
                # imagined child beneath the gameplay child before a single
                # simulation had judged them (and on a negative-value game
                # would rank an in-flight child ABOVE every evaluated one).
                if child.visit_count > 0:
                    onward = child.q_value()
                elif child.done:
                    onward = 0.0
                else:
                    onward = child.expanded_value
                score = stats.normalize(child.reward + discount * onward)
                if child.virtual_loss > 0:
                    # Virtual loss: each in-flight simulation counts as a
                    # completed visit at the normalized MINIMUM (0 once the
                    # stats have a range; identity before), fading as real
                    # visits accumulate, so a
                    # genuinely dominant action can still take several
                    # simulations of one batch. An unvisited child's network
                    # estimate counts as its single pseudo-visit. For
                    # score >= 0 this is algebraically identical to raw-space
                    # visits pinned at min_max_stats.minimum (normalize is
                    # affine); a BELOW-minimum child normalizes NEGATIVE and
                    # scaling it toward 0 would be a bonus, so min() holds it
                    # at its own score instead -- no penalty, never a boost
                    # (2026-08-30 re-audit).
                    n = max(edge(child), 1)
                    score = min(score, score * n / (n + child.virtual_loss))
                score += u
                if score > best_score:
                    best_score = score
                    best_candidates = [child]
                elif score == best_score:
                    best_candidates.append(child)
            # Randomly select one of the best candidates in case of a tie.
            chosen_child = random.choice(best_candidates)
            async with chosen_child.lock:
                chosen_child.virtual_loss += 1
            node = chosen_child
        path.append(node)
        await self.selection_results.put((node, path))

    async def expand_selected(self, discount, stats_by_root=None):
        simulations = []
        while not self.selection_results.empty():
            simulations.append(await self.selection_results.get())

        # Expand each distinct leaf once; batch_expand_nodes drops the terminal
        # and already-expanded ones itself.
        unique_leaves = []
        seen = set()
        for leaf, _ in simulations:
            if leaf.id not in seen:
                seen.add(leaf.id)
                unique_leaves.append(leaf)
        self.batch_expand_nodes(unique_leaves)

        # Back up EVERY simulation, not just the ones that caused an expansion.
        # Simulations dropped here previously (a duplicate leaf within the
        # batch, or a leaf that was already expanded) never reached
        # backup_simulation -- the only place virtual_loss is cleared. So
        # virtual loss accumulated monotonically on the most-visited nodes and
        # progressively steered the search away from precisely the actions it
        # had found to be good. Their visits were lost too, so the effective
        # simulation budget was well under num_simulations.
        for leaf, path in simulations:
            leaf_value = 0.0 if leaf.done else leaf.expanded_value
            # Each simulation's range update goes to ITS search's stats: in a
            # communal window the paths in one drain belong to different
            # roots, keyed here by the root that heads the path.
            stats = self.min_max_stats
            if stats_by_root is not None:
                stats = stats_by_root.get(path[0].id, self.min_max_stats)
            self.backup_simulation(path, leaf_value, discount=discount,
                                   min_max_stats=stats)

    def seed_min_max_stats(self, root, discount, stats=None):
        """Prime the per-search normalisation range from values already in the tree.

        MinMaxStats starts empty and normalize() is the identity until it has
        seen a range, so the first selection round used to score on RAW values:
        the gameplay child's backed-up Monte Carlo return (order hundreds)
        against exploration terms of order 0.3, which sent the entire first
        batch of simulations down the behaviour trajectory. Seeding from the
        root's children puts round one on the same [0,1] footing as every
        later round.
        """
        stats = self.min_max_stats if stats is None else stats
        for child in root.children.values():
            if child.visit_count > 0:
                onward = child.q_value()
            elif child.done:
                onward = 0.0
            else:
                onward = child.expanded_value
            stats.update(child.reward + discount * onward)

    def release_root(self, node):
        """Prune imagined children and release the latent once a search is done.

        Holding one latent per timestep for a whole episode is the memory cost
        this avoids. Assigning None rather than `del` keeps the attribute
        defined, so searching the same index twice -- which is what
        re-analysing a stored trajectory means -- raises nothing.
        """
        for action in [a for a, child in node.children.items() if not child.is_gameplay]:
            del node.children[action]
        node.latent = None

    def release_span(self, start_idx, end_idx):
        """Communal-window teardown: free chain nodes start_idx..end_idx.

        Traffic only flows FORWARD along the chain (a search from root i
        reaches at most i + MAX_DEPTH), so once every root <= end_idx has
        finished, nothing can descend into this span again and its imagined
        subtrees and latents can all go -- including those hanging off
        interior, never-searched chain nodes, which per-root release_root
        never visited and which previously lingered for the trajectory's
        whole lifetime. Nodes PAST end_idx keep their expansions: they are
        the next window's reuse. Visit statistics are kept everywhere (they
        are communal by design, and q_value readouts may still want them).
        """
        for i in range(max(start_idx, 0), min(end_idx + 1, len(self.current_trajectory))):
            node = self.current_trajectory[i]
            for action in [a for a, child in node.children.items()
                           if not child.is_gameplay]:
                del node.children[action]
            node.latent = None

    def drop_non_gameplay_children_and_compute_improvements(self, idx, baseline=None,
                                                            release=True):
        # release=False: the communal window keeps every root's subtree alive
        # until the whole window has finished (concurrent neighbours may still
        # descend through it); release_span then frees the window in one pass.
        baseline = baseline or {}
        node = self.current_trajectory[idx]
        # Improved policy from the visits THIS search allocated. Raw counts also
        # include the chain backup's bookkeeping visit and simulations from
        # earlier searches, which reach this node only along the gameplay chain
        # -- both sit on the behaviour action, so counting them biases every
        # policy target toward the action the agent already took. The inherited
        # visits still warm the Q estimates; they just don't get read out as
        # this search's conclusion.
        visit_counts = np.array([
            node.children[a].visit_count - baseline.get(a, 0) if a in node.children else 0
            for a in range(self.action_space)
        ])
        has_policy = bool(visit_counts.sum() > 0)
        if has_policy:
            # RAW visit fractions, per the paper. MuZero's policy target is
            # always the untempered visit distribution; temperature exists only
            # for action SELECTION during self-play. Sharpening targets on a
            # schedule -- what a decaying temperature once did here -- trains the
            # head toward the argmax of a 32-simulation count vector, i.e.
            # toward search noise, increasingly confidently as training runs.
            improved_policy = visit_counts / visit_counts.sum()
        else:
            # No visits to read -- a terminal root has no successors to search.
            # The uniform vector is a shape placeholder only: has_policy False
            # masks it out of the loss, rather than training the policy head
            # toward uniform on terminal observations.
            improved_policy = np.ones(self.action_space) / self.action_space
        root_value = node.q_value()
        if release:
            self.release_root(node)
        return improved_policy, root_value, has_policy

    # -----------------------------
    # Full MCTS: Run multiple simulations, batch expand unexpanded leaves,
    # then backup their values.
    # -----------------------------
    async def run_full_mcts(self, index, num_simulations=128, sim_batches=8, c_puct=1.0, discount=None, max_depth=None, root_noise=True):
        discount = DISCOUNT if discount is None else discount
        max_depth = MAX_DEPTH if max_depth is None else max_depth
        if USE_GUMBEL_SEARCH:
            # sim_batches is PUCT-only: the Gumbel path ignores it entirely.
            # A probe comparing batch widths with USE_GUMBEL_SEARCH left True
            # measures Gumbel twice and reports identical results
            # (2026-08-30 audit trap).
            return await self.run_gumbel_mcts(index, num_simulations=num_simulations,
                                              discount=discount, max_depth=max_depth)
        # Now that expansion is lazy the root has to be expanded explicitly:
        # selection stops at unexpanded nodes, and add_noise_to_node writes the
        # noisy prior through to the children, which must therefore exist.
        root = self.current_trajectory[index]
        # Value normalisation is per-search, as in the MuZero paper.
        self.min_max_stats = MinMaxStats()
        # The prefix accumulates from the root, so it restarts at every search.
        root.prefix = 0.0
        root.prefix_state = None
        root.prefix_len = 0
        if not root.is_expanded:
            self.batch_expand_nodes([root])
        if root_noise:
            # Exploration noise belongs to SELF-PLAY roots; the papers disable
            # it at evaluation, and act_search(greedy=True) passes False so
            # eval returns are not systematically noise-degraded.
            self.add_noise_to_node(index)
        self.seed_min_max_stats(root, discount)
        # Visits the root's children carry INTO this search: the chain backup's
        # bookkeeping visit plus traffic from earlier searches through the
        # shared tree. Snapshotted so the improved policy can count only what
        # this search's own budget decides.
        baseline_visits = {a: child.visit_count for a, child in root.children.items()}
        # Spend the budget EXACTLY: the old `sims // batches` rounds silently
        # dropped the remainder (50 sims at batch 8 ran 48). Fail fast on a
        # non-positive batch size -- the loop would otherwise spin forever,
        # and MCTS_process swallows exceptions but cannot log a hang.
        assert sim_batches > 0, f"sim_batches must be positive, got {sim_batches}"
        remaining = max(int(num_simulations), 1)
        while remaining > 0:
            batch_n = min(sim_batches, remaining)
            futures = [
                self.selection_task_no_expand(index, c_puct, discount, max_depth)
                for _ in range(batch_n)
            ]
            await asyncio.gather(*futures)
            await self.expand_selected(discount)
            remaining -= batch_n
        return self.drop_non_gameplay_children_and_compute_improvements(
            index, baseline=baseline_visits)

    async def run_communal_mcts(self, indices, num_simulations, c_puct, discount,
                                chain_seeded=True):
        """One communal window: concurrent lockstep searches over `indices`.

        Each root runs the papers' sequential schedule -- one simulation per
        round, statistics updated before its next selection -- while the GPU
        batch is formed ACROSS the window's roots, so throughput matches the
        old within-tree sim_batches shape without any selection running on
        stale statistics. The shared chain tree's statistics are COMMUNAL:
        traffic from earlier or concurrent searches stays in the counts and
        is read out rather than baseline-subtracted; only the chain backup's
        bookkeeping visit (one per gameplay child, from
        create_tree_from_game, allocated by no search) is removed from the
        policy target -- and, under COMMUNAL_EDGE_VISITS, so are the
        behaviour child's own ROOT visits: that child is the next chain
        node, whose search in this window incremented its visit_count
        once per simulation without ever crossing this root's edge.
        Roots are NOT released here -- concurrent neighbours
        may still descend through them; the caller frees the whole span with
        release_span once the window is done.

        Returns {index: (improved_policy, root_value, has_policy)}.
        """
        indices = sorted(indices)
        to_expand = [self.current_trajectory[idx] for idx in indices
                     if not self.current_trajectory[idx].is_expanded]
        self.batch_expand_nodes(to_expand)
        stats_by_index = {}
        stats_by_root = {}
        for idx in indices:
            root = self.current_trajectory[idx]
            # The prefix accumulates from each search's own root.
            root.prefix = 0.0
            root.prefix_state = None
            root.prefix_len = 0
            self.add_noise_to_node(idx)
            stats = MinMaxStats()
            self.seed_min_max_stats(root, discount, stats=stats)
            stats_by_index[idx] = stats
            stats_by_root[root.id] = stats
        for _ in range(max(int(num_simulations), 1)):
            futures = [
                self.selection_task_no_expand(idx, c_puct, discount,
                                              min_max_stats=stats_by_index[idx])
                for idx in indices
            ]
            await asyncio.gather(*futures)
            await self.expand_selected(discount, stats_by_root=stats_by_root)
        results = {}
        for idx in indices:
            root = self.current_trajectory[idx]
            # Chains built with seed_values=True carry the MC backup's one
            # bookkeeping visit on each gameplay child; subtract exactly
            # that. A neutrally-seeded chain (chain_seeded=False) has no
            # such visit -- subtracting would undercount real traffic.
            bookkeeping = ({a: 1 for a, child in root.children.items()
                            if child.is_gameplay} if chain_seeded else {})
            if COMMUNAL_EDGE_VISITS:
                for a, child in root.children.items():
                    if child.is_gameplay:
                        bookkeeping[a] = bookkeeping.get(a, 0) + child.root_visits
            results[idx] = self.drop_non_gameplay_children_and_compute_improvements(
                idx, baseline=bookkeeping, release=False)
        return results

    # -----------------------------
    # Gumbel search (Danihelka et al. 2022): sequential halving at the root,
    # deterministic visit-matching below it, and a policy target built from
    # completed Q-values instead of visit fractions.
    # -----------------------------
    async def run_gumbel_mcts(self, index, num_simulations=32, discount=None, max_depth=None):
        discount = DISCOUNT if discount is None else discount
        max_depth = MAX_DEPTH if max_depth is None else max_depth
        root = self.current_trajectory[index]
        self.min_max_stats = MinMaxStats()
        root.prefix = 0.0
        root.prefix_state = None
        if not root.is_expanded:
            self.batch_expand_nodes([root])
        if not root.children:
            # Terminal root: nothing to search, no real policy target, and no
            # informed action either.
            self.last_search_action = int(np.random.randint(self.action_space))
            return (np.ones(self.action_space) / self.action_space,
                    root.q_value(), False)
        self.seed_min_max_stats(root, discount)
        # No Dirichlet noise: the Gumbel draws ARE the exploration, fresh per
        # search, and unlike the noise they never contaminate node.policy.
        logits = np.log(np.asarray(root.policy, dtype=np.float64) + 1e-9)
        gumbels = np.random.gumbel(size=self.action_space)
        baseline = {a: child.visit_count for a, child in root.children.items()}

        m = max(1, min(GUMBEL_MAX_CANDIDATES, self.action_space, num_simulations))
        available = [int(a) for a in np.argsort(-(gumbels + logits)) if a in root.children]
        current = available[:m]

        # Sequential halving: split the budget over ceil(log2(m)) phases, visit
        # each surviving candidate equally within a phase, keep the top half by
        # g + logits + sigma(qhat). The last phase spends whatever remains.
        total_phases = max(1, int(np.ceil(np.log2(len(current))))) if len(current) > 1 else 1
        sims_left = num_simulations
        for phase in range(total_phases):
            if not current or sims_left <= 0:
                break
            if phase == total_phases - 1 or len(current) == 1:
                per_candidate = max(1, sims_left // len(current))
            else:
                per_candidate = max(1, int(num_simulations / (total_phases * len(current))))
            schedule = [a for a in current for _ in range(per_candidate)][:sims_left]
            for i in range(0, len(schedule), len(current)):
                batch = schedule[i:i + len(current)]
                await asyncio.gather(*[
                    self.gumbel_selection_task(index, a, discount, max_depth)
                    for a in batch])
                await self.expand_selected(discount)
            sims_left -= len(schedule)
            if len(current) > 1:
                current = self._halve_candidates(root, current, gumbels, logits,
                                                 discount, baseline)

        improved_policy = self._gumbel_improved_policy(root, logits, discount,
                                                       baseline=baseline)
        root_value = root.q_value()
        # The paper's acting rule: the sequential-halving survivor, scored by
        # g + logits + sigma(qhat). Stashed rather than returned so the
        # reanalysis callers' 3-tuple contract stays put; act_search reads it
        # immediately after the call (one search at a time by design).
        sigma = self._sigma_scale(root, baseline)
        self.last_search_action = max(
            current, key=lambda a: gumbels[a] + logits[a]
            + sigma * self._normalized_q(root.children[a], discount))
        self.release_root(root)
        return improved_policy, root_value, True

    async def act_search(self, obs_stack, num_simulations=None, discount=None,
                         greedy=False):
        """One search from a live observation; returns (action, policy, value).

        The papers' acting mode (USE_SEARCH_ACTING; the shipped product keeps
        it off): a fresh one-node tree per step, dispatching PUCT or Gumbel by
        USE_GUMBEL_SEARCH. PUCT mode samples the action from the visit
        distribution at the staged paper temperature (argmax when greedy);
        Gumbel mode acts with the halving survivor (improved-policy argmax
        when greedy). The returned policy/value are the search policy target
        and root value the paper data path records per step. Neither reads nor
        disturbs reanalysis state; current_trajectory is saved and restored.
        """
        num_simulations = ACT_SEARCH_SIMS if num_simulations is None else num_simulations
        latents, policies, values = self.network_manager.batch_inference(
            np.asarray(obs_stack)[None])
        root = MCTSNode(latent=latents[0:1], prior=1.0)
        root.policy = policies[0]
        root.expanded_value = float(values[0])
        saved = getattr(self, "current_trajectory", None)
        self.current_trajectory = [root]
        try:
            policy, value, has_policy = await self.run_full_mcts(
                0, num_simulations=num_simulations,
                sim_batches=min(MCTS_BATCHES, num_simulations),
                c_puct=C_PUCT, discount=discount,
                root_noise=not greedy)
            if not has_policy:
                action = int(np.random.randint(self.action_space))
            elif USE_GUMBEL_SEARCH:
                action = (int(np.argmax(policy)) if greedy
                          else int(self.last_search_action))
            else:
                temp = 1e-4 if greedy else visit_softmax_temperature(
                    getattr(self.network_manager, "training_steps", 0))
                action = int(np.random.choice(
                    self.action_space, p=apply_temperature(policy, temp)))
            return action, policy, float(value)
        finally:
            self.current_trajectory = saved

    async def gumbel_selection_task(self, index, root_action, discount, max_depth=None):
        """One simulation with the root action forced by the halving schedule;
        below the root, selection is the paper's deterministic visit-matching
        rule (no Gumbel, no PUCT)."""
        max_depth = MAX_DEPTH if max_depth is None else max_depth
        node = self.current_trajectory[index]
        path = []
        forced = root_action
        while len(path) < max_depth:
            if not node.is_expanded or node.done:
                break
            path.append(node)
            if forced is not None:
                a, forced = forced, None
            else:
                a = self._nonroot_select(node, discount)
            child = node.children[a]
            async with child.lock:
                child.virtual_loss += 1
            node = child
        path.append(node)
        await self.selection_results.put((node, path))

    def _normalized_q(self, child, discount):
        """The same composite selection scores: r + discount * V, normalized.
        Unvisited children read their network value, as in PUCT selection;
        in-flight simulations count as completed visits at the normalized
        minimum (see selection_task_no_expand), not as zero-value visits."""
        if child.visit_count > 0:
            onward = child.q_value()
        elif child.done:
            onward = 0.0
        else:
            onward = child.expanded_value
        qn = self.min_max_stats.normalize(child.reward + discount * onward)
        if child.virtual_loss > 0:
            n = max(child.visit_count, 1)
            # min() mirrors the PUCT clamp: a below-minimum child
            # normalizes NEGATIVE, and scaling it toward 0 would raise it
            # (2026-08-30 verification round).
            qn = min(qn, qn * n / (n + child.virtual_loss))
        return qn

    def _sigma_scale(self, node, baseline=None):
        baseline = baseline or {}
        max_n = max((child.visit_count - baseline.get(a, 0)
                     for a, child in node.children.items()), default=0)
        return (GUMBEL_C_VISIT + max_n) * GUMBEL_C_SCALE

    def _halve_candidates(self, root, current, gumbels, logits, discount, baseline):
        sigma = self._sigma_scale(root, baseline)
        scored = sorted(current,
                        key=lambda a: gumbels[a] + logits[a]
                        + sigma * self._normalized_q(root.children[a], discount),
                        reverse=True)
        return scored[:max(1, len(current) // 2)]

    def _gumbel_improved_policy(self, node, logits, discount, baseline=None):
        """softmax(logits + sigma(completedQ)) over ALL actions -- the paper's
        policy target. Visited actions contribute their search Q; unvisited
        ones are completed with v_mix, the visit-weighted blend of the node's
        own network value and the values its visited children returned. This
        is what stays informative at small budgets: the target is built from
        value ESTIMATES, not from a handful of noisy visit counts."""
        baseline = baseline or {}
        prior = np.asarray(node.policy, dtype=np.float64)
        visited = {a: child for a, child in node.children.items()
                   if child.visit_count - baseline.get(a, 0) > 0}
        qhat = {a: self._normalized_q(child, discount) for a, child in visited.items()}
        own_value = node.expanded_value if node.expanded_value is not None else node.q_value()
        v_hat = self.min_max_stats.normalize(own_value)
        if visited:
            n_total = sum(child.visit_count - baseline.get(a, 0)
                          for a, child in visited.items())
            pi_visited = max(sum(prior[a] for a in visited), 1e-9)
            weighted = sum(prior[a] * qhat[a] for a in visited) / pi_visited
            v_mix = (v_hat + n_total * weighted) / (1.0 + n_total)
        else:
            v_mix = v_hat
        completed = np.full(self.action_space, v_mix)
        for a, q in qhat.items():
            completed[a] = q
        z = logits + self._sigma_scale(node, baseline) * completed
        z -= z.max()
        pi = np.exp(z)
        return pi / pi.sum()

    def _nonroot_select(self, node, discount):
        """Deterministic visit-matching: pick the action whose visit share lags
        its improved-policy probability the most. Distributes descents in
        proportion to pi' without any sampling noise; virtual loss counts as a
        visit, which is what diversifies concurrent simulations."""
        pi = self._gumbel_improved_policy(
            node, np.log(np.asarray(node.policy, dtype=np.float64) + 1e-9), discount)
        counts = np.zeros(self.action_space)
        for a, child in node.children.items():
            counts[a] = child.visit_count + child.virtual_loss
        total = counts.sum()
        score = pi - counts / (1.0 + total)
        return max(node.children.keys(), key=lambda a: score[a])

    def select_reanalysis_indices(self, length, stride=None):
        """Timesteps to run a search on, evenly spaced with a random offset.

        A stride rather than a random sample, so coverage along the episode is
        uniform; a random offset so successive passes over the buffer land on
        different positions and the whole episode gets covered over time.
        """
        stride = REANALYSE_STRIDE if stride is None else stride
        if length <= 0:
            return set()
        if stride <= 1:
            return set(range(length))
        # Cap the offset at the trajectory length: an episode shorter than the
        # stride would otherwise draw an offset past its end and be reanalysed
        # with no searched timesteps at all, contributing zero policy targets.
        offset = random.randrange(min(stride, length))
        return set(range(offset, length, stride))

    def get_value(self, idx):
        return self.current_trajectory[idx].q_value()

    def nstep_value(self, idx, n=None, discount=None, search_values=None):
        """MuZero-Reanalyze value target: m real rewards, then a bootstrap.

            z_t = r_{t+1} + g*r_{t+2} + ... + g^(m-1)*r_{t+m} + g^m * v_{t+m}

        The real rewards are the anchor: they enter the target no matter what
        the network believes, so a systematic value-head error cannot confirm
        itself through its own targets. The paper's v is a SEARCH value, so
        the bootstrap prefers the nearest searched root value at or past
        idx+n (search_values, collected in pass 1 of this trajectory's
        reanalysis) -- stretching the horizon m by up to REANALYSE_STRIDE-1
        steps to land on a searched index, which the discount accounts for.
        Where no search lies ahead it falls back to the current network's
        expanded_value at exactly idx+n; where the episode ends inside the
        horizon it is the pure Monte Carlo return, since nothing follows a
        terminal state. A search value at idx itself is never consulted:
        a position's own target always opens with real rewards.
        """
        n = TD_STEPS if n is None else n
        discount = DISCOUNT if discount is None else discount
        traj = self.current_trajectory
        tail = idx + n
        if search_values:
            ahead = [j for j in search_values if j >= tail]
            if ahead:
                tail = min(ahead)
        # Clamp the horizon to the trajectory: the tail node's own done flag
        # then decides terminal-vs-truncated. The unclamped early return below
        # skipped that check entirely, so a TRUNCATED (capped) tail still got
        # the implicit V=0 of a pure-MC cutoff -- the very bug the truncation
        # plumbing was built to remove, resurfacing one consumer downstream.
        # For idx at the tail itself this bootstraps the node's own network
        # value (an identity target), which is the standard truncation
        # boundary treatment.
        tail = min(tail, len(traj) - 1)
        value, g = 0.0, 1.0
        for k in range(1, tail - idx + 1):
            if idx + k >= len(traj):
                return value
            value += g * traj[idx + k].reward
            g *= discount
        node = traj[tail]
        if node.done:
            return value
        # tail != idx guards the invariant three comments in this repo exist
        # to protect: a position is NEVER trained on its own search value.
        # The horizon clamp made tail == idx reachable (a truncated tail), and
        # if that index was searched, the un-guarded lookup handed the
        # position its own root value at full weight with zero reward anchor.
        if search_values and tail in search_values and tail != idx:
            return value + g * search_values[tail]
        return value + g * node.expanded_value

class TrainingHelper:
    def __init__(self, action_space, device, network_ref, storage_ref, enable_logging=False):
        self.network_helper = network_ref
        self.storage_manager = StorageManager(storage_ref)  
        self.device = device
        self.action_space = action_space
        self.training_steps = 0
        self.representation = build_representation().to(device)
        self.dynamics = build_dynamics(action_space).to(device)
        self.prediction = build_prediction(action_space).to(device)
        self.value_prefix = build_value_prefix().to(device)
        self.projector = build_projector().to(device)
        self.rep_optimizer = make_optimizer(self.representation.parameters())
        self.dyn_optimizer = make_optimizer(self.dynamics.parameters())
        self.pred_optimizer = make_optimizer(self.prediction.parameters())
        self.prefix_optimizer = make_optimizer(self.value_prefix.parameters())
        self.proj_optimizer = make_optimizer(self.projector.parameters())
        self.logger = configure_actor_logger("TrainingHelper", enable_logging)
        self.logger.info("Initialized TrainingHelper logger.")

    async def setup(self):
        await self.storage_manager.setup()
        await self.load_networks()
    
    async def load_networks(self):
        state = await self.network_helper.get_state.remote()
        self.training_steps = state["training_steps"]
        self.network_version = state["version"]
        self.representation.load_state_dict(state["representation"])
        self.dynamics.load_state_dict(state["dynamics"])
        self.prediction.load_state_dict(state["prediction"])
        self.value_prefix.load_state_dict(state["value_prefix"])
        self.projector.load_state_dict(state["projector"])
        self.rep_optimizer.load_state_dict(state["rep_optimizer"])
        self.dyn_optimizer.load_state_dict(state["dyn_optimizer"])
        self.pred_optimizer.load_state_dict(state["pred_optimizer"])
        self.prefix_optimizer.load_state_dict(state["prefix_optimizer"])
        self.proj_optimizer.load_state_dict(state["proj_optimizer"])

    async def save_networks(self):
        await self.network_helper.update_state.remote({
            "representation":   self.representation.state_dict(),
            "dynamics":         self.dynamics.state_dict(),
            "prediction":       self.prediction.state_dict(),
            "value_prefix":     self.value_prefix.state_dict(),
            "projector":        self.projector.state_dict(),
            "rep_optimizer":    self.rep_optimizer.state_dict(),
            "dyn_optimizer":    self.dyn_optimizer.state_dict(),
            "pred_optimizer":   self.pred_optimizer.state_dict(),
            "prefix_optimizer": self.prefix_optimizer.state_dict(),
            "proj_optimizer":   self.proj_optimizer.state_dict(),
            "training_steps":   self.training_steps,
        })
        
    
    # -----------------------------
    # Training Step: remote method to update networks
    # -----------------------------
    async def train_step(self, batch):
        """
        Multi-step unrolling training step.
        
        Expects a batch dictionary with keys:
        "obs": (B, unroll_steps+1, C, H, W) uint8 images,
        "actions": (B, unroll_steps) ints,
        "rewards": (B, unroll_steps+1) floats,
        "target_policy": (B, unroll_steps+1, action_space) probabilities,
        "target_value": (B, unroll_steps+1) floats.
        """
        # Determine the number of unroll steps from the batch.
        unroll_steps = batch["actions"].shape[1]
        
        # Convert inputs to tensors and move them to the appropriate device.
        obs = torch.tensor(batch["obs"], dtype=torch.float32, device=self.device) / 255.0
        if USE_AUGMENTATION:
            obs = augment_observations(obs)
        actions = torch.tensor(batch["actions"], dtype=torch.long, device=self.device)
        rewards = torch.tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        target_policy = torch.tensor(batch["target_policy"], dtype=torch.float32, device=self.device)
        target_value = torch.tensor(batch["target_value"], dtype=torch.float32, device=self.device)
        # Positions whose target_policy is a real improved policy rather than a
        # placeholder. Reanalysis only searches every REANALYSE_STRIDE-th step,
        # and trajectory padding is never a real target.
        policy_mask = torch.tensor(np.asarray(batch["has_policy_target"]),
                                   dtype=torch.float32, device=self.device)
        # PER importance-sampling weights (paper mode). Applied to the ENTIRE
        # per-sample loss -- EfficientZero's weighted_loss = (weights * loss)
        # .mean(); a first version weighted only the value loss, which leaves
        # the policy/prefix/consistency gradients biased by the non-uniform
        # sampling with no correction.
        is_weights = None
        if "is_weights" in batch:
            is_weights = torch.tensor(batch["is_weights"], dtype=torch.float32,
                                      device=self.device)

        def wmean(per_sample):
            if is_weights is None:
                return per_sample.mean()
            return (per_sample * is_weights).mean()
        
        # EfficientZero's consistency mask (CONSISTENCY_MASK_PADDING): unroll
        # positions past the trajectory's end contribute no consistency loss.
        real_mask = None
        if CONSISTENCY_MASK_PADDING:
            if "real_mask" not in batch:
                raise KeyError("CONSISTENCY_MASK_PADDING needs the paper "
                               "batch's real_mask; this batch has none")
            real_mask = torch.tensor(np.asarray(batch["real_mask"]),
                                     dtype=torch.float32, device=self.device)

        # Linear lr warmup (LR_WARMUP_STEPS; a no-op when 0 or past it).
        apply_lr_warmup((self.rep_optimizer, self.pred_optimizer,
                         self.dyn_optimizer, self.prefix_optimizer,
                         self.proj_optimizer),
                        getattr(self, "training_steps", 0))

        # Zero out the gradients for all optimizers.
        self.rep_optimizer.zero_grad()
        self.pred_optimizer.zero_grad()
        self.dyn_optimizer.zero_grad()
        self.prefix_optimizer.zero_grad()
        self.proj_optimizer.zero_grad()

        total_loss = 0.0

        # ---------------------
        # Step 0: Initial Inference
        # ---------------------
        init_obs = obs[:, 0]
        latent = self.representation(init_obs)
        policy_logits, decoded_v0, value_logits = self.prediction(latent)
        
        converted_target_rewards = scalar_categorical.scalar_to_categorical_batch(
            rewards, self.dynamics.support_size, self.dynamics.num_bins, self.device,
            transform=USE_H_TRANSFORM)

        # Value prefix targets: the cumulative discounted reward sum along the
        # unroll, which is what the LSTM head predicts in one shot instead of
        # per-step rewards that are summed afterwards.
        # UNDISCOUNTED (discount=1.0), matching what the tree consumes: an edge
        # reward there is the difference between a child's prefix and its
        # parent's, and that identity only holds without discounting baked in.
        # backup_simulation applies the discount, as it does for real rewards.
        target_prefix = discounted_prefix_targets(rewards, discount=1.0)
        converted_target_prefix = scalar_categorical.scalar_to_categorical_batch(
            target_prefix, self.value_prefix.support_size, self.value_prefix.num_bins, self.device,
            transform=USE_H_TRANSFORM)
        # The hidden state IS the accumulated reward, so it starts empty at the
        # beginning of every unroll. Same reason it must reset at a search root.
        prefix_state = self.value_prefix.initial_state(obs.size(0), self.device)

        # Compute losses for the initial step (no reward loss here).
        policy_loss_0 = masked_policy_loss(policy_logits, target_policy[:, 0, :],
                                           policy_mask[:, 0], weights=is_weights)
        converted_target_values = scalar_categorical.scalar_to_categorical_batch(
            target_value, self.prediction.support_size, self.prediction.num_bins, self.device,
            transform=USE_H_TRANSFORM)
        value_ce_0 = -torch.sum(converted_target_values[:, 0, :] * F.log_softmax(value_logits, dim=1), dim=1)
        value_loss_0 = wmean(value_ce_0)
        step0_loss = (POLICY_LOSS_WEIGHT * policy_loss_0
                      + VALUE_LOSS_WEIGHT * value_loss_0)
        if EFFZERO_LOSS_SCALING:
            # EfficientZero scales the whole loss, step 0 included, by 1/K.
            step0_loss = scale_gradient(step0_loss, 1.0 / unroll_steps)
        total_loss += step0_loss

        # ---------------------
        # Unroll for each subsequent step
        # ---------------------
        for i in range(1, unroll_steps + 1):
            # Use the action from the previous step to roll forward the latent state.
            latent, _, reward_logits = self.dynamics(latent, actions[:, i - 1])
            if EFFZERO_LOSS_SCALING:
                # EfficientZero's hidden_state hook: halves ALL gradient into
                # this latent, this step's heads included (the per-step reward
                # head reads the pre-return latent inside dynamics and is not
                # halved; REWARD_LOSS_WEIGHT is 0 in the paper arms).
                latent = scale_gradient(latent, LATENT_GRADIENT_SCALE)
            # SimSiam consistency (EfficientZero 4.1), replacing plain latent
            # MSE. Online branch: the dynamics-predicted latent through
            # projection AND prediction. Target branch: the representation of
            # the real observation through projection only, gradients stopped.
            # Cosine rather than MSE, so latents cannot drift small to make
            # the matching easier.
            with torch.no_grad():
                true_latent = self.representation(obs[:, i])
                target_proj = self.projector(true_latent, with_prediction=False)
            online_proj = self.projector(latent, with_prediction=True)
            cons_weights = is_weights
            if real_mask is not None:
                cons_weights = (real_mask[:, i] if is_weights is None
                                else is_weights * real_mask[:, i])
            latent_loss = consistency_loss(online_proj, target_proj,
                                           weights=cons_weights)
            
            # Loss for reward prediction. (reward[0] not used since it is reward for entering obs)
            reward_loss = wmean(-torch.sum(converted_target_rewards[:, i, :] * F.log_softmax(reward_logits, dim=1), dim=1))

            # Value prefix over the same unroll. Both heads are trained on the
            # same trunk for now so they can be compared directly on identical
            # representations; the per-step head goes away once the tree search
            # is migrated to consume prefixes.
            _, prefix_logits, prefix_state = self.value_prefix(latent, prefix_state)
            prefix_loss = wmean(-torch.sum(converted_target_prefix[:, i, :] * F.log_softmax(prefix_logits, dim=1), dim=1))

            policy_logits, _, value_logits = self.prediction(latent)

            # Loss for policy prediction, over the positions that have one.
            # Only the POLICY is masked here, and deliberately: a padded position
            # carries a uniform placeholder policy, which is fabricated, but its
            # target_value of 0.0 and reward of 0.0 are TRUE -- padding repeats
            # the terminal observation, and a terminal state really is worth
            # nothing. Masking those would discard correct signal, not noise.
            policy_loss = masked_policy_loss(policy_logits, target_policy[:, i, :],
                                             policy_mask[:, i], weights=is_weights)
            value_ce = -torch.sum(converted_target_values[:, i, :] * F.log_softmax(value_logits, dim=1), dim=1)
            value_loss = wmean(value_ce)

            step_loss = (REWARD_LOSS_WEIGHT * reward_loss
                         + PREFIX_LOSS_WEIGHT * prefix_loss
                         + POLICY_LOSS_WEIGHT * policy_loss
                         + VALUE_LOSS_WEIGHT * value_loss
                         + CONSISTENCY_LOSS_WEIGHT * latent_loss)

            # Each unroll step contributes 1/K of the gradient, so the total does
            # not scale with how far the unroll runs.
            total_loss += scale_gradient(step_loss, 1.0 / unroll_steps)

            # Halve the gradient on the RECURRENT path only, AFTER this step's
            # heads have consumed the latent at full strength -- MuZero appendix
            # G's placement. Scaling before the dynamics call (as this used to)
            # also scaled the representation's output on step one, so every
            # unrolled loss -- including the whole SimSiam consistency signal
            # -- reached the representation at half the specified strength:
            # 0.5^k instead of 0.5^(k-1). (EFFZERO_LOSS_SCALING reinstates
            # 0.5^k deliberately: it is what EfficientZero's code does.)
            if not EFFZERO_LOSS_SCALING:
                latent = scale_gradient(latent, LATENT_GRADIENT_SCALE)

        # Backpropagate the total loss.
        total_loss.backward()

        # One global norm across every module, EfficientZero's max_grad_norm=5.
        grad_norm = torch.nn.utils.clip_grad_norm_(
            itertools.chain(self.representation.parameters(),
                            self.dynamics.parameters(),
                            self.prediction.parameters(),
                            self.value_prefix.parameters(),
                            self.projector.parameters()),
            MAX_GRAD_NORM)
        if SKIP_NONFINITE_GRAD_STEPS and not torch.isfinite(grad_norm):
            # A NaN/Inf norm has just scaled every gradient by NaN; stepping
            # would poison the weights and SGD momentum for good (and the
            # resume checkpoint with them). Skip the update and the priority
            # writeback, as EfficientZero's AMP GradScaler skips such steps.
            # BN running statistics updated by this forward are not undone.
            self.nonfinite_skips = getattr(self, "nonfinite_skips", 0) + 1
            print(f"  [nonfinite] grad norm {float(grad_norm)} at training "
                  f"step {getattr(self, 'training_steps', '?')}: update skipped "
                  f"({self.nonfinite_skips} so far)", flush=True)
            for opt in (self.rep_optimizer, self.dyn_optimizer,
                        self.prefix_optimizer, self.pred_optimizer,
                        self.proj_optimizer):
                opt.zero_grad()
            self._last_step = (total_loss.item(), grad_norm)
            return self._last_step[0]

        # Step the optimizers.
        self.rep_optimizer.step()
        self.dyn_optimizer.step()
        self.prefix_optimizer.step()
        self.pred_optimizer.step()
        self.proj_optimizer.step()

        # PER priority writeback: new priority = |v(s_0) - z_0| on the fresh
        # forward, as in MuZero appendix G. sample_refs are the live
        # transition dicts, so the buffer updates in place.
        if batch.get("sample_refs"):
            with torch.no_grad():
                errs = (decoded_v0.detach().squeeze(-1)
                        - target_value[:, 0]).abs().cpu().numpy()
            for ref, e in zip(batch["sample_refs"], errs):
                ref["priority"] = float(e) + PER_EPS

        # (loss, pre-clip grad norm TENSOR) for harness diagnostics; the norm
        # stays on-device so arms that never read it pay no sync.
        self._last_step = (total_loss.item(), grad_norm)
        return self._last_step[0]

    async def training_process(self):
        try:
            while True:
                await self.storage_manager.update_if_needed()
                # Buffer may have changed: drop the bootstrap cache (id() reuse).
                self.invalidate_bootstrap_cache()
                if len(self.storage_manager.replay_buffer) < MAX_REPLAY // 2:
                    self.logger.debug("Training sleeping")
                    await asyncio.sleep(1)
                    continue
                batch = await self.prepare_batch(self.storage_manager.replay_buffer, TRAINING_BATCHES, UNROLL_STEPS, self.action_space)
                loss = await self.train_step(batch)
                self.training_steps += 1
                self.logger.info(f"Step: {self.training_steps} Training loss: {loss}")
                if self.training_steps % 100 == 0 and self.training_steps > 1:
                    self.logger.info(f"Saving network at {self.training_steps} training steps")
                    await self.save_networks()
        except Exception:
            self.logger.exception("training_process failed")
        self.logger.info("Training process exiting")
    
    def _ensure_target_model(self):
        """The FULL lagged target model (EfficientZero's, interval 200): all
        four function networks, because their reanalyze both bootstraps values
        AND re-searches policies with it. Lazy, so the harness's
        object.__new__ construction and older tests need nothing.

        DOUBLE-BUFFERED as in their train.py ("the interval of the current
        model and the target model is between x and 2x"): each sync loads the
        snapshot taken at the PREVIOUS sync, so target staleness is one-to-two
        intervals (200-400 steps), not the 0-200 a direct copy would give.
        """
        if getattr(self, "target_representation", None) is None:
            # EVERY geometry field mirrored from the LIVE nets -- never the
            # module constants -- so the strict load_state_dict below can
            # never meet a shape mismatch, even when constants changed
            # between net construction and the first sync (audit
            # 2026-08-28: width and trunk depth were still
            # constant-sourced here while the comment claimed otherwise).
            latent = self.representation.conv4.out_channels
            self.target_representation = RepresentationNetwork(
                input_channels=12, latent_channels=latent,
                trunk_blocks=len(self.representation.trunk),
                renormalize=self.representation.renormalize_output).to(self.device)
            self.target_prediction = PredictionNetwork(
                latent_channels=latent, action_space=self.action_space,
                support_size=self.prediction.support_size,
                num_bins=self.prediction.num_bins,
                transform=self.prediction.transform,
                head_channels=self.prediction.head_channels,
                fc_hidden=self.prediction.fc_hidden).to(self.device)
            self.target_dynamics = DynamicsNetwork(
                latent_channels=latent,
                action_space=self.action_space,
                support_size=self.dynamics.support_size,
                num_bins=self.dynamics.num_bins,
                transform=self.dynamics.transform,
                blocks=len(self.dynamics.blocks),
                action_encoding=self.dynamics.action_encoding,
                reward_channels=self.dynamics.reward_channels,
                renormalize=self.dynamics.renormalize_output).to(self.device)
            self.target_value_prefix = ValuePrefixNetwork(
                latent_channels=self.value_prefix.conv.in_channels,
                hidden_size=self.value_prefix.hidden_size,
                support_size=self.value_prefix.support_size,
                num_bins=self.value_prefix.num_bins,
                transform=self.value_prefix.transform,
                head_channels=self.value_prefix.head_channels,
                fc_hidden=self.value_prefix.fc_hidden).to(self.device)
            # Structural, not conventional: target nets never see an
            # optimizer, and every forward is no-grad anyway.
            for net in (self.target_representation, self.target_prediction,
                        self.target_dynamics, self.target_value_prefix):
                net.eval()
                for p in net.parameters():
                    p.requires_grad_(False)
            self._target_synced_at = None
            self._target_snapshot = None
        if (self._target_synced_at is None
                or self.training_steps - self._target_synced_at >= TARGET_MODEL_INTERVAL):
            online = ((self.representation, self.target_representation),
                      (self.prediction, self.target_prediction),
                      (self.dynamics, self.target_dynamics),
                      (self.value_prefix, self.target_value_prefix))
            if self._target_snapshot is None:
                self._target_snapshot = [
                    {k: v.detach().clone() for k, v in net.state_dict().items()}
                    for net, _ in online]
            for (net, target), snap in zip(online, self._target_snapshot):
                target.load_state_dict(snap)
                target.eval()
            self._target_snapshot = [
                {k: v.detach().clone() for k, v in net.state_dict().items()}
                for net, _ in online]
            self._target_synced_at = self.training_steps

    def assemble_batch_paper(self, replay_buffer, batch_size, unroll_steps,
                             action_space, training_steps=None):
        """The papers' batch: PER-sampled window roots with importance
        weights, and value targets computed NOW from stored rewards + stored
        root values under the age-adjusted horizon (EfficientZero 4.3).
        Policy targets are the recorded acting-search policies -- every real
        position carries one."""
        # Pure CPU half (safe on a prefetch worker): no CUDA, no target-net
        # access, no replay mutation. `training_steps` is explicit so a
        # prefetched batch i uses base + i like the serial loop would.
        ts = self.training_steps if training_steps is None else training_steps
        trajectories = list(replay_buffer)
        roots, priorities = [], []
        for traj in trajectories:
            # The last record is excluded as a window ROOT (it still appears
            # inside earlier windows and as the value bootstrap): for a
            # terminated trajectory it carried no search; for a short
            # truncated one it IS searched, but its stored action is a
            # placeholder, so rooting a window there would unroll a fake
            # action at full policy weight.
            limit = min(len(traj), max(1, len(traj) - 1))
            if traj[-1].get("truncated") and len(traj) > unroll_steps:
                limit = len(traj) - unroll_steps
            for idx in range(limit):
                roots.append((traj, idx))
                priorities.append(traj[idx].get("priority", PER_EPS))
        priorities = np.asarray(priorities, dtype=np.float64)
        if PER_ENABLED:
            probs = priorities ** PER_ALPHA
            probs = probs / probs.sum()
        else:
            probs = np.full(len(roots), 1.0 / len(roots))
        # Without replacement when the population allows, as EfficientZero's
        # replay buffer samples.
        replace = batch_size > len(roots)
        chosen = np.random.choice(len(roots), size=batch_size, p=probs,
                                  replace=replace)
        # w = (N * P)^-beta with the annealed beta, normalized by the max.
        weights = (len(roots) * probs[chosen]) ** (-per_beta(ts))
        weights = weights / weights.max()

        use_tm = USE_TARGET_MODEL
        # SVE (VALUE_TARGET_MODE "mixed"): a root is STALE when its global
        # buffer index lies before the newest SVE_FRESH_TRANSITIONS
        # transitions. The TD target is still assembled for every position
        # (so the row diagnostics can measure |SVE - TD| and today's batch
        # is untouched up to that point); finish_batch_paper then swaps
        # the stale real positions' values for their stored root values.
        mixed = VALUE_TARGET_MODE == "mixed"
        sve_on = mixed and ts >= SVE_START_TRAIN_STEPS
        buf_frac = 0.0
        if sve_on:
            # Counted in TRANSITIONS (actions), as EZ-V2's
            # GameTrajectory.__len__: a trajectory's last record (terminal
            # or truncated tail) is not one. A root at global transition
            # index g is stale iff g <= N - T2 -- EZ-V2's
            # fresh = idx > collected - mixed_value_threshold
            # (ez/worker/batch_worker.py:704), so the fresh window holds
            # T2 - 1 transitions (SVE audit r1, fidelity L2).
            offsets, acc = {}, 0
            for traj in trajectories:
                offsets[id(traj)] = acc
                acc += max(0, len(traj) - 1)
            sve_cutoff = acc - SVE_FRESH_TRANSITIONS
            # The buffer's stale share (uniform over transitions): the
            # reference for sve_frac, which PER shifts either way.
            buf_frac = min(acc, max(0, sve_cutoff + 1)) / max(1, acc)
        # (row, step, traj, pos, stored value, stored value's age)
        sve = []
        n_real = 0
        # (traj_id, tail) -> [tail_obs, [(row, step, g), ...]] for the
        # batch-time bootstrap refresh; deduped since windows share tails.
        pending = {}
        batch_obs, batch_actions, batch_rewards = [], [], []
        batch_policy, batch_value, batch_has, refs = [], [], [], []
        batch_real = []
        for flat in chosen:
            traj, start = roots[flat]
            refs.append(traj[start])
            age = max(0, ts - traj[start].get("born_step", ts))
            # EZ-V2 drops EfficientZero's age-shrunk horizon whenever the
            # mixed target is configured, before T1 too
            # (ez/worker/batch_worker.py:645-648: delta_td = 0 for
            # value_target 'mixed'; SVE audit r1, fidelity L1).
            horizon = TD_STEPS if mixed else paper_horizon(age)
            L = len(traj)
            stale = sve_on and offsets[id(traj)] + start <= sve_cutoff
            obs, actions, rewards = [], [], []
            pols, vals, has = [], [], []
            real = []
            for step in range(unroll_steps + 1):
                pos = start + step
                # A real position (not padding past the trajectory's end):
                # EfficientZero's mask_batch, for CONSISTENCY_MASK_PADDING.
                real.append(pos < L)
                if pos < L:
                    t = traj[pos]
                    obs.append(t["obs"])
                    rewards.append(t["reward"])
                    pols.append(t["target_policy"])
                    if use_tm:
                        rsum, g, tail = paper_value_parts(traj, pos, horizon, DISCOUNT)
                        tail_t = traj[tail]
                        terminal_tail = (tail == L - 1
                                         and not tail_t.get("truncated"))
                        vals.append(rsum)
                        if not terminal_tail:
                            entry = pending.setdefault(
                                (id(traj), tail), [tail_t["obs"], []])
                            entry[1].append((len(batch_value), step, g))
                    else:
                        vals.append(paper_value_target(traj, pos, horizon, DISCOUNT))
                    has.append(t.get("has_policy_target", True))
                    n_real += 1
                    if stale:
                        vstep = t.get("root_value_step", t.get("born_step", ts))
                        sve.append((len(batch_value), step, traj, pos,
                                    float(t["root_value"]), max(0, ts - vstep)))
                    if step < unroll_steps:
                        actions.append(t["action"])
                else:
                    t = traj[-1]
                    obs.append(t["obs"])
                    rewards.append(0.0)
                    pols.append(np.ones(action_space) / action_space)
                    # Absorbing padding: 0 is true past a terminal; truncated
                    # windows only reach here on short trajectories, where the
                    # tail's value stands in -- refreshed through the target
                    # model like every other bootstrap, not the aging stored
                    # one.
                    if t.get("truncated") and use_tm:
                        vals.append(0.0)
                        entry = pending.setdefault(
                            (id(traj), L - 1), [t["obs"], []])
                        entry[1].append((len(batch_value), step, 1.0))
                    else:
                        vals.append(t["root_value"] if t.get("truncated") else 0.0)
                    has.append(False)
                    if step < unroll_steps:
                        actions.append(0)
            batch_obs.append(obs)
            batch_actions.append(actions)
            batch_rewards.append(rewards)
            batch_policy.append(pols)
            batch_value.append(vals)
            batch_has.append(has)
            batch_real.append(real)

        return {"pending": pending, "weights": weights, "refs": refs,
                "sve": (sve, n_real, buf_frac) if mixed else None,
                "use_tm": use_tm, "obs": batch_obs, "actions": batch_actions,
                "rewards": batch_rewards, "policy": batch_policy,
                "value": batch_value, "has": batch_has, "real": batch_real}

    def finish_batch_paper(self, a):
        """Main-thread half: the target-model bootstrap forward (CUDA,
        target nets) and the final arrays. Runs at the exact serial point
        -- after the previous train_step's target sync, before this one."""
        pending, weights, refs = a["pending"], a["weights"], a["refs"]
        batch_obs, batch_actions, batch_rewards = a["obs"], a["actions"], a["rewards"]
        batch_policy, batch_value, batch_has = a["policy"], a["value"], a["has"]
        if a["use_tm"]:
            self._ensure_target_model()
        if pending:
            # One (chunked) forward through the LAGGED target model supplies
            # every bootstrap value; z = reward_sum + g * v_target(tail).
            # BOOTSTRAP CACHE (2026-09-11): the target nets are frozen
            # between syncs (TARGET_MODEL_INTERVAL steps), so v_target of a
            # stored tail is a constant over that window -- and the same
            # tails recur every step (256 lookups/step over a ~30k-position
            # buffer). Only MISSES go through the forward. Keyed on
            # (id(traj), tail); the cache is dropped whenever the target
            # sync stamp changes and at every training-block start
            # (invalidate_bootstrap_cache), and trajectories are never
            # allocated/evicted inside a block, so an id() can never be
            # reused while its entry is live. Numerics: a value computed in
            # a different chunk composition differs at ~1e-6 in fp32 and
            # up to ~1e-3 relative under PyTorch's TF32 conv defaults --
            # the same class of wobble the uncached chunked forward
            # already had from batch to batch.
            stamp = self._target_synced_at
            if not BOOTSTRAP_CACHE:
                # Per-call scratch: every key misses, nothing is retained
                # across steps -- the uncached builder's exact behaviour.
                cache = {}
                self._boot_cache = None
            else:
                cache = getattr(self, "_boot_cache", None)
                if (cache is None
                        or getattr(self, "_boot_cache_stamp", None) != stamp):
                    cache = self._boot_cache = {}
                    self._boot_cache_stamp = stamp
            keys = list(pending)
            miss = [k for k in keys if k not in cache]
            self.boot_cache_hits = getattr(self, "boot_cache_hits", 0) + (len(keys) - len(miss))
            self.boot_cache_misses = getattr(self, "boot_cache_misses", 0) + len(miss)
            if miss:
                obs_arr = np.stack([pending[k][0] for k in miss])
                with torch.no_grad():
                    for lo in range(0, len(miss), 256):
                        chunk = torch.tensor(obs_arr[lo:lo + 256],
                                             dtype=torch.float32,
                                             device=self.device) / 255.0
                        _, v, _ = self.target_prediction(
                            self.target_representation(chunk))
                        vals = v.squeeze(-1).cpu().numpy()
                        for k, val in zip(miss[lo:lo + 256], vals):
                            cache[k] = float(val)
            for k in keys:
                val = cache[k]
                for row, step, g in pending[k][1]:
                    batch_value[row][step] += g * val

        if a.get("sve") is not None:
            # SVE swap (VALUE_TARGET_MODE "mixed"), after every TD value is
            # complete, so the diagnostics compare against the exact target
            # it replaces. Accumulated per row window (Run._merge_row_stats).
            sve, n_real, buf_frac = a["sve"]
            w = getattr(self, "_sve_window", None) or {
                "n_real": 0, "n_sve": 0, "gap": 0.0, "bias": 0.0, "age": 0.0,
                "batches": 0, "buf": 0.0, "searches": 0, "search_s": 0.0}
            w["n_real"] += n_real
            w["batches"] += 1
            w["buf"] += buf_frac
            searched = (self._sve_search_values(sve, w)
                        if SVE_SOURCE == "search" and sve else None)
            for i, (row, step, _traj, _pos, sv, vage) in enumerate(sve):
                if searched is not None:
                    sv, vage = searched[i]
                d = sv - batch_value[row][step]
                w["n_sve"] += 1
                w["gap"] += abs(d)
                w["bias"] += d
                w["age"] += vage
                batch_value[row][step] = sv
            self._sve_window = w

        return {
            "obs": np.array(batch_obs),
            "actions": np.array(batch_actions),
            "rewards": np.array(batch_rewards),
            "target_policy": np.array(batch_policy),
            "target_value": np.array(batch_value),
            "has_policy_target": np.array(batch_has),
            "real_mask": np.array(a["real"]),
            "is_weights": weights.astype(np.float32),
            "sample_refs": refs,
        }

    def prepare_batch_paper(self, replay_buffer, batch_size, unroll_steps, action_space):
        return self.finish_batch_paper(self.assemble_batch_paper(
            replay_buffer, batch_size, unroll_steps, action_space))

    def invalidate_bootstrap_cache(self):
        """Drop cached target-model bootstrap values -- and the SVE search
        cache, keyed the same way. Called at every training-block start
        (the only points where the replay buffer may have changed) -- see
        finish_batch_paper."""
        self._boot_cache = None
        self._boot_cache_stamp = None
        self._sve_cache = None
        self._sve_cache_stamp = None

    def _sve_search_values(self, sve, w):
        """Stage-2 SVE values (SVE_SOURCE "search"), one (value, age) per
        entry of `sve`: a fresh-root PUCT search of each stale position
        through the lagged TARGET nets at SVE_SEARCH_SIMS, no root noise
        (a deliberate deviation: EZ-V2's reanalysis search is noisy),
        deterministic tie-breaks -- the global RNG is never drawn, so the
        training streams cannot depend on the cache's state (a resume
        starts it empty). The value is EZ-V2's root mean over N+1
        estimates, (v(root) + n * q) / (n + 1), n the root's visits
        (ez/mcts/ctree/cnode.cpp prepare/get_value); vtree's q averages
        the n simulation backups only. Cached per (id(traj), pos) until the target sync
        stamp changes, and dropped at every block start like the bootstrap
        cache (trajectories are never evicted inside a block, so an id()
        cannot be reused while its entry lives). The terminal record of a
        TERMINATED trajectory is absorbing: 0, never searched; a truncated
        tail is searched (deliberate: EZ-V2 zeroes every chunk end). The age is
        training steps since the sync (the double-buffered weights are one
        interval older still). Main thread only (finish_batch_paper)."""
        import muzero_vtree
        self._ensure_target_model()
        stamp = self._target_synced_at
        cache = getattr(self, "_sve_cache", None)
        if cache is None or getattr(self, "_sve_cache_stamp", None) != stamp:
            cache = self._sve_cache = {}
            self._sve_cache_stamp = stamp
        keys, miss = [], {}
        for _row, _step, traj, pos, _stored, _age in sve:
            terminal = pos == len(traj) - 1 and not traj[-1].get("truncated")
            k = None if terminal else (id(traj), pos)
            keys.append(k)
            if k is not None and k not in cache and k not in miss:
                miss[k] = traj[pos]["obs"]
        if miss:
            t0 = time.perf_counter()
            engine = muzero_vtree.VectorizedPUCT(
                {"representation": self.target_representation,
                 "dynamics": self.target_dynamics,
                 "prediction": self.target_prediction,
                 "value_prefix": self.target_value_prefix},
                self.action_space, self.device)
            todo = list(miss)
            chunk = max(1, int(VTREE_REFRESH_BATCH))
            for lo in range(0, len(todo), chunk):
                part = todo[lo:lo + chunk]
                obs = np.stack([miss[k] for k in part])
                _, q, counts = engine.search(
                    obs, num_simulations=SVE_SEARCH_SIMS, c_puct=C_PUCT,
                    discount=DISCOUNT, root_noise=False, tie_break="first")
                _, _, v0 = engine._initial(obs)
                n = np.asarray(counts, dtype=np.float64).sum(axis=1)
                vals = (v0.cpu().numpy() + n * np.asarray(q)) / (n + 1.0)
                for k, v in zip(part, vals):
                    cache[k] = float(v)
            w["searches"] += len(todo)
            w["search_s"] += time.perf_counter() - t0
        age = max(0, self.training_steps - (stamp if stamp is not None
                                            else self.training_steps))
        return [(0.0 if k is None else cache[k], age) for k in keys]

    def assemble_batch(self, replay_buffer, batch_size, unroll_steps,
                       action_space, training_steps=None):
        """Prefetch-worker half of prepare_batch for either target mode.
        The plain (non-paper) builder is pure CPU already."""
        if PAPER_TARGETS:
            return ("paper", self.assemble_batch_paper(
                replay_buffer, batch_size, unroll_steps, action_space,
                training_steps=training_steps))
        return ("plain", asyncio.run(self.prepare_batch(
            replay_buffer, batch_size, unroll_steps, action_space)))

    def finish_batch(self, item):
        kind, payload = item
        return self.finish_batch_paper(payload) if kind == "paper" else payload

    async def prepare_batch(self, replay_buffer, batch_size, unroll_steps, action_space):
        """
        Prepares a training batch for multi-step unrolling.
        
        Args:
        replay_buffer: list of trajectories. Each trajectory is a list of transitions.
                        Each transition is a tuple: 
                        (obs, action, reward, next_obs, target_policy, target_value)
        batch_size: number of samples to include in the batch.
        unroll_steps: number of unroll steps to sample.
        action_space: number of possible actions.
        
        Returns:
        A dictionary with keys:
            "obs": (B, unroll_steps, C, H, W) initial observations,
            "actions": (B, unroll_steps) sequence of actions,
            "rewards": (B, unroll_steps+1) sequence of rewards,
            "target_policy": (B, unroll_steps+1, action_space) target policies,
            "target_value": (B, unroll_steps+1) target values,
            "has_policy_target": (B, unroll_steps+1) bool, whether that position
                carries a real improved policy. False for timesteps reanalysis
                skipped and for padding past the end of a trajectory; the
                policy loss is masked there.
        """
        if PAPER_TARGETS:
            return self.prepare_batch_paper(replay_buffer, batch_size,
                                            unroll_steps, action_space)
        batch_obs = []
        batch_actions = []
        batch_rewards = []
        batch_target_policy = []
        batch_target_value = []
        batch_has_policy = []

        # Sample uniformly over TRANSITIONS, not trajectories: weighting each
        # trajectory by its length. Picking a trajectory uniformly first
        # oversamples the transitions of short episodes in proportion to how
        # short they are -- a 100-step death was sampled 6x as often per
        # transition as a 600-step run.
        trajectories = list(replay_buffer)

        def legal_roots(t):
            # A truncated trajectory long enough for window avoidance offers
            # only len - unroll legal window roots; weighting it by its full
            # length would oversample every legal root by len/(len - unroll).
            if t and t[-1].get("truncated") and len(t) > unroll_steps:
                return len(t) - unroll_steps
            return len(t)

        lengths = np.array([legal_roots(t) for t in trajectories], dtype=np.float64)
        traj_probs = lengths / lengths.sum()
        for _ in range(batch_size):
            traj = trajectories[np.random.choice(len(trajectories), p=traj_probs)]
            traj_length = len(traj)
            truncated = bool(traj[-1].get("truncated"))
            if truncated and traj_length > unroll_steps:
                # Keep unroll windows off a TRUNCATED tail entirely: padding
                # there fabricates rewards and teaches the dynamics a frozen
                # mid-game frame the game never produced. (Terminated tails
                # keep full-range starts -- their absorbing-state padding is
                # true.) Tail positions still train as interior steps of
                # earlier windows.
                start_index = np.random.randint(0, traj_length - unroll_steps)
            else:
                start_index = np.random.randint(0, traj_length)
            # Fallback for truncated trajectories too short to avoid padding:
            # pad with the tail's frozen value target rather than a fabricated
            # terminal V=0.
            pad_value = traj[-1]["target_value"] if truncated else 0.0
            
            # Prepare lists for unroll data.
            obs = []
            actions = []
            rewards = []
            target_policy = []
            target_value = []
            has_policy = []
            
            # We need unroll_steps+1 target entries (for step 0 and each unroll step).
            for step in range(unroll_steps + 1):
                if start_index + step < traj_length:
                    obs.append(traj[start_index + step]["obs"])
                    node = traj[start_index + step]
                    # Append the target policy and value for each step.
                    target_policy.append(node["target_policy"])
                    target_value.append(node["target_value"])
                    # Trajectories written before reanalysis subsampling have no
                    # such key; every position in them was searched.
                    has_policy.append(node.get("has_policy_target", True))
                    # reward for entering node (reward[0] will not be used)
                    rewards.append(node["reward"])
                    if step < unroll_steps:
                        # For actions, only collect for unroll_steps.
                        actions.append(node["action"])
                else:
                    obs.append(traj[traj_length - 1]["obs"])
                    # If the trajectory doesn't have enough transitions, pad.
                    target_policy.append(np.ones(action_space) / action_space)
                    target_value.append(pad_value)
                    rewards.append(0.0)
                    has_policy.append(False)   # padding is not a real target
                    if step < unroll_steps:
                        actions.append(0)         # or a default action
            
            batch_obs.append(obs)
            batch_actions.append(actions)
            batch_rewards.append(rewards)
            batch_target_policy.append(target_policy)
            batch_target_value.append(target_value)
            batch_has_policy.append(has_policy)
        
        # Convert lists to numpy arrays.
        batch_obs = np.array(batch_obs)  # (B, H, W, C)
        batch_actions = np.array(batch_actions)  # (B, unroll_steps)
        batch_rewards = np.array(batch_rewards)  # (B, unroll_steps)
        batch_target_policy = np.array(batch_target_policy)  # (B, unroll_steps+1, action_space)
        batch_target_value = np.array(batch_target_value)  # (B, unroll_steps+1)
        batch_has_policy = np.array(batch_has_policy)  # (B, unroll_steps+1)
        
        return {
            "obs": batch_obs,
            "actions": batch_actions,
            "rewards": batch_rewards,
            "target_policy": batch_target_policy,
            "target_value": batch_target_value,
            "has_policy_target": batch_has_policy
        }

class BatchPrefetcher:
    """Overlap CPU batch assembly with GPU training for a block of exactly
    n_steps gradient steps (see TRAIN_PREFETCH). Contract: the caller
    must not mutate `replay` (append/evict/refresh) while the prefetcher
    is open -- open it around a contiguous training block and close() it
    before collect/refresh run; close() always joins the worker, even if
    the consumer stopped early (an exception in train_step). A worker
    exception surfaces on the next next()."""

    def __init__(self, trainer, replay, batch_size, unroll_steps,
                 action_space, n_steps, depth=None):
        import queue
        import threading
        self.trainer, self.replay = trainer, replay
        self.args = (batch_size, unroll_steps, action_space)
        self.n_steps = int(n_steps)
        self.base = trainer.training_steps
        self.q = queue.Queue(maxsize=max(1, int(depth or TRAIN_PREFETCH_DEPTH)))
        self.err = None
        self._stop = threading.Event()
        self.produced = self.consumed = 0
        self.thread = threading.Thread(target=self._work, name="batch-prefetch",
                                       daemon=True)
        self.thread.start()

    def _work(self):
        try:
            for i in range(self.n_steps):
                if self._stop.is_set():
                    return
                item = self.trainer.assemble_batch(
                    self.replay, *self.args, training_steps=self.base + i)
                while True:  # bounded put that still honours a stop request
                    try:
                        self.q.put(item, timeout=0.1)
                        break
                    except Exception:
                        if self._stop.is_set():
                            return
                self.produced += 1
        except BaseException as e:  # surfaced by next(); never lost
            self.err = e
            while not self._stop.is_set():  # keep offering the sentinel
                try:
                    self.q.put(None, timeout=0.1)
                    break
                except Exception:
                    continue

    def next(self):
        if self.consumed >= self.n_steps:
            raise RuntimeError("BatchPrefetcher exhausted")
        import queue as _queue
        while True:
            try:
                item = self.q.get(timeout=1.0)
                break
            except _queue.Empty:
                # Audit F1: a worker that died without a sentinel landing
                # (queue full at the moment of failure) must not leave the
                # consumer blocked forever (un-interruptible on Windows,
                # invisible to the supervisor) -- surface it.
                if not self.thread.is_alive() and self.q.empty():
                    item = None
                    break
        if item is None:
            self._err_raised = True
            raise RuntimeError("batch prefetch worker failed") from self.err
        self.consumed += 1
        return self.trainer.finish_batch(item)

    def close(self):
        self._stop.set()
        while True:  # unblock a worker parked on a full queue
            try:
                self.q.get_nowait()
            except Exception:
                break
        self.thread.join()
        # A worker failure that next() already surfaced is not raised
        # twice; one that nobody saw (block abandoned early) is.
        if (self.err is not None and not getattr(self, "_err_raised", False)
                and self.consumed < self.n_steps):
            raise RuntimeError("batch prefetch worker failed") from self.err

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


#Agent will be single threaded for real time learning
class MuZeroAgent:
    def __init__(self, action_space, device, network_ref, storage_ref, enable_logging=False):
        self.network_manager = NetworkManager(network_ref, action_space=action_space, device=device, load_dynamics=False)
        # The agent only ever writes to the gameplay buffer, so it has no reason
        # to mirror the replay buffer.
        self.storage_manager = StorageManager(storage_ref, update_replay=False)
        self.device = device
        self.action_space = action_space
        self.training_steps = 0
        self.logger = configure_actor_logger("MuZeroAgent", enable_logging)
        self.logger.info("Initialized actor logger.")

    async def setup(self):
        await self.network_manager.setup()
        await self.storage_manager.setup()

    async def gameplay_process(self):
        try:
            if not hasattr(np, 'bool8'):
                np.bool8 = np.bool_
            env = gym.make("MsPacman-v4")
            env = EpisodicLifeEnv(env)
            period = real_time_step_period(env) if SELF_PLAY_REAL_TIME else 0.0
            self.logger.info("Self-play paced at %.1f decisions/s (%s)",
                             (1.0 / period) if period else float("inf"),
                             "real time" if period else "unlimited")
            while True:
                # Refresh weights between episodes. Without this the behaviour
                # policy stayed frozen at whatever setup() loaded, so self-play
                # kept generating data from the initial network forever and the
                # policy improvement loop never closed. (Refreshing per episode
                # rather than per step keeps acting real-time.)
                await self.network_manager.update_if_needed()
                self.logger.debug("Starting new game")
                limiter = RateLimiter(period)
                obs, info = await asyncio.to_thread(env.reset)
                preprocessed_obs = preprocess_frame(obs, resize_shape=IMAGE_RESHAPE)
                prev_reward = 0.0
                done = False
                episode_reward = 0
                game_trajectory = []  # store tuples of (obs, latent, action)
                frame_stack = initial_frame_stack(preprocessed_obs)
                
                # Play one episode.
                while not done:
                    preprocessed_obs = preprocess_frame(obs, resize_shape=IMAGE_RESHAPE)
                    frame_stack = push_frame(frame_stack, preprocessed_obs)
                    with torch.no_grad():
                        policy, value = self.network_manager.initial_inference(frame_stack)
                    if ACTING_TEMPERATURE != 1.0:
                        policy = policy ** (1.0 / ACTING_TEMPERATURE)
                        policy = policy / policy.sum()
                    action = np.random.choice(np.arange(self.action_space), p=policy)
                    obs, reward, terminated, truncated, _ = await asyncio.to_thread(env.step, action)
                    done = terminated or truncated
                    await limiter.wait()
                    # obs, chosen action, reward for entering obs
                    reward = clip_reward(reward)
                    game_trajectory.append((frame_stack, action, prev_reward))
                    episode_reward += reward
                    prev_reward = reward

                # Record the terminal state. The loop above assigns the final
                # step's reward to prev_reward and then exits, so without this
                # the last reward of every episode -- often the informative one
                # -- was dropped, and create_tree_from_game ended up marking a
                # non-terminal node as done. The action is a placeholder; no
                # action is ever taken from a terminal state.
                preprocessed_obs = preprocess_frame(obs, resize_shape=IMAGE_RESHAPE)
                frame_stack = push_frame(frame_stack, preprocessed_obs)
                game_trajectory.append((frame_stack, 0, prev_reward))

                # Carry the terminal/truncated distinction to reanalysis: ALE's
                # TimeLimit truncation must not fabricate a terminal V=0.
                await self.storage_manager.append_gameplay_buffer(
                    (game_trajectory, episode_reward, terminated))
        except Exception:
            self.logger.exception("gameplay_process failed")
        self.logger.info("Gameplay process exiting")


def initial_frame_stack(frame, num_frames=4):
    """Build the starting (num_frames*C, H, W) stack by repeating one frame.

    np.tile, not np.repeat: repeat interleaves the channel planes into
    [R,R,R,R,G,G,G,G,B,B,B,B], whereas push_frame below treats the stack as
    consecutive whole frames. The mismatch corrupted the first observation of
    every episode and stayed wrong for num_frames steps as it shifted through.
    """
    return np.tile(frame, (num_frames, 1, 1))


def push_frame(frame_stack, frame):
    """Drop the oldest frame from the stack and append the newest."""
    channels = frame.shape[0]
    return np.concatenate([frame_stack[channels:, :, :], frame], axis=0)


def preprocess_frame(frame, resize_shape=(84, 84)):
    """
    Preprocess a single game frame.

    Args:
        frame (np.array): Input image, expected to be in RGB format.
        resize_shape (tuple): Target size as (width, height). Default is (84, 84).

    Returns:
        np.array: Processed image.
    """
    # Convert from RGB to grayscale.
    #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Resize with INTER_AREA -- the standard Atari choice. The flag had been
    # commented out, falling back to INTER_LINEAR, whose 2x2 sampling skips
    # input rows on a 210->84 reduction: 1-2px sprites (pellets, the Pong
    # ball) alias in and out of existence depending on sub-pixel phase.
    frame = cv2.resize(frame, resize_shape, interpolation=cv2.INTER_AREA)
    frame = np.transpose(frame, (2, 0, 1))
    return frame


# ---------------------------------------------------------------------------
# Ray actor wrappers
#
# The classes above are plain Python so they can be constructed directly in
# tests -- @ray.remote returns an ActorClass, which cannot be instantiated
# normally. Decorating here keeps both paths available: tests import the class,
# main uses the actor.
#
# max_concurrency governs concurrent *Ray* calls into an actor. MCTSHelper's
# was tied to MCTS_BATCHES on the assumption that it limited the simulation
# coroutines, but those are created by asyncio.gather inside MCTS_process and
# are not Ray calls at all. Worse, MCTSHelper keeps per-search state on the
# instance (current_trajectory, min_max_stats, selection_results), so two
# concurrent Ray calls into one actor would silently corrupt each other's
# searches. The design is one long-running MCTS_process call per actor, after
# a setup call that completes first; max_concurrency=1 enforces it.
# ---------------------------------------------------------------------------
StorageActor  = ray.remote(max_concurrency=8)(StorageHelper)
NetworkActor  = ray.remote(max_concurrency=8)(NetworkHelper)
MCTSActor     = ray.remote(max_concurrency=1)(MCTSHelper)
TrainingActor = ray.remote(TrainingHelper)
AgentActor    = ray.remote(max_concurrency=1)(MuZeroAgent)


def resource_plan(num_mcts_actors=NUM_MCTS_ACTORS, use_cuda=True):
    """Total CPU and GPU the actor fleet will request."""
    cpu = (CPU_STORAGE + CPU_NETWORK + CPU_TRAINING + CPU_AGENT
           + CPU_MCTS * num_mcts_actors)
    gpu = 0.0
    if use_cuda:
        gpu = (GPU_NETWORK + GPU_TRAINING + GPU_AGENT
               + mcts_gpu_share(num_mcts_actors) * num_mcts_actors)
    return cpu, gpu


def check_cluster_resources(num_mcts_actors=NUM_MCTS_ACTORS, use_cuda=True):
    """Fail loudly when the fleet cannot possibly be scheduled.

    Ray does not raise when an actor's resource request exceeds what the
    cluster has -- it queues the actor indefinitely, so ray.get() on its
    setup() call simply never returns. Turning that silent hang into an error
    at startup is the difference between a one-line config fix and an evening
    spent wondering which actor deadlocked.
    """
    resources = ray.cluster_resources()
    have_cpu = resources.get("CPU", 0)
    have_gpu = resources.get("GPU", 0)
    want_cpu, want_gpu = resource_plan(num_mcts_actors, use_cuda)

    problems = []
    if want_cpu > have_cpu:
        problems.append(f"CPU: requesting {want_cpu:g}, cluster has {have_cpu:g}")
    if want_gpu > have_gpu + 1e-9:
        problems.append(f"GPU: requesting {want_gpu:g}, cluster has {have_gpu:g}")
    if problems:
        raise RuntimeError(
            "Actor resource requests exceed the Ray cluster. Ray would queue "
            "the surplus actors forever rather than raising:\n  "
            + "\n  ".join(problems)
            + f"\nLower NUM_MCTS_ACTORS (currently {num_mcts_actors}) or the "
              "per-actor budgets at the top of muzero.py."
        )
    print(f"Resource plan: {want_cpu:g}/{have_cpu:g} CPU, "
          f"{want_gpu:g}/{have_gpu:g} GPU, {num_mcts_actors} MCTS actor(s)")


# -----------------------------
# Main Loop Example with Full MCTS Sample Generation
# -----------------------------
if __name__ == '__main__':
    ray.init(include_dashboard=True, dashboard_port=8265)#log_to_driver=True)
    ACTION_SPACE = 5
    ENABLE_LOGGING = True
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    def gpu(fraction):
        # Requesting a GPU on a machine that has none queues the actor forever,
        # exactly like over-subscribing one. Ask for nothing instead.
        return fraction if use_cuda else 0

    check_cluster_resources(NUM_MCTS_ACTORS, use_cuda)

    network = NetworkActor.options(
                                num_cpus=CPU_NETWORK,
                                num_gpus=gpu(GPU_NETWORK)).remote(
                                action_space=ACTION_SPACE,
                                device=device,
                                enable_logging=ENABLE_LOGGING)
    storage = StorageActor.options(num_cpus=CPU_STORAGE).remote()
    agent = AgentActor.options(
                                num_cpus=CPU_AGENT,
                                num_gpus=gpu(GPU_AGENT)).remote(
                                action_space=ACTION_SPACE,
                                device=device,
                                network_ref=network,
                                storage_ref=storage,
                                enable_logging=ENABLE_LOGGING)
    mcts_instances = [
        MCTSActor.options(
                                num_cpus=CPU_MCTS,
                                num_gpus=gpu(GPU_MCTS)).remote(
                                action_space=ACTION_SPACE,
                                device=device,
                                network_ref=network,
                                storage_ref=storage,
                                enable_logging=ENABLE_LOGGING,
                                name=f'MCTSHelper{i}')
        for i in range(NUM_MCTS_ACTORS)
    ]
    training_instance = TrainingActor.options(
                                num_cpus=CPU_TRAINING,
                                num_gpus=gpu(GPU_TRAINING)).remote(
                                action_space=ACTION_SPACE,
                                device=device,
                                network_ref=network,
                                storage_ref=storage,
                                enable_logging=ENABLE_LOGGING)

    ray.get(agent.setup.remote())
    ray.get([m.setup.remote() for m in mcts_instances])
    ray.get(training_instance.setup.remote())

    gameplay_process = agent.gameplay_process.remote()
    mcts_processes = [m.MCTS_process.remote() for m in mcts_instances]
    training_process = training_instance.training_process.remote()

    viewenv = gym.make("MsPacman-v4", render_mode='human')
    viewenv = EpisodicLifeEnv(viewenv)
    try:
        while True:
            obs, _ = viewenv.reset()
            preprocessed_obs = preprocess_frame(obs, resize_shape=IMAGE_RESHAPE)
            done = False
            episode_reward = 0
            frame_stack = initial_frame_stack(preprocessed_obs)
            while not done:
                preprocessed_obs = preprocess_frame(obs, resize_shape=IMAGE_RESHAPE)
                frame_stack = push_frame(frame_stack, preprocessed_obs)
                with torch.no_grad():
                    policy, value = ray.get(network.initial_inference.remote(frame_stack))
                    action = np.random.choice(np.arange(ACTION_SPACE), p=policy)
                    obs, reward, terminated, truncated, _ = viewenv.step(action)
                    done = terminated or truncated
                    episode_reward += reward
            print(f"Reward: {episode_reward}")
            print_memory_diagnostics()
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Exiting gracefully...")
        ray.cancel(gameplay_process)
        for mcts_process in mcts_processes:
            ray.cancel(mcts_process)
        ray.cancel(training_process)