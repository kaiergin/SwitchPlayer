"""Test G, stage 1: the blind PILOT (CURRICULUM "Next experiment
(pre-registered) -- Test G: does the master-student value gap point at real
weakness?", Kai 2026-11-07). A go or no-go before any value net trains and
before g exists, so it cannot bias the verdict: nothing here builds a net or
reads g. Measurement only: the emulator scores, and no branch feeds a model
(the E line's rule).

PLAYERS, Test B2's lineage, seed 0, read only from Y:\\SwitchPlayer-testb2-run:
T0 at iteration 45 (the student, 36 tokens, deployed 746) and T2 at iteration 42
(the master, 18 tokens, deployed 1699). The two vocabularies need different nets,
so each player gets its own arm context in its own process, and saved states move
between them as ale_py.ALEState(bytes) with restoreState.

THE PASSES, each a fresh process (`pilot` runs them in order and stops at the
first VOID; a pass whose output exists is reused, so a stopped pilot resumes, but
every file carries a stamp of the knobs and a hash of the play path's source, and
one made by anything else is refused rather than mixed in):
  g1 T0, g1 T2: G1. The 30 plain games, against the snapshot's exact-eval row
      (probe_exhaustive_eval.play, the deployed acting states, PROBE_THREADS 2 as
      the rows ran). Each game is also played through this probe's own loop, which
      every main line and branch below uses, and must equal the row too: the
      registered check says the row still reproduces, this one says the loop is
      the deployed path, for both players.
  main: the 240 main lines. Game s: noop start 1 + s % 30, opening_moves(s) on raw
      ALE with a pooled stack carried through (Test O's read (8)), then T0's
      deployed policy to the game's end, the 3000 cap per life. Base steps count
      from the life's start, the opening included. The opening's reward is left
      out of the score, as read (8) leaves it out, and printed beside it. A life
      lost in the opening does not end the game: it plays on (registered at 100
      moves; none is lost at 80 or 100 for these 240 seeds on raw ALE). Every
      decision after the opening holds its serialized state, stack, life index and
      base steps until the game ends; the game's 6 states are then drawn at random
      with a fixed seed and kept, with each decision's token and stack digest and
      the rewards per base step, which is what later clusters and checks read.
      G6 on the score alone: 1 / sum p^2 over the distinct scores, at least 30, or
      the opening is lengthened once to 100 moves and all 240 replayed; a second
      failure is VOID.
  select: DROPS, counted, in the registration's order: FREE states, where all 9
      moves held for 32 frames give identical frames (token_census on the nine
      (a, 32) entries, at its own 92-frame window), then identical stacks, keeping
      the first. STRATA: life index x 20-base-step bins. The seeded draw of 60
      states, 3 in each of 20 strata, among strata holding at least 3 after drops;
      fewer than 20 such strata is VOID.
  branches T0, branches T2: for each of the 60 states, delays 0-15 NOOP frames on
      raw ALE, then the player's deployed policy to the life's end, capped at 1,536
      frames, delay frames included (rewards stop at the last whole base step
      inside the cap, so a token is cut there). A pooled frame enters the stack
      every 4 delay frames, the
      delay's rewards count with the first base step, and a life lost during a
      delay ends the branch. The wrappers stay in the path (ablation.play_token),
      and restore() resets EpisodicLifeEnv.lives after every restoreState: left
      stale, a later life's first step reads as a death, or a real death is missed.
  g4 T0, g4 T2: G4. One seeded delay per state, the whole branch again in a fresh
      process, policy included: the same tokens, rewards and stacks, and through
      probe_action_census.play_window the same raw frames and end state. Also, for
      T0, every delay-0 branch must be the main line it left: same state, stack
      and policy, so the same tokens, stacks and rewards. Any failure is VOID.
THE READ, on the 60 states at the primary head (0.997 per frame, the student's
clip): a return discounts base step j by gamma ** (4 j) from the first base step,
which carries the delay's rewards, on the clip (T0's ceiling 1, REWARD_UNIT 10, per
base step) and raw.
  - The six Pearson correlations: residuals (a branch minus its state's 16-delay
    mean), pooled over the delay pairs at lag 1, 4 or 8, per player; null -1/15.
    STOP if any exceeds 0.3.
  - G5: rho_w of G from delays 0-7 against delays 8-15, Spearman-Brown; STOP under
    0.3, the pilot's bar (the full sample's is 0.5).
  - rho_w(x, y): inside each stratum of n states, centred ranks (r - 0.5) / n - 0.5,
    ties averaged; their Pearson correlation pooled over all states; a stratum of
    one drops out.
Prints GO, STOP with the reason, or VOID. An undefined statistic (no spread at all)
is a STOP, since the registration has no GO for it. Nothing else moves: strata stay
20 wide, K stays 16.

usage: python probe_gap_check.py pilot [--out DIR]
       python probe_gap_check.py g1|branches|g4 --player T0|T2 [--out DIR]
       python probe_gap_check.py main [--opening 80|100] [--out DIR]
       python probe_gap_check.py select [--out DIR]
DIR defaults to probe_gap_check-pilot next to this file; it ends with pilot.json,
which is never overwritten. Smoke knobs (any non-default marks the record smoke and
moves the default DIR to probe_gap_check-pilot-smoke): GAP_GAMES (240),
GAP_PLAIN_GAMES (30), GAP_STRATA (20), GAP_G6_BAR (30). CPU only, 2 threads.
Imported as a module (the tests) it changes nothing process-wide: the env settings
and probe_exhaustive_eval's import happen only when it runs as a script."""
import os
ENV_ID = "MsPacmanNoFrameskip-v4"
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # "" deletes the var on Windows
    os.environ["PROBE_ENV"] = ENV_ID
    os.environ["PROBE_ACTING_STATES"] = "1"     # the deployed policy
    os.environ["PROBE_THREADS"] = "2"           # as the rows ran
import argparse
import contextlib
import hashlib
import json
import pathlib
import pickle
import random
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT / "experiments", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import ale_py
import numpy as np
import torch
import muzero
import ablation
import probe_action_census as pc
import probe_e3a_sequences as e3

RESULTS = pathlib.Path(r"Y:\SwitchPlayer-testb2-run\benchmarks\results")
PLAYERS = {"T0": {"run": "mspacman-two-t0-s0-2026-10-29", "iteration": 45,
                  "tokens": 36},
           "T2": {"run": "mspacman-two-t2-s0-2026-10-29", "iteration": 42,
                  "tokens": 18}}
BASE = muzero.TOKEN_BASE_FRAMES              # raw frames per base step (4)
EVAL_SEED = 777                              # probe_exhaustive_eval.play's reset
LIFE_CAP = 3000                              # base steps a life, the exact eval's
MOVES = 9                                    # the minimal action set
OPENING, OPENING_LONG = 80, 100              # G6's one lengthening
STATES_PER_GAME = 6
STRATUM_STEPS = 20                           # base steps a stratum's bin spans
PER_STRATUM = 3
DELAYS = tuple(range(16))                    # K = 16 NOOP-frame delays
HALF = 8                                     # G5's split: delays 0-7 against 8-15
FRAME_CAP = 1536                             # a branch's frames, delay included
DISCOUNTS = (0.997, 0.99, 0.98, 0.95)        # per frame; the first is primary
SCALES = ("clip", "raw")                     # the first is primary
CLIP_UNIT, CLIP_CEILING = 10.0, 1.0          # the student's: T0's spec, checked
LAGS = (1, 4, 8)
LAG_NULL = -1.0 / (len(DELAYS) - 1)
LAG_BAR = 0.3                                # a correlation above this stops it
RELIABILITY_BAR = 0.3                        # the pilot's G5 bar
FREE_GRID = tuple((a, 32) for a in range(MOVES))
SEED = 20261107                              # the fixed seed of every draw
GAMES = int(os.environ.get("GAP_GAMES") or 240)
PLAIN_GAMES = int(os.environ.get("GAP_PLAIN_GAMES") or 30)
N_STRATA = int(os.environ.get("GAP_STRATA") or 20)
G6_BAR = float(os.environ.get("GAP_G6_BAR") or 30)
SMOKE = GAMES != 240 or PLAIN_GAMES != 30 or N_STRATA != 20 or G6_BAR != 30
# What a pass's numbers depend on: this probe and the code it plays through.
SOURCES = (pathlib.Path(__file__).resolve(), HERE / "probe_action_census.py",
           HERE / "probe_e3a_sequences.py", HERE / "probe_exhaustive_eval.py",
           ROOT / "experiments" / "ablation.py", ROOT / "muzero.py",
           ROOT / "atari_wrappers.py", ROOT / "EpisodicLifeEnv.py")


def source_hash(paths=SOURCES):
    h = hashlib.blake2b(digest_size=16)
    for path in paths:
        h.update(path.read_bytes())
    return h.hexdigest()


SOURCE = source_hash()                       # as this process imported them


# --------------------------------------------------------------------------
# the statistics
# --------------------------------------------------------------------------
def centred_ranks(x):
    """Ranks inside one stratum, ties averaged, centred: (r - 0.5) / n - 0.5."""
    x = np.asarray(x, dtype=np.float64)
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    average = np.cumsum(counts) - (counts - 1) / 2.0
    return (average[inverse] - 0.5) / len(x) - 0.5


def rho_w(x, y, strata):
    """STATISTIC: the Pearson correlation, pooled over all states, of x's and
    y's centred ranks inside each stratum. A stratum of one drops out. None
    when either side has no spread inside any stratum."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    groups = {}
    for i, k in enumerate(strata):
        groups.setdefault(k, []).append(i)
    rx, ry = [], []
    for members in groups.values():
        if len(members) > 1:
            rx.append(centred_ranks(x[members]))
            ry.append(centred_ranks(y[members]))
    if not rx:
        return None
    rx, ry = np.concatenate(rx), np.concatenate(ry)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    scale = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / scale) if scale > 0 else None


def lag_correlation(returns, lag):
    """The pilot's statistic: `returns` (states, delays) at one head, each
    branch minus its state's mean over the delays, the Pearson correlation
    pooled over every pair of delays `lag` apart. None without spread."""
    r = np.asarray(returns, dtype=np.float64)
    r = r - r.mean(axis=1, keepdims=True)
    x, y = r[:, :-lag].ravel(), r[:, lag:].ravel()
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spearman_brown(r):
    """A half-length correlation stepped up to the whole; None at r = -1 or
    without an r."""
    return None if r is None or r <= -1 else 2.0 * r / (1.0 + r)


def split_half(gap, strata):
    """G5: `gap` (states, delays), T2's return minus T0's at the same delay;
    (rho_w of the mean gap over delays 0-7 against delays 8-15, its
    Spearman-Brown reliability)."""
    gap = np.asarray(gap, dtype=np.float64)
    r = rho_w(gap[:, :HALF].mean(axis=1), gap[:, HALF:].mean(axis=1), strata)
    return r, spearman_brown(r)


def pilot_verdict(correlations, reliability):
    """("GO" | "STOP", reasons). `correlations` {player: {lag: r}}."""
    reasons = []
    for player, lags in correlations.items():
        for lag, r in lags.items():
            if r is None:
                reasons.append(f"{player} lag {lag}: undefined, no residuals")
            elif r > LAG_BAR:
                reasons.append(f"{player} lag {lag}: {r:+.3f} above {LAG_BAR}")
    if reliability is None:
        reasons.append("G5: reliability undefined")
    elif reliability < RELIABILITY_BAR:
        reasons.append(f"G5: reliability {reliability:.3f} under "
                       f"{RELIABILITY_BAR}")
    return ("STOP", reasons) if reasons else ("GO", [])


def pilot_read(t0, t2, strata):
    """The pilot's numbers from each player's (states, delays) returns at the
    primary head. G is never printed: only its reliability is read here."""
    correlations = {"T0": {lag: lag_correlation(t0, lag) for lag in LAGS},
                    "T2": {lag: lag_correlation(t2, lag) for lag in LAGS}}
    half, reliability = split_half(np.asarray(t2) - np.asarray(t0), strata)
    verdict, reasons = pilot_verdict(correlations, reliability)
    return {"correlations": correlations, "null": LAG_NULL, "rho_half": half,
            "reliability": reliability, "verdict": verdict, "reasons": reasons}


def effective_games(scores):
    """G6: 1 / sum p^2 over the games' distinct scores."""
    _, counts = np.unique(np.asarray(scores, dtype=np.float64),
                          return_counts=True)
    p = counts / counts.sum()
    return float(1.0 / (p * p).sum())


def g6_next(effective):
    """What G6 does after the openings played so far, `effective` their
    effective games in order: "pass", "lengthen" (once), or "void"."""
    if effective[-1] >= G6_BAR:
        return "pass"
    return "lengthen" if len(effective) == 1 else "void"


def discounted(rewards, gamma):
    """Rewards per base step, discounted per FRAME from the first base step:
    base step j weighs gamma ** (BASE * j)."""
    r = np.asarray(rewards, dtype=np.float64)
    return float((r * gamma ** (BASE * np.arange(len(r)))).sum())


def head_returns(rewards):
    """A branch's raw rewards per base step on every head: (scales,
    discounts), the student's clip per base step, then raw points."""
    clip = [e3.clipped(r, CLIP_UNIT, CLIP_CEILING) for r in rewards]
    return np.array([[discounted(s, g) for g in DISCOUNTS]
                     for s in (clip, rewards)])


def primary_returns(branches):
    """(states, delays) at the primary head from a pass's branches."""
    return np.array([[head_returns(b["rewards"])[0, 0] for b in row]
                     for row in branches])


# --------------------------------------------------------------------------
# the draws, all seeded
# --------------------------------------------------------------------------
def opening_moves(game, n):
    """Game `game`'s opening: n seeded random moves. The generator of the
    measurement behind the opening (session scratchpad peek_ready.py), so
    100 moves are the same game's 80 and 20 more."""
    r = random.Random(game)
    return [r.randrange(MOVES) for _ in range(n)]


def game_noops(game):
    return 1 + game % 30


def draw_game_states(game, n):
    """The decisions kept from a game of n decisions after the opening."""
    rng = np.random.default_rng((SEED, 1, game))
    return sorted(int(i) for i in rng.choice(n, min(STATES_PER_GAME, n),
                                             replace=False))


def stratum(life, steps):
    return int(life), int(steps) // STRATUM_STEPS


def draw_pilot(keys, n_strata=None, per=PER_STRATUM):
    """The pilot's states from each state's stratum key: (their indices, the
    strata holding at least `per`), the indices None under `n_strata` such
    strata. Strata are drawn first, then `per` states inside each."""
    n_strata = N_STRATA if n_strata is None else n_strata
    groups = {}
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)
    eligible = sorted(k for k, v in groups.items() if len(v) >= per)
    if len(eligible) < n_strata:
        return None, eligible
    rng = np.random.default_rng((SEED, 2))
    picked = [eligible[int(j)]
              for j in sorted(rng.choice(len(eligible), n_strata,
                                         replace=False))]
    chosen = []
    for k in picked:
        chosen += [groups[k][int(j)]
                   for j in sorted(rng.choice(len(groups[k]), per,
                                              replace=False))]
    return chosen, eligible


def g4_delays(n):
    """The one delay per state G4 plays again, the same for both players."""
    rng = np.random.default_rng((SEED, 3))
    return [int(d) for d in rng.integers(0, len(DELAYS), size=n)]


def select_states(states, is_free):
    """The drops, in the registration's order: FREE states, then identical
    stacks, keeping the first (`merged` lists the games a kept state stands
    for). Returns (kept, counts)."""
    kept, seen, free, merged = [], {}, 0, 0
    for s in states:
        if is_free(s):
            free += 1
            continue
        key = pc._digest(s["stack"].tobytes())
        if key in seen:
            merged += 1
            seen[key]["merged"].append(s["game"])
            continue
        seen[key] = {**s, "merged": []}
        kept.append(seen[key])
    return kept, {"sampled": len(states), "free": free, "merged": merged,
                  "kept": len(kept)}


# --------------------------------------------------------------------------
# play: one loop for the main lines and the branches
# --------------------------------------------------------------------------
def raw_env():
    """The deployed wrapper stack with no nets (ablation.build_env's, for the
    passes that need only the emulator, and the tests)."""
    from atari_wrappers import paper_atari_env
    from EpisodicLifeEnv import EpisodicLifeEnv
    env = EpisodicLifeEnv(paper_atari_env(ENV_ID, skip=BASE))
    env.reset(seed=EVAL_SEED)
    return env


def deployed(run):
    """The greedy prior argmax, as Run.play_episode's evaluation takes it."""
    manager = run.searcher.network_manager

    def decide(stack):
        policy, _ = manager.initial_inference(stack)
        return int(policy.argmax())

    return decide


def raw_frames(ale, action, frames, stack, stop=True):
    """`frames` frames of `action` on raw ALE, keeping the rewards: a pooled
    frame (MaxAndSkipEnv's max of a base step's last two) enters `stack`
    every BASE frames, and a remainder enters nothing. With `stop`, a lost
    life ends it at once. Returns (stack, reward, life lost)."""
    lives = ale.lives()
    total, prev = 0.0, None
    for t in range(frames):
        total += float(ale.act(action))
        if stop and (ale.lives() < lives or ale.game_over()):
            return stack, total, True
        k = t % BASE
        if k == BASE - 2:
            prev = np.array(ale.getScreenRGB(), copy=True)
        elif k == BASE - 1:
            stack = e3.reached_stack(stack,
                                     [np.maximum(prev, ale.getScreenRGB())])
    return stack, total, False


def restore(env, state):
    """A saved state onto the deployed wrappers. EpisodicLifeEnv keeps its own
    count of the lives, which restoreState does not touch."""
    ale = env.unwrapped.ale
    ale.restoreState(state)
    env.lives = ale.lives()
    env.was_real_done = False


def play_life(env, decide, stack, steps=0, cap=LIFE_CAP, on_decision=None):
    """The policy from `stack`, which holds the newest frame, until the life
    ends or `steps` reaches `cap`: Run.play_episode's greedy loop through
    ablation.play_token. Returns (raw reward per base step, steps, whether
    the life ended)."""
    rewards = []
    while True:
        token = decide(stack)
        if on_decision is not None:
            on_decision(stack, token, steps)
        inner = []
        obs, _, term, trunc, stack, n = ablation.play_token(env, token, stack,
                                                            inner)
        rewards += inner
        steps += n
        if term or trunc:
            return rewards, steps, True
        if steps >= cap:
            return rewards, steps, False
        stack = e3.reached_stack(stack, [obs])


def start_game(env, noops, moves):
    """A game up to its first decision: probe_exhaustive_eval.play's seeded
    double reset and noop start, then `moves` as 4-frame moves on raw ALE.
    Returns (stack, life index, base steps, the opening's reward)."""
    u = env.unwrapped
    ale, actions = u.ale, list(u._action_set)
    u.reset(seed=EVAL_SEED)
    u.reset()
    for _ in range(noops):
        ale.act(actions[0])
    stack = ablation.initial_frame_stack(ablation.preprocess_frame(
        ale.getScreenRGB(), ablation.IMAGE_RESHAPE))
    life, steps, reward = 0, 0, 0.0
    for a in moves:
        before = ale.lives()
        stack, r, _ = raw_frames(ale, actions[a], BASE, stack, stop=False)
        reward += r
        steps += 1
        if ale.lives() < before:
            life, steps = life + 1, 0
    env.lives = ale.lives()
    env.was_real_done = False
    return stack, life, steps, reward


def main_game(env, decide, game, moves):
    """One whole game: its record, and every decision's (serialized state,
    stack) until the caller has drawn from them. Lives follow Run.play_game:
    EpisodicLifeEnv's reset plays one 4-frame NOOP whose reward is dropped,
    and the next life's stack tiles its frame."""
    ale = env.unwrapped.ale
    stack, life, steps, opening_reward = start_game(env, game_noops(game),
                                                    moves)
    held, rewards = [], {}
    rec = {"game": game, "noops": game_noops(game), "moves": len(moves),
           "opening_reward": opening_reward, "lost_in_opening": life,
           "life": [], "steps": [], "tokens": [], "digests": []}

    def note(stack, token, steps):
        held.append((ale.cloneState().serialize(), stack))
        rec["life"].append(life)
        rec["steps"].append(steps)
        rec["tokens"].append(token)
        rec["digests"].append(pc._digest(stack.tobytes()))

    while True:
        played, steps, ended = play_life(env, decide, stack, steps,
                                         on_decision=note)
        rewards[life] = played
        if not ended or env.was_real_done:
            break
        obs, _ = env.reset()
        assert not env.was_real_done, "a life's NOOP ended the game"
        stack = ablation.initial_frame_stack(ablation.preprocess_frame(
            obs, ablation.IMAGE_RESHAPE))
        life, steps = life + 1, 0
    rec.update(rewards=rewards, terminated=ended, decisions=len(held),
               score=float(sum(sum(v) for v in rewards.values())),
               length=len(moves) + sum(len(v) for v in rewards.values()))
    return rec, held


def plain_game(env, decide, noops):
    """A plain game through the loop above, in the exact-eval row's form:
    [return, base steps, terminated, decisions]."""
    rec, _ = main_game(env, decide, noops - 1, [])
    return rec["score"], rec["length"], rec["terminated"], rec["decisions"]


def branch(env, decide, state, stack, delay, cap=FRAME_CAP):
    """One branch from a saved state: `delay` NOOP frames on raw ALE, then
    `decide` to the life's end, at most `cap` frames in all. `rewards`: raw
    per base step, the delay's in the first."""
    ale = env.unwrapped.ale
    restore(env, state)
    start = ale.getEpisodeFrameNumber()
    stack, first, lost = raw_frames(ale, env.unwrapped._action_set[0], delay,
                                    stack)
    tokens, digests = [], []

    def note(stack, token, steps):
        tokens.append(token)
        digests.append(pc._digest(stack.tobytes()))

    if lost:
        rewards, ended = [first], "delay"
    else:
        budget = (cap - delay) // BASE
        rewards, _, died = play_life(env, decide, stack, cap=budget,
                                     on_decision=note)
        ended = "life" if died and len(rewards) <= budget else "cap"
        rewards = rewards[:budget]
        rewards[0] += first
    return {"delay": delay, "rewards": rewards, "ended": ended,
            "tokens": tokens, "digests": digests,
            "frames": ale.getEpisodeFrameNumber() - start,
            "end": pc._digest(ale.cloneState().serialize())}


def branch_inputs(actions, b):
    """A played branch's ALE action per frame, delay and tokens, up to where
    it really stopped (a life's last token is often cut short)."""
    inputs = [actions[0]] * b["delay"]
    for t in b["tokens"]:
        inputs += e3.input_frames(muzero.token_inputs(muzero.TOKEN_VOCAB[t]),
                                  actions)
    return inputs[:b["frames"]]


def frames_of(env, state, b):
    """G4's digests of a played branch: (every raw RGB frame, the end state),
    replayed on raw ALE as the census plays a window."""
    actions = list(env.unwrapped._action_set)
    inputs = branch_inputs(actions, b)
    return pc.play_window(env.unwrapped.ale, state, inputs, len(inputs),
                          actions[0])


def same_branch(a, b):
    """The fields that must repeat in a fresh process; the ones that differ."""
    return [k for k in ("delay", "rewards", "ended", "tokens", "digests",
                        "frames", "end") if a[k] != b[k]]


def continuation(game, decision):
    """The main line from `decision` to its life's end: (tokens, stack
    digests, raw rewards per base step)."""
    life = game["life"][decision]
    members = [i for i, x in enumerate(game["life"]) if x == life]
    skip = game["steps"][decision] - game["steps"][members[0]]
    last = members[-1] + 1
    return (game["tokens"][decision:last], game["digests"][decision:last],
            game["rewards"][life][skip:])


def main_line_reasons(states, games, zero_delay):
    """T0's delay-0 branches against the main lines they left. `games` by
    game index; a capped branch is a prefix, one that ended its life is the
    whole continuation."""
    reasons = []
    for s, b in zip(states, zero_delay):
        tokens, digests, rewards = continuation(games[s["game"]],
                                                s["decision"])
        n = len(b["tokens"])
        if b["tokens"] != tokens[:n] or b["digests"] != digests[:n]:
            reasons.append(f"game {s['game']} decision {s['decision']}: the "
                           f"branch's decisions left the main line")
        elif b["rewards"] != rewards[:len(b["rewards"])] or (
                b["ended"] == "life" and len(b["rewards"]) != len(rewards)):
            reasons.append(f"game {s['game']} decision {s['decision']}: the "
                           f"branch's rewards left the main line")
    return reasons


# --------------------------------------------------------------------------
# the passes
# --------------------------------------------------------------------------
def snapshot(name):
    p = PLAYERS[name]
    found = sorted((RESULTS / p["run"]).glob(
        f"snapshots-*/weights-iter{p['iteration']:03d}.pth"))
    if len(found) != 1:
        raise SystemExit(f"{name}: {len(found)} snapshots at iteration "
                         f"{p['iteration']} under {RESULTS / p['run']}")
    return found[0].resolve()


def check_player(name, spec):
    """Refuses a snapshot that is not the registered player, and a student
    whose clip is not the clip the returns use."""
    n = len(spec.get("TOKEN_VOCAB") or ())
    if n != PLAYERS[name]["tokens"]:
        raise SystemExit(f"{name}: {n} tokens, not {PLAYERS[name]['tokens']}")
    if name == "T0" and not (
            spec.get("REWARD_UNIT") == CLIP_UNIT
            and spec.get("REWARD_GROWTH_START") == CLIP_CEILING
            and spec.get("REWARD_GROWTH_CEILING") == CLIP_CEILING):
        raise SystemExit(f"T0's clip is not unit {CLIP_UNIT}, ceiling "
                         f"{CLIP_CEILING}: {spec.get('REWARD_UNIT')}, "
                         f"{spec.get('REWARD_GROWTH_START')}, "
                         f"{spec.get('REWARD_GROWTH_CEILING')}")


@contextlib.contextmanager
def player(name):
    """A player's arm context and deployed Run, as the census builds its own:
    the snapshot's tokens and prefix layout, its weights, its acting states."""
    import probe_exhaustive_eval as pe    # sets ablation's env globals
    if pe.SEARCH_SIMS:
        raise SystemExit("the pilot plays the prior argmax: unset "
                         "PROBE_SEARCH_SIMS")
    path = snapshot(name)
    vocab = pe.snapshot_vocab(path)
    pc.check_snapshot(vocab, pe.ENV)
    with ablation.arm_config(pe.ARM), ablation._run_length_globals(90, 2000), \
            pe.snapshot_tokens(vocab), \
            pe.snapshot_prefix_layout(pe.snapshot_prefix_bn(path)):
        run = ablation.Run(pe.ARM, 0, ablation.ARMS[pe.ARM])
        try:
            assert muzero.USE_SEARCH_ACTING is False   # greedy = prior argmax
            d = pe.load(run, path)
            check_player(name, d["spec"])
            pe.apply_acting_states(run, d)
            yield run
        finally:
            run.env.close()


def knobs():
    return {"games": GAMES, "plain_games": PLAIN_GAMES, "strata": N_STRATA,
            "g6_bar": G6_BAR}


def stamp():
    """What made a pass's file: the knobs and the source."""
    return {"knobs": knobs(), "source": SOURCE}


def write(path, obj):
    """Whole or not at all, so a stopped pass leaves nothing to reuse."""
    obj = {**obj, "stamp": stamp()}
    tmp = path.with_name(path.name + ".tmp")
    if path.suffix == ".json":
        tmp.write_text(json.dumps(obj, indent=1))
    else:
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def read(path):
    """A pass's file, refused unless this run would have written it: existing
    is not enough to be reused."""
    if path.suffix == ".json":
        obj = json.loads(path.read_text())
    else:
        with open(path, "rb") as f:
            obj = pickle.load(f)
    if obj.get("stamp") != stamp():
        raise SystemExit(f"{path} was made by {obj.get('stamp')}, this run is "
                         f"{stamp()}: move it aside, passes are never mixed")
    return obj


def main_files(out, opening):
    return out / f"main-{opening:03d}.json", out / f"main-{opening:03d}.pkl"


def stage_g1(out, name):
    import probe_exhaustive_eval as pe
    path = snapshot(name)
    row = pc.exact_eval_row(path)
    if row is None:
        raise SystemExit(f"no deployed exact-eval row for {path}")
    reasons, games, t0 = [], {}, time.time()
    with player(name) as run:
        decide = deployed(run)
        for n in range(1, PLAIN_GAMES + 1):
            want = row["games"].get(str(n), ())
            eval_game = pe.play(run, n)
            own_game = plain_game(run.env, decide, n)
            games[n] = {"eval": list(eval_game), "own": list(own_game)}
            if not pc.same_game(eval_game, want):
                reasons.append(f"game {n} {list(eval_game)} != the row {want}")
            if not pc.same_game(own_game, want):
                reasons.append(f"game {n}: the probe's loop {list(own_game)} "
                               f"!= the row {want}")
    print(f"G1 {name}: {PLAIN_GAMES} games, "
          f"{'OK' if not reasons else 'FAILED: ' + '; '.join(reasons[:4])} "
          f"[{time.time() - t0:.0f}s]", flush=True)
    write(out / f"g1-{name}.json",
          {"ok": not reasons, "reasons": reasons, "games": games,
           "threads": torch.get_num_threads(), "row_mean": row["exact_mean"]})


def stage_main(out, opening):
    t0 = time.time()
    games, states = [], []
    with player("T0") as run:
        decide = deployed(run)
        for g in range(GAMES):
            rec, held = main_game(run.env, decide, g, opening_moves(g, opening))
            for i in draw_game_states(g, len(held)):
                states.append({"game": g, "decision": i,
                               "life": rec["life"][i],
                               "steps": rec["steps"][i],
                               "state": held[i][0], "stack": held[i][1]})
            games.append(rec)
            if (g + 1) % 40 == 0:
                print(f"main {opening}: {g + 1} games, {len(states)} states "
                      f"[{time.time() - t0:.0f}s]", flush=True)
    scores = [r["score"] for r in games]
    summary = {"opening": opening, "games": len(games),
               "effective": effective_games(scores),
               "distinct": len(set(scores)),
               "effective_with_opening": effective_games(
                   [r["score"] + r["opening_reward"] for r in games]),
               "scores": scores, "lengths": [r["length"] for r in games],
               "decisions": sum(r["decisions"] for r in games),
               "lost_in_opening": sum(r["lost_in_opening"] > 0 for r in games),
               "uncapped": all(r["terminated"] for r in games),
               "secs": round(time.time() - t0)}
    print(f"G6 at {opening} moves: {summary['effective']:.1f} effective games "
          f"({summary['distinct']} distinct scores), bar {G6_BAR:g}; with the "
          f"opening's reward {summary['effective_with_opening']:.1f}; "
          f"{summary['lost_in_opening']} lost a life in the opening "
          f"[{summary['secs']}s]", flush=True)
    summary_path, states_path = main_files(out, opening)
    write(states_path, {"opening": opening, "games": games, "states": states})
    write(summary_path, summary)


def stage_select(out, opening):
    data = read(main_files(out, opening)[1])
    env = raw_env()
    try:
        ale = env.unwrapped.ale
        actions = list(env.unwrapped._action_set)

        def is_free(s):
            part, _, _ = pc.token_census(ale, actions,
                                         ale_py.ALEState(s["state"]),
                                         grid=FREE_GRID)
            return len(set(part)) == 1

        kept, counts = select_states(data["states"], is_free)
    finally:
        env.close()
    keys = [stratum(s["life"], s["steps"]) for s in kept]
    chosen, eligible = draw_pilot(keys)
    counts.update(strata=len(set(keys)), eligible=len(eligible))
    print(f"states: {counts['sampled']} sampled, {counts['free']} FREE, "
          f"{counts['merged']} merged, {counts['kept']} kept; {counts['strata']} "
          f"strata, {counts['eligible']} holding {PER_STRATUM}+", flush=True)
    write(out / "select.pkl",
          {"opening": opening, "counts": counts, "kept": kept,
           "chosen": chosen, "keys": keys})


def chosen_states(out):
    sel = read(out / "select.pkl")
    return [sel["kept"][i] for i in sel["chosen"]], \
        [sel["keys"][i] for i in sel["chosen"]]


def stage_branches(out, name):
    states, _ = chosen_states(out)
    checked = g4_delays(len(states))
    t0, table, reasons = time.time(), [], []
    with player(name) as run:
        env, decide = run.env, deployed(run)
        for i, s in enumerate(states):
            saved = ale_py.ALEState(s["state"])
            row = [branch(env, decide, saved, s["stack"], d) for d in DELAYS]
            b = row[checked[i]]
            b["replay"], end = frames_of(env, saved, b)
            if end != b["end"]:
                reasons.append(f"state {i} delay {checked[i]}: the raw replay "
                               f"ended elsewhere")
            table.append(row)
            if (i + 1) % 10 == 0:
                print(f"branches {name}: {i + 1} states "
                      f"[{time.time() - t0:.0f}s]", flush=True)
    write(out / f"branches-{name}.pkl",
          {"table": table, "reasons": reasons, "threads":
           torch.get_num_threads(), "secs": round(time.time() - t0)})


def stage_g4(out, name):
    states, _ = chosen_states(out)
    data = read(out / f"branches-{name}.pkl")
    table, checked = data["table"], g4_delays(len(states))
    reasons = list(data["reasons"])
    t0 = time.time()
    with player(name) as run:
        env, decide = run.env, deployed(run)
        for i, s in enumerate(states):
            saved = ale_py.ALEState(s["state"])
            first = table[i][checked[i]]
            again = branch(env, decide, saved, s["stack"], checked[i])
            frames, end = frames_of(env, saved, again)
            differ = same_branch(first, again)
            if frames != first["replay"]:
                differ.append("raw frames")
            if end != again["end"]:
                differ.append("raw replay's end")
            if differ:
                reasons.append(f"state {i} delay {checked[i]}: {differ}")
    if name == "T0":
        sel = read(out / "select.pkl")
        games = read(main_files(out, sel["opening"])[1])["games"]
        reasons += main_line_reasons(states, games,
                                     [row[0] for row in table])
    print(f"G4 {name}: {len(states)} branches in a fresh process, "
          f"{'OK' if not reasons else 'FAILED: ' + '; '.join(reasons[:4])} "
          f"[{time.time() - t0:.0f}s]", flush=True)
    write(out / f"g4-{name}.json", {"ok": not reasons, "reasons": reasons})


def run_stage(out, stage, *args):
    """A pass in its own process; a failed pass stops the pilot."""
    cmd = [sys.executable, "-u", str(pathlib.Path(__file__).resolve()), stage,
           "--out", str(out), *args]
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode:
        raise SystemExit(f"pass {stage} {' '.join(args)}: rc {r.returncode}")


def finish(out, verdict, reasons, **fields):
    rec = {"verdict": verdict, "reasons": reasons, "smoke": SMOKE, **fields}
    write(out / "pilot.json", rec)
    line = f"PILOT: {verdict}" + (f" ({'; '.join(reasons)})" if reasons else "")
    print(("SMOKE " if SMOKE else "") + line, flush=True)
    return rec


def pilot(out):
    if (out / "pilot.json").exists():
        raise SystemExit(f"{out / 'pilot.json'} exists: a pilot is never "
                         f"overwritten (move it aside first)")
    for name in PLAYERS:
        snapshot(name)
    fields = {"knobs": knobs()}
    t0 = time.time()
    for name in PLAYERS:
        if not (out / f"g1-{name}.json").exists():
            run_stage(out, "g1", "--player", name)
        g1 = read(out / f"g1-{name}.json")
        if not g1["ok"]:
            return finish(out, "VOID", [f"G1 {name}: {g1['reasons'][0]}"],
                          **fields)
    effective, opening = [], OPENING
    while True:
        if not main_files(out, opening)[0].exists():
            run_stage(out, "main", "--opening", str(opening))
        effective.append(read(main_files(out, opening)[0])["effective"])
        step = g6_next(effective)
        if step == "pass":
            break
        if step == "void":
            return finish(out, "VOID", [
                f"G6: {effective[-1]:.1f} effective games at {opening} moves, "
                f"bar {G6_BAR:g}, after the one lengthening"],
                effective=effective, **fields)
        opening = OPENING_LONG
    fields.update(effective=effective, opening=opening)
    if not (out / "select.pkl").exists():
        run_stage(out, "select", "--opening", str(opening))
    sel = read(out / "select.pkl")
    fields["counts"] = sel["counts"]
    if sel["chosen"] is None:
        return finish(out, "VOID", [
            f"{sel['counts']['eligible']} strata hold {PER_STRATUM} states, "
            f"{N_STRATA} needed"], **fields)
    for name in PLAYERS:
        if not (out / f"branches-{name}.pkl").exists():
            run_stage(out, "branches", "--player", name)
    for name in PLAYERS:
        if not (out / f"g4-{name}.json").exists():
            run_stage(out, "g4", "--player", name)
        g4 = read(out / f"g4-{name}.json")
        if not g4["ok"]:
            return finish(out, "VOID", [f"G4 {name}: {g4['reasons'][0]}"],
                          **fields)
    _, keys = chosen_states(out)
    table = {n: read(out / f"branches-{n}.pkl")["table"] for n in PLAYERS}
    read_out = pilot_read(primary_returns(table["T0"]),
                          primary_returns(table["T2"]), keys)
    for name, lags in read_out["correlations"].items():
        for lag, r in lags.items():
            print(f"residual correlation {name} lag {lag}: "
                  f"{'undefined' if r is None else format(r, '+.3f')} "
                  f"(null {LAG_NULL:+.4f}, bar {LAG_BAR})", flush=True)
    reliability = read_out["reliability"]
    print(f"G5: rho_w of the halves {read_out['rho_half']}, Spearman-Brown "
          f"{reliability}, bar {RELIABILITY_BAR}", flush=True)
    return finish(out, read_out["verdict"], read_out["reasons"], **fields,
                  **{k: read_out[k] for k in ("correlations", "null",
                                              "rho_half", "reliability")},
                  ended={n: {e: sum(b["ended"] == e for row in table[n]
                                    for b in row)
                             for e in ("life", "cap", "delay")}
                         for n in PLAYERS},
                  secs=round(time.time() - t0))


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("pilot", "g1", "main", "select",
                                      "branches", "g4"))
    ap.add_argument("--player", choices=tuple(PLAYERS))
    ap.add_argument("--opening", type=int, default=OPENING,
                    choices=(OPENING, OPENING_LONG))
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    out = pathlib.Path(args.out) if args.out else \
        HERE / f"probe_gap_check-pilot{'-smoke' if SMOKE else ''}"
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "pilot":
        pilot(out)
    elif args.stage == "main":
        stage_main(out, args.opening)
    elif args.stage == "select":
        stage_select(out, args.opening)
    else:
        if args.player is None:
            raise SystemExit(f"{args.stage} needs --player")
        {"g1": stage_g1, "branches": stage_branches,
         "g4": stage_g4}[args.stage](out, args.player)


if __name__ == "__main__":
    main(sys.argv[1:])
