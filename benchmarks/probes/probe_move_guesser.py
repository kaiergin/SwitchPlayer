"""Test H (CURRICULUM "Next experiment (pre-registered) -- Test H", registration
audit loop closed 2026-11-11 at e306a3f4): does knowing where a player got to
help guess the move they made?

STAGE A, own-play data (this file's `data` command). O0's kept snapshots at
iterations 10, 20, ..., 160 play 10 whole games each, OF's at 10, ..., 150 and
159 play 2 each, under training play's own acting rule (Run.play_episode, not
greedy): the prior read with the snapshot's acting BN states, tempered by the
staged acting temperature, no search. Each game:
  - starts with the training reset (noop starts, its count drawn from the game's
    seed) and then Test G's opening, OPENING uniformly random moves of the 9,
    which are not the player's (no decision index, never samples or scored);
  - plays every life through the training wrapper stack, with play_episode's
    frame-stack rule and life changes (EpisodicLifeEnv's NOOP step, the stack
    re-tiled), each life capped at LIFE_CAP base steps;
  - draws each decision's move as the tempered prior's inverse CDF at a uniform
    u from a counter-based stream keyed by (run, snapshot, game, decision
    index), the distribution legacy np.random.choice draws from.
Recorded per event (a life start or a base step): the pooled 84x84x3 frame, the
move, its kind, u, the reward, the life, the step's raw-frame digests, and at
decisions the raw prior and the temperature. Held-out games (O0's 8-9, all of
OF's) also keep, at every decision, the emulator state and MaxAndSkip's
two-frame buffer (outside the ALE state) for stage B's truth.

The temperature is pinned from the runs, not the snapshot spec:
TEMPERATURE_TOTAL_STEPS 300,000 (the logs' planned run), SCHEDULE_ENV_TARGET
100,000 (--stop-env-steps), SCHEDULE_ENV_STEPS row N's env_steps; the stage
refuses a snapshot whose T differs from an independent recomputation, or is not
among the beh_T_values of the row that played those weights (row N+5; O0's 160:
row 162; OF's 159: none). After all games it checks the opening's diversity:
first-decision stacks byte-distinct in at least 90% of games, overall and within
each noop phase, or it stops (Kai's call).

usage (from the run tree, under the lock, CPU):
  python benchmarks/probes/probe_move_guesser.py data [--out DIR]
      [--player O0|OF] [--iterations 10,20] [--games N] [--life-cap N]
The optional flags make a SMOKE run, which writes under a "smoke" directory and
never into the real output. Games already written with this source's stamp are
skipped, so a stopped stage resumes. CPU only, THREADS threads, batch-1
forwards: stage B replays these forwards byte for byte. Imported as a module
(the tests, stage B, the trainer) it changes nothing process-wide: the env
settings happen only when it runs as a script, and probe_exhaustive_eval is
imported only inside player()."""
import os
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # "" deletes the var on Windows
    os.environ["PROBE_ENV"] = "MsPacmanNoFrameskip-v4"
    os.environ["PROBE_THREADS"] = "2"
import argparse
import contextlib
import ctypes
import hashlib
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT / "experiments", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import gymnasium as gym
import numpy as np
import torch
import muzero
import ablation
from atari_wrappers import MaxAndSkipEnv, NoopResetEnv
from EpisodicLifeEnv import EpisodicLifeEnv

ENV_ID = "MsPacmanNoFrameskip-v4"
MOVES = 9                                   # the minimal action set
BASE = 4                                    # raw frames per base step
LIFE_CAP = 3000                             # base steps a life, as in training
OPENING = 80                                # Test G's random opening
PLANNED_ITERATIONS, TRAIN_STEPS = 150, 2000  # TEMPERATURE_TOTAL_STEPS 300,000
ENV_TARGET = 100_000                        # both runs' --stop-env-steps
STAGES = ((50_000, 1.0), (75_000, 0.5))     # env steps below: stage factor
LATE_FACTOR = 0.25
DIVERSITY_BAR = 0.9
FREE_COMMIT_GB = 8.0
THREADS = 2                                 # torch CPU threads, data and truth alike
ARM_SCALE = 0.5                             # PRIOR_ACT_TEMP_SCALE of both runs
DIGEST = 16                                 # bytes per raw-frame digest
RUNS = {
    "O0": {"tree": os.environ.get("TESTH_O0_TREE") or r"Y:\SwitchPlayer-testo-run",
           "run": "mspacman-plain-o0-s0-2026-11-07",
           "iterations": tuple(range(10, 161, 10)), "games": 10,
           "held": (8, 9)},
    "OF": {"tree": os.environ.get("TESTH_OF_TREE") or r"Y:\SwitchPlayer-testof-run",
           "run": "mspacman-plain-of-s0-2026-11-07",
           "iterations": tuple(range(10, 151, 10)) + (159,), "games": 2,
           "held": (0, 1)},
}
SPLIT = {"fit": (0, 1, 2, 3, 4, 5), "val": (6, 7), "held": (8, 9)}
KIND_START, KIND_OPEN, KIND_MOVE = 0, 1, 2
# What the data depend on: this probe and the code it plays through.
SOURCES = (pathlib.Path(__file__).resolve(), HERE / "probe_exhaustive_eval.py",
           ROOT / "experiments" / "ablation.py", ROOT / "muzero.py",
           ROOT / "atari_wrappers.py", ROOT / "EpisodicLifeEnv.py",
           ROOT / "networks.py", ROOT / "residual_block.py",
           ROOT / "scalar_categorical.py")


def source_hash(paths=SOURCES):
    h = hashlib.blake2b(digest_size=16)
    for path in paths:
        h.update(path.read_bytes())
    return h.hexdigest()


SOURCE = source_hash()                      # as this process imported them


# --------------------------------------------------------------------------
# seeds and uniforms: counter-based, independent of play
# --------------------------------------------------------------------------
def game_key(run, iteration, game):
    """The game's 63-bit key: its noop count, opening and uniforms all derive
    from it, so no draw depends on how play went."""
    d = hashlib.blake2b(f"testh|{run}|{int(iteration)}|{int(game)}".encode(),
                        digest_size=8).digest()
    return int.from_bytes(d, "little") & (2 ** 63 - 1)


def _rng(key, tag, index=0):
    return np.random.Generator(np.random.PCG64(
        np.random.SeedSequence([int(key) & 0xFFFFFFFF, int(key) >> 32,
                                int(tag), int(index)])))


TAG_NOOP, TAG_OPEN, TAG_U = 1, 2, 3


def noop_count(key):
    """1..30, as NoopResetEnv draws."""
    return int(_rng(key, TAG_NOOP).integers(1, 31))


def opening_move(key, j):
    return int(_rng(key, TAG_OPEN, j).integers(0, MOVES))


def uniform(key, decision):
    """The decision's uniform in [0, 1): the stream stage B's branches reuse
    at the same decision indices (common random numbers)."""
    return float(_rng(key, TAG_U, decision).random())


# --------------------------------------------------------------------------
# the acting rule
# --------------------------------------------------------------------------
def tempered(policy, t):
    """play_episode's tempering, the same float32 numpy operations."""
    if t != 1.0:
        policy = policy ** (1.0 / t)
        policy = policy / policy.sum()
    return policy


def sample(policy, u):
    """Legacy np.random.choice(len(p), p=policy) at its one uniform u: p is
    converted to float64, cdf = cumsum / cdf[-1], searchsorted side right; the
    same refusals (non-finite, negative, a sum off 1 by more than the input
    dtype's sqrt(eps))."""
    src = np.asarray(policy)
    p = src.astype(np.float64)
    atol = np.sqrt(np.finfo(np.float64).eps)
    if np.issubdtype(src.dtype, np.floating):
        atol = max(atol, float(np.sqrt(np.finfo(src.dtype).eps)))
    if not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError("probabilities are not non-negative and finite")
    if abs(p.sum() - 1.0) > atol:
        raise ValueError("probabilities do not sum to 1")
    cdf = p.cumsum()
    cdf /= cdf[-1]
    return int(cdf.searchsorted(u, side="right"))


def expected_temperature(env_steps, scale):
    """The independent recomputation: the staged factor at env-step progress
    (50k and 75k of 100k), times the arm's scale."""
    for bound, factor in STAGES:
        if env_steps < bound:
            return factor * scale
    return LATE_FACTOR * scale


# --------------------------------------------------------------------------
# the env: the training wrapper stack, with a tap on the raw frames
# --------------------------------------------------------------------------
class RawTap(gym.Wrapper):
    """Transparent: passes every call through and keeps a digest of every raw
    frame stepped since the last take()."""

    def __init__(self, env):
        super().__init__(env)
        self.digests = []

    def step(self, action):
        out = self.env.step(action)
        self.digests.append(hashlib.blake2b(np.ascontiguousarray(out[0]).tobytes(),
                                            digest_size=DIGEST).digest())
        return out

    def take(self):
        d, self.digests = self.digests, []
        return d


def make_env():
    """build_env's USE_ATARI_WRAPPERS stack, raw -> NoopReset(30) ->
    MaxAndSkip(4) -> EpisodicLife, with RawTap under NoopReset. Returns
    (env, tap, noop wrapper, MaxAndSkip wrapper)."""
    raw = gym.make(ENV_ID, frameskip=1)
    tap = RawTap(raw)
    noop = NoopResetEnv(tap, noop_max=30)
    skip = MaxAndSkipEnv(noop, skip=BASE)
    return EpisodicLifeEnv(skip), tap, noop, skip


@contextlib.contextmanager
def fixed_noops(noop_env, noops):
    """NoopResetEnv.reset with a fixed count, as probe_exhaustive_eval.play."""

    def fixed(self, **kw):
        obs, info = self.env.reset(**kw)
        for _ in range(noops):
            obs, _, _, _, info = self.env.step(0)
        return obs, info

    noop_env.reset = fixed.__get__(noop_env)
    try:
        yield
    finally:
        del noop_env.reset


def frame_of(obs):
    return ablation.preprocess_frame(obs, ablation.IMAGE_RESHAPE)


# --------------------------------------------------------------------------
# one game
# --------------------------------------------------------------------------
def play_game(env, tap, skip, choose, opening, cap=LIFE_CAP, keep_states=False,
              on_decision=None):
    """One whole game from env's real reset (the caller set the noops and
    was_real_done), lives as Run.play_game plays them, each life as
    Run.play_episode (not greedy): reset, tile the frame, then per base step
    push the newest frame and choose. `choose(stack, j, decision)` returns
    (move, u, prior, t): j counts the game's base steps (life-change NOOPs
    excluded), the first `opening` of them are opening moves (decision None);
    after them decision counts the player's moves from 0. Returns the events
    as arrays and, with keep_states, each decision's (serialized emulator
    state, MaxAndSkip buffer, lives)."""
    ale = env.unwrapped.ale
    ev = {k: [] for k in ("kind", "move", "u", "decision", "life", "reward",
                          "end", "frame", "digests", "prior", "temp")}
    states = []
    env.was_real_done = True
    life, j, decision = 0, 0, 0
    while True:
        tap.take()
        obs, _ = env.reset()
        if life > 0 and env.was_real_done:
            # the life-change NOOP ended the game and the wrapper reset it:
            # never seen in Ms. Pac-Man (Test G's assertion), refused here
            raise RuntimeError("the life-change NOOP ended the game")
        frame = frame_of(obs)
        stack = muzero.initial_frame_stack(frame)
        _event(ev, KIND_START, -1, np.nan, -1, life, 0.0, 0, frame, tap.take(),
               None, np.nan)
        steps, done = 0, False
        while not done and steps < cap:
            stack = muzero.push_frame(stack, frame)
            if j < opening:
                move, u, prior, t, d, kind = opening_choice(choose, stack, j)
            else:
                if keep_states:
                    states.append((decision, ale.cloneState().serialize(),
                                   skip._buffer.copy(), int(env.lives)))
                if on_decision is not None:
                    on_decision(stack, decision)
                move, u, prior, t = choose(stack, j, decision)
                d, kind = decision, KIND_MOVE
                decision += 1
            obs, reward, term, trunc, _ = env.step(move)
            done = term or trunc
            frame = frame_of(obs)
            steps += 1
            j += 1
            _event(ev, kind, move, u, d, life, float(reward),
                   1 if done else (2 if steps >= cap else 0), frame,
                   tap.take(), prior, t)
        if not done:
            env.was_real_done = True        # a capped life ends the game
        if not done or env.was_real_done:
            break
        life += 1
    return _arrays(ev), states


def opening_choice(choose, stack, j):
    move, _, _, _ = choose(stack, j, None)
    return move, np.nan, None, np.nan, -1, KIND_OPEN


def _event(ev, kind, move, u, decision, life, reward, end, frame, digests,
           prior, temp):
    ev["kind"].append(kind)
    ev["move"].append(move)
    ev["u"].append(u)
    ev["decision"].append(decision)
    ev["life"].append(life)
    ev["reward"].append(reward)
    ev["end"].append(end)
    ev["frame"].append(frame)
    ev["digests"].append(digests)
    ev["prior"].append(np.full(MOVES, np.nan, np.float32) if prior is None
                       else np.asarray(prior, np.float32))
    ev["temp"].append(temp)


def _arrays(ev):
    n = len(ev["kind"])
    dig = np.zeros((n, BASE + 30, DIGEST), np.uint8)   # a reset can hold noops
    ndig = np.zeros(n, np.int16)
    for i, ds in enumerate(ev["digests"]):
        ds = ds[-dig.shape[1]:]
        ndig[i] = len(ds)
        for k, d in enumerate(ds):
            dig[i, k] = np.frombuffer(d, np.uint8)
    return {"kind": np.array(ev["kind"], np.int8),
            "move": np.array(ev["move"], np.int8),
            "u": np.array(ev["u"], np.float64),
            "decision": np.array(ev["decision"], np.int32),
            "life": np.array(ev["life"], np.int8),
            "reward": np.array(ev["reward"], np.float32),
            "end": np.array(ev["end"], np.int8),
            "frames": np.stack(ev["frame"]).astype(np.uint8),
            "digests": dig, "ndigests": ndig,
            "prior": np.stack(ev["prior"]),
            "temp": np.array(ev["temp"], np.float64)}


def stacks_from_frames(arr):
    """Every decision's stack rebuilt from the recorded frames by
    play_episode's rule (a life's start frame tiled, then each step's frame
    pushed before the next choice): the input stage B and the trainer use.
    Returns {decision: stack}."""
    out = {}
    stack = frame = None
    for i in range(len(arr["kind"])):
        k = int(arr["kind"][i])
        if k == KIND_START:
            frame = arr["frames"][i]
            stack = muzero.initial_frame_stack(frame)
            continue
        stack = muzero.push_frame(stack, frame)
        if k == KIND_MOVE:
            out[int(arr["decision"][i])] = stack
        frame = arr["frames"][i]
    return out


# --------------------------------------------------------------------------
# snapshots, rows, the temperature gate
# --------------------------------------------------------------------------
def snapshot_path(player, iteration):
    spec = RUNS[player]
    found = sorted((pathlib.Path(spec["tree"]) / "benchmarks" / "results" /
                    spec["run"]).glob(f"snapshots-*/weights-iter{iteration:03d}.pth"))
    if len(found) != 1:
        raise SystemExit(f"{player} @{iteration}: {len(found)} snapshots")
    return found[0]


def rows(player):
    spec = RUNS[player]
    path = (pathlib.Path(spec["tree"]) / "benchmarks" / "results" / spec["run"] /
            "ablation-mspacmannoframeskip-skip4.jsonl")
    out = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if "iteration" in r and "env_steps" in r:
                out[int(r["iteration"])] = r
    return out


def temperature_gate(player, iteration, table, t_code, scale=ARM_SCALE):
    """(env_steps, the playing row or None, message). SystemExit when T
    differs from the recomputation, or is not among the playing row's
    beh_T_values. The playing row is the next row after N (rows come every 5
    iterations); OF's 159 has none."""
    if iteration not in table:
        raise SystemExit(f"{player} @{iteration}: no jsonl row")
    env_steps = int(table[iteration]["env_steps"])
    t_exp = expected_temperature(env_steps, scale)
    if abs(t_code - t_exp) > 1e-12:
        raise SystemExit(f"{player} @{iteration}: T {t_code} differs from the "
                         f"recomputation {t_exp} at {env_steps} env steps")
    later = sorted(i for i in table if i > iteration)
    if not later:
        return env_steps, None, (f"T {t_code} at {env_steps} env steps; no row "
                                 f"played these weights, env steps alone")
    play = later[0]
    vals = table[play].get("beh_T_values")
    if not vals or not any(abs(float(v) - t_code) < 1e-9 for v in vals):
        raise SystemExit(f"{player} @{iteration}: T {t_code} not among row "
                         f"{play}'s beh_T_values {vals}")
    return env_steps, play, (f"T {t_code} at {env_steps} env steps; row {play} "
                             f"beh_T_values {vals}")


@contextlib.contextmanager
def player(path, env_steps):
    """The snapshot's deployed acting nets (its acting BN states) in a Run,
    with the temperature pins set; everything restored after.
    probe_exhaustive_eval is imported here, not at module import, since its
    import sets process-wide globals (Test G's rule)."""
    os.environ["PROBE_ENV"] = ENV_ID
    import probe_exhaustive_eval as pe
    vocab = pe.snapshot_vocab(path)
    if vocab is not None:
        raise SystemExit(f"{path.name}: a token snapshot; Test H plays plain moves")
    saved = (muzero.PRIOR_ACT_STAGED_TEMP, muzero.PRIOR_ACT_TEMP_SCALE,
             muzero.SCHEDULE_ENV_TARGET, muzero.SCHEDULE_ENV_STEPS)
    with ablation.arm_config(pe.ARM), \
            ablation._run_length_globals(PLANNED_ITERATIONS, TRAIN_STEPS), \
            pe.snapshot_tokens(None), \
            pe.snapshot_prefix_layout(pe.snapshot_prefix_bn(path)), \
            pe.snapshot_option_layout(pe.snapshot_option_length(path)):
        run = ablation.Run(pe.ARM, 0, ablation.ARMS[pe.ARM])
        try:
            if not muzero.USE_ATARI_WRAPPERS or ablation.ACTIONS != MOVES:
                raise SystemExit("the arm does not play the 9-move wrapper stack")
            if muzero.USE_SEARCH_ACTING:
                raise SystemExit("the arm acts by search; Test H plays the prior")
            d = pe.load(run, path)
            spec = d.get("spec") or {}
            if not spec.get("BN_RECAL_ACTING"):
                raise SystemExit(f"{path.name}: not a BN_RECAL_ACTING run")
            pe.apply_acting_states(run, d)
            if float(spec["PRIOR_ACT_TEMP_SCALE"]) != ARM_SCALE or \
                    not spec["PRIOR_ACT_STAGED_TEMP"]:
                raise SystemExit(f"{path.name}: acting temperature "
                                 f"{spec.get('PRIOR_ACT_STAGED_TEMP')} x "
                                 f"{spec.get('PRIOR_ACT_TEMP_SCALE')}, not staged x "
                                 f"{ARM_SCALE}")
            muzero.PRIOR_ACT_STAGED_TEMP = bool(spec["PRIOR_ACT_STAGED_TEMP"])
            muzero.PRIOR_ACT_TEMP_SCALE = float(spec["PRIOR_ACT_TEMP_SCALE"])
            muzero.SCHEDULE_ENV_TARGET = ENV_TARGET
            muzero.SCHEDULE_ENV_STEPS = int(env_steps)
            if muzero.TEMPERATURE_TOTAL_STEPS != PLANNED_ITERATIONS * TRAIN_STEPS:
                raise SystemExit("TEMPERATURE_TOTAL_STEPS is not 300,000")
            if muzero.token_long_bias(run.trainer.training_steps):
                raise SystemExit("a long-token bias in a plain run")
            yield run, d
        finally:
            (muzero.PRIOR_ACT_STAGED_TEMP, muzero.PRIOR_ACT_TEMP_SCALE,
             muzero.SCHEDULE_ENV_TARGET, muzero.SCHEDULE_ENV_STEPS) = saved
            run.env.close()


def chooser(run, key, t):
    """The game's choose(): opening moves from the key's stream, then the
    acting rule at the key's uniforms."""
    manager = run.searcher.network_manager

    def choose(stack, j, decision):
        if decision is None:
            return opening_move(key, j), None, None, None
        policy, _ = manager.initial_inference(stack)
        u = uniform(key, decision)
        return sample(tempered(policy, t), u), u, policy, t

    return choose


# --------------------------------------------------------------------------
# files
# --------------------------------------------------------------------------
def free_commit_gb():
    """Windows: the commit still available (ullAvailPageFile), in GB."""

    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
        raise OSError("GlobalMemoryStatusEx failed")
    return m.ullAvailPageFile / 2 ** 30


def commit_gate():
    if sys.platform != "win32":
        return
    free = free_commit_gb()
    if free < FREE_COMMIT_GB:
        raise SystemExit(f"free commit {free:.1f} GB < {FREE_COMMIT_GB} GB")


def game_name(player, iteration, game):
    return f"{player}-it{iteration:03d}-g{game}"


def written(out, name, cap=None):
    """The game's meta if it is written by this source (and, given `cap`, at
    that life cap and this opening), else None."""
    meta = out / f"{name}.json"
    if not meta.exists() or not (out / f"{name}.npz").exists():
        return None
    m = json.loads(meta.read_text())
    if m.get("source") != SOURCE or m.get("threads") != THREADS:
        return None
    if cap is not None and (m.get("life_cap") != cap or m.get("opening") != OPENING):
        return None
    return m


def pin_process():
    """CPU, THREADS threads: refuses a process whose nets would run elsewhere."""
    if ablation.DEVICE.type != "cpu":
        raise SystemExit(f"the probe runs on the CPU, not {ablation.DEVICE}")
    os.environ["PROBE_THREADS"] = str(THREADS)  # probe_exhaustive_eval's import reads it
    torch.set_num_threads(THREADS)


def check_threads():
    if torch.get_num_threads() != THREADS:
        raise SystemExit(f"torch threads {torch.get_num_threads()}, not {THREADS}")


def write_game(out, name, arr, states, meta):
    """npz, the states file (held-out games) and the json last: a stopped
    write leaves no json, so it is redone."""
    tmp = out / f"{name}.npz.tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arr)
    os.replace(tmp, out / f"{name}.npz")
    if states is not None:
        blob = b"".join(s[1] for s in states)
        offs = np.cumsum([0] + [len(s[1]) for s in states]).astype(np.int64)
        tmp = out / f"{name}-states.npz.tmp"
        with open(tmp, "wb") as f:
            np.savez_compressed(
                f, decision=np.array([s[0] for s in states], np.int32),
                blob=np.frombuffer(blob, np.uint8), offsets=offs,
                buffer=np.stack([s[2] for s in states]) if states else
                np.zeros((0, 2, 210, 160, 3), np.uint8),
                lives=np.array([s[3] for s in states], np.int8))
        os.replace(tmp, out / f"{name}-states.npz")
    tmp = out / f"{name}.json.tmp"
    tmp.write_text(json.dumps(meta, indent=1))
    os.replace(tmp, out / f"{name}.json")


def split_of(player, game):
    if player == "OF":
        return "held"
    for k, games in SPLIT.items():
        if game in games:
            return k
    raise ValueError(game)


# --------------------------------------------------------------------------
# stage A
# --------------------------------------------------------------------------
def stage_data(out, players, iterations=None, games=None, cap=LIFE_CAP):
    out.mkdir(parents=True, exist_ok=True)
    commit_gate()
    pin_process()
    for p in players:
        spec = RUNS[p]
        table = rows(p)
        its = spec["iterations"] if iterations is None else iterations
        n_games = spec["games"] if games is None else games
        for it in its:
            names = [game_name(p, it, g) for g in range(n_games)]
            if all(written(out, n, cap) for n in names):
                continue
            path = snapshot_path(p, it)
            env_steps = int(table[it]["env_steps"])
            with player(path, env_steps) as (run, d):
                steps_at = run.searcher.network_manager.training_steps
                t = float(muzero.prior_acting_temperature(steps_at))
                env_steps, play_row, msg = temperature_gate(p, it, table, t)
                print(f"{p} @{it}: {msg}", flush=True)
                check_threads()
                env, tap, noop, skip = make_env()
                try:
                    for g, name in enumerate(names):
                        if written(out, name, cap):
                            continue
                        commit_gate()
                        key = game_key(spec["run"], it, g)
                        noops = noop_count(key)
                        held = split_of(p, g) == "held"
                        t0 = time.time()
                        env.reset(seed=key % (2 ** 31))
                        with fixed_noops(noop, noops):
                            arr, states = play_game(
                                env, tap, skip, chooser(run, key, t), OPENING,
                                cap=cap, keep_states=held)
                        n_dec = int((arr["kind"] == KIND_MOVE).sum())
                        meta = {"player": p, "run": spec["run"], "iteration": it,
                                "game": g, "split": split_of(p, g), "key": key,
                                "noops": noops, "env_steps": env_steps,
                                "play_row": play_row, "temperature": t,
                                "training_steps": int(d.get("training_steps", 0)),
                                "decisions": n_dec,
                                "base_steps": int((arr["kind"] != KIND_START).sum()),
                                "lives": int(arr["life"].max()) + 1,
                                "return": float(arr["reward"].sum()),
                                "life_cap": cap, "opening": OPENING,
                                "env_seed": key % (2 ** 31), "threads": THREADS,
                                "device": str(ablation.DEVICE),
                                "source": SOURCE, "secs": round(time.time() - t0, 1)}
                        write_game(out, name, arr, states if held else None, meta)
                        print(f"  {name}: {n_dec} decisions, return "
                              f"{meta['return']:.0f}, {meta['lives']} lives, "
                              f"noops {noops} [{meta['secs']}s]", flush=True)
                finally:
                    env.close()
    return diversity(out, players, iterations, games)


def diversity(out, players, iterations=None, games=None):
    """The opening check: first-decision stacks byte-distinct in at least
    DIVERSITY_BAR of the games, overall and within each noop phase (noops mod
    4). Returns the report; SystemExit (Kai's call) below the bar."""
    keys, phase = [], {}
    for p in players:
        spec = RUNS[p]
        its = spec["iterations"] if iterations is None else iterations
        n_games = spec["games"] if games is None else games
        for it in its:
            for g in range(n_games):
                name = game_name(p, it, g)
                meta = written(out, name)
                if meta is None:
                    raise SystemExit(f"{name}: missing")
                with np.load(out / f"{name}.npz") as z:
                    arr = {k: z[k] for k in ("kind", "frames", "decision")}
                first = stacks_from_frames(arr).get(0)
                h = None if first is None else hashlib.blake2b(
                    first.tobytes(), digest_size=16).hexdigest()
                keys.append(h)
                phase.setdefault(meta["noops"] % 4, []).append(h)

    def share(hs):
        hs = [h for h in hs if h is not None]
        return (len(set(hs)) / len(hs)) if hs else 0.0

    rep = {"games": len(keys), "no_decision": sum(h is None for h in keys),
           "distinct_share": share(keys),
           "by_phase": {str(k): {"games": len(v), "distinct_share": share(v)}
                        for k, v in sorted(phase.items())}}
    low = [k for k, v in rep["by_phase"].items()
           if v["distinct_share"] < DIVERSITY_BAR]
    rep["passed"] = not (rep["distinct_share"] < DIVERSITY_BAR or low)
    print("diversity:", json.dumps(rep), flush=True)
    (out / "diversity.json").write_text(json.dumps({**rep, "source": SOURCE},
                                                   indent=1))
    if not rep["passed"]:
        raise SystemExit(f"STOP (Kai's call): first-decision stacks distinct in "
                         f"{rep['distinct_share']:.3f} of games; phases below "
                         f"{DIVERSITY_BAR}: {low}")
    return rep


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["data"])
    ap.add_argument("--out")
    ap.add_argument("--player", choices=sorted(RUNS))
    ap.add_argument("--iterations")
    ap.add_argument("--games", type=int)
    ap.add_argument("--life-cap", type=int)
    a = ap.parse_args(argv)
    smoke = bool(a.player or a.iterations or a.games or a.life_cap)
    base = pathlib.Path(a.out) if a.out else ROOT / "benchmarks" / "probes" / "testh"
    out = base / ("smoke" if smoke else "data")
    players = [a.player] if a.player else ["O0", "OF"]
    its = (tuple(int(x) for x in a.iterations.split(",")) if a.iterations
           else None)
    stage_data(out, players, its, a.games, a.life_cap or LIFE_CAP)
    print("DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
