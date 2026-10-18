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
reproducible. ~5.5 min per snapshot (prior argmax; a search agent costs
the search per step). PROBE_SEARCH_SIMS / PROBE_SEARCH_ENGINE: the search
agent instead of the prior argmax (below). PROBE_ENV (default Pong-v4): the
env id the snapshots were trained on (Ms. Pac-Man, 2026-10-16:
MsPacmanNoFrameskip-v4); the action count comes from the env. Every game is
a whole GAME (Run.play_game: all lives of a multi-life game, the 3000 cap per
life); a Pong game is one life, so Pong records are unchanged."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # "" deletes the var on Windows
import contextlib
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
ENV = os.environ.get("PROBE_ENV") or "Pong-v4"


def env_actions(env_id):
    """The env's action count (Pong-v4: 6, the value hardcoded before
    PROBE_ENV)."""
    import gymnasium as gym
    probe = gym.make(env_id)
    try:
        return int(probe.action_space.n)
    finally:
        probe.close()


ablation.ENV_NAME, ablation.FRAMESKIP, ablation.EPISODE_CAP, ablation.ACTIONS = \
    ENV, 4, 3000, env_actions(ENV)
HERE = pathlib.Path(__file__).parent


def load(run, path):
    d = torch.load(str(path), map_location="cpu", weights_only=False)
    v = (d.get("spec") or {}).get("TOKEN_VOCAB")
    v = None if v is None else tuple(tuple(t) for t in v)
    if v != muzero.TOKEN_VOCAB:
        # One vocabulary per probe call (main sets it from the first file).
        raise SystemExit(f"{path.name}: TOKEN_VOCAB {v!r} differs from the "
                         f"probe's {muzero.TOKEN_VOCAB!r}")
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


# PROBE_ACTING_STATES=1 (BN stabilizer, 2026-10-08): evaluate the DEPLOYED
# policy -- the snapshot's shipped representation / prediction acting
# states (recalibrated BN statistics) instead of the trainer's -- and write
# probe_exhaustive_eval-<run>-acting.jsonl. A snapshot without them refuses.
ACTING = os.environ.get("PROBE_ACTING_STATES") == "1"


def apply_acting_states(run, d):
    acting = d.get("acting_states")
    if not acting:
        raise SystemExit("PROBE_ACTING_STATES=1 but the snapshot has no "
                         "acting_states (not a BN_RECAL_ACTING run)")
    for n, sd in acting.items():
        getattr(run.trainer, n).load_state_dict(sd)
    run.set_train_mode(False)


# PROBE_SEARCH_SIMS=N (Gumbel search acting, 2026-10-12): evaluate the
# SEARCH agent a search-acting run would deploy instead of the prior argmax
# -- every step a greedy N-sim act_search through play_episode:
# PROBE_SEARCH_ENGINE "puct" (argmax of the visits, no root noise) or
# "gumbel" (argmax of the Gumbel improved policy; ACTOR_GUMBEL_SEARCH).
# Gumbel's halving still draws Gumbel noise, so np.random is reseeded per
# game: a snapshot's result is reproducible. Writes
# probe_exhaustive_eval-<run>-<engine><N>.jsonl. Not with
# PROBE_ACTING_STATES (root-only BN statistics are wrong inside a search).
SEARCH_SIMS = int(os.environ.get("PROBE_SEARCH_SIMS") or 0)
SEARCH_ENGINE = os.environ.get("PROBE_SEARCH_ENGINE", "puct")
ENGINE_SET = "PROBE_SEARCH_ENGINE" in os.environ
# Smoke runs only: play noop counts 1..MAX_NOOP (the readout uses all 30);
# a smoke record goes to a "-noops<N>" file, never the readout's.
MAX_NOOP = int(os.environ.get("PROBE_MAX_NOOP") or 30)


def check_mode():
    """Refuses a search engine without a budget (it would silently run the
    prior argmax) and a noop range outside 1..30."""
    if ENGINE_SET and not SEARCH_SIMS:
        raise SystemExit("PROBE_SEARCH_ENGINE without PROBE_SEARCH_SIMS would "
                         "evaluate the prior argmax; set both")
    if not 1 <= MAX_NOOP <= 30:
        raise SystemExit(f"PROBE_MAX_NOOP must be 1..30, got {MAX_NOOP}")


def configure_search():
    """The search agent's flags (inside the probe's arm_config); refuses
    a bad engine or budget, and the acting-states combination."""
    if SEARCH_ENGINE not in ("puct", "gumbel") or SEARCH_SIMS < 1:
        raise SystemExit(f"PROBE_SEARCH_ENGINE must be puct|gumbel and "
                         f"PROBE_SEARCH_SIMS >= 1, got {SEARCH_ENGINE!r} / "
                         f"{SEARCH_SIMS!r}")
    if ACTING:
        raise SystemExit("PROBE_SEARCH_SIMS with PROBE_ACTING_STATES: the "
                         "root-only BN statistics are wrong inside a search")
    muzero.USE_SEARCH_ACTING = True      # greedy play_episode searches ...
    muzero.PAPER_TARGETS = True          # ... (only in paper mode) ...
    muzero.EVAL_PRIOR_ONLY = False       # ... per step
    muzero.SEARCH_ACT_BIRTH_TARGETS = False
    muzero.ACT_SEARCH_SIMS = SEARCH_SIMS  # greedy searches use this budget
    muzero.USE_GUMBEL_SEARCH = False
    muzero.ACTOR_GUMBEL_SEARCH = SEARCH_ENGINE == "gumbel"


def snapshot_vocab(path):
    """The TOKEN_VOCAB a snapshot's run trained with (curriculum P2a; the
    weights file's spec), as tuples -- None for a plain-action run."""
    d = torch.load(str(path), map_location="cpu", weights_only=False)
    v = (d.get("spec") or {}).get("TOKEN_VOCAB")
    return None if v is None else tuple(tuple(t) for t in v)


@contextlib.contextmanager
def snapshot_tokens(vocab):
    """Play the snapshots' tokens: TOKEN_VOCAB and the vocabulary-sized
    action space for the probe's Run, restored after. None = unchanged."""
    saved = muzero.TOKEN_VOCAB, ablation.ACTIONS
    if vocab is not None:
        muzero.TOKEN_VOCAB = vocab
        ablation.ACTIONS = len(vocab)
    try:
        yield
    finally:
        muzero.TOKEN_VOCAB, ablation.ACTIONS = saved


def play(run, noops):
    """One noop-start game: (return, base steps, terminated), plus the
    decisions played under TOKEN_VOCAB (where base steps count frames / 4
    and decisions count tokens)."""
    ne = _noop_env(run.env)
    if SEARCH_SIMS:
        np.random.seed(1000 + noops)
    tokens = muzero.TOKEN_VOCAB is not None
    decisions = [0]
    if tokens:
        inner = run.play_episode

        def counted(*a, **k):
            out = inner(*a, **k)
            decisions[0] += len(out[0]) - 1   # records minus the final one
            return out

        run.play_episode = counted

    def fixed(self, **kw):
        obs, info = self.env.reset(**kw)
        for _ in range(noops):
            obs, _, _, _, info = self.env.step(0)
        return obs, info

    ne.reset = fixed.__get__(ne)
    try:
        run.env.was_real_done = True     # every game starts from a real reset
        run.env.reset(seed=777)
        # Unbound, as the Ray eval actor calls it: any object with env and
        # play_episode plays a game.
        ret, steps, term = ablation.Run.play_game(run, cap=3000)
    finally:
        del ne.reset
        if tokens:
            del run.play_episode          # back to the class's method
    if tokens:
        return float(ret), int(steps), bool(term), int(decisions[0])
    return float(ret), int(steps), bool(term)


def main(argv):
    check_mode()
    vocab = snapshot_vocab(argv[0]) if argv else None
    with ablation.arm_config(ARM), ablation._run_length_globals(90, 2000), \
            snapshot_tokens(vocab):
        run = ablation.Run(ARM, 0, ablation.ARMS[ARM])
        assert muzero.USE_SEARCH_ACTING is False       # greedy = prior argmax
        if SEARCH_SIMS:
            configure_search()                         # greedy = the search agent
        for arg in argv:
            path = pathlib.Path(arg)
            d = load(run, path)
            if ACTING:
                apply_acting_states(run, d)
            t0 = time.time()
            games = {n: play(run, n) for n in range(1, MAX_NOOP + 1)}
            rets = np.array([games[n][0] for n in games])
            phase = {k: float(np.mean([games[n][0] for n in games if n % 4 == k]))
                     for k in range(4) if any(n % 4 == k for n in games)}
            rec = {"snapshot": str(path), "run": path.parent.parent.name,
                   "iteration": int(path.stem[-3:]) if path.stem[-3:].isdigit() else None,
                   "exact_mean": round(float(rets.mean()), 3),
                   "sd": round(float(rets.std()), 2),
                   "min": float(rets.min()),
                   "phase_means": phase,
                   "phases_won": sum(1 for v in phase.values() if v > 0),
                   "capped": sum(1 for g in games.values() if not g[2]),
                   "games": {n: list(g) for n, g in games.items()},
                   "secs": round(time.time() - t0),
                   "acting_states": ACTING,
                   "variant": ("acting" if ACTING else
                               f"search-{SEARCH_ENGINE}{SEARCH_SIMS}" if SEARCH_SIMS
                               else "raw")}
            if ENV != "Pong-v4":
                rec["env"] = ENV
            if vocab is not None:
                # TOKEN_VOCAB: each game is [return, base steps, terminated,
                # decisions]; the deployed player's raw frames per decision,
                # pooled over the row's games (P2a read 4: decisions per
                # second = 60 / this; audit r3 A-NIT: pooled, not a mean of
                # per-game ratios).
                rec["tokens"] = len(vocab)
                rec["frames_per_decision"] = round(
                    sum(g[1] for g in games.values()) * muzero.TOKEN_BASE_FRAMES
                    / max(1, sum(g[3] for g in games.values())), 3)
            if SEARCH_SIMS:
                rec["search"] = {"engine": SEARCH_ENGINE, "sims": SEARCH_SIMS}
            if MAX_NOOP != 30:
                rec["max_noop"] = MAX_NOOP
            print(f"{rec['run']} @{rec['iteration']}: exact {rec['exact_mean']:+.2f} "
                  f"(sd {rec['sd']}, min {rec['min']:+.0f}) phases "
                  f"{[round(v, 1) for v in phase.values()]} won {rec['phases_won']}/4 "
                  f"capped {rec['capped']}/{len(games)} [{rec['secs']}s]", flush=True)
            suffix = ("-acting" if ACTING else
                      f"-{SEARCH_ENGINE}{SEARCH_SIMS}" if SEARCH_SIMS else "")
            if MAX_NOOP != 30:
                suffix += f"-noops{MAX_NOOP}"     # a smoke run
            with open(HERE / f"probe_exhaustive_eval-{rec['run']}{suffix}.jsonl",
                      "a") as f:
                f.write(json.dumps(rec) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
