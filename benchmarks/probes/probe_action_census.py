"""E0, the action-equivalence census (CURRICULUM "Queued experiment
(pre-registered) -- E0", Kai 2026-10-30): on the agent's own play, how many
distinct outcomes do the tokens and short input sequences produce per
state? The main line is the exact eval's (probe_exhaustive_eval.play: the
deployed acting states, prior argmax, noop starts 1..N, the 3000 cap per
life). At every decision the emulator state is saved, branches are played
on the raw ALE from it, and the state is restored before the agent's own
token plays:
  - the 36 grid tokens (9 actions x 4 / 8 / 16 / 32 frames), each followed
    by NOOP to a common window of 32 + 60 frames;
  - at every SEQ_EVERY-th decision of a game (decisions 0, SEQ_EVERY, ...),
    every sequence of 1-3 base steps (4 frames each), each followed by NOOP
    to a common window of 12 + 60 frames, expanded breadth-first with
    identical (emulator state, frames so far) merged -- such branches have
    identical futures under the same inputs, so the counts are exact.
Two branches are EQUIVALENT when every raw RGB frame of the window is
identical (Kai); STRICT: the same serialized emulator state at the window's
end. Measurement only: no training, and the emulator's branches never feed
a model (Kai's rule for the E line).

usage: python probe_action_census.py SNAPSHOT.pth
Writes probe_action_census-<run>.jsonl (the summary: reads (1)-(6)) and
probe_action_census-<run>-decisions.jsonl (one line per decision, with the
36 grid tokens' class partition, for E1's scoring) next to itself.
CENSUS_GAMES (default 30, the exact eval's noop starts 1..30),
CENSUS_SEQ_EVERY (default 10), CENSUS_MAX_DECISIONS (smoke runs only: stop
each game's census after that many decisions; the main line plays on). A
smoke run's files carry "-smoke". CPU only, as the exact eval.
Imported as a module (the tests) it changes nothing process-wide: the env
settings and probe_exhaustive_eval's import (which sets ablation's env
globals) happen only when it runs as a script."""
import os
ENV_ID = "MsPacmanNoFrameskip-v4"
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # "" deletes the var on Windows
    os.environ["PROBE_ENV"] = ENV_ID
    os.environ["PROBE_ACTING_STATES"] = "1"     # the deployed policy
import contextlib
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
import numpy as np
import torch
import muzero
import ablation

DURATIONS = (4, 8, 16, 32)
GRID = tuple((a, f) for a in range(9) for f in DURATIONS)   # TK's 36 order
HORIZON = 60                                 # frames after the longest input
WINDOW_TOKENS = max(DURATIONS) + HORIZON     # 92
SEQ_DEPTH = 3
WINDOW_SEQ = SEQ_DEPTH * muzero.TOKEN_BASE_FRAMES + HORIZON   # 72
GAMES = int(os.environ.get("CENSUS_GAMES") or 30)
SEQ_EVERY = int(os.environ.get("CENSUS_SEQ_EVERY") or 10)
MAX_DECISIONS = int(os.environ.get("CENSUS_MAX_DECISIONS") or 0)
SMOKE = GAMES != 30 or SEQ_EVERY != 10 or MAX_DECISIONS > 0


def _digest(data):
    return hashlib.blake2b(data, digest_size=16).digest()


def partition(keys):
    """Class ids by first appearance: equal keys share an id."""
    ids = {}
    return [ids.setdefault(k, len(ids)) for k in keys]


def play_window(ale, start, inputs, window, noop):
    """From `start`, play `inputs` (one ALE action per frame) then `noop` to
    `window` frames; (the frames' digest, the end state's digest)."""
    ale.restoreState(start)
    h = hashlib.blake2b(digest_size=16)
    for t in range(window):
        ale.act(inputs[t] if t < len(inputs) else noop)
        h.update(ale.getScreenRGB().tobytes())
    return h.digest(), _digest(ale.cloneState().serialize())


def token_census(ale, actions, start, grid=GRID, window=WINDOW_TOKENS):
    """Every grid token from `start`, each followed by NOOP to `window`
    frames: (frame partition, strict partition) over `grid`. Restores
    `start`."""
    frames, strict = [], []
    for a, f in grid:
        fk, sk = play_window(ale, start, [actions[a]] * f, window, actions[0])
        frames.append(fk)
        strict.append(sk)
    ale.restoreState(start)
    return partition(frames), partition(strict)


def sequence_census(ale, actions, start, depth=SEQ_DEPTH,
                    base=muzero.TOKEN_BASE_FRAMES, window=WINDOW_SEQ):
    """Every sequence of 1..depth base steps over `actions` from `start`,
    each followed by NOOP to `window` frames: the distinct frame windows
    and distinct end states per length, ([k_1..k_depth], [strict...]).
    Breadth-first; a node is (emulator state, frames so far), and nodes
    equal in both are merged (identical futures). Restores `start`."""
    noop = actions[0]
    level = [(start, hashlib.blake2b(digest_size=16))]
    k_frames, k_strict = [], []
    for length in range(1, depth + 1):
        nxt = {}
        for state, hist in level:
            for a in actions:
                ale.restoreState(state)
                h = hist.copy()
                for _ in range(base):
                    ale.act(a)
                    h.update(ale.getScreenRGB().tobytes())
                st = ale.cloneState()
                nxt.setdefault((_digest(st.serialize()), h.digest()), (st, h))
        level = list(nxt.values())
        fk, sk = set(), set()
        for state, hist in level:
            ale.restoreState(state)
            h = hist.copy()
            for _ in range(window - length * base):
                ale.act(noop)
                h.update(ale.getScreenRGB().tobytes())
            fk.add(h.digest())
            sk.add(_digest(ale.cloneState().serialize()))
        k_frames.append(len(fk))
        k_strict.append(len(sk))
    ale.restoreState(start)
    return k_frames, k_strict


def entropy(p):
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def prior_split(prior, vocab_grid, frame_part):
    """Read (5): the prior's entropy over the agent's tokens, and over the
    same mass summed within the 36-token frame classes."""
    p = np.asarray(prior, dtype=np.float64)
    p = p / p.sum()
    mass = {}
    for i, g in enumerate(vocab_grid):
        c = frame_part[g]
        mass[c] = mass.get(c, 0.0) + p[i]
    return entropy(p), entropy(list(mass.values()))


def decision_record(game, index, token, vocab_grid, frame_part, strict_part,
                    prior, seq=None):
    """One decision's census line. vocab_grid: the agent's tokens' indices
    in GRID; the partitions are over GRID."""
    sub = [frame_part[g] for g in vocab_grid]
    sub_s = [strict_part[g] for g in vocab_grid]
    by_dur = {str(f): len({frame_part[a * len(DURATIONS) + j]
                           for a in range(9)})
              for j, f in enumerate(DURATIONS)}
    per_action = [len({frame_part[a * len(DURATIONS) + j]
                       for j in range(len(DURATIONS))}) for a in range(9)]
    h_tok, h_cls = prior_split(prior, vocab_grid, frame_part)
    chosen = vocab_grid[token]
    rec = {"game": game, "d": index, "token": int(token),
           "classes_vocab": len(set(sub)), "classes_grid": len(set(frame_part)),
           "strict_vocab": len(set(sub_s)),
           "strict_grid": len(set(strict_part)),
           "by_duration": by_dur, "durations_per_action": per_action,
           "chosen_class_size": sum(1 for c in frame_part
                                    if c == frame_part[chosen]),
           "h_tokens": round(h_tok, 5), "h_classes": round(h_cls, 5),
           "partition": frame_part, "strict_partition": strict_part}
    if seq is not None:
        rec["seq"] = {"frames": seq[0], "strict": seq[1]}
    return rec


def summarize(recs, n_vocab):
    """Reads (1)-(6) over the decision records."""
    cv = np.array([r["classes_vocab"] for r in recs], dtype=np.float64)
    cg = np.array([r["classes_grid"] for r in recs], dtype=np.float64)

    def dist(x):
        return {"mean": round(float(x.mean()), 3),
                "median": float(np.median(x)),
                "p10": float(np.percentile(x, 10)),
                "p90": float(np.percentile(x, 90))}

    pairs = {}
    for j, f in enumerate(DURATIONS):
        for k in range(j + 1, len(DURATIONS)):
            g = DURATIONS[k]
            same = [r["partition"][a * len(DURATIONS) + j]
                    == r["partition"][a * len(DURATIONS) + k]
                    for r in recs for a in range(9)]
            pairs[f"{f}={g}"] = round(float(np.mean(same)), 4)
    seqs = [r["seq"] for r in recs if "seq" in r]
    out = {"decisions": len(recs), "tokens_vocab": n_vocab,
           "tokens_grid": len(GRID),
           # (1) PRIMARY: the effective branching factor of the vocabulary
           "classes_vocab": dist(cv),
           "ratio_vocab": round(float(cv.mean()) / n_vocab, 4),
           "classes_grid": dist(cg),
           "ratio_grid": round(float(cg.mean()) / len(GRID), 4),
           # (2) by duration; a short token + NOOP against a longer one
           "by_duration": {str(f): round(float(np.mean(
               [r["by_duration"][str(f)] for r in recs])), 3)
               for f in DURATIONS},
           "durations_per_action": round(float(np.mean(
               [x for r in recs for x in r["durations_per_action"]])), 3),
           "same_class_by_durations": pairs,
           # (4) free decisions
           "free_vocab": round(float(np.mean(cv == 1)), 4),
           "free_grid": round(float(np.mean(cg == 1)), 4),
           "at_most_two_vocab": round(float(np.mean(cv <= 2)), 4),
           # (5) the prior's split
           "h_tokens": round(float(np.mean([r["h_tokens"] for r in recs])), 4),
           "h_classes": round(float(np.mean([r["h_classes"] for r in recs])),
                              4),
           # (6) frames against emulator state
           "strict_vocab_mean": round(float(np.mean(
               [r["strict_vocab"] for r in recs])), 3),
           "strict_grid_mean": round(float(np.mean(
               [r["strict_grid"] for r in recs])), 3)}
    if seqs:
        # (3) sequences
        kf = np.array([s["frames"] for s in seqs], dtype=np.float64)
        ks = np.array([s["strict"] for s in seqs], dtype=np.float64)
        out["seq_decisions"] = len(seqs)
        out["seq_frames_mean"] = [round(float(x), 3) for x in kf.mean(0)]
        out["seq_strict_mean"] = [round(float(x), 3) for x in ks.mean(0)]
        out["seq_nominal"] = [9 ** (L + 1) for L in range(kf.shape[1])]
        out["seq_growth"] = [round(float(kf.mean(0)[L] / kf.mean(0)[L - 1]),
                                   3) for L in range(1, kf.shape[1])]
    r = out["ratio_vocab"]
    out["verdict"] = ("E1" if r <= 0.5 else
                      "KAI'S CALL (does not pay)" if r > 0.75 else
                      "KAI'S CALL (in between)")
    return out


class Census:
    """The main line's hook: wraps ablation.play_token so that each
    decision is censused from the saved emulator state before the agent's
    token plays."""

    def __init__(self, run, vocab, out):
        self.run, self.out = run, out
        self.vocab_grid = [GRID.index(tuple(t)) for t in vocab]
        self.game, self.index, self.recs = 0, 0, []

    def decision(self, env, token, stack):
        ale = env.unwrapped.ale
        actions = list(env.unwrapped._action_set)
        start = ale.cloneState()
        frame_part, strict_part = token_census(ale, actions, start)
        seq = (sequence_census(ale, actions, start)
               if self.index % SEQ_EVERY == 0 else None)
        ale.restoreState(start)
        with torch.no_grad():
            prior, _ = self.run.searcher.network_manager.initial_inference(
                stack)
        prior = (prior.detach().cpu().numpy() if hasattr(prior, "detach")
                 else np.asarray(prior))
        rec = decision_record(self.game, self.index, token, self.vocab_grid,
                              frame_part, strict_part, prior, seq)
        self.recs.append(rec)
        self.out.write(json.dumps(rec) + "\n")
        self.index += 1

    @contextlib.contextmanager
    def hooked(self):
        orig = ablation.play_token

        def play_token(env, token, stack, inner=None):
            if not MAX_DECISIONS or self.index < MAX_DECISIONS:
                self.decision(env, token, stack)
            return orig(env, token, stack, inner)

        ablation.play_token = play_token
        try:
            yield
        finally:
            ablation.play_token = orig


def check_snapshot(vocab, env):
    """The census needs a token snapshot on Ms. Pac-Man whose tokens are
    grid tokens."""
    if env != ENV_ID:
        raise SystemExit(f"PROBE_ENV must be {ENV_ID}, got {env}")
    if vocab is None:
        raise SystemExit("the census needs a TOKEN_VOCAB snapshot")
    bad = [t for t in vocab if tuple(t) not in GRID]
    if bad:
        raise SystemExit(f"tokens outside the 36-token grid: {bad}")


def exact_eval_mean(path):
    """The snapshot's deployed exact-eval row (probe_exhaustive_eval's
    -acting file next to this probe, or the run tree's), or None. The
    census's main line must reproduce it: a difference means a branch
    leaked into the main line, and the census is void."""
    run = path.parent.parent.name
    it = int(path.stem[-3:]) if path.stem[-3:].isdigit() else None
    for d in (HERE, path.parents[3] / "probes"):
        f = d / f"probe_exhaustive_eval-{run}-acting.jsonl"
        if f.exists():
            for line in f.read_text().splitlines():
                r = json.loads(line)
                if (r.get("iteration") == it and "max_noop" not in r
                        and r.get("env") == ENV_ID):
                    return float(r["exact_mean"])
    return None


def main(argv):
    import probe_exhaustive_eval as pe    # sets ablation's env globals
    path = pathlib.Path(argv[0])
    vocab = pe.snapshot_vocab(path)
    check_snapshot(vocab, pe.ENV)
    pbn = pe.snapshot_prefix_bn(path)
    run_name = path.parent.parent.name
    sfx = "-smoke" if SMOKE else ""
    with ablation.arm_config(pe.ARM), ablation._run_length_globals(90, 2000), \
            pe.snapshot_tokens(vocab), pe.snapshot_prefix_layout(pbn):
        run = ablation.Run(pe.ARM, 0, ablation.ARMS[pe.ARM])
        assert muzero.USE_SEARCH_ACTING is False       # greedy = prior argmax
        d = pe.load(run, path)
        pe.apply_acting_states(run, d)
        t0 = time.time()
        games = {}
        dec = HERE / f"probe_action_census-{run_name}{sfx}-decisions.jsonl"
        if dec.exists() and not SMOKE:
            raise SystemExit(f"{dec} exists: a census is never overwritten "
                             f"(move it aside first)")
        with open(dec, "w") as out:
            census = Census(run, vocab, out)
            with census.hooked():
                for n in range(1, GAMES + 1):
                    census.game, census.index = n, 0
                    games[n] = pe.play(run, n)
                    print(f"game {n}: return {games[n][0]:.0f}, "
                          f"{games[n][3]} decisions, {len(census.recs)} "
                          f"censused [{time.time() - t0:.0f}s]", flush=True)
        rec = {"snapshot": str(path), "run": run_name,
               "iteration": (int(path.stem[-3:]) if path.stem[-3:].isdigit()
                             else None),
               "env": ENV_ID, "games": GAMES, "seq_every": SEQ_EVERY,
               "max_decisions": MAX_DECISIONS or None,
               "window_tokens": WINDOW_TOKENS, "window_seq": WINDOW_SEQ,
               "exact_mean": round(float(np.mean([g[0] for g in
                                                  games.values()])), 3),
               "secs": round(time.time() - t0),
               **summarize(census.recs, len(vocab))}
        ref = exact_eval_mean(path)
        rec["exact_eval_mean"] = ref
        rec["main_line_ok"] = (None if ref is None or GAMES != 30 else
                               abs(rec["exact_mean"] - ref) < 1e-6)
        if rec["main_line_ok"] is False:
            rec["verdict"] = "VOID (the main line differs from the exact eval)"
        with open(HERE / f"probe_action_census-{run_name}{sfx}.jsonl",
                  "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps({k: v for k, v in rec.items()
                          if k not in ("snapshot",)}), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
