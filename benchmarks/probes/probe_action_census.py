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
identical (Kai); STRICT: equivalent AND the same serialized emulator state
at the window's end (a refinement: it also separates state the screen does
not show yet). Measurement only: no training, and the emulator's branches
never feed a model (Kai's rule for the E line).

VALIDITY (A3): every censused game must equal the exact eval's game of the
same noop start ([return, base steps, terminated, decisions]; the
snapshot's deployed exact-eval row, looked up BEFORE anything plays: a
full run refuses without one). Also checked per decision: the hook's prior
argmax is the agent's token, and the agent's own grid token replays to the
digest the census gave it (branch-order independence). Any failure makes
the verdict VOID.

usage: python probe_action_census.py SNAPSHOT.pth
Writes probe_action_census-<run>.jsonl (the summary: reads (1)-(6)) and
probe_action_census-<run>-decisions.jsonl (one line per decision, with the
36 grid tokens' class partitions, for E1's scoring) next to itself.
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
        h.update(ale.getScreenRGB())
    return h.digest(), _digest(ale.cloneState().serialize())


def token_census(ale, actions, start, grid=GRID, window=WINDOW_TOKENS):
    """Every grid token from `start`, each followed by NOOP to `window`
    frames: (frame partition, strict partition, frame digests) over `grid`.
    Strict keys are (frames, end state), so strict refines frames. Restores
    `start`."""
    frames, strict = [], []
    for a, f in grid:
        fk, sk = play_window(ale, start, [actions[a]] * f, window, actions[0])
        frames.append(fk)
        strict.append((fk, sk))
    ale.restoreState(start)
    return partition(frames), partition(strict), frames


def sequence_census(ale, actions, start, depth=SEQ_DEPTH,
                    base=muzero.TOKEN_BASE_FRAMES, window=WINDOW_SEQ):
    """Every sequence of 1..depth base steps over `actions` from `start`,
    each followed by NOOP to `window` frames: per length, the number of
    distinct frame windows and of distinct (frame window, end state) pairs,
    ([k_1..k_depth], [strict...]). Breadth-first; a node is (emulator
    state, frames so far), and nodes equal in both are merged (identical
    futures under identical inputs). Restores `start`."""
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
                    h.update(ale.getScreenRGB())
                st = ale.cloneState()
                nxt.setdefault((_digest(st.serialize()), h.digest()), (st, h))
        level = list(nxt.values())
        fk, sk = set(), set()
        for state, hist in level:
            ale.restoreState(state)
            h = hist.copy()
            for _ in range(window - length * base):
                ale.act(noop)
                h.update(ale.getScreenRGB())
            fk.add(h.digest())
            sk.add((h.digest(), _digest(ale.cloneState().serialize())))
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


def verdict(ratio):
    """The pre-registered rule on the (unrounded) ratio."""
    return ("E1" if ratio <= 0.5 else
            "KAI'S CALL (does not pay)" if ratio > 0.75 else
            "KAI'S CALL (in between)")


def summarize(recs, n_vocab):
    """Reads (1)-(6) over the decision records, and (1b) (A6) over the
    decisions where some input changes something (classes_grid > 1)."""
    cv = np.array([r["classes_vocab"] for r in recs], dtype=np.float64)
    cg = np.array([r["classes_grid"] for r in recs], dtype=np.float64)

    def dist(x):
        return {"mean": round(float(x.mean()), 3),
                "median": float(np.median(x)),
                "p10": float(np.percentile(x, 10)),
                "p90": float(np.percentile(x, 90))}

    # (2) the duration read leaves NOOP out: its four tokens are one input
    # stream (NOOP throughout), equal by construction (audit r1a L2)
    moving = range(1, 9)
    pairs = {}
    for j, f in enumerate(DURATIONS):
        for k in range(j + 1, len(DURATIONS)):
            g = DURATIONS[k]
            same = [r["partition"][a * len(DURATIONS) + j]
                    == r["partition"][a * len(DURATIONS) + k]
                    for r in recs for a in moving]
            pairs[f"{f}={g}"] = round(float(np.mean(same)), 4)
    seqs = [r["seq"] for r in recs if "seq" in r]
    ratio = float(cv.mean()) / n_vocab
    live = cg > 1
    out = {"decisions": len(recs), "tokens_vocab": n_vocab,
           "tokens_grid": len(GRID),
           # (1) PRIMARY: the effective branching factor of the vocabulary
           "classes_vocab": dist(cv),
           "ratio_vocab": round(ratio, 4),
           "classes_grid": dist(cg),
           "ratio_grid": round(float(cg.mean()) / len(GRID), 4),
           # (1b) (A6): the decisions where some input matters
           "live_share": round(float(live.mean()), 4),
           # (2) by duration; a short token + NOOP against a longer one
           "by_duration": {str(f): round(float(np.mean(
               [r["by_duration"][str(f)] for r in recs])), 3)
               for f in DURATIONS},
           "durations_per_action": round(float(np.mean(
               [r["durations_per_action"][a] for r in recs
                for a in moving])), 3),
           "same_class_by_durations": pairs,
           # (4) free decisions
           "free_vocab": round(float(np.mean(cv == 1)), 4),
           "free_grid": round(float(np.mean(cg == 1)), 4),
           "at_most_two_vocab": round(float(np.mean(cv <= 2)), 4),
           # (5) the prior's split
           "h_tokens": round(float(np.mean([r["h_tokens"] for r in recs])), 4),
           "h_classes": round(float(np.mean([r["h_classes"] for r in recs])),
                              4),
           # (6) frames against frames + emulator state
           "strict_vocab_mean": round(float(np.mean(
               [r["strict_vocab"] for r in recs])), 3),
           "strict_grid_mean": round(float(np.mean(
               [r["strict_grid"] for r in recs])), 3)}
    if live.any():
        ratio_b = float(cv[live].mean()) / n_vocab
        out["classes_vocab_live"] = dist(cv[live])
        out["ratio_vocab_live"] = round(ratio_b, 4)
    if seqs:
        # (3) sequences; the growth is a ratio of the per-length means
        kf = np.array([s["frames"] for s in seqs], dtype=np.float64)
        ks = np.array([s["strict"] for s in seqs], dtype=np.float64)
        out["seq_decisions"] = len(seqs)
        out["seq_frames_mean"] = [round(float(x), 3) for x in kf.mean(0)]
        out["seq_strict_mean"] = [round(float(x), 3) for x in ks.mean(0)]
        out["seq_nominal"] = [9 ** (L + 1) for L in range(kf.shape[1])]
        out["seq_growth"] = [round(float(kf.mean(0)[L] / kf.mean(0)[L - 1]),
                                   3) for L in range(1, kf.shape[1])]
    # A6: the verdict is (1)'s; when (1b) falls on another side of a
    # threshold, the decision is Kai's
    v = verdict(ratio)
    out["verdict_1"] = v
    if live.any():
        vb = verdict(ratio_b)
        out["verdict_1b"] = vb
        if vb != v:
            v = f"KAI'S CALL ((1) {v}; (1b) {vb})"
    out["verdict"] = v
    return out


class Census:
    """The main line's hook: wraps ablation.play_token so that each
    decision is censused from the saved emulator state before the agent's
    token plays."""

    def __init__(self, run, vocab, out):
        self.run, self.out = run, out
        self.vocab_grid = [GRID.index(tuple(t)) for t in vocab]
        self.game, self.index, self.recs = 0, 0, []
        self.argmax_mismatches = 0     # the hook's prior against the token
        self.replay_mismatches = 0     # the chosen token's replayed digest

    def decision(self, env, token, stack):
        ale = env.unwrapped.ale
        actions = list(env.unwrapped._action_set)
        start = ale.cloneState()
        frame_part, strict_part, digests = token_census(ale, actions, start)
        seq = (sequence_census(ale, actions, start)
               if self.index % SEQ_EVERY == 0 else None)
        # branch-order independence (audit r1a L4): the agent's own grid
        # token, replayed after every other branch, gives its census digest
        chosen = self.vocab_grid[int(token)]
        a, f = GRID[chosen]
        again, _ = play_window(ale, start, [actions[a]] * f, WINDOW_TOKENS,
                               actions[0])
        self.replay_mismatches += int(again != digests[chosen])
        ale.restoreState(start)
        with torch.no_grad():
            prior, _ = self.run.searcher.network_manager.initial_inference(
                stack)
        prior = (prior.detach().cpu().numpy() if hasattr(prior, "detach")
                 else np.asarray(prior))
        prior = np.asarray(prior, dtype=np.float64).reshape(-1)
        self.argmax_mismatches += int(int(np.argmax(prior)) != int(token))
        rec = decision_record(self.game, self.index, token, self.vocab_grid,
                              frame_part, strict_part, prior, seq)
        self.recs.append(rec)
        self.out.write(json.dumps(rec) + "\n")
        self.out.flush()
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


def exact_eval_row(path):
    """The snapshot's deployed exact-eval row (probe_exhaustive_eval's
    -acting file next to this probe, or in the run tree's probes folder),
    or None. `path` is resolved by the caller."""
    run = path.parent.parent.name
    it = int(path.stem[-3:]) if path.stem[-3:].isdigit() else None
    dirs = [HERE] + ([path.parents[3] / "probes"]
                     if len(path.parents) > 3 else [])
    for d in dirs:
        f = d / f"probe_exhaustive_eval-{run}-acting.jsonl"
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if (r.get("iteration") == it and "max_noop" not in r
                    and r.get("env") == ENV_ID and "games" in r):
                return r
    return None


def same_game(played, ref):
    """A censused game against the exact eval's game of the same noop start
    ([return, base steps, terminated, decisions])."""
    return (len(ref) == len(played) == 4
            and float(ref[0]) == float(played[0])
            and int(ref[1]) == int(played[1])
            and bool(ref[2]) == bool(played[2])
            and int(ref[3]) == int(played[3]))


def validity(games, ref, censused, argmax_mismatches, replay_mismatches):
    """A3: (ok, reasons). `games` {noop: played tuple}; `ref` the exact-eval
    row or None (a smoke run without one: ok None, nothing compared);
    `censused` {noop: decisions censused} (checked against the game's
    decisions when the census was not cut short)."""
    reasons = []
    if ref is not None:
        for n, g in games.items():
            r = ref["games"].get(str(n))
            if r is None or not same_game(g, r):
                reasons.append(f"game {n} {list(g)} != exact eval {r}")
    if not MAX_DECISIONS:
        for n, g in games.items():
            if censused.get(n) != int(g[3]):
                reasons.append(f"game {n}: {censused.get(n)} decisions "
                               f"censused of {int(g[3])}")
    if argmax_mismatches:
        reasons.append(f"{argmax_mismatches} decisions whose prior argmax "
                       f"is not the agent's token")
    if replay_mismatches:
        reasons.append(f"{replay_mismatches} replays that differ from their "
                       f"census digest")
    if reasons:
        return False, reasons
    return (None if ref is None else True), []


def decisions_file(run_name, smoke):
    """The decisions file; a full census is never overwritten."""
    p = HERE / f"probe_action_census-{run_name}{'-smoke' if smoke else ''}" \
               f"-decisions.jsonl"
    if p.exists() and not smoke:
        raise SystemExit(f"{p} exists: a census is never overwritten (move "
                         f"it aside first)")
    return p


def main(argv):
    import probe_exhaustive_eval as pe    # sets ablation's env globals
    if pe.SEARCH_SIMS:
        raise SystemExit("the census plays the prior argmax: unset "
                         "PROBE_SEARCH_SIMS")
    path = pathlib.Path(argv[0]).resolve()
    vocab = pe.snapshot_vocab(path)
    check_snapshot(vocab, pe.ENV)
    ref = exact_eval_row(path)
    if ref is None and not SMOKE:
        raise SystemExit(f"no deployed exact-eval row for {path} (A3: a "
                         f"full census needs one to check its main line)")
    pbn = pe.snapshot_prefix_bn(path)
    run_name = path.parent.parent.name
    dec = decisions_file(run_name, SMOKE)
    with ablation.arm_config(pe.ARM), ablation._run_length_globals(90, 2000), \
            pe.snapshot_tokens(vocab), pe.snapshot_prefix_layout(pbn):
        run = ablation.Run(pe.ARM, 0, ablation.ARMS[pe.ARM])
        assert muzero.USE_SEARCH_ACTING is False       # greedy = prior argmax
        d = pe.load(run, path)
        pe.apply_acting_states(run, d)
        t0 = time.time()
        games, censused = {}, {}
        with open(dec, "w") as out:
            census = Census(run, vocab, out)
            with census.hooked():
                for n in range(1, GAMES + 1):
                    census.game, census.index = n, 0
                    games[n] = pe.play(run, n)
                    censused[n] = census.index
                    print(f"game {n}: return {games[n][0]:.0f}, "
                          f"{games[n][3]} decisions, {len(census.recs)} "
                          f"censused [{time.time() - t0:.0f}s]", flush=True)
        ok, reasons = validity(games, ref, censused,
                               census.argmax_mismatches,
                               census.replay_mismatches)
        rec = {"snapshot": str(path), "run": run_name,
               "iteration": (int(path.stem[-3:]) if path.stem[-3:].isdigit()
                             else None),
               "env": ENV_ID, "games": {n: list(g) for n, g in games.items()},
               "n_games": GAMES, "seq_every": SEQ_EVERY,
               "max_decisions": MAX_DECISIONS or None,
               "window_tokens": WINDOW_TOKENS, "window_seq": WINDOW_SEQ,
               "exact_mean": round(float(np.mean([g[0] for g in
                                                  games.values()])), 3),
               "exact_eval_mean": None if ref is None else ref["exact_mean"],
               "secs": round(time.time() - t0),
               **summarize(census.recs, len(vocab)),
               "main_line_ok": ok, "void_reasons": reasons}
        if ok is False:
            rec["verdict"] = "VOID (" + "; ".join(reasons)[:300] + ")"
        with open(HERE / f"probe_action_census-{run_name}"
                         f"{'-smoke' if SMOKE else ''}.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps({k: v for k, v in rec.items()
                          if k not in ("snapshot", "games")}), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
