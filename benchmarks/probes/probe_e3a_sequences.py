"""E3a: does grouping by latents make sense for sequences mined from the
agent's own play? (CURRICULUM "Queued experiment (pre-registered) -- E3",
its E3a part and amendments (A1)-(A8), Kai 2026-11-04.) Measurement only:
the emulator only scores, and nothing here trains a model (Kai's rule for
the E line).

SOURCE (A1): Test B2's T2 -- its iteration-42 snapshot (E0's and E1's;
deployed exact eval 1699.0) and its own-play log over its generation
(iterations 1-42, at raw-step resolution with token boundaries; located as
E1 locates it: probe_e1_equivalence.lineage).

ADVANTAGE (A2) of a logged decision: the life's discounted return from it
minus T2's value estimate on the decision's stack. The return is the
prep's value check's (distill.discounted_return_max): T2's raw per-base-
step rewards / REWARD_UNIT, clipped (sign kept) at T2's constant ceiling,
discounted by T2's DISCOUNT per base step (4 raw frames) from the
decision's first base step to the life's end, with no bootstrap. The value
is T2's prediction head on the decision's stack through the deployed
acting states, decoded to value units. Ranked over the whole log; the
HIGH half is the floor(n / 2) largest advantages, ties to the earlier
decision in log order.

CANDIDATES (A3): every contiguous run of 2, 3 or 4 decisions inside one
logged life whose FIRST decision is in the high half and whose decisions
all played their full frames (a life's last token can be cut short by its
end); its key is its token tuple ((action, frames), ...), its LENGTH the
sum of its frames, at most 64. A run of ONE action whose length equals a
token of T2's vocabulary (that action at that length) is dropped: it
already exists as a duration token. The 32 most frequent keys are the
candidates, ties by first occurrence in log order.

TRUTH (A4): the main line is the exact eval of the 30 noop starts, replayed
exactly as E0's census and E1 replay it (probe_exhaustive_eval.play, the
deployed acting states, the prior argmax), each game checked against its
deployed exact-eval row and E0's census. At each LIVE decision t (E0's
definition: its census grid partition has more than one class), the run T2
actually plays next -- its own next 2, 3 and 4 tokens, all inside t's life,
each played in full, the last not ending the game -- is a CANDIDATE PLAY
when its key is a candidate. For each candidate play, every candidate of
the same LENGTH L is played on the raw ALE from t's saved emulator state,
then NOOP to a common window of L + 60 frames (E0's branch play); two
candidates are EQUIVALENT at t when every raw RGB frame of the window is
identical (E0's digest). Each (t, L) is scored once. The branches run when
T2's next tokens are known (MAX_RUN decisions later, or at the life's end),
and the emulator is restored to the main line's state after them. Checked
at every candidate play (A4, VOID otherwise): the played candidate,
replayed after all the others, gives its digest again (branch-order
independence, E0's check), and the stack its branch reached -- the max of
each base step's last two raw frames (atari_wrappers.MaxAndSkipEnv),
preprocessed and pushed onto t's stack -- is the main line's own next
stack (the branch replays the main line's frames).

READS
  (1) (A5) the DUPLICATE-SPELLING RATE: the share of candidate plays for
      which another same-length candidate is equivalent at t. Descriptive:
      the equivalent share of all same-length candidate pairs scored; the
      classes per (t, L) against the candidates per (t, L); by length; the
      rate over the plays that have another same-length candidate.
  (2) (A6) PREDICTED latents: z_c = projector(g(...g(h(s_t), c_1)..., c_n),
      with_prediction=True): h on t's stack with the deployed acting states,
      T2's dynamics unrolled over candidate c's tokens, the online
      projection (E1's forward model); the cosine distance 1 - cos between
      two candidates. A pair is GROUPED when its distance is below tau_L
      (strictly); the predicted classes are the connected components of the
      grouped pairs (E1's). tau_L is the median L-length prediction error
      over T2's logged runs of total length L (every contiguous run of 2-4
      decisions inside a logged life, each decision played in full, of any
      start and any key; run sizes pooled): 1 - cos(the run's unrolled online
      projection, projector(h(the stack the run reached), False)) -- E1's
      one-step tau, by length. Scored as pairs against the truth, pooled
      over the scored (t, L): merge precision and recall, beside the trivial
      rules (all one class; all distinct), as E1.
  (3) (A7) REAL-OUTCOME latents (an upper bound): projector(h(the stack each
      candidate's branch reached, before the NOOP tail), False), the target
      projection the calibration measures against; grouped below the same
      tau_L, scored the same way. Descriptive: the median distance of
      equivalent against non-equivalent pairs (also for (2)).
  Reported, not decided on: (2) and (3) at tau_L x 0.5 and x 2, and without
  the components (each pair on its own distance); whether (3) passes (2)'s
  bars ("If (3) cannot separate outcomes, latent grouping is not worth
  pursuing").
DECISION (A8), on the unrounded reads:
  - "KAI_CALL_TOO_FEW": under 100 candidate plays to read;
  - "GROUP_BY_PREDICTED": (2) has merge precision >= 0.7 at recall >= 0.2
    and precision strictly above every trivial rule's (all distinct has no
    precision: no bar) -> E3 groups mined sequences by predicted latents;
  - "MINE_WITHOUT_GROUPING": else, (1) under 10% -> E3 mines without
    grouping (Kai's fallback);
  - "KAI_CALL": else (duplicates common, latents fail).
  A None precision (no predicted merge) or recall (no true merge) fails.
  VOID when the main line is not the census's (E1's rules: the census
  valid, every game the exact eval's game and the census's decision count,
  every scored decision found in the census with its token and the hook's
  prior argmax) or a played candidate fails A4's checks.

usage: python probe_e3a_sequences.py SNAPSHOT.pth   (T2 iteration 42)
Refuses before anything plays: no full census of the snapshot next to the
probe (E0's decisions file gives the live decisions), another snapshot's
census, a thread count other than the census's, a full run without the
exact eval's row (E1's preflight), a T2 spec whose clip, unit or discount
is not its prep's, a value decoder that is not T2's.
Writes, next to itself: probe_e3a_sequences-<run>-meta.jsonl BEFORE the
replay (the value terms, the advantage, the candidates, the mining counts,
tau_L and its calibration); probe_e3a_sequences-<run>-decisions.jsonl (one
line per scored decision: live, T2's runs, per candidate play the members,
the truth, both distance lists, the checks); probe_e3a_sequences-<run>.jsonl
(the summary: reads (1)-(3), the verdict, main_line_ok, void_reasons). A
full run never overwrites its meta or decisions file. Smoke knobs (any
non-default writes "-smoke" files): E3A_GAMES (default 30), E3A_MAX_DECISIONS
(score each game's first N decisions; the main line plays on), E3A_LOG_LIVES
(read only the log's first N lives). CPU only, 2 threads
(probe_exhaustive_eval's PROBE_THREADS default), as the census.
Imported as a module (the tests) it changes nothing process-wide: the env
settings and probe_exhaustive_eval's import happen only when it runs as a
script."""
import os
ENV_ID = "MsPacmanNoFrameskip-v4"
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # "" deletes the var on Windows
    os.environ["PROBE_ENV"] = ENV_ID
    os.environ["PROBE_ACTING_STATES"] = "1"     # the deployed policy
import collections
import contextlib
import hashlib
import json
import math
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
import torch.nn.functional as F
import muzero
import ablation
import scalar_categorical
import probe_action_census as pc
import probe_e1_equivalence as e1

BASE = muzero.TOKEN_BASE_FRAMES              # raw frames per base step (4)
RUN_SIZES = (2, 3, 4)                        # (A3): decisions per run
MAX_RUN = max(RUN_SIZES)
MAX_LENGTH = 64                              # (A3): frames per run, at most
N_CANDIDATES = 32                            # (A3): the most frequent keys
HORIZON = pc.HORIZON                         # (A4): NOOP frames after a run
# (A8): (2)'s bars (E1's low bars), (1)'s fallback bar, the plays to read
GROUP_P, GROUP_R = 0.7, 0.2
DUP_BAR = 0.10
MIN_PLAYS = 100
TAU_MULTS = {"0.5": 0.5, "2": 2.0}           # robustness, reported only
CALIB_BATCH = e1.CALIB_BATCH
GAMES = int(os.environ.get("E3A_GAMES") or 30)
MAX_DECISIONS = int(os.environ.get("E3A_MAX_DECISIONS") or 0)
LOG_LIVES = int(os.environ.get("E3A_LOG_LIVES") or 0)
SMOKE = GAMES != 30 or MAX_DECISIONS > 0 or LOG_LIVES > 0


def _key(tokens):
    """A run's key: its tokens as a tuple of (action, frames) int pairs."""
    return tuple((int(a), int(f)) for a, f in tokens)


def _length(key):
    return sum(f for _, f in key)


# --------------------------------------------------------------------------
# (A2) the advantage
# --------------------------------------------------------------------------
def clipped(r, unit, ceiling):
    """A raw reward in units, clipped at `ceiling` with its sign kept
    (muzero.grown_reward's per-base-step clip, as the prep's value check
    applies it)."""
    r = float(r) / float(unit)
    return min(r, ceiling) if r >= 0 else -min(-r, ceiling)


def decision_returns(traj, unit, ceiling, discount):
    """(A2) Each acted decision's discounted return to the life's end:
    decision i's base-step rewards are the raw tuple record i + 1 carries
    (REWARD_GROWTH's records), every base step clipped, discounted by
    `discount` per base step from decision i's first base step (the prep's
    value check, distill.discounted_return_max, per decision). Returns
    (returns, base steps played per decision)."""
    m = len(traj) - 1
    for i in range(m):
        if not isinstance(traj[i + 1][2], (tuple, list)):
            raise SystemExit("the own-play log's records carry no raw "
                             "per-base-step rewards (a REWARD_GROWTH log is "
                             "needed)")
    steps = [len(traj[i + 1][2]) for i in range(m)]
    flat = [clipped(r, unit, ceiling) for i in range(m)
            for r in traj[i + 1][2]]
    g = np.zeros(len(flat) + 1)
    for j in range(len(flat) - 1, -1, -1):
        g[j] = flat[j] + discount * g[j + 1]
    starts = np.cumsum([0] + steps[:-1]) if m else []
    return np.array([g[int(s)] for s in starts], dtype=np.float64), steps


def high_half(adv):
    """(A2) The HIGH half: the floor(n / 2) largest advantages, ties to the
    earlier decision in log order. Returns (their indices, the smallest
    advantage among them or None)."""
    adv = np.asarray(adv, dtype=np.float64)
    k = len(adv) // 2
    order = np.argsort(-adv, kind="stable")[:k]
    return ({int(i) for i in order},
            float(adv[order[-1]]) if k else None)


# --------------------------------------------------------------------------
# (A3) the candidates
# --------------------------------------------------------------------------
def logged_runs(slots, vocab, sizes=RUN_SIZES):
    """Every contiguous run of n decisions (n in `sizes`) of one life whose
    decisions are `slots`: (start, n, key, length in frames)."""
    toks = [_key([vocab[int(s)]])[0] for s in slots]
    for i in range(len(toks)):
        for n in sizes:
            if i + n <= len(toks):
                key = tuple(toks[i:i + n])
                yield i, n, key, _length(key)


def is_duration_token(key, vocab_set):
    """(A3) A run of ONE action whose length equals a token of the
    vocabulary (that action at that length): already a duration token."""
    return (len({a for a, _ in key}) == 1
            and (key[0][0], _length(key)) in vocab_set)


def mine_candidates(lives, vocab, high, top=N_CANDIDATES,
                    max_length=MAX_LENGTH):
    """(A3) The candidates from the log's lives ({"slots", "full",
    "offset"}: slot per decision, whether it played its full frames, its
    first decision's index in the log), in log order; `high` the log
    indices of the high-half decisions. Counts every run whose FIRST
    decision is high, whose decisions all played in full and whose length
    is at most `max_length`, minus the duration tokens; the `top` most
    frequent keys, ties by first occurrence. Returns (candidates [{"id",
    "key", "length", "n", "count"}], the dropped duration tokens {key:
    count}, counts of the runs looked at)."""
    vocab_set = set(_key(vocab))
    counts, dropped = {}, {}
    stats = collections.Counter()
    for life in lives:
        full = life["full"]
        for i, n, key, length in logged_runs(life["slots"], vocab):
            if life["offset"] + i not in high:
                continue
            stats["high_runs"] += 1
            if not all(full[i:i + n]):
                stats["cut_short"] += 1
            elif length > max_length:
                stats["over_length"] += 1
            elif is_duration_token(key, vocab_set):
                dropped[key] = dropped.get(key, 0) + 1
                stats["duration_tokens"] += 1
            else:
                counts[key] = counts.get(key, 0) + 1
                stats["counted"] += 1
    # dicts keep insertion order and sorted() is stable: ties stay in the
    # order of their first occurrence
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])[:top]
    cands = [{"id": j, "key": key, "length": _length(key), "n": len(key),
              "count": int(c)} for j, (key, c) in enumerate(ranked)]
    stats["keys"] = len(counts)
    return cands, dropped, dict(stats)


def by_length(cands):
    """{length: candidate ids of that length, in candidate order}."""
    out = {}
    for c in cands:
        out.setdefault(int(c["length"]), []).append(int(c["id"]))
    return out


# --------------------------------------------------------------------------
# (A6) the forward model and its calibration
# --------------------------------------------------------------------------
@torch.no_grad()
def life_latents(rep, traj, batch=CALIB_BATCH):
    """h of every stack of a life (its m acted decisions and the final
    record), batched."""
    return torch.cat([rep(e1._floats([r[0] for r in traj[i:i + batch]]))
                      for i in range(0, len(traj), batch)])


@torch.no_grad()
def latent_values(pred, lat, batch=CALIB_BATCH):
    """The prediction head's decoded values (value units) on latents."""
    if not len(lat):
        return np.zeros(0)
    return torch.cat([pred(lat[i:i + batch])[1].reshape(-1)
                      for i in range(0, len(lat), batch)]).double().numpy()


@torch.no_grad()
def life_run_errors(nets, lat, slots, sizes=RUN_SIZES, batch=CALIB_BATCH):
    """(A6) The calibration errors of one life: for every start i and run
    size n with i + n <= m (m = len(slots)), 1 - cos(proj(g(...g(lat[i],
    s_i)..., s_(i+n-1)), True), proj(lat[i + n], False)) -- train_step's
    multi-step consistency pairing. `lat`: h of the life's m + 1 stacks.
    Returns {(i, n): error}."""
    m = len(slots)
    out = {}
    if m < min(sizes):
        return out
    target = torch.cat([nets.proj(lat[i:i + batch], with_prediction=False)
                        for i in range(0, m + 1, batch)])
    s = torch.as_tensor([int(x) for x in slots], dtype=torch.long)
    z = lat
    for k in range(1, max(sizes) + 1):
        starts = m - k + 1                  # i + k <= m
        if starts <= 0:
            break
        zin, sl = z[:starts], s[k - 1:m]
        z = torch.cat([nets.dyn(zin[i:i + batch], sl[i:i + batch])[0]
                       for i in range(0, starts, batch)])
        if k in sizes:
            online = torch.cat([nets.proj(z[i:i + batch],
                                          with_prediction=True)
                                for i in range(0, starts, batch)])
            err = 1.0 - F.cosine_similarity(online, target[k:k + starts],
                                             dim=1)
            for i, e in enumerate(err.double().numpy()):
                out[(i, k)] = float(e)
    return out


def calibrate_lengths(lengths, sizes, errors):
    """(A6) tau_L per total length L: the median error over the logged
    runs of that length (run sizes pooled), with the calibration read
    (E1's) and the read per run size. {L: {"tau", ..., "by_size"}}."""
    lengths = np.asarray(lengths, dtype=np.int64)
    sizes = np.asarray(sizes, dtype=np.int64)
    errors = np.asarray(errors, dtype=np.float64)
    out = {}
    for L in sorted(set(lengths.tolist())):
        sel = lengths == L
        out[int(L)] = {"tau": float(np.median(errors[sel])),
                       **e1.calibration_read(errors[sel]),
                       "by_size": {str(n): e1.calibration_read(
                           errors[sel & (sizes == n)])
                           for n in sorted(set(sizes[sel].tolist()))}}
    return out


@torch.no_grad()
def unrolled_projections(nets, stack, seqs):
    """(A6) For each slot sequence c of `seqs`, proj(g(...g(h(stack),
    c_1)..., c_n), True): (len(seqs), d), in order (sequences of one size
    unrolled as one batch)."""
    lat = nets.rep(e1._floats(stack[None]))
    rows = [None] * len(seqs)
    by_n = {}
    for j, q in enumerate(seqs):
        by_n.setdefault(len(q), []).append(j)
    for n, js in sorted(by_n.items()):
        z = lat.repeat(len(js), 1, 1, 1)
        s = torch.as_tensor([[int(x) for x in seqs[j]] for j in js],
                            dtype=torch.long)
        for k in range(n):
            z = nets.dyn(z, s[:, k])[0]
        p = nets.proj(z, with_prediction=True)
        for r, j in enumerate(js):
            rows[j] = p[r]
    return torch.stack(rows)


@torch.no_grad()
def stack_projections(nets, stacks):
    """(A7) proj(h(stack), False) for each stack: (len(stacks), d)."""
    return nets.proj(nets.rep(e1._floats(np.stack(stacks))),
                     with_prediction=False)


def pair_distances(z):
    """1 - cos over every pair of rows (e1.all_pairs order), as floats."""
    d = e1.cos_distances(z)
    return [float(d[a, b]) for a, b in e1.all_pairs(len(z))]


# --------------------------------------------------------------------------
# (A4) the truth: branches on the raw ALE
# --------------------------------------------------------------------------
def token_inputs(key, actions):
    """A run's raw inputs: each token's ALE action once per frame."""
    return [actions[int(a)] for a, f in key for _ in range(int(f))]


def play_branch(ale, start, inputs, window, noop, base=BASE):
    """E0's branch (probe_action_census.play_window: from `start`, `inputs`
    one ALE action per frame, then `noop` to `window` frames; the same
    digest of the window's raw RGB frames) together with, for each base
    step inside `inputs`, the observation the wrapper stack returns there:
    the element-wise max of its last two raw frames
    (atari_wrappers.MaxAndSkipEnv). Returns (digest, observations)."""
    if base < 2 or len(inputs) % base or window < len(inputs):
        raise ValueError(f"{len(inputs)} inputs, base {base}, window "
                         f"{window}")
    ale.restoreState(start)
    h = hashlib.blake2b(digest_size=16)
    obs, prev = [], None
    for t in range(window):
        ale.act(inputs[t] if t < len(inputs) else noop)
        screen = ale.getScreenRGB()
        h.update(screen)
        if t < len(inputs):
            k = t % base
            if k == base - 2:
                prev = np.array(screen, copy=True)
            elif k == base - 1:
                obs.append(np.maximum(prev, screen))
    return h.digest(), obs


def reached_stack(stack, observations):
    """The network input after `observations` (raw wrapper observations,
    one per base step) are played from `stack`, as play_episode and
    play_token push them."""
    s = stack
    for o in observations:
        s = ablation.push_frame(s, ablation.preprocess_frame(
            o, ablation.IMAGE_RESHAPE))
    return s


# --------------------------------------------------------------------------
# the main line's hook
# --------------------------------------------------------------------------
class Prober:
    """Wraps ablation.play_token on the main line: saves each decision's
    emulator state before its token plays and records what it played; when
    T2's next tokens are known (MAX_RUN decisions later, or at the life's
    or game's end) it scores the decision's candidate plays (A4-A7) and
    restores the emulator to the main line's state."""

    def __init__(self, run, nets, vocab, truth, cands, taus, out):
        self.run, self.nets, self.out = run, nets, out
        self.vocab = _key(vocab)
        self.slot = {t: s for s, t in enumerate(self.vocab)}
        self.truth, self.taus = truth, {int(k): float(v)
                                        for k, v in taus.items()}
        self.cands = list(cands)
        self.cand_of = {_key(c["key"]): int(c["id"]) for c in self.cands}
        self.by_length = by_length(self.cands)
        self.game, self.index, self.scored, self.recs = 0, 0, 0, []
        self.pending = []           # the current life's unscored decisions
        self.token_mismatches = 0   # the replayed token against the census's
        self.argmax_mismatches = 0  # the hook's prior argmax against it
        self.missing = 0            # decisions the census has no line for
        self.replay_mismatches = 0  # (A4) a played candidate's replay digest
        self.stack_mismatches = 0   # (A4) its reached stack: the main line's

    def after(self, env, token, stack, state, out):
        """Records one played decision; scores what has become scorable."""
        token = int(token)
        ended = bool(out[2] or out[3])
        self.pending.append({
            "d": self.index, "token": token, "stack": stack, "state": state,
            # the next decision's stack, as play_episode builds it (the
            # final stack at a life's end)
            "nxt": ablation.push_frame(out[4], ablation.preprocess_frame(
                out[0], ablation.IMAGE_RESHAPE)),
            "full": int(out[5]) == self.vocab[token][1] // BASE,
            "ended": ended,
            # a real game over (or the time limit): EpisodicLife's
            # was_real_done after the step; a lost life alone is not
            "game_end": ended and bool(getattr(env, "was_real_done", True))})
        self.index += 1
        if ended:
            self.flush(env)
        while len(self.pending) >= MAX_RUN:
            self.process(env)

    def flush(self, env):
        """Scores every pending decision (the life or the game ended)."""
        while self.pending:
            self.process(env)

    def process(self, env):
        """Scores pending[0] -- its life's next decisions are pending[1:] --
        and drops it."""
        if not MAX_DECISIONS or self.pending[0]["d"] < MAX_DECISIONS:
            self.score(env, self.pending)
        self.pending = self.pending[1:]

    def runs(self, life):
        """T2's runs from life[0] (A4): [(n, key, length, eligible,
        candidate id or None)]. Eligible: inside the life, every token
        played in full, the last not ending the game."""
        out = []
        for n in RUN_SIZES:
            if n > len(life):
                break
            seq = life[:n]
            key = tuple(self.vocab[d["token"]] for d in seq)
            ok = (all(d["full"] for d in seq)
                  and not any(d["ended"] for d in seq[:-1])
                  and not seq[-1]["game_end"])
            out.append((n, key, _length(key), ok, self.cand_of.get(key)))
        return out

    def score(self, env, life):
        """One decision's record (life[0]; the rest of its life after it):
        the census checks (token, prior argmax, a missing line), LIVE, T2's
        runs, and at a live decision a group per candidate play."""
        dec = life[0]
        rec = {"game": self.game, "d": dec["d"], "token": dec["token"]}
        self.scored += 1
        prior, _ = self.run.searcher.network_manager.initial_inference(
            dec["stack"])
        prior = np.asarray(prior.detach().cpu().numpy()
                           if hasattr(prior, "detach") else prior,
                           dtype=np.float64).reshape(-1)
        self.argmax_mismatches += int(int(np.argmax(prior)) != dec["token"])
        truth = self.truth.get((self.game, dec["d"]))
        if truth is None:
            self.missing += 1
            rec["missing"] = True
        else:
            self.token_mismatches += int(int(truth[0]) != dec["token"])
            # LIVE: the census's 36-token partition has more than one class
            rec["live"] = len(set(truth[1])) > 1
            rec["runs"], rec["groups"] = [], []
            seen = set()
            for n, key, length, ok, cid in self.runs(life):
                rec["runs"].append({"n": n, "length": length,
                                    "eligible": ok, "candidate": cid})
                if rec["live"] and ok and cid is not None \
                        and length not in seen:
                    seen.add(length)              # each (t, L) once
                    rec["groups"].append(self.group(env, life, n, cid,
                                                    length))
        self.recs.append(rec)
        if self.out is not None:
            self.out.write(json.dumps(rec) + "\n")
            self.out.flush()

    def group(self, env, life, n, cid, length):
        """One candidate play (A4-A7): every candidate of its length from
        the decision's saved state, the truth, both latent distances, the
        replay and reached-stack checks; the emulator restored."""
        dec = life[0]
        ale = env.unwrapped.ale
        actions = list(env.unwrapped._action_set)
        members = self.by_length[length]
        p = members.index(cid)
        window = length + HORIZON
        now = ale.cloneState()
        digests, reached = [], []
        for c in members:
            dg, obs = play_branch(ale, dec["state"], token_inputs(
                self.cands[c]["key"], actions), window, actions[0])
            digests.append(dg)
            reached.append(reached_stack(dec["stack"], obs))
        # branch-order independence (E0's check): the played candidate,
        # replayed after every other branch, gives its digest again
        again, _ = play_branch(ale, dec["state"], token_inputs(
            self.cands[cid]["key"], actions), window, actions[0])
        ale.restoreState(now)
        replay_ok = again == digests[p]
        # the branch replays the main line: its reached stack is the main
        # line's own next stack after the run
        stack_ok = bool(np.array_equal(reached[p], life[n - 1]["nxt"]))
        self.replay_mismatches += int(not replay_ok)
        self.stack_mismatches += int(not stack_ok)
        truth = pc.partition(digests)
        seqs = [[self.slot[t] for t in _key(self.cands[c]["key"])]
                for c in members]
        return {"length": length, "n": n, "candidate": cid,
                "members": members, "played": p, "truth": truth,
                "duplicate": any(truth[j] == truth[p]
                                 for j in range(len(members)) if j != p),
                "tau": self.taus[length],
                "dist_pred": pair_distances(unrolled_projections(
                    self.nets, dec["stack"], seqs)),
                "dist_real": pair_distances(stack_projections(self.nets,
                                                              reached)),
                "replay_ok": bool(replay_ok), "stack_ok": stack_ok}

    @contextlib.contextmanager
    def hooked(self):
        orig = ablation.play_token

        def play_token(env, token, stack, inner=None):
            state = env.unwrapped.ale.cloneState()
            out = orig(env, token, stack, inner)
            self.after(env, token, stack, state, out)
            return out

        ablation.play_token = play_token
        try:
            yield
        finally:
            ablation.play_token = orig


# --------------------------------------------------------------------------
# the reads
# --------------------------------------------------------------------------
def scored_groups(recs):
    """The candidate plays of the live decisions."""
    return [g for r in recs if r.get("live") is True
            for g in r.get("groups", [])]


def _pairs_of(g):
    return e1.all_pairs(len(g["truth"]))


def read_1(groups):
    """(A5) The duplicate-spelling rate and its descriptive reads."""
    plays = len(groups)
    dups = sum(1 for g in groups if g["duplicate"])
    multi = [g for g in groups if len(g["members"]) > 1]
    pairs = eq = 0
    by_len = {}
    for g in groups:
        t = g["truth"]
        e = sum(1 for i, j in _pairs_of(g) if t[i] == t[j])
        pairs += len(_pairs_of(g))
        eq += e
        b = by_len.setdefault(str(g["length"]), {"plays": 0, "duplicates": 0,
                                                 "members": len(g["members"])})
        b["plays"] += 1
        b["duplicates"] += int(g["duplicate"])
    for b in by_len.values():
        b["rate"] = e1._round(b["duplicates"] / b["plays"])
    rate = dups / plays if plays else None
    # (A11) the DECISION rate: over the plays whose length holds another
    # candidate (a play with none cannot be a duplicate)
    rate_multi = (sum(1 for g in multi if g["duplicate"]) / len(multi)
                  if multi else None)
    return {"plays": plays, "duplicates": dups, "rate": e1._round(rate),
            "_rate": rate, "_rate_multi": rate_multi,
            "plays_with_another_candidate": len(multi),
            "rate_with_another_candidate": e1._round(rate_multi),
            "pairs": pairs, "equivalent_pairs": eq,
            "equivalent_pair_share": e1._round(eq / pairs if pairs else None),
            "classes_mean": (round(float(np.mean([len(set(g["truth"]))
                                                  for g in groups])), 4)
                             if groups else None),
            "candidates_mean": (round(float(np.mean([len(g["members"])
                                                     for g in groups])), 4)
                                if groups else None),
            "by_length": dict(sorted(by_len.items(), key=lambda kv:
                                     int(kv[0])))}


def latent_counts(groups, key, mult=1.0, closure=True):
    """Pooled pair counts (tp, fp, fn, tn) of a latent grouping against the
    truth over `groups`: a pair is grouped when its distance (`key`) is
    below tau_L x mult; with `closure` the predicted classes are the
    components of the grouped pairs (E1's), else each pair stands alone.
    Also the trivial rules' counts {"one", "distinct"}."""
    counts = np.zeros(4, dtype=np.int64)
    floors = {k: np.zeros(4, dtype=np.int64) for k in ("one", "distinct")}
    for g in groups:
        t, pairs = g["truth"], _pairs_of(g)
        if not pairs:
            continue
        tau = float(g["tau"]) * mult
        close = [float(x) < tau for x in g[key]]
        if len(close) != len(pairs):
            raise ValueError(f"{len(close)} distances for {len(pairs)} pairs")
        if closure:
            pred = e1.components(len(t), [pr for pr, c in zip(pairs, close)
                                          if c])
            counts += e1.pair_counts(t, pred)
        else:
            for (i, j), c in zip(pairs, close):
                same = t[i] == t[j]
                counts[0 if same and c else 1 if c else 2 if same else 3] += 1
        for k, triv in e1.trivial_partitions(len(t)).items():
            floors[k] += e1.pair_counts(t, triv)
    return counts, floors


def grouping_passes(precision, recall, trivial_precisions):
    """(A8) (2)'s bar: precision >= GROUP_P at recall >= GROUP_R
    (inclusive; None fails) and strictly above every trivial rule's
    precision (a rule with no precision -- all distinct -- sets none)."""
    return bool(e1.meets(precision, recall, GROUP_P, GROUP_R)
                and all(precision > tp for tp in trivial_precisions
                        if tp is not None))


def _prf_read(counts):
    p, r, f1 = e1.prf(*counts[:3])
    return {"tp": int(counts[0]), "fp": int(counts[1]), "fn": int(counts[2]),
            "precision": e1._round(p), "recall": e1._round(r),
            "f1": round(f1, 4), "_p": p, "_r": r}


def median_distances(groups, key):
    """The median distance of the equivalent and of the non-equivalent
    pairs (None when there are none)."""
    eq, ne = [], []
    for g in groups:
        t = g["truth"]
        for (i, j), x in zip(_pairs_of(g), g[key]):
            (eq if t[i] == t[j] else ne).append(float(x))
    return {"equivalent": (round(float(np.median(eq)), 6) if eq else None),
            "non_equivalent": (round(float(np.median(ne)), 6) if ne
                               else None),
            "pairs": {"equivalent": len(eq), "non_equivalent": len(ne)}}


def latent_read(groups, key):
    """(A6) / (A7) one latent grouping, pooled over the candidate plays:
    the decision read (components below tau_L), the trivial rules, the bar
    (A8); reported: the pairwise read, tau_L x 0.5 and x 2, the median
    distances."""
    counts, floors = latent_counts(groups, key)
    out = {"groups": sum(1 for g in groups if len(g["truth"]) > 1),
           **_prf_read(counts)}
    triv = {k: e1.prf(*c[:3]) for k, c in floors.items()}
    out["trivial"] = {k: {"precision": e1._round(v[0]),
                          "recall": e1._round(v[1]), "f1": round(v[2], 4)}
                      for k, v in triv.items()}
    out["_trivial_p"] = [v[0] for v in triv.values()]
    out["above_trivial"] = bool(out["_p"] is not None and all(
        out["_p"] > v for v in out["_trivial_p"] if v is not None))
    out["meets_bars"] = grouping_passes(out["_p"], out["_r"],
                                        out["_trivial_p"])
    out["pairwise"] = _prf_read(latent_counts(groups, key,
                                              closure=False)[0])
    out["robustness"] = {k: _prf_read(latent_counts(groups, key, m)[0])
                         for k, m in TAU_MULTS.items()}
    out["median_distance"] = median_distances(groups, key)
    return out


def verdict(plays, dup_rate, precision, recall, trivial_precisions):
    """(A8) on the unrounded reads."""
    if plays < MIN_PLAYS:
        return "KAI_CALL_TOO_FEW"
    if grouping_passes(precision, recall, trivial_precisions):
        return "GROUP_BY_PREDICTED"
    if dup_rate is not None and dup_rate < DUP_BAR:
        return "MINE_WITHOUT_GROUPING"
    return "KAI_CALL"


def summarize(recs):
    """Reads (1)-(3) and the verdict (main_line_ok aside) from the decision
    records (also from the decisions file read back)."""
    groups = scored_groups(recs)
    r1, r2, r3 = read_1(groups), latent_read(groups, "dist_pred"), \
        latent_read(groups, "dist_real")
    live = [r for r in recs if r.get("live") is True]
    runs = [x for r in live for x in r.get("runs", [])]
    return {"decisions": len(recs),
            "live_decisions": len(live),
            "runs_on_live": {
                str(n): {"eligible": sum(1 for x in runs if x["n"] == n
                                         and x["eligible"]),
                         "candidate_plays": sum(
                             1 for x in runs if x["n"] == n and x["eligible"]
                             and x["candidate"] is not None)}
                for n in RUN_SIZES},
            "checks": {"replay_ok": sum(1 for g in groups if g["replay_ok"]),
                       "stack_ok": sum(1 for g in groups if g["stack_ok"]),
                       "plays": len(groups)},
            "read_1": e1._public(r1), "read_2": e1._public(r2),
            "read_3": e1._public(r3),
            # "If (3) cannot separate outcomes, latent grouping is not worth
            # pursuing": reported, not decided on
            "read_3_meets_bars": r3["meets_bars"],
            # (A11) the plays with another same-length candidate and their
            # duplicate rate decide; the all-plays rate is descriptive
            "verdict": verdict(r1["plays_with_another_candidate"],
                               r1["_rate_multi"], r2["_p"], r2["_r"],
                               r2["_trivial_p"])}


# --------------------------------------------------------------------------
# validity, inputs, files
# --------------------------------------------------------------------------
def validity(games, ref, truth_games, scored, counts, census_ok,
             n_games=None):
    """(ok, reasons). E1's rules: the census valid; every game the exact
    eval's game of the same noop start (`ref`, the deployed exact-eval row;
    None in a smoke without one: ok None) and the census's decision count;
    the decisions scored (`scored` {noop: n}, at most MAX_DECISIONS each);
    no scored decision missing from the census, none whose token is not the
    census's or the hook's prior argmax; a full run plays `n_games`. And
    (A4)'s: no played candidate whose replay digest or reached stack
    differs. `counts` holds the Prober's five counters by name."""
    reasons = []
    if census_ok is not True:
        reasons.append(f"the census's main_line_ok is {census_ok!r}")
    if n_games is not None and len(games) < n_games:
        reasons.append(f"stopped after {len(games)} of {n_games} games")
    if ref is not None:
        for n, g in games.items():
            r = ref["games"].get(str(n))
            if r is None or not pc.same_game(g, r):
                reasons.append(f"game {n} {list(g)} != exact eval {r}")
    for n, g in games.items():
        if truth_games.get(n) != int(g[3]):
            reasons.append(f"game {n}: {int(g[3])} decisions played, the "
                           f"census has {truth_games.get(n)}")
        want = (int(g[3]) if not MAX_DECISIONS
                else min(MAX_DECISIONS, int(g[3])))
        if scored.get(n) != want:
            reasons.append(f"game {n}: {scored.get(n)} decisions scored of "
                           f"{want}")
    what = {"token_mismatches": "decisions whose token is not the census's",
            "argmax_mismatches": "decisions whose prior argmax is not the "
                                 "agent's token",
            "missing": "decisions the census has no line for",
            "replay_mismatches": "played candidates whose replay differs "
                                 "from their first digest",
            "stack_mismatches": "played candidates whose reached stack is "
                                "not the main line's"}
    for k, text in what.items():
        if counts.get(k):
            reasons.append(f"{counts[k]} {text}")
    if reasons:
        return False, reasons
    return (None if ref is None else True), []


def value_terms(spec, prep):
    """(A2) T2's reward unit, constant clip ceiling and discount per base
    step, from its snapshot's spec, which must equal its prep's vocab.json
    (the prep chose them: reward_unit, student_ceiling, student_discount).
    SystemExit otherwise."""
    def num(x):
        return (not isinstance(x, bool) and isinstance(x, (int, float))
                and math.isfinite(x))

    if not spec.get("REWARD_GROWTH"):
        raise SystemExit("T2's spec has no REWARD_GROWTH (no raw per-base-"
                         "step rewards to clip)")
    unit, c, d = (spec.get("REWARD_UNIT"), spec.get("_student_ceiling"),
                  spec.get("DISCOUNT"))
    if not (num(unit) and unit > 0):
        raise SystemExit(f"T2's REWARD_UNIT {unit!r} is not positive")
    if not (num(c) and c > 0 and spec.get("REWARD_GROWTH_START") == c
            and spec.get("REWARD_GROWTH_CEILING") == c):
        raise SystemExit(f"T2's ceiling is not a constant _student_ceiling "
                         f"({c!r})")
    if not (num(d) and 0 < d < 1):
        raise SystemExit(f"T2's DISCOUNT {d!r} is not in (0, 1)")
    theirs = (prep.get("reward_unit"), prep.get("student_ceiling"),
              prep.get("student_discount"))
    if theirs != (unit, c, d):
        raise SystemExit(f"T2's unit / ceiling / discount {(unit, c, d)} "
                         f"are not its prep's {theirs}")
    return float(unit), float(c), float(d)


def check_value_head(prediction, spec):
    """(A2) The probe's value decoder must be T2's: the h-transform over
    the support of T2's VALUE_CEILING (ablation.build_networks)."""
    if spec.get("_linear_support") or not getattr(prediction, "transform",
                                                  False):
        raise SystemExit("T2's value head decodes without the h-transform "
                         "(the probe's would not)")
    want = scalar_categorical.support_for(spec.get("VALUE_CEILING",
                                                   muzero.VALUE_CEILING))
    if abs(float(prediction.support_size) - want) > 1e-9:
        raise SystemExit(f"the value support {prediction.support_size} is "
                         f"not T2's {want}")


def acting_nets(run, d, apply_acting):
    """(A2, A6, A7) The deployed nets: `apply_acting`
    (probe_exhaustive_eval.apply_acting_states, which the main line needs
    anyway) gives the representation and prediction the snapshot's acting
    states; the dynamics and projector stay the trainer's (refused if an
    acting state touched them); all in eval mode. Returns (Nets,
    prediction)."""
    extra = set(d.get("acting_states") or {}) - {"representation",
                                                  "prediction"}
    if extra:
        raise SystemExit(f"acting states for {sorted(extra)}: the dynamics "
                         f"/ projector would not be the trainer's")
    apply_acting(run, d)
    nets = e1.Nets(run.trainer.representation, run.trainer.dynamics,
                   run.trainer.projector)
    pred = run.trainer.prediction
    if any(n.training for n in (*nets, pred)):
        raise SystemExit("the probe's nets must be in eval mode")
    return nets, pred


def own_play_pass(lives_iter, vocab, nets, pred, unit, ceiling, discount,
                  batch=CALIB_BATCH):
    """T2's log, one pass, one life at a time: per life its slots, full
    flags and log offset (A3); every decision's return, value and advantage
    (A2); every logged run's (length, size, error) for tau_L (A6), runs of
    full decisions at most MAX_LENGTH frames long. Returns (lives,
    advantages, returns, values, calibration (lengths, sizes, errors),
    counts)."""
    vocab = _key(vocab)
    lives, rets, vals = [], [], []
    lengths, sizes, errors = [], [], []
    counts = collections.Counter()
    for it, life in lives_iter:
        traj = life["traj"]
        m = len(traj) - 1
        counts["lives"] += 1
        counts["truncated_lives"] += int(not life.get("terminated", True))
        if m < 1:
            continue
        slots = [int(r[1]) for r in traj[:-1]]
        frames = [vocab[s][1] for s in slots]
        g, steps = decision_returns(traj, unit, ceiling, discount)
        full = [steps[i] == frames[i] // BASE for i in range(m)]
        lat = life_latents(nets.rep, traj, batch)
        v = latent_values(pred, lat[:m], batch)
        for (i, n), e in sorted(life_run_errors(nets, lat, slots,
                                                batch=batch).items()):
            length = sum(frames[i:i + n])
            if all(full[i:i + n]) and length <= MAX_LENGTH:
                lengths.append(length)
                sizes.append(n)
                errors.append(e)
        lives.append({"iteration": int(it), "slots": slots, "full": full,
                      "offset": len(rets)})
        rets.extend(g.tolist())
        vals.extend(v.tolist())
        counts["decisions"] += m
        counts["cut_short_decisions"] += sum(1 for f in full if not f)
    rets, vals = np.asarray(rets), np.asarray(vals)
    return (lives, rets - vals, rets, vals, (lengths, sizes, errors),
            dict(counts))


def _dist(x):
    x = np.asarray(x, dtype=np.float64)
    if not len(x):
        return None
    q = np.percentile(x, [10, 50, 90])
    return {"mean": round(float(x.mean()), 4), "p10": round(float(q[0]), 4),
            "p50": round(float(q[1]), 4), "p90": round(float(q[2]), 4)}


def check_knobs(games, max_decisions, log_lives):
    """The env knobs' ranges (refusals before anything loads)."""
    if games < 1 or games > 30 or max_decisions < 0 or log_lives < 0:
        raise SystemExit(f"E3A_GAMES {games} must be 1..30, "
                         f"E3A_MAX_DECISIONS {max_decisions} and "
                         f"E3A_LOG_LIVES {log_lives} >= 0")


def out_files(run_name, smoke):
    """(summary, decisions, meta); a full run's decisions and meta are never
    overwritten (the summary is appended to, as E0's and E1's)."""
    tag = f"probe_e3a_sequences-{run_name}{'-smoke' if smoke else ''}"
    dec = HERE / f"{tag}-decisions.jsonl"
    meta = HERE / f"{tag}-meta.jsonl"
    for p in (dec, meta):
        if p.exists() and not smoke:
            raise SystemExit(f"{p} exists: a full E3a is never overwritten "
                             f"(move it aside first)")
    return HERE / f"{tag}.jsonl", dec, meta


def _jsonable_cands(cands):
    return [{**c, "key": [list(t) for t in c["key"]]} for c in cands]


def main(argv):
    import probe_exhaustive_eval as pe    # sets ablation's env globals
    if pe.SEARCH_SIMS:
        raise SystemExit("E3a replays the prior argmax: unset "
                         "PROBE_SEARCH_SIMS")
    if not pe.ACTING:
        raise SystemExit("E3a replays the deployed policy: "
                         "PROBE_ACTING_STATES=1")
    check_knobs(GAMES, MAX_DECISIONS, LOG_LIVES)
    t_start = time.time()
    path = pathlib.Path(argv[0]).resolve()
    vocab = pe.snapshot_vocab(path)
    pc.check_snapshot(vocab, pe.ENV)
    run_name = path.parent.parent.name
    iteration = int(path.stem[-3:]) if path.stem[-3:].isdigit() else None
    summary_path, truth_path = e1.census_files(run_name)
    census = e1.census_row(summary_path, run_name, iteration)
    if census is None or not truth_path.exists():
        raise SystemExit(f"no full census of {run_name} @{iteration} next to "
                         f"the probe (E0's live decisions)")
    truth, truth_games = e1.read_truth(truth_path)
    ref = pc.exact_eval_row(path)
    snapshot_note = e1.preflight(path, census, truth, ref, SMOKE,
                                 torch.get_num_threads())
    t2 = e1.lineage(path, vocab, iteration)[0]          # (A1) T2's log
    prep = json.loads((path.parents[2] / e1.PREPS[1] / "vocab.json")
                      .read_text())
    pbn = pe.snapshot_prefix_bn(path)
    summary_file, dec_file, meta_file = out_files(run_name, SMOKE)
    knobs = {"games": GAMES, "max_decisions": MAX_DECISIONS or None,
             "log_lives": LOG_LIVES or None}
    secs = {}
    with ablation.arm_config(pe.ARM), ablation._run_length_globals(90, 2000), \
            pe.snapshot_tokens(vocab), pe.snapshot_prefix_layout(pbn):
        run = ablation.Run(pe.ARM, 0, ablation.ARMS[pe.ARM])
        assert muzero.USE_SEARCH_ACTING is False       # greedy = prior argmax
        d = pe.load(run, path)
        spec = d.get("spec") or {}
        unit, ceiling, discount = value_terms(spec, prep)
        check_value_head(run.trainer.prediction, spec)
        nets, pred = acting_nets(run, d, pe.apply_acting_states)
        del d
        # 1. own play: the advantage, the candidates, tau_L
        t0 = time.time()
        lives, adv, rets, vals, (lengths, sizes, errors), log_counts = \
            own_play_pass(e1.iter_own_play(t2["folder"], t2["last"],
                                           LOG_LIVES),
                          vocab, nets, pred, unit, ceiling, discount)
        if not len(adv):
            raise SystemExit("no logged decision in T2's own-play log")
        high, threshold = high_half(adv)
        cands, dropped, mining = mine_candidates(lives, vocab, high)
        if not cands:
            raise SystemExit("no candidate mined from T2's own-play log")
        calibration = calibrate_lengths(lengths, sizes, errors)
        missing_tau = sorted({c["length"] for c in cands} - set(calibration))
        if missing_tau:
            raise SystemExit(f"no logged run to calibrate tau_L for the "
                             f"lengths {missing_tau}")
        taus = {L: calibration[L]["tau"] for L in sorted(by_length(cands))}
        secs["own_play"] = round(time.time() - t0)
        shown = {k: round(v, 5) for k, v in taus.items()}
        print(f"T2 log: {log_counts.get('lives')} lives, "
              f"{log_counts.get('decisions')} decisions; {len(cands)} "
              f"candidates; tau_L by length {shown} "
              f"[{secs['own_play']}s]", flush=True)
        advantage = {"decisions": int(len(adv)), "high": len(high),
                     "high_threshold": threshold,
                     "advantage": _dist(adv), "returns": _dist(rets),
                     "values": _dist(vals),
                     "corr_return_value": (round(float(np.corrcoef(
                         rets, vals)[0, 1]), 4) if len(adv) > 2 else None)}
        value = {"reward_unit": unit, "ceiling": ceiling,
                 "discount_per_base_step": discount}
        cand_rec = _jsonable_cands(cands)
        drop_rec = [{"key": [list(t) for t in k], "count": v}
                    for k, v in sorted(dropped.items(),
                                       key=lambda kv: -kv[1])]
        # the own-play results before anything replays: they survive a
        # failed summary write (E1's N1)
        meta = {"snapshot": str(path), "run": run_name,
                "iteration": iteration, "smoke": SMOKE, "knobs": knobs,
                "log": {"folder": str(t2["folder"]), "last": t2["last"],
                        **log_counts},
                "value_terms": value, "advantage": advantage,
                "mining": mining, "dropped_duration_tokens": drop_rec,
                "candidates": cand_rec,
                "taus": {str(k): v for k, v in taus.items()},
                "calibration": {str(k): v for k, v in calibration.items()},
                "secs": dict(secs), "memory": e1._memory()}
        with open(meta_file, "w") as f:
            f.write(json.dumps(meta) + "\n")
        # 2. the main line, replayed; the candidate plays scored
        t0 = time.time()
        games, scored = {}, {}
        with open(dec_file, "w") as out:
            prober = Prober(run, nets, vocab, truth, cands, taus, out)
            with prober.hooked():
                for n in range(1, GAMES + 1):
                    prober.game, prober.index, prober.scored = n, 0, 0
                    games[n] = pe.play(run, n)
                    prober.flush(run.env)        # the game's last decisions
                    scored[n] = prober.scored
                    same = (None if ref is None else pc.same_game(
                        games[n], ref["games"].get(str(n), ())))
                    plays = len(scored_groups(prober.recs))
                    print(f"game {n}: return {games[n][0]:.0f}, "
                          f"{games[n][3]} decisions, {len(prober.recs)} "
                          f"scored, {plays} candidate plays "
                          f"[{time.time() - t0:.0f}s]"
                          + ("" if same is None else " MATCH" if same else
                             f" MISMATCH (exact eval "
                             f"{ref['games'].get(str(n))})"), flush=True)
                    if same is False and not SMOKE:
                        break                  # VOID whatever follows
        secs["replay"] = round(time.time() - t0)
        counts = {k: getattr(prober, k) for k in (
            "token_mismatches", "argmax_mismatches", "missing",
            "replay_mismatches", "stack_mismatches")}
        ok, reasons = validity(games, ref, truth_games, scored, counts,
                               census.get("main_line_ok"),
                               n_games=None if SMOKE else GAMES)
        reads = summarize(prober.recs)
        secs["total"] = round(time.time() - t_start)
        rec = {"snapshot": str(path), "run": run_name,
               "iteration": iteration, "env": ENV_ID, "smoke": SMOKE,
               "knobs": knobs,
               "games": {n: list(g) for n, g in games.items()},
               "exact_mean": round(float(np.mean([g[0] for g in
                                                  games.values()])), 3),
               "exact_eval_mean": None if ref is None else ref["exact_mean"],
               "census": {"file": truth_path.name,
                          "main_line_ok": census.get("main_line_ok"),
                          "decisions": len(truth),
                          "threads": census.get("threads"),
                          "snapshot_note": snapshot_note},
               "value_terms": value, "advantage": advantage,
               "mining": mining, "candidates": cand_rec,
               "taus": {str(k): v for k, v in taus.items()},
               "meta_file": meta_file.name,
               **reads, "counts": counts,
               "main_line_ok": ok, "void_reasons": reasons,
               "secs": secs, "threads": torch.get_num_threads(),
               "memory": e1._memory()}
        e1.finalize(rec, ok, reasons)
        with open(summary_file, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps({k: rec[k] for k in (
            "exact_mean", "read_1", "verdict", "main_line_ok",
            "void_reasons")}), flush=True)
        print("READS " + json.dumps({k: {x: rec[k][x] for x in (
            "precision", "recall", "trivial", "meets_bars")}
            for k in ("read_2", "read_3")}), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
