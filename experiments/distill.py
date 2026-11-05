"""Curriculum P3's distillation step (Kai, 2026-10-21; pre-registered in
CURRICULUM.md): the data side of a master -> student generation swap.

`prep` runs in ONE process, in the MASTER's world (its arm, its run length,
its tokens). A master that is itself a P3 student (generation 2, Kai
2026-10-27: DS2) runs as its snapshot's spec (master_spec: its student
vocabulary and constant ceiling, which student_arm applied), and its final
snapshot comes after its offline steps too (master_offline_steps):

  1. RECORD: the master snapshot plays fresh games with its training-time
     acting policy -- Run.play_episode(greedy=False): the prior at the
     master's final temperature (its training step under its own run
     length), through its acting states (the snapshot's BN-recalibrated
     representation / prediction) -- and every life is kept at RAW-STEP
     resolution: the decision records play_episode returns (stack, token,
     raw per-base-step rewards) plus every inner base frame play_token
     pushed (ablation.INNER_FRAMES), which the replay never stores. Whole
     games until >= BASE_STEPS base steps. check_life proves each
     decision's inner frames rebuild the next decision's stack. Test B
     (--from-log, Kai 2026-10-27): no recording -- the master's own play
     over its whole generation, logged by its trainer in the same form
     (ablation.OwnPlayLog; log_lives), is the set, at no env steps.
  2. VOCABULARY: student_vocabulary drops every token whose share of the
     recorded decisions is below DROP_SHARE, never a 4-frame one (Kai's
     10-17 rule, "drop the least used"; a dropped token must be expressible
     as raw 4-frame steps). The master's slot order is kept.
  2b. THE STUDENT'S CLIP (Test 2, Kai 2026-10-24: the clip loosens only at a
     swap): under REWARD_GROWTH, the student's constant ceiling is the
     smallest that removes HALF the share of reward units the master's
     ceiling (its spec's, at its final step: G0's sign clip, DS2's
     constant c1) removes on the recorded games (halved_ceiling); written
     to vocab.json as "student_ceiling", which the generation-1 and -2
     student arms apply (ablation.student_arm).
  3. RE-TOKENIZE (retokenize): a kept token stays one record; a dropped
     token (a, f) becomes its m played base steps as m records of (a, 4),
     each on the stack at that base step and carrying its own raw reward.
  4. TARGETS: the master's birth search (Run._birth_search: the arm's own
     target recipe -- fresh root, 50 sims, Dirichlet noise, RAW weights, a
     greedy search on a truncated tail) over its tokens on every student
     decision state; visit fractions renormalized over the student's tokens
     (masked_targets). The student's value is never distilled.
  5. WRITE <out>/set.pkl (header + one pickle frame per life) and
     <out>/vocab.json, each atomically; set.pkl last but one, vocab.json
     last, so vocab.json's existence means the prep finished.

E3 (Kai 2026-11-05; CURRICULUM "E3", amendments E3-A2..A9), --mine: after
the 1% rule, tokens MINED from the master's own play (mine: runs of 2-4
decisions whose first is in the high-advantage half, keyed by raw inputs,
scored count x (k - 1) x effect consistency, prefix-free, no grouping) are
APPENDED to the student's vocabulary, at most 8 and at most the slots the
rule freed (E3-A3, E3-A4); re-tokenization RELABELS the master's stretches
that play a mined token's inputs as one record of it (E3-A6); every
record's target gives each mined token the master's path mass q(t1) x
prod pi(tj | ...) moved from t1's slot (E3-A5). A master holding mined
tokens (the second swap) treats them as tokens (E3-A8): kept, or dropped
and replayed as their base steps, each with its own action. Without
--mine and without a master's mined tokens every step above is unchanged,
bit for bit.

The student run (ablation.py --distill-set) reads set.pkl with read_set and
pre-fills its replay through Run.paper_targets (ablation.distill_prefill).

Usage:
  python experiments/distill.py --master SNAPSHOT.pth --master-iterations N
      --out DIR [--seed 0] [--base-steps 31250] [--from-log DIR] [--mine]
"""
import argparse
import contextlib
import json
import math
import os
import pathlib
import pickle
import sys
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import muzero  # noqa: E402

# ablation is imported where the prep needs it, never at module level: the
# trainer runs ablation.py as __main__, and its read_set / read_header must
# not execute a second copy of it (audit r1 A-NIT2).

ENV = "MsPacmanNoFrameskip-v4"
FRAMESKIP = 4
EPISODE_CAP = 3000          # per life, base steps (the TK recipe)
TRAIN_STEPS = 2000          # per iteration (the TK recipe)
BASE_STEPS = 31_250         # the replay cap, in base steps
DROP_SHARE = 0.01           # tokens below 1% of the recorded decisions
SET_MAGIC = b"DISTILLSET1\n"
FORMAT = 1


# -- 1. recording -------------------------------------------------------------
def load_master(run, blob):
    """The master's nets from its snapshot blob into `run` (all five, strict),
    its latent renormalisation flag and training step; returns the RAW
    representation / prediction states (the acting states are overlaid for
    recording and these restored for the search)."""
    import ablation
    for n in ablation.NET_NAMES:
        getattr(run.trainer, n).load_state_dict(blob[n])
    renorm = blob["spec"].get("LATENT_RENORMALIZE", True)
    run.trainer.representation.renormalize_output = renorm
    run.trainer.dynamics.renormalize_output = renorm
    run.trainer.training_steps = int(blob["training_steps"])
    run.searcher.network_manager.training_steps = run.trainer.training_steps
    run.set_train_mode(False)
    return {n: {k: v.detach().clone() for k, v in
                getattr(run.trainer, n).state_dict().items()}
            for n in ("representation", "prediction")}


def set_rep_pred(run, states):
    for n, sd in states.items():
        getattr(run.trainer, n).load_state_dict(sd)
    run.set_train_mode(False)


def check_life(traj, inner, frames=4):
    """Raw-step consistency of one recorded life: one inner-frame list per
    acted decision, len(inner[i]) == base steps played - 1 (the raw reward
    tuple the next record carries), and decision i's stack with its inner
    frames pushed equals decision i+1's stack minus its newest frame (a
    stack holds `frames` frames: muzero.initial_frame_stack's default).
    Raises AssertionError otherwise."""
    channels = traj[0][0].shape[0] // frames
    assert len(inner) == len(traj) - 1, (len(inner), len(traj))
    for i in range(len(traj) - 1):
        m = len(traj[i + 1][2])
        assert m >= 1 and len(inner[i]) == m - 1, (i, m, len(inner[i]))
        s = traj[i][0]
        for f in inner[i]:
            s = muzero.push_frame(s, f)
        assert np.array_equal(s[channels:], traj[i + 1][0][:-channels]), i


def record(run, base_steps):
    """The master plays whole games (lives through Run.play_episode, sampled
    acting) until >= base_steps base steps; returns (lives, game returns,
    base steps). Each life: {"traj": 3-field decision records, "inner":
    per-decision inner frames, "terminated", "game", "steps"}."""
    import ablation
    lives, games, total, g_ret = [], [], 0, 0.0
    while True:
        ablation.INNER_FRAMES = []
        try:
            traj, ret, steps, terminated = run.play_episode(greedy=False)
            inner = ablation.INNER_FRAMES
        finally:
            ablation.INNER_FRAMES = None
        check_life(traj, inner)
        real_done = bool(getattr(run.env, "was_real_done", True))
        lives.append({"traj": traj, "inner": inner,
                      "terminated": bool(terminated), "game": len(games),
                      "steps": int(steps),
                      # E3 (mining): a terminated life that ended the game
                      "game_end": bool(terminated) and real_done})
        total += steps
        g_ret += float(ret)
        if not terminated or real_done:
            games.append(g_ret)
            g_ret = 0.0
            if total >= base_steps:
                return lives, games, total


# -- 2. vocabulary --------------------------------------------------------------
def token_shares(lives, n_tokens):
    """Each master slot's share of the recorded decisions (acted records)."""
    counts = np.zeros(n_tokens, dtype=np.int64)
    for life in lives:
        for rec in life["traj"][:-1]:
            counts[int(rec[1])] += 1
    return counts / max(1, counts.sum()), counts


def student_vocabulary(master_vocab, shares, drop_share=DROP_SHARE):
    """Kai's rule: drop every token below drop_share of the decisions,
    except the TOKEN_BASE_FRAMES ones; keep the master's slot order.
    Returns (student vocab, dropped tokens), canonical (muzero.canon_token).
    E3-A8: a master's MINED token is a token like any other (never a base
    one: it lasts >= 2 base steps)."""
    base = muzero.TOKEN_BASE_FRAMES
    keep, dropped = [], []
    for tok, share in zip(master_vocab, shares):
        tok = muzero.canon_token(tok)
        is_base = (not muzero.token_is_mined(tok)) and tok[1] == base
        (keep if is_base or share >= drop_share else dropped).append(tok)
    return tuple(keep), tuple(dropped)


# -- 2b. the student's clip (Test 2, Kai 2026-10-24) ----------------------------
def reward_magnitudes(lives, unit):
    """Every recorded base step's raw reward magnitude in REWARD_UNITs (each
    base step once: the records' incoming raw-reward tuples)."""
    return np.asarray([abs(float(r)) / float(unit) for life in lives
                       for rec in life["traj"] for r in rec[2]],
                      dtype=np.float64)


def clip_share(mags, c):
    """S(c): the share of the reward units a per-base-step ceiling c removes
    (1 - sum(min(m, c)) / sum(m)); 0 when nothing was earned."""
    total = float(np.sum(mags))
    if total <= 0:
        return 0.0
    return float(1.0 - np.minimum(mags, c).sum() / total)


def halved_ceiling(mags, c0):
    """The smallest ceiling c >= c0 with S(c) <= S(c0) / 2 (Kai's "decrease
    the amount clipped", halved at each swap). sum(min(m, c)) is continuous,
    piecewise linear and non-decreasing in c, so the threshold is solved
    exactly on the sorted magnitudes; c0 itself when it clips nothing."""
    m = np.sort(np.asarray(mags, dtype=np.float64))
    m = m[m > 0]
    if not len(m):
        return float(c0)
    prefix = np.concatenate([[0.0], np.cumsum(m)])
    total = float(prefix[-1])
    s0 = clip_share(m, c0)
    if total <= 0 or s0 <= 0:
        return float(c0)
    want = total * (1.0 - s0 / 2.0)          # the kept units to reach
    n = len(m)
    # kept(c) = prefix[j] + c * (n - j) for m[j-1] <= c <= m[j]; at the
    # breakpoint m[j] it is prefix[j + 1] + m[j] * (n - j - 1), which is
    # total (>= want) at j = n - 1, so the first segment reaching `want`
    # always exists, and c is clamped into it against rounding (audit r1
    # A-L1: a window test on c itself could miss both neighbours of a
    # duplicated magnitude and fall through to "clip nothing").
    at = prefix[1:] + m * (n - 1 - np.arange(n))
    j = int(np.argmax(at >= want))
    lo = m[j - 1] if j else 0.0
    c = min(max((want - prefix[j]) / (n - j), lo), m[j])
    return float(max(c, c0))


# -- 2c. Test B: the student's discount, the value support, the own-play log ------
# the value support's headroom over the largest discounted return seen
VALUE_HEADROOM = 2.0


def student_discount_rule(g):
    """Test B (Kai, 2026-10-27): the student's discount per base step -- the
    master's gap 1 - g halved, capped at BBF's final 0.997
    (ablation.STUDENT_DISCOUNT_CAP): 0.97, 0.985, 0.9925, 0.99625, 0.997."""
    import ablation
    return float(min(ablation.STUDENT_DISCOUNT_CAP, 1.0 - (1.0 - g) / 2.0))


# Test B2 (Kai, 2026-10-29): the swap's discount rule, named by the master's
# arm: Test B's "halve" (its arms carry no _discount_rule) or Test B2's
# "keep" (the student keeps the master's discount).
DISCOUNT_RULES = ("halve", "keep")


def discount_rule(spec):
    """The swap's discount rule in the master's spec (`_discount_rule`,
    default "halve"); SystemExit on any other value."""
    rule = spec.get("_discount_rule", "halve")
    if rule not in DISCOUNT_RULES:
        raise SystemExit(f"_discount_rule {rule!r} is not one of "
                         f"{DISCOUNT_RULES}")
    return rule


def swap_discount(g, rule):
    """The student's discount per base step under `rule`: the master's `g`
    ("keep") or student_discount_rule(g) ("halve"); SystemExit on any other
    rule."""
    if rule == "keep":
        import ablation
        if float(g) > ablation.STUDENT_DISCOUNT_CAP:
            # (the student would refuse it: ablation.student_discount)
            raise SystemExit(f"a kept discount {g!r} past the cap "
                             f"{ablation.STUDENT_DISCOUNT_CAP}")
        return float(g)
    if rule == "halve":
        return student_discount_rule(g)
    raise SystemExit(f"_discount_rule {rule!r} is not one of "
                     f"{DISCOUNT_RULES}")


def discounted_return_max(lives, unit, ceiling, g):
    """The largest |discounted return| at any base step of any life, its
    raw rewards clipped per base step at `ceiling` REWARD_UNITs (sign
    kept, as grown_reward clips them) and discounted by `g` per base step
    to the life's end -- what a value target can reach in these games."""
    best = 0.0
    for life in lives:
        steps = [r for rec in life["traj"] for r in rec[2]]
        ret = 0.0
        for r in reversed(steps):
            r = float(r) / float(unit)
            r = min(r, ceiling) if r >= 0 else -min(-r, ceiling)
            ret = r + g * ret
            best = max(best, abs(ret))
    return float(best)


def log_lives(folder, last_iteration):
    """Test B: a master's own-play log (ablation.OwnPlayLog, iterations
    1..last_iteration) as record()'s lives -- each life checked
    (check_life), grouped into games as record() groups them (a game ends
    with a life that was truncated or ended it: the log's real_done). A
    game still running when the log ends is no game return. Returns (lives,
    game returns (raw), base steps, source header fields)."""
    import ablation
    entries = ablation.read_own_play(folder, last_iteration)
    lives, games, total, g_ret = [], [], 0, 0.0
    temps = {}
    for e in entries:
        check_life(e["traj"], e["inner"])
        lives.append({"traj": e["traj"], "inner": e["inner"],
                      "terminated": bool(e["terminated"]),
                      "game": len(games), "steps": int(e["steps"]),
                      # E3 (mining): a terminated life that ended the game
                      "game_end": bool(e["terminated"])
                      and bool(e["real_done"])})
        total += int(e["steps"])
        g_ret += sum(float(r) for rec in e["traj"] for r in rec[2])
        t = repr(float(e["temperature"]))
        temps[t] = temps.get(t, 0) + 1
        if not e["terminated"] or e["real_done"]:
            games.append(g_ret)
            g_ret = 0.0
    if not lives:
        raise SystemExit(f"own-play log {folder}: no lives")
    return lives, games, total, {
        "source": "own-play log", "log_dir": str(folder),
        "log_iterations": int(last_iteration), "log_lives": len(lives),
        "log_temperatures": temps,
        "log_partial_game_lives": sum(1 for l in lives
                                      if l["game"] == len(games))}


# -- 2d. E3: tokens mined from the master's own play (E3-A3, E3-A4) --------------
# Ported from benchmarks/probes/probe_e3a_sequences.py (E3a's A2, A3, A9, A10,
# A13), generalized to a master holding mined tokens (E3-A8).
MINE_RUN_SIZES = (2, 3, 4)          # decisions per run (E3a's A3)
MINE_BATCH = 128                    # stacks per representation forward
MINE_SCORED_SHOWN = 32              # the top-scored candidates vocab.json lists
# E3-A13: the tolerance of the disjointness check (float64 path products)
OVERLAP_TOL = 1e-9


def clipped(r, unit, ceiling):
    """A raw reward in units, clipped at `ceiling` with its sign kept
    (muzero.grown_reward's per-base-step clip)."""
    r = float(r) / float(unit)
    return min(r, ceiling) if r >= 0 else -min(-r, ceiling)


def decision_returns(traj, unit, ceiling, discount):
    """(E3a's A2, A9) Each acted decision's discounted return to the life's
    end: decision i's base-step rewards are the raw tuple record i + 1
    carries, every base step clipped, discounted by `discount` per base step
    from decision i's first base step (discounted_return_max's arithmetic);
    a life cut short is not bootstrapped. Returns (returns, base steps
    played per decision)."""
    m = len(traj) - 1
    for i in range(m):
        if not isinstance(traj[i + 1][2], (tuple, list)):
            raise SystemExit("the master's play carries no raw per-base-step "
                             "rewards (mining needs a REWARD_GROWTH log)")
    steps = [len(traj[i + 1][2]) for i in range(m)]
    flat = [clipped(r, unit, ceiling) for i in range(m)
            for r in traj[i + 1][2]]
    g = np.zeros(len(flat) + 1)
    for j in range(len(flat) - 1, -1, -1):
        g[j] = flat[j] + discount * g[j + 1]
    starts = np.cumsum([0] + steps[:-1]) if m else []
    return np.array([g[int(s)] for s in starts], dtype=np.float64), steps


def high_half(adv):
    """(E3a's A2, A9) The HIGH half: the floor(n / 2) largest advantages,
    ties to the earlier decision. Returns (their indices, the smallest
    advantage among them or None)."""
    adv = np.asarray(adv, dtype=np.float64)
    k = len(adv) // 2
    order = np.argsort(-adv, kind="stable")[:k]
    return ({int(i) for i in order},
            float(adv[order[-1]]) if k else None)


def run_inputs(tokens):
    """(E3a's A13) A run's KEY: the per-base-step inputs of its tokens,
    concatenated (muzero.token_inputs; a mined token's steps each with its
    own action)."""
    return tuple(x for t in tokens for x in muzero.token_inputs(t))


def run_frames(tokens):
    """A run's LENGTH: its tokens' total frames."""
    return sum(muzero.token_total_frames(t) for t in tokens)


def logged_runs(slots, vocab, sizes=MINE_RUN_SIZES):
    """Every contiguous run of n decisions (n in `sizes`) of one life whose
    decisions are `slots` (slots of `vocab`): (start, n, key = its token
    tuple, length in frames)."""
    toks = [vocab[int(s)] for s in slots]
    for i in range(len(toks)):
        for n in sizes:
            if i + n <= len(toks):
                key = tuple(toks[i:i + n])
                yield i, n, key, run_frames(key)


def spelling(spellings):
    """(E3a's A13) A key's spelling: the token tuple played most often among
    `spellings` {token tuple: count} (insertion order = first occurrence;
    max() keeps the first of equal counts)."""
    return max(spellings.items(), key=lambda kv: kv[1])[0]


def mine_candidates(lives, vocab, high, student_vocab,
                    max_length=muzero.TOKEN_MINED_MAX_FRAMES,
                    sizes=MINE_RUN_SIZES):
    """(E3-A3) The candidates from the master's lives ({"slots", "full",
    "offset"}: slot per decision, whether it played its full frames, its
    first decision's index in the log), in log order; `high` the log indices
    of the high-half decisions. Counts every run whose FIRST decision is
    high, whose decisions all played in full and whose length is at most
    `max_length`, by its raw INPUTS; dropped: a key equal in inputs to a
    token of `student_vocab` (the vocabulary after the 1% rule). Every key
    is a candidate (no count cut: the score ranks them, E3-A3), in count
    order, ties by first occurrence. Returns (candidates [{"id", "inputs",
    "key" (the spelling), "length", "n" (the spelling's decisions), "count",
    "spellings" [(token tuple, count)], "occurrences" [(life index, start,
    n)]}], the dropped keys {inputs: count}, counts of the runs looked
    at)."""
    import collections
    taken = {muzero.token_inputs(t) for t in student_vocab}
    counts, spells, dropped, occ = {}, {}, {}, {}
    stats = collections.Counter()
    for li, life in enumerate(lives):
        full = life["full"]
        for i, n, key, length in logged_runs(life["slots"], vocab, sizes):
            if life["offset"] + i not in high:
                continue
            stats["high_runs"] += 1
            inputs = run_inputs(key)
            if not all(full[i:i + n]):
                stats["cut_short"] += 1
            elif length > max_length:
                stats["over_length"] += 1
            elif inputs in taken:
                dropped[inputs] = dropped.get(inputs, 0) + 1
                stats["vocab_equal"] += 1
            else:
                counts[inputs] = counts.get(inputs, 0) + 1
                s = spells.setdefault(inputs, {})
                s[key] = s.get(key, 0) + 1
                occ.setdefault(inputs, []).append((li, i, n))
                stats["counted"] += 1
    # dicts keep insertion order and sorted() is stable: ties stay in the
    # order of their first occurrence
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    cands = []
    for j, (inputs, c) in enumerate(ranked):
        key = spelling(spells[inputs])
        cands.append({"id": j, "inputs": inputs, "key": key,
                      "length": muzero.TOKEN_BASE_FRAMES * len(inputs),
                      "n": len(key), "count": int(c),
                      "spellings": [(k, int(v)) for k, v in
                                    spells[inputs].items()],
                      "occurrences": occ[inputs]})
    stats["keys"] = len(counts)
    stats["merged_spellings"] = sum(len(s) - 1 for s in spells.values())
    return cands, dropped, dict(stats)


def effect_consistency(deltas):
    """(E3-A3) EFFECT CONSISTENCY of a candidate's occurrences: c = 1 - the
    mean cosine distance of each occurrence's latent change from the mean
    change (= the mean cosine similarity to it; a zero vector's cosine is 0).
    None without an occurrence."""
    if not len(deltas):
        return None
    d = np.asarray(deltas, dtype=np.float64).reshape(len(deltas), -1)
    mu = d.mean(axis=0)
    den = np.linalg.norm(d, axis=1) * np.linalg.norm(mu)
    cos = np.divide(d @ mu, den, out=np.zeros(len(d)), where=den > 0)
    return float(1.0 - np.mean(1.0 - cos))


def mined_score(count, k, c):
    """(E3-A3) SCORE = count x (decisions saved per use, k - 1) x effect
    consistency c (k: the decisions of the spelling that stands for the
    key). 0 without a consistency."""
    if c is None:
        return 0.0
    return float(count) * (int(k) - 1) * float(c)


def _is_prefix(a, b):
    return len(a) <= len(b) and tuple(b[:len(a)]) == tuple(a)


def select_mined(cands, cap):
    """(E3-A3, E3-A4) The appended tokens: candidates by score (highest
    first, ties in candidate order), a positive score only, PREFIX-FREE --
    a key whose inputs are a prefix of a higher-scored selected key's, or
    extend one, is skipped -- at most `cap`. Returns (selected candidates,
    skip counts)."""
    order = sorted(range(len(cands)), key=lambda j: -cands[j]["score"])
    chosen, skipped = [], {"prefix": 0, "score": 0}
    for j in order:
        c = cands[j]
        if len(chosen) >= cap:
            break
        if not c["score"] > 0:
            skipped["score"] += 1
            continue
        if any(_is_prefix(c["inputs"], s["inputs"])
               or _is_prefix(s["inputs"], c["inputs"]) for s in chosen):
            skipped["prefix"] += 1
            continue
        chosen.append(c)
    return chosen, skipped


def mined_token(spelled):
    """A candidate's token (E3-A2): its spelling's steps, flattened (a
    master's mined token contributes its own steps), canonical."""
    return muzero.canon_token(tuple(s for t in spelled
                                    for s in muzero.token_steps(t)))


def stack_latents_values(run, stacks, batch=MINE_BATCH, latents=True):
    """The master's representation of each stack (flattened, float32) and
    its decoded value, through run.trainer's representation / prediction
    as they stand (the caller sets the acting states), in eval mode;
    initial_inference's /255 input. `latents` False: (None, values) -- no
    latent kept."""
    rep, pred = run.trainer.representation, run.trainer.prediction
    if rep.training or pred.training:
        raise SystemExit("mining reads the master's nets in eval mode")
    dev = run.trainer.device
    hs, vs = [], []
    with torch.no_grad():
        for i in range(0, len(stacks), batch):
            x = torch.tensor(np.stack(stacks[i:i + batch]),
                             dtype=torch.float32, device=dev) / 255.0
            z = rep(x)
            vs.append(pred(z)[1].reshape(-1).double().cpu().numpy())
            if latents:
                hs.append(z.reshape(len(z), -1).float().cpu().numpy())
    return (np.concatenate(hs) if latents else None), np.concatenate(vs)


def mine_tokens(run, lives, master_vocab, student_vocab, cap, unit, ceiling,
                discount, batch=MINE_BATCH):
    """(E3-A3, E3-A4) Mining on the master's own play (`lives`, as record /
    log_lives give them), its nets carrying the ACTING states (the caller's
    set_rep_pred). The advantage of a decision: its discounted return (the
    master's clip `ceiling` in `unit`s, `discount` per base step) minus the
    master's value on its stack; the high half; the candidates; each one's
    effect consistency over its occurrences' latent changes h(stack after
    the run) - h(stack before) -- an occurrence ending at the final record
    of a life that ended the GAME is left out of c (MaxAndSkip stops before
    refreshing its buffer there, so that stack is stale; E3a's L1), its
    count kept; the score; the selection (at most `cap`; none computed at
    a cap of 0). Returns (the selected candidates, each with "token",
    "score", "consistency", "consistency_occurrences" and "path" (its
    spelling as master slots), the record vocab.json keeps).

    MEMORY: two passes over the log -- the values first (no latent kept),
    then the latents of only the stacks the counted runs start and end on
    (float32), the ones the consistency reads."""
    import collections
    if cap <= 0:
        # (audit r1b NIT) no slot freed: nothing to mine, nothing computed
        return [], {"cap": 0, "candidates": 0, "skipped": "no slot freed"}
    master_vocab = muzero.canon_vocab(master_vocab)
    slot_of = {t: i for i, t in enumerate(master_vocab)}
    base = muzero.TOKEN_BASE_FRAMES
    infos, rets, vals = [], [], []
    counts = collections.Counter()
    for li, life in enumerate(lives):
        traj = life["traj"]
        m = len(traj) - 1
        counts["lives"] += 1
        if m < 1:
            continue
        slots = [int(r[1]) for r in traj[:-1]]
        g, steps = decision_returns(traj, unit, ceiling, discount)
        full = [steps[i] == muzero.token_total_frames(master_vocab[s]) // base
                for i, s in enumerate(slots)]
        _, v = stack_latents_values(run, [r[0] for r in traj[:-1]], batch,
                                    latents=False)
        infos.append({"slots": slots, "full": full, "offset": len(rets),
                      "life": li, "game_end": bool(life.get("game_end"))})
        rets.extend(g.tolist())
        vals.extend(v.tolist())
        counts["decisions"] += m
        counts["truncated_lives"] += int(not life["terminated"])
    rets, vals = np.asarray(rets), np.asarray(vals)
    adv = rets - vals
    high, threshold = high_half(adv)
    cands, dropped, stats = mine_candidates(infos, master_vocab, high,
                                            student_vocab)
    # the second pass: the latents of the counted runs' start / end stacks
    need = {}
    for c in cands:
        for li, i, n in c["occurrences"]:
            need.setdefault(li, set()).update((i, i + n))
    lat = {}
    for li, idx in need.items():
        idx = sorted(idx)
        traj = lives[infos[li]["life"]]["traj"]
        h, _ = stack_latents_values(run, [traj[i][0] for i in idx], batch)
        lat.update({(li, i): h[r] for r, i in enumerate(idx)})
    counts["latents_kept"] = len(lat)
    stale = 0
    for c in cands:
        deltas = []
        for li, i, n in c["occurrences"]:
            life = infos[li]
            if life["game_end"] and i + n == len(life["slots"]):
                stale += 1
                continue
            deltas.append(lat[(li, i + n)] - lat[(li, i)])
        c["consistency"] = effect_consistency(deltas)
        c["consistency_occurrences"] = len(deltas)
        c["score"] = mined_score(c["count"], c["n"], c["consistency"])
    chosen, skipped = select_mined(cands, cap)
    for c in chosen:
        c["token"] = mined_token(c["key"])
        c["path"] = tuple(slot_of[t] for t in c["key"])

    def summary(x):
        if not len(x):
            return None
        q = np.percentile(x, [10, 50, 90])
        return {"mean": round(float(np.mean(x)), 4),
                "p10": round(float(q[0]), 4), "p50": round(float(q[1]), 4),
                "p90": round(float(q[2]), 4)}

    by_score = sorted(cands, key=lambda c: -c["score"])[:MINE_SCORED_SHOWN]
    record = {
        **dict(counts), **stats, "high": len(high),
        "high_threshold": threshold, "cap": int(cap),
        "candidates": len(cands), "skipped": skipped,
        "stale_occurrences": stale,
        "value_terms": {"reward_unit": float(unit), "ceiling": float(ceiling),
                        "discount_per_base_step": float(discount)},
        "advantage": summary(adv), "returns": summary(rets),
        "values": summary(vals),
        "vocab_equal_keys": len(dropped),
        "scored": [{"inputs": list(c["inputs"]),
                    "spelling": [muzero.token_json(t) for t in c["key"]],
                    "count": c["count"], "k": c["n"],
                    "consistency": c["consistency"], "score": c["score"]}
                   for c in by_score]}
    return chosen, record


def mined_entry(c, slot):
    """vocab.json's record of one mined token (E3-A4): the token, its
    student slot, spelling and master path, inputs, count, k, consistency,
    score, and every spelling played with its count."""
    return {"token": muzero.token_json(c["token"]), "slot": int(slot),
            "frames": muzero.token_total_frames(c["token"]),
            "inputs": [int(a) for a in c["inputs"]],
            "spelling": [muzero.token_json(t) for t in c["key"]],
            "path": [int(s) for s in c["path"]], "k": int(c["n"]),
            "count": int(c["count"]), "consistency": c["consistency"],
            "consistency_occurrences": int(c["consistency_occurrences"]),
            "score": float(c["score"]),
            "spellings": [{"spelling": [muzero.token_json(t) for t in k],
                           "count": int(v)} for k, v in c["spellings"]],
            # E3-A13: the master paths whose path products sum to P(m)
            "target_spellings": [{"path": [int(s) for s in p],
                                  "logged": int(n)}
                                 for p, n in c.get("target_spellings", ())]}


def target_spellings(lives, master_vocab, tokens):
    """(E3-A13) Every spelling of each mined token's inputs that is a path of
    master tokens: the master's own token with exactly those inputs (a
    one-token spelling: a dropped (a, 32), or a dropped master mined token
    re-mined -- its q counts directly), and every run of the master's log
    (any start, inside one life, every decision played in full) whose
    concatenated inputs are the token's. Returns, per token (in `tokens`'
    order), [(path of master slots, times logged)] -- the one-token spelling
    first, then the runs in first-occurrence order. Two spellings of the
    same inputs are never slot-prefixes of one another (the longer would
    add inputs), and the tokens' inputs are prefix-free (select_mined): the
    paths from any one slot are disjoint."""
    master_vocab = muzero.canon_vocab(master_vocab)
    base = muzero.TOKEN_BASE_FRAMES
    want = {muzero.token_inputs(t): j for j, t in enumerate(tokens)}
    out = [{} for _ in tokens]
    if not want:
        return []
    for s, t in enumerate(master_vocab):
        j = want.get(muzero.token_inputs(t))
        if j is not None:
            out[j].setdefault((s,), 0)
    max_steps = max(len(k) for k in want)
    for life in lives:
        traj = life["traj"]
        for i in range(len(traj) - 1):
            acc = ()
            for e in range(i, len(traj) - 1):
                tok = master_vocab[int(traj[e][1])]
                if len(traj[e + 1][2]) != muzero.token_total_frames(tok) // base:
                    break                               # not played in full
                acc = acc + muzero.token_inputs(tok)
                if len(acc) > max_steps:
                    break
                j = want.get(acc)
                if j is not None:
                    path = tuple(int(traj[x][1]) for x in range(i, e + 1))
                    out[j][path] = out[j].get(path, 0) + 1
    return [list(d.items()) for d in out]


# -- 3. re-tokenization -----------------------------------------------------------
def _relabel_match(traj, master_vocab, i, mined, max_steps):
    """(E3-A6) The longest mined token (`mined` {inputs: student slot})
    whose inputs equal the concatenated inputs of the master's decisions i,
    i + 1, ..., i + k - 1, every one played in full: (k, slot) or None."""
    base = muzero.TOKEN_BASE_FRAMES
    acc, best = (), None
    for j in range(i, len(traj) - 1):
        tok = master_vocab[int(traj[j][1])]
        if len(traj[j + 1][2]) != muzero.token_total_frames(tok) // base:
            break                                   # not played in full
        acc = acc + muzero.token_inputs(tok)
        if len(acc) > max_steps:
            break
        if acc in mined:
            best = (j - i + 1, mined[acc])
    return best


def retokenize(traj, inner, master_vocab, student_vocab, mined=None,
               stats=None):
    """One recorded life in the student's vocabulary: 3-field records
    (stack, student slot, raw reward tuple entering the record) and the
    final record (stack, 0, its incoming tuple). A kept token stays one
    record; a dropped token (a, f) that played m base steps becomes m
    records of (a, TOKEN_BASE_FRAMES): the first on the decision's stack
    with the decision's incoming rewards, the j-th on that stack with inner
    frames 1..j-1 pushed, entering with base step j-1's raw reward; the
    record after it enters with the last base step's. Every raw reward and
    base step is kept exactly once.

    E3-A8: a dropped MINED token is split the same way, each base step's
    record carrying that step's own action's base token. E3-A6, `mined`
    ({inputs: student slot} of the tokens mined at this swap; None: the
    rule above, exactly): at each of the master's decision boundaries,
    greedy left to right, the longest mined token whose inputs equal the
    next k decisions' concatenated inputs (all played in full) replaces
    them with ONE record -- the start stack, the mined slot, entering with
    the incoming rewards; the record after it enters with the k decisions'
    raw per-base-step rewards concatenated (the learner sums them, each
    base step clipped). The interior stacks leave the set. `stats` (a dict)
    counts the relabels: "records" {slot: n}, "decisions" (master
    decisions replaced) -- and "split", the dropped-token decisions split,
    "split_extra" the records their splits added."""
    student_vocab = muzero.canon_vocab(student_vocab)
    master_vocab = muzero.canon_vocab(master_vocab)
    index = {t: i for i, t in enumerate(student_vocab)}
    base = muzero.TOKEN_BASE_FRAMES
    max_steps = max((len(k) for k in mined), default=0) if mined else 0
    out, incoming = [], traj[0][2]
    i = 0
    while i < len(traj) - 1:
        if mined:
            match = _relabel_match(traj, master_vocab, i, mined, max_steps)
            if match is not None:
                k, slot = match
                out.append((traj[i][0], slot, incoming))
                incoming = tuple(r for j in range(i, i + k)
                                 for r in traj[j + 1][2])
                if stats is not None:
                    rec = stats.setdefault("records", {})
                    rec[slot] = rec.get(slot, 0) + 1
                    stats["decisions"] = stats.get("decisions", 0) + k
                i += k
                continue
        stack, slot = traj[i][0], int(traj[i][1])
        tok = master_vocab[slot]
        raw = traj[i + 1][2]
        if tok in index:
            out.append((stack, index[tok], incoming))
            incoming = raw
            i += 1
            continue
        inputs = muzero.token_inputs(tok)
        if stats is not None:
            stats["split"] = stats.get("split", 0) + 1
            stats["split_extra"] = stats.get("split_extra", 0) + len(raw) - 1
        s = stack
        out.append((s, index[(inputs[0], base)], incoming))
        for j in range(1, len(raw)):
            s = muzero.push_frame(s, inner[i][j - 1])
            out.append((s, index[(inputs[j], base)], (raw[j - 1],)))
        incoming = (raw[-1],)
        i += 1
    out.append((traj[-1][0], 0, incoming))
    return out


# -- 4. targets -------------------------------------------------------------------
def masked_targets(pols, master_vocab, student_vocab, mined=None,
                   weights=None):
    """Master visit fractions over its tokens -> the student's: the student
    tokens' entries, renormalized (visits to dropped tokens discarded).
    A row with no mass on any student token falls back to uniform; returns
    (targets, fallbacks, the mass discarded per row -- the search spreads
    visits over every child, so the read-out sharpens the targets by
    1 / (1 - discarded); audit r1 A's note).

    E3-A5 (the key rule) and E3-A13, `mined` (None: the rule above,
    exactly): for each token mined at this swap, in the order they close
    the student vocabulary, the master slot t1 each of its spellings starts
    from (target_spellings; a bare int: one spelling), and `weights[i]` the
    spellings' path products prod_{j>=2} pi(tj | the master's latent after
    t1..tj-1) at record i, flat in that order (mined_path_weights; 1 for a
    one-token spelling). At EVERY record each mined token m gets P(m) = the
    SUM over its spellings of q(t1) x the spelling's product, each moved
    from its own t1's slot -- the paths from one slot are disjoint (checked,
    ValueError otherwise: their products sum to at most 1 + OVERLAP_TOL and
    no slot goes below -OVERLAP_TOL x its q; the float residue is clamped);
    then the dropped tokens' mass is discarded
    and the rest renormalized, as above. A mined slot carries the master's
    own preference for the whole sequence everywhere, never a forced 0."""
    student_vocab = muzero.canon_vocab(student_vocab)
    master = list(muzero.canon_vocab(master_vocab))
    n_new = 0 if mined is None else len(mined)
    idx = [master.index(t)
           for t in student_vocab[:len(student_vocab) - n_new]]
    out, fallbacks, gone = [], 0, []
    if n_new == 0:
        for p in pols:
            full = np.asarray(p, dtype=np.float64)
            q = full[idx]
            s = q.sum()
            gone.append(float(full.sum() - s))
            if s > 0:
                out.append(q / s)
            else:
                out.append(np.ones(len(idx)) / len(idx))
                fallbacks += 1
        return out, fallbacks, gone
    groups = [[int(x)] if np.ndim(x) == 0 else [int(y) for y in x]
              for x in mined]
    if any(not g for g in groups):
        raise ValueError("a mined token without a spelling")
    firsts = np.asarray([s for g in groups for s in g], dtype=np.int64)
    owner = np.repeat(np.arange(n_new), [len(g) for g in groups])
    for i, p in enumerate(pols):
        full = np.asarray(p, dtype=np.float64)
        w = np.asarray(weights[i], dtype=np.float64).reshape(-1)
        if len(w) != len(firsts):
            raise ValueError(f"{len(w)} path products for {len(firsts)} "
                             f"spellings")
        per_slot = np.zeros(len(full))
        np.add.at(per_slot, firsts, w)
        moved = full[firsts] * w
        rest = full.copy()
        np.subtract.at(rest, firsts, moved)
        # the disjointness of the paths from one slot (E3-A13): their
        # products sum to at most 1, so no slot goes below 0 -- one
        # tolerance for both, OVERLAP_TOL (relative to the slot's q for the
        # second), past float64 residue; a real overlap raises (also at a
        # slot whose q is 0), only the residue is clamped
        if (per_slot.max() > 1.0 + OVERLAP_TOL
                or np.any(rest < -OVERLAP_TOL * full)):
            raise ValueError(f"overlapping mined paths at record {i}: path "
                             f"products sum to {per_slot.max()!r} at slot "
                             f"{int(per_slot.argmax())} (at most 1)")
        rest = np.maximum(rest, 0.0)
        mass = np.bincount(owner, weights=moved, minlength=n_new)
        q = np.concatenate([rest[idx], mass])
        s = q.sum()
        gone.append(float(full.sum() - s))
        if s > 0:
            out.append(q / s)
        else:
            out.append(np.ones(len(q)) / len(q))
            fallbacks += 1
    return out, fallbacks, gone


def mined_path_weights(nets, stacks, paths, device, batch=MINE_BATCH):
    """(E3-A5) For each stack and each mined token's master path (s1, ...,
    sk) (master slots: its spelling), prod_{j>=2} pi(s_j | g(...g(h(stack),
    s1)..., s_{j-1})) -- the master's PRIOR (the prediction head's softmax)
    on its own dynamics-unrolled latents, through `nets` (representation /
    dynamics / prediction: the nets its search ran, in eval mode for the
    call, restored after). [len(stacks), len(paths)] float64."""
    out = np.ones((len(stacks), len(paths)), dtype=np.float64)
    if not len(stacks) or not len(paths):
        return out
    rep, dyn, pred = (nets["representation"], nets["dynamics"],
                      nets["prediction"])
    modes = [n.training for n in (rep, dyn, pred)]
    for n in (rep, dyn, pred):
        n.eval()
    try:
        with torch.no_grad():
            for i0 in range(0, len(stacks), batch):
                x = torch.tensor(np.stack(stacks[i0:i0 + batch]),
                                 dtype=torch.float32, device=device) / 255.0
                z0 = rep(x)
                B = len(z0)
                for p, path in enumerate(paths):
                    z, w = z0, np.ones(B, dtype=np.float64)
                    for j in range(1, len(path)):
                        a = torch.full((B,), int(path[j - 1]),
                                       dtype=torch.long, device=device)
                        z = dyn(z, a)[0]
                        # float64 (audit r2): disjoint paths' products then
                        # sum to at most 1 within OVERLAP_TOL
                        prior = torch.softmax(pred(z)[0].double(), dim=1)
                        w = w * prior[:, int(path[j])].cpu().numpy()
                    out[i0:i0 + B, p] = w
    finally:
        for n, m in zip((rep, dyn, pred), modes):
            n.train(m)
    return out


def master_search(run, recs, terminated):
    """The master's birth search on every record's stack (the actions are
    irrelevant to a fresh-root search): (policies over the master's tokens,
    root values)."""
    dummy = [(r[0], 0, ()) for r in recs]
    searched = run._birth_search(dummy, terminated)
    return [r[3] for r in searched], [float(r[4]) for r in searched]


# -- 5. the set file ----------------------------------------------------------------
def write_set(path, header, lives):
    """header pickle, then one pickle frame per life ({"traj": 5-field
    records, "terminated"}); atomic (tmp + replace)."""
    path = pathlib.Path(path)
    # per-process tmp name: two preps can never interleave one file (audit
    # r1 B-M1; the supervisor's guard also counts a live prep)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb", buffering=16 * 1024 * 1024) as f:
            f.write(SET_MAGIC)
            pickle.dump({**header, "lives": len(lives)}, f, protocol=5)
            for life in lives:
                pickle.dump(life, f, protocol=5)
        os.replace(tmp, path)
    finally:
        if tmp.exists():                # a failed write leaves no ~1 GB tmp
            try:
                tmp.unlink()
            except OSError:             # never mask the write's own error
                pass


def read_header(path):
    with open(path, "rb") as f:
        if f.read(len(SET_MAGIC)) != SET_MAGIC:
            raise SystemExit(f"{path}: not a distillation set")
        header = pickle.load(f)
    if header.get("format") != FORMAT:
        raise SystemExit(f"{path}: distillation set format "
                         f"{header.get('format')!r}, expected {FORMAT}")
    return header


def read_set(path):
    """(header, lives) of a set written by write_set; refuses a truncated
    file (fewer life frames than the header counts)."""
    with open(path, "rb") as f:
        if f.read(len(SET_MAGIC)) != SET_MAGIC:
            raise SystemExit(f"{path}: not a distillation set")
        header = pickle.load(f)
        if header.get("format") != FORMAT:
            raise SystemExit(f"{path}: distillation set format "
                             f"{header.get('format')!r}, expected {FORMAT}")
        try:
            lives = [pickle.load(f) for _ in range(header["lives"])]
        except (EOFError, pickle.UnpicklingError):
            raise SystemExit(f"{path}: truncated (header counts "
                             f"{header['lives']} lives)")
        if f.read(1):
            raise SystemExit(f"{path}: data past the {header['lives']} "
                             f"lives its header counts")
    return header, lives


# -- the master's spec (generation 2: a student master) ---------------------------
# the keys ablation.student_arm sets on a P3 student arm (--token-vocab)
STUDENT_KEYS = ("TOKEN_VOCAB", "_token_vocab_sha256", "_vocab_set_sha256",
                "REWARD_GROWTH_START", "REWARD_GROWTH_CEILING",
                "_student_ceiling", "_master_vocab",
                # Test B: the prep's per-generation discount
                "DISCOUNT", "_student_discount",
                # E3: the declared mined tails (student_arm, E3-A8)
                "_mined_tokens", "_master_mined")


def master_spec(arm, blob, master):
    """The master's runtime spec. A master that is not a P3 student (G0)
    ran as its arm: ARMS[arm]. A STUDENT master (generation 2, Kai
    2026-10-27: DS2) ran its arm with --token-vocab applied
    (ablation.student_arm: the vocabulary and constant ceiling of its own
    prep), which only its snapshot's spec records. That spec is taken when
    every key but student_arm's (STUDENT_KEYS) equals the arm's, its
    vocabulary is a student vocabulary of the vocabulary its own master
    played -- the arm's, or, when that master was itself a student, its
    `_master_vocab`, required then and only then (student_master_vocab's
    rule; audit r1 A-L2) -- and its ceiling is the constant
    `_student_ceiling` of a `_ceiling_from_prep` arm."""
    import ablation
    static = ablation.ARMS[arm]
    if static.get("_student") is None:
        return static
    spec = blob.get("spec") or {}
    # the discount is a student key only where the prep sets it (Test B)
    excused = (STUDENT_KEYS if static.get("_discount_from_prep") else
               tuple(k for k in STUDENT_KEYS
                     if k not in ("DISCOUNT", "_student_discount")))
    bad = sorted(k for k in set(static) | set(spec) if k not in excused
                 and (k not in static or k not in spec
                      or spec[k] != static[k]))
    if bad:
        raise SystemExit(f"{master}: its spec differs from its arm's beyond "
                         f"the student keys: {bad}")
    if not static.get("_ceiling_from_prep"):
        raise SystemExit(f"{master}: a student master must hold its prep's "
                         f"constant ceiling (a _ceiling_from_prep arm)")
    parent = static.get("_master_arm")
    if parent is not None and ablation.ARMS.get(parent, {}).get("_student"):
        if "_master_vocab" not in spec:
            raise SystemExit(f"{master}: no _master_vocab, though its own "
                             f"master {parent!r} is a student")
        # E3-A8: its master's own mined tokens are that vocabulary's tail
        played = ablation.validate_student_vocab(
            spec["_master_vocab"], static["TOKEN_VOCAB"],
            mined=spec.get("_master_mined"))
    elif "_master_vocab" in spec or spec.get("_master_mined"):
        raise SystemExit(f"{master}: a _master_vocab, though its own master "
                         f"{parent!r} is no student")
    else:
        played = static["TOKEN_VOCAB"]
    # E3-A8: a master holding the mined tokens its prep appended
    mined = spec.get("_mined_tokens")
    ablation.validate_student_vocab(
        spec.get("TOKEN_VOCAB") or (), played, mined=mined,
        max_mined=muzero.TOKEN_MINED_PER_SWAP if mined else None)
    c = spec.get("_student_ceiling")
    if (isinstance(c, bool) or not isinstance(c, (int, float))
            or not math.isfinite(c) or c <= 0
            or spec.get("REWARD_GROWTH_START") != c
            or spec.get("REWARD_GROWTH_CEILING") != c):
        raise SystemExit(f"{master}: its ceiling is not a constant "
                         f"_student_ceiling ({c!r})")
    if static.get("_discount_from_prep"):
        d = spec.get("_student_discount")
        if (isinstance(d, bool) or not isinstance(d, (int, float))
                or not 0 < d < 1 or spec.get("DISCOUNT") != d):
            raise SystemExit(f"{master}: its DISCOUNT is not its prep's "
                             f"_student_discount ({d!r})")
    return dict(spec)


def master_offline_steps(spec):
    """A distilled student master's offline gradient steps, which precede
    its first iteration (generation 2's DS2 ends at 20,000 + 90 x 2,000);
    0 for any other master."""
    if spec.get("_student") != "distill":
        return 0
    return int(spec.get("_distill_offline_steps") or 0)


@contextlib.contextmanager
def installed(arm, spec):
    """ARMS[arm] = spec for the prep's recording and searches (arm_config
    and Run read it), restored after."""
    import ablation
    saved = ablation.ARMS[arm]
    ablation.ARMS[arm] = spec
    try:
        yield
    finally:
        ablation.ARMS[arm] = saved


def master_checks(master, master_iterations):
    """The prep's checks on its master snapshot, before anything is
    recorded: a known arm, its runtime spec (master_spec), its vocabulary,
    its acting states, its FINAL step (after its offline steps). Also the
    generation-2 preflight's, on the real DS2 (audit r1 A-L1 / B-L2: an
    edit to DS2's arm would otherwise surface only as three failed preps).
    Returns (blob, arm, spec, offline steps); SystemExit otherwise."""
    import ablation
    blob = torch.load(str(master), map_location="cpu", weights_only=False)
    arm = blob.get("arm")
    if arm not in ablation.ARMS:
        raise SystemExit(f"{master}: arm {arm!r} unknown")
    mspec = master_spec(arm, blob, master)
    # (E3: canonical tokens -- a master may hold mined ones)
    master_vocab = muzero.canon_vocab(blob["spec"].get("TOKEN_VOCAB") or ())
    if not master_vocab or master_vocab != muzero.canon_vocab(
            mspec["TOKEN_VOCAB"]):
        raise SystemExit(f"{master}: its TOKEN_VOCAB is not its arm's")
    if not blob.get("acting_states"):
        raise SystemExit(f"{master}: no acting_states (a snapshot of a "
                         f"BN_RECAL_ACTING run is required)")
    offline = master_offline_steps(mspec)
    if (int(blob["training_steps"])
            != offline + master_iterations * TRAIN_STEPS
            or blob.get("iteration") != master_iterations):
        raise SystemExit(f"{master}: training_steps {blob['training_steps']} "
                         f"is not the end of a {master_iterations}-iteration "
                         f"run (the master is its FINAL snapshot"
                         + (f", after {offline} offline steps" if offline
                            else "") + ")")
    return blob, arm, mspec, offline


# -- steps 2-4 on the master's lives --------------------------------------------------
def distill_lives(run, lives, master_vocab, raw_states, acting_states, clip,
                  discount, drop_share=DROP_SHARE, mine=False, log=print,
                  t0=None):
    """Steps 2-4 (module docstring) on the master's `lives` (record /
    log_lives), the prep's core: the 1% rule; with `mine` (E3), mining on
    the master's ACTING states (its deployed values and latents; `clip`'s
    reward unit and master_ceiling, `discount` per base step: the master's)
    and the appended tokens; the reward / value-prefix support check of the
    student's ceiling for a record of the vocabulary's longest token (E3-A9;
    whenever REWARD_GROWTH set a ceiling: a master's kept mined token counts
    too); then, on the master's RAW states (the nets its search runs),
    re-tokenization with relabeling (E3-A6), the master's search on every
    record and the targets (E3-A5 / A13: every spelling's path product on
    the re-tokenized records' stacks). `run` provides trainer.{representation,
    dynamics, prediction, device}, set_train_mode and _birth_search (Run's).
    Returns a dict: "vocab", "dropped", "shares", "counts", "chosen",
    "mining", "n_kept", "out_lives", "records", "splits", "split_extra",
    "net_extra", "fallbacks", "gone_mass", "rstats", "trule"."""
    import ablation
    t0 = time.time() if t0 is None else t0
    master_vocab = muzero.canon_vocab(master_vocab)
    shares, counts = token_shares(lives, len(master_vocab))
    vocab, dropped = student_vocabulary(master_vocab, shares, drop_share)
    log(f"student vocabulary: {len(vocab)} of {len(master_vocab)} "
        f"tokens; dropped {list(dropped)}")
    chosen, mining = [], None
    if mine:
        # E3-A3 / E3-A4: mined on the master's own play with its acting
        # states, appended after the kept tokens
        if "student_ceiling" not in clip:
            raise SystemExit("--mine needs REWARD_GROWTH (raw per-base-step "
                             "rewards, the master's clip)")
        set_rep_pred(run, acting_states)
        cap = min(muzero.TOKEN_MINED_PER_SWAP, len(dropped))
        chosen, mining = mine_tokens(
            run, lives, master_vocab, vocab, cap, clip["reward_unit"],
            clip["master_ceiling"], discount)
        vocab = vocab + tuple(c["token"] for c in chosen)
        ablation.validate_student_vocab(
            vocab, master_vocab, mined=tuple(c["token"] for c in chosen),
            max_mined=muzero.TOKEN_MINED_PER_SWAP)
        log(f"mined {len(chosen)} of {mining['candidates']} candidates (cap "
            f"{cap}): "
            + "; ".join(f"{muzero.token_key(c['token'])} (count "
                        f"{c['count']}, k {c['n']}, c "
                        f"{c['consistency']:.3f}, score {c['score']:.1f})"
                        for c in chosen)
            + f" [{time.time() - t0:.0f}s]")
    if "student_ceiling" in clip:
        # E3-A9 (audit r1b): always -- 1 base step for a duration-only
        # vocabulary (the check the prep always ran), the longest token's
        # once it holds a mined token (appended, or the master's kept)
        steps = ablation.support_steps(vocab)
        if not ablation.ceiling_fits_supports(clip["student_ceiling"], steps):
            raise SystemExit(
                f"the student's ceiling {clip['student_ceiling']!r} lies past "
                f"the reward supports for a record of {steps} base steps (its "
                f"longest token)")
    set_rep_pred(run, raw_states)
    n_kept = len(vocab) - len(chosen)
    # E3-A6: the mined tokens' inputs -> their student slots
    relabel = {muzero.token_inputs(c["token"]): n_kept + j
               for j, c in enumerate(chosen)} or None
    # E3-A13: every spelling of each mined token, its path's first slot
    spellings = target_spellings(lives, master_vocab,
                                 [c["token"] for c in chosen])
    for c, sp in zip(chosen, spellings):
        c["target_spellings"] = sp
    paths = [p for sp in spellings for p, _n in sp]
    groups = [[p[0] for p, _n in sp] for sp in spellings] or None
    firsts = np.asarray([p[0] for p in paths], dtype=np.int64)
    owner = np.repeat(np.arange(len(chosen)),
                      [len(sp) for sp in spellings]).astype(np.int64)
    nets = {k: getattr(run.trainer, k) for k in
            ("representation", "dynamics", "prediction")}
    rstats = {}
    M = len(chosen)
    trule = {"records": 0, "moved": np.zeros(M), "target": np.zeros(M),
             "nonzero": np.zeros(M), "labelled": np.zeros(M),
             "at_label": np.zeros(M), "moved_spelling": np.zeros(len(paths))}
    out_lives, records, net_extra, fallbacks = [], 0, 0, 0
    gone_mass = []
    for life in lives:
        recs = retokenize(life["traj"], life["inner"], master_vocab, vocab,
                          mined=relabel, stats=rstats)
        pols, vals = master_search(run, recs, life["terminated"])
        weights = None
        n_s = len(recs) - (1 if life["terminated"] else 0)
        if paths:
            # E3-A5 / A13: the master's prior along each spelling, on the
            # nets its search ran (the raw states), at the RE-TOKENIZED
            # records' stacks
            weights = mined_path_weights(nets, [r[0] for r in recs], paths,
                                         run.trainer.device)
            for i in range(n_s):
                moved = np.asarray(pols[i], dtype=np.float64)[firsts] \
                    * weights[i]
                trule["moved_spelling"] += moved
                trule["moved"] += np.bincount(owner, weights=moved,
                                              minlength=M)
        pols, fb, gone = masked_targets(pols, master_vocab, vocab,
                                        mined=groups, weights=weights)
        if paths:
            for i in range(n_s):
                t = np.asarray(pols[i])[n_kept:]
                trule["target"] += t
                trule["nonzero"] += t > 0
                s = int(recs[i][1]) - n_kept
                if s >= 0:
                    trule["labelled"][s] += 1
                    trule["at_label"][s] += t[s]
            trule["records"] += n_s
        fallbacks += fb
        # acted records only (a terminated tail's uniform is no search)
        gone_mass += gone[:-1] if life["terminated"] else gone
        out_lives.append({
            "traj": [(r[0], r[1], tuple(r[2]), p, v)
                     for r, p, v in zip(recs, pols, vals)],
            "terminated": life["terminated"]})
        records += len(recs)
        net_extra += len(recs) - len(life["traj"])
        life["inner"] = None          # free as we go
    splits, split_extra = rstats.get("split", 0), rstats.get("split_extra", 0)
    log(f"re-tokenized + master-searched {records} records "
        f"({splits} dropped-token decisions split into {split_extra} "
        f"extra records; {fallbacks} uniform fallbacks; mean visit "
        f"mass discarded on dropped tokens "
        f"{np.mean(gone_mass) if gone_mass else 0.0:.3f}"
        + (f"; {rstats.get('decisions', 0)} decisions relabeled as "
           f"{sum(rstats.get('records', {}).values())} mined records"
           if mine else "")
        + f") [{time.time() - t0:.0f}s]")
    return {"vocab": vocab, "dropped": dropped, "shares": shares,
            "counts": counts, "chosen": chosen, "mining": mining,
            "n_kept": n_kept, "out_lives": out_lives, "records": records,
            "splits": splits, "split_extra": split_extra,
            "net_extra": net_extra, "fallbacks": fallbacks,
            "gone_mass": gone_mass, "rstats": rstats, "trule": trule}


def mined_header(got):
    """(E3) The header keys of a mined prep (distill_lives' result `got`):
    "mine"; "mined" (E3-A4: the appended tokens, in the order they close the
    student vocabulary, their spellings, counts, scores, consistency and
    E3-A13's target spellings); "mining"; "relabeled" and
    "relabeled_decisions" (E3-A6); "net_extra_records" (the records the
    re-tokenization added net of the relabels; split_extra_records counts
    the splits' alone); "target_rule" (E3-A5 / A13, over the searched
    records: per mined token the mean mass moved to it before
    renormalizing, its mean target, the share of records where it is
    nonzero, its mean target at its own relabeled records, and each
    spelling's mean moved mass)."""
    chosen, n_kept = got["chosen"], got["n_kept"]
    trule, rstats = got["trule"], got["rstats"]
    n = max(1, trule["records"])
    keys = [muzero.token_key(c["token"]) for c in chosen]
    moved_sp, idx = [], 0
    for c in chosen:
        sp = []
        for p, logged in c["target_spellings"]:
            sp.append({"path": [int(s) for s in p], "logged": int(logged),
                       "moved_mean": round(float(
                           trule["moved_spelling"][idx] / n), 6)})
            idx += 1
        moved_sp.append(sp)
    return {
        "mine": True,
        "mined": [mined_entry(c, n_kept + j) for j, c in enumerate(chosen)],
        "mining": got["mining"],
        "relabeled": {keys[s - n_kept]: int(v) for s, v in sorted(
            rstats.get("records", {}).items())},
        "relabeled_decisions": int(rstats.get("decisions", 0)),
        "net_extra_records": int(got["net_extra"]),
        "target_rule": {
            "records": int(trule["records"]),
            "moved_mean": {k: round(float(v / n), 6) for k, v in
                           zip(keys, trule["moved"])},
            "target_mean": {k: round(float(v / n), 6) for k, v in
                            zip(keys, trule["target"])},
            "nonzero_share": {k: round(float(v / n), 6) for k, v in
                              zip(keys, trule["nonzero"])},
            "target_at_relabeled_mean": {
                k: (round(float(a / c), 6) if c else None) for k, a, c
                in zip(keys, trule["at_label"], trule["labelled"])},
            "spellings": dict(zip(keys, moved_sp))}}


# -- the prep ---------------------------------------------------------------------
def prep(master, master_iterations, out, seed=0, base_steps=BASE_STEPS,
         drop_share=DROP_SHARE, log=print, from_log=None, mine=False):
    """Steps 1-5 (module docstring) for one master snapshot; returns the
    header written. `from_log` (Test B): the master's own-play log folder
    (ablation.OwnPlayLog, iterations 1..master_iterations) replaces the
    recording of step 1 -- no env step is played (base_steps unused).
    `mine` (E3, --mine): tokens mined from the master's play are appended
    to the student's vocabulary, its stretches relabeled, every target
    under E3-A5 (module docstring); the header then carries "mine",
    "mined", "mining", "relabeled" and "target_rule". A master holding
    mined tokens adds "master_mined" (E3-A8)."""
    import ablation
    t0 = time.time()
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    blob, arm, mspec, offline = master_checks(master, master_iterations)
    master_vocab = muzero.canon_vocab(mspec["TOKEN_VOCAB"])
    ablation.ENV_NAME, ablation.FRAMESKIP, ablation.EPISODE_CAP = \
        ENV, FRAMESKIP, EPISODE_CAP
    with installed(arm, mspec), ablation.arm_config(arm) as spec, \
            ablation._run_length_globals(master_iterations, TRAIN_STEPS), \
            ablation.token_actions():
        run = ablation.Run(arm, seed, spec)
        try:
            raw_states = load_master(run, blob)
            temp = muzero.prior_acting_temperature(run.trainer.training_steps)
            bias = muzero.token_long_bias(run.trainer.training_steps)
            if from_log is None:
                set_rep_pred(run, blob["acting_states"])
                log(f"recording: master {pathlib.Path(master).name} ({arm}, "
                    f"{run.trainer.training_steps} steps), temperature "
                    f"{temp}, long-token bias {bias}, until >= {base_steps} "
                    f"base steps")
                lives, games, total = record(run, base_steps)
                source = {"source": "recording"}
            else:
                lives, games, total, source = log_lives(from_log,
                                                        master_iterations)
                log(f"own-play log: {from_log}, iterations 1-"
                    f"{master_iterations} of master "
                    f"{pathlib.Path(master).name} ({arm}, "
                    f"{run.trainer.training_steps} steps); temperatures "
                    f"{source['log_temperatures']}")
            log(f"{'recorded' if from_log is None else 'logged'} "
                f"{len(games)} games, {len(lives)} lives, {total} base "
                f"steps; game returns mean "
                f"{np.mean(games) if games else float('nan'):.1f} "
                f"[{time.time() - t0:.0f}s]")
            clip = {}
            if muzero.REWARD_GROWTH:
                # Test 2: the student's ceiling halves the share of reward
                # units the master's ceiling removed, on its own recorded
                # games
                unit = float(muzero.REWARD_UNIT)
                c0 = float(muzero.reward_ceiling(run.trainer.training_steps))
                mags = reward_magnitudes(lives, unit)
                c1 = halved_ceiling(mags, c0)
                if not ablation.ceiling_fits_supports(c1):
                    raise SystemExit(f"the halved-clip ceiling {c1!r} lies "
                                     f"past the reward supports")
                earned = mags[mags > 0]
                clip = {
                    "reward_unit": unit, "master_ceiling": c0,
                    "student_ceiling": c1,
                    "clip_share_master": round(clip_share(mags, c0), 6),
                    "clip_share_student": round(clip_share(mags, c1), 6),
                    "clipped_step_share_master": round(float(
                        np.mean(earned > c0)) if len(earned) else 0.0, 6),
                    "clipped_step_share_student": round(float(
                        np.mean(earned > c1)) if len(earned) else 0.0, 6),
                    # over every recorded base step (the pre-registered
                    # "share of base steps clipped"; the two above are
                    # over the reward-earning ones)
                    "clipped_base_share_master": round(float(
                        np.mean(mags > c0)) if len(mags) else 0.0, 6),
                    "clipped_base_share_student": round(float(
                        np.mean(mags > c1)) if len(mags) else 0.0, 6),
                    "reward_steps": int(len(earned)),
                    "reward_units": round(float(mags.sum()), 3)}
                log(f"student clip: the master's ceiling {c0:g} removes "
                    f"{clip['clip_share_master']:.3f} of {mags.sum():.0f} "
                    f"reward units; the student's ceiling {c1:.4g} removes "
                    f"{clip['clip_share_student']:.3f}")
            disc = {}
            if mspec.get("_env_stop"):
                # Test B: the student's discount (the gap 1 - g halved,
                # capped; Test B2: the master's, kept), and the value
                # support checked against the largest discounted clipped
                # return in these games under the student's ceiling and
                # discount (x2 headroom)
                if "student_ceiling" not in clip:
                    raise SystemExit("a Test B prep needs REWARD_GROWTH "
                                     "(the student's ceiling)")
                rule = discount_rule(mspec)
                g0 = float(muzero.DISCOUNT)
                g1 = swap_discount(g0, rule)
                vmax = discounted_return_max(lives, clip["reward_unit"],
                                             clip["student_ceiling"], g1)
                if VALUE_HEADROOM * vmax > float(muzero.VALUE_CEILING):
                    raise SystemExit(
                        f"the value support ({muzero.VALUE_CEILING} units) "
                        f"does not hold {VALUE_HEADROOM} x the largest "
                        f"discounted return {vmax:.1f} under the student's "
                        f"ceiling {clip['student_ceiling']:.4g} and "
                        f"discount {g1}")
                disc = {"master_discount": g0, "student_discount": g1,
                        "discount_rule": rule,
                        "value_max_student": round(vmax, 3),
                        "value_ceiling": float(muzero.VALUE_CEILING)}
                log(f"student discount: {g0} -> {g1} (rule {rule}); the "
                    f"largest discounted return {vmax:.1f} units (value "
                    f"support {muzero.VALUE_CEILING})")
            got = distill_lives(run, lives, master_vocab, raw_states,
                                blob["acting_states"], clip,
                                float(muzero.DISCOUNT), drop_share=drop_share,
                                mine=mine, log=log, t0=t0)
        finally:
            run.env.close()
    vocab, dropped, shares, counts = (got["vocab"], got["dropped"],
                                      got["shares"], got["counts"])
    header = {
        "format": FORMAT, "env": ENV, "seed": seed,
        "master_snapshot": str(master), "master_arm": arm,
        "master_sha256": ablation.file_sha256(master),
        "master_iterations": master_iterations,
        "master_training_steps": int(blob["training_steps"]),
        "master_offline_steps": offline,
        # (token_json: [a, f] as ever; a mined token as nested lists)
        "master_vocab": [muzero.token_json(t) for t in master_vocab],
        "vocab": [muzero.token_json(t) for t in vocab],
        "dropped": [muzero.token_json(t) for t in dropped],
        "drop_share": drop_share,
        "shares": {muzero.token_key(t): round(float(s), 5)
                   for t, s in zip(master_vocab, shares)},
        "decisions": int(counts.sum()),
        "games": len(games), "game_returns": [float(g) for g in games],
        "base_steps": int(total), "records": int(got["records"]),
        "split_decisions": int(got["splits"]),
        "split_extra_records": int(got["split_extra"]),
        "uniform_fallbacks": int(got["fallbacks"]),
        "discarded_visit_mass_mean": (
            round(float(np.mean(got["gone_mass"])), 4)
            if got["gone_mass"] else 0.0),
        "acting_temperature": float(temp), "long_token_bias": float(bias),
        "lives": len(got["out_lives"]),
        **clip,
        **disc,
        **source,
    }
    master_mined = [t for t in master_vocab if muzero.token_is_mined(t)]
    if master_mined:
        # E3-A8: the master's own mined tokens (its vocabulary's tail), which
        # its student's vocabulary checks read
        header["master_mined"] = [muzero.token_json(t) for t in master_mined]
    if mine:
        header.update(mined_header(got))
    write_set(out / "set.pkl", header, got["out_lives"])
    doc = {k: v for k, v in header.items()}
    doc["set_sha256"] = ablation.file_sha256(out / "set.pkl")
    tmp = out / f"vocab.json.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
    os.replace(tmp, out / "vocab.json")
    log(f"wrote {out / 'set.pkl'} and vocab.json [{time.time() - t0:.0f}s]")
    return header


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True)
    ap.add_argument("--master-iterations", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base-steps", type=int, default=BASE_STEPS)
    # Test B: distil from the master's own-play log instead of recording
    ap.add_argument("--from-log", default=None, metavar="DIR")
    # E3: mine tokens from the master's play and append them (E3-A3..A6)
    ap.add_argument("--mine", action="store_true")
    args = ap.parse_args(argv)
    prep(args.master, args.master_iterations, args.out, seed=args.seed,
         base_steps=args.base_steps,
         log=lambda m: print(time.strftime("%m-%d %H:%M ") + m, flush=True),
         from_log=args.from_log, mine=args.mine)


if __name__ == "__main__":
    main()
