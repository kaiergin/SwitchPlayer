"""Curriculum P3's distillation step (Kai, 2026-10-21; pre-registered in
CURRICULUM.md): the data side of a master -> student generation swap.

`prep` runs in ONE process, in the MASTER's world (its arm, its run length,
its 36 tokens):

  1. RECORD: the master snapshot plays fresh games with its training-time
     acting policy -- Run.play_episode(greedy=False): the prior at the
     master's final temperature (its training step under its own run
     length), through its acting states (the snapshot's BN-recalibrated
     representation / prediction) -- and every life is kept at RAW-STEP
     resolution: the decision records play_episode returns (stack, token,
     raw per-base-step rewards) plus every inner base frame play_token
     pushed (ablation.INNER_FRAMES), which the replay never stores. Whole
     games until >= BASE_STEPS base steps. check_life proves each
     decision's inner frames rebuild the next decision's stack.
  2. VOCABULARY: student_vocabulary drops every token whose share of the
     recorded decisions is below DROP_SHARE, never a 4-frame one (Kai's
     10-17 rule, "drop the least used"; a dropped token must be expressible
     as raw 4-frame steps). The master's slot order is kept.
  2b. THE STUDENT'S CLIP (Test 2, Kai 2026-10-24: the clip loosens only at a
     swap): under REWARD_GROWTH, the student's constant ceiling is the
     smallest that removes HALF the share of reward units the master's
     ceiling (its arm's, at its final step) removes on the recorded games
     (halved_ceiling); written to vocab.json as "student_ceiling", which
     the generation-1 student arms apply (ablation.student_arm).
  3. RE-TOKENIZE (retokenize): a kept token stays one record; a dropped
     token (a, f) becomes its m played base steps as m records of (a, 4),
     each on the stack at that base step and carrying its own raw reward.
  4. TARGETS: the master's birth search (Run._birth_search: the arm's own
     target recipe -- fresh root, 50 sims, Dirichlet noise, RAW weights, a
     greedy search on a truncated tail) over its 36 tokens on every student
     decision state; visit fractions renormalized over the student's tokens
     (masked_targets). The student's value is never distilled.
  5. WRITE <out>/set.pkl (header + one pickle frame per life) and
     <out>/vocab.json, each atomically; set.pkl last but one, vocab.json
     last, so vocab.json's existence means the prep finished.

The student run (ablation.py --distill-set) reads set.pkl with read_set and
pre-fills its replay through Run.paper_targets (ablation.distill_prefill).

Usage:
  python experiments/distill.py --master SNAPSHOT.pth --master-iterations N
      --out DIR [--seed 0] [--base-steps 31250]
"""
import argparse
import json
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
        lives.append({"traj": traj, "inner": inner,
                      "terminated": bool(terminated), "game": len(games),
                      "steps": int(steps)})
        total += steps
        g_ret += float(ret)
        if not terminated or getattr(run.env, "was_real_done", True):
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
    Returns (student vocab, dropped tokens)."""
    base = muzero.TOKEN_BASE_FRAMES
    keep, dropped = [], []
    for tok, share in zip(master_vocab, shares):
        tok = (int(tok[0]), int(tok[1]))
        (keep if tok[1] == base or share >= drop_share else dropped).append(tok)
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


# -- 3. re-tokenization -----------------------------------------------------------
def retokenize(traj, inner, master_vocab, student_vocab):
    """One recorded life in the student's vocabulary: 3-field records
    (stack, student slot, raw reward tuple entering the record) and the
    final record (stack, 0, its incoming tuple). A kept token stays one
    record; a dropped token (a, f) that played m base steps becomes m
    records of (a, TOKEN_BASE_FRAMES): the first on the decision's stack
    with the decision's incoming rewards, the j-th on that stack with inner
    frames 1..j-1 pushed, entering with base step j-1's raw reward; the
    record after it enters with the last base step's. Every raw reward and
    base step is kept exactly once."""
    index = {t: i for i, t in enumerate(student_vocab)}
    base = muzero.TOKEN_BASE_FRAMES
    out, incoming = [], traj[0][2]
    for i in range(len(traj) - 1):
        stack, slot = traj[i][0], int(traj[i][1])
        tok = tuple(master_vocab[slot])
        raw = traj[i + 1][2]
        if tok in index:
            out.append((stack, index[tok], incoming))
            incoming = raw
            continue
        b = index[(tok[0], base)]
        s = stack
        out.append((s, b, incoming))
        for j in range(1, len(raw)):
            s = muzero.push_frame(s, inner[i][j - 1])
            out.append((s, b, (raw[j - 1],)))
        incoming = (raw[-1],)
    out.append((traj[-1][0], 0, incoming))
    return out


# -- 4. targets -------------------------------------------------------------------
def masked_targets(pols, master_vocab, student_vocab):
    """Master visit fractions over its tokens -> the student's: the student
    tokens' entries, renormalized (visits to dropped tokens discarded).
    A row with no mass on any student token falls back to uniform; returns
    (targets, fallbacks, the mass discarded per row -- the search spreads
    visits over every child, so the read-out sharpens the targets by
    1 / (1 - discarded); audit r1 A's note)."""
    idx = [list(map(tuple, master_vocab)).index(t) for t in student_vocab]
    out, fallbacks, gone = [], 0, []
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


# -- the prep ---------------------------------------------------------------------
def prep(master, master_iterations, out, seed=0, base_steps=BASE_STEPS,
         drop_share=DROP_SHARE, log=print):
    """Steps 1-5 (module docstring) for one master snapshot; returns the
    header written."""
    import ablation
    t0 = time.time()
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    blob = torch.load(str(master), map_location="cpu", weights_only=False)
    arm = blob.get("arm")
    if arm not in ablation.ARMS:
        raise SystemExit(f"{master}: arm {arm!r} unknown")
    master_vocab = tuple(tuple(int(x) for x in t)
                         for t in (blob["spec"].get("TOKEN_VOCAB") or ()))
    if not master_vocab or master_vocab != tuple(ablation.ARMS[arm]["TOKEN_VOCAB"]):
        raise SystemExit(f"{master}: its TOKEN_VOCAB is not its arm's")
    if not blob.get("acting_states"):
        raise SystemExit(f"{master}: no acting_states (a snapshot of a "
                         f"BN_RECAL_ACTING run is required)")
    if (int(blob["training_steps"]) != master_iterations * TRAIN_STEPS
            or blob.get("iteration") != master_iterations):
        raise SystemExit(f"{master}: training_steps {blob['training_steps']} "
                         f"is not the end of a {master_iterations}-iteration "
                         f"run (the master is its FINAL snapshot)")
    ablation.ENV_NAME, ablation.FRAMESKIP, ablation.EPISODE_CAP = \
        ENV, FRAMESKIP, EPISODE_CAP
    with ablation.arm_config(arm) as spec, \
            ablation._run_length_globals(master_iterations, TRAIN_STEPS), \
            ablation.token_actions():
        run = ablation.Run(arm, seed, spec)
        try:
            raw_states = load_master(run, blob)
            temp = muzero.prior_acting_temperature(run.trainer.training_steps)
            bias = muzero.token_long_bias(run.trainer.training_steps)
            set_rep_pred(run, blob["acting_states"])
            log(f"recording: master {pathlib.Path(master).name} ({arm}, "
                f"{run.trainer.training_steps} steps), temperature {temp}, "
                f"long-token bias {bias}, until >= {base_steps} base steps")
            lives, games, total = record(run, base_steps)
            log(f"recorded {len(games)} games, {len(lives)} lives, {total} "
                f"base steps; game returns mean {np.mean(games):.1f} "
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
            shares, counts = token_shares(lives, len(master_vocab))
            vocab, dropped = student_vocabulary(master_vocab, shares,
                                                drop_share)
            log(f"student vocabulary: {len(vocab)} of {len(master_vocab)} "
                f"tokens; dropped {list(dropped)}")
            set_rep_pred(run, raw_states)
            out_lives, records, splits, extra, fallbacks = [], 0, 0, 0, 0
            gone_mass = []
            dropped_set = set(dropped)
            for life in lives:
                splits += sum(1 for r in life["traj"][:-1]
                              if tuple(master_vocab[int(r[1])]) in dropped_set)
                recs = retokenize(life["traj"], life["inner"], master_vocab,
                                  vocab)
                pols, vals = master_search(run, recs, life["terminated"])
                pols, fb, gone = masked_targets(pols, master_vocab, vocab)
                fallbacks += fb
                # acted records only (a terminated tail's uniform is no
                # search)
                gone_mass += gone[:-1] if life["terminated"] else gone
                out_lives.append({
                    "traj": [(r[0], r[1], tuple(r[2]), p, v)
                             for r, p, v in zip(recs, pols, vals)],
                    "terminated": life["terminated"]})
                records += len(recs)
                extra += len(recs) - len(life["traj"])
                life["inner"] = None          # free as we go
            log(f"re-tokenized + master-searched {records} records "
                f"({splits} dropped-token decisions split into {extra} "
                f"extra records; {fallbacks} uniform fallbacks; mean visit "
                f"mass discarded on dropped tokens "
                f"{np.mean(gone_mass) if gone_mass else 0.0:.3f}) "
                f"[{time.time() - t0:.0f}s]")
        finally:
            run.env.close()
    header = {
        "format": FORMAT, "env": ENV, "seed": seed,
        "master_snapshot": str(master), "master_arm": arm,
        "master_sha256": ablation.file_sha256(master),
        "master_iterations": master_iterations,
        "master_training_steps": int(blob["training_steps"]),
        "master_vocab": [list(t) for t in master_vocab],
        "vocab": [list(t) for t in vocab],
        "dropped": [list(t) for t in dropped],
        "drop_share": drop_share,
        "shares": {f"{a},{f}": round(float(s), 5)
                   for (a, f), s in zip(master_vocab, shares)},
        "decisions": int(counts.sum()),
        "games": len(games), "game_returns": [float(g) for g in games],
        "base_steps": int(total), "records": int(records),
        "split_decisions": int(splits), "split_extra_records": int(extra),
        "uniform_fallbacks": int(fallbacks),
        "discarded_visit_mass_mean": (round(float(np.mean(gone_mass)), 4)
                                      if gone_mass else 0.0),
        "acting_temperature": float(temp), "long_token_bias": float(bias),
        "lives": len(out_lives),
        **clip,
    }
    write_set(out / "set.pkl", header, out_lives)
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
    args = ap.parse_args(argv)
    prep(args.master, args.master_iterations, args.out, seed=args.seed,
         base_steps=args.base_steps,
         log=lambda m: print(time.strftime("%m-%d %H:%M ") + m, flush=True))


if __name__ == "__main__":
    main()
