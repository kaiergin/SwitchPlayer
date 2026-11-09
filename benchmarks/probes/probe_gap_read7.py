"""Test G, READ (7): is there anything for exploration or search to find?
(CURRICULUM "Next experiment (pre-registered) -- Test G: does the
master-student value gap point at real weakness?", its read (7), Kai
2026-11-07: "the first look at Kai's question, whether exploration or search
makes a difference".) Descriptive: no verdict and no gate of its own. No net
trains; the emulator measures outcomes only and no branch feeds a model (the
E line's rule). One decision only, so a lower bound on what search does.

REFUSES, before anything plays, unless stage 2b's summary (stage2b.json,
stamp-checked) ran past its gates: its verdict HOLDS or NOT SHOWN (never
VOID), stage 2a's gates PASS-SO-FAR (gates.json re-read: G3, G4, G5), G2's
first clause passed (g2.json re-read) and G2's inside clause passed (both
lower bounds in stage2b.json above 0: this read ranks states by g, and nets
that cannot rank states answer nothing here).

THE STATES: stage 2b's primary sample (the kept states minus those whose
stack is in the master's or the twin's training stacks: stage 2b's own
assemble and exclusion, its count checked against stage2b.json). g at the
primary head (0.997 per frame, the student's clip) is ranked inside each
stratum (life index x 20-base-step bins) among the sample's states, ties
averaged, as rho_w ranks them: a state of average rank r among n is in the
TOP FIFTH when (r - 0.5) / n >= 4/5 and in the BOTTOM FIFTH when it is <= 1/5
(rho_w's centred rank +0.3 or more, -0.3 or less; a stratum of one or two
holds neither). PER_FIFTH = 120 states are drawn at random from each fifth
without replacement, seed DRAW_SEED (a fifth holding fewer is taken whole).

AT EACH DRAWN STATE, from its saved emulator state (the pilot's ALEState):
  CLASSES: T0's slots grouped by outcome, the census's classes
      (probe_action_census.token_census over T0's vocabulary in slot order,
      each token then NOOP to the census's 92-frame window; two slots are one
      class when every raw RGB frame is equal). Each class is played as its
      highest acting-prior slot (the deployed prior on the snapshot's acting
      BN states, probe_gap_check.player; ties to the lowest slot), so the
      deployed slot, the prior's argmax, stands for its class.
  BRANCHES: a class's slot played at s through the deployed wrappers
      (ablation.play_token, as the policy plays a token), then the delay
      (NOOP frames on raw ALE, stage 2a's way: a pooled frame enters the stack
      every 4 delay frames, a life lost in the delay ends the branch, the
      delay's rewards count with the next base step), then T0's deployed
      policy to the life's end, capped at 1,536 frames, slot and delay
      included (rewards stop at the last whole base step inside the cap).
      Delays 0-3 CHOOSE and 8-11 SCORE: no state's choice reads a scoring
      delay. Returns on stage 2a's 8 heads (probe_gap_check.head_returns:
      base step j from s weighs gamma ** (4 j), the clip per base step); the
      reads use the primary head. CHECK: the deployed class at delay 0 is
      stage 2a's T0 branch at delay 0 (the same state, stack and policy), so
      its returns must equal gap.pkl's on all 8 heads exactly, and the
      deployed slot must be t0.pkl's prior argmax; else the read is VOID.
  SEARCH: T0's trainer weights in eval mode as birth search runs them
      (probe_prefix_bn.snapshot_run, no overlay; the four nets birth search
      takes deep-copied and set to eval before the acting overlay is loaded
      into the run, which then gives the deployed prior again: a cross-check
      against the play pass's prior, max |p - p'| <= CROSS_TOL, or VOID).
      muzero_vtree.VectorizedPUCT.search at 50 and at 200 simulations with
      root noise (birth search's), the snapshot's own C_PUCT, DISCOUNT, depth
      and prefix knobs; ROOTS = 8 roots a chunk for both budgets, chunk c
      being drawn states 8c..8c+7 whatever the pass; before each search
      torch's global RNG (the Dirichlet draw) is seeded per chunk, the same
      for both budgets, and the tie-break generator per chunk too. Its top
      slot is the most visited (np.argmax: ties to the lowest slot). THE RAW
      PRIOR: one forward pass on the same eval-mode BN (initial_inference
      before the overlay; the acting BN states differ), its argmax.
THE PER-STATE READS, primary head, a class's mean return over its delays:
  ROOM: the class with the best mean on 0-3 (ties to the deployed class,
      then to the class whose played slot is lowest), its mean on 8-11 minus
      the deployed class's; 0 when it is the deployed class.
  MEAN CLASS MINUS DEPLOYED: the mean over the state's classes (the deployed
      included) of their 8-11 means, minus the deployed class's.
  SEARCH GAIN, for the raw prior and for 50 and 200 simulations: the top
      slot's class, its 8-11 mean minus the deployed class's.
  Beside: the classes a state, and the share of states whose chosen or top
      slot's class is not the deployed class.
TOP AGAINST BOTTOM FIFTH: each read's mean in each fifth, the difference
(top minus bottom), and both fifths together, each with its two-sided 95%
percentile interval over DRAWS cluster draws (probe_gap_stage2a.
cluster_draws: the drawn states' clusters from gap.pkl, as many as there are,
with replacement, each bringing all its drawn states), seed BOOT_SEED. A
statistic undefined on any draw (a draw without one fifth) has no interval.

THE PASSES, each a fresh process; a file whose stamp (this read's knobs and
source, stage 2b's stamp with stage 2a's, the pilot's and the nets' inside,
and the digests of stage 2b's files read) is not this run's is refused, and a
pass whose file exists is reused, so a stopped read resumes:
  draw: the fifths and the drawn states (draw.pkl), with what the checks
      compare against (gap.pkl's T0 delay-0 returns, t0.pkl's argmax).
  play K: chunk K's states (CHUNK a pass): census, deployed prior, branches.
  search K: chunk K's states: the raw prior, the acting cross-check prior,
      both searches.
`read7` runs the missing passes in order and reads them: read7.json, the
summary, never overwritten (a VOID one when a check fails).

usage: python probe_gap_read7.py read7 [--out DIR] [--pilot DIR] [--gap DIR]
                                 [--nets DIR] [--stage2b DIR]
       python probe_gap_read7.py draw | play --chunk K | search --chunk K [...]
Defaults: the pilot's, stage 2a's, the nets' and stage 2b's real output
directories (PILOT, GAP, NETS, STAGE2B below); DIR probe_gap_read7 next to
this file. Knobs: GAP7_CHUNK (states a pass, 40, a multiple of ROOTS),
GAP7_STATES (the first N drawn states only; 0, all), GAP7_DRAWS (10,000);
the last two, or any of stage 2b's smoke knobs, mark the record smoke and move
the default DIR to probe_gap_read7-smoke. CPU only, 2 threads, stage 2a's
environment. Imported as a module (the tests) it changes nothing
process-wide."""
import os
# probe_gap_check's own settings when it runs as a script (stage 2a's ENV),
# made before any import: ablation picks its device when imported
ENV = {"CUDA_VISIBLE_DEVICES": "-1",          # "" deletes the var on Windows
       "PROBE_ENV": "MsPacmanNoFrameskip-v4",
       "PROBE_ACTING_STATES": "1",              # the deployed policy
       "PROBE_THREADS": "2"}                    # as the rows ran
if __name__ == "__main__":
    os.environ.update(ENV)
import argparse
import copy
import pathlib
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
import muzero_vtree
import ablation
import probe_action_census as pc
import probe_e3a_sequences as e3
import probe_gap_check as gc
import probe_gap_nets as gn
import probe_gap_stage2a as s2
import probe_gap_stage2b as s2b

# the real outputs this read consumes, read only
PILOT = pathlib.Path(r"Y:\SwitchPlayer-testg-run\benchmarks\probes"
                     r"\probe_gap_check-pilot")
GAP = pathlib.Path(r"Y:\SwitchPlayer-testg2a-run\benchmarks\probes"
                   r"\probe_gap_stage2a")
NETS = pathlib.Path(r"Y:\SwitchPlayer-testg-nets-run\benchmarks\probes"
                    r"\probe_gap_nets-out")
STAGE2B = pathlib.Path(r"Y:\SwitchPlayer-testg2b-run\benchmarks\probes"
                       r"\probe_gap_stage2b")
PRIMARY = s2.PRIMARY                         # (scale, discount): clip, 0.997
FIFTH_TOP, FIFTH_BOTTOM = (4, 5), (1, 5)     # quantile bounds, as fractions
PER_FIFTH = 120
CHOOSE = (0, 1, 2, 3)                        # delays that choose ...
SCORE = (8, 9, 10, 11)                       # ... and delays that score
DELAYS = CHOOSE + SCORE
FRAME_CAP = gc.FRAME_CAP                     # 1,536 frames, slot and delay in
SIMS = (50, 200)
ROOTS = 8                                    # roots a search chunk
SEARCH_NETS = ("representation", "dynamics", "prediction", "value_prefix")
DRAW_SEED = (gc.SEED, 6)                     # 1-3 the pilot's, 4 stage 2a's,
BOOT_SEED = (gc.SEED, 7)                     # 5 stage 2b's
NOISE_STREAM, TIE_STREAM = 8, 9              # the searches' seeds, per chunk
CROSS_TOL = s2b.CROSS_TOL
FIFTHS = ("top", "bottom")
READS = ("room", "mean_minus_deployed", "gain_raw", "gain_50", "gain_200")
BESIDE = ("classes", "moved", "moved_raw", "moved_50", "moved_200")
PARTS = ("top", "bottom", "top_minus_bottom", "both")
SUMMARY = "read7.json"
THREADS = 2
CHUNK = int(os.environ.get("GAP7_CHUNK") or 40)
LIMIT = int(os.environ.get("GAP7_STATES") or 0)
DRAWS = int(os.environ.get("GAP7_DRAWS") or 10_000)
SMOKE = LIMIT != 0 or DRAWS != 10_000 or s2b.SMOKE
# this read's source and the search engine; the stamps of stage 2b, stage
# 2a, the pilot and the nets (in every stamp) cover the rest
SOURCES = (pathlib.Path(__file__).resolve(), ROOT / "muzero_vtree.py")
SOURCE = gc.source_hash(SOURCES)


# --------------------------------------------------------------------------
# the fifths and the draw
# --------------------------------------------------------------------------
def fifth_masks(g, sid):
    """(top, bottom) boolean masks: inside each stratum of n states, g's
    average rank r (1..n, ties averaged); top when (r - 0.5) / n >= 4/5,
    bottom when <= 1/5, in integers (2 r is whole)."""
    g = np.asarray(g, np.float64)
    if not np.all(np.isfinite(g)):
        raise SystemExit("g is not finite on every state of the sample")
    top, bottom = np.zeros(len(g), bool), np.zeros(len(g), bool)
    groups = {}
    for i, k in enumerate(sid):
        groups.setdefault(k, []).append(i)
    (tn, td), (bn, bd) = FIFTH_TOP, FIFTH_BOTTOM
    for m in groups.values():
        m = np.asarray(m)
        _, inverse, counts = np.unique(g[m], return_inverse=True,
                                       return_counts=True)
        twice = (2 * np.cumsum(counts) - (counts - 1))[inverse.ravel()]
        n = len(m)
        # (r - 0.5) / n = (2 r - 1) / (2 n)
        top[m] = td * (twice - 1) >= 2 * tn * n
        bottom[m] = bd * (twice - 1) <= 2 * bn * n
    return top, bottom


def draw_fifths(g, sid, per=None, seed=DRAW_SEED):
    """{fifth: (sorted indices drawn, the fifth's size)}: `per` of each at
    random without replacement, the top's draw first; a smaller fifth
    whole."""
    per = PER_FIFTH if per is None else per
    top, bottom = fifth_masks(g, sid)
    rng = np.random.default_rng(seed)
    out = {}
    for name, mask in zip(FIFTHS, (top, bottom)):
        pool = np.flatnonzero(mask)
        pick = rng.choice(pool, min(per, len(pool)), replace=False) \
            if len(pool) else pool
        out[name] = (np.sort(pick).astype(int), int(len(pool)))
    return out


# --------------------------------------------------------------------------
# one state: classes, representatives, branches
# --------------------------------------------------------------------------
def representatives(partition, prior):
    """Each class's played slot, by class id: its highest-prior slot, ties to
    the lowest."""
    prior = np.asarray(prior, np.float64)
    reps = []
    for c in range(max(partition) + 1):
        members = [i for i, x in enumerate(partition) if x == c]
        best = max(prior[i] for i in members)
        reps.append(min(i for i in members if prior[i] == best))
    return reps


def class_branch(env, decide, state, stack, slot, delay, cap=None):
    """One branch: `slot` played at the saved state through the deployed
    wrappers, then `delay` NOOP frames on raw ALE, then `decide` to the
    life's end, at most `cap` frames in all. `rewards`: raw per base step,
    the delay's added to the base step after it; `tokens` every token played,
    the slot first."""
    cap = FRAME_CAP if cap is None else cap
    ale = env.unwrapped.ale
    gc.restore(env, state)
    start = ale.getEpisodeFrameNumber()
    budget = (cap - delay) // gc.BASE
    inner = []
    obs, _, term, trunc, stack, n = ablation.play_token(env, slot, stack,
                                                        inner)
    tokens = [int(slot)]
    if term or trunc:                        # the life ended in the slot
        rewards, ended = inner[:budget], "life"
    elif n >= budget:                        # only a tiny cap: no room left
        rewards, ended = inner[:budget], "cap"
    else:
        stack = e3.reached_stack(stack, [obs])
        stack, first, lost = gc.raw_frames(ale, env.unwrapped._action_set[0],
                                           delay, stack)
        if lost:
            rewards, ended = inner + [first], "delay"
        else:
            played, _, died = gc.play_life(
                env, decide, stack, steps=n, cap=budget,
                on_decision=lambda st, token, steps: tokens.append(int(token)))
            rewards = inner + played
            ended = "life" if died and len(rewards) <= budget else "cap"
            rewards = rewards[:budget]
            rewards[n] += first
    return {"slot": int(slot), "delay": int(delay), "rewards": rewards,
            "ended": ended, "tokens": tokens,
            "frames": ale.getEpisodeFrameNumber() - start}


def play_state(env, decide, prior_of, s, vocab, delays=DELAYS, cap=None):
    """One drawn state: the census's classes over `vocab` (slot order), the
    deployed prior, each class's slot, its branches at `delays` and their
    returns (classes, delays, scales, discounts)."""
    ale = env.unwrapped.ale
    actions = list(env.unwrapped._action_set)
    saved = ale_py.ALEState(s["state"])
    part, strict, _ = pc.token_census(ale, actions, saved, grid=vocab,
                                      window=pc.WINDOW_TOKENS)
    prior = np.asarray(prior_of(s["stack"]), np.float64).reshape(-1)
    if len(prior) != len(vocab):
        raise SystemExit(f"a prior over {len(prior)} slots, {len(vocab)} "
                         f"tokens")
    deployed = int(np.argmax(prior))
    if decide(s["stack"]) != deployed:
        raise SystemExit("the deployed policy is not the prior's argmax")
    reps = representatives(part, prior)
    rows = [[class_branch(env, decide, saved, s["stack"], slot, d, cap)
             for d in delays] for slot in reps]
    returns = np.array([[gc.head_returns(b["rewards"]) for b in row]
                        for row in rows], np.float64)
    return {"partition": [int(x) for x in part],
            "strict": [int(x) for x in strict], "prior": prior,
            "deployed": deployed, "reps": reps, "delays": list(delays),
            "branches": rows, "returns": returns}


# --------------------------------------------------------------------------
# the search
# --------------------------------------------------------------------------
def seed_of(stream, chunk):
    """A chunk's seed for torch (uint32 from numpy's SeedSequence)."""
    return int(np.random.SeedSequence((gc.SEED, stream, int(chunk)))
               .generate_state(1)[0])


def forward_priors(run, stacks):
    """initial_inference's softmax for each stack: one forward pass."""
    m = run.searcher.network_manager
    return np.array([np.asarray(m.initial_inference(x)[0], np.float64)
                     .reshape(-1) for x in stacks], np.float64)


def overlay(run, blob):
    """The acting states into the run's trainer nets, as
    probe_prefix_bn.snapshot_run(acting=True) loads them."""
    acting = blob.get("acting_states")
    if not acting:
        raise SystemExit("the snapshot has no acting_states")
    for n, sd in acting.items():
        getattr(run.trainer, n).load_state_dict(sd)
    run.set_train_mode(False)


def same_weights(net, state_dict):
    """Whether `net` holds `state_dict` exactly (every tensor)."""
    mine = net.state_dict()
    return set(mine) == set(state_dict) and all(
        torch.equal(mine[k].cpu(), torch.as_tensor(state_dict[k]).cpu())
        for k in mine)


def search_states(run, blob, stacks, first_chunk, sims=SIMS, roots=ROOTS):
    """The raw prior, the acting prior (the cross-check) and each budget's
    search on `stacks`, drawn states first_chunk * roots onwards. The search
    nets are deep copies of the run's trainer nets set to eval, taken before
    the acting overlay, and must still hold the snapshot's own weights."""
    nets = {n: copy.deepcopy(getattr(run.trainer, n)) for n in SEARCH_NETS}
    for net in nets.values():
        net.eval()
    run.set_train_mode(False)                 # snapshot_run's eval mode
    raw = forward_priors(run, stacks)
    overlay(run, blob)
    acting = forward_priors(run, stacks)
    for n, net in nets.items():
        if not same_weights(net, blob[n]):
            raise SystemExit(f"the search's {n} is not the snapshot's own "
                             f"weights")
        if net.training:
            raise SystemExit(f"the search's {n} is in train mode")
    engine = muzero_vtree.VectorizedPUCT(nets, ablation.ACTIONS, "cpu")
    out = {k: {"counts": [], "values": [], "chunks": []} for k in sims}
    for j in range(0, len(stacks), roots):
        chunk = first_chunk + j // roots
        obs = np.stack(stacks[j:j + roots])
        for k in sims:
            torch.manual_seed(seed_of(NOISE_STREAM, chunk))
            gen = torch.Generator().manual_seed(seed_of(TIE_STREAM, chunk))
            _, values, counts = engine.search(
                obs, num_simulations=k, c_puct=muzero.C_PUCT,
                discount=muzero.DISCOUNT, root_noise=True, generator=gen,
                tie_break="random")
            out[k]["counts"] += [np.asarray(c, np.float64) for c in counts]
            out[k]["values"] += [float(v) for v in values]
            out[k]["chunks"] += [chunk] * len(obs)
    top = {"raw": [int(np.argmax(p)) for p in raw]}
    for k in sims:
        top[str(k)] = [int(np.argmax(c)) for c in out[k]["counts"]]
    return {"raw": raw, "acting": acting,
            "search": {str(k): {"counts": np.array(v["counts"]),
                                "values": v["values"], "chunks": v["chunks"]}
                       for k, v in out.items()},
            "top": top}


# --------------------------------------------------------------------------
# the per-state reads
# --------------------------------------------------------------------------
def choose_class(means, deployed, reps):
    """The class with the best mean on the choosing delays: ties to the
    deployed class, then to the class whose played slot is lowest."""
    means = np.asarray(means, np.float64)
    tied = [int(c) for c in np.flatnonzero(means == means.max())]
    if deployed in tied:
        return deployed
    return min(tied, key=lambda c: reps[c])


def state_reads(rec, top):
    """One state's reads from its play record and its top slots {"raw",
    "50", "200"}: ROOM, mean class minus deployed, the search gains, and the
    beside counts."""
    part = rec["partition"]
    delays = list(rec["delays"])
    choose = [delays.index(d) for d in CHOOSE]
    score = [delays.index(d) for d in SCORE]
    r = np.asarray(rec["returns"], np.float64)[:, :, PRIMARY[0], PRIMARY[1]]
    choose_mean, score_mean = r[:, choose].mean(axis=1), r[:, score].mean(axis=1)
    c0 = part[rec["deployed"]]
    chosen = choose_class(choose_mean, c0, rec["reps"])
    out = {"room": float(score_mean[chosen] - score_mean[c0]),
           "mean_minus_deployed": float(score_mean.mean() - score_mean[c0]),
           "classes": float(len(rec["reps"])),
           "moved": float(chosen != c0), "chosen": int(chosen),
           "deployed_class": int(c0)}
    for k in ("raw", "50", "200"):
        c = part[top[k]]
        out[f"gain_{k}"] = float(score_mean[c] - score_mean[c0])
        out[f"moved_{k}"] = float(c != c0)
    return out


# --------------------------------------------------------------------------
# top against bottom: the cluster bootstrap
# --------------------------------------------------------------------------
def parts(values, top, idx):
    """{part: value} of one read on the states `idx` (with repeats)."""
    t, b = idx[top[idx]], idx[~top[idx]]
    mt = float(values[t].mean()) if len(t) else np.nan
    mb = float(values[b].mean()) if len(b) else np.nan
    return {"top": mt, "bottom": mb, "top_minus_bottom": mt - mb,
            "both": float(values[idx].mean()) if len(idx) else np.nan}


def boot(values, fifth, clusters, draws=None, seed=BOOT_SEED):
    """Each read's parts on the drawn states and on every cluster draw:
    {name: {"point" {part}, "intervals" {part: [lo, hi] | None},
    "undefined" {part: draws}}}. `values` {name: (n,)}; `fifth` the states'
    "top" | "bottom"."""
    draws = DRAWS if draws is None else draws
    top = np.asarray(fifth) == "top"
    vals = {k: np.asarray(v, np.float64) for k, v in values.items()}
    n = len(top)
    every = np.arange(n)
    rows = {k: [] for k in vals}
    for idx in s2.cluster_draws(clusters, draws, seed=seed):
        for k, v in vals.items():
            p = parts(v, top, idx)
            rows[k].append([p[q] for q in PARTS])
    out = {}
    for k, v in vals.items():
        a = np.array(rows[k], np.float64).reshape(len(rows[k]), len(PARTS))
        out[k] = {"point": parts(v, top, every),
                  "intervals": {q: s2b.interval(a[:, j])
                                for j, q in enumerate(PARTS)},
                  "undefined": {q: int((~np.isfinite(a[:, j])).sum())
                                for j, q in enumerate(PARTS)}}
    return out


# --------------------------------------------------------------------------
# files, stamps, the gate
# --------------------------------------------------------------------------
def knobs():
    return {"chunk": CHUNK, "states": LIMIT, "draws": DRAWS}


INPUTS_2B = ("stage2b.json",) + tuple(f"{n}.pkl" for n in s2b.PASSES)


def stamp(st2b, stage2b):
    """What made a file: this read's knobs and source, stage 2b's stamp, and
    the digests of stage 2b's files it reads."""
    return {"knobs": knobs(), "smoke": SMOKE, "source": SOURCE,
            "stage2b": st2b,
            "inputs": {n: s2.file_digest(pathlib.Path(stage2b) / n)
                       for n in INPUTS_2B}}


write, read = s2.write, s2.read              # whole or nothing; stamp-checked


def guard(out, *read_only):
    """No output inside any input (stage 2b's guard: the pilot's, the run
    tree, and these)."""
    s2b.guard(out, *read_only)


def draw_file(out):
    return pathlib.Path(out) / "draw.pkl"


def play_file(out, k):
    return pathlib.Path(out) / f"play-{k:03d}.pkl"


def search_file(out, k):
    return pathlib.Path(out) / f"search-{k:03d}.pkl"


def stage2b_reasons(rec):
    """Why read (7) refuses, from stage2b.json: [] when stage 2b ran past its
    gates (a verdict of HOLDS or NOT SHOWN, stage 2a passed so far, G2's
    first clause passed, both of G2's inside-strata lower bounds above 0)."""
    verdict = rec.get("verdict")
    if verdict not in ("HOLDS", "NOT SHOWN"):
        return [f"stage 2b's verdict is {verdict!r}"]
    reasons = []
    gates = rec.get("gates") or {}
    if gates.get("verdict") != "PASS-SO-FAR":
        reasons.append(f"stage 2b's record of stage 2a: {gates.get('verdict')}")
    for k in ("G3", "G5"):
        if (gates.get(k) or {}).get("ok") is not True:
            reasons.append(f"stage 2b's record of {k}: not ok")
    g2 = rec.get("G2_first") or {}
    if g2.get("clause") != "first" or g2.get("verdict") != gn.G2_FIRST_PASS:
        reasons.append(f"stage 2b's record of G2's first clause: "
                       f"{g2.get('clause')!r} {g2.get('verdict')!r}")
    iv = (((rec.get("reads") or {}).get("primary") or {})
          .get("intervals") or {})
    for name in ("g2_master", "g2_twin"):
        lo = (iv.get(name) or [None])[0]
        if lo is None or not lo > 0:
            reasons.append(f"G2 inside strata ({name}): lower bound {lo}, "
                           f"not above 0")
    return reasons


class Context:
    """Every pass's inputs, read and checked: the kept states, stage 2b's
    stamp, its summary (gates passed) and this read's stamp."""

    def __init__(self, out, pilot, gap, nets, stage2b):
        self.out, self.pilot, self.gap, self.nets, self.stage2b = (
            pathlib.Path(p) for p in (out, pilot, gap, nets, stage2b))
        guard(self.out, self.pilot, self.gap, self.nets, self.stage2b)
        summary = self.stage2b / s2b.SUMMARY
        if not summary.exists():
            raise SystemExit(f"{summary}: stage 2b has not finished, read (7) "
                             f"does not run")
        _, self.states, self.keys = s2.load_states(self.pilot)
        self.st2b = s2b.stamp(self.states, self.gap, self.nets)
        self.rec2b = read(summary, self.st2b)
        reasons = stage2b_reasons(self.rec2b)
        reasons += s2b.read_gates(self.gap, self.nets, self.states)[3]
        if reasons:
            raise SystemExit(f"stage 2b did not pass its gates "
                             f"({'; '.join(reasons)}): read (7) does not run")
        self.st = stamp(self.st2b, self.stage2b)

    def drawn(self):
        """The draw file's states, in order (the first GAP7_STATES only)."""
        d = read(draw_file(self.out), self.st)
        n = len(d["drawn"]) if not LIMIT else min(LIMIT, len(d["drawn"]))
        if n < 1:
            raise SystemExit("no drawn states")
        return d, d["drawn"][:n]


def chunks(n):
    """[(k, first, end)]: every drawn state once, CHUNK at a time."""
    if CHUNK < 1 or CHUNK % ROOTS:
        raise SystemExit(f"GAP7_CHUNK must be a positive multiple of {ROOTS}, "
                         f"got {CHUNK}")
    return [(k, lo, min(n, lo + CHUNK))
            for k, lo in enumerate(range(0, n, CHUNK))]


def chunk_range(n, k):
    found = [c for c in chunks(n) if c[0] == k]
    if not found:
        raise SystemExit(f"chunk {k}: {n} states make {len(chunks(n))}")
    return found[0][1:]


# --------------------------------------------------------------------------
# the passes
# --------------------------------------------------------------------------
def stage_draw(ctx):
    t0 = time.time()
    gap = read(ctx.gap / "gap.pkl", s2.stamp(ctx.states))
    recs = {n: read(s2b.pass_file(ctx.stage2b, n), ctx.st2b)
            for n in s2b.PASSES}
    data, training = s2b.assemble(ctx.states, ctx.keys, gap, recs["nets"],
                                  recs["t0"], recs["table"])
    left = np.zeros(len(ctx.states), bool)
    for n in s2b.EXCLUDED_BY:
        left |= training[n]
    sample = np.flatnonzero(~left)
    if len(sample) != ctx.rec2b["sample"]["states"]:
        raise SystemExit(f"{len(sample)} states in the sample, stage 2b read "
                         f"{ctx.rec2b['sample']['states']}")
    g, sid = data["g"][sample], data["sid"][sample]
    picks = draw_fifths(g, sid)
    drawn = sorted([(int(sample[i]), name) for name in FIFTHS
                    for i in picks[name][0]])
    kept = [k for k, _ in drawn]
    write(draw_file(ctx.out), {
        "sample": int(len(sample)), "strata": len(set(sid.tolist())),
        "pools": {n: picks[n][1] for n in FIFTHS},
        "drawn_per_fifth": {n: int(len(picks[n][0])) for n in FIFTHS},
        "drawn": kept, "fifth": [f for _, f in drawn],
        "ids": s2.ids([ctx.states[k] for k in kept]),
        "g": [float(data["g"][k]) for k in kept],
        "keys": [tuple(ctx.keys[k]) for k in kept],
        "clusters": [gap["clusters"][k] for k in kept],
        "t0_argmax": [int(recs["t0"]["prior_argmax"][k]) for k in kept],
        "t0_delay0": np.asarray(gap["returns"]["T0"], np.float64)[kept, 0],
        "secs": round(time.time() - t0, 1)}, ctx.st)


def stage_play(ctx, k):
    draw, kept = ctx.drawn()
    lo, hi = chunk_range(len(kept), k)
    t0, recs = time.time(), []
    with gc.player("T0") as run:
        env, decide = run.env, gc.deployed(run)
        manager = run.searcher.network_manager
        vocab = tuple(tuple(int(x) for x in t) for t in muzero.TOKEN_VOCAB)
        for i in range(lo, hi):
            s = ctx.states[kept[i]]
            t1 = time.time()
            rec = play_state(env, decide,
                             lambda x: manager.initial_inference(x)[0], s,
                             vocab)
            rec.update(kept=kept[i], id=[s["game"], s["decision"]],
                       secs=round(time.time() - t1, 2))
            recs.append(rec)
            if (i + 1 - lo) % 10 == 0:
                print(f"play chunk {k}: {i + 1 - lo} of {hi - lo} states "
                      f"[{time.time() - t0:.0f}s]", flush=True)
    write(play_file(ctx.out, k),
          {"chunk": k, "range": [lo, hi], "kept": kept[lo:hi],
           "states": recs, "threads": torch.get_num_threads(),
           "secs": round(time.time() - t0, 1), "memory": s2.memory()}, ctx.st)


def load_blob(path):
    return torch.load(str(path), map_location="cpu", weights_only=False)


def trainer_run(blob):
    """probe_prefix_bn.snapshot_run, no overlay: the snapshot's own arm, its
    trainer weights in eval mode (stage 2b's trainer_run)."""
    return s2b.trainer_run(blob, acting=False)


def stage_search(ctx, k):
    draw, kept = ctx.drawn()
    lo, hi = chunk_range(len(kept), k)
    stacks = [ctx.states[kept[i]]["stack"] for i in range(lo, hi)]
    t0 = time.time()
    blob = load_blob(gc.snapshot("T0"))
    with trainer_run(blob) as (run, missing):
        res = search_states(run, blob, stacks, lo // ROOTS)
    del blob
    write(search_file(ctx.out, k),
          {"chunk": k, "range": [lo, hi], "kept": kept[lo:hi], **res,
           "missing_spec_keys": list(missing),
           "threads": torch.get_num_threads(),
           "secs": round(time.time() - t0, 1), "memory": s2.memory()}, ctx.st)


def run_pass(ctx, stage, k=None):
    """A pass in its own process; a failed pass stops the read."""
    cmd = [sys.executable, "-u", str(pathlib.Path(__file__).resolve()), stage,
           "--out", str(ctx.out), "--pilot", str(ctx.pilot), "--gap",
           str(ctx.gap), "--nets", str(ctx.nets), "--stage2b",
           str(ctx.stage2b)] + ([] if k is None else ["--chunk", str(k)])
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode:
        raise SystemExit(f"pass {stage}{'' if k is None else f' {k}'}: rc "
                         f"{r.returncode}")


def read7(out, pilot, gap, nets, stage2b):
    out = pathlib.Path(out)
    guard(out, pilot, gap, nets, stage2b)
    if (out / SUMMARY).exists():
        raise SystemExit(f"{out / SUMMARY} exists: the summary is never "
                         f"overwritten (move it aside first)")
    t0 = time.time()
    ctx = Context(out, pilot, gap, nets, stage2b)
    if not draw_file(out).exists():
        run_pass(ctx, "draw")
    _, kept = ctx.drawn()
    for k, _, _ in chunks(len(kept)):
        if not play_file(out, k).exists():
            run_pass(ctx, "play", k)
    for k, _, _ in chunks(len(kept)):
        if not search_file(out, k).exists():
            run_pass(ctx, "search", k)
    return finish(ctx, t0)


def collect(ctx, kept):
    """Every chunk's play and search records, checked against the draw."""
    plays, searches, cost = [], [], {"play": [], "search": []}
    for k, lo, hi in chunks(len(kept)):
        p = read(play_file(ctx.out, k), ctx.st)
        s = read(search_file(ctx.out, k), ctx.st)
        for name, f in (("play", p), ("search", s)):
            if f["range"] != [lo, hi] or f["kept"] != kept[lo:hi]:
                raise SystemExit(f"{name} chunk {k}: not drawn states "
                                 f"{lo}-{hi}")
            cost[name].append({"secs": f["secs"], **f.get("memory", {})})
        plays += p["states"]
        for j in range(hi - lo):
            searches.append({
                "raw": s["raw"][j], "acting": s["acting"][j],
                "top": {t: s["top"][t][j] for t in s["top"]},
                "chunk": {t: s["search"][t]["chunks"][j] for t in s["search"]},
                "visits": {t: float(np.sum(s["search"][t]["counts"][j]))
                           for t in s["search"]}})
    return plays, searches, cost


def checks(draw, plays, searches):
    """The path checks: [] when the deployed slot is t0.pkl's argmax, the
    deployed class at delay 0 gives stage 2a's T0 returns exactly, the
    acting prior through snapshot_run is the play pass's, and each budget
    spent its simulations on its chunk."""
    reasons = []
    zero = DELAYS.index(0)
    for i, (p, s) in enumerate(zip(plays, searches)):
        where = f"drawn state {i} (game {p['id'][0]} decision {p['id'][1]})"
        if p["deployed"] != draw["t0_argmax"][i]:
            reasons.append(f"{where}: deployed slot {p['deployed']}, t0.pkl's "
                           f"argmax {draw['t0_argmax'][i]}")
        c0 = p["partition"][p["deployed"]]
        if p["reps"][c0] != p["deployed"] or list(p["delays"]) != list(DELAYS):
            reasons.append(f"{where}: the deployed slot does not stand for "
                           f"its class at the registered delays")
        elif not np.array_equal(np.asarray(p["returns"])[c0, zero],
                                np.asarray(draw["t0_delay0"][i])):
            reasons.append(f"{where}: the deployed class at delay 0 is not "
                           f"stage 2a's T0 branch")
        diff = float(np.abs(np.asarray(s["acting"]) -
                            np.asarray(p["prior"])).max())
        if not diff <= CROSS_TOL:
            reasons.append(f"{where}: snapshot_run's acting prior differs "
                           f"from the deployed prior by {diff}")
        for t in ("50", "200"):
            if s["visits"][t] != float(t) or s["chunk"][t] != i // ROOTS:
                reasons.append(f"{where}: the {t}-simulation search spent "
                               f"{s['visits'][t]} visits in chunk "
                               f"{s['chunk'][t]}")
    return reasons


def finish(ctx, t0):
    draw, kept = ctx.drawn()
    plays, searches, cost = collect(ctx, kept)
    n = len(kept)
    fifth = np.array(draw["fifth"][:n])
    clusters = list(draw["clusters"][:n])
    reasons = checks(draw, plays, searches)
    base = {"smoke": SMOKE, "knobs": knobs(), "states": n,
            "draw": {k: draw[k] for k in ("sample", "strata", "pools",
                                          "drawn_per_fifth")},
            "stage2b": {"verdict": ctx.rec2b["verdict"]}}
    if reasons:
        rec = {"status": "VOID", "reasons": reasons, **base,
               "secs": round(time.time() - t0)}
        print(("SMOKE " if SMOKE else "") + f"READ (7): VOID "
              f"({'; '.join(reasons[:4])}; {len(reasons)} in all): Kai's "
              f"call", flush=True)
        return save(ctx, rec)
    per = [state_reads(p, s["top"]) for p, s in zip(plays, searches)]
    values = {k: np.array([r[k] for r in per]) for k in READS + BESIDE}
    reads = boot(values, fifth, clusters)
    rec = {"status": "READ", "reasons": [], **base,
           "fifth_counts": {f: int((fifth == f).sum()) for f in FIFTHS},
           "clusters": len(set(clusters)),
           "g_mean": {f: float(np.mean(np.asarray(draw["g"][:n])[fifth == f]))
                      for f in FIFTHS if (fifth == f).any()},
           "reads": reads,
           "per_state": [{"kept": kept[i], "fifth": str(fifth[i]),
                          **{k: per[i][k] for k in per[i]},
                          "top": searches[i]["top"]} for i in range(n)],
           "ended": {e: sum(b["ended"] == e for p in plays
                            for row in p["branches"] for b in row)
                     for e in ("life", "cap", "delay")},
           "cost": {"play_secs_per_state": round(sum(
                        c["secs"] for c in cost["play"]) / n, 2),
                    "search_secs_per_state": round(sum(
                        c["secs"] for c in cost["search"]) / n, 2),
                    "peak_wset_mb": max([c.get("peak_wset_mb", 0)
                                         for c in cost["play"] +
                                         cost["search"]] or [0]),
                    "peak_commit_mb": max([c.get("peak_commit_mb", 0)
                                           for c in cost["play"] +
                                           cost["search"]] or [0])},
           "memory": s2.memory(), "secs": round(time.time() - t0)}
    report(rec)
    return save(ctx, rec)


def save(ctx, rec):
    if (ctx.out / SUMMARY).exists():
        raise SystemExit(f"{ctx.out / SUMMARY} appeared: never overwritten")
    write(ctx.out / SUMMARY, rec, ctx.st)
    return rec


def _fmt(r, part):
    p, iv = r["point"][part], r["intervals"][part]
    ci = "[undefined]" if iv is None else f"[{iv[0]:+.3f}, {iv[1]:+.3f}]"
    return f"{p:+.3f} {ci}"


def report(rec):
    d = rec["draw"]
    print(f"the sample {d['sample']} states in {d['strata']} strata; fifths "
          f"by g inside strata: top {d['pools']['top']}, bottom "
          f"{d['pools']['bottom']}; drawn {d['drawn_per_fifth']}; "
          f"{rec['clusters']} clusters; mean g {rec['g_mean']}", flush=True)
    print(f"branches ended: {rec['ended']}; play {rec['cost']['play_secs_per_state']}"
          f" s a state, search {rec['cost']['search_secs_per_state']} s a "
          f"state, peak {rec['cost']['peak_wset_mb']} MB", flush=True)
    for k in READS + BESIDE:
        r = rec["reads"][k]
        print(f"(7) {k}: " + "; ".join(f"{q} {_fmt(r, q)}" for q in PARTS),
              flush=True)
    print(("SMOKE " if rec["smoke"] else "") + "READ (7): descriptive, read",
          flush=True)


def default_out():
    return HERE / f"probe_gap_read7{'-smoke' if SMOKE else ''}"


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("read7", "draw", "play", "search"))
    ap.add_argument("--chunk", type=int)
    for name in ("out", "pilot", "gap", "nets", "stage2b"):
        ap.add_argument("--" + name)
    args = ap.parse_args(argv)
    if LIMIT < 0 or DRAWS < 1:
        raise SystemExit(f"GAP7_STATES must be 0 or more and GAP7_DRAWS at "
                         f"least 1, got {LIMIT}, {DRAWS}")
    torch.set_num_threads(THREADS)
    res = lambda a, d: pathlib.Path(a).resolve() if a else d   # noqa: E731
    out = res(args.out, default_out())
    dirs = (res(args.pilot, PILOT), res(args.gap, GAP), res(args.nets, NETS),
            res(args.stage2b, STAGE2B))
    guard(out, *dirs)
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "read7":
        read7(out, *dirs)
        return
    ctx = Context(out, *dirs)
    if args.stage == "draw":
        stage_draw(ctx)
        return
    if args.chunk is None:
        raise SystemExit(f"{args.stage} needs --chunk")
    {"play": stage_play, "search": stage_search}[args.stage](ctx, args.chunk)


if __name__ == "__main__":
    main(sys.argv[1:])
