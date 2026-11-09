"""Test G, stage 2b: the value nets meet the kept states, and reads (1)-(6)
(CURRICULUM "Next experiment (pre-registered) -- Test G: does the
master-student value gap point at real weakness?", Kai 2026-11-07). Read (7)
is a later stage's. No net trains here and no emulator runs: the nets are
probe_gap_nets', the real gap G is stage 2a's (gap.pkl), the states are the
pilot's kept states (select.pkl).

THE GATES FIRST. Nothing runs unless stage 2a's gates.json says PASS-SO-FAR
(G3, G4, G5 each ok) and the nets' g2.json passed G2's first clause (as
amended 2026-11-07: R2 on all held-out stacks; read by g2_first_clause, which
needs "clause": "first", the verdict probe_gap_nets.G2_FIRST_PASS, and the
master's and the twin's own reads passed). On any
VOID it prints VOID with the reasons, writes the summary saying so, and stops
before any net meets a state: g is never formed (the GATES paragraph: VOID is
reported and goes to Kai, no retuning). Every pass checks the gates again.

THE PASSES, each a fresh process; a pass whose file exists is reused (its
stamp checked), so a stopped stage resumes:
  nets: master, twin and late twin (probe_gap_nets.load: 8 heads each,
      de-standardised, averaged over 3 seeds) on every kept state's stack,
      and whether its digest (probe_gap_nets.stack_digest) is in each net's
      training stacks.
  t0: T0's snapshot (probe_gap_check.snapshot, the one the pilot plays) on
      every kept stack. Prior entropy: the deployed prior (probe_gap_check.
      player: the acting BN states; network_manager.initial_inference's
      softmax, temperature 1). Value head: initial_inference's value on the
      trainer's eval-mode BN (probe_prefix_bn.snapshot_run, no acting
      overlay: the nets birth search runs in eval mode). Cross-check:
      snapshot_run with the acting overlay must give the deployed prior again
      (max |p - p'| <= CROSS_TOL), or the pass refuses.
  table: the position table, from the logs and no frames: each own-play log
      as the nets read it (probe_gap_nets.read_log's lives through the cut and
      life index, its totals asserted on a full run), every base step's
      measured return at the primary head in its stratum; the mean per
      stratum, T2's log and T0's.
THE READS, in this process, each group its own file (reused when there):
  g(s) = master - twin, per discount and scale (HEADS matched by name to G's
  (scale, discount) axes). PRIMARY HEAD 0.997 per frame, the student's clip.
  THE SAMPLE: the kept states minus those whose stack is in the master's or
  the twin's training stacks; those are printed beside (their count, and
  G2's inside clause and read (1) on all kept states). The late twin has its
  own split, so read (3) also leaves out the sample's states whose stack is
  in the late twin's training stacks (all of read (3), so its comparisons
  share their states); counted and printed.
  Intervals: two-sided 95% percentile, DRAWS cluster draws
  (probe_gap_stage2a.cluster_draws: the sample's clusters from gap.pkl, as
  many as there are, with replacement, each with all its states), seed
  BOOT_SEED, every statistic recomputed on each draw (rho_w's ranks inside
  strata included). The same draws serve every statistic, so "inside each
  draw" holds across them. A statistic undefined on any draw has no interval,
  and a verdict needing it fails.
  G2, inside strata: 95% lower bounds of rho_w(master net, T2's mean return)
      and rho_w(twin, T0's mean return), both above 0, else NOT SHOWN.
  (1) PRIMARY. Baselines: student prior entropy, twin value (primary head),
      T0's value head, base steps since the life began; each oriented once by
      the sign of its rho_w with G on the sample (0 or undefined: +1).
      HOLDS iff G2's inside clause passed, the lower bound of rho_w(g, G) is
      above 0 and the lower bound of rho_w(g, G) minus the best baseline's
      rho_w, the best taken inside each draw, is above 0. Else NOT SHOWN.
      Beside: the ceiling, sqrt of G5's reliability (gates.json), and G5 on
      the sample.
  (2) rho_w(master, G); rho_w(g, G) - rho_w(master, G); rho_w(g, T2's mean
      return); rho_w(g, -T0's mean return); rho_w(-T0's mean return, G).
  (3) rho_w(master - late twin, G); ranks only for T0's value head: inside
      each stratum the master's centred rank minus the value head's, its
      rho_w with G; the same rank gap with the twin beside it, like for like.
  (4) At g <= 0 against g > 0: mean G, the share with G <= 0, and the
      differences; read only with at least READ4_MIN states at g <= 0.
  (5) rho_w(g, G) at each discount on the clip, the raw scale beside; how
      far the four g agree: rho_w between each pair of discounts, per scale,
      and G's pairs as the reference.
  (6) Plain Spearman (one stratum) of g and of the position-table gap (T2's
      table minus T0's, the state's stratum) against G; states whose stratum
      either log lacks are left out of the table's read, and g is also read
      on those states alone.
Writes stage2b.json, the summary, never overwritten.

usage: python probe_gap_stage2b.py stage2b [--out DIR] [--gap DIR]
                                   [--nets DIR] [--pilot DIR]
       python probe_gap_stage2b.py nets|t0|table [the same options]
--gap is stage 2a's output (default probe_gap_stage2a next to this file),
--nets probe_gap_nets' (its default), --pilot the pilot's; DIR defaults to
probe_gap_stage2b. Knob GAP2B_DRAWS (10,000) marks a smoke run, as do stage
2a's and the nets' smoke knobs; smoke moves the default DIR to
probe_gap_stage2b-smoke. CPU only, 2 threads. Imported as a module (the
tests) it changes nothing process-wide."""
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
import itertools
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT / "experiments", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import numpy as np
import torch
import probe_e1_equivalence as e1
import probe_gap_check as gc
import probe_gap_nets as gn
import probe_gap_stage2a as s2

GAP = HERE / "probe_gap_stage2a"             # stage 2a's files, read only
PRIMARY = s2.PRIMARY                         # (scale, discount): clip, 0.997
NETS = ("master", "twin", "late")
EXCLUDED_BY = ("master", "twin")             # "either net"
BASELINES = ("prior_entropy", "twin_value", "value_head", "steps")
DRAWS = int(os.environ.get("GAP2B_DRAWS") or 10_000)
BOOT_SEED = (gc.SEED, 5)                     # 1-3 the pilot's, 4 stage 2a's
READ4_MIN = 30
CROSS_TOL = 1e-6
THREADS = 2
SUMMARY = "stage2b.json"
PASSES = ("nets", "t0", "table")
GROUPS = ("primary", "read2", "read3", "read4", "read5", "read6")
SMOKE = DRAWS != 10_000 or s2.SMOKE or gn.SMOKE
# This stage's source and the loader it reads T0's value head through; the
# stamps of stage 2a, the pilot and the nets (in every stamp) cover the rest.
SOURCES = (pathlib.Path(__file__).resolve(), HERE / "probe_prefix_bn.py")
SOURCE = gc.source_hash(SOURCES)


# --------------------------------------------------------------------------
# g, the sample, the baselines
# --------------------------------------------------------------------------
def head(scale, discount):
    """The nets' column for G's (scale, discount) axes."""
    return gn.HEADS.index((gc.SCALES[scale], gc.DISCOUNTS[discount]))


def gap_of(a, b):
    """a - b per head, laid out as G: (states, scales, discounts)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return np.stack([np.stack([a[:, head(s, d)] - b[:, head(s, d)]
                               for d in range(len(gc.DISCOUNTS))], axis=1)
                     for s in range(len(gc.SCALES))], axis=1)


def prior_entropy(policy):
    """Each row's entropy, nats (0 log 0 = 0)."""
    p = np.asarray(policy, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log(p), 0.0)
    return -terms.sum(axis=1)


def strata_ids(keys):
    """Each state's stratum as a small int (rho_w only groups by it)."""
    ids = {}
    return np.array([ids.setdefault(tuple(k), len(ids)) for k in keys])


def rank_gap(a, b, sid):
    """Inside each stratum, a's centred rank minus b's (ties averaged)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    out, groups = np.zeros(len(a)), {}
    for i, k in enumerate(sid):
        groups.setdefault(k, []).append(i)
    for m in groups.values():
        out[m] = gc.centred_ranks(a[m]) - gc.centred_ranks(b[m])
    return out


def _rho(x, y, sid):
    r = gc.rho_w(x, y, sid)
    return np.nan if r is None else r


def orientation(data):
    """Each baseline's sign, once, on the whole sample: that of its rho_w
    with G (+1 at 0 or undefined)."""
    sid = list(data["sid"])
    signs = {}
    for b in BASELINES:
        r = gc.rho_w(data[b], data["G"], sid)
        signs[b] = -1.0 if r is not None and r < 0 else 1.0
    return signs


# --------------------------------------------------------------------------
# the reads: each a function of the data and one draw's indices
# --------------------------------------------------------------------------
def read_primary(d, idx, signs):
    """G2's inside clause and read (1) on the states `idx`."""
    sid = d["sid"][idx].tolist()
    G = d["G"][idx]
    out = {"g2_master": _rho(d["master"][idx], d["t2_mean"][idx], sid),
           "g2_twin": _rho(d["twin"][idx], d["t0_mean"][idx], sid),
           "g": _rho(d["g"][idx], G, sid)}
    for b in BASELINES:
        out["b:" + b] = _rho(signs[b] * d[b][idx], G, sid)
    best = [out["b:" + b] for b in BASELINES]
    out["best"] = max(best) if np.all(np.isfinite(best)) else np.nan
    out["g_minus_best"] = out["g"] - out["best"]
    return out


def read_2(d, idx, signs=None):
    sid = d["sid"][idx].tolist()
    G, g = d["G"][idx], d["g"][idx]
    out = {"master": _rho(d["master"][idx], G, sid), "g": _rho(g, G, sid)}
    out["g_minus_master"] = out["g"] - out["master"]
    out["g_vs_t2"] = _rho(g, d["t2_mean"][idx], sid)
    out["g_vs_neg_t0"] = _rho(g, -d["t0_mean"][idx], sid)
    out["neg_t0_vs_G"] = _rho(-d["t0_mean"][idx], G, sid)
    return out


def read_3(d, idx, signs=None):
    """On the states not in the late twin's training stacks only."""
    idx = idx[d["late_ok"][idx]]
    sid = d["sid"][idx].tolist()
    G, m = d["G"][idx], d["master"][idx]
    return {"twin": _rho(d["g"][idx], G, sid),
            "late": _rho(d["g_late"][idx], G, sid),
            "value_head_ranks": _rho(rank_gap(m, d["value_head"][idx], sid),
                                     G, sid),
            "twin_ranks": _rho(rank_gap(m, d["twin"][idx], sid), G, sid)}


def _mean(x):
    return float(np.mean(x)) if len(x) else np.nan


def read_4(d, idx, signs=None):
    g, G = d["g"][idx], d["G"][idx]
    low = g <= 0
    out = {"n_low": float(low.sum()),
           "mean_G_low": _mean(G[low]), "mean_G_rest": _mean(G[~low]),
           "share_low": _mean(G[low] <= 0), "share_rest": _mean(G[~low] <= 0)}
    out["mean_diff"] = out["mean_G_low"] - out["mean_G_rest"]
    out["share_diff"] = out["share_low"] - out["share_rest"]
    return out


def read4_counts(g):
    """Read (4) is read only with at least READ4_MIN states at g <= 0."""
    low = int((np.asarray(g) <= 0).sum())
    return {"low": low, "min": READ4_MIN, "read": low >= READ4_MIN}


def read_5(d, idx, signs=None):
    sid = d["sid"][idx].tolist()
    g, G = d["g_all"][idx], d["G_all"][idx]
    out = {}
    for s, sc in enumerate(gc.SCALES):
        for k, dc in enumerate(gc.DISCOUNTS):
            out[f"{sc} {dc}"] = _rho(g[:, s, k], G[:, s, k], sid)
    for s, sc in enumerate(gc.SCALES):
        for a, b in itertools.combinations(range(len(gc.DISCOUNTS)), 2):
            pair = f"{sc} {gc.DISCOUNTS[a]}-{gc.DISCOUNTS[b]}"
            out["g agree " + pair] = _rho(g[:, s, a], g[:, s, b], sid)
            out["G agree " + pair] = _rho(G[:, s, a], G[:, s, b], sid)
    return out


def read_6(d, idx, signs=None):
    """Plain Spearman: rho_w with one stratum."""
    G, g = d["G"][idx], d["g"][idx]
    t = d["has_table"][idx]
    one = [0] * len(idx)
    return {"g": _rho(g, G, one),
            "table": _rho(d["table"][idx][t], G[t], one[:int(t.sum())]),
            "g_on_table_states": _rho(g[t], G[t], one[:int(t.sum())])}


READS = {"primary": read_primary, "read2": read_2, "read3": read_3,
         "read4": read_4, "read5": read_5, "read6": read_6}


def interval(values):
    """The two-sided 95% percentile interval; None when any draw is
    undefined."""
    v = np.asarray(values, np.float64)
    if not len(v) or not np.all(np.isfinite(v)):
        return None
    lo, hi = np.percentile(v, s2.INTERVAL)
    return [float(lo), float(hi)]


def draws_of(clusters, draws=None):
    return s2.cluster_draws(clusters, DRAWS if draws is None else draws,
                            seed=BOOT_SEED)


def boot(read, d, clusters, signs, draws=None):
    """A read on the whole sample and on every cluster draw: {"names",
    "point" {name: value}, "draws" (draws, names), "intervals" {name:
    [lo, hi] | None}, "undefined" {name: draws undefined}}."""
    point = read(d, np.arange(len(d["G"])), signs)
    names = list(point)
    rows = [[r[n] for n in names] for r in
            (read(d, idx, signs) for idx in draws_of(clusters, draws))]
    vals = np.array(rows, np.float64).reshape(len(rows), len(names))
    return {"names": names, "point": {n: float(point[n]) for n in names},
            "draws": vals,
            "intervals": {n: interval(vals[:, j]) for j, n in enumerate(names)},
            "undefined": {n: int((~np.isfinite(vals[:, j])).sum())
                          for j, n in enumerate(names)}}


def primary_verdict(iv):
    """("HOLDS" | "NOT SHOWN", reasons) from read_primary's intervals."""
    def low(name):
        return None if iv.get(name) is None else iv[name][0]

    reasons = []
    for name, what in (("g2_master", "rho_w(master net, T2's mean return)"),
                       ("g2_twin", "rho_w(twin, T0's mean return)")):
        lo = low(name)
        if lo is None or not lo > 0:
            reasons.append(f"G2 inside strata: {what} lower bound {lo}, not "
                           f"above 0")
    for name, what in (("g", "rho_w(g, G)"),
                       ("g_minus_best", "rho_w(g, G) minus the best "
                                        "baseline's")):
        lo = low(name)
        if lo is None or not lo > 0:
            reasons.append(f"(1): {what} lower bound {lo}, not above 0")
    return ("NOT SHOWN", reasons) if reasons else ("HOLDS", [])


def sample_g5(d):
    """G5 on the sample: probe_gap_check.split_half of the per-delay gap."""
    r, rel = gc.split_half(d["per_delay"], list(d["sid"]))
    return {"rho_half": r, "reliability": rel,
            "ceiling": None if rel is None or rel < 0 else float(np.sqrt(rel))}


# --------------------------------------------------------------------------
# files
# --------------------------------------------------------------------------
def knobs():
    return {"draws": DRAWS}


def input_files(gap, nets):
    gap, nets = pathlib.Path(gap), pathlib.Path(nets)
    return [gap / "gap.pkl", gap / "gates.json", nets / "g2.json"] + [
        gn.net_file(nets, n, s) for n in NETS for s in gn.SEEDS]


def stamp(states, gap, nets, files=None):
    """What made a file: this stage's knobs and source, stage 2a's stamp (the
    pilot's and the states' inside it), the nets' stamp, and the digest of
    every input file read."""
    files = input_files(gap, nets) if files is None else files
    missing = [str(f) for f in files if not pathlib.Path(f).exists()]
    if missing:
        raise SystemExit(f"missing inputs: {missing}")
    return {"knobs": knobs(), "smoke": SMOKE, "source": SOURCE,
            "gap": s2.stamp(states), "nets": gn.stamp(),
            "inputs": {pathlib.Path(f).name: s2.file_digest(f)
                       for f in files}}


write, read = s2.write, s2.read              # whole or nothing; stamp-checked


def guard(out, *read_only):
    """No output inside the pilot's, stage 2a's or the nets' files, nor in
    the read-only run tree."""
    o = pathlib.Path(out).resolve()
    keep = {pathlib.Path(p).resolve() for p in read_only}
    keep |= {s2.PILOT.resolve(), gc.RESULTS.resolve().parents[1]}
    for p in keep:
        if o == p or p in o.parents:
            raise SystemExit(f"{out} is inside {p}, which this stage reads "
                             f"and never writes")


def pass_file(out, name):
    return pathlib.Path(out) / f"{name}.pkl"


def group_file(out, name):
    return pathlib.Path(out) / f"reads-{name}.pkl"


# --------------------------------------------------------------------------
# the gates
# --------------------------------------------------------------------------
def gate_reasons(gates, g2=None):
    """VOID's reasons, [] when stage 2a passed so far and (g2 given) G2's
    first clause passed."""
    reasons = []
    if gates.get("verdict") != "PASS-SO-FAR":
        reasons += [f"stage 2a: {r}" for r in gates.get("reasons") or
                    [f"verdict {gates.get('verdict')}"]]
    else:
        failed = [k for k in ("G3", "G5") if not gates[k]["ok"]] + [
            f"G4 {n}" for n, g in gates["G4"].items() if not g["ok"]]
        if failed or set(gates["G4"]) != set(s2.PLAYERS):
            reasons.append(f"stage 2a: PASS-SO-FAR, but {failed} failed or "
                           f"G4 is not both players'")
    if reasons or g2 is None:
        return reasons
    return g2_first_clause(g2)


def g2_first_clause(g2):
    """G2's first clause from the nets' g2.json (amended 2026-11-07: R2 on
    all held-out stacks): VOID's reasons, [] when it passed. The file must
    say it is the first clause ("clause": "first") and its verdict must be
    probe_gap_nets.G2_FIRST_PASS exactly (G2_FIRST_VOID or anything else
    fails); and whatever the verdict says, the master and the twin must each
    have passed on their own read: "pass" true and "r2" (the all-held-out
    R2) above 0. The late twin gates nothing."""
    verdict = str(g2.get("verdict"))
    if g2.get("clause") != "first":
        return [f"G2: g2.json's clause is {g2.get('clause')!r}, not 'first'"]
    if verdict != gn.G2_FIRST_PASS:
        return [f"G2: {r}" for r in g2.get("reasons") or
                [f"verdict {verdict}, not {gn.G2_FIRST_PASS}"]]
    reasons = []
    for n in ("master", "twin"):
        r = (g2.get("reads") or {}).get(n) or {}
        r2 = r.get("r2")
        if r.get("pass") is not True or isinstance(r2, bool) or \
                not isinstance(r2, (int, float)) or not r2 > 0:
            reasons.append(f"G2: verdict {verdict}, but {n}'s read is pass "
                           f"{r.get('pass')}, R2 {r2}")
    return reasons


def read_gates(gap, nets, states):
    """(gates, g2 or None, the files read, VOID's reasons). g2.json is read
    only when stage 2a passed so far."""
    gap, nets = pathlib.Path(gap), pathlib.Path(nets)
    if not (gap / "gates.json").exists():
        raise SystemExit(f"{gap / 'gates.json'}: stage 2a has not finished")
    gates = read(gap / "gates.json", s2.stamp(states))
    reasons, g2, files = gate_reasons(gates), None, [gap / "gates.json"]
    if not reasons:
        if not (nets / "g2.json").exists():
            raise SystemExit(f"{nets / 'g2.json'}: the nets have not "
                             f"finished")
        g2 = gn.read(nets / "g2.json")
        files.append(nets / "g2.json")
        reasons = gate_reasons(gates, g2)
    return gates, g2, files, reasons


def require_pass(gap, nets, states):
    """A pass forms nothing on a VOID."""
    _, _, _, reasons = read_gates(gap, nets, states)
    if reasons:
        raise SystemExit(f"VOID ({'; '.join(reasons)}): stage 2b does not "
                         f"run")


# --------------------------------------------------------------------------
# the passes
# --------------------------------------------------------------------------
def setup(out, gap, nets, pilot):
    guard(out, pilot, gap, nets)
    _, states, keys = s2.load_states(pilot)
    require_pass(gap, nets, states)
    return states, keys, stamp(states, gap, nets)


def stage_nets(out, gap, nets, pilot):
    states, _, st = setup(out, gap, nets, pilot)
    t0 = time.time()
    stacks = np.stack([s["stack"] for s in states])
    digests = [gn.stack_digest(x) for x in stacks]
    pred, training = {}, {}
    for net in NETS:
        model = gn.load(net, nets)
        pred[net] = np.asarray(model.predict(stacks), np.float64)
        training[net] = np.array([x in model.train_digests for x in digests])
        if pred[net].shape != (len(states), len(gn.HEADS)):
            raise SystemExit(f"{net}: predictions {pred[net].shape}")
    write(pass_file(out, "nets"),
          {"states": s2.ids(states), "heads": [list(h) for h in gn.HEADS],
           "pred": pred, "in_training": training,
           "secs": round(time.time() - t0, 1), "memory": s2.memory()}, st)


def deployed_priors(run, stacks):
    """initial_inference's softmax (temperature 1) for each stack."""
    m = run.searcher.network_manager
    return np.array([m.initial_inference(x)[0] for x in stacks], np.float64)


def value_heads(run, stacks):
    m = run.searcher.network_manager
    return np.array([m.initial_inference(x)[1] for x in stacks], np.float64)


def load_blob(path):
    return torch.load(str(path), map_location="cpu", weights_only=False)


def trainer_run(blob, acting=False):
    """probe_prefix_bn.snapshot_run: the snapshot's own arm, its trainer
    weights in eval mode; acting=True overlays its acting states."""
    import probe_exhaustive_eval  # noqa: F401  sets ablation's env globals
    import probe_prefix_bn as pb
    return pb.snapshot_run(blob, acting=acting)


def stage_t0(out, gap, nets, pilot):
    states, _, st = setup(out, gap, nets, pilot)
    t0 = time.time()
    stacks = [s["stack"] for s in states]
    with gc.player("T0") as run:
        prior = deployed_priors(run, stacks)
    blob = load_blob(gc.snapshot("T0"))
    with trainer_run(blob) as (run, missing):
        value = value_heads(run, stacks)
    with trainer_run(blob, acting=True) as (run, _):
        again = deployed_priors(run, stacks)
    del blob
    if prior.shape != again.shape:
        raise SystemExit(f"the deployed prior {prior.shape}, snapshot_run's "
                         f"acting prior {again.shape}")
    diff = float(np.abs(prior - again).max()) if prior.size else 0.0
    if not diff <= CROSS_TOL:
        raise SystemExit(f"snapshot_run's acting prior differs from the "
                         f"deployed prior by {diff} (> {CROSS_TOL}): its "
                         f"value head is not T0 as the pilot plays it")
    write(pass_file(out, "t0"),
          {"states": s2.ids(states), "prior_entropy": prior_entropy(prior),
           "prior_argmax": prior.argmax(axis=1), "value_head": value,
           "cross_check": diff, "missing_spec_keys": list(missing),
           "snapshot": str(gc.snapshot("T0")),
           "secs": round(time.time() - t0, 1), "memory": s2.memory()}, st)


def log_returns(player, results=None, max_lives=None):
    """A player's own-play log as probe_gap_nets.read_log reads it -- the
    lives through the cut, each with its place in its game -- without
    frames: ([{"index", "returns" at the primary head per base step}],
    totals)."""
    spec = gn.LOGS[player]
    folder = gn.log_folder(player, results)
    gn.check_files(folder, spec["last"])
    max_lives = gn.LIVES if max_lives is None else max_lives
    lives, index = [], 0
    totals = {"lives": 0, "base_steps": 0, "raw": 0.0, "later_lives": 0}
    for it, life in e1.iter_own_play(folder, spec["last"], max_lives):
        where = f"{player}'s log, iteration {it}, life {totals['lives'] + 1}"
        if int(life["iteration"]) != it:
            raise SystemExit(f"{where}: logged as iteration "
                             f"{life['iteration']}")
        try:
            rewards = gn.reward_tuples(life)
        except ValueError as err:
            raise SystemExit(f"{where}: {err}")
        flat = [float(x) for r in rewards for x in r]
        totals["lives"] += 1
        totals["base_steps"] += len(flat)
        totals["raw"] += float(sum(flat))
        if it <= spec["cut"]:
            if len(flat) != int(life["steps"]):      # as read_log refuses it
                raise SystemExit(f"{where}: {len(flat)} base steps by the "
                                 f"rewards, {life['steps']} logged")
            lives.append({"index": index, "returns":
                          gn.life_returns(flat)[:, gn.PRIMARY]})
        else:
            totals["later_lives"] += 1
        index = 0 if life["real_done"] else index + 1
    return lives, totals


def position_table(lives):
    """{stratum: [mean return, base steps]} over every logged base step."""
    acc = {}
    for life in lives:
        for t, r in enumerate(life["returns"]):
            k = gc.stratum(life["index"], t)
            s, n = acc.get(k, (0.0, 0))
            acc[k] = (s + float(r), n + 1)
    return {k: [s / n, n] for k, (s, n) in acc.items()}


def stage_table(out, gap, nets, pilot):
    states, _, st = setup(out, gap, nets, pilot)
    t0, tables, totals = time.time(), {}, {}
    for player in ("T2", "T0"):
        lives, totals[player] = log_returns(player)
        if not gn.SMOKE:
            gn.check_totals(player, lives, totals[player])
        tables[player] = position_table(lives)
        del lives
    write(pass_file(out, "table"),
          {"tables": tables, "totals": totals,
           "secs": round(time.time() - t0, 1), "memory": s2.memory()}, st)


def table_gap(tables, keys):
    """Each state's position-table gap, T2's stratum mean minus T0's; NaN
    where either log lacks the stratum."""
    out = np.full(len(keys), np.nan)
    for i, k in enumerate(keys):
        k = tuple(k)
        if k in tables["T2"] and k in tables["T0"]:
            out[i] = tables["T2"][k][0] - tables["T0"][k][0]
    return out


# --------------------------------------------------------------------------
# the stage
# --------------------------------------------------------------------------
def run_pass(out, gap, nets, pilot, name):
    """A pass in its own process; a failed pass stops the stage."""
    cmd = [sys.executable, "-u", str(pathlib.Path(__file__).resolve()), name,
           "--out", str(out), "--gap", str(gap), "--nets", str(nets),
           "--pilot", str(pilot)]
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode:
        raise SystemExit(f"pass {name}: rc {r.returncode}")


def void(out, states, gates, g2, files, reasons, t0):
    st = stamp(states, None, None, files=files)
    rec = {"verdict": "VOID", "reasons": reasons, "smoke": SMOKE,
           "knobs": knobs(), "states": len(states),
           "gates": {"verdict": gates["verdict"],
                     "reasons": gates["reasons"]},
           "G2": None if g2 is None else {"clause": g2.get("clause"),
                                           "verdict": g2.get("verdict"),
                                           "reasons": g2.get("reasons")},
           "secs": round(time.time() - t0)}
    print(("SMOKE " if SMOKE else "") + f"STAGE 2b: VOID ({'; '.join(reasons)})"
          f": no g is formed; Kai's call", flush=True)
    if (pathlib.Path(out) / SUMMARY).exists():
        raise SystemExit(f"{pathlib.Path(out) / SUMMARY} appeared: never "
                         f"overwritten")
    write(pathlib.Path(out) / SUMMARY, rec, st)
    return rec


def stage2b(out, gap, nets, pilot):
    out, gap, nets, pilot = (pathlib.Path(p) for p in (out, gap, nets, pilot))
    guard(out, pilot, gap, nets)
    if (out / SUMMARY).exists():
        raise SystemExit(f"{out / SUMMARY} exists: the summary is never "
                         f"overwritten (move it aside first)")
    t0 = time.time()
    _, states, _ = s2.load_states(pilot)
    gates, g2, files, reasons = read_gates(gap, nets, states)
    if reasons:
        return void(out, states, gates, g2, files, reasons, t0)
    for name in PASSES:
        if not pass_file(out, name).exists():
            run_pass(out, gap, nets, pilot, name)
    return finish(out, gap, nets, pilot, gates, g2, t0)


def assemble(states, keys, gap, nets_rec, t0_rec, table_rec):
    """Every input, per kept state, checked against the states: the arrays
    the reads take (all kept states) and the training-overlap masks."""
    ids = s2.ids(states)
    for name, rec in (("gap.pkl", gap), ("nets.pkl", nets_rec),
                      ("t0.pkl", t0_rec)):
        if rec["states"] != ids:
            raise SystemExit(f"{name}: not the kept states")
    if [tuple(k) for k in gap["keys"]] != [tuple(k) for k in keys] or \
            gap["delays"] != list(s2.DELAYS) or \
            list(gap["scales"]) != list(gc.SCALES) or \
            list(gap["discounts"]) != list(gc.DISCOUNTS):
        raise SystemExit("gap.pkl: other strata, delays or heads")
    if [tuple(h) for h in nets_rec["heads"]] != list(gn.HEADS):
        raise SystemExit("nets.pkl: other heads")
    t0r = np.asarray(gap["returns"]["T0"], np.float64)
    t2r = np.asarray(gap["returns"]["T2"], np.float64)
    G = np.asarray(gap["G"], np.float64)
    if G.shape != (len(states), len(gc.SCALES), len(gc.DISCOUNTS)) or \
            not np.allclose(G, s2.real_gap(t0r, t2r), rtol=0, atol=1e-9):
        raise SystemExit("gap.pkl: G is not T2's mean return minus T0's")
    p = {n: np.asarray(nets_rec["pred"][n], np.float64) for n in NETS}
    g, g_late = gap_of(p["master"], p["twin"]), gap_of(p["master"], p["late"])
    sc, dc = PRIMARY
    h = head(sc, dc)
    data = {"G": G[:, sc, dc], "g": g[:, sc, dc], "g_late": g_late[:, sc, dc],
            "G_all": G, "g_all": g,
            "master": p["master"][:, h], "twin": p["twin"][:, h],
            "late": p["late"][:, h],
            "t2_mean": t2r[:, :, sc, dc].mean(axis=1),
            "t0_mean": t0r[:, :, sc, dc].mean(axis=1),
            "per_delay": t2r[:, :, sc, dc] - t0r[:, :, sc, dc],
            "prior_entropy": np.asarray(t0_rec["prior_entropy"], np.float64),
            "twin_value": p["twin"][:, h],
            "value_head": np.asarray(t0_rec["value_head"], np.float64),
            "steps": np.array([s["steps"] for s in states], np.float64),
            "table": table_gap(table_rec["tables"], keys),
            "sid": strata_ids(keys),
            "clusters": np.asarray(gap["clusters"])}
    data["has_table"] = np.isfinite(data["table"])
    training = {n: np.asarray(nets_rec["in_training"][n], bool)
                for n in NETS}
    data["late_ok"] = ~training["late"]          # read (3)'s states
    return data, training


def subset(data, mask):
    return {k: v[mask] for k, v in data.items()}


def read_groups(out, data, signs, st, names=GROUPS, prefix="", draws=None):
    """Each group's boot(), from its file when there."""
    res = {}
    for name in names:
        path = group_file(out, prefix + name)
        if path.exists():
            res[name] = read(path, st)["boot"]
            continue
        t0 = time.time()
        res[name] = boot(READS[name], data, data["clusters"], signs, draws)
        write(path, {"boot": res[name], "signs": signs,
                     "secs": round(time.time() - t0, 1)}, st)
        print(f"reads {prefix + name}: {DRAWS if draws is None else draws} "
              f"draws [{time.time() - t0:.0f}s]", flush=True)
    return res


def finish(out, gap_dir, nets_dir, pilot, gates, g2, t0):
    _, states, keys = s2.load_states(pilot)
    st = stamp(states, gap_dir, nets_dir)
    gap = read(pathlib.Path(gap_dir) / "gap.pkl", s2.stamp(states))
    recs = {n: read(pass_file(out, n), st) for n in PASSES}
    data, training = assemble(states, keys, gap, recs["nets"], recs["t0"],
                              recs["table"])
    left = np.zeros(len(states), bool)
    for n in EXCLUDED_BY:
        left |= training[n]
    sample = subset(data, ~left)
    if not len(sample["G"]):
        raise SystemExit("every kept state is in a net's training stacks")
    signs = orientation(sample)
    res = read_groups(out, sample, signs, st)
    beside = (read_groups(out, data, signs, st, ("primary",), "all-")
              ["primary"] if left.any() else None)
    verdict, reasons = primary_verdict(res["primary"]["intervals"])
    r4 = read4_counts(sample["g"])
    rel = gates["G5"]["reliability"]
    rec = {
        "verdict": verdict, "reasons": reasons, "smoke": SMOKE,
        "knobs": knobs(), "states": len(states),
        "sample": {"states": int((~left).sum()),
                   "strata": len(set(sample["sid"].tolist())),
                   "clusters": len(set(sample["clusters"].tolist()))},
        "left_out": {**{n: int(training[n].sum()) for n in EXCLUDED_BY},
                     "either": int(left.sum()),
                     "late_in_sample": int(training["late"][~left].sum()),
                     "states": [i for i, x in zip(s2.ids(states), left)
                                if x]},
        "read3_states": int(sample["late_ok"].sum()),
        "gates": {"verdict": gates["verdict"], "G3": gates["G3"],
                  "G5": gates["G5"]},
        "G2_first": {"clause": g2.get("clause"), "verdict": g2.get("verdict"),
                     **{n: (g2.get("reads") or {}).get(n, {}).get("r2")
                        for n in gn.NETS}},
        "ceiling": None if rel is None or rel < 0 else float(np.sqrt(rel)),
        "G5_sample": sample_g5(sample), "signs": signs,
        "reads": {n: {k: res[n][k] for k in ("point", "intervals",
                                             "undefined")} for n in GROUPS},
        "beside": None if beside is None else {
            k: beside[k] for k in ("point", "intervals", "undefined")},
        "read4": r4,
        "table_states": int(sample["has_table"].sum()),
        "t0": {"cross_check": recs["t0"]["cross_check"],
               "missing_spec_keys": recs["t0"]["missing_spec_keys"]},
        "cost": {n: {"secs": recs[n]["secs"], "memory": recs[n]["memory"]}
                 for n in PASSES},
        "memory": s2.memory(), "secs": round(time.time() - t0)}
    if not rec["read4"]["read"]:
        rec["reads"]["read4"] = None
    report(rec)
    if (pathlib.Path(out) / SUMMARY).exists():
        raise SystemExit(f"{pathlib.Path(out) / SUMMARY} appeared: never "
                         f"overwritten")
    write(pathlib.Path(out) / SUMMARY, rec, st)
    return rec


def _fmt(read, name):
    p, iv = read["point"][name], read["intervals"][name]
    ci = "[undefined]" if iv is None else f"[{iv[0]:+.3f}, {iv[1]:+.3f}]"
    return f"{p:+.3f} {ci}"


def report(rec):
    r, lo = rec["reads"], rec["left_out"]
    p = r["primary"]
    print(f"{rec['states']} kept states; left out for training overlap "
          f"{lo['either']} (master {lo['master']}, twin {lo['twin']}); late "
          f"twin's training stacks in the sample {lo['late_in_sample']} (left "
          f"out of read (3) only, which reads {rec['read3_states']}); the "
          f"sample {rec['sample']['states']} states in "
          f"{rec['sample']['strata']} strata, {rec['sample']['clusters']} "
          f"clusters", flush=True)
    print("G2 first clause (g2.json), R2 at the primary head on all held-out "
          "stacks: " + ", ".join(f"{n} {v}" for n, v in
                                 rec["G2_first"].items()), flush=True)
    print(f"G2 inside strata: rho_w(master net, T2's mean return) "
          f"{_fmt(p, 'g2_master')}; rho_w(twin, T0's mean return) "
          f"{_fmt(p, 'g2_twin')}", flush=True)
    print("(1) baselines, oriented on the sample: " + ", ".join(
        f"{b} {'+' if rec['signs'][b] > 0 else '-'}" for b in BASELINES),
        flush=True)
    print(f"(1) rho_w(g, G) {_fmt(p, 'g')}; " + "; ".join(
        f"{b} {_fmt(p, 'b:' + b)}" for b in BASELINES), flush=True)
    print(f"(1) rho_w(g, G) minus the best baseline's (the best inside each "
          f"draw) {_fmt(p, 'g_minus_best')}; ceiling sqrt(G5) "
          f"{rec['ceiling']}; G5 on the sample "
          f"{rec['G5_sample']['reliability']} (ceiling "
          f"{rec['G5_sample']['ceiling']})", flush=True)
    if rec["beside"] is not None:
        b = rec["beside"]
        print(f"    beside, all {rec['states']} kept states: rho_w(g, G) "
              f"{_fmt(b, 'g')}; minus the best {_fmt(b, 'g_minus_best')}; "
              f"G2 inside {_fmt(b, 'g2_master')}, {_fmt(b, 'g2_twin')}",
              flush=True)
    for n, label in (("read2", "(2)"), ("read3", "(3)"), ("read4", "(4)"),
                     ("read5", "(5)"), ("read6", "(6)")):
        if r[n] is None:
            print(f"(4) {rec['read4']['low']} states at g <= 0, under "
                  f"{READ4_MIN}: not read", flush=True)
            continue
        for k in r[n]["point"]:
            print(f"{label} {k} {_fmt(r[n], k)}", flush=True)
    print(f"(6) the table's read on {rec['table_states']} states whose "
          f"stratum both logs hold", flush=True)
    line = f"STAGE 2b PRIMARY: {rec['verdict']}" + (
        f" ({'; '.join(rec['reasons'])})" if rec["reasons"] else "")
    print(("SMOKE " if rec["smoke"] else "") + line, flush=True)


def default_out():
    return HERE / f"probe_gap_stage2b{'-smoke' if SMOKE else ''}"


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("stage2b",) + PASSES)
    ap.add_argument("--out")
    ap.add_argument("--gap")
    ap.add_argument("--nets")
    ap.add_argument("--pilot")
    args = ap.parse_args(argv)
    if DRAWS < 1:
        raise SystemExit(f"GAP2B_DRAWS must be at least 1, got {DRAWS}")
    torch.set_num_threads(THREADS)
    res = lambda a, d: pathlib.Path(a).resolve() if a else d   # noqa: E731
    out = res(args.out, default_out())
    gap, nets = res(args.gap, GAP), res(args.nets, gn.default_out())
    pilot = res(args.pilot, s2.PILOT)
    guard(out, pilot, gap, nets)
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "stage2b":
        stage2b(out, gap, nets, pilot)
        return
    {"nets": stage_nets, "t0": stage_t0, "table": stage_table}[args.stage](
        out, gap, nets, pilot)


if __name__ == "__main__":
    main(sys.argv[1:])
