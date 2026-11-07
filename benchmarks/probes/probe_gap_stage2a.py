"""Test G, stage 2a: the REAL GAP on every kept state, and the gates that need
only branches (CURRICULUM "Next experiment (pre-registered) -- Test G: does
the master-student value gap point at real weakness?", Kai 2026-11-07). No
value net, no g, no read (1)-(7): those are a later stage's. Measurement only:
the emulator scores, and no branch feeds a model (the E line's rule).

It consumes the pilot's files (probe_gap_check.py, verdict GO) and never
writes them: select.pkl's kept states (1,121; the pilot's 60 are among them and
are played again), main-080.pkl's main lines (G4's delay-0 check and the
clusters), pilot.json and g1-*.json (G1 and G6, which the pilot read). They are
read through probe_gap_check.read, so a pilot file made by other code or knobs
is refused. This stage lives in its own file because the pilot's stamps carry
a hash of probe_gap_check.py: editing that file would lock its own files.

REAL GAP: each kept state gets both players' K = 16 branches through
probe_gap_check.branch, exactly as the pilot's branch pass plays them: 0-15 NOOP
frames on raw ALE, the same 16 delays for both players, then the player's
deployed policy (the prior argmax on the snapshot's acting BN states) to the
life's end, capped at 1,536 frames with the delay. Each branch's return on all
8 heads is kept: discounts 0.997, 0.99, 0.98, 0.95 per frame (base step j weighs
gamma ** (4 j), the delay's rewards in the first base step), on the student's
clip (T0's ceiling 1 per base step, in units of REWARD_UNIT 10) and raw.
G(s) = T2's mean over the 16 delays minus T0's, per discount and scale.

THE PASSES, each a fresh process, CHUNK states at a time. `stage2a` runs the
missing ones in order and then reads the gates. A pass whose file exists is
reused, so a stopped stage resumes; every file is stamped with this stage's
knobs and source hash, the pilot's stamp and a fingerprint of the states it
consumed, and one made otherwise is refused rather than mixed in:
  branches P K: chunk K's states, P's 16 branches each. One seeded delay per
      state (probe_gap_check.g4_delays over all the stage's states) is also
      replayed on raw ALE (probe_action_census.play_window), its frames kept.
  g4 P K: G4. That delay's branch again in a fresh process, policy included:
      the same tokens, rewards, stacks, raw frames and end state. For T0 also
      every delay-0 branch against the main line it left
      (probe_gap_check.main_line_reasons). Any failure is VOID. The file
      names the branches file it replayed by its digest, so a branches file
      made again after its G4 is refused.
THE GATES (G1 and G6 were read by the pilot; G2 needs the nets):
  G3: mean G at the primary head (0.997 per frame, the clip) above 0, and the
      lower end of its two-sided 95% percentile interval above 0. The interval
      resamples CLUSTERS: games connected through shared stacks after the
      opening (any decision stack two main lines share joins them, and chains
      join). 10,000 draws, each of as many clusters as there are, with
      replacement, each bringing all its states (twice if drawn twice); the
      statistic is the mean over the states drawn.
  G5: probe_gap_check.split_half on the per-delay gap at the primary head:
      rho_w of the mean over delays 0-7 against delays 8-15, inside the strata
      (life index x 20-base-step bins; a stratum of one drops out),
      Spearman-Brown at least 0.5. The lag profile prints beside it: the
      residual correlation (a branch minus its state's 16-delay mean, pooled
      over the delay pairs) at every lag 1-15, for each player and for the
      per-delay gap; the null is -1/15.
  Any failure, or an undefined statistic: VOID, with its reasons. Else
  PASS-SO-FAR: G2 is still to read.
Writes gap.pkl (the per-branch returns, G, each state's stratum and cluster,
for the next stage) and gates.json, the summary, which is never overwritten.

usage: python probe_gap_stage2a.py stage2a [--out DIR] [--pilot DIR]
       python probe_gap_stage2a.py branches|g4 --player T0|T2 --chunk K [...]
DIR defaults to probe_gap_stage2a next to this file; --pilot to
probe_gap_check-pilot. Knobs: GAP2_CHUNK (states a pass, 100), GAP2_STATES (the
first N kept states only; 0, all). GAP2_STATES marks the record smoke and moves
the default DIR to probe_gap_stage2a-smoke. CPU only, 2 threads, as the
pilot's passes ran. Imported as a module (the tests) it changes nothing
process-wide."""
import os
# probe_gap_check's own settings when it runs as a script: its passes ran so
ENV = {"CUDA_VISIBLE_DEVICES": "-1",          # "" deletes the var on Windows
       "PROBE_ENV": "MsPacmanNoFrameskip-v4",
       "PROBE_ACTING_STATES": "1",              # the deployed policy
       "PROBE_THREADS": "2"}                    # as the rows ran
if __name__ == "__main__":
    os.environ.update(ENV)
import argparse
import hashlib
import json
import pathlib
import pickle
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
import probe_gap_check as gc

PILOT = HERE / "probe_gap_check-pilot"       # the pilot's files, read only
PLAYERS = tuple(gc.PLAYERS)                  # ("T0", "T2"): student, master
DELAYS = gc.DELAYS                           # the same 16 for both players
FRAME_CAP = gc.FRAME_CAP                     # 1,536 frames, delay included
PRIMARY = (0, 0)                             # (scale, discount): clip, 0.997
G5_BAR = 0.5                                 # the full sample's bar
DRAWS = 10_000
INTERVAL = (2.5, 97.5)                       # two-sided 95% percentiles
BOOT_SEED = (gc.SEED, 4)                     # streams 1-3 are the pilot's
LAG_PROFILE = tuple(range(1, len(DELAYS)))
SUMMARY = "gates.json"
CHUNK = int(os.environ.get("GAP2_CHUNK") or 100)
LIMIT = int(os.environ.get("GAP2_STATES") or 0)
SMOKE = LIMIT != 0
# This stage's own source and the nets it plays through, which the pilot's
# hash left out; the pilot's stamp (in every stamp below) covers the rest.
SOURCES = (pathlib.Path(__file__).resolve(), ROOT / "networks.py",
           ROOT / "scalar_categorical.py")
SOURCE = gc.source_hash(SOURCES)


# --------------------------------------------------------------------------
# returns and the real gap
# --------------------------------------------------------------------------
def branch_returns(table):
    """Every branch's return on every head: (states, delays, scales,
    discounts), from its raw rewards per base step (probe_gap_check's
    head_returns: the clip per base step, then raw; per-frame discounts)."""
    return np.array([[gc.head_returns(b["rewards"]) for b in row]
                     for row in table], dtype=np.float64)


def real_gap(t0, t2):
    """G(s): T2's mean return over the delays minus T0's, per head. `t0`, `t2`
    (states, delays, ...) from branch_returns, the same states and delays."""
    t0, t2 = np.asarray(t0, dtype=np.float64), np.asarray(t2, dtype=np.float64)
    if t0.shape != t2.shape:
        raise SystemExit(f"T0's returns {t0.shape} and T2's {t2.shape} differ")
    return t2.mean(axis=1) - t0.mean(axis=1)


def check_rows(table, n, name):
    """Each of the n states holds the 16 registered delays, in order."""
    if len(table) != n:
        raise SystemExit(f"{name}: branches for {len(table)} states, {n} kept")
    for i, row in enumerate(table):
        delays = [b["delay"] for b in row]
        if delays != list(DELAYS):
            raise SystemExit(f"{name} state {i}: delays {delays}, not the "
                             f"registered {list(DELAYS)}")


def ended_counts(table):
    return {e: sum(b["ended"] == e for row in table for b in row)
            for e in ("life", "cap", "delay")}


# --------------------------------------------------------------------------
# clusters and the interval
# --------------------------------------------------------------------------
def game_clusters(games):
    """CLUSTERS: games connected through shared stacks after the opening. Two
    main lines sharing any decision stack (their recorded digests; the opening
    records none) are one cluster, and so is a chain of them. {game: its
    cluster, named by its lowest game}."""
    parent = {g["game"]: g["game"] for g in games}

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    first = {}
    for g in games:
        for d in g["digests"]:
            if d not in first:
                first[d] = g["game"]
                continue
            a, b = root(first[d]), root(g["game"])
            if a != b:
                parent[max(a, b)] = min(a, b)
    return {g: root(g) for g in parent}


def state_clusters(states, games):
    """Each kept state's cluster: its game's. A merged state's games share
    its stack, so they must be one cluster already."""
    labels = game_clusters(games)
    out = []
    for s in states:
        c = labels[s["game"]]
        if any(labels[m] != c for m in s.get("merged", ())):
            raise SystemExit(f"game {s['game']} decision {s['decision']}: the "
                             f"games merged into it are not one cluster")
        out.append(c)
    return out


def cluster_draws(clusters, draws=DRAWS, seed=BOOT_SEED):
    """The bootstrap's draws, as index arrays into the states: as many
    clusters as there are, with replacement, each bringing all its states (a
    cluster drawn twice brings them twice)."""
    _, inverse = np.unique(np.asarray(clusters), return_inverse=True)
    inverse = inverse.ravel()
    members = [np.flatnonzero(inverse == c) for c in range(inverse.max() + 1)]
    rng = np.random.default_rng(seed)
    for _ in range(draws):
        pick = rng.integers(0, len(members), size=len(members))
        yield np.concatenate([members[c] for c in pick])


def cluster_interval(stat, clusters, draws=DRAWS):
    """The two-sided 95% percentile interval of stat(indices) over the
    cluster draws."""
    values = np.array([stat(idx) for idx in cluster_draws(clusters, draws)],
                      dtype=np.float64)
    lo, hi = np.percentile(values, INTERVAL)
    return float(lo), float(hi)


# --------------------------------------------------------------------------
# the gates
# --------------------------------------------------------------------------
def lag_profile(t0, t2):
    """The residual correlation at every lag, for each player's (states,
    delays) returns and for the per-delay gap."""
    t0, t2 = np.asarray(t0, dtype=np.float64), np.asarray(t2, dtype=np.float64)
    return {who: {lag: gc.lag_correlation(x, lag) for lag in LAG_PROFILE}
            for who, x in (("T0", t0), ("T2", t2), ("G", t2 - t0))}


def gates_verdict(g3, g4, g5):
    """("PASS-SO-FAR" | "VOID", reasons). `g4` {player: its reasons}."""
    reasons = []
    if not g3["ok"]:
        lo, hi = g3["interval"]
        reasons.append(f"G3: mean G {g3['mean']:+.4f} at the primary head, 95% "
                       f"interval [{lo:+.4f}, {hi:+.4f}]: T2 is not shown to "
                       f"beat T0 from T0's states")
    for name, rs in g4.items():
        if rs:
            reasons.append(f"G4 {name}: {rs[0]} ({len(rs)} in all)")
    if g5["reliability"] is None:
        reasons.append("G5: reliability undefined")
    elif not g5["ok"]:
        reasons.append(f"G5: reliability {g5['reliability']:.3f} under "
                       f"{G5_BAR}")
    return ("VOID", reasons) if reasons else ("PASS-SO-FAR", [])


def read_gates(t0, t2, keys, clusters, g4, draws=DRAWS):
    """G3, G4 and G5 from each player's branch_returns (states, delays,
    scales, discounts), the states' strata and clusters, and each player's G4
    reasons. Nothing here reads a value net or g."""
    s, d = PRIMARY
    p0 = np.asarray(t0, dtype=np.float64)[:, :, s, d]
    p2 = np.asarray(t2, dtype=np.float64)[:, :, s, d]
    gap = real_gap(t0, t2)[:, s, d]
    mean = float(gap.mean())
    lo, hi = cluster_interval(lambda idx: gap[idx].mean(), clusters, draws)
    g3 = {"mean": mean, "interval": [lo, hi], "draws": draws,
          "clusters": len(set(clusters)), "ok": mean > 0 and lo > 0}
    rho_half, reliability = gc.split_half(p2 - p0, keys)
    g5 = {"rho_half": rho_half, "reliability": reliability, "bar": G5_BAR,
          "ok": reliability is not None and reliability >= G5_BAR}
    verdict, reasons = gates_verdict(g3, g4, g5)
    return {"verdict": verdict, "reasons": reasons, "G3": g3,
            "G4": {n: {"ok": not r, "reasons": r} for n, r in g4.items()},
            "G5": g5, "lags": lag_profile(p0, p2), "null": gc.LAG_NULL}


# --------------------------------------------------------------------------
# files
# --------------------------------------------------------------------------
def knobs():
    return {"chunk": CHUNK, "states": LIMIT}


def fingerprint(states):
    """The states a pass consumed: their places, saved states and stacks."""
    h = hashlib.blake2b(digest_size=16)
    for s in states:
        h.update(repr((s["game"], s["decision"], s["life"],
                       s["steps"])).encode())
        h.update(s["state"])
        h.update(s["stack"].tobytes())
    return h.hexdigest()


def stamp(states):
    """What made a file: this stage's knobs and source, the pilot's stamp
    (its knobs and its play path's source), the states."""
    return {"knobs": knobs(), "source": SOURCE, "pilot": gc.stamp(),
            "states": fingerprint(states)}


def write(path, obj, st):
    """Whole or not at all, so a stopped pass leaves nothing to reuse."""
    obj = {**obj, "stamp": st}
    tmp = path.with_name(path.name + ".tmp")
    try:
        if path.suffix == ".json":
            tmp.write_text(json.dumps(obj, indent=1))
        else:
            with open(tmp, "wb") as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def read(path, st):
    """A pass's file, refused unless this stage would have written it."""
    if path.suffix == ".json":
        obj = json.loads(path.read_text())
    else:
        with open(path, "rb") as f:
            obj = pickle.load(f)
    if obj.get("stamp") != st:
        raise SystemExit(f"{path} was made by {obj.get('stamp')}, this stage "
                         f"is {st}: move it aside, passes are never mixed")
    return obj


def guard(out, pilot):
    """The pilot's files are read, never written: no output inside them."""
    o = pathlib.Path(out).resolve()
    for p in {pathlib.Path(pilot).resolve(), PILOT.resolve()}:
        if o == p or p in o.parents:
            raise SystemExit(f"{out} is inside the pilot's {p}: the pilot's "
                             f"files are never written")


def load_states(pilot):
    """The pilot's select.pkl (stamp checked) and the stage's states: all the
    kept states, or the first GAP2_STATES. (select, states, their strata)."""
    sel = gc.read(pathlib.Path(pilot) / "select.pkl")
    n = len(sel["kept"]) if not LIMIT else min(LIMIT, len(sel["kept"]))
    if n < 1:
        raise SystemExit("no kept states")
    return sel, sel["kept"][:n], [tuple(k) for k in sel["keys"][:n]]


def chunks(n):
    """[(k, first state, end)]: every state once, CHUNK at a time."""
    if CHUNK < 1:
        raise SystemExit(f"GAP2_CHUNK must be at least 1, got {CHUNK}")
    return [(k, lo, min(n, lo + CHUNK))
            for k, lo in enumerate(range(0, n, CHUNK))]


def chunk_range(n, k):
    found = [c for c in chunks(n) if c[0] == k]
    if not found:
        raise SystemExit(f"chunk {k}: {n} states make {len(chunks(n))}")
    return found[0][1:]


def branch_file(out, name, k):
    return pathlib.Path(out) / f"branches-{name}-{k:03d}.pkl"


def g4_file(out, name, k):
    return pathlib.Path(out) / f"g4-{name}-{k:03d}.json"


def ids(states):
    return [[s["game"], s["decision"]] for s in states]


def file_digest(path):
    """A G4 file names the branches file it replayed by this."""
    return hashlib.blake2b(pathlib.Path(path).read_bytes(),
                           digest_size=16).hexdigest()


def memory():
    """This process's peak working set and peak commit, MB."""
    try:
        import psutil
    except ImportError:
        return {}
    m = psutil.Process().memory_info()
    return {"peak_wset_mb": round(getattr(m, "peak_wset", m.rss) / 2 ** 20),
            "peak_commit_mb": round(getattr(m, "peak_pagefile", 0) / 2 ** 20)}


# --------------------------------------------------------------------------
# the passes
# --------------------------------------------------------------------------
def play_state(env, decide, s, checked):
    """One state's 16 branches, as the pilot's branch pass plays them; the
    `checked` delay's branch also replayed on raw ALE, its frames kept.
    (row, a failure or None)."""
    saved = ale_py.ALEState(s["state"])
    row = [gc.branch(env, decide, saved, s["stack"], d, cap=FRAME_CAP)
           for d in DELAYS]
    b = row[checked]
    b["replay"], end = gc.frames_of(env, saved, b)
    return row, (None if end == b["end"] else
                 f"delay {checked}: the raw replay ended elsewhere")


def stage_branches(out, pilot, name, k):
    guard(out, pilot)
    _, states, _ = load_states(pilot)
    st = stamp(states)
    lo, hi = chunk_range(len(states), k)
    checked = gc.g4_delays(len(states))
    t0, table, reasons = time.time(), [], []
    with gc.player(name) as run:
        env, decide = run.env, gc.deployed(run)
        for i in range(lo, hi):
            row, reason = play_state(env, decide, states[i], checked[i])
            if reason:
                reasons.append(f"state {i} {reason}")
            table.append(row)
            if (i + 1 - lo) % 10 == 0:
                print(f"branches {name} chunk {k}: {i + 1 - lo} of {hi - lo} "
                      f"states [{time.time() - t0:.0f}s]", flush=True)
    write(branch_file(out, name, k),
          {"player": name, "chunk": k, "range": [lo, hi],
           "states": ids(states[lo:hi]), "table": table, "reasons": reasons,
           "threads": torch.get_num_threads(),
           "secs": round(time.time() - t0, 1), "memory": memory()}, st)


def stage_g4(out, pilot, name, k):
    guard(out, pilot)
    sel, states, _ = load_states(pilot)
    st = stamp(states)
    lo, hi = chunk_range(len(states), k)
    data = read(branch_file(out, name, k), st)
    replayed = file_digest(branch_file(out, name, k))
    if data["range"] != [lo, hi] or data["player"] != name:
        raise SystemExit(f"{branch_file(out, name, k)}: not {name}'s states "
                         f"{lo}-{hi}")
    table, checked = data["table"], gc.g4_delays(len(states))
    reasons = list(data["reasons"])
    t0 = time.time()
    with gc.player(name) as run:
        env, decide = run.env, gc.deployed(run)
        for i in range(lo, hi):
            s = states[i]
            saved = ale_py.ALEState(s["state"])
            first = table[i - lo][checked[i]]
            again = gc.branch(env, decide, saved, s["stack"], checked[i],
                              cap=FRAME_CAP)
            frames, end = gc.frames_of(env, saved, again)
            differ = gc.same_branch(first, again)
            if frames != first["replay"]:
                differ.append("raw frames")
            if end != again["end"]:
                differ.append("raw replay's end")
            if differ:
                reasons.append(f"state {i} delay {checked[i]}: {differ}")
    if name == "T0":
        games = gc.read(gc.main_files(pathlib.Path(pilot),
                                      sel["opening"])[1])["games"]
        reasons += gc.main_line_reasons(states[lo:hi], games,
                                        [row[0] for row in table])
    print(f"G4 {name} chunk {k}: {hi - lo} branches in a fresh process, "
          f"{'OK' if not reasons else 'FAILED: ' + '; '.join(reasons[:4])} "
          f"[{time.time() - t0:.0f}s]", flush=True)
    write(g4_file(out, name, k),
          {"ok": not reasons, "reasons": reasons, "range": [lo, hi],
           "branches": replayed, "secs": round(time.time() - t0, 1),
           "memory": memory()}, st)


def run_pass(out, pilot, stage, name, k):
    """A pass in its own process; a failed pass stops the stage."""
    cmd = [sys.executable, "-u", str(pathlib.Path(__file__).resolve()), stage,
           "--player", name, "--chunk", str(k), "--out", str(out),
           "--pilot", str(pilot)]
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode:
        raise SystemExit(f"pass {stage} {name} {k}: rc {r.returncode}")


def pilot_record(pilot):
    """The pilot's verdict and its G1 and G6, which this stage builds on."""
    pil = gc.read(pilot / "pilot.json")
    if pil["verdict"] != "GO":
        raise SystemExit(f"the pilot's verdict is {pil['verdict']}, not GO: "
                         f"stage 2 does not run")
    g1 = {n: gc.read(pilot / f"g1-{n}.json")["ok"] for n in PLAYERS}
    if not all(g1.values()):
        raise SystemExit(f"the pilot's G1 failed: {g1}")
    return {"verdict": pil["verdict"], "G1": g1, "G6": pil["effective"][-1],
            "opening": pil["opening"], "counts": pil["counts"]}


def stage2a(out, pilot):
    out, pilot = pathlib.Path(out), pathlib.Path(pilot)
    guard(out, pilot)
    if (out / SUMMARY).exists():
        raise SystemExit(f"{out / SUMMARY} exists: the summary is never "
                         f"overwritten (move it aside first)")
    pil = pilot_record(pilot)
    _, states, _ = load_states(pilot)
    n, t0 = len(states), time.time()
    for name in PLAYERS:
        for k, _, _ in chunks(n):
            if not branch_file(out, name, k).exists():
                run_pass(out, pilot, "branches", name, k)
    for name in PLAYERS:
        for k, _, _ in chunks(n):
            if not g4_file(out, name, k).exists():
                run_pass(out, pilot, "g4", name, k)
    return finish(out, pilot, pil, t0)


def finish(out, pilot, pil, t0):
    """Every pass's file in, stamps checked: the real gap and the gates."""
    sel, states, keys = load_states(pilot)
    if sel["opening"] != pil["opening"]:
        raise SystemExit(f"select's opening {sel['opening']} is not the "
                         f"pilot's {pil['opening']}")
    st, n = stamp(states), len(states)
    tables, g4, cost = {}, {}, {}
    for name in PLAYERS:
        tables[name], g4[name] = [], []
        cost[name] = {"branch_secs": 0.0, "g4_secs": 0.0, "peak_wset_mb": 0,
                      "peak_commit_mb": 0}
        for k, lo, hi in chunks(n):
            b = read(branch_file(out, name, k), st)
            g = read(g4_file(out, name, k), st)
            if b["range"] != [lo, hi] or g["range"] != [lo, hi] \
                    or b["player"] != name:
                raise SystemExit(f"{name} chunk {k}: not states {lo}-{hi}")
            if g["branches"] != file_digest(branch_file(out, name, k)):
                raise SystemExit(f"{g4_file(out, name, k)} replayed another "
                                 f"{branch_file(out, name, k).name}: run its "
                                 f"G4 again")
            tables[name] += b["table"]
            g4[name] += g["reasons"]
            cost[name]["branch_secs"] += b["secs"]
            cost[name]["g4_secs"] += g["secs"]
            for f in (b, g):
                for key, v in f.get("memory", {}).items():
                    cost[name][key] = max(cost[name][key], v)
        check_rows(tables[name], n, name)
        cost[name]["secs_per_state"] = round(cost[name]["branch_secs"] / n, 2)
    returns = {name: branch_returns(tables[name]) for name in PLAYERS}
    gap = real_gap(returns["T0"], returns["T2"])
    games = gc.read(gc.main_files(pilot, sel["opening"])[1])["games"]
    clusters = state_clusters(states, games)
    rec = read_gates(returns["T0"], returns["T2"], keys, clusters, g4)
    sizes = {}
    for c in clusters:
        sizes[c] = sizes.get(c, 0) + 1
    strata = {}
    for k in keys:
        strata[k] = strata.get(k, 0) + 1
    write(out / "gap.pkl",
          {"states": ids(states), "merged": [s["merged"] for s in states],
           "keys": keys, "clusters": clusters, "delays": list(DELAYS),
           "scales": list(gc.SCALES), "discounts": list(gc.DISCOUNTS),
           "returns": returns, "G": gap}, st)
    rec.update(
        smoke=SMOKE, knobs=knobs(), states=n, pilot=pil,
        strata={"count": len(strata),
                "singletons": sum(v == 1 for v in strata.values())},
        clusters={"count": len(sizes), "largest": max(sizes.values())},
        G_by_head={sc: {str(dc): float(gap[:, i, j].mean())
                        for j, dc in enumerate(gc.DISCOUNTS)}
                   for i, sc in enumerate(gc.SCALES)},
        ended={name: ended_counts(tables[name]) for name in PLAYERS},
        cost=cost, memory=memory(), secs=round(time.time() - t0))
    report(rec)
    if (out / SUMMARY).exists():
        raise SystemExit(f"{out / SUMMARY} appeared: never overwritten")
    write(out / SUMMARY, rec, st)
    return rec


def report(rec):
    g3, g5 = rec["G3"], rec["G5"]
    lo, hi = g3["interval"]
    print(f"{rec['states']} states, {rec['strata']['count']} strata "
          f"({rec['strata']['singletons']} of one), {rec['clusters']['count']} "
          f"clusters (largest {rec['clusters']['largest']} states)", flush=True)
    print(f"G3: mean G at the primary head {g3['mean']:+.4f} clipped units, "
          f"95% interval [{lo:+.4f}, {hi:+.4f}] ({g3['draws']} cluster "
          f"draws): {'OK' if g3['ok'] else 'FAILED'}", flush=True)
    for name, g in rec["G4"].items():
        print(f"G4 {name}: {'OK' if g['ok'] else 'FAILED: ' + g['reasons'][0]}",
              flush=True)
    print(f"G5: rho_w of the halves {g5['rho_half']}, Spearman-Brown "
          f"{g5['reliability']}, bar {G5_BAR}: "
          f"{'OK' if g5['ok'] else 'FAILED'}", flush=True)
    for who, lags in rec["lags"].items():
        print(f"lag profile {who} (null {rec['null']:+.4f}): " + " ".join(
            f"{lag}:{'undef' if r is None else format(r, '+.3f')}"
            for lag, r in lags.items()), flush=True)
    for sc, by in rec["G_by_head"].items():
        print(f"mean G, {sc}: " + ", ".join(f"{dc} {v:+.4f}"
                                            for dc, v in by.items()), flush=True)
    for name, e in rec["ended"].items():
        print(f"{name} branches ended: {e}", flush=True)
    line = f"STAGE 2a: {rec['verdict']}" + (
        f" ({'; '.join(rec['reasons'])})" if rec["reasons"] else "")
    print(("SMOKE " if rec["smoke"] else "") + line, flush=True)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("stage2a", "branches", "g4"))
    ap.add_argument("--player", choices=PLAYERS)
    ap.add_argument("--chunk", type=int)
    ap.add_argument("--out")
    ap.add_argument("--pilot")
    args = ap.parse_args(argv)
    if LIMIT < 0:
        raise SystemExit(f"GAP2_STATES must be 0 or more, got {LIMIT}")
    pilot = pathlib.Path(args.pilot).resolve() if args.pilot else PILOT
    out = pathlib.Path(args.out).resolve() if args.out else \
        HERE / f"probe_gap_stage2a{'-smoke' if SMOKE else ''}"
    guard(out, pilot)
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "stage2a":
        stage2a(out, pilot)
        return
    if args.player is None or args.chunk is None:
        raise SystemExit(f"{args.stage} needs --player and --chunk")
    {"branches": stage_branches, "g4": stage_g4}[args.stage](
        out, pilot, args.player, args.chunk)


if __name__ == "__main__":
    main(sys.argv[1:])
